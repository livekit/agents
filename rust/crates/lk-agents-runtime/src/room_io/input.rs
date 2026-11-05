//! Executes a participant input's track selection: opens the tracks voice selects, pumps their
//! frames into the bounded queue the session reads, and retires the streams voice retires.

type PublicationSnapshot = (Vec<Publication>, Vec<Arc<dyn TrackPublication>>);

use crate::telemetry;
use lk_agents_abi::voice::io::{AudioInput, InputFrame, IoError, Result, VideoInput};
use lk_agents_abi::voice::room_io::AudioInputConfig;
use lk_agents_abi::voice::rtc::TrackSource;
use lk_agents_abi::voice::rtc::{
    AudioInputStream, AudioStreamItem, AudioStreamOptions, AudioTrackFeature, ParticipantIdentity,
    ProcessorFailure, RemoteParticipant, RoomEvent, RtcRoom, TrackPublication, TrackSid,
    VideoFrameRef,
};
use lk_agents_abi::{AudioFrame, Clock, Closed, HOST_TEARDOWN_TIMEOUT, Spawner, StreamOutput, Ts};
use lk_agents_media::media_queue::{
    self, AudioReceiver, AudioSender, Item, Stuck, VideoReceiver, VideoSender,
};
use lk_agents_net::audio_drops::AudioDrops;
use lk_agents_net::task::TaskSet;
use lk_agents_voice::room_io::machine::input::{
    AUDIO_SOURCES, Admit, Open, Owner, ParticipantInput, ProcessorSpans, Publication,
    VIDEO_SOURCES, end_of_track, replayed_head,
};
use parking_lot::Mutex;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

use super::pre_connect_audio::PreConnectAudioHandler;

const AUDIO_POLICY: media_queue::Policy = media_queue::Policy::CatchUp(lk_agents_abi::AUDIO_QUEUE);

/// What travels in a reader's queue besides frames.
enum InputEvent {
    /// The selection or attachment changed: what was queued before no longer reaches the session.
    SourceChanged,
    Failed(IoError),
}

/// How a track reader hands its frames and events to the session's media queue.
trait ReaderSink: Send + Sync + 'static {
    type Frame: Send + 'static;
    type Receiver: Send + 'static;
    /// Returns the audio the push dropped.
    fn push(&self, frame: Self::Frame) -> Duration;
    fn event(&self, event: InputEvent) -> std::result::Result<(), Stuck>;
    fn flush(&self, event: InputEvent) -> std::result::Result<(), Stuck>;
    fn is_closed(&self) -> bool;
    fn recv(
        receiver: &mut Self::Receiver,
    ) -> impl Future<Output = Option<Item<Self::Frame, InputEvent>>> + Send;
    /// The pre-connect audio received so far.
    fn head(receiver: &Self::Receiver) -> Duration;
}

impl ReaderSink for AudioSender<InputEvent> {
    type Frame = AudioFrame;
    type Receiver = AudioReceiver<InputEvent>;
    fn push(&self, frame: AudioFrame) -> Duration {
        AudioSender::push(self, frame)
    }
    fn event(&self, event: InputEvent) -> std::result::Result<(), Stuck> {
        AudioSender::event(self, event)
    }
    fn flush(&self, event: InputEvent) -> std::result::Result<(), Stuck> {
        AudioSender::flush(self, event)
    }
    fn is_closed(&self) -> bool {
        AudioSender::is_closed(self)
    }
    fn recv(
        receiver: &mut AudioReceiver<InputEvent>,
    ) -> impl Future<Output = Option<Item<AudioFrame, InputEvent>>> + Send {
        receiver.recv()
    }
    fn head(receiver: &AudioReceiver<InputEvent>) -> Duration {
        receiver.head().received
    }
}

impl ReaderSink for VideoSender<InputEvent> {
    type Frame = VideoFrameRef;
    type Receiver = VideoReceiver<InputEvent>;
    fn push(&self, frame: VideoFrameRef) -> Duration {
        VideoSender::push(self, frame);
        Duration::ZERO
    }
    fn event(&self, event: InputEvent) -> std::result::Result<(), Stuck> {
        VideoSender::event(self, event)
    }
    fn flush(&self, event: InputEvent) -> std::result::Result<(), Stuck> {
        VideoSender::flush(self, event)
    }
    fn is_closed(&self) -> bool {
        VideoSender::is_closed(self)
    }
    fn recv(
        receiver: &mut VideoReceiver<InputEvent>,
    ) -> impl Future<Output = Option<Item<VideoFrameRef, InputEvent>>> + Send {
        receiver.recv()
    }
    fn head(_: &VideoReceiver<InputEvent>) -> Duration {
        Duration::ZERO
    }
}

struct State<Q> {
    readers: ParticipantInput,
    queue: Option<Q>,
}

/// The selection, the queue it admits frames into, and one wake for every reader.
struct Readers<Q: ReaderSink> {
    state: Mutex<State<Q>>,
    queue: tokio::sync::Mutex<Q::Receiver>,
    changed: Notify,
    /// Where an audio queue's drops are reported.
    drops: Option<AudioDrops>,
}

impl<Q: ReaderSink> Readers<Q> {
    fn new(
        sources: &'static [TrackSource],
        (tx, rx): (Q, Q::Receiver),
        drops: Option<AudioDrops>,
    ) -> Self {
        Self {
            state: Mutex::new(State { readers: ParticipantInput::new(sources), queue: Some(tx) }),
            queue: tokio::sync::Mutex::new(rx),
            changed: Notify::new(),
            drops,
        }
    }

    /// Applies a decision, flushes what a source change left queued, and wakes the readers it may
    /// have retired.
    fn decide<R>(&self, decision: impl FnOnce(&mut ParticipantInput) -> R) -> R {
        let outcome = {
            let mut state = self.state.lock();
            let source = state.readers.source();
            let outcome = decision(&mut state.readers);
            if state.readers.source() != source
                && let Some(queue) = &state.queue
            {
                // A stuck flush still drops the old source's frames, and nothing reads the event.
                let _ = queue.flush(InputEvent::SourceChanged);
            }
            outcome
        };
        self.changed.notify_waiters();
        outcome
    }

    fn reading(&self) -> Option<TrackSid> {
        self.state.lock().readers.reading().cloned()
    }

    /// `false` stops the reader.
    fn deliver(&self, generation: u64, item: Result<Q::Frame>) -> bool {
        self.admitted(generation, |queue| match item {
            Ok(frame) => queue.push(frame),
            Err(error) => {
                let _ = queue.event(InputEvent::Failed(error));
                Duration::ZERO
            }
        })
    }

    /// Hands a frame of the reader of `generation` to `deliver` as voice admits it, and reports
    /// the audio its push dropped. `false` stops the reader.
    fn admitted(&self, generation: u64, deliver: impl FnOnce(&Q) -> Duration) -> bool {
        let (open, dropped) = {
            let state = self.state.lock();
            match (state.readers.admit(generation), &state.queue) {
                (Admit::Stop, _) | (Admit::Deliver, None) => (false, Duration::ZERO),
                (Admit::Drop, _) => (true, Duration::ZERO),
                (Admit::Deliver, Some(queue)) => {
                    let dropped = deliver(queue);
                    (!queue.is_closed(), dropped)
                }
            }
        };
        if let Some(drops) = &self.drops {
            drops.record(dropped);
        }
        open
    }

    /// Resolves once voice retires the reader of `generation`.
    async fn retired(&self, generation: u64) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.state.lock().readers.producing(generation) {
                return;
            }
            changed.await;
        }
    }

    /// The next frame, and whether it replays pre-connect audio.
    async fn recv(&self) -> Result<Option<(Q::Frame, bool)>> {
        let mut queue = self.queue.lock().await;
        loop {
            let head = Q::head(&queue);
            match Q::recv(&mut queue).await {
                Some(Item::Media(frame)) => return Ok(Some((frame, Q::head(&queue) > head))),
                Some(Item::Event(InputEvent::Failed(error))) => return Err(error),
                Some(Item::Gap(_) | Item::Event(InputEvent::SourceChanged)) => {}
                None => return Ok(None),
            }
        }
    }

    /// Readers keep retiring their streams after the waiter reaches the deadline.
    async fn close(&self, clock: &dyn Clock, deadline: Ts) -> Closed {
        self.decide(ParticipantInput::close);
        let queue = self.state.lock().queue.take();
        drop(queue);
        let settled = async {
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if let Some(outcome) = self.state.lock().readers.settled() {
                    return outcome;
                }
                changed.await;
            }
        };
        let remaining = deadline.saturating_duration_since(clock.now());
        lk_agents_abi::timeout(clock, remaining, settled).await.unwrap_or(Closed::TimedOut)
    }
}

/// The participant's publications as voice reads them, with the SDK handles to open.
fn publications(participant: &dyn RemoteParticipant) -> Result<PublicationSnapshot> {
    let handles = participant.track_publications()?;
    let listed = handles
        .iter()
        .map(|publication| {
            Ok(Publication {
                sid: publication.sid().clone(),
                source: publication.source(),
                subscribed: publication.is_subscribed()?,
            })
        })
        .collect::<Result<_>>()?;
    Ok((listed, handles))
}

/// One kind of input's SDK reader: opens the selected track and pumps it.
trait Reader: Sized + Send + Sync + 'static {
    type Queue: ReaderSink;

    fn readers(&self) -> &Readers<Self::Queue>;
    fn room(&self) -> &dyn RtcRoom;
    /// Opens the stream and spawns its reader, which reports `retired` when its stream is closed.
    fn open(
        self: &Arc<Self>,
        participant: &dyn RemoteParticipant,
        publication: &Arc<dyn TrackPublication>,
        open: &Open,
    ) -> Result<()>;
}

fn perform<R: Reader>(
    reader: &Arc<R>,
    participant: &dyn RemoteParticipant,
    handles: &[Arc<dyn TrackPublication>],
    open: Option<Open>,
) -> Result<()> {
    let Some(open) = open else { return Ok(()) };
    let publication = handles.iter().find(|publication| *publication.sid() == open.sid);
    let opened = match publication {
        Some(publication) => reader.open(participant, publication, &open),
        None => Ok(()),
    };
    if publication.is_none() || opened.is_err() {
        reader.readers().decide(|readers| readers.retired(open.generation, Closed::Clean));
    }
    opened
}

fn link<R: Reader>(reader: &Arc<R>, identity: Option<ParticipantIdentity>) -> Result<()> {
    let participant = match &identity {
        Some(identity) => reader.room().participant(identity)?,
        None => None,
    };
    let (listed, handles) = match &participant {
        Some(participant) => publications(participant.as_ref())?,
        None => Default::default(),
    };
    let owner = participant.as_ref().map(|p| Owner { identity: p.identity(), sid: p.sid() });
    let open = reader.readers().decide(|readers| readers.link(identity.clone(), owner, &listed));
    match participant {
        Some(participant) => perform(reader, participant.as_ref(), &handles, open),
        None => Ok(()),
    }
}

fn route<R: Reader>(reader: &Arc<R>, event: &RoomEvent) -> Result<()> {
    let (participant, publication, unsubscribed) = match event {
        RoomEvent::TrackSubscribed { participant, publication } => {
            let owner = Owner { identity: participant.identity(), sid: participant.sid() };
            let listed = Publication {
                sid: publication.sid().clone(),
                source: publication.source(),
                subscribed: true,
            };
            let open = reader.readers().decide(|readers| readers.subscribed(owner, &listed));
            return perform(reader, participant.as_ref(), std::slice::from_ref(publication), open);
        }
        RoomEvent::TrackUnpublished { participant, publication } => {
            (participant, publication, false)
        }
        RoomEvent::TrackUnsubscribed { participant, publication } => {
            (participant, publication, true)
        }
        _ => return Ok(()),
    };
    if reader.readers().reading().as_ref() != Some(publication.sid()) {
        return Ok(());
    }
    let (listed, handles) = publications(participant.as_ref())?;
    let owner = Owner { identity: participant.identity(), sid: participant.sid() };
    let open = reader
        .readers()
        .decide(|readers| readers.removed(owner, publication.sid(), unsubscribed, &listed));
    perform(reader, participant.as_ref(), &handles, open)
}

/// Reads a stream until voice retires its reader or it ends, then retires the stream. `admit`
/// turns a stream item into the frame it delivers, if any.
async fn pump<S: Send + 'static, Q: ReaderSink, A: Future<Output = Option<Q::Frame>>>(
    readers: &Readers<Q>,
    clock: &dyn Clock,
    generation: u64,
    mut stream: Box<dyn StreamOutput<S>>,
    replay: impl Future<Output = bool>,
    end: impl FnOnce() -> Option<Q::Frame>,
    mut admit: impl FnMut(S) -> A,
) {
    let reading = async {
        if !replay.await {
            return;
        }
        loop {
            match stream.next().await {
                Ok(Some(item)) => {
                    if let Some(frame) = admit(item).await
                        && !readers.deliver(generation, Ok(frame))
                    {
                        return;
                    }
                }
                Ok(None) => {
                    if let Some(frame) = end() {
                        readers.deliver(generation, Ok(frame));
                    }
                    return;
                }
                Err(error) => {
                    readers.deliver(generation, Err(IoError::from(error)));
                    return;
                }
            }
        }
    };
    tokio::select! {
        biased;
        () = readers.retired(generation) => {},
        () = reading => {},
    }
    let cleanup = stream.close(clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
    readers.decide(|readers| readers.retired(generation, cleanup));
}

pub struct ParticipantVideoInput {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    readers: Readers<VideoSender<InputEvent>>,
    /// A track's pump, which ends once its generation is replaced or its stream ends.
    pumps: Arc<TaskSet>,
}

impl ParticipantVideoInput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        Arc::new(Self {
            room,
            spawner,
            clock,
            readers: Readers::new(
                VIDEO_SOURCES,
                media_queue::video(lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES),
                None,
            ),
            pumps: Arc::default(),
        })
    }

    pub fn reading(&self) -> Option<TrackSid> {
        self.readers.reading()
    }

    pub fn set_participant(self: &Arc<Self>, identity: Option<ParticipantIdentity>) -> Result<()> {
        link(self, identity)
    }

    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        route(self, event)
    }

    pub async fn aclose(&self) -> Closed {
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        self.readers.close(self.clock.as_ref(), deadline).await
    }
}

impl Reader for ParticipantVideoInput {
    type Queue = VideoSender<InputEvent>;

    fn readers(&self) -> &Readers<VideoSender<InputEvent>> {
        &self.readers
    }

    fn room(&self) -> &dyn RtcRoom {
        self.room.as_ref()
    }

    fn open(
        self: &Arc<Self>,
        _: &dyn RemoteParticipant,
        publication: &Arc<dyn TrackPublication>,
        open: &Open,
    ) -> Result<()> {
        let stream = publication.open_video_stream()?;
        let (owner, generation) = (self.clone(), open.generation);
        self.pumps.spawn(self.spawner.as_ref(), "video track pump", async move {
            let readers = &owner.readers;
            let clock = owner.clock.as_ref();
            let admit = |frame| std::future::ready(Some(frame));
            pump(readers, clock, generation, stream, async { true }, || None, admit).await;
        });
        Ok(())
    }
}

impl VideoInput for ParticipantVideoInput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_frame<'a>(
        &'a self,
        _cx: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<VideoFrameRef>>> {
        Box::pin(async move { Ok(self.readers.recv().await?.map(|(frame, _)| frame)) })
    }
    fn on_attached(&self) {
        self.readers.decide(|readers| readers.set_attached(true));
    }
    fn on_detached(&self) {
        self.readers.decide(|readers| readers.set_attached(false));
    }
}

pub struct ParticipantAudioInput {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    config: AudioInputConfig,
    pre_connect: Option<Arc<PreConnectAudioHandler>>,
    /// Where a processor's failing span is reported to the session.
    failures: tokio::sync::mpsc::Sender<ProcessorFailure>,
    readers: Readers<AudioSender<InputEvent>>,
    /// A track's pump, which ends once its generation is replaced or its stream ends.
    pumps: Arc<TaskSet>,
    /// The session's `wait_for_audio_track`, until the first frame; with the subscription's time.
    first_frame: Mutex<Option<(telemetry::ScopedSpan<telemetry::WaitForAudioTrack>, Option<u64>)>>,
}

/// `rtc.TrackSource.Name(..)`.
fn source_name(source: TrackSource) -> &'static str {
    match source {
        TrackSource::Unknown => "SOURCE_UNKNOWN",
        TrackSource::Camera => "SOURCE_CAMERA",
        TrackSource::Microphone => "SOURCE_MICROPHONE",
        TrackSource::Screenshare => "SOURCE_SCREENSHARE",
        TrackSource::ScreenshareAudio => "SOURCE_SCREENSHARE_AUDIO",
    }
}

impl ParticipantAudioInput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        config: AudioInputConfig,
        pre_connect: Option<Arc<PreConnectAudioHandler>>,
        failures: tokio::sync::mpsc::Sender<ProcessorFailure>,
    ) -> Arc<Self> {
        Arc::new(Self {
            room,
            spawner,
            clock,
            config,
            pre_connect,
            failures,
            readers: Readers::new(
                AUDIO_SOURCES,
                media_queue::audio(AUDIO_POLICY),
                Some(AudioDrops::new(telemetry::AudioConsumer::RoomInput, AUDIO_POLICY)),
            ),
            pumps: Arc::default(),
            first_frame: Mutex::new(None),
        })
    }

    /// Records the linked participant's track and first frame on `span`, which the first frame
    /// ends.
    pub fn wait_for_track(&self, span: telemetry::ScopedSpan<telemetry::WaitForAudioTrack>) {
        *self.first_frame.lock() = Some((span, None));
    }

    /// Reports the audio the input drops under `cx`'s job.
    pub fn observe(&self, cx: telemetry::Cx) {
        if let Some(drops) = &self.readers.drops {
            drops.observe(cx);
        }
    }

    fn track_subscribed(&self, sid: &TrackSid, source: TrackSource) {
        if let Some((span, subscribed)) = self.first_frame.lock().as_mut() {
            let track_id = sid.as_str().to_owned();
            telemetry::event(
                span,
                telemetry::TrackSubscribed { track_id, source: source_name(source).to_owned() },
            );
            *subscribed = Some(span.cx().now_nanos());
        }
    }

    fn frame_admitted(&self) {
        let Some((span, subscribed)) = self.first_frame.lock().take() else { return };
        telemetry::event(&span, telemetry::FirstFrame);
        let now = span.cx().now_nanos();
        let delay = subscribed.map(|at| telemetry::Seconds(now.saturating_sub(at) as f64 / 1e9));
        telemetry::end(span, telemetry::WaitForAudioTrackEnd { first_frame_delay: delay });
    }

    pub fn reading(&self) -> Option<TrackSid> {
        self.readers.reading()
    }

    pub fn set_participant(self: &Arc<Self>, identity: Option<ParticipantIdentity>) -> Result<()> {
        link(self, identity)
    }

    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        route(self, event)
    }

    /// A frame to deliver, or a processor failure reported when it opens that processor's span.
    async fn admit(
        &self,
        spans: &Mutex<ProcessorSpans>,
        item: AudioStreamItem,
    ) -> Option<AudioFrame> {
        match item {
            AudioStreamItem::Frame(frame) => {
                spans.lock().frame();
                self.frame_admitted();
                Some(frame)
            }
            AudioStreamItem::ProcessorFailed(failure) => {
                // A span reports once, so its report waits for room rather than drop; a closed
                // channel is a session that is gone.
                if spans.lock().failed(&failure.label) {
                    let _ = self.failures.send(failure).await;
                }
                None
            }
        }
    }

    /// The publisher's pre-connect head, through the track's processing, ahead of the live
    /// frames. `false` stops the reader.
    async fn replay(
        &self,
        generation: u64,
        sid: &TrackSid,
        input_stream: Option<Arc<dyn AudioInputStream>>,
        spans: &Mutex<ProcessorSpans>,
    ) -> bool {
        let (Some(input_stream), Some(handler)) = (input_stream, &self.pre_connect) else {
            return true;
        };
        let frames = match replayed_head(handler.wait_for_data(sid).await) {
            Ok(frames) => frames,
            Err(error) => {
                self.readers.deliver(generation, Err(error));
                return false;
            }
        };
        if let Some((span, _)) = self.first_frame.lock().as_ref() {
            let duration: Duration =
                frames.iter().map(lk_agents_abi::voice::rtc::frame_duration).sum();
            telemetry::event(
                span,
                telemetry::PreConnectAudio { duration: telemetry::Seconds(duration.as_secs_f64()) },
            );
        }
        for frame in frames {
            match input_stream.process_audio(&telemetry::Cx::NONE, vec![frame]).await {
                Ok(items) => {
                    for item in items {
                        if let Some(frame) = self.admit(spans, item).await
                            && !self.readers.admitted(generation, |queue| {
                                queue.head(frame);
                                Duration::ZERO
                            })
                        {
                            return false;
                        }
                    }
                }
                Err(error) => {
                    self.readers.deliver(generation, Err(error.into()));
                    return false;
                }
            }
            crate::process::yield_now().await;
        }
        true
    }

    pub async fn aclose(&self) -> Closed {
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        self.readers.close(self.clock.as_ref(), deadline).await
    }
}

impl Reader for ParticipantAudioInput {
    type Queue = AudioSender<InputEvent>;

    fn readers(&self) -> &Readers<AudioSender<InputEvent>> {
        &self.readers
    }

    fn room(&self) -> &dyn RtcRoom {
        self.room.as_ref()
    }

    fn open(
        self: &Arc<Self>,
        participant: &dyn RemoteParticipant,
        publication: &Arc<dyn TrackPublication>,
        open: &Open,
    ) -> Result<()> {
        let input_stream = publication.open_audio_stream(AudioStreamOptions {
            sample_rate: self.config.sample_rate,
            num_channels: self.config.num_channels,
            frame_size_ms: Some(self.config.frame_size_ms),
            noise_cancellation: self.config.noise_cancellation.clone(),
            auto_gain_control: self.config.auto_gain_control,
            participant_identity: participant.identity().clone(),
            queue_size_ms: lk_agents_abi::AUDIO_QUEUE.as_millis() as u32,
        })?;
        let stream = input_stream.frames()?;
        self.track_subscribed(&open.sid, publication.source());
        let owner = Owner { identity: participant.identity(), sid: participant.sid() };
        let replay = publication.audio_features().contains(&AudioTrackFeature::PreconnectBuffer)
            && self.readers.decide(|readers| readers.replay_pre_connect(owner, &open.sid));
        let (input, generation, sid) = (self.clone(), open.generation, open.sid.clone());
        self.pumps.spawn(self.spawner.as_ref(), "audio track pump", async move {
            let (sample_rate, num_channels) = (input.config.sample_rate, input.config.num_channels);
            let spans = Mutex::new(ProcessorSpans::default());
            let replay = input.replay(generation, &sid, replay.then_some(input_stream), &spans);
            let end = || Some(end_of_track(sample_rate, num_channels));
            let admit = |item| input.admit(&spans, item);
            pump(&input.readers, input.clock.as_ref(), generation, stream, replay, end, admit)
                .await;
        });
        Ok(())
    }
}

impl AudioInput for ParticipantAudioInput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_frame<'a>(
        &'a self,
        _cx: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<AudioFrame>>> {
        Box::pin(async move { Ok(self.readers.recv().await?.map(|(frame, _)| frame)) })
    }
    fn next_input<'a>(
        &'a self,
        _cx: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<InputFrame>>> {
        Box::pin(async move {
            Ok(self.readers.recv().await?.map(|(frame, head)| InputFrame { frame, head }))
        })
    }
    fn on_attached(&self) {
        self.readers.decide(|readers| readers.set_attached(true));
    }
    fn on_detached(&self) {
        self.readers.decide(|readers| readers.set_attached(false));
    }
}

impl lk_agents_abi::voice::room_owner::RoomAudioInputPort for ParticipantAudioInput {
    fn reading(&self) -> Option<TrackSid> {
        ParticipantAudioInput::reading(self)
    }
    fn aclose<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Closed> {
        Box::pin(async move { ParticipantAudioInput::aclose(self).await })
    }
}

impl lk_agents_abi::voice::room_owner::RoomVideoInputPort for ParticipantVideoInput {
    fn reading(&self) -> Option<TrackSid> {
        ParticipantVideoInput::reading(self)
    }
    fn aclose<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Closed> {
        Box::pin(async move { ParticipantVideoInput::aclose(self).await })
    }
}
