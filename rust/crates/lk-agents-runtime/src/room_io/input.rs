//! Executes a participant input's track selection: opens the tracks voice selects, pumps their
//! frames into the bounded queue the session reads, and retires the streams voice retires.

type PublicationSnapshot = (Vec<Publication>, Vec<Arc<dyn TrackPublication>>);

use crate::telemetry;
use lk_agents_abi::voice::io::{AudioInput, IoError, Result, VideoInput};
use lk_agents_abi::voice::room_io::AudioInputConfig;
use lk_agents_abi::voice::rtc::TrackSource;
use lk_agents_abi::voice::rtc::{
    AudioInputStream, AudioStreamItem, AudioStreamOptions, AudioTrackFeature, ParticipantIdentity,
    ProcessorFailure, RemoteParticipant, RoomEvent, RtcRoom, TrackPublication, TrackSid,
    VideoFrameRef,
};
use lk_agents_abi::{AudioFrame, Clock, Closed, HOST_TEARDOWN_TIMEOUT, Spawner, StreamOutput, Ts};
use lk_agents_net::leaky;
use lk_agents_net::task::TaskSet;
use lk_agents_voice::room_io::machine::input::{
    AUDIO_SOURCES, Admit, InputIdentity, Open, Owner, ParticipantInput, ProcessorSpans,
    Publication, VIDEO_SOURCES, end_of_track, replayed_head,
};
use parking_lot::Mutex;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Notify;

use super::pre_connect_audio::PreConnectAudioHandler;

/// Where a reader's admitted frames wait for the session, tagged with what admitted them.
trait Queue: Send + 'static {
    type Frame: Send + 'static;
    type Rx: Send + 'static;

    fn channel() -> (Self, Self::Rx)
    where
        Self: Sized;
    /// `false` once the session's side is gone.
    fn push(&mut self, identity: InputIdentity, item: Result<Self::Frame>) -> bool;
    /// The next item `accepts` admits; `None` once the producers are gone and it is drained.
    fn recv(
        rx: &mut Self::Rx,
        accepts: impl Fn(InputIdentity) -> bool + Send,
    ) -> impl Future<Output = Result<Option<Self::Frame>>> + Send;
}

struct Delivery<T> {
    identity: InputIdentity,
    item: Result<T>,
}

/// Video keeps the newest few frames: each held frame pins a decoded host frame.
struct VideoQueue(leaky::VideoSender<Delivery<VideoFrameRef>>);

impl Queue for VideoQueue {
    type Frame = VideoFrameRef;
    type Rx = leaky::VideoReceiver<Delivery<VideoFrameRef>>;

    fn channel() -> (Self, Self::Rx) {
        let (tx, rx) = leaky::video_channel(lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES);
        (VideoQueue(tx), rx)
    }

    fn push(&mut self, identity: InputIdentity, item: Result<VideoFrameRef>) -> bool {
        self.0.push(Delivery { identity, item });
        !self.0.is_closed()
    }

    async fn recv(
        rx: &mut Self::Rx,
        accepts: impl Fn(InputIdentity) -> bool + Send,
    ) -> Result<Option<VideoFrameRef>> {
        while let Some(frame) = rx.recv().await {
            if accepts(frame.identity) {
                return frame.item.map(Some);
            }
        }
        Ok(None)
    }
}

/// What a frame's admission changed, in order with the audio.
enum Mark {
    /// The frames that follow were admitted under this identity.
    Admitted(InputIdentity),
    Failed(IoError),
}

/// Audio goes through a `leaky` channel; a new identity clears what the old one queued, which
/// the session would refuse anyway.
struct AudioQueue {
    tx: leaky::Sender<Mark>,
    admitted: Option<InputIdentity>,
}

struct AudioRx {
    rx: leaky::Receiver<Mark>,
    batch: std::collections::VecDeque<leaky::Item<Mark>>,
    admitted: Option<InputIdentity>,
}

impl AudioQueue {
    fn admit(&mut self, identity: InputIdentity) -> bool {
        if self.admitted != Some(identity) {
            self.tx.clear();
            if self.tx.control(Mark::Admitted(identity)).is_err() {
                return false;
            }
            self.admitted = Some(identity);
        }
        !self.tx.is_closed()
    }

    fn head(&mut self, identity: InputIdentity, frame: AudioFrame) -> bool {
        let admitted = self.admit(identity);
        self.tx.head(frame);
        admitted
    }
}

impl Queue for AudioQueue {
    type Frame = AudioFrame;
    type Rx = AudioRx;

    fn channel() -> (Self, AudioRx) {
        let policy = leaky::Policy::CatchUp { max: lk_agents_abi::AUDIO_QUEUE };
        let (tx, rx) = leaky::audio_channel(policy);
        let rx = AudioRx { rx, batch: Default::default(), admitted: None };
        (AudioQueue { tx, admitted: None }, rx)
    }

    fn push(&mut self, identity: InputIdentity, item: Result<AudioFrame>) -> bool {
        if !self.admit(identity) {
            return false;
        }
        match item {
            Ok(frame) => self.tx.push(frame),
            Err(error) => return self.tx.control(Mark::Failed(error)).is_ok(),
        }
        true
    }

    async fn recv(
        rx: &mut AudioRx,
        accepts: impl Fn(InputIdentity) -> bool + Send,
    ) -> Result<Option<AudioFrame>> {
        loop {
            while let Some(item) = rx.batch.pop_front() {
                let accepted = rx.admitted.is_some_and(&accepts);
                match item {
                    leaky::Item::Audio(frame) if accepted => return Ok(Some(frame)),
                    leaky::Item::Control(Mark::Failed(error)) if accepted => return Err(error),
                    leaky::Item::Control(Mark::Admitted(identity)) => rx.admitted = Some(identity),
                    _ => {}
                }
            }
            match rx.rx.recv().await {
                Some(batch) => rx.batch.extend(batch),
                None => return Ok(None),
            }
        }
    }
}

struct State<Q> {
    readers: ParticipantInput,
    queue: Option<Q>,
}

/// The selection, the queue it admits frames into, and one wake for every reader.
struct Readers<Q: Queue> {
    state: Mutex<State<Q>>,
    queue: tokio::sync::Mutex<Q::Rx>,
    changed: Notify,
}

impl<Q: Queue> Readers<Q> {
    fn new(sources: &'static [lk_agents_abi::voice::rtc::TrackSource]) -> Self {
        let (queue, rx) = Q::channel();
        Self {
            state: Mutex::new(State {
                readers: ParticipantInput::new(sources),
                queue: Some(queue),
            }),
            queue: tokio::sync::Mutex::new(rx),
            changed: Notify::new(),
        }
    }

    /// Applies a decision and wakes the readers it may have retired.
    fn decide<R>(&self, decision: impl FnOnce(&mut ParticipantInput) -> R) -> R {
        let outcome = decision(&mut self.state.lock().readers);
        self.changed.notify_waiters();
        outcome
    }

    fn reading(&self) -> Option<TrackSid> {
        self.state.lock().readers.reading().cloned()
    }

    /// `false` stops the reader.
    fn deliver(&self, generation: u64, item: Result<Q::Frame>) -> bool {
        self.admitted(generation, |queue, identity| queue.push(identity, item))
    }

    /// Hands a frame of the reader of `generation` to `deliver` as voice admits it.
    fn admitted(
        &self,
        generation: u64,
        deliver: impl FnOnce(&mut Q, InputIdentity) -> bool,
    ) -> bool {
        let mut state = self.state.lock();
        match state.readers.admit(generation) {
            Admit::Stop => false,
            Admit::Drop => true,
            Admit::Deliver(identity) => {
                state.queue.as_mut().is_some_and(|queue| deliver(queue, identity))
            }
        }
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

    async fn recv(&self) -> Result<Option<Q::Frame>> {
        let mut queue = self.queue.lock().await;
        Q::recv(&mut queue, |identity| self.state.lock().readers.accepts(identity)).await
    }

    /// Readers keep retiring their streams after the waiter reaches the deadline.
    async fn close(&self, clock: &dyn Clock, deadline: Ts) -> Closed {
        {
            let mut state = self.state.lock();
            state.readers.close();
            state.queue.take();
        }
        self.changed.notify_waiters();
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
    type Queue: Queue;

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
async fn pump<S: Send + 'static, Q: Queue, A: Future<Output = Option<Q::Frame>>>(
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
    readers: Readers<VideoQueue>,
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
            readers: Readers::new(VIDEO_SOURCES),
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
    type Queue = VideoQueue;

    fn readers(&self) -> &Readers<VideoQueue> {
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
        Box::pin(async move { self.readers.recv().await })
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
    readers: Readers<AudioQueue>,
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
            readers: Readers::new(AUDIO_SOURCES),
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
        if let Some(queue) = &self.readers.state.lock().queue {
            queue.tx.observe(cx, telemetry::AudioConsumer::RoomInput);
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
        telemetry::end_with(span, telemetry::WaitForAudioTrackEnd { first_frame_delay: delay });
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
            let duration: std::time::Duration =
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
                            && !self.readers.admitted(generation, |queue, id| queue.head(id, frame))
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
    type Queue = AudioQueue;

    fn readers(&self) -> &Readers<AudioQueue> {
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
        Box::pin(async move { self.readers.recv().await })
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
