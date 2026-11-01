//! Executes a participant input's track selection: opens the tracks voice selects, pumps their
//! frames into the bounded queue the session reads, and retires the streams voice retires.

use crate::room_io::bounded_lossy::{BoundedLossy, LossyTx, Pushed};
type PublicationSnapshot = (Vec<Publication>, Vec<Arc<dyn TrackPublication>>);

use lk_agents_abi::voice::io::{AudioInput, IoError, Result, VideoInput};
use lk_agents_abi::voice::room_io::AudioInputConfig;
use lk_agents_abi::voice::rtc::{
    AudioInputStream, AudioStreamItem, AudioStreamOptions, AudioTrackFeature, ParticipantIdentity,
    ProcessorFailure, RemoteParticipant, RoomEvent, RtcRoom, TrackPublication, TrackSid,
    VideoFrameRef,
};
use lk_agents_abi::{AudioFrame, Clock, Closed, HOST_TEARDOWN_TIMEOUT, Spawner, StreamOutput, Ts};
use lk_agents_net::task::TaskSet;
use lk_agents_voice::room_io::input_state::{
    AUDIO_SOURCES, Admit, InputIdentity, Open, Owner, ProcessorSpans, Publication, TrackReaders,
    VIDEO_SOURCES, end_of_track, replayed_head,
};
use parking_lot::Mutex;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Notify;

use super::pre_connect_audio::PreConnectAudioHandler;

struct Delivery<T> {
    identity: InputIdentity,
    item: Result<T>,
}

struct State<T> {
    readers: TrackReaders,
    queue: Option<LossyTx<Delivery<T>>>,
}

/// The selection, the queue it admits frames into, and one wake for every reader.
struct Readers<T> {
    state: Mutex<State<T>>,
    queue: tokio::sync::Mutex<BoundedLossy<Delivery<T>>>,
    changed: Notify,
}

impl<T> Readers<T> {
    fn new(sources: &'static [lk_agents_abi::voice::rtc::TrackSource], capacity: usize) -> Self {
        let (queue, rx) = BoundedLossy::channel(capacity);
        Self {
            state: Mutex::new(State { readers: TrackReaders::new(sources), queue: Some(queue) }),
            queue: tokio::sync::Mutex::new(rx),
            changed: Notify::new(),
        }
    }

    /// Applies a decision and wakes the readers it may have retired.
    fn decide<R>(&self, decision: impl FnOnce(&mut TrackReaders) -> R) -> R {
        let outcome = decision(&mut self.state.lock().readers);
        self.changed.notify_waiters();
        outcome
    }

    fn reading(&self) -> Option<TrackSid> {
        self.state.lock().readers.reading().cloned()
    }

    /// `false` stops the reader.
    fn deliver(&self, generation: u64, item: Result<T>) -> bool {
        let state = self.state.lock();
        match state.readers.admit(generation) {
            Admit::Stop => false,
            Admit::Drop => true,
            Admit::Deliver(identity) => state
                .queue
                .as_ref()
                .is_some_and(|queue| queue.push(Delivery { identity, item }) != Pushed::Closed),
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

    async fn recv(&self) -> Result<Option<T>> {
        let mut queue = self.queue.lock().await;
        while let Some(frame) = queue.recv().await {
            if self.state.lock().readers.accepts(frame.identity) {
                return frame.item.map(Some);
            }
        }
        Ok(None)
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
    type Frame: Send + 'static;

    fn readers(&self) -> &Readers<Self::Frame>;
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
async fn pump<S: Send + 'static, T: Send + 'static, A: Future<Output = Option<T>>>(
    readers: &Readers<T>,
    clock: &dyn Clock,
    generation: u64,
    mut stream: Box<dyn StreamOutput<S>>,
    replay: impl Future<Output = bool>,
    end: impl FnOnce() -> Option<T>,
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
    readers: Readers<VideoFrameRef>,
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
            readers: Readers::new(VIDEO_SOURCES, lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES),
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
    type Frame = VideoFrameRef;

    fn readers(&self) -> &Readers<VideoFrameRef> {
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
    fn next_frame<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<Option<VideoFrameRef>>> {
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
    readers: Readers<AudioFrame>,
    /// A track's pump, which ends once its generation is replaced or its stream ends.
    pumps: Arc<TaskSet>,
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
            readers: Readers::new(AUDIO_SOURCES, lk_agents_abi::AUDIO_QUEUE_FRAMES),
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

    /// A frame to deliver, or a processor failure reported when it opens that processor's span.
    async fn admit(
        &self,
        spans: &Mutex<ProcessorSpans>,
        item: AudioStreamItem,
    ) -> Option<AudioFrame> {
        match item {
            AudioStreamItem::Frame(frame) => {
                spans.lock().frame();
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
        for frame in frames {
            match input_stream.process_audio(vec![frame]).await {
                Ok(items) => {
                    for item in items {
                        if let Some(frame) = self.admit(spans, item).await
                            && !self.readers.deliver(generation, Ok(frame))
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
    type Frame = AudioFrame;

    fn readers(&self) -> &Readers<AudioFrame> {
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
            queue_capacity: lk_agents_abi::AUDIO_QUEUE_FRAMES,
        })?;
        let stream = input_stream.frames()?;
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
    fn next_frame<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<Option<AudioFrame>>> {
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
