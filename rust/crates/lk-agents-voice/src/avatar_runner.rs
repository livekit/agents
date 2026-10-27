//! `voice/avatar/_runner.py`'s `AvatarRunner`, the avatar worker's side: the agent's audio from an
//! `AudioReceiver` into the user's `VideoGenerator`, what it generates out through the SDK's
//! `rtc.AVSynchronizer` (publishing the tracks on the first frame), and each segment's playback
//! reported back.
//!
//! [`RunnerCore`] makes every decision as a state machine: `handle(event, now) -> effects`, no
//! timers of its own. The frames are the host's ([`MediaRef`]): the core decides where each one
//! goes, never what is in it. The I/O layer below it ([`AvatarIo::perform`] and the
//! [`AvatarRunner`] queue owner) runs each [`IoOp`] against the host proxies and answers it with
//! an [`Event`]; it decides nothing.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Clock, MediaRef, Spawner, SpawnerExt, TaskSet, Ts};
use lk_agents_rtc::{RoomEvent, RtcRoom};
use tokio::sync::{mpsc, oneshot};

const LOG: &str = "livekit.agents";

// ----------------------------------------------------------------------------- the host's side

/// What an avatar's audio receiver or video generator yields. The frames are the host's
/// (`rtc.AudioFrame`, `rtc.VideoFrame`): the runner never reads one.
#[derive(Debug, Clone, PartialEq)]
pub enum AvatarFrame {
    /// An audio frame, and how long it plays.
    Audio { frame: MediaRef, duration: f64 },
    /// A video frame, or anything else that is neither audio nor a segment end.
    Video(MediaRef),
    /// `AudioSegmentEnd`.
    SegmentEnd,
}

/// `AudioReceiver`: the agent's audio as the avatar worker receives it, and where its playback is
/// reported. A host `AudioReceiver` carries this surface (`voice/avatar.py`).
#[lk_agents_abi::host(proxy(cancel_on_drop))]
#[async_trait]
pub trait AvatarAudioReceiver: Send + Sync {
    /// Begin iterating (`__aiter__`).
    async fn open_audio(&self) -> Result<(), ProviderError>;
    /// The next frame or segment end; `None` once the receiver is exhausted.
    async fn next_audio(&self) -> Result<Option<AvatarFrame>, ProviderError>;
    async fn notify_playback_started(&self) -> Result<(), ProviderError>;
    async fn notify_playback_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
    ) -> Result<(), ProviderError>;
    async fn aclose(&self) -> Result<(), ProviderError>;
}

/// `VideoGenerator`: the avatar's model, fed the audio and yielding the frames to publish. A host
/// `VideoGenerator` carries this surface (`voice/avatar.py`).
#[lk_agents_abi::host(proxy(cancel_on_drop))]
#[async_trait]
pub trait AvatarVideoGenerator: Send + Sync {
    /// An audio frame, or `None` for `AudioSegmentEnd`.
    async fn push_audio(&self, frame: Option<MediaRef>) -> Result<(), ProviderError>;
    async fn clear_buffer(&self) -> Result<(), ProviderError>;
    /// Begin iterating (`__aiter__`).
    async fn open_frames(&self) -> Result<(), ProviderError>;
    /// The next frame or segment end; `None` once the generator is exhausted.
    async fn next_frame(&self) -> Result<Option<AvatarFrame>, ProviderError>;
}

/// The SDK's half: the avatar's tracks in the room and the `rtc.AVSynchronizer` pacing what is
/// pushed to them. Python and Node each have their own.
#[lk_agents_abi::host(proxy(cancel_on_drop))]
#[async_trait]
pub trait AvatarMedia: Send + Sync {
    /// Publish the audio track, wait for its subscription, then publish the video track.
    async fn publish(&self) -> Result<(), ProviderError>;
    /// One frame into the synchronizer, which paces it out.
    async fn push(&self, frame: MediaRef) -> Result<(), ProviderError>;
    async fn aclose(&self) -> Result<(), ProviderError>;
}

// ----------------------------------------------------------------------------- the core

/// A host call into the runner, answered by [`Effect::Answer`] with the same id.
pub type CallId = u64;

/// Why the runner refused a call, or what its loops ended with (`wait_for_complete`).
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RunnerError {
    #[error("AvatarRunner not started")]
    NotStarted,
    #[host(kind = "asyncio.CancelledError")]
    #[error("the avatar runner was closed")]
    Closed,
    /// What the receiver, the generator or the synchronizer raised.
    #[host(delegate)]
    #[error(transparent)]
    Failed(ProviderError),
}

/// What happened: a host call, the room, or the answer to an [`IoOp`].
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// `start()`, once the receiver has started; whether the room is connected now.
    Start {
        call: CallId,
        connected: bool,
    },
    WaitForComplete {
        call: CallId,
    },
    /// The receiver's `clear_buffer` event.
    ClearBuffer,
    /// `aclose()`.
    Close {
        call: CallId,
    },
    /// The room's connection state changed.
    Connection {
        connected: bool,
    },
    AudioOpened(Result<(), ProviderError>),
    Audio(Result<Option<AvatarFrame>, ProviderError>),
    AudioPushed(Result<(), ProviderError>),
    FramesOpened(Result<(), ProviderError>),
    Frame(Result<Option<AvatarFrame>, ProviderError>),
    Published(Result<(), ProviderError>),
    Pushed(Result<(), ProviderError>),
    Reported(Result<(), ProviderError>),
    GeneratorCleared {
        clear: u64,
        result: Result<(), ProviderError>,
    },
    ReceiverClosed(Result<(), ProviderError>),
    /// Every [`IoOp`] in flight was cancelled ([`Effect::CancelIo`]).
    IoCancelled,
    MediaClosed(Result<(), ProviderError>),
}

/// What the runner asks of its receiver, generator and media.
#[derive(Debug, Clone, PartialEq)]
pub enum IoOp {
    OpenAudio,
    NextAudio,
    PushAudio(Option<MediaRef>),
    OpenFrames,
    NextFrame,
    Publish,
    Push(MediaRef),
    ReportStarted,
    ReportFinished { position: f64, interrupted: bool },
    ClearGenerator { clear: u64 },
    CloseReceiver,
    CloseMedia,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    /// Run this; its answer comes back as an [`Event`].
    Io(IoOp),
    /// Cancel every [`IoOp`] in flight and wait for them to be gone, then [`Event::IoCancelled`].
    CancelIo,
    Answer {
        call: CallId,
        result: Result<(), RunnerError>,
    },
}

/// Where one of the two loops (`_read_audio`, `_forward_video`) is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Pump {
    #[default]
    Idle,
    /// An op in flight; its answer moves the loop on.
    Busy,
    /// Waiting for the tracks to be published before pushing the held frame.
    Publishing,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Publish {
    Never,
    /// Asked, waiting for the room to connect.
    Waiting,
    InFlight,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Closing {
    /// The receiver is closing; the loops still run, as the reference's do.
    Receiver,
    Cancelling,
    Media,
}

/// The runner's decisions, one event at a time.
#[derive(Debug)]
pub struct RunnerCore {
    lazy_publish: bool,
    /// `_room_connected_fut`.
    connected: bool,
    publish: Publish,
    /// The `start()` waiting on a publish.
    start_call: Option<CallId>,
    started: bool,
    read: Pump,
    forward: Pump,
    /// The frame the forward loop holds while the tracks publish, and its duration if audio.
    held: Option<(MediaRef, Option<f64>)>,
    /// The audio frame being pushed, and its duration: reported once the push returns.
    pushing: Option<f64>,
    /// `_playback_position`, `_audio_playing`.
    position: f64,
    playing: bool,
    /// Each clear in flight, and whether audio was playing when it arrived.
    clears: BTreeMap<u64, bool>,
    next_clear: u64,
    /// How the loops ended, as `asyncio.gather` sees them: the first failure, else both done.
    finished: u8,
    failed: Option<RunnerError>,
    waiting: Vec<CallId>,
    closing: Option<(Closing, CallId)>,
    closed: bool,
}

impl RunnerCore {
    pub fn new(lazy_publish: bool) -> Self {
        RunnerCore {
            lazy_publish,
            connected: false,
            publish: Publish::Never,
            start_call: None,
            started: false,
            read: Pump::Idle,
            forward: Pump::Idle,
            held: None,
            pushing: None,
            position: 0.0,
            playing: false,
            clears: BTreeMap::new(),
            next_clear: 0,
            finished: 0,
            failed: None,
            waiting: Vec::new(),
            closing: None,
            closed: false,
        }
    }

    /// The runner has no timers.
    pub fn wake_at(&self) -> Option<Ts> {
        None
    }

    /// Whether the loops are being cut: no answer moves them on any more.
    fn cut(&self) -> bool {
        self.closed || matches!(self.closing, Some((Closing::Cancelling | Closing::Media, _)))
    }

    pub fn handle(&mut self, event: Event, _now: Ts) -> Vec<Effect> {
        let mut out = Vec::new();
        match event {
            Event::Start { call, connected } => {
                self.connected |= connected;
                if self.lazy_publish {
                    self.begin(&mut out);
                    out.push(Effect::Answer { call, result: Ok(()) });
                } else {
                    self.start_call = Some(call);
                    self.ask_publish(&mut out);
                }
            }
            Event::WaitForComplete { call } => {
                self.waiting.push(call);
                self.answer_waiting(&mut out);
            }
            Event::Connection { connected } => {
                if connected && !self.connected {
                    self.connected = true;
                    if self.publish == Publish::Waiting {
                        self.publish = Publish::InFlight;
                        out.push(Effect::Io(IoOp::Publish));
                    }
                }
            }
            Event::ClearBuffer if !self.cut() => {
                let clear = self.next_clear;
                self.next_clear += 1;
                self.clears.insert(clear, std::mem::take(&mut self.playing));
                out.push(Effect::Io(IoOp::ClearGenerator { clear }));
            }
            Event::GeneratorCleared { clear, result } if !self.cut() => {
                let playing = self.clears.remove(&clear).unwrap_or(false);
                match result {
                    Ok(()) if playing => {
                        let position = std::mem::take(&mut self.position);
                        out.push(Effect::Io(IoOp::ReportFinished { position, interrupted: true }));
                    }
                    Ok(()) => {}
                    Err(e) => log_error("_handle_clear_buffer", &e),
                }
            }
            Event::AudioOpened(result) if !self.cut() => match result {
                Ok(()) => out.push(Effect::Io(IoOp::NextAudio)),
                Err(e) => self.loop_ended(true, Err(e), &mut out),
            },
            Event::Audio(result) if !self.cut() => match result {
                Ok(Some(item)) => {
                    let frame = match item {
                        AvatarFrame::Audio { frame, .. } => {
                            self.playing = true;
                            Some(frame)
                        }
                        AvatarFrame::Video(frame) => Some(frame),
                        AvatarFrame::SegmentEnd => None,
                    };
                    out.push(Effect::Io(IoOp::PushAudio(frame)));
                }
                Ok(None) => self.loop_ended(true, Ok(()), &mut out),
                Err(e) => self.loop_ended(true, Err(e), &mut out),
            },
            Event::AudioPushed(result) if !self.cut() => match result {
                Ok(()) => out.push(Effect::Io(IoOp::NextAudio)),
                Err(e) => self.loop_ended(true, Err(e), &mut out),
            },
            Event::FramesOpened(result) if !self.cut() => match result {
                Ok(()) => out.push(Effect::Io(IoOp::NextFrame)),
                Err(e) => self.loop_ended(false, Err(e), &mut out),
            },
            Event::Frame(result) if !self.cut() => match result {
                Ok(Some(AvatarFrame::SegmentEnd)) => {
                    // the agent hears the segment finished; the forwarding goes on
                    if std::mem::take(&mut self.playing) {
                        let position = std::mem::take(&mut self.position);
                        out.push(Effect::Io(IoOp::ReportFinished { position, interrupted: false }));
                    }
                    out.push(Effect::Io(IoOp::NextFrame));
                }
                Ok(Some(AvatarFrame::Audio { frame, duration })) => {
                    self.forward(frame, Some(duration), &mut out)
                }
                Ok(Some(AvatarFrame::Video(frame))) => self.forward(frame, None, &mut out),
                Ok(None) => self.loop_ended(false, Ok(()), &mut out),
                Err(e) => self.loop_ended(false, Err(e), &mut out),
            },
            Event::Published(result) if !self.cut() => {
                self.publish = if result.is_ok() { Publish::Done } else { Publish::Never };
                if let Some(call) = self.start_call.take() {
                    match result {
                        Ok(()) => {
                            self.begin(&mut out);
                            out.push(Effect::Answer { call, result: Ok(()) });
                        }
                        Err(e) => {
                            out.push(Effect::Answer { call, result: Err(RunnerError::Failed(e)) })
                        }
                    }
                } else if self.forward == Pump::Publishing {
                    match result {
                        Ok(()) => {
                            if let Some((frame, duration)) = self.held.take() {
                                self.push(frame, duration, &mut out);
                            }
                        }
                        Err(e) => {
                            self.held = None;
                            self.loop_ended(false, Err(e), &mut out);
                        }
                    }
                }
            }
            Event::Pushed(result) if !self.cut() => match result {
                Ok(()) => {
                    if let Some(duration) = self.pushing.take() {
                        let started = self.position == 0.0 && duration > 0.0;
                        self.position += duration;
                        if started {
                            out.push(Effect::Io(IoOp::ReportStarted));
                        }
                    }
                    out.push(Effect::Io(IoOp::NextFrame));
                }
                Err(e) => {
                    self.pushing = None;
                    self.loop_ended(false, Err(e), &mut out);
                }
            },
            Event::Reported(Err(e)) => log_error("the playback report", &e),
            Event::Close { call } => match self.closing {
                _ if self.closed => out.push(Effect::Answer { call, result: Ok(()) }),
                // a second close while one is under way waits for nothing more: the first answers
                Some(_) => out.push(Effect::Answer { call, result: Ok(()) }),
                None => {
                    self.closing = Some((Closing::Receiver, call));
                    out.push(Effect::Io(IoOp::CloseReceiver));
                }
            },
            Event::ReceiverClosed(result) => {
                if let Some((Closing::Receiver, call)) = self.closing {
                    match result {
                        Ok(()) => {
                            self.closing = Some((Closing::Cancelling, call));
                            out.push(Effect::CancelIo);
                        }
                        // as the reference's `aclose`, which raises before cancelling anything
                        Err(e) => {
                            self.closing = None;
                            out.push(Effect::Answer { call, result: Err(RunnerError::Failed(e)) });
                        }
                    }
                }
            }
            Event::IoCancelled => {
                if let Some((Closing::Cancelling, call)) = self.closing {
                    self.held = None;
                    self.pushing = None;
                    self.clears.clear();
                    for pump in [&mut self.read, &mut self.forward] {
                        if *pump != Pump::Done {
                            *pump = Pump::Done;
                            self.failed.get_or_insert(RunnerError::Closed);
                        }
                    }
                    if !self.started {
                        self.failed.get_or_insert(RunnerError::Closed);
                    }
                    self.answer_waiting(&mut out);
                    self.closing = Some((Closing::Media, call));
                    out.push(Effect::Io(IoOp::CloseMedia));
                }
            }
            Event::MediaClosed(result) => {
                if let Some((Closing::Media, call)) = self.closing.take() {
                    self.closed = true;
                    out.push(Effect::Answer { call, result: result.map_err(RunnerError::Failed) });
                }
            }
            // an answer to an op the close cut, or a report that went through
            _ => {}
        }
        out
    }

    /// Start both loops.
    fn begin(&mut self, out: &mut Vec<Effect>) {
        self.started = true;
        self.read = Pump::Busy;
        self.forward = Pump::Busy;
        out.push(Effect::Io(IoOp::OpenAudio));
        out.push(Effect::Io(IoOp::OpenFrames));
        self.answer_waiting(out);
    }

    /// `_publish_track`: once the room has connected.
    fn ask_publish(&mut self, out: &mut Vec<Effect>) {
        if self.connected {
            self.publish = Publish::InFlight;
            out.push(Effect::Io(IoOp::Publish));
        } else {
            self.publish = Publish::Waiting;
        }
    }

    /// A generated frame: publish the tracks first if they are not up.
    fn forward(&mut self, frame: MediaRef, duration: Option<f64>, out: &mut Vec<Effect>) {
        if self.publish == Publish::Done {
            self.push(frame, duration, out);
            return;
        }
        self.forward = Pump::Publishing;
        self.held = Some((frame, duration));
        if self.publish == Publish::Never {
            self.ask_publish(out);
        }
    }

    fn push(&mut self, frame: MediaRef, duration: Option<f64>, out: &mut Vec<Effect>) {
        self.forward = Pump::Busy;
        self.pushing = duration;
        out.push(Effect::Io(IoOp::Push(frame)));
    }

    /// One loop's end: `read` or forward.
    fn loop_ended(&mut self, read: bool, result: Result<(), ProviderError>, out: &mut Vec<Effect>) {
        let (pump, name) = if read {
            (&mut self.read, "_read_audio")
        } else {
            (&mut self.forward, "_forward_video")
        };
        *pump = Pump::Done;
        match result {
            Ok(()) => self.finished += 1,
            Err(e) => {
                log_error(name, &e);
                self.failed.get_or_insert(RunnerError::Failed(e));
            }
        }
        self.answer_waiting(out);
    }

    /// `wait_for_complete`'s answers, once the outcome is known.
    fn answer_waiting(&mut self, out: &mut Vec<Effect>) {
        let result = match &self.failed {
            Some(e) => Err(e.clone()),
            None if !self.started && self.start_call.is_none() => Err(RunnerError::NotStarted),
            None if self.finished == 2 => Ok(()),
            None => return,
        };
        for call in self.waiting.drain(..) {
            out.push(Effect::Answer { call, result: result.clone() });
        }
    }
}

fn log_error(what: &str, e: &ProviderError) {
    log::error!(target: LOG, error = e.to_string().as_str(); "Error in {what}");
}

// ----------------------------------------------------------------------------- the I/O layer

/// The host objects the runner's ops run against.
#[derive(Clone)]
pub struct AvatarIo {
    pub receiver: Arc<dyn AvatarAudioReceiver>,
    pub generator: Arc<dyn AvatarVideoGenerator>,
    pub media: Arc<dyn AvatarMedia>,
}

impl AvatarIo {
    /// Run one op; its answer.
    pub async fn perform(&self, op: IoOp) -> Event {
        match op {
            IoOp::OpenAudio => Event::AudioOpened(self.receiver.open_audio().await),
            IoOp::NextAudio => Event::Audio(self.receiver.next_audio().await),
            IoOp::PushAudio(frame) => Event::AudioPushed(self.generator.push_audio(frame).await),
            IoOp::OpenFrames => Event::FramesOpened(self.generator.open_frames().await),
            IoOp::NextFrame => Event::Frame(self.generator.next_frame().await),
            IoOp::Publish => Event::Published(self.media.publish().await),
            IoOp::Push(frame) => Event::Pushed(self.media.push(frame).await),
            IoOp::ReportStarted => Event::Reported(self.receiver.notify_playback_started().await),
            IoOp::ReportFinished { position, interrupted } => {
                Event::Reported(self.receiver.notify_playback_finished(position, interrupted).await)
            }
            IoOp::ClearGenerator { clear } => {
                Event::GeneratorCleared { clear, result: self.generator.clear_buffer().await }
            }
            IoOp::CloseReceiver => Event::ReceiverClosed(self.receiver.aclose().await),
            IoOp::CloseMedia => Event::MediaClosed(self.media.aclose().await),
        }
    }
}

type Answers = Arc<Mutex<HashMap<CallId, oneshot::Sender<Result<(), RunnerError>>>>>;

/// `AvatarRunner`: the queue owner of one [`RunnerCore`]. Host calls and the room become events,
/// each effect runs as a task whose answer is the next event.
pub struct AvatarRunner {
    room: Arc<dyn RtcRoom>,
    events: mpsc::UnboundedSender<Event>,
    answers: Answers,
    next_call: std::sync::atomic::AtomicU64,
    spawner: Arc<dyn Spawner>,
    io_tasks: Arc<TaskSet>,
    owner: lk_agents_abi::AbortHandle,
}

impl Drop for AvatarRunner {
    fn drop(&mut self) {
        self.owner.abort();
        self.io_tasks.abort_all();
    }
}

impl AvatarRunner {
    /// The runner, its ready events taken in the order `order(len)` picks (an index below
    /// `len`): arrival order in production, the simulation's own choice under test.
    pub fn scheduled(
        room: Arc<dyn RtcRoom>,
        io: AvatarIo,
        lazy_publish: bool,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        mut order: impl FnMut(usize) -> usize + Send + 'static,
    ) -> Self {
        let (events, mut queue) = mpsc::unbounded_channel();
        let answers: Answers = Arc::default();
        let io_tasks: Arc<TaskSet> = Arc::default();
        let owner = {
            let (events, answers, io_tasks, spawner) =
                (events.clone(), Arc::clone(&answers), Arc::clone(&io_tasks), Arc::clone(&spawner));
            spawner.clone().spawn_unboxed(async move {
                let mut core = RunnerCore::new(lazy_publish);
                let mut ready = std::collections::VecDeque::new();
                loop {
                    if ready.is_empty() {
                        match queue.recv().await {
                            Some(event) => ready.push_back(event),
                            None => return,
                        }
                    }
                    while let Ok(event) = queue.try_recv() {
                        ready.push_back(event);
                    }
                    let pick = order(ready.len()).min(ready.len() - 1);
                    let Some(event) = ready.remove(pick) else { continue };
                    for effect in core.handle(event, clock.now()) {
                        match effect {
                            Effect::Io(op) => {
                                let (io, events) = (io.clone(), events.clone());
                                io_tasks.spawn(&*spawner, "avatar runner op", async move {
                                    let _ = events.send(io.perform(op).await);
                                });
                            }
                            Effect::CancelIo => {
                                let (io_tasks, events, clock) =
                                    (Arc::clone(&io_tasks), events.clone(), Arc::clone(&clock));
                                spawner.spawn_unboxed(async move {
                                    io_tasks.cancel_and_wait(&*clock).await;
                                    let _ = events.send(Event::IoCancelled);
                                });
                            }
                            Effect::Answer { call, result } => {
                                let tx =
                                    answers.lock().unwrap_or_else(|p| p.into_inner()).remove(&call);
                                if let Some(tx) = tx {
                                    let _ = tx.send(result);
                                }
                            }
                        }
                    }
                }
            })
        };
        AvatarRunner {
            room,
            events,
            answers,
            next_call: Default::default(),
            spawner,
            io_tasks,
            owner,
        }
    }

    /// Whether an op, a report or the room watch is still running: none once closed.
    pub fn io_in_flight(&self) -> bool {
        !self.io_tasks.is_empty()
    }

    /// Hand the core one host call and wait for its answer.
    async fn call(&self, event: impl FnOnce(CallId) -> Event) -> Result<(), RunnerError> {
        let call = self.next_call.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.answers.lock().unwrap_or_else(|p| p.into_inner()).insert(call, tx);
        let _ = self.events.send(event(call));
        rx.await.unwrap_or(Err(RunnerError::Closed))
    }
}

#[lk_agents_abi::host(object(cell = "arc"))]
impl AvatarRunner {
    /// `room` is the avatar worker's `rtc.Room`; `media` the SDK half (`rtc.AVSynchronizer` and
    /// the tracks); `lazy_publish` holds the tracks back until the first frame.
    #[host(new(kw_only = "receiver"))]
    pub fn new(
        room: Arc<dyn RtcRoom>,
        receiver: Arc<dyn AvatarAudioReceiver>,
        generator: Arc<dyn AvatarVideoGenerator>,
        media: Arc<dyn AvatarMedia>,
        lazy_publish: bool,
        #[host(ambient)] clock: Arc<dyn Clock>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let io = AvatarIo { receiver, generator, media };
        AvatarRunner::scheduled(room, io, lazy_publish, clock, spawner, |_| 0)
    }

    /// `start()`, once the receiver has started (`await audio_recv.start()`).
    #[host(method)]
    pub async fn start(&self) -> Result<(), RunnerError> {
        // subscribed before the state is read, so no change falls between the two
        let mut room_events = self.room.events();
        let connected = self.room.connection_state().is_connected();
        let events = self.events.clone();
        self.io_tasks.spawn(&*self.spawner, "avatar room connection", async move {
            while let Some(ev) = room_events.recv().await {
                if let RoomEvent::ConnectionStateChanged { state } = ev {
                    let _ = events.send(Event::Connection { connected: state.is_connected() });
                }
            }
        });
        self.call(|call| Event::Start { call, connected }).await
    }

    /// Both loops ran out; the first failure as soon as one fails; `CancelledError` once closed.
    #[host(method)]
    pub async fn wait_for_complete(&self) -> Result<(), RunnerError> {
        self.call(|call| Event::WaitForComplete { call }).await
    }

    /// The receiver's `clear_buffer` event.
    #[host(method)]
    pub fn clear_buffer(&self) {
        let _ = self.events.send(Event::ClearBuffer);
    }

    /// `aclose()`: the receiver, then the loops and what is in flight, then the media.
    #[host(method)]
    pub async fn aclose(&self) -> Result<(), RunnerError> {
        self.call(|call| Event::Close { call }).await
    }
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use super::AvatarFrame;

    /// What the receiver and the generator yield: an `AudioSegmentEnd`, an `rtc.AudioFrame`, or
    /// anything else, which goes on as video (`_forward_video`'s `isinstance` tests).
    impl<'a, 'py> FromPyObject<'a, 'py> for AvatarFrame {
        type Error = PyErr;

        fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py = obj.py();
            let end = py.import("livekit.agents.voice.avatar")?.getattr("AudioSegmentEnd")?;
            if obj.is_instance(&end)? {
                return Ok(AvatarFrame::SegmentEnd);
            }
            let frame = obj.extract::<lk_agents_abi::MediaRef>()?;
            if obj.is_instance(&py.import("livekit.rtc")?.getattr("AudioFrame")?)? {
                let duration = obj.getattr("duration")?.extract::<f64>()?;
                return Ok(AvatarFrame::Audio { frame, duration });
            }
            Ok(AvatarFrame::Video(frame))
        }
    }
}

#[cfg(test)]
#[path = "avatar_runner_tests.rs"]
mod tests;
