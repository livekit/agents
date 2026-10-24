//! `voice/agent_session.py`'s `AgentSession` as its host holds it: [`SessionApi`], assembled from
//! the host's arguments, opened at `start()`, commanded, and closed; with the host-side traits and
//! errors it speaks through. Its host methods are in `host`.

use crate::agent_activity::options::{
    RECORDING_ALL_OFF, RecordingRequest, records_anything, resolve_recording_options,
};
use crate::agent_session::builder::DriverBuilder;
use crate::agent_session::calls::Calls;
use crate::agent_session::commands::{
    AgentUpdate, Answer, Close, Command, HostRecorder, NewActivity, Origin, PreviousActivity,
    RecorderCommand, RequestId, SayText, SpeechCommand,
};
use crate::agent_session::driver::{DriverError, Peer, Rx, SessionHandle, Tx, start_ivr};
use crate::agent_session::host_events::HostEvent;
use crate::agent_session::time::Timebase;
use crate::events::SessionEvent;
use crate::generation::Instructions;
use crate::tool_executor::Toolset;
use crate::trace::{CoreStep, Step, Trace};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::HostWire;
use lk_agents_abi::voice::session_effects::StartPlan;
use lk_agents_abi::{Clock, Llm, Maybe, Spawner, SpawnerExt, Stt, Tts, Vad};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// `voice/agent_session.py`'s `AgentSession`, over the driver: assembled at construction, opened at
/// `start()`, closed at `aclose()`; every other method is one command. `frozen`, so there is no
/// borrow flag: `aclose()` can be awaited while the event drain calls back into the same object.
///
/// The host's end of a session: assembled by [`host_new`](Self::host_new), opened once, commanded through
/// [`send`](Self::send), closed through [`close`](Self::close). Every method takes `&self`; one
/// `Mutex` around a small enum, off every per-frame path.
#[lk_agents_abi::class(name = "AgentSession")]
pub struct SessionApi {
    /// `Arc`, so `open`'s future can write the running loop back after the borrow ended.
    stage: Arc<Mutex<Stage>>,
    /// The driver's legs, kept in step with every provider swap ([`swap`](Self::swap)).
    legs: Mutex<crate::agent_activity::options::LegSet>,
    /// The driver's option stack, kept in step with every `Configure` and frame update a host
    /// sends, so `session.options` resolves the instant a change is sent.
    stack: Mutex<crate::agent_activity::options::ConfigStack>,
    room_io: Arc<crate::room_io::RoomIo>,
    /// The transcript synchroniser, held only so `close_io` can close it.
    synchronizer: Option<Arc<crate::transcription::synchronizer::TranscriptSynchronizer>>,
    spawner: Arc<dyn Spawner>,
    /// The driver's clock: `close` on a virtual clock moves it.
    time: Timebase,
    /// The read-back guard of the last DTMF collection, read by
    /// [`dtmf_reply_running`](Self::dtmf_reply_running).
    dtmf_reply: Mutex<Option<Arc<std::sync::atomic::AtomicBool>>>,
    /// The monotonic-to-wall pair every `Ts`-carrying event converts against, read once so events
    /// never disagree about when `Ts::ZERO` was.
    anchor: lk_agents_abi::WallAnchor,
    /// What only a host-built session carries ([`host_new`](Self::host_new)).
    host: HostParts,
}

impl SessionApi {
    pub fn time(&self) -> &Timebase {
        &self.time
    }

    pub fn legs(&self) -> crate::agent_activity::options::LegSet {
        *self.legs.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn stack(&self) -> crate::agent_activity::options::ConfigStack {
        *self.stack.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Merged here too, as the driver does.
    pub fn configure(&self, patch: &crate::agent_activity::options::SessionPatch) {
        self.stack.lock().unwrap_or_else(|p| p.into_inner()).session.merge(patch);
    }

    /// Its `turn_handling` becomes the agent layer, replaced whole.
    pub fn set_agent_layer(&self, agent: Option<crate::agent_activity::options::SessionPatch>) {
        self.stack.lock().unwrap_or_else(|p| p.into_inner()).agent = agent;
    }

    /// Refuses synchronously when there is nothing to open; does not wait for a participant.
    /// `drain` is spawned beside the driver so `close` can abort both.
    pub fn open(
        &self,
        capture_run: bool,
        drain: impl FnOnce(SessionRunner) -> lk_agents_abi::BoxFuture<'static, ()> + Send + 'static,
    ) -> Result<
        impl std::future::Future<Output = Result<(), DriverError>> + Send + 'static,
        OpenRefused,
    > {
        // Its own statement: an inline `MutexGuard` would live through the `match`, and the
        // already-started arm's relock would deadlock.
        let current = std::mem::replace(&mut *self.lock(), Stage::Closed);
        let ready = match current {
            Stage::Ready(ready) => *ready,
            Stage::Running(r) => {
                *self.lock() = Stage::Running(r);
                return Err(OpenRefused::AlreadyStarted);
            }
            Stage::Closed => return Err(OpenRefused::Closed),
        };

        let room_io = Arc::clone(&self.room_io);
        let spawner = Arc::clone(&self.spawner);
        let stage = Arc::clone(&self.stage);
        let anchor = self.anchor;
        let time = self.time.clone();
        Ok(async move {
            let Ready { mut builder, interruption } = ready;
            if capture_run {
                builder.capture_run = Some(crate::run_result::RunSpec {
                    user_input: None,
                    expects_output: false,
                    max_retries: 0,
                    retry_instructions: String::new(),
                });
            }
            if let Some(open_port) = interruption {
                builder.interruption = Some(open_port(anchor));
            }

            // The room's own IO first; its events join the driver's queue, one stream, one order.
            room_io.start();
            builder.room_io_events = room_io.events();

            let (mut driver, handle) = builder.start().await?;
            let SessionHandle { api, events, answers, trace, peer, ivr_detection } = handle;
            // The IVR navigator's task, subscribed before the opening cycle.
            start_ivr(&peer, ivr_detection);
            // The opening cycle: `open` resolves only once the drain took its events, so a host's
            // `start()` sees `listening` dispatched first.
            driver.open(time.read());
            let (opening_tx, opening_rx) = tokio::sync::oneshot::channel();
            let opening = Opening { left: events.len(), delivered: Some(opening_tx) };

            let mut tasks = Vec::new();
            tasks.push(spawner.spawn_unboxed(async move { driver.run().await }));
            let calls = Calls::default();
            tasks.push(spawner.spawn_unboxed(calls.clone().pump(answers)));
            // Drained whether or not anybody listens: the queue is unbounded.
            let (drained_tx, drained) = tokio::sync::oneshot::channel();
            let runner = SessionRunner {
                events,
                room_io: Arc::clone(&room_io),
                api: api.clone(),
                drained: Some(drained_tx),
                opening,
            };
            tasks.push(spawner.spawn(drain(runner)));

            *stage.lock().unwrap_or_else(|p| p.into_inner()) = Stage::Running(Running {
                api,
                calls,
                trace,
                peer,
                amd: None,
                tasks,
                drained: Some(drained),
            });
            // A drain that returned without reading drops the sender: nothing to wait for.
            let _ = opening_rx.await;
            Ok(())
        })
    }

    /// `None` detaches. Safe before or after `open`.
    pub fn attach_recorder(
        &self,
        recorder: Option<Arc<dyn lk_agents_media::recorder::AudioRecorder>>,
    ) {
        match &mut *self.lock() {
            Stage::Ready(ready) => ready.builder.recorder = recorder,
            Stage::Running(r) => {
                let cmd = match recorder {
                    Some(tap) => RecorderCommand::Attach(HostRecorder(tap)),
                    None => RecorderCommand::Detach,
                };
                let _ = r.api.send(Command::Recorder(cmd));
            }
            Stage::Closed => {}
        }
    }

    /// `false`: the loop did not take it (not started or closed), and the image is dropped.
    pub fn push_video(&self, image: lk_agents_abi::ImageContent) -> bool {
        match &*self.lock() {
            Stage::Running(r) => r.api.send(Command::Video(image)).is_ok(),
            _ => false,
        }
    }

    /// Refused before `start()` and after the close.
    fn running<R>(
        &self,
        f: impl FnOnce(&mut Running) -> Result<R, SessionRefused>,
    ) -> Result<R, SessionRefused> {
        match &mut *self.lock() {
            Stage::Running(r) => f(r),
            Stage::Ready(_) => Err(SessionRefused::NotStarted),
            Stage::Closed => Err(SessionRefused::Closed),
        }
    }

    pub fn api(&self) -> Result<Tx<Command>, SessionRefused> {
        self.running(|r| Ok(r.api.clone()))
    }

    /// A call to await: its id goes in the command, `pick` takes its answer.
    pub fn call<T: Send + 'static>(
        &self,
        pick: fn(Answer) -> Option<T>,
    ) -> Result<(RequestId, tokio::sync::oneshot::Receiver<T>), SessionRefused> {
        self.running(|r| Ok(r.calls.open(pick)))
    }

    pub fn send(&self, cmd: impl Into<Command>) -> Result<(), SessionRefused> {
        let cmd = cmd.into();
        self.running(|r| r.api.send(cmd).map_err(|_| SessionRefused::LoopEnded))
    }

    pub fn send_from(&self, origin: Origin, cmd: impl Into<Command>) -> Result<(), SessionRefused> {
        let cmd = cmd.into();
        self.running(|r| r.api.send_from(origin, cmd).map_err(|_| SessionRefused::LoopEnded))
    }

    /// `AMD._run`. One per session; the verdict comes back as `amd_result`.
    pub fn attach_amd(&self, setup: crate::amd::detector::AmdSetup) -> Result<(), SessionRefused> {
        self.running(|r| {
            if r.amd.is_some() {
                r.peer.journal(Step::Core(CoreStep::PortFailure("amd: already attached")));
            } else {
                r.amd = Some(crate::amd::detector::AmdSystem::spawn(r.peer.clone(), setup));
            }
            Ok(())
        })
    }

    /// None attached or already closed: nothing raised.
    fn amd_input(&self, input: crate::amd::classifier::In) -> Result<(), SessionRefused> {
        self.running(|r| {
            if let Some(handle) = &r.amd {
                let _ = handle.input(input);
            }
            Ok(())
        })
    }

    /// Cloned: the loop is still writing. Empty until the loop runs.
    pub fn steps(&self) -> Vec<Step> {
        match &*self.lock() {
            Stage::Running(r) => r.trace.lock().unwrap_or_else(|p| p.into_inner()).steps().to_vec(),
            _ => Vec::new(),
        }
    }

    /// `_aclose_impl`: resolves once the loop returned and the drain handed over the last event.
    /// The stage stays `Running` meanwhile (hooks and tools still answer through it). Idempotent.
    pub fn close(
        &self,
        reason: lk_agents_abi::voice::events::CloseReason,
        drain: bool,
        error: Option<ProviderError>,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        let running = match &mut *self.lock() {
            Stage::Running(r) => Some((r.api.clone(), r.drained.take(), r.tasks.clone())),
            Stage::Ready(_) | Stage::Closed => None,
        };
        let stage = Arc::clone(&self.stage);
        let time = self.time.clone();
        async move {
            let Some((api, drained, tasks)) = running else { return };
            let _ = api.send(Command::Close(Close { reason, drain, error }));
            // Every close phase is bounded on the session clock; this watchdog is a backstop. On a
            // virtual clock the host stopped advancing, so the clock is moved here.
            if time.is_virtual() {
                time.advance_until(api.closed(), CLOSE_TICK, CLOSE_TICK_BUDGET).await;
            } else {
                let _ = time.timeout(CLOSE_WATCHDOG, api.closed()).await;
            }
            // `Closed` **before** awaiting the drain: `Running` holds the `Peer`, a sender on the
            // host stream, and the drain only ends when every sender is gone.
            *stage.lock().unwrap_or_else(|p| p.into_inner()) = Stage::Closed;
            // Then the drain: the host's `close` is queued before this future completes.
            if let Some(drained) = drained {
                let _ = time.timeout(CLOSE_TIMEOUT, drained).await;
            }
            for task in tasks {
                task.abort();
            }
        }
    }

    /// After the host's `close` handlers ran. The synchroniser first, or its rotation tasks publish
    /// into a closed stream.
    pub fn close_io(&self) -> impl std::future::Future<Output = ()> + Send + 'static {
        let room_io = Arc::clone(&self.room_io);
        let synchronizer = self.synchronizer.clone();
        let time = self.time.clone();
        async move {
            if let Some(sync) = &synchronizer {
                let _ = time.timeout(CLOSE_TIMEOUT, sync.aclose()).await;
            }
            let _ = time.timeout(CLOSE_TIMEOUT, room_io.aclose()).await;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Stage> {
        self.stage.lock().unwrap_or_else(|p| p.into_inner())
    }
}

type Leg<T> = Maybe<Option<Arc<T>>>;

impl SessionApi {
    /// Given present, given `None` clears, not-given keeps. The speaker filter follows the STT: the
    /// options given, else none for a swapped STT (only a `MultiSpeakerAdapter` brings some), kept:
    /// `None`.
    #[allow(clippy::type_complexity)]
    fn swap(
        &self,
        llm: Leg<dyn Llm>,
        tts: Leg<dyn Tts>,
        stt: Leg<dyn Stt>,
        vad: &Leg<dyn Vad>,
        speaker_filter: Maybe<Option<crate::audio::SpeakerFilterConfig>>,
    ) -> (Leg<dyn Llm>, Leg<dyn Tts>, Leg<dyn Stt>, Option<Option<crate::audio::SpeakerFilterConfig>>)
    {
        fn map<T: ?Sized>(leg: Leg<T>, f: impl FnOnce(Arc<T>) -> Arc<T>) -> Leg<T> {
            match leg {
                Maybe::Given(Some(v)) => Maybe::Given(Some(f(v))),
                other => other,
            }
        }
        fn slot<T>(slot: &mut bool, arg: &Maybe<Option<T>>) {
            if let Maybe::Given(given) = arg {
                *slot = given.is_some();
            }
        }
        let clock = self.time.clock();
        let (llm, tts, stt) =
            (map(llm, llm_leg), map(tts, |t| tts_leg(t, clock)), map(stt, stt_leg));
        {
            let mut legs = self.legs.lock().unwrap_or_else(|p| p.into_inner());
            slot(&mut legs.stt, &stt);
            slot(&mut legs.vad, vad);
            slot(&mut legs.llm, &llm);
            slot(&mut legs.tts, &tts);
        }
        let filter = match (&stt, speaker_filter) {
            (_, Maybe::Given(config)) => Some(config),
            (Maybe::Given(_), Maybe::NotGiven) => Some(None),
            (Maybe::NotGiven, Maybe::NotGiven) => None,
        };
        (llm, tts, stt, filter)
    }

    /// Shared by `update_agent`, `hook_described` and `enter_task`. A host frame change closes the
    /// frame it leaves and starts the one it installs; the driver sets an inline task's pair
    /// itself.
    fn frame(
        &self,
        frame: HostFrame,
    ) -> (AgentUpdate, Option<Option<crate::audio::SpeakerFilterConfig>>) {
        let HostFrame {
            instructions,
            agent,
            tools,
            tools_host,
            tool_runner,
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            speaker_filter,
            turn_handling,
        } = frame;
        let options = turn_handling.as_ref().map(crate::turn::agent_patch).unwrap_or_default();
        self.set_agent_layer(Some(options));
        let (llm, tts, stt, speaker_filter) = self.swap(llm, tts, stt, &vad, speaker_filter);
        let update = AgentUpdate {
            agent,
            instructions,
            toolset: host_toolset(tools, tools_host, tool_runner),
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            options,
            previous: PreviousActivity::Close,
            next: NewActivity::Start,
            call: None,
            task: None,
        };
        (update, speaker_filter)
    }

    fn speak(
        &self,
        text: SayText,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<crate::speech_handle::HostSpeech, CommandRefused> {
        let origin = origin_of(origin)?;
        let (call, rx) = self.call(Answer::speech)?;
        let speech =
            crate::speech_handle::HostSpeech::pending(self.api()?, rx, allow_interruptions);
        let call = Some(call);
        self.send_from(
            origin,
            SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call },
        )?;
        Ok(speech)
    }
}

/// A second session asked to record, explicitly. The reference raises with how to proceed.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "Only one `AgentSession` can be the primary at a time. If you want to ignore primary \
     designation, use session.start(record=False)."
)]
pub struct NotPrimary;

/// `{recording, is_primary, hosting, records}`; the refusal raises `RuntimeError`.
#[lk_agents_abi::host(export(
    name = "plan_session_start",
    kw_only,
    defaults(
        job_default = false,
        has_job_context = false,
        another_session_is_primary = false,
        text_only = false
    )
))]
pub fn plan_session_start(
    requested: Option<RecordingRequest>,
    job_default: bool,
    has_job_context: bool,
    another_session_is_primary: bool,
    session_host: Option<bool>,
    text_only: bool,
) -> Result<StartPlan, NotPrimary> {
    // The reference's order: absent `record` takes the job's setting; `text_only` forces audio off
    // after; primacy is claimed unless held; a non-primary recorder is refused if asked, downgraded
    // if defaulted.
    let mut recording =
        resolve_recording_options(requested.unwrap_or(RecordingRequest::All(job_default)));
    if text_only {
        recording.audio = false;
    }
    let is_primary = !has_job_context || !another_session_is_primary;
    if !is_primary && records_anything(&recording) {
        if requested.is_some() {
            return Err(NotPrimary);
        }
        recording = RECORDING_ALL_OFF;
    }
    Ok(StartPlan {
        recording,
        is_primary,
        hosting: is_primary && session_host.unwrap_or(true),
        records: records_anything(&recording),
    })
}

// A binding is argument extraction and serialisation only: `host_new` assembles, `start()` opens
// with a drain that serialises each `HostEvent`, every other method sends a command.

/// Decided before assembly but *opened* at [`SessionApi::open`], on the runtime (opening spawns).
pub type InterruptionPortFactory = Box<
    dyn FnOnce(lk_agents_abi::WallAnchor) -> Arc<dyn crate::agent_activity::pause::InterruptionPort>
        + Send,
>;

/// The loop's close watchdogs are 5 s; also the bound on every host teardown after the loop.
pub const CLOSE_TIMEOUT: Duration = Duration::from_secs(5);
/// The host's last line against a runtime that stopped ticking (above the close machine's bounds).
const CLOSE_WATCHDOG: Duration = Duration::from_secs(30);
/// One frame of virtual time per real millisecond while a virtual-clock close runs.
const CLOSE_TICK: Duration = Duration::from_millis(20);
const CLOSE_TICK_BUDGET: u32 = 30_000;

/// A command on a session that cannot take it. Each `Display` is the message a host raises.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SessionRefused {
    #[error("the session has not been started")]
    NotStarted,
    #[error("the session is closed")]
    Closed,
    /// The loop dropped its receiver: the command went nowhere.
    #[error("the session loop has ended")]
    LoopEnded,
}

/// [`SessionApi::open`] on a session that is not waiting to be opened.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OpenRefused {
    #[error("this AgentSession is already started")]
    AlreadyStarted,
    #[error("this AgentSession is closed")]
    Closed,
}

/// One enum, so "started twice" is a named error.
enum Stage {
    Ready(Box<Ready>),
    Running(Running),
    Closed,
}

struct Ready {
    builder: DriverBuilder,
    interruption: Option<InterruptionPortFactory>,
}

struct Running {
    api: Tx<Command>,
    calls: Calls,
    trace: Arc<Mutex<Trace>>,
    peer: Peer,
    /// One per session.
    amd: Option<crate::amd::detector::AmdHandle>,
    tasks: Vec<lk_agents_abi::AbortHandle>,
    /// Resolves when the host's drain has handed over the last event, so `close` handlers are
    /// queued before the close completes.
    drained: Option<tokio::sync::oneshot::Receiver<()>>,
}

/// Owns the turn-detector stream it closes once the loop is gone (the driver's legs never owned
/// it); a session built with no options object answers `None`.
#[derive(Default)]
struct HostParts {
    events: Option<Arc<dyn SessionEvents>>,
    eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    has_options: bool,
}

/// One stream, in emission order, so ordering between items, handoffs and completions needs no
/// barrier. Dropping the runner releases a waiting [`close`](SessionApi::close).
pub struct SessionRunner {
    events: Rx<HostEvent>,
    room_io: Arc<crate::room_io::RoomIo>,
    api: Tx<Command>,
    drained: Option<tokio::sync::oneshot::Sender<()>>,
    opening: Opening,
}

struct Opening {
    left: usize,
    delivered: Option<tokio::sync::oneshot::Sender<()>>,
}

impl SessionRunner {
    /// `None` once the loop has returned. The room's transcript leg is fed before the event is
    /// handed back so one utterance's segments land in order.
    pub async fn next_host_event(&mut self) -> Option<HostEvent> {
        // Asked again: the event handed back last was handled by the host.
        if self.opening.left == 0
            && let Some(delivered) = self.opening.delivered.take()
        {
            let _ = delivered.send(());
        }
        let ev = self.events.recv().await?;
        self.opening.left = self.opening.left.saturating_sub(1);
        if let HostEvent::Session(session) = &ev {
            forward_to_room_io(&self.room_io, session).await;
        }
        Some(ev)
    }

    pub fn api(&self) -> &Tx<Command> {
        &self.api
    }
}

impl Drop for SessionRunner {
    fn drop(&mut self) {
        if let Some(drained) = self.drained.take() {
            let _ = drained.send(());
        }
    }
}

/// What the room I/O mirrors: the agent's state edge (written as `lk.agent.state`) and the user's
/// transcript.
async fn forward_to_room_io(room_io: &Arc<crate::room_io::RoomIo>, ev: &SessionEvent) {
    match ev {
        SessionEvent::AgentStateChanged { new, .. } => room_io.on_agent_state_changed(new.as_str()),
        SessionEvent::UserTranscript(t) => {
            let text = t.ev.best_alternative().map(|a| a.text.as_str()).unwrap_or_default();
            let _ = room_io.on_user_input_transcribed(text, t.is_final).await;
        }
        _ => {}
    }
}

// --- the host's commands --------------------------------------------------------------------------

/// A dropped reply channel is resolved here, once.
pub struct Reply<T>(lk_agents_abi::BoxFuture<'static, T>);

impl<T> Reply<T> {
    fn new(f: impl std::future::Future<Output = T> + Send + 'static) -> Self {
        Reply(Box::pin(f))
    }
}

impl<T> std::future::Future for Reply<T> {
    type Output = T;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<T> {
        self.0.as_mut().poll(cx)
    }
}

/// A command refused before it was sent: its arguments (`ValueError`) or the session
/// (`RuntimeError`).
#[lk_agents_abi::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandRefused {
    #[error(transparent)]
    #[host(kind = "RuntimeError")]
    Session(#[from] SessionRefused),
    #[error("{0}")]
    Invalid(String),
}

/// `interrupt()` refused by the current speech (`allow_interruptions=False`, no `force`).
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("This generation handle does not allow interruptions")]
pub struct NotInterruptible;

fn origin_of(origin: Option<&str>) -> Result<Origin, CommandRefused> {
    origin.map_or(Ok(Origin::Host), |o| o.parse().map_err(CommandRefused::Invalid))
}

/// Where a host session's events go: one call per event, in driver order, on the host loop
/// (`call_soon_threadsafe`: a callback touching `asyncio` from a tokio worker lands in `_ready`
/// without waking the selector). A plain callable is seen through `_CoreEventSink`, which calls it.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.agent_session._CoreEventSink"))]
pub trait SessionEvents: Send + Sync {
    fn event(&self, wire: &HostWire);
    /// The stream has ended: said once, so a host awaiting a `speech_done` that can no longer come
    /// is released.
    fn stream_closed(&self, wire: &lk_agents_abi::voice::events::StreamClosedWire);
}

/// Timers and playout must share one clock: a fake room brings its own virtual clock, a host
/// `rtc.Room` is a [`HostRoom`] on the wall clock.
pub trait SessionRoom: Send + Sync {
    fn rtc(&self) -> Arc<dyn lk_agents_rtc::RtcRoom>;
    fn timebase(&self) -> Timebase;
}

/// A host `rtc.Room`, seen through `_core_rtc.Room`, whose `_core_rtc` wraps it.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.room_io._core_rtc.Room"))]
pub trait HostRoom: Send + Sync {
    /// The room, as the seam sees it.
    fn rtc(&self) -> Arc<dyn lk_agents_rtc::RtcRoom>;
}

/// On the clock the proxy was built on, else the ambient one (the wall clock, or the running
/// loop's on virtual time).
#[cfg(feature = "python")]
struct HostedRoom(Arc<dyn HostRoom>, Timebase);

#[cfg(feature = "python")]
impl HostedRoom {
    fn new(room: Arc<dyn HostRoom>, clock: Option<Arc<dyn lk_agents_abi::Clock>>) -> Self {
        let clock = clock.unwrap_or_else(
            <Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient,
        );
        HostedRoom(room, Timebase::of_clock(clock))
    }
}

#[cfg(feature = "python")]
impl SessionRoom for HostedRoom {
    fn rtc(&self) -> Arc<dyn lk_agents_rtc::RtcRoom> {
        self.0.rtc()
    }

    fn timebase(&self) -> Timebase {
        self.1.clone()
    }
}

/// A native room (registered by its `as_dyn`) as itself, anything else as a host room.
#[cfg(feature = "python")]
impl lk_agents_abi::python::HostProxy for dyn SessionRoom {
    fn from_host(
        obj: &pyo3::Bound<'_, pyo3::PyAny>,
        clock: Option<Arc<dyn lk_agents_abi::Clock>>,
    ) -> pyo3::PyResult<Arc<Self>> {
        if let Some(native) = lk_agents_abi::python::native::<dyn SessionRoom>(obj) {
            return Ok(native);
        }
        let host =
            <dyn HostRoom as lk_agents_abi::python::HostProxy>::from_host(obj, clock.clone())?;
        Ok(Arc::new(HostedRoom::new(host, clock)))
    }

    fn view(
        handle: lk_agents_abi::python::HostHandle,
        clock: Option<Arc<dyn lk_agents_abi::Clock>>,
    ) -> pyo3::PyResult<Box<Self>> {
        let host = <dyn HostRoom as lk_agents_abi::python::HostProxy>::view(handle, clock.clone())?;
        Ok(Box::new(HostedRoom::new(Arc::from(host), clock)))
    }
}

/// The session's clock scopes the host objects converted after the room (`HostClock`).
#[cfg(feature = "python")]
impl lk_agents_abi::python::HostClock for dyn SessionRoom {
    fn host_clock(&self) -> Arc<dyn lk_agents_abi::Clock> {
        self.timebase().clock()
    }
}

/// The session's clock scopes the host providers its commands take (`HostClock`).
#[cfg(feature = "python")]
impl lk_agents_abi::python::HostClock for SessionApi {
    fn host_clock(&self) -> Arc<dyn lk_agents_abi::Clock> {
        self.time().clock()
    }
}

/// `ValueError` for any other object (a newer package's metric type too).
pub struct HostMetricsRecord(pub lk_agents_abi::metrics::base::AgentMetrics);

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for HostMetricsRecord {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        lk_agents_abi::metrics::base::AgentMetrics::extract_known(&obj.to_owned())?
            .map(HostMetricsRecord)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("host_metrics: not an AgentMetrics record")
            })
    }
}

pub struct FixedRoom {
    pub rtc: Arc<dyn lk_agents_rtc::RtcRoom>,
    pub time: Timebase,
}

impl SessionRoom for FixedRoom {
    fn rtc(&self) -> Arc<dyn lk_agents_rtc::RtcRoom> {
        Arc::clone(&self.rtc)
    }

    fn timebase(&self) -> Timebase {
        self.time.clone()
    }
}

/// `Agent.transcription_node` / `Agent.realtime_audio_output_node`, as assembly calls them: handed
/// the core leg below, each builds the host node over it (`voice/agent_nodes.py`'s factories; a
/// plain callable is seen through `_NodeFactory`, which calls it). Called under the GIL, at
/// assembly; what one raises comes back as the exception it raised.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.agent_nodes._NodeFactory"))]
pub trait NodeFactory: Send + Sync {
    fn text_node(
        &self,
        below: crate::io::CoreTextOutput,
    ) -> Result<Arc<dyn crate::io::TextNode>, lk_agents_abi::errors::HostError>;
    fn audio_node(
        &self,
        below: crate::io::CoreAudioSink,
    ) -> Result<Arc<dyn crate::io::RealtimeAudioNode>, lk_agents_abi::errors::HostError>;
}

/// The frame a host `Agent` describes (`AgentActivity._frame_kwargs`), as `update_agent`,
/// `hook_described` and `enter_task` take it. Each provider is three-state: absent keeps the leg,
/// `None` removes it, an object replaces it. `chat_ctx` replaces when present. The tools are one
/// toolset, replaced together because the runner is per-agent. `turn_handling` is the agent layer.
#[lk_agents_abi::host(kwargs)]
pub struct HostFrame {
    pub instructions: Instructions,
    /// The new frame's agent id.
    pub agent: String,
    #[host(kwarg(default = None))]
    pub tools: Option<Vec<lk_agents_abi::ToolSpec>>,
    #[host(kwarg(default = None))]
    pub tools_host: Option<lk_agents_abi::HostRef>,
    #[host(kwarg(default = None))]
    pub tool_runner: Option<Arc<dyn crate::tool_executor::HostToolRunner>>,
    #[host(kwarg(default = None))]
    pub chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
    pub llm: Maybe<Option<Arc<dyn Llm>>>,
    pub tts: Maybe<Option<Arc<dyn Tts>>>,
    pub stt: Maybe<Option<Arc<dyn Stt>>>,
    pub vad: Maybe<Option<Arc<dyn Vad>>>,
    /// `MultiSpeakerAdapter`'s options when the new STT is one, `None` for a plain one; given with
    /// `stt` only (the filter follows the STT).
    pub speaker_filter: Maybe<Option<crate::audio::SpeakerFilterConfig>>,
    #[host(kwarg(default = None))]
    pub turn_handling: Option<crate::agent_activity::options::TurnHandlingKwargs>,
}

/// A host-backed LLM needs the host's own tool objects: Rust cannot rebuild a `Tool` from a spec.
fn host_toolset(
    tools: Option<Vec<lk_agents_abi::ToolSpec>>,
    host: Option<lk_agents_abi::HostRef>,
    runner: Option<Arc<dyn crate::tool_executor::HostToolRunner>>,
) -> Toolset {
    Toolset {
        specs: tools.unwrap_or_default(),
        host,
        runner: runner.map(|r| {
            Arc::new(crate::tool_executor::HostTools(r))
                as Arc<dyn crate::tool_executor::ToolRunner>
        }),
    }
}

/// The objects are the pool a host-backed LLM is handed beside the frame's.
fn session_tools(
    tools: Option<Vec<lk_agents_abi::ToolSpec>>,
    host: Option<lk_agents_abi::HostRef>,
) -> Option<crate::tool_executor::SessionTools> {
    tools.map(|specs| crate::tool_executor::SessionTools { specs, host })
}

/// STT and VAD keep the host's budgets (nothing in Rust retries them); LLM and TTS run at zero
/// because the engine's `Attempt` loop owns their retries, and two loops would multiply.
fn engine_retries() -> lk_agents_abi::ApiConnectOptions {
    lk_agents_abi::ApiConnectOptions { max_retry: 0, ..lk_agents_abi::ApiConnectOptions::default() }
}

fn stt_leg(stt: Arc<dyn Stt>) -> Arc<dyn Stt> {
    stt.with_conn_options(&lk_agents_abi::ApiConnectOptions::default()).unwrap_or(stt)
}

fn llm_leg(llm: Arc<dyn Llm>) -> Arc<dyn Llm> {
    llm.with_conn_options(&engine_retries()).unwrap_or(llm)
}

/// A non-streaming TTS is split into sentences with BlingFire, as the reference.
fn tts_leg(tts: Arc<dyn Tts>, clock: Arc<dyn Clock>) -> Arc<dyn Tts> {
    let tts = tts.with_conn_options(&engine_retries()).unwrap_or(tts);
    lk_agents_abi::StreamAdapter::adapted(
        tts,
        lk_agents_media::blingfire::SentenceBuffer::factory(),
        clock,
    )
}

/// Why a host session could not be assembled. A node factory that raised comes back as the
/// exception it raised: the host wrote the factory and catches its own type.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
pub enum SessionBuildError {
    #[error(transparent)]
    #[host(delegate)]
    Node(lk_agents_abi::errors::HostError),
    #[error("{0}")]
    Assembly(String),
}

/// `start()`'s awaited half failed.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, thiserror::Error)]
#[error("the session failed to start: {0}")]
pub struct StartFailed(DriverError);

/// `aclose(reason=..)` with a reason that is no `CloseReason`.
#[lk_agents_abi::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown CloseReason {0:?}")]
pub struct UnknownCloseReason(String);

/// `push_video(..)` handed a part that is not an image.
#[lk_agents_abi::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "push_video takes an image content part -- `ChatContent.image_frame(..)`, which is what \
     `llm.ImageContent(image=frame)` becomes."
)]
pub struct NotAnImage;

pub struct TextPusher(std::sync::Mutex<Option<tokio::sync::mpsc::Sender<String>>>);

#[lk_agents_abi::host(object(cell = "arc"))]
impl TextPusher {
    /// Queue one chunk. Awaited: a full channel waits rather than dropping (every chunk is spoken,
    /// in order). Resolves `False` once the speech is gone, so the host stops pulling.
    #[host(method)]
    pub async fn push(&self, text: String) -> bool {
        let tx = self.0.lock().unwrap_or_else(|p| p.into_inner()).clone();
        match tx {
            Some(tx) => tx.send(text).await.is_ok(),
            None => false,
        }
    }

    /// No more words. This is what finishes the speech. Idempotent.
    #[host(method)]
    pub fn end(&self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).take();
    }
}

pub struct FramePusher(
    std::sync::Mutex<Option<tokio::sync::mpsc::Sender<lk_agents_abi::AudioFrame>>>,
);

#[lk_agents_abi::host(object(cell = "arc"))]
impl FramePusher {
    /// Queue one `rtc.AudioFrame`. Awaited: a full channel is backpressure, never a dropped frame.
    /// Resolves `False` once the speech is gone, so the host stops pulling.
    #[host(method)]
    pub async fn push(&self, frame: lk_agents_abi::AudioFrame) -> bool {
        let tx = self.0.lock().unwrap_or_else(|p| p.into_inner()).clone();
        match tx {
            Some(tx) => tx.send(frame).await.is_ok(),
            None => false,
        }
    }

    /// No more frames. This is what finishes the speech. Idempotent.
    #[host(method)]
    pub fn end(&self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).take();
    }
}

// After `HostFrame`, whose kwargs macro the host methods expand against.
mod host;
#[cfg(test)]
mod tests;
