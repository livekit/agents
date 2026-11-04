//! `_core.VoiceSession` as the host holds it: assembled from the host's arguments, started,
//! commanded and closed. It owns what runs beside the session's loop (the host event drain, hook
//! answers, handoffs, a streamed `say`'s feed); the decisions behind each command are voice's.

mod builder;
pub(super) mod calls;
mod class;
mod host_drain;
mod options;
pub mod owner;
pub(super) mod shared;
pub mod storage;
#[cfg(test)]
#[path = "../session_tests.rs"]
mod tests;
mod time;

pub use builder::*;
pub use calls::*;
pub use options::SessionOptions;
pub use shared::*;
pub use time::Timebase;

use host_drain::HostDrain;
#[cfg(test)]
use host_drain::answer_hook;

use crate::telemetry;
use crate::voice::nodes::Nodes;
use crate::voice::nodes::OutputNodes;
use crate::voice::resources::Resources;
use crate::voice::speech_handle::{SayStream, SpeechHandle, Speeches};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::events::{CloseReason, HostWire};
use lk_agents_abi::voice::room_owner::RoomIo;
use lk_agents_abi::voice::session::{
    CommandRefused, OpenRefused, SessionRefused, StartFailed, UpdateAgentArgs,
};
use lk_agents_abi::voice::session_effects::{FillerReply, SessionEvents, SessionHooks};
use lk_agents_abi::voice::session_options::SessionConnOptions;
use lk_agents_abi::worker::remote_session::{SessionHosting, SessionPort};
use lk_agents_abi::{
    BoxFuture, HOST_TEARDOWN_TIMEOUT, HostRef, Llm, Maybe, Spawner, Stt, Tts, Vad,
};
use lk_agents_net::task::{TaskId, TaskSet};
use lk_agents_voice::agent_activity::options::{ConfigStack, SessionLegs};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::lifecycle::{CloseAction, RecorderTarget, SessionLifecycle};
use lk_agents_voice::agent_session::{
    AgentUpdate, Answer, Close, Command, DriverError, HookCall, RecorderCommand, RecorderHandle,
    RequestId, SayText, SessionView, SpeechCall, SpeechCommand, VoiceCommandOrigin, VoiceHostEvent,
};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::Instructions;
use lk_agents_voice::tool_executor::Toolset;
use lk_agents_voice::trace::{CoreStep, Step};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The host's session handle: assembled at construction, opened once, commanded, and closed.
/// Startup owns the runtime resources; accepted core effects supply the readable views.
pub struct VoiceSession {
    tool_bodies: calls::ToolBodies,
    /// `Arc`, so `open`'s future can write the running loop back after the borrow ended.
    stage: Arc<Mutex<Stage>>,
    view: tokio::sync::watch::Receiver<Arc<SessionView>>,
    /// Every speech's live handle, from its host call or creation to the session's end.
    speeches: Arc<Speeches>,
    room_io: Arc<dyn RoomIo>,
    /// The transcript synchroniser, held only so `close_io` can close it.
    outputs: crate::voice::io::SessionOutputs,
    spawner: Arc<dyn Spawner>,
    /// What runs beside the loop: its drain, hook answers, handoffs, fed `say`s, a `shutdown`.
    tasks: Arc<TaskSet>,
    /// The driver's clock: `close` on a virtual clock moves it.
    time: Timebase,
    /// The last `with_filler` scope numbered by `filler_start`.
    last_filler: AtomicU64,
    /// Every `Agent` by its id, as the host installed it: the events naming an agent hand it
    /// back.
    agents: HostAgents,
    /// The monotonic-to-wall pair every `Ts`-carrying event converts against, read once so events
    /// never disagree about when `Ts::ZERO` was.
    anchor: lk_agents_abi::WallAnchor,
    host: HostParts,
    /// Every provider handed over is registered here; the driver performs against it.
    resources: Resources,
    /// `conn_options=`: every leg handed over, at start or later, is bound to its budget.
    conn: SessionConnOptions,
    /// The session's report as its loop closed.
    report: FinalReport,
}

/// What a deterministic simulation needs from a session it drives, installed before `start()`.
#[cfg(feature = "dst")]
pub struct Probe {
    /// Every journal step, in order, from the first the session wrote.
    pub step: Box<dyn FnMut(&Step) + Send>,
    /// Every host event, typed, as the drain takes it and before the host sees its wire form.
    pub host_event: HostEventProbe,
}

#[cfg(feature = "dst")]
pub type HostEventProbe = Box<dyn FnMut(&VoiceHostEvent) + Send>;

/// Decided before assembly but opened at `open`, on the runtime (opening spawns).
type InterruptionPortFactory = Box<
    dyn FnOnce(
            lk_agents_abi::WallAnchor,
        )
            -> Arc<dyn lk_agents_voice::agent_activity::false_interruption::InterruptionPort>
        + Send,
>;

/// The host's last line against a runtime that stopped ticking (above the close machine's bounds).
const CLOSE_WATCHDOG: Duration = Duration::from_secs(30);
/// One frame of virtual time per real millisecond while a virtual-clock close runs.
const CLOSE_TICK: Duration = Duration::from_millis(20);
const CLOSE_TICK_BUDGET: u32 = 30_000;

enum Stage {
    Ready(Box<Ready>),
    Opening { remote: Option<Arc<dyn SessionHosting>>, cancelled: tokio::sync::watch::Sender<bool> },
    Running(Box<Running>),
    Closed(Option<Arc<dyn SessionHosting>>),
}

impl Stage {
    fn lifecycle(&self) -> SessionLifecycle {
        match self {
            Self::Ready(_) => SessionLifecycle::Ready,
            Self::Opening { .. } => SessionLifecycle::Opening,
            Self::Running(_) => SessionLifecycle::Running,
            Self::Closed(_) => SessionLifecycle::Closed,
        }
    }
}

struct Ready {
    builder: DriverBuilder,
    /// The opening agent's `tts_node` and `stt_node`, run over the opening legs.
    nodes: Nodes,
    interruption: Option<InterruptionPortFactory>,
    #[cfg(feature = "dst")]
    probe: Option<HostEventProbe>,
}

struct Running {
    remote: Option<Arc<dyn SessionHosting>>,
    api: Tx,
    /// The replies of the calls the host awaits, until the core answers them.
    calls: Arc<Calls>,
    /// `session.keyterms`, as the driver last pushed them.
    keyterms: tokio::sync::watch::Receiver<Option<Vec<String>>>,
    /// `Agent.chat_ctx`: the driving agent's view, as the driver last published it.
    chat_ctx: tokio::sync::watch::Receiver<Arc<lk_agents_abi::ChatContext>>,
    peer: PeerHandle,
    /// One per session.
    amd: Option<Arc<dyn lk_agents_abi::voice::amd::AmdPort>>,
    /// The loop and the host event drain, cancelled once the close has drained.
    loop_tasks: Vec<TaskId>,
    /// Resolves when the host's drain has handed over the last event, so `close` handlers are
    /// queued before the close completes.
    drained: Option<tokio::sync::oneshot::Receiver<()>>,
    /// The loop's id counter: a speech call's handle has its id before the loop creates it.
    speech_ids: Arc<AtomicU64>,
    /// The executor's, for every TTS swapped in.
    tts_adapter: TtsAdapter,
    /// The executor's, for every STT swapped in.
    stt_adapter: SttAdapter,
    _stalls: crate::telemetry::StallWatch,
}

impl Running {
    /// A host call's id, from the session's counter.
    fn next_call(&self) -> RequestId {
        RequestId(self.speech_ids.fetch_add(1, Ordering::Relaxed))
    }
}

/// Live inputs handed to the runtime when a remote session host is opened.
#[derive(Clone)]
pub struct RemoteSessionConfig {
    pub port: Arc<dyn SessionPort>,
    pub room: Option<Arc<dyn lk_agents_abi::voice::rtc::RtcRoom>>,
    pub console: Option<Arc<dyn lk_agents_abi::voice::console::ConsolePort>>,
}

/// What only a host-built session carries.
#[derive(Default)]
struct HostParts {
    events: Option<Arc<dyn SessionEvents>>,
    /// Without it the hook calls are the embedder's to answer (`Command::HookDone`).
    hooks: Option<Arc<dyn SessionHooks>>,
    /// Set once the close sequence ran; a second `aclose()` waits for the first.
    closed: Arc<tokio::sync::OnceCell<()>>,
    /// Handoffs in call order: one waits for the previous one's toolset setup.
    handoffs: Arc<tokio::sync::Mutex<()>>,
    /// The turn-detector stream, closed once the loop is gone (the driver's legs never owned it).
    eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// The options object the session was built over: it mints `override_options` layer ids.
    /// `resolved_options()` answers `None` for a session built with none.
    options: Option<SessionOptions>,
    /// Where `AMD` awaits the callee: the call's room, `None` on an offline (console) session.
    call_room: Option<Arc<dyn lk_agents_abi::voice::rtc::RtcRoom>>,
    /// The remote session protocol this session serves, if it does.
    remote: Option<RemoteSessionConfig>,
    /// The job whose room the session runs in.
    job: Option<Arc<dyn lk_agents_abi::worker::job::SessionJob>>,
    /// The job this primary session ends once it closes.
    job_run: Option<Arc<dyn lk_agents_abi::worker::ipc_host::JobExecution>>,
}

/// The providers a host call hands over, registered.
struct ModelRefs {
    llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
    tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
    stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
    vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
}

type Leg<T> = Maybe<Option<Arc<T>>>;

impl VoiceSession {
    pub fn time(&self) -> &Timebase {
        &self.time
    }

    /// The scheduler supplied when the session was assembled.
    pub fn spawner(&self) -> Arc<dyn Spawner> {
        self.spawner.clone()
    }

    pub fn legs(&self) -> SessionLegs {
        self.view.borrow().config.legs
    }

    pub fn stack(&self) -> ConfigStack {
        self.view.borrow().config.stack.clone()
    }

    /// Every provider and leg the session still holds.
    #[cfg(feature = "dst")]
    pub fn resources(&self) -> &Resources {
        &self.resources
    }

    /// Every step the driver journals, from the first, to `observer`. Before `start()`.
    #[cfg(any(test, feature = "dst"))]
    pub fn observe(&self, observer: impl FnMut(&Step) + Send + 'static) {
        if let Stage::Ready(ready) = &mut *self.stage.lock() {
            ready.builder.journal = shared::Journal::observed(observer);
        }
    }

    #[cfg(feature = "dst")]
    pub fn probe(&self, Probe { step, host_event }: Probe) {
        if let Stage::Ready(ready) = &mut *self.stage.lock() {
            ready.builder.journal = shared::Journal::observed(step);
            ready.probe = Some(host_event);
        }
    }

    /// Refuses synchronously when there is nothing to open; does not wait for a participant.
    /// The loop and `drain` run in the session's task set, so `close` can cancel both.
    pub fn open(
        &self,
        mut executor: Box<dyn SessionExecutor>,
        remote: Option<Arc<dyn SessionHosting>>,
        drain: impl FnOnce(SessionRunner) -> BoxFuture<'static, ()> + Send + 'static,
    ) -> Result<impl Future<Output = Result<(), DriverError>> + Send + 'static, OpenRefused> {
        let (cancel, cancelled) = tokio::sync::watch::channel(false);
        let ready = {
            let mut stage = self.stage.lock();
            stage.lifecycle().open()?;
            let Stage::Ready(ready) = std::mem::replace(
                &mut *stage,
                Stage::Opening { remote: remote.clone(), cancelled: cancel },
            ) else {
                unreachable!("opening consumes ready resources under the same lock")
            };
            *ready
        };
        let mut admission = builder::OpeningSession::new(self, cancelled);

        let room_io = Arc::clone(&self.room_io);
        let spawner = Arc::clone(&self.spawner);
        let anchor = self.anchor;
        let time = self.time.clone();
        let speeches = Arc::clone(&self.speeches);
        Ok(async move {
            admission.check_open()?;
            #[cfg(feature = "dst")]
            let Ready { mut builder, nodes, interruption, probe } = ready;
            #[cfg(not(feature = "dst"))]
            let Ready { mut builder, nodes, interruption } = ready;
            if let Some(open_port) = interruption {
                builder.interruption = Some(open_port(anchor));
            }

            // The room's own IO first, under `session_start`; its events join the driver's queue,
            // one stream, one order.
            room_io.clone().start(builder.startup());
            let (job, session_span) =
                (builder.trace.job.clone(), builder.opening().session.context);
            builder.room_io_events = Some(room_io.clone());

            let tts_adapter = executor.tts_adapter();
            let stt_adapter = executor.stt_adapter();
            prepare_opening_legs(&mut builder, &nodes, &tts_adapter, &stt_adapter)?;
            let (cores, handle) = tokio::select! {
                biased;
                _ = admission.cancelled.wait_for(|cancelled| *cancelled) => {
                    return Err(DriverError::StartupClosed);
                }
                started = builder.start(&mut *executor) => started?,
            };
            let SessionHandle {
                api,
                events,
                keyterms,
                chat_ctx,
                peer,
                ivr_detection: _,
                speech_ids,
                calls,
            } = handle;
            let mut owner = executor.own(cores);
            admission.check_open()?;
            // The opening cycle: `open` resolves only once the drain took its events, so a host's
            // `start()` sees `listening` dispatched first.
            owner.open(time.read());
            let (opening_tx, opening_rx) = tokio::sync::oneshot::channel();
            let opening = Opening { left: events.len(), delivered: Some(opening_tx) };

            if let Some(run) = owner.run() {
                admission.spawn(spawner.as_ref(), "voice session loop", run)?;
            }
            // Drained whether or not anybody listens: the queue is unbounded.
            let (drained_tx, drained) = tokio::sync::oneshot::channel();
            let runner = SessionRunner {
                events,
                speeches,
                room_io: Arc::clone(&room_io),
                api: api.clone(),
                drained: Some(drained_tx),
                opening,
                #[cfg(feature = "dst")]
                host_event: probe,
            };
            admission.spawn(spawner.as_ref(), "host event drain", drain(runner))?;
            // The job's loop stalls are counted on this session.
            let stalled = peer.clone();
            let stalls = crate::telemetry::watch_stalls(
                job,
                session_span,
                Arc::new(move |duration| {
                    let _ =
                        stalled.send(VoiceCommandOrigin::Host, Command::EventLoopBlocked(duration));
                }),
            );
            admission.install(Running {
                api,
                calls,
                keyterms,
                chat_ctx,
                peer,
                _stalls: stalls,
                amd: None,
                remote,
                loop_tasks: Vec::new(),
                drained: Some(drained),
                speech_ids,
                tts_adapter,
                stt_adapter,
            })?;
            // A drain that returned without reading drops the sender: nothing to wait for.
            let _ = opening_rx.await;
            Ok(())
        })
    }

    /// Opens the providers through `executor` and runs it beside the host event drain: one event
    /// per call, in driver order; a hook call handed to the host loop in that order, then awaited
    /// on its own task. The job's room connects beside the opening; the remote session starts
    /// once both are done, so what it queued meanwhile goes out on a connected room.
    pub fn start_with(
        &self,
        executor: Box<dyn SessionExecutor>,
    ) -> Result<BoxFuture<'static, Result<(), StartFailed>>, OpenRefused> {
        let job = self.host.job.clone();
        // The room connects under `session_start`, reserved before the driver opens it.
        let startup = match &mut *self.stage.lock() {
            Stage::Ready(ready) => {
                let startup = ready.builder.startup();
                ready.builder.tracer.resume(&startup)
            }
            _ => telemetry::Cx::NONE,
        };
        let remote = self.host.remote.clone().map(|config| {
            crate::voice::remote_session::hosting(config, self.time.clone(), self.spawner.clone())
        });
        let drain = HostDrain {
            events: self.host.events.clone(),
            hooks: self.host.hooks.clone(),
            remote: remote.clone(),
            job_run: self.host.job_run.clone(),
            view: self.view.clone(),
            anchor: self.anchor,
            agents: self.agents.clone(),
            spawner: Arc::clone(&self.spawner),
            tasks: Arc::clone(&self.tasks),
        };
        let opening =
            self.open(executor, remote.clone(), move |runner| Box::pin(drain.run(runner)))?;
        Ok(Box::pin(async move {
            let connect = async {
                match &job {
                    Some(job) => job.connect(&startup).await,
                    None => Ok(()),
                }
            };
            let (opened, connected) = tokio::join!(opening, connect);
            opened.map_err(|e| StartFailed::Driver(e.to_string()))?;
            connected.map_err(StartFailed::Host)?;
            if let Some(remote) = remote {
                remote.start().await.map_err(StartFailed::Host)?;
            }
            Ok(())
        }))
    }

    /// `None` detaches. Safe before or after `open`.
    pub fn attach_recorder(
        &self,
        mut recorder: Option<Arc<dyn lk_agents_abi::recorder::AudioRecorder>>,
    ) {
        let running = {
            let mut stage = self.stage.lock();
            match (stage.lifecycle().recorder_target(), &mut *stage) {
                (RecorderTarget::Builder, Stage::Ready(ready)) => {
                    std::mem::swap(&mut ready.builder.recorder, &mut recorder);
                    false
                }
                (RecorderTarget::Running, Stage::Running(_)) => true,
                _ => false,
            }
        };
        if running {
            let cmd = match recorder {
                Some(tap) => {
                    RecorderCommand::Attach(RecorderHandle(self.resources.register_recorder(tap)))
                }
                None => RecorderCommand::Detach,
            };
            // Use normal command admission: a closed queue must retire handed-over resources.
            let _ = self.send(Command::Recorder(cmd));
        }
    }

    /// `false`: the loop did not take it (not started or closed), and the image is dropped.
    pub fn push_video(&self, image: lk_agents_abi::ImageContent) -> bool {
        self.running(|running| Ok(running.api.send(Command::Video(image)).is_ok())).unwrap_or(false)
    }

    /// Refused before `start()` and after the close.
    fn running<R>(
        &self,
        f: impl FnOnce(&mut Running) -> Result<R, SessionRefused>,
    ) -> Result<R, SessionRefused> {
        let mut stage = self.stage.lock();
        stage.lifecycle().running()?;
        let Stage::Running(running) = &mut *stage else { unreachable!("running resources") };
        f(running)
    }

    fn remote_host(&self) -> Option<Arc<dyn SessionHosting>> {
        match &*self.stage.lock() {
            Stage::Opening { remote, .. } | Stage::Closed(remote) => remote.clone(),
            Stage::Ready(_) => None,
            Stage::Running(running) => running.remote.clone(),
        }
    }

    /// The live handle of speech `id`, whenever the session held it.
    pub fn speech(&self, id: SpeechId) -> Option<Arc<SpeechHandle>> {
        self.speeches.get(id)
    }

    pub fn api(&self) -> Result<Tx, SessionRefused> {
        self.running(|r| Ok(r.api.clone()))
    }

    /// A call to await: its id goes in the command, which carries its reply; `pick` takes its
    /// answer.
    pub fn call<T>(
        &self,
        pick: fn(Answer) -> Option<T>,
    ) -> Result<(Call, Awaited<T>), SessionRefused> {
        self.running(|r| Ok(open(r.next_call(), pick)))
    }

    /// A `say()` / `generate_reply()`: the handle the host gets now, with its id and the
    /// interruptibility it is created with (`allow_interruptions`, else the options' as they
    /// stand), and the call that brings it to the loop.
    pub fn speech_call(
        &self,
        allow_interruptions: Option<bool>,
    ) -> Result<(Arc<SpeechHandle>, SpeechCall, Call), SessionRefused> {
        let allow = allow_interruptions
            .unwrap_or_else(|| self.stack().resolve(&self.legs()).turn.interruption.enabled());
        self.running(|r| {
            let id = SpeechId(r.speech_ids.fetch_add(1, Ordering::Relaxed));
            let speech = SpeechHandle::pending(self.speeches.requests(r.api.clone()), id, allow);
            self.speeches.register(Arc::clone(&speech));
            let reply_to = ReplyTo::Speech(SpeechReply(Arc::clone(&speech)));
            let call = Call { id: r.next_call(), reply_to };
            Ok((speech, SpeechCall { id: call.id, speech: id }, call))
        })
    }

    pub fn send(&self, cmd: impl Into<Command>) -> Result<(), SessionRefused> {
        self.send_from(VoiceCommandOrigin::Host, cmd)
    }

    pub fn send_from(
        &self,
        origin: VoiceCommandOrigin,
        cmd: impl Into<Command>,
    ) -> Result<(), SessionRefused> {
        self.send_command(origin, cmd.into(), None)
    }

    /// The command `make` builds around a new call's id, sent with its reply; `pick` takes the
    /// answer.
    pub fn ask<T, C: Into<Command>>(
        &self,
        origin: VoiceCommandOrigin,
        pick: fn(Answer) -> Option<T>,
        make: impl FnOnce(RequestId) -> C,
    ) -> Result<Awaited<T>, SessionRefused> {
        let (call, answer) = self.call(pick)?;
        self.send_awaited(origin, make(call.id), call)?;
        Ok(answer)
    }

    /// A command naming `call`, sent with its reply.
    pub fn send_awaited(
        &self,
        origin: VoiceCommandOrigin,
        cmd: impl Into<Command>,
        call: Call,
    ) -> Result<(), SessionRefused> {
        self.send_command(origin, cmd.into(), Some(call))
    }

    /// A command the loop never takes releases the providers it hands over.
    fn send_command(
        &self,
        origin: VoiceCommandOrigin,
        cmd: Command,
        call: Option<Call>,
    ) -> Result<(), SessionRefused> {
        let pending = self.resources.pending(cmd.resources());
        let sent = self.running(|r| {
            r.calls.send_command(&r.api, origin, cmd, call).map_err(|_| SessionRefused::LoopEnded)
        });
        if sent.is_ok() {
            pending.submitted();
        }
        sent
    }

    /// `AMD._run`. One per session; the verdict comes back as `amd_result`.
    pub fn attach_amd(
        &self,
        setup: lk_agents_voice::amd::AmdSetup,
    ) -> Result<bool, SessionRefused> {
        self.running(|r| {
            if r.amd.is_some() {
                r.peer.journal(Step::Core(CoreStep::PortFailure("amd: already attached")));
                Ok(false)
            } else {
                r.amd = Some(crate::voice::amd::spawn(r.peer.clone(), setup));
                Ok(true)
            }
        })
    }

    /// `_aclose_impl`: resolves once the loop returned and the drain handed over the last event.
    /// The stage stays `Running` meanwhile (hooks and tools still answer through it). Idempotent.
    pub fn close(
        &self,
        reason: CloseReason,
        drain: bool,
        error: Option<lk_agents_voice::events::ErrorEvent>,
    ) -> impl Future<Output = ()> + Send + 'static {
        let (running, retired) = {
            let mut stage = self.stage.lock();
            match (stage.lifecycle().close(), &mut *stage) {
                (CloseAction::Running, Stage::Running(running)) => (
                    Some((
                        running.api.clone(),
                        running.drained.take(),
                        running.loop_tasks.clone(),
                        running.remote.clone(),
                    )),
                    None,
                ),
                (CloseAction::Opening, Stage::Opening { remote, cancelled }) => {
                    cancelled.send_replace(true);
                    let closed = Stage::Closed(remote.clone());
                    (None, Some(std::mem::replace(&mut *stage, closed)))
                }
                (CloseAction::Ready, Stage::Ready(_)) => {
                    (None, Some(std::mem::replace(&mut *stage, Stage::Closed(None))))
                }
                (CloseAction::None, _) => (None, None),
                _ => unreachable!("the close acts on the stage it was decided on"),
            }
        };
        drop(retired);
        let (stage, tasks) = (Arc::clone(&self.stage), Arc::clone(&self.tasks));
        let time = self.time.clone();
        async move {
            let Some((api, drained, loop_tasks, remote)) = running else { return };
            let _ = api.send(Command::Close(Close { reason, drain, error }));
            // Every close phase is bounded on the session clock; this watchdog is a backstop. On a
            // virtual clock the host stopped advancing, so the clock is moved here.
            if time.is_virtual() {
                advance_until(&time, api.closed()).await;
            } else {
                let _ = time.timeout(CLOSE_WATCHDOG, api.closed()).await;
            }
            // `Closed` **before** awaiting the drain: `Running` holds the `PeerHandle`, a sender on
            // the host stream, and the drain only ends when every sender is gone.
            {
                let mut stage = stage.lock();
                *stage = Stage::Closed(remote);
            }
            // Then the drain: the host's `close` is queued before this future completes.
            if let Some(drained) = drained {
                let _ = time.timeout(HOST_TEARDOWN_TIMEOUT, drained).await;
            }
            loop_tasks.into_iter().for_each(|task| tasks.cancel(task));
        }
    }

    /// After the host's `close` handlers ran. The synchroniser first, or its rotation tasks publish
    /// into a closed stream.
    pub fn close_io(&self) -> impl Future<Output = ()> + Send + 'static {
        let room_io = Arc::clone(&self.room_io);
        let outputs = self.outputs.clone();
        let time = self.time.clone();
        async move {
            outputs.aclose().await;
            let _ = time.timeout(HOST_TEARDOWN_TIMEOUT, room_io.aclose()).await;
        }
    }

    /// Given present, given `None` clears, not-given keeps. Each provider given is wrapped, under
    /// the agent's node when it overrides one, and registered: the command carries its reference.
    fn swap(
        &self,
        llm: Leg<dyn Llm>,
        tts: Leg<dyn Tts>,
        stt: Leg<dyn Stt>,
        vad: Leg<dyn Vad>,
        nodes: &Nodes,
    ) -> ModelRefs {
        fn map<T: ?Sized, R>(leg: Leg<T>, f: impl FnOnce(Arc<T>) -> R) -> Maybe<Option<R>> {
            match leg {
                Maybe::Given(given) => Maybe::Given(given.map(f)),
                Maybe::NotGiven => Maybe::NotGiven,
            }
        }
        let clock = self.time.clock();
        let r = &self.resources;
        // Before start nothing reaches the loop (the command is refused and the leg released).
        let adapters = self.running(|r| Ok((r.tts_adapter.clone(), r.stt_adapter.clone()))).ok();
        let vad_now = match &vad {
            Maybe::Given(given) => given.clone(),
            Maybe::NotGiven => {
                let id = self.view.borrow().vad.as_ref().map(|v| v.id);
                id.and_then(|id| self.resources.vad(id))
            }
        };
        ModelRefs {
            llm: map(llm, |l| r.register_llm(nodes.llm(llm_leg(l, self.conn.llm)))),
            tts: map(tts, |t| {
                let t = match &adapters {
                    Some((adapter, _)) => tts_leg(t, adapter, clock.clone(), self.conn.tts),
                    None => t,
                };
                r.register_tts(nodes.tts(t))
            }),
            stt: map(stt, |s| {
                let s = match &adapters {
                    Some((_, adapter)) => {
                        stt_leg(s, adapter, vad_now, clock.clone(), self.conn.stt)
                    }
                    None => s,
                };
                r.register_stt(nodes.stt(s))
            }),
            vad: map(vad, |v| r.register_vad(v)),
        }
    }

    /// The agent `frame` describes, its providers registered, as `update_agent` installs it.
    pub fn frame(&self, frame: UpdateAgentArgs) -> AgentUpdate {
        let UpdateAgentArgs {
            instructions,
            dynamic_instructions,
            agent,
            agent_host,
            tools,
            tools_host,
            voice_agent,
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            turn_handling,
            min_consecutive_speech_delay,
            use_tts_aligned_transcript,
            expressive,
            llm_node,
            tts_node,
            stt_node,
            transcription_node,
        } = frame;
        self.agents.install(&agent, voice_agent.toolset_scope(), agent_host);
        let options = lk_agents_voice::turn::agent_layer(
            turn_handling.as_ref(),
            min_consecutive_speech_delay,
            use_tts_aligned_transcript,
        );
        let nodes = Nodes { llm: llm_node, tts: tts_node, stt: stt_node };
        let ModelRefs { llm, tts, stt, vad } = self.swap(llm, tts, stt, vad, &nodes);
        AgentUpdate {
            agent,
            instructions: Instructions::plain(instructions),
            dynamic_instructions: dynamic_instructions
                .map(|value| self.resources.register_instructions(value)),
            toolset: Toolset {
                specs: tools.unwrap_or_default(),
                host: tools_host,
                agent: Some(voice_agent),
            },
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            options,
            expressive: expressive.given(),
            output_nodes: lk_agents_voice::agent_session::OutputNodes {
                transcription: transcription_node
                    .map(|value| self.resources.register_transcription_node(value)),
            },
            call: None,
            activation: None,
        }
    }

    /// `update`'s toolsets set up first, in call order: the turn is taken now. Then the command
    /// `wrap` makes of it is sent as `origin`.
    fn send_frame(
        &self,
        origin: VoiceCommandOrigin,
        mut update: AgentUpdate,
        call: Call,
        wrap: impl FnOnce(AgentUpdate) -> Command + Send + 'static,
        failed: fn(HostError) -> Answer,
    ) -> Result<(), SessionRefused> {
        let (Some(hooks), Some(tools)) = (self.host.hooks.clone(), update.toolset.host.clone())
        else {
            return self.send_awaited(origin, wrap(update), call);
        };
        let pending = self.resources.pending(update.resources().collect());
        let (api, calls, session) =
            self.running(|r| Ok((r.api.clone(), Arc::clone(&r.calls), r.peer.session.clone())))?;
        let scope = update.toolset.agent.as_ref().map_or(0, |agent| agent.toolset_scope());
        // The turn is queued for now, in call order: the lock's waiter list is entered by this
        // first poll (a tokio lock takes the real waker at the next one). Only the lock is polled
        // here; the setup's host bridge runs in the task, with its own waker.
        let mut turn = Box::pin(Arc::clone(&self.host.handoffs).lock_owned());
        let held =
            turn.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
        // Tuple fields retire in order even if this future is never polled: release an already
        // acquired turn before pending callbacks, whose destructors may reenter the session.
        let admission = (held, turn, pending);
        let handoff = async move {
            let (held, turn, pending) = admission;
            let handoff_turn = match held {
                std::task::Poll::Ready(guard) => guard,
                std::task::Poll::Pending => turn.await,
            };
            // The arriving agent's activity starts here: its toolsets are set up inside it, under
            // the `update_agent` the handoff opens.
            let reserved = telemetry::reserve::<telemetry::UpdateAgent>(&session);
            let activity = telemetry::reserve(
                &session.clone().under::<telemetry::UpdateAgent>(reserved.context),
            );
            update.activation = Some(telemetry::Activation { update: reserved, activity });
            let span = session.under::<telemetry::StartAgentActivity>(activity.context);
            let span = telemetry::start(&span, telemetry::SetupToolsets);
            let scopes = hooks.toolset_scopes().clone();
            let setup = async {
                let activation = scopes.acquire(scope, hooks.toolset_resources(&tools)?).await?;
                match hooks.tool_infos(&tools) {
                    Ok(infos) => Ok((activation, infos)),
                    Err(error) => {
                        if let Err(close) = scopes.discard(activation).await {
                            log::warn!("the refused agent toolsets could not close: {close}");
                        }
                        Err(error)
                    }
                }
            };
            let (activation, set_up) = match telemetry::check(&span, setup.await) {
                Ok(ready) => {
                    telemetry::end_with(span, ());
                    ready
                }
                Err(error) => {
                    hooks.hook_failed("setup_tools", error.host_ref.clone());
                    call.reply_to.settle(failed(error));
                    return;
                }
            };
            let toolset = std::mem::take(&mut update.toolset);
            update.toolset = Toolset::set_up(set_up, toolset.host, toolset.agent);
            let cmd = wrap(update);
            if calls.send_command(&api, origin, cmd, Some(call)).is_ok() {
                drop(handoff_turn);
                pending.submitted();
            } else {
                if let Err(error) = scopes.discard(activation).await {
                    log::warn!("the refused agent toolsets could not close: {error}");
                }
            }
        };
        self.tasks
            .try_spawn(self.spawner.as_ref(), "handoff", handoff)
            .map(|_| ())
            .ok_or(SessionRefused::LoopEnded)
    }

    fn speak(
        &self,
        (text, stream): (SayText, Option<SayStream>),
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<Arc<SpeechHandle>, CommandRefused> {
        let origin = origin_of(origin);
        let (speech, call, reply) = self.speech_call(allow_interruptions)?;
        // The handle holds what `say()` streams until its speech's task takes it.
        if let Some(stream) = stream {
            speech.hold_say(stream);
        }
        let call = Some(call);
        self.send_awaited(
            origin,
            SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call },
            reply,
        )?;
        Ok(speech)
    }

    /// A `say` of streamed words: the speech and its feeding end.
    pub fn say_text_channel(
        &self,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<(Arc<SpeechHandle>, Pusher<String>), CommandRefused> {
        // Backpressure against a runaway producer, not a buffer.
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let said = (SayText::Stream, Some(SayStream::Text(rx)));
        let speech = self.speak(said, allow_interruptions, add_to_chat_ctx, origin)?;
        Ok((speech, Pusher(Mutex::new(Some(tx)))))
    }

    /// A `say` of pre-synthesised frames, `text` still committed: the speech and its feeding end.
    pub fn say_audio_channel(
        &self,
        text: String,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<(Arc<SpeechHandle>, Pusher<lk_agents_abi::AudioFrame>), CommandRefused> {
        // Deeper than the text channel: at 20 ms a frame, 64 would be 1.3 s of audio.
        let (tx, rx) = tokio::sync::mpsc::channel(512);
        let said = (SayText::Audio { text }, Some(SayStream::Audio(rx)));
        let speech = self.speak(said, allow_interruptions, add_to_chat_ctx, origin)?;
        Ok((speech, Pusher(Mutex::new(Some(tx)))))
    }

    /// `source` read into `pusher` as one of the session's tasks.
    fn feed<T: Send + 'static>(
        &self,
        pusher: Pusher<T>,
        source: Box<dyn lk_agents_abi::StreamOutput<T>>,
    ) {
        let clock = self.time.clock();
        self.tasks.spawn(self.spawner.as_ref(), "say feed", pusher.feed(source, clock));
    }
}

/// For a host that stopped moving time (awaiting `aclose()`). Paced on real time (1 ms per step) so
/// host tasks can run between steps.
async fn advance_until(time: &Timebase, done: impl Future<Output = ()>) {
    let pacer = lk_agents_net::clock::RealClock::new();
    let mut done = std::pin::pin!(done);
    for _ in 0..=CLOSE_TICK_BUDGET {
        let tick = lk_agents_abi::Clock::wait_until(
            &pacer,
            lk_agents_abi::Clock::now(&pacer).saturating_add(Duration::from_millis(1)),
        );
        tokio::select! {
            biased;
            () = &mut done => return,
            () = tick => time.advance(CLOSE_TICK),
        }
    }
}

fn origin_of(origin: Option<VoiceCommandOrigin>) -> VoiceCommandOrigin {
    origin.unwrap_or(VoiceCommandOrigin::Host)
}

/// One stream, in emission order, so ordering between items, handoffs and completions needs no
/// barrier. Dropping the runner releases a waiting [`close`](VoiceSession::close).
pub struct SessionRunner {
    events: tokio::sync::mpsc::UnboundedReceiver<VoiceHostEvent>,
    /// The host's objects for the speeches the events name.
    speeches: Arc<Speeches>,
    room_io: Arc<dyn RoomIo>,
    api: Tx,
    drained: Option<tokio::sync::oneshot::Sender<()>>,
    opening: Opening,
    #[cfg(feature = "dst")]
    host_event: Option<HostEventProbe>,
}

struct Opening {
    left: usize,
    delivered: Option<tokio::sync::oneshot::Sender<()>>,
}

impl SessionRunner {
    /// `None` once the loop has returned. The room's transcript leg is fed before the event is
    /// handed back so one utterance's segments land in order.
    pub async fn next_host_event(&mut self) -> Option<VoiceHostEvent> {
        // Asked again: the event handed back last was handled by the host.
        if self.opening.left == 0
            && let Some(delivered) = self.opening.delivered.take()
        {
            let _ = delivered.send(());
        }
        let ev = self.events.recv().await?;
        self.opening.left = self.opening.left.saturating_sub(1);
        #[cfg(feature = "dst")]
        if let Some(observe) = &mut self.host_event {
            observe(&ev);
        }
        // What the room I/O mirrors: the agent's state edge and the user's transcript.
        match &ev {
            VoiceHostEvent::Session(SessionEvent::AgentStateChanged { new, .. }) => {
                self.room_io.clone().on_agent_state_changed(new.as_str())
            }
            VoiceHostEvent::Session(SessionEvent::UserInputTranscribed(t)) => {
                let text = t.ev.best_alternative().map(|a| a.text.as_str()).unwrap_or_default();
                let _ = self.room_io.on_user_input_transcribed(text, t.is_final).await;
            }
            // Every event naming it was wired before this one.
            VoiceHostEvent::SpeechRetired(speech) => self.speeches.retire(*speech),
            _ => {}
        }
        Some(ev)
    }

    pub fn api(&self) -> &Tx {
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

/// The host's `Agent` objects by agent id. The hooks and the handoff record name an agent by
/// id; the host reads the object it installed.
#[derive(Clone, Default)]
struct HostAgents(Arc<Mutex<std::collections::BTreeMap<(String, u64), HostRef>>>);

impl HostAgents {
    fn install(&self, agent: &str, scope: u64, host: Option<HostRef>) {
        if let Some(host) = host {
            let retired = self.0.lock().insert((agent.to_owned(), scope), host);
            drop(retired);
        }
    }

    fn get(&self, agent: &str, scope: u64) -> Option<HostRef> {
        self.0.lock().get(&(agent.to_owned(), scope)).cloned()
    }
}

/// The feeding end of a streamed `say`: what the speech reads, chunk by chunk. Dropping it, or
/// [`end`](Self::end), finishes the speech.
pub struct Pusher<T>(Mutex<Option<tokio::sync::mpsc::Sender<T>>>);

impl<T: Send + 'static> Pusher<T> {
    /// Queue one item. A full channel waits rather than dropping (every chunk is spoken, in
    /// order). `false` once the speech is gone.
    pub async fn push(&self, item: T) -> bool {
        let tx = self.0.lock().clone();
        match tx {
            Some(tx) => tx.send(item).await.is_ok(),
            None => false,
        }
    }

    /// No more items. Idempotent.
    pub fn end(&self) {
        self.0.lock().take();
    }

    /// Resolves once the speech stopped reading.
    async fn closed(&self) {
        let tx = self.0.lock().clone();
        if let Some(tx) = tx {
            tx.closed().await;
        }
    }

    /// `source` read into the speech to its end, or until the speech is gone; then closed. A
    /// failing source ends the speech with what it gave.
    async fn feed(
        self,
        mut source: Box<dyn lk_agents_abi::StreamOutput<T>>,
        clock: Arc<dyn lk_agents_abi::Clock>,
    ) {
        loop {
            let next = tokio::select! {
                biased;
                next = source.next() => next,
                () = self.closed() => break,
            };
            match next {
                Ok(Some(item)) => {
                    if !self.push(item).await {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    log::error!("say(): the stream it speaks raised: {e}");
                    break;
                }
            }
        }
        self.end();
        source.close(clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
    }
}
