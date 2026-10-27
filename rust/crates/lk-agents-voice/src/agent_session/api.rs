//! `voice/agent_session.py`'s `AgentSession` as its host holds it: [`SessionApi`], assembled from
//! the host's arguments, opened at `start()`, commanded, and closed; with the host-side traits and
//! errors it speaks through. Its host methods are in `host`.

use crate::agent_activity::options::{
    AgentSessionOptions, RECORDING_ALL_OFF, RecordingRequest, records_anything,
    resolve_recording_options,
};
use crate::agent_session::builder::DriverBuilder;
use crate::agent_session::commands::{
    AgentUpdate, Answer, Close, Command, HostRecorder, Origin, RecorderCommand, RequestId, SayText, SpeechCall,
    SpeechCommand,
};
use crate::agent_session::driver::{DriverError, PeerHandle, Rx, SessionHandle, Tx, start_ivr};
use crate::agent_session::host_events::HostEvent;
use crate::agent_session::nodes::Nodes;
use crate::agent_session::time::Timebase;
use crate::events::SessionEvent;
use crate::generation::Instructions;
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::Toolset;
use crate::trace::{CoreStep, Step};
use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::voice::events::HostWire;
use lk_agents_abi::voice::session_effects::StartPlan;
use lk_agents_abi::{
    Clock, Llm, LlmStream, Maybe, Spawner, SpawnerExt, Stt, SttSink, SttSource, Tts, TtsSink,
    TtsSource, Vad,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

/// `voice/agent_session.py`'s `AgentSession`, over the driver: assembled at construction, opened at
/// `start()`, closed at `aclose()`; every other method is one command. `frozen`, so there is no
/// borrow flag: `aclose()` can be awaited while the event drain calls back into the same object.
///
/// The host's end of a session: assembled by [`host_new`](Self::host_new), opened once, commanded through
/// [`send`](Self::send), closed through [`close`](Self::close). Every method takes `&self`; one
/// `Mutex` around a small enum, off every per-frame path.
#[lk_agents_abi::class(name = "VoiceSession")]
pub struct SessionApi {
    /// `Arc`, so `open`'s future can write the running loop back after the borrow ended.
    stage: Arc<Mutex<Stage>>,
    /// The driver's legs, kept in step with every provider swap ([`swap`](Self::swap)).
    legs: Mutex<crate::agent_activity::options::LegSet>,
    /// The driver's option stack, kept in step with every `Configure` and agent update a host
    /// sends, so `session.options` resolves the instant a change is sent.
    stack: Mutex<crate::agent_activity::options::ConfigStack>,
    room_io: Arc<crate::room_io::RoomIo>,
    /// The VAD the session runs on now, which a recognize-only STT given later goes over.
    pub(super) vad: Mutex<Option<Arc<dyn Vad>>>,
    /// The transcript synchroniser, held only so `close_io` can close it.
    synchronizer: Option<Arc<crate::transcription::synchronizer::TranscriptSynchronizer>>,
    spawner: Arc<dyn Spawner>,
    /// The driver's clock: `close` on a virtual clock moves it.
    time: Timebase,
    /// The read-back guard of the last DTMF collection, read by
    /// [`dtmf_reply_running`](Self::dtmf_reply_running).
    dtmf_reply: Mutex<Option<Arc<std::sync::atomic::AtomicBool>>>,
    /// The last `with_filler` scope numbered by [`filler_start`](Self::filler_start).
    last_filler: std::sync::atomic::AtomicU64,
    /// Every `Agent` by its id, as the host installed it: the events naming an agent hand
    /// it back ([`HostAgents::resolve`]).
    agents: HostAgents,
    /// The monotonic-to-wall pair every `Ts`-carrying event converts against, read once so events
    /// never disagree about when `Ts::ZERO` was.
    anchor: lk_agents_abi::WallAnchor,
    /// What only a host-built session carries ([`host_new`](Self::host_new)).
    host: HostParts,
    /// Every provider handed over is registered here; the driver performs against it.
    resources: lk_agents_abi::resources::Resources,
    #[cfg(feature = "dst")]
    probe: Mutex<Option<(HostEventProbe, oneshot::Sender<crate::runtime::Runtime>)>>,
}

/// Test-only (`dst`): what a deterministic simulation needs from a session it drives through this
/// API, and nothing a host can reach. Installed before `start()`.
#[cfg(feature = "dst")]
pub struct Probe {
    /// Every journal step, in order, from the first the session wrote.
    pub step: Box<dyn FnMut(&Step) + Send>,
    /// Every host event, typed, as the drain takes it and before the host sees its wire form.
    pub host_event: HostEventProbe,
    /// The session's cores after the opening cycle, for a queue owner other than `Runtime::run`.
    pub runtime: tokio::sync::oneshot::Sender<crate::runtime::Runtime>,
}

#[cfg(feature = "dst")]
pub type HostEventProbe = Box<dyn FnMut(&HostEvent) + Send>;

#[cfg(any(test, feature = "dst"))]
impl SessionApi {
    /// Every step the driver journals, from the first, to `observer`. Before `start()`.
    pub fn observe(&self, observer: impl FnMut(&Step) + Send + 'static) {
        if let Stage::Ready(ready) = &mut *self.lock() {
            ready.builder.journal = crate::trace::Journal::observed(observer);
        }
    }
}

#[cfg(feature = "dst")]
impl SessionApi {
    pub fn probe(&self, Probe { step, host_event, runtime }: Probe) {
        self.observe(step);
        *self.probe.lock().unwrap_or_else(|p| p.into_inner()) = Some((host_event, runtime));
    }

    /// Every provider and leg the session still holds.
    pub fn resources(&self) -> &lk_agents_abi::resources::Resources {
        &self.resources
    }
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
    pub fn configure(&self, update: &AgentSessionOptions) {
        self.stack.lock().unwrap_or_else(|p| p.into_inner()).session.merge(update);
    }

    /// Its `turn_handling` becomes the agent layer, replaced whole.
    pub fn set_agent_layer(&self, agent: Option<AgentSessionOptions>) {
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
        #[cfg(feature = "dst")]
        let probe = self.probe.lock().unwrap_or_else(|p| p.into_inner()).take();
        Ok(async move {
            let Ready { mut builder, interruption } = ready;
            if capture_run {
                builder.capture_run = Some(crate::run_result::RunSpec {
                    user_input: None,
                    expects_output: false,
                    max_retries: 0,
                    retry_instructions: String::new(),
                    host: None,
                });
            }
            if let Some(open_port) = interruption {
                builder.interruption = Some(open_port(anchor));
            }

            // The room's own IO first; its events join the driver's queue, one stream, one order.
            room_io.start();
            builder.room_io_events = room_io.events();

            let (mut runtime, handle) = builder.start().await?;
            let SessionHandle {
                api,
                events,
                answers,
                keyterms,
                activity,
                scheduling,
                peer,
                ivr_detection,
            } = handle;
            #[cfg(feature = "dst")]
            let (host_event, queue_owner) = probe.unzip();
            // The IVR navigator's task, subscribed before the opening cycle.
            start_ivr(&peer, ivr_detection);
            // The opening cycle: `open` resolves only once the drain took its events, so a host's
            // `start()` sees `listening` dispatched first.
            runtime.driver.open(time.read());
            let (opening_tx, opening_rx) = tokio::sync::oneshot::channel();
            let opening = Opening { left: events.len(), delivered: Some(opening_tx) };

            let mut tasks = Vec::new();
            #[cfg(feature = "dst")]
            let runtime = match queue_owner {
                Some(owner) => {
                    let _ = owner.send(runtime);
                    None
                }
                None => Some(runtime),
            };
            #[cfg(not(feature = "dst"))]
            let runtime = Some(runtime);
            if let Some(mut runtime) = runtime {
                tasks.push(spawner.spawn_unboxed(async move { runtime.run().await }));
            }
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
                #[cfg(feature = "dst")]
                host_event,
            };
            tasks.push(spawner.spawn(drain(runner)));

            *stage.lock().unwrap_or_else(|p| p.into_inner()) = Stage::Running(Running {
                api,
                calls,
                keyterms,
                activity,
                scheduling,
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

    pub fn api(&self) -> Result<Tx, SessionRefused> {
        self.running(|r| Ok(r.api.clone()))
    }

    /// A call to await: its id goes in the command, `pick` takes its answer.
    pub fn call<T: Send + 'static>(
        &self,
        pick: fn(Answer) -> Option<T>,
    ) -> Result<(RequestId, tokio::sync::oneshot::Receiver<T>), SessionRefused> {
        self.running(|r| Ok(r.calls.open(pick)))
    }

    /// A `say()` / `generate_reply()`: the handle the host gets now, and the call that brings it
    /// to the loop.
    pub fn speech_call(
        &self,
        allow_interruptions: Option<bool>,
    ) -> Result<(Arc<SpeechHandle>, SpeechCall), SessionRefused> {
        self.running(|r| {
            let speech = SpeechHandle::pending(r.api.clone(), allow_interruptions);
            let id = r.calls.open_speech(&speech);
            Ok((Arc::clone(&speech), SpeechCall { id, speech }))
        })
    }

    pub fn send(&self, cmd: impl Into<Command>) -> Result<(), SessionRefused> {
        self.send_from(Origin::Host, cmd)
    }

    /// A command the loop never takes releases the providers it hands over.
    pub fn send_from(&self, origin: Origin, cmd: impl Into<Command>) -> Result<(), SessionRefused> {
        let cmd = cmd.into();
        let handed = cmd.resources();
        let sent =
            self.running(|r| r.api.send_from(origin, cmd).map_err(|_| SessionRefused::LoopEnded));
        if sent.is_err() {
            handed.into_iter().for_each(|id| self.resources.release(id));
        }
        sent
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
                handle.input(input);
            }
            Ok(())
        })
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
            // `Closed` **before** awaiting the drain: `Running` holds the `PeerHandle`, a sender on the
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

/// The providers a host call hands over, registered.
pub(super) struct ModelRefs {
    pub(super) llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
    pub(super) tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
    pub(super) stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
    pub(super) vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
}

impl SessionApi {
    /// Given present, given `None` clears, not-given keeps.
    /// Each provider given is wrapped, under the agent's node when it overrides one, and
    /// registered: the command carries its reference.
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
        fn slot<T>(slot: &mut bool, arg: &Maybe<Option<T>>) {
            if let Maybe::Given(given) = arg {
                *slot = given.is_some();
            }
        }
        let clock = self.time.clock();
        let r = &self.resources;
        let vad_now = {
            let mut current = self.vad.lock().unwrap_or_else(|p| p.into_inner());
            if let Maybe::Given(given) = &vad {
                current.clone_from(given);
            }
            current.clone()
        };
        let models = ModelRefs {
            llm: map(llm, |l| r.register_llm(nodes.llm(llm_leg(l)))),
            tts: map(tts, |t| r.register_tts(nodes.tts(tts_leg(t, clock)))),
            stt: map(stt, |s| r.register_stt(nodes.stt(stt_leg(s, vad_now)))),
            vad: map(vad, |v| r.register_vad(v)),
        };
        {
            let mut legs = self.legs.lock().unwrap_or_else(|p| p.into_inner());
            slot(&mut legs.stt, &models.stt);
            slot(&mut legs.vad, &models.vad);
            slot(&mut legs.llm, &models.llm);
            slot(&mut legs.tts, &models.tts);
        }
        models
    }

    /// Used by `update_agent`.
    fn frame(&self, frame: UpdateAgentArgs) -> AgentUpdate {
        let UpdateAgentArgs {
            instructions,
            dynamic_instructions,
            agent,
            agent_host,
            tools,
            tools_host,
            tool_runner,
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
        } = frame;
        self.agents.install(&agent, agent_host);
        let mut options =
            turn_handling.as_ref().map(crate::turn::agent_layer).unwrap_or_default();
        options.speech.min_consecutive_speech_delay = min_consecutive_speech_delay
            .and_then(|s| std::time::Duration::try_from_secs_f64(s).ok());
        options.speech.use_tts_aligned_transcript = use_tts_aligned_transcript;
        self.set_agent_layer(Some(options));
        let nodes = Nodes { llm: llm_node, tts: tts_node, stt: stt_node };
        let ModelRefs { llm, tts, stt, vad } = self.swap(llm, tts, stt, vad, &nodes);
        AgentUpdate {
            agent,
            instructions,
            dynamic_instructions,
            toolset: host_toolset(tools, tools_host, tool_runner),
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            options,
            expressive: expressive.given(),
            call: None,
        }
    }

    fn speak(
        &self,
        text: SayText,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<Arc<SpeechHandle>, CommandRefused> {
        let origin = origin_of(origin)?;
        self.refuse_paused(&origin)?;
        let (speech, call) = self.speech_call(allow_interruptions)?;
        let call = Some(call);
        self.send_from(
            origin,
            SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call },
        )?;
        Ok(speech)
    }

    /// `_schedule_speech`'s `RuntimeError` while scheduling drains, raised by the call itself. A
    /// pause that lands after this check refuses the speech in the loop instead (interrupted).
    fn refuse_paused(&self, origin: &Origin) -> Result<(), CommandRefused> {
        let gate = match &*self.lock() {
            Stage::Running(r) => *r.scheduling.borrow(),
            Stage::Ready(_) | Stage::Closed => return Ok(()),
        };
        match gate.refusal(origin) {
            Some(rejected) => Err(CommandRefused::Rejected(rejected)),
            None => Ok(()),
        }
    }

    /// A `say` of streamed words: the speech and its feeding end.
    pub fn say_text_channel(
        &self,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<(Arc<SpeechHandle>, Pusher<String>), CommandRefused> {
        // Backpressure against a runaway producer, not a buffer.
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let speech =
            self.speak(SayText::Stream(rx), allow_interruptions, add_to_chat_ctx, origin)?;
        Ok((speech, Pusher(Mutex::new(Some(tx)))))
    }

    /// A `say` of pre-synthesised frames, `text` still committed: the speech and its feeding end.
    pub fn say_audio_channel(
        &self,
        text: String,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<(Arc<SpeechHandle>, Pusher<lk_agents_abi::AudioFrame>), CommandRefused> {
        // Deeper than the text channel: at 20 ms a frame, 64 would be 1.3 s of audio.
        let (tx, rx) = tokio::sync::mpsc::channel(512);
        let speech =
            self.speak(SayText::Audio { text, rx }, allow_interruptions, add_to_chat_ctx, origin)?;
        Ok((speech, Pusher(Mutex::new(Some(tx)))))
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
    dyn FnOnce(
            lk_agents_abi::WallAnchor,
        ) -> Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>
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
    api: Tx,
    calls: Calls,
    /// `session.keyterms`, as the driver last pushed them.
    keyterms: tokio::sync::watch::Receiver<Option<Vec<String>>>,
    /// `Agent.chat_ctx`: the driving agent's view, as the driver last published it.
    activity: tokio::sync::watch::Receiver<crate::agent_session::ActivitySnapshot>,
    scheduling: tokio::sync::watch::Receiver<crate::agent_session::SchedulingGate>,
    peer: PeerHandle,
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
    /// Without it the hook calls are the embedder's to answer (`Command::HookDone`).
    hooks: Option<Arc<dyn SessionHooks>>,
    /// Set once the close sequence ran; a second `aclose()` waits for the first.
    closed: Arc<tokio::sync::OnceCell<()>>,
    /// Handoffs in call order: one waits for the previous one's toolset setup.
    handoffs: Arc<tokio::sync::Mutex<()>>,
    eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    has_options: bool,
    /// Where `AMD` awaits the callee: the call's room, `None` on an offline (console) session.
    call_room: Option<Arc<dyn lk_agents_rtc::RtcRoom>>,
    /// The remote session protocol this session serves, if it does.
    remote: Option<Arc<crate::remote_session::driver::Hosting>>,
    /// The job whose room the session runs in.
    job: Option<Arc<dyn SessionJob>>,
    /// `session.usage`: every metrics record the host is handed, folded per model.
    usage: Arc<Mutex<lk_agents_abi::metrics::usage::ModelUsageCollector>>,
}

/// One stream, in emission order, so ordering between items, handoffs and completions needs no
/// barrier. Dropping the runner releases a waiting [`close`](SessionApi::close).
pub struct SessionRunner {
    events: Rx<HostEvent>,
    room_io: Arc<crate::room_io::RoomIo>,
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
    pub async fn next_host_event(&mut self) -> Option<HostEvent> {
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
        if let HostEvent::Session(session) = &ev {
            forward_to_room_io(&self.room_io, session).await;
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

/// What the room I/O mirrors: the agent's state edge (written as `lk.agent.state`) and the user's
/// transcript.
async fn forward_to_room_io(room_io: &Arc<crate::room_io::RoomIo>, ev: &SessionEvent) {
    match ev {
        SessionEvent::AgentStateChanged { new, .. } => room_io.on_agent_state_changed(new.as_str()),
        SessionEvent::UserInputTranscribed(t) => {
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
    #[error(transparent)]
    #[host(kind = "RuntimeError")]
    Rejected(#[from] crate::agent_activity::scheduling::Rejected),
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
/// without waking the selector).
#[lk_agents_abi::host(proxy)]
pub trait SessionEvents: Send + Sync {
    fn event(&self, wire: &HostWire);

    /// `str()` of the error record `error` is emitted with: the message the remote session sends.
    fn error_message(&self, error: &lk_agents_abi::voice::events::SessionErrorWire) -> String {
        error.error.to_string()
    }
}

/// The job a session runs in, as `start()` sequences it: the room connected beside the opening,
/// before the remote session host starts on it.
#[lk_agents_abi::host(proxy(cancel_on_drop))]
#[async_trait::async_trait]
pub trait SessionJob: Send + Sync {
    /// `JobContext.connect()`: the room connected, once however many callers.
    async fn connect(&self) -> Result<(), HostError>;
}

/// The agent's hooks, run by the host and awaited by the core: each hook call is answered with
/// what its method returns, or `Failed` when it raises (after `hook_failed`). A hook still running
/// when the loop has returned is cancelled.
#[lk_agents_abi::host(proxy(cancel_on_drop))]
#[async_trait::async_trait]
pub trait SessionHooks: Send + Sync {
    /// `Agent.on_enter()` for `agent`, the `Agent` the host installed.
    async fn on_enter(&self, agent: &lk_agents_abi::HostRef) -> Result<(), HostError>;
    /// `Agent.on_exit()`.
    async fn on_exit(&self, agent: &lk_agents_abi::HostRef) -> Result<(), HostError>;
    /// The toolsets of `agent` (after its `on_exit` was answered), or of the session (`None`, as
    /// it closes) closed.
    async fn close_toolsets(&self, agent: Option<&lk_agents_abi::HostRef>);
    /// The toolsets among `tools` (an agent's and the session's, a list) set up; the specs of every
    /// tool they hold then.
    async fn setup_tools(
        &self,
        tools: &lk_agents_abi::HostRef,
    ) -> Result<Vec<lk_agents_abi::ToolSpec>, HostError>;
    /// The session closes: what runs beside it (AMD) closes first.
    async fn closing(&self) -> Result<(), HostError>;
    /// `agent.on_user_turn_completed(turn_ctx, new_message)`, `turn_ctx` built from `chat_ctx`
    /// (a copy of the agent's own context): the message's text as the hook left it, the turn's
    /// items when it changed them, and whether it raised `StopResponse`.
    async fn on_user_turn_completed(
        &self,
        agent: &lk_agents_abi::HostRef,
        chat_ctx: Vec<lk_agents_abi::ChatItem>,
        new_transcript: &str,
        transcript_confidence: f64,
    ) -> Result<(Option<String>, Option<Vec<lk_agents_abi::ChatItem>>, bool), HostError>;
    /// The `function_tools_executed` event emitted: each output's `reply_required` and the event's
    /// `handoff_required`, as its handlers left them.
    async fn function_tools_executed(
        &self,
        function_calls: Vec<lk_agents_abi::ChatItem>,
        function_call_outputs: Vec<lk_agents_abi::ChatItem>,
        handoff_required: bool,
    ) -> Result<(Vec<bool>, bool), HostError>;
    /// A filler's callable `source` called for `step`, awaited when it returns an awaitable.
    async fn filler_source(
        &self,
        source: &lk_agents_abi::HostRef,
        step: u32,
    ) -> Result<FillerReply, HostError>;
    /// Hook `hook` raised `error` (the exception itself, when there is one): logged.
    fn hook_failed(&self, hook: &str, error: Option<lk_agents_abi::errors::CapturedError>);
}

/// What a filler's callable answered: its text to say, a speech it scheduled itself, or nothing.
pub enum FillerReply {
    Text(String),
    Spoke,
    Declined,
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for FillerReply {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods as _;
        Ok(if obj.is_none() {
            FillerReply::Declined
        } else if let Ok(text) = obj.extract::<String>() {
            FillerReply::Text(text)
        } else {
            FillerReply::Spoke
        })
    }
}

/// Runs the hook `call` asks for through `hooks` and answers it. Calls the host first, before any
/// other await, so a caller polling it once has handed the call to the host loop in event order.
async fn answer_hook(
    hooks: Arc<dyn SessionHooks>,
    call: crate::agent_session::host_events::HookCall,
    agents: HostAgents,
    api: Tx,
    spawner: Arc<dyn Spawner>,
) {
    use crate::agent::TurnEdit;
    use crate::agent_session::commands::{HookId, HookOutcome};
    use crate::agent_session::host_events::HookCall;
    let id = call.id();
    let kind = call.kind();
    let outcome: Result<HookOutcome, HostError> = match call {
        HookCall::OnEnter { agent, .. } => match agents.get(&agent) {
            Some(agent) => hooks.on_enter(&agent).await.map(|()| HookOutcome::Returned),
            None => Ok(HookOutcome::Failed),
        },
        HookCall::OnExit { agent, .. } => match agents.get(&agent) {
            Some(agent) => {
                let outcome = hooks.on_exit(&agent).await.map(|()| HookOutcome::Returned);
                // The toolsets close after the answer, and outlive the hook's cancellation.
                let hooks = Arc::clone(&hooks);
                spawner.spawn_unboxed(async move { hooks.close_toolsets(Some(&agent)).await });
                outcome
            }
            None => Ok(HookOutcome::Failed),
        },
        HookCall::OnUserTurnCompleted {
            agent, chat_ctx, new_transcript, transcript_confidence, ..
        } => match agents.get(&agent) {
            Some(agent) => hooks
                .on_user_turn_completed(&agent, chat_ctx, &new_transcript, transcript_confidence)
                .await
                .map(|(user_text, items, stop)| {
                    let mut edits = Vec::new();
                    if stop {
                        edits.push(TurnEdit::StopResponse);
                    }
                    if let Some(user_text) = user_text.filter(|t| *t != new_transcript) {
                        edits.push(TurnEdit::UpdateUserMessage(user_text));
                    }
                    if let Some(items) = items {
                        edits.push(TurnEdit::ReplaceItems(items));
                    }
                    HookOutcome::Turn(edits)
                }),
            None => Ok(HookOutcome::Failed),
        },
        HookCall::FunctionToolsExecuted { event, .. } => {
            use lk_agents_abi::ChatItem;
            let calls = event.function_calls.into_iter().map(ChatItem::FunctionCall).collect();
            let outputs =
                event.function_call_outputs.into_iter().map(ChatItem::FunctionCallOutput).collect();
            hooks.function_tools_executed(calls, outputs, event.handoff_required).await.map(
                |(reply_required, handoff_required)| HookOutcome::FunctionToolsExecuted {
                    reply_required,
                    handoff_required,
                },
            )
        }
        HookCall::FillerSource { source: None, .. } => {
            Ok(HookOutcome::Filler { text: None, spoke: false })
        }
        HookCall::FillerSource { source: Some(source), step, .. } => {
            hooks.filler_source(&source, step).await.map(|reply| match reply {
                FillerReply::Text(text) => HookOutcome::Filler { text: Some(text), spoke: false },
                FillerReply::Spoke => HookOutcome::Filler { text: None, spoke: true },
                FillerReply::Declined => HookOutcome::Filler { text: None, spoke: false },
            })
        }
        // Unanswered: the reference's templates.
        HookCall::RenderToolReply { .. } => Ok(HookOutcome::Failed),
    };
    let outcome = outcome.unwrap_or_else(|e| {
        hooks.hook_failed(kind.as_str(), e.host_ref.clone());
        HookOutcome::Failed
    });
    let _ = api.send_from(Origin::Host, Command::HookDone { id: HookId(id.0), outcome });
}

/// Timers and playout must share one clock: a fake room brings its own virtual clock, a host
/// `rtc.Room` is a [`HostRoom`] on the wall clock.
pub trait SessionRoom: Send + Sync {
    fn rtc(&self) -> Arc<dyn lk_agents_rtc::RtcRoom>;
    fn timebase(&self) -> Timebase;
    /// No call behind the room (`OfflineRoom`): the reference's session without a room io.
    fn offline(&self) -> bool {
        false
    }
}

/// A host `rtc.Room`, seen through `_core_rtc.Room`, whose `_core_rtc` wraps it.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.room_io.Room"))]
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

/// The host's `Agent` objects by agent id. The hooks and the handoff record name an agent by
/// id; the host reads the object it installed.
#[derive(Clone, Default)]
pub struct HostAgents(Arc<Mutex<std::collections::HashMap<String, lk_agents_abi::HostRef>>>);

impl HostAgents {
    pub fn install(&self, agent: &str, host: Option<lk_agents_abi::HostRef>) {
        if let Some(host) = host {
            self.0.lock().unwrap_or_else(|p| p.into_inner()).insert(agent.to_owned(), host);
        }
    }

    pub(crate) fn get(&self, agent: &str) -> Option<lk_agents_abi::HostRef> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).get(agent).cloned()
    }

    /// Fills in the `Agent`s of a handoff item's wire record.
    pub fn resolve(&self, wire: &mut lk_agents_abi::voice::events::HostWire) {
        use lk_agents_abi::voice::events::HostWire;
        if let HostWire::ItemAdded(w) = wire
            && let lk_agents_abi::ChatItem::AgentHandoff(h) = &w.item
        {
            w.old_agent_host = h.old_agent_id.as_deref().and_then(|a| self.get(a));
            w.new_agent_host = self.get(&h.new_agent_id);
        }
    }
}

/// An `Agent`'s overridden nodes, as the core calls them. `transcription_node` and
/// `realtime_audio_output_node` are built once at assembly over the core leg below (called under
/// the GIL; what one raises comes back as the exception it raised). `llm_node`, `tts_node` and
/// `stt_node` are opened per request in place of the provider ([`Node`](super::nodes)): one
/// completion, one segment's synthesis, one recognition stream.
#[lk_agents_abi::host(proxy(clock))]
#[async_trait::async_trait]
pub trait NodeFactory: Send + Sync {
    fn text_node(
        &self,
        below: crate::io::CoreTextOutput,
    ) -> Result<Arc<dyn crate::io::TextNode>, lk_agents_abi::errors::HostError>;
    fn audio_node(
        &self,
        below: crate::io::CoreAudioSink,
    ) -> Result<Arc<dyn crate::io::RealtimeAudioNode>, lk_agents_abi::errors::HostError>;
    /// `llm_node(chat_ctx, tools, model_settings)`: its chunks, strings and flushes as chunks.
    async fn llm_node(&self, req: &lk_agents_abi::ChatRequest) -> Result<LlmStream, ProviderError>;
    /// `tts_node(text, model_settings)`: text pushed in, its frames out.
    async fn tts_node(&self) -> Result<(TtsSink, TtsSource), ProviderError>;
    /// `stt_node(audio, model_settings)`: frames pushed in, its speech events out.
    async fn stt_node(&self) -> Result<(SttSink, SttSource), ProviderError>;
}

/// The agent a host `Agent` describes (`AgentActivity._frame_kwargs`), as `update_agent`,
/// Each provider is three-state: absent keeps the leg,
/// `None` removes it, an object replaces it. `chat_ctx` replaces when present. The tools are one
/// toolset, replaced together because the runner is per-agent. `turn_handling` is the agent layer.
/// `llm_node`, `tts_node` and `stt_node` are the agent's overrides, each over the leg given beside
/// it: a leg kept from the previous agent keeps that agent's node.
#[lk_agents_abi::host(kwargs)]
pub struct UpdateAgentArgs {
    pub instructions: Instructions,
    #[host(kwarg(default = None))]
    pub dynamic_instructions: Option<Arc<dyn crate::generation::HostInstructions>>,
    /// The new agent's id.
    pub agent: String,
    /// The new agent's `Agent`, handed back by the events that name it.
    #[host(kwarg(default = None))]
    pub agent_host: Option<lk_agents_abi::HostRef>,
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
    #[host(kwarg(default = None))]
    pub turn_handling: Option<crate::agent_activity::options::TurnHandlingKwargs>,
    /// The agent's `min_consecutive_speech_delay` in seconds; `None` defers to the session's.
    #[host(kwarg(default = None))]
    pub min_consecutive_speech_delay: Option<f64>,
    /// The agent's `use_tts_aligned_transcript`; `None` defers to the session's.
    #[host(kwarg(default = None))]
    pub use_tts_aligned_transcript: Option<bool>,
    /// The agent's `expressive`; `NOT_GIVEN` defers to the session's.
    pub expressive: Maybe<crate::agent_activity::options::Expressive>,
    #[host(kwarg(default = None))]
    pub llm_node: Option<Arc<dyn NodeFactory>>,
    #[host(kwarg(default = None))]
    pub tts_node: Option<Arc<dyn NodeFactory>>,
    #[host(kwarg(default = None))]
    pub stt_node: Option<Arc<dyn NodeFactory>>,
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
        runner,
    }
}

/// The objects are the pool a host-backed LLM is handed beside the agent's.
fn session_tools(
    tools: Option<Vec<lk_agents_abi::ToolSpec>>,
    host: Option<lk_agents_abi::HostRef>,
) -> Option<Toolset> {
    tools.map(|specs| Toolset { specs, host, runner: None })
}

/// LLM and TTS run at zero because the engine's `Attempt` loop owns their retries, and two loops
/// would multiply; a host STT keeps its budget, which its `RetryingStt` spends.
fn engine_retries() -> lk_agents_abi::ApiConnectOptions {
    lk_agents_abi::ApiConnectOptions { max_retry: 0, ..lk_agents_abi::ApiConnectOptions::default() }
}

/// A recognize-only STT goes over `vad` (`Agent.stt_node`'s default `StreamAdapter`).
fn stt_leg(stt: Arc<dyn Stt>, vad: Option<Arc<dyn Vad>>) -> Arc<dyn Stt> {
    let stt = stt.with_conn_options(&lk_agents_abi::ApiConnectOptions::default()).unwrap_or(stt);
    lk_agents_abi::stt::stream_adapter::SttStreamAdapter::adapted(stt, vad)
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
}

/// `start()`'s awaited half failed: the loop, or what the host does beside it (the room's
/// connection, the remote session's transport), which raises as itself.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, thiserror::Error)]
pub enum StartFailed {
    #[error("the session failed to start: {0}")]
    Driver(DriverError),
    #[error(transparent)]
    #[host(delegate)]
    Host(HostError),
}

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

/// The feeding end of a streamed `say`: what the speech reads, chunk by chunk. Dropping it, or
/// [`end`](Self::end), finishes the speech.
pub struct Pusher<T>(std::sync::Mutex<Option<tokio::sync::mpsc::Sender<T>>>);

impl<T: Send + 'static> Pusher<T> {
    /// Queue one item. A full channel waits rather than dropping (every chunk is spoken, in
    /// order). `false` once the speech is gone.
    pub async fn push(&self, item: T) -> bool {
        let tx = self.0.lock().unwrap_or_else(|p| p.into_inner()).clone();
        match tx {
            Some(tx) => tx.send(item).await.is_ok(),
            None => false,
        }
    }

    /// No more items. Idempotent.
    pub fn end(&self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).take();
    }

    /// Resolves once the speech stopped reading.
    async fn closed(&self) {
        let tx = self.0.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if let Some(tx) = tx {
            tx.closed().await;
        }
    }

    /// `source` read into the speech to its end, or until the speech is gone; then closed. A
    /// failing source ends the speech with what it gave.
    async fn feed(
        self,
        mut source: Box<dyn lk_agents_abi::StreamSource<T>>,
        clock: Arc<dyn Clock>,
    ) {
        loop {
            let next = tokio::select! {
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
        source.close(clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)).await;
    }
}

// After `UpdateAgentArgs`, whose kwargs macro the host methods expand against.
mod host;

/// The caller's side of awaited commands: a call gets an id for its command and a typed receiver,
/// settled from the session's answer stream.
type Settle = Box<dyn FnOnce(Answer) + Send>;

/// One session's open calls, shared by everyone who sends it commands.
#[derive(Clone, Default)]
pub struct Calls(Arc<Mutex<Table>>);

#[derive(Default)]
struct Table {
    next: u64,
    open: BTreeMap<RequestId, Settle>,
    closed: bool,
}

impl Calls {
    /// `pick` takes the answer's payload; an answer of another kind, or none before the session
    /// ends, closes the receiver.
    pub fn open<T: Send + 'static>(
        &self,
        pick: fn(Answer) -> Option<T>,
    ) -> (RequestId, oneshot::Receiver<T>) {
        let (tx, rx) = oneshot::channel();
        let call = self.insert(Box::new(move |answer| {
            if let Some(value) = pick(answer) {
                let _ = tx.send(value);
            }
        }));
        (call, rx)
    }

    /// A speech call: its answer marks `speech` created; another answer, or none before the
    /// session ends, marks it lost.
    pub fn open_speech(&self, speech: &Arc<SpeechHandle>) -> RequestId {
        struct Answering(Arc<SpeechHandle>);
        impl Drop for Answering {
            fn drop(&mut self) {
                self.0.answered(false);
            }
        }
        let answering = Answering(Arc::clone(speech));
        self.insert(Box::new(move |answer| {
            answering.0.answered(matches!(answer, Answer::Speech));
        }))
    }

    fn insert(&self, settle: Settle) -> RequestId {
        let mut table = self.lock();
        let call = RequestId(table.next);
        table.next += 1;
        if !table.closed {
            table.open.insert(call, settle);
        }
        call
    }


    /// A late or unknown call's answer is dropped.
    pub fn settle(&self, call: RequestId, answer: Answer) {
        let settle = self.lock().open.remove(&call);
        if let Some(settle) = settle {
            settle(answer);
        }
    }

    pub fn close(&self) {
        let open = {
            let mut table = self.lock();
            table.closed = true;
            std::mem::take(&mut table.open)
        };
        drop(open);
    }

    /// Settles every answer until the session's stream ends, then closes what is left.
    pub async fn pump(self, mut answers: Rx<(RequestId, Answer)>) {
        while let Some((call, answer)) = answers.recv().await {
            self.settle(call, answer);
        }
        self.close();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Table> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[cfg(test)]
mod tests;
