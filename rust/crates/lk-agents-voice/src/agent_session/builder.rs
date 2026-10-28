//! `DriverBuilder`: assembles a session's cores ([`SessionCores`]: its driver and data plane) from
//! its providers, legs and options. The [`SessionExecutor`] opens the streams and runs the tasks
//! that feed them.

use super::driver::*;
use crate::agent_activity::false_interruption::PausedSpeech;
use crate::agent_activity::scheduling::SpeechScheduler;
use crate::agent_activity::{AgentActivity, DriverConfig, PIPELINE_SAMPLE_RATE};
use crate::agent_session::SessionShared;
use crate::agent_session::VoiceSession;
use crate::agent_session::commands::Origin;
use crate::agent_session::host_events::HostEvent;
use crate::agent_session::time::{Now, Timebase};
use crate::audio::AudioChannel;
use crate::audio_recognition::AudioRecognition;
use crate::data_plane::{AudioForwarder, ForwarderCommand};
use crate::generation::Instructions;
use crate::trace::{CoreStep, Journal, RealtimeStep, Step};
use lk_agents_abi::resources::{ResourceId, Resources};
use lk_agents_abi::utils::frame::FrameTable;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, TextOutput};
use lk_agents_abi::voice::rtc::RoomEvent;
use lk_agents_abi::{
    AudioFrame, EotDetector, FrameId, Llm, Spawner, StreamSink, Stt, TimerSet, Ts, Tts, Vad,
};
use std::sync::Arc;
use std::time::Duration;

/// A started session's two cores, the decision core ([`Driver`]) and the data plane
/// ([`AudioForwarder`]), with what performing their effects needs. The cores decide and the queue
/// owner performs: the opening effects are still in the driver's outbox.
pub struct SessionCores {
    pub driver: Driver,
    pub forwarder: AudioForwarder,
    /// Every source's events, in arrival order, and where the effects' reports join them.
    pub queue: (Tx, Rx<Event>),
    /// The microphone the session starts with.
    pub source: Option<ResourceId>,
    /// The turn detector, asked for predictions by the driver's effects.
    pub eot: Option<Arc<dyn EotDetector>>,
    /// The interruption detector, whose verdicts the executor pumps into the queue.
    pub interruption: Option<Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>>,
    /// The room's events and the room IO's, forwarded into the queue until `Closing`.
    pub room_events:
        (Option<Rx<RoomEvent>>, Option<Arc<dyn lk_agents_abi::voice::room_owner::RoomIo>>),
    /// The camera the runtime reads from the start.
    pub video: Option<Arc<dyn lk_agents_abi::voice::io::VideoInput>>,
    pub shared: Arc<SessionShared>,
    /// Where host events and taps go.
    pub peer: PeerHandle,
    pub storage: Option<Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>>,
}

/// How a TTS with no duplex path is made to stream: sentence by sentence on the session's clock,
/// each sentence retried within the budget. The executor supplies it with its speech text.
pub type TtsAdapter = Arc<
    dyn Fn(
            Arc<dyn Tts>,
            Arc<dyn lk_agents_abi::Clock>,
            lk_agents_abi::ApiConnectOptions,
        ) -> Arc<dyn Tts>
        + Send
        + Sync,
>;

/// How a recognize-only STT is made to stream over the session's VAD (`Agent.stt_node`'s default
/// `StreamAdapter`): each utterance the VAD cuts recognized once it ends. A streaming STT, or one
/// with no VAD, is kept as it is. The executor supplies it.
pub type SttAdapter = Arc<
    dyn Fn(Arc<dyn Stt>, Option<Arc<dyn Vad>>, Arc<dyn lk_agents_abi::Clock>) -> Arc<dyn Stt>
        + Send
        + Sync,
>;

/// What runs a session: the production runtime, or a simulation's own executor. It opens the
/// opening STT and VAD streams before the cores exist, then takes the cores over.
pub trait SessionExecutor: Send {
    /// Adapts each TTS handed over, at start and on every later swap.
    fn tts_adapter(&self) -> TtsAdapter;

    /// Adapts each STT handed over, at start and on every later swap.
    fn stt_adapter(&self) -> SttAdapter;

    /// Opens the streams and registers their sinks in `resources`; their sources are read once
    /// the executor owns the cores.
    fn open_legs<'a>(
        &'a mut self,
        resources: &'a Resources,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<OpeningLegs, DriverError>>;

    /// Takes the started cores over.
    fn own(self: Box<Self>, cores: SessionCores) -> Box<dyn QueueOwner>;
}

/// The opening streams' sinks, registered.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpeningLegs {
    pub stt: Option<ResourceId>,
    pub vad: Option<ResourceId>,
}

/// What steps a session's cores once its executor owns them.
pub trait QueueOwner: Send {
    /// The opening cycle, at the instant the session opens.
    fn open(&mut self, now: Now);
    /// The loop that steps the cores from then on; `None` when the owner steps them itself.
    fn run(self: Box<Self>) -> Option<lk_agents_abi::BoxFuture<'static, ()>>;
}

pub struct DriverBuilder {
    pub view: tokio::sync::watch::Sender<Arc<crate::agent_session::SessionView>>,
    pub time: Timebase,
    pub spawner: Arc<dyn Spawner>,
    pub config: DriverConfig,
    pub agent: String,
    pub instructions: Instructions,
    pub dynamic_instructions: Option<Arc<dyn crate::generation::HostInstructions>>,
    /// The opening agent's `transcription_node` and `realtime_audio_output_node`.
    pub output_nodes: crate::agent_session::nodes::OutputNodes,
    /// The session starts from the agent's constructor context. Seeded before the instructions, so
    /// the system message lands in it.
    pub chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
    /// `Some(bound)`: the hooks the sequence needs wait up to `bound`; an overrun is journalled and
    /// skipped. `None`: each hook is finished in place.
    pub hook_bound: Option<Duration>,
    /// Opened before the opening agent's handoff record, so that record and the opening `on_enter`
    /// are the run's.
    pub capture_run: Option<crate::run_result::RunSpec>,
    /// Consecutive unrecoverable errors of one leg tolerated before closing (default `3`).
    pub max_unrecoverable_errors: u32,
    /// `conn_options=`: the budgets the engine retries the LLM and the TTS on.
    pub conn: lk_agents_abi::voice::session_options::SessionConnOptions,
    /// `expressive=`, the session's and the opening agent's.
    pub expressive: crate::generation::ExpressiveLayers,
    /// Where the providers below are registered, and every one handed over later.
    pub resources: lk_agents_abi::resources::Resources,
    pub llm: Option<Arc<dyn Llm>>,
    /// Adapted at start by the executor's [`TtsAdapter`], under `tts_node`.
    pub tts: Option<Arc<dyn Tts>>,
    pub tts_node: Option<Arc<dyn super::api::NodeFactory>>,
    /// Adapted at start by the executor's [`SttAdapter`] over `vad`, under `stt_node`.
    pub stt: Option<Arc<dyn Stt>>,
    pub stt_node: Option<Arc<dyn super::api::NodeFactory>>,
    /// `None`: nothing is pushed.
    pub keyterms: Option<crate::keyterm_detection::KeytermSetup>,
    /// `stt_context_options["forward_chat_context"]`: conversation turns reach an STT with the
    /// `chat_context` capability.
    pub forward_chat_context: bool,
    pub vad: Option<Arc<dyn Vad>>,
    pub audio_in: Option<Arc<dyn AudioInput>>,
    pub video_in: Option<Arc<dyn lk_agents_abi::voice::io::VideoInput>>,
    pub recorder: Option<Arc<dyn lk_agents_abi::recorder::AudioRecorder>>,
    pub audio_out: Option<Arc<dyn AudioOutput>>,
    pub text_out: Option<Arc<dyn TextOutput>>,

    pub room_events: Option<Rx<RoomEvent>>,
    pub room_io_events: Option<Arc<dyn lk_agents_abi::voice::room_owner::RoomIo>>,
    pub room_link: Option<crate::room_io::RoomLinkConfig>,
    /// `None` commits turns on silence alone.
    pub eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// `None` is VAD-only interruption: every burst past `min_interruption_duration`, a "mm-hm"
    /// included, stops the agent.
    pub interruption: Option<Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>>,
    /// `None` forwards **every** frame the host pushes.
    pub video_sampler: Option<lk_agents_voice::room_io::video_sampler::VideoSampler>,
    /// Instead of an LLM+TTS pair: `llm`/`tts` are ignored while it is set.
    pub realtime: Option<Arc<dyn lk_agents_abi::Realtime>>,
    pub toolset: crate::tool_executor::Toolset,
    /// Prefixes every chat item id the session mints, so items from two calls never collide.
    pub session_id: String,
    pub storage: Option<Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>>,
    /// Where the driver's steps go: nowhere in production.
    pub journal: Journal,
    /// Test-only: what a simulation installed before `start()`.
    #[cfg(feature = "dst")]
    pub probe: Option<super::HostEventProbe>,
}

impl DriverBuilder {
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    /// `async` because the executor opens the providers' streams first.
    pub async fn start(
        mut self,
        executor: &mut dyn SessionExecutor,
    ) -> Result<(SessionCores, SessionHandle), DriverError> {
        // `agent_session` and `session_start` run from here.
        let built = self.time.wall_secs();
        let keyterms = self.keyterms.take().map(|mut setup| {
            let llm = setup.llm.take().map(|llm| self.resources.register_llm(llm).id);
            (setup, llm)
        });
        let forward_chat_context = self.forward_chat_context;
        // An advertised tool with nothing to run it is refused here, not at the first call.
        if !self.toolset.specs.is_empty() && self.toolset.runner.is_none() {
            return Err(DriverError::Missing("tool_runner"));
        }
        if let Some(llm) = &self.llm {
            llm.prewarm();
        }
        let nodes =
            super::nodes::Nodes { llm: None, tts: self.tts_node.take(), stt: self.stt_node.take() };
        let adapter = executor.stt_adapter();
        self.stt = self.stt.take().map(|stt| {
            let vad = self.vad.clone();
            nodes.stt(super::api::stt_leg(stt, &adapter, vad, self.time.clock(), &self.conn.stt))
        });
        if let Some(stt) = &self.stt {
            stt.prewarm();
        }
        let adapter = executor.tts_adapter();
        self.tts = self.tts.take().map(|tts| {
            nodes.tts(super::api::tts_leg(tts, &adapter, self.time.clock(), &self.conn.tts))
        });
        if let Some(tts) = &self.tts {
            tts.prewarm();
        }
        let (queue_tx, queue_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let queue = Tx::new(queue_tx);
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<HostEvent>();
        let watch = WatchHub::new();
        let closing = tokio::sync::watch::channel(false).0;

        let mut sinks = Sinks {
            by_channel: std::collections::BTreeMap::new(),
            forwarder: AudioForwarder::new(PIPELINE_SAMPLE_RATE),
        };
        let opened = executor
            .open_legs(
                &self.resources,
                self.time.clone(),
                Arc::clone(&self.spawner),
                self.stt.clone(),
                self.vad.clone(),
            )
            .await?;
        self.install_legs(&mut sinks, opened);
        let realtime = self.open_realtime(&mut sinks, &queue).await?;
        let Sinks { by_channel: sinks, forwarder } = sinks;

        // The microphone, read by the runtime.
        let audio_in =
            self.audio_in.clone().map(|input| self.resources.register_audio_input(input));
        let resources = &self.resources;
        let llm = self.llm.clone().map(|l| resources.register_llm(l));
        let tts = self.tts.clone().map(|t| resources.register_tts(t));
        let stt = self.stt.clone().map(|s| resources.register_stt(s));
        let vad = self.vad.clone().map(|v| resources.register_vad(v));
        let outputs = crate::agent_session::Outputs::new(
            self.audio_out.clone().map(|o| resources.register_audio_output(o)),
            self.text_out.clone().map(|o| resources.register_text_output(o)),
        );
        let shared = self.shared(&queue, closing, outputs.clone());

        let paused_speech = {
            // The sink's facts at start; re-read at every pause decision.
            let (sink, audio_enabled) = outputs.pause_facts();
            PausedSpeech::new(&self.config.effective, sink, audio_enabled)
        };
        // The constructor context as the agent's view, not announced (the host has it). The prompt
        // is rendered per request.
        let mut session = VoiceSession::new(
            self.config.clone(),
            crate::agent_session::SessionParts {
                llm,
                tts,
                stt,
                vad,
                outputs,
                max_unrecoverable_errors: self.max_unrecoverable_errors,
                start_recorder: self.recorder.take(),
                room_link: self.room_link.take(),
                agent: self.agent.clone(),
                chat_ctx: self.chat_ctx.clone(),
                session_id: self.session_id.clone(),
                storage_enabled: self.storage.is_some(),
            },
        );
        session.expressive = self.expressive.clone();
        let journal = self.journal.clone();
        if journal.is_on() {
            journal.record(Step::Core(CoreStep::ConfigApplied {
                layer: crate::agent_activity::options::ConfigLayer::Session,
                effective: Box::new(session.config.effective),
            }));
            // The wall clock at `Ts::ZERO`: the session may already be past it.
            let since_zero = self.time.now().saturating_duration_since(Ts::ZERO);
            journal.record(Step::Core(CoreStep::WallEpoch {
                at: self.time.wall().checked_sub(since_zero).unwrap_or(std::time::UNIX_EPOCH),
            }));
            if let Some(realtime) = &realtime {
                let turn_detection_disabled = realtime.disabled;
                journal.record(Step::Realtime(RealtimeStep::Opened { turn_detection_disabled }));
            }
        }

        let resolved_mode = match &realtime {
            Some(realtime) => realtime.resolved,
            None => session.config.effective.turn.owner(),
        };
        let requested_detection = session.config.effective.turn.detection();
        let input = crate::agent_activity::io::AgentInput::new(
            sinks,
            self.video_in.as_ref().map(|input| Arc::as_ptr(input).cast::<()>() as usize),
            self.video_sampler.take(),
        );
        let time = self.time.clone();
        let capture_run = self.capture_run.take();
        let interruption = self.interruption.take();
        let eot = self.eot.take();
        let activity = self.activity(
            eot.as_ref(),
            realtime,
            paused_speech,
            resolved_mode,
            requested_detection,
        );
        let mut driver = Driver {
            activity_changed: Default::default(),
            outbox: Vec::new(),
            view: Arc::clone(&self.view.borrow()),
            session,
            activity,
            input,
            current: FrameId::ROOT,
            timers: TimerSet::new(),
            journalled: journal.is_on(),
            queue: queue.clone(),
            command_origin: Origin::Host,
            step: crate::agent_session::time::StepClock::new(),
            stopped: false,
            opened: false,
        };
        driver.session.trace.built = built;
        // What the session opens with is released like anything handed over later.
        driver.session.providers = driver.providers_held();
        // The opening cycles run at the instant the session is built.
        driver.step.set(time.read());
        driver.publish_activity();
        driver.flush_activity();

        driver.open_session(capture_run, forward_chat_context, keyterms);
        // `ivr_detection=True` is the IVR system's to start; the core starts no system.
        let ivr_detection = driver.session.config.effective.turn.ivr_detection();
        let keyterms_pushed = shared.keyterms.subscribe();
        let chat_ctx = shared.chat_ctx.subscribe();
        let speech_ids = driver.session.ids.speech_ids();
        let peer = PeerHandle {
            queue: queue.clone(),
            events: ev_tx,
            journal,
            time,
            spawner: Arc::clone(&self.spawner),
            watch,
            resources: self.resources.clone(),
            closing: shared.closing.subscribe(),
        };

        let cores = SessionCores {
            driver,
            forwarder,
            queue: (queue.clone(), queue_rx),
            source: audio_in,
            eot,
            interruption,
            room_events: (self.room_events.take(), self.room_io_events.take()),
            video: self.video_in.take(),
            shared,
            peer: peer.clone(),
            storage: self.storage.take(),
        };
        Ok((
            cores,
            SessionHandle {
                api: queue,
                events: ev_rx,
                keyterms: keyterms_pushed,
                chat_ctx,
                peer,
                ivr_detection,
                speech_ids,
            },
        ))
    }

    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    /// What the spawned tasks share with the loop.
    fn shared(
        &self,
        queue: &Tx,
        closing: tokio::sync::watch::Sender<bool>,
        outputs: crate::agent_session::Outputs,
    ) -> Arc<SessionShared> {
        Arc::new(SessionShared {
            view: self.view.clone(),
            time: self.time.clone(),
            output: tokio::sync::watch::channel(outputs).0,
            recorder: Arc::new(std::sync::RwLock::new(None)),
            tts_transforms: lk_agents_voice::transcription::TransformSpec::of_set(
                self.config.effective.speech.tts_text_transforms(),
            ),
            conn: self.conn,
            queue: queue.clone(),
            closing,
            draining: tokio::sync::watch::channel(false).0,
            hook_host: self.hook_bound.is_some(),
            spawner: Arc::clone(&self.spawner),
            frames: Arc::new(FrameTable::new()),
            gen_ids: std::sync::atomic::AtomicU64::new(0),
            session_id: self.session_id.clone(),
            resources: self.resources.clone(),
            chat_ctx: tokio::sync::watch::channel(Arc::new(lk_agents_abi::ChatContext::default()))
                .0,
            keyterms: tokio::sync::watch::channel(None).0,
            journal: self.journal.clone(),
        })
    }

    /// The opening streams' sinks and the detectors' audio, installed. The detector's presence
    /// has one source, the session's `eot`.
    fn install_legs(&mut self, sinks: &mut Sinks, opened: OpeningLegs) {
        if let Some(sink) = opened.stt {
            sinks.insert(AudioChannel::Stt, sink);
        }
        if let Some(sink) = opened.vad {
            sinks.insert(AudioChannel::Vad, sink);
        }
        {
            use crate::agent_activity::options::DetectorKind;
            let eot = self.eot.is_some();
            self.config.edit(|_, legs| legs.detector = eot.then_some(DetectorKind::Streaming));
        }
        if let Some(detector) = &self.eot {
            let audio = Arc::new(EotAudioSink(Arc::clone(detector)));
            sinks.install(&self.resources, AudioChannel::Eot, audio);
        }
        // The interruption detector's verdicts and audio leg come off one object, so neither
        // exists without the other.
        if let Some(detector) = &self.interruption {
            sinks.install(&self.resources, AudioChannel::Interruption, detector.audio());
        }
    }

    /// The realtime session, opened once for the driver's life (a handoff updates it).
    async fn open_realtime(
        &mut self,
        sinks: &mut Sinks,
        queue: &Tx,
    ) -> Result<Option<RealtimeStart>, DriverError> {
        let Some(model) = &self.realtime else {
            return Ok(None);
        };
        // Who ends the turn, decided once from the model's capabilities and the session's
        // inputs (`_resolve_rt_turn_detection_enabled`).
        let caps = model.capabilities();
        let side = crate::agent_activity::realtime::SessionSide {
            stt: self.stt.is_some(),
            vad: self.vad.is_some(),
            detection: self.config.effective.turn.detection(),
            streaming_detector: self.config.legs.detector
                == Some(crate::agent_activity::options::DetectorKind::Streaming),
            interruption_given: self.config.effective.turn.interruption.mode.is_some(),
            turn_detection_mode: self.config.effective.turn.owner(),
        };
        let disabled = crate::agent_activity::realtime::turn_detection_disabled(&caps, side);
        // The refined mode goes into the config; the pre-refinement answer is kept so a later
        // re-resolution is compared against it.
        let owner =
            crate::agent_activity::realtime::realtime_turn_detection_mode(&caps, side, disabled);
        self.config.effective.turn.owner = Some(owner);
        let session = model.session(disabled).await?;
        let sink = crate::agent_activity::realtime::RealtimeAudioSink::new(Arc::clone(&session));
        sinks.install(&self.resources, AudioChannel::Realtime, Arc::new(sink));
        // The opening agent is sent ungated by `mutable_*` (they govern mid-session changes)
        // through `push_frame` once the driver exists.
        crate::agent_activity::realtime::spawn_event_pump(
            &self.spawner,
            Arc::clone(&session),
            queue.clone(),
        );
        Ok(Some(RealtimeStart { session, caps, disabled, resolved: side.turn_detection_mode }))
    }

    /// The opening agent's activity: nothing spoken, nothing pending.
    fn activity(
        &mut self,
        eot: Option<&Arc<dyn EotDetector>>,
        realtime: Option<RealtimeStart>,
        paused_speech: PausedSpeech,
        resolved_mode: crate::audio_recognition::TurnDetectionMode,
        requested_detection: lk_agents_abi::voice::session_options::TurnDetection,
    ) -> AgentActivity {
        let effective = &self.config.effective;
        let mut audio_recognition = AudioRecognition::new(effective, &self.config.legs);
        audio_recognition.set_thresholds(eot.and_then(|detector| detector.thresholds()));
        // `_adaptive_interruption_active` is "a detector was supplied"; without it the sentinels
        // are unreachable.
        audio_recognition.set_interruption_active(self.config.legs.adaptive_detector);
        audio_recognition.set_eot_provider(eot.map(|detector| detector.id()));
        let (rt_session, caps, disabled) = match realtime {
            Some(rt) => (Some(self.resources.register_realtime(rt.session)), rt.caps, rt.disabled),
            None => (None, Default::default(), false),
        };
        AgentActivity {
            agent: std::mem::take(&mut self.agent),
            instructions: std::mem::take(&mut self.instructions),
            dynamic_instructions: self.dynamic_instructions.take(),
            output_nodes: std::mem::take(&mut self.output_nodes),
            toolset: Arc::new(std::mem::take(&mut self.toolset)),
            tools: Default::default(),
            audio_recognition,
            speech_scheduler: SpeechScheduler::new(),
            paused_speech,
            speech_agents: Default::default(),
            finished: Default::default(),
            tools_committed: Default::default(),
            reply_plans: Default::default(),
            tools_in_flight: Default::default(),
            spoke: Default::default(),
            announced_current_speech: None,
            interrupt_waiters: Vec::new(),
            hook_bound: self.hook_bound,
            hooks: Default::default(),
            handoff: Default::default(),
            on_exit: Default::default(),
            drain: Default::default(),
            voice_tasks: Default::default(),
            dtmf: Default::default(),
            end_call: None,
            pending_turn: None,
            turn_limit: Default::default(),
            waiting_turns: Default::default(),
            last_turn: None,
            live_speech_at_final: false,
            preemptive_turn: None,
            preemptive_record_version: 0,
            rt_session,
            realtime_state: crate::agent_activity::realtime::RealtimeSessionState::new(caps),
            realtime_players: Default::default(),
            launching: Default::default(),
            realtime_turn_detection_disabled: disabled,
            realtime_resolved_mode: resolved_mode,
            realtime_requested_detection: requested_detection,
        }
    }
}

impl Driver {
    /// The opening cycles: the first facts, in order, the gate, then the opening agent's records
    /// (the handoff from nobody, the starting config), and the opening agent on the realtime
    /// socket; then the static keyterms.
    fn open_session(
        &mut self,
        capture_run: Option<crate::run_result::RunSpec>,
        forward_chat_context: bool,
        keyterms: Option<(
            crate::keyterm_detection::KeytermSetup,
            Option<lk_agents_abi::resources::ResourceId>,
        )>,
    ) {
        self.publish_gate();
        {
            let mut out = self.out();
            // `start(capture_run=True)`'s run exists before the opening handoff is recorded, so the
            // handoff and the opening `on_enter` are the run's.
            if let Some(spec) = capture_run {
                out.trace(CoreStep::CommandReceived { origin: Origin::Host, kind: "Command::Run" });
                self.on_run(self.step.now, spec, &mut out);
            }
            self.record_handoff(None, &mut out);
            if self.activity.rt_session.is_some() {
                let opening = self.session.chat_ctx.for_realtime();
                self.push_activity(Some(opening), &mut out);
            }
            let effects = self.settle_step(out);
            self.outbox.extend(effects);
        }
        self.session.forward_chat_context = forward_chat_context;
        // `KeytermDetector.start`: the static terms reach the recogniser now.
        if let Some((setup, llm)) = keyterms {
            let mut out = self.out();
            self.keyterms_start(setup, llm, &mut out);
            let effects = self.settle_step(out);
            self.outbox.extend(effects);
        }
    }
}

/// The sinks the microphone feeds, by channel: the decision core and the forwarder each learn
/// every one installed at start.
struct Sinks {
    by_channel: std::collections::BTreeMap<AudioChannel, lk_agents_abi::resources::ResourceId>,
    forwarder: AudioForwarder,
}

impl Sinks {
    fn install(
        &mut self,
        resources: &Resources,
        channel: AudioChannel,
        sink: Arc<dyn StreamSink<AudioFrame>>,
    ) {
        self.insert(channel, resources.register_audio_sink(sink));
    }

    fn insert(&mut self, channel: AudioChannel, sink: ResourceId) {
        self.by_channel.insert(channel, sink);
        // Nothing is replaced yet, so there is nothing to perform.
        let _ = self.forwarder.handle(ForwarderCommand::Install { channel, sink });
    }
}

/// The turn detector's audio leg: every frame is `push_audio`. `flush` is not the detector's
/// `flush`, which is a turn boundary the machine decides.
struct EotAudioSink(Arc<dyn EotDetector>);

impl StreamSink<AudioFrame> for EotAudioSink {
    fn push(&self, frame: AudioFrame) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.0.push_audio(frame);
        Ok(())
    }

    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        // The detector's stream outlives a handoff.
    }
}

/// The realtime session opened at start, and who ends the turn under it.
struct RealtimeStart {
    session: Arc<dyn lk_agents_abi::RealtimeSession>,
    caps: lk_agents_abi::llm::realtime::RealtimeCapabilities,
    disabled: bool,
    /// The turn owner before the model's refinement, which a later re-resolution compares with.
    resolved: crate::audio_recognition::TurnDetectionMode,
}
