//! `DriverBuilder`: assembles a session's cores ([`SessionCores`]: its driver and data plane) from
//! its providers, legs and options. The [`SessionExecutor`] opens the streams and runs the tasks
//! that feed them.

/// A realtime session and its registered resource identity.
pub type RealtimeLeg = (Arc<dyn lk_agents_abi::RealtimeSession>, ResourceId);

use super::calls::Calls;
use super::shared::{Journal, PeerHandle, Rx, SessionHandle, SessionShared, WatchHub};
use super::time::Timebase;
use crate::telemetry;
use crate::voice::VoiceRuntimeEvent;
use crate::voice::resources::Resources;
use crate::voice::session::Tx;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, TextOutput};
use lk_agents_abi::voice::rtc::RoomEvent;
use lk_agents_abi::{AudioFrame, EotDetector, Llm, Spawner, StreamInput, Stt, Tts, Vad};
use lk_agents_voice::agent_activity::{DriverConfig, PIPELINE_SAMPLE_RATE};
use lk_agents_voice::agent_session::{
    DriverError, InitialSession, Now, SessionParts, VoiceDriver, VoiceHostEvent,
};
use lk_agents_voice::audio::AudioChannel;
use lk_agents_voice::data_plane::{AudioInputRouter, AudioInputRouterCommand};
use lk_agents_voice::generation::Instructions;
use std::sync::Arc;
use std::time::Duration;

/// A started session's two cores, the decision core ([`VoiceDriver`]) and the data plane
/// ([`AudioInputRouter`]), with what performing their effects needs. The cores decide and the queue
/// owner performs: the opening effects are still in the driver's outbox.
pub struct SessionCores {
    pub(crate) publications: super::shared::Publications,
    pub driver: VoiceDriver,
    pub forwarder: AudioInputRouter,
    /// Every source's events, in arrival order, and where the effects' reports join them.
    pub queue: (Tx, Rx<VoiceRuntimeEvent>),
    /// The microphone the session starts with.
    pub source: Option<ResourceId>,
    /// The turn detector, asked for predictions by the driver's effects.
    pub eot: Option<Arc<dyn EotDetector>>,
    /// The interruption detector, whose verdicts the executor pumps into the queue.
    pub interruption:
        Option<Arc<dyn lk_agents_voice::agent_activity::false_interruption::InterruptionPort>>,
    /// The opened realtime session whose events are read after the executor adopts the cores.
    pub realtime_events: Option<Arc<dyn lk_agents_abi::RealtimeSession>>,
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
    /// `cx`: the first activity the STT stream runs under (`start_agent_activity`).
    fn open_legs<'a>(
        &'a mut self,
        resources: &'a Resources,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
        cx: telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'a, Result<OpeningLegs, DriverError>>;

    /// Opens the model with the core's turn-detection choice and registers its audio sink.
    fn open_realtime<'a>(
        &'a mut self,
        resources: &'a Resources,
        model: Arc<dyn lk_agents_abi::Realtime>,
        disabled: bool,
    ) -> lk_agents_abi::BoxFuture<'a, Result<RealtimeLeg, DriverError>>;

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
    pub(crate) tool_bodies: super::calls::ToolBodies,
    /// Every speech's live handle, for the session's whole life.
    pub speeches: Arc<crate::voice::speech_handle::Speeches>,
    pub view: tokio::sync::watch::Sender<Arc<lk_agents_voice::agent_session::SessionView>>,
    pub time: Timebase,
    pub spawner: Arc<dyn Spawner>,
    pub config: DriverConfig,
    pub agent: String,
    pub instructions: Instructions,
    pub dynamic_instructions: Option<Arc<dyn lk_agents_voice::generation::HostInstructions>>,
    /// The opening agent's `transcription_node`.
    pub output_nodes: crate::voice::nodes::OutputNodes,
    /// The session starts from the agent's constructor context. Seeded before the instructions, so
    /// the system message lands in it.
    pub chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
    /// `Some(bound)`: the hooks the sequence needs wait up to `bound`; an overrun is journalled and
    /// skipped. `None`: each hook is finished in place.
    pub hook_bound: Option<Duration>,
    /// Consecutive unrecoverable errors of one leg tolerated before closing (default `3`).
    pub max_unrecoverable_errors: u32,
    /// `conn_options=`: the budgets the engine retries the LLM and the TTS on.
    pub conn: lk_agents_abi::voice::session_options::SessionConnOptions,
    /// `expressive=`, the session's and the opening agent's.
    pub expressive: lk_agents_voice::generation::ExpressiveLayers,
    /// Where the providers below are registered, and every one handed over later.
    pub resources: crate::voice::resources::Resources,
    pub llm: Option<Arc<dyn Llm>>,
    /// The runtime hands it over as the session runs it: adapted, under the agent's `tts_node`.
    pub tts: Option<Arc<dyn Tts>>,
    /// The runtime hands it over as the session runs it: adapted over `vad`, under `stt_node`.
    pub stt: Option<Arc<dyn Stt>>,
    /// `None`: nothing is pushed.
    pub keyterms: Option<lk_agents_voice::keyterm_detection::KeytermSetup>,
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
    pub room_link: Option<lk_agents_voice::agent_session::room::RoomLinkConfig>,
    /// `None` commits turns on silence alone.
    pub eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// `None` is VAD-only interruption: every burst past `min_interruption_duration`, a "mm-hm"
    /// included, stops the agent.
    pub interruption:
        Option<Arc<dyn lk_agents_voice::agent_activity::false_interruption::InterruptionPort>>,
    /// `None` forwards **every** frame the host pushes.
    pub video_sampler: Option<lk_agents_voice::agent_activity::video_sampler::VideoSampler>,
    /// Instead of an LLM+TTS pair: `llm`/`tts` are ignored while it is set.
    pub realtime: Option<Arc<dyn lk_agents_abi::Realtime>>,
    pub toolset: lk_agents_voice::tool_executor::Toolset,
    /// Prefixes every chat item id the session mints, so items from two calls never collide.
    /// `None` mints one at start.
    pub session_id: Option<String>,
    pub storage: Option<Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>>,
    /// Where the driver's steps go: nowhere in production.
    pub journal: Journal,
    /// Where the session's spans and measurements go: the process tracer in production.
    pub tracer: telemetry::Tracer,
    /// The span the session continues, the job it records for and its conversation.
    pub trace: telemetry::Cx,
    /// The job's ids, which `agent_session` repeats.
    pub job_ids: telemetry::JobIds,
    /// The session's spans reserved at `start()`; drawn at the opening when `None`.
    pub opening: Option<telemetry::Opening>,
    /// Where the loop leaves the session's report as it closes.
    pub report: FinalReport,
}

/// The session's report as its loop closed.
pub type FinalReport = Arc<std::sync::OnceLock<telemetry::SessionReport>>;

impl DriverBuilder {
    /// A session of `agent` with `config` and nothing else: no provider, leg, node or history.
    pub fn new(
        agent: String,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
        config: DriverConfig,
    ) -> Self {
        let view = lk_agents_voice::agent_session::SessionView {
            launch_holds: Default::default(),
            config: config.clone(),
            agent: agent.clone(),
            agent_scope: 0,
            vad: None,
            usage: Default::default(),
            usage_revision: 0,
        };
        let tracer = telemetry::Tracer::disabled(time.clock());
        DriverBuilder {
            tool_bodies: Default::default(),
            speeches: Default::default(),
            view: tokio::sync::watch::Sender::new(Arc::new(view)),
            time,
            spawner,
            config,
            agent,
            instructions: Instructions::default(),
            dynamic_instructions: None,
            output_nodes: Default::default(),
            chat_ctx: None,
            hook_bound: None,
            max_unrecoverable_errors: 3,
            conn: Default::default(),
            expressive: Default::default(),
            resources: Default::default(),
            llm: None,
            tts: None,
            stt: None,
            keyterms: None,
            forward_chat_context: true,
            vad: None,
            audio_in: None,
            video_in: None,
            recorder: None,
            audio_out: None,
            text_out: None,
            room_events: None,
            room_io_events: None,
            room_link: None,
            eot: None,
            interruption: None,
            video_sampler: None,
            realtime: None,
            toolset: Default::default(),
            session_id: None,
            storage: None,
            journal: Default::default(),
            trace: tracer.root(),
            tracer,
            job_ids: Default::default(),
            opening: None,
            report: Default::default(),
        }
    }

    /// The session's startup spans: those `start()` reserved, else reserved now.
    pub fn opening(&mut self) -> telemetry::Opening {
        *self.opening.get_or_insert_with(|| {
            let cx = &self.trace;
            let session = telemetry::reserve::<telemetry::AgentSession>(cx);
            let start = telemetry::reserve(&cx.clone().under::<telemetry::AgentSession>(session.context));
            telemetry::Opening { session, start, activity: None }
        })
    }

    /// The context startup work runs under: the session's `session_start`.
    pub fn startup(&mut self) -> telemetry::Cx {
        let start = self.opening().start.context;
        self.trace.clone().under::<telemetry::SessionStart>(start)
    }

    /// `gen_ai.conversation.id`: the call's thread, else the job's room session id; the session's
    /// own id only outside a job.
    fn conversation(&self, session_id: &str) -> String {
        self.trace.conversation().unwrap_or(session_id).to_owned()
    }

    /// The session's context: what its systems' spans (AMD's, a handoff's) open under.
    pub(crate) fn session_cx(&mut self, session_id: &str) -> telemetry::Cx {
        let root = self.opening().session.context;
        let conversation = self.conversation(session_id);
        self.trace
            .clone()
            .with_conversation(Some(conversation.as_str()))
            .under::<telemetry::AgentSession>(root)
    }

    /// `async` because the executor opens the providers' streams first.
    pub async fn start(
        mut self,
        executor: &mut dyn SessionExecutor,
    ) -> Result<(SessionCores, SessionHandle), DriverError> {
        let session_id =
            self.session_id.take().unwrap_or_else(|| lk_agents_net::strings::shortuuid("AS_"));
        let keyterms = self.keyterms.take().map(|mut setup| {
            let llm = setup.llm.take().map(|llm| self.resources.register_llm(llm).id);
            (setup, llm)
        });
        let forward_chat_context = self.forward_chat_context;
        let (queue_tx, queue_rx) = tokio::sync::mpsc::unbounded_channel::<VoiceRuntimeEvent>();
        let queue = Tx::new(queue_tx);
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<VoiceHostEvent>();
        let watch = WatchHub::new();
        let closing = tokio::sync::watch::channel(false).0;

        let mut sinks = Sinks {
            by_channel: std::collections::BTreeMap::new(),
            forwarder: AudioInputRouter::new(PIPELINE_SAMPLE_RATE),
        };
        // The STT pump runs under the first agent's activity, as the reference's does.
        let opening = self.opening();
        let activity = opening.activity.map_or(opening.start.context, |activity| activity.context);
        let cx = self.trace.clone().continuing(Some(activity));
        let opened = executor
            .open_legs(&self.resources, self.stt.clone(), self.vad.clone(), cx.clone())
            .await?;
        self.install_legs(&mut sinks, opened);
        let realtime = self.open_realtime(executor, &mut sinks).await?;
        if let Some(realtime) = &realtime {
            realtime.session.observe(cx);
        }
        let realtime_events = realtime.as_ref().map(|realtime| Arc::clone(&realtime.session));
        let Sinks { by_channel: sinks, forwarder } = sinks;

        // The microphone, read by the runtime.
        let audio_in =
            self.audio_in.clone().map(|input| self.resources.register_audio_input(input));
        let resources = &self.resources;
        let llm = self.llm.clone().map(|l| resources.register_llm(l));
        let tts = self.tts.clone().map(|t| resources.register_tts(t));
        let stt = self.stt.clone().map(|s| resources.register_stt(s));
        let vad = self.vad.clone().map(|v| resources.register_vad(v));
        let outputs = lk_agents_voice::agent_session::Outputs::new(
            self.audio_out.clone().map(|o| resources.register_audio_output(o)),
            self.text_out.clone().map(|o| resources.register_text_output(o)),
        );
        let (shared, publications) =
            self.shared(&queue, closing, outputs.clone(), session_id.clone());
        let session_cx = self.session_cx(&session_id);
        let shared_calls = Arc::clone(&shared.calls);
        // The session's id sequence: the core's speeches and items, and a host speech call's id.
        let speech_ids = Arc::new(std::sync::atomic::AtomicU64::new(1));

        let journal = self.journal.clone();
        let time = self.time.clone();
        let interruption = self.interruption.take();
        let eot = self.eot.take();
        let rt_model = self.realtime.as_ref().map(|model| model.id());
        shared.realtime_spans.set_model(rt_model.clone());
        let (rt_session, rt_caps, rt_disabled, rt_resolved) = match realtime.zip(rt_model) {
            Some((realtime, model)) => (
                Some(self.resources.register_realtime(realtime.session, model)),
                realtime.caps,
                realtime.disabled,
                realtime.resolved,
            ),
            None => (None, Default::default(), false, self.config.effective.turn.owner()),
        };
        let driver = VoiceDriver::new(InitialSession {
            config: self.config.clone(),
            parts: SessionParts {
                llm,
                tts,
                stt,
                vad,
                outputs,
                max_unrecoverable_errors: self.max_unrecoverable_errors,
                start_recorder: self
                    .recorder
                    .take()
                    .map(|value| self.resources.register_recorder(value)),
                room_link: self.room_link.take(),
                agent: self.agent.clone(),
                chat_ctx: self.chat_ctx.clone(),
                session_id,
                storage_enabled: self.storage.is_some(),
                telemetry: telemetry::SessionTelemetry {
                    parent: self.trace.span(),
                    job: self.trace.job().cloned(),
                    conversation: self.trace.conversation().map(str::to_owned),
                    job_ids: self.job_ids.clone(),
                    // Ids unique across processes in a job; seeded from the session id otherwise.
                    ids: (self.trace.span().is_some() || self.trace.job().is_some())
                        .then(|| telemetry::seeded_ids(telemetry::random_seed())),
                    opening: Some(self.opening()),
                },
            },
            instructions: std::mem::take(&mut self.instructions),
            dynamic_instructions: self
                .dynamic_instructions
                .take()
                .map(|value| self.resources.register_instructions(value)),
            output_nodes: lk_agents_voice::agent_session::OutputNodes {
                transcription: self
                    .output_nodes
                    .transcription
                    .take()
                    .map(|value| self.resources.register_transcription_node(value)),
            },
            toolset: std::mem::take(&mut self.toolset),
            hook_bound: self.hook_bound,
            expressive: self.expressive.clone(),
            sinks,
            video_instance: self
                .video_in
                .as_ref()
                .map(|input| self.resources.register_video_input(Arc::clone(input))),
            video_sampler: self.video_sampler.take(),
            eot_thresholds: eot.as_ref().and_then(|detector| detector.thresholds()),
            eot_provider: eot.as_ref().map(|detector| detector.id()),
            rt_session,
            rt_caps,
            rt_disabled,
            rt_resolved,
            view: Arc::clone(&self.view.borrow()),
            next_id: {
                let ids = Arc::clone(&speech_ids);
                Box::new(move || ids.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            },
            now: time.read(),
            journalled: journal.is_on(),
            forward_chat_context,
            keyterms,
        });
        // `ivr_detection=True` is the IVR system's to start; the core starts no system.
        let ivr_detection = driver.ivr_detection();
        let keyterms_pushed = shared.keyterms.clone();
        let chat_ctx = shared.chat_ctx.clone();
        let peer = PeerHandle {
            queue: queue.clone(),
            events: ev_tx,
            journal,
            time,
            spawner: Arc::clone(&self.spawner),
            watch,
            resources: self.resources.clone(),
            closing: shared.closing.clone(),
            session: session_cx,
        };

        let cores = SessionCores {
            publications,
            driver,
            forwarder,
            queue: (queue.clone(), queue_rx),
            source: audio_in,
            eot,
            interruption,
            realtime_events,
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
                calls: Arc::clone(&shared_calls),
            },
        ))
    }

    /// What the spawned tasks share with the loop.
    fn shared(
        &self,
        queue: &Tx,
        closing: tokio::sync::watch::Sender<bool>,
        outputs: lk_agents_voice::agent_session::Outputs,
        session_id: String,
    ) -> (Arc<SessionShared>, super::shared::Publications) {
        let publications = super::shared::Publications {
            closing,
            recorder: tokio::sync::watch::channel(None).0,
            view: self.view.clone(),
            output: tokio::sync::watch::channel(outputs).0,
            chat_ctx: tokio::sync::watch::channel(Arc::new(lk_agents_abi::ChatContext::default()))
                .0,
            keyterms: tokio::sync::watch::channel(None).0,
        };
        let trace = (self.trace.clone().continuing(None))
            .with_conversation(Some(self.conversation(&session_id).as_str()));
        let shared = Arc::new(SessionShared {
            tool_bodies: self.tool_bodies.clone(),
            view: publications.view.subscribe(),
            time: self.time.clone(),
            output: publications.output.subscribe(),
            recorder: publications.recorder.subscribe(),
            tts_transforms: lk_agents_voice::transcription::TransformSpec::of_set(
                self.config.effective.speech.tts_text_transforms(),
            ),
            conn: self.conn,
            queue: queue.clone(),
            calls: Arc::new(Calls::default()),
            speeches: Arc::clone(&self.speeches),
            closing: publications.closing.subscribe(),
            spawner: Arc::clone(&self.spawner),
            gen_ids: std::sync::atomic::AtomicU64::new(0),
            session_id,
            resources: self.resources.clone(),
            chat_ctx: publications.chat_ctx.subscribe(),
            keyterms: publications.keyterms.subscribe(),
            journal: self.journal.clone(),
            trace,
            realtime_spans: Default::default(),
            tracer: self.tracer.clone(),
            report: self.report.clone(),
        });
        (shared, publications)
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
            use lk_agents_voice::agent_activity::options::DetectorKind;
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
        executor: &mut dyn SessionExecutor,
        sinks: &mut Sinks,
    ) -> Result<Option<RealtimeStart>, DriverError> {
        let Some(model) = &self.realtime else {
            return Ok(None);
        };
        // Who ends the turn, decided once from the model's capabilities and the session's
        // inputs (`_resolve_rt_turn_detection_enabled`).
        let caps = model.capabilities();
        let side = lk_agents_voice::agent_activity::realtime::SessionSide {
            stt: self.stt.is_some(),
            vad: self.vad.is_some(),
            detection: self.config.effective.turn.detection(),
            streaming_detector: self.config.legs.detector
                == Some(lk_agents_voice::agent_activity::options::DetectorKind::Streaming),
            interruption_given: self.config.effective.turn.interruption.mode.is_some(),
            turn_detection_mode: self.config.effective.turn.owner(),
        };
        let disabled =
            lk_agents_voice::agent_activity::realtime::turn_detection_disabled(&caps, side);
        // The refined mode goes into the config; the pre-refinement answer is kept so a later
        // re-resolution is compared against it.
        let owner = lk_agents_voice::agent_activity::realtime::realtime_turn_detection_mode(
            &caps, side, disabled,
        );
        self.config.effective.turn.owner = Some(owner);
        let (session, sink) =
            executor.open_realtime(&self.resources, Arc::clone(model), disabled).await?;
        sinks.insert(AudioChannel::Realtime, sink);
        Ok(Some(RealtimeStart { session, caps, disabled, resolved: side.turn_detection_mode }))
    }
}

/// The opening STT and TTS as the session runs them: one attempt per call within the session's
/// budgets, adapted by the executor and under the agent's nodes, once the core accepted the
/// session's tools.
pub(super) fn prepare_opening_legs(
    builder: &mut DriverBuilder,
    nodes: &crate::voice::nodes::Nodes,
    tts_adapter: &TtsAdapter,
    stt_adapter: &SttAdapter,
) -> Result<(), DriverError> {
    builder.toolset.refuse_unrunnable()?;
    if let Some(llm) = &builder.llm {
        llm.prewarm();
    }
    let clock = builder.time.clock();
    let (vad, conn) = (builder.vad.clone(), builder.conn.stt);
    builder.stt = builder
        .stt
        .take()
        .map(|stt| nodes.stt(stt_leg(stt, stt_adapter, vad, clock.clone(), conn)));
    if let Some(stt) = &builder.stt {
        stt.prewarm();
    }
    // The engine's TTS budget: the session's, else the TTS's own policy (models.md).
    builder.conn.tts =
        builder.conn.tts.or_else(|| builder.tts.as_ref().and_then(|t| t.conn_options()));
    let conn = builder.conn.tts;
    builder.tts = builder.tts.take().map(|tts| nodes.tts(tts_leg(tts, tts_adapter, clock, conn)));
    if let Some(tts) = &builder.tts {
        tts.prewarm();
    }
    Ok(())
}

/// The sinks the microphone feeds, by channel: the decision core and the forwarder each learn
/// every one installed at start.
struct Sinks {
    by_channel: std::collections::BTreeMap<AudioChannel, lk_agents_abi::resources::ResourceId>,
    forwarder: AudioInputRouter,
}

impl Sinks {
    fn install(
        &mut self,
        resources: &Resources,
        channel: AudioChannel,
        sink: Arc<dyn StreamInput<AudioFrame>>,
    ) {
        self.insert(channel, resources.register_audio_sink(sink));
    }

    fn insert(&mut self, channel: AudioChannel, sink: ResourceId) {
        self.by_channel.insert(channel, sink);
        // Nothing is replaced yet, so there is nothing to perform.
        let _ = self.forwarder.handle(AudioInputRouterCommand::Install { channel, sink });
    }
}

/// The turn detector's audio leg: every frame is `push_audio`. `flush` is not the detector's
/// `flush`, which is a turn boundary the machine decides.
struct EotAudioSink(Arc<dyn EotDetector>);

impl StreamInput<AudioFrame> for EotAudioSink {
    fn push(&self, frame: AudioFrame) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.0.push_audio(frame);
        Ok(())
    }

    fn head(&self, frame: AudioFrame) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.0.push_head(frame);
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
    resolved: lk_agents_voice::audio_recognition::TurnDetectionMode,
}

/// A host provider is attempted once per call with the leg's timeout: the engine's `Attempt`
/// loop owns the LLM's and the TTS's retries (on `SessionShared::conn`), and two loops would
/// multiply.
fn one_attempt(conn: &lk_agents_abi::ApiConnectOptions) -> lk_agents_abi::ApiConnectOptions {
    lk_agents_abi::ApiConnectOptions { max_retry: 0, ..*conn }
}

/// A host STT spends its budget in its `RetryingStt`; a recognize-only one is adapted by the
/// executor over `vad`.
pub fn stt_leg(
    stt: Arc<dyn Stt>,
    adapter: &SttAdapter,
    vad: Option<Arc<dyn Vad>>,
    clock: Arc<dyn lk_agents_abi::Clock>,
    conn: Option<lk_agents_abi::ApiConnectOptions>,
) -> Arc<dyn Stt> {
    let conn = conn.or_else(|| stt.conn_options()).unwrap_or_default();
    adapter(stt.with_conn_options(&conn).unwrap_or(stt), vad, clock)
}

pub fn llm_leg(llm: Arc<dyn Llm>, conn: Option<lk_agents_abi::ApiConnectOptions>) -> Arc<dyn Llm> {
    llm.with_conn_options(&one_attempt(&conn.unwrap_or_default())).unwrap_or(llm)
}

/// One attempt per call, adapted by the executor when it has no duplex path.
pub fn tts_leg(
    tts: Arc<dyn Tts>,
    adapter: &TtsAdapter,
    clock: Arc<dyn lk_agents_abi::Clock>,
    conn: Option<lk_agents_abi::ApiConnectOptions>,
) -> Arc<dyn Tts> {
    let conn = conn.or_else(|| tts.conn_options()).unwrap_or_default();
    adapter(tts.with_conn_options(&one_attempt(&conn)).unwrap_or(tts), clock, conn)
}

/// Startup owns its registrations and admitted tasks until it installs Running. A dropped or
/// refused startup cannot leave tasks behind or overwrite a terminal session phase.
pub(super) struct OpeningSession {
    stage: Arc<parking_lot::Mutex<super::Stage>>,
    tasks: Arc<lk_agents_net::task::TaskSet>,
    loop_tasks: Vec<lk_agents_net::task::TaskId>,
    resources: Resources,
    pub(super) cancelled: tokio::sync::watch::Receiver<bool>,
    installed: bool,
    spawner: Arc<dyn Spawner>,
    cleanup: Option<lk_agents_abi::BoxFuture<'static, ()>>,
}

impl OpeningSession {
    pub(super) fn new(
        session: &super::VoiceSession,
        cancelled: tokio::sync::watch::Receiver<bool>,
    ) -> Self {
        Self {
            stage: session.stage.clone(),
            tasks: session.tasks.clone(),
            loop_tasks: Vec::new(),
            resources: session.resources.clone(),
            cancelled,
            installed: false,
            spawner: session.spawner.clone(),
            cleanup: Some(Box::pin(session.close_io())),
        }
    }

    pub(super) fn check_open(&self) -> Result<(), DriverError> {
        if self.stage.lock().lifecycle().accepts_opened() {
            Ok(())
        } else {
            Err(DriverError::StartupClosed)
        }
    }

    pub(super) fn spawn(
        &mut self,
        spawner: &dyn Spawner,
        name: &str,
        work: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> Result<(), DriverError> {
        let task = self.tasks.try_spawn(spawner, name, work).ok_or(DriverError::StartupClosed)?;
        self.loop_tasks.push(task);
        Ok(())
    }

    pub(super) fn install(&mut self, mut running: super::Running) -> Result<(), DriverError> {
        let retired = {
            let mut stage = self.stage.lock();
            if !stage.lifecycle().accepts_opened() {
                return Err(DriverError::StartupClosed);
            }
            running.loop_tasks = std::mem::take(&mut self.loop_tasks);
            self.installed = true;
            std::mem::replace(&mut *stage, super::Stage::Running(Box::new(running)))
        };
        drop(retired);
        Ok(())
    }
}

impl Drop for OpeningSession {
    fn drop(&mut self) {
        if self.installed {
            return;
        }
        let retired = {
            let mut stage = self.stage.lock();
            match &*stage {
                super::Stage::Opening { remote, .. } => {
                    let closed = super::Stage::Closed(remote.clone());
                    Some(std::mem::replace(&mut *stage, closed))
                }
                _ => None,
            }
        };
        drop(retired);
        for task in self.loop_tasks.drain(..) {
            self.tasks.cancel(task);
        }
        self.resources.clear();
        if let Some(cleanup) = self.cleanup.take() {
            use lk_agents_abi::SpawnerExt as _;
            self.spawner.spawn_detached(cleanup);
        }
    }
}
