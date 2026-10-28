//! `DriverBuilder`: assembles a session's runtime (its driver and data plane) from its providers,
//! legs and options, and spawns the tasks that feed it.

use super::driver::*;
use crate::agent_activity::false_interruption::PausedSpeech;
use crate::agent_activity::scheduling::SpeechScheduler;
use crate::agent_activity::{AgentActivity, DriverConfig, PIPELINE_SAMPLE_RATE};
use crate::agent_session::SessionShared;
use crate::agent_session::VoiceSession;
use crate::agent_session::commands::Origin;
use crate::agent_session::host_events::HostEvent;
use crate::agent_session::time::Timebase;
use crate::audio::{AudioChannel, spawn_drain};
use crate::audio_recognition::AudioRecognition;
use crate::data_plane::{AudioForwarder, ForwarderCommand};
use crate::generation::Instructions;
use crate::trace::{CoreStep, Journal, RealtimeStep, Step};
use lk_agents_abi::turn::detector::DetectorLeg;
use lk_agents_abi::utils::frame::FrameTable;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, TextOutput};
use lk_agents_abi::{FrameId, Llm, Spawner, SpawnerExt, Stt, TimerSet, Ts, Tts, Vad};
use lk_agents_rtc::RoomEvent;
use std::sync::Arc;
use std::time::Duration;

type EotLeg = lk_agents_abi::EotLeg<dyn lk_agents_abi::EotDetector>;

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
    /// A TTS with no duplex path arrives wrapped in `StreamAdapter`.
    pub tts: Option<Arc<dyn Tts>>,
    pub stt: Option<Arc<dyn Stt>>,
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
    pub room_io_events: Option<Rx<crate::room_io::RoomIoEvent>>,
    pub room_link: Option<crate::room_io::RoomLinkConfig>,
    /// `None` commits turns on silence alone.
    pub eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// `None` is VAD-only interruption: every burst past `min_interruption_duration`, a "mm-hm"
    /// included, stops the agent.
    pub interruption: Option<Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>>,
    /// `None` forwards **every** frame the host pushes.
    pub video_sampler: Option<Arc<lk_agents_voice::room_io::_input::VideoSampler>>,
    /// Instead of an LLM+TTS pair: `llm`/`tts` are ignored while it is set.
    pub realtime: Option<Arc<dyn lk_agents_abi::Realtime>>,
    pub toolset: crate::tool_executor::Toolset,
    /// Prefixes every chat item id the session mints, so items from two calls never collide.
    pub session_id: String,
    pub storage: Option<Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>>,
    /// Where the driver's steps go: nowhere in production.
    pub journal: Journal,
}

impl DriverBuilder {
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    /// `async` because the provider factories must run on the host loop.
    pub async fn start(mut self) -> Result<(crate::runtime::Runtime, SessionHandle), DriverError> {
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
        if let Some(stt) = &self.stt {
            stt.prewarm();
        }
        if let Some(tts) = &self.tts {
            tts.prewarm();
        }
        let (queue_tx, queue_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let queue = Tx::new(queue_tx);
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<HostEvent>();
        let watch = WatchHub::new();
        // Created first so forwarders and drains can subscribe.
        let closing = tokio::sync::watch::channel(false).0;
        // Forwarders end at `Closing`, so no task outlives teardown.
        if let Some(rx) = self.room_io_events.take() {
            spawn_forward(&self.spawner, rx, queue.clone(), closing.subscribe());
        }
        if let Some(rx) = self.room_events.take() {
            spawn_forward(&self.spawner, rx, queue.clone(), closing.subscribe());
        }

        let mut sinks = Sinks {
            by_channel: std::collections::BTreeMap::new(),
            forwarder: AudioForwarder::new(PIPELINE_SAMPLE_RATE),
        };
        let eot = self.open_legs(&mut sinks, &queue, &closing).await?;
        let realtime = self.open_realtime(&mut sinks, &queue).await?;
        let Sinks { by_channel: sinks, forwarder } = sinks;

        // The microphone, read by the runtime.
        let audio_in =
            self.audio_in.clone().map(|input| self.resources.register_audio_input(input));
        // The camera, the same way: one task per source, replaced whole by `SetVideoInput`.
        let video_parts = crate::audio::VideoForwarderParts {
            closing: closing.subscribe(),
            queue: queue.clone(),
        };
        let video_forwarder = self
            .video_in
            .clone()
            .map(|input| self.spawner.spawn_unboxed(video_parts.clone().pump(input)));

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

        let mut runtime = crate::runtime::Runtime::new(
            driver,
            forwarder,
            (queue.clone(), queue_rx),
            audio_in,
            eot,
            interruption,
            (video_parts, video_forwarder, self.video_in.take()),
            shared,
            peer.events.clone(),
            peer.watch.clone(),
            self.storage.take(),
        );
        let effects = std::mem::take(&mut runtime.driver.outbox);
        runtime.perform_voice(effects);
        Ok((
            runtime,
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
            recorder_observer: std::sync::Mutex::new(None),
            launch: crate::audio::LaunchGate::new(),
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

    /// The STT and VAD streams, drained into the loop, and the detectors' legs, pumped into it.
    async fn open_legs(
        &mut self,
        sinks: &mut Sinks,
        queue: &Tx,
        closing: &tokio::sync::watch::Sender<bool>,
    ) -> Result<Option<Arc<EotLeg>>, DriverError> {
        if let Some(stt) = &self.stt {
            let (sink, mut source) =
                stt.stream(None, None).await?.ok_or(DriverError::Missing("a streaming STT"))?;
            source.report_retries(crate::audio::report_to(queue.clone()));
            let sink = sinks.install(&self.resources, AudioChannel::Stt, Arc::from(sink));
            spawn_drain(
                &self.spawner,
                self.time.clone(),
                closing.subscribe(),
                crate::audio::tagged(sink, source),
                queue.clone(),
                (AudioChannel::Stt, sink),
            );
        }

        if let Some(vad) = &self.vad {
            let (sink, source) = vad.stream().await?;
            let sink = sinks.install(&self.resources, AudioChannel::Vad, Arc::from(sink));
            spawn_drain(
                &self.spawner,
                self.time.clone(),
                closing.subscribe(),
                source,
                queue.clone(),
                (AudioChannel::Vad, sink),
            );
        }

        // Detector presence has one source, the leg.
        {
            use crate::agent_activity::options::DetectorKind;
            let eot = self.eot.is_some();
            self.config.edit(|_, legs| legs.detector = eot.then_some(DetectorKind::Streaming));
        }
        let eot = self
            .eot
            .clone()
            .map(|d| Arc::new(lk_agents_abi::EotLeg::new(d, Arc::clone(&self.spawner))));
        if let Some(leg) = &eot {
            if let Some(audio) = leg.audio() {
                sinks.install(&self.resources, AudioChannel::Eot, audio);
            }
            spawn_leg_pump(&self.spawner, Arc::clone(leg), queue.clone(), closing.subscribe());
        }
        // The interruption detector's verdicts and audio leg come off one object, so neither
        // exists without the other.
        if let Some(detector) = self.interruption.clone() {
            sinks.install(&self.resources, AudioChannel::Interruption, detector.audio());
            spawn_leg_pump(&self.spawner, detector, queue.clone(), closing.subscribe());
        }
        Ok(eot)
    }

    /// The realtime session, opened once for the driver's life (a handoff updates it).
    async fn open_realtime(
        &mut self,
        sinks: &mut Sinks,
        queue: &Tx,
    ) -> Result<Option<RealtimeStart>, DriverError> {
        let Some(model) = &self.realtime else { return Ok(None) };
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
        eot: Option<&Arc<EotLeg>>,
        realtime: Option<RealtimeStart>,
        paused_speech: PausedSpeech,
        resolved_mode: crate::audio_recognition::TurnDetectionMode,
        requested_detection: lk_agents_abi::voice::session_options::TurnDetection,
    ) -> AgentActivity {
        let effective = &self.config.effective;
        let mut audio_recognition = AudioRecognition::new(effective, &self.config.legs);
        audio_recognition.set_thresholds(eot.as_ref().and_then(|leg| leg.thresholds()));
        // `_adaptive_interruption_active` is "a detector was supplied"; without it the sentinels
        // are unreachable.
        audio_recognition.set_interruption_active(self.config.legs.adaptive_detector);
        audio_recognition.set_eot_provider(eot.map(|leg| leg.id()));
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
        resources: &lk_agents_abi::resources::Resources,
        channel: AudioChannel,
        sink: Arc<dyn lk_agents_abi::StreamSink<lk_agents_abi::AudioFrame>>,
    ) -> lk_agents_abi::resources::ResourceId {
        let sink = resources.register_audio_sink(sink);
        self.by_channel.insert(channel, sink);
        // Nothing is replaced yet, so there is nothing to perform.
        let _ = self.forwarder.handle(ForwarderCommand::Install { channel, sink });
        sink
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

#[allow(clippy::disallowed_macros, reason = "S0: runtime migration")]
/// Ends on `Closed` (forwarded first), when the loop is gone, or at `Closing`. None is an error:
/// the session goes on with VAD-only interruption.
pub(super) fn spawn_leg_pump<L>(
    spawner: &Arc<dyn Spawner>,
    leg: Arc<L>,
    tx: Tx,
    mut closing: tokio::sync::watch::Receiver<bool>,
) where
    L: DetectorLeg + ?Sized + 'static,
    L::Verdict: 'static,
    lk_agents_abi::turn::detector::LegEvent<L::Verdict>: Into<Event>,
{
    spawner.spawn_unboxed(async move {
        loop {
            let ev = tokio::select! {
                biased;
                _ = closing.wait_for(|c| *c) => break,
                ev = leg.next() => ev,
            };
            let closed = matches!(ev, lk_agents_abi::turn::detector::LegEvent::Closed);
            if tx.send(ev).is_err() || closed {
                break;
            }
        }
    });
}

#[allow(clippy::disallowed_macros, reason = "S0: runtime migration")]
fn spawn_forward<T: Into<Event> + Send + 'static>(
    spawner: &Arc<dyn Spawner>,
    mut rx: Rx<T>,
    tx: Tx,
    mut closing: tokio::sync::watch::Receiver<bool>,
) {
    spawner.spawn_unboxed(async move {
        loop {
            let ev = tokio::select! {
                biased;
                ev = rx.recv() => ev,
                _ = closing.wait_for(|c| *c) => None,
            };
            if ev.is_none_or(|ev| tx.send(ev).is_err()) {
                break;
            }
        }
    });
}
