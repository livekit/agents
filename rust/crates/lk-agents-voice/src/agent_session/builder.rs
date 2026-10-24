//! `DriverBuilder`: assembles a session's driver from its providers, legs and options, and
//! spawns the pumps that feed it.

use super::driver::*;
use crate::agent_activity::pause::{PausedSpeech, sink_facts};
use crate::agent_activity::scheduling::SpeechScheduler;
use crate::agent_activity::{ActivityParts, DriverConfig, GateSnapshot, PIPELINE_SAMPLE_RATE};
use crate::agent_session::LiveSession;
use crate::agent_session::SessionShared;
use crate::agent_session::commands::{Command, Origin};
use crate::agent_session::host_events::HostEvent;
use crate::agent_session::time::Timebase;
use crate::audio::{AudioForwarderParts, LegEnd, LegId, Legs, spawn_drain};
use crate::audio_recognition::AudioRecognition;
use crate::generation::GenEvent;
use crate::generation::Instructions;
use crate::trace::{CoreStep, RealtimeStep, Step, Trace};
use lk_agents_abi::turn::detector::DetectorLeg;
use lk_agents_abi::utils::frame::FrameTable;
use lk_agents_abi::{
    FrameId, Llm, Spawner, SpawnerExt, SpeechEvent, Stt, TimerSet, Ts, Tts, Vad, VadEvent,
};
use lk_agents_rtc::RoomEvent;
use lk_agents_voice::io::{AudioInput, AudioOutput, TextOutput};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct DriverBuilder {
    pub time: Timebase,
    pub spawner: Arc<dyn Spawner>,
    pub config: DriverConfig,
    pub thresholds: lk_agents_abi::tokenize::language::ThresholdOptions,
    pub agent: String,
    pub instructions: Instructions,
    /// The session starts from the agent's constructor context. Seeded before the instructions, so
    /// the system message lands in it.
    pub chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
    /// `Some(bound)`: the hooks the sequence needs wait up to `bound`; an overrun is journalled and
    /// skipped. `None`: each hook is finished in place, and a tool's handoff cannot be described.
    pub hook_bound: Option<Duration>,
    /// Opened before the opening frame's handoff record, so that record and the opening `on_enter`
    /// are the run's.
    pub capture_run: Option<crate::run_result::RunSpec>,
    /// Consecutive unrecoverable errors of one leg tolerated before closing (default `3`).
    pub max_unrecoverable_errors: u32,
    pub llm: Option<Arc<dyn Llm>>,
    /// A TTS with no duplex path arrives wrapped in `StreamAdapter`.
    pub tts: Option<Arc<dyn Tts>>,
    pub stt: Option<Arc<dyn Stt>>,
    pub speaker_filter: Option<crate::audio::SpeakerFilterConfig>,
    /// `None`: nothing is pushed.
    pub keyterms: Option<crate::keyterm_detection::KeytermSetup>,
    pub vad: Option<Arc<dyn Vad>>,
    pub audio_in: Option<Arc<dyn AudioInput>>,
    pub video_in: Option<Arc<dyn lk_agents_voice::io::VideoInput>>,
    pub recorder: Option<Arc<dyn lk_agents_media::recorder::AudioRecorder>>,
    pub audio_out: Option<Arc<dyn AudioOutput>>,
    pub tts_text_transforms: Vec<lk_agents_voice::transcription::text_transforms::TransformSpec>,
    pub realtime_audio_node: Option<Arc<dyn crate::io::RealtimeAudioNode>>,
    pub text_out: Option<Arc<dyn TextOutput>>,
    pub room_events: Option<Rx<RoomEvent>>,
    pub room_io_events: Option<Rx<crate::room_io::RoomIoEvent>>,
    pub room_link: Option<crate::room_io::link::RoomLinkConfig>,
    /// `None` commits turns on silence alone.
    pub eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// `None` is VAD-only interruption: every burst past `min_interruption_duration`, a "mm-hm"
    /// included, stops the agent.
    pub interruption: Option<Arc<dyn crate::agent_activity::pause::InterruptionPort>>,
    /// `None` forwards **every** frame the host pushes.
    pub video_sampler: Option<Arc<lk_agents_voice::room_io::_input::VideoSampler>>,
    /// Instead of an LLM+TTS pair: `llm`/`tts` are ignored while it is set.
    pub realtime: Option<Arc<dyn lk_agents_abi::Realtime>>,
    pub toolset: crate::tool_executor::Toolset,
}

impl DriverBuilder {
    /// `async` because the provider factories must run on the host loop.
    pub async fn start(mut self) -> Result<(Driver, SessionHandle), DriverError> {
        let keyterms = self.keyterms.take();
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
        let (queue_tx, queue_rx) = tokio::sync::mpsc::unbounded_channel::<Input>();
        let stt_tx: Tx<crate::audio::Tagged<SpeechEvent>> = Tx::new(queue_tx.clone(), Input::Stt);
        let vad_tx: Tx<VadEvent> = Tx::new(queue_tx.clone(), Input::Vad);
        let api_tx: Tx<Command> = Tx::new(queue_tx.clone(), host_input);
        let gen_tx: Tx<GenEvent> = Tx::new(queue_tx.clone(), Input::Gen);
        let legs_tx: Tx<LegEnd> = Tx::new(queue_tx.clone(), Input::LegEnded);
        let leg_opens: Tx<crate::room_io::LegOpened> = Tx::new(queue_tx.clone(), Input::LegOpened);
        let video_tx: Tx<lk_agents_rtc::VideoFrameRef> = Tx::new(queue_tx.clone(), Input::Video);
        let eot_events: Tx<lk_agents_abi::turn::detector::LegEvent<lk_agents_abi::EotPrediction>> =
            Tx::new(queue_tx.clone(), eot_input);
        let verdict_tx: Tx<
            lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict>,
        > = Tx::new(queue_tx.clone(), Input::Overlap);
        let rt_tx: Tx<crate::agent_activity::realtime::RealtimeWake> =
            Tx::new(queue_tx.clone(), Input::Realtime);
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<HostEvent>();
        let (answers_tx, answers_rx) = tokio::sync::mpsc::unbounded_channel();
        let watch = WatchHub::new();
        let ev_tx_for_peers = ev_tx.clone();
        // Created first so forwarders and drains can subscribe.
        let closing = tokio::sync::watch::channel(false).0;
        // Forwarders end at `Closing`, so no task outlives teardown.
        if let Some(rx) = self.room_io_events.take() {
            let tx = Tx::new(queue_tx.clone(), Input::RoomIo);
            spawn_forward(&self.spawner, rx, tx, closing.subscribe());
        }
        if let Some(rx) = self.room_events.take() {
            let tx = Tx::new(queue_tx.clone(), Input::Room);
            spawn_forward(&self.spawner, rx, tx, closing.subscribe());
        }

        // --- the one leg table: presence is read off it and nowhere else ---
        let legs = Legs::new();
        if let Some(stt) = &self.stt {
            let (sink, source) =
                stt.stream(None, None).await.ok_or(DriverError::Missing("a streaming STT"))??;
            // The speaker filter rides the STT row, a fresh detector per stream.
            let ticket = legs.install_stt(
                Arc::from(sink),
                self.speaker_filter.as_ref().map(crate::audio::speaker_filter),
            );
            spawn_drain(
                &self.spawner,
                self.time.clone(),
                closing.subscribe(),
                crate::audio::tagged(ticket, source),
                stt_tx.clone(),
                Arc::clone(&legs),
                ticket,
                legs_tx.clone(),
            );
        }

        if let Some(vad) = &self.vad {
            let (sink, source) = vad.stream().await?;
            let ticket = legs.install(LegId::Vad, Arc::from(sink));
            spawn_drain(
                &self.spawner,
                self.time.clone(),
                closing.subscribe(),
                source,
                vad_tx.clone(),
                Arc::clone(&legs),
                ticket,
                legs_tx.clone(),
            );
        }

        // --- the realtime session, opened once for the driver's life (a handoff updates it) ---
        // Detector presence has one source, the leg; with a leg and no protocol named, batch.
        {
            use crate::agent_activity::options::DetectorKind;
            let eot = self.eot.is_some();
            self.config.edit(|_, legs| {
                legs.detector = match (eot, legs.detector) {
                    (false, None | Some(DetectorKind::Batch | DetectorKind::Streaming)) => None,
                    (true, None) => Some(DetectorKind::Batch),
                    (true, Some(kind @ (DetectorKind::Batch | DetectorKind::Streaming))) => {
                        Some(kind)
                    }
                };
            });
        }
        let eot = self
            .eot
            .clone()
            .map(|d| Arc::new(lk_agents_abi::EotLeg::new(d, Arc::clone(&self.spawner))));
        if let Some(leg) = &eot {
            if let Some(audio) = leg.audio() {
                legs.install(LegId::Eot, audio);
            }
            spawn_leg_pump(&self.spawner, Arc::clone(leg), eot_events, closing.subscribe());
        }
        // The interruption detector's verdicts and audio leg come off one object, so neither
        // exists without the other.
        if let Some(detector) = self.interruption.clone() {
            legs.install(LegId::Interruption, detector.audio());
            spawn_leg_pump(&self.spawner, detector, verdict_tx, closing.subscribe());
        }

        let mut realtime_session = None;
        let mut realtime_caps = None;
        let mut realtime_disabled = None;
        let mut realtime_resolved = None;
        if let Some(model) = &self.realtime {
            // Who ends the turn, decided once from the model's capabilities and the session's
            // inputs (`_resolve_rt_turn_detection_enabled`).
            let caps = model.capabilities();
            let side = crate::agent_activity::realtime::SessionSide {
                stt: self.stt.is_some(),
                vad: self.vad.is_some(),
                detection: self.config.effective.turn.detection,
                streaming_detector: self.config.legs.detector
                    == Some(crate::agent_activity::options::DetectorKind::Streaming),
                interruption_given: self.config.effective.turn.interruption.mode.is_some(),
                turn_detection_mode: self.config.effective.turn.owner,
            };
            let disabled = crate::agent_activity::realtime::turn_detection_disabled(&caps, side);
            realtime_disabled = Some(disabled);
            // The refined mode goes into the config; the pre-refinement answer is kept so a later
            // re-resolution is compared against it.
            realtime_resolved = Some(side.turn_detection_mode);
            self.config.effective.turn.owner =
                crate::agent_activity::realtime::realtime_turn_detection_mode(
                    &caps, side, disabled,
                );
            let session = model.session(disabled).await?;
            realtime_caps = Some(caps);
            legs.install(
                LegId::Realtime,
                Arc::new(crate::agent_activity::realtime::RealtimeAudioSink::new(Arc::clone(
                    &session,
                ))),
            );
            // The opening frame is sent ungated by `mutable_*` (they govern mid-session changes)
            // through `push_frame` once the driver exists.
            crate::agent_activity::realtime::spawn_event_pump(
                &self.spawner,
                Arc::clone(&session),
                rt_tx.clone(),
            );
            realtime_session = Some(session);
        }

        // --- the one audio conversion, fanned out ----------------------------------------------
        let gate = GateSnapshot::new();
        // Held, so `SetAudioInput` can point a fresh pump at the same table.
        let recorder_cell = Arc::new(std::sync::RwLock::new(None));
        let pump_parts = AudioForwarderParts {
            last_input: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            first_input_at: Arc::new(std::sync::atomic::AtomicU64::new(crate::audio::NO_INPUT_YET)),
            input_enabled: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            recorder: Arc::clone(&recorder_cell),
            time: self.time.clone(),
            gate: Arc::clone(&gate),
            legs,
            rate: PIPELINE_SAMPLE_RATE,
            closing: closing.subscribe(),
            ended: legs_tx.clone(),
        };
        let audio_forwarder = self
            .audio_in
            .clone()
            .map(|input| self.spawner.spawn_unboxed(pump_parts.clone().pump(input)));
        // The camera, the same way: one task per source, replaced whole by `SetVideoInput`.
        let video_parts = crate::audio::VideoForwarderParts {
            closing: closing.subscribe(),
            ended: legs_tx.clone(),
            frames: video_tx,
        };
        let video_forwarder = self
            .video_in
            .clone()
            .map(|input| self.spawner.spawn_unboxed(video_parts.clone().pump(input)));

        let shared = Arc::new(SessionShared {
            time: self.time.clone(),
            output: crate::agent_session::AgentOutput::new(
                self.audio_out.clone(),
                self.text_out.clone(),
            ),
            recorder: recorder_cell,
            recorder_observer: std::sync::Mutex::new(None),
            launch: crate::audio::LaunchGate::new(),
            tts_transforms: self.tts_text_transforms.clone(),
            realtime_audio: self.realtime_audio_node.clone(),
            events: gen_tx,
            closing,
            draining: tokio::sync::watch::channel(false).0,
            hook_host: self.hook_bound.is_some(),
            spawner: Arc::clone(&self.spawner),
            frames: Arc::new(FrameTable::new()),
            gen_ids: std::sync::atomic::AtomicU64::new(0),
            frame: tokio::sync::watch::channel(crate::agent_session::FrameView::default()).0,
        });
        let tts = self.tts.clone();

        let config = self.config.clone();
        let mut audio_recognition =
            AudioRecognition::new(&config.effective, &config.legs, self.thresholds);
        let paused_speech = {
            // The sink's facts at start; re-read at every pause decision.
            let (sink, audio_enabled) = sink_facts(self.audio_out.as_deref());
            PausedSpeech::new(&config.effective, sink, audio_enabled)
        };
        let speech_scheduler =
            SpeechScheduler::new(config.effective.speech.min_consecutive_speech_delay);

        // `_adaptive_interruption_active` is "a detector was supplied"; without it the sentinels
        // are unreachable.
        audio_recognition.set_interruption_active(self.interruption.is_some());

        // The constructor context as the agent's view, not announced (the host has it). The prompt
        // is rendered per request.
        let session = LiveSession::new(
            config,
            crate::agent_session::SessionParts {
                llm: self.llm,
                tts,
                stt: self.stt,
                vad: self.vad,
                max_unrecoverable_errors: self.max_unrecoverable_errors,
                start_options: crate::agent_session::StartOptions {
                    recorder: self.recorder,
                    run: self.capture_run,
                },
                room_link: self.room_link,
                agent: self.agent.clone(),
                chat_ctx: self.chat_ctx.clone(),
            },
        );

        let trace = Arc::new(Mutex::new(Trace::new()));
        {
            let mut t = trace.lock().unwrap_or_else(|p| p.into_inner());
            t.push(Step::Core(CoreStep::ConfigApplied {
                layer: crate::agent_activity::options::ConfigLayer::Session,
                effective: Box::new(session.config.effective),
            }));
            // The wall clock at `Ts::ZERO`: the session may already be past it.
            let since_zero = self.time.now().saturating_duration_since(Ts::ZERO);
            t.push(Step::Core(CoreStep::WallEpoch {
                at: self.time.wall().checked_sub(since_zero).unwrap_or(std::time::UNIX_EPOCH),
            }));
            if let Some(turn_detection_disabled) = realtime_disabled {
                t.push(Step::Realtime(RealtimeStep::Opened { turn_detection_disabled }));
            }
        }

        let resolved_mode = realtime_resolved.unwrap_or(session.config.effective.turn.owner);
        audio_recognition.set_detectors(eot, self.interruption);
        let activity = crate::agent_activity::AgentActivity::new(ActivityParts {
            agent: self.agent,
            instructions: self.instructions,
            toolset: self.toolset,
            audio_recognition,
            speech_scheduler,
            paused_speech,
            hook_bound: self.hook_bound,
            rt_session: realtime_session,
            realtime_caps: realtime_caps.unwrap_or_default(),
            realtime_turn_detection_disabled: realtime_disabled.unwrap_or(false),
            realtime_resolved_mode: resolved_mode,
            realtime_requested_detection: session.config.effective.turn.detection,
        });
        let input = crate::io::AgentInput::new(
            (pump_parts, audio_forwarder, self.audio_in),
            (video_parts, video_forwarder, self.video_in),
            self.video_sampler,
            gate,
            self.speaker_filter,
        );
        let mut driver = Driver {
            session,
            activity,
            input,
            current: FrameId::ROOT,
            timers: TimerSet::new(),
            shared,
            events: ev_tx,
            answers: answers_tx,
            inputs: queue_rx,
            trace: Arc::clone(&trace),
            watch: watch.clone(),
            realtime_tx: rt_tx,
            stt_tx,
            vad_tx,
            leg_opens,
            command_origin: Origin::Host,
            step: crate::agent_session::time::StepClock::new(),
            stopped: false,
            opened: false,
        };
        // The opening cycles run at the instant the session is built.
        driver.step.set(self.time.read());
        driver.publish_frame();

        // The first facts, in order: the gate, then the opening frame's records (the handoff from
        // nobody, the starting config), and the opening frame on the realtime socket.
        driver.publish_gate();
        {
            let mut out = driver.out();
            // `start(capture_run=True)`'s run exists before the opening handoff is recorded, so the
            // handoff and the opening `on_enter` are the run's.
            if let Some(spec) = driver.session.start_options.run.take() {
                out.trace(CoreStep::CommandReceived { origin: Origin::Host, kind: "Command::Run" });
                driver.on_run(driver.step.now, spec, &mut out);
            }
            driver.record_frame(None, &mut out);
            if driver.activity.rt_session.is_some() {
                let opening = driver.session.record.for_realtime();
                driver.push_frame(Some(opening), &mut out);
            }
            driver.execute(out);
        }
        // `KeytermDetector.start`: the static terms reach the recogniser now.
        if let Some(setup) = keyterms {
            let mut out = driver.out();
            driver.keyterms_start(setup, &mut out);
            driver.execute(out);
        }
        // `ivr_detection=True` is the IVR system's to start; the core starts no system.
        let ivr_detection = driver.session.config.effective.turn.ivr_detection;
        let peer = PeerHandle {
            api: api_tx.clone(),
            events: ev_tx_for_peers,
            trace: Arc::clone(&trace),
            time: self.time.clone(),
            spawner: Arc::clone(&self.spawner),
            watch,
            legs: Arc::clone(driver.input.legs()),
            closing: driver.shared.closing.subscribe(),
        };

        Ok((
            driver,
            SessionHandle {
                api: api_tx,
                events: ev_rx,
                answers: answers_rx,
                trace,
                peer,
                ivr_detection,
            },
        ))
    }
}

/// Ends on `Closed` (forwarded first), when the loop is gone, or at `Closing`. None is an error:
/// the session goes on with VAD-only interruption.
pub(super) fn spawn_leg_pump<L>(
    spawner: &Arc<dyn Spawner>,
    leg: Arc<L>,
    tx: Tx<lk_agents_abi::turn::detector::LegEvent<L::Verdict>>,
    mut closing: tokio::sync::watch::Receiver<bool>,
) where
    L: DetectorLeg + ?Sized + 'static,
    L::Verdict: 'static,
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

fn spawn_forward<T: Send + 'static>(
    spawner: &Arc<dyn Spawner>,
    mut rx: Rx<T>,
    tx: Tx<T>,
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

/// `Closed` is the pump's own end.
fn eot_input(ev: lk_agents_abi::turn::detector::LegEvent<lk_agents_abi::EotPrediction>) -> Input {
    use lk_agents_abi::turn::detector::{LegEvent, ReqId};
    match ev {
        LegEvent::Verdict { req, verdict } => Input::Eot(req.unwrap_or(ReqId(0)), Ok(verdict)),
        LegEvent::Error { req, recoverable: _, err } => {
            Input::Eot(req.unwrap_or(ReqId(0)), Err(err))
        }
        LegEvent::Closed => Input::LegEnded(LegEnd {
            leg: "eot",
            open: 0,
            cause: crate::trace::LegEndCause::SourceEnded,
            closed: None,
            error: None,
        }),
    }
}
