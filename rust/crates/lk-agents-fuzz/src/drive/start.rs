//! Building the session, and the product objects a case's words become.
//!
//! [`Run::start`] wires a real driver to the fakes and installs the trace observer before the
//! loop is spawned. Below it are the constructors the start and the commands share -- a toolset,
//! a frame, a history, a patch -- the live objects a command carries, built from the case's terms.
use super::*;

/// Journal steps one virtual instant may hold before the run counts as a livelock.
const NO_PROGRESS_STEPS: u32 = 100_000;

impl Run {
    /// Start a real driver with the fakes attached, and record what it opens on the way up.
    ///
    /// The trace observer is installed before the loop is spawned, so the first step the driver
    /// decides is stamped from the same counter as everything else the case observes. That one
    /// counter is what makes the recording a happens-before order rather than a pile of clocks.
    pub async fn start(setup: &Setup) -> Run {
        let time = Timebase::simulated();
        let stamps = Stamps::new(time.clone());
        let (registry, records) = Registry::new(stamps.clone());
        let tasks = Arc::new(Tasks::default());

        let mut config = ConfigStack::default();
        config.session.turn.detection = match setup.turn {
            TurnMode::Default => None,
            TurnMode::Stt => Some(TurnDetection::Stt),
            TurnMode::Manual => Some(TurnDetection::Manual),
        };
        if setup.preemptive {
            config.session.turn.preemptive.enabled = Some(true);
            config.session.turn.preemptive.preemptive_tts = Some(true);
        }
        config.session.speech.max_tool_steps = setup.max_tool_steps;
        config.session.speech.min_consecutive_speech_delay =
            setup.speech_gap_ms.map(Duration::from_millis);
        config.session.turn.user_turn_limit.max_words =
            setup.user_turn_max_words.map(Nullable::Value);
        config.session.turn.user_turn_limit.max_duration =
            setup.user_turn_max_ms.map(|ms| Nullable::Value(Duration::from_millis(ms)));
        config.session.timers.transcription_timeout =
            setup.transcription_timeout_ms.map(|ms| Nullable::Value(Duration::from_millis(ms)));
        config.session.turn.ivr_detection = setup.ivr;
        config.session.turn.endpointing.mode = Some(if setup.dynamic_endpointing {
            EndpointingMode::Dynamic
        } else {
            EndpointingMode::Fixed
        });
        // The host asked for the adaptive detector, and the host's own gate -- the driver's
        // calculator, condition for condition -- says whether it gets one. A port under `Manual`
        // turn detection, say, is a session no host ever assembles: the gate refuses it because
        // there is nothing to withhold transcripts with, and so does this harness.
        let mut adaptive = false;
        if setup.interruption {
            config.session.turn.interruption.enabled = Some(true);
            config.session.turn.interruption.mode = Some(InterruptionMode::Adaptive);
            let gate = resolve_interruption_detector(&InterruptionGateInputs {
                stt: (setup.stt && setup.realtime.is_none())
                    .then_some(SttGate { aligned_transcript: true, streaming: true }),
                realtime: setup
                    .realtime
                    .is_some()
                    .then_some(RealtimeGate { server_turn_detection: !setup.vad && !setup.stt }),
                has_vad: setup.vad,
                turn_detection: config.session.turn.detection.unwrap_or(TurnDetection::Auto),
                allow_interruptions: true,
                asked: [None, Some(InterruptionMode::Adaptive)],
                deployment: Deployment::Dev,
            });
            adaptive = matches!(gate, InterruptionGate::Build);
        }
        // The legs the driver resolves its defaults and derived switches from -- the turn owner,
        // the endpointing base, whether an adaptive detector judges overlaps. What is handed to it
        // below and what it is told it has must agree: a VAD it was not told about is one it
        // never trusts for timing, and a detector it was not told about judges nothing.
        let connect = Duration::from_millis(setup.connect_ms.unwrap_or(0));
        // The two outputs, and the synchroniser between them when the session has one: the same
        // wiring `SessionSpec` does, one layer up, so what the driver forwards to is the node
        // whose word timing is under test.
        let audio_leaf = fakes::FakeAudioOutput::new(registry.clone(), setup.pausable_output);
        let text_leaf = fakes::FakeTextOutput::new(registry.clone());
        let (audio_out, text_out, sync): (Arc<dyn AudioOutput>, Arc<dyn TextOutput>, _) =
            if setup.sync_transcription {
                use lk_agents_voice::transcription::synchronizer::{
                    Liang, SyncConfig, TranscriptSynchronizer,
                };
                let sync = TranscriptSynchronizer::new(SyncConfig {
                    audio_next: audio_leaf,
                    text_next: Some(text_leaf),
                    speed: 1.0,
                    hyphenator: Arc::new(Liang),
                    clock: time.clock(),
                    spawner: tasks.clone(),
                });
                (sync.audio_output(), sync.text_output(), Some(sync))
            } else {
                (audio_leaf, text_leaf, None)
            };
        let refs: Arc<dyn RefRegistry> = Arc::new(NoopRefs);
        let legs = LegSet {
            stt: setup.stt,
            vad: setup.vad,
            llm: true,
            // The realtime socket is the voice: no TTS leg, and the model owns the reply.
            tts: setup.tts && setup.realtime.is_none(),
            realtime: setup.realtime.is_some(),
            // The fake predictor takes the user's audio as it arrives and answers a request.
            detector: setup.eot.then_some(DetectorKind::Streaming),
            adaptive_detector: adaptive,
            sink_can_pause: setup.pausable_output,
            audio_output: true,
            audio_enabled: true,
            ..LegSet::none()
        };
        // The room's two channels are the harness's to feed, and the link's are the session's to
        // answer, so both ends are kept here for as long as the run lasts.
        let (room_tx, room_rx) = tokio::sync::mpsc::unbounded_channel();
        let (room_io_tx, room_io_rx) = tokio::sync::mpsc::unbounded_channel();
        let room = setup.room.map(|room| RoomLinkConfig {
            close_on_disconnect: room.close_on_disconnect,
            delete_room_on_close: room.delete_room_on_close,
            room: "room".into(),
        });

        // A voice call always has a microphone; the legs -- STT, VAD, the end-of-turn model, the
        // realtime socket -- are what listens to it. Minted like an output so its `nth` is what a
        // `Ref<Mic>` names.
        let mic = {
            let (mic, frames) = fakes::FakeAudioInput::new();
            Some((mic, (registry.new_mic(), frames)))
        };
        let (mut driver, session) = DriverBuilder {
            time,
            spawner: tasks.clone(),
            config: DriverConfig::new(config, legs),
            thresholds: lk_agents_abi::tokenize::language::ThresholdOptions::new(
                lk_agents_abi::tokenize::language::TurnDetectorModel::V1Mini,
                None,
                setup
                    .backchannel_threshold
                    .map(|percent| ThresholdOverride::All(f64::from(percent) / 100.0)),
            ),
            agent: "agent".into(),
            instructions: instructions(&setup.instructions),
            chat_ctx: None,
            hook_bound: setup.hook_bound_ms.map(Duration::from_millis),
            capture_run: None,
            max_unrecoverable_errors: setup.max_unrecoverable_errors,
            llm: (setup.realtime.is_none())
                .then(|| fakes::FakeLlm::new(registry.clone()) as Arc<Llm>),
            tts: (setup.tts && setup.realtime.is_none())
                .then(|| fakes::FakeTts::new(registry.clone()) as Arc<Tts>),
            stt: setup.stt.then(|| fakes::FakeStt::new(registry.clone(), connect) as Arc<Stt>),
            audio_in: mic.as_ref().map(|(mic, _)| mic.clone() as Arc<AudioIn>),
            audio_out: Some(audio_out),
            text_out: Some(text_out),
            toolset: toolset(&registry, &setup.tools),
            speaker_filter: setup.speaker_filter.then(speaker_filter),
            keyterms: (!setup.keyterms.is_empty()).then(|| KeytermSetup {
                detection: KeytermDetection::default(),
                static_keyterms: setup.keyterms.clone(),
                llm: None,
                tools_host: None,
            }),
            vad: setup.vad.then(|| fakes::FakeVad::new(registry.clone()) as Arc<VadLeg>),
            video_in: setup
                .video
                .then(|| fakes::FakeVideoInput::new(&registry, refs.clone()) as Arc<VideoIn>),
            recorder: None,
            tts_text_transforms: vec![],
            realtime_audio_node: None,
            room_events: room.is_some().then_some(room_rx),
            room_io_events: room.is_some().then_some(room_io_rx),
            room_link: room,
            eot: setup.eot.then(|| fakes::FakeEot::new(registry.clone()) as Arc<EotDet>),
            interruption: adaptive.then(|| {
                fakes::FakeInterruption::new(registry.clone()) as Arc<dyn InterruptionPort>
            }),
            video_sampler: None,
            realtime: setup.realtime.map(|server| {
                fakes::FakeRealtime::new(registry.clone(), server.turn_detection)
                    as Arc<dyn Realtime>
            }),
        }
        .start()
        .await
        .expect("start session");

        // Tool bodies command the driver from their own origin through the session's queue.
        registry.attach_api(session.api.clone());

        let steps: Arc<Mutex<Vec<(Observed, Step)>>> = Arc::default();
        {
            let (steps, stamps) = (steps.clone(), stamps.clone());
            let mut trace = session.trace.lock().unwrap();
            // What the driver wrote while it was being built -- the config it resolved, the wall
            // epoch, the realtime socket's opening -- is journal too, and came before anyone
            // was listening. Taken under the same lock the observer is installed under, so no
            // step lands in between.
            for step in trace.steps() {
                steps.lock().unwrap().push((stamps.next(), step));
            }
            // Virtual time only advances when every task is idle, so a driver that keeps feeding
            // itself at one instant never returns control to the case: fail the run by name.
            let mut instant = (tokio::time::Instant::now(), 0u32);
            trace.observe(Box::new(move |step| {
                let now = tokio::time::Instant::now();
                instant = if now == instant.0 { (now, instant.1 + 1) } else { (now, 0) };
                assert!(
                    instant.1 < NO_PROGRESS_STEPS,
                    "no progress: {NO_PROGRESS_STEPS} journal steps without virtual time moving, last {step:?}"
                );
                let at = stamps.next();
                steps.lock().unwrap().push((at, step.clone()));
            }));
        }
        // The systems are the host's to start, right after `start()`: the navigator so an `Ivr`
        // command has a reader, the AMD as the reference's `await amd.execute()`.
        if setup.ivr.is_some() {
            lk_agents_voice::agent_session::start(&session.peer, session.ivr_detection);
        }
        if setup.amd {
            let llm = fakes::FakeLlm::new(registry.clone());
            let amd =
                AmdSystem::spawn(session.peer.clone(), AmdSetup::new(AmdConfig::default(), llm));
            // `await amd.execute()`: the host arms the detection budget and opens the listening
            // window as soon as the call is up. Attached but never started, the machine holds the
            // agent's authorization and hears nothing, which no host leaves it doing.
            let _ = amd.input(AmdIn::Start(AmdStart::DetectionTimer));
            let _ = amd.input(AmdIn::Start(AmdStart::Listening));
        }
        let driver = tokio::spawn(async move { driver.run().await });
        let mut run = Run {
            sync,
            room: room_tx,
            room_io: room_io_tx,
            session,
            driver: Some(driver),
            tasks,
            stamps,
            registry,
            records,
            controls: BTreeMap::new(),
            segments: BTreeMap::new(),
            steps,
            pending: BTreeMap::new(),
            streams: BTreeMap::new(),
            handles: BTreeMap::new(),
            utterances: BTreeMap::new(),
            mics: BTreeMap::new(),
            next_filler: 1,
            next_task: 1,
            refs,
            setup: setup.clone(),
            record: Recording::default(),
        };
        // The two outputs the builder attached exist before any input is applied; collecting here
        // is what registers them, together with whatever the first cycle has already decided.
        if let Some((_, (nth, frames))) = mic {
            run.mics.insert(nth, (frames, 0));
        }
        run.collect();
        run
    }
}

pub(super) type Stt = dyn lk_agents_abi::Stt;
pub(super) type Llm = dyn lk_agents_abi::Llm;
pub(super) type Tts = dyn lk_agents_abi::Tts;
pub(super) type AudioIn = dyn lk_agents_voice::io::AudioInput;
pub(super) type VadLeg = dyn lk_agents_abi::Vad;
pub(super) type EotDet = dyn lk_agents_abi::EotDetector;
pub(super) type VideoIn = dyn lk_agents_voice::io::VideoInput;

/// `Maybe` spells "not mentioned" and "set to nothing" differently, and a leg the case did not
/// name must be the former: `Given(None)` would tear the provider out.
pub(super) fn given<T>(value: Option<T>) -> Maybe<Option<T>> {
    value.map_or(Maybe::NotGiven, |value| Maybe::Given(Some(value)))
}

pub(super) fn instructions(common: &str) -> Instructions {
    Instructions { common: common.into(), audio: None, text: None }
}

/// A toolset over the controlled runner. Every frame gets a fresh runner on the same registry, so
/// a tool call after a handoff registers exactly like one before it.
pub(super) fn toolset(registry: &Arc<Registry>, tools: &[String]) -> Toolset {
    Toolset {
        specs: tools
            .iter()
            .map(|name| lk_agents_abi::ToolSpec {
                name: name.clone(),
                description: format!("fixture tool {name}"),
                json_schema: serde_json::json!({"type": "object", "properties": {}}),
                flags: Default::default(),
            })
            .collect(),
        host: None,
        runner: Some(fakes::FakeTools::new(registry.clone())),
    }
}

/// A new frame with the same fakes behind it: a handoff and an inline task differ only in how they
/// treat the frame they leave, which their own arms set.
pub(super) fn frame(
    registry: &Arc<Registry>,
    agent: &str,
    instructions: &str,
    tools: &[String],
) -> AgentUpdate {
    AgentUpdate {
        agent: agent.into(),
        instructions: self::instructions(instructions),
        toolset: toolset(registry, tools),
        chat_ctx: None,
        llm: Maybe::NotGiven,
        tts: Maybe::NotGiven,
        stt: Maybe::NotGiven,
        vad: Maybe::NotGiven,
        options: SessionPatch::default(),
        previous: PreviousActivity::Pause,
        next: NewActivity::Start,
        done: None,
        task: None,
    }
}

/// History the host supplies: one item per role, all carrying the same words, because what is
/// being varied is the shape of the context and not its content.
pub(super) fn history(roles: &[Role], text: &str) -> Vec<ChatItem> {
    roles
        .iter()
        .enumerate()
        .map(|(n, role)| {
            let role = match role {
                Role::System => lk_agents_abi::Role::System,
                Role::User => lk_agents_abi::Role::User,
                Role::Assistant => lk_agents_abi::Role::Assistant,
            };
            ChatItem::Message(ChatMessage::text(
                format!("case_item_{n}"),
                role,
                text.to_string(),
                std::time::UNIX_EPOCH,
            ))
        })
        .collect()
}

pub(super) fn speaker_filter() -> lk_agents_abi::voice::audio::SpeakerFilterConfig {
    lk_agents_abi::voice::audio::SpeakerFilterConfig {
        detect_primary_speaker: true,
        suppress_background_speaker: true,
        options: Default::default(),
        formats: Default::default(),
    }
}

/// Metrics from a leg the host owns. The values say nothing; that the driver accepts and forwards
/// them is the whole of what `HostMetrics` exercises.
pub(super) fn vad_metrics() -> AgentMetrics {
    AgentMetrics::Vad(lk_agents_abi::metrics::base::VadMetrics {
        label: "fake::vad".into(),
        timestamp: 0.0,
        idle_time: 0.0,
        inference_duration_total: 0.0,
        inference_count: 0,
        metadata: None,
    })
}

/// One live option, changed alone. The knob picks the field and the payload supplies whichever
/// value that field takes, so a finding names the one option that mattered.
pub(super) fn patch(
    knob: Knob,
    mode: TurnMode,
    ms: Millis,
    on: bool,
    words: Count,
) -> SessionPatch {
    let mut patch = SessionPatch::default();
    let delay = Some(Duration::from_millis(ms.0));
    match knob {
        Knob::Detection => {
            patch.turn.detection = Some(match mode {
                TurnMode::Default => TurnDetection::Auto,
                TurnMode::Stt => TurnDetection::Stt,
                TurnMode::Manual => TurnDetection::Manual,
            })
        }
        Knob::MinEndpointing => patch.turn.endpointing.min_delay = delay,
        Knob::MaxEndpointing => patch.turn.endpointing.max_delay = delay,
        Knob::Interruptions => patch.turn.interruption.enabled = Some(on),
        Knob::MinInterruptionWords => patch.turn.interruption.min_words = Some(words.0),
        Knob::UserAway => patch.timers.user_away = delay.map(Nullable::Value),
    }
    patch
}
