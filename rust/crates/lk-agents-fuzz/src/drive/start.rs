//! Building the session through the host API, and the product objects a case's words become.
//!
//! [`Run::start`] assembles a real `LiveSession` over the fake room and the fakes, as a host does,
//! and installs the probe before `start()`. Below it are the constructors the start and the calls
//! share: an agent, a history, a patch, built from the case's terms.
use super::*;

impl Run {
    /// Build a session through the host's API, over the fake room and the fakes, and start it.
    ///
    /// The probe is installed before `start()`, so the first step the driver decides is stamped
    /// from the same counter as everything else the case observes. That one counter is what makes
    /// the recording a happens-before order rather than a pile of clocks.
    pub async fn start(setup: &Setup, schedule: u64, policy: Policy) -> Run {
        let time = Timebase::simulated();
        let stamps = Stamps::new(time.clone());
        let (registry, records) = Registry::new(stamps.clone());
        let tasks = Arc::new(Tasks { stamps: stamps.clone(), spawned: Mutex::default() });
        let room = lk_agents_rtc::FakeRtc::new(time.clock()).room("room");

        // The host's own gate (`agent_session.py`) decides whether an adaptive detector is handed
        // over: a port under `Manual` turn detection, say, is a session no host ever assembles.
        let adaptive = setup.interruption
            && matches!(
                resolve_interruption_detector(&InterruptionGateInputs {
                    stt: (setup.stt && setup.realtime.is_none())
                        .then_some(SttGate { aligned_transcript: true, streaming: true }),
                    realtime: setup.realtime.is_some().then_some(RealtimeGate {
                        server_turn_detection: !setup.vad && !setup.stt
                    }),
                    has_vad: setup.vad,
                    turn_detection: turn_detection(setup.turn).unwrap_or(TurnDetection::Auto),
                    allow_interruptions: true,
                    asked: [None, Some(InterruptionMode::Adaptive)],
                    deployment: Deployment::Dev,
                }),
                InterruptionGate::Build
            );
        let connect = Duration::from_millis(setup.connect_ms.unwrap_or(0));
        let refs: Arc<dyn RefRegistry> = Arc::new(NoopRefs);
        // A voice call always has a microphone; the legs are what listens to it. Minted like an
        // output so its `nth` is what a `Ref<Mic>` names.
        let (mic, frames) = fakes::FakeAudioInput::new();
        let mic_nth = registry.new_mic();
        let options = HostOptions::new(session_kwargs(setup));
        let room_io = RoomIoConfig {
            close_on_disconnect: setup.room.is_some_and(|r| r.close_on_disconnect),
            delete_room_on_close: setup.room.is_some_and(|r| r.delete_room_on_close),
            // The host's own legs replace the room's, which `agent_session.py` turns off.
            audio_input: None,
            audio_output: None,
            text_output: Some(TextOutputConfig {
                sync_transcription: Some(setup.sync_transcription),
                ..TextOutputConfig::default()
            }),
            ..RoomIoConfig::default()
        };
        let pipeline = setup.realtime.is_none();
        let api = Arc::new(
            SessionApi::host_new(
                Arc::new(FixedRoom { rtc: room.clone(), time: time.clone() }),
                Some(instructions(&setup.instructions)),
                None,
                "agent".into(),
                None,
                setup.stt.then(|| fakes::FakeStt::new(registry.clone(), connect) as Arc<Stt>),
                setup.vad.then(|| fakes::FakeVad::new(registry.clone()) as Arc<VadLeg>),
                pipeline.then(|| fakes::FakeLlm::new(registry.clone()) as Arc<Llm>),
                setup.realtime.map(|server| {
                    fakes::FakeRealtime::new(registry.clone(), server.turn_detection)
                        as Arc<dyn Realtime>
                }),
                None,
                (setup.tts && pipeline).then(|| fakes::FakeTts::new(registry.clone()) as Arc<Tts>),
                Some(Arc::new(WireSink)),
                None,
                true,
                SAMPLE_RATE,
                SAMPLE_RATE,
                Some(&options),
                None,
                setup.eot.then(|| fakes::FakeEot::new(registry.clone()) as Arc<EotDet>),
                true,
                adaptive.then(|| {
                    Arc::new(fakes::FakeInterruptionSource(registry.clone()))
                        as Arc<dyn InterruptionSource>
                }),
                true,
                Some(room_io),
                Some(tool_specs(&setup.tools)),
                None,
                Some(fakes::FakeTools::new(registry.clone()) as Arc<dyn HostToolRunner>),
                None,
                None,
                None,
                Some(mic as Arc<AudioIn>),
                Some(fakes::FakeAudioOutput::new(registry.clone(), setup.pausable_output) as _),
                setup
                    .video
                    .then(|| fakes::FakeVideoInput::new(&registry, refs.clone()) as Arc<VideoIn>),
                None,
                false,
                Some(fakes::FakeTextOutput::new(registry.clone()) as _),
                false,
                setup.hook_bound_ms.map(|ms| ms as f64 / 1000.0),
                setup.max_unrecoverable_errors,
                (!setup.keyterms.is_empty()).then(|| SttContextConfig {
                    keyterms: Some(setup.keyterms.clone()),
                    ..SttContextConfig::default()
                }),
                None,
                None,
                tasks.clone(),
            )
            .expect("assemble the session"),
        );
        registry.attach_session(api.clone());

        let steps: Arc<Mutex<Vec<(Observed, Step)>>> = Arc::default();
        let host: Arc<Mutex<Vec<(Observed, Host)>>> = Arc::default();
        let livelock: Arc<Mutex<Option<String>>> = Arc::default();
        let (runtime_tx, runtime_rx) = oneshot::channel();
        {
            let (steps, step_stamps, livelock) = (steps.clone(), stamps.clone(), livelock.clone());
            // A step that journals forever never returns to the run's loop, which counts its own
            // iterations: the observer is the only one that sees it, and stops it the only way
            // anything can stop a step, by unwinding out of it.
            let mut instant = (tokio::time::Instant::now(), 0u32);
            let step = Box::new(move |step: &Step| {
                let now = tokio::time::Instant::now();
                instant = if now == instant.0 { (now, instant.1 + 1) } else { (now, 0) };
                let at = step_stamps.next();
                if instant.1 >= NO_PROGRESS_STEPS {
                    let detail = format!(
                        "{NO_PROGRESS_STEPS} journal steps at {} ms without virtual time moving, \
                         the last {step:?}",
                        at.at_ms
                    );
                    *livelock.lock().unwrap() = Some(detail.clone());
                    panic!("{detail}");
                }
                steps.lock().unwrap().push((at, step.clone()));
            });
            let (host, stamps) = (host.clone(), stamps.clone());
            let host_event = Box::new(move |event: &HostEvent| {
                let seen = match event {
                    HostEvent::Session(event) => Host::Session(event.clone()),
                    HostEvent::Hook(call) => hook_of(call),
                    HostEvent::Room(event) => Host::Room(event.clone()),
                };
                host.lock().unwrap().push((stamps.next(), seen));
            });
            api.probe(Probe { step, host_event, runtime: runtime_tx });
        }
        api.start(false).expect("a new session starts").await.expect("the session started");
        let runtime = runtime_rx.await.expect("the probe takes the session's cores");
        // The answering-machine detector, as `await amd.execute()` runs it: attached, its budget
        // armed and its listening window opened as soon as the call is up.
        if setup.amd {
            let llm = fakes::FakeLlm::new(registry.clone());
            api.host_attach_amd(Default::default(), llm, None, None).expect("attach the AMD");
            api.amd_start_detection_timer().expect("a running session");
            api.amd_start_listening().expect("a running session");
        }
        let mut run = Run {
            api,
            room,
            closing: None,
            host,
            runtime: Some(runtime),
            queue: super::queue::Queue::new(schedule, policy),
            time,
            tasks,
            stamps,
            registry,
            records,
            controls: BTreeMap::new(),
            segments: BTreeMap::new(),
            steps,
            livelock,
            pending: BTreeMap::new(),
            streams: BTreeMap::new(),
            handles: BTreeMap::new(),
            utterances: BTreeMap::new(),
            mics: BTreeMap::from([(mic_nth, (frames, 0))]),
            next_filler: 1,
            next_task: 1,
            refs,
            setup: setup.clone(),
            record: Recording::default(),
        };
        run.collect();
        run
    }
}

/// `on_event`: the host's sink, handed each event's wire form. What the run reads is the typed
/// event the probe saw first.
struct WireSink;

impl SessionEvents for WireSink {
    fn event(&self, _wire: &lk_agents_abi::voice::events::HostWire) {}
    fn stream_closed(&self, _wire: &lk_agents_abi::voice::events::StreamClosedWire) {}
}

pub(super) fn turn_detection(mode: TurnMode) -> Option<TurnDetection> {
    match mode {
        TurnMode::Default => None,
        TurnMode::Stt => Some(TurnDetection::Stt),
        TurnMode::Manual => Some(TurnDetection::Manual),
    }
}

/// `AgentSession(**kwargs)` for a setup: the reference's defaults, and what the setup changes.
fn session_kwargs(setup: &Setup) -> SessionKwargs {
    let secs = |ms: u64| ms as f64 / 1000.0;
    let limit = setup.user_turn_max_words.is_some() || setup.user_turn_max_ms.is_some();
    SessionKwargs {
        turn_handling: Some(TurnHandlingKwargs {
            turn_detection: turn_detection(setup.turn).map(TurnDetectionKwarg::Mode),
            endpointing: Some(EndpointingKwargs {
                mode: Some(if setup.dynamic_endpointing {
                    EndpointingMode::Dynamic
                } else {
                    EndpointingMode::Fixed
                }),
                ..EndpointingKwargs::default()
            }),
            interruption: setup.interruption.then(|| InterruptionKwargs {
                enabled: Some(true),
                mode: Some(InterruptionMode::Adaptive),
                ..InterruptionKwargs::default()
            }),
            preemptive_generation: setup.preemptive.then(|| PreemptiveKwargs {
                enabled: Some(true),
                preemptive_tts: Some(true),
                ..PreemptiveKwargs::default()
            }),
            user_turn_limit: limit.then(|| UserTurnLimitKwargs {
                max_words: Maybe::Given(setup.user_turn_max_words.map(i64::from)),
                max_duration: Maybe::Given(setup.user_turn_max_ms.map(secs)),
            }),
        }),
        stt_context_options: HostValue::NotGiven,
        tool_handling: HostValue::NotGiven,
        max_tool_steps: setup.max_tool_steps.map_or(3, i64::from),
        use_tts_aligned_transcript: Maybe::NotGiven,
        tts_text_transforms: Maybe::NotGiven,
        min_consecutive_speech_delay: secs(setup.speech_gap_ms.unwrap_or(0)),
        expressive: false,
        aec_warmup_duration: Maybe::NotGiven,
        ivr_detection: setup.ivr == Some(true),
        user_away_timeout: Some(15.0),
        transcription_timeout: setup.transcription_timeout_ms.map(secs),
        session_close_transcript_timeout: 2.0,
        conn_options: HostValue::NotGiven,
        deprecated: Default::default(),
        keyterms_options: HostValue::NotGiven,
    }
}

pub(super) type Stt = dyn lk_agents_abi::Stt;
pub(super) type Llm = dyn lk_agents_abi::Llm;
pub(super) type Tts = dyn lk_agents_abi::Tts;
pub(super) type AudioIn = dyn lk_agents_abi::voice::io::AudioInput;
pub(super) type VadLeg = dyn lk_agents_abi::Vad;
pub(super) type EotDet = dyn lk_agents_abi::EotDetector;
pub(super) type VideoIn = dyn lk_agents_abi::voice::io::VideoInput;

pub(super) fn instructions(common: &str) -> Instructions {
    Instructions { common: common.into(), audio: None, text: None }
}

/// The tools advertised to the model; each is run by the controlled tool runner.
pub(super) fn tool_specs(tools: &[String]) -> Vec<lk_agents_abi::ToolSpec> {
    tools
        .iter()
        .map(|name| lk_agents_abi::ToolSpec {
            name: name.clone(),
            description: format!("fixture tool {name}"),
            json_schema: serde_json::json!({"type": "object", "properties": {}}),
            flags: Default::default(),
        })
        .collect()
}

/// A new agent with the same fakes behind it, as `update_agent` and `enter_task` take it. Every
/// frame gets a fresh runner on the same registry, so a tool call after a handoff registers
/// exactly like one before it.
pub(super) fn frame(
    registry: &Arc<Registry>,
    agent: &str,
    instructions: &str,
    tools: &[String],
) -> UpdateAgentArgs {
    UpdateAgentArgs {
        instructions: self::instructions(instructions),
        dynamic_instructions: None,
        agent: agent.into(),
        tools: Some(tool_specs(tools)),
        tools_host: None,
        tool_runner: Some(fakes::FakeTools::new(registry.clone())),
        chat_ctx: None,
        llm: Maybe::NotGiven,
        tts: Maybe::NotGiven,
        stt: Maybe::NotGiven,
        vad: Maybe::NotGiven,
        turn_handling: None,
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
        Knob::MaxToolSteps => patch.speech.max_tool_steps = Some(words.0),
    }
    patch
}
