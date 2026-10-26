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
        // The deprecated keywords have no interruption mode to ask for adaptive with.
        let adaptive = setup.interruption
            && !setup.deprecated_turn
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
        // A voice call always has a microphone; the legs are what listens to it. The host's own
        // is minted like an output, so its `nth` is what a `Ref<Mic>` names; the room's is the
        // person's track, minted when they join.
        let legs = setup.room_legs;
        // The console's socket is a leg of the host's like the fakes, so the room's stay off.
        let console = setup.console.then(|| fakes::FakeCli::connect(Arc::new(TokioSpawner::current())));
        let host_mic = (legs.is_none() && console.is_none()).then(|| {
            let (mic, frames) = fakes::FakeAudioInput::new();
            (mic, frames, registry.new_mic())
        });
        let session_host = console.as_ref().map(|_| {
            let host = RemoteSessionHost::new(time.clock(), Arc::new(TokioSpawner::current()));
            host.set_port(Arc::new(fakes::FakeConsolePort));
            Arc::new(host)
        });
        let options = HostOptions::new(session_kwargs(setup));
        let room_io = RoomIoConfig {
            close_on_disconnect: setup.room.is_some_and(|r| r.close_on_disconnect),
            delete_room_on_close: setup.room.is_some_and(|r| r.delete_room_on_close),
            // With its own legs, the host has turned the room's off, as `agent_session.py` does.
            audio_input: legs.map(|legs| AudioInputConfig {
                sample_rate: SAMPLE_RATE,
                pre_connect_audio: legs.pre_connect.is_some(),
                // Shorter than the half second of frames the track's queue holds, so the frames
                // spoken while the session waits for a buffer that never comes are not dropped.
                pre_connect_audio_timeout: Duration::from_millis(200),
                ..AudioInputConfig::default()
            }),
            audio_output: legs.map(|_| AudioOutputConfig {
                publish: lk_agents_rtc::TrackPublishOptions {
                    sample_rate: SAMPLE_RATE,
                    ..Default::default()
                },
            }),
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
                Some(Arc::new(WireSink { console: session_host.clone(), time: time.clone() })),
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
                None,
                None,
                match &console {
                    Some((io, _)) => Some(io.as_audio_input()),
                    None => host_mic.as_ref().map(|(mic, _, _)| mic.clone() as Arc<AudioIn>),
                },
                match &console {
                    Some((io, _)) => Some(io.as_audio_output()),
                    None => legs.is_none().then(|| {
                        fakes::FakeAudioOutput::new(
                            registry.clone(),
                            setup.pausable_output,
                            setup.lagging_output,
                        ) as _
                    }),
                },
                setup
                    .video
                    .then(|| fakes::FakeVideoInput::new(&registry, refs.clone()) as Arc<VideoIn>),
                setup.video_sampler.then_some((1.0, 0.3)),
                false,
                legs.is_none().then(|| fakes::FakeTextOutput::new(registry.clone()) as _),
                false,
                setup.hook_bound_ms.map(|ms| ms as f64 / 1000.0),
                setup.max_unrecoverable_errors,
                (!setup.keyterms.is_empty()).then(|| SttContextConfig {
                    keyterms: Some(setup.keyterms.clone()),
                    ..SttContextConfig::default()
                }),
                None,
                None,
                lk_agents_abi::Maybe::NotGiven,
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
        // `ConsoleCall.start()`: the host runs on the socket, which then carries the microphone.
        let console = match (console, session_host) {
            (Some((io, cli)), Some(host)) => {
                host.start(io.as_transport(), Some(io.as_console()))
                    .await
                    .expect("the console's socket is open");
                Some((host, cli))
            }
            _ => None,
        };
        let console_mic = console.as_ref().map(|(_, cli)| (registry.new_mic(), cli.clone()));
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
            mics: host_mic
                .into_iter()
                .map(|(_, frames, nth)| (nth, (Mic::Host(frames), 0)))
                .collect(),
            host_mic: false,
            joins: 0,
            console,
            next_request: 0,
            background: None,
            next_filler: 1,
            refs,
            setup: setup.clone(),
            record: Recording::default(),
        };
        if let Some((nth, cli)) = console_mic {
            run.mics.insert(nth, (Mic::Console(cli), 0));
        }
        // `BackgroundAudioPlayer().start(room=, agent_session=)` with an ambient sound.
        if setup.background_audio {
            let player = Arc::new(BackgroundAudioPlayer::for_host(SAMPLE_RATE, 1, 2_400, 200));
            let room = Arc::new(FixedRoom { rtc: run.room.clone(), time: run.time.clone() });
            if player.start_for_host(room, None).await.is_ok() {
                let ambient = fakes::FakeSoundStream::sound(20);
                let _ = player.play(vec![ambient], true, false, true, &mut || 0.5);
            }
            run.background = Some(player);
        }
        // The person joins the call the agent just answered.
        if legs.is_some() {
            run.join_caller();
        }
        run.collect();
        run
    }

    /// The person joins the room. With the room's legs they publish a microphone, which is then
    /// the session's (unless the host replaced it), and their app sends what it buffered before
    /// the agent was there, as the setup says.
    pub(super) fn join_caller(&mut self) {
        let caller = self.room.add_participant(CALLER, ParticipantKind::Standard);
        let Some(legs) = self.setup.room_legs else { return };
        if self.host_mic {
            return;
        }
        self.joins += 1;
        let track = format!("TR_mic_{}", self.joins);
        let mic = caller.publish_microphone(track.as_str());
        if let Some(pre_connect) = legs.pre_connect {
            mic.set_audio_features(vec![lk_agents_rtc::AudioTrackFeature::PreconnectBuffer]);
            let mut attributes = std::collections::HashMap::from([("trackId".into(), track)]);
            if pre_connect != PreConnect::Malformed {
                attributes.insert("sampleRate".into(), SAMPLE_RATE.to_string());
                attributes.insert("channels".into(), "1".into());
            }
            let info = lk_agents_rtc::StreamInfo {
                id: format!("pre_connect_{}", self.joins),
                topic: PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
                mime_type: "audio/pcm".into(),
                total_length: None,
                attributes,
                timestamp: 0,
                attachments: Vec::new(),
            };
            // Half a second of what the person said before the agent joined: samples no case
            // frame carries, so nothing mistakes them for the microphone's.
            let buffered = vec![0u8; SAMPLE_RATE as usize];
            if pre_connect != PreConnect::Never {
                let _ = self.room.send_byte_stream(CALLER, info, [buffered]);
            }
        }
        let nth = self.registry.new_mic();
        self.mics.insert(nth, (Mic::Room { track: mic, burst: (0, 0) }, 0));
    }
}

/// The person on the call: who joins, leaves and speaks in the room.
pub(super) const CALLER: &str = "caller";

/// `on_event`: the host's sink, handed each event's wire form. What the run reads is the typed
/// event the probe saw first; a console forwards what the CLI renders.
struct WireSink {
    console: Option<Arc<RemoteSessionHost>>,
    time: Timebase,
}

impl SessionEvents for WireSink {
    fn event(&self, wire: &lk_agents_abi::voice::events::HostWire) {
        let (Some(host), Some(event)) = (&self.console, console_event(wire)) else { return };
        let _ = host.send_wire(event, Some(self.time.wall_secs()));
    }
    fn stream_closed(&self, _wire: &lk_agents_abi::voice::events::StreamClosedWire) {}
}

/// `ConsoleCall.forward` and `_tools_executed`: the events the CLI renders, in its shapes.
fn console_event(wire: &lk_agents_abi::voice::events::HostWire) -> Option<WireEvent> {
    use lk_agents_abi::voice::events::{HostWire as W, ToolUpdate};
    use lk_agents_voice::remote_session::{
        ErrorWire, FunctionCallOutputRef, FunctionCallRef, FunctionToolsExecutedWire,
        ToolCallStartedWire, ToolExecutionUpdatedWire, ToolUpdateWire,
    };
    Some(match wire {
        W::AgentStateChanged(ev) => WireEvent::AgentStateChanged(ev.clone()),
        W::UserStateChanged(ev) => WireEvent::UserStateChanged(ev.clone()),
        W::UserTranscript(ev) => WireEvent::UserInputTranscribed(ev.clone()),
        W::EotPrediction(ev) => WireEvent::EotPrediction(ev.clone()),
        W::AgentFalseInterruption(ev) => WireEvent::AgentFalseInterruption(ev.clone()),
        W::ItemAdded(ev)
            if matches!(ev.item, ChatItem::Message(_) | ChatItem::AgentHandoff(_)) =>
        {
            WireEvent::ConversationItemAdded(ev.clone())
        }
        W::Error(ev) => WireEvent::Error(ErrorWire { message: ev.error.to_string() }),
        W::ToolExecutionUpdated(ev) => WireEvent::ToolExecutionUpdated(ToolExecutionUpdatedWire {
            update: match &ev.update {
                ToolUpdate::Started(started) => {
                    let call = &started.function_call;
                    ToolUpdateWire::Started(ToolCallStartedWire {
                        id: call.id.clone(),
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    })
                }
                ToolUpdate::Updated(updated) => ToolUpdateWire::CallUpdated(updated.clone()),
                ToolUpdate::Ended(ended) => ToolUpdateWire::Ended(ended.clone()),
                ToolUpdate::Reply(reply) => ToolUpdateWire::ReplyUpdated(reply.clone()),
            },
        }),
        W::FunctionToolsExecuted(ev) => WireEvent::FunctionToolsExecuted(FunctionToolsExecutedWire {
            function_calls: ev
                .function_calls
                .iter()
                .map(|c| FunctionCallRef {
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                    call_id: c.call_id.clone(),
                })
                .collect(),
            function_call_outputs: ev
                .function_call_outputs
                .iter()
                .map(|o| FunctionCallOutputRef {
                    call_id: o.call_id.clone(),
                    name: o.name.clone(),
                    output: o.output.clone(),
                    is_error: o.is_error,
                })
                .collect(),
        }),
        _ => return None,
    })
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
        // A host writing the deprecated keywords writes no `turn_handling`: the core migrates them.
        turn_handling: (!setup.deprecated_turn).then(|| TurnHandlingKwargs {
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
        tts_text_transforms: setup.text_transforms.map_or(Maybe::NotGiven, Maybe::Given),
        min_consecutive_speech_delay: secs(setup.speech_gap_ms.unwrap_or(0)),
        expressive: Default::default(),
        aec_warmup_duration: Maybe::NotGiven,
        ivr_detection: setup.ivr == Some(true),
        user_away_timeout: Some(15.0),
        transcription_timeout: setup.transcription_timeout_ms.map(secs),
        session_close_transcript_timeout: 2.0,
        conn_options: HostValue::NotGiven,
        deprecated: if setup.deprecated_turn {
            DeprecatedTurnKwargs {
                turn_detection: turn_detection(setup.turn)
                    .map_or(Maybe::NotGiven, |mode| Maybe::Given(TurnDetectionKwarg::Mode(mode))),
                min_endpointing_delay: Maybe::Given(0.4),
                max_endpointing_delay: Maybe::Given(4.0),
                false_interruption_timeout: Maybe::Given(Some(1.5)),
                discard_audio_if_uninterruptible: Maybe::Given(true),
                min_interruption_duration: Maybe::Given(0.4),
                min_interruption_words: Maybe::Given(1),
                allow_interruptions: Maybe::Given(true),
                resume_false_interruption: Maybe::Given(true),
                agent_false_interruption_timeout: Maybe::Given(Some(4.0)),
                preemptive_generation: Maybe::Given(setup.preemptive),
                ..DeprecatedTurnKwargs::default()
            }
        } else {
            Default::default()
        },
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
        expressive: Maybe::NotGiven,
        llm_node: None,
        tts_node: None,
        stt_node: None,
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
