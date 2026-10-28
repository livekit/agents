//! Building the session through the host API, and the product objects a case's words become.
//!
//! [`Run::start`] assembles a real `VoiceSession` over the fake room and the fakes, as a host does,
//! and installs the probe before `start()`. Below it are the constructors the start and the calls
//! share: an agent, a history, a patch, built from the case's terms.
use super::*;
use lk_agents_abi::voice::console::ConsolePort;

impl Run {
    /// Build a session through the host's API, over the fake room and the fakes, and start it.
    ///
    /// The probe is installed before `start()`, so the first step the driver decides is stamped
    /// from the same counter as everything else the case observes. That one counter is what makes
    /// the recording a happens-before order rather than a pile of clocks.
    pub async fn start(setup: &Setup, schedule: u64, policy: Policy) -> Run {
        let time = Timebase::simulated();
        let stamps = Stamps::new(time.clone());
        let gate = (policy == Policy::TaskBridge).then(|| Arc::new(super::queue::Gate::new()));
        let tasks = Arc::new(Tasks {
            stamps: stamps.clone(),
            spawned: Mutex::default(),
            world: Mutex::default(),
            gate,
        });
        let world: Arc<dyn Spawner> = Arc::new(World(tasks.clone()));
        let (registry, records) = Registry::new(stamps.clone(), world.clone());
        // Named by the schedule: a room's name keys process-wide tables (the avatar's RPC routes),
        // and the survey runs seeds side by side in one process.
        let room = lk_agents_test_support::rtc::FakeRtc::new(time.clock())
            .room(&format!("room-{schedule}"));

        // The host's own gate (`agent_session.py`) decides whether an adaptive detector is handed
        // over: a port under `Manual` turn detection, say, is a session no host ever assembles.
        let adaptive = setup.interruption
            && matches!(
                resolve_interruption_detector(&InterruptionGateInputs {
                    stt: (setup.stt && setup.realtime.is_none()).then_some(SttGate {
                        aligned_transcript: true,
                        streaming: true
                    }),
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
        let console = setup
            .console
            .then(|| fakes::FakeCli::connect(world.clone(), time.clock()));
        let host_mic = (legs.is_none() && console.is_none()).then(|| {
            let (mic, frames) = fakes::FakeAudioInput::new();
            (mic, frames, registry.new_mic())
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
                publish: lk_agents_test_support::rtc::TrackPublishOptions {
                    sample_rate: SAMPLE_RATE,
                    ..Default::default()
                },
            }),
            video_input: setup.room_camera,
            text_output: Some(TextOutputConfig {
                sync_transcription: Some(setup.sync_transcription),
                ..TextOutputConfig::default()
            }),
            ..RoomIoConfig::default()
        };
        let pipeline = setup.realtime.is_none();
        // The providers, and the nodes an agent overriding them opens over each.
        let stt = setup.stt.then(|| session_stt(setup, &registry, connect));
        let llm = pipeline.then(|| fakes::FakeLlm::new(registry.clone()) as Arc<Llm>);
        // A realtime session has a voice too when the developer gives one: it speaks `say()`, and
        // a text-only model's replies.
        let tts = setup.tts.then(|| {
            fakes::FakeTts::new(
                registry.clone(),
                setup.tts_latency,
                !setup.chunked_tts,
                setup.tts_alignment,
            ) as Arc<Tts>
        });
        let providers = Legs {
            llm: llm.clone(),
            tts: tts.clone(),
            stt: stt.clone(),
        };
        let nodes = agent_nodes(setup, &registry, OPENING_AGENT, &providers);
        // The avatar worker is in the room before its stream is opened to it. Its reports cross
        // as they are sent, so the sink gives up on one after a clear as in production.
        let avatar_sink = (setup.avatar && console.is_none()).then(|| {
            room.add_participant(AVATAR, ParticipantKind::Agent);
            Arc::new(lk_agents_test_support::avatar_output::FakeAvatarOutput::new(
                room.clone(),
                AVATAR.into(),
                Some(SAMPLE_RATE),
                None,
                Some(Duration::from_secs(2)),
                setup.avatar_reports_start,
                time.clock(),
                tasks.clone(),
            ))
        });
        let api = Arc::new(
            SessionApi::host_new(
                Arc::new(FixedRoom {
                    rtc: room.clone(),
                    time: time.clone(),
                }),
                Some(instructions(&setup.instructions)),
                None,
                OPENING_AGENT.into(),
                None,
                None,
                // The thread's storage: the voice storage core runs in the storage survey.
                None,
                stt,
                setup
                    .vad
                    .then(|| fakes::FakeVad::new(registry.clone()) as Arc<VadLeg>),
                llm,
                setup.realtime.map(|server| {
                    let vad = server.turn_detection.then_some(if server.semantic {
                        fakes::RealtimeVad::Semantic
                    } else {
                        fakes::RealtimeVad::Server
                    });
                    let fake = |seed: u64, connect: Duration| {
                        let setup = fakes::ServerSetup {
                            vad,
                            seed,
                            connect,
                            tool_calls: server.tool_calls,
                            text_only: server.text_only,
                        };
                        fakes::FakeRealtime::new(
                            registry.clone(),
                            setup,
                            time.clock(),
                            tasks.clone(),
                        ) as Arc<dyn Realtime>
                    };
                    if server.duplex {
                        let clock = time.clock();
                        let duplex = fakes::FakeDuplex::new(
                            registry.clone(),
                            server.seed,
                            clock,
                            tasks.clone(),
                        );
                        return duplex as Arc<dyn Realtime>;
                    }
                    if !server.fallback {
                        return fake(server.seed, connect);
                    }
                    let children = vec![
                        fake(server.seed, connect),
                        // The fallback is a healthy model, connecting at once.
                        fake(server.seed.rotate_left(17), Duration::ZERO),
                    ];
                    fakes::FallbackRealtime::new(children, time.clock()) as Arc<dyn Realtime>
                }),
                None,
                tts,
                Some(Arc::new(WireSink)),
                // The run answers the hook calls itself (`HookAnswer`).
                None,
                None,
                true,
                SAMPLE_RATE,
                SAMPLE_RATE,
                Some(&options),
                None,
                None,
                setup.agent_aligned_transcript,
                setup
                    .eot
                    .then(|| fakes::FakeEot::new(registry.clone()) as Arc<EotDet>),
                adaptive.then(|| {
                    Arc::new(fakes::FakeInterruptionSource(registry.clone()))
                        as Arc<dyn InterruptionSource>
                }),
                true,
                Some(room_io),
                Some(tool_specs(&setup.tools)),
                None,
                Some(fakes::FakeTools::new(registry.clone()) as Arc<dyn HostToolRunner>),
                nodes.transcription,
                nodes.realtime_audio,
                nodes.llm,
                nodes.tts,
                nodes.stt,
                match &console {
                    Some((io, _)) => Some(io.clone().as_audio_input()),
                    None => host_mic
                        .as_ref()
                        .map(|(mic, _, _)| mic.clone() as Arc<AudioIn>),
                },
                match &console {
                    Some((io, _)) => Some(io.clone().as_audio_output()),
                    None if setup.avatar => avatar_sink
                        .as_ref()
                        .map(|sink| sink.clone() as Arc<dyn lk_agents_abi::voice::io::AudioOutput>),
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
                legs.is_none()
                    .then(|| fakes::FakeTextOutput::new(registry.clone()) as _),
                false,
                setup.hook_bound_ms.map(|ms| ms as f64 / 1000.0),
                setup.max_unrecoverable_errors,
                (!setup.keyterms.is_empty() || setup.keyterm_detection.is_some()).then(|| {
                    SttContextConfig {
                        keyterms: Some(setup.keyterms.clone()),
                        keyterm_detection: setup.keyterm_detection.map(|k| {
                            KeytermDetectionConfig {
                                enabled: Some(true),
                                llm: Some(DetectionLlm::Instance),
                                turn_interval: Some(k.turn_interval),
                                max_keyterms: k.max_keyterms,
                                instructions: None,
                                timeout: k.timeout_ms.map(|ms| ms as f64 / 1000.0),
                            }
                        }),
                        ..SttContextConfig::default()
                    }
                }),
                // `KeytermDetectionOptions(llm=...)`: the detection LLM the host resolved.
                setup
                    .keyterm_detection
                    .map(|_| fakes::FakeLlm::new(registry.clone()) as Arc<Llm>),
                None,
                lk_agents_abi::Maybe::NotGiven,
                Some("AS_fuzz".into()),
                // `ConsoleCall`: the session hosts the CLI's requests on the console's socket.
                console.as_ref().map(|_| {
                    Arc::new(fakes::FakeConsolePort)
                        as Arc<dyn lk_agents_voice::remote_session::SessionPort>
                }),
                console.as_ref().map(|(io, _)| {
                    io.clone() as Arc<dyn lk_agents_abi::voice::console::ConsolePort>
                }),
                None,
                tasks.clone(),
            )
            .expect("the configured voice fixture is valid"),
        );
        registry.attach_session(api.clone());

        let steps: Arc<Mutex<Vec<(Observed, Step)>>> = Arc::default();
        let host: Arc<Mutex<Vec<(Observed, Host)>>> = Arc::default();
        let livelock: Arc<Mutex<Option<String>>> = Arc::default();
        let (cores_tx, cores_rx) = oneshot::channel();
        {
            let (steps, step_stamps, livelock) = (steps.clone(), stamps.clone(), livelock.clone());
            // A step that journals forever never returns to the run's loop, which counts its own
            // iterations: the observer is the only one that sees it, and stops it the only way
            // anything can stop a step, by unwinding out of it.
            let mut instant = (tokio::time::Instant::now(), 0u32);
            let step = Box::new(move |step: &Step| {
                let now = tokio::time::Instant::now();
                instant = if now == instant.0 {
                    (now, instant.1 + 1)
                } else {
                    (now, 0)
                };
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
            api.probe(Probe { step, host_event });
        }
        api.start(false, Box::new(super::executor::Handover::new(cores_tx)))
            .expect("a new session starts")
            .await
            .expect("the session started");
        let cores = cores_rx.await.expect("the probe takes the session's cores");
        // The session started its host on the console's socket, which then carries the microphone.
        let console = console.map(|(_, cli)| cli);
        let console_mic = console
            .as_ref()
            .map(|cli| (registry.new_mic(), cli.clone()));
        // The answering-machine detector, as `AMD.__aenter__` attaches it once the call is up:
        // its config as the host resolves it; it then awaits the callee in the room.
        if let Some(amd) = setup.amd {
            use lk_agents_abi::voice::amd::{AmdConfigKwargs, TranscriptSource};
            let llm = fakes::FakeLlm::new(registry.clone());
            let stt = amd
                .own_stt
                .then(|| fakes::FakeStt::for_amd(registry.clone(), connect) as Arc<Stt>);
            let activity = api
                .resolved_options()
                .map(|o| Duration::from_secs_f64(o.endpointing.max_delay));
            let cfg = AmdConfigKwargs {
                max_endpointing_delay: Some(
                    lk_agents_voice::amd::classifier::resolve_max_endpointing_delay_for_host(
                        None, activity,
                    )
                    .as_secs_f64(),
                ),
                source: Some(if amd.own_stt {
                    TranscriptSource::AmdStt
                } else {
                    TranscriptSource::SessionStt
                }),
                wait_until_finished: amd.wait_until_finished,
                interrupt_on_machine: amd.interrupt_on_machine,
                ivr_detection: amd.ivr_detection,
                ..AmdConfigKwargs::default()
            };
            api.host_attach_amd(cfg, llm, None, stt, None)
                .expect("attach the AMD");
        }
        let mut run = Run {
            api,
            room,
            closing: None,
            host,
            cores: Some(cores),
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
            busiest_instant: 0,
            deadlock: None,
            pending: BTreeMap::new(),
            task_awaits: BTreeMap::new(),
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
            avatar: None,
            avatar_sink,
            refs,
            legs: providers,
            setup: setup.clone(),
            record: Recording::default(),
        };
        run.record
            .set_conn(setup.conn.unwrap_or(ConnSetup::DEFAULT), setup.chunked_tts);
        run.record.set_transcript(TranscriptSetup {
            synchronized: setup.sync_transcription,
            aligned: setup.agent_aligned_transcript.or(setup.aligned_transcript) == Some(true),
            // A streaming voice stamping its words. A voice that does not stream is adapted sentence
            // by sentence, the adapter stamping: not held to it (seed 31412, a three-line text,
            // came back with no stamp).
            stamps: setup.tts_alignment == TtsAlignment::Words && !setup.chunked_tts,
            adapted: setup.chunked_tts,
            late_reports: setup.avatar,
        });
        if let Some((nth, cli)) = console_mic {
            run.mics.insert(nth, (Mic::Console(cli), 0));
        }
        // `BackgroundAudioPlayer().start(room=, agent_session=)` with an ambient sound.
        if setup.background_audio {
            let player = Arc::new(BackgroundAudioPlayer::for_host(SAMPLE_RATE, 1, 2_400, 200));
            let room = Arc::new(FixedRoom {
                rtc: run.room.clone(),
                time: run.time.clone(),
            });
            let ambient = vec![fakes::FakeSoundStream::sound(20)];
            let _ = player
                .start_for_host(room, fakes::HALF, ambient, true, None)
                .await;
            run.background = Some(player);
        }
        if setup.avatar {
            let name = format!("avatar-{schedule}");
            let (fps, latency) = (setup.avatar_fps, setup.avatar_latency_ms);
            let worker = fakes::FakeAvatarWorker::start(
                run.registry.world.clone(),
                run.room.clone(),
                AVATAR,
                run.time.clock(),
                run.stamps.clone(),
                &name,
                schedule,
                fps,
                latency,
            )
            .await;
            run.avatar = Some(worker);
        }
        // The person joins the call the agent just answered; the callee AMD awaits may not have.
        if setup
            .amd
            .map_or(legs.is_some(), |amd| amd.answer != AmdAnswer::Missing)
        {
            run.join_caller();
        }
        run.collect();
        run
    }

    /// The person joins the room. With the room's legs they publish a microphone, which is then
    /// the session's (unless the host replaced it), and their app sends what it buffered before
    /// the agent was there, as the setup says. AMD's callee joins as the setup's answer says: a
    /// SIP call still ringing, and with their audio even when no leg of the session hears it.
    pub(super) fn join_caller(&mut self) {
        let sip = self
            .setup
            .amd
            .is_some_and(|amd| amd.answer == AmdAnswer::Sip);
        let kind = if sip {
            ParticipantKind::Sip
        } else {
            ParticipantKind::Standard
        };
        let caller = self.room.add_participant(CALLER, kind);
        if sip {
            caller.set_attribute(lk_agents_abi::voice::amd::SIP_CALL_STATUS_ATTR, "ringing");
        }
        let Some(legs) = self.setup.room_legs else {
            if self.setup.amd.is_some() {
                caller.publish_microphone("TR_callee");
            }
            return;
        };
        if self.host_mic {
            return;
        }
        self.joins += 1;
        let track = format!("TR_mic_{}", self.joins);
        let mic = caller.publish_microphone(track.as_str());
        if let Some(pre_connect) = legs.pre_connect {
            mic.set_audio_features(vec![
                lk_agents_test_support::rtc::AudioTrackFeature::PreconnectBuffer,
            ]);
            let mut attributes = std::collections::BTreeMap::from([("trackId".into(), track)]);
            if pre_connect != PreConnect::Malformed {
                attributes.insert("sampleRate".into(), SAMPLE_RATE.to_string());
                attributes.insert("channels".into(), "1".into());
            }
            let info = lk_agents_test_support::rtc::StreamInfo {
                name: None,
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
        self.mics.insert(
            nth,
            (
                Mic::Room {
                    track: mic,
                    burst: (0, 0),
                    read: None,
                },
                0,
            ),
        );
        // `RoomOptions(video_input=True)`: their camera, which the case feeds like the host's.
        if self.setup.room_camera {
            let camera = caller.publish_track(
                format!("TR_cam_{}", self.joins).as_str(),
                lk_agents_test_support::rtc::TrackSource::Camera,
            );
            fakes::room_camera(&self.registry, camera, self.refs.clone());
        }
    }
}

/// The avatar worker the agent's voice streams to.
pub(super) const AVATAR: &str = "avatar";

/// The person on the call: who joins, leaves and speaks in the room.
pub(super) const CALLER: &str = "caller";

/// `on_event`: the host's sink, handed each event's wire form. What the run reads is the typed
/// event the probe saw first; a console's events are the session's own to forward.
struct WireSink;

impl SessionEvents for WireSink {
    fn event(&self, _wire: lk_agents_abi::voice::events::HostWire) {}
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
        use_tts_aligned_transcript: setup
            .aligned_transcript
            .map_or(Maybe::NotGiven, Maybe::Given),
        tts_text_transforms: setup.text_transforms.map_or(Maybe::NotGiven, Maybe::Given),
        min_consecutive_speech_delay: secs(setup.speech_gap_ms.unwrap_or(0)),
        expressive: Default::default(),
        aec_warmup_duration: Maybe::NotGiven,
        ivr_detection: setup.ivr == Some(true),
        user_away_timeout: Some(15.0),
        transcription_timeout: setup.transcription_timeout_ms.map(secs),
        session_close_transcript_timeout: 2.0,
        conn_options: setup.conn.map_or_else(SessionConnOptions::default, |c| {
            let leg = |max_retry| lk_agents_abi::ApiConnectOptions {
                max_retry,
                retry_interval: secs(c.interval_ms),
                ..lk_agents_abi::ApiConnectOptions::default()
            };
            SessionConnOptions {
                stt: leg(c.stt),
                llm: leg(c.llm),
                tts: leg(c.tts),
            }
        }),
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
    Instructions {
        common: common.into(),
        audio: None,
        text: None,
    }
}

/// The tools advertised to the model; each is run by the controlled tool runner.
pub(super) fn tool_specs(tools: &[String]) -> Vec<lk_agents_abi::ToolSpec> {
    tools
        .iter()
        .map(|name| fakes::tools::tool_spec(name))
        .collect()
}

/// A new agent with the same fakes behind it, as `update_agent` and `enter_task` take it. Every
/// frame gets a fresh runner on the same registry, so a tool call after a handoff registers
/// exactly like one before it.
/// The session's providers as the host holds them.
#[derive(Clone, Default)]
pub(super) struct Legs {
    pub(super) llm: Option<Arc<Llm>>,
    pub(super) tts: Option<Arc<Tts>>,
    pub(super) stt: Option<Arc<Stt>>,
}

/// The node factories an agent installs: its `llm_node`, `tts_node` and `stt_node`, each over the
/// provider given beside it, and its `transcription_node` and `realtime_audio_output_node`.
#[derive(Default)]
pub(super) struct AgentNodes {
    pub(super) llm: Option<Arc<dyn NodeFactory>>,
    pub(super) tts: Option<Arc<dyn NodeFactory>>,
    pub(super) stt: Option<Arc<dyn NodeFactory>>,
    pub(super) transcription: Option<Arc<dyn NodeFactory>>,
    pub(super) realtime_audio: Option<Arc<dyn NodeFactory>>,
}

/// The nodes `agent` overrides: the opening agent's are the setup's, and every other agent flips
/// some of them by its name, so a handoff changes which agent's nodes must run.
pub(crate) fn overrides(setup: &Setup, agent: &str) -> Option<NodeSetup> {
    let on = setup.nodes?;
    if agent == OPENING_AGENT {
        return Some(on);
    }
    let bits = agent
        .bytes()
        .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(u32::from(b)));
    let flip = |bit: u32| bits & (1 << bit) != 0;
    Some(NodeSetup {
        llm: on.llm ^ flip(0),
        tts: on.tts ^ flip(1),
        stt: on.stt ^ flip(2),
        transcription: on.transcription ^ flip(3),
        ..on
    })
}

pub(super) fn agent_nodes(
    setup: &Setup,
    registry: &Arc<Registry>,
    agent: &str,
    legs: &Legs,
) -> AgentNodes {
    let Some(on) = overrides(setup, agent) else {
        return AgentNodes::default();
    };
    let node = |wanted: bool| {
        wanted.then(|| {
            fakes::FakeNodes::new(
                registry.clone(),
                agent,
                legs.llm.clone(),
                legs.tts.clone(),
                legs.stt.clone(),
                on.llm_text,
            ) as Arc<dyn NodeFactory>
        })
    };
    AgentNodes {
        llm: node(on.llm && legs.llm.is_some()),
        tts: node(on.tts && legs.tts.is_some()),
        stt: node(on.stt && legs.stt.is_some()),
        transcription: node(on.transcription),
        realtime_audio: node(on.realtime_audio),
    }
}

/// A handoff's frame: the agent's tools and, when its agents override nodes, every leg a node
/// wraps given again with the node over it (`_frame` gives the session's providers each time).
pub(super) fn frame(
    setup: &Setup,
    legs: &Legs,
    registry: &Arc<Registry>,
    agent: &str,
    instructions: &str,
    tools: &[String],
) -> UpdateAgentArgs {
    let mut args = plain_frame(registry, agent, instructions, tools);
    args.use_tts_aligned_transcript = setup.agent_aligned_transcript;
    if setup.nodes.is_some() {
        let nodes = agent_nodes(setup, registry, agent, legs);
        args.llm_node = nodes.llm;
        args.tts_node = nodes.tts;
        args.stt_node = nodes.stt;
        args.transcription_node = nodes.transcription;
        args.realtime_audio_output_node = nodes.realtime_audio;
        args.stt = Maybe::Given(legs.stt.clone());
        args.llm = if setup.realtime.is_none() {
            Maybe::Given(legs.llm.clone())
        } else {
            Maybe::NotGiven
        };
        args.tts = Maybe::Given(legs.tts.clone());
    }
    args
}

fn plain_frame(
    registry: &Arc<Registry>,
    agent: &str,
    instructions: &str,
    tools: &[String],
) -> UpdateAgentArgs {
    UpdateAgentArgs {
        instructions: self::instructions(instructions),
        dynamic_instructions: None,
        agent: agent.into(),
        agent_host: None,
        tools: Some(tool_specs(tools)),
        tools_host: None,
        tool_runner: Some(fakes::FakeTools::new(registry.clone())),
        chat_ctx: None,
        llm: Maybe::NotGiven,
        tts: Maybe::NotGiven,
        stt: Maybe::NotGiven,
        vad: Maybe::NotGiven,
        turn_handling: None,
        min_consecutive_speech_delay: None,
        use_tts_aligned_transcript: None,
        expressive: Maybe::NotGiven,
        llm_node: None,
        tts_node: None,
        stt_node: None,
        transcription_node: None,
        realtime_audio_output_node: None,
    }
}

/// The session's recogniser: one that takes keyterms when the setup's detection says so.
pub(super) fn session_stt(setup: &Setup, registry: &Arc<Registry>, connect: Duration) -> Arc<Stt> {
    if setup
        .keyterm_detection
        .is_some_and(|k| k.stt_takes_keyterms)
    {
        fakes::FakeStt::taking_keyterms(registry.clone(), connect)
    } else {
        fakes::FakeStt::new(registry.clone(), connect)
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
) -> AgentSessionOptions {
    let mut patch = AgentSessionOptions::default();
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
        Knob::UserAway => patch.timers.user_away = lk_agents_abi::Maybe::Given(delay),
        Knob::MaxToolSteps => patch.speech.max_tool_steps = Some(words.0),
    }
    patch
}
