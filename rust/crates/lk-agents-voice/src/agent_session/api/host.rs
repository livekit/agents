//! [`SessionApi`]'s host methods: the one `#[host(object)]` block the Python and Node
//! `AgentSession` call, each assembling, opening, closing or sending one command.

use super::{
    CLOSE_TIMEOUT, CommandRefused, HostParts,
    InterruptionPortFactory, ModelRefs, NodeFactory, NotAnImage, NotInterruptible, OpenRefused,
    Ready, Reply, SessionApi, SessionEvents, SessionRefused, SessionRoom,
    SessionJob, SessionRunner, Stage, StartFailed, UnknownCloseReason, UpdateAgentArgs,
    host_toolset, llm_leg, origin_of, session_tools, stt_leg, tts_leg,
};
use crate::agent_activity::DriverConfig;
use crate::agent_session::DriverBuilder;
use crate::agent_session::commands::{
    ActivityCommand, Command, DrainRefused, DtmfCommand, FillerSource, HoldId,
    HostAudio, HostVideo, IdleRefused, IoKind, IvrCommand, Origin, SayText, SpeechCommand,
    ToolUpdate, TurnCommand,
};
use crate::agent_session::nodes::{Nodes, OutputNodes};
use crate::agent_session::{Answer, CommitSuperseded, HostEvent};
use crate::filler_scheduler::FillerId;
use crate::generation::Instructions;
use crate::speech_handle::SpeechHandle;
use lk_agents_abi::voice::events::HostWire;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, TextOutput};
use lk_agents_abi::{Llm, Maybe, Spawner, SpawnerExt, Stt, Tts, Vad};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// `voice/agent_session.py`'s `AgentSession`, over the driver: assembled at construction, opened at
/// `start()`, closed at `aclose()`; every other method is one command. `frozen`, so there is no
/// borrow flag: `aclose()` can be awaited while the event drain calls back into the same object.
#[lk_agents_abi::host(object(name = "VoiceSession"))]
impl SessionApi {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// Assemble a session; the loop begins at `start()`.
    ///
    /// `room` is a `livekit.rtc.Room`, a `_core.FakeRoom` or a `_core.OfflineRoom`. `on_event`
    /// receives one `dict` per session event, on the calling loop. Must be called inside a running
    /// asyncio loop: the host objects' loop is captured here.
    #[host(new(
        kw_only = "instructions",
        defaults(
            agent = String::new(),
            auto_gain_control = true,
            input_sample_rate = 24_000,
            output_sample_rate = 24_000,
            text_output = true,
            host_video_sampler = false,
            using_default_vad = false,
            max_unrecoverable_errors = 3,
            agent_expressive = Maybe::NotGiven
        )
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn host_new(
        room: Arc<dyn SessionRoom>,
        instructions: Option<Instructions>,
        dynamic_instructions: Option<Arc<dyn crate::generation::HostInstructions>>,
        agent: String,
        // The first agent's `Agent`, handed back by the events that name it.
        agent_host: Option<lk_agents_abi::HostRef>,
        chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
        llm: Option<Arc<dyn Llm>>,
        // The native realtime model; a plugin's `RealtimeModel` is `realtime_host`.
        realtime: Option<Arc<dyn lk_agents_abi::Realtime>>,
        realtime_host: Option<Arc<dyn lk_agents_abi::llm::realtime::HostRealtime>>,
        tts: Option<Arc<dyn Tts>>,
        on_event: Option<Arc<dyn SessionEvents>>,
        hooks: Option<Arc<dyn super::SessionHooks>>,
        participant_identity: Option<String>,
        auto_gain_control: bool,
        input_sample_rate: u32,
        output_sample_rate: u32,
        // The session layer `_core.SessionOptions(kwargs)` built; none is an empty session layer.
        options: Option<&crate::agent_activity::options::HostOptions>,
        // The first agent's own `turn_handling`, as written, `min_consecutive_speech_delay` in
        // seconds and `use_tts_aligned_transcript`: the agent layer of the stack.
        agent_turn_handling: Option<crate::agent_activity::options::TurnHandlingKwargs>,
        agent_min_consecutive_speech_delay: Option<f64>,
        agent_use_tts_aligned_transcript: Option<bool>,
        // `_core_view.TurnDetector` over the user's audio `turn_detection`.
        turn_detector: Option<Arc<dyn lk_agents_abi::EotDetector>>,
        // The adaptive-interruption detector: an object means the host's gate said yes.
        interruption_detector: Option<
            Arc<dyn lk_agents_abi::turn::interruption::InterruptionSource>,
        >,
        text_output: bool,
        // `RoomOptions._to_rust_config()`, resolved per leg; without it the scalars above.
        room_io_config: Option<crate::room_io::RoomIoConfig>,
        tools: Option<Vec<lk_agents_abi::ToolSpec>>,
        tools_host: Option<lk_agents_abi::HostRef>,
        tool_runner: Option<Arc<dyn crate::tool_executor::HostToolRunner>>,
        transcription_node: Option<Arc<dyn NodeFactory>>,
        realtime_audio_output_node: Option<Arc<dyn NodeFactory>>,
        llm_node: Option<Arc<dyn NodeFactory>>,
        tts_node: Option<Arc<dyn NodeFactory>>,
        stt_node: Option<Arc<dyn NodeFactory>>,
        // The host's own io legs, replacing the room's.
        host_audio_input: Option<Arc<dyn AudioInput>>,
        host_audio_output: Option<Arc<dyn AudioOutput>>,
        host_video_input: Option<Arc<dyn lk_agents_abi::voice::io::VideoInput>>,
        // `VoiceActivityVideoSampler(speaking_fps, silent_fps)`'s two rates; `None`: every frame.
        video_sampler: Option<(f64, f64)>,
        // A custom `video_sampler` callable: the host pumps and gates; the driver reads no camera.
        host_video_sampler: bool,
        host_transcription_output: Option<Arc<dyn TextOutput>>,
        using_default_vad: bool,
        // The end-of-turn hook's bound, in seconds; `None` is no hook and no wait.
        hook_bound: Option<f64>,
        max_unrecoverable_errors: u32,
        stt_context: Option<lk_agents_abi::turn::keyterms::SttContextConfig>,
        keyterm_llm: Option<Arc<dyn Llm>>,
        keyterm_tools: Option<lk_agents_abi::HostRef>,
        // The first agent's own `expressive`; `NOT_GIVEN` defers to the session's.
        agent_expressive: Maybe<crate::agent_activity::options::Expressive>,
        // Prefixes the ids the session mints for chat items; `None` mints a random one.
        session_id: Option<String>,
        // Serve the remote session protocol (`lk agent simulate`, the dashboard, `lk agent
        // console`) on the room's byte streams, or on `console`: what a client asks that needs the
        // host's objects is the port's to answer.
        remote_session: Option<Arc<dyn crate::remote_session::SessionPort>>,
        // `lk agent console`'s socket: the session's audio input and output, and the remote
        // session's transport.
        console: Option<Arc<crate::remote_session::console::ConsoleIo>>,
        // The job whose room this is: `start()` connects it beside the opening.
        job: Option<Arc<dyn SessionJob>>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        use crate::room_io::{
            AudioInputConfig, AudioOutputConfig, RoomIoConfig, TextOutputConfig,
        };
        let agents = super::HostAgents::default();
        agents.install(&agent, agent_host);
        let time = room.timebase();
        let clock = time.clock();
        let anchor = lk_agents_abi::WallAnchor::of(&*clock);
        let mut config = room_io_config.unwrap_or_else(|| RoomIoConfig {
            participant_identity: participant_identity.map(Into::into),
            audio_input: Some(AudioInputConfig {
                sample_rate: input_sample_rate,
                auto_gain_control,
                ..AudioInputConfig::default()
            }),
            audio_output: Some(AudioOutputConfig {
                publish: lk_agents_rtc::TrackPublishOptions {
                    sample_rate: output_sample_rate,
                    ..Default::default()
                },
            }),
            text_output: text_output.then(TextOutputConfig::default),
            ..RoomIoConfig::default()
        });
        let host_audio_input =
            host_audio_input.or_else(|| console.as_ref().map(|c| c.as_audio_input()));
        let host_audio_output =
            host_audio_output.or_else(|| console.as_ref().map(|c| c.as_audio_output()));
        // `start()`: an audio leg the host set replaces the room's, which is then neither published
        // nor subscribed ("RoomIO audio output is enabled but output.audio is already set,
        // ignoring"). The room's text output stays: it paces a host transcription leg.
        if host_audio_input.is_some() {
            config.audio_input = None;
        }
        if host_audio_output.is_some() {
            config.audio_output = None;
        }
        // A host sink's playback reports are stamped on the session's clock.
        if let Some(out) = &host_audio_output {
            out.tracker().anchor_to(anchor);
        }
        let realtime = realtime.or_else(|| {
            realtime_host.map(|host| {
                Arc::new(lk_agents_abi::llm::realtime::HostRealtimeModel::new(
                    host,
                    anchor,
                    spawner.clone(),
                ))
                    as Arc<dyn lk_agents_abi::Realtime>
            })
        });
        let expressive = crate::generation::ExpressiveLayers {
            session: options.map(|o| o.expressive()).unwrap_or_default(),
            agent: agent_expressive.given(),
        };
        let session_layer = options.map(|o| o.session()).unwrap_or_default();
        let conn = options.map(|o| o.conn()).unwrap_or_default();
        let forward_chat_context =
            stt_context.as_ref().and_then(|cfg| cfg.forward_chat_context).unwrap_or(true);
        let keyterms = stt_context.map(|cfg| crate::keyterm_detection::KeytermSetup {
            detection: crate::keyterm_detection::resolve_detection(cfg.keyterm_detection.as_ref()),
            static_keyterms: cfg.keyterms.unwrap_or_default(),
            llm: keyterm_llm.map(|l| llm_leg(l, &conn.llm)),
            tools_host: keyterm_tools,
        });
        let stack = crate::agent_activity::options::ConfigStack {
            session: session_layer,
            agent: Some(crate::turn::agent_layer(
                agent_turn_handling.as_ref(),
                agent_min_consecutive_speech_delay,
                agent_use_tts_aligned_transcript,
            )),
        };
        let nodes = Nodes { llm: llm_node, tts: tts_node, stt: stt_node };
        let stt = stt.map(|s| nodes.stt(stt_leg(s, vad.clone(), &conn.stt, clock.clone())));
        let llm = llm.map(|l| nodes.llm(llm_leg(l, &conn.llm)));
        let tts = tts.map(|t| nodes.tts(tts_leg(t, clock.clone(), &conn.tts)));
        let vad_leg = vad.clone();
        let interruption = interruption_detector
            .map(|source| Box::new(move |anchor| source.open(anchor)) as InterruptionPortFactory);
        let toolset = host_toolset(tools, tools_host, tool_runner);
        let offline = room.offline();
        let room = room.rtc();
        let remote = remote_session.map(|port| {
            let transport = match &console {
                Some(console) => console.as_transport(),
                None => Arc::new(crate::remote_session::room::RoomTransport::new(
                    Arc::clone(&room),
                    spawner.clone(),
                )) as Arc<dyn crate::remote_session::SessionTransport>,
            };
            let wall = time.clone();
            Arc::new(crate::remote_session::driver::Hosting::new(
                port,
                transport,
                console.as_ref().map(|c| c.as_console()),
                clock.clone(),
                spawner.clone(),
                Box::new(move || wall.wall_secs()),
            ))
        });

        // **Nothing starts here**; the loop begins at `start()`. Synchronous so a host can bind the
        // room's legs and recorder before starting. Read before the config moves into `RoomIo`.
        let sync_speed = config.text_output.as_ref().map(|t| t.transcription_speed_factor);
        let room_link = crate::room_io::RoomLinkConfig {
            close_on_disconnect: config.close_on_disconnect,
            delete_room_on_close: config.delete_room_on_close,
            room: room.name().to_owned(),
        };
        let room_io = crate::room_io::RoomIo::new(
            Arc::clone(&room),
            clock.clone(),
            spawner.clone(),
            Arc::new(crate::room_io::_output::transcription::SegmentIds::default()),
            config,
            None,
        );

        // An avatar's sink is an audio output the room does not know; the transcript syncs to it.
        let has_external_audio_output = host_audio_output.is_some();

        let mut synchronizer = None;
        let (audio_out, text_out) = match (
            host_audio_output.or_else(|| room_io.audio_output()),
            host_transcription_output.or_else(|| room_io.transcription_output()),
        ) {
            (Some(audio), Some(text))
                if room_io.wants_transcript_sync(has_external_audio_output) =>
            {
                use crate::transcription::synchronizer::{
                    Liang, SyncConfig, TranscriptSynchronizer,
                };
                let sync = TranscriptSynchronizer::new(SyncConfig {
                    audio_next: audio,
                    text_next: Some(text),
                    speed: sync_speed.unwrap_or(1.0),
                    hyphenator: Arc::new(Liang),
                    clock: clock.clone(),
                    spawner: spawner.clone(),
                });
                let legs = (Some(sync.audio_output()), Some(sync.text_output()));
                synchronizer = Some(sync);
                legs
            }
            // No text leg, no audio leg, or `sync_transcription=False`: unpaced, as the reference.
            legs => legs,
        };

        // Facts, not options: the legs the two leg-dependent defaults need.
        let legs = crate::agent_activity::options::LegSet {
            stt: stt.is_some(),
            vad: vad.is_some(),
            using_default_vad,
            llm: llm.is_some(),
            tts: tts.is_some(),
            realtime: realtime.is_some(),
            detector: turn_detector
                .as_ref()
                .map(|_| crate::agent_activity::options::DetectorKind::Streaming),
            adaptive_detector: interruption.is_some(),
            audio_enabled: true,
            audio_output: true,
            sink_can_pause: true,
        };

        let resources = lk_agents_abi::resources::Resources::default();
        let builder = DriverBuilder {
            journal: Default::default(),
            resources: resources.clone(),
            session_id: session_id.unwrap_or_else(crate::agent_session::mint_session_id),
            agent,
            time: time.clone(),
            spawner: spawner.clone(),
            interruption: None,
            video_sampler: video_sampler.map(|(speaking, silent)| {
                Arc::new(crate::room_io::_input::VideoSampler::new(speaking, silent))
            }),
            // **The one resolution**; no default restated beside it.
            config: DriverConfig::new(stack, legs),
            instructions: instructions.unwrap_or_default(),
            dynamic_instructions,
            // Opened at each use, above the synchroniser: the node rewrites, the synchroniser paces.
            output_nodes: OutputNodes {
                transcription: transcription_node,
                realtime_audio: realtime_audio_output_node,
            },
            chat_ctx,
            hook_bound: hook_bound.map(Duration::from_secs_f64),
            capture_run: None,
            max_unrecoverable_errors,
            conn,
            expressive,
            eot: turn_detector.clone(),
            llm,
            realtime,
            tts,
            stt,
            keyterms,
            forward_chat_context,
            vad,
            audio_in: host_audio_input.or_else(|| room_io.audio_input()),
            // Only with the driver's sampler; a custom callable keeps frames host-side.
            video_in: if host_video_sampler {
                None
            } else {
                host_video_input.or_else(|| {
                    room_io
                        .video_input()
                        .map(|v| v as Arc<dyn lk_agents_abi::voice::io::VideoInput>)
                })
            },
            recorder: None,
            audio_out,
            text_out,
            room_events: Some(room.events()),
            // Installed at `open`, once `RoomIo` has been started and its stream taken.
            room_io_events: None,
            room_link: Some(room_link),
            toolset,
        };

        let mut api = SessionApi {
            stage: Arc::new(Mutex::new(Stage::Ready(Box::new(Ready { builder, interruption })))),
            legs: Mutex::new(legs),
            vad: Mutex::new(vad_leg),
            stack: Mutex::new(stack),
            room_io,
            synchronizer,
            spawner,
            anchor: lk_agents_abi::WallAnchor::of(&*clock),
            time,
            dtmf_reply: Mutex::new(None),
            last_filler: std::sync::atomic::AtomicU64::new(0),
            agents,
            host: HostParts::default(),
            resources,
            conn,
            #[cfg(feature = "dst")]
            probe: Mutex::new(None),
        };
        api.host = HostParts {
            events: on_event,
            hooks,
            closed: Arc::default(),
            handoffs: Arc::default(),
            eot: turn_detector,
            has_options: options.is_some(),
            call_room: (!offline).then(|| Arc::clone(&room)),
            remote,
            job,
            usage: Arc::default(),
        };
        api
    }

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    /// Publish the agent's track, open every provider stream, and start the loop. Awaitable, since
    /// the provider factories must be opened on the loop.
    ///
    /// Does not wait for a participant; await `wait_until_ready` for that. `capture_run=True` opens
    /// `start(capture_run=True)`'s run before the opening agent's `AgentHandoff` record and its
    /// `on_enter`.
    #[host(method(kw_only, defaults(capture_run = false)))]
    pub fn start(&self, capture_run: bool) -> Result<Reply<Result<(), StartFailed>>, OpenRefused> {
        let job = self.host.job.clone();
        let events = self.host.events.clone();
        let remote = self.host.remote.clone();
        let forwarding = remote.clone();
        let usage = Arc::clone(&self.host.usage);
        let hooks = self.host.hooks.clone();
        let anchor = self.anchor;
        let agents = self.agents.clone();
        let spawner = Arc::clone(&self.spawner);
        // The drain: one event per call, in driver order; a hook call handed to the host loop in
        // that order, then awaited on its own task.
        let opening = self.open(capture_run, move |mut runner: SessionRunner| {
            Box::pin(async move {
                let mut running: Vec<lk_agents_abi::AbortHandle> = Vec::new();
                while let Some(ev) = runner.next_host_event().await {
                    match ev {
                        HostEvent::Hook(call) => {
                            use crate::agent_session::HookCall;
                            if let (Some(remote), HookCall::FunctionToolsExecuted { event, .. }) =
                                (&forwarding, &call)
                            {
                                remote.forward_tools_executed(event);
                            }
                            let Some(hooks) = &hooks else { continue };
                            let answer = super::answer_hook(
                                Arc::clone(hooks),
                                call,
                                agents.clone(),
                                runner.api().clone(),
                                Arc::clone(&spawner),
                            );
                            let mut answer = Box::pin(answer);
                            let first = std::future::poll_fn(|cx| {
                                std::task::Poll::Ready(answer.as_mut().poll(cx))
                            })
                            .await;
                            if first.is_pending() {
                                running.retain(|task| !task.is_finished());
                                running.push(spawner.spawn(answer));
                            }
                        }
                        ev => {
                            let Some(wire) = ev.wire(anchor) else { continue };
                            // `_on_metrics_collected`: a record is folded into the session's
                            // usage, then `session_usage_updated` follows it.
                            let updated = session_usage_updated(&usage, &wire);
                            for mut wire in std::iter::once(wire).chain(updated) {
                                match (&forwarding, &events, &wire) {
                                    (Some(remote), Some(events), HostWire::Error(error)) => {
                                        remote.forward_error(events.error_message(error));
                                    }
                                    (Some(remote), ..) => remote.forward(&wire),
                                    (None, ..) => {}
                                }
                                if let Some(events) = &events {
                                    agents.resolve(&mut wire);
                                    events.event(&wire);
                                }
                            }
                        }
                    }
                }
                running.iter().for_each(lk_agents_abi::AbortHandle::abort);
                // Dropping the runner tells `aclose()` the host has seen everything.
                drop(runner);
                // However the session closed, its remote session host closes with it.
                if let Some(remote) = forwarding {
                    remote.aclose().await;
                }
            })
        })?;
        // The job's room connects beside the opening; the remote session starts once both are
        // done, so what it queued meanwhile goes out on a connected room.
        Ok(Reply::new(async move {
            let connect = async {
                match &job {
                    Some(job) => job.connect().await,
                    None => Ok(()),
                }
            };
            let (opened, connected) = tokio::join!(opening, connect);
            opened.map_err(StartFailed::Driver)?;
            connected.map_err(StartFailed::Host)?;
            if let Some(remote) = remote {
                remote.start().await.map_err(StartFailed::Host)?;
            }
            Ok(())
        }))
    }

    /// `session.usage.model_usage`: the usage of every model the session was billed for.
    #[host(getter)]
    pub fn usage(&self) -> Vec<lk_agents_abi::metrics::usage::ModelUsage> {
        self.host.usage.lock().unwrap_or_else(|p| p.into_inner()).flatten()
    }

    /// The session's clock, as the handle a provider's `_bind_session_clock` takes.
    #[host(getter)]
    pub fn session_clock(&self) -> lk_agents_abi::utils::clock::SessionClock {
        lk_agents_abi::utils::clock::SessionClock(self.time.clock())
    }

    /// The room's microphone leg, unless `RoomIoConfig(audio_input=False)`, else `None`. A fresh
    /// handle each call over the same leg; safe before or after `start()`.
    #[host(getter)]
    pub fn audio_input(&self) -> Option<Arc<crate::room_io::_input::ParticipantAudioInput>> {
        self.room_io.audio_input_leg()
    }

    /// The room's video leg, when `RoomIoConfig(video_input=True)` asked for one, else `None`. A
    /// fresh handle each call over the same leg; safe before or after `start()`.
    #[host(getter)]
    pub fn video_input(&self) -> Option<Arc<crate::room_io::_input::ParticipantVideoInput>> {
        self.room_io.video_input()
    }

    /// Record the session into `recorder`; `None` detaches (`RecorderIO.start()`/`aclose()`). The
    /// recorder taps the driver's audio path, so no chain node wraps a leg. Safe before or after
    /// `start()`; the file's `finish` stays the host's.
    #[host(method(name = "attach_recorder"))]
    pub fn host_attach_recorder(
        &self,
        #[host(required)] recorder: Option<Arc<dyn lk_agents_media::recorder::AudioRecorder>>,
    ) {
        self.attach_recorder(recorder);
    }

    /// Resolve once `RoomIo` has linked a participant and published the agent's track.
    #[host(method)]
    pub fn wait_until_ready(&self) -> Reply<()> {
        let room_io = Arc::clone(&self.room_io);
        Reply::new(async move { room_io.wait_for_ready().await })
    }

    /// `Agent.update_options(stt=.., vad=.., llm=.., tts=..)` on a running agent: an in-place
    /// provider swap, not a handoff (same agent, no hook, no drain). Each leg is three-state; a
    /// given one goes under the agent's node when it overrides one (`llm_node=` and the others).
    #[host(method(kw_only))]
    #[allow(clippy::too_many_arguments)]
    pub fn update_models(
        &self,
        llm: Maybe<Option<Arc<dyn Llm>>>,
        tts: Maybe<Option<Arc<dyn Tts>>>,
        stt: Maybe<Option<Arc<dyn Stt>>>,
        vad: Maybe<Option<Arc<dyn Vad>>>,
        llm_node: Option<Arc<dyn NodeFactory>>,
        tts_node: Option<Arc<dyn NodeFactory>>,
        stt_node: Option<Arc<dyn NodeFactory>>,
    ) -> Result<(), SessionRefused> {
        let nodes = Nodes { llm: llm_node, tts: tts_node, stt: stt_node };
        let ModelRefs { llm, tts, stt, vad } = self.swap(llm, tts, stt, vad, &nodes);
        self.send(ActivityCommand::UpdateModels { stt, vad, llm, tts })
    }

    /// The handoff: replace the agent, keep the session. Resolves `True` once the agent is
    /// installed (`on_enter` issued, not returned), `False` when it was refused (closing).
    ///
    /// The driver owns the providers, scheduler and buffered frames and outlives the swap, so there
    /// is no `ReusableResources` handover. An omitted provider keeps its stream.
    #[host(method(kw_only))]
    pub fn update_agent(
        &self,
        #[host(kwargs)] frame: UpdateAgentArgs,
    ) -> Result<Reply<bool>, SessionRefused> {
        // Always `_update_activity(previous_activity="close", new_activity="start")`.
        // The answer first: a refused handoff registers no provider it would never release.
        let (call, rx) = self.call(Answer::handoff)?;
        let mut update = self.frame(frame);
        update.call = Some(call);
        let answer = Reply::new(async move { matches!(rx.await, Ok(Ok(()))) });
        self.send_frame(Origin::Host, update, |u| ActivityCommand::Update(Box::new(u)).into())?;
        Ok(answer)
    }

    /// `VoiceTask.run()`: `update_agent`'s keywords for the task's agent, sent from inside a tool
    /// (`origin` is its `"tool:<call_id>"`). The agent holds the floor until one of its tools
    /// completes with a value the host validated, which this resolves to; the parent comes back
    /// without its hooks. Raises when refused or ended without a value. A cancelled await ends
    /// the task, and every task it started, and puts the parent back.
    #[host(method(kw_only = "frame", defaults(opens_with_reply = false)))]
    pub fn run_voice_task(
        &self,
        #[host(kwargs)] frame: UpdateAgentArgs,
        origin: Option<&str>,
        opens_with_reply: bool,
    ) -> Result<
        Reply<Result<lk_agents_abi::HostRef, crate::agent_activity::voice_task::VoiceTaskEnded>>,
        CommandRefused,
    > {
        use crate::agent_activity::voice_task::{VoiceTaskCommand, VoiceTaskEnded, VoiceTaskRun};
        let origin = origin_of(origin)?;
        let (call, rx) = self.call(Answer::voice_task)?;
        let tx = self.api()?;
        // `session.options` keeps the parent's layer: the task's is temporary.
        let vad = self.vad.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (stack, legs) = (self.stack(), self.legs());
        let update = self.frame(frame);
        *self.stack.lock().unwrap_or_else(|p| p.into_inner()) = stack;
        *self.legs.lock().unwrap_or_else(|p| p.into_inner()) = legs;
        *self.vad.lock().unwrap_or_else(|p| p.into_inner()) = vad;
        self.send_frame(origin, update, move |update| {
            let run = VoiceTaskRun { update: Box::new(update), opens_with_reply, call };
            VoiceTaskCommand::Run(Box::new(run)).into()
        })?;
        // Dropped unanswered (the awaiter was cancelled): the task ends.
        struct Awaiting(Option<(crate::agent_session::Tx, crate::agent_session::RequestId)>);
        impl Drop for Awaiting {
            fn drop(&mut self) {
                if let Some((tx, call)) = self.0.take() {
                    let _ = tx.send_from(Origin::Host, VoiceTaskCommand::Cancel { call }.into());
                }
            }
        }
        let awaiting = Awaiting(Some((tx, call)));
        Ok(Reply::new(async move {
            let mut awaiting = awaiting;
            let answer = rx.await;
            awaiting.0 = None;
            answer.unwrap_or(Err(VoiceTaskEnded::Closing))
        }))
    }

    /// `update`'s toolsets set up first, in call order: the turn is taken now. Then the command
    /// `wrap` makes of it is sent as `origin`.
    fn send_frame(
        &self,
        origin: Origin,
        mut update: crate::agent_session::AgentUpdate,
        wrap: impl FnOnce(crate::agent_session::AgentUpdate) -> Command + Send + 'static,
    ) -> Result<(), SessionRefused> {
        let (Some(hooks), Some(tools)) = (self.host.hooks.clone(), update.toolset.host.clone())
        else {
            return self.send_from(origin, wrap(update));
        };
        let api = self.running(|r| Ok(r.api.clone()))?;
        let resources = self.resources.clone();
        // The turn is queued for now, in call order: the lock's waiter list is entered by this
        // first poll (a tokio lock takes the real waker at the next one). Only the lock is polled
        // here; the setup's host bridge runs in the task, with its own waker.
        let mut turn = Box::pin(Arc::clone(&self.host.handoffs).lock_owned());
        let held = turn.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
        let handoff = async move {
            let _turn = match held {
                std::task::Poll::Ready(guard) => guard,
                std::task::Poll::Pending => turn.await,
            };
            update.toolset.specs = hooks.setup_tools(&tools).await.unwrap_or_else(|e| {
                hooks.hook_failed("setup_tools", e.host_ref.clone());
                Vec::new()
            });
            let cmd = wrap(update);
            let handed = cmd.resources();
            if api.send_from(origin, cmd).is_err() {
                handed.into_iter().for_each(|id| resources.release(id));
            }
        };
        self.spawner.spawn(Box::pin(handoff));
        Ok(())
    }

    /// `session.input.audio = ..` after `start()`; `None` detaches. The reference delivers audio
    /// from a source attached after start, so this must reach the driver.
    #[host(method(send = ActivityCommand::SetAudioInput(source.map(HostAudio))))]
    pub fn set_audio_input(
        &self,
        #[host(required)] source: Option<Arc<dyn AudioInput>>,
    ) -> Result<(), SessionRefused>;

    /// `session.input.video = ..` after `start()`: the driver's video pump reads the new source (the
    /// room's own leg is a no-op, a host `io.VideoInput` is wrapped), or stops on `None`.
    #[host(method(send = ActivityCommand::SetVideoInput(source.map(HostVideo))))]
    pub fn set_video_input(
        &self,
        #[host(required)] source: Option<Arc<dyn lk_agents_abi::voice::io::VideoInput>>,
    ) -> Result<(), SessionRefused>;

    /// `session.output.audio = ..` after `start()`: the next speech plays into the new sink, one
    /// in flight is interrupted. `None` takes the sink away. The room's own leg handed back is the
    /// sink the driver already plays into, as the reference's `sink is self._audio_sink` return.
    #[host(method)]
    pub fn set_audio_output(
        &self,
        #[host(required)] sink: Option<Arc<dyn AudioOutput>>,
    ) -> Result<(), SessionRefused> {
        if let Some(sink) = &sink {
            if self.room_io.audio_output().is_some_and(|room| Arc::ptr_eq(&room, sink)) {
                return Ok(());
            }
            sink.tracker().anchor_to(self.anchor);
        }
        let sink = sink.map(|s| self.resources.register_audio_output(s));
        self.send(ActivityCommand::SetAudioOutput(sink))
    }

    /// `session.output.transcription = ..` after `start()`: the next reply forwards into the new
    /// leg; one in flight keeps the leg it started with.
    #[host(method)]
    pub fn set_transcription_output(
        &self,
        #[host(required)] sink: Option<Arc<dyn TextOutput>>,
    ) -> Result<(), SessionRefused> {
        let sink = sink.map(|s| self.resources.register_text_output(s));
        self.send(ActivityCommand::SetTranscriptionOutput(sink))
    }

    /// `Agent.update_tools()` on a running agent. All three parts go together: the runner is
    /// per-agent. The reference's follow-up `update_chat_ctx(chat_ctx.copy(tools=tools))` stays on
    /// the host side.
    #[host(method(kw_only = "tools_host", send = ActivityCommand::UpdateTools(host_toolset(tools, tools_host, tool_runner))))]
    pub fn update_tools(
        &self,
        tools: Option<Vec<lk_agents_abi::ToolSpec>>,
        tools_host: Option<lk_agents_abi::HostRef>,
        tool_runner: Option<Arc<dyn crate::tool_executor::HostToolRunner>>,
    ) -> Result<(), SessionRefused>;

    /// `AgentSession.update_options(..)` on a running session: the raw update, read into a patch
    /// the driver merges into its session layer and re-resolves; `expressive` reaches the next
    /// reply, `keyterms` replace the user's list. An update that says nothing is a no-op. A turn
    /// detector object is refused: swapping the detector of a running session is not ported.
    #[host(method)]
    pub fn update_turn_options(
        &self,
        update: crate::agent_activity::options::UpdateKwargs,
    ) -> Result<(), CommandRefused> {
        if let Maybe::Given(crate::agent_activity::options::TurnDetectionKwarg::Detector { .. }) =
            update.turn_detection
        {
            return Err(CommandRefused::Invalid(
                "update_options(turn_detection=) takes a mode on a running session; a turn \
                 detector object is set with VoiceSession(turn_handling=)"
                    .to_owned(),
            ));
        }
        let patch = crate::agent_activity::options::update_layer(&update);
        if let Maybe::Given(expressive) = update.expressive {
            self.send(ActivityCommand::UpdateExpressive(expressive))?;
        }
        if let Some(terms) = update.keyterms {
            self.send(Command::SetStaticKeyterms(terms))?;
        }
        if patch == crate::agent_activity::options::AgentSessionOptions::default() {
            return Ok(());
        }
        self.configure(&patch);
        Ok(self.send(TurnCommand::Configure(patch))?)
    }

    /// The outbound-SIP re-evaluation (`_on_room_io_participant_linked`): the warmup leaf of the
    /// session layer, re-resolved by the driver.
    #[host(method)]
    pub fn set_aec_warmup(
        &self,
        #[host(required)] duration: Option<f64>,
    ) -> Result<(), SessionRefused> {
        let patch = crate::agent_activity::options::aec_warmup_layer(duration);
        self.configure(&patch);
        self.send(TurnCommand::Configure(patch))
    }

    /// The configuration the driver runs on, as the dict `session.options` views, current the
    /// instant a change is sent. `None` for a session built with no options object.
    #[host(method)]
    pub fn resolved_options(
        &self,
    ) -> Option<crate::agent_activity::options::ResolvedSessionOptions> {
        if !self.host.has_options {
            return None;
        }
        let stack = self.stack();
        Some(crate::agent_activity::options::resolved_session_options(
            &stack.resolve(&self.legs()),
            &stack.session,
        ))
    }

    /// `AgentActivity.push_video(frame)`: one sampled frame, on its way to the model, from
    /// `ChatContent.image_frame(..)` (the frame is already interned; only a `MediaRef` moves).
    /// `False` when the loop did not take it (not started or closed);
    /// a non-image part raises.
    #[host(method(name = "push_video"))]
    pub fn host_push_video(&self, content: lk_agents_abi::ChatContent) -> Result<bool, NotAnImage> {
        let lk_agents_abi::ChatContent::Image(image) = content else { return Err(NotAnImage) };
        Ok(self.push_video(image))
    }

    /// an accepted output from inside `speech`'s tool: the output the run
    /// ending on that speech reads. `error` is the `isinstance(result, Exception)` arm.
    #[host(method(kw_only = "value"))]
    pub fn set_run_output(
        &self,
        speech: &SpeechHandle,
        value: Option<lk_agents_abi::HostRef>,
        error: Option<lk_agents_abi::HostRef>,
    ) -> Result<(), SessionRefused> {
        // A handle the loop has not created yet has no speech: the caller is outside any speech.
        let Some(speech) = speech.created() else { return Ok(()) };
        let output = match (value, error) {
            (_, Some(err)) => crate::run_result::RunOutput::Error(err),
            (Some(v), None) => crate::run_result::RunOutput::Value(v),
            (None, None) => return Ok(()),
        };
        self.send(SpeechCommand::SetRunOutput { speech, output })
    }

    /// `AgentSession.generate_reply()`: reply now, with no end of turn behind it.
    ///
    /// The overrides apply to this reply only. `tool_choice=None` lets the provider decide, unlike
    /// `"none"`; `chat_ctx=` replaces the history for planning; `tools=`, `tools_host=` and
    /// `tool_runner=` go together, and only a passed `tools` pins the reply's toolset.
    #[host(method(kw_only, defaults(on_enter = false)))]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_reply(
        &self,
        user_input: Option<String>,
        instructions: Option<String>,
        allow_interruptions: Option<bool>,
        tool_choice: Option<lk_agents_abi::ToolChoice>,
        chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
        tools: Option<Vec<lk_agents_abi::ToolSpec>>,
        tools_host: Option<lk_agents_abi::HostRef>,
        tool_runner: Option<Arc<dyn crate::tool_executor::HostToolRunner>>,
        // Decided by the host: the reference tests a `ContextVar` plus agent/session identity.
        on_enter: bool,
        origin: Option<&str>,
    ) -> Result<Arc<SpeechHandle>, CommandRefused> {
        let origin = origin_of(origin)?;
        self.refuse_paused(&origin)?;
        // An empty toolset would tell the model about no tool at all; `None` keeps the agent's.
        let toolset =
            tools.map(|specs| Arc::new(host_toolset(Some(specs), tools_host, tool_runner)));
        let (speech, call) = self.speech_call(allow_interruptions)?;
        self.send_from(
            origin,
            SpeechCommand::GenerateReply {
                user_input,
                instructions,
                allow_interruptions,
                tool_choice,
                chat_ctx: chat_ctx.map(lk_agents_abi::ChatContext::new),
                toolset,
                on_enter,
                call: Some(call),
            },
        )?;
        Ok(speech)
    }

    /// `IVRActivity.start()`: the IVR navigator runs in the driver. `max_silence` is
    /// `max_silence_duration` in seconds; `tools` is `IVRActivity.tools`. A second start is
    /// journalled, not run.
    #[host(method(kw_only = "tools_host", defaults(max_silence = 5.0)))]
    pub fn ivr_start(
        &self,
        max_silence: f64,
        tools: Option<Vec<lk_agents_abi::ToolSpec>>,
        tools_host: Option<lk_agents_abi::HostRef>,
    ) -> Result<(), CommandRefused> {
        if !(max_silence.is_finite() && max_silence >= 0.0) {
            return Err(CommandRefused::Invalid(
                "ivr_start: max_silence must be a finite, non-negative number of seconds"
                    .to_owned(),
            ));
        }
        // `_start_ivr_detection`: the session's tools extended, then the navigator started.
        if let Some(tools) = session_tools(tools, tools_host) {
            self.send(Command::Ivr(IvrCommand::AddTools(tools)))?;
        }
        Ok(self.send(Command::Ivr(IvrCommand::Start {
            max_silence: Duration::from_secs_f64(max_silence),
        }))?)
    }

    /// `AMD._run`: the answering-machine detector, started as a peer of this session over its
    /// config, classifier LLM, tools and optional STT. It listens once the callee answers in the
    /// session's room (only `participant_identity`'s audio, when given); the verdict comes back as
    /// `amd_result`. A second attach is refused.
    #[host(method(name = "attach_amd"))]
    pub fn host_attach_amd(
        &self,
        cfg: lk_agents_abi::voice::amd::AmdConfigKwargs,
        llm: Arc<dyn Llm>,
        tools: Option<lk_agents_abi::HostRef>,
        stt: Option<Arc<dyn Stt>>,
        participant_identity: Option<String>,
    ) -> Result<(), CommandRefused> {
        let config = crate::amd::classifier::AmdConfig::resolve(&cfg)
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let setup = crate::amd::detector::AmdSetup {
            config,
            prompt: cfg.prompt.unwrap_or_else(|| lk_agents_abi::voice::amd::AMD_PROMPT.to_owned()),
            llm: llm_leg(llm, &self.conn.llm),
            tools_host: tools,
            stt: stt.map(|s| stt_leg(s, None, &self.conn.stt, self.time.clock())),
            room: self.host.call_room.clone(),
            participant: participant_identity,
        };
        Ok(self.attach_amd(setup)?)
    }

    /// `_aclose_impl`: the hooks' `closing` (AMD), `Command::Close { reason, drain, error }`, the
    /// turn-detector stream, the remote session host, the room io, then the session's toolsets.
    /// Resolves once all of it ran; the loop's `close` handlers ran before the room io (the
    /// reference's "close room io after close event is emitted"). An unknown `reason` is a
    /// `ValueError`. Idempotent: a second call waits for the first.
    #[host(method(kw_only, defaults(reason = "user_initiated", drain = false)))]
    pub fn aclose(&self, reason: &str, drain: bool) -> Result<Reply<()>, UnknownCloseReason> {
        let reason = lk_agents_abi::voice::events::CloseReason::from_wire(reason)
            .ok_or_else(|| UnknownCloseReason(reason.to_owned()))?;
        // A session never started has nothing to wait for, and its turn-detector stream stays the
        // host's to close.
        let was_running = self.api().is_ok();
        let closing = self.close(reason, drain, None);
        let close_io = self.close_io();
        let clock = self.time.clock();
        let eot = self.host.eot.clone().filter(|_| was_running);
        let hooks = self.host.hooks.clone().filter(|_| was_running);
        let remote = self.host.remote.clone();
        let closed = Arc::clone(&self.host.closed);
        Ok(Reply::new(async move {
            closed
                .get_or_init(|| async move {
                    if let Some(hooks) = &hooks
                        && let Err(e) = hooks.closing().await
                    {
                        hooks.hook_failed("closing", e.host_ref.clone());
                    }
                    closing.await;
                    if let Some(eot) = eot {
                        let _ = lk_agents_abi::timeout(&*clock, CLOSE_TIMEOUT, eot.aclose()).await;
                    }
                    if let Some(remote) = remote {
                        remote.aclose().await;
                    }
                    close_io.await;
                    if let Some(hooks) = &hooks {
                        hooks.close_toolsets(None).await;
                    }
                })
                .await;
        }))
    }

    /// `AgentSession.shutdown(drain=)`: [`aclose`](Self::aclose) without waiting for it.
    #[host(method(kw_only, defaults(drain = true)))]
    pub fn shutdown(&self, drain: bool) {
        if let Ok(closing) = self.aclose("user_initiated", drain) {
            self.spawner.spawn_unboxed(closing);
        }
    }

    /// The identity `RoomIo` linked to, once it has.
    #[host(getter)]
    pub fn linked_participant(&self) -> Option<String> {
        self.room_io.linked_participant().map(|p| p.0)
    }

    /// `RoomIO.set_participant(identity)`: point the audio input and the user transcript at
    /// someone else, mid-session. `None` unlinks, like `unset_participant()`. The old link is
    /// forgotten first, then the new participant is looked up in the room.
    #[host(method)]
    pub fn set_participant(&self, #[host(required)] identity: Option<String>) -> Reply<()> {
        let room_io = Arc::clone(&self.room_io);
        Reply::new(async move { room_io.set_participant(identity.map(Into::into)).await })
    }

    /// `RoomIO.unset_participant()`. Idempotent.
    #[host(method)]
    pub fn unset_participant(&self) -> Reply<()> {
        let room_io = Arc::clone(&self.room_io);
        Reply::new(async move { room_io.unset_participant().await })
    }

    /// `RunContext.with_filler(..)` entered for tool `call_id` of `speech_id`: after `delay` s idle
    /// say `text` (or ask `source`, the host's callable), then repeat every `interval`. Returns the
    /// scope's number, which `filler_stop` takes.
    #[host(method(kw_only = "delay"))]
    #[allow(clippy::too_many_arguments)]
    pub fn filler_start(
        &self,
        call_id: String,
        speech_id: &str,
        delay: f64,
        interval: Option<f64>,
        max_steps: Option<u32>,
        text: Option<String>,
        source: Option<lk_agents_abi::HostRef>,
    ) -> Result<u64, CommandRefused> {
        use crate::filler_scheduler::FillerError;
        let invalid = |e: FillerError| CommandRefused::Invalid(e.message().to_owned());
        if delay < 0.0 || delay.is_nan() {
            return Err(invalid(FillerError::NegativeDelay));
        }
        if interval.is_some_and(|i| i < 0.0 || i.is_nan()) {
            return Err(invalid(FillerError::NegativeInterval));
        }
        let speech = speech_id
            .parse::<crate::agent_activity::scheduling::SpeechId>()
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let source = match text {
            Some(text) => FillerSource::Text(text),
            None => FillerSource::Host(source),
        };
        let id = self.last_filler.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let origin = Origin::Tool { call_id: call_id.clone() };
        let cmd = SpeechCommand::FillerStart {
            id: FillerId(id),
            call_id,
            speech,
            delay: Duration::from_secs_f64(delay),
            interval: interval.map(Duration::from_secs_f64),
            max_steps,
            source,
        };
        self.send_from(origin, cmd)?;
        Ok(id)
    }

    /// `with_filler`'s exit.
    #[host(method(send = SpeechCommand::FillerStop { id: FillerId(id) }))]
    pub fn filler_stop(&self, id: u64) -> Result<(), SessionRefused>;

    /// `AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it,
    /// including a realtime session's server-side buffer.
    #[host(method(send = TurnCommand::Clear))]
    pub fn clear_user_turn(&self) -> Result<(), SessionRefused>;

    /// `Agent.update_instructions()` on the agent that is already driving. Changes only the
    /// prompt; `update_agent` would replace the whole agent.
    #[host(method(send = ActivityCommand::UpdateInstructions(instructions)))]
    pub fn update_instructions(&self, instructions: Instructions) -> Result<(), SessionRefused>;

    /// `input.set_audio_enabled` / `output.set_audio_enabled` / `output.set_transcription_enabled`.
    /// `leg` is `"audio_input"`, `"audio_output"` or `"transcription"`.
    #[host(method)]
    pub fn set_io_enabled(&self, leg: &str, enabled: bool) -> Result<(), CommandRefused> {
        let leg = match leg {
            "audio_input" => IoKind::AudioInput,
            "audio_output" => IoKind::AudioOutput,
            "transcription" => IoKind::Transcription,
            other => return Err(CommandRefused::Invalid(format!("unknown io leg {other:?}"))),
        };
        Ok(self.send(ActivityCommand::SetIoEnabled { leg, enabled })?)
    }

    /// `RunContext.update()` after the first (an `update` outcome), or a detached tool's end
    /// (any other).
    #[host(method)]
    pub fn tool_update(
        &self,
        call_id: String,
        report: crate::tool_executor::ToolReport,
    ) -> Result<(), CommandRefused> {
        let update = ToolUpdate::from(report);
        let origin = Origin::Tool { call_id: call_id.clone() };
        Ok(self.send_from(origin, SpeechCommand::ToolUpdate { call_id, update })?)
    }

    /// `Agent.update_chat_ctx()` on a running agent: replace the context.
    /// `exclude_invalid_function_calls` is applied host-side before the items arrive.
    #[host(method(send = ActivityCommand::UpdateChatCtx(items)))]
    pub fn update_chat_ctx(
        &self,
        items: Vec<lk_agents_abi::ChatItem>,
    ) -> Result<(), SessionRefused>;

    /// `AgentSession.interrupt(force=)`: interrupt the current and queued speeches up to the first
    /// uninterruptible; answers when they are done, or with the refusal.
    #[host(method(kw_only, defaults(force = false)))]
    pub fn interrupt(
        &self,
        force: bool,
    ) -> Result<Reply<Result<(), NotInterruptible>>, SessionRefused> {
        let (call, rx) = self.call(Answer::interrupt)?;
        self.send(SpeechCommand::Interrupt { force, call: Some(call) })?;
        Ok(Reply::new(async move {
            match rx.await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(_)) => Err(NotInterruptible),
                // The loop ended before the walk: nothing is speaking any more.
                Err(_) => Ok(()),
            }
        }))
    }

    /// `AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
    /// speech is left; the session stays open. Answers `None`, `"activity_changing"` or `"closing"`.
    #[host(method)]
    pub fn drain(&self) -> Result<Reply<Option<String>>, SessionRefused> {
        let (call, rx) = self.call(Answer::drain)?;
        self.send(ActivityCommand::Drain { call: Some(call) })?;
        Ok(Reply::new(async move {
            match rx.await {
                Ok(Ok(())) => None,
                Ok(Err(refused)) => Some(refused.as_str().to_owned()),
                Err(_) => Some(DrainRefused::Closing.as_str().to_owned()),
            }
        }))
    }

    /// DTMF collection start's subscriptions, run by the driver. Answers once, with the outcome
    /// (`kind`: `complete`, `incomplete` or `stopped`).
    #[host(method(
        kw_only,
        defaults(ask_for_confirmation = false, dtmf_input_timeout = 4.0, dtmf_stop_event = "#".to_owned())
    ))]
    pub fn dtmf_start(
        &self,
        num_digits: usize,
        ask_for_confirmation: bool,
        dtmf_input_timeout: f64,
        dtmf_stop_event: String,
    ) -> Result<Reply<lk_agents_abi::voice::workflows::DtmfOutcomeWire>, CommandRefused> {
        use crate::workflows::dtmf::{DtmfCollector, DtmfEvent, DtmfOutcome};
        let stop = DtmfEvent::parse(&dtmf_stop_event).ok_or_else(|| {
            CommandRefused::Invalid(format!("{dtmf_stop_event:?} is not a DTMF key"))
        })?;
        let timeout = Duration::try_from_secs_f64(dtmf_input_timeout)
            .map_err(|e| CommandRefused::Invalid(format!("dtmf_input_timeout: {e}")))?;
        let collector = DtmfCollector::new(num_digits, ask_for_confirmation, timeout, stop)
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let guard = collector.reply_flag();
        let (call, rx) = self.call(Answer::dtmf)?;
        self.send(DtmfCommand::Start { collector, call: Some(call) })?;
        *self.dtmf_reply.lock().unwrap_or_else(|p| p.into_inner()) = Some(guard);
        Ok(Reply::new(async move { rx.await.unwrap_or(DtmfOutcome::Stopped).wire() }))
    }

    /// DTMF read-back activity: whether the last collection is inside its debounced
    /// reply (for the confirm arm, through the read-back speech).
    #[host(method(pid = false))]
    pub fn dtmf_reply_running(&self) -> bool {
        self.dtmf_reply
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .is_some_and(|guard| guard.load(std::sync::atomic::Ordering::SeqCst))
    }

    /// `EndCallTool`: close the session once `speech`, the tool's, is done (with what is left to
    /// say), or once a realtime model's own reply to the tool result is.
    #[host(method)]
    pub fn end_call(&self, speech: &SpeechHandle) -> Result<(), SessionRefused> {
        let Some(speech) = speech.created() else { return Ok(()) };
        self.send(SpeechCommand::EndCall { speech })
    }

    /// `sip_dtmf_received`: the raw `ev.digit`, for the running collection.
    #[host(method(send = DtmfCommand::Digit { digit }))]
    pub fn dtmf_digit(&self, digit: String) -> Result<(), SessionRefused>;

    /// DTMF collection stop: the collection ends, an outcome still owed resolves `stopped`. A
    /// closed loop has stopped the collection already.
    #[host(method(send = DtmfCommand::Stop))]
    pub fn dtmf_stop(&self);

    /// `AgentSession.wait_for_idle()`: answers when the session is idle (no hold but `hold`).
    /// Refused with [`IdleRefused::Closing`] if it closes first.
    #[host(method(kw_only))]
    pub fn wait_for_idle(
        &self,
        hold: Option<u64>,
    ) -> Result<Reply<Result<(), IdleRefused>>, SessionRefused> {
        let (call, rx) = self.call(Answer::idle)?;
        self.send(SpeechCommand::WaitForIdle { hold: hold.map(HoldId), call: Some(call) })?;
        Ok(Reply::new(async move { rx.await.unwrap_or(Err(IdleRefused::Closing)) }))
    }

    /// `RunContext.foreground()`: wait for idle, then hold the floor until
    /// [`release_idle_hold`](Self::release_idle_hold). Answers the hold's id.
    #[host(method(kw_only))]
    pub fn wait_for_idle_and_hold(
        &self,
        origin: Option<&str>,
    ) -> Result<Reply<Result<u64, IdleRefused>>, CommandRefused> {
        let (call, rx) = self.call(Answer::hold)?;
        self.send_from(origin_of(origin)?, SpeechCommand::WaitForIdleAndHold { call: Some(call) })?;
        Ok(Reply::new(async move { rx.await.unwrap_or(Err(IdleRefused::Closing)).map(|h| h.0) }))
    }

    /// `_wait_for_idle_and_hold`'s exit. A hold the loop no longer has is a no-op, and so is a
    /// closed loop: it has released everything already.
    #[host(method(send = SpeechCommand::ReleaseIdleHold { hold: HoldId(hold) }))]
    pub fn release_idle_hold(&self, hold: u64);

    /// `AgentSession._claim_user_turn()`'s enter: pins the user state to `speaking` and holds
    /// `wait_for_idle` open. Reentrant.
    #[host(method(send = SpeechCommand::ClaimUserTurn))]
    pub fn claim_user_turn(&self) -> Result<(), SessionRefused>;

    /// `_claim_user_turn()`'s exit: the last release re-derives the user state from the audio
    /// path. Releasing on a closed loop has nothing left to release.
    #[host(method(send = SpeechCommand::ReleaseUserTurn))]
    pub fn release_user_turn(&self);

    /// `AgentSession.run(user_input=, output_type=, output_options=)`, answered on the stream:
    /// every run event carries `host`, the host's `RunResult`.
    #[host(method(kw_only))]
    pub fn run(
        &self,
        user_input: Option<String>,
        expects_output: bool,
        max_retries: u32,
        retry_instructions: String,
        origin: Option<&str>,
        host: Option<lk_agents_abi::HostRef>,
    ) -> Result<(), CommandRefused> {
        let spec = crate::run_result::RunSpec {
            user_input,
            expects_output,
            max_retries,
            retry_instructions,
            host,
        };
        Ok(self.send_from(origin_of(origin)?, Command::Run(Box::new(spec)))?)
    }

    /// `AgentSession.say(text)`: speak these words, no LLM. Synchronous; the handle is the promise
    /// the loop fulfils.
    #[host(method(kw_only = "allow_interruptions", defaults(add_to_chat_ctx = true)))]
    pub fn say(
        &self,
        text: String,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<Arc<SpeechHandle>, CommandRefused> {
        self.speak(SayText::Text(text), allow_interruptions, add_to_chat_ctx, origin)
    }

    /// `AgentSession.say(AsyncIterable[str])`: the speech reads `text` to its end, one segment per
    /// chunk.
    #[host(method(kw_only = "allow_interruptions", defaults(add_to_chat_ctx = true)))]
    pub fn say_stream(
        &self,
        text: Box<dyn lk_agents_abi::StreamSource<String>>,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<Arc<SpeechHandle>, CommandRefused> {
        let (speech, pusher) = self.say_text_channel(allow_interruptions, add_to_chat_ctx, origin)?;
        self.spawner.spawn_unboxed(pusher.feed(text, self.time.clock()));
        Ok(speech)
    }

    /// `AgentSession.say(text, audio=..)`: pre-synthesised frames, read from `audio` to its end,
    /// instead of TTS; `text` is still committed.
    #[host(method(kw_only = "allow_interruptions", defaults(add_to_chat_ctx = true)))]
    pub fn say_audio(
        &self,
        text: String,
        audio: Box<dyn lk_agents_abi::StreamSource<lk_agents_abi::AudioFrame>>,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<Arc<SpeechHandle>, CommandRefused> {
        let (speech, pusher) =
            self.say_audio_channel(text, allow_interruptions, add_to_chat_ctx, origin)?;
        self.spawner.spawn_unboxed(pusher.feed(audio, self.time.clock()));
        Ok(speech)
    }

    /// A metrics record only a plugin can measure, emitted as `metrics_collected`. Engine-computed
    /// kinds are refused (they would count twice).
    #[host(method)]
    pub fn host_metrics(
        &self,
        record: lk_agents_abi::metrics::base::AgentMetrics,
    ) -> Result<(), CommandRefused> {
        use lk_agents_abi::metrics::base::AgentMetrics as M;
        match &record {
            M::Llm(_) | M::Tts(_) | M::Stt(_) | M::Eou(_) => {
                return Err(CommandRefused::Invalid(format!(
                    "host_metrics: {} records are the engine's own; handing one in would count it twice",
                    record.type_literal()
                )));
            }
            M::Vad(_)
            | M::EotInference(_)
            | M::RealtimeModel(_)
            | M::Interruption(_)
            | M::Avatar(_) => {}
        }
        Ok(self.send(Command::HostMetrics(Box::new(record)))?)
    }

    /// `_start_ivr_detection(transcript)`: a final the navigator counts as if heard: AMD's
    /// greeting transcript, so the first menu repeat is the second chunk, not the first.
    #[host(method(send = IvrCommand::SeedTranscript { text: text.to_owned() }))]
    pub fn ivr_final(&self, text: &str) -> Result<(), SessionRefused>;

    /// `IVRActivity.aclose()`: the countdown cancelled, the transcripts no longer counted; no
    /// nudge after this. A no-op when no navigator runs.
    #[host(method(send = IvrCommand::Close))]
    pub fn ivr_close(&self) -> Result<(), SessionRefused>;

    /// `AMD.started`: the detector listens (the callee answered, no verdict out yet).
    #[host(method)]
    pub fn amd_listening(&self) -> bool {
        self.running(|r| Ok(r.amd.as_ref().is_some_and(|amd| amd.listening()))).unwrap_or(false)
    }

    /// `AMD.execute()`'s interrupt and `_resume_authorization()`, after its IVR hand-off: a
    /// `machine-ivr` verdict holds the session until then.
    #[host(method)]
    pub fn amd_execute(&self) -> Result<(), SessionRefused> {
        self.amd_input(crate::amd::classifier::In::Execute)
    }

    /// `AMD.aclose()`: the machine closed. Idempotent: a system already gone is a closed AMD.
    #[host(method)]
    pub fn amd_close(&self) -> Result<(), SessionRefused> {
        self.running(|r| {
            if let Some(handle) = r.amd.take() {
                handle.close();
            }
            Ok(())
        })
    }

    /// `session.keyterms`: the list the recogniser was last handed (static + confirmed); `None`
    /// before any push.
    #[host(method)]
    pub fn keyterms(&self) -> Option<Vec<String>> {
        match &*self.lock() {
            Stage::Running(r) => r.keyterms.borrow().clone(),
            _ => None,
        }
    }

    /// `Agent.chat_ctx` of the driving agent: its own view of the conversation, not the session's
    /// history of every agent. Empty before `start()`.
    #[host(method)]
    pub fn agent_chat_ctx(&self) -> Vec<lk_agents_abi::ChatItem> {
        match &*self.lock() {
            Stage::Running(r) => r.activity.borrow().chat_ctx.items.clone(),
            _ => Vec::new(),
        }
    }

    /// `KeytermDetector.set_static_keyterms` / `update_options(keyterms=)`: the user's list,
    /// replaced; the detected set stays and the effective list is pushed to the STT at once.
    #[host(method(send = Command::SetStaticKeyterms(terms)))]
    pub fn set_static_keyterms(&self, terms: Vec<String>) -> Result<(), SessionRefused>;

    /// `AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
    /// transcript. Runs whether or not awaited; a newer commit supersedes it.
    #[host(method(
        kw_only,
        defaults(transcript_timeout = 2.0, stt_flush_duration = 2.0, skip_reply = false)
    ))]
    pub fn commit_user_turn(
        &self,
        transcript_timeout: f64,
        stt_flush_duration: f64,
        skip_reply: bool,
    ) -> Result<Reply<Result<String, CommitSuperseded>>, SessionRefused> {
        let (call, rx) = self.call(Answer::commit)?;
        self.send(SpeechCommand::CommitUserTurn {
            transcript_timeout: Duration::from_secs_f64(transcript_timeout.max(0.0)),
            stt_flush_duration: Duration::from_secs_f64(stt_flush_duration.max(0.0)),
            skip_reply,
            call: Some(call),
        })?;
        Ok(Reply::new(async move { rx.await.unwrap_or(Err(CommitSuperseded)) }))
    }
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// The `session_usage_updated` a `metrics_collected` record brings, once folded into `usage`. The
/// end-of-utterance and avatar records are reported beside the providers' (no usage follows them).
fn session_usage_updated(
    usage: &Mutex<lk_agents_abi::metrics::usage::ModelUsageCollector>,
    wire: &HostWire,
) -> Option<HostWire> {
    use lk_agents_abi::metrics::base::AgentMetrics;
    let HostWire::MetricsCollected(m) = wire else { return None };
    if matches!(m.metrics, AgentMetrics::Eou(_) | AgentMetrics::Avatar(_)) {
        return None;
    }
    let mut usage = usage.lock().unwrap_or_else(|p| p.into_inner());
    usage.collect(&m.metrics);
    let model_usage = usage.flatten();
    Some(HostWire::SessionUsageUpdated(lk_agents_abi::voice::events::SessionUsageWire { model_usage }))
}
