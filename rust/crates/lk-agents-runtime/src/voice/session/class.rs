//! The declared `_core.VoiceSession` over [`VoiceSession`]: assembly from the host's arguments,
//! and every other method one command to the loop.

use super::*;
use crate::telemetry;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, TextOutput, VideoInput};
use lk_agents_abi::voice::nodes::NodeFactory;
use lk_agents_abi::voice::session::{
    CommitSuperseded, IdleRefused, NotAnImage, NotInterruptible, UnknownCloseReason,
    VoiceSessionArgs, VoiceTaskEnded,
};
use lk_agents_abi::voice::session_options::{ResolvedSessionOptions, UpdateKwargs};
use lk_agents_abi::{ChatItem, ToolSpec};
use lk_agents_voice::agent_activity::DriverConfig;
use lk_agents_voice::agent_activity::options::{
    AgentSessionOptions, DetectorKind, OverrideLayer, running_layer,
};
use lk_agents_voice::agent_session::{
    ActivityCommand, AudioInputHandle, DrainRefused, FillerSource, HoldId, IoKind, IvrCommand,
    TurnCommand,
};
use lk_agents_voice::filler_scheduler::FillerId;

type Speech = Arc<SpeechHandle>;

impl lk_agents_abi::voice::session::VoiceSession for VoiceSession {
    type Env = lk_agents_net::env::ProcessEnv;
    type Options = SessionOptions;
    type Speech = SpeechHandle;
    type MicLeg = crate::room_io::RoomAudioInput;
    type CameraLeg = crate::room_io::RoomVideoInput;

    /// Assemble a session; nothing starts here: synchronous, so a host can bind the room's legs
    /// and recorder before starting.
    fn new(
        env: &Self::Env,
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        args: VoiceSessionArgs,
        options: Option<SessionOptions>,
    ) -> Result<Self, lk_agents_abi::voice::rtc::RtcError> {
        let spawner = (env.spawner)().map_err(|e| {
            lk_agents_abi::voice::rtc::RtcError::Stream(
                lk_agents_abi::errors::ApiError::new(e).into(),
            )
        })?;
        let agents = HostAgents::default();
        agents.install(
            &args.agent,
            args.voice_agent.as_ref().map_or(0, |agent| agent.toolset_scope()),
            args.agent_host,
        );
        let time = match room.virtual_clock() {
            Some(clock) => Timebase::from_virtual(clock),
            None => Timebase::of_clock(room.clock()),
        };
        let clock = time.clock();
        let anchor = lk_agents_abi::WallAnchor::of(&*clock);
        let mut config = args.room_io_config;
        let host_audio_input = args
            .host_audio_input
            .or_else(|| args.console.as_ref().map(|c| c.clone().as_audio_input()));
        let host_audio_output = args
            .host_audio_output
            .or_else(|| args.console.as_ref().map(|c| c.clone().as_audio_output()));
        let rtc_room = room.rtc();
        // An audio leg the host set replaces the room's, which is then neither published nor
        // subscribed. The room's text output stays: it paces a host transcription leg.
        if host_audio_input.is_some() {
            config.audio_input = None;
        }
        if host_audio_output.is_some() || rtc_room.is_none() {
            config.audio_output = None;
        }
        // A host sink's playback reports are stamped on the session's clock.
        if let Some(out) = &host_audio_output {
            out.tracker().anchor_to(anchor);
        }
        let conn = options.as_ref().map(|o| o.conn()).unwrap_or_default();
        let keyterms =
            args.stt_context.as_ref().map(|cfg| lk_agents_voice::keyterm_detection::KeytermSetup {
                detection: lk_agents_voice::keyterm_detection::resolve_detection(
                    cfg.keyterm_detection.as_ref(),
                ),
                static_keyterms: cfg.keyterms.clone().unwrap_or_default(),
                llm: args.keyterm_llm.map(|l| llm_leg(l, conn.llm)),
                tools_host: args.keyterm_tools,
            });
        let remote = args
            .remote_session
            .map(|port| {
                if args.console.is_none() && rtc_room.is_none() {
                    return Err(lk_agents_abi::voice::rtc::RtcError::Closed("remote RTC session"));
                }
                Ok(RemoteSessionConfig {
                    port,
                    room: rtc_room.clone(),
                    console: args.console.clone(),
                })
            })
            .transpose()?;

        // Read before the config moves into `RoomIo`.
        let sync_speed = config.text_output.as_ref().map(|t| t.transcription_speed_factor);
        let room_link = lk_agents_voice::agent_session::room::RoomLinkConfig {
            close_on_disconnect: config.close_on_disconnect,
            delete_room_on_close: config.delete_room_on_close,
            room: rtc_room.as_ref().map(|room| room.name()).transpose()?.unwrap_or_default(),
        };
        let room_io = room.room_io(spawner.clone(), config);

        let outputs = crate::voice::io::SessionOutputs::new(
            room_io.as_ref(),
            host_audio_output,
            args.host_transcription_output,
            sync_speed.unwrap_or(1.0),
            time.clone(),
            spawner.clone(),
        );

        let interruption = args.interruption_detector.map(|source| {
            Box::new(move |anchor, cx: &telemetry::Cx| source.open(anchor, cx))
                as InterruptionPortFactory
        });
        let nodes = Nodes { llm: args.llm_node, tts: None, stt: None };
        let llm = args.llm.map(|l| nodes.llm(llm_leg(l, conn.llm)));
        // Facts, not options: the legs the two leg-dependent defaults need.
        let legs = SessionLegs {
            stt: args.stt.is_some(),
            vad: args.vad.is_some(),
            using_default_vad: args.using_default_vad,
            llm: llm.is_some(),
            tts: args.tts.is_some(),
            realtime: args.realtime.is_some(),
            detector: args.turn_detector.as_ref().map(|_| DetectorKind::Streaming),
            adaptive_detector: interruption.is_some(),
            audio_enabled: true,
            audio_output: true,
            sink_can_pause: true,
        };
        // Past the last step that can fail: a failed build leaves the layers for a retried start.
        let held = options.as_ref().map(SessionOptions::take_layers).unwrap_or_default();
        let stack = ConfigStack {
            session: held.session,
            overrides: held.overrides,
            agent: Some(lk_agents_voice::turn::agent_layer(
                args.agent_turn_handling.as_ref(),
                args.agent_min_consecutive_speech_delay,
                args.agent_use_tts_aligned_transcript,
            )),
        };
        let mut builder = DriverBuilder::new(
            args.agent,
            time.clone(),
            spawner.clone(),
            DriverConfig::new(stack, legs),
        );
        let view = builder.view.subscribe();
        let resources = builder.resources.clone();
        let tool_bodies = builder.tool_bodies.clone();
        let speeches = Arc::clone(&builder.speeches);
        builder.session_id = args.session_id;
        builder.tracer = env.tracer.clone();
        builder.video_sampler = args.video_sampler.map(|(speaking, silent)| {
            lk_agents_voice::agent_activity::video_sampler::VideoSampler::new(speaking, silent)
        });
        builder.instructions = args.instructions.map(Instructions::plain).unwrap_or_default();
        builder.dynamic_instructions = args.dynamic_instructions;
        // Opened at each use, above the synchroniser: the node rewrites, the synchroniser paces.
        builder.output_nodes = OutputNodes { transcription: args.transcription_node };
        builder.chat_ctx = args.chat_ctx;
        builder.storage = args.storage;
        builder.hook_bound = args.hook_bound.map(Duration::from_secs_f64);
        builder.max_unrecoverable_errors = args.max_unrecoverable_errors;
        builder.conn = conn;
        builder.expressive = lk_agents_voice::generation::ExpressiveLayers {
            session: options.as_ref().map(|o| o.expressive()).unwrap_or_default(),
            agent: args.agent_expressive.given(),
        };
        builder.eot = args.turn_detector.clone();
        builder.llm = llm;
        builder.realtime = args.realtime;
        builder.tts = args.tts;
        builder.stt = args.stt;
        builder.forward_chat_context =
            args.stt_context.as_ref().and_then(|cfg| cfg.forward_chat_context).unwrap_or(true);
        builder.keyterms = keyterms;
        builder.vad = args.vad;
        builder.audio_in = host_audio_input.or_else(|| room_io.audio_input());
        // Only with the driver's sampler; a custom callable keeps frames host-side.
        builder.video_in = if args.host_video_sampler {
            None
        } else {
            args.host_video_input
                .or_else(|| room_io.video_input().map(|v| v as Arc<dyn VideoInput>))
        };
        builder.audio_out = outputs.audio.clone();
        builder.text_out = outputs.text.clone();
        builder.room_link = Some(room_link);
        builder.toolset =
            Toolset::set_up(args.tools.unwrap_or_default(), args.tools_host, args.voice_agent);

        let report = builder.report.clone();
        Ok(VoiceSession {
            report,
            tool_bodies,
            stage: Arc::new(Mutex::new(Stage::Ready(Box::new(Ready {
                builder,
                nodes: Nodes { llm: None, tts: args.tts_node, stt: args.stt_node },
                interruption,
                #[cfg(feature = "dst")]
                probe: None,
            })))),
            view,
            speeches,
            room_io,
            outputs,
            tasks: Arc::default(),
            spawner,
            anchor,
            time,
            last_filler: AtomicU64::new(0),
            agents,
            host: HostParts {
                events: args.on_event,
                hooks: args.hooks,
                closed: Arc::default(),
                handoffs: Arc::default(),
                eot: args.turn_detector,
                options: options.clone(),
                call_room: rtc_room,
                remote,
                job: args.job,
                job_run: args.job_run,
            },
            resources,
            conn,
            delivered: Default::default(),
        })
    }

    fn start(&self) -> Result<BoxFuture<'static, Result<(), StartFailed>>, OpenRefused> {
        self.start_with(Box::new(crate::voice::Executor::new(self.time.clone(), self.spawner())))
    }

    fn usage(&self) -> lk_agents_abi::usage::Usage {
        self.view.borrow().usage.clone()
    }

    fn current_agent(&self) -> Option<HostRef> {
        let view = self.view.borrow();
        self.agents.get(&view.agent, view.agent_scope)
    }

    fn audio_input(&self) -> Option<crate::room_io::RoomAudioInput> {
        self.room_io.audio_input_leg().map(crate::room_io::RoomAudioInput)
    }

    fn video_input(&self) -> Option<crate::room_io::RoomVideoInput> {
        self.room_io.video_input().map(crate::room_io::RoomVideoInput)
    }

    fn attach_recorder(&self, recorder: Option<Arc<dyn lk_agents_abi::recorder::AudioRecorder>>) {
        VoiceSession::attach_recorder(self, recorder);
    }

    fn update_models(
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
        self.send(ActivityCommand::UpdateModels(Box::new(
            lk_agents_voice::agent_session::ModelUpdate { stt, vad, llm, tts },
        )))
    }

    fn update_agent(
        &self,
        frame: UpdateAgentArgs,
    ) -> Result<BoxFuture<'static, bool>, SessionRefused> {
        // Always `_update_activity(previous_activity="close", new_activity="start")`. The answer
        // first: a refused handoff registers no provider it would never release.
        let (call, rx) = self.call(Answer::handoff)?;
        let mut update = self.frame(frame);
        update.call = Some(call.id);
        self.send_frame(
            VoiceCommandOrigin::Host,
            update,
            call,
            |u| ActivityCommand::Update(Box::new(u)).into(),
            |_| Answer::Handoff(Err(lk_agents_voice::agent_session::HandoffRefused::SetupFailed)),
        )?;
        Ok(Box::pin(async move { matches!(rx.await, Ok(Ok(()))) }))
    }

    fn run_voice_task(
        &self,
        frame: UpdateAgentArgs,
        origin: Option<VoiceCommandOrigin>,
        opens_with_reply: bool,
        dtmf: Option<lk_agents_abi::voice::session::DtmfTaskOptions>,
    ) -> Result<
        BoxFuture<'static, Result<lk_agents_abi::voice::session::VoiceTaskValue, VoiceTaskEnded>>,
        CommandRefused,
    > {
        use lk_agents_voice::agent_activity::voice_task::{VoiceTaskCommand, VoiceTaskRun};
        let dtmf = dtmf
            .map(|options| {
                use lk_agents_voice::workflows::dtmf::{DtmfCollector, DtmfEvent};
                let stop = DtmfEvent::parse(&options.stop_event).ok_or_else(|| {
                    CommandRefused::Invalid(format!("{:?} is not a DTMF key", options.stop_event))
                })?;
                let timeout = Duration::try_from_secs_f64(options.timeout)
                    .map_err(|error| CommandRefused::Invalid(format!("timeout: {error}")))?;
                DtmfCollector::new(options.num_digits, options.ask_for_confirmation, timeout, stop)
                    .map_err(|error| CommandRefused::Invalid(error.to_string()))
            })
            .transpose()?;
        let origin = origin_of(origin);
        let (awaited, rx) = self.call(Answer::voice_task)?;
        let call = awaited.id;
        let tx = self.api()?;
        let update = self.frame(frame);
        self.send_frame(
            origin,
            update,
            awaited,
            move |update| {
                let run = VoiceTaskRun { update: Box::new(update), opens_with_reply, call, dtmf };
                VoiceTaskCommand::Run(Box::new(run)).into()
            },
            |error| Answer::VoiceTask(Err(VoiceTaskEnded::Setup(error))),
        )?;
        // Dropped unanswered (the awaiter was cancelled): the task ends.
        struct Awaiting(Option<(Tx, RequestId)>);
        impl Drop for Awaiting {
            fn drop(&mut self) {
                if let Some((tx, call)) = self.0.take() {
                    let _ = tx.send_from(
                        VoiceCommandOrigin::Host,
                        VoiceTaskCommand::Cancel { call }.into(),
                    );
                }
            }
        }
        let awaiting = Awaiting(Some((tx, call)));
        Ok(Box::pin(async move {
            let mut awaiting = awaiting;
            let answer = rx.await;
            awaiting.0 = None;
            answer.unwrap_or(Err(VoiceTaskEnded::Closing))
        }))
    }

    fn collect_dtmf(
        &self,
        num_digits: usize,
        ask_for_confirmation: bool,
        timeout: f64,
        stop_event: String,
    ) -> Result<BoxFuture<'static, Result<String, VoiceTaskEnded>>, CommandRefused> {
        use lk_agents_voice::agent_session::DtmfCommand;
        use lk_agents_voice::workflows::dtmf::{DtmfCollector, DtmfEvent, DtmfOutcome};
        let stop = DtmfEvent::parse(&stop_event)
            .ok_or_else(|| CommandRefused::Invalid(format!("{stop_event:?} is not a DTMF key")))?;
        let timeout = Duration::try_from_secs_f64(timeout)
            .map_err(|e| CommandRefused::Invalid(format!("timeout: {e}")))?;
        let collector = DtmfCollector::new(num_digits, ask_for_confirmation, timeout, stop)
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let (call, rx) = self.call(Answer::dtmf)?;
        let id = call.id;
        let tx = self.api()?;
        self.send_awaited(
            VoiceCommandOrigin::Host,
            DtmfCommand::Start { collector, call: Some(id) },
            call,
        )?;
        // Dropped unanswered (the awaiter was cancelled): the collection stops.
        struct Collecting(Option<(Tx, RequestId)>);
        impl Drop for Collecting {
            fn drop(&mut self) {
                if let Some((tx, call)) = self.0.take() {
                    let _ =
                        tx.send_from(VoiceCommandOrigin::Host, DtmfCommand::Cancel { call }.into());
                }
            }
        }
        let collecting = Collecting(Some((tx, id)));
        Ok(Box::pin(async move {
            let mut collecting = collecting;
            let outcome = rx.await;
            collecting.0 = None;
            match outcome.unwrap_or(DtmfOutcome::Stopped) {
                DtmfOutcome::Complete(result) => Ok(result.user_input),
                DtmfOutcome::Incomplete { message } => Err(VoiceTaskEnded::Failed(message)),
                DtmfOutcome::Stopped => Err(VoiceTaskEnded::Cancelled),
            }
        }))
    }

    fn report_avatar_metrics(
        &self,
        metrics: lk_agents_abi::metrics::base::AvatarMetrics,
    ) -> Result<(), SessionRefused> {
        let record = lk_agents_abi::metrics::base::AgentMetrics::Avatar(metrics);
        self.send(Command::HostMetrics(Box::new(record)))
    }

    fn set_audio_input(&self, source: Option<Arc<dyn AudioInput>>) -> Result<(), SessionRefused> {
        self.send(ActivityCommand::SetAudioInput(
            source.map(|source| AudioInputHandle(self.resources.register_audio_input(source))),
        ))
    }

    fn set_video_input(&self, source: Option<Arc<dyn VideoInput>>) -> Result<(), SessionRefused> {
        let source = source.map(|s| self.resources.register_video_input(s));
        self.send(ActivityCommand::SetVideoInput(source))
    }

    fn set_audio_output(&self, sink: Option<Arc<dyn AudioOutput>>) -> Result<(), SessionRefused> {
        if let Some(sink) = &sink {
            sink.tracker().anchor_to(self.anchor);
        }
        let sink = sink.map(|s| self.resources.register_audio_output(s));
        self.send(ActivityCommand::SetAudioOutput(sink))
    }

    fn set_transcription_output(
        &self,
        sink: Option<Arc<dyn TextOutput>>,
    ) -> Result<(), SessionRefused> {
        let sink = sink.map(|s| self.resources.register_text_output(s));
        self.send(ActivityCommand::SetTranscriptionOutput(sink))
    }

    fn update_tools(
        &self,
        tools: Option<Vec<ToolSpec>>,
        tools_host: Option<HostRef>,
        voice_agent: Option<Arc<dyn lk_agents_abi::agent::AgentFrame>>,
    ) -> Result<(), SessionRefused> {
        let toolset =
            Toolset { specs: tools.unwrap_or_default(), host: tools_host, agent: voice_agent };
        self.send(ActivityCommand::UpdateTools(toolset))
    }

    fn update_turn_options(&self, update: UpdateKwargs) -> Result<(), CommandRefused> {
        let patch = running_layer(&update)?;
        if let Maybe::Given(expressive) = update.expressive {
            self.send(ActivityCommand::UpdateExpressive(expressive))?;
        }
        if let Some(terms) = update.keyterms {
            self.send(Command::SetStaticKeyterms(terms))?;
        }
        if patch == AgentSessionOptions::default() {
            return Ok(());
        }
        Ok(self.send(TurnCommand::Configure(Box::new(patch)))?)
    }

    fn override_options(&self, update: UpdateKwargs) -> Result<u64, CommandRefused> {
        let options = running_layer(&update)?;
        let Some(session_options) = &self.host.options else {
            return Err(CommandRefused::Invalid(
                "override_options needs a session built with options".to_owned(),
            ));
        };
        let id = session_options.next_layer_id();
        self.send(TurnCommand::Override(Box::new(OverrideLayer { id, options })))?;
        Ok(id)
    }

    fn release_override(&self, layer: u64) {
        // Refused only once the session closed, when nothing is left to restore.
        let _ = self.send(TurnCommand::ReleaseOverride(layer));
    }

    fn resolved_options(&self) -> Option<ResolvedSessionOptions> {
        self.host.options.as_ref()?;
        let view = Arc::clone(&self.view.borrow());
        Some(lk_agents_voice::agent_activity::options::resolved_session_options(
            &view.config.effective,
            &view.config.stack.session,
        ))
    }

    fn push_video(&self, content: lk_agents_abi::ChatContent) -> Result<bool, NotAnImage> {
        let lk_agents_abi::ChatContent::Image(image) = content else { return Err(NotAnImage) };
        Ok(VoiceSession::push_video(self, image))
    }

    fn generate_reply(
        &self,
        user_input: Option<String>,
        instructions: Option<String>,
        allow_interruptions: Option<bool>,
        tool_choice: Option<lk_agents_abi::ToolChoice>,
        chat_ctx: Option<Vec<ChatItem>>,
        tools: Option<Vec<String>>,
        on_enter: bool,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<Speech, CommandRefused> {
        let origin = origin_of(origin);
        let (speech, call, reply) = self.speech_call(allow_interruptions)?;
        self.send_awaited(
            origin,
            SpeechCommand::GenerateReply {
                user_input,
                instructions,
                allow_interruptions,
                tool_choice,
                chat_ctx: chat_ctx.map(lk_agents_abi::ChatContext::new),
                toolset: None,
                tool_names: tools,
                on_enter,
                call: Some(call),
            },
            reply,
        )?;
        Ok(speech)
    }

    fn ivr_start(
        &self,
        max_silence: f64,
        tools: Option<Vec<ToolSpec>>,
        tools_host: Option<HostRef>,
    ) -> Result<(), CommandRefused> {
        if !(max_silence.is_finite() && max_silence >= 0.0) {
            return Err(CommandRefused::Invalid(
                "ivr_start: max_silence must be a finite, non-negative number of seconds"
                    .to_owned(),
            ));
        }
        // `_start_ivr_detection`: the session's tools extended, then the navigator started.
        if let Some(specs) = tools {
            self.send(Command::Ivr(IvrCommand::AddTools(Toolset {
                specs,
                host: tools_host,
                agent: None,
            })))?;
        }
        Ok(self.send(Command::Ivr(IvrCommand::Start {
            max_silence: Duration::from_secs_f64(max_silence),
        }))?)
    }

    fn attach_amd(
        &self,
        cfg: lk_agents_abi::voice::amd::AmdConfigKwargs,
        llm: Arc<dyn Llm>,
        tools: Option<HostRef>,
        stt: Option<Arc<dyn Stt>>,
        participant_identity: Option<String>,
    ) -> Result<bool, CommandRefused> {
        let config = lk_agents_voice::amd::machine::classifier::AmdConfig::resolve(&cfg)
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let setup = lk_agents_voice::amd::AmdSetup {
            config,
            prompt: cfg
                .detection
                .prompt
                .unwrap_or_else(|| lk_agents_abi::voice::amd::AMD_PROMPT.to_owned()),
            llm: llm_leg(llm, self.conn.llm),
            tools_host: tools,
            stt: stt.map(|s| {
                let conn = self.conn.stt.or_else(|| s.conn_options()).unwrap_or_default();
                s.with_conn_options(&conn).unwrap_or(s)
            }),
            room: self.host.call_room.clone(),
            participant: participant_identity,
        };
        Ok(VoiceSession::attach_amd(self, setup)?)
    }

    fn aclose(
        &self,
        reason: String,
        drain: bool,
    ) -> Result<BoxFuture<'static, ()>, UnknownCloseReason> {
        let reason = CloseReason::from_wire(&reason).ok_or(UnknownCloseReason(reason))?;
        Ok(self.close_session(reason, drain))
    }

    fn report(&self) -> Result<BoxFuture<'static, telemetry::SessionReport>, SessionRefused> {
        if let Some(report) = self.report.get() {
            let report = report.clone();
            return Ok(Box::pin(async move { report }));
        }
        let (reply, read) = tokio::sync::oneshot::channel();
        self.api()?
            .send(crate::voice::VoiceRuntimeEvent::Report(reply))
            .map_err(|_| SessionRefused::LoopEnded)?;
        // A loop that closed before reading it leaves its final report.
        let closed = self.report.clone();
        Ok(Box::pin(async move {
            read.await.unwrap_or_else(|_| closed.get().cloned().unwrap_or_default())
        }))
    }

    fn shutdown(&self, drain: bool) {
        if let Ok(closing) = self.aclose("user_initiated".to_owned(), drain) {
            self.tasks.spawn(self.spawner.as_ref(), "shutdown", closing);
        }
    }

    fn set_participant(
        &self,
        identity: Option<String>,
    ) -> BoxFuture<'static, lk_agents_abi::voice::io::Result<()>> {
        let room_io = Arc::clone(&self.room_io);
        Box::pin(async move { room_io.set_participant(identity.map(Into::into)).await })
    }

    fn filler_start(
        &self,
        call_id: String,
        speech_id: String,
        delay: f64,
        interval: Option<f64>,
        max_steps: Option<u32>,
        text: Option<String>,
        source: Option<HostRef>,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<u64, CommandRefused> {
        use lk_agents_voice::filler_scheduler::FillerError;
        let invalid = |e: FillerError| CommandRefused::Invalid(e.message().to_owned());
        if delay < 0.0 || delay.is_nan() {
            return Err(invalid(FillerError::NegativeDelay));
        }
        if interval.is_some_and(|i| i < 0.0 || i.is_nan()) {
            return Err(invalid(FillerError::NegativeInterval));
        }
        let speech =
            speech_id.parse::<SpeechId>().map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let source = match text {
            Some(text) => FillerSource::Text(text),
            None => FillerSource::Host(source),
        };
        let id = self.last_filler.fetch_add(1, Ordering::Relaxed) + 1;
        let origin = origin_of(origin);
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

    fn filler_stop(&self, id: u64) -> Result<(), SessionRefused> {
        self.send(SpeechCommand::FillerStop { id: FillerId(id) })
    }

    fn clear_user_turn(&self) -> Result<(), SessionRefused> {
        self.send(TurnCommand::Clear)
    }

    fn update_instructions(&self, instructions: String) -> Result<(), SessionRefused> {
        self.send(ActivityCommand::UpdateInstructions(Instructions::plain(instructions)))
    }

    fn set_io_enabled(&self, leg: String, enabled: bool) -> Result<(), CommandRefused> {
        let leg = match leg.as_str() {
            "audio_input" => IoKind::AudioInput,
            "audio_output" => IoKind::AudioOutput,
            "transcription" => IoKind::Transcription,
            other => return Err(CommandRefused::Invalid(format!("unknown io leg {other:?}"))),
        };
        Ok(self.send(ActivityCommand::SetIoEnabled { leg, enabled })?)
    }

    fn tool_update(
        &self,
        invocation_id: String,
        operation: u64,
        call_id: String,
        report: lk_agents_abi::llm::tool_executor::ToolReport,
    ) -> Result<(), CommandRefused> {
        self.api()?;
        let report =
            match self.tool_bodies.first_tool_progress(&invocation_id, operation, &call_id, report)
            {
                Ok(()) => return Ok(()),
                Err(report) => report,
            };
        let update = lk_agents_voice::agent_session::ToolUpdate::from(report);
        let origin = VoiceCommandOrigin::Tool {
            invocation_id: invocation_id.clone(),
            operation,
            call_id: call_id.clone(),
        };
        Ok(self.send_from(
            origin,
            SpeechCommand::ToolUpdate { invocation_id, operation, call_id, update },
        )?)
    }

    fn update_chat_ctx(&self, items: Vec<ChatItem>) -> Result<(), SessionRefused> {
        self.send(ActivityCommand::UpdateChatCtx(items))
    }

    fn interrupt(
        &self,
        force: bool,
    ) -> Result<BoxFuture<'static, Result<(), NotInterruptible>>, SessionRefused> {
        let rx = self.ask(VoiceCommandOrigin::Host, Answer::interrupt, |call| {
            SpeechCommand::Interrupt { force, call: Some(call) }
        })?;
        Ok(Box::pin(async move {
            match rx.await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(_)) => Err(NotInterruptible),
                // The loop ended before the walk: nothing is speaking any more.
                Err(_) => Ok(()),
            }
        }))
    }

    fn drain(&self) -> Result<BoxFuture<'static, Option<String>>, SessionRefused> {
        let rx = self.ask(VoiceCommandOrigin::Host, Answer::drain, |call| {
            ActivityCommand::Drain { call: Some(call) }
        })?;
        Ok(Box::pin(async move {
            match rx.await {
                Ok(Ok(())) => None,
                Ok(Err(refused)) => Some(refused.as_str().to_owned()),
                Err(_) => Some(DrainRefused::Closing.as_str().to_owned()),
            }
        }))
    }

    fn wait_for_idle(
        &self,
        hold: Option<u64>,
    ) -> Result<BoxFuture<'static, Result<(), IdleRefused>>, SessionRefused> {
        let rx = self.ask(VoiceCommandOrigin::Host, Answer::idle, |call| {
            SpeechCommand::WaitForIdle { hold: hold.map(HoldId), call: Some(call) }
        })?;
        Ok(Box::pin(async move { rx.await.unwrap_or(Err(IdleRefused::Closing)) }))
    }

    fn wait_for_idle_and_hold(
        &self,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<BoxFuture<'static, Result<u64, IdleRefused>>, CommandRefused> {
        let origin = origin_of(origin);
        let rx = self
            .ask(origin, Answer::hold, |c| SpeechCommand::WaitForIdleAndHold { call: Some(c) })?;
        Ok(Box::pin(async move { rx.await.unwrap_or(Err(IdleRefused::Closing)).map(|h| h.0) }))
    }

    fn release_idle_hold(&self, hold: u64) {
        // A closed loop has released everything already.
        let _ = self.send(SpeechCommand::ReleaseIdleHold { hold: HoldId(hold) });
    }

    fn claim_user_turn(&self) -> Result<(), SessionRefused> {
        self.send(SpeechCommand::ClaimUserTurn)
    }

    fn release_user_turn(&self) {
        // A closed loop has nothing left to release.
        let _ = self.send(SpeechCommand::ReleaseUserTurn);
    }

    fn run(
        &self,
        user_input: Option<String>,
        origin: Option<VoiceCommandOrigin>,
        host: Option<HostRef>,
    ) -> Result<(), CommandRefused> {
        let spec = lk_agents_voice::run_result::RunSpec { user_input, host };
        Ok(self.send_from(origin_of(origin), Command::Run(Box::new(spec)))?)
    }

    fn say(
        &self,
        text: String,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<Speech, CommandRefused> {
        self.speak((SayText::Text(text), None), allow_interruptions, add_to_chat_ctx, origin)
    }

    fn say_stream(
        &self,
        text: Box<dyn lk_agents_abi::StreamOutput<String>>,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<Speech, CommandRefused> {
        let (speech, pusher) =
            self.say_text_channel(allow_interruptions, add_to_chat_ctx, origin)?;
        self.feed(pusher, text);
        Ok(speech)
    }

    fn say_audio(
        &self,
        text: String,
        audio: Box<dyn lk_agents_abi::StreamOutput<lk_agents_abi::AudioFrame>>,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<Speech, CommandRefused> {
        let (speech, pusher) =
            self.say_audio_channel(text, allow_interruptions, add_to_chat_ctx, origin)?;
        self.feed(pusher, audio);
        Ok(speech)
    }

    fn ivr_final(&self, text: String) -> Result<(), SessionRefused> {
        self.send(IvrCommand::SeedTranscript { text })
    }

    fn amd_listening(&self) -> bool {
        self.running(|r| Ok(r.amd.as_ref().is_some_and(|amd| amd.listening()))).unwrap_or(false)
    }

    fn amd_execute(&self) -> Result<(), SessionRefused> {
        self.running(|r| {
            if let Some(handle) = &r.amd {
                handle.execute();
            }
            Ok(())
        })
    }

    fn amd_close(&self) -> Result<(), SessionRefused> {
        self.running(|r| {
            if let Some(handle) = r.amd.take() {
                handle.close();
            }
            Ok(())
        })
    }

    fn keyterms(&self) -> Option<Vec<String>> {
        match &*self.stage.lock() {
            Stage::Running(r) => r.keyterms.borrow().clone(),
            _ => None,
        }
    }

    fn agent_chat_ctx(&self) -> Vec<ChatItem> {
        let context = match &*self.stage.lock() {
            Stage::Running(r) => Some(Arc::clone(&r.chat_ctx.borrow())),
            _ => None,
        };
        context.map(|context| context.items.clone()).unwrap_or_default()
    }

    fn commit_user_turn(
        &self,
        transcript_timeout: f64,
        stt_flush_duration: f64,
        skip_reply: bool,
    ) -> Result<BoxFuture<'static, Result<String, CommitSuperseded>>, SessionRefused> {
        let rx = self.ask(VoiceCommandOrigin::Host, Answer::commit, |call| {
            SpeechCommand::CommitUserTurn {
                transcript_timeout: Duration::from_secs_f64(transcript_timeout.max(0.0)),
                stt_flush_duration: Duration::from_secs_f64(stt_flush_duration.max(0.0)),
                skip_reply,
                call: Some(call),
            }
        })?;
        Ok(Box::pin(async move { rx.await.unwrap_or(Err(CommitSuperseded)) }))
    }
}

impl VoiceSession {
    /// The span the session's spans continue, the job they record for and the spans `start()`
    /// reserved; before `start()`.
    pub(crate) fn trace_under(
        &self,
        trace: telemetry::Cx,
        job_ids: telemetry::JobIds,
        opening: telemetry::Opening,
    ) {
        if let Stage::Ready(ready) = &mut *self.stage.lock() {
            ready.builder.trace = trace;
            ready.builder.job_ids = job_ids;
            ready.builder.opening = Some(opening);
        }
    }

    /// Native callers already hold a validated close reason.
    pub(crate) fn close_session(&self, reason: CloseReason, drain: bool) -> BoxFuture<'static, ()> {
        // A session never started has nothing to wait for, and its turn-detector stream stays the
        // host's to close.
        let was_running = self.api().is_ok();
        let remote = self.remote_host();
        let closing = self.close(reason, drain, None);
        let close_io = self.close_io();
        let clock = self.time.clock();
        let eot = self.host.eot.clone().filter(|_| was_running);
        let hooks = self.host.hooks.clone().filter(|_| was_running);
        let closed = Arc::clone(&self.host.closed);
        Box::pin(async move {
            closed
                .get_or_init(|| async move {
                    if let Some(hooks) = &hooks
                        && let Err(e) = hooks.closing(&telemetry::Cx::NONE).await
                    {
                        hooks.hook_failed("closing", e.host_ref.clone());
                    }
                    closing.await;
                    if let Some(eot) = eot {
                        let _ = lk_agents_abi::timeout(
                            &*clock,
                            HOST_TEARDOWN_TIMEOUT,
                            eot.aclose(&telemetry::Cx::NONE),
                        )
                        .await;
                    }
                    if let Some(remote) = remote {
                        remote.aclose().await;
                    }
                    close_io.await;
                    if let Some(hooks) = &hooks
                        && let Err(error) = hooks.toolset_scopes().release(None).await
                    {
                        log::warn!("the session toolsets could not close: {error}");
                    }
                })
                .await;
        })
    }
}
