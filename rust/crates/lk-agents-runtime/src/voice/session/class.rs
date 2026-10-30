//! The declared `_core.VoiceSession` over [`VoiceSession`]: assembly from the host's arguments,
//! and every other method one command to the loop.

use super::*;
use crate::voice::session::transcript_sync::{TranscriptSyncConfig, TranscriptSynchronizer};
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, TextOutput, VideoInput};
use lk_agents_abi::voice::nodes::NodeFactory;
use lk_agents_abi::voice::session::{
    CommitSuperseded, IdleRefused, NotAnImage, NotInterruptible, UnknownCloseReason,
    VoiceSessionArgs, VoiceTaskEnded,
};
use lk_agents_abi::voice::session_options::{ResolvedSessionOptions, UpdateKwargs};
use lk_agents_abi::{ChatItem, ToolSpec};
use lk_agents_voice::agent_activity::DriverConfig;
use lk_agents_voice::agent_activity::options::{AgentSessionOptions, DetectorKind, OverrideLayer};
use lk_agents_voice::agent_session::{
    ActivityCommand, DrainRefused, FillerSource, HoldId, HostAudio, HostVideo, IoKind, IvrCommand,
    TurnCommand,
};
use lk_agents_voice::filler_scheduler::FillerId;

type Speech = Arc<SpeechHandle>;

impl lk_agents_abi::voice::session::VoiceSession for VoiceSession {
    type Env = lk_agents_net::env::ProcessEnv;
    type Options = SessionOptions;
    type Speech = Speech;
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
        agents.install(&args.agent, args.agent_host);
        let time = match room.virtual_clock() {
            Some(clock) => Timebase::from_virtual(clock),
            None => Timebase::of_clock(room.clock()),
        };
        let clock = time.clock();
        let anchor = lk_agents_abi::WallAnchor::of(&*clock);
        let mut config = lk_agents_voice::room_io::options::resolve_config(
            args.room_io_config,
            args.participant_identity,
            args.input_sample_rate,
            args.output_sample_rate,
            args.auto_gain_control,
            args.text_output,
        );
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
        let room_link = lk_agents_voice::room_io::RoomLinkConfig {
            close_on_disconnect: config.close_on_disconnect,
            delete_room_on_close: config.delete_room_on_close,
            room: rtc_room.as_ref().map(|room| room.name()).transpose()?.unwrap_or_default(),
        };
        let room_io = room.room_io(spawner.clone(), config);

        // An avatar's sink is an audio output the room does not know; the transcript syncs to it.
        let has_external_audio_output = host_audio_output.is_some();
        let mut synchronizer = None;
        let (audio_out, text_out) = match (
            host_audio_output.or_else(|| room_io.audio_output()),
            args.host_transcription_output.or_else(|| room_io.transcription_output()),
        ) {
            (Some(audio), Some(text))
                if room_io.wants_transcript_sync(has_external_audio_output) =>
            {
                let sync = TranscriptSynchronizer::new(TranscriptSyncConfig {
                    audio_next: audio,
                    text_next: Some(text),
                    speed: sync_speed.unwrap_or(1.0),
                    hyphenator: Arc::new(lk_agents_voice::transcription::synchronizer::Liang),
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

        let interruption = args
            .interruption_detector
            .map(|source| Box::new(move |anchor| source.open(anchor)) as InterruptionPortFactory);
        let nodes = Nodes { llm: args.llm_node, tts: None, stt: None };
        let llm = args.llm.map(|l| nodes.llm(llm_leg(l, conn.llm)));
        // Facts, not options: the legs the two leg-dependent defaults need.
        let legs = LegSet {
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
        let speeches = Arc::clone(&builder.speeches);
        builder.session_id = args.session_id;
        builder.video_sampler = args.video_sampler.map(|(speaking, silent)| {
            lk_agents_voice::room_io::video_sampler::VideoSampler::new(speaking, silent)
        });
        builder.instructions = args.instructions.map(Instructions::plain).unwrap_or_default();
        builder.dynamic_instructions = args.dynamic_instructions;
        // Opened at each use, above the synchroniser: the node rewrites, the synchroniser paces.
        builder.output_nodes = OutputNodes {
            transcription: args.transcription_node,
            realtime_audio: args.realtime_audio_output_node,
        };
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
        builder.audio_out = audio_out;
        builder.text_out = text_out;
        builder.room_link = Some(room_link);
        builder.toolset =
            Toolset::set_up(args.tools.unwrap_or_default(), args.tools_host, args.voice_agent);

        Ok(VoiceSession {
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
            synchronizer,
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
            },
            resources,
            conn,
        })
    }

    fn start(&self) -> Result<BoxFuture<'static, Result<(), StartFailed>>, OpenRefused> {
        self.start_with(Box::new(crate::voice::Executor::new(self.time.clone(), self.spawner())))
    }

    fn usage(&self) -> lk_agents_abi::usage::Usage {
        self.view.borrow().usage.clone()
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
        self.send(ActivityCommand::UpdateModels { stt, vad, llm, tts })
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
        self.send_frame(Origin::Host, update, call, |u| {
            ActivityCommand::Update(Box::new(u)).into()
        })?;
        Ok(Box::pin(async move { matches!(rx.await, Ok(Ok(()))) }))
    }

    fn run_voice_task(
        &self,
        frame: UpdateAgentArgs,
        origin: Option<String>,
        opens_with_reply: bool,
    ) -> Result<BoxFuture<'static, Result<HostRef, VoiceTaskEnded>>, CommandRefused> {
        use lk_agents_voice::agent_activity::voice_task::{VoiceTaskCommand, VoiceTaskRun};
        let origin = origin_of(origin)?;
        let (awaited, rx) = self.call(Answer::voice_task)?;
        let call = awaited.id;
        let tx = self.api()?;
        let update = self.frame(frame);
        self.send_frame(origin, update, awaited, move |update| {
            let run = VoiceTaskRun { update: Box::new(update), opens_with_reply, call };
            VoiceTaskCommand::Run(Box::new(run)).into()
        })?;
        // Dropped unanswered (the awaiter was cancelled): the task ends.
        struct Awaiting(Option<(Tx, RequestId)>);
        impl Drop for Awaiting {
            fn drop(&mut self) {
                if let Some((tx, call)) = self.0.take() {
                    let _ = tx.send_from(Origin::Host, VoiceTaskCommand::Cancel { call }.into());
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
        self.send_awaited(Origin::Host, DtmfCommand::Start { collector, call: Some(id) }, call)?;
        // Dropped unanswered (the awaiter was cancelled): the collection stops.
        struct Collecting(Option<(Tx, RequestId)>);
        impl Drop for Collecting {
            fn drop(&mut self) {
                if let Some((tx, call)) = self.0.take() {
                    let _ = tx.send_from(Origin::Host, DtmfCommand::Cancel { call }.into());
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
        self.send(ActivityCommand::SetAudioInput(source.map(HostAudio)))
    }

    fn set_video_input(&self, source: Option<Arc<dyn VideoInput>>) -> Result<(), SessionRefused> {
        self.send(ActivityCommand::SetVideoInput(source.map(HostVideo)))
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
        Ok(self.send(TurnCommand::Configure(patch))?)
    }

    fn override_options(&self, update: UpdateKwargs) -> Result<u64, CommandRefused> {
        let options = running_layer(&update)?;
        let Some(session_options) = &self.host.options else {
            return Err(CommandRefused::Invalid(
                "override_options needs a session built with options".to_owned(),
            ));
        };
        let id = session_options.next_layer_id();
        self.send(TurnCommand::Override(OverrideLayer { id, options }))?;
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
        tools: Option<Vec<ToolSpec>>,
        tools_host: Option<HostRef>,
        voice_agent: Option<Arc<dyn lk_agents_abi::agent::AgentFrame>>,
        on_enter: bool,
        origin: Option<String>,
    ) -> Result<Speech, CommandRefused> {
        let origin = origin_of(origin)?;
        // An empty toolset would tell the model about no tool at all; `None` keeps the agent's.
        let toolset =
            tools.map(|specs| Arc::new(Toolset { specs, host: tools_host, agent: voice_agent }));
        let (speech, call, reply) = self.speech_call(allow_interruptions)?;
        self.send_awaited(
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
    ) -> Result<(), CommandRefused> {
        let config = lk_agents_voice::amd::classifier::AmdConfig::resolve(&cfg)
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let setup = lk_agents_voice::amd::AmdSetup {
            config,
            prompt: cfg.prompt.unwrap_or_else(|| lk_agents_abi::voice::amd::AMD_PROMPT.to_owned()),
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
        Ok(Box::pin(async move {
            closed
                .get_or_init(|| async move {
                    if let Some(hooks) = &hooks
                        && let Err(e) = hooks.closing().await
                    {
                        hooks.hook_failed("closing", e.host_ref.clone());
                    }
                    closing.await;
                    if let Some(eot) = eot {
                        let _ =
                            lk_agents_abi::timeout(&*clock, HOST_TEARDOWN_TIMEOUT, eot.aclose())
                                .await;
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
        call_id: String,
        report: lk_agents_abi::llm::tool_executor::ToolReport,
    ) -> Result<(), CommandRefused> {
        let update = lk_agents_voice::agent_session::ToolUpdate::from(report);
        let origin = Origin::Tool { call_id: call_id.clone() };
        Ok(self.send_from(origin, SpeechCommand::ToolUpdate { call_id, update })?)
    }

    fn update_chat_ctx(&self, items: Vec<ChatItem>) -> Result<(), SessionRefused> {
        self.send(ActivityCommand::UpdateChatCtx(items))
    }

    fn interrupt(
        &self,
        force: bool,
    ) -> Result<BoxFuture<'static, Result<(), NotInterruptible>>, SessionRefused> {
        let rx = self.ask(Origin::Host, Answer::interrupt, |call| SpeechCommand::Interrupt {
            force,
            call: Some(call),
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
        let rx = self
            .ask(Origin::Host, Answer::drain, |call| ActivityCommand::Drain { call: Some(call) })?;
        Ok(Box::pin(async move {
            match rx.await {
                Ok(Ok(())) => None,
                Ok(Err(refused)) => Some(refused.as_str().to_owned()),
                Err(_) => Some(DrainRefused::Closing.as_str().to_owned()),
            }
        }))
    }

    fn end_call(&self, speech_id: String) -> Result<(), CommandRefused> {
        // A handle constructed with no id names no speech: nothing to end.
        if speech_id.is_empty() {
            return Ok(());
        }
        let speech =
            speech_id.parse::<SpeechId>().map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        Ok(self.send(SpeechCommand::EndCall { speech })?)
    }

    fn wait_for_idle(
        &self,
        hold: Option<u64>,
    ) -> Result<BoxFuture<'static, Result<(), IdleRefused>>, SessionRefused> {
        let rx = self.ask(Origin::Host, Answer::idle, |call| SpeechCommand::WaitForIdle {
            hold: hold.map(HoldId),
            call: Some(call),
        })?;
        Ok(Box::pin(async move { rx.await.unwrap_or(Err(IdleRefused::Closing)) }))
    }

    fn wait_for_idle_and_hold(
        &self,
        origin: Option<String>,
    ) -> Result<BoxFuture<'static, Result<u64, IdleRefused>>, CommandRefused> {
        let origin = origin_of(origin)?;
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
        origin: Option<String>,
        host: Option<HostRef>,
    ) -> Result<(), CommandRefused> {
        let spec = lk_agents_voice::run_result::RunSpec { user_input, host };
        Ok(self.send_from(origin_of(origin)?, Command::Run(Box::new(spec)))?)
    }

    fn say(
        &self,
        text: String,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<String>,
    ) -> Result<Speech, CommandRefused> {
        self.speak((SayText::Text(text), None), allow_interruptions, add_to_chat_ctx, origin)
    }

    fn say_stream(
        &self,
        text: Box<dyn lk_agents_abi::StreamSource<String>>,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<String>,
    ) -> Result<Speech, CommandRefused> {
        let (speech, pusher) =
            self.say_text_channel(allow_interruptions, add_to_chat_ctx, origin)?;
        self.feed(pusher, text);
        Ok(speech)
    }

    fn say_audio(
        &self,
        text: String,
        audio: Box<dyn lk_agents_abi::StreamSource<lk_agents_abi::AudioFrame>>,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<String>,
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
        let rx = self.ask(Origin::Host, Answer::commit, |call| SpeechCommand::CommitUserTurn {
            transcript_timeout: Duration::from_secs_f64(transcript_timeout.max(0.0)),
            stt_flush_duration: Duration::from_secs_f64(stt_flush_duration.max(0.0)),
            skip_reply,
            call: Some(call),
        })?;
        Ok(Box::pin(async move { rx.await.unwrap_or(Err(CommitSuperseded)) }))
    }
}

/// What `update_options` / `override_options` say to a running session; a turn detector object
/// is refused: swapping the detector of a running session is not ported.
fn running_layer(update: &UpdateKwargs) -> Result<AgentSessionOptions, CommandRefused> {
    use lk_agents_abi::voice::session_options::TurnDetectionKwarg;
    if let Maybe::Given(TurnDetectionKwarg::Detector { .. }) = update.turn_detection {
        return Err(CommandRefused::Invalid(
            "update_options(turn_detection=) takes a mode on a running session; a turn \
             detector object is set with VoiceSession(turn_handling=)"
                .to_owned(),
        ));
    }
    Ok(lk_agents_voice::agent_activity::options::update_layer(update))
}
