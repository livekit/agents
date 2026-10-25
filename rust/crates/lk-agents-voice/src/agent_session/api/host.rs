//! [`SessionApi`]'s host methods: the one `#[host(object)]` block the Python and Node
//! `AgentSession` call, each assembling, opening, closing or sending one command.

use super::{
    CLOSE_TIMEOUT, CommandRefused, FramePusher, HostMetricsRecord, HostParts,
    InterruptionPortFactory, ModelRefs, NodeFactory, NotAnImage, NotInterruptible, OpenRefused,
    Ready, Reply, SessionApi, SessionBuildError, SessionEvents, SessionRefused, SessionRoom,
    SessionRunner, Stage, StartFailed, TextPusher, UnknownCloseReason, UpdateAgentArgs,
    host_toolset, llm_leg, origin_of, session_tools, stt_leg, tts_leg,
};
use crate::agent_activity::DriverConfig;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::DriverBuilder;
use crate::agent_session::commands::{
    ActivityCommand, Command, DrainRefused, DtmfCommand, FillerSource, HoldId, HookId, HookOutcome,
    HostAudio, HostAudioOut, HostTextOut, HostVideo, IdleRefused, IoKind, IvrCommand, Origin,
    SayText, SpeechCommand, TaskCommand, TaskEntry, ToolUpdate, TurnCommand,
};
use crate::agent_session::time::Timebase;
use crate::agent_session::{Answer, CommitSuperseded};
use crate::filler_scheduler::FillerId;
use crate::generation::Instructions;
use crate::trace::Step;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::HostWire;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, TextOutput};
use lk_agents_abi::{Llm, Maybe, Spawner, Stt, Tts, Vad};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// `voice/agent_session.py`'s `AgentSession`, over the driver: assembled at construction, opened at
/// `start()`, closed at `aclose()`; every other method is one command. `frozen`, so there is no
/// borrow flag: `aclose()` can be awaited while the event drain calls back into the same object.
#[lk_agents_abi::host(object(name = "LiveSession"))]
impl SessionApi {
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
            turn_detector_streaming = false,
            text_output = true,
            host_video_sampler = false,
            using_default_vad = false,
            max_unrecoverable_errors = 3
        )
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn host_new(
        room: Arc<dyn SessionRoom>,
        instructions: Option<Instructions>,
        agent: String,
        chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
        stt: Option<Arc<dyn Stt>>,
        speaker_filter: Option<crate::audio::SpeakerFilterConfig>,
        vad: Option<Arc<dyn Vad>>,
        llm: Option<Arc<dyn Llm>>,
        // The native realtime model; a plugin's `RealtimeModel` is `realtime_host`.
        realtime: Option<Arc<dyn lk_agents_abi::Realtime>>,
        realtime_host: Option<Arc<dyn lk_agents_abi::llm::realtime::HostRealtime>>,
        tts: Option<Arc<dyn Tts>>,
        on_event: Option<Arc<dyn SessionEvents>>,
        participant_identity: Option<String>,
        auto_gain_control: bool,
        input_sample_rate: u32,
        output_sample_rate: u32,
        // The session layer `_core.SessionOptions(kwargs)` built; none is an empty session layer.
        options: Option<&crate::agent_activity::options::HostOptions>,
        // The first agent's own `turn_handling`, as written: the agent layer of the stack.
        agent_turn_handling: Option<crate::agent_activity::options::TurnHandlingKwargs>,
        // `voice.turn._CoreTurnDetector` over the user's `turn_detection`, and which protocol it
        // speaks (`hasattr(detector, "stream")`).
        turn_detector: Option<Arc<dyn lk_agents_abi::EotDetector>>,
        turn_detector_streaming: bool,
        // The adaptive-interruption detector: an object means the host's gate said yes.
        interruption_detector: Option<
            Arc<dyn lk_agents_abi::turn::interruption::InterruptionSource>,
        >,
        text_output: bool,
        // `RoomOptions._to_rust_config()`, resolved per leg; without it the scalars above.
        room_io_config: Option<crate::room_io::types::RoomIoConfig>,
        tools: Option<Vec<lk_agents_abi::ToolSpec>>,
        tools_host: Option<lk_agents_abi::HostRef>,
        tool_runner: Option<Arc<dyn crate::tool_executor::HostToolRunner>>,
        transcription_node: Option<Arc<dyn NodeFactory>>,
        realtime_audio_output_node: Option<Arc<dyn NodeFactory>>,
        tts_text_transforms: Option<crate::transcription::text_transforms::TransformSpecs>,
        // The host's own io legs, replacing the room's (which `agent_session.py` has turned off).
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
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Result<Self, SessionBuildError> {
        use crate::room_io::types::{
            AudioInputConfig, AudioOutputConfig, RoomIoConfig, TextOutputConfig,
        };
        let time = room.timebase();
        let clock = time.clock();
        let anchor = lk_agents_abi::WallAnchor::of(&*clock);
        let config = room_io_config.unwrap_or_else(|| RoomIoConfig {
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
        // A host sink's playback reports are stamped on the session's clock.
        if let Some(out) = &host_audio_output {
            out.tracker().anchor_to(anchor);
        }
        let realtime = realtime.or_else(|| {
            realtime_host.map(|host| {
                Arc::new(lk_agents_abi::llm::realtime::HostRealtimeModel::new(host, anchor))
                    as Arc<dyn lk_agents_abi::Realtime>
            })
        });
        let (session_layer, diagnostics) = match options {
            Some(o) => (o.session_patch(), o.take_diagnostics()),
            None => (crate::agent_activity::options::SessionPatch::default(), Vec::new()),
        };
        let keyterms = stt_context.map(|cfg| crate::keyterm_detection::KeytermSetup {
            detection: crate::keyterm_detection::resolve_detection(cfg.keyterm_detection.as_ref()),
            static_keyterms: cfg.keyterms.unwrap_or_default(),
            llm: keyterm_llm.map(llm_leg),
            tools_host: keyterm_tools,
        });
        let stack = crate::agent_activity::options::ConfigStack {
            session: session_layer,
            agent: agent_turn_handling.as_ref().map(crate::turn::agent_patch),
        };
        let stt = stt.map(stt_leg);
        let llm = llm.map(llm_leg);
        let tts = tts.map(|t| tts_leg(t, clock.clone()));
        let interruption = interruption_detector
            .map(|source| Box::new(move |anchor| source.open(anchor)) as InterruptionPortFactory);
        let toolset = host_toolset(tools, tools_host, tool_runner);
        let room = room.rtc();

        // **Nothing starts here**; the loop begins at `start()`. Synchronous so a host can bind the
        // room's legs and recorder before starting. Read before the config moves into `RoomIo`.
        let sync_speed = config.text_output.as_ref().map(|t| t.transcription_speed_factor);
        let room_link = crate::room_io::link::RoomLinkConfig {
            close_on_disconnect: config.close_on_disconnect,
            delete_room_on_close: config.delete_room_on_close,
            room: room.name().to_owned(),
        };
        let room_io = crate::room_io::RoomIo::new(
            Arc::clone(&room),
            clock.clone(),
            spawner.clone(),
            Arc::new(crate::room_io::_output::transcription::CountingSegmentIds::default()),
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

        // `realtime_audio_output_node`, over the synchroniser's node or the room's output; with no
        // audio leg it has nothing to play into. What a factory raises comes back as itself.
        let realtime_audio_node = match (realtime_audio_output_node, &audio_out) {
            (Some(factory), Some(next)) => Some(
                factory
                    .audio_node(crate::io::CoreAudioSink::new(Arc::clone(next)))
                    .map_err(SessionBuildError::Node)?,
            ),
            _ => None,
        };

        // `Agent.transcription_node`, above the synchroniser (the node rewrites, the synchroniser
        // paces). With no leg below it is dropped: `text_output=False` is legitimate.
        let text_out: Option<Arc<dyn TextOutput>> = match (transcription_node, text_out) {
            (Some(factory), Some(next)) => {
                let node = factory
                    .text_node(crate::io::CoreTextOutput::new(Arc::clone(&next)))
                    .map_err(SessionBuildError::Node)?;
                Some(Arc::new(crate::io::NodeTextOutput::new(node, next)))
            }
            (Some(_), None) => None,
            (None, next) => next,
        };

        // Facts, not options: the legs the two leg-dependent defaults need.
        let legs = crate::agent_activity::options::LegSet {
            stt: stt.is_some(),
            vad: vad.is_some(),
            using_default_vad,
            llm: llm.is_some(),
            tts: tts.is_some(),
            realtime: realtime.is_some(),
            detector: turn_detector.as_ref().map(|_| {
                if turn_detector_streaming {
                    crate::agent_activity::options::DetectorKind::Streaming
                } else {
                    crate::agent_activity::options::DetectorKind::Batch
                }
            }),
            adaptive_detector: interruption.is_some(),
            audio_enabled: true,
            audio_output: true,
            sink_can_pause: true,
        };

        let resources = lk_agents_abi::resources::Resources::default();
        let builder = DriverBuilder {
            resources: resources.clone(),
            agent,
            time: time.clone(),
            spawner: spawner.clone(),
            interruption: None,
            video_sampler: video_sampler.map(|(speaking, silent)| {
                Arc::new(crate::room_io::_input::VideoSampler::new(speaking, silent))
            }),
            // **The one resolution**; no default restated beside it.
            config: DriverConfig::new(stack, legs),
            thresholds: lk_agents_abi::tokenize::language::ThresholdOptions::new(
                lk_agents_abi::tokenize::language::TurnDetectorModel::V1Mini,
                None,
                None,
            ),
            instructions: instructions.unwrap_or_default(),
            chat_ctx,
            hook_bound: hook_bound.map(Duration::from_secs_f64),
            capture_run: None,
            max_unrecoverable_errors,
            eot: turn_detector.clone(),
            llm,
            realtime,
            tts,
            stt,
            speaker_filter,
            keyterms,
            vad,
            tts_text_transforms: tts_text_transforms.map(|t| t.0).unwrap_or_default(),
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
            realtime_audio_node,
            room_events: Some(room.events()),
            // Installed at `open`, once `RoomIo` has been started and its stream taken.
            room_io_events: None,
            room_link: Some(room_link),
            toolset,
        };

        let mut api = SessionApi {
            stage: Arc::new(Mutex::new(Stage::Ready(Box::new(Ready { builder, interruption })))),
            legs: Mutex::new(legs),
            stack: Mutex::new(stack),
            room_io,
            synchronizer,
            spawner,
            anchor: lk_agents_abi::WallAnchor::of(&*clock),
            time,
            dtmf_reply: Mutex::new(None),
            host: HostParts::default(),
            resources,
        };
        // The reference's `@deprecate_params` warnings, as `Diagnostic` events, not Python logs.
        if let Some(events) = &on_event {
            for d in diagnostics {
                events.event(&HostWire::Diagnostic(d.wire()));
            }
        }
        api.host =
            HostParts { events: on_event, eot: turn_detector, has_options: options.is_some() };
        Ok(api)
    }

    /// Publish the agent's track, open every provider stream, and start the loop. Awaitable, since
    /// the provider factories must be opened on the loop.
    ///
    /// Does not wait for a participant; await `wait_until_ready` for that. `capture_run=True` opens
    /// `start(capture_run=True)`'s run before the opening frame's `AgentHandoff` record and its
    /// `on_enter`.
    #[host(method(kw_only, defaults(capture_run = false)))]
    pub fn start(&self, capture_run: bool) -> Result<Reply<Result<(), StartFailed>>, OpenRefused> {
        let events = self.host.events.clone();
        let anchor = self.anchor;
        // The drain: one event per call, in driver order, then one last word.
        let opening = self.open(capture_run, move |mut runner: SessionRunner| {
            Box::pin(async move {
                let api = runner.api().clone();
                while let Some(ev) = runner.next_host_event().await {
                    if let Some(events) = &events {
                        events.event(&ev.wire(anchor, &api));
                    }
                }
                // Dropping the runner after it tells `aclose()` the host has seen everything.
                if let Some(events) = &events {
                    events.stream_closed(&lk_agents_abi::voice::events::StreamClosedWire {});
                }
            })
        })?;
        Ok(Reply::new(async move { opening.await.map_err(StartFailed) }))
    }

    /// The session's clock, as the handle a provider's `_bind_session_clock` takes.
    #[host(getter)]
    pub fn session_clock(&self) -> Timebase {
        self.time.clone()
    }

    /// The room's video leg, when `RoomIoConfig(video_input=True)` asked for one, else `None`. A
    /// fresh handle each call over the same leg; safe before or after `start()`.
    #[host(getter)]
    pub fn video_input(&self) -> Option<Arc<crate::room_io::_input::ParticipantVideoInput>> {
        self.room_io.video_input()
    }

    /// The room's publishing audio leg, as `_core.RoomAudioOutput`, when `audio_output` is on. A
    /// fresh handle each call over the same leg.
    #[host(getter)]
    pub fn audio_output(&self) -> Option<Arc<crate::room_io::_output::RoomAudioLeg>> {
        self.room_io
            .audio_output()
            .map(|out| Arc::new(crate::room_io::_output::RoomAudioLeg::of(out, self.anchor)))
    }

    /// The agent's transcription leg, as `_core.RoomTranscriptionOutput`, when `text_output` is on.
    #[host(getter)]
    pub fn transcription_output(
        &self,
    ) -> Option<Arc<crate::room_io::_output::transcription::RoomTranscriptionLeg>> {
        self.room_io
            .agent_transcription()
            .map(|t| Arc::new(crate::room_io::_output::transcription::RoomTranscriptionLeg::of(t)))
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
    /// provider swap, not a handoff (same frame, no hook, no drain). Each leg is three-state; the
    /// primary-speaker filter follows the STT, on the same queue (a swapped-in STT that is no
    /// `MultiSpeakerAdapter` clears it).
    #[host(method(kw_only))]
    pub fn update_models(
        &self,
        llm: Maybe<Option<Arc<dyn Llm>>>,
        tts: Maybe<Option<Arc<dyn Tts>>>,
        stt: Maybe<Option<Arc<dyn Stt>>>,
        vad: Maybe<Option<Arc<dyn Vad>>>,
        speaker_filter: Maybe<Option<crate::audio::SpeakerFilterConfig>>,
    ) -> Result<(), SessionRefused> {
        let (ModelRefs { llm, tts, stt, vad }, filter) =
            self.swap(llm, tts, stt, vad, speaker_filter);
        self.send(ActivityCommand::UpdateModels { stt, vad, llm, tts })?;
        if let Some(config) = filter {
            self.send(ActivityCommand::SetSpeakerFilter(config))?;
        }
        Ok(())
    }

    /// `describe_agent` answered: the frame of the `Agent` a tool returned, with the keywords
    /// `update_agent` takes. The driver performs the handoff from here.
    #[host(method(kw_only = "frame"))]
    pub fn hook_described(
        &self,
        id: u64,
        #[host(kwargs)] frame: UpdateAgentArgs,
    ) -> Result<(), SessionRefused> {
        let (update, filter) = self.frame(frame);
        self.send(Command::HookDone {
            id: HookId(id),
            outcome: HookOutcome::Described(Box::new(update)),
        })?;
        if let Some(config) = filter {
            self.send(ActivityCommand::SetSpeakerFilter(config))?;
        }
        Ok(())
    }

    /// The handoff: replace the agent, keep the session. Resolves `True` once the frame is
    /// installed (`on_enter` issued, not returned), `False` when it was refused (closing).
    ///
    /// The driver owns the providers, scheduler and buffered frames and outlives the swap, so there
    /// is no `ReusableResources` handover. An omitted provider keeps its stream.
    #[host(method(kw_only))]
    pub fn update_agent(
        &self,
        #[host(kwargs)] frame: UpdateAgentArgs,
    ) -> Result<Reply<bool>, SessionRefused> {
        // Always `_update_activity(previous_activity="close", new_activity="start")`; the
        // pause/resume pair is an inline task's (`enter_task`).
        let (mut update, filter) = self.frame(frame);
        let (call, rx) = self.call(Answer::handoff)?;
        update.call = Some(call);
        self.send(ActivityCommand::Update(Box::new(update)))?;
        // The driver rebuilds the detector on the reopened leg when the swap lands.
        if let Some(config) = filter {
            self.send(ActivityCommand::SetSpeakerFilter(config))?;
        }
        Ok(Reply::new(async move { matches!(rx.await, Ok(Ok(()))) }))
    }

    /// `await task` inside a tool body or a hook (`AgentTask.__await_impl`); the
    /// awaiter parks on `handle.joined()` while the driver pauses the parent frame, runs the task's
    /// frame, and re-installs the parent once the handle settles.
    ///
    /// Frame keywords are `update_agent`'s; `speech_id` is the awaiting speech (`None` from
    /// `on_enter`/`on_exit`); `parent` is the awaiter's agent id, naming the slot the task takes.
    #[host(method(kw_only = "frame", defaults(preserve_function_call_history = false)))]
    pub fn enter_task(
        &self,
        handle: &crate::workflows::HostTaskHandle,
        #[host(kwargs)] frame: UpdateAgentArgs,
        speech_id: Option<u64>,
        preserve_function_call_history: bool,
        parent: Option<String>,
        origin: Option<&str>,
    ) -> Result<(), CommandRefused> {
        let origin = origin_of(origin)?;
        let (frame, _) = self.frame(frame);
        let entry = TaskEntry {
            handle: handle.clone(),
            frame,
            speech: speech_id.map(SpeechId),
            preserve_function_call_history,
            parent,
        };
        Ok(self.send_from(origin, TaskCommand::Enter(Box::new(entry)))?)
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
        self.send(ActivityCommand::SetAudioOutput(sink.map(HostAudioOut)))
    }

    /// `session.output.transcription = ..` after `start()`: the next reply forwards into the new
    /// leg; one in flight keeps the leg it started with.
    #[host(method(send = ActivityCommand::SetTranscriptionOutput(sink.map(HostTextOut))))]
    pub fn set_transcription_output(
        &self,
        #[host(required)] sink: Option<Arc<dyn TextOutput>>,
    ) -> Result<(), SessionRefused>;

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
    /// the driver merges into its session layer and re-resolves. An update that says nothing is a
    /// no-op (`keyterms`/`expressive` never reach the driver).
    #[host(method)]
    pub fn update_turn_options(
        &self,
        update: crate::agent_activity::options::UpdateKwargs,
    ) -> Result<(), SessionRefused> {
        let (patch, diagnostics) = crate::agent_activity::options::update_patch(&update);
        if let Some(events) = &self.host.events {
            for d in diagnostics {
                events.event(&HostWire::Diagnostic(d.wire()));
            }
        }
        if patch == crate::agent_activity::options::SessionPatch::default() {
            return Ok(());
        }
        self.configure(&patch);
        self.send(TurnCommand::Configure(patch))
    }

    /// The outbound-SIP re-evaluation (`_on_room_io_participant_linked`): the warmup leaf of the
    /// session layer, re-resolved by the driver.
    #[host(method)]
    pub fn set_aec_warmup(
        &self,
        #[host(required)] duration: Option<f64>,
    ) -> Result<(), SessionRefused> {
        let patch = crate::agent_activity::options::aec_warmup_patch(duration);
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

    /// `AgentTask.complete(result)` from inside `speech`'s tool: the output the run
    /// ending on that speech reads. `error` is the `isinstance(result, Exception)` arm.
    #[host(method(kw_only = "value"))]
    pub fn set_run_output(
        &self,
        speech: &crate::speech_handle::HostSpeech,
        value: Option<lk_agents_abi::HostRef>,
        error: Option<lk_agents_abi::HostRef>,
    ) -> Result<(), SessionRefused> {
        // A handle the loop has not created yet has no speech: the caller is outside any speech.
        let Some(handle) = speech.peek() else { return Ok(()) };
        let output = match (value, error) {
            (_, Some(err)) => crate::run_result::RunOutput::Error(err),
            (Some(v), None) => crate::run_result::RunOutput::Value(v),
            (None, None) => return Ok(()),
        };
        self.send(SpeechCommand::SetRunOutput { speech: handle.id(), output })
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
    ) -> Result<crate::speech_handle::HostSpeech, CommandRefused> {
        let origin = origin_of(origin)?;
        // An empty toolset would tell the model about no tool at all; `None` keeps the frame's.
        let toolset =
            tools.map(|specs| Arc::new(host_toolset(Some(specs), tools_host, tool_runner)));
        let (call, rx) = self.call(Answer::speech)?;
        let speech =
            crate::speech_handle::HostSpeech::pending(self.api()?, rx, allow_interruptions);
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
    /// config, classifier LLM, tools and optional STT. The verdict comes back as `amd_result`; a
    /// second attach is refused.
    #[host(method(name = "attach_amd"))]
    pub fn host_attach_amd(
        &self,
        cfg: lk_agents_abi::voice::amd::AmdConfigKwargs,
        llm: Arc<dyn Llm>,
        tools: Option<lk_agents_abi::HostRef>,
        stt: Option<Arc<dyn Stt>>,
    ) -> Result<(), CommandRefused> {
        let config = crate::amd::classifier::AmdConfig::resolve(&cfg)
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let setup = crate::amd::detector::AmdSetup {
            config,
            prompt: cfg.prompt.unwrap_or_else(|| lk_agents_abi::voice::amd::AMD_PROMPT.to_owned()),
            llm: llm_leg(llm),
            tools_host: tools,
            stt: stt.map(stt_leg),
        };
        Ok(self.attach_amd(setup)?)
    }

    /// `_aclose_impl`'s close: `Command::Close { reason, drain, error }`. Resolves once the loop has
    /// returned and the host loop has received `close`, so a `close` handler runs before the
    /// caller's next line. `has_error` says whether an error close carries one (the object stays
    /// with the host). An unknown `reason` is a `ValueError`. Idempotent. The room io's teardown is
    /// `close_io`, called after the `close` handlers.
    #[host(method(kw_only, defaults(reason = "user_initiated", drain = false, has_error = false)))]
    pub fn aclose(
        &self,
        reason: &str,
        drain: bool,
        has_error: bool,
    ) -> Result<Reply<()>, UnknownCloseReason> {
        let reason = lk_agents_abi::voice::events::CloseReason::from_wire(reason)
            .ok_or_else(|| UnknownCloseReason(reason.to_owned()))?;
        // The error object is the host's; the driver carries only the fact of it.
        let error = has_error.then(|| {
            ProviderError::Local(lk_agents_abi::errors::ApiError::new("the host's close error"))
        });
        // A session never started has nothing to wait for, and its turn-detector stream stays the
        // host's to close.
        let was_running = self.api().is_ok();
        let closing = self.close(reason, drain, error);
        let clock = self.time.clock();
        let eot = self.host.eot.clone();
        Ok(Reply::new(async move {
            closing.await;
            if let (true, Some(eot)) = (was_running, eot) {
                let _ = lk_agents_abi::timeout(&*clock, CLOSE_TIMEOUT, eot.aclose()).await;
            }
        }))
    }

    /// The room io's teardown, after the host's `close` handlers ran (the reference's "close room
    /// io after close event is emitted"). Bounded, so a wedged host generator cannot hang shutdown.
    #[host(method(name = "close_io"))]
    pub fn host_close_io(&self) -> Reply<()> {
        Reply::new(self.close_io())
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

    /// The driver's decision log, one `repr` per step.
    #[host(method(name = "steps"))]
    pub fn host_steps(&self) -> Vec<String> {
        self.steps().iter().map(|s| format!("{s:?}")).collect()
    }

    /// `on_enter` / `on_exit` returned for hook call `id` (a `hook_call` event's `id`).
    #[host(method(send = Command::HookDone { id: HookId(id), outcome: HookOutcome::Returned }))]
    pub fn hook_returned(&self, id: u64) -> Result<(), SessionRefused>;

    /// The hook `id` raised or could not be run; the sequence goes on as after `log_exceptions`.
    #[host(method(send = Command::HookDone { id: HookId(id), outcome: HookOutcome::Failed }))]
    pub fn hook_failed(&self, id: u64) -> Result<(), SessionRefused>;

    /// `render_tool_reply` answered for hook call `id`: the deferred tool reply's instructions, as
    /// the host's template rendered them.
    #[host(method(send = Command::HookDone { id: HookId(id), outcome: HookOutcome::Rendered(text) }))]
    pub fn hook_rendered(&self, id: u64, text: String) -> Result<(), SessionRefused>;

    /// `function_tools_executed` handled for hook call `id`: each output's `reply_required`, in
    /// order, and `handoff_required`, as the event's handlers left them (`cancel_tool_reply()`,
    /// `cancel_agent_handoff()`).
    #[host(method(send = Command::HookDone {
        id: HookId(id),
        outcome: HookOutcome::FunctionToolsExecuted { reply_required, handoff_required },
    }))]
    pub fn hook_tools_executed(
        &self,
        id: u64,
        reply_required: Vec<bool>,
        handoff_required: bool,
    ) -> Result<(), SessionRefused>;

    /// `filler_source` answered for hook call `id`: the callable's text for the driver to say,
    /// `spoke` when it scheduled a speech itself, neither when it declined this fire.
    #[host(method(
        kw_only = "text",
        defaults(spoke = false),
        send = Command::HookDone { id: HookId(id), outcome: HookOutcome::Filler { text, spoke } }
    ))]
    pub fn hook_filler(
        &self,
        id: u64,
        text: Option<String>,
        spoke: bool,
    ) -> Result<(), SessionRefused>;

    /// `Agent.on_user_turn_completed` returned for hook call `id`, with its edits to the turn's
    /// request. A late answer is journalled and dropped.
    #[host(method(kw_only = "instructions", defaults(stop = false)))]
    pub fn hook_turn_done(
        &self,
        id: u64,
        instructions: Option<String>,
        items: Option<Vec<lk_agents_abi::ChatItem>>,
        user_text: Option<String>,
        stop: bool,
    ) -> Result<(), SessionRefused> {
        use crate::agent::TurnEdit;
        let mut edits = Vec::new();
        if stop {
            edits.push(TurnEdit::StopResponse);
        }
        if let Some(text) = instructions {
            edits.push(TurnEdit::UpdateInstructions(text));
        }
        if let Some(text) = user_text {
            edits.push(TurnEdit::UpdateUserMessage(text));
        }
        if let Some(items) = items {
            edits.push(TurnEdit::ReplaceItems(items));
        }
        self.send(Command::HookDone { id: HookId(id), outcome: HookOutcome::Turn(edits) })
    }

    /// `RunContext.with_filler(..)` entered for tool `call_id` of `speech_id`: after `delay` s idle
    /// say `text` (or ask the callable), then repeat every `interval`.
    #[host(method(kw_only = "delay"))]
    #[allow(clippy::too_many_arguments)]
    pub fn filler_start(
        &self,
        id: u64,
        call_id: String,
        speech_id: &str,
        delay: f64,
        interval: Option<f64>,
        max_steps: Option<u32>,
        text: Option<String>,
    ) -> Result<(), CommandRefused> {
        use crate::filler_scheduler::FillerError;
        let invalid = |e: FillerError| CommandRefused::Invalid(e.message().to_owned());
        if delay < 0.0 || delay.is_nan() {
            return Err(invalid(FillerError::NegativeDelay));
        }
        if interval.is_some_and(|i| i < 0.0 || i.is_nan()) {
            return Err(invalid(FillerError::NegativeInterval));
        }
        let speech = speech_id
            .parse::<crate::speech_handle::SpeechWireId>()
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?
            .0;
        let source = match text {
            Some(text) => FillerSource::Text(text),
            None => FillerSource::Host,
        };
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
        Ok(self.send_from(origin, cmd)?)
    }

    /// `with_filler`'s exit.
    #[host(method(send = SpeechCommand::FillerStop { id: FillerId(id) }))]
    pub fn filler_stop(&self, id: u64) -> Result<(), SessionRefused>;

    /// `AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it,
    /// including a realtime session's server-side buffer.
    #[host(method(send = TurnCommand::Clear))]
    pub fn clear_user_turn(&self) -> Result<(), SessionRefused>;

    /// `Agent.update_instructions()` on the agent that is already driving. Changes only the
    /// prompt; `update_agent` would replace the whole frame.
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

    /// `RunContext.update()` after the first, or a detached tool's return. `kind` is `"progress"`
    /// or `"final"`.
    #[host(method(kw_only = "message"))]
    pub fn tool_update(
        &self,
        call_id: String,
        kind: &str,
        message: Option<String>,
        raw: Option<String>,
        answer: Option<crate::tool_executor::ToolAnswer>,
    ) -> Result<(), CommandRefused> {
        let missing = |what: &str| CommandRefused::Invalid(what.to_owned());
        let update = match kind {
            "progress" => ToolUpdate::Progress {
                message: message.ok_or_else(|| missing("a progress update carries `message`"))?,
                raw: raw.ok_or_else(|| missing("a progress update carries `raw`"))?,
            },
            "final" => ToolUpdate::Final {
                answer: answer.ok_or_else(|| missing("a final update carries `answer`"))?,
            },
            other => {
                return Err(CommandRefused::Invalid(format!(
                    "tool_update kind must be \"progress\" or \"final\", got {other:?}"
                )));
            }
        };
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

    /// `AgentSession.drain()`: the driving frame's `on_exit`, then scheduling paused until no
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

    /// `GetDtmfTask.on_enter`'s subscriptions, run by the driver. Answers once, with the outcome
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
        use crate::workflows::builtins::dtmf::{DtmfCollector, DtmfEvent, DtmfOutcome};
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

    /// `GetDtmfTask.dtmf_reply_running()`: whether the last collection is inside its debounced
    /// reply (for the confirm arm, through the read-back speech).
    #[host(method(pid = false))]
    pub fn dtmf_reply_running(&self) -> bool {
        self.dtmf_reply
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .is_some_and(|guard| guard.load(std::sync::atomic::Ordering::SeqCst))
    }

    /// `sip_dtmf_received`: the raw `ev.digit`, for the running collection.
    #[host(method(send = DtmfCommand::Digit { digit }))]
    pub fn dtmf_digit(&self, digit: String) -> Result<(), SessionRefused>;

    /// `GetDtmfTask.on_exit`: the collection ends, an outcome still owed resolves `stopped`. A
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

    /// `AgentSession.run(user_input=, output_type=, output_options=)`, answered on the stream.
    #[host(method(kw_only))]
    pub fn run(
        &self,
        user_input: Option<String>,
        expects_output: bool,
        max_retries: u32,
        retry_instructions: String,
        origin: Option<&str>,
    ) -> Result<(), CommandRefused> {
        let spec = crate::run_result::RunSpec {
            user_input,
            expects_output,
            max_retries,
            retry_instructions,
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
    ) -> Result<crate::speech_handle::HostSpeech, CommandRefused> {
        self.speak(SayText::Text(text), allow_interruptions, add_to_chat_ctx, origin)
    }

    /// `AgentSession.say(AsyncIterable[str])`: answers the pending speech and the chunk sender;
    /// dropping the sender ends the speech.
    #[host(method(kw_only, defaults(add_to_chat_ctx = true)))]
    pub fn say_stream(
        &self,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<(crate::speech_handle::HostSpeech, TextPusher), CommandRefused> {
        // Backpressure against a runaway producer, not a buffer.
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let speech =
            self.speak(SayText::Stream(rx), allow_interruptions, add_to_chat_ctx, origin)?;
        Ok((speech, TextPusher(std::sync::Mutex::new(Some(tx)))))
    }

    /// `AgentSession.say(text, audio=..)`: pre-synthesised frames instead of TTS, pushed like
    /// [`say_stream`](Self::say_stream)'s chunks; `text` is still committed.
    #[host(method(kw_only = "allow_interruptions", defaults(add_to_chat_ctx = true)))]
    pub fn say_audio(
        &self,
        text: String,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<&str>,
    ) -> Result<(crate::speech_handle::HostSpeech, FramePusher), CommandRefused> {
        // Deeper than the text channel: at 20 ms a frame, 64 would be 1.3 s of audio.
        let (tx, rx) = tokio::sync::mpsc::channel(512);
        let speech =
            self.speak(SayText::Audio { text, rx }, allow_interruptions, add_to_chat_ctx, origin)?;
        Ok((speech, FramePusher(std::sync::Mutex::new(Some(tx)))))
    }

    /// A metrics record only a plugin can measure, emitted as `metrics_collected`. Engine-computed
    /// kinds are refused (they would count twice).
    #[host(method)]
    pub fn host_metrics(&self, record: HostMetricsRecord) -> Result<(), CommandRefused> {
        let HostMetricsRecord(record) = record;
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

    /// `_AMDClassifier.start_detection_timer()`: the overall detection budget, armed (or moved)
    /// from now: at *answer*, not at dial, so pre-answer ringback cannot burn it.
    #[host(method)]
    pub fn amd_start_detection_timer(&self) -> Result<(), SessionRefused> {
        use crate::amd::classifier::{In, Start};
        self.amd_input(In::Start(Start::DetectionTimer))
    }

    /// `_AMDClassifier.start_listening()`: the input gate opens and the no-speech timer is armed.
    #[host(method)]
    pub fn amd_start_listening(&self) -> Result<(), SessionRefused> {
        use crate::amd::classifier::{In, Start};
        self.amd_input(In::Start(Start::Listening))
    }

    /// `AMD.push_audio(frame)`: a frame for AMD's own STT while it listens; dropped otherwise.
    #[host(method)]
    pub fn amd_push_audio(&self, frame: lk_agents_abi::AudioFrame) -> Result<(), SessionRefused> {
        self.running(|r| {
            if let Some(handle) = &r.amd {
                let _ = handle.push_audio(frame);
            }
            Ok(())
        })
    }

    /// `_AMDClassifier.settle(category, reason)`: a fallback verdict from the host (e.g.
    /// `uncertain`, `participant_missing`). Works before listening ever opened.
    #[host(method)]
    pub fn amd_settle(&self, category: &str, reason: &str) -> Result<(), CommandRefused> {
        use crate::amd::classifier::{In, Verdict};
        use lk_agents_abi::voice::amd::{AmdCategory, AmdReason};
        let category = AmdCategory::from_wire(category)
            .ok_or_else(|| CommandRefused::Invalid(format!("unknown AMD category {category:?}")))?;
        let reason = AmdReason::from_wire(reason)
            .ok_or_else(|| CommandRefused::Invalid(format!("unknown AMD reason {reason:?}")))?;
        Ok(self.amd_input(In::Verdict(Verdict::Settle {
            category,
            reason,
            speech_duration: None,
        }))?)
    }

    /// `_AMDClassifier.end_input()`: no more transcripts, the classification in flight cancelled.
    /// The system does this itself at release; a host may do it earlier.
    #[host(method)]
    pub fn amd_end_input(&self) -> Result<(), SessionRefused> {
        self.amd_input(crate::amd::classifier::In::EndInput)
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

    /// `session.keyterms`: the list the recogniser was last handed (static + confirmed), read off
    /// the journal's last `KeytermsPushed`; `None` before any push.
    #[host(method)]
    pub fn keyterms(&self) -> Option<Vec<String>> {
        match &*self.lock() {
            Stage::Running(r) => {
                r.trace.lock().unwrap_or_else(|p| p.into_inner()).steps().iter().rev().find_map(
                    |s| match s {
                        Step::Detector(crate::trace::DetectorStep::KeytermsPushed { terms }) => {
                            Some(terms.clone())
                        }
                        _ => None,
                    },
                )
            }
            _ => None,
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
