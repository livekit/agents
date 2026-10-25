//! GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`); see `napi.rs` in `lk-agents-codegen`.
//!
//! The Node twin of `lk-agents-py/src/generated`: 284 core items bound (120 functions and consts, 78 classes, 85 records, 1 proxies).
//!
//! Not bound, and why -- each is a gap in the generator or in a declaration, not in this crate:
//!
//! - `lk_agents_abi::tts::markup::summarize_plan_of`: parameter `items`: Vec(Declared { name: "ChatItem", by_ref: false }) borrows inside a container
//! - `lk_agents_voice::keyterm_detection::resolve_keyterm_detection`: field `llm`: `DetectionLlm` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_voice::turn::migrate_turn_handling_for_host`: parameter `turn_detection`: `TurnDetectionArg` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_activity::options::resolve_recording_options_for_host`: parameter `record`: `RecordingRequest` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_activity::options::resolve_interruption_detector_for_host`: parameter `agent_mode`: `InterruptionMode` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_session::plan_session_start`: parameter `requested`: `RecordingRequest` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_abi::gen_ai::chat_ctx_gen_ai_events`: parameter `items`: Vec(Declared { name: "ChatItem", by_ref: false }) borrows inside a container
//! - `lk_agents_abi::tts::markup::steering_instructions_of`: field `nonverbal_sounds`: `NonverbalSounds` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_abi::compute_chat_ctx_diff_of`: parameter `old`: Vec(Declared { name: "ChatItem", by_ref: false }) borrows inside a container
//! - `lk_agents_abi::llm::provider_format::host::to_provider_format`: parameter `items`: Vec(Declared { name: "ChatItem", by_ref: false }) borrows inside a container
//! - `lk_agents_abi::tokenize::search::keyword_search`: parameter `docs`: Slice(Declared { name: "Document", by_ref: false }) borrows inside a container
//! - `lk_agents_abi::adapter_stream`: parameter `provider`: not yet: `Arc<dyn Tts>` (a host object behind a proxy)
//! - `lk_agents_abi::stt::stream_adapter::capabilities`: field `aligned_transcript`: `AlignedTranscript` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_abi::stt::stream_adapter::recognized`: parameter `event`: `SpeechEvent` cannot be built from JS: `kind`: a declared enum is read from JS only as a parameter (it has no `FromStr` to rely on)
//! - `lk_agents_abi::types::retry::decide_for_host`: parameter `kind`: not yet: `RequestKind` is an open string (#[host(string)])
//! - `lk_agents_abi::telemetry::provider::stt_batch_for_host`: parameter `event`: `SpeechEvent` cannot be built from JS: `kind`: a declared enum is read from JS only as a parameter (it has no `FromStr` to rely on)
//! - `lk_agents_abi::telemetry::provider::stt_stream_for_host`: parameter `event`: `SpeechEvent` cannot be built from JS: `kind`: a declared enum is read from JS only as a parameter (it has no `FromStr` to rely on)
//! - `lk_agents_inference::mcp::resolve_tool_options`: field `flags`: `ToolFlag` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_voice::agent_activity::options::HostOptions::new`: field `mode`: `EndpointingMode` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_voice::agent_activity::options::HostOptions::update`: field `mode`: `EndpointingMode` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_voice::agent_activity::options::HostOptions::resolved`: field `detector`: `lk_agents_abi::voice::session_options::DetectorKind` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_voice::workflows::HostTaskHandle::host_complete`: parameter `result`: not yet: a `HostRef` (a host object in the side table)
//! - `lk_agents_voice::workflows::HostTaskHandle::host_joined`: return: not yet: a `HostRef` (a host object in the side table)
//! - `lk_agents_voice::workflows::HostTaskHandle::host_result`: return: not yet: a `HostRef` (a host object in the side table)
//! - `lk_agents_voice::workflows::HostTaskHandle::host_repr`: not yet: `__repr__` is a Python protocol method
//! - `lk_agents_voice::speech_handle::HostSpeech::chat_items`: return: not yet: returning an exposed class
//! - `lk_agents_voice::speech_handle::HostSpeech::exception`: return: `SpeechException` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::speech_handle::HostSpeech::interrupt`: parameter `spawner`: not yet: `Arc<dyn Spawner>` (a host object behind a proxy)
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::email`: parameter `instructions`: `Instructions` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::address`: parameter `instructions`: `Instructions` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::phone_number`: parameter `tools`: `ToolSpec` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::name`: parameter `tools`: `ToolSpec` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::dob`: parameter `tools`: `ToolSpec` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::card_number`: return: `Self` outside a constructor
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::security_code`: return: `Self` outside a constructor
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::expiration_date`: return: `Self` outside a constructor
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::dtmf`: return: `Self` outside a constructor
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::tools`: no core path: the IR names none (give the declaration #[node(from = "..")] or a binding target)
//! - `lk_agents_voice::workflows::builtins::HostWorkflow::call`: return: not yet: returning an exposed class
//! - `lk_agents_voice::workflows::builtins::WorkflowReply::result`: return: `CoreObject` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::workflows::builtins::WorkflowReply::tools`: no core path: the IR names none (give the declaration #[node(from = "..")] or a binding target)
//! - `lk_agents_voice::workflows::builtins::WorkflowReply::host_repr`: not yet: `__repr__` is a Python protocol method
//! - `lk_agents_voice::workflows::builtins::dtmf::DtmfResult::host_repr`: not yet: `__repr__` is a Python protocol method
//! - `lk_agents_voice::workflows::builtins::dtmf::DtmfResult::from_dtmf_inputs`: return: `Self` outside a constructor
//! - `lk_agents_voice::workflows::builtins::warm_transfer::HostWarmTransfer::new`: parameter `instructions`: `Instructions` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::workflows::builtins::warm_transfer::HostWarmTransfer::tools`: no core path: the IR names none (give the declaration #[node(from = "..")] or a binding target)
//! - `lk_agents_abi::voice::workflows::GetDOBResult::for_host`: parameter `date_of_birth`: `Date` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_abi::voice::workflows::GetDOBResult::host_date_of_birth`: return: `Date` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_abi::voice::workflows::GetDOBResult::host_time_of_birth`: return: `Time` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_abi::voice::workflows::GetDOBResult::host_eq`: not yet: `__eq__` is a Python protocol method
//! - `lk_agents_abi::voice::workflows::GetDOBResult::host_repr`: not yet: `__repr__` is a Python protocol method
//! - `lk_agents_voice::workflows::builtins::task_group::TaskGroupSchedule::host_start`: arm `Run`: no core path: the IR names none (give the declaration #[node(from = "..")] or a binding target)
//! - `lk_agents_voice::workflows::builtins::task_group::TaskGroupSchedule::host_advance`: arm `Run`: no core path: the IR names none (give the declaration #[node(from = "..")] or a binding target)
//! - `lk_agents_voice::workflows::builtins::task_group::TaskGroupSchedule::host_summarized`: arm `Run`: no core path: the IR names none (give the declaration #[node(from = "..")] or a binding target)
//! - `lk_agents_voice::agent_session::Timebase::host_repr`: not yet: `__repr__` is a Python protocol method
//! - `lk_agents_voice::agent_session::FramePusher::push`: parameter `frame`: `AudioFrame` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_session::SessionApi::host_new`: parameter `room`: not yet: `Arc<dyn SessionRoom>` (a host object behind a proxy)
//! - `lk_agents_voice::agent_session::SessionApi::session_clock`: return: not yet: returning an exposed class
//! - `lk_agents_voice::agent_session::SessionApi::video_input`: return: not yet: `Arc<ParticipantVideoInput>` (a shared handle to an exposed class)
//! - `lk_agents_voice::agent_session::SessionApi::audio_output`: return: not yet: `Arc<RoomAudioLeg>` (a shared handle to an exposed class)
//! - `lk_agents_voice::agent_session::SessionApi::transcription_output`: return: not yet: `Arc<RoomTranscriptionLeg>` (a shared handle to an exposed class)
//! - `lk_agents_voice::agent_session::SessionApi::host_attach_recorder`: parameter `recorder`: not yet: `Arc<dyn AudioRecorder>` (a host object behind a proxy)
//! - `lk_agents_voice::agent_session::SessionApi::update_models`: parameter `llm`: not yet: `Arc<dyn Llm>` (a host object behind a proxy)
//! - `lk_agents_voice::agent_session::SessionApi::hook_described`: parameter `frame`: not yet: `UpdateAgentArgs` is a keyword record (spread into keywords in Python; an options object in Node)
//! - `lk_agents_voice::agent_session::SessionApi::update_agent`: parameter `frame`: not yet: `UpdateAgentArgs` is a keyword record (spread into keywords in Python; an options object in Node)
//! - `lk_agents_voice::agent_session::SessionApi::enter_task`: parameter `frame`: not yet: `UpdateAgentArgs` is a keyword record (spread into keywords in Python; an options object in Node)
//! - `lk_agents_voice::agent_session::SessionApi::set_audio_input`: parameter `source`: not yet: `Arc<dyn AudioInput>` (a host object behind a proxy)
//! - `lk_agents_voice::agent_session::SessionApi::set_video_input`: parameter `source`: not yet: `Arc<dyn VideoInput>` (a host object behind a proxy)
//! - `lk_agents_voice::agent_session::SessionApi::set_audio_output`: parameter `sink`: not yet: `Arc<dyn AudioOutput>` (a host object behind a proxy)
//! - `lk_agents_voice::agent_session::SessionApi::set_transcription_output`: parameter `sink`: not yet: `Arc<dyn TextOutput>` (a host object behind a proxy)
//! - `lk_agents_voice::agent_session::SessionApi::update_tools`: parameter `tools`: `ToolSpec` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_session::SessionApi::update_turn_options`: field `mode`: `EndpointingMode` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_voice::agent_session::SessionApi::host_push_video`: parameter `content`: an exposed `ChatContent` is lent, never moved; take `&ChatContent`
//! - `lk_agents_voice::agent_session::SessionApi::set_run_output`: parameter `value`: not yet: a `HostRef` (a host object in the side table)
//! - `lk_agents_voice::agent_session::SessionApi::generate_reply`: parameter `tool_choice`: `ToolChoice` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_session::SessionApi::ivr_start`: parameter `tools`: `ToolSpec` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_session::SessionApi::host_attach_amd`: field `source`: `TranscriptSource` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_voice::agent_session::SessionApi::hook_turn_done`: parameter `items`: Option(Vec(Declared { name: "ChatItem", by_ref: false })) borrows inside a container
//! - `lk_agents_voice::agent_session::SessionApi::update_instructions`: parameter `instructions`: `Instructions` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_session::SessionApi::tool_update`: parameter `answer`: `ToolAnswer` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_session::SessionApi::update_chat_ctx`: parameter `items`: Vec(Declared { name: "ChatItem", by_ref: false }) borrows inside a container
//! - `lk_agents_voice::agent_session::SessionApi::say`: return: not yet: returning an exposed class
//! - `lk_agents_voice::agent_session::SessionApi::say_stream`: return: not yet: returning an exposed class
//! - `lk_agents_voice::agent_session::SessionApi::say_audio`: return: not yet: returning an exposed class
//! - `lk_agents_voice::agent_session::SessionApi::host_metrics`: parameter `record`: `HostMetricsRecord` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::agent_session::SessionApi::amd_push_audio`: parameter `frame`: `AudioFrame` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_media::StreamBuffer::len`: not yet: `__len__` is a Python protocol method
//! - `lk_agents_voice::remote_session::driver::HostRuntime::new`: parameter `clock`: not yet: `Arc<dyn Clock>` (a host object behind a proxy)
//! - `lk_agents_voice::remote_session::driver::HostRuntime::set_port`: parameter `port`: not yet: `Arc<dyn SessionPort>` (a host object behind a proxy)
//! - `lk_agents_voice::remote_session::driver::HostRuntime::send_wire`: arm `FunctionToolsExecuted`: `FunctionToolsExecutedWire` is not declared
//! - `lk_agents_voice::remote_session::driver::HostRuntime::start`: parameter `transport`: not yet: `Arc<dyn SessionTransport>` (a host object behind a proxy)
//! - `lk_agents_voice::remote_session::driver::ClientRuntime::new`: parameter `clock`: not yet: `Arc<dyn Clock>` (a host object behind a proxy)
//! - `lk_agents_voice::remote_session::driver::ClientRuntime::start`: parameter `transport`: not yet: `Arc<dyn SessionTransport>` (a host object behind a proxy)
//! - `lk_agents_abi::tts::markup::Markup::llm_instructions`: field `nonverbal_sounds`: `NonverbalSounds` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_abi::ChatContent::text`: return: `Self` outside a constructor
//! - `lk_agents_abi::ChatContent::image_url`: return: `Self` outside a constructor
//! - `lk_agents_abi::ChatContent::image_frame`: parameter `frame`: `MediaRef` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_abi::ChatContent::audio`: parameter `frames`: `MediaRef` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_abi::ChatItem::message`: parameter `content`: Vec(Declared { name: "ChatContent", by_ref: false }) borrows inside a container
//! - `lk_agents_abi::ChatItem::function_call`: return: `Self` outside a constructor
//! - `lk_agents_abi::ChatItem::function_call_output`: return: `Self` outside a constructor
//! - `lk_agents_abi::ChatItem::agent_handoff`: return: `Self` outside a constructor
//! - `lk_agents_abi::ChatItem::agent_config_update`: parameter `tools`: not yet: a `HostRef` (a host object in the side table)
//! - `lk_agents_abi::ChatItem::host_tools`: return: not yet: a `HostRef` (a host object in the side table)
//! - `lk_agents_abi::tokenize::search::Bm25::build`: parameter `documents`: Slice(Declared { name: "Document", by_ref: false }) borrows inside a container
//! - `lk_agents_abi::tokenize::search::Bm25::search`: parameter `docs`: Slice(Declared { name: "Document", by_ref: false }) borrows inside a container
//! - `lk_agents_voice::io::CoreAudioSink::capture_frame`: parameter `frame`: `AudioFrame` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_rtc::RoomEventBus`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::ByteStreamSender`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::ByteChunks`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::TextStreamSender`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::TextChunks`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::AudioFrameSender`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::VideoFrameSender`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::PlayoutLedger`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_abi::voice::rtc::AudioStreamOptions::noise_cancellation`: return: not yet: a `HostRef` (a host object in the side table)
//! - `lk_agents_abi::voice::rtc::VideoFrameRef::new`: parameter `frame`: `MediaRef` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_abi::voice::rtc::StreamTextOptions::sender_identity`: return: not yet: `ParticipantIdentity` is an open string (#[host(string)])
//! - `lk_agents_abi::voice::rtc::Transcription::participant_identity`: return: not yet: `ParticipantIdentity` is an open string (#[host(string)])
//! - `lk_agents_abi::voice::rtc::Transcription::track_sid`: return: not yet: `TrackSid` is an open string (#[host(string)])
//! - `lk_agents_voice::room_io::_output::RoomAudioLeg::standalone`: parameter `room`: not yet: `Arc<dyn RtcRoom>` (a host object behind a proxy)
//! - `lk_agents_voice::room_io::_output::RoomAudioLeg::capture_frame`: parameter `frame`: `AudioFrame` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::room_io::_output::RoomAudioLeg::subscribe`: parameter `listener`: not yet: `Arc<dyn PlaybackListener>` (a host object behind a proxy)
//! - `lk_agents_voice::room_io::_output::transcription::RoomTranscriptionLeg::standalone`: parameter `room`: not yet: `Arc<dyn RtcRoom>` (a host object behind a proxy)
//! - `lk_agents_voice::room_io::_output::transcription::RoomTranscriptionLeg::set_participant`: parameter `identity`: not yet: `ParticipantIdentity` is an open string (#[host(string)])
//! - `lk_agents_voice::room_io::rooms::ScriptedRoom::session_clock`: return: not yet: returning an exposed class
//! - `lk_agents_voice::background_audio::Sound::for_host`: parameter `stream`: not yet: `Arc<dyn FrameStream>` (a host object behind a proxy)
//! - `lk_agents_voice::background_audio::BackgroundAudioPlayer::start_for_host`: parameter `participant`: not yet: `Arc<dyn LocalParticipant>` (a host object behind a proxy)
//! - `lk_agents_voice::background_audio::BackgroundAudioPlayer::play_for_host`: parameter `sounds`: Vec(Declared { name: "Sound", by_ref: false }) borrows inside a container
//! - `lk_agents_voice::background_audio::BackgroundAudioPlayer::on_agent_state_for_host`: parameter `sounds`: Vec(Declared { name: "Sound", by_ref: false }) borrows inside a container
//! - `lk_agents_inference::fallback::FallbackLlm::for_host`: parameter `llms`: not yet: `Arc<dyn Llm>` (a host object behind a proxy)
//! - `lk_agents_inference::fallback::FallbackLlm::chat_for_host`: parameter `chat_ctx_host`: not yet: a `HostRef` (a host object in the side table)
//! - `lk_agents_inference::fallback::FallbackStt::for_host`: parameter `stts`: not yet: `Arc<dyn Stt>` (a host object behind a proxy)
//! - `lk_agents_inference::fallback::FallbackStt::merged_capabilities`: field `aligned_transcript`: `AlignedTranscript` has no Node shape (not a declared record, enum or JSON)
//! - `lk_agents_inference::fallback::FallbackStt::recognize_for_host`: parameter `frames`: `AudioFrame` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_inference::fallback::FallbackStt::stream_for_host`: parameter `language`: not yet: `LanguageCode` is an open string (#[host(string)])
//! - `lk_agents_inference::fallback::FallbackTts::for_host`: parameter `ttss`: not yet: `Arc<dyn Tts>` (a host object behind a proxy)
//! - `lk_agents_inference::fallback::FallbackTts::synthesize_for_host`: return: not yet: a stream half returned as a stream class
//! - `lk_agents_inference::fallback::FallbackTts::stream_for_host`: return: not yet: a stream half returned as a stream class
//! - `lk_agents_inference::stt::NativeStt::update_options`: parameter `extra`: `ExtraKwargs` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_inference::stt::NativeStt::stream_for_host`: return: `SttSinkHandle` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_inference::vad::NativeVad::for_host`: parameter `model_factory`: not yet: `Arc<dyn VadModelFactory>` (a host object behind a proxy)
//! - `lk_agents_inference::vad::NativeVad::stream_for_host`: return: `VadSinkHandle` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_inference::tts::NativeTts::update_options`: parameter `extra`: `ExtraKwargs` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_inference::tts::NativeTts::stream_for_host`: return: not yet: a stream half returned as a stream class
//! - `lk_agents_inference::realtime::NativeRealtime::for_host`: parameter `dialect`: not yet: `Dialect` is an open string (#[host(string)])
//! - `lk_agents_voice::room_io::_input::ParticipantVideoInput::reading`: return: not yet: `TrackSid` is an open string (#[host(string)])
//! - `lk_agents_voice::room_io::_input::ParticipantVideoInput::next_host_frame`: return: `MediaRef` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_voice::transcription::synchronizer::TranscriptSynchronizer::for_host`: parameter `next_in_chain_audio`: not yet: `Arc<dyn AudioOutput>` (a host object behind a proxy)
//! - `lk_agents_voice::transcription::synchronizer::TranscriptSynchronizer::audio_leg`: return: not yet: returning an exposed class
//! - `lk_agents_voice::transcription::synchronizer::TranscriptSynchronizer::text_leg`: return: not yet: returning an exposed class
//! - `lk_agents_inference::mcp::session::McpSession::attach_client`: parameter `client`: not yet: `Arc<dyn McpClient>` (a host object behind a proxy)
//! - `lk_agents_inference::mcp::session::McpSession::connect_host`: parameter `transport`: not yet: `Arc<dyn McpTransport>` (a host object behind a proxy)
//! - `lk_agents_inference::mcp::session::McpSession::call_tool`: parameter `progress`: not yet: `Arc<dyn McpProgress>` (a host object behind a proxy)
//! - `FallbackSttStream`: input: `AudioFrame` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `InferenceSttStream`: input: `AudioFrame` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `InferenceVadStream`: output: `VadOutput` is not a #[derive(Host)] type or a #[host(wire_enum)]
//! - `lk_agents_inference::llm::LlmClientConfig`: not yet: a #[host(client)] constructor lands on a hand-written pyclass
//! - `lk_agents_inference::tts::TtsClientConfig`: not yet: a #[host(client)] constructor lands on a hand-written pyclass
//! - `lk_agents_voice::tool_executor::HostToolRunner`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::agent_session::SessionEvents`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::agent_session::HostRoom`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::agent_session::NodeFactory`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::remote_session::SessionPort`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::remote_session::SessionTransport`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::remote_session::ConsoleSink`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::remote_session::ClientSink`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::io::PlaybackListener`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::io::AudioInput`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::io::VideoInput`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::io::AudioOutput`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::io::TextOutput`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::io::TextNode`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_abi::Vad`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_rtc::RtcRoom`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::LocalParticipant`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::TextStreamWriter`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::AudioSource`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::RemoteParticipant`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_rtc::TrackPublication`: `lk_agents_rtc` is not linked by lk-agents-node
//! - `lk_agents_voice::background_audio::FrameStream`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::background_audio::RandomDraw`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_abi::Tts`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_abi::Stt`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_abi::Llm`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_abi::EotDetector`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_inference::fallback::ChainObserver`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_abi::llm::realtime::HostRealtime`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_abi::llm::realtime::HostRealtimeSession`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_voice::transcription::synchronizer::Hyphenator`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_inference::mcp::session::McpTransport`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_inference::mcp::session::McpClient`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `lk_agents_inference::mcp::session::McpProgress`: not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)
//! - `VadEvent (record VADEvent)`: field `frames`: `MediaRef` has no Node shape (not a declared record, enum or JSON)
//! - `AudioInputConfig (record AudioInputOptions)`: field `noise_cancellation`: `lk_agents_abi::HostRef` has no Node shape (not a declared record, enum or JSON)
//! - `NoiseCancellationParams (record NoiseCancellationParams)`: field `participant`: `crate::HostRef` has no Node shape (not a declared record, enum or JSON)
//! - `TextInputOptions (record TextInputOptions)`: field `text_input_cb`: `crate::HostRef` has no Node shape (not a declared record, enum or JSON)
//! - `AudioOutputOptions (record AudioOutputOptions)`: field `track_publish_options`: `crate::HostRef` has no Node shape (not a declared record, enum or JSON)
//! - `RoomInputOptions (record RoomInputOptions)`: field `noise_cancellation`: `crate::HostRef` has no Node shape (not a declared record, enum or JSON)
//! - `RoomOutputOptions (record RoomOutputOptions)`: field `audio_publish_options`: `crate::HostRef` has no Node shape (not a declared record, enum or JSON)
//! - `TextInput (record TextInputEvent)`: field `participant`: `ParticipantIdentity` has no Node shape (not a declared record, enum or JSON)
//! - `GetDOBResult (record GetDOBResult)`: field `date_of_birth`: `Date` has no Node shape (not a declared record, enum or JSON)

pub mod records;
pub mod functions;
pub mod objects;
pub mod errors;
pub mod proxies;
pub mod streams;
