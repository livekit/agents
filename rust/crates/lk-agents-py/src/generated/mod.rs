//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The declared classes' pyo3 adapters, one module per ABI subject.

use pyo3::prelude::*;

pub mod a2a;
pub mod accessors;
pub mod agent;
pub mod cli;
pub mod errors;
pub mod evals;
pub mod implementations;
pub mod inference;
pub mod invoke;
pub mod llm;
pub mod media;
pub mod metrics;
pub mod native_enums;
pub mod recorder;
pub mod records;
pub mod storage;
pub mod stt;
pub mod telemetry;
pub mod text_session;
pub mod tokenize;
pub mod tts;
pub mod turn;
pub mod types;
pub mod utils;
pub mod vad;
pub mod voice;
pub mod worker;

/// The implementation of each declared class, selected from its concrete Rust impl.
pub trait Implementations: 'static {
    type Credentials: ::lk_agents_abi::a2a::auth::Credentials;
    type AuthPermissions: ::lk_agents_abi::a2a::auth::AuthPermissions;
    type EndpointCard: ::lk_agents_abi::a2a::card::EndpointCard;
    type MessageContent: ::lk_agents_abi::a2a::client::MessageContent;
    type Reply: ::lk_agents_abi::a2a::client::Reply;
    type AgentClient: ::lk_agents_abi::a2a::client::AgentClient<Reply = Self::Reply>;
    type A2ADoor: ::lk_agents_abi::a2a::door::A2ADoor<Response = Self::A2AResponse>;
    type A2AResponse: ::lk_agents_abi::a2a::door::A2AResponse;
    type RemoteInteraction: ::lk_agents_abi::a2a::remote::RemoteInteraction;
    type RemoteConfiguration: ::lk_agents_abi::a2a::remote::RemoteConfiguration;
    type Turn: ::lk_agents_abi::a2a::turn::Turn<Thread = Self::Thread>;
    type HandlerOutcomes: ::lk_agents_abi::a2a::turn::HandlerOutcomes;
    type VoiceAgent: ::lk_agents_abi::agent::VoiceAgent;
    type RunEvents: ::lk_agents_abi::agent::assertions::RunEvents;
    type AgentConfig: ::lk_agents_abi::agent::authoring::AgentConfig;
    type RunResources: ::lk_agents_abi::agent::resources::RunResources<
            Namespace = Self::Storage,
            Activation = Self::ContextActivation,
        >;
    type AgentRunner: ::lk_agents_abi::agent::run::AgentRunner<
            Stream = Self::RunStream,
            Resources = Self::RunResources,
        >;
    type RunStream: ::lk_agents_abi::agent::stream::RunStream<
            Events = Self::StreamEvents,
            Text = Self::StreamText,
        >;
    type StreamEvents: ::lk_agents_abi::agent::stream::StreamEvents;
    type StreamText: ::lk_agents_abi::agent::stream::StreamText;
    type Cli: ::lk_agents_abi::cli::Cli;
    type CliShutdown: ::lk_agents_abi::cli::CliShutdown;
    type ApiErrorFunctions: ::lk_agents_abi::errors::ApiErrorFunctions;
    type JudgeRunner: ::lk_agents_abi::evals::JudgeRunner;
    type Evaluations: ::lk_agents_abi::evals::Evaluations;
    type InferenceCredentials: ::lk_agents_abi::inference::InferenceCredentials;
    type InferenceModels: ::lk_agents_abi::inference::InferenceModels;
    type InferenceAvatar: ::lk_agents_abi::inference::avatar_session::InferenceAvatar;
    type InferenceTts: ::lk_agents_abi::inference::tts_client::InferenceTts<Stream = Self::TtsStream>;
    type Logging: ::lk_agents_abi::invoke::Logging;
    type RemoteChatContext: ::lk_agents_abi::llm::chat_context::RemoteChatContext;
    type ChatContextPlans: ::lk_agents_abi::llm::chat_context::ChatContextPlans;
    type ChatContextDiffs: ::lk_agents_abi::llm::chat_context::ChatContextDiffs;
    type FixedGate: ::lk_agents_abi::llm::duplex::FixedGate;
    type AdaptiveNoiseGate: ::lk_agents_abi::llm::duplex::AdaptiveNoiseGate;
    type DuplexConfiguration: ::lk_agents_abi::llm::duplex::DuplexConfiguration;
    type DuplexRealtimeModel: ::lk_agents_abi::llm::duplex::DuplexRealtimeModel;
    type FallbackLlm: ::lk_agents_abi::llm::fallback::FallbackLlm;
    type FallbackTts: ::lk_agents_abi::llm::fallback::FallbackTts<Stream = Self::TtsStream>;
    type RealtimeFallbackModel: ::lk_agents_abi::llm::fallback::RealtimeFallbackModel;
    type TtsTokenLog: ::lk_agents_abi::llm::fallback::TtsTokenLog;
    type InferenceLlm: ::lk_agents_abi::llm::inference::InferenceLlm;
    type ChoiceParser: ::lk_agents_abi::llm::inference::ChoiceParser;
    type CompletionWire: ::lk_agents_abi::llm::inference::CompletionWire;
    type McpSession: ::lk_agents_abi::llm::mcp::McpSession;
    type McpTools: ::lk_agents_abi::llm::mcp::McpTools;
    type LlmAlgorithms: ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms;
    type HostRealtimeModel: ::lk_agents_abi::llm::realtime::HostRealtimeModel;
    type OpenAIRealtimeModel: ::lk_agents_abi::llm::realtime::OpenAIRealtimeModel;
    type ChatContextDict: ::lk_agents_abi::llm::to_dict::ChatContextDict;
    type ChatRequestView: ::lk_agents_abi::llm::types::ChatRequestView;
    type ToolsetLifetime: ::lk_agents_abi::llm::toolsets::ToolsetLifetime;
    type ToolsetGroup: ::lk_agents_abi::llm::toolsets::ToolsetGroup;
    type ToolsetScopes: ::lk_agents_abi::llm::toolsets::ToolsetScopes;
    type FrameReaderInstall: ::lk_agents_abi::media::images::FrameReaderInstall;
    type ImageFunctions: ::lk_agents_abi::media::images::ImageFunctions;
    type AudioEmitter: ::lk_agents_abi::media::emitter::AudioEmitter;
    type AudioDecoder: ::lk_agents_abi::media::codecs::AudioDecoder;
    type MediaTokenization: ::lk_agents_abi::media::codecs::MediaTokenization;
    type RuntimeLoad: ::lk_agents_abi::metrics::RuntimeLoad;
    type MetricsRecords: ::lk_agents_abi::metrics::MetricsRecords;
    type ProviderMetrics: ::lk_agents_abi::metrics::provider::ProviderMetrics;
    type LlmMetricState: ::lk_agents_abi::metrics::provider::LlmMetricState;
    type TtsMetricState: ::lk_agents_abi::metrics::provider::TtsMetricState;
    type SessionRecorder: ::lk_agents_abi::recorder::SessionRecorder;
    type Storage: ::lk_agents_abi::storage::Storage<
            Thread = Self::Thread,
            ContextActivation = Self::ContextActivation,
            VoiceStore = Self::VoiceStore,
            ThreadServer = Self::ThreadServer,
        >;
    type ContextActivation: ::lk_agents_abi::storage::context::ContextActivation<Transaction = Self::AppTransaction>;
    type AppTransaction: ::lk_agents_abi::storage::context::AppTransaction;
    type Thread: ::lk_agents_abi::storage::handles::Thread<Run = Self::Run>;
    type Run: ::lk_agents_abi::storage::handles::Run;
    type ThreadServer: ::lk_agents_abi::storage::http::ThreadServer;
    type VoiceStore: ::lk_agents_abi::storage::voice::VoiceStore<ContextActivation = Self::ContextActivation>;
    type FallbackStt: ::lk_agents_abi::stt::fallback::FallbackStt;
    type InferenceStt: ::lk_agents_abi::stt::inference::InferenceStt;
    type SttStreamAdapter: ::lk_agents_abi::stt::stream_adapter::SttStreamAdapter;
    type HostSpan: ::lk_agents_abi::telemetry::host::HostSpan;
    type Telemetry: ::lk_agents_abi::telemetry::host::Telemetry;
    type Tagger: ::lk_agents_abi::telemetry::report::Tagger;
    type EventRetention: ::lk_agents_abi::telemetry::report::EventRetention;
    type TextSessionRunner: ::lk_agents_abi::text_session::TextSessionRunner<
            Stream = Self::RunStream,
            Resources = Self::RunResources,
        >;
    type TokenizeFunctions: ::lk_agents_abi::tokenize::bindings::TokenizeFunctions;
    type HostTokenization: ::lk_agents_abi::tokenize::bindings::HostTokenization;
    type WordTokenizer: ::lk_agents_abi::tokenize::bindings::WordTokenizer;
    type SentenceTokenizer: ::lk_agents_abi::tokenize::bindings::SentenceTokenizer;
    type BufferedTokens: ::lk_agents_abi::tokenize::bindings::BufferedTokens;
    type LanguageFunctions: ::lk_agents_abi::tokenize::language::LanguageFunctions;
    type SearchDocument: ::lk_agents_abi::tokenize::search::SearchDocument;
    type SearchIndex: ::lk_agents_abi::tokenize::search::SearchIndex<Document = Self::SearchDocument>;
    type ToolSearch: ::lk_agents_abi::tokenize::search::ToolSearch<Document = Self::SearchDocument>;
    type MarkupFunctions: ::lk_agents_abi::tts::markup::MarkupFunctions;
    type MarkupClass: ::lk_agents_abi::tts::markup::MarkupClass;
    type TranscriptStripper: ::lk_agents_abi::tts::markup::TranscriptStripper;
    type StreamAdapterFunctions: ::lk_agents_abi::StreamAdapterFunctions<Stream = Self::TtsStream>;
    type SentencePacer: ::lk_agents_abi::tts::stream_pacer::SentencePacer;
    type TtsStream: ::lk_agents_abi::tts::streams::TtsStream;
    type SynthesisChecks: ::lk_agents_abi::tts::streams::SynthesisChecks;
    type InferenceEot: ::lk_agents_abi::turn::inference::InferenceEot<Stream = Self::EotStream>;
    type EotStream: ::lk_agents_abi::turn::inference::EotStream;
    type InterruptionDetector: ::lk_agents_abi::turn::inference::InterruptionDetector;
    type KeytermFunctions: ::lk_agents_abi::turn::keyterms::KeytermFunctions;
    type ConnectOptionsFunctions: ::lk_agents_abi::types::connect::ConnectOptionsFunctions;
    type ProviderAttempts: ::lk_agents_abi::types::provider::ProviderAttempts;
    type AudioByteStream: ::lk_agents_abi::utils::byte_stream::AudioByteStream;
    type ConnectionPoolState: ::lk_agents_abi::utils::connection_pool::ConnectionPoolState;
    type ExpFilter: ::lk_agents_abi::utils::filters::ExpFilter;
    type MovingAverage: ::lk_agents_abi::utils::filters::MovingAverage;
    type InferenceVad: ::lk_agents_abi::vad::inference::InferenceVad;
    type AmdFunctions: ::lk_agents_abi::voice::amd::AmdFunctions;
    type DataStreamAudioOutput: ::lk_agents_abi::voice::avatar::DataStreamAudioOutput;
    type DataStreamAudioReceiver: ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver;
    type AvatarRunner: ::lk_agents_abi::voice::avatar::AvatarRunner<Synchronizer = Self::AVSynchronizer>;
    type QueueAudio: ::lk_agents_abi::voice::avatar_queue::QueueAudio;
    type PlayState: ::lk_agents_abi::voice::background_audio::PlayState;
    type BackgroundAudioPlayer: ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer<State = Self::PlayState>;
    type Playback: ::lk_agents_abi::voice::io::Playback;
    type PlaybackReports: ::lk_agents_abi::voice::io::PlaybackReports;
    type AgentNodes: ::lk_agents_abi::voice::nodes::AgentNodes;
    type NodeCalls: ::lk_agents_abi::voice::nodes::NodeCalls;
    type LlmNodeChunks: ::lk_agents_abi::voice::nodes::LlmNodeChunks;
    type TtsNodeFrames: ::lk_agents_abi::voice::nodes::TtsNodeFrames;
    type SttNodeEvents: ::lk_agents_abi::voice::nodes::SttNodeEvents;
    type DefaultNodes: ::lk_agents_abi::voice::nodes::DefaultNodes<
            Chunks = Self::LlmNodeChunks,
            Frames = Self::TtsNodeFrames,
            Events = Self::SttNodeEvents,
        >;
    type RoomEventBus: ::lk_agents_abi::voice::rtc::RoomEventBus;
    type RtcRuntime: ::lk_agents_abi::voice::rtc::events::RtcRuntime;
    type AVSynchronizer: ::lk_agents_abi::voice::rtc::media::AVSynchronizer;
    type PlayoutLedger: ::lk_agents_abi::voice::rtc::media::PlayoutLedger;
    type ByteStreamSender: ::lk_agents_abi::voice::rtc::ports::ByteStreamSender;
    type TextStreamSender: ::lk_agents_abi::voice::rtc::ports::TextStreamSender;
    type RpcHandler: ::lk_agents_abi::voice::rtc::ports::RpcHandler;
    type RoomRpcRouter: ::lk_agents_abi::voice::rtc::ports::RoomRpcRouter;
    type VoiceRun: ::lk_agents_abi::voice::run_view::VoiceRun;
    type SessionOptions: ::lk_agents_abi::voice::session::SessionOptions;
    type VoiceSession: ::lk_agents_abi::voice::session::VoiceSession<
            Options = Self::SessionOptions,
            MicLeg = Self::RoomAudioInput,
            CameraLeg = Self::RoomVideoInput,
            Speech = Self::SpeechHandle,
        >;
    type SessionStart: ::lk_agents_abi::voice::session_effects::SessionStart;
    type InterruptionGateFunctions: ::lk_agents_abi::voice::session_options::InterruptionGateFunctions;
    type SessionPreparation: ::lk_agents_abi::voice::session_owner::SessionPreparation;
    type SessionOwner: ::lk_agents_abi::voice::session_owner::SessionOwner<
            Options = Self::SessionOptions,
            Storage = Self::SessionStorage,
            Session = Self::VoiceSession,
        >;
    type SessionStorage: ::lk_agents_abi::voice::session_storage::SessionStorage<
            Namespace = Self::Storage,
            Activation = Self::ContextActivation,
            VoiceStore = Self::VoiceStore,
        >;
    type SpeechHandle: ::lk_agents_abi::voice::speech_handle::SpeechHandle;
    type WorkflowChecks: ::lk_agents_abi::voice::workflows::WorkflowChecks;
    type DtmfIo: ::lk_agents_abi::voice::workflows::dtmf::DtmfIo;
    type WarmTransfer: ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer;
    type ConsoleIo: ::lk_agents_abi::voice::console::ConsoleIo;
    type RoomAudioInput: ::lk_agents_abi::voice::room_owner::RoomAudioInput;
    type RoomVideoInput: ::lk_agents_abi::voice::room_owner::RoomVideoInput;
    type RoomSession: ::lk_agents_abi::voice::room_owner::RoomSession;
    type RoomRuntime: ::lk_agents_abi::voice::room_owner::RoomRuntime<Room = Self::RoomSession>;
    type JoinTokens: ::lk_agents_abi::worker::JoinTokens;
    type ByteStream: ::lk_agents_abi::worker::http::ByteStream;
    type HttpServerCapacity: ::lk_agents_abi::worker::http::HttpServerCapacity;
    type ServerEnvironment: ::lk_agents_abi::worker::types::ServerEnvironment;
    type JobSupervisor: ::lk_agents_abi::worker::ipc_host::JobSupervisor;
    type IpcClient: ::lk_agents_abi::worker::ipc_host::IpcClient;
    type ProcessPool: ::lk_agents_abi::worker::ipc_host::ProcessPool;
    type JobRun: ::lk_agents_abi::worker::ipc_host::JobRun;
    type ProcessBootstrap: ::lk_agents_abi::worker::ipc_host::ProcessBootstrap;
    type ProcessInitialization: ::lk_agents_abi::worker::ipc_host::ProcessInitialization;
    type ServerConfig: ::lk_agents_abi::worker::server_host::ServerConfig;
    type ServerRuntime: ::lk_agents_abi::worker::server_host::ServerRuntime;
}

/// Adds every declared class to `_core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<a2a::CoreCredentials>()?;
    m.add_function(wrap_pyfunction!(a2a::_core_auth_grant_allows, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_unauthenticated_principal, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_extensions_header, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_agent_card, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_message_text, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_message_data, m)?)?;
    m.add_class::<a2a::Reply>()?;
    m.add_class::<a2a::AgentClient>()?;
    m.add_class::<a2a::CoreA2ADoor>()?;
    m.add_class::<a2a::CoreA2AResponse>()?;
    m.add_class::<a2a::CoreRemoteInteraction>()?;
    m.add_function(wrap_pyfunction!(a2a::_core_delegation_tool, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_remote_token, m)?)?;
    m.add_class::<a2a::Turn>()?;
    m.add_function(wrap_pyfunction!(a2a::normalize_handler_output, m)?)?;
    m.add_class::<agent::VoiceAgent>()?;
    m.add_class::<agent::RunEvents>()?;
    m.add_class::<agent::CoreAgentConfig>()?;
    m.add_class::<agent::CoreRunResources>()?;
    m.add_class::<agent::AgentRunner>()?;
    m.add_class::<agent::AgentRunUpdates>()?;
    m.add_class::<agent::CoreRunStream>()?;
    m.add_class::<agent::CoreStreamEvents>()?;
    m.add_class::<agent::CoreStreamText>()?;
    m.add_function(wrap_pyfunction!(cli::parse_cli, m)?)?;
    m.add_function(wrap_pyfunction!(cli::format_log_record, m)?)?;
    m.add_function(wrap_pyfunction!(cli::dev_channel, m)?)?;
    m.add_class::<cli::Shutdown>()?;
    m.add_function(wrap_pyfunction!(errors::http_error_display, m)?)?;
    m.add_class::<evals::CoreJudgeRunner>()?;
    m.add_function(wrap_pyfunction!(evals::_core_evaluation_score, m)?)?;
    m.add_function(wrap_pyfunction!(evals::_core_judge_tool_schemas, m)?)?;
    m.add_function(wrap_pyfunction!(inference::inference_resolve_api_key, m)?)?;
    m.add_function(wrap_pyfunction!(inference::inference_resolve_api_secret, m)?)?;
    m.add_function(wrap_pyfunction!(inference::inference_quota_usage, m)?)?;
    m.add_function(wrap_pyfunction!(inference::inference_default_url, m)?)?;
    m.add_function(wrap_pyfunction!(inference::local_models_init, m)?)?;
    m.add_function(wrap_pyfunction!(inference::tts_parse_model_string, m)?)?;
    m.add_function(wrap_pyfunction!(inference::tts_has_aligned_transcript, m)?)?;
    m.add_function(wrap_pyfunction!(inference::tts_markup_provider_key, m)?)?;
    m.add_function(wrap_pyfunction!(inference::stt_parse_model_string, m)?)?;
    m.add_function(wrap_pyfunction!(inference::min_reasoning_effort, m)?)?;
    m.add_function(wrap_pyfunction!(inference::amd_auto_select_models, m)?)?;
    m.add_function(wrap_pyfunction!(inference::amd_is_evaluated_model, m)?)?;
    m.add_function(wrap_pyfunction!(inference::llm_unsupported_params, m)?)?;
    m.add_function(wrap_pyfunction!(inference::eot_select, m)?)?;
    m.add_function(wrap_pyfunction!(inference::stt_resolve_vad_for_model, m)?)?;
    m.add_function(wrap_pyfunction!(inference::stt_normalize_fallback_str, m)?)?;
    m.add_function(wrap_pyfunction!(inference::tts_normalize_fallback_str, m)?)?;
    m.add_function(wrap_pyfunction!(inference::stt_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(inference::stt_fallback_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(inference::tts_fallback_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(inference::merge_realtime_capabilities, m)?)?;
    m.add_class::<inference::InferenceAvatar>()?;
    m.add_class::<inference::InferenceTts>()?;
    m.add_function(wrap_pyfunction!(invoke::flush_logs, m)?)?;
    m.add_class::<llm::CoreRemoteChatContext>()?;
    m.add_function(wrap_pyfunction!(llm::chat_ctx_copy_plan, m)?)?;
    m.add_function(wrap_pyfunction!(llm::chat_ctx_truncate_plan, m)?)?;
    m.add_function(wrap_pyfunction!(llm::chat_ctx_merge_plan, m)?)?;
    m.add_function(wrap_pyfunction!(llm::remove_instructions, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_chat_ctx_diff, m)?)?;
    m.add_class::<llm::FixedGate>()?;
    m.add_class::<llm::AdaptiveNoiseGate>()?;
    m.add_function(wrap_pyfunction!(llm::duplex_realtime_capabilities, m)?)?;
    m.add_class::<llm::DuplexRealtimeModel>()?;
    m.add_class::<llm::FallbackLlm>()?;
    m.add_class::<llm::FallbackLlmChunks>()?;
    m.add_class::<llm::FallbackTts>()?;
    m.add_class::<llm::RealtimeFallbackModel>()?;
    m.add_class::<llm::TtsTokenLog>()?;
    m.add_class::<llm::InferenceLlm>()?;
    m.add_class::<llm::InferenceLlmChunks>()?;
    m.add_class::<llm::LlmChoiceParser>()?;
    m.add_function(wrap_pyfunction!(llm::llm_usage_chunk, m)?)?;
    m.add_class::<llm::McpSession>()?;
    m.add_function(wrap_pyfunction!(llm::resolve_mcp_tool_options, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_raw_schema, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_render_result, m)?)?;
    m.add_function(wrap_pyfunction!(llm::to_provider_format, m)?)?;
    m.add_function(wrap_pyfunction!(llm::convert_mid_conversation_instructions, m)?)?;
    m.add_function(wrap_pyfunction!(llm::to_fnc_ctx, m)?)?;
    m.add_function(wrap_pyfunction!(llm::to_strict_json_schema, m)?)?;
    m.add_function(wrap_pyfunction!(llm::inject_schema_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(llm::build_legacy_openai_schema, m)?)?;
    m.add_function(wrap_pyfunction!(llm::build_strict_openai_schema, m)?)?;
    m.add_function(wrap_pyfunction!(llm::to_openai_response_format, m)?)?;
    m.add_function(wrap_pyfunction!(llm::parse_function_arguments, m)?)?;
    m.add_function(wrap_pyfunction!(llm::inject_confirm_duplicate, m)?)?;
    m.add_class::<llm::HostRealtimeModel>()?;
    m.add_class::<llm::OpenAIRealtimeModel>()?;
    m.add_function(wrap_pyfunction!(llm::chat_ctx_to_dict, m)?)?;
    m.add_class::<llm::ChatRequest>()?;
    m.add_class::<llm::CoreToolsetLifetime>()?;
    m.add_class::<llm::CoreToolsetGroup>()?;
    m.add_class::<llm::CoreToolsetScopes>()?;
    m.add_function(wrap_pyfunction!(media::_core_install_frame_reader, m)?)?;
    m.add_function(wrap_pyfunction!(media::encode_image, m)?)?;
    m.add_function(wrap_pyfunction!(media::serialize_image_frame, m)?)?;
    m.add_function(wrap_pyfunction!(media::serialize_image_url, m)?)?;
    m.add_class::<media::AudioEmitter>()?;
    m.add_class::<media::AudioDecoder>()?;
    m.add_function(wrap_pyfunction!(media::split_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::runtime_busy_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::metrics_log_fields, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::stt_batch_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::stt_stream_metrics, m)?)?;
    m.add_class::<metrics::LlmMetricState>()?;
    m.add_class::<metrics::TtsMetricState>()?;
    m.add_class::<recorder::SessionRecorder>()?;
    m.add_class::<storage::Storage>()?;
    m.add_class::<storage::CoreContextActivation>()?;
    m.add_class::<storage::CoreAppTransaction>()?;
    m.add_class::<storage::Thread>()?;
    m.add_class::<storage::Run>()?;
    m.add_class::<storage::CoreThreadServer>()?;
    m.add_class::<storage::CoreVoiceStore>()?;
    m.add_class::<stt::FallbackStt>()?;
    m.add_class::<stt::FallbackRecognizeStream>()?;
    m.add_class::<stt::InferenceStt>()?;
    m.add_class::<stt::InferenceSttStream>()?;
    m.add_class::<stt::SttStreamAdapter>()?;
    m.add_class::<stt::SttAdapterStream>()?;
    m.add_class::<telemetry::TelemetrySpan>()?;
    m.add_function(wrap_pyfunction!(telemetry::_core_telemetry_install, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_telemetry_configure, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_telemetry_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_telemetry_conflict, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_set_capture_content, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_telemetry_shutdown, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_event_loop_blocked, m)?)?;
    m.add_class::<telemetry::Tagger>()?;
    m.add_class::<telemetry::CoreEventRetention>()?;
    m.add_class::<text_session::TextSessionRunner>()?;
    m.add_class::<text_session::TextSessionUpdates>()?;
    m.add_function(wrap_pyfunction!(tokenize::split_words, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::replace_words, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::tokenize_paragraphs, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::hyphenate_word, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::xml_tokenize, m)?)?;
    m.add_class::<tokenize::BasicWordTokenizer>()?;
    m.add_class::<tokenize::BasicSentenceTokenizer>()?;
    m.add_class::<tokenize::TokenBuffer>()?;
    m.add_function(wrap_pyfunction!(tokenize::normalize_language, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::base_language, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::language_region, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::language_iso, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::language_name, m)?)?;
    m.add_class::<tokenize::SearchDocument>()?;
    m.add_class::<tokenize::BM25Index>()?;
    m.add_function(wrap_pyfunction!(tokenize::keyword_search, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::tool_search_validate, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::tool_search_loaded, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::tool_proxy_arguments, m)?)?;
    m.add_function(wrap_pyfunction!(tts::chat_message_text_content, m)?)?;
    m.add_function(wrap_pyfunction!(tts::supported_nonverbals, m)?)?;
    m.add_function(wrap_pyfunction!(tts::split_all_markup, m)?)?;
    m.add_function(wrap_pyfunction!(tts::strip_all_markup, m)?)?;
    m.add_function(wrap_pyfunction!(tts::strip_expr_markup, m)?)?;
    m.add_function(wrap_pyfunction!(tts::split_expr_markup, m)?)?;
    m.add_function(wrap_pyfunction!(tts::to_xml, m)?)?;
    m.add_class::<tts::Markup>()?;
    m.add_class::<tts::TranscriptMarkupStripper>()?;
    m.add_function(wrap_pyfunction!(tts::tts_adapter_stream, m)?)?;
    m.add_class::<tts::SentencePacer>()?;
    m.add_class::<tts::TtsStream>()?;
    m.add_function(wrap_pyfunction!(tts::validate_synthesis, m)?)?;
    m.add_class::<turn::InferenceEot>()?;
    m.add_class::<turn::EotStream>()?;
    m.add_class::<turn::InterruptionDetector>()?;
    m.add_function(wrap_pyfunction!(turn::resolve_keyterm_detection, m)?)?;
    m.add_function(wrap_pyfunction!(types::api_connect_options_check, m)?)?;
    m.add_function(wrap_pyfunction!(types::api_retry_interval, m)?)?;
    m.add_class::<types::CoreProviderAttempts>()?;
    m.add_class::<utils::AudioByteStream>()?;
    m.add_class::<utils::ConnectionPoolState>()?;
    m.add_class::<utils::ExpFilter>()?;
    m.add_class::<utils::MovingAverage>()?;
    m.add_class::<vad::InferenceVad>()?;
    m.add_class::<vad::InferenceVadStream>()?;
    m.add_function(wrap_pyfunction!(voice::amd_constants, m)?)?;
    m.add_function(wrap_pyfunction!(voice::amd_resolve_max_endpointing_delay, m)?)?;
    m.add_function(wrap_pyfunction!(voice::amd_prediction_is_human, m)?)?;
    m.add_function(wrap_pyfunction!(voice::amd_prediction_is_machine, m)?)?;
    m.add_class::<voice::DataStreamAudioOutput>()?;
    m.add_class::<voice::DataStreamAudioReceiver>()?;
    m.add_class::<voice::AvatarRunner>()?;
    m.add_class::<voice::CoreQueueAudio>()?;
    m.add_class::<voice::PlayState>()?;
    m.add_class::<voice::BackgroundAudioPlayer>()?;
    m.add_class::<voice::PlaybackTracker>()?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_begin, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_current_segment, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_note_capture, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_started, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_progressed, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_finished, m)?)?;
    m.add_class::<voice::AgentNodes>()?;
    m.add_class::<voice::CoreNodeCalls>()?;
    m.add_class::<voice::LlmNodeChunks>()?;
    m.add_class::<voice::TtsNodeFrames>()?;
    m.add_class::<voice::SttNodeEvents>()?;
    m.add_function(wrap_pyfunction!(voice::llm_node, m)?)?;
    m.add_function(wrap_pyfunction!(voice::tts_node, m)?)?;
    m.add_function(wrap_pyfunction!(voice::stt_node, m)?)?;
    m.add_class::<voice::RoomEventBus>()?;
    m.add_function(wrap_pyfunction!(voice::_core_room_track_subscribed, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_track_unsubscribed, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_track_published, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_track_unpublished, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_local_track_published, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_participant_connected, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_participant_active, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_participant_attributes_changed, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_participant_disconnected, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_connection_state_changed, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_disconnected, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_room_sip_dtmf_received, m)?)?;
    m.add_class::<voice::AVSynchronizer>()?;
    m.add_class::<voice::PlayoutLedger>()?;
    m.add_class::<voice::ByteStreamSender>()?;
    m.add_class::<voice::TextStreamSender>()?;
    m.add_class::<voice::RpcHandler>()?;
    m.add_class::<voice::CoreRoomRpcRouter>()?;
    m.add_class::<voice::CoreVoiceRun>()?;
    m.add_class::<voice::SessionOptions>()?;
    m.add_class::<voice::VoiceSession>()?;
    m.add_function(wrap_pyfunction!(voice::reply_tool_indices, m)?)?;
    m.add_function(wrap_pyfunction!(voice::default_session_models, m)?)?;
    m.add_function(wrap_pyfunction!(voice::user_turn_exceeded_reply, m)?)?;
    m.add_function(wrap_pyfunction!(voice::plan_session_start, m)?)?;
    m.add_function(wrap_pyfunction!(voice::rt_server_turn_detection, m)?)?;
    m.add_function(wrap_pyfunction!(voice::resolve_interruption_detector, m)?)?;
    m.add_class::<voice::CoreSessionBindings>()?;
    m.add_class::<voice::CoreSessionOwner>()?;
    m.add_class::<voice::CoreSessionStorage>()?;
    m.add_class::<voice::SpeechHandle>()?;
    m.add_function(wrap_pyfunction!(voice::dtmf_capture_plan, m)?)?;
    m.add_function(wrap_pyfunction!(voice::email_capture_plan, m)?)?;
    m.add_function(wrap_pyfunction!(voice::email_read_back, m)?)?;
    m.add_function(wrap_pyfunction!(voice::validate_card_number, m)?)?;
    m.add_function(wrap_pyfunction!(voice::card_expired, m)?)?;
    m.add_function(wrap_pyfunction!(voice::normalize_date_of_birth, m)?)?;
    m.add_function(wrap_pyfunction!(voice::normalize_email_address, m)?)?;
    m.add_function(wrap_pyfunction!(voice::normalize_phone_number, m)?)?;
    m.add_function(wrap_pyfunction!(voice::dtmf_code, m)?)?;
    m.add_function(wrap_pyfunction!(voice::format_dtmf, m)?)?;
    m.add_function(wrap_pyfunction!(voice::send_dtmf_events, m)?)?;
    m.add_function(wrap_pyfunction!(voice::receive_dtmf, m)?)?;
    m.add_class::<voice::WarmTransfer>()?;
    m.add_class::<voice::ConsoleIo>()?;
    m.add_class::<voice::RoomAudioInput>()?;
    m.add_class::<voice::RoomVideoInput>()?;
    m.add_class::<voice::CoreRoomSession>()?;
    m.add_function(wrap_pyfunction!(voice::_core_create_room_session, m)?)?;
    m.add_function(wrap_pyfunction!(voice::wait_avatar_join, m)?)?;
    m.add_function(wrap_pyfunction!(voice::wait_for_participant, m)?)?;
    m.add_function(wrap_pyfunction!(voice::wait_for_agent, m)?)?;
    m.add_function(wrap_pyfunction!(voice::wait_for_track_publication, m)?)?;
    m.add_function(wrap_pyfunction!(worker::room_join_token, m)?)?;
    m.add_class::<worker::ByteStream>()?;
    m.add_class::<worker::CoreHttpServerCapacity>()?;
    m.add_function(wrap_pyfunction!(worker::cpu_count, m)?)?;
    m.add_class::<worker::JobSupervisor>()?;
    m.add_class::<worker::IpcClient>()?;
    m.add_class::<worker::ProcPool>()?;
    m.add_class::<worker::JobRun>()?;
    m.add_function(wrap_pyfunction!(worker::ipc_pair, m)?)?;
    m.add_function(wrap_pyfunction!(worker::init, m)?)?;
    m.add_class::<worker::ServerConfig>()?;
    m.add_class::<worker::ServerRuntime>()?;
    m.add_class::<voice::NodeText>()?;
    m.add_class::<voice::NodeAudio>()?;
    m.add_class::<voice::NodeTranscript>()?;
    m.add_function(wrap_pyfunction!(text_session::_core_persist_text_session_state, m)?)?;
    m.add_function(wrap_pyfunction!(text_session::_core_restore_text_session_state, m)?)?;
    m.add("TELEMETRY_MAX_DEPTH", ::lk_agents_abi::telemetry::policy::MAX_DEPTH)?;
    m.add_function(wrap_pyfunction!(voice::rtc_wire_names, m)?)?;
    m.add_function(wrap_pyfunction!(voice::video_queue_frames, m)?)?;
    m.add(
        "WARM_TRANSFER_IDENTITY",
        ::lk_agents_abi::voice::workflows::warm_transfer::HUMAN_AGENT_IDENTITY,
    )?;
    m.add(
        "TWILIO_RINGING_TIMEOUT",
        ::lk_agents_abi::voice::workflows::warm_transfer::TWILIO_RINGING_TIMEOUT,
    )?;
    m.add("StaleHandle", m.py().get_type::<utils::StaleHandle>())?;
    m.add("SessionClosing", m.py().get_type::<voice::SessionClosing>())?;
    native_enums::register(m)?;
    Ok(())
}

/// A native implementation a proxy call passes, handed to the host as its declared class.
pub(crate) trait NativeToHost {
    fn to_host<'py>(py: Python<'py>, native: ::std::sync::Arc<Self>)
    -> PyResult<Bound<'py, PyAny>>;
}

impl NativeToHost for dyn ::lk_agents_abi::voice::nodes::ProviderCalls {
    fn to_host<'py>(
        py: Python<'py>,
        native: ::std::sync::Arc<Self>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::generated::implementations::Selected as Implementations>::NodeCalls>() {
            Ok(native) => Ok(Py::new(py, voice::CoreNodeCalls(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::voice::nodes::ProviderCalls` is no declared class")),
        }
    }
}

impl NativeToHost for dyn ::lk_agents_abi::voice::rtc::ByteStreamIngress {
    fn to_host<'py>(
        py: Python<'py>,
        native: ::std::sync::Arc<Self>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::generated::implementations::Selected as Implementations>::ByteStreamSender>() {
            Ok(native) => Ok(Py::new(py, voice::ByteStreamSender(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::voice::rtc::ByteStreamIngress` is no declared class")),
        }
    }
}

impl NativeToHost for dyn ::lk_agents_abi::voice::rtc::RpcCallback {
    fn to_host<'py>(
        py: Python<'py>,
        native: ::std::sync::Arc<Self>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::generated::implementations::Selected as Implementations>::RpcHandler>() {
            Ok(native) => Ok(Py::new(py, voice::RpcHandler(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::voice::rtc::RpcCallback` is no declared class")),
        }
    }
}

impl NativeToHost for dyn ::lk_agents_abi::voice::rtc::TextStreamIngress {
    fn to_host<'py>(
        py: Python<'py>,
        native: ::std::sync::Arc<Self>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::generated::implementations::Selected as Implementations>::TextStreamSender>() {
            Ok(native) => Ok(Py::new(py, voice::TextStreamSender(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::voice::rtc::TextStreamIngress` is no declared class")),
        }
    }
}

impl NativeToHost for dyn ::lk_agents_abi::voice::speech_handle::Speech {
    fn to_host<'py>(
        py: Python<'py>,
        native: ::std::sync::Arc<Self>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::generated::implementations::Selected as Implementations>::SpeechHandle>() {
            Ok(native) => Ok(Py::new(py, voice::SpeechHandle(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::voice::speech_handle::Speech` is no declared class")),
        }
    }
}

impl NativeToHost for dyn ::lk_agents_abi::worker::http::ByteTransport {
    fn to_host<'py>(
        py: Python<'py>,
        native: ::std::sync::Arc<Self>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::generated::implementations::Selected as Implementations>::ByteStream>() {
            Ok(native) => Ok(Py::new(py, worker::ByteStream(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::worker::http::ByteTransport` is no declared class")),
        }
    }
}
