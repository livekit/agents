//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The declared classes' pyo3 adapters, one module per ABI subject.

use pyo3::prelude::*;

pub mod a2a;
pub mod agent;
pub mod errors;
pub mod evals;
pub mod inference;
pub mod llm;
pub mod media;
pub mod metrics;
pub mod recorder;
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
pub mod records;

/// The implementation of each declared class. The registration shell implements it.
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
    type VoiceAgent: ::lk_agents_abi::agent::VoiceAgent;
    type RunEvents: ::lk_agents_abi::agent::assertions::RunEvents;
    type AgentRunner: ::lk_agents_abi::agent::run::AgentRunner;
    type ApiErrorFunctions: ::lk_agents_abi::errors::ApiErrorFunctions;
    type JudgeRunner: ::lk_agents_abi::evals::JudgeRunner;
    type Evaluations: ::lk_agents_abi::evals::Evaluations;
    type InferenceCredentials: ::lk_agents_abi::inference::InferenceCredentials;
    type InferenceModels: ::lk_agents_abi::inference::InferenceModels;
    type InferenceAvatar: ::lk_agents_abi::inference::avatar_session::InferenceAvatar;
    type InferenceTts: ::lk_agents_abi::inference::tts_client::InferenceTts<Stream = Self::InferenceTtsStream>;
    type ChatContextPlans: ::lk_agents_abi::llm::chat_context::ChatContextPlans;
    type ChatContextDiffs: ::lk_agents_abi::llm::ctx_diff::ChatContextDiffs;
    type FixedGate: ::lk_agents_abi::llm::duplex::FixedGate;
    type AdaptiveNoiseGate: ::lk_agents_abi::llm::duplex::AdaptiveNoiseGate;
    type DuplexRealtimeModel: ::lk_agents_abi::llm::duplex::DuplexRealtimeModel;
    type FallbackLlm: ::lk_agents_abi::llm::fallback::FallbackLlm;
    type FallbackTts: ::lk_agents_abi::llm::fallback::FallbackTts<Stream = Self::FallbackTtsStream>;
    type TtsTokenLog: ::lk_agents_abi::llm::fallback::TtsTokenLog;
    type InferenceLlm: ::lk_agents_abi::llm::inference::InferenceLlm;
    type ChoiceParser: ::lk_agents_abi::llm::inference::ChoiceParser;
    type CompletionWire: ::lk_agents_abi::llm::inference::CompletionWire;
    type McpSession: ::lk_agents_abi::llm::mcp::McpSession;
    type McpTools: ::lk_agents_abi::llm::mcp::McpTools;
    type LlmAlgorithms: ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms;
    type HostRealtimeModel: ::lk_agents_abi::llm::realtime::HostRealtimeModel;
    type WebSocketRealtimeModel: ::lk_agents_abi::llm::realtime_native::WebSocketRealtimeModel;
    type RealtimeFallbackModel: ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel;
    type ChatContextDict: ::lk_agents_abi::llm::to_dict::ChatContextDict;
    type FrameReaderInstall: ::lk_agents_abi::media::images::FrameReaderInstall;
    type ImageFunctions: ::lk_agents_abi::media::images::ImageFunctions;
    type AudioEmitter: ::lk_agents_abi::media::emitter::AudioEmitter;
    type AudioDecoder: ::lk_agents_abi::media::codecs::AudioDecoder;
    type StreamBuffer: ::lk_agents_abi::media::codecs::StreamBuffer;
    type MediaTokenization: ::lk_agents_abi::media::codecs::MediaTokenization;
    type MetricsRecords: ::lk_agents_abi::metrics::MetricsRecords;
    type UsageCollector: ::lk_agents_abi::metrics::usage::UsageCollector;
    type SessionRecorder: ::lk_agents_abi::recorder::SessionRecorder;
    type Storage: ::lk_agents_abi::storage::Storage<ContextActivation = Self::ContextActivation, VoiceStore = Self::VoiceStore>;
    type ContextActivation: ::lk_agents_abi::storage::context::ContextActivation<Transaction = Self::AppTransaction>;
    type AppTransaction: ::lk_agents_abi::storage::context::AppTransaction;
    type VoiceStore: ::lk_agents_abi::storage::voice::VoiceStore<ContextActivation = Self::ContextActivation>;
    type FallbackStt: ::lk_agents_abi::stt::fallback::FallbackStt;
    type InferenceStt: ::lk_agents_abi::stt::inference::InferenceStt;
    type SttStreamAdapter: ::lk_agents_abi::stt::stream_adapter::SttStreamAdapter;
    type TelemetryPolicy: ::lk_agents_abi::telemetry::policy::TelemetryPolicy;
    type TelemetryAdmission: ::lk_agents_abi::telemetry::policy::TelemetryAdmission;
    type MetricSeries: ::lk_agents_abi::telemetry::policy::MetricSeries;
    type ProviderMetrics: ::lk_agents_abi::telemetry::provider::ProviderMetrics;
    type LlmMetricState: ::lk_agents_abi::telemetry::provider::LlmMetricState;
    type TtsMetricState: ::lk_agents_abi::telemetry::provider::TtsMetricState;
    type Tagger: ::lk_agents_abi::telemetry::report::Tagger;
    type UploadGate: ::lk_agents_abi::telemetry::report::UploadGate;
    type ReportRecords: ::lk_agents_abi::telemetry::report::ReportRecords;
    type SessionReportProjection: ::lk_agents_abi::telemetry::report::projection::SessionReportProjection;
    type EventRetention: ::lk_agents_abi::telemetry::observation::EventRetention;
    type TraceParents: ::lk_agents_abi::telemetry::transport::TraceParents;
    type SessionReportUpload: ::lk_agents_abi::telemetry::upload::SessionReportUpload;
    type PreparedTranscript: ::lk_agents_abi::telemetry::upload::PreparedTranscript;
    type TextSessionRunner: ::lk_agents_abi::text_session::TextSessionRunner;
    type TokenizeFunctions: ::lk_agents_abi::tokenize::bindings::TokenizeFunctions;
    type HostTokenization: ::lk_agents_abi::tokenize::bindings::HostTokenization;
    type WordTokenizer: ::lk_agents_abi::tokenize::bindings::WordTokenizer;
    type SentenceTokenizer: ::lk_agents_abi::tokenize::bindings::SentenceTokenizer;
    type BufferedTokens: ::lk_agents_abi::tokenize::bindings::BufferedTokens;
    type LanguageFunctions: ::lk_agents_abi::tokenize::language::LanguageFunctions;
    type SearchDocument: ::lk_agents_abi::tokenize::search::SearchDocument;
    type SearchIndex: ::lk_agents_abi::tokenize::search::SearchIndex<Document = Self::SearchDocument>;
    type ToolSearch: ::lk_agents_abi::tokenize::search::ToolSearch<Document = Self::SearchDocument>;
    type MarkupFunctions: ::lk_agents_abi::tts::markup::bindings::MarkupFunctions;
    type MarkupClass: ::lk_agents_abi::tts::markup::bindings::MarkupClass;
    type TranscriptStripper: ::lk_agents_abi::tts::markup::bindings::TranscriptStripper;
    type StreamAdapterFunctions: ::lk_agents_abi::StreamAdapterFunctions<Stream = Self::FallbackTtsStream>;
    type SentencePacer: ::lk_agents_abi::tts::stream_pacer::SentencePacer;
    type InferenceTtsStream: ::lk_agents_abi::tts::streams::InferenceTtsStream;
    type FallbackTtsStream: ::lk_agents_abi::tts::streams::FallbackTtsStream;
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
    type AvatarRunner: ::lk_agents_abi::voice::avatar::AvatarRunner;
    type PlayState: ::lk_agents_abi::voice::background_audio::PlayState;
    type BackgroundAudioPlayer: ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer<State = Self::PlayState>;
    type PlaybackReports: ::lk_agents_abi::voice::io::PlaybackReports;
    type AgentNodes: ::lk_agents_abi::voice::nodes::AgentNodes;
    type NodeCalls: ::lk_agents_abi::voice::nodes::NodeCalls;
    type TtsNodeFrames: ::lk_agents_abi::voice::nodes::TtsNodeFrames;
    type SttNodeEvents: ::lk_agents_abi::voice::nodes::SttNodeEvents;
    type DefaultNodes: ::lk_agents_abi::voice::nodes::DefaultNodes<Frames = Self::TtsNodeFrames, Events = Self::SttNodeEvents>;
    type RtcRuntime: ::lk_agents_abi::voice::rtc::events::RtcRuntime;
    type PlayoutLedger: ::lk_agents_abi::voice::rtc::playout::PlayoutLedger;
    type SessionOptions: ::lk_agents_abi::voice::session::SessionOptions;
    type VoiceSession: ::lk_agents_abi::voice::session::VoiceSession<Options = Self::SessionOptions>;
    type SessionStart: ::lk_agents_abi::voice::session_effects::SessionStart;
    type InterruptionGateFunctions: ::lk_agents_abi::voice::session_options::InterruptionGateFunctions;
    type WorkflowChecks: ::lk_agents_abi::voice::workflows::WorkflowChecks;
    type DtmfResult: ::lk_agents_abi::voice::workflows::dtmf::DtmfResult;
    type DtmfIo: ::lk_agents_abi::voice::workflows::dtmf::DtmfIo<Result = Self::DtmfResult>;
    type WarmTransfer: ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer;
    type ConsoleIo: ::lk_agents_abi::voice::console::ConsoleIo;
    type RoomRuntime: ::lk_agents_abi::voice::room_owner::RoomRuntime;
    type EndpointStream: ::lk_agents_abi::worker::endpoint_host::EndpointStream;
    type EndpointServer: ::lk_agents_abi::worker::endpoint_host::EndpointServer;
    type AsgiReceive: ::lk_agents_abi::worker::endpoint_host::AsgiReceive;
    type AsgiSend: ::lk_agents_abi::worker::endpoint_host::AsgiSend;
    type JobTrace: ::lk_agents_abi::worker::job::JobTrace;
    type ServerEnvironment: ::lk_agents_abi::worker::worker::ServerEnvironment;
    type WorkerCli: ::lk_agents_abi::worker::cli::WorkerCli;
    type CliSignals: ::lk_agents_abi::worker::cli::CliSignals;
    type ProcessSupervisor: ::lk_agents_abi::worker::ipc_host::ProcessSupervisor;
    type IpcClient: ::lk_agents_abi::worker::ipc_host::IpcClient;
    type ProcessPool: ::lk_agents_abi::worker::ipc_host::ProcessPool;
    type JobRun: ::lk_agents_abi::worker::ipc_host::JobRun;
    type ServerConfig: ::lk_agents_abi::worker::server_host::ServerConfig;
    type ServerRuntime: ::lk_agents_abi::worker::server_host::ServerRuntime;
}

/// Adds every declared class to `_core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<a2a::_core_Credentials>()?;
    m.add_function(wrap_pyfunction!(a2a::_core_auth_grant_allows, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_extensions_header, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_agent_card, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_message_text, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_message_data, m)?)?;
    m.add_class::<a2a::Reply>()?;
    m.add_class::<a2a::AgentClient>()?;
    m.add_class::<a2a::_core_A2ADoor>()?;
    m.add_class::<a2a::_core_A2AResponse>()?;
    m.add_class::<a2a::_core_RemoteInteraction>()?;
    m.add_function(wrap_pyfunction!(a2a::_core_remote_token, m)?)?;
    m.add_class::<agent::VoiceAgent>()?;
    m.add_class::<agent::RunEvents>()?;
    m.add_class::<agent::AgentRunner>()?;
    m.add_class::<agent::AgentRunUpdates>()?;
    m.add_function(wrap_pyfunction!(errors::http_error_display, m)?)?;
    m.add_class::<evals::_core_JudgeRunner>()?;
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
    m.add_function(wrap_pyfunction!(llm::chat_ctx_copy_plan, m)?)?;
    m.add_function(wrap_pyfunction!(llm::chat_ctx_truncate_plan, m)?)?;
    m.add_function(wrap_pyfunction!(llm::chat_ctx_merge_plan, m)?)?;
    m.add_function(wrap_pyfunction!(llm::remove_instructions, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_chat_ctx_diff, m)?)?;
    m.add_class::<llm::FixedGate>()?;
    m.add_class::<llm::AdaptiveNoiseGate>()?;
    m.add_class::<llm::DuplexRealtimeModel>()?;
    m.add_class::<llm::FallbackLlm>()?;
    m.add_class::<llm::FallbackLlmChunks>()?;
    m.add_class::<llm::FallbackTts>()?;
    m.add_class::<llm::TtsTokenLog>()?;
    m.add_class::<llm::InferenceLlm>()?;
    m.add_class::<llm::InferenceLlmChunks>()?;
    m.add_class::<llm::LlmChoiceParser>()?;
    m.add_function(wrap_pyfunction!(llm::llm_usage_chunk, m)?)?;
    m.add_class::<llm::McpSession>()?;
    m.add_function(wrap_pyfunction!(llm::resolve_mcp_tool_options, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_raw_schema, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_plan_setup, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_http_transport, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_render_result, m)?)?;
    m.add_function(wrap_pyfunction!(llm::_core_mcp_timeout_check, m)?)?;
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
    m.add_class::<llm::NativeRealtimeModel>()?;
    m.add_class::<llm::RealtimeFallbackModel>()?;
    m.add_function(wrap_pyfunction!(llm::chat_ctx_to_dict, m)?)?;
    m.add_function(wrap_pyfunction!(media::_core_install_frame_reader, m)?)?;
    m.add_function(wrap_pyfunction!(media::encode_image, m)?)?;
    m.add_function(wrap_pyfunction!(media::serialize_image_frame, m)?)?;
    m.add_function(wrap_pyfunction!(media::serialize_image_url, m)?)?;
    m.add_class::<media::AudioEmitter>()?;
    m.add_class::<media::AudioDecoder>()?;
    m.add_class::<media::StreamBuffer>()?;
    m.add_function(wrap_pyfunction!(media::split_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::metrics_log_fields, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::otel_turn_records, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::otel_usage_records, m)?)?;
    m.add_class::<metrics::ModelUsageCollector>()?;
    m.add_class::<recorder::SessionRecorder>()?;
    m.add_class::<storage::Storage>()?;
    m.add_class::<storage::_core_ContextActivation>()?;
    m.add_class::<storage::_core_AppTransaction>()?;
    m.add_class::<storage::_core_VoiceStore>()?;
    m.add_class::<stt::FallbackStt>()?;
    m.add_class::<stt::FallbackRecognizeStream>()?;
    m.add_class::<stt::InferenceStt>()?;
    m.add_class::<stt::InferenceSttStream>()?;
    m.add_class::<stt::SttStreamAdapter>()?;
    m.add_class::<stt::SttAdapterStream>()?;
    m.add_class::<telemetry::_core_TelemetryPolicy>()?;
    m.add_function(wrap_pyfunction!(telemetry::_core_export_action, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_value_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_record_bytes, m)?)?;
    m.add_class::<telemetry::_core_MetricSeries>()?;
    m.add_function(wrap_pyfunction!(telemetry::stt_batch_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::stt_stream_metrics, m)?)?;
    m.add_class::<telemetry::LlmMetricState>()?;
    m.add_class::<telemetry::TtsMetricState>()?;
    m.add_class::<telemetry::Tagger>()?;
    m.add_class::<telemetry::UploadGate>()?;
    m.add_function(wrap_pyfunction!(telemetry::recording_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_session_chat_record, m)?)?;
    m.add_class::<telemetry::_core_SessionReportProjection>()?;
    m.add_class::<telemetry::_core_EventRetention>()?;
    m.add_function(wrap_pyfunction!(telemetry::_core_format_trace_parent, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry::_core_trace_parent, m)?)?;
    m.add_class::<telemetry::_core_SessionReportUpload>()?;
    m.add_class::<telemetry::_core_PreparedTranscript>()?;
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
    m.add_function(wrap_pyfunction!(tts::chat_history_summary_xml, m)?)?;
    m.add_function(wrap_pyfunction!(tts::chat_ctx_summarize_plan, m)?)?;
    m.add_class::<tts::Markup>()?;
    m.add_class::<tts::TranscriptMarkupStripper>()?;
    m.add_function(wrap_pyfunction!(tts::tts_adapter_stream, m)?)?;
    m.add_class::<tts::SentencePacer>()?;
    m.add_class::<tts::InferenceTtsStream>()?;
    m.add_class::<tts::FallbackTtsStream>()?;
    m.add_function(wrap_pyfunction!(tts::validate_synthesis, m)?)?;
    m.add_class::<turn::InferenceEot>()?;
    m.add_class::<turn::EotStream>()?;
    m.add_class::<turn::InterruptionDetector>()?;
    m.add_function(wrap_pyfunction!(turn::resolve_keyterm_detection, m)?)?;
    m.add_function(wrap_pyfunction!(types::api_connect_options_check, m)?)?;
    m.add_function(wrap_pyfunction!(types::api_retry_interval, m)?)?;
    m.add_class::<types::_core_ProviderAttempts>()?;
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
    m.add_class::<voice::PlayState>()?;
    m.add_class::<voice::BackgroundAudioPlayer>()?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_begin, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_current_segment, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_note_capture, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_started, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_progressed, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_playback_finished, m)?)?;
    m.add_class::<voice::AgentNodes>()?;
    m.add_class::<voice::_core_NodeCalls>()?;
    m.add_class::<voice::TtsNodeFrames>()?;
    m.add_class::<voice::SttNodeEvents>()?;
    m.add_function(wrap_pyfunction!(voice::tts_node, m)?)?;
    m.add_function(wrap_pyfunction!(voice::stt_node, m)?)?;
    m.add_function(wrap_pyfunction!(voice::_core_create_room_event_bus, m)?)?;
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
    m.add_class::<voice::PlayoutLedger>()?;
    m.add_class::<voice::SessionOptions>()?;
    m.add_class::<voice::VoiceSession>()?;
    m.add_function(wrap_pyfunction!(voice::plan_session_start, m)?)?;
    m.add_function(wrap_pyfunction!(voice::rt_server_turn_detection, m)?)?;
    m.add_function(wrap_pyfunction!(voice::resolve_interruption_detector, m)?)?;
    m.add_function(wrap_pyfunction!(voice::validate_card_number, m)?)?;
    m.add_function(wrap_pyfunction!(voice::card_expired, m)?)?;
    m.add_function(wrap_pyfunction!(voice::normalize_date_of_birth, m)?)?;
    m.add_function(wrap_pyfunction!(voice::normalize_email_address, m)?)?;
    m.add_function(wrap_pyfunction!(voice::normalize_phone_number, m)?)?;
    m.add_function(wrap_pyfunction!(voice::dtmf_code, m)?)?;
    m.add_function(wrap_pyfunction!(voice::format_dtmf, m)?)?;
    m.add_class::<voice::GetDtmfResult>()?;
    m.add_function(wrap_pyfunction!(voice::send_dtmf_events, m)?)?;
    m.add_function(wrap_pyfunction!(voice::receive_dtmf, m)?)?;
    m.add_class::<voice::WarmTransfer>()?;
    m.add_class::<voice::ConsoleIo>()?;
    m.add_function(wrap_pyfunction!(voice::_core_create_room_session, m)?)?;
    m.add_function(wrap_pyfunction!(voice::wait_avatar_join, m)?)?;
    m.add_function(wrap_pyfunction!(voice::wait_for_participant, m)?)?;
    m.add_function(wrap_pyfunction!(voice::wait_for_agent, m)?)?;
    m.add_function(wrap_pyfunction!(voice::wait_for_track_publication, m)?)?;
    m.add_class::<worker::EndpointStream>()?;
    m.add_class::<worker::EndpointServer>()?;
    m.add_class::<worker::_core_AsgiReceive>()?;
    m.add_class::<worker::_core_AsgiSend>()?;
    m.add_function(wrap_pyfunction!(worker::job_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(worker::cpu_count, m)?)?;
    m.add_function(wrap_pyfunction!(worker::endpoint_url_base, m)?)?;
    m.add_function(wrap_pyfunction!(worker::parse_cli, m)?)?;
    m.add_function(wrap_pyfunction!(worker::entrypoint_module, m)?)?;
    m.add_function(wrap_pyfunction!(worker::format_log_record, m)?)?;
    m.add_function(wrap_pyfunction!(worker::dev_channel, m)?)?;
    m.add_function(wrap_pyfunction!(worker::ipc_pair, m)?)?;
    m.add_class::<worker::ExitSignals>()?;
    m.add_class::<worker::ProcSupervisor>()?;
    m.add_class::<worker::IpcClient>()?;
    m.add_class::<worker::ProcPool>()?;
    m.add_class::<worker::JobRun>()?;
    m.add_class::<worker::ServerConfig>()?;
    m.add_class::<worker::ServerRuntime>()?;
    m.add_class::<voice::NodeText>()?;
    m.add_class::<voice::NodeAudio>()?;
    m.add_class::<voice::NodeTranscript>()?;
    m.add_function(wrap_pyfunction!(text_session::_core_persist_text_session_state, m)?)?;
    m.add_function(wrap_pyfunction!(text_session::_core_restore_text_session_state, m)?)?;
    Ok(())
}

/// A native implementation a proxy call passes, handed to the host as its declared class.
pub(crate) trait NativeToHost {
    fn to_host<'py>(py: Python<'py>, native: ::std::sync::Arc<Self>) -> PyResult<Bound<'py, PyAny>>;
}

impl NativeToHost for dyn ::lk_agents_abi::voice::nodes::ProviderCalls {
    fn to_host<'py>(py: Python<'py>, native: ::std::sync::Arc<Self>) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::Selected as Implementations>::NodeCalls>() {
            Ok(native) => Ok(Py::new(py, voice::_core_NodeCalls(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::voice::nodes::ProviderCalls` is no declared class")),
        }
    }
}

impl NativeToHost for dyn ::lk_agents_abi::worker::endpoint_host::AsgiReceiver {
    fn to_host<'py>(py: Python<'py>, native: ::std::sync::Arc<Self>) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::Selected as Implementations>::AsgiReceive>() {
            Ok(native) => Ok(Py::new(py, worker::_core_AsgiReceive(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::worker::endpoint_host::AsgiReceiver` is no declared class")),
        }
    }
}

impl NativeToHost for dyn ::lk_agents_abi::worker::endpoint_host::AsgiSender {
    fn to_host<'py>(py: Python<'py>, native: ::std::sync::Arc<Self>) -> PyResult<Bound<'py, PyAny>> {
        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;
        match native.downcast::<<crate::Selected as Implementations>::AsgiSend>() {
            Ok(native) => Ok(Py::new(py, worker::_core_AsgiSend(native))?.into_bound(py).into_any()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("this `::lk_agents_abi::worker::endpoint_host::AsgiSender` is no declared class")),
        }
    }
}
