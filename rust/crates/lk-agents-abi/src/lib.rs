//! The provider seam and the data vocabulary every other crate is written against. Native providers
//! and host plugins implement the same traits. Nothing here performs I/O, spawns or blocks.
//!
//! One module per subject, named like the reference package (`livekit/agents/`).

// `Host` stays in this crate; `Vocabulary`, `host` and `class` are the only items re-exported from
// `lk-agents-host-derive` (`scripts/check_layout.py`).
pub use lk_agents_host_derive::Vocabulary;
pub use lk_agents_host_derive::class;
pub use lk_agents_host_derive::host;

/// So `lk_agents_abi::X` in a `#[derive(Host)]` field and `::lk_agents_abi::python` in an expansion
/// resolve in ABI as in every other crate.
extern crate self as lk_agents_abi;

pub mod errors;
pub mod evals;
pub mod inference;
pub mod llm;
pub mod metrics;
#[cfg(feature = "python")]
pub mod python;
pub mod resources;
pub mod stt;
pub mod telemetry;
pub mod tokenize;
pub mod tts;
pub mod turn;
pub mod types;
pub mod utils;
pub mod vad;
pub mod voice;
pub mod worker;

#[cfg(test)]
mod testkit;

pub use telemetry::gen_ai;
// The session report (`traces.py`, `observability.py`). Its `RecordingOptions` is not
// re-exported: `voice::session_options` has a type of that name.
pub(crate) use telemetry::attributes;
pub use telemetry::report::{
    DISABLED_MARKERS, EvaluationRecord, LogRecordWire, LoggerName, Outcome, TAG_FAIL, TAG_SUCCESS,
    TagEntry, Tagger, UploadGate, session_records, usage_attributes,
};
pub use worker::job::{
    CallbackOutcome, SHUTDOWN_CALLBACK_TIMEOUT, observability_url, run_shutdown_callbacks,
};

#[cfg(feature = "python")]
pub use llm::llm::InferenceLlmStream;
#[cfg(feature = "python")]
pub use stt::stt::FallbackSttStream;
pub use tts::emitter::{AudioEmitter, EmitterError, SegmentDump};
#[cfg(feature = "python")]
pub use tts::tts::{FallbackTtsStream, InferenceTtsStream};
pub use utils::resample::{LinearResampler, Quality, ResampleError, Resampler};

pub use llm::llm::{ChatRequest, Llm, LlmStream, ToolChoice, ToolFlags, ToolSpec};
pub use llm::realtime::{
    CancelReply, GenerateReply, PushAudio, PushItem, Say, Truncate, UpdateChatCtx,
    UpdateInstructions, UpdateTools,
};
pub use llm::realtime::{Realtime, RealtimeCommand, RealtimeCommandKind, RealtimeSession};
pub use tts::stream_adapter::{SentenceSplitter, SplitterFactory, StreamAdapter, adapter_stream};
pub use turn::detector::{DetectorLeg, EotDetector, EotLeg, EotRequest, LegEvent, ReqId};
pub use types::identity::ProviderId;
pub use utils::audio::{AudioBuffer, AudioFrame};
// The `Event` payloads. Generated code resolves them through `use lk_agents_abi::*`, so each must
// be re-exported here.
pub use llm::realtime::{
    AudioDelta, FunctionCallReady, GenerationCreated, GenerationDone, GenerationStatus,
    InputSpeechStopped, InputTranscription, MessageDone, MessageModalities, MessageStarted,
    Modalities, TextDelta,
};
pub use stt::stt::{AlignedTranscript, Stt, SttCapabilities, SttSink, SttSource};
pub use tts::tts::{
    AudioEncoding, SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource, no_duplex_path,
};
pub use utils::stream::{Closed, HOST_TEARDOWN_TIMEOUT, StreamSink, StreamSource};
pub use vad::vad::{Vad, VadCapabilities, VadSink, VadSource};

/// At the root because generated extract code names it through `use lk_agents_abi::*`.
pub use tokenize::language::LanguageCode;

pub use attributes::{
    ATTRIBUTE_AGENT_NAME, ATTRIBUTE_AGENT_STATE, ATTRIBUTE_PUBLISH_ON_BEHALF,
    ATTRIBUTE_REDACTION_ENABLED, ATTRIBUTE_SIMULATION_ENABLED, ATTRIBUTE_SIMULATOR,
    ATTRIBUTE_SIMULATOR_DISPATCH, ATTRIBUTE_TRANSCRIPTION_EXPRESSION,
    ATTRIBUTE_TRANSCRIPTION_FINAL, ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID, TOPIC_CHAT, TOPIC_TRANSCRIPTION, USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
};
pub use llm::chat_context::{
    AgentConfigUpdate, AgentHandoffItem, AudioContent, ChatContent, ChatItem, ChatMessage,
    FunctionCall, FunctionCallOutput, ImageContent, ImageSource, InferenceDetail, ItemKind,
    MetricsMetadata, MetricsReport, Modality, Role, UnknownLiteral,
};
pub use llm::chat_context::{
    ChatContext, CopyOptions, HostFacts, ItemFacts, MergeOptions, copy_excludes_facts, copy_plan,
    copy_plan_of, merge_plan, merge_plan_of, truncate_plan, truncate_plan_of,
};
pub use llm::chat_context::{Extra, chat_message_raw_text_content, chat_message_text_content};
pub use llm::ctx_diff::{Anchored, DiffOps, compute_chat_ctx_diff, compute_chat_ctx_diff_of};
pub use llm::events::{
    ChoiceDelta, CompletionUsage, FunctionToolCall, LlmChunk, chat_chunk_has_response,
};
pub use stt::events::{RecognitionUsage, SpeechData, SpeechEvent, SpeechEventType};
pub use turn::events::EotPrediction;
pub use types::connect::{
    ApiConnectOptions, ApiConnectOptionsError, DEFAULT_API_CONNECT_OPTIONS, FIRST_RETRY_INTERVAL,
    api_retry_interval,
};
pub use utils::host_ref::{CoreObject, HostRef, MediaRef, RefRegistry};
pub use vad::events::{VadEvent, VadEventType};
// `metrics::base::Metadata` is deliberately not at the root, beside the unrelated `MetricsMetadata`.
pub use llm::to_dict::{
    DumpCx, DumpMap, DumpValue, Dumped, ModelDump, ToDictError, ToDictHost, ToDictOptions,
    epoch_secs, item_to_dict, wall_from_secs,
};
pub use metrics::base::{EotInferenceMetrics, InterruptionMetrics};
pub use tts::timed::TimedString;

pub use utils::bound_clock::BoundClock;
pub use utils::byte_stream::{AudioByteStream, Flushed, Frame};
pub use utils::clock::{
    Clock, Elapsed, RealClock, SimClock, Ts, VirtualClock, WallAnchor, simulated_epoch, timeout,
};
pub use utils::duplex::{WebSocketSink, WebSocketStream};
pub use utils::filters::{ExpFilter, FilterError, MovingAverage};
pub use utils::frame::FrameId;
pub use utils::maybe::Maybe;
pub use utils::strings::{camel_to_snake_case, is_cloud_url};
pub use utils::timer::{TimerId, TimerSet};

pub use utils::bounded_lossy::{AUDIO_QUEUE_FRAMES, BoundedLossy, LossyTx, Pushed};
pub use utils::monotone::Monotone;
pub use utils::spawner::{Abort, AbortHandle, BoxFuture, Spawner, SpawnerExt, TokioSpawner};
