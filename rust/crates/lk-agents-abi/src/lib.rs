#![allow(
    clippy::disallowed_macros,
    reason = "Clippy checks macro bans at crate scope; exact legacy macro sites are ratcheted."
)]

//! The provider seam and the data vocabulary every other crate is written against. Native providers
//! and host plugins implement the same traits. Nothing here performs I/O, spawns or blocks.
//!
//! One module per subject, named like the reference package (`livekit/agents/`).

// `Host` stays in this crate; `Vocabulary`, `host` and `class` are the only items re-exported from
// `lk-agents-host-derive`.
/// The A2A protocol records (`a2a-lf`) the declared A2A and task records are built from.
pub use ::a2a as a2a_protocol;
pub use inventory;
/// The `livekit.agent` records (chat items, threads) the declared records are built from.
pub use livekit_protocol;
pub use lk_agents_host_derive::Vocabulary;
pub use lk_agents_host_derive::class;
pub use lk_agents_host_derive::host;

/// So `lk_agents_abi::X` in a `#[derive(Host)]` field and `::lk_agents_abi::python` in an expansion
/// resolve in ABI as in every other crate.
extern crate self as lk_agents_abi;

pub mod a2a;
pub mod agent;
pub mod errors;
pub mod evals;
pub mod inference;
pub mod invoke;
pub mod llm;
pub mod media;
pub mod metrics;
#[cfg(feature = "python")]
pub mod python;
pub mod recorder;
pub mod resources;
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

#[cfg(test)]
mod testkit;

// The session report (`traces.py`, `observability.py`). Its `RecordingOptions` is not
// re-exported: `voice::session_options` has a type of that name.
pub(crate) use telemetry::attributes;
pub use telemetry::report::{EvaluationRecord, LogRecordWire, Outcome, TAG_FAIL, TAG_SUCCESS};

#[cfg(feature = "python")]
pub use media::emitter::EmitterError;
pub use utils::resample::{Quality, ResampleError, Resampler};

pub use llm::llm::{ChatRequest, Llm, LlmStream, ToolChoice, ToolFlags, ToolSpec};
pub use llm::realtime::{
    CancelReply, GenerateReply, PushAudio, PushItem, Say, Truncate, UpdateChatCtx,
    UpdateInstructions, UpdateTools,
};
pub use llm::realtime::{Realtime, RealtimeCommand, RealtimeCommandKind, RealtimeSession};
pub use tts::stream_adapter::{
    SentenceSplitter, SentenceTextProcessor, SplitterFactory, StreamAdapterFunctions,
    TextProcessorFactory,
};
pub use turn::detector::{EotDetector, LegEvent, ReqId};
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
pub use utils::stream::{Closed, HOST_TEARDOWN_TIMEOUT, RetryReport, StreamSink, StreamSource};
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
pub use llm::chat_context::{Extra, chat_message_raw_text_content};
pub use llm::events::{ChoiceDelta, CompletionUsage, FunctionToolCall, LlmChunk};
pub use stt::events::{RecognitionUsage, SpeechData, SpeechEvent, SpeechEventType};
pub use turn::events::EotPrediction;
pub use types::connect::{
    ApiConnectOptions, ApiConnectOptionsError, DEFAULT_API_CONNECT_OPTIONS, FIRST_RETRY_INTERVAL,
};
pub use utils::host_ref::{CoreObject, HostRef, MediaRef, RefRegistry};
pub use vad::events::{VadEvent, VadEventType};
// `metrics::base::Metadata` is deliberately not at the root, beside the unrelated `MetricsMetadata`.
pub use llm::to_dict::{
    DumpCx, DumpMap, DumpValue, Dumped, ModelDump, ToDictError, ToDictHost, ToDictOptions,
    epoch_secs, wall_from_secs,
};
pub use metrics::base::{EotInferenceMetrics, InterruptionMetrics};
pub use tts::timed::TimedString;

pub use utils::bound_clock::BoundClock;
pub use utils::clock::{
    Clock, Elapsed, RealClock, SimClock, Ts, VirtualClock, WallAnchor, before, simulated_epoch,
    timeout, unix_now, unix_secs,
};
pub use utils::filters::FilterError;
pub use utils::frame::FrameId;
pub use utils::lock;
pub use utils::maybe::Maybe;
pub use utils::timer::{TimerId, TimerSet};

pub use utils::bounded_lossy::{AUDIO_QUEUE_FRAMES, BoundedLossy, LossyTx, Pushed};
pub use utils::monotone::Monotone;
pub use utils::spawner::{
    Abort, AbortHandle, BoxFuture, Spawner, SpawnerExt, TaskSet, TokioSpawner,
};
