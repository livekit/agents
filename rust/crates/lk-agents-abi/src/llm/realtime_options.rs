use serde::{Serialize, Serializer};

/// A wire string, serialized as itself.
macro_rules! serialize_as_wire {
    ($($t:ty),*) => {$(
        impl Serialize for $t {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.as_str())
            }
        }
    )*};
}

/// The OpenAI realtime models.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "RealtimeModels"
))]
pub enum OpenAiRealtimeModel {
    #[host(rename = "gpt-realtime")]
    GptRealtime,
    #[host(rename = "gpt-realtime-1.5")]
    GptRealtime15,
    #[host(rename = "gpt-realtime-2")]
    GptRealtime2,
    #[host(rename = "gpt-realtime-2025-08-28")]
    GptRealtime20250828,
    #[host(rename = "gpt-4o-realtime-preview")]
    Gpt4oRealtimePreview,
}

/// Transcription of the caller's audio.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioTranscription {
    /// The transcription model, e.g. ``"gpt-4o-mini-transcribe"``, ``"whisper-1"``, or Voice
    /// Live's ``"azure-speech"``.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The input language (ISO-639-1, or Voice Live's ``"en-US,fr-FR"``).
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Text guiding the transcription's style or vocabulary.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

/// OpenAI's input noise reduction: a close-talking or a far-field microphone.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "NoiseReductionType",
    extract
))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseReductionType {
    NearField,
    FarField,
}

/// Azure Voice Live's deep noise suppression of the caller's audio.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "azure_deep_noise_suppression")]
pub struct AzureDeepNoiseSuppression {}

/// Input noise reduction: OpenAI's by microphone type, or Voice Live's.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "NoiseReduction",
    extract,
    transparent
))]
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseReduction {
    Microphone(NoiseReductionType),
    Azure(AzureDeepNoiseSuppression),
}

impl Serialize for NoiseReduction {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Typed {
            r#type: NoiseReductionType,
        }
        match self {
            NoiseReduction::Microphone(r#type) => Typed { r#type: *r#type }.serialize(s),
            NoiseReduction::Azure(azure) => azure.serialize(s),
        }
    }
}

/// Server turn detection on the input's volume.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "server_vad")]
pub struct ServerVad {
    /// The activation threshold, 0.0 to 1.0; higher needs louder audio.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Audio kept before the detected speech, in milliseconds.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_padding_ms: Option<u32>,
    /// The silence that ends speech, in milliseconds.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub silence_duration_ms: Option<u32>,
    /// The silence after which the model replies on its own, in milliseconds.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_timeout_ms: Option<u32>,
    /// Reply when speech ends; the server's default is on.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_response: Option<bool>,
    /// Cut the reply when speech starts; the server's default is on.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interrupt_response: Option<bool>,
}

/// How readily semantic turn detection ends the caller's turn.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "Eagerness",
    extract
))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eagerness {
    Low,
    Medium,
    High,
    Auto,
}

/// Server turn detection on what the caller says.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "semantic_vad")]
pub struct SemanticVad {
    /// ``"auto"`` (the server's default) is ``"medium"``.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eagerness: Option<Eagerness>,
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_response: Option<bool>,
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interrupt_response: Option<bool>,
}

/// Azure Voice Live's semantic turn detection models.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "AzureSemanticVadType",
    extract
))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AzureSemanticVadType {
    AzureSemanticVad,
    AzureSemanticVadEn,
    AzureSemanticVadMultilingual,
}

/// Azure Voice Live's semantic turn detection.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AzureSemanticVad {
    #[host(default = "azure_semantic_vad")]
    pub r#type: AzureSemanticVadType,
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_padding_ms: Option<u32>,
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub silence_duration_ms: Option<u32>,
    /// Ignore fillers ("um", "uh") when deciding the turn ended.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remove_filler_words: Option<bool>,
    /// The languages ``azure_semantic_vad_multilingual`` listens for.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<String>>,
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_response: Option<bool>,
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interrupt_response: Option<bool>,
}

/// Server turn detection.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "TurnDetection",
    extract,
    transparent
))]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum TurnDetection {
    Server(ServerVad),
    Semantic(SemanticVad),
    Azure(AzureSemanticVad),
}

impl TurnDetection {
    /// Reply when speech ends; `None` is the server's default (on).
    pub fn create_response(&self) -> Option<bool> {
        match self {
            TurnDetection::Server(td) => td.create_response,
            TurnDetection::Semantic(td) => td.create_response,
            TurnDetection::Azure(td) => td.create_response,
        }
    }

    /// Cut the reply when speech starts; `None` is the server's default (on).
    pub fn interrupt_response(&self) -> Option<bool> {
        match self {
            TurnDetection::Server(td) => td.interrupt_response,
            TurnDetection::Semantic(td) => td.interrupt_response,
            TurnDetection::Azure(td) => td.interrupt_response,
        }
    }
}

/// An OpenAI custom voice.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CustomVoice {
    pub id: String,
}

#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "AzureVoiceType",
    extract
))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AzureVoiceType {
    #[host(rename = "azure-standard")]
    AzureStandard,
    #[host(rename = "azure-custom")]
    AzureCustom,
}

/// An Azure Voice Live voice.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AzureVoice {
    /// The voice, e.g. ``"en-US-AvaMultilingualNeural"``.
    pub name: String,
    #[host(default = "azure-standard")]
    pub r#type: AzureVoiceType,
    /// The one language it speaks, e.g. ``"en-US"``.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    /// A custom voice's deployment.
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_id: Option<String>,
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// The voice of the model's audio: a name, or a voice object.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "Voice",
    extract,
    transparent
))]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Voice {
    Name(String),
    Custom(CustomVoice),
    Azure(AzureVoice),
}

/// The tracing a `Tracing` with nothing set sends: the server's defaults.
const TRACING_AUTO: &str = "auto";

/// OpenAI's traces of the session. Nothing set is the server's defaults (``"auto"``).
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct Tracing {
    /// The trace's name in the Traces Dashboard.
    #[host(default = None)]
    pub workflow_name: Option<String>,
    #[host(default = None)]
    pub group_id: Option<String>,
    #[host(default = None)]
    pub metadata: Option<serde_json::Value>,
}

impl Serialize for Tracing {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Configured<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            workflow_name: &'a Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            group_id: &'a Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            metadata: &'a Option<serde_json::Value>,
        }
        let Tracing { workflow_name, group_id, metadata } = self;
        if workflow_name.is_none() && group_id.is_none() && metadata.is_none() {
            return s.serialize_str(TRACING_AUTO);
        }
        Configured { workflow_name, group_id, metadata }.serialize(s)
    }
}

/// How the server drops conversation over the input token limit: ``"auto"``, or never.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "TruncationStrategy",
    extract
))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    Auto,
    Disabled,
}

/// Truncation that keeps a fraction of the conversation, amortizing truncations over turns.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct RetentionRatio {
    /// The fraction of the post-instruction conversation kept, 0.0 to 1.0.
    pub retention_ratio: f64,
    /// The conversation tokens allowed after the instructions; the model's limit by default.
    #[host(default = None)]
    pub post_instructions_token_limit: Option<u32>,
}

impl Serialize for RetentionRatio {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Limits {
            post_instructions: u32,
        }
        #[derive(Serialize)]
        #[serde(tag = "type", rename = "retention_ratio")]
        struct Wire {
            retention_ratio: f64,
            #[serde(skip_serializing_if = "Option::is_none")]
            token_limits: Option<Limits>,
        }
        Wire {
            retention_ratio: self.retention_ratio,
            token_limits: self
                .post_instructions_token_limit
                .map(|n| Limits { post_instructions: n }),
        }
        .serialize(s)
    }
}

/// Conversation truncation over the input token limit.
#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "Truncation",
    extract,
    transparent
))]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Truncation {
    Strategy(TruncationStrategy),
    RetentionRatio(RetentionRatio),
}

#[crate::host(record(
    module = "realtime_options",
    path = "livekit.agents.llm._realtime",
    name = "ReasoningEffort",
    extract
))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

/// Reasoning of a reasoning model such as ``gpt-realtime-2``.
#[crate::host(record(module = "realtime_options", path = "livekit.agents.llm._realtime", extract))]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Reasoning {
    #[host(default = None)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
}

/// The LiveKit Inference gateway's routing of a realtime session, set by `inference.RealtimeModel`.
#[crate::host(record(module = "realtime_options", extract))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InferenceGateway {
    #[host(default = None)]
    pub api_secret: Option<String>,
    /// The provider the gateway routes to.
    #[host(default = None)]
    pub provider: Option<String>,
    #[host(default = None)]
    pub inference_class: Option<String>,
}

serialize_as_wire!(
    NoiseReductionType,
    Eagerness,
    AzureSemanticVadType,
    AzureVoiceType,
    TruncationStrategy,
    ReasoningEffort
);
