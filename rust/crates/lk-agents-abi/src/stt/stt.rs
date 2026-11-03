//! `Stt`: the abstract path is batch (`stt/stt.py:204`); `stream()` is optional. `recognize` is
//! always callable; `SttCapabilities::offline_recognize` says whether it works.

use crate::SpeechEvent;
use crate::errors::ProviderError;
use crate::tokenize::language::LanguageCode;

use crate::utils::stream::{ProviderStream, StreamInput};
use crate::{ApiConnectOptions, AudioFrame};

/// `Literal["word", "chunk", False]`. `None` is the boolean `False` on the host, never a string;
/// unknown textual values retain the existing disabled-alignment fallback.
#[crate::host(codec(
    extract,
    emit,
    host = "typing.Literal[\"word\", \"chunk\", False]",
    via = serde_json::Value
))]
#[crate::host(wire_strs(
    err = crate::llm::chat_context::UnknownLiteral,
    msg = crate::llm::chat_context::UnknownLiteral { kind: "aligned_transcript", value: s.to_owned() },
))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AlignedTranscript {
    #[default]
    #[host(rename = "")]
    None,
    Word,
    Chunk,
}

impl From<serde_json::Value> for AlignedTranscript {
    fn from(value: serde_json::Value) -> Self {
        value.as_str().and_then(|s| s.parse().ok()).unwrap_or_default()
    }
}

impl From<AlignedTranscript> for serde_json::Value {
    fn from(value: AlignedTranscript) -> Self {
        match value {
            AlignedTranscript::None => false.into(),
            other => other.as_str().into(),
        }
    }
}

// Field `///`s are the reference's. `impl Default` and the model's defaults agree on purpose.
#[crate::host(record(
    module = "stt",
    path = "livekit.agents.stt",
    name = "STTCapabilities",
    dataclass,
    extract,
    emit,
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SttCapabilities {
    pub streaming: bool,
    pub interim_results: bool,
    #[host(default = false)]
    pub diarization: bool,
    #[host(default = false)]
    pub aligned_transcript: AlignedTranscript,
    /// Whether the STT supports batch recognition via recognize() method
    #[host(default = true)]
    pub offline_recognize: bool,
    /// Whether the STT supports keyterm prompting
    #[host(default = false)]
    pub keyterms: bool,
    /// Whether the STT can natively consume conversation context (see STT._push_conversation_item)
    #[host(default = false)]
    pub chat_context: bool,
}

impl Default for SttCapabilities {
    fn default() -> Self {
        SttCapabilities {
            streaming: false,
            interim_results: false,
            diarization: false,
            aligned_transcript: AlignedTranscript::None,
            offline_recognize: true,
            keyterms: false,
            chat_context: false,
        }
    }
}

/// May drop: the driver feeds it from a drop-oldest queue.
pub type SttSink = Box<dyn StreamInput<AudioFrame>>;
/// What a recognition stream yields: its speech events.
pub type SttSource = Box<dyn ProviderStream<SpeechEvent>>;

// A host `stt.STT` is this trait through its generated proxy (`SttProxy`), calling the `_core_*`
// methods of its hand-written base class (`stt.py`), which keeps the plugin API's shape
// (`_recognize_impl`, `conn_options`) there.
/// A speech-to-text provider.
///
/// Plugins implement `_recognize_impl` for batch recognition, and override `stream()` when the
/// service recognizes live audio.
#[crate::host(proxy(host = "livekit.agents.stt.STT", base = "stt.py", generic))]
pub trait Stt: Send + Sync {
    /// The host object's identity, independent of proxy instances and bound connection options;
    /// `None` for a native provider.
    #[host(read_once, name = "_core_instance_id")]
    fn instance_id(&self) -> Option<usize> {
        None
    }

    /// The name logs and metrics give this instance: its class path.
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> String;

    /// What this instance supports, fixed at construction.
    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> SttCapabilities;

    /// The model and provider labelling this instance's turn metrics.
    // `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// Open the connection to the service early; a provider with nothing to warm ignores it.
    #[host(notify, name = "_core_prewarm")]
    fn prewarm(&self) {}

    /// Set the session's keyterms: its configured ones and those confirmed in the conversation,
    /// kept apart from the user's own. Providers with the `keyterms` capability override it.
    ///
    /// Args:
    /// - keyterms: Every keyterm the session holds now.
    // The default drops it; an adapter forwards it.
    #[host(notify, name = "_core_update_keyterms")]
    fn update_keyterms(&self, _keyterms: &[String]) {}

    /// Feed a conversation turn to bias recognition. Providers with the `chat_context`
    /// capability override it.
    ///
    /// Args:
    /// - role: Who spoke the turn: ``"user"`` or ``"assistant"``.
    /// - text: What was said.
    // The default drops it; an adapter forwards it.
    #[host(notify, name = "_core_push_conversation_item")]
    fn push_conversation_item(&self, _role: &str, _text: &str) {}

    // The framework's report of a failed attempt of a call it retries: the provider's `error`
    // event, `recoverable` while the call is retried. A wrapper forwards it.
    #[host(notify, name = "_core_failed")]
    fn failed(&self, _error: ProviderError, _recoverable: bool) {}

    // A view of this provider whose calls carry `conn` -- how a fallback chain gives each entry
    // its per-attempt retry budget and timeout. `None`: the provider takes no per-call options,
    // and the chain uses it as it is. A host STT is retried by the core (`RetryingStt`).
    #[host(bind(wrap = lk_agents_runtime::inference::stt::retrying::RetryingStt))]
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Stt>> {
        None
    }

    // The connection policy this provider was given at construction, which applies to the
    // calls of a session or run that supplies none (models.md); `None`: the defaults.
    #[host(read_once, name = "_core_conn_options")]
    fn conn_options(&self) -> Option<ApiConnectOptions> {
        None
    }

    /// Recognize a recording, its attempts retried per `conn_options`.
    ///
    /// Args:
    /// - buffer: The audio to recognize.
    /// - language: The language to recognize; the provider's default when not given.
    /// - conn_options: The timeout and retries of the request.
    // Always present. `language: None` is `NOT_GIVEN`, the provider's default.
    #[host(await, name = "_core_recognize")]
    fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> crate::BoxFuture<'_, Result<SpeechEvent, ProviderError>>;

    /// Open a stream recognizing live audio. Providers with the `streaming` capability override
    /// it; any other is wrapped in a `StreamAdapter`.
    ///
    /// Args:
    /// - language: The language to recognize; the provider's default when not given.
    /// - conn_options: The timeout and retries of the connection.
    // The duplex path, if this instance has one. `Ok(None)` selects the STT stream adapter (VAD +
    // batch `recognize`); `Err` is a construction failure. `async` as on [`Tts`](crate::Tts).
    //
    // `start_time_offset`: the stream's `RecognizeStream.start_time_offset`, in seconds -- where
    // its timestamps sit on the caller's input timeline (a fallback leg opened mid-input). `None`
    // leaves the stream's own (`0`). `trace`: the span the stream runs under (the activity's, as
    // the reference's STT pump); a node or an adapter that records spans opens them there.
    #[host(await, name = "_core_stream")]
    fn stream(
        &self,
        language: Option<&LanguageCode>,
        start_time_offset: Option<f64>,
        #[host(context)] trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'_, Result<Option<crate::DuplexStream<SttSink, SttSource>>, ProviderError>>;
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "stt_availability", path = "livekit.agents.stt", dataclass))]
pub struct AvailabilityChangedEvent {
    #[host(annotation = ".stt.STT")]
    pub stt: crate::HostRef,
    pub available: bool,
}

/// The public provider error event; its Python exception is retained but never serialized.
#[crate::host(record(
    module = "stt",
    name = "STTError",
    path = "livekit.agents.stt",
    pydantic,
    arbitrary_types,
    constant(type = "stt_error")
))]
pub struct SttErrorEvent {
    pub timestamp: f64,
    pub label: String,
    #[host(annotation = "typing.Annotated[Exception, pydantic.Field(exclude=True)]")]
    pub error: crate::HostRef,
    pub recoverable: bool,
}
