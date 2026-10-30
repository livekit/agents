//! `Stt`: the abstract path is batch (`stt/stt.py:204`); `stream()` is optional. `recognize` is
//! always callable; `SttCapabilities::offline_recognize` says whether it works.

use crate::SpeechEvent;
use crate::errors::ProviderError;
use crate::tokenize::language::LanguageCode;

use crate::utils::stream::{StreamSink, StreamSource};
use crate::{ApiConnectOptions, AudioFrame};

/// `Literal["word", "chunk", False]`. `None` is the boolean `False` on the host, never a string;
/// anything but a known word (`False` included) reads as none, like `isinstance(x, str)` host-side.
#[crate::host(handwritten(extract, emit, host = "typing.Literal[\"word\", \"chunk\", False]", py))]
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

/// May drop: the driver feeds it from a `BoundedLossy`.
pub type SttSink = Box<dyn StreamSink<AudioFrame>>;
/// What a recognition stream yields: its speech events.
pub type SttSource = Box<dyn StreamSource<SpeechEvent>>;

// A host `stt.STT` is this trait through its generated proxy (`SttProxy`), calling the `_core_*`
// methods of the base class `lk-agents-codegen` writes from the `#[python]` markers, which keeps
// the plugin API's shape (`_recognize_impl`, `conn_options`) there.
/// A speech-to-text provider.
///
/// Plugins implement `_recognize_impl` for batch recognition, and override `stream()` when the
/// service recognizes live audio.
#[crate::host(proxy(host = "livekit.agents.stt.STT"))]
#[python(provider(
    module = "stt",
    path = "livekit.agents.stt",
    events = "metrics_collected, error",
    error(name = "STTError", kind = "stt_error")
))]
pub trait Stt: Send + Sync {
    /// The host object's identity, independent of proxy instances and bound connection options;
    /// `None` for a native provider.
    #[python(value = "id(self)")]
    #[host(read_once, name = "_core_instance_id")]
    fn instance_id(&self) -> Option<usize> {
        None
    }

    /// The name logs and metrics give this instance: its class path.
    #[python(label)]
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> String;

    /// What this instance supports, fixed at construction.
    #[python(init)]
    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> SttCapabilities;

    /// The model and provider labelling this instance's turn metrics.
    // `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    #[python(identity)]
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// Open the connection to the service early; a provider with nothing to warm ignores it.
    #[python(hook)]
    #[host(notify, name = "_core_prewarm")]
    fn prewarm(&self) {}

    /// Set the session's keyterms: its configured ones and those confirmed in the conversation,
    /// kept apart from the user's own. Providers with the `keyterms` capability override it.
    ///
    /// Args:
    /// - keyterms: Every keyterm the session holds now.
    // The default drops it; an adapter forwards it.
    #[python(hook = "_update_session_keyterms")]
    #[host(notify, name = "_core_update_keyterms")]
    fn update_keyterms(&self, _keyterms: &[String]) {}

    /// Feed a conversation turn to bias recognition. Providers with the `chat_context`
    /// capability override it.
    ///
    /// Args:
    /// - role: Who spoke the turn: ``"user"`` or ``"assistant"``.
    /// - text: What was said.
    // The default drops it; an adapter forwards it.
    #[python(hook = "_push_conversation_item")]
    #[host(notify, name = "_core_push_conversation_item")]
    fn push_conversation_item(&self, _role: &str, _text: &str) {}

    // The framework's report of a failed attempt of a call it retries: the provider's `error`
    // event, `recoverable` while the call is retried. A wrapper forwards it.
    #[python(report)]
    #[host(notify, name = "_core_failed")]
    fn failed(&self, _error: ProviderError, _recoverable: bool) {}

    // A view of this provider whose calls carry `conn` -- how a fallback chain gives each entry
    // its per-attempt retry budget and timeout. `None`: the provider takes no per-call options,
    // and the chain uses it as it is. A host STT is retried by the core (`RetryingStt`).
    #[python(conn)]
    #[host(bind(wrap = lk_agents_runtime::inference::stt::retrying::RetryingStt))]
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Stt>> {
        None
    }

    // The connection policy this provider was given at construction, which applies to the
    // calls of a session or run that supplies none (models.md); `None`: the defaults.
    #[python(value = "None")]
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
    #[python(hook = "recognize", impl = "_recognize_impl", retry = "recognize")]
    #[host(await, name = "_core_recognize")]
    fn recognize(
        &self,
        #[python(ty = "AudioBuffer", import = "..utils:AudioBuffer")] buffer: &[AudioFrame],
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
    // leaves the stream's own (`0`).
    #[python(
        hook,
        when = "streaming",
        unsupported = "streaming is not supported by this STT, please use a different STT or use a StreamAdapter",
        stream(
            name = "RecognizeStream",
            kind = "recognize",
            alias = "SpeechStream",
            doc = "A live recognition: audio in through ``push_frame``, ``SpeechEvent`` out.\n\nPlugins implement ``_run``, reading ``input`` and handing each event to ``send``."
        )
    )]
    #[host(await, name = "_core_stream")]
    fn stream(
        &self,
        language: Option<&LanguageCode>,
        #[python(attr)] start_time_offset: Option<f64>,
    ) -> crate::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>>;
}
