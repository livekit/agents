#![cfg_attr(
    feature = "python",
    allow(
        clippy::disallowed_types,
        reason = "S0: generated host siblings need module scope; exact type sites are ratcheted."
    )
)]

//! `Stt`: the abstract path is batch (`stt/stt.py:204`); `stream()` is optional. `recognize` is
//! always callable; `SttCapabilities::offline_recognize` says whether it works.

use crate::SpeechEvent;
use crate::tokenize::language::LanguageCode;
use async_trait::async_trait;
use crate::errors::ProviderError;
use lk_agents_host_derive::Host;

use crate::{AudioFrame, ApiConnectOptions};
use crate::utils::stream::{StreamSink, StreamSource};

/// `Literal["word", "chunk", False]`. `None` is the boolean `False` on the host, never a string.
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

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for AlignedTranscript {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<pyo3::Bound<'py, pyo3::PyAny>, std::convert::Infallible> {
        Ok(match self {
            AlignedTranscript::None => pyo3::types::PyBool::new(py, false).to_owned().into_any(),
            other => pyo3::types::PyString::new(py, other.as_str()).into_any(),
        })
    }
}

/// Anything but a known word (`False` included) is none, like `isinstance(x, str)` host-side.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for AlignedTranscript {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(obj.extract::<String>().ok().and_then(|s| s.parse().ok()).unwrap_or_default())
    }
}

// Field `///`s are the reference's. The dict is generated because a forgotten key would silently
// default `offline_recognize` to `True`. `impl Default` and `#[python(default)]` agree on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
// The extract direction reads a plugin's `stt.STTCapabilities` attribute by attribute.
#[binding(emit, extract)]
#[python(dataclass, module = "stt", path = "livekit.agents.stt", name = "STTCapabilities")]
pub struct SttCapabilities {
    pub streaming: bool,
    pub interim_results: bool,
    #[python(default = "False")]
    pub diarization: bool,
    // `#[python(ty)]` because `False` is neither `"none"` nor `None`; the gate compares annotation
    // text.
    #[python(ty = "Literal[\"word\", \"chunk\", False]", default = "False")]
    pub aligned_transcript: AlignedTranscript,
    /// Whether the STT supports batch recognition via recognize() method
    #[python(default = "True")]
    pub offline_recognize: bool,
    /// Whether the STT supports keyterm prompting
    #[python(default = "False")]
    pub keyterms: bool,
    /// Whether the STT can natively consume conversation context (see STT._push_conversation_item)
    #[python(default = "False")]
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
#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
/// Drained by `FallbackSttStream`: every frame fans out to the serving stream and recovery probes.
#[crate::host(stream(name = "FallbackSttStream", doc = "One `RecognizeStream`'s worth of the chain: push audio in, pull `SpeechEvent`s out.", sink = dyn lk_agents_abi::StreamSink<lk_agents_abi::AudioFrame>,
    push = "push_frame", input = AudioFrame,
    input_py = "rtc.AudioFrame", stub_import = "from livekit import rtc", end_input,
    output = SpeechEvent))]
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
#[async_trait]
pub trait Stt: Send + Sync {
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

    /// Recognize a recording, its attempts retried per `conn_options`.
    ///
    /// Args:
    /// - buffer: The audio to recognize.
    /// - language: The language to recognize; the provider's default when not given.
    /// - conn_options: The timeout and retries of the request.
    // Always present. `language: None` is `NOT_GIVEN`, the provider's default.
    #[python(hook = "recognize", impl = "_recognize_impl", retry = "recognize")]
    #[host(await, name = "_core_recognize")]
    async fn recognize(
        &self,
        #[python(ty = "AudioBuffer", import = "..utils:AudioBuffer")] buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError>;

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
    async fn stream(
        &self,
        language: Option<&LanguageCode>,
        #[python(attr)] start_time_offset: Option<f64>,
    ) -> Result<Option<(SttSink, SttSource)>, ProviderError>;
}
