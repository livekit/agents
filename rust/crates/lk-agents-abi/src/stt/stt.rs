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
#[python(dataclass, module = "stt", path = "livekit.agents.stt.stt", name = "STTCapabilities")]
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
/// Drained by `FallbackSttStream`: every frame fans out to the serving stream and recovery probes.
#[crate::host(stream(name = "FallbackSttStream", doc = "One `RecognizeStream`'s worth of the chain: push audio in, pull `SpeechEvent`s out.", sink = dyn lk_agents_abi::StreamSink<lk_agents_abi::AudioFrame>,
    push = "push_frame", input = AudioFrame,
    input_py = "rtc.AudioFrame", stub_import = "from livekit import rtc", end_input,
    output = SpeechEvent))]
pub type SttSource = Box<dyn StreamSource<SpeechEvent>>;

/// A speech-to-text provider. A host `stt.STT` is this trait through its bare `#[host(proxy)]`
/// (`PyStt`): each method calls the base class's `_core_<method>`, which keeps the plugin API's
/// quirks (`_recognize_impl`, `stream()`'s `NotImplementedError`, `conn_options`) host-side.
#[crate::host(proxy(host = "livekit.agents.stt.STT", clock))]
#[async_trait]
pub trait Stt: Send + Sync {
    fn label(&self) -> String;

    /// Read per instance, at construction.
    fn capabilities(&self) -> SttCapabilities;

    /// `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// `prewarm()`: open the connection pool early; a provider with nothing to warm ignores it.
    fn prewarm(&self) {}

    /// `STT._update_session_keyterms(keyterms)`: the session's static plus confirmed detected
    /// keyterms. The default drops it; an adapter forwards it.
    fn update_keyterms(&self, _keyterms: &[String]) {}

    /// A view of this provider whose calls carry `conn` -- how a fallback chain gives each entry
    /// its per-attempt retry budget and timeout. `None`: the provider takes no per-call options,
    /// and the chain uses it as it is.
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Stt>> {
        None
    }

    /// `_recognize_impl`, always present. `language: None` is `NOT_GIVEN`, the provider's default.
    async fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError>;

    /// The duplex path, if this instance has one. `None` selects the STT stream adapter (VAD +
    /// batch `recognize`); `Some(Err)` is a construction failure. `async` as on [`Tts`](crate::Tts).
    ///
    /// `start_time_offset`: the stream's `RecognizeStream.start_time_offset`, in seconds -- where
    /// its timestamps sit on the caller's input timeline (a fallback leg opened mid-input). `None`
    /// leaves the stream's own (`0`).
    async fn stream(
        &self,
        language: Option<&LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> Option<Result<(SttSink, SttSource), ProviderError>>;
}
