//! `Tts`: `synthesize()` is universal; `stream()` is optional per instance (`tts/tts.py:229`), so
//! non-streaming ones get [`StreamAdapter`](crate::StreamAdapter).

use crate::TimedString;
use async_trait::async_trait;
use crate::errors::{ApiError, ProviderError};
use lk_agents_host_derive::Host;

use crate::{AudioFrame, ApiConnectOptions};
use crate::utils::stream::{StreamSink, StreamSource};

/// `TTSCapabilities` (`tts/tts.py`).
///
/// The Python class stays hand-written; the payload `FallbackTts`'s `capabilities` getter builds by name
/// is generated, since a forgotten key would silently take the reference's default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
// Both directions.
#[binding(emit, extract)]
pub struct TtsCapabilities {
    /// Whether this instance has a duplex path; read at construction (some plugins compute it).
    pub streaming: bool,
    pub aligned_transcript: bool,
}

/// `tts.SynthesizedAudio` (`tts/tts.py:39-51`) plus the two fields a Python plugin carries in
/// `rtc.AudioFrame.userdata` (`USERDATA_TIMED_TRANSCRIPT`, `USERDATA_TTS_STARTED_TIME`), lifted by
/// the host class's `_core_*` views. Not what `Agent.tts_node` yields (a bare frame).
#[derive(Debug, Clone, PartialEq, Host)]
// Read by hand (below): the host class's own views of the frame's `userdata`.
#[binding(emit)]
pub struct SynthesizedAudio {
    /// A real `rtc.AudioFrame` both ways: [`AudioFrame`]'s twin is `construct`.
    pub frame: AudioFrame,
    /// One segment can be made up of several requests.
    pub request_id: String,
    /// Last frame of the segment.
    pub is_final: bool,
    /// Segments are separated by a flush. Streaming only, `""` otherwise.
    pub segment_id: String,
    /// Streaming only, `""` otherwise.
    pub delta_text: String,
    /// `USERDATA_TTS_STARTED_TIME`: `perf_counter()` when the text was first sent to the provider.
    /// `None` when absent (`generation.py:426` branches on it); `0.0` is legitimate.
    /// Read off `_core_started_time`, the host class's view of the frame's `userdata`.
    pub started_time: Option<f64>,
    /// `USERDATA_TIMED_TRANSCRIPT`: the words this frame covers. Empty is the common case.
    /// Host `TimedString`s both ways; read off `_core_timed_transcript` (the `userdata` list, words
    /// only -- a plain `str` there is skipped, as `generation.py:432-434` does).
    pub timed_transcript: Vec<TimedString>,
    /// What `frame.data` holds. Host plugins always deliver PCM, so it does not cross.
    #[binding(skip, extract_skip)]
    pub encoding: AudioEncoding,
}

/// `started_time` and `timed_transcript` come off `_core_started_time` / `_core_timed_transcript`.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for SynthesizedAudio {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{attr, attr_or};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "SynthesizedAudioFromPy";
        let obj = &*obj;
        Ok(SynthesizedAudio {
            frame: attr(obj, OWNER, "frame")?,
            request_id: attr(obj, OWNER, "request_id")?,
            is_final: attr(obj, OWNER, "is_final")?,
            segment_id: attr(obj, OWNER, "segment_id")?,
            delta_text: attr(obj, OWNER, "delta_text")?,
            started_time: attr_or(obj, OWNER, "_core_started_time", |v| v.extract(), || None)?,
            timed_transcript: attr_or(obj, OWNER, "_core_timed_transcript", |v| v.extract(), Vec::new)?,
            encoding: AudioEncoding::default(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AudioEncoding {
    /// Interleaved 16-bit little-endian PCM.
    #[default]
    Pcm,
    /// Decoded once where the provider enters the engine.
    Encoded { mime: String },
}

impl SynthesizedAudio {
    pub fn new(frame: AudioFrame, request_id: impl Into<String>) -> Self {
        SynthesizedAudio {
            frame,
            request_id: request_id.into(),
            is_final: false,
            segment_id: String::new(),
            delta_text: String::new(),
            started_time: None,
            timed_transcript: Vec::new(),
            encoding: AudioEncoding::Pcm,
        }
    }
}

/// May not drop: a dropped token silently mangles speech.
pub type TtsSink = Box<dyn StreamSink<String>>;
/// Both TTS paths yield this. Drained by the gateway's `InferenceTtsStream` and the fallback
/// chain's `FallbackTtsStream` (no sink for a one-shot `synthesize()`).
#[crate::host(stream(name = "InferenceTtsStream", doc = "One live synthesis.", sink = dyn lk_agents_abi::StreamSink<String>,
    input = String, output = SynthesizedAudio,
    end_input, close_ends_input))]
#[crate::host(stream(name = "FallbackTtsStream", doc = "One synthesis, served by whichever entry the chain picks.", sink = dyn lk_agents_abi::StreamSink<String>,
    push = "push_text", arg = "token", input = String, end_input,
    optional_sink = "this is a synthesize() stream; its text was given at the call",
    output = SynthesizedAudio))]
pub type TtsSource = Box<dyn StreamSource<SynthesizedAudio>>;

/// The factory methods are `async`: base-class stream constructors call `asyncio.create_task`,
/// which needs the host loop's thread, so a host provider awaits them there.
///
/// A host `tts.TTS` is this trait through its bare `#[host(proxy)]` (`PyTts`).
#[crate::host(proxy(host = "livekit.agents.tts.TTS", clock))]
#[async_trait]
pub trait Tts: Send + Sync {
    fn label(&self) -> String;

    /// Read per instance, at construction. Not a class property.
    fn capabilities(&self) -> TtsCapabilities;

    /// `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// `prewarm()`: open the connection pool early; a provider with nothing to warm ignores it.
    fn prewarm(&self) {}

    /// The provider's native output rate, which the driver resamples from.
    fn sample_rate(&self) -> u32;
    fn num_channels(&self) -> u16;

    /// A view of this provider whose calls carry `conn`; see [`Stt::with_conn_options`](crate::Stt::with_conn_options).
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Tts>> {
        None
    }

    /// Always present: `synthesize()` is the abstract method.
    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError>;

    /// The duplex path.
    ///
    /// An `Err` is a failed attempt the engine's retry budget handles; nothing downgrades silently.
    /// A provider without one answers [`no_duplex_path`] and is wrapped in
    /// [`StreamAdapter`](crate::StreamAdapter) where it enters the engine.
    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError>;
}

/// Non-retryable: wrap it in [`StreamAdapter`](crate::StreamAdapter) instead.
pub fn no_duplex_path(label: &str) -> ProviderError {
    ProviderError::Local(ApiError {
        retryable: false,
        ..ApiError::new(format!("{label} has no duplex path; wrap it in StreamAdapter"))
    })
}
