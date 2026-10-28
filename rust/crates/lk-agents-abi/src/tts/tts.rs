#![cfg_attr(
    feature = "python",
    allow(
        clippy::disallowed_types,
        reason = "S0: generated host siblings need module scope; exact type sites are ratcheted."
    )
)]

//! `Tts`: `synthesize()` is universal; `stream()` is optional per instance (`tts/tts.py:229`), so
//! non-streaming ones get [`StreamAdapter`](crate::StreamAdapter).

use crate::TimedString;
use async_trait::async_trait;
use crate::errors::{ApiError, ProviderError};
use lk_agents_host_derive::Host;

use crate::{AudioFrame, ApiConnectOptions};
use crate::utils::stream::{StreamSink, StreamSource};

// `TTSCapabilities` (`tts/tts.py`). The payload `FallbackTts`'s `capabilities` getter builds by
// name is generated too, since a forgotten key would silently take the reference's default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[python(dataclass, module = "tts", path = "livekit.agents.tts", name = "TTSCapabilities", with_bases)]
// Both directions.
#[binding(emit, extract)]
pub struct TtsCapabilities {
    /// Whether this TTS supports streaming (generally using websockets)
    // Whether this instance has a duplex path; read at construction (some plugins compute it).
    pub streaming: bool,
    /// Whether this TTS supports aligned transcripts with word timestamps
    #[python(default = "False")]
    pub aligned_transcript: bool,
}

/// What the expressive markup pipeline can do with a given voice.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(dataclass, module = "tts", path = "livekit.agents.tts", name = "MarkupInfo", with_bases)]
pub struct MarkupInfo {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// ``NonverbalOptions`` field -> the labels it governs; an absent field is a no-op
    #[python(default_factory = "dict")]
    pub nonverbals: std::collections::HashMap<String, Vec<String>>,
}

// `tts.SynthesizedAudio` (`tts/tts.py:39-51`) plus the two fields a Python plugin carries in
// `rtc.AudioFrame.userdata` (`USERDATA_TIMED_TRANSCRIPT`, `USERDATA_TTS_STARTED_TIME`). Not what
// `Agent.tts_node` yields (a bare frame). The reference has no class docstring.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(dataclass, module = "tts", path = "livekit.agents.tts", name = "SynthesizedAudio", with_bases)]
// Read by hand (below): the two fields are the frame's `userdata`.
#[host(handwritten(extract))]
#[binding(emit)]
pub struct SynthesizedAudio {
    /// Synthesized audio frame
    // A real `rtc.AudioFrame` both ways: [`AudioFrame`]'s twin is `construct`.
    #[python(ty = "rtc.AudioFrame", import = "livekit:rtc")]
    pub frame: AudioFrame,
    /// Request ID (one segment could be made up of multiple requests)
    pub request_id: String,
    /// Whether this is latest frame of the segment
    #[python(default = "False")]
    pub is_final: bool,
    /// Segment ID, each segment is separated by a flush (streaming only)
    #[python(default = "\"\"")]
    pub segment_id: String,
    /// Current segment of the synthesized audio (streaming only)
    #[python(default = "\"\"")]
    pub delta_text: String,
    // `USERDATA_TTS_STARTED_TIME`: when the text was first sent to the provider, in wall-clock
    // seconds (a host's `perf_counter()` stamp is converted where it is read). `None` when absent
    // (`generation.py:426` branches on it); `0.0` is legitimate.
    #[python(skip)]
    pub started_time: Option<f64>,
    // `USERDATA_TIMED_TRANSCRIPT`: the words this frame covers. Empty is the common case. Host
    // `TimedString`s both ways; a plain `str` in the host's list is skipped, as
    // `generation.py:432-434` does.
    #[python(skip)]
    pub timed_transcript: Vec<TimedString>,
    // What `frame.data` holds. Host plugins always deliver PCM, so it does not cross.
    #[binding(skip, extract_skip)]
    #[python(skip)]
    pub encoding: AudioEncoding,
}

/// `started_time` and `timed_transcript` come off the frame's `userdata`.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for SynthesizedAudio {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::attr;
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "SynthesizedAudioFromPy";
        let obj = &*obj;
        let py = obj.py();
        let frame = obj.getattr("frame")?;
        let types = py.import("livekit.agents.types")?;
        let userdata = frame.getattr("userdata")?;
        let get = |key: &str| -> pyo3::PyResult<Option<pyo3::Bound<'py, pyo3::PyAny>>> {
            let v = userdata.call_method1("get", (types.getattr(key)?,))?;
            Ok((!v.is_none()).then_some(v))
        };
        // a `perf_counter()` stamp, as wall-clock seconds
        let started_time = match get("USERDATA_TTS_STARTED_TIME")? {
            Some(v) => {
                let started: f64 = v.extract()?;
                let time = py.import("time")?;
                let wall: f64 = time.call_method0("time")?.extract()?;
                let perf: f64 = time.call_method0("perf_counter")?.extract()?;
                Some(wall - (perf - started))
            }
            None => None,
        };
        let timed_transcript = match get("USERDATA_TIMED_TRANSCRIPT")? {
            Some(words) => {
                let word = types.getattr("TimedString")?;
                let words = words.try_iter()?.flatten().filter(|w| w.is_instance(&word).unwrap_or(false));
                words.map(|w| w.extract()).collect::<pyo3::PyResult<_>>()?
            }
            None => Vec::new(),
        };
        Ok(SynthesizedAudio {
            frame: frame.extract()?,
            request_id: attr(obj, OWNER, "request_id")?,
            is_final: attr(obj, OWNER, "is_final")?,
            segment_id: attr(obj, OWNER, "segment_id")?,
            delta_text: attr(obj, OWNER, "delta_text")?,
            started_time,
            timed_transcript,
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
#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
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

// The factory methods are `async`: base-class stream constructors call `asyncio.create_task`,
// which needs the host loop's thread, so a host provider awaits them there.
//
// A host `tts.TTS` is this trait through its generated proxy (`TtsProxy`), calling the `_core_*`
// methods of the base class `lk-agents-codegen` writes from the `#[python]` markers.
/// A text-to-speech provider.
///
/// Plugins implement `synthesize()` for one-shot synthesis, and override `stream()` when the
/// service synthesizes text as it arrives.
#[crate::host(proxy(host = "livekit.agents.tts.TTS"))]
#[python(provider(
    module = "tts",
    path = "livekit.agents.tts",
    events = "metrics_collected, error",
    error(name = "TTSError", kind = "tts_error")
))]
#[async_trait]
pub trait Tts: Send + Sync {
    /// The name logs and metrics give this instance: its class path.
    #[python(label)]
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> String;

    /// What this instance supports, fixed at construction.
    // Read per instance, at construction. Not a class property.
    #[python(init)]
    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> TtsCapabilities;

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

    /// The rate of the audio the provider produces, which the framework resamples from.
    #[python(init)]
    #[host(read_once, name = "_core_sample_rate")]
    fn sample_rate(&self) -> u32;

    /// The channels of the audio the provider produces.
    #[python(init)]
    #[host(read_once, name = "_core_num_channels")]
    fn num_channels(&self) -> u16;

    /// Access TTS markup capabilities (instructions for LLM, text stripping).
    // `markup._provider_key()`: the markup dialect this TTS speaks, `""` for none.
    #[python(markup)]
    #[host(read_once, name = "_core_markup_provider")]
    fn markup_provider(&self) -> String {
        String::new()
    }

    // Whether the framework lowers this TTS's markup on a duplex path it owns (the gateway's
    // stream, a `StreamAdapter`); a natively streaming plugin owns its input and does not.
    #[python(value = "False")]
    #[host(read_once, name = "_core_lowers_markup")]
    fn lowers_markup(&self) -> bool {
        false
    }

    /// Mark whether expressive is active for this turn (set by the framework before each
    /// synthesis); a TTS tokenizing its own input batches larger chunks when it is.
    ///
    /// Args:
    /// - enabled: Whether markup is active for the synthesis about to open.
    #[python(hook = "_set_expressive", store = "_expressive", default = "False")]
    #[host(notify, name = "_core_set_expressive")]
    fn set_expressive(&self, _enabled: bool) {}

    // A view of this provider whose calls carry `conn`; see [`Stt::with_conn_options`](crate::Stt::with_conn_options).
    #[python(conn)]
    #[host(bind)]
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Tts>> {
        None
    }

    // The framework's report of a failed attempt of a call it retries: the provider's `error`
    // event, `recoverable` while the call is retried. A wrapper forwards it.
    #[python(report)]
    #[host(notify, name = "_core_failed")]
    fn failed(&self, _error: ProviderError, _recoverable: bool) {}

    /// Synthesize `text`, its attempts retried per `conn_options`.
    ///
    /// Args:
    /// - text: What to say.
    /// - conn_options: The timeout and retries of the request.
    // Always present: `synthesize()` is the abstract method.
    #[python(
        hook,
        stream(
            name = "ChunkedStream",
            kind = "chunked",
            doc = "One synthesis of a whole text: ``SynthesizedAudio`` out.\n\nPlugins implement ``_run``, pushing the provider's audio into the ``AudioEmitter`` it is given; its attempts are retried per ``conn_options``."
        )
    )]
    #[host(await, name = "_core_synthesize")]
    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError>;

    /// Open a stream synthesizing text as it is pushed. Providers with the `streaming`
    /// capability override it; any other is wrapped in a `StreamAdapter`.
    ///
    /// Args:
    /// - conn_options: The timeout and retries of the connection.
    // The duplex path. An `Err` is a failed attempt the engine's retry budget handles; nothing
    // downgrades silently. A provider without one answers [`no_duplex_path`] and is wrapped in
    // [`StreamAdapter`](crate::StreamAdapter) where it enters the engine.
    #[python(
        hook,
        unsupported = "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter",
        stream(
            name = "SynthesizeStream",
            kind = "synthesize",
            doc = "A live synthesis: text in through ``push_text``, ``SynthesizedAudio`` out.\n\nPlugins implement ``_run``, reading ``input`` and pushing the provider's audio into the ``AudioEmitter`` it is given; its attempts are retried per ``conn_options``."
        )
    )]
    #[host(await, name = "_core_stream")]
    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError>;
}

/// Non-retryable: wrap it in [`StreamAdapter`](crate::StreamAdapter) instead.
pub fn no_duplex_path(label: &str) -> ProviderError {
    ProviderError::Local(ApiError {
        retryable: false,
        ..ApiError::new(format!("{label} has no duplex path; wrap it in StreamAdapter"))
    })
}
