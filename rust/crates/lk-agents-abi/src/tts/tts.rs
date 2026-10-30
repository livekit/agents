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
use crate::errors::{ApiError, ProviderError};

use crate::utils::stream::{StreamSink, StreamSource};
use crate::{ApiConnectOptions, AudioFrame};

// `TTSCapabilities` (`tts/tts.py`). The payload `FallbackTts`'s `capabilities` getter builds by
// name is generated too, since a forgotten key would silently take the reference's default.
#[crate::host(record(
    module = "tts",
    path = "livekit.agents.tts",
    name = "TTSCapabilities",
    dataclass,
    extract,
    emit,
))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TtsCapabilities {
    /// Whether this TTS supports streaming (generally using websockets)
    // Whether this instance has a duplex path; read at construction (some plugins compute it).
    pub streaming: bool,
    /// Whether this TTS supports aligned transcripts with word timestamps
    #[host(default = false)]
    pub aligned_transcript: bool,
}

/// What the expressive markup pipeline can do with a given voice.
#[crate::host(record(module = "tts", path = "livekit.agents.tts", dataclass, emit))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MarkupInfo {
    /// ``NonverbalOptions`` field -> the labels it governs; an absent field is a no-op
    #[host(default_factory = "dict")]
    pub nonverbals: std::collections::BTreeMap<crate::tts::markup::NonverbalField, Vec<String>>,
}

// `tts.SynthesizedAudio` (`tts/tts.py:39-51`) plus the two fields a Python plugin carries in
// `rtc.AudioFrame.userdata` (`USERDATA_TIMED_TRANSCRIPT`, `USERDATA_TTS_STARTED_TIME`). Not what
// `Agent.tts_node` yields (a bare frame). The reference has no class docstring.
// Crosses by hand (`conversions::synthesized_audio_{from,to}_py`): two fields are the frame's
// `userdata`.
#[crate::host(record(
    module = "tts",
    path = "livekit.agents.tts",
    dataclass,
    extract(py),
    emit(py)
))]
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesizedAudio {
    /// Synthesized audio frame
    pub frame: AudioFrame,
    /// Request ID (one segment could be made up of multiple requests)
    pub request_id: String,
    /// Whether this is latest frame of the segment
    #[host(default = false)]
    pub is_final: bool,
    /// Segment ID, each segment is separated by a flush (streaming only)
    #[host(default = "")]
    pub segment_id: String,
    /// Current segment of the synthesized audio (streaming only)
    #[host(default = "")]
    pub delta_text: String,
    // `USERDATA_TTS_STARTED_TIME`: when the text was first sent to the provider, in wall-clock
    // seconds (a host's `perf_counter()` stamp is converted where it is read). `None` when absent
    // (`generation.py:426` branches on it); `0.0` is legitimate.
    #[host(skip)]
    pub started_time: Option<f64>,
    // `USERDATA_TIMED_TRANSCRIPT`: the words this frame covers. Empty is the common case. Host
    // `TimedString`s both ways; a plain `str` in the host's list is skipped on the way in, as
    // `generation.py:432-434` does.
    #[host(skip)]
    pub timed_transcript: Vec<TimedString>,
    // What `frame.data` holds. Host plugins always deliver PCM, so it does not cross.
    #[host(skip)]
    pub encoding: AudioEncoding,
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
/// Both TTS paths yield this, drained by a `TtsStream` (no sink for a one-shot `synthesize()`).
pub type TtsSource = Box<dyn StreamSource<SynthesizedAudio>>;

// The factory methods are `async`: base-class stream constructors call `asyncio.create_task`,
// which needs the host loop's thread, so a host provider awaits them there.
//
// A host `tts.TTS` is this trait through its generated proxy (`TtsProxy`), calling the `_core_*`
// methods of its hand-written base class (`tts.py`).
/// A text-to-speech provider.
///
/// Plugins implement `synthesize()` for one-shot synthesis, and override `stream()` when the
/// service synthesizes text as it arrives.
#[crate::host(proxy(host = "livekit.agents.tts.TTS", base = "tts.py", generic))]
pub trait Tts: Send + Sync {
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
    // Read per instance, at construction. Not a class property.
    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> TtsCapabilities;

    /// The model and provider labelling this instance's turn metrics.
    // `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// Open the connection to the service early; a provider with nothing to warm ignores it.
    #[host(notify, name = "_core_prewarm")]
    fn prewarm(&self) {}

    /// The rate of the audio the provider produces, which the framework resamples from.
    #[host(read_once, name = "_core_sample_rate")]
    fn sample_rate(&self) -> u32;

    /// The channels of the audio the provider produces.
    #[host(read_once, name = "_core_num_channels")]
    fn num_channels(&self) -> u16;

    /// Access TTS markup capabilities (instructions for LLM, text stripping).
    // `markup._provider_key()`: the markup dialect this TTS speaks, `""` for none.
    #[host(read_once, name = "_core_markup_provider")]
    fn markup_provider(&self) -> String {
        String::new()
    }

    // Whether the framework lowers this TTS's markup on a duplex path it owns (the gateway's
    // stream, a `StreamAdapter`); a natively streaming plugin owns its input and does not.
    #[host(read_once, name = "_core_lowers_markup")]
    fn lowers_markup(&self) -> bool {
        false
    }

    // A view of this provider whose calls carry `conn`; see [`Stt::with_conn_options`](crate::Stt::with_conn_options).
    #[host(bind)]
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Tts>> {
        None
    }

    // The connection policy this provider was given at construction, which applies to the
    // calls of a session or run that supplies none (models.md); `None`: the defaults.
    #[host(read_once, name = "_core_conn_options")]
    fn conn_options(&self) -> Option<ApiConnectOptions> {
        None
    }

    // The framework's report of a failed attempt of a call it retries: the provider's `error`
    // event, `recoverable` while the call is retried. A wrapper forwards it.
    #[host(notify, name = "_core_failed")]
    fn failed(&self, _error: ProviderError, _recoverable: bool) {}

    /// Synthesize `text`, its attempts retried per `conn_options`.
    ///
    /// Args:
    /// - text: What to say.
    /// - conn_options: The timeout and retries of the request.
    // Always present: `synthesize()` is the abstract method.
    #[host(await, name = "_core_synthesize")]
    fn synthesize<'a>(
        &'a self,
        text: &'a str,
    ) -> crate::BoxFuture<'a, Result<TtsSource, ProviderError>>;

    /// Open a stream synthesizing text as it is pushed. Providers with the `streaming`
    /// capability override it; any other is wrapped in a `StreamAdapter`.
    ///
    /// Args:
    /// - conn_options: The timeout and retries of the connection.
    // The duplex path. An `Err` is a failed attempt the engine's retry budget handles; nothing
    // downgrades silently. A provider without one answers [`no_duplex_path`] and is wrapped in
    // [`StreamAdapter`](crate::StreamAdapter) where it enters the engine.
    //
    // `_expressive`: whether this synthesis carries expressive markup, set on the stream (a TTS
    // tokenizing its own input batches larger chunks when it does).
    #[host(await, name = "_core_stream")]
    fn stream(
        &self,
        _expressive: bool,
    ) -> crate::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>>;
}

/// Non-retryable: wrap it in [`StreamAdapter`](crate::StreamAdapter) instead.
pub fn no_duplex_path(label: &str) -> ProviderError {
    ProviderError::Local(ApiError {
        retryable: false,
        ..ApiError::new(format!("{label} has no duplex path; wrap it in StreamAdapter"))
    })
}
