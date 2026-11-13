//! `Tts`: `synthesize()` is universal; `stream()` is optional per instance (`tts/tts.py:229`), so
//! non-streaming ones get Python’s `tts.StreamAdapter`.

use crate::TranscriptChunk;
use crate::errors::{ApiError, ProviderError};

use crate::utils::stream::{ProviderStream, StreamInput};
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

impl TtsCapabilities {
    /// How a TTS is paced until its options say otherwise: one without streaming is sent one
    /// request per batch (through `StreamAdapter`), and pacing gives each more context; a
    /// streaming one is sent each chunk as it comes, never left idle.
    pub fn default_pacing(&self) -> Option<crate::tts::stream_pacer::SentenceStreamPacer> {
        (!self.streaming).then(crate::tts::stream_pacer::SentenceStreamPacer::default)
    }
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
// Generated conversion includes the two declared paths into the frame's userdata.
#[crate::host(record(
    module = "tts",
    path = "livekit.agents.tts",
    dataclass,
    boundary,
    extract,
    emit
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
    /// Token usage the provider reported for the segment, on its final frame
    #[host(default = None)]
    pub usage: Option<Box<crate::usage::TokenUsage>>,
    /// Characters the provider billed for the segment, on its final frame
    #[host(default = None)]
    pub characters_count: Option<u64>,
    /// The provider's request ids learnt since the previous frame, in the order learnt
    #[host(default_factory = "list")]
    pub provider_request_ids: Vec<String>,
    // `USERDATA_TTS_STARTED_TIME`: when the text was first sent to the provider, in wall-clock
    // seconds (a host's `perf_counter()` stamp is converted where it is read). `None` when absent
    // (`generation.py:426` branches on it); `0.0` is legitimate.
    #[host(
        hidden,
        source(
            attrs(frame, userdata),
            constant = "livekit.agents.types.USERDATA_TTS_STARTED_TIME"
        ),
        extract_with = "py_values::perf_to_wall",
        emit_with = "py_values::wall_to_perf",
        omit_none
    )]
    pub started_time: Option<f64>,
    // The typed transcript chunks carried by this audio frame. The whole list is converted;
    // malformed entries raise instead of silently disappearing.
    #[host(
        hidden,
        source(
            attrs(frame, userdata),
            constant = "livekit.agents.types.USERDATA_TIMED_TRANSCRIPT"
        ),
        omit_empty
    )]
    pub timed_transcript: Vec<TranscriptChunk>,
    // What `frame.data` holds. Host plugins always deliver PCM, so it does not cross.
    #[host(skip)]
    pub encoding: AudioEncoding,
    // The connection a native provider acquired, on the first frame it synthesized over it.
    #[host(skip)]
    pub connection: Option<crate::metrics::base::ConnectionAcquired>,
}

/// `tts.TTSOptions`: how the framework prepares a TTS's text.
#[crate::host(record(
    module = "tts",
    path = "livekit.agents.tts",
    name = "TTSOptions",
    dataclass,
    extract
))]
#[derive(Clone)]
pub struct TtsOptions {
    /// Splits the text into the chunks the TTS is sent: the built-in ``SentenceTokenizer`` by
    /// default, with the TTS's own tags kept whole and its request limit applied on top. A TTS
    /// that takes words is sent words whatever this is.
    #[host(
        default_factory = "..tokenize.SentenceTokenizer",
        annotation = "..tokenize.SentenceTokenizer"
    )]
    pub sentence_tokenizer: crate::tokenize::bindings::SentenceTokenizerLeg,
    /// A ``SentenceStreamPacer`` holds later chunks until the audio plays out, for more context
    /// per request; ``None`` sends each as soon as it is split. A TTS without streaming is paced
    /// by default, a streaming one is not.
    #[host(default = None)]
    pub pacing: Option<crate::tts::stream_pacer::SentenceStreamPacer>,
}

impl Default for TtsOptions {
    /// `SentenceTokenizer()`, unpaced.
    fn default() -> Self {
        let settings = crate::tokenize::bindings::SentenceSettings::default();
        TtsOptions {
            sentence_tokenizer: crate::tokenize::bindings::SentenceTokenizerLeg::Native(
                std::sync::Arc::new(settings),
            ),
            pacing: None,
        }
    }
}

/// A TTS's options as each synthesis over it reads them: the value current when its stage opens.
#[crate::host(native(refused = "expected a `_core.TtsOptionsCell`"))]
pub trait SharedTtsOptions: Send + Sync {
    fn current(&self) -> TtsOptions;
}

/// Options that never change: a native TTS's own.
impl SharedTtsOptions for TtsOptions {
    fn current(&self) -> TtsOptions {
        self.clone()
    }
}

/// `_core.TtsOptionsCell`: the options a `tts.TTS` owns (`tts.options`), shared with every
/// synthesis over it. A synthesis reads them when its stage opens; one already open keeps them.
#[crate::host(class(name = "TtsOptionsCell", native = SharedTtsOptions))]
pub trait TtsOptionsCell: SharedTtsOptions + Sized + 'static {
    /// Split by `sentence_tokenizer`, paced as a TTS with `capabilities` is by default.
    #[host(new)]
    fn new(sentence_tokenizer: OptionsTokenizer, capabilities: TtsCapabilities) -> Self;
    /// The tokenizer as it was given; `None` only for the options of a TTS no host built.
    #[host(getter)]
    fn sentence_tokenizer(&self) -> Option<crate::HostRef>;
    #[host(getter)]
    fn pacing(&self) -> Option<crate::tts::stream_pacer::SentenceStreamPacer>;
    /// Replace both at once.
    #[host(call)]
    fn set(
        &self,
        sentence_tokenizer: OptionsTokenizer,
        pacing: Option<crate::tts::stream_pacer::SentenceStreamPacer>,
    );
}

/// A host `tokenize.SentenceTokenizer` as a TTS's options hold it: the object itself, read back as
/// `options.sentence_tokenizer`, and how the stage splits by it.
#[crate::host(record(module = "tts", host_type = "builtins.object", boundary, extract))]
#[derive(Clone)]
pub struct OptionsTokenizer {
    #[host(own_value, extract_with = "py_values::host_ref")]
    pub host: crate::HostRef,
    #[host(own_value)]
    pub leg: crate::tokenize::bindings::SentenceTokenizerLeg,
}

/// What each chunk of a TTS stream's input holds.
#[crate::host(string)]
#[crate::host(wire_strs(msg = format!("unknown text unit {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextUnit {
    /// Whole sentences, batched to a spoken length.
    #[default]
    Sentences,
    /// Words, for a provider that buffers text itself; an open wrapping tag holds its words.
    Words,
    /// The text as pushed, for a TTS that prepares it per provider itself (a fallback chain):
    /// each chunk's `text` is as written and its `said` empty, the words known only downstream.
    Text,
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
            usage: None,
            characters_count: None,
            provider_request_ids: Vec::new(),
            started_time: None,
            timed_transcript: Vec::new(),
            encoding: AudioEncoding::Pcm,
            connection: None,
        }
    }
}

/// One chunk of a TTS's input, as the stage prepared it.
#[crate::host(record(module = "tts", path = "livekit.agents.tts", dataclass, emit))]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpeechChunk {
    /// What the provider is sent: its markup in the provider's syntax.
    pub text: String,
    /// The words it says, without markup: what its audio is aligned to.
    pub said: String,
}

/// A TTS's input: the chunks its stage prepared. May not drop: a lost chunk mangles speech.
pub type TtsSink = Box<dyn StreamInput<SpeechChunk>>;
/// A synthesis request's input: the text as written, which its stage prepares.
pub type TextSink = Box<dyn StreamInput<String>>;
/// Both TTS paths yield this, drained by a `TtsStream` (no sink for a one-shot `synthesize()`).
pub type TtsSource = Box<dyn ProviderStream<SynthesizedAudio>>;

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

    /// Open the connection to the service early, for the job `cx` runs for; a provider with
    /// nothing to warm ignores it.
    #[host(notify, name = "_core_prewarm")]
    fn prewarm(&self, _cx: &crate::telemetry::Cx) {}

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

    // What each chunk of a stream's input holds: a provider that buffers text itself takes words.
    #[host(read_once, name = "_core_text_unit")]
    fn text_unit(&self) -> TextUnit {
        TextUnit::Sentences
    }

    // Whether it retries its own attempts within the connection policy it is given (a stream
    // adapter retries each chunk, a fallback chain its members): a request over it runs once and
    // hands it the whole policy. `_core_native_retries`, for a TTS.
    #[host(read_once, name = "_core_retries_itself")]
    fn retries_itself(&self) -> bool {
        false
    }

    // The most characters one request may carry; `None` takes its dialect's known limit, else
    // the stage's default ceiling.
    #[host(read_once, name = "_core_max_text_len")]
    fn max_text_len(&self) -> Option<usize> {
        None
    }

    // How its text is split and paced: `tts.options`, which it shares with every synthesis over
    // it; each reads the current value when its stage opens.
    #[host(read_once, name = "_core_options")]
    fn options(&self) -> std::sync::Arc<dyn SharedTtsOptions> {
        std::sync::Arc::new(TtsOptions {
            pacing: self.capabilities().default_pacing(),
            ..TtsOptions::default()
        })
    }

    // A view of this provider whose calls carry `conn`; see [`Stt::with_conn_options`](crate::Stt::with_conn_options).
    #[host(bind)]
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Tts>> {
        None
    }

    // The connection policy this provider was given at construction, which applies to the
    // calls of a session or run that supplies none (models.md); a bound view's is its binding;
    // `None`: the defaults.
    #[host(read_once, name = "_core_conn_options")]
    fn conn_options(&self) -> Option<ApiConnectOptions> {
        None
    }

    // The framework's report of a failed attempt of a call it retries: the provider's `error`
    // event, `recoverable` while the call is retried. A wrapper forwards it.
    #[host(notify, name = "_core_failed")]
    fn failed(&self, _error: ProviderError, _recoverable: bool) {}

    // Native only: what a request to this TTS records as.
    fn request_span(&self) -> crate::telemetry::spans::inference::RequestSpan {
        crate::telemetry::spans::inference::RequestSpan::Request
    }

    /// One provider attempt at a whole text, as the stage prepared it.
    // Always present: `_synthesize()` is the abstract method.
    #[host(await, name = "_core_synthesize")]
    fn synthesize<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        chunk: &'a SpeechChunk,
    ) -> crate::BoxFuture<'a, Result<TtsSource, ProviderError>>;

    /// Open a stream synthesizing text as it is pushed. Providers with the `streaming`
    /// capability override it; any other is wrapped in a `StreamAdapter`.
    ///
    /// Args:
    /// - conn_options: The timeout and retries of the connection.
    // The duplex path. An `Err` is a failed attempt the engine's retry budget handles; nothing
    // downgrades silently. A provider without one answers [`no_duplex_path`] and is wrapped in
    // Python’s `tts.StreamAdapter` where it enters the engine.
    // Its input is the chunks its stage prepared; `cx` is the span the request runs under, where
    // an adapter that records its own spans opens them.
    #[host(await, name = "_core_stream")]
    fn stream(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<crate::DuplexStream<TtsSink, TtsSource>, ProviderError>>;
}

/// Non-retryable: wrap it in Python’s `tts.StreamAdapter` instead.
pub fn no_duplex_path(label: &str) -> ProviderError {
    ProviderError::Local(ApiError {
        retryable: false,
        ..ApiError::new(format!("{label} has no duplex path; wrap it in StreamAdapter"))
    })
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "tts_availability", path = "livekit.agents.tts", dataclass))]
pub struct AvailabilityChangedEvent {
    #[host(annotation = ".tts.TTS")]
    pub tts: crate::HostRef,
    pub available: bool,
}

/// The public provider error event; its Python exception is retained but never serialized.
#[crate::host(record(
    module = "tts",
    name = "TTSError",
    path = "livekit.agents.tts",
    pydantic,
    arbitrary_types,
    constant(type = "tts_error")
))]
pub struct TtsErrorEvent {
    pub timestamp: f64,
    pub label: String,
    #[host(annotation = "typing.Annotated[Exception, pydantic.Field(exclude=True)]")]
    pub error: crate::HostRef,
    pub recoverable: bool,
}
