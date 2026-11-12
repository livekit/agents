/// A transcript word or chunk with optional alignment, confidence and speaker metadata.
///
/// Times are seconds on the owning audio stream's timeline. Stream adapters apply the source
/// offset when normalizing provider events; `start_time_offset` records that offset and must
/// not be added again to normalized start/end times. Missing metadata is `None`.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(
    module = "transcript",
    path = "livekit.agents.types",
    dataclass,
    extract,
    emit
))]
pub struct TranscriptChunk {
    pub text: String,
    /// Start time in seconds on the audio stream's timeline.
    #[host(default = None)]
    pub start_time: Option<f64>,
    /// End time in seconds on the audio stream's timeline.
    #[host(default = None)]
    pub end_time: Option<f64>,
    /// Provider-reported confidence, when available.
    #[host(default = None)]
    pub confidence: Option<f64>,
    /// The source offset already applied by the stream adapter.
    #[host(default = None)]
    pub start_time_offset: Option<f64>,
    #[host(default = None)]
    pub speaker_id: Option<String>,
    /// Whether the voice says this text. `False` is text the transcript shows with no speech
    /// (markdown syntax, an emoji the voice skips, code, a link's URL): shown as soon as what comes
    /// before it was heard, and never paced.
    #[host(default = true)]
    pub spoken: bool,
    /// The delivery the model asked for from here on (an expression marker's label): what the
    /// room publishes as `lk.expression`. The marker itself never reaches a transcript.
    #[host(default = None)]
    pub expression: Option<String>,
}

impl Default for TranscriptChunk {
    fn default() -> Self {
        TranscriptChunk {
            text: String::new(),
            start_time: None,
            end_time: None,
            confidence: None,
            start_time_offset: None,
            speaker_id: None,
            spoken: true,
            expression: None,
        }
    }
}

impl TranscriptChunk {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), ..Default::default() }
    }

    /// Text the transcript shows with no speech.
    pub fn silent(text: impl Into<String>) -> Self {
        Self { text: text.into(), spoken: false, ..Default::default() }
    }
}
