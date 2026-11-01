//! Transcript text and alignment metadata shared by every language binding.

/// A transcript word or chunk with optional alignment, confidence and speaker metadata.
///
/// Times are seconds on the owning audio stream's timeline. Stream adapters apply the source
/// offset when normalizing provider events; `start_time_offset` records that offset and must
/// not be added again to normalized start/end times. Missing metadata is `None`.
#[derive(Debug, Clone, Default, PartialEq)]
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
}

impl TranscriptChunk {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), ..Default::default() }
    }
}
