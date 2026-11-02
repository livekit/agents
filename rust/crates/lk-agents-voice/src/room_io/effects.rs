//! Room publication work from `machine::RoomTranscriptionWriter`, performed by the runtime
//! room transcription output. TranscriptSync owns upstream word pacing, not this protocol.

/// Ordered room publication work requested by [`RoomTranscriptionWriter`](super::machine::RoomTranscriptionWriter).
/// A replacement is its own short stream; everything else closes final.
#[derive(Debug, Clone, PartialEq)]
pub enum RoomTranscriptionEffect {
    /// Open the segment's text stream; only the replacement mode's closing stream opens final.
    OpenStream {
        segment: String,
        is_final: bool,
        expression: Option<(&'static str, String)>,
    },
    Write {
        payload: String,
    },
    CloseStream {
        is_final: bool,
        expression: Option<(&'static str, String)>,
    },
    /// Publish the segment's text so far on the deprecated API.
    PublishLegacy {
        segment: String,
        text: String,
        is_final: bool,
    },
}
