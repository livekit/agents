//! Ordered transcript synchronization work and private pacing results.
use super::Heard;
use super::machine::lifecycle::TranscriptOutput;
use super::machine::pacer::TranscriptPacerWarning;
use lk_agents_abi::{TranscriptChunk, Ts};

/// The transcript synchronizer's ordered runtime work and heard-text reports.
/// Segment matching and per-segment pacing remain private to this owner.
pub enum TranscriptSyncEffect {
    /// A transcript word released against the audio playhead.
    PublishWord {
        word: String,
        start_time: f64,
        end_time: f64,
    },
    FlushTranscript,
    ReportHeard(Heard),
    Warn(TranscriptPacerWarning),
    Capture {
        text: TranscriptChunk,
        aligned: bool,
    },
    FlushText,
    AttachText,
    RetireText {
        admitted: bool,
    },
    Queue(TranscriptOutput),
    CloseSender,
    BeginWait,
    WaitUntil(Ts),
}

#[must_use = "an effect not performed is a word the user never sees"]
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub(in crate::transcription) enum TranscriptPacerEffect {
    /// One raw token for the text output, with where its audio starts and ends.
    Forward {
        word: String,
        start_time: f64,
        end_time: f64,
    },
    Flush,
    /// Once per playout: at its finish, and at each split.
    Heard(Heard),
    /// An event the segment was not in a state to take. Nothing changed.
    Warn(TranscriptPacerWarning),
}
