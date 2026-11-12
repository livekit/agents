use super::Heard;
use super::machine::pacer::TranscriptPacerWarning;
use super::machine::synchronizer::TranscriptOutput;
use lk_agents_abi::voice::io::PlaybackSegment;
use lk_agents_abi::{TranscriptChunk, Ts};

/// The transcript synchronizer's ordered runtime work and heard-text reports.
#[must_use = "an effect not performed is a word the user never sees"]
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptSyncEffect {
    /// A piece of the transcript released against the audio playhead, with where its audio starts
    /// and ends.
    PublishWord {
        chunk: TranscriptChunk,
        start_time: f64,
        end_time: f64,
    },
    FlushTranscript,
    /// What was heard of a playout: once at its finish, and at each split.
    ReportHeard(Heard),
    /// The sink's audio was cleared under `segment`'s playout: what was heard of it by then, as
    /// its finish would report it. It shows no more words; its finish may never come.
    ReportCleared {
        segment: PlaybackSegment,
        heard: Heard,
    },
    /// An event a segment was not in a state to take. Nothing changed.
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
    /// Feed [`TranscriptSyncEvent::Tick`](super::machine::synchronizer::TranscriptSyncEvent::Tick)
    /// once the clock reaches this instant: the playing segment's next word is due. It replaces
    /// a tick still pending, which was asked for later.
    TickAt(Ts),
}
