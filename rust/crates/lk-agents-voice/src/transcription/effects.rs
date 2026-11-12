use super::Heard;
use super::machine::lifecycle::TranscriptOutput;
use super::machine::pacer::TranscriptPacerWarning;
use lk_agents_abi::voice::io::PlaybackSegment;
use lk_agents_abi::{TranscriptChunk, Ts};

/// The transcript synchronizer's ordered runtime work and heard-text reports.
pub enum TranscriptSyncEffect {
    /// A piece of the transcript released against the audio playhead.
    PublishWord {
        chunk: TranscriptChunk,
        start_time: f64,
        end_time: f64,
    },
    FlushTranscript,
    ReportHeard(Heard),
    /// The sink's audio was cleared under `segment`'s playout: what was heard of it by then, as
    /// its finish would report it. It shows no more words; its finish may never come.
    ReportCleared {
        segment: PlaybackSegment,
        heard: Heard,
    },
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
    /// Feed [`TranscriptSyncEvent::Tick`](super::machine::lifecycle::TranscriptSyncEvent::Tick)
    /// once the clock reaches this instant: the playing segment's next word is due. It replaces
    /// a tick still pending, which was asked for later.
    TickAt(Ts),
}

#[must_use = "an effect not performed is a word the user never sees"]
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub(in crate::transcription) enum TranscriptPacerEffect {
    /// One piece of the transcript for the text output, with where its audio starts and ends.
    Forward {
        chunk: TranscriptChunk,
        start_time: f64,
        end_time: f64,
    },
    Flush,
    /// Once per playout: at its finish, and at each split.
    Heard(Heard),
    /// A playout the sink's clear stopped: what was heard of `segment` by then.
    Cleared {
        segment: PlaybackSegment,
        heard: Heard,
    },
    /// An event the segment was not in a state to take. Nothing changed.
    Warn(TranscriptPacerWarning),
}
