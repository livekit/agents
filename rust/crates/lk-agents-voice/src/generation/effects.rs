use super::machine::pipeline::Finished;
use super::machine::playout::{SegmentEnd, SourceFailed};
use crate::trace::PlayoutStep;
use crate::transcription::{Heard, Visible};
use lk_agents_abi::llm::realtime as rt;
use lk_agents_abi::voice::io::PlaybackFinished;
use lk_agents_abi::{AudioFrame, TranscriptChunk, Ts};

/// The ordered operations caused by one text input.
pub enum StreamingTtsEffect {
    Capture(Visible),
    Push(String),
    EndInput,
    FlushText,
}

pub enum AlignedTranscriptEffect {
    CaptureTimed(TranscriptChunk),
    Capture(String),
    WarnUnaligned,
    Flush,
}

pub enum SynthesisLossEffect {
    Fail(crate::agent_session::DriverError),
    ReleaseVoice,
    CutShort,
    Capture(String),
    Drain { opening: Visible, text: DrainedText },
    Heard(Heard),
}

/// What happens to the model text drained after a synthesis loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainedText {
    /// No accepted stamps stand for what was heard: the text goes on to the transcript.
    Forward,
    /// Nothing was lost, or the stamps already say what was heard.
    Discard,
}

/// An accepted observation's publications.
pub(in crate::generation) enum AudioPlaybackAccountingEffect {
    Trace(PlayoutStep),
    Started,
}

/// One physical operation requested by the cut receipt policy.
pub(in crate::generation) enum AudioPlaybackCutEffect {
    Probe,
    Yield,
    WaitUntil(Ts),
    Complete(Option<PlaybackFinished>),
    Unheard,
}

/// Terminal I/O is sequenced by the voice core, after the resampler's tail was handled.
pub(in crate::generation) enum AudioPlaybackFinishEffect {
    Drain,
    Accept(PlaybackFinished),
    Cut(String),
    Confirm,
    Complete(Result<crate::transcription::Heard, Box<SourceFailed>>),
}

/// The ordered source, conversion and capture operations for one forwarding.
pub(in crate::generation) enum AudioPlaybackFlowEffect {
    CheckInterruption,
    Read,
    WaitLaunch,
    Frame(lk_agents_abi::AudioFrame),
    Tail,
    FlushBoundary,
    Finish(SegmentEnd),
}

pub enum RealtimeGenerationEffect {
    OpenText,
    Text(String),
    Audio(AudioFrame),
    VoiceText { segment: usize, text: String },
    Boundary,
    End,
    ToolCall(rt::FunctionCallReady),
    Failed(String),
}

/// Physical work requested by the playback owner, plus its publications and terminal result.
#[must_use = "playback work and its completion must be handled"]
pub enum AudioPlaybackEffect {
    CheckInterruption,
    ReadSource,
    WaitForLaunchPermission,
    ForwardFrame(AudioFrame),
    ForwardResamplerTail,
    FlushBoundary,
    /// Report the last pushed batch before taking the source's final transcript snapshot.
    ReadSourceTranscript,
    WaitForPlaybackReceipt,
    AcceptPlaybackReceipt(PlaybackFinished),
    /// Clear synchronously, then fold reports before acknowledging this operation.
    ClearOutput,
    PollCutReceipt,
    YieldForCutReceipt,
    WaitForCutReceiptUntil(Ts),
    /// Fold queued reports only when this receipt is present, then return its projection facts.
    ResolveCutReceipt(Option<PlaybackFinished>),
    /// Fold reports/settle the source, then publish a physical finish only when playback
    /// started and `reported` is present. A sink can report a whole finish despite our cut.
    FinishCut {
        heard: Heard,
        reported: Option<bool>,
    },
    ConfirmPlayed,
    Trace(PlayoutStep),
    Started,
    Complete(Result<Heard, Box<SourceFailed>>),
}

pub enum GenerationPreparationEffect {
    WaitForReadiness,
    WaitForPlaybackAuthorization,
    Start,
    Interrupted,
}

pub enum ReplyStartupEffect {
    RequestPlan(super::ReplyPlan),
    Run,
    Finished(Finished),
}

pub enum TurnAccumulatorEffect {
    Read,
    Forward(crate::transcription::Visible),
    End { interrupted: bool },
}
