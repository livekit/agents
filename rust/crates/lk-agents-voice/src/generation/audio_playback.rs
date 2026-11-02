//! One audio segment's playback lifecycle: forwarding, interruption, receipts, and completion.
//!
//! The runtime supplies physical outcomes and performs requested work. The owner keeps the
//! forwarding/receipt/finish states private, including when the source transcript is sampled.

use super::playout::{
    Completion, CutEffect, CutEvent, CutReceipt, FinishEffect, FinishEvent, FlowEffect, FlowEvent,
    Playout, PlayoutEffect, SegmentFinish, SegmentFlow, drained,
};
pub use super::playout::{Next, SegmentEnd, SinkReport, SourceFailed};
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{DriverError, Now};
use crate::trace::PlayoutStep;
use crate::transcription::Heard;
use lk_agents_abi::voice::io::{PlaybackFinished, PlaybackSegment};
use lk_agents_abi::{AudioFrame, Ts};
use std::time::Duration;

/// Results of the physical work requested by [`AudioPlaybackEffect`]. Sink observations use
/// [`AudioPlayback::on_report`] so they can progress alongside an outstanding source read.
pub enum AudioPlaybackEvent {
    InterruptionChecked {
        interrupted: bool,
    },
    SourceRead {
        next: Option<Result<Next, DriverError>>,
        interrupted: bool,
    },
    LaunchPermissionResolved {
        allowed: bool,
    },
    FrameForwarded(Result<(), SegmentEnd>),
    ResamplerTailForwarded(Result<(), SegmentEnd>),
    BoundaryFlushed,
    SourceTranscriptRead(String),
    PlaybackReceiptReceived(Result<Option<PlaybackFinished>, DriverError>),
    PlaybackReceiptAccepted(PlaybackFinished),
    OutputCleared,
    CutReceiptPolled(Option<PlaybackFinished>),
    CutReceiptYielded {
        closing: bool,
    },
    CutReceiptWaited(Option<PlaybackFinished>),
    /// Available reports were folded before this projection when `report` is present. With
    /// no report, `position` is the sink tracker's current lower bound.
    CutReceiptResolved {
        report: Option<PlaybackFinished>,
        position: Duration,
    },
    CutFinished(Heard),
    PlayedConfirmed,
}

/// Physical work requested by the playback owner, plus its publications and terminal result.
/// There is no runtime-facing forwarding, cut, or finish sub-protocol.
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

// These states group existing policies without exposing their instruction enums to the runtime.
enum PlaybackPhase {
    Initial { stop_at_boundary: bool },
    Forwarding(SegmentFlow),
    ReadingTranscript(SegmentEnd),
    Finishing(SegmentFinish),
    Clearing { finish: SegmentFinish, text: String },
    Cutting { finish: SegmentFinish, text: String, receipt: CutReceipt },
    ResolvingCut { finish: SegmentFinish, text: String },
    FinishingCut(SegmentFinish),
    Done,
}

/// Owns a segment's playback decisions and accounting; it has no sink, clock, task, or I/O.
///
/// Speech/floor scheduling and generation-step authorization remain separate owners. A source
/// without an output still uses this lifecycle to drain until its boundary/end/interruption.
pub struct AudioPlayback {
    phase: PlaybackPhase,
    has_sink: bool,
    playout: Playout,
}

impl AudioPlayback {
    pub fn new(
        speech: SpeechId,
        segment_key: Option<PlaybackSegment>,
        has_sink: bool,
        stop_at_boundary: bool,
    ) -> Self {
        Self {
            phase: PlaybackPhase::Initial { stop_at_boundary },
            has_sink,
            playout: Playout::new(speech, segment_key),
        }
    }

    pub fn begin(&mut self) -> AudioPlaybackEffect {
        let PlaybackPhase::Initial { stop_at_boundary } = self.phase else {
            panic!("audio playback was already begun");
        };
        self.phase = PlaybackPhase::Forwarding(SegmentFlow::new(self.has_sink, stop_at_boundary));
        AudioPlaybackEffect::CheckInterruption
    }

    /// Apply the outcome of the requested physical operation. Terminal/late outcomes are
    /// ignored after completion; they cannot perform cleanup or publish a second result.
    pub fn handle(&mut self, event: AudioPlaybackEvent, now: Ts) -> Option<AudioPlaybackEffect> {
        let phase = std::mem::replace(&mut self.phase, PlaybackPhase::Done);
        let effect = match (phase, event) {
            (PlaybackPhase::Done, _) => return None,
            (PlaybackPhase::Forwarding(mut flow), event) => {
                let event = match event {
                    AudioPlaybackEvent::InterruptionChecked { interrupted } => {
                        FlowEvent::Ready { interrupted }
                    }
                    AudioPlaybackEvent::SourceRead { next, interrupted } => {
                        FlowEvent::Source { next, interrupted }
                    }
                    AudioPlaybackEvent::LaunchPermissionResolved { allowed } => {
                        FlowEvent::Launched(allowed)
                    }
                    AudioPlaybackEvent::FrameForwarded(result) => FlowEvent::Forwarded(result),
                    AudioPlaybackEvent::ResamplerTailForwarded(result) => {
                        FlowEvent::TailPushed(result)
                    }
                    AudioPlaybackEvent::BoundaryFlushed => FlowEvent::BoundaryFlushed,
                    _ => panic!("audio playback outcome does not match forwarding work"),
                };
                let effect = flow.handle(event, self.playout.pushed);
                self.forwarding_effect(flow, effect)
            }
            (
                PlaybackPhase::ReadingTranscript(end),
                AudioPlaybackEvent::SourceTranscriptRead(text),
            ) => {
                if !self.has_sink {
                    AudioPlaybackEffect::Complete(drained(end, text))
                } else {
                    let (finish, effect) = SegmentFinish::begin(end, text, &self.playout);
                    self.finishing_effect(finish, effect)
                }
            }
            (PlaybackPhase::Finishing(mut finish), event) => {
                let event = match event {
                    AudioPlaybackEvent::PlaybackReceiptReceived(result) => {
                        FinishEvent::Receipt(result)
                    }
                    AudioPlaybackEvent::PlaybackReceiptAccepted(receipt) => {
                        FinishEvent::Accepted(receipt)
                    }
                    AudioPlaybackEvent::PlayedConfirmed => FinishEvent::Confirmed,
                    _ => panic!("audio playback outcome does not match completion work"),
                };
                let effect = finish.handle(event, &self.playout);
                self.finishing_effect(finish, effect)
            }
            (PlaybackPhase::Clearing { finish, text }, AudioPlaybackEvent::OutputCleared) => {
                let (receipt, effect) = CutReceipt::begin(self.playout.pushed);
                self.cut_effect(finish, text, receipt, effect)
            }
            (PlaybackPhase::Cutting { finish, text, mut receipt }, event) => {
                let event = match event {
                    AudioPlaybackEvent::CutReceiptPolled(report) => CutEvent::Probed(report),
                    AudioPlaybackEvent::CutReceiptYielded { closing } => {
                        CutEvent::Yielded { closing }
                    }
                    AudioPlaybackEvent::CutReceiptWaited(report) => CutEvent::Waited(report),
                    _ => panic!("audio playback outcome does not match cut receipt work"),
                };
                let effect = receipt.handle(event, now);
                self.cut_effect(finish, text, receipt, effect)
            }
            (
                PlaybackPhase::ResolvingCut { finish, text },
                AudioPlaybackEvent::CutReceiptResolved { report, position },
            ) => {
                let reported = report.as_ref().map(|receipt| receipt.interrupted);
                let completion =
                    report.map(Completion::Cut).unwrap_or(Completion::CutUnreported { position });
                let heard = self.playout.heard(text, completion);
                self.phase = PlaybackPhase::FinishingCut(finish);
                AudioPlaybackEffect::FinishCut { heard, reported }
            }
            (PlaybackPhase::FinishingCut(mut finish), AudioPlaybackEvent::CutFinished(heard)) => {
                let effect = finish.handle(FinishEvent::Cut(heard), &self.playout);
                self.finishing_effect(finish, effect)
            }
            _ => panic!("audio playback outcome does not match its requested operation"),
        };
        Some(effect)
    }

    fn forwarding_effect(&mut self, flow: SegmentFlow, effect: FlowEffect) -> AudioPlaybackEffect {
        self.phase = PlaybackPhase::Forwarding(flow);
        match effect {
            FlowEffect::CheckInterruption => AudioPlaybackEffect::CheckInterruption,
            FlowEffect::Read => AudioPlaybackEffect::ReadSource,
            FlowEffect::WaitLaunch => AudioPlaybackEffect::WaitForLaunchPermission,
            FlowEffect::Frame(frame) => AudioPlaybackEffect::ForwardFrame(frame),
            FlowEffect::Tail => AudioPlaybackEffect::ForwardResamplerTail,
            FlowEffect::FlushBoundary => AudioPlaybackEffect::FlushBoundary,
            FlowEffect::Finish(end) => {
                self.phase = PlaybackPhase::ReadingTranscript(end);
                AudioPlaybackEffect::ReadSourceTranscript
            }
        }
    }

    fn finishing_effect(
        &mut self,
        finish: SegmentFinish,
        effect: FinishEffect,
    ) -> AudioPlaybackEffect {
        match effect {
            FinishEffect::Drain => {
                self.phase = PlaybackPhase::Finishing(finish);
                AudioPlaybackEffect::WaitForPlaybackReceipt
            }
            FinishEffect::Accept(receipt) => {
                self.phase = PlaybackPhase::Finishing(finish);
                AudioPlaybackEffect::AcceptPlaybackReceipt(receipt)
            }
            FinishEffect::Cut(text) => {
                self.phase = PlaybackPhase::Clearing { finish, text };
                AudioPlaybackEffect::ClearOutput
            }
            FinishEffect::Confirm => {
                self.phase = PlaybackPhase::Finishing(finish);
                AudioPlaybackEffect::ConfirmPlayed
            }
            FinishEffect::Complete(result) => {
                self.phase = PlaybackPhase::Done;
                AudioPlaybackEffect::Complete(result)
            }
        }
    }

    fn cut_effect(
        &mut self,
        finish: SegmentFinish,
        text: String,
        receipt: CutReceipt,
        effect: CutEffect,
    ) -> AudioPlaybackEffect {
        match effect {
            CutEffect::Probe => {
                self.phase = PlaybackPhase::Cutting { finish, text, receipt };
                AudioPlaybackEffect::PollCutReceipt
            }
            CutEffect::Yield => {
                self.phase = PlaybackPhase::Cutting { finish, text, receipt };
                AudioPlaybackEffect::YieldForCutReceipt
            }
            CutEffect::WaitUntil(at) => {
                self.phase = PlaybackPhase::Cutting { finish, text, receipt };
                AudioPlaybackEffect::WaitForCutReceiptUntil(at)
            }
            CutEffect::Complete(report) => {
                self.phase = PlaybackPhase::ResolvingCut { finish, text };
                AudioPlaybackEffect::ResolveCutReceipt(report)
            }
            CutEffect::Unheard => {
                self.phase = PlaybackPhase::FinishingCut(finish);
                AudioPlaybackEffect::FinishCut { heard: Heard::default(), reported: Some(true) }
            }
        }
    }

    /// Fold independent sink facts, in arrival order, without consuming the pending I/O result.
    pub fn on_report(&mut self, report: SinkReport, now: Now) -> Vec<AudioPlaybackEffect> {
        if matches!(self.phase, PlaybackPhase::Done) {
            return Vec::new();
        }
        self.playout
            .handle(report, now)
            .into_iter()
            .map(|effect| match effect {
                PlayoutEffect::Trace(step) => AudioPlaybackEffect::Trace(step),
                PlayoutEffect::Started => AudioPlaybackEffect::Started,
            })
            .collect()
    }

    pub fn captured(&mut self, key: Option<PlaybackSegment>, duration: Duration) {
        self.playout.captured(key, duration);
    }

    pub fn take_flush(&mut self) -> bool {
        self.playout.take_flush()
    }

    pub fn flushed(&mut self, now: Ts) -> PlayoutStep {
        self.playout.flushed(now)
    }

    pub fn take_started(&mut self) -> Option<f64> {
        self.playout.take_started()
    }

    pub fn completed_audio(&self) -> Duration {
        self.playout.completed
    }

    pub fn played_audio(&self) -> Duration {
        self.playout.played
    }

    pub fn segment_key(&self) -> Option<PlaybackSegment> {
        self.playout.segment_key
    }

    pub fn pending_frames(&self) -> u64 {
        self.playout.batch
    }

    pub fn pushed_frames(&self) -> u64 {
        self.playout.pushed
    }

    pub fn has_started(&self) -> bool {
        self.playout.started
    }

    pub fn finish_deadline(&self) -> Option<Ts> {
        self.playout.finish_due
    }
}

#[cfg(test)]
#[path = "audio_playback_tests.rs"]
mod tests;
