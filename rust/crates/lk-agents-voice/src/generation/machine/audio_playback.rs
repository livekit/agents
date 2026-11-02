//! One audio segment's playback lifecycle: forwarding, interruption, receipts, and completion.
//!
//! The runtime supplies physical outcomes and performs requested work. The owner keeps the
//! forwarding/receipt/finish states private, including when the source transcript is sampled.

use super::playout::{
    AudioPlaybackCutEvent, AudioPlaybackFinishEvent, AudioPlaybackFlowEvent, Completion,
    CutReceipt, Playout, SegmentFinish, SegmentFlow, drained,
};
pub use super::playout::{Next, SegmentEnd, SinkReport, SourceFailed};
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{DriverError, Now};
use crate::generation::effects::AudioPlaybackEffect;
use crate::generation::effects::{
    AudioPlaybackAccountingEffect, AudioPlaybackCutEffect, AudioPlaybackFinishEffect,
    AudioPlaybackFlowEffect,
};
use crate::trace::PlayoutStep;
use crate::transcription::Heard;
use lk_agents_abi::Ts;
use lk_agents_abi::voice::io::{PlaybackFinished, PlaybackSegment};
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
                        AudioPlaybackFlowEvent::Ready { interrupted }
                    }
                    AudioPlaybackEvent::SourceRead { next, interrupted } => {
                        AudioPlaybackFlowEvent::Source { next, interrupted }
                    }
                    AudioPlaybackEvent::LaunchPermissionResolved { allowed } => {
                        AudioPlaybackFlowEvent::Launched(allowed)
                    }
                    AudioPlaybackEvent::FrameForwarded(result) => {
                        AudioPlaybackFlowEvent::Forwarded(result)
                    }
                    AudioPlaybackEvent::ResamplerTailForwarded(result) => {
                        AudioPlaybackFlowEvent::TailPushed(result)
                    }
                    AudioPlaybackEvent::BoundaryFlushed => AudioPlaybackFlowEvent::BoundaryFlushed,
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
                        AudioPlaybackFinishEvent::Receipt(result)
                    }
                    AudioPlaybackEvent::PlaybackReceiptAccepted(receipt) => {
                        AudioPlaybackFinishEvent::Accepted(receipt)
                    }
                    AudioPlaybackEvent::PlayedConfirmed => AudioPlaybackFinishEvent::Confirmed,
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
                    AudioPlaybackEvent::CutReceiptPolled(report) => {
                        AudioPlaybackCutEvent::Probed(report)
                    }
                    AudioPlaybackEvent::CutReceiptYielded { closing } => {
                        AudioPlaybackCutEvent::Yielded { closing }
                    }
                    AudioPlaybackEvent::CutReceiptWaited(report) => {
                        AudioPlaybackCutEvent::Waited(report)
                    }
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
                let effect = finish.handle(AudioPlaybackFinishEvent::Cut(heard), &self.playout);
                self.finishing_effect(finish, effect)
            }
            _ => panic!("audio playback outcome does not match its requested operation"),
        };
        Some(effect)
    }

    fn forwarding_effect(
        &mut self,
        flow: SegmentFlow,
        effect: AudioPlaybackFlowEffect,
    ) -> AudioPlaybackEffect {
        self.phase = PlaybackPhase::Forwarding(flow);
        match effect {
            AudioPlaybackFlowEffect::CheckInterruption => AudioPlaybackEffect::CheckInterruption,
            AudioPlaybackFlowEffect::Read => AudioPlaybackEffect::ReadSource,
            AudioPlaybackFlowEffect::WaitLaunch => AudioPlaybackEffect::WaitForLaunchPermission,
            AudioPlaybackFlowEffect::Frame(frame) => AudioPlaybackEffect::ForwardFrame(frame),
            AudioPlaybackFlowEffect::Tail => AudioPlaybackEffect::ForwardResamplerTail,
            AudioPlaybackFlowEffect::FlushBoundary => AudioPlaybackEffect::FlushBoundary,
            AudioPlaybackFlowEffect::Finish(end) => {
                self.phase = PlaybackPhase::ReadingTranscript(end);
                AudioPlaybackEffect::ReadSourceTranscript
            }
        }
    }

    fn finishing_effect(
        &mut self,
        finish: SegmentFinish,
        effect: AudioPlaybackFinishEffect,
    ) -> AudioPlaybackEffect {
        match effect {
            AudioPlaybackFinishEffect::Drain => {
                self.phase = PlaybackPhase::Finishing(finish);
                AudioPlaybackEffect::WaitForPlaybackReceipt
            }
            AudioPlaybackFinishEffect::Accept(receipt) => {
                self.phase = PlaybackPhase::Finishing(finish);
                AudioPlaybackEffect::AcceptPlaybackReceipt(receipt)
            }
            AudioPlaybackFinishEffect::Cut(text) => {
                self.phase = PlaybackPhase::Clearing { finish, text };
                AudioPlaybackEffect::ClearOutput
            }
            AudioPlaybackFinishEffect::Confirm => {
                self.phase = PlaybackPhase::Finishing(finish);
                AudioPlaybackEffect::ConfirmPlayed
            }
            AudioPlaybackFinishEffect::Complete(result) => {
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
        effect: AudioPlaybackCutEffect,
    ) -> AudioPlaybackEffect {
        match effect {
            AudioPlaybackCutEffect::Probe => {
                self.phase = PlaybackPhase::Cutting { finish, text, receipt };
                AudioPlaybackEffect::PollCutReceipt
            }
            AudioPlaybackCutEffect::Yield => {
                self.phase = PlaybackPhase::Cutting { finish, text, receipt };
                AudioPlaybackEffect::YieldForCutReceipt
            }
            AudioPlaybackCutEffect::WaitUntil(at) => {
                self.phase = PlaybackPhase::Cutting { finish, text, receipt };
                AudioPlaybackEffect::WaitForCutReceiptUntil(at)
            }
            AudioPlaybackCutEffect::Complete(report) => {
                self.phase = PlaybackPhase::ResolvingCut { finish, text };
                AudioPlaybackEffect::ResolveCutReceipt(report)
            }
            AudioPlaybackCutEffect::Unheard => {
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
                AudioPlaybackAccountingEffect::Trace(step) => AudioPlaybackEffect::Trace(step),
                AudioPlaybackAccountingEffect::Started => AudioPlaybackEffect::Started,
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
