//! Speaking starts with delivered audio, or the first words when no audio output exists.

use super::TurnAccumulator;
use crate::generation::effects::TurnAccumulatorEffect;

pub enum TurnAccumulatorOutputEvent<'a> {
    SegmentOpened { has_audio_output: bool, interrupted: bool },
    TextDelta { delta: &'a str, has_audio_output: bool, interrupted: bool },
    Audio { at: f64 },
}

impl TurnAccumulator {
    /// The first output determines the speaking stamp for the entire generation step.
    /// Returns whether text-only delivery must publish the first-frame notification. Audio
    /// delivery records its stamp here but publishes through its existing playback path.
    pub fn on_forwarded(&mut self, event: TurnAccumulatorOutputEvent<'_>, now: f64) -> bool {
        match event {
            TurnAccumulatorOutputEvent::TextDelta { delta, has_audio_output, interrupted } => {
                if delta.is_empty() {
                    return false;
                }
                self.on_forwarded(
                    TurnAccumulatorOutputEvent::SegmentOpened { has_audio_output, interrupted },
                    now,
                )
            }
            TurnAccumulatorOutputEvent::SegmentOpened { has_audio_output, interrupted } => {
                if has_audio_output || interrupted {
                    return false;
                }
                self.on_first_frame(None, now)
            }
            TurnAccumulatorOutputEvent::Audio { at } => {
                self.on_first_frame(Some(at), now);
                false
            }
        }
    }
}

/// A transcript segment flushes once even when several completion paths meet.
#[derive(Clone, Copy, Default)]
pub struct TextLegState {
    forwarded: bool,
    flushed: bool,
}

impl TextLegState {
    /// Empty deltas never reach transcript output.
    pub fn should_forward(&self, empty: bool) -> bool {
        !empty
    }

    /// Record actual forwarding, so unvoiced completion only ends a delivered segment.
    pub fn record_forwarded(&mut self) {
        self.forwarded = true;
    }

    /// Claims the segment's single flush. Competing completion paths become no-ops.
    pub fn flush_once(&mut self) -> bool {
        !std::mem::replace(&mut self.flushed, true)
    }
}

/// Inputs to the generation's sequential segment forwarding.
pub enum TurnAccumulatorEvent {
    Ready { interrupted: bool },
    Chunk { words: crate::transcription::Visible, caller_frames: bool, interrupted: bool },
    Boundary,
    Closed { interrupted: bool },
    Played { heard: crate::transcription::Heard, interrupted: bool },
}

impl TurnAccumulator {
    /// Empty word segments are skipped; caller-supplied audio plays even without words.
    pub fn on_segment(&mut self, input: TurnAccumulatorEvent) -> TurnAccumulatorEffect {
        match input {
            TurnAccumulatorEvent::Ready { interrupted: true } => {
                TurnAccumulatorEffect::End { interrupted: true }
            }
            TurnAccumulatorEvent::Ready { interrupted: false } | TurnAccumulatorEvent::Boundary => {
                TurnAccumulatorEffect::Read
            }
            TurnAccumulatorEvent::Closed { interrupted } => {
                TurnAccumulatorEffect::End { interrupted }
            }
            TurnAccumulatorEvent::Chunk { words, caller_frames, interrupted } => {
                if !caller_frames && words.raw.is_empty() && words.text.is_empty() {
                    TurnAccumulatorEffect::Read
                } else if interrupted {
                    TurnAccumulatorEffect::End { interrupted: true }
                } else {
                    TurnAccumulatorEffect::Forward(words)
                }
            }
            TurnAccumulatorEvent::Played { heard, interrupted } => {
                self.record(&heard.text);
                if interrupted {
                    self.cut_at(heard.position);
                    return TurnAccumulatorEffect::End { interrupted: true };
                }
                if !heard.complete {
                    self.cut_short();
                }
                TurnAccumulatorEffect::Read
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum VoiceKind {
    Synthesis,
    CallerFrames,
    Silent,
}

#[derive(Clone, Copy)]
pub enum SegmentRoute {
    Synthesis,
    CallerFrames,
    Text,
}

impl SegmentRoute {
    /// Only synthesis starts before activation; the other routes are decided again once the
    /// activated binding names its outputs.
    pub fn of(kind: VoiceKind, audio_output: bool) -> Self {
        match (kind, audio_output) {
            (VoiceKind::Synthesis, true) => SegmentRoute::Synthesis,
            (VoiceKind::CallerFrames, true) => SegmentRoute::CallerFrames,
            _ => SegmentRoute::Text,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TranscriptPlan {
    /// The words queue behind the segment's activation, which attaches the transcription output:
    /// a synthesised segment's words never reach an output before its attach.
    pub deferred: bool,
    pub aligned: bool,
}

/// The binding's fixed output facts decide each segment's audio and transcript path together.
pub struct SegmentPlan {
    pub route: SegmentRoute,
    pub transcript: Option<TranscriptPlan>,
    audio_output: bool,
    text_output: bool,
}

impl SegmentPlan {
    pub fn new(
        kind: VoiceKind,
        audio_output: bool,
        text_output: bool,
        aligned_requested: bool,
        aligned_provider: bool,
    ) -> Self {
        let route = SegmentRoute::of(kind, audio_output);
        let synthesis = matches!(route, SegmentRoute::Synthesis);
        Self {
            route,
            transcript: (synthesis || text_output).then_some(TranscriptPlan {
                deferred: synthesis,
                aligned: synthesis && aligned_requested && aligned_provider,
            }),
            audio_output,
            text_output,
        }
    }

    /// A transcript-only segment still ends the bound audio leg when all its text was forwarded
    /// and `delivered` through the text chain. A chain cancelled with the words still in it (a cut
    /// `transcription_node`) ends nothing: the synchroniser would take an audio end with no words
    /// as a segment of its own, which the next speech's words join (seed 24984).
    pub fn end_unvoiced(
        &self,
        heard: &crate::transcription::Heard,
        leg: Option<TextLegState>,
        delivered: bool,
    ) -> bool {
        self.audio_output
            && self.text_output
            && delivered
            && leg.is_some_and(|leg| leg.forwarded)
            && heard.frames == 0
            && heard.complete
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription::Heard;
    use std::time::Duration;

    #[test]
    fn text_only_start_is_published_once_and_not_for_empty_or_interrupted_output() {
        let mut acc = TurnAccumulator::new(std::time::SystemTime::UNIX_EPOCH, None);
        assert!(!acc.on_forwarded(
            TurnAccumulatorOutputEvent::TextDelta {
                delta: "",
                has_audio_output: false,
                interrupted: false,
            },
            1.0
        ));
        assert!(!acc.on_forwarded(
            TurnAccumulatorOutputEvent::SegmentOpened {
                has_audio_output: false,
                interrupted: true,
            },
            1.0
        ));
        assert!(acc.on_forwarded(
            TurnAccumulatorOutputEvent::TextDelta {
                delta: "hello",
                has_audio_output: false,
                interrupted: false,
            },
            2.0
        ));
        assert!(!acc.on_forwarded(
            TurnAccumulatorOutputEvent::SegmentOpened {
                has_audio_output: false,
                interrupted: false,
            },
            3.0
        ));
    }

    #[test]
    fn competing_text_completion_paths_flush_once() {
        let mut leg = TextLegState::default();
        assert!(!leg.should_forward(true));
        assert!(leg.should_forward(false));
        assert!(leg.flush_once());
        assert!(!leg.flush_once());
    }

    // Catches: a transcript-only segment whose words a cut `transcription_node` never delivered
    // ended its audio leg; the synchroniser took that audio end as a segment of its own, and the
    // next speech's words joined it and were shown whole before their audio (seed 24984).
    #[test]
    fn only_delivered_words_end_the_audio_leg_unvoiced() {
        let plan = SegmentPlan::new(VoiceKind::Synthesis, true, true, false, false);
        let mut leg = TextLegState::default();
        leg.record_forwarded();
        let silent = Heard::complete(String::from("Hello."), Duration::ZERO, 0);
        assert!(plan.end_unvoiced(&silent, Some(leg), true));
        assert!(!plan.end_unvoiced(&silent, Some(leg), false));
    }
}
