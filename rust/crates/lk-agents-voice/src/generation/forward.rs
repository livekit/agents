//! Speaking starts with delivered audio, or the first words when no audio output exists.

use super::TurnAccumulator;

pub enum ForwardEvent<'a> {
    SegmentOpened { has_audio_output: bool, interrupted: bool },
    TextDelta { delta: &'a str, has_audio_output: bool, interrupted: bool },
    Audio { at: f64 },
}

pub enum ForwardEffect {
    StartedSpeaking,
}

impl TurnAccumulator {
    /// The first output determines the speaking stamp for the entire generation step.
    pub fn on_forwarded(&mut self, event: ForwardEvent<'_>, now: f64) -> Option<ForwardEffect> {
        match event {
            ForwardEvent::TextDelta { delta, has_audio_output, interrupted } => {
                if delta.is_empty() {
                    return None;
                }
                self.on_forwarded(
                    ForwardEvent::SegmentOpened { has_audio_output, interrupted },
                    now,
                )
            }
            ForwardEvent::SegmentOpened { has_audio_output, interrupted } => {
                if has_audio_output || interrupted {
                    return None;
                }
                self.on_first_frame(None, now).then_some(ForwardEffect::StartedSpeaking)
            }
            ForwardEvent::Audio { at } => {
                self.on_first_frame(Some(at), now);
                None
            }
        }
    }
}

/// Authorization belongs to a generation step, including all its synthesis segments.
#[derive(Default)]
pub struct PlaybackGate {
    pub at: Option<std::time::SystemTime>,
}

pub enum PlaybackEvent {
    Observe { interrupted: bool },
    Authorized { at: std::time::SystemTime },
    Interrupted,
}

pub enum PlaybackEffect {
    Wait,
    /// The step's authorization is held: this gate consumed it, here or before.
    Play,
    Refuse,
}

impl PlaybackGate {
    pub fn handle(&mut self, event: PlaybackEvent) -> PlaybackEffect {
        match event {
            PlaybackEvent::Observe { interrupted: true } | PlaybackEvent::Interrupted => {
                PlaybackEffect::Refuse
            }
            PlaybackEvent::Observe { interrupted: false } => {
                if self.at.is_some() {
                    PlaybackEffect::Play
                } else {
                    PlaybackEffect::Wait
                }
            }
            PlaybackEvent::Authorized { at } => {
                self.at = Some(at);
                PlaybackEffect::Play
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

pub enum TextLegEvent {
    Capture { empty: bool },
    Forwarded,
    Flush,
}

pub enum TextLegEffect {
    Forward,
    Flush,
}

impl TextLegState {
    pub fn handle(&mut self, event: TextLegEvent) -> Option<TextLegEffect> {
        match event {
            TextLegEvent::Capture { empty } => (!empty).then_some(TextLegEffect::Forward),
            TextLegEvent::Forwarded => {
                self.forwarded = true;
                None
            }
            TextLegEvent::Flush if !self.flushed => {
                self.flushed = true;
                Some(TextLegEffect::Flush)
            }
            TextLegEvent::Flush => None,
        }
    }
}

/// Inputs to the generation's sequential segment forwarding.
pub enum SegmentInput {
    Ready { interrupted: bool },
    Chunk { words: crate::transcription::Visible, caller_frames: bool, interrupted: bool },
    Boundary,
    Closed { interrupted: bool },
    Played { heard: crate::transcription::Heard, interrupted: bool },
}

pub enum SegmentEffect {
    Read,
    Forward(crate::transcription::Visible),
    End { interrupted: bool },
}

impl TurnAccumulator {
    /// Empty word segments are skipped; caller-supplied audio plays even without words.
    pub fn on_segment(&mut self, input: SegmentInput) -> SegmentEffect {
        match input {
            SegmentInput::Ready { interrupted: true } => SegmentEffect::End { interrupted: true },
            SegmentInput::Ready { interrupted: false } | SegmentInput::Boundary => {
                SegmentEffect::Read
            }
            SegmentInput::Closed { interrupted } => SegmentEffect::End { interrupted },
            SegmentInput::Chunk { words, caller_frames, interrupted } => {
                if !caller_frames && words.raw.is_empty() && words.text.is_empty() {
                    SegmentEffect::Read
                } else if interrupted {
                    SegmentEffect::End { interrupted: true }
                } else {
                    SegmentEffect::Forward(words)
                }
            }
            SegmentInput::Played { heard, interrupted } => {
                self.record(&heard.text);
                if interrupted {
                    return SegmentEffect::End { interrupted: true };
                }
                if !heard.complete {
                    self.cut_short();
                }
                SegmentEffect::Read
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

#[derive(Default)]
pub struct SynthesisPreparation {
    phase: PreparationPhase,
}

#[derive(Default)]
enum PreparationPhase {
    #[default]
    Initial,
    Readiness,
    InitialPlayback,
    FinalPlayback,
    Started,
    Refused,
}

pub enum PreparationEvent {
    Begin {
        has_tts: bool,
        audio_enabled: bool,
        has_audio_output: bool,
        llm_reply: bool,
        preemptive_tts: bool,
        authorized: bool,
    },
    Ready {
        ready: bool,
        authorized: bool,
    },
    Playback {
        accepted: bool,
        authorized: bool,
    },
}

pub enum PreparationEffect {
    WaitReady,
    WaitPlayback,
    Start,
    Refuse,
}

impl SynthesisPreparation {
    /// Speculative synthesis may precede playback authorization; other output waits for playback.
    pub fn handle(&mut self, event: PreparationEvent) -> PreparationEffect {
        use PreparationEffect as Effect;
        use PreparationEvent as Event;
        use PreparationPhase as Phase;
        match (&self.phase, event) {
            (
                Phase::Initial,
                Event::Begin {
                    has_tts,
                    audio_enabled,
                    has_audio_output,
                    llm_reply,
                    preemptive_tts,
                    authorized,
                },
            ) => {
                if has_tts && audio_enabled && has_audio_output {
                    if llm_reply && preemptive_tts {
                        self.check_authorization(authorized)
                    } else {
                        self.phase = Phase::Readiness;
                        Effect::WaitReady
                    }
                } else {
                    self.phase = Phase::InitialPlayback;
                    Effect::WaitPlayback
                }
            }
            (Phase::Readiness, Event::Ready { ready: true, authorized })
            | (Phase::InitialPlayback, Event::Playback { accepted: true, authorized }) => {
                self.check_authorization(authorized)
            }
            (Phase::FinalPlayback, Event::Playback { accepted: true, .. }) => {
                self.phase = Phase::Started;
                Effect::Start
            }
            (Phase::Started, _) => Effect::Start,
            _ => {
                self.phase = Phase::Refused;
                Effect::Refuse
            }
        }
    }

    fn check_authorization(&mut self, authorized: bool) -> PreparationEffect {
        if authorized {
            self.phase = PreparationPhase::FinalPlayback;
            PreparationEffect::WaitPlayback
        } else {
            self.phase = PreparationPhase::Started;
            PreparationEffect::Start
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription::Heard;
    use std::time::Duration;

    // Catches: a transcript-only segment whose words a cut `transcription_node` never delivered
    // ended its audio leg; the synchroniser took that audio end as a segment of its own, and the
    // next speech's words joined it and were shown whole before their audio (seed 24984).
    #[test]
    fn only_delivered_words_end_the_audio_leg_unvoiced() {
        let plan = SegmentPlan::new(VoiceKind::Synthesis, true, true, false, false);
        let mut leg = TextLegState::default();
        let _ = leg.handle(TextLegEvent::Forwarded);
        let silent = Heard::complete(String::from("Hello."), Duration::ZERO, 0);
        assert!(plan.end_unvoiced(&silent, Some(leg), true));
        assert!(!plan.end_unvoiced(&silent, Some(leg), false));
    }
}
