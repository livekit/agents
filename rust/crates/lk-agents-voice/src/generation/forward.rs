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
    pub audio_enabled: bool,
    pub text_enabled: bool,
}

pub enum PlaybackEvent {
    Observe { interrupted: bool },
    Authorized { at: std::time::SystemTime, audio_enabled: bool, text_enabled: bool },
    Interrupted,
}

pub enum PlaybackEffect {
    Wait,
    Play { consume_authorization: bool },
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
                    PlaybackEffect::Play { consume_authorization: false }
                } else {
                    PlaybackEffect::Wait
                }
            }
            PlaybackEvent::Authorized { at, audio_enabled, text_enabled } => {
                self.at = Some(at);
                self.audio_enabled = audio_enabled;
                self.text_enabled = text_enabled;
                PlaybackEffect::Play { consume_authorization: true }
            }
        }
    }
}

/// A transcript segment flushes once even when several completion paths meet.
#[derive(Clone, Copy, Default)]
pub struct TextLegState {
    pub forwarded: bool,
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
