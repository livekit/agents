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
