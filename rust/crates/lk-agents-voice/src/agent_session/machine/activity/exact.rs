use crate::agent_session::machine::activity::scheduling::{
    SpeechId, SpeechKind, VoiceRunParticipation,
};
use crate::agent_session::{SayText, SpeechCall, SpeechCommand, VoiceDriver};
use lk_agents_abi::Ts;

pub(super) const EXACT_TTS_REQUIRED: &str =
    "exact tool output requires an explicit TTS provider or TTS node in a realtime session";

impl VoiceDriver {
    /// Exact tool output cannot delegate wording to a generative realtime model. Registered
    /// TTS includes a configured TTS node wrapping its provider; ordinary say stays separate.
    fn exact_text_supported(&mut self) -> bool {
        if self.activity.realtime().is_some() && self.session.tts.is_none() {
            self.record_trace(crate::trace::SpeechStep::SayRefused {
                why: crate::trace::SayRefusal::NoVoice,
            });
            return false;
        }
        true
    }

    /// Framework verbatim results, questions and approval read-backs share exact eligibility.
    pub(crate) fn say_tool_text(
        &mut self,
        now: Ts,
        text: String,
        voice_run: VoiceRunParticipation,
    ) -> Option<SpeechId> {
        if !self.exact_text_supported() {
            return None;
        }
        self.say_text(now, text, voice_run)
    }

    pub(crate) fn say_exact(&mut self, now: Ts, text: String, call: Option<SpeechCall>) {
        if !self.exact_text_supported() {
            self.fail_waiter(call, SpeechKind::Say, EXACT_TTS_REQUIRED);
            return;
        }
        self.on_speech(
            now,
            SpeechCommand::Say {
                text: SayText::Text(text),
                allow_interruptions: None,
                add_to_chat_ctx: true,
                call,
            },
        );
    }
}

#[cfg(test)]
#[path = "exact_tests.rs"]
mod tests;
