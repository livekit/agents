//! What the session's spans learn from the driver: the edges of the user's speech and turn, with
//! the STT and the endpointing only the driver knows.

use crate::agent_session::VoiceDriver;
use crate::audio_recognition::TurnDetectionMode;
use crate::events::SessionEvent;
use crate::telemetry::{Endpointing, Stt};
use lk_agents_abi::Ts;
use lk_agents_abi::telemetry::spans::user_turn::UserTurnOrigin;
use lk_agents_core::telemetry::gen_ai::provider_name;

impl VoiceDriver {
    pub(crate) fn span_stt(&self) -> Stt {
        let id = self.stt_identity();
        Stt {
            model: id.as_ref().map(|id| id.model.clone()),
            provider: id.and_then(|id| provider_name(&id.provider)),
        }
    }

    fn span_endpointing(&self) -> Endpointing {
        let recognition = &self.activity.audio_recognition;
        let source = match recognition.turn_detection_mode() {
            TurnDetectionMode::Vad => "vad",
            TurnDetectionMode::Stt => "stt",
            TurnDetectionMode::Manual => "manual",
            TurnDetectionMode::RealtimeLlm => "realtime_llm",
        };
        Endpointing { source, delay: recognition.min_delay().as_secs_f64() }
    }

    /// The spans a session event opens, ends or describes, before the event reaches the host.
    pub(crate) fn observe_spans(&mut self, now: Ts, ev: &SessionEvent) {
        let stt = self.span_stt();
        let endpointing = self.span_endpointing();
        let server =
            self.activity.audio_recognition.turn_detection_mode() == TurnDetectionMode::RealtimeLlm;
        let wall = self.step.wall_secs_at(now);
        self.session.spans.turns(now, |trace, root, user, _| match ev {
            SessionEvent::StartOfSpeech(e) => {
                let origin = match (e.from_vad, server) {
                    (true, _) => UserTurnOrigin::Speech,
                    (false, true) => UserTurnOrigin::ServerSpeech,
                    (false, false) => UserTurnOrigin::Transcript,
                };
                user.speech_started(trace, root, e.at, origin, &stt);
            }
            SessionEvent::EndOfSpeech(e) => user.speech_ended(trace, e.at, endpointing),
            SessionEvent::UserInputTranscribed(t) => {
                let text = t.ev.best_alternative().map_or("", |a| a.text.as_str());
                let request = t.ev.request_id.as_str();
                user.transcript(trace, root, now, wall, t.is_final, text, request, &stt);
            }
            SessionEvent::EotPrediction(e) => user.prediction(trace, now, e, endpointing),
            _ => {}
        });
        self.session.spans.observe(now, ev);
    }
}
