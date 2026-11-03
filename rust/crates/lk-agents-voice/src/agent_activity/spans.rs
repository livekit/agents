//! What the session's spans learn from the driver: the edges of the user's speech and turn, with
//! the STT, the endpointing and the conversation only the driver knows.

use crate::agent_session::VoiceDriver;
use crate::audio_recognition::{EouDetected, TurnDetectionMode};
use crate::events::SessionEvent;
use crate::telemetry::{self, Endpointing, Stt};
use lk_agents_abi::llm::chat_context::CopyOptions;
use lk_agents_abi::telemetry::spans::user_turn::{UserTurnOrigin, UserTurnOutcome};
use lk_agents_abi::{ChatContext, ChatMessage, Role, ToDictHost, ToDictOptions};
use lk_agents_abi::{SpeechEventType, Ts};
use lk_agents_core::llm::chat_dict::to_dict;
use lk_agents_core::llm::history::ChatContextExt;

/// `_EOU_MAX_HISTORY_TURNS`: the items an `eou_detection` span records.
const EOU_HISTORY: usize = 6;

impl VoiceDriver {
    /// The STT's plugin model and provider: `user_turn` records them as 1.8.10 does, and they
    /// label its measurements.
    pub(crate) fn span_stt(&self) -> Stt {
        let id = self.stt_identity();
        Stt { model: id.as_ref().map(|id| id.model.clone()), provider: id.map(|id| id.provider) }
    }

    fn span_endpointing(&self) -> Endpointing {
        let recognition = &self.activity.audio_recognition;
        let source = match recognition.turn_detection_mode() {
            TurnDetectionMode::Vad => "vad",
            TurnDetectionMode::Stt => "stt",
            TurnDetectionMode::Manual => crate::telemetry::MANUAL,
            TurnDetectionMode::RealtimeLlm => "realtime_llm",
        };
        Endpointing { source, delay: recognition.min_delay().as_secs_f64() }
    }

    /// The spans a session event opens, ends or describes, before the event reaches the host:
    /// the heard user turn's evidence, then the session's.
    pub(crate) fn observe_spans(&mut self, now: Ts, ev: &SessionEvent) {
        let endpointing = self.span_endpointing();
        let recognition = &self.activity.audio_recognition;
        let mode = recognition.turn_detection_mode();
        let speaking = recognition.user_speaking();
        let wall = self.step.wall_secs_at(now);
        match ev {
            SessionEvent::StartOfSpeech(e) => {
                let origin = match (e.from_vad, mode == TurnDetectionMode::RealtimeLlm) {
                    (true, _) => UserTurnOrigin::Speech,
                    (false, true) => UserTurnOrigin::ServerSpeech,
                    (false, false) => UserTurnOrigin::Transcript,
                };
                let participant = self.session.spans.linked_participant().cloned();
                self.hear_turn(e.at, origin);
                if let Some(turn) = &mut self.activity.user_turn {
                    turn.speech_started(&mut self.session.spans.trace, e.at, participant.as_ref());
                }
            }
            SessionEvent::EndOfSpeech(e) => {
                if let Some(turn) = &mut self.activity.user_turn {
                    turn.speech_ended(&mut self.session.spans.trace, e.at, endpointing);
                }
            }
            SessionEvent::UserInputTranscribed(t) => {
                let text = t.ev.best_alternative().map_or("", |a| a.text.as_str());
                // A transcript opens a turn no speech edge opened.
                if text.is_empty() && self.activity.user_turn.is_none() {
                    return self.session.spans.observe(now, ev);
                }
                let received = t.ev.created_at.unwrap_or(wall);
                let kind = match t.ev.kind {
                    SpeechEventType::FinalTranscript => "final_transcript",
                    SpeechEventType::PreflightTranscript => "preflight_transcript",
                    _ => "interim_transcript",
                };
                self.hear_turn(now, UserTurnOrigin::Transcript);
                if let Some(turn) = &mut self.activity.user_turn {
                    turn.transcript(received, kind, text, t.ev.request_id.as_str());
                    // A final heard after the speech re-arms the countdown (1.8.10's `stt`
                    // trigger).
                    if t.is_final && !speaking && mode != TurnDetectionMode::Manual {
                        let stt = Endpointing { source: "stt", ..endpointing };
                        turn.retriggered(&mut self.session.spans.trace, stt);
                    }
                }
            }
            SessionEvent::OverlappingSpeech(e) => {
                if let Some(turn) = &mut self.activity.user_turn {
                    turn.overlap(e);
                }
            }
            SessionEvent::UserTranscriptionTimeout(_) => {
                if let Some(mut turn) = self.activity.user_turn.take() {
                    let outcome = UserTurnOutcome::TranscriptionTimeout;
                    turn.end(&mut self.session.spans.trace, outcome);
                }
            }
            // Each user turn the activity still owns ends `closed`, before the session.
            SessionEvent::Close { .. } => self.close_user_turns(),
            _ => {}
        }
        self.session.spans.observe(now, ev);
    }

    /// Evidence at `at` opens the user turn being heard, when none is.
    fn hear_turn(&mut self, at: Ts, origin: UserTurnOrigin) {
        if self.activity.user_turn.is_none() {
            let stt = self.span_stt();
            let spans = &mut self.session.spans;
            let participant = spans.linked_participant().cloned();
            let turn = telemetry::UserTurnSpan::open(
                &mut spans.trace,
                at,
                origin,
                participant.as_ref(),
                &stt,
            );
            self.activity.user_turn = Some(turn);
        }
    }

    /// The session closes: the heard turn and every decided one still waiting or parked on its
    /// hook end `closed`.
    pub(crate) fn close_user_turns(&mut self) {
        let trace = &mut self.session.spans.trace;
        let closed = UserTurnOutcome::Closed;
        if let Some(mut turn) = self.activity.user_turn.take() {
            turn.end(trace, closed);
        }
        for turn in &mut self.activity.waiting_turns {
            turn.span.end(trace, closed);
        }
        if let Some(pending) = &mut self.activity.pending_turn {
            pending.span.end(trace, closed);
        }
    }

    /// The countdown's detector: `eou_detection` records the conversation it read.
    pub(crate) fn observe_detection(&mut self, detected: &EouDetected) {
        let predicted = matches!(detected, EouDetected::Resolved { prediction: Some(_), .. });
        let chat_ctx = predicted.then(|| {
            let transcript = self.activity.audio_recognition.audio_transcript();
            let mut items = self.session.chat_ctx.agent_chat_ctx().items;
            if !transcript.is_empty() {
                let wall = std::time::SystemTime::UNIX_EPOCH;
                items
                    .push(ChatMessage::text("eou_transcript", Role::User, transcript, wall).into());
            }
            let from = items.len().saturating_sub(EOU_HISTORY);
            let recent = ChatContext::new(items.split_off(from)).copy(&CopyOptions {
                exclude_function_call: true,
                exclude_instructions: true,
                exclude_empty_message: true,
                exclude_handoff: true,
                exclude_config_update: true,
                tools: None,
            });
            let opts = ToDictOptions {
                exclude_image: true,
                exclude_audio: true,
                exclude_timestamp: true,
                exclude_function_call: true,
                exclude_metrics: true,
                exclude_config_update: true,
            };
            to_dict(&recent.items, &opts, &ToDictHost::default())
                .map(serde_json::Value::Object)
                .unwrap_or_default()
        });
        if let Some(turn) = &mut self.activity.user_turn {
            turn.detection(&mut self.session.spans.trace, detected, chat_ctx.unwrap_or_default());
        }
    }
}
