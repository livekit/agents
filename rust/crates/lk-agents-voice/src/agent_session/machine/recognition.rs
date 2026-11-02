//! Translate recognition's ordered domain outputs into driver-owned requests and callbacks.
//! Only the driver knows the private outbox; recognition never receives a collector.

use super::coordination::VoiceCallback;
use crate::agent_session::VoiceDriver;
use crate::agent_session::effects::VoiceEffect;
use crate::audio_recognition::effects::AudioRecognitionEffect;
use crate::audio_recognition::observation::RecognitionObservation;
use crate::trace::Step;

impl VoiceDriver {
    pub(crate) fn apply_recognition_effects(&mut self, effects: Vec<AudioRecognitionEffect>) {
        for effect in effects {
            match effect {
                AudioRecognitionEffect::SetDeadline(timer, at) => self.set_deadline(timer, at),
                AudioRecognitionEffect::ClearDeadline(timer) => self.clear_deadline(timer),
                AudioRecognitionEffect::FlushInput(input) => self.flush_input(input),
                AudioRecognitionEffect::PublishSessionEvent(event) => self.publish_event(event),
                AudioRecognitionEffect::RequestEot(req, timeout) => {
                    self.record_effect(VoiceEffect::EotPredict { req, timeout });
                }
                AudioRecognitionEffect::CancelEot(req, timed_out) => {
                    self.record_effect(VoiceEffect::EotCancel { req, timed_out });
                }
                AudioRecognitionEffect::NotifyInterruption(signal) => {
                    self.notify_interruption(signal)
                }
                AudioRecognitionEffect::FlushTurnDetector(reason) => {
                    self.flush_turn_detector(reason)
                }
                AudioRecognitionEffect::EndOfTurn(info) => {
                    self.record_trace(Step::Decided(RecognitionObservation::EndOfTurn(
                        info.clone(),
                    )));
                    self.defer_callback(VoiceCallback::EndOfTurn(info));
                }
                AudioRecognitionEffect::TurnSettled(settled) => {
                    self.record_trace(Step::Decided(RecognitionObservation::TurnSettled(settled)));
                    self.defer_callback(VoiceCallback::TurnSettled(settled));
                }
                AudioRecognitionEffect::AudioInterruptGuard(guard) => {
                    self.record_trace(Step::Decided(RecognitionObservation::AudioInterruptGuard(
                        guard,
                    )));
                    self.defer_callback(VoiceCallback::AudioInterruptGuard(guard));
                }
                AudioRecognitionEffect::InterruptByAudioActivity(trigger) => {
                    self.record_trace(Step::Decided(
                        RecognitionObservation::InterruptByAudioActivity(trigger),
                    ));
                    self.defer_callback(VoiceCallback::InterruptByAudioActivity(trigger));
                }
                AudioRecognitionEffect::EndpointingDelayUpdated { min_delay } => {
                    self.record_trace(Step::Decided(
                        RecognitionObservation::EndpointingDelayUpdated { min_delay },
                    ));
                }
                AudioRecognitionEffect::TurnDeclared { owner } => {
                    self.record_trace(Step::Decided(RecognitionObservation::TurnDeclared {
                        owner,
                    }));
                }
            }
        }
    }
}

impl VoiceDriver {
    pub(crate) fn apply_user_state_updates(
        &mut self,
        updates: Vec<crate::agent_session::agent_user_state::AgentUserStateUpdate>,
    ) {
        use crate::agent_session::agent_user_state::AgentUserStateUpdate;
        use crate::agent_session::effects::Timer;
        for update in updates {
            match update {
                AgentUserStateUpdate::PublishSessionEvent(event) => self.publish_event(event),
                AgentUserStateUpdate::SetAwayDeadline(at) => self.set_deadline(Timer::UserAway, at),
                AgentUserStateUpdate::ClearAwayDeadline => self.clear_deadline(Timer::UserAway),
            }
        }
    }
}
