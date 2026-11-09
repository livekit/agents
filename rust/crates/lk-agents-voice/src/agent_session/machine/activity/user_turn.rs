//! User turns opened from code or a client's button (`start_user_turn()`), and push-to-talk.
//!
//! An open turn holds the turn machine to explicit turns: only its `end()` or `cancel()` closes
//! it. The turn's STT stream is the one installed at the press, or the one that lands for it. On
//! release the router feeds that stream a short tail and ends its input; the turn commits once the
//! stream has delivered every final and closed, and a fresh stream is opened for what follows. A
//! cancelled turn's stream is aborted, so nothing it heard is ever finalized.

use std::time::Duration;

use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::events::{UserTurnEndedEvent, UserTurnResult, UserTurnStartedEvent};
use lk_agents_abi::voice::session::UserTurnFailed;
use lk_agents_abi::{ChatMessage, Maybe, Ts};

use crate::agent_session::effects::{Timer, VoiceEffect};
use crate::agent_session::{Answer, RequestId, VoiceDriver};
use crate::audio::AudioChannel;
use crate::audio_recognition::{TurnDetectionMode, TurnId};
use crate::data_plane::AudioInputRouterCommand;
use crate::events::SessionEvent;
use crate::telemetry;
use crate::trace::TurnStep;

/// How long a released turn waits for its STT stream to deliver every final; past it, the turn
/// commits what arrived.
pub const USER_TURN_END_TIMEOUT: Duration = Duration::from_secs(5);

/// A released turn's tail ends on this much session time even if the microphone went quiet:
/// audio flows in real time, so twice the tail is the most it can lag.
const USER_TURN_TAIL_BOUND: Duration = Duration::from_millis(400);

/// The session's user turn: it outlives an agent handoff.
#[derive(Default)]
pub(crate) struct UserTurns {
    open: Option<OpenTurn>,
}

struct OpenTurn {
    id: u64,
    /// Set by `end()`.
    ending: Option<Ending>,
}

struct Ending {
    /// The stream that heard the turn, until it closed or was dropped.
    stream: Option<ResourceId>,
    reply: bool,
    call: Option<RequestId>,
    /// The recogniser's turn the release committed; the turn ends when its message is added.
    decided: Option<TurnId>,
}

impl UserTurns {
    /// The released turn's stream, whose items still belong to the turn.
    pub(crate) fn ending_stream(&self) -> Option<ResourceId> {
        self.open.as_ref()?.ending.as_ref()?.stream
    }

    pub(crate) fn is_open(&self) -> bool {
        self.open.is_some()
    }
}

impl VoiceDriver {
    fn push_to_talk(&self) -> bool {
        self.session.config.effective().turn.push_to_talk()
    }

    pub(crate) fn route_audio(&mut self, command: AudioInputRouterCommand) {
        self.record_effect(VoiceEffect::UpdateAudioInputRouting(command));
    }

    /// Push-to-talk is on, or a turn is open: only a turn's `end()` ends it.
    pub(crate) fn explicit_turns(&self) -> bool {
        self.push_to_talk() || self.session.user_turns.is_open()
    }

    /// The turn machine runs explicit turns under push-to-talk and while a turn is open, else
    /// the configured detection.
    pub(crate) fn sync_turn_mode(&mut self, now: Ts) {
        let mode = if self.explicit_turns() {
            TurnDetectionMode::Manual
        } else {
            self.session.config.effective().turn.owner()
        };
        let eff = self.activity.audio_recognition.set_turn_detection_mode(now, mode);
        self.apply_recognition_effects(eff);
    }

    /// `push_to_talk` changed: off ends an open turn as a release would.
    pub(crate) fn set_push_to_talk(&mut self, now: Ts, on: bool) {
        if !on && let Some(id) = self.session.user_turns.open.as_ref().map(|turn| turn.id) {
            self.end_user_turn(now, id, true, None);
        }
        self.route_audio(AudioInputRouterCommand::SetPushToTalk(on));
        self.sync_turn_mode(now);
    }

    pub(crate) fn start_user_turn(&mut self, now: Ts, id: u64) {
        if self.session.user_turns.is_open() || self.session.closing.is_some() {
            return self.record_trace(TurnStep::UserTurnRefused { turn: id });
        }
        self.interrupt_for_user_turn(now);
        // Words heard before the press belong to no turn.
        let eff = self.activity.audio_recognition.reset_user_turn(now);
        self.apply_recognition_effects(eff);
        // A stream that listened before the press may still be finishing words said before it:
        // the turn gets a fresh one, and its first audio waits for it. Under push-to-talk the
        // stream heard nothing outside a turn.
        if !self.push_to_talk()
            && let Some(heard) = self.input.live_stt.take()
        {
            self.input.sinks.remove(&AudioChannel::Stt);
            self.route_audio(AudioInputRouterCommand::Remove(heard));
        }
        self.session.user_turns.open = Some(OpenTurn { id, ending: None });
        self.sync_turn_mode(now);
        self.route_audio(AudioInputRouterCommand::StartTurn);
        self.claim_user_turn(now);
        if self.input.live_stt.is_none() {
            self.reopen_stt();
        }
        self.record_trace(TurnStep::UserTurnStarted { turn: id });
        self.publish_event(SessionEvent::UserTurnStarted(UserTurnStartedEvent {
            turn_id: id,
            created_at: 0.0,
        }));
    }

    pub(crate) fn end_user_turn(&mut self, now: Ts, id: u64, reply: bool, call: Option<RequestId>) {
        let open = self.session.user_turns.open.as_mut();
        let Some(turn) = open.filter(|turn| turn.id == id && turn.ending.is_none()) else {
            return self.answer(call, Answer::UserTurn(Box::new(Err(UserTurnFailed::Ended))));
        };
        // The router takes the stream for the turn's tail; the next turn gets a fresh one.
        let stream = self.input.live_stt.take();
        if stream.is_some() {
            self.input.sinks.remove(&AudioChannel::Stt);
        }
        turn.ending = Some(Ending { stream, reply, call, decided: None });
        self.route_audio(AudioInputRouterCommand::EndTurn);
        self.release_user_turn(now);
        self.record_trace(TurnStep::UserTurnReleased { turn: id, reply });
        if stream.is_none() {
            return self.commit_released_turn(now);
        }
        self.set_deadline(Timer::UserTurnEnd, now.saturating_add(USER_TURN_END_TIMEOUT));
        self.set_deadline(Timer::UserTurnTail, now.saturating_add(USER_TURN_TAIL_BOUND));
        self.reopen_stt();
    }

    pub(crate) fn cancel_user_turn(&mut self, now: Ts, id: u64) {
        let open = self.session.user_turns.open.as_ref();
        if !open.is_some_and(|turn| turn.id == id && turn.ending.is_none()) {
            return;
        }
        self.session.user_turns.open = None;
        self.route_audio(AudioInputRouterCommand::CancelTurn);
        if self.input.live_stt.take().is_some() {
            self.input.sinks.remove(&AudioChannel::Stt);
        }
        self.drop_user_turn_words(now);
        self.release_user_turn(now);
        self.record_trace(TurnStep::UserTurnCancelled { turn: id });
        self.publish_ended(id, None);
        self.sync_turn_mode(now);
        self.reopen_stt();
    }

    /// The close cancels an open turn; a released one's `end()` fails.
    pub(crate) fn close_user_turn(&mut self, now: Ts) {
        let Some(turn) = self.session.user_turns.open.take() else { return };
        self.clear_deadline_if_set(Timer::UserTurnEnd);
        self.clear_deadline_if_set(Timer::UserTurnTail);
        match turn.ending {
            None => {
                self.route_audio(AudioInputRouterCommand::CancelTurn);
                self.drop_user_turn_words(now);
                self.release_user_turn(now);
            }
            Some(ending) => {
                if let Some(stream) = ending.stream {
                    self.route_audio(AudioInputRouterCommand::Abort(stream));
                }
                self.answer(ending.call, Answer::UserTurn(Box::new(Err(UserTurnFailed::Closed))));
            }
        }
        self.record_trace(TurnStep::UserTurnCancelled { turn: turn.id });
        self.publish_ended(turn.id, None);
    }

    /// `user_turn_limit` passed while a turn is open: the turn ends as a release would. Whether
    /// a turn was open.
    pub(crate) fn user_turn_limit_reached(&mut self, now: Ts) -> bool {
        let open = self.session.user_turns.open.as_ref();
        let Some(id) = open.filter(|turn| turn.ending.is_none()).map(|turn| turn.id) else {
            return open.is_some();
        };
        self.end_user_turn(now, id, true, None);
        true
    }

    /// A stream ended: the released turn's commits.
    pub(crate) fn user_turn_stream_ended(&mut self, now: Ts, stream: ResourceId) {
        if self.session.user_turns.ending_stream() != Some(stream) {
            return;
        }
        if let Some(ending) = self.ending_mut() {
            ending.stream = None;
        }
        self.commit_released_turn(now);
    }

    pub(crate) fn user_turn_end_timed_out(&mut self, now: Ts) {
        let Some(stream) = self.ending_mut().and_then(|ending| ending.stream.take()) else {
            return;
        };
        self.route_audio(AudioInputRouterCommand::Abort(stream));
        self.publish_event(SessionEvent::Diagnostic(crate::events::DiagnosticEvent::warning(
            crate::events::diagnostics::USER_TURN_END_TIMEOUT,
        )));
        self.commit_released_turn(now);
    }

    /// The recogniser's turn `turn` added `message`: the released turn it came from ends.
    pub(crate) fn user_turn_message_added(&mut self, turn: TurnId, message: &ChatMessage) {
        if self.ending_mut().is_some_and(|ending| ending.decided == Some(turn)) {
            self.settle_user_turn(Some(message.clone()), false);
        }
    }

    /// The recogniser's turn `turn` ended with no message (the hook stopped it, or it was
    /// dropped behind an uninterruptible speech).
    pub(crate) fn user_turn_dropped(&mut self, turn: TurnId) {
        if self.ending_mut().is_some_and(|ending| ending.decided == Some(turn)) {
            self.settle_user_turn(None, false);
        }
    }

    /// The recogniser's turn `from` was merged into `to`, which now carries its words.
    pub(crate) fn user_turn_renamed(&mut self, from: TurnId, to: TurnId) {
        if let Some(ending) = self.ending_mut().filter(|ending| ending.decided == Some(from)) {
            ending.decided = Some(to);
        }
    }

    fn ending_mut(&mut self) -> Option<&mut Ending> {
        self.session.user_turns.open.as_mut()?.ending.as_mut()
    }

    /// Every final arrived (or the wait ran out): what the turn holds commits now.
    fn commit_released_turn(&mut self, now: Ts) {
        self.clear_deadline_if_set(Timer::UserTurnEnd);
        self.clear_deadline_if_set(Timer::UserTurnTail);
        let Some(reply) = self.ending_mut().map(|ending| ending.reply) else { return };
        let released = telemetry::Endpointing { source: telemetry::EouSource::Manual, delay: 0.0 };
        if let Some(turn) = &mut self.activity.user_turn {
            turn.retriggered(&mut self.telemetry, released);
        }
        let (decided, eff) = self.activity.audio_recognition.commit_release(now, !reply);
        match decided {
            // Named before the effects run: the message may be added within them.
            Some(turn) => {
                if let Some(ending) = self.ending_mut() {
                    ending.decided = Some(turn);
                }
                self.apply_recognition_effects(eff);
            }
            None => {
                self.apply_recognition_effects(eff);
                if let Some(mut turn) = self.activity.user_turn.take() {
                    turn.end(&mut self.telemetry, telemetry::UserTurnOutcome::Cleared);
                }
                self.settle_user_turn(None, true);
            }
        }
        self.cancel_preemptive_generation(
            now,
            crate::agent_session::machine::activity::scheduling::PreemptiveDiscard::TurnAbandoned,
        );
    }

    fn settle_user_turn(&mut self, message: Option<ChatMessage>, empty: bool) {
        let Some(turn) = self.session.user_turns.open.take() else { return };
        let call = turn.ending.and_then(|ending| ending.call);
        let result = UserTurnResult { message, empty };
        self.record_trace(TurnStep::UserTurnEnded { turn: turn.id, empty });
        self.publish_ended(turn.id, Some(&result));
        self.answer(call, Answer::UserTurn(Box::new(Ok(result))));
        self.sync_turn_mode(self.step.now);
    }

    fn publish_ended(&mut self, turn_id: u64, result: Option<&UserTurnResult>) {
        self.publish_event(SessionEvent::UserTurnEnded(UserTurnEndedEvent {
            turn_id,
            cancelled: result.is_none(),
            empty: result.is_some_and(|result| result.empty),
            created_at: 0.0,
        }));
    }

    /// The words a turn heard so far are dropped, uncommitted.
    fn drop_user_turn_words(&mut self, now: Ts) {
        if let Some(mut turn) = self.activity.user_turn.take() {
            turn.end(&mut self.telemetry, telemetry::UserTurnOutcome::Cleared);
        }
        let eff = self.activity.audio_recognition.reset_user_turn(now);
        self.apply_recognition_effects(eff);
    }

    /// A fresh stream from the session's recogniser, for the next turn.
    fn reopen_stt(&mut self) {
        if let Some(stt) = self.session.stt.clone() {
            self.reopen_legs(Maybe::Given(Some(stt)), Maybe::NotGiven);
        }
    }
}
