//! Idle predicates and waiter ownership shared by speech, turns and tools.

use super::AgentActivity;
use crate::agent_activity::scheduling::{SpeechId, Status};
use crate::agent_session::{IdleRefused, IdleWaiter, RequestId, VoiceDriver};
use crate::trace::{SpeechStep, TurnStep};
use lk_agents_abi::Ts;

impl VoiceDriver {
    /// `wait_for_idle`: nothing is speaking, queued, winding down or deciding a turn, and no user
    /// turn is claimed. A turn held for the agent a handoff brings in does not count: the reference
    /// has no such turn (it drops it), and the drain it waits on may itself wait on idle, through
    /// a tool of the speech being drained.
    pub(crate) fn is_idle(&self) -> bool {
        self.activity.is_idle()
            && !self.activity.audio_recognition.is_deciding_turn()
            && self.session.states.user_turn_claims() == 0
    }

    pub(crate) fn settle_waiters(&mut self, now: Ts) {
        for waiter in self.activity.take_settled_interrupt_waiters() {
            self.record_trace(crate::trace::InterruptStep::InterruptSettled {
                speeches: waiter.speeches,
            });
            self.answer(waiter.call, crate::agent_session::Answer::Interrupt(Ok(())));
        }

        // A pending deferred tool reply takes the idle instant first.
        self.try_deliver_tool_reply();
        self.try_present_experts(now);
        self.pump_fillers(now);
        self.settle_turn_limit(now);
        if !self.session.idle.has_waiters() || !self.is_idle() {
            return;
        }
        // `wait_for_idle` raises `ActivityClosedError` when it wakes on a closing session: idle
        // reached during a close's drain grants nothing, or a hold would outlive the session.
        if self.session.closing.is_some() {
            self.refuse_idle_waiters();
            return;
        }
        for waiter in self.session.idle.admit() {
            self.record_trace(SpeechStep::IdleAnswered { hold: waiter.grant, closing: false });
            self.answer(waiter.call, waiter.answer(Ok(())));
        }
    }

    /// `RunContext.foreground()`. The hold is minted at the ask, so the run a tool's hold belongs
    /// to can own it before idle.
    pub(crate) fn wait_for_idle_and_hold(&mut self, call: Option<RequestId>) {
        let id = self.session.idle.mint_hold();
        let speech = self.tool_speech(&self.command_origin);
        let scheduler = &self.activity.speech_scheduler;
        let run = self
            .session
            .runs
            .current()
            .filter(|run| speech.is_some_and(|speech| scheduler.belongs_to_run(speech, run.id)))
            .map(|run| run.id);
        self.session.idle.ask(IdleWaiter { hold: None, speech, grant: Some(id), call, run });
    }

    /// Pins the user state to `speaking` and holds idle open. Reentrant.
    pub(crate) fn claim_user_turn(&mut self, now: Ts) {
        let (depth, eff) = self.session.states.claim_user_turn(now);
        self.apply_user_state_updates(eff);
        self.record_trace(TurnStep::UserTurnClaims { depth });
    }

    /// The last release re-derives the user state from what the recognizer hears.
    pub(crate) fn release_user_turn(&mut self, now: Ts) {
        let speaking = self.activity.audio_recognition.user_speaking();
        let (depth, eff) = self.session.states.release_user_turn(now, speaking);
        self.apply_user_state_updates(eff);
        self.record_trace(TurnStep::UserTurnClaims { depth });
    }

    pub(crate) fn withdraw_idle_asks(&mut self, speech: SpeechId) {
        for w in self.session.idle.withdraw(speech) {
            self.record_trace(SpeechStep::IdleAskWithdrawn { hold: w.grant, speech });
            self.answer(w.call, w.answer(Err(IdleRefused::Closing)));
        }
    }

    pub(crate) fn refuse_idle_waiters(&mut self) {
        for waiter in self.session.idle.take_waiters() {
            // A refused hold was never granted, so none is named.
            self.record_trace(SpeechStep::IdleAnswered { hold: None, closing: true });
            // A hold refused was never granted: whatever run owned the ask lets it go.
            self.answer(waiter.call, waiter.answer(Err(IdleRefused::Closing)));
        }
    }
}

impl AgentActivity {
    pub(crate) fn is_idle(&self) -> bool {
        let activity = self;
        // Busy: every held handle is in flight, except one whose tools run (the current speech is
        // released for `foreground()` from inside a tool) and a live preemptive generation.
        let scheduling = &activity.speech_scheduler;
        let winding_down = scheduling.speech_ids().any(|id| match scheduling.status(id) {
            None => true,
            Some(Status::Working) => !activity.tools_in_flight.contains(&id),
            Some(Status::Preemptive) => false,
            Some(Status::Current | Status::Queued | Status::Discarded) => true,
        });
        scheduling.no_current_speech()
            && scheduling.speech_q().is_empty()
            && !winding_down
            && !activity.audio_recognition.phase().is_running()
            && activity.pending_turn.is_none()
    }
}
