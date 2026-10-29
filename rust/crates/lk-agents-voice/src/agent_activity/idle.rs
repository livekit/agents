//! Idle predicates and waiter ownership shared by speech, turns and tools.

use crate::agent_activity::scheduling::{SpeechId, Status};
use crate::agent_session::effects::Out;
use crate::agent_session::{Driver, IdleRefused};
use crate::trace::SpeechStep;
use lk_agents_abi::Ts;

impl Driver {
    /// `wait_for_idle`: nothing is speaking, queued, winding down or deciding a turn, and no user
    /// turn is claimed. A turn held for the agent a handoff brings in does not count: the reference
    /// has no such turn (it drops it), and the drain it waits on may itself wait on idle, through
    /// a tool of the speech being drained.
    pub(crate) fn is_idle(&self) -> bool {
        use crate::audio_recognition::Phase;
        let activity = &self.activity;
        // The user is busy while speaking or while an end-of-turn countdown runs
        // (`_audio_recognition._speaking`, `_end_of_turn_task`). A turn that ended with no
        // countdown waits on words that may never come, and does not hold idle off.
        let deciding = match activity.audio_recognition.phase() {
            Phase::Idle | Phase::Ended { .. } => false,
            Phase::Speaking { .. }
            | Phase::Awaiting { .. }
            | Phase::Delaying { .. }
            | Phase::ManualCommitted { .. } => true,
            Phase::Committed { speaking, .. } => *speaking,
        };
        self.agent_idle() && !deciding && self.session.states.user_turn_claims() == 0
    }

    /// `wait_for_idle(wait_for_agent=True, wait_for_user=False)`: no speech current, queued or
    /// winding down, no end-of-turn countdown running and no turn parked on its hook.
    pub(crate) fn agent_idle(&self) -> bool {
        let activity = &self.activity;
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

    pub(crate) fn settle_waiters(&mut self, now: Ts, out: &mut Out) {
        self.settle_interrupt_waiters(out);

        // A pending deferred tool reply takes the idle instant first.
        self.try_deliver_tool_reply(now, out);
        self.try_present_experts(now, out);
        self.pump_fillers(now, out);
        self.settle_turn_limit(now, out);
        if !self.session.idle.has_waiters() || !self.is_idle() {
            return;
        }
        // `wait_for_idle` raises `ActivityClosedError` when it wakes on a closing session: idle
        // reached during a close's drain grants nothing, or a hold would outlive the session.
        if self.session.closing.is_some() {
            self.refuse_idle_waiters(out);
            return;
        }
        for waiter in self.session.idle.admit() {
            out.trace(SpeechStep::IdleAnswered { hold: waiter.grant, closing: false });
            out.answer(waiter.call, waiter.answer(Ok(())));
        }
    }

    pub(crate) fn withdraw_idle_asks(&mut self, speech: SpeechId, out: &mut Out) {
        for w in self.session.idle.withdraw(speech) {
            out.trace(SpeechStep::IdleAskWithdrawn { hold: w.grant, speech });
            if let Some(id) = w.grant {
                self.run_release(crate::run_result::Owned::Hold(id), out);
            }
            out.answer(w.call, w.answer(Err(IdleRefused::Closing)));
        }
    }

    pub(crate) fn refuse_idle_waiters(&mut self, out: &mut Out) {
        for waiter in self.session.idle.take_waiters() {
            // A refused hold was never granted, so none is named.
            out.trace(SpeechStep::IdleAnswered { hold: None, closing: true });
            // A hold refused was never granted: whatever run owned the ask lets it go.
            if let Some(id) = waiter.grant {
                self.run_release(crate::run_result::Owned::Hold(id), out);
            }
            out.answer(waiter.call, waiter.answer(Err(IdleRefused::Closing)));
        }
    }
}
