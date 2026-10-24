//! `wait_for_idle()` and `hold_floor()`: the asks waiting for an idle instant, and the floor holds
//! that only their own asks pass (`_idle_holds`).

use std::collections::BTreeSet;

use crate::agent_activity::scheduling::SpeechId;

use super::{Answer, CallId, HoldId, IdleRefused};

pub(crate) struct IdleWaiter {
    /// The hold the caller is inside, which it may pass through.
    pub(crate) hold: Option<HoldId>,
    /// The speech whose tool asked; its interruption withdraws the ask. `None` for the host.
    pub(crate) speech: Option<SpeechId>,
    /// `hold_floor()`'s hold: minted at the ask, granted when idle answers it.
    pub(crate) grant: Option<HoldId>,
    pub(crate) call: Option<CallId>,
}

impl IdleWaiter {
    pub(crate) fn answer(&self, result: Result<(), IdleRefused>) -> Answer {
        match self.grant {
            None => Answer::Idle(result),
            Some(id) => Answer::Hold(result.map(|()| id)),
        }
    }
}

#[derive(Default)]
pub(crate) struct IdleGate {
    waiters: Vec<IdleWaiter>,
    /// While non-empty, only a waiter naming one of them passes.
    holds: BTreeSet<HoldId>,
    next_hold: u64,
}

impl IdleGate {
    /// A floor hold's id, minted at the ask.
    pub(crate) fn mint_hold(&mut self) -> HoldId {
        self.next_hold += 1;
        HoldId(self.next_hold)
    }

    /// Answered at the first idle instant the ask passes.
    pub(crate) fn ask(&mut self, waiter: IdleWaiter) {
        self.waiters.push(waiter);
    }

    pub(crate) fn has_waiters(&self) -> bool {
        !self.waiters.is_empty()
    }

    /// `_idle_holds > 0`.
    pub(crate) fn held(&self) -> bool {
        !self.holds.is_empty()
    }

    /// The asks the holds let through, in ask order, their holds granted as they pass: a hold
    /// granted holds off the asks after it. While a hold is held only an ask inside it passes
    /// (`_idle_holds > 0 and not _IdleHoldContextVar.get()`).
    pub(crate) fn admit(&mut self) -> Vec<IdleWaiter> {
        let mut admitted = Vec::new();
        for waiter in std::mem::take(&mut self.waiters) {
            if !self.holds.is_empty() && !waiter.hold.is_some_and(|h| self.holds.contains(&h)) {
                self.waiters.push(waiter);
                continue;
            }
            self.holds.extend(waiter.grant);
            admitted.push(waiter);
        }
        admitted
    }

    /// The asks `speech`'s tools made, withdrawn with it.
    pub(crate) fn withdraw(&mut self, speech: SpeechId) -> Vec<IdleWaiter> {
        let (gone, kept) =
            std::mem::take(&mut self.waiters).into_iter().partition(|w| w.speech == Some(speech));
        self.waiters = kept;
        gone
    }

    /// Every ask still waiting.
    pub(crate) fn take_waiters(&mut self) -> Vec<IdleWaiter> {
        std::mem::take(&mut self.waiters)
    }

    /// `release_floor(hold)`: whether it was held.
    pub(crate) fn release(&mut self, hold: HoldId) -> bool {
        self.holds.remove(&hold)
    }

    /// `hold` is still asked for or held: its `foreground()` scope has not exited.
    pub(crate) fn live(&self, hold: HoldId) -> bool {
        self.holds.contains(&hold) || self.waiters.iter().any(|w| w.grant == Some(hold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(gate: &mut IdleGate, hold: Option<HoldId>, grant: Option<HoldId>) {
        gate.ask(IdleWaiter { hold, speech: None, grant, call: None });
    }

    // Catches: holds checked against the state before the pass, so a wait asked after a
    // `hold_floor()` slips past the hold granted in the same idle instant.
    #[test]
    fn a_hold_granted_in_a_pass_holds_off_the_asks_after_it() {
        let mut gate = IdleGate::default();
        let floor = gate.mint_hold();
        ask(&mut gate, None, Some(floor));
        ask(&mut gate, None, None);
        ask(&mut gate, Some(floor), None);
        let admitted: Vec<_> = gate.admit().iter().map(|w| (w.hold, w.grant)).collect();
        assert_eq!(admitted, [(None, Some(floor)), (Some(floor), None)]);
        assert!(gate.held() && gate.has_waiters());
        assert!(gate.release(floor));
        assert_eq!(gate.admit().len(), 1, "released, the outside ask passes");
    }
}
