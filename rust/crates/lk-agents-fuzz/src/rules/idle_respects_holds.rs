//! `wait_for_idle` is not answered while another caller's floor hold is outstanding.
//!
//! A hold is live from the moment the driver answered the `HoldFloor` that took it until it
//! releases it; the run opens and closes a `Hold` resource on exactly those two journal steps. A
//! waiter that names the hold passes through; every other waiter must wait.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{Input, Ref},
    record::{Answer, Detail, Kind, Recording},
};

pub struct IdleRespectsHolds;

impl Rule for IdleRespectsHolds {
    const CONTRACT: Contract = Contract {
        id: "idle_respects_holds",
        guarantee: "WaitForIdle is answered Ok only when no other caller's floor hold is outstanding.",
        examples: &[idle_after_release],
        counterexamples: &[idle_while_held],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        for call in host.api_calls() {
            let Some((at, Answer::IdleOk)) = call.returned() else { continue };
            let Input::WaitForIdle { hold: own } = call.input() else { continue };

            for hold in host.holds() {
                // A hold taken after the call, or the waiter's own hold, does not bind it.
                if hold.taken_at() > at || own.map(|o| o.nth) == Some(hold.nth()) {
                    continue;
                }
                // Every other hold taken before the call must have been released before it too.
                let released = hold.released_at().is_some_and(|released| released < at);
                check.expect(
                    &call,
                    released,
                    format!("idle answered while hold {} was outstanding", hold.nth()),
                );
            }
        }
    }
}

/// A hold was taken and released before the idle call.
fn idle_after_release() -> Recording {
    Authored::new()
        .sent_opening(Input::HoldFloor {}, &[(Kind::Hold, Detail::None)])
        .sent(Input::ReleaseFloor { hold: Ref::new(0) })
        .closed(Kind::Hold, 0)
        .sent_opening(Input::WaitForIdle { hold: None }, &[(Kind::Answer, Detail::None)])
        .result(0, Answer::IdleOk)
        .done()
}

/// One caller holds the floor, another asks for idle without naming it, and is answered.
fn idle_while_held() -> Recording {
    Authored::new()
        .sent_opening(Input::HoldFloor {}, &[(Kind::Hold, Detail::None)])
        .sent_opening(Input::WaitForIdle { hold: None }, &[(Kind::Answer, Detail::None)])
        .result(0, Answer::IdleOk)
        .done()
}
