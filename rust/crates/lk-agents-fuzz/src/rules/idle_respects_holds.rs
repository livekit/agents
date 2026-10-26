//! `wait_for_idle` is not answered while another caller's floor hold is outstanding.
//!
//! A hold is live from the moment the driver answered the `WaitForIdleAndHold` that took it until it
//! releases it; the run opens and closes a `Hold` resource on exactly those two journal steps. A
//! waiter that names the hold passes through; every other waiter must wait.
use lk_agents_voice::trace::{SpeechStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{Input, Ref},
    record::{Answer, Detail, Kind, Observed, Recording},
};

pub struct IdleRespectsHolds;

impl Rule for IdleRespectsHolds {
    const CONTRACT: Contract = Contract {
        id: "idle_respects_holds",
        guarantee: "WaitForIdle is answered Ok only when no other caller's floor hold is outstanding.",
        examples: &[idle_after_release, admitted_before_the_grant],
        counterexamples: &[idle_while_held, admitted_while_held],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        // Where the driver admitted each wait: its `IdleAnswered`, journalled as it answers. The
        // answer reaches the host after the whole pass, so a hold granted later in the same pass
        // (to an ask queued behind this one) is stamped before the answer but did not bind it.
        let admissions: Vec<Observed> = r
            .journal_where(|step| match step {
                Step::Speech(SpeechStep::IdleAnswered { hold: None, closing: false }) => Some(()),
                _ => None,
            })
            .map(|(at, ())| at)
            .collect();
        for call in host.api_calls() {
            let Some((answered, Answer::IdleOk)) = call.returned() else { continue };
            let Input::WaitForIdle { hold: own } = call.input() else { continue };
            let at = admissions.iter().rev().find(|a| **a < answered).copied().unwrap_or(answered);

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
        .sent_opening(Input::WaitForIdleAndHold {}, &[(Kind::Hold, Detail::None)])
        .sent(Input::ReleaseIdleHold { hold: Ref::new(0) })
        .closed(Kind::Hold, 0)
        .sent_opening(Input::WaitForIdle { hold: None }, &[(Kind::Answer, Detail::None)])
        .result(0, Answer::IdleOk)
        .done()
}

/// One caller holds the floor, another asks for idle without naming it, and is answered.
fn idle_while_held() -> Recording {
    Authored::new()
        .sent_opening(Input::WaitForIdleAndHold {}, &[(Kind::Hold, Detail::None)])
        .sent_opening(Input::WaitForIdle { hold: None }, &[(Kind::Answer, Detail::None)])
        .result(0, Answer::IdleOk)
        .done()
}

fn admitted() -> Step {
    Step::Speech(SpeechStep::IdleAnswered { hold: None, closing: false })
}

/// A released hold, then the wait is admitted and a hold asked after it is granted in the same
/// pass; the answer reaches the host after both.
fn admitted_before_the_grant() -> Recording {
    Authored::new()
        .sent_opening(Input::WaitForIdleAndHold {}, &[(Kind::Hold, Detail::None)])
        .sent(Input::ReleaseIdleHold { hold: Ref::new(0) })
        .closed(Kind::Hold, 0)
        .sent_opening(Input::WaitForIdle { hold: None }, &[(Kind::Answer, Detail::None)])
        .journal(admitted())
        .sent_opening(Input::WaitForIdleAndHold {}, &[(Kind::Hold, Detail::None)])
        .result(0, Answer::IdleOk)
        .done()
}

/// The hold is granted, then the wait is admitted while it is still held.
fn admitted_while_held() -> Recording {
    Authored::new()
        .sent_opening(Input::WaitForIdleAndHold {}, &[(Kind::Hold, Detail::None)])
        .sent_opening(Input::WaitForIdle { hold: None }, &[(Kind::Answer, Detail::None)])
        .journal(admitted())
        .result(0, Answer::IdleOk)
        .done()
}
