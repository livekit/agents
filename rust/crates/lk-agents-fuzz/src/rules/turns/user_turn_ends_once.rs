use std::collections::BTreeMap;

use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{Step, TurnStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

// A user turn is the host's (`start_user_turn()`): one at a time, and each is told ended once,
// released or cancelled, by the time the session closed.

pub struct UserTurnEndsOnce;

impl Rule for UserTurnEndsOnce {
    const CONTRACT: Contract = Contract {
        id: "user_turn_ends_once",
        guarantee: "One user turn is open at a time, and each that started is told ended exactly once, by the session's close at the latest.",
        examples: &[released, cancelled_then_another],
        counterexamples: &[ended_twice, never_ended_at_close, two_open],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let mut ended: BTreeMap<u64, u32> = BTreeMap::new();
        let mut open: Option<u64> = None;
        let mut closed = false;
        for (at, step) in r.journal() {
            match step {
                Step::Turn(TurnStep::UserTurnStarted { turn }) => {
                    check.expect_with(Subject::Session, open.is_none(), || {
                        format!("turn {turn} opened at {at:?} while turn {open:?} was open")
                    });
                    open = Some(*turn);
                    ended.entry(*turn).or_default();
                }
                Step::Emitted(SessionEvent::UserTurnEnded(e)) => {
                    *ended.entry(e.turn_id).or_default() += 1;
                    if open == Some(e.turn_id) {
                        open = None;
                    }
                }
                Step::Emitted(SessionEvent::Close { .. }) => closed = true,
                _ => {}
            }
        }
        for (turn, told) in ended {
            check.expect_with(Subject::Session, told <= 1, || {
                format!("turn {turn} told ended {told} times")
            });
            if closed {
                check.expect_with(Subject::Session, told == 1, || {
                    format!("turn {turn} still open when the session closed")
                });
            }
        }
    }
}

fn started(turn: u64) -> Step {
    Step::Turn(TurnStep::UserTurnStarted { turn })
}

fn ended(turn: u64, released: bool) -> Step {
    Step::Emitted(SessionEvent::UserTurnEnded(lk_agents_abi::voice::events::UserTurnEndedEvent {
        turn_id: turn,
        cancelled: !released,
        empty: released,
        created_at: 0.0,
    }))
}

fn close() -> Step {
    Step::Emitted(SessionEvent::Close {
        reason: lk_agents_abi::voice::events::CloseReason::UserInitiated,
        error: None,
    })
}

fn released() -> Recording {
    Authored::new().journal(started(1)).journal(ended(1, true)).journal(close()).done()
}

fn cancelled_then_another() -> Recording {
    Authored::new()
        .journal(started(1))
        .journal(ended(1, false))
        .journal(started(2))
        .journal(ended(2, true))
        .done()
}

fn ended_twice() -> Recording {
    Authored::new().journal(started(1)).journal(ended(1, true)).journal(ended(1, false)).done()
}

fn never_ended_at_close() -> Recording {
    Authored::new().journal(started(1)).journal(close()).done()
}

fn two_open() -> Recording {
    Authored::new().journal(started(1)).journal(started(2)).done()
}
