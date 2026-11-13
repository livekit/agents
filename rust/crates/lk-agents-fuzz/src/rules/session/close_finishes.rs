use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::Input,
    record::Recording,
};

// Every close phase is bounded, so a session not exited a virtual day later is stuck, not slow.
// Without this rule every rule judging a closed session would skip such a run in silence.

pub struct CloseFinishes;

impl Rule for CloseFinishes {
    const CONTRACT: Contract = Contract {
        id: "close_finishes",
        guarantee: "A close, once begun, ends with the session's exit: it never stalls with \
                    nothing left that could wake it.",
        examples: &[exited],
        counterexamples: &[never_exited],
        probe: Some(a_close),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let Some(begun) = r.close_begun() else { return };
        let awaited =
            r.applied().iter().filter(|a| a.at > begun && matches!(a.input, Input::AwaitExit {}));
        for wait in awaited {
            check.expect_with(Subject::Session, wait.accepted, || {
                let stepped = r.journal().last().map_or(0, |(at, _)| at.at_ms);
                format!(
                    "the close begun at {} ms had not finished a virtual day after the wait for \
                     it at {} ms; the session last stepped at {stepped} ms",
                    begun.at_ms, wait.at.at_ms
                )
            });
        }
    }
}

fn exited() -> Recording {
    Authored::new()
        .sent(Input::Close { drain: false })
        .sent(Input::AwaitExit {})
        .driver_finished()
        .done()
}

fn never_exited() -> Recording {
    Authored::new().sent(Input::Close { drain: true }).refused(Input::AwaitExit {}).done()
}

/// A session closed with nothing in flight exits.
fn a_close() -> Case {
    Case {
        setup: Setup::default(),
        inputs: vec![Input::Close { drain: true }, Input::AwaitExit {}],
        ..Case::default()
    }
}
