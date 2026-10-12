//! Every API call gets its result.
//!
//! A call that returns a value -- `say`, `generate_reply`, `interrupt`, `commit_user_turn`,
//! `wait_for_idle`, `hold_floor`, `handoff`, `drain`, a DTMF start -- is a future the developer
//! awaits, and the driver owes it a result even while closing: a pending drain joins the close's
//! own, a handoff is refused as closing, a DTMF collection is answered stopped. The one future the
//! driver drops on purpose is a `say` or `generate_reply` that reached the session after the host
//! asked it to drain or close, or after the driver began closing on its own: it never spoke, and
//! the dropped future says so. The other is a `commit_user_turn` a newer commit superseded, which
//! the API reports as superseded. Any other dropped future, and any call with no result once the
//! driver has exited, is a call the driver forgot.
use lk_agents_voice::trace::{Step, TurnStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{AgentText, Input, Millis, SayInput},
    record::{Answer, Recording},
};

pub struct ApiCallsReturn;

impl Rule for ApiCallsReturn {
    const CONTRACT: Contract = Contract {
        id: "api_calls_return",
        guarantee: "Every API call gets its result; only a say or generate_reply made after the session began closing may have its future dropped.",
        examples: &[returned, refused_after_closing_began, commit_superseded],
        counterexamples: &[dropped_before_closing_began, no_result_at_exit],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // A run the driver never exited cannot judge a call; `session_settles` and
        // `run_terminates` report those runs.
        if !r.driver_finished() {
            return;
        }
        let began_closing = [
            r.inputs_view().closing_asked_at(),
            r.internal_view().close_requested().map(|(at, _)| at),
        ]
        .into_iter()
        .flatten()
        .min();

        for call in r.host_view().api_calls() {
            let verdict = match call.returned() {
                Some((_, Answer::Lost)) => {
                    let speaks = matches!(call.input(), Input::Say { .. } | Input::GenerateReply { .. });
                    let after_closing_began = began_closing.is_some_and(|began| call.called_at() > began);
                    let superseded = matches!(call.input(), Input::CommitUserTurn { .. })
                        && r.internal_view().commit_superseded_after(call.called_at());
                    ((speaks && after_closing_began) || superseded)
                        .then_some(())
                        .ok_or("its future was dropped")
                }
                Some(_) => Ok(()),
                None => Err("it never got a result"),
            };
            check.expect(&call, verdict.is_ok(), format!("{:?}: {}", call.input(), verdict.err().unwrap_or("")));
        }
    }
}

fn say() -> Input {
    Input::Say {
        text: AgentText("hello".into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    }
}

/// `say` returned its speech.
fn returned() -> Recording {
    Authored::new().api_call(say()).returns(Answer::Speech(1)).driver_finished().done()
}

/// The host asked to drain, then called `say`: the driver refused it by dropping the future.
fn refused_after_closing_began() -> Recording {
    Authored::new()
        .api_call(Input::Drain { wait: true })
        .returns(Answer::DrainOk)
        .api_call(say())
        .returns(Answer::Lost)
        .driver_finished()
        .done()
}

/// A second `commit_user_turn` superseded the first, whose future the driver dropped.
fn commit_superseded() -> Recording {
    let commit = Input::CommitUserTurn {
        transcript_timeout: Millis(1000),
        stt_flush: Millis(100),
        skip_reply: false,
    };
    Authored::new()
        .api_call(commit.clone())
        .api_call(commit)
        .journal(Step::Turn(TurnStep::CommitSuperseded))
        .result(0, Answer::Lost)
        .returns(Answer::Committed("hello".into()))
        .driver_finished()
        .done()
}

/// Nobody asked to close and the `say`'s future was dropped.
fn dropped_before_closing_began() -> Recording {
    Authored::new().api_call(say()).returns(Answer::Lost).driver_finished().done()
}

/// The driver exited and the `say` never got a result.
fn no_result_at_exit() -> Recording {
    Authored::new().api_call(say()).driver_finished().done()
}
