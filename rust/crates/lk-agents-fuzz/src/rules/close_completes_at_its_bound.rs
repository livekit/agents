//! A session closes within its cleanup bound even when a host callback refuses its cancellation.
//!
//! A tool body that catches its cancellation (`except CancelledError:`) runs on after the close
//! cancelled it. The session must not wait for it: its loop exits, and its close cancels what is
//! still running at `HOST_TEARDOWN_TIMEOUT` past the stop (`Runtime::close`), so a refusal costs
//! the close at most that bound.
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::Step;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{Input, Millis, Ref, ToolArgs},
    record::Recording,
};

pub struct CloseCompletesAtItsBound;

impl Rule for CloseCompletesAtItsBound {
    const CONTRACT: Contract = Contract {
        id: "close_completes_at_its_bound",
        guarantee: "A session whose host refused a cancellation still closes, its close \
                    cancelling what runs on within the cleanup bound past the stop.",
        examples: &[cancelled_at_the_bound],
        counterexamples: &[never_closed, cancelled_past_the_bound],
        probe: Some(a_refusing_tool_at_the_close),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if r.cancels_refused().is_empty() {
            return;
        }
        let stopped = r.journal().iter().find_map(|(at, step)| {
            matches!(step, Step::Emitted(SessionEvent::Close { .. })).then_some(at.at_ms)
        });
        check.expect(
            Subject::Session,
            r.driver_finished() && stopped.is_some(),
            "a host refused a cancellation and the session never closed",
        );
        let Some(stopped) = stopped else { return };
        let bound = lk_agents_abi::HOST_TEARDOWN_TIMEOUT.as_millis() as u64;
        for (at, name) in r.late_tasks() {
            check.expect_with(Subject::Session, at.at_ms <= stopped + bound, || {
                format!(
                    "the close cancelled {name:?} at {} ms, past its bound {} ms after the stop \
                     at {stopped} ms",
                    at.at_ms, bound
                )
            });
        }
    }
}

fn closed() -> Step {
    Step::Emitted(SessionEvent::Close {
        reason: lk_agents_abi::voice::events::CloseReason::UserInitiated,
        error: None,
    })
}

/// The refused call's task is cancelled at the bound, and the session is gone.
fn cancelled_at_the_bound() -> Recording {
    Authored::new()
        .cancel_refused("call_1")
        .journal(closed())
        .later(4_990)
        .late_task("tool call")
        .driver_finished()
        .done()
}

/// A refusal held the session open.
fn never_closed() -> Recording {
    Authored::new().cancel_refused("call_1").done()
}

/// The close waited on the refusing body past its bound.
fn cancelled_past_the_bound() -> Recording {
    Authored::new()
        .cancel_refused("call_1")
        .journal(closed())
        .later(9_000)
        .late_task("tool call")
        .driver_finished()
        .done()
}

/// A tool runs when the session closes, and its body refuses the cancel.
fn a_refusing_tool_at_the_close() -> Case {
    Case {
        setup: Setup {
            tools: vec!["lookup".into()],
            tools_refuse_cancel: true,
            ..Setup::default()
        },
        inputs: vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(20_000) },
            Input::AwaitExit {},
        ],
        ..Case::default()
    }
}
