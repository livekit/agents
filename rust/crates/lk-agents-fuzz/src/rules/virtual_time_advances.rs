//! The session never works without bound at one virtual instant.
//!
//! The clock is paused and moves only when nothing can run, so a session that keeps feeding
//! itself -- an event that schedules the next, a task that wakes itself, a step that journals
//! forever -- never lets time move and never hands control back to the case. The run counts
//! steps at one instant, in the journal and in its own loop, and stops the session past
//! `NO_PROGRESS_STEPS`: a livelock, reported here rather than as a hang or a stray panic.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, Ref},
    record::Recording,
};

pub struct VirtualTimeAdvances;

impl Rule for VirtualTimeAdvances {
    const CONTRACT: Contract = Contract {
        id: "virtual_time_advances",
        guarantee: "The session never does unbounded work at one virtual instant: time moves, or \
                    it goes idle.",
        examples: &[went_idle],
        counterexamples: &[livelocked],
        probe: Some(a_reply),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        check.expect_with(Subject::Session, r.livelock().is_none(), || {
            format!("livelock: {}", r.livelock().unwrap_or_default())
        });
    }
}

fn went_idle() -> Recording {
    Authored::new().sent(Input::Close { drain: false }).driver_finished().done()
}

fn livelocked() -> Recording {
    Authored::new().livelocked("100000 steps at 20 ms").done()
}

/// A reply runs to its end with the clock moving under it.
fn a_reply() -> Case {
    Case {
        setup: Setup::default(),
        inputs: vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Hello there.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ],
        ..Case::default()
    }
}
