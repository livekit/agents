//! Nothing a turn opened is still open once the driver has exited.
//!
//! The soak's oracle, held over every run rather than eight long ones. A floor hold, a tool body,
//! a hook call, a filler, a segment awaiting its playout report, an utterance, a prediction, a
//! realtime turn or a stream a `say` opened belongs to a turn; a driver that exits with one still
//! open has left it with nobody to release, answer or stop it. What lives as long as the session
//! (the legs, the outputs, the socket) is rightly still there, and speeches, answers and provider
//! streams have contracts of their own (`speech_settles`, `api_calls_return`,
//! `provider_streams_closed`).
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, Ref},
    record::{Kind, Phase, Recording},
};

pub struct NothingLeftLive;

/// What a turn opens, with no other contract judging what is left of it.
const OWNED_BY_A_TURN: [Kind; 9] = [
    Kind::Hold,
    Kind::Filler,
    Kind::Hook,
    Kind::Tool,
    Kind::Segment,
    Kind::Utterance,
    Kind::Prediction,
    Kind::Generation,
    Kind::Stream,
];

impl Rule for NothingLeftLive {
    const CONTRACT: Contract = Contract {
        id: "nothing_left_live",
        guarantee: "Once the driver has exited, nothing a turn opened (a floor hold, tool, hook \
                    call, filler, segment, utterance, prediction, realtime turn or say stream) is \
                    still open.",
        examples: &[closed_before_exit],
        counterexamples: &[open_at_exit],
        probe: Some(a_reply),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if r.phase() != Phase::Closed {
            return;
        }
        for kind in OWNED_BY_A_TURN {
            for nth in 0..r.live().opened(kind) {
                check.expect(
                    Subject::Resource(kind, nth),
                    !r.live().has(kind, nth),
                    format!("{kind:?} #{nth} still open after the driver exited"),
                );
            }
        }
    }
}

/// A tool ended before the driver exited.
fn closed_before_exit() -> Recording {
    Authored::new()
        .tool_call("call-1", 1)
        .closed(Kind::Tool, 0)
        .sent(Input::Close { drain: false })
        .driver_finished()
        .done()
}

/// A detached tool was still running when the driver exited.
fn open_at_exit() -> Recording {
    Authored::new()
        .tool_call("call-1", 1)
        .sent(Input::Close { drain: false })
        .driver_finished()
        .done()
}

/// A reply runs to its end and the session closes.
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
