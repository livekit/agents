//! A user turn past its limit hands the developer's `on_user_turn_exceeded` over once no reply is
//! on its way, as the reference's `_user_turn_exceeded_task` does: it waits for the reply the end
//! of turn may bring and gives up if the agent starts speaking, one callback at a time. Fired at
//! once, the default callback's reply races the end of turn's (two replies), and an override fires
//! on an agent that is already answering.
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::events::{AgentState, UserTurnExceededEvent};
use lk_agents_voice::agent_activity::hooks::TurnLimitSkip;
use lk_agents_voice::agent_session::{HookId, HookKind};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{HandoffStep, HookEnd, Step, TurnStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{Input, Millis, Ref, TurnMode, UserText},
    record::Recording,
};

pub struct TurnLimitCutsInOnce;

impl Rule for TurnLimitCutsInOnce {
    const CONTRACT: Contract = Contract {
        id: "turn_limit_cuts_in_once",
        guarantee: "A user turn past its limit gets on_user_turn_exceeded, or is skipped, by the close; never while the agent speaks or while the previous callback runs.",
        examples: &[cut_in, skipped_for_the_reply],
        counterexamples: &[never_handed_over, while_speaking, twice_at_once],
        probe: Some(long_manual_turn),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let mut agent = AgentState::Initializing;
        // An exceeded turn not yet handed over nor skipped, and the callback running.
        let (mut seen, mut owed) = (false, false);
        let mut running: Option<HookId> = None;
        for (at, step) in r.journal() {
            match step {
                Step::Emitted(SessionEvent::AgentStateChanged { new, .. }) => agent = *new,
                Step::Emitted(SessionEvent::UserTurnExceeded(_)) => (seen, owed) = (true, true),
                Step::Turn(TurnStep::TurnLimitSkipped(_)) => owed = false,
                Step::Handoff(HandoffStep::HookIssued {
                    hook,
                    kind: HookKind::OnUserTurnExceeded,
                    ..
                }) => {
                    check.expect(
                        Subject::Session,
                        agent != AgentState::Speaking,
                        format!(
                            "on_user_turn_exceeded issued while the agent speaks (seq {})",
                            at.seq
                        ),
                    );
                    check.expect(
                        Subject::Session,
                        running.is_none(),
                        format!(
                            "on_user_turn_exceeded issued while {running:?} runs (seq {})",
                            at.seq
                        ),
                    );
                    owed = false;
                    running = Some(*hook);
                }
                Step::Handoff(HandoffStep::HookFinished {
                    hook,
                    kind: HookKind::OnUserTurnExceeded,
                    ..
                }) if running == Some(*hook) => running = None,
                // A handoff abandons the leaving agent's callback.
                Step::Handoff(HandoffStep::AgentUpdated { .. }) => running = None,
                Step::Emitted(SessionEvent::Close { .. }) if seen => {
                    check.expect(
                        Subject::Session,
                        !owed,
                        "a user turn past its limit was neither handed over nor skipped by the close",
                    );
                }
                _ => {}
            }
        }
    }
}

fn exceeded() -> Step {
    Step::Emitted(SessionEvent::UserTurnExceeded(UserTurnExceededEvent {
        transcript: "one two three four five".into(),
        accumulated_transcript: "one two three four five".into(),
        accumulated_word_count: 5,
        duration: std::time::Duration::from_secs(2),
        created_at: 0.0,
    }))
}

fn issued(hook: u64) -> Step {
    Step::Handoff(HandoffStep::HookIssued {
        hook: HookId(hook),
        kind: HookKind::OnUserTurnExceeded,
        agent: Some("agent".into()),
        speech: None,
        turn: None,
    })
}

fn finished(hook: u64) -> Step {
    Step::Handoff(HandoffStep::HookFinished {
        hook: HookId(hook),
        kind: HookKind::OnUserTurnExceeded,
        how: HookEnd::Returned { edits: 0 },
    })
}

fn state(old: AgentState, new: AgentState) -> Step {
    Step::Emitted(SessionEvent::AgentStateChanged { old, new })
}

fn close() -> Step {
    Step::Emitted(SessionEvent::Close { reason: CloseReason::UserInitiated, error: None })
}

/// No reply on its way: the callback runs, returns, then the session closes.
fn cut_in() -> Recording {
    Authored::new()
        .journal(state(AgentState::Initializing, AgentState::Listening))
        .journal(exceeded())
        .journal(issued(1))
        .journal(finished(1))
        .journal(exceeded())
        .journal(issued(2))
        .journal(close())
        .done()
}

/// The end of turn replied first: the callback is skipped.
fn skipped_for_the_reply() -> Recording {
    Authored::new()
        .journal(exceeded())
        .journal(state(AgentState::Listening, AgentState::Speaking))
        .journal(Step::Turn(TurnStep::TurnLimitSkipped(TurnLimitSkip::AgentSpeaking)))
        .journal(close())
        .done()
}

/// The callback was never handed over.
fn never_handed_over() -> Recording {
    Authored::new().journal(exceeded()).journal(close()).done()
}

/// Handed over while the agent answered the turn.
fn while_speaking() -> Recording {
    Authored::new()
        .journal(state(AgentState::Listening, AgentState::Speaking))
        .journal(exceeded())
        .journal(issued(1))
        .done()
}

/// A second callback while the first runs.
fn twice_at_once() -> Recording {
    Authored::new()
        .journal(exceeded())
        .journal(issued(1))
        .journal(exceeded())
        .journal(issued(2))
        .done()
}

/// Prerequisite: a manual turn (no end of turn replies) whose final passes the word limit, on a
/// session whose host runs hooks, then the close.
fn long_manual_turn() -> Case {
    Case {
        setup: Setup {
            stt: true,
            turn: TurnMode::Manual,
            user_turn_max_words: Some(4),
            hook_bound_ms: Some(2_000),
            ..Setup::default()
        },
        inputs: vec![
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("my name is Theo and I live in Paris".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(100) },
            Input::Close { drain: false },
        ],
        ..Case::default()
    }
}
