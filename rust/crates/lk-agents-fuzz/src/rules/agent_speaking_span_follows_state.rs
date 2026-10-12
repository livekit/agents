//! The host's `agent_speaking` span opens on every edge into `speaking` and closes on every edge
//! out of it, and on no other.
//!
//! The span is decided by the labels the driver runs (`agent_session.AgentSession
//! ._update_agent_state`), so a host opens and closes its OTel span on these two events and keeps
//! no label state of its own. Each is emitted immediately before the `AgentStateChanged` it belongs
//! to; read in the order the host received them, a span edge with no matching state edge next, or
//! a state edge into or out of `speaking` with no span edge right before it, is a span the host
//! would hold open forever or never open.
use lk_agents_abi::voice::{
    events::AgentState,
    session_effects::{Span, SpanEnded, SpanStarted},
};
use lk_agents_voice::events::SessionEvent;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, SayInput},
    record::Recording,
};

pub struct AgentSpeakingSpanFollowsState;

impl Rule for AgentSpeakingSpanFollowsState {
    const CONTRACT: Contract = Contract {
        id: "agent_speaking_span_follows_state",
        guarantee: "An agent_speaking span edge is emitted right before, and only before, each AgentStateChanged into or out of speaking.",
        examples: &[spoke_once],
        counterexamples: &[never_opened, never_closed, opened_twice],
        probe: Some(the_agent_speaks),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let events = host.events();
        let mut open = false;
        for (i, (at, event)) in events.iter().enumerate() {
            let previous = i.checked_sub(1).map(|p| events[p].1);
            let next = events.get(i + 1).map(|(_, e)| *e);
            match event {
                // An edge into speaking has its span opened right before it.
                SessionEvent::AgentStateChanged { old, new } if *new == AgentState::Speaking => {
                    let spanned = matches!(
                        previous,
                        Some(SessionEvent::SpanStarted(SpanStarted { span: Span::AgentSpeaking, .. }))
                    );
                    check.expect(
                        Subject::Session,
                        spanned,
                        format!("{old:?} -> speaking at seq {} with no span opened for it", at.seq),
                    );
                }
                // An edge out of speaking has its span closed right before it.
                SessionEvent::AgentStateChanged { old: AgentState::Speaking, new } => {
                    let spanned = matches!(
                        previous,
                        Some(SessionEvent::SpanEnded(SpanEnded { span: Span::AgentSpeaking, .. }))
                    );
                    check.expect(
                        Subject::Session,
                        spanned,
                        format!("speaking -> {new:?} at seq {} with no span closed for it", at.seq),
                    );
                }
                // A span opens only for the speaking edge that follows it, and never twice.
                SessionEvent::SpanStarted(SpanStarted { span: Span::AgentSpeaking, .. }) => {
                    let belongs = matches!(
                        next,
                        Some(SessionEvent::AgentStateChanged { new: AgentState::Speaking, .. })
                    );
                    check.expect(
                        Subject::Session,
                        belongs && !open,
                        format!(
                            "agent_speaking opened at seq {} (open already: {open}) off a speaking edge",
                            at.seq
                        ),
                    );
                    open = true;
                }
                // A span closes only for the edge out of speaking that follows it, and only if open.
                SessionEvent::SpanEnded(SpanEnded { span: Span::AgentSpeaking, .. }) => {
                    let belongs = matches!(
                        next,
                        Some(SessionEvent::AgentStateChanged { old: AgentState::Speaking, .. })
                    );
                    check.expect(
                        Subject::Session,
                        belongs && open,
                        format!(
                            "agent_speaking closed at seq {} (open: {open}) off a speaking edge",
                            at.seq
                        ),
                    );
                    open = false;
                }
                _ => {}
            }
        }
    }
}

fn state(old: AgentState, new: AgentState) -> SessionEvent {
    SessionEvent::AgentStateChanged { old, new }
}

fn opened() -> SessionEvent {
    SessionEvent::SpanStarted(SpanStarted { span: Span::AgentSpeaking, start_time: None })
}

fn closed() -> SessionEvent {
    SessionEvent::SpanEnded(SpanEnded { span: Span::AgentSpeaking, end_time: None })
}

/// Thinking, speaking with its span, back to listening with the span closed.
fn spoke_once() -> Recording {
    Authored::new()
        .session(state(AgentState::Listening, AgentState::Thinking))
        .session(opened())
        .session(state(AgentState::Thinking, AgentState::Speaking))
        .session(closed())
        .session(state(AgentState::Speaking, AgentState::Listening))
        .done()
}

/// The agent spoke and the host was never told to open the span.
fn never_opened() -> Recording {
    Authored::new().session(state(AgentState::Thinking, AgentState::Speaking)).done()
}

/// The agent stopped speaking and the span stayed open.
fn never_closed() -> Recording {
    Authored::new()
        .session(opened())
        .session(state(AgentState::Thinking, AgentState::Speaking))
        .session(state(AgentState::Speaking, AgentState::Listening))
        .done()
}

/// A second open with no close between, not on a speaking edge.
fn opened_twice() -> Recording {
    Authored::new()
        .session(opened())
        .session(state(AgentState::Thinking, AgentState::Speaking))
        .session(opened())
        .done()
}

/// Prerequisite: the agent actually speaks -- a `say()` the runner lets play out -- so the label
/// goes into `speaking` and back.
fn the_agent_speaks() -> Case {
    Case {
        setup: Setup::default(),
        inputs: vec![
            Input::Say {
                text: AgentText("Hello there.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(200) },
        ],
        schedule: 0,
    }
}
