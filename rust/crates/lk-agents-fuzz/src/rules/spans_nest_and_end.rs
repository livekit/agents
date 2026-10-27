//! The session's trace is a tree the host can materialise: one root, every span named as in the
//! reference and opened under a parent the reference allows (`telemetry::SPAN_PARENTS`, the
//! rows of `tests/trace_schema.py`) that is still open, ended once and not before its children
//! (bar `MAY_OUTLIVE_PARENT`), and nothing left open once the session closed. `agent_speaking`
//! opens right before each `AgentStateChanged` into `speaking` and is closed by the one out of it.
//!
//! The host turns each record into an OTel span and keeps nothing else (`VoiceSession`'s span
//! handlers), so a record out of this shape is a span lost, orphaned, misplaced or never ended
//! on the dashboard.
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::voice::events::{AgentState, SpanEndedWire, SpanStartedWire};
use lk_agents_voice::{
    events::SessionEvent,
    telemetry::{MAY_OUTLIVE_PARENT, SPAN_PARENTS},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, SayInput},
    record::Recording,
};

pub struct SpansNestAndEnd;

impl Rule for SpansNestAndEnd {
    const CONTRACT: Contract = Contract {
        id: "spans_nest_and_end",
        guarantee: "Span records form one tree of reference-named spans under reference parents, each opened under an open parent and ended once after its children, none left open after close; agent_speaking brackets the speaking state.",
        examples: &[spoke_once],
        counterexamples: &[
            orphan,
            unknown_name,
            wrong_parent,
            parent_ends_first,
            never_closed,
            speaking_without_span,
            ended_twice,
        ],
        probe: Some(the_agent_speaks),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let events = r.host_view().events();
        // id -> (name, parent id), for the spans open now.
        let mut open: BTreeMap<&str, (&str, Option<&str>)> = BTreeMap::new();
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut root = false;
        let mut closed = false;
        for (i, (at, event)) in events.iter().enumerate() {
            let previous = i.checked_sub(1).map(|p| events[p].1);
            match event {
                SessionEvent::SpanStarted(s) => {
                    let allowed = SPAN_PARENTS.iter().find(|(n, _)| *n == s.name).map(|(_, p)| *p);
                    check.expect(
                        Subject::Session,
                        allowed.is_some(),
                        format!("span {:?} at seq {} is no reference span", s.name, at.seq),
                    );
                    check.expect(
                        Subject::Session,
                        seen.insert(s.span_id.as_str()),
                        format!("span id {} reused at seq {}", s.span_id, at.seq),
                    );
                    let parent = s.parent_id.as_deref();
                    let parent_name = match parent {
                        None => (!std::mem::replace(&mut root, true)).then_some(None),
                        Some(p) => open.get(p).map(|(name, _)| Some(*name)),
                    };
                    check.expect(
                        Subject::Session,
                        parent_name.is_some(),
                        format!("{} at seq {} under {parent:?}, not an open span", s.name, at.seq),
                    );
                    if let (Some(allowed), Some(parent_name)) = (allowed, parent_name) {
                        check.expect(
                            Subject::Session,
                            allowed.contains(&parent_name),
                            format!("{} at seq {} under {parent_name:?}", s.name, at.seq),
                        );
                    }
                    open.insert(s.span_id.as_str(), (s.name.as_str(), parent));
                }
                SessionEvent::SpanEnded(s) => {
                    let ended = open.remove(s.span_id.as_str());
                    check.expect(
                        Subject::Session,
                        ended.is_some(),
                        format!("span {} ended at seq {} while not open", s.span_id, at.seq),
                    );
                    let Some((name, _)) = ended else { continue };
                    let orphaned: Vec<&str> = open
                        .values()
                        .filter(|(child, parent)| {
                            *parent == Some(s.span_id.as_str())
                                && !MAY_OUTLIVE_PARENT.contains(&(*child, name))
                        })
                        .map(|(child, _)| *child)
                        .collect();
                    check.expect(
                        Subject::Session,
                        orphaned.is_empty(),
                        format!("{name} ended at seq {} before its children {orphaned:?}", at.seq),
                    );
                }
                SessionEvent::AgentStateChanged { old, new } if old != new => {
                    if *new == AgentState::Speaking {
                        let spanned = matches!(previous,
                            Some(SessionEvent::SpanStarted(s)) if s.name == "agent_speaking");
                        check.expect(
                            Subject::Session,
                            spanned,
                            format!(
                                "{old:?} -> speaking at seq {} with no span opened for it",
                                at.seq
                            ),
                        );
                    }
                    if *old == AgentState::Speaking {
                        check.expect(
                            Subject::Session,
                            !open.values().any(|(name, _)| *name == "agent_speaking"),
                            format!(
                                "speaking -> {new:?} at seq {} with agent_speaking open",
                                at.seq
                            ),
                        );
                    }
                }
                SessionEvent::Close { .. } => closed = true,
                _ => {}
            }
        }
        if closed && r.driver_finished() {
            check.expect(
                Subject::Session,
                open.is_empty(),
                format!("spans left open after the close: {:?}", open.values().collect::<Vec<_>>()),
            );
        }
    }
}

fn start(id: &str, parent: Option<&str>, name: &str) -> SessionEvent {
    SessionEvent::SpanStarted(Box::new(SpanStartedWire {
        span_id: id.to_owned(),
        parent_id: parent.map(str::to_owned),
        name: name.to_owned(),
        start_time: 0.0,
        attributes: Default::default(),
    }))
}

fn end(id: &str) -> SessionEvent {
    SessionEvent::SpanEnded(Box::new(SpanEndedWire {
        span_id: id.to_owned(),
        end_time: 0.0,
        attributes: Default::default(),
        error: None,
        events: Vec::new(),
        metrics: Vec::new(),
    }))
}

fn state(old: AgentState, new: AgentState) -> SessionEvent {
    SessionEvent::AgentStateChanged { old, new }
}

/// The root, a turn, the agent speaking inside it, everything ended.
fn spoke_once() -> Recording {
    Authored::new()
        .session(start("agent_session", None, "agent_session"))
        .session(state(AgentState::Listening, AgentState::Thinking))
        .session(start("agent_turn_2", Some("agent_session"), "agent_turn"))
        .session(start("agent_speaking_3", Some("agent_turn_2"), "agent_speaking"))
        .session(state(AgentState::Thinking, AgentState::Speaking))
        .session(end("agent_speaking_3"))
        .session(state(AgentState::Speaking, AgentState::Listening))
        .session(end("agent_turn_2"))
        .session(end("agent_session"))
        .done()
}

/// A span under a parent the host never opened: it would hang off nothing.
fn orphan() -> Recording {
    Authored::new()
        .session(start("agent_session", None, "agent_session"))
        .session(start("agent_turn_2", Some("agent_turn_9"), "agent_turn"))
        .done()
}

/// A span under a parent the reference never gives it.
fn wrong_parent() -> Recording {
    Authored::new()
        .session(start("agent_session", None, "agent_session"))
        .session(start("llm_node_2", Some("agent_session"), "llm_node"))
        .done()
}

/// A turn ended while its tool span was still open.
fn parent_ends_first() -> Recording {
    Authored::new()
        .session(start("agent_session", None, "agent_session"))
        .session(start("agent_turn_2", Some("agent_session"), "agent_turn"))
        .session(start("function_tool_3", Some("agent_turn_2"), "function_tool"))
        .session(end("agent_turn_2"))
        .done()
}

/// A span the reference never emits.
fn unknown_name() -> Recording {
    Authored::new().session(start("agent_session", None, "agent_sesion")).done()
}

/// The session closed with a span still open.
fn never_closed() -> Recording {
    Authored::new()
        .session(start("agent_session", None, "agent_session"))
        .session(SessionEvent::Close {
            reason: lk_agents_abi::voice::events::CloseReason::UserInitiated,
            error: None,
        })
        .driver_finished()
        .done()
}

/// The agent spoke and the host was never told to open the span.
fn speaking_without_span() -> Recording {
    Authored::new().session(state(AgentState::Thinking, AgentState::Speaking)).done()
}

/// One span ended twice.
fn ended_twice() -> Recording {
    Authored::new()
        .session(start("agent_session", None, "agent_session"))
        .session(end("agent_session"))
        .session(end("agent_session"))
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
        ..Case::default()
    }
}
