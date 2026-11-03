//! The session's spans form a tree the exporter can ship: every span the voice core records (the
//! session tracer's records, observed as `SessionEvent::Telemetry`) ends exactly once, under a
//! parent its declaration allows, within that parent unless declared to outlive it; the session
//! ends its spans itself, so none reaches the close sweep, and a failed span is never recorded as
//! cancelled. A span out of this shape is a turn lost, orphaned, misplaced or never ended on the
//! dashboard. The runtime's own spans are outside this rule.
use std::collections::BTreeMap;

use lk_agents_abi::telemetry::decl::SpanDecl;
use lk_agents_abi::telemetry::record::{
    Key, Record, SpanId, SpanRecord, SpanRef, Status, TraceId, Value,
};
use lk_agents_abi::telemetry::spans::{agent_turn, session, user_turn};
use lk_agents_voice::events::SessionEvent;

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
        guarantee: "Every voice span ends exactly once, under a declared parent and within it unless declared to outlive it; the close sweep ends nothing and no failure is recorded as cancelled.",
        examples: &[spoke_once],
        counterexamples: &[
            wrong_parent,
            parent_ends_first,
            ended_twice,
            swept,
            failure_cancelled,
            never_closed,
        ],
        probe: Some(the_agent_speaks),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let events = r.host_view().events();
        let spans: Vec<&SpanRecord> = events
            .iter()
            .filter_map(|(_, event)| match event {
                SessionEvent::Telemetry(records) => Some(records.iter()),
                _ => None,
            })
            .flatten()
            .filter_map(|record| match record {
                Record::Span(span) => Some(span),
                _ => None,
            })
            .collect();
        let decls = declared();
        let decl = |name: &str| decls.iter().find(|d| d.0 == name);
        let mut by_id: BTreeMap<SpanId, &SpanRecord> = BTreeMap::new();
        for span in &spans {
            if decl(&span.name).is_none() {
                continue;
            }
            check.expect(
                Subject::Session,
                by_id.insert(span.context.span_id, span).is_none(),
                format!("{} {:?} ended twice", span.name, span.context.span_id),
            );
            let cancelled =
                span.attributes.contains(&(Key::Borrowed("lk.cancelled"), Value::Bool(true)));
            check.expect(
                Subject::Session,
                !cancelled,
                format!("{} reached the close sweep", span.name),
            );
            check.expect(
                Subject::Session,
                !(cancelled && matches!(span.status, Status::Error(_))),
                format!("{} failed and was recorded as cancelled", span.name),
            );
        }
        for span in by_id.values() {
            let Some((_, parents, outlives)) = decl(&span.name) else { continue };
            // A parent outside the session's spans is the span the session continues.
            let Some(parent) = span.parent.and_then(|id| by_id.get(&id)) else { continue };
            check.expect(
                Subject::Session,
                parents.contains(&&*parent.name),
                format!("{} under {}, not a declared parent", span.name, parent.name),
            );
            check.expect(
                Subject::Session,
                *outlives || span.end_unix_nanos <= parent.end_unix_nanos,
                format!("{} ended after its parent {}", span.name, parent.name),
            );
        }
        let closed = events.iter().any(|(_, e)| matches!(e, SessionEvent::Close { .. }));
        if closed && r.driver_finished() {
            check.expect(
                Subject::Session,
                spans.iter().any(|s| s.name == session::AgentSession::NAME),
                "the session closed and agent_session never ended".to_owned(),
            );
        }
    }
}

/// The voice core's spans: name, declared parents, may outlive its parent.
fn declared() -> Vec<(&'static str, &'static [&'static str], bool)> {
    fn decl<D: SpanDecl>() -> (&'static str, &'static [&'static str], bool) {
        (D::NAME, D::PARENTS, D::OUTLIVES_PARENT.is_some())
    }
    vec![
        decl::<session::AgentSession>(),
        decl::<session::SessionStart>(),
        decl::<session::SessionClose>(),
        decl::<session::UpdateAgent>(),
        decl::<session::StartAgentActivity>(),
        decl::<session::OnEnter>(),
        decl::<session::DrainAgentActivity>(),
        decl::<session::OnExit>(),
        decl::<session::PauseAgentActivity>(),
        decl::<session::ResumeAgentActivity>(),
        decl::<session::Amd>(),
        decl::<session::KeytermDetection>(),
        decl::<user_turn::UserTurn>(),
        decl::<user_turn::UserSpeaking>(),
        decl::<user_turn::EouWait>(),
        decl::<user_turn::EouDetection>(),
        decl::<user_turn::OnUserTurnCompleted>(),
        decl::<agent_turn::AgentTurn>(),
        decl::<agent_turn::AgentSpeaking>(),
    ]
}

fn span(id: u64, parent: Option<u64>, name: &'static str, end: u64) -> SpanRecord {
    SpanRecord {
        context: SpanRef { trace_id: TraceId(1), span_id: SpanId(id), sampled: true },
        parent: parent.map(SpanId),
        name: Key::Borrowed(name),
        kind: Default::default(),
        start_unix_nanos: 0,
        end_unix_nanos: end,
        attributes: Vec::new(),
        events: Vec::new(),
        links: Vec::new(),
        status: Status::Unset,
        scope: Key::Borrowed("livekit-agents"),
        job: None,
    }
}

fn ended(spans: Vec<SpanRecord>) -> SessionEvent {
    SessionEvent::Telemetry(std::sync::Arc::new(spans.into_iter().map(Record::Span).collect()))
}

fn session_root(end: u64) -> SpanRecord {
    span(1, Some(99), "agent_session", end)
}

/// The session, a turn, the agent speaking inside it, everything ended.
fn spoke_once() -> Recording {
    Authored::new()
        .session(ended(vec![
            span(3, Some(2), "agent_speaking", 5),
            span(2, Some(1), "agent_turn", 6),
        ]))
        .session(ended(vec![session_root(9)]))
        .done()
}

/// A span under a parent its declaration does not allow.
fn wrong_parent() -> Recording {
    Authored::new()
        .session(ended(vec![span(2, Some(1), "agent_speaking", 1), session_root(2)]))
        .done()
}

/// A turn ended before the agent stopped speaking in it.
fn parent_ends_first() -> Recording {
    Authored::new()
        .session(ended(vec![
            span(2, Some(1), "agent_turn", 3),
            span(3, Some(2), "agent_speaking", 4),
        ]))
        .session(ended(vec![session_root(9)]))
        .done()
}

/// One span ended twice.
fn ended_twice() -> Recording {
    Authored::new()
        .session(ended(vec![session_root(1)]))
        .session(ended(vec![session_root(2)]))
        .done()
}

/// A span the session left for the close sweep.
fn swept() -> Recording {
    let mut turn = span(2, Some(1), "user_turn", 1);
    turn.attributes.push((Key::Borrowed("lk.cancelled"), Value::Bool(true)));
    Authored::new().session(ended(vec![turn, session_root(2)])).done()
}

/// A failed hook recorded as cancelled.
fn failure_cancelled() -> Recording {
    let mut hook = span(2, Some(1), "on_user_turn_completed", 1);
    hook.status = Status::Error("boom".into());
    hook.attributes.push((Key::Borrowed("lk.cancelled"), Value::Bool(true)));
    Authored::new().session(ended(vec![hook])).done()
}

/// The session closed and its span never ended.
fn never_closed() -> Recording {
    Authored::new()
        .session(ended(vec![span(2, Some(1), "session_start", 1)]))
        .session(SessionEvent::Close {
            reason: lk_agents_abi::voice::events::CloseReason::UserInitiated,
            error: None,
        })
        .driver_finished()
        .done()
}

/// Prerequisite: the agent actually speaks, a `say()` the runner lets play out, so a turn and
/// its speaking span are recorded.
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
