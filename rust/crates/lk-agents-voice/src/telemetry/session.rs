//! The session's own span entities: the spans that hold agent activities (`session_start`,
//! `update_agent`), a lifecycle callback's span with the activity it runs in, the AMD detection,
//! the event-loop stalls `agent_session` ends with, and the turn span of each chat item.

use crate::agent_activity::scheduling::SpeechId;
use crate::audio_recognition::TurnId;
use lk_agents_abi::telemetry::decl::{Seconds, SpanDecl, SpanContext};
use lk_agents_abi::telemetry::errors::SpanError;
use lk_agents_abi::telemetry::spans::session::{
    AgentSessionEnd, Amd, DrainAgentActivity, OnEnter, OnExit, StartAgentActivity, UpdateAgent,
};
use lk_agents_core::telemetry::span::{ManualSpan, Reserved, Trace};
use std::collections::BTreeMap;

/// The span an agent's activity runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Under {
    Start,
    Update,
    Close,
}

/// A span that holds agent activities (`session_start`, `update_agent`): it ends once its own
/// work is `done` and no activity under it still runs.
pub(crate) struct Holding<D: SpanDecl> {
    pub(crate) span: ManualSpan<D>,
    pub(crate) done: bool,
}

/// A lifecycle callback's span and the activity span it runs in: `on_enter` in
/// `start_agent_activity`, `on_exit` in `drain_agent_activity`. Its hook record owns it.
pub(crate) enum HookSpan {
    Enter { under: Under, activity: ManualSpan<StartAgentActivity>, hook: ManualSpan<OnEnter> },
    Exit { under: Under, drain: ManualSpan<DrainAgentActivity>, hook: ManualSpan<OnExit> },
}

impl std::fmt::Debug for HookSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("HookSpan").field(&self.under()).finish()
    }
}

impl HookSpan {
    /// The span the activity runs in.
    pub(crate) fn under(&self) -> Under {
        match self {
            HookSpan::Enter { under, .. } | HookSpan::Exit { under, .. } => *under,
        }
    }

    /// The callback's span, which the host makes current while it runs.
    pub(crate) fn context(&self, trace: &Trace) -> Option<SpanContext> {
        match self {
            HookSpan::Enter { hook, .. } => trace.context(hook),
            HookSpan::Exit { hook, .. } => trace.context(hook),
        }
    }

    /// The callback returned, raised, timed out or was abandoned: it ends, then its activity.
    /// Returns the span the activity ran in.
    pub(crate) fn end(self, trace: &mut Trace, failure: Option<&dyn SpanError>) -> Under {
        match self {
            HookSpan::Enter { under, activity, hook } => {
                trace.end(hook, failure.map_or(Ok(()), Err));
                trace.end(activity, ());
                under
            }
            HookSpan::Exit { under, drain, hook } => {
                trace.end(hook, failure.map_or(Ok(()), Err));
                trace.end(drain, ());
                under
            }
        }
    }
}

/// The spans an arriving agent's activity was reserved under when its toolsets were set up: its
/// `update_agent` (none for the first agent's) and its `start_agent_activity`.
#[derive(Debug, Clone)]
pub(crate) struct Arriving {
    pub(crate) agent: String,
    pub(crate) update: Option<Reserved<UpdateAgent>>,
    pub(crate) activity: Reserved<StartAgentActivity>,
}

/// The answering-machine detection the runtime's detector runs, from its attach to its verdict.
pub(crate) struct AmdDetection {
    pub(crate) span: ManualSpan<Amd>,
}

/// The event-loop stalls a session saw, which `agent_session` ends with.
#[derive(Default)]
pub(crate) struct Blocking {
    count: u32,
    total: f64,
    max: f64,
}

impl Blocking {
    /// One stall of `seconds`.
    pub(crate) fn stalled(&mut self, seconds: f64) {
        self.count += 1;
        self.total += seconds;
        self.max = self.max.max(seconds);
    }

    /// `agent_session`'s end.
    pub(crate) fn session_end(&self) -> AgentSessionEnd {
        let stalled = self.count > 0;
        AgentSessionEnd {
            blocking_count: stalled.then_some(self.count),
            blocking_total: stalled.then_some(Seconds(self.total)),
            blocking_max: stalled.then_some(Seconds(self.max)),
            sip_phone_number: None,
        }
    }
}

/// The turn span of each chat item, by item id, for the session report; and of the committed user
/// turns and the speeches that produce items.
#[derive(Default)]
pub(crate) struct ItemTurns {
    items: BTreeMap<String, SpanContext>,
    user_turns: BTreeMap<TurnId, SpanContext>,
    speeches: BTreeMap<SpeechId, SpanContext>,
}

impl ItemTurns {
    /// A committed user turn: its chat items record its span.
    pub(crate) fn user_turn(&mut self, turn: TurnId, span: SpanContext) {
        self.user_turns.insert(turn, span);
    }

    /// A speech's turn opened: its chat items record its span.
    pub(crate) fn speech(&mut self, speech: SpeechId, span: SpanContext) {
        self.speeches.insert(speech, span);
    }

    pub(crate) fn of_user_turn(&self, turn: TurnId) -> Option<SpanContext> {
        self.user_turns.get(&turn).copied()
    }

    pub(crate) fn of_speech(&self, speech: SpeechId) -> Option<SpanContext> {
        self.speeches.get(&speech).copied()
    }

    /// Item `item` was produced in `span`.
    pub(crate) fn item(&mut self, item: &str, span: SpanContext) {
        self.items.insert(item.to_owned(), span);
    }

    /// The turn span of each chat item, by item id.
    pub(crate) fn items(&self) -> &BTreeMap<String, SpanContext> {
        &self.items
    }
}
