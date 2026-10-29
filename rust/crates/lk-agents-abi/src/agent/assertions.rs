//! Run event assertions (testing.md "Programmatic tests"): the cursor `result.expect` walks. The
//! matching, the cursor and the failure context are `lk-agents-core`'s.

use serde_json::Value;

use crate::ChatItem;

/// One event of a tested invocation, as the assertions read it.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone)]
pub struct ObservedEvent {
    pub item: ChatItem,
    /// For a handoff, the qualified names of the new agent's class and its bases.
    pub new_agent_types: Vec<String>,
}

/// What an assertion expects of an event: its kind and the fields it names.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone, PartialEq)]
pub enum EventExpectation {
    /// `role` is the `ChatRole` literal.
    Message {
        role: Option<String>,
    },
    /// `arguments` holds the keys the call must carry with equal values; others may be present.
    FunctionCall {
        name: Option<String>,
        arguments: Option<Value>,
    },
    FunctionCallOutput {
        output: Option<String>,
        is_error: Option<bool>,
    },
    AgentHandoff {
        new_agent_type: Option<String>,
    },
}

/// A failed expectation: what was expected and the events around it.
#[crate::host(error(kind = "AssertionError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct AssertionFailure(pub String);

/// The observed events of one invocation and the position `next_event` reads from. Each call
/// returns the index of the event it matched.
#[crate::host(class(name = "RunEvents"))]
pub trait RunEvents: Send + Sized + 'static {
    #[host(new)]
    fn new(events: Vec<ObservedEvent>) -> Self;

    /// The event at `index` (from the end when negative), meeting `expected` when given.
    #[host(call)]
    fn at(&self, index: i64, expected: Option<EventExpectation>)
    -> Result<usize, AssertionFailure>;

    /// Advances to the next event of `expected`'s kind, or the next event when `None`.
    #[host(call)]
    fn next_event(&mut self, expected: Option<EventExpectation>)
    -> Result<usize, AssertionFailure>;

    /// Advances past the next event only if it meets `expected`.
    #[host(call)]
    fn skip_next_event_if(&mut self, expected: EventExpectation) -> Option<usize>;

    #[host(call)]
    fn skip_next(&mut self, count: usize) -> Result<(), AssertionFailure>;

    #[host(call)]
    fn no_more_events(&self) -> Result<(), AssertionFailure>;

    /// The first event of `start..stop` meeting `expected`.
    #[host(call)]
    fn contains(
        &self,
        start: usize,
        stop: usize,
        expected: EventExpectation,
    ) -> Result<usize, AssertionFailure>;

    /// The message of a failure about the event at `index`, or the current one: `message` and
    /// the events with that one marked.
    #[host(call)]
    fn failure(&self, message: String, index: Option<usize>) -> String;
}
