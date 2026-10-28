//! One model request's lifecycle: its correlation, the text and calls it streams, its usage.

use lk_agents_abi::{CompletionUsage, FunctionToolCall, LlmChunk};

use super::result::RequestId;

/// What the request is for.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Purpose {
    /// The model's next step.
    Turn,
    /// The facing agent words a question; `ask` is what it asks.
    Phrase(super::Ask),
}

#[derive(Debug)]
pub(super) struct ModelStep {
    pub(super) request: RequestId,
    pub(super) purpose: Purpose,
    /// The id of the assistant message its text becomes.
    pub(super) message_id: String,
    text: String,
    calls: Vec<FunctionToolCall>,
    usage: Option<CompletionUsage>,
}

/// The finished response.
#[derive(Debug)]
pub(super) struct Response {
    pub(super) text: String,
    pub(super) calls: Vec<FunctionToolCall>,
    pub(super) usage: Option<CompletionUsage>,
}

impl ModelStep {
    pub(super) fn new(request: RequestId, purpose: Purpose, message_id: String) -> Self {
        ModelStep { request, purpose, message_id, text: String::new(), calls: Vec::new(), usage: None }
    }

    /// Takes one chunk; the text it adds, to stream as it arrives.
    pub(super) fn chunk(&mut self, chunk: LlmChunk) -> Option<String> {
        if chunk.usage.is_some() {
            self.usage = chunk.usage;
        }
        let delta = chunk.delta?;
        // A phrasing request offers no tools; a call it makes anyway is not one to run.
        if self.purpose == Purpose::Turn {
            self.calls.extend(delta.tool_calls);
        }
        let text = delta.content.filter(|t| !t.is_empty())?;
        self.text.push_str(&text);
        Some(text)
    }

    pub(super) fn has_text(&self) -> bool {
        !self.text.is_empty()
    }

    pub(super) fn end(self) -> Response {
        Response { text: self.text, calls: self.calls, usage: self.usage }
    }
}
