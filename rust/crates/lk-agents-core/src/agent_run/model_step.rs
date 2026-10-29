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
    /// Calls the response made so far: each is admitted as it arrives.
    calls: usize,
    usage: Option<CompletionUsage>,
}

/// The finished response; its calls were taken as they arrived.
#[derive(Debug)]
pub(super) struct Response {
    pub(super) text: String,
    pub(super) usage: Option<CompletionUsage>,
}

/// What one chunk adds: text to stream, and the calls it made with their index in the response.
#[derive(Debug, Default)]
pub(super) struct Added {
    pub(super) text: Option<String>,
    pub(super) calls: Vec<(usize, FunctionToolCall)>,
}

impl ModelStep {
    pub(super) fn new(request: RequestId, purpose: Purpose, message_id: String) -> Self {
        ModelStep { request, purpose, message_id, text: String::new(), calls: 0, usage: None }
    }

    /// Takes one chunk: the text it adds, to stream, and the calls it makes, to admit, as they
    /// arrive.
    pub(super) fn chunk(&mut self, chunk: LlmChunk) -> Added {
        if chunk.usage.is_some() {
            self.usage = chunk.usage;
        }
        let Some(delta) = chunk.delta else { return Added::default() };
        let mut added = Added::default();
        // A phrasing request offers no tools; a call it makes anyway is not one to run.
        if self.purpose == Purpose::Turn {
            for call in delta.tool_calls {
                added.calls.push((self.calls, call));
                self.calls += 1;
            }
        }
        added.text = delta.content.filter(|t| !t.is_empty());
        if let Some(text) = &added.text {
            self.text.push_str(text);
        }
        added
    }

    pub(super) fn has_text(&self) -> bool {
        !self.text.is_empty()
    }

    pub(super) fn end(self) -> Response {
        Response { text: self.text, usage: self.usage }
    }
}
