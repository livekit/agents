//! `AgentRun` driven directly, event by event, with no executor: each test plays the runtime's
//! part and asserts the effects and the settled result.
//!
//! - `mod.rs`: the driver and builders the tests share.
//! - `flow.rs`: model steps, tools, outputs and questions.
//! - `approval.rs`: the approval gate across invocations.
//! - `coordinated.rs`: a coordinator planning the requests and deciding each step.
//! - `lifecycle.rs`: sibling calls, duplicates, cancellation, deadlines and arrival orders.

mod approval;
mod checkpoints;
mod coordinated;
mod flow;
mod lifecycle;

use std::time::Duration;

use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::{
    ChatContext, ChatItem, ChatMessage, ChoiceDelta, CompletionUsage, FunctionToolCall, LlmChunk,
    ProviderId, Role, ToolSpec, Ts, WallAnchor,
};
use serde_json::{Map, Value, json};

use super::*;
use lk_agents_abi::agent::{AgentDefinition, FunctionToolInfo, OutputSpec, Retries};

pub(super) const LLM: ResourceId = ResourceId(7);

pub(super) fn tool(name: &str, description: &str, schema: Value) -> FunctionToolInfo {
    FunctionToolInfo {
        spec: ToolSpec {
            provider_tool: None,
            name: name.into(),
            description: description.into(),
            json_schema: schema,
            flags: Default::default(),
        },
        raw_schema: false,
        needs_approval: None,
        on_duplicate: DuplicateMode::Allow,
        duplicate_scope: DuplicateScope::Name,
        retries: None,
        remote: None,
    }
}

pub(super) fn string_tool(name: &str, parameter: &str) -> FunctionToolInfo {
    tool(
        name,
        &format!("Use {name}."),
        json!({"type": "object", "properties": {parameter: {"type": "string"}}, "required": [parameter]}),
    )
}

pub(super) fn agent(tools: Vec<FunctionToolInfo>, output: OutputSpec) -> AgentDefinition {
    AgentDefinition {
        name: "scheduler".into(),
        instructions: "Book appointments.".into(),
        dynamic_instructions: false,
        tools,
        output,
        retries: Retries::default(),
    }
}

pub(super) fn text_agent(tools: Vec<FunctionToolInfo>) -> AgentDefinition {
    agent(tools, OutputSpec::Text { validated: false })
}

pub(super) fn user(id: &str, text: &str) -> ChatMessage {
    ChatMessage::text(id, Role::User, text, std::time::UNIX_EPOCH)
}

/// The runtime's side of one invocation: it delivers events at a supplied instant and keeps every
/// effect the run returned.
pub(super) struct Driver {
    pub(super) run: AgentRun,
    pub(super) now: Ts,
    pub(super) effects: Vec<Effect>,
    /// Calls whose body the driver settled.
    settled: Vec<CallId>,
}

impl Driver {
    pub(super) fn new(agent: AgentDefinition, message: &str, history: Vec<ChatItem>) -> Driver {
        Driver::with_input(
            agent,
            RunInput {
                invocation_id: "run_1".into(),
                opening: Opening::Message(user("msg_1", message)),
                history,
                answer_tool: None,
                delegated: false,
                mode: crate::tool_outcome::Mode::Text,
                llm: Some(LlmRef { id: LLM, provider: ProviderId::of_label("fake") }),
                wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
                cleanup_timeout: Duration::from_secs(5),
            },
        )
    }

    pub(super) fn with_input(agent: AgentDefinition, input: RunInput) -> Driver {
        let mut driver = Driver {
            run: AgentRun::new(agent, input).unwrap(),
            now: Ts::ZERO,
            effects: Vec::new(),
            settled: Vec::new(),
        };
        driver.send(Event::Start);
        driver
    }

    /// Delivers one event; the effects it caused.
    pub(super) fn send(&mut self, event: Event) -> Vec<Effect> {
        if let Event::ToolSettled { call, .. } = &event {
            self.settled.push(*call);
        }
        let fx = self.run.handle(event, self.now);
        self.effects.extend(fx.iter().cloned());
        fx
    }

    pub(super) fn advance(&mut self, by: Duration) {
        self.now = self.now.saturating_add(by);
    }

    /// The last model request: its id, history and tools.
    pub(super) fn request(&self) -> (RequestId, ChatContext, Vec<ToolSpec>) {
        self.effects
            .iter()
            .rev()
            .find_map(|e| match e {
                Effect::LlmRequestStart { request, llm, chat_ctx, tools } => {
                    assert_eq!(*llm, LLM);
                    Some((*request, chat_ctx.clone(), tools.clone()))
                }
                _ => None,
            })
            .expect("a model request")
    }

    pub(super) fn requests(&self) -> usize {
        self.effects.iter().filter(|e| matches!(e, Effect::LlmRequestStart { .. })).count()
    }

    /// The model answers the last request with text and calls `(name, arguments)`.
    pub(super) fn reply(&mut self, text: &str, calls: &[(&str, Value)]) -> Vec<Effect> {
        let (request, _, _) = self.request();
        let n = self.requests();
        if !text.is_empty() {
            self.send(Event::LlmChunk { request, chunk: chunk(Some(text), vec![], None) });
        }
        let calls = calls
            .iter()
            .enumerate()
            .map(|(i, (name, arguments))| FunctionToolCall {
                name: (*name).into(),
                arguments: arguments.to_string(),
                call_id: format!("call_{n}_{i}"),
                extra: None,
            })
            .collect();
        let usage =
            CompletionUsage { prompt_tokens: 10, completion_tokens: 3, ..Default::default() };
        self.send(Event::LlmChunk { request, chunk: chunk(None, calls, Some(usage)) });
        self.send(Event::LlmEnded { request })
    }

    /// Every `ToolPrepare` so far: the call and its arguments.
    pub(super) fn prepares(&self) -> Vec<(CallId, Map<String, Value>)> {
        self.effects
            .iter()
            .filter_map(|e| match e {
                Effect::ToolPrepare { call, arguments, .. } => Some((*call, arguments.clone())),
                _ => None,
            })
            .collect()
    }

    pub(super) fn executes(&self) -> Vec<(CallId, ResourceId, lk_agents_abi::FunctionCall)> {
        self.effects
            .iter()
            .filter_map(|e| match e {
                Effect::ToolExecute { call, prepared, function_call, .. } => {
                    Some((*call, *prepared, function_call.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// The host prepares the call with `identity` as its validated arguments.
    pub(super) fn prepared(&mut self, call: CallId, identity: Value) -> Vec<Effect> {
        let prepared = ResourceId(1000 + call.0);
        self.send(Event::ToolPrepared {
            call,
            prepared,
            identity: ArgumentIdentity::Json(identity),
        })
    }

    /// Prepares every pending call with its arguments as sent, then settles each body with
    /// `outcome`.
    pub(super) fn run_tools(&mut self, outcome: impl Fn(&str) -> ToolOutcome) {
        for (call, arguments) in self.prepares() {
            if self.executes().iter().all(|(c, ..)| *c != call) {
                self.prepared(call, Value::Object(arguments));
            }
        }
        for (call, _, function_call) in self.executes() {
            if !self.settled(call) {
                self.send(Event::ToolSettled { call, outcome: outcome(&function_call.name) });
            }
        }
    }

    fn settled(&self, call: CallId) -> bool {
        self.settled.contains(&call)
    }

    pub(super) fn finished(&self) -> Option<&Result<RunResult, RunError>> {
        self.effects.iter().find_map(|e| match e {
            Effect::Finished(r) => Some(r),
            _ => None,
        })
    }

    pub(super) fn result(&self) -> &RunResult {
        match self.finished() {
            Some(Ok(result)) => result,
            other => panic!("expected a result, got {other:?}"),
        }
    }
}

pub(super) fn chunk(
    text: Option<&str>,
    tool_calls: Vec<FunctionToolCall>,
    usage: Option<CompletionUsage>,
) -> LlmChunk {
    LlmChunk {
        id: "chunk".into(),
        delta: Some(ChoiceDelta {
            role: Some(Role::Assistant),
            content: text.map(Into::into),
            tool_calls,
            extra: None,
        }),
        usage,
        flush: false,
        node_text: false,
    }
}

pub(super) fn value(text: &str) -> ToolOutcome {
    ToolOutcome::Value { text: text.into(), reply_required: true, verbatim: false, more: None }
}

/// The history as `(kind, text)` pairs: messages by role, calls by name, outputs by text.
pub(super) fn transcript(items: &[ChatItem]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|item| match item {
            ChatItem::Message(m) => {
                (format!("{:?}", m.role).to_lowercase(), m.raw_text_content().unwrap_or_default())
            }
            ChatItem::FunctionCall(c) => ("call".into(), format!("{}({})", c.name, c.arguments)),
            ChatItem::FunctionCallOutput(o) => ("output".into(), o.output.clone()),
            other => ("other".into(), other.id().to_owned()),
        })
        .collect()
}

pub(super) fn pairs(rows: &[(&str, &str)]) -> Vec<(String, String)> {
    rows.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect()
}
