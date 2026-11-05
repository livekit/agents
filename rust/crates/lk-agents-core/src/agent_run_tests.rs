//! `AgentRun` driven directly, event by event, with no executor: each test plays the runtime's
//! part and asserts the effects and the settled result.
//!
//! The shared driver, model steps, tools, outputs and managed checkpoints live here; questions,
//! coordinated (voice) runs and cancellation have sibling files.

#[path = "agent_run_cancellation_tests.rs"]
mod cancellation;
#[path = "agent_run_questions_tests.rs"]
mod questions;
#[path = "agent_run_voice_tests.rs"]
mod voice;

use lk_agents_abi::agent::ToolTemplate;
use std::time::Duration;

use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{
    ChatContext, ChatItem, ChatMessage, ChoiceDelta, FunctionToolCall, LlmChunk, ProviderId, Role,
    ToolSpec, Ts, WallAnchor,
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

/// A text-mode invocation opening with `opening` on `history`.
pub(super) fn input(opening: Opening, history: Vec<ChatItem>) -> RunInput {
    RunInput {
        invocation_id: "run_1".into(),
        opening,
        history,
        answer_tool: None,
        delegated: false,
        mode: crate::tool_outcome::Mode::Text,
        llm: Some(LlmRef { id: LLM, provider: ProviderId::of_label("fake") }),
        wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
        cleanup_timeout: Duration::from_secs(5),
    }
}

/// The runtime's side of one invocation: it delivers events at a supplied instant and keeps every
/// effect the run returned.
pub(super) struct Driver {
    pub(super) run: AgentRun,
    pub(super) now: Ts,
    pub(super) effects: Vec<AgentRunEffect>,
    /// Calls whose body the driver settled.
    settled: Vec<CallId>,
}

impl Driver {
    pub(super) fn new(agent: AgentDefinition, message: &str, history: Vec<ChatItem>) -> Driver {
        Driver::with_input(
            agent,
            input(Opening::Message(Box::new(user("msg_1", message))), history),
        )
    }

    pub(super) fn with_input(agent: AgentDefinition, input: RunInput) -> Driver {
        Driver::start(AgentRun::new(agent, input).unwrap())
    }

    pub(super) fn start(run: AgentRun) -> Driver {
        let mut driver = Driver { run, now: Ts::ZERO, effects: Vec::new(), settled: Vec::new() };
        driver.send(AgentRunEvent::Start);
        driver
    }

    /// Delivers one event; the effects it caused.
    pub(super) fn send(&mut self, event: AgentRunEvent) -> Vec<AgentRunEffect> {
        if let AgentRunEvent::ToolSettled { call, .. } = &event {
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
                AgentRunEffect::LlmRequestStart { request, llm, chat_ctx, tools } => {
                    assert_eq!(*llm, LLM);
                    Some((*request, chat_ctx.clone(), tools.clone()))
                }
                _ => None,
            })
            .expect("a model request")
    }

    /// The latest output candidate put to validation.
    pub(super) fn validation(&self) -> ValidationId {
        (self.effects.iter().rev())
            .find_map(|e| match e {
                AgentRunEffect::ValidateOutput { validation, .. } => Some(*validation),
                _ => None,
            })
            .expect("an output candidate")
    }

    pub(super) fn requests(&self) -> usize {
        self.effects.iter().filter(|e| matches!(e, AgentRunEffect::LlmRequestStart { .. })).count()
    }

    /// The model answers the last request with text and calls `(name, arguments)`.
    pub(super) fn reply(&mut self, text: &str, calls: &[(&str, Value)]) -> Vec<AgentRunEffect> {
        let (request, _, _) = self.request();
        let n = self.requests();
        if !text.is_empty() {
            self.send(AgentRunEvent::LlmChunk { request, chunk: chunk(Some(text), vec![], None) });
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
        let usage = TokenUsage { input_tokens: 10, output_tokens: 3, ..Default::default() };
        self.send(AgentRunEvent::LlmChunk { request, chunk: chunk(None, calls, Some(usage)) });
        self.send(AgentRunEvent::LlmEnded { request })
    }

    /// Every `ToolPrepare` so far: the call and its arguments.
    pub(super) fn prepares(&self) -> Vec<(CallId, Map<String, Value>)> {
        self.effects
            .iter()
            .filter_map(|e| match e {
                AgentRunEffect::ToolPrepare { call, arguments, .. } => {
                    Some((*call, arguments.clone()))
                }
                _ => None,
            })
            .collect()
    }

    pub(super) fn executes(&self) -> Vec<(CallId, ResourceId, lk_agents_abi::FunctionCall)> {
        self.effects
            .iter()
            .filter_map(|e| match e {
                AgentRunEffect::ToolExecute { call, prepared, function_call, .. } => {
                    Some((*call, *prepared, function_call.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// The host prepares the call with `identity` as its validated arguments.
    pub(super) fn prepared(&mut self, call: CallId, identity: Value) -> Vec<AgentRunEffect> {
        let prepared = ResourceId(1000 + call.0);
        self.send(AgentRunEvent::ToolPrepared {
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
                self.send(AgentRunEvent::ToolSettled {
                    call,
                    outcome: outcome(&function_call.name),
                });
            }
        }
    }

    fn settled(&self, call: CallId) -> bool {
        self.settled.contains(&call)
    }

    pub(super) fn finished(&self) -> Option<&Result<RunResult, RunError>> {
        self.effects.iter().find_map(|e| match e {
            AgentRunEffect::Finished { result: r, .. } => Some(r.as_ref()),
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
    usage: Option<TokenUsage>,
) -> LlmChunk {
    LlmChunk {
        id: "chunk".into(),
        delta: Some(ChoiceDelta {
            role: Some(Role::Assistant),
            content: text.map(Into::into),
            tool_calls,
            extra: None,
        }),
        usage: usage.map(Box::new),
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

// Catches: a tool result not reaching the next request, the incoming message missing from
// `new_messages`, instructions leaking into `all_messages`, or usage counted per run instead of
// per request.
#[test]
fn a_tool_runs_and_the_model_continues_from_its_output() {
    let mut d =
        Driver::new(text_agent(vec![string_tool("find_slot", "day")]), "Book me Monday", vec![]);
    let (_, ctx, tools) = d.request();
    assert_eq!(
        transcript(&ctx.items),
        pairs(&[("system", "Book appointments."), ("user", "Book me Monday")])
    );
    assert_eq!(tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["find_slot"]);

    d.reply("Let me look.", &[("find_slot", json!({"day": "Monday"}))]);
    d.run_tools(|_| value("9:00 is free"));
    let (_, ctx, _) = d.request();
    assert_eq!(
        transcript(&ctx.items)[2..],
        pairs(&[
            ("assistant", "Let me look."),
            ("call", r#"find_slot({"day":"Monday"})"#),
            ("output", "9:00 is free"),
        ])
    );

    d.reply("Monday 9:00 is free.", &[]);
    let result = d.result();
    assert_eq!(
        result.outcome,
        RunOutcome::Completed { output: Output::Text("Monday 9:00 is free.".into()) }
    );
    assert_eq!(transcript(&result.new_messages)[0], ("user".into(), "Book me Monday".into()));
    assert_eq!(result.new_messages.len(), 5);
    assert_eq!(result.all_messages, result.new_messages);
    let usage = &result.usage;
    assert_eq!(
        (usage.requests, usage.input_tokens(), usage.output_tokens(), usage.tool_calls),
        (2, 20, 6, 1)
    );
    assert!(usage.settled && usage.complete, "{usage:?}");
}

// Catches: text withheld until the response ends (no incremental output), or a step's text
// stream not closed before its tools run.
#[test]
fn text_streams_as_it_arrives_before_the_model_ends() {
    let mut d = Driver::new(text_agent(vec![]), "hi", vec![]);
    let (request, _, _) = d.request();
    let fx = d.send(AgentRunEvent::LlmChunk { request, chunk: chunk(Some("Hel"), vec![], None) });
    assert!(
        matches!(&fx[..], [AgentRunEffect::Emit(StreamEvent::TextDelta { delta, .. })] if delta == "Hel")
    );
    assert!(d.finished().is_none());
    d.send(AgentRunEvent::LlmChunk { request, chunk: chunk(Some("lo"), vec![], None) });
    let fx = d.send(AgentRunEvent::LlmEnded { request });
    assert!(matches!(
        &fx[0],
        AgentRunEffect::Emit(StreamEvent::TextEnd { interrupted: false, .. })
    ));
    assert_eq!(d.result().outcome, RunOutcome::Completed { output: Output::Text("Hello".into()) });
}

// Catches: identical inputs producing different ids, effects or history (ambient time, entropy
// or iteration order in a decision).
#[test]
fn identical_inputs_replay_identically() {
    let play = || {
        let mut d = Driver::new(
            text_agent(vec![string_tool("a", "x"), string_tool("b", "y")]),
            "go",
            vec![],
        );
        d.reply("", &[("b", json!({"y": "2"})), ("a", json!({"x": "1"}))]);
        d.run_tools(value);
        d.reply("done", &[]);
        d.effects
    };
    assert_eq!(play(), play());
}

// Catches: a direct run ignoring `ToolResult(reply_required=False)` and asking the
// model again.
#[test]
fn a_tool_that_asks_for_no_reply_ends_the_turn() {
    let mut d =
        Driver::new(text_agent(vec![string_tool("send_dtmf", "digits")]), "press 1", vec![]);
    d.reply("", &[("send_dtmf", json!({"digits": "1"}))]);
    d.run_tools(|_| ToolOutcome::Value {
        text: "sent".into(),
        reply_required: false,
        verbatim: false,
        more: None,
    });
    assert_eq!(d.requests(), 1);
    assert_eq!(d.result().outcome, RunOutcome::NoOutput);
}

// Catches: a tool returning `None` ending the turn; `None` is a value the model reads and goes
// on from (tools.md), unlike an explicit no-reply result.
#[test]
fn none_return_continues() {
    let mut d = Driver::new(text_agent(vec![string_tool("log_visit", "note")]), "log it", vec![]);
    d.reply("", &[("log_visit", json!({"note": "seen"}))]);
    d.run_tools(|_| ToolOutcome::Nothing);
    assert_eq!(d.requests(), 2);
    d.reply("Logged.", &[]);
    assert_eq!(
        d.result().outcome,
        RunOutcome::Completed { output: Output::Text("Logged.".into()) }
    );
}

// Catches: `StopResponse` continuing the loop, or a verbatim result losing its delivered text or
// its exact words on the result.
#[test]
fn stop_response_and_verbatim_results_end_or_lead_the_turn() {
    let mut d = Driver::new(text_agent(vec![string_tool("stop", "x")]), "stop", vec![]);
    d.reply("", &[("stop", json!({"x": "1"}))]);
    d.run_tools(|_| ToolOutcome::StopResponse);
    assert_eq!((d.requests(), &d.result().outcome), (1, &RunOutcome::NoOutput));

    let mut d = Driver::new(text_agent(vec![string_tool("code", "x")]), "my code?", vec![]);
    d.reply("", &[("code", json!({"x": "1"}))]);
    d.run_tools(|_| ToolOutcome::Value {
        text: "Your code is 1234.".into(),
        reply_required: true,
        verbatim: true,
        more: None,
    });
    let (_, ctx, _) = d.request();
    assert_eq!(
        ctx.items.last().map(|i| transcript(std::slice::from_ref(i))[0].1.clone()).unwrap(),
        "delivered to the person as written: Your code is 1234."
    );
    d.reply("Anything else?", &[]);
    let result = d.result();
    assert_eq!(
        result.outcome,
        RunOutcome::Completed {
            output: Output::Text("Your code is 1234.\n\nAnything else?".into())
        }
    );
    assert_eq!(result.verbatim.as_deref(), Some("Your code is 1234."));
}

// Catches: `ModelRetry` treated as an internal error, or retried without a budget.
#[test]
fn model_retry_is_feedback_within_the_tool_retry_budget() {
    let mut d = Driver::new(text_agent(vec![string_tool("book", "day")]), "book", vec![]);
    d.reply("", &[("book", json!({"day": "someday"}))]);
    d.run_tools(|_| ToolOutcome::ModelRetry("use a weekday".into()));
    let (_, ctx, _) = d.request();
    let ChatItem::FunctionCallOutput(output) = ctx.items.last().unwrap() else { panic!() };
    assert_eq!((output.output.as_str(), output.is_error), ("use a weekday", true));

    d.reply("", &[("book", json!({"day": "never"}))]);
    d.run_tools(|_| ToolOutcome::ModelRetry("use a weekday".into()));
    assert_eq!(
        d.finished(),
        Some(&Err(RunError::ToolRetriesExceeded { tool: "book".into(), retries: 1 }))
    );
}

// Catches: invalid arguments running the body, or an unknown tool failing the run instead of
// telling the model.
#[test]
fn invalid_arguments_and_unknown_tools_are_feedback_and_never_run_a_body() {
    let mut d = Driver::new(text_agent(vec![string_tool("book", "day")]), "book", vec![]);
    d.reply("", &[("book", json!({"day": 5})), ("cancel", json!({}))]);
    let (call, _) = d.prepares()[0].clone();
    d.send(AgentRunEvent::ToolPrepareFailed {
        call,
        failure: PrepareFailure::Refused("day: Input should be a valid string".into()),
    });
    assert!(d.executes().is_empty());
    let (_, ctx, _) = d.request();
    let outputs: Vec<_> =
        transcript(&ctx.items).into_iter().filter(|(k, _)| k == "output").collect();
    assert_eq!(
        outputs,
        pairs(&[
            ("output", "day: Input should be a valid string"),
            ("output", "unknown tool `cancel`")
        ])
    );
}

// Catches: a non-object output schema offered unwrapped, its value not unwrapped, an accepted
// candidate not settling the run, or a rejected one not fed back.
#[test]
fn structured_output_is_accepted_through_final_result() {
    let mut d = Driver::new(
        agent(vec![], OutputSpec::Structured { schema: json!({"type": "integer"}) }),
        "count",
        vec![],
    );
    let (_, _, tools) = d.request();
    assert_eq!(tools[0].name, "final_result");
    assert_eq!(tools[0].json_schema["properties"]["result"], json!({"type": "integer"}));

    d.reply("", &[("final_result", json!({"result": 3}))]);
    let validation = d.validation();
    assert!(d.effects.contains(&AgentRunEffect::ValidateOutput {
        validation,
        candidate: OutputCandidate::Value(json!(3))
    }));
    d.send(AgentRunEvent::OutputValidated {
        validation,
        verdict: OutputVerdict::Retry("must be even".into()),
    });
    let (_, ctx, _) = d.request();
    assert_eq!(transcript(&ctx.items).last().unwrap().1, "invalid final_result: must be even");

    d.reply("", &[("final_result", json!({"result": 4}))]);
    let validation = d.validation();
    d.send(AgentRunEvent::OutputValidated {
        validation,
        verdict: OutputVerdict::Accepted { text: None },
    });
    assert_eq!(d.result().outcome, RunOutcome::Completed { output: Output::Validated(validation) });
}

// Catches: an output validator's replacement value discarded.
#[test]
fn a_validator_replacement_is_the_accepted_output() {
    let mut d = Driver::new(agent(vec![], OutputSpec::Text { validated: true }), "hi", vec![]);
    d.reply("hello", &[]);
    d.send(AgentRunEvent::OutputValidated {
        validation: d.validation(),
        verdict: OutputVerdict::Accepted { text: Some("Hello!".into()) },
    });
    assert_eq!(d.result().outcome, RunOutcome::Completed { output: Output::Text("Hello!".into()) });
}

// Catches: conversational text where a typed output is expected settling as output, or a
// question being worded twice.
#[test]
fn text_where_a_typed_output_is_expected_asks_the_person() {
    let mut d = Driver::new(
        agent(vec![], OutputSpec::Structured { schema: json!({"type": "object"}) }),
        "book",
        vec![],
    );
    d.reply("Which day works for you?", &[]);
    assert_eq!(d.requests(), 1);
    assert_eq!(
        d.result().outcome,
        RunOutcome::InputRequired {
            question: "Which day works for you?".into(),
            verbatim: false,
            answer_tool: None
        }
    );
}

// Catches: the facing agent passing a tool's question on unworded, a delegated run wording it,
// or the worded question missing from the history.
#[test]
fn a_tool_question_is_worded_by_the_facing_agent_only() {
    let ask = |_: &str| ToolOutcome::InputRequired {
        question: "photo of the damage".into(),
        verbatim: false,
        tool: None,
    };
    let mut d = Driver::new(text_agent(vec![string_tool("photo", "x")]), "claim", vec![]);
    d.reply("", &[("photo", json!({"x": "1"}))]);
    d.run_tools(ask);
    let (_, ctx, tools) = d.request();
    assert!(tools.is_empty());
    assert!(
        transcript(&ctx.items).last().unwrap().1.ends_with("nothing else: photo of the damage")
    );
    d.reply("Could you send a photo of the damage?", &[]);
    let result = d.result();
    assert_eq!(
        result.outcome,
        RunOutcome::InputRequired {
            question: "Could you send a photo of the damage?".into(),
            verbatim: false,
            answer_tool: None
        }
    );
    assert_eq!(
        transcript(&result.all_messages).last().unwrap().1,
        "Could you send a photo of the damage?"
    );

    let mut input = Driver::new(text_agent(vec![]), "x", vec![]).run.input.clone();
    input.delegated = true;
    input.opening = Opening::Message(Box::new(user("m", "claim")));
    let mut d = Driver::with_input(text_agent(vec![string_tool("photo", "x")]), input);
    d.reply("", &[("photo", json!({"x": "1"}))]);
    d.run_tools(ask);
    assert_eq!(d.requests(), 1);
    assert!(
        matches!(&d.result().outcome, RunOutcome::InputRequired { question, .. } if question == "photo of the damage")
    );
}

// Catches: `InputRequired(tool=)`'s answer going to the model instead of the named tool, or the
// waiting placeholder left in the history once answered.
#[test]
fn an_answer_for_a_named_tool_is_handed_to_it_without_a_model_turn() {
    let tools =
        || vec![string_tool("ask_expert", "question"), string_tool("reply_expert", "answer")];
    let mut d = Driver::new(text_agent(tools()), "is it covered?", vec![]);
    d.reply("", &[("ask_expert", json!({"question": "covered?"}))]);
    d.run_tools(|_| ToolOutcome::InputRequired {
        question: "Which policy?".into(),
        verbatim: true,
        tool: Some("reply_expert".into()),
    });
    let first = d.result().clone();
    let RunOutcome::InputRequired { answer_tool, .. } = &first.outcome else { panic!() };
    assert_eq!(answer_tool.as_deref(), Some("reply_expert"));

    let mut input = d.run.input.clone();
    input.history = first.all_messages.clone();
    input.opening = Opening::Message(Box::new(user("msg_2", "Gold plan")));
    input.answer_tool = answer_tool.clone();
    let mut d = Driver::with_input(text_agent(tools()), input);
    assert_eq!(d.requests(), 0);
    assert_eq!(d.prepares()[0].1, Map::from_iter([("answer".to_owned(), json!("Gold plan"))]));
    d.run_tools(|_| value("covered"));
    let (_, ctx, _) = d.request();
    let outputs: Vec<_> =
        transcript(&ctx.items).into_iter().filter(|(k, _)| k == "output").collect();
    assert_eq!(outputs, pairs(&[("output", "Gold plan"), ("output", "covered")]));
}

// Catches: an implicit loop cap ending a long, finite model/tool sequence.
#[test]
fn a_long_finite_tool_sequence_completes_without_a_loop_cap() {
    let mut d = Driver::new(text_agent(vec![string_tool("step", "n")]), "go", vec![]);
    for n in 0..40 {
        d.reply("", &[("step", json!({"n": n.to_string()}))]);
        d.run_tools(|_| value("ok"));
    }
    d.reply("done", &[]);
    assert_eq!(d.result().usage.requests, 41);
}

// Catches: dynamic instructions not rendered before each request, or rendered ones entering the
// history.
#[test]
fn dynamic_instructions_are_rendered_before_every_request() {
    let mut a = text_agent(vec![string_tool("t", "x")]);
    a.dynamic_instructions = true;
    let mut d = Driver::new(a, "hi", vec![]);
    for (n, rendered) in ["Patient: none.", "Patient: Ada."].into_iter().enumerate() {
        let render = d.effects.iter().rev().find_map(|e| match e {
            AgentRunEffect::RenderInstructions { render } => Some(*render),
            _ => None,
        });
        assert_eq!(d.requests(), n);
        d.send(AgentRunEvent::InstructionsRendered {
            render: render.unwrap(),
            result: Ok(rendered.into()),
        });
        let (_, ctx, _) = d.request();
        assert_eq!(transcript(&ctx.items)[0].1, format!("Book appointments.\n\n{rendered}"));
        if n == 0 {
            d.reply("", &[("t", json!({"x": "1"}))]);
            d.run_tools(|_| value("ok"));
        }
    }
    d.reply("hello", &[]);
    assert!(d.result().all_messages.iter().all(|i| !i.is_instruction()));
}

// A host tool's retained remote-task identity must survive into the history the next run loads.
#[test]
fn tool_metadata_is_recorded_with_its_original_call() {
    let mut driver =
        Driver::new(text_agent(vec![string_tool("expert", "request")]), "Help", vec![]);
    driver.reply("", &[("expert", json!({"request": "Check coverage"}))]);
    let (call, arguments) = driver.prepares()[0].clone();
    driver.prepared(call, Value::Object(arguments));
    driver.send(AgentRunEvent::ToolCallUpdated {
        call,
        extra: [("lk.delegation.task_id".into(), json!("remote-task"))].into_iter().collect(),
    });
    driver.send(AgentRunEvent::ToolSettled {
        call,
        outcome: ToolOutcome::InputRequired {
            question: "Which payer?".into(),
            verbatim: false,
            tool: None,
        },
    });
    driver.reply("Which payer?", &[]);
    let call = driver
        .result()
        .new_messages
        .iter()
        .find_map(|item| match item {
            ChatItem::FunctionCall(call) => Some(call),
            _ => None,
        })
        .unwrap();
    assert_eq!(call.extra["lk.delegation.task_id"], "remote-task");
}

// Catches: a tool's own cancellation (a remote task cancelled) not cancelling the caller's run.
#[test]
fn an_unsolicited_tool_cancellation_cancels_the_run() {
    let mut driver =
        Driver::new(text_agent(vec![string_tool("expert", "request")]), "Help", vec![]);
    driver.reply("", &[("expert", json!({"request": "Check coverage"}))]);
    driver.run_tools(|_| ToolOutcome::Cancelled);
    assert_eq!(driver.finished(), Some(&Err(RunError::Cancelled)));
    assert_eq!(driver.requests(), 1);
}

// Catches: a failed validator leaving the run waiting on, or cancelling, its own settled callback.
#[test]
fn a_failed_text_validator_does_not_wait_for_its_own_already_received_result() {
    let mut d = Driver::new(agent(vec![], OutputSpec::Text { validated: true }), "hi", vec![]);
    d.reply("hello", &[]);
    let effects = d.send(AgentRunEvent::OutputValidated {
        validation: d.validation(),
        verdict: OutputVerdict::Failed("validator failed".into()),
    });
    assert_eq!(d.finished(), Some(&Err(RunError::Host("validator failed".into()))));
    assert!(d.run.wake_at().is_none(), "the failed callback is no longer cleanup owed");
    assert!(
        !effects.iter().any(|effect| matches!(effect, AgentRunEffect::OutputValidateCancel { .. }))
    );
}

mod checkpoints {
    //! Managed checkpoints settle before model admission and accepted answer delivery.

    use super::*;

    fn managed(tools: Vec<FunctionToolInfo>) -> Driver {
        let input = input(Opening::Message(Box::new(user("message", "Find a room"))), Vec::new());
        Driver::start(AgentRun::new(text_agent(tools), input).unwrap().with_checkpoints())
    }

    fn checkpoint(driver: &Driver) -> CheckpointId {
        driver
            .effects
            .iter()
            .rev()
            .find_map(|effect| match effect {
                AgentRunEffect::HostCheckpoint { checkpoint, .. } => Some(*checkpoint),
                _ => None,
            })
            .expect("a managed checkpoint")
    }

    fn verdict(driver: &mut Driver, yielded: bool) -> Vec<AgentRunEffect> {
        driver.send(AgentRunEvent::HostCheckpointDone {
            checkpoint: checkpoint(driver),
            result: Ok(yielded),
            tools: None,
        })
    }

    #[test]
    fn a_waiter_after_a_tool_preserves_its_result_without_starting_the_next_model() {
        let mut driver = managed(vec![string_tool("lookup", "room")]);
        assert_eq!(driver.requests(), 0);
        let first = checkpoint(&driver);
        assert!(
            driver
                .send(AgentRunEvent::HostCheckpointDone {
                    checkpoint: CheckpointId(first.0 + 1),
                    result: Ok(false),
                    tools: None,
                })
                .is_empty(),
            "a stale verdict cannot launch a request"
        );
        verdict(&mut driver, false);
        driver.reply("", &[("lookup", json!({"room":"suite"}))]);
        driver.run_tools(|_| value("suite found"));
        assert_eq!(driver.requests(), 1, "the next model still waits for admission");
        let effects = verdict(&mut driver, true);
        let result = driver.result();
        assert!(result.interrupted);
        assert_eq!(result.usage.requests, 1);
        assert_eq!(result.usage.tool_calls, 1);
        assert_eq!(result.outcome, RunOutcome::NoOutput);
        assert!(result.new_messages.iter().any(|item| matches!(item,
            ChatItem::FunctionCallOutput(output) if output.output == "suite found"
        )));
        assert!(!effects.iter().any(|effect| matches!(
            effect,
            AgentRunEffect::LlmRequestStart { .. }
                | AgentRunEffect::Emit(StreamEvent::TextDelta { .. })
        )));
    }

    #[test]
    fn the_accepted_result_waits_for_its_delivery_verdict() {
        let mut driver = managed(Vec::new());
        verdict(&mut driver, false);
        driver.reply("the answer", &[]);
        assert!(driver.finished().is_none());
        verdict(&mut driver, true);
        assert!(driver.result().interrupted);
        assert!(driver.result().new_messages.iter().any(|item| matches!(item,
            ChatItem::Message(message) if message.raw_text_content().as_deref() == Some("the answer")
        )));

        let mut continued = managed(Vec::new());
        verdict(&mut continued, false);
        continued.reply("the accepted answer", &[]);
        verdict(&mut continued, false);
        assert!(!continued.result().interrupted);
        assert_eq!(
            continued.result().outcome,
            RunOutcome::Completed { output: Output::Text("the accepted answer".into()) }
        );
    }

    #[test]
    fn cancellation_waits_for_a_pending_checkpoint_callback_to_settle() {
        let mut driver = managed(Vec::new());
        let pending = checkpoint(&driver);
        let effects = driver.send(AgentRunEvent::Cancel);
        assert!(effects.contains(&AgentRunEffect::HostCheckpointCancel { checkpoint: pending }));
        assert!(driver.finished().is_none());
        verdict(&mut driver, false);
        assert!(matches!(driver.finished(), Some(Err(RunError::Cancelled))));
        assert_eq!(driver.requests(), 0);
        assert!(
            driver
                .send(AgentRunEvent::HostCheckpointDone {
                    checkpoint: pending,
                    result: Ok(false),
                    tools: None
                })
                .is_empty()
        );
    }

    #[test]
    fn discovered_tools_replace_the_previous_steps_tools_before_request_and_execution() {
        let mut driver = managed(vec![string_tool("search", "query")]);
        verdict(&mut driver, false);
        driver.reply("", &[("search", json!({"query":"room"}))]);
        driver.run_tools(|_| value("found book"));
        let pending = checkpoint(&driver);
        driver.send(AgentRunEvent::HostCheckpointDone {
            checkpoint: pending,
            result: Ok(false),
            tools: Some(vec![string_tool("book", "room")]),
        });
        let (_, _, offered) = driver.request();
        assert_eq!(offered.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["book"]);
        driver.reply("", &[("book", json!({"room":"suite"}))]);
        driver.run_tools(|_| value("booked"));
        verdict(&mut driver, false);
        driver.reply("Reserved.", &[]);
        verdict(&mut driver, false);
        assert!(driver.result().new_messages.iter().any(|item| matches!(item,
            ChatItem::FunctionCallOutput(output) if output.output == "booked" && !output.is_error
        )));
    }

    #[test]
    fn an_invalid_discovery_fails_before_the_model_request() {
        let mut driver = managed(Vec::new());
        let tool = string_tool("book", "room");
        driver.send(AgentRunEvent::HostCheckpointDone {
            checkpoint: checkpoint(&driver),
            result: Ok(false),
            tools: Some(vec![tool.clone(), tool]),
        });
        assert_eq!(driver.requests(), 0);
        assert!(
            matches!(driver.finished(), Some(Err(RunError::Host(error))) if error.contains("book"))
        );
    }
}
