//! The driver against a scripted model and a fake host: correlated host callbacks under their
//! invocation, cancellation through the host, the cleanup deadline, and release of what the host
//! handed over.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::agent::run::{RunError, RunOutcome, StreamEvent};
use lk_agents_abi::agent::{AgentDefinition, FunctionToolInfo, OutputSpec, Retries};
use lk_agents_abi::agent::{
    ArgumentIdentity, OutputCandidate, PreparedArguments, ReadBackAnswer, ToolArgumentError,
};
use lk_agents_abi::errors::{HostError, ProviderError, RefRegistry};
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::llm::tool_executor::{ToolOutcome, ToolReport};
use lk_agents_abi::resources::LlmRef;
use lk_agents_abi::{
    ChatMessage, ChoiceDelta, Closed, FunctionToolCall, LlmChunk, LlmStream, ProviderId, RealClock,
    Role, StreamSource, ToolSpec, WallAnchor,
};
use lk_agents_core::tool_outcome::Mode;
use serde_json::{Map, Value, json};
use tokio::sync::Notify;

use super::*;

/// Counts host values released, so a test sees nothing the host handed over is kept.
#[derive(Default)]
struct Registry {
    released: AtomicUsize,
}

impl RefRegistry for Registry {
    fn release(&self, _id: u64) {
        self.released.fetch_add(1, Ordering::SeqCst);
    }
}

/// One scripted model response.
enum Reply {
    Text(&'static str),
    Call(&'static str, Value),
    /// Never ends until closed.
    Hang,
}

struct ScriptedLlm {
    replies: Mutex<VecDeque<Reply>>,
    closed: Arc<AtomicUsize>,
}

struct Scripted {
    chunks: VecDeque<LlmChunk>,
    hang: bool,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource<LlmChunk> for Scripted {
    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        match self.chunks.pop_front() {
            Some(chunk) => Ok(Some(chunk)),
            None if self.hang => std::future::pending().await,
            None => Ok(None),
        }
    }

    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Closed::Clean
    }
}

#[async_trait]
impl Llm for ScriptedLlm {
    fn label(&self) -> String {
        "scripted".into()
    }

    async fn chat(&self, _req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        let reply = self.replies.lock().unwrap().pop_front().expect("a scripted reply");
        let delta = |content: Option<&str>, tool_calls| LlmChunk {
            id: "c".into(),
            delta: Some(ChoiceDelta {
                role: Some(Role::Assistant),
                content: content.map(Into::into),
                tool_calls,
                extra: None,
            }),
            usage: None,
            flush: false,
            node_text: false,
        };
        let (chunks, hang) = match reply {
            Reply::Text(text) => (vec![delta(Some(text), vec![])], false),
            Reply::Call(name, arguments) => {
                let call = FunctionToolCall {
                    name: name.into(),
                    arguments: arguments.to_string(),
                    call_id: format!("call_{name}"),
                    extra: None,
                };
                (vec![delta(None, vec![call])], false)
            }
            Reply::Hang => (vec![delta(Some("Thinking"), vec![])], true),
        };
        Ok(Box::new(Scripted { chunks: chunks.into(), hang, closed: self.closed.clone() }))
    }
}

/// How the fake host answers a tool body.
#[derive(Clone, Copy, PartialEq)]
enum Body {
    Returns,
    /// Runs until cancelled, then reports the cancellation.
    UntilCancelled,
    /// Ignores cancellation: never settles.
    Stuck,
}

struct FakeHost {
    registry: Arc<dyn RefRegistry>,
    next_ref: AtomicUsize,
    body: Body,
    /// Every invocation a callback ran under: (callback, operation, tool, approval).
    seen: Mutex<Vec<(&'static str, Invocation)>>,
    cancelled: Mutex<Vec<u64>>,
    cancel: Notify,
}

impl FakeHost {
    fn new(body: Body, registry: Arc<dyn RefRegistry>) -> Arc<Self> {
        Arc::new(FakeHost {
            registry,
            next_ref: AtomicUsize::new(1),
            body,
            seen: Mutex::new(Vec::new()),
            cancelled: Mutex::new(Vec::new()),
            cancel: Notify::new(),
        })
    }

    fn host_ref(&self) -> HostRef {
        HostRef::new(self.next_ref.fetch_add(1, Ordering::SeqCst) as u64, &self.registry)
    }

    fn saw(&self, callback: &'static str, invocation: &Invocation) {
        self.seen.lock().unwrap().push((callback, invocation.clone()));
    }
}

#[async_trait]
impl AgentHost for FakeHost {
    async fn render_instructions(&self, invocation: &Invocation) -> Result<String, HostError> {
        self.saw("instructions", invocation);
        Ok(String::new())
    }

    async fn prepare_tool(
        &self,
        invocation: &Invocation,
        arguments: Map<String, Value>,
    ) -> Result<Result<PreparedArguments, ToolArgumentError>, HostError> {
        self.saw("prepare", invocation);
        Ok(Ok(PreparedArguments {
            prepared: self.host_ref(),
            identity: ArgumentIdentity::Json(Value::Object(arguments)),
        }))
    }

    async fn read_back(
        &self,
        invocation: &Invocation,
        _prepared: &HostRef,
    ) -> Result<ReadBackAnswer, HostError> {
        self.saw("read_back", invocation);
        Ok(ReadBackAnswer::Question("Sure?".into()))
    }

    async fn run_tool(
        &self,
        invocation: &Invocation,
        _prepared: &HostRef,
    ) -> Result<ToolReport, HostError> {
        self.saw("run", invocation);
        match self.body {
            Body::Returns => {}
            Body::UntilCancelled => {
                self.cancel.notified().await;
                return Ok(ToolOutcome::Cancelled.into());
            }
            Body::Stuck => std::future::pending::<()>().await,
        }
        Ok(ToolOutcome::Value {
            text: "9:00 is free".into(),
            reply_required: true,
            verbatim: false,
            more: None,
        }
        .into())
    }

    async fn validate_output(
        &self,
        invocation: &Invocation,
        candidate: OutputCandidate,
    ) -> Result<ValidatedOutput, HostError> {
        self.saw("validate", invocation);
        let OutputCandidate::Value(_) = candidate else {
            return Ok(ValidatedOutput::Retry("not text".into()));
        };
        Ok(ValidatedOutput::Accepted { text: None, value: Some(self.host_ref()) })
    }

    fn cancel(&self, invocation: &Invocation) {
        self.cancelled.lock().unwrap().push(invocation.operation);
        self.cancel.notify_waiters();
    }
}

fn tool(name: &str) -> FunctionToolInfo {
    FunctionToolInfo {
        spec: ToolSpec {
            name: name.into(),
            description: "Find a slot.".into(),
            json_schema: json!({"type": "object", "properties": {"day": {"type": "string"}}}),
            flags: Default::default(),
        },
        raw_schema: false,
        needs_approval: None,
        on_duplicate: DuplicateMode::Allow,
        duplicate_scope: DuplicateScope::Name,
    }
}

struct Setup {
    handle: AgentRunHandle,
    updates: mpsc::Receiver<RunUpdate>,
    host: Arc<FakeHost>,
    closed: Arc<AtomicUsize>,
}

fn setup(
    replies: Vec<Reply>,
    body: Body,
    output: OutputSpec,
    registry: Arc<dyn RefRegistry>,
) -> Setup {
    let closed = Arc::new(AtomicUsize::new(0));
    let llm = Arc::new(ScriptedLlm { replies: Mutex::new(replies.into()), closed: closed.clone() });
    let host = FakeHost::new(body, registry.clone());
    let agent = AgentDefinition {
        name: "scheduler".into(),
        instructions: "Book.".into(),
        dynamic_instructions: false,
        tools: vec![tool("find_slot")],
        output,
        retries: Retries::default(),
    };
    let input = RunInput {
        invocation_id: "run_1".into(),
        message: ChatMessage::text("m1", Role::User, "Monday?", std::time::UNIX_EPOCH),
        history: vec![],
        answer_tool: None,
        delegated: false,
        mode: Mode::Text,
        llm: Some(LlmRef { id: ResourceId(1), provider: ProviderId::of_label("scripted") }),
        wall: WallAnchor { wall: 0.0, at: Ts::ZERO },
        cleanup_timeout: Duration::from_secs(5),
    };
    let entry = Invocation {
        operation: 0,
        invocation_id: "run_1".into(),
        agent_name: "scheduler".into(),
        function_call: None,
        approval: None,
        host_context: Some(HostRef::new(0, &registry)),
    };
    let ports = Ports {
        llm,
        host: host.clone(),
        clock: Arc::new(RealClock::new()),
        conn_options: Default::default(),
        entry,
    };
    let (handle, updates) =
        start(&tokio::runtime::Handle::current(), agent, input, ports, 4).unwrap();
    Setup { handle, updates, host, closed }
}

async fn drain(updates: &mut mpsc::Receiver<RunUpdate>) -> Vec<RunUpdate> {
    let mut all = Vec::new();
    while let Some(update) = updates.recv().await {
        all.push(update);
    }
    all
}

fn finished(updates: &[RunUpdate]) -> Result<&RunSettled, &RunError> {
    match updates.last() {
        Some(RunUpdate::Finished(settled)) => Ok(settled),
        Some(RunUpdate::Failed(error)) => Err(error),
        other => panic!("the last update is not the result: {other:?}"),
    }
}

// Catches: a callback run without its call's invocation scope (or a scope leaking into another
// callback), the operation entry not captured once, or a host value kept after the run.
#[tokio::test(start_paused = true)]
async fn each_callback_runs_under_its_own_invocation_and_everything_is_released() {
    let registry = Arc::new(Registry::default());
    let mut s = setup(
        vec![Reply::Call("find_slot", json!({"day": "Monday"})), Reply::Text("Monday 9:00.")],
        Body::Returns,
        OutputSpec::Text { validated: false },
        registry.clone(),
    );
    let updates = drain(&mut s.updates).await;
    let Ok(settled) = finished(&updates) else { panic!() };
    assert_eq!(
        settled.result.outcome,
        RunOutcome::Completed { output: Output::Text("Monday 9:00.".into()) }
    );
    assert!(updates.iter().any(|u| matches!(u, RunUpdate::Stream(StreamEvent::ToolResult { output, .. }) if output == "9:00 is free")));

    let seen = s.host.seen.lock().unwrap().clone();
    let callbacks: Vec<_> = seen
        .iter()
        .map(|(c, i)| (*c, i.function_call.as_ref().map(|f| f.call_id.clone())))
        .collect();
    assert_eq!(
        callbacks,
        [("prepare", Some("call_find_slot".into())), ("run", Some("call_find_slot".into()))]
    );
    assert!(
        seen.iter().all(|(_, i)| i.invocation_id == "run_1"
            && i.host_context.as_ref().map(HostRef::id) == Some(0))
    );
    assert_ne!(seen[0].1.operation, seen[1].1.operation);
    drop(seen);
    drop(s);
    // The prepared arguments, and every copy of the captured entry context, released.
    assert_eq!(registry.released.load(Ordering::SeqCst), 2);
}

// Catches: a validated output value lost between the host and the result.
#[tokio::test(start_paused = true)]
async fn the_validated_output_reaches_the_result() {
    let registry = Arc::new(Registry::default());
    let mut s = setup(
        vec![Reply::Call("final_result", json!({"day": "Monday"}))],
        Body::Returns,
        OutputSpec::Structured {
            schema: json!({"type": "object", "properties": {"day": {"type": "string"}}}),
        },
        registry,
    );
    let updates = drain(&mut s.updates).await;
    let Ok(settled) = finished(&updates) else { panic!() };
    assert!(settled.output.is_some());
}

// Catches (decision 16): cancellation not reaching the running body through the host, or the run
// settling before the body reported back.
#[tokio::test(start_paused = true)]
async fn cancel_reaches_the_running_tool_through_the_host_and_waits_for_it() {
    let mut s = setup(
        vec![Reply::Call("find_slot", json!({"day": "Monday"}))],
        Body::UntilCancelled,
        OutputSpec::Text { validated: false },
        Arc::new(Registry::default()),
    );
    while !s.host.seen.lock().unwrap().iter().any(|(c, _)| *c == "run") {
        tokio::task::yield_now().await;
    }
    s.handle.cancel();
    let updates = drain(&mut s.updates).await;
    assert!(matches!(finished(&updates), Err(RunError::Cancelled)));
    assert!(!updates.iter().any(|u| matches!(u, RunUpdate::IncompleteCleanup(_))));
    let run_operation =
        s.host.seen.lock().unwrap().iter().find(|(c, _)| *c == "run").unwrap().1.operation;
    assert_eq!(*s.host.cancelled.lock().unwrap(), [run_operation]);
}

// Catches (decision 14): a body that never settles holding the run forever, or the deadline
// passing silently.
#[tokio::test(start_paused = true)]
async fn a_body_that_ignores_cancel_is_reported_incomplete_at_the_deadline() {
    let mut s = setup(
        vec![Reply::Call("find_slot", json!({"day": "Monday"}))],
        Body::Stuck,
        OutputSpec::Text { validated: false },
        Arc::new(Registry::default()),
    );
    while !s.host.seen.lock().unwrap().iter().any(|(c, _)| *c == "run") {
        tokio::task::yield_now().await;
    }
    s.handle.cancel();
    let mut updates = Vec::new();
    while let Some(update) = s.updates.recv().await {
        let last = matches!(update, RunUpdate::Finished(_) | RunUpdate::Failed(_));
        updates.push(update);
        if last {
            break;
        }
    }
    assert!(
        matches!(&updates[updates.len() - 2], RunUpdate::IncompleteCleanup(pending) if pending.len() == 1)
    );
    assert!(matches!(finished(&updates), Err(RunError::Cancelled)));
}

// Catches: a cancelled model request left streaming, or its stream dropped without its close.
#[tokio::test(start_paused = true)]
async fn cancel_closes_the_model_stream_in_flight() {
    let mut s = setup(
        vec![Reply::Hang],
        Body::Returns,
        OutputSpec::Text { validated: false },
        Arc::new(Registry::default()),
    );
    let Some(RunUpdate::Stream(StreamEvent::TextDelta { .. })) = s.updates.recv().await else {
        panic!()
    };
    s.handle.cancel();
    let updates = drain(&mut s.updates).await;
    assert!(matches!(finished(&updates), Err(RunError::Cancelled)));
    assert_eq!(s.closed.load(Ordering::SeqCst), 1);
}

// Catches: the host's runner accepting a definition the core refuses (the error surfacing only on
// the first run), or a run whose stream ends before the settled result, or whose result is not the
// model's text.
#[tokio::test]
async fn the_runner_refuses_an_invalid_definition_and_streams_a_run_to_its_result() {
    use lk_agents_abi::agent::run::{AgentRunner as _, Output, RunRequest};

    let registry: Arc<dyn RefRegistry> = Arc::new(Registry::default());
    let llm = || -> Arc<dyn Llm> {
        Arc::new(ScriptedLlm {
            replies: Mutex::new(vec![Reply::Text("Monday works.")].into()),
            closed: Arc::new(AtomicUsize::new(0)),
        })
    };
    let host = FakeHost::new(Body::Returns, registry.clone());
    let definition = |tools: Vec<FunctionToolInfo>| AgentDefinition {
        name: "scheduler".into(),
        instructions: "Book.".into(),
        dynamic_instructions: false,
        tools,
        output: OutputSpec::Text { validated: false },
        retries: Retries::default(),
    };
    let invalid = definition(vec![tool("find_slot"), tool("find_slot")]);
    let refused = super::Runner::new(invalid, llm(), host.clone()).err().expect("refused");
    assert!(refused.0.contains("find_slot"), "{refused}");

    let runner = super::Runner::new(definition(vec![]), llm(), host).unwrap();
    let mut updates = runner
        .run(RunRequest {
            invocation_id: "run_1".into(),
            message: lk_agents_abi::ChatItem::Message(ChatMessage::text(
                "m1",
                Role::User,
                "Monday?",
                std::time::UNIX_EPOCH,
            )),
            history: vec![],
            answer_tool: None,
            delegated: false,
            conn_options: Default::default(),
            host_context: HostRef::new(0, &registry),
        })
        .unwrap();
    let mut last = None;
    while let Some(update) = updates.next().await.unwrap() {
        last = Some(update);
    }
    let Some(RunUpdate::Finished(settled)) = last else { panic!("{last:?}") };
    let RunOutcome::Completed { output: Output::Text(text) } = settled.result.outcome else {
        panic!("{:?}", settled.result.outcome)
    };
    assert_eq!(text, "Monday works.");
}
