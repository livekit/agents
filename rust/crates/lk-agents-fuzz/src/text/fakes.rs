use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value, json};
use tokio::sync::watch;

use super::drive::Env;
use lk_agents_abi::a2a::turn::{HandlerOutput, HandlerPort, TurnContext, TurnHost};
use lk_agents_abi::agent::{
    AgentCallbacks, ArgumentIdentity, FunctionToolInfo, Invocation, OutputCandidate,
    PreparedArguments, ReadBackAnswer, ToolArgumentError, ToolCompletion, ValidatedOutput,
};
use lk_agents_abi::errors::{HostError, HostErrorKind, ProviderError};
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::{
    BoxFuture, ChatItem, ChatMessage, ChatRequest, ChoiceDelta, Closed, FunctionToolCall, HostRef,
    Llm, LlmChunk, LlmStream, Role, StreamOutput, ToolSpec, Ts,
};

// The model and the host answer, panic or never answer as a scenario says; a fake that never
// answers still honours cancellation, as an asyncio task does.

/// A host callback or model boundary a fault can hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Callback {
    Checkpoint,
    ListTools,
    Prepare,
    RunTool,
    Validate,
    LlmOpen,
    LlmChunk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    Answer,
    Panic,
    /// Never answers until cancelled.
    Hang,
}

/// What one run's fakes share: their faults, cancellations and the handler port.
pub struct RunFakes {
    pub id: u32,
    env: Arc<Env>,
    rng: Mutex<fastrand::Rng>,
    /// Forced faults not yet spent, each on its callback's first call.
    forced: Mutex<Vec<(Callback, Fault)>>,
    random: bool,
    requests: Mutex<u32>,
    calls: u32,
    cancelled: watch::Sender<BTreeSet<u64>>,
    /// The A2A handler's turn, whose checkpoint the run's checkpoint answers.
    port: Option<Arc<dyn HandlerPort>>,
}

impl RunFakes {
    pub fn new(env: Arc<Env>, id: u32, port: Option<Arc<dyn HandlerPort>>) -> Arc<Self> {
        let (seed, forced, random) = env.fakes_for(id);
        let mut rng = fastrand::Rng::with_seed(seed);
        let calls = rng.u32(0..=2);
        Arc::new(Self {
            id,
            env,
            rng: Mutex::new(rng),
            forced: Mutex::new(forced),
            random,
            requests: Mutex::new(0),
            calls,
            cancelled: watch::Sender::new(BTreeSet::new()),
            port,
        })
    }

    fn fault(&self, callback: Callback) -> Fault {
        let forced = {
            let mut forced = self.forced.lock();
            let at = forced.iter().position(|(c, _)| *c == callback);
            at.map(|at| forced.remove(at).1)
        };
        let fault =
            forced.unwrap_or_else(|| match self.random.then(|| self.rng.lock().u32(0..32)) {
                Some(0) => Fault::Panic,
                Some(1) => Fault::Hang,
                _ => Fault::Answer,
            });
        match fault {
            Fault::Panic => self.env.mark(self.id, |r| r.panicked = true),
            Fault::Hang => self.env.mark(self.id, |r| r.hung = true),
            Fault::Answer => {}
        }
        fault
    }

    /// `value`, or the fault's: a panic, or `cancelled` once the host cancels `operation`.
    async fn answer<T>(
        &self,
        callback: Callback,
        operation: u64,
        value: impl Future<Output = T>,
        cancelled: T,
    ) -> T {
        match self.fault(callback) {
            Fault::Answer => value.await,
            Fault::Panic => panic!("dst fault: {callback:?} panicked"),
            Fault::Hang => {
                let mut seen = self.cancelled.subscribe();
                let _ = seen.wait_for(|ops| ops.contains(&operation)).await;
                cancelled
            }
        }
    }

    fn host_ref(&self) -> HostRef {
        HostRef::new(1, &self.env.refs)
    }
}

pub fn tool() -> FunctionToolInfo {
    FunctionToolInfo {
        spec: ToolSpec {
            provider_tool: None,
            name: "lookup".into(),
            description: "Look a booking up.".into(),
            json_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
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

fn cancelled_error() -> HostError {
    HostError { kind: HostErrorKind::Cancelled, ..HostError::new("cancelled") }
}

pub struct FakeHost(pub Arc<RunFakes>);

impl AgentCallbacks for FakeHost {
    fn list_tools<'a>(
        &'a self,
        _cx: &Cx,
        inv: &'a Invocation,
    ) -> BoxFuture<'a, Result<Vec<FunctionToolInfo>, HostError>> {
        let value = async { Ok(vec![tool()]) };
        Box::pin(self.0.answer(Callback::ListTools, inv.operation, value, Err(cancelled_error())))
    }

    fn checkpoint<'a>(
        &'a self,
        _cx: &Cx,
        inv: &'a Invocation,
    ) -> BoxFuture<'a, Result<bool, HostError>> {
        let port = self.0.port.clone();
        let value = async move {
            match port {
                Some(port) => Ok(port.checkpoint().await),
                None => Ok(false),
            }
        };
        Box::pin(self.0.answer(Callback::Checkpoint, inv.operation, value, Err(cancelled_error())))
    }

    fn render_instructions<'a>(
        &'a self,
        _cx: &Cx,
        _inv: &'a Invocation,
    ) -> BoxFuture<'a, Result<String, HostError>> {
        Box::pin(async { Ok(String::new()) })
    }

    fn prepare_tool<'a>(
        &'a self,
        _cx: &Cx,
        inv: &'a Invocation,
        arguments: Map<String, Value>,
    ) -> BoxFuture<'a, Result<Result<PreparedArguments, ToolArgumentError>, HostError>> {
        let prepared = PreparedArguments {
            prepared: self.0.host_ref(),
            identity: ArgumentIdentity::Json(Value::Object(arguments)),
        };
        let value = async { Ok(Ok(prepared)) };
        Box::pin(self.0.answer(Callback::Prepare, inv.operation, value, Err(cancelled_error())))
    }

    fn read_back<'a>(
        &'a self,
        _cx: &Cx,
        _inv: &'a Invocation,
        _prepared: &'a HostRef,
    ) -> BoxFuture<'a, Result<ReadBackAnswer, HostError>> {
        Box::pin(async { Ok(ReadBackAnswer::Question("Sure?".into())) })
    }

    fn run_tool<'a>(
        &'a self,
        _cx: &Cx,
        inv: &'a Invocation,
        _prepared: &'a HostRef,
    ) -> BoxFuture<'a, Result<ToolCompletion, HostError>> {
        let completion = |outcome: ToolOutcome| ToolCompletion {
            report: outcome.into(),
            extra: Default::default(),
            usage: None,
        };
        let found = ToolOutcome::Value {
            text: "booked for 9:00".into(),
            reply_required: true,
            verbatim: false,
            more: None,
        };
        let value = async move { Ok(completion(found)) };
        let cancelled = Ok(completion(ToolOutcome::Cancelled));
        Box::pin(self.0.answer(Callback::RunTool, inv.operation, value, cancelled))
    }

    fn validate_output<'a>(
        &'a self,
        _cx: &Cx,
        inv: &'a Invocation,
        _candidate: OutputCandidate,
    ) -> BoxFuture<'a, Result<ValidatedOutput, HostError>> {
        let value =
            async { Ok(ValidatedOutput::Accepted { text: None, value: None, serialized: None }) };
        Box::pin(self.0.answer(Callback::Validate, inv.operation, value, Err(cancelled_error())))
    }

    fn cancel(&self, invocation: &Invocation) {
        self.0.cancelled.send_modify(|ops| {
            ops.insert(invocation.operation);
        });
    }
}

/// The model: a step of tool calls when the run's script has some, then a text answer.
pub struct FakeLlm(pub Arc<RunFakes>);

impl Llm for FakeLlm {
    fn label(&self) -> String {
        "dst".into()
    }

    fn chat<'a>(
        &'a self,
        _cx: &Cx,
        _req: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            let fakes = &self.0;
            let opened = fakes.fault(Callback::LlmOpen);
            let request = {
                let mut requests = fakes.requests.lock();
                *requests += 1;
                *requests
            };
            let calls = if request == 1 { fakes.calls } else { 0 };
            let tool_calls: Vec<FunctionToolCall> = (0..calls)
                .map(|i| FunctionToolCall {
                    name: "lookup".into(),
                    arguments: json!({"q": format!("booking {i}")}).to_string(),
                    call_id: format!("call_{}_{i}", fakes.id),
                    extra: None,
                })
                .collect();
            let content = tool_calls.is_empty().then(|| "Your booking is at 9:00.".to_owned());
            let chunk = LlmChunk {
                id: format!("chunk_{request}"),
                delta: Some(ChoiceDelta {
                    role: Some(Role::Assistant),
                    content,
                    tool_calls,
                    extra: None,
                }),
                usage: None,
                flush: false,
                node_text: false,
            };
            match opened {
                Fault::Panic => panic!("dst fault: the model panicked opening its stream"),
                Fault::Hang => std::future::pending().await,
                Fault::Answer => {}
            }
            let stream = Scripted { fakes: fakes.clone(), chunks: VecDeque::from([chunk]) };
            Ok(Box::new(stream) as LlmStream)
        })
    }
}

struct Scripted {
    fakes: Arc<RunFakes>,
    chunks: VecDeque<LlmChunk>,
}

impl StreamOutput<LlmChunk> for Scripted {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            let Some(chunk) = self.chunks.pop_front() else { return Ok(None) };
            match self.fakes.fault(Callback::LlmChunk) {
                Fault::Answer => Ok(Some(chunk)),
                Fault::Panic => panic!("dst fault: the model stream panicked"),
                Fault::Hang => std::future::pending().await,
            }
        })
    }

    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Scripted {}

/// The `a2a_session` handler: one AgentRun on the turn's message, its checkpoints the turn's.
pub struct FakeHandler(pub Arc<Env>);

impl TurnHost for FakeHandler {
    fn handle<'a>(
        &'a self,
        _cx: &Cx,
        ctx: &'a TurnContext,
        turn: Arc<dyn HandlerPort>,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'a, Result<HandlerOutput, ProviderError>> {
        Box::pin(async move {
            let env = self.0.clone();
            let (finished, _stop) = env.start_run(super::RunKind::Handler, Some(turn), None);
            let finished = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    return Err(ProviderError::Host {
                        error: cancelled_error(),
                        api_kind: None,
                        retryable: false,
                        status_code: None,
                    });
                }
                finished = finished => finished.unwrap_or(false),
            };
            if !finished {
                return Ok(HandlerOutput { error: Some("the run failed".into()), ..empty() });
            }
            let turns = ctx.state.as_ref().and_then(|s| s["turns"].as_i64()).unwrap_or(0);
            let answer = ChatMessage::text(
                format!("answer_{}", ctx.task_id),
                Role::Assistant,
                "Your booking is at 9:00.",
                env.wall(),
            );
            Ok(HandlerOutput {
                output: Some(json!("Your booking is at 9:00.")),
                new_items: vec![ChatItem::Message(answer)],
                state: Some(json!({"turns": turns + 1})),
                ..empty()
            })
        })
    }
}

fn empty() -> HandlerOutput {
    HandlerOutput {
        output: None,
        new_items: Vec::new(),
        state: None,
        scope_writes: Vec::new(),
        app_writes: Vec::new(),
        usage: Vec::new(),
        input_required: None,
        pending_call: None,
        interrupted: false,
        metadata: Map::new(),
        error: None,
        tool_executions: Vec::new(),
        status: None,
    }
}
