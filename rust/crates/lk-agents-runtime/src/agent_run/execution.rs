//! I/O for one invocation, shared by the direct runner and TextSession. Coordinators own
//! decisions and update delivery; this executor owns host work and provider requests.
use super::*;
use crate::host::HostInvoker;
use crate::telemetry;
use lk_agents_abi::agent::run::{RunInputRequirement, RunOutcome, RunResult};
use lk_agents_abi::agent::{FunctionToolInfo, OutputCandidate, ToolArgumentError, ToolTemplate};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::{ChatContext, ToolSpec};
use serde_json::{Map, Value};

#[derive(Clone)]
pub(crate) struct Reporter {
    pub id: String,
    pub tx: mpsc::Sender<(String, Report)>,
}

impl Reporter {
    async fn send(&self, report: Report) -> Result<(), ()> {
        self.tx.send((self.id.clone(), report)).await.map_err(|_| ())
    }
}

pub(crate) enum Report {
    Core(AgentRunEvent),
    /// A failed model request, with the provider's error kept for the run's caller.
    LlmFailed {
        request: RequestId,
        error: ProviderError,
    },
    Prepared {
        call: CallId,
        resource: ResourceId,
        prepared: PreparedArguments,
    },
    Validated {
        validation: ValidationId,
        verdict: OutputVerdict,
        value: Option<HostRef>,
        serialized: Option<serde_json::Value>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Work {
    Checkpoint(CheckpointId),
    Call(CallId),
    Render(RenderId),
    Validation(ValidationId),
}

pub(crate) struct Execution {
    pub invoker: HostInvoker,
    /// The invocation's `agent_run` span, ended with its result; dropped (cancelled) with an
    /// abandoned invocation. Its tool calls nest under it.
    span: Option<telemetry::ScopedSpan<telemetry::AgentRun>>,
    /// The span's context, kept past its end for work still settling.
    cx: telemetry::Cx,
    pub(super) setup: RunSetup,
    runtime: tokio::runtime::Handle,
    events_tx: Reporter,
    pub requests: BTreeMap<RequestId, Option<oneshot::Sender<()>>>,
    operations: BTreeMap<Work, u64>,
    calls: BTreeMap<CallId, FunctionCall>,
    prepared: BTreeMap<ResourceId, HostRef>,
    /// The last model request's tool descriptions, by name, for its calls' `function_tool`.
    descriptions: BTreeMap<String, String>,
    pub outputs: BTreeMap<ValidationId, HostRef>,
    next_resource: u64,
    /// The provider error of the last failed model request.
    pub failure: Option<ProviderError>,
}

impl Execution {
    pub fn new(
        setup: RunSetup,
        runtime: tokio::runtime::Handle,
        events_tx: Reporter,
        span: telemetry::ScopedSpan<telemetry::AgentRun>,
    ) -> Self {
        Self {
            invoker: HostInvoker::new(setup.entry.clone()),
            cx: span.cx().clone(),
            span: Some(span),
            setup,
            runtime,
            events_tx,
            requests: BTreeMap::new(),
            operations: BTreeMap::new(),
            calls: BTreeMap::new(),
            prepared: BTreeMap::new(),
            descriptions: BTreeMap::new(),
            outputs: BTreeMap::new(),
            next_resource: 0,
            failure: None,
        }
    }

    pub fn idle(&self) -> bool {
        self.invoker.is_idle() && self.requests.is_empty()
    }

    pub fn progress(&self, operation: u64, message: String) -> Option<AgentRunEvent> {
        self.operations.iter().find_map(|(work, op)| match work {
            Work::Call(call) if *op == operation && self.invoker.live(*op).is_some() => {
                Some(AgentRunEvent::ToolProgress { call: *call, message: message.clone() })
            }
            _ => None,
        })
    }

    pub fn report(&mut self, report: Report) -> (AgentRunEvent, Option<serde_json::Value>) {
        let (event, serialized) = self.adopt(report);
        if let AgentRunEvent::LlmEnded { request } | AgentRunEvent::LlmFailed { request, .. } =
            &event
        {
            self.requests.remove(request);
        }
        (event, serialized)
    }
    pub(crate) async fn execute(&mut self, effect: AgentRunEffect) -> Option<RunUpdate> {
        match effect {
            // A direct run opens on its message: it plans and decides its own steps.
            AgentRunEffect::PlanRequest { .. }
            | AgentRunEffect::HandleToolResults { .. }
            | AgentRunEffect::CancelToolExecution { .. } => {
                unreachable!("only a coordinated run asks its coordinator")
            }
            AgentRunEffect::HostCheckpoint { checkpoint, list_tools } => {
                self.check_host(checkpoint, list_tools)
            }
            AgentRunEffect::HostCheckpointCancel { checkpoint } => {
                self.cancel(Work::Checkpoint(checkpoint))
            }
            AgentRunEffect::LlmRequestStart { request, chat_ctx, tools, .. } => {
                self.start_llm_request(request, chat_ctx, tools)
            }
            AgentRunEffect::LlmRequestCancel { request } => {
                if let Some(cancel) = self.requests.get_mut(&request).and_then(Option::take) {
                    let _ = cancel.send(());
                }
            }
            AgentRunEffect::RenderInstructions { render } => self.render_instructions(render),
            AgentRunEffect::ToolPrepare { call, arguments, function_call } => {
                self.prepare_tool(call, arguments, function_call)
            }
            AgentRunEffect::ReadBackRender { call, prepared, .. } => {
                self.render_read_back(call, prepared)?
            }
            AgentRunEffect::RenderToolHandlingTemplate { call, template } => {
                self.render_tool_handling(call, template)
            }
            AgentRunEffect::ToolExecute { call, prepared, function_call, approval } => {
                self.run_tool(call, prepared, function_call, approval)?
            }
            AgentRunEffect::ToolCancel { call } => self.cancel(Work::Call(call)),
            AgentRunEffect::InstructionsCancel { render } => self.cancel(Work::Render(render)),
            AgentRunEffect::OutputValidateCancel { validation } => {
                self.cancel(Work::Validation(validation))
            }
            AgentRunEffect::ToolArgumentsRelease { prepared } => {
                self.prepared.remove(&prepared);
            }
            AgentRunEffect::ValidateOutput { validation, candidate } => {
                self.validate_output(validation, candidate)
            }
            AgentRunEffect::Emit(event) => {
                return Some(RunUpdate::Stream(event));
            }
            AgentRunEffect::ArgumentsRepaired { call_id, error, repaired } => {
                log_repaired_arguments(&call_id, &error, repaired)
            }
            AgentRunEffect::IncompleteCleanup { pending } => {
                log::error!(target: "livekit.agents", "agent run cleanup missed its deadline: {}", pending.join(", "));
                return Some(RunUpdate::IncompleteCleanup(pending));
            }
            AgentRunEffect::Finished { result, input_requirements } => {
                return Some(self.finish(*result, input_requirements));
            }
        }
        None
    }

    /// The managed turn's verdict and, when the run goes on and asked, the step's tools.
    fn check_host(&mut self, checkpoint: CheckpointId, list_tools: bool) {
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        self.invoke(Work::Checkpoint(checkpoint), None, None, move |inv, events| async move {
            let (result, tools) = checkpoint_host(host.as_ref(), &cx, &inv, list_tools).await;
            let event = AgentRunEvent::HostCheckpointDone { checkpoint, result, tools };
            let _ = events.send(Report::Core(event)).await;
        });
    }

    fn start_llm_request(
        &mut self,
        request: RequestId,
        chat_ctx: ChatContext,
        tools: Vec<ToolSpec>,
    ) {
        let (cancel, cancelled) = oneshot::channel();
        self.requests.insert(request, Some(cancel));
        self.descriptions = tools.iter().map(|t| (t.name.clone(), t.description.clone())).collect();
        let req = ChatRequest {
            tools_host: self.setup.tools_host.clone(),
            chat_ctx,
            tools,
            conn_options: self.setup.conn_options,
            ..Default::default()
        };
        let llm = self.setup.llm.clone();
        let clock = self.setup.clock.clone();
        let events = self.events_tx.clone();
        let trace = telemetry::LlmTrace::start(&self.cx, llm.as_ref(), &req);
        self.runtime.spawn(stream_request(request, req, llm, clock, events, cancelled, trace));
    }

    fn render_instructions(&mut self, render: RenderId) {
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        self.invoke(Work::Render(render), None, None, move |inv, events| async move {
            let result = host.render_instructions(&cx, &inv).await.map_err(|e| e.to_string());
            let event = AgentRunEvent::InstructionsRendered { render, result };
            let _ = events.send(Report::Core(event)).await;
        });
    }

    fn prepare_tool(
        &mut self,
        call: CallId,
        arguments: Map<String, Value>,
        function_call: FunctionCall,
    ) {
        self.calls.insert(call, function_call.clone());
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        let resource = self.mint();
        let work = Work::Call(call);
        self.invoke(work, Some(function_call), None, move |inv, events| async move {
            let prepared = host.prepare_tool(&cx, &inv, arguments).await;
            let _ = events.send(prepare_report(call, resource, prepared)).await;
        });
    }

    /// Renders the tool's read-back; `None` when the prepared arguments were released.
    fn render_read_back(&mut self, call: CallId, prepared: ResourceId) -> Option<()> {
        let args = self.prepared.get(&prepared).cloned()?;
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        let function_call = self.calls.get(&call).cloned();
        self.invoke(Work::Call(call), function_call, None, move |inv, events| async move {
            let result = host.read_back(&cx, &inv, &args).await.map_err(|e| e.to_string());
            let event = AgentRunEvent::ReadBackRendered { call, result };
            let _ = events.send(Report::Core(event)).await;
        });
        Some(())
    }

    fn render_tool_handling(&mut self, call: CallId, template: ToolTemplate) {
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        let function_call = self.calls.get(&call).cloned();
        self.invoke(Work::Call(call), function_call, None, move |inv, events| async move {
            let text = host.render_template(&cx, &inv, template).await.unwrap_or_else(|e| {
                log::error!(target: "livekit.agents", "a tool_handling template failed: {e}");
                None
            });
            let event = AgentRunEvent::ToolHandlingTemplateRendered { call, text };
            let _ = events.send(Report::Core(event)).await;
        });
    }

    /// Runs the tool's body under its `function_tool` span; `None` when the prepared arguments
    /// were released.
    fn run_tool(
        &mut self,
        call: CallId,
        prepared: ResourceId,
        function_call: FunctionCall,
        approval: Option<Approval>,
    ) -> Option<()> {
        let args = self.prepared.remove(&prepared)?;
        let span = crate::telemetry::function_tool(
            &self.cx,
            &function_call,
            &self.setup.entry.agent_name,
            self.descriptions.get(&function_call.name).map(String::as_str),
        );
        let started = span.cx().now_nanos();
        let run = ToolRun {
            host: self.setup.host.clone(),
            span,
            args,
            function_call: function_call.clone(),
            call,
            started,
        };
        self.invoke(Work::Call(call), Some(function_call), approval, move |inv, events| {
            run.run(inv, events)
        });
        Some(())
    }

    fn validate_output(&mut self, validation: ValidationId, candidate: OutputCandidate) {
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        self.invoke(Work::Validation(validation), None, None, move |inv, events| async move {
            let (verdict, value, serialized) =
                crate::output_validation::project(host.validate_output(&cx, &inv, candidate).await);
            let _ = events.send(Report::Validated { validation, verdict, value, serialized }).await;
        });
    }

    /// The run's update for its result: its validated output kept, a model failure's provider
    /// error attached, and its span ended.
    fn finish(
        &mut self,
        result: Result<RunResult, RunError>,
        input_requirements: Vec<RunInputRequirement>,
    ) -> RunUpdate {
        let settled = result.map(|result| {
            let output = match result.outcome {
                RunOutcome::Completed { output: Output::Validated(v) } => self.outputs.remove(&v),
                _ => None,
            };
            RunSettled { result, output }
        });
        self.outputs.clear();
        self.prepared.clear();
        self.end_span(&settled);
        match settled {
            Ok(settled) => RunUpdate::Finished(Box::new(settled)),
            Err(error) => {
                let provider_error =
                    if matches!(error, RunError::Model(_)) { self.failure.take() } else { None };
                RunUpdate::Failed { error, input_requirements, provider_error }
            }
        }
    }

    /// Ends the invocation's span with its result.
    fn end_span(&mut self, settled: &Result<RunSettled, RunError>) {
        let Some(span) = self.span.take() else { return };
        match settled {
            Ok(settled) => {
                let status = match settled.result.outcome {
                    RunOutcome::InputRequired { .. } => telemetry::RunStatus::InputRequired,
                    _ => telemetry::RunStatus::Completed,
                };
                telemetry::end(span, telemetry::AgentRunEnd { status });
            }
            Err(error) => {
                let end = telemetry::AgentRunEnd { status: telemetry::RunStatus::Failed };
                let error = telemetry::Failure {
                    kind: telemetry::ErrorType::RunError,
                    message: error.to_string(),
                };
                telemetry::end(span, telemetry::Failed { end, error: &error });
            }
        }
    }

    /// Keeps the host value a report carries and hands the core its event.
    pub(crate) fn adopt(&mut self, report: Report) -> (AgentRunEvent, Option<serde_json::Value>) {
        match report {
            Report::Core(event) => (event, None),
            Report::LlmFailed { request, error } => {
                let message = error.to_string();
                self.failure = Some(error);
                (AgentRunEvent::LlmFailed { request, error: message }, None)
            }
            Report::Prepared { call, resource, prepared } => {
                self.prepared.insert(resource, prepared.prepared);
                (
                    AgentRunEvent::ToolPrepared {
                        call,
                        prepared: resource,
                        identity: prepared.identity,
                    },
                    None,
                )
            }
            Report::Validated { validation, verdict, value, serialized } => {
                if let Some(value) = value {
                    self.outputs.insert(validation, value);
                }
                (AgentRunEvent::OutputValidated { validation, verdict }, serialized)
            }
        }
    }

    /// One host callback through the invoker, under the work it serves.
    fn invoke<F, Fut>(
        &mut self,
        work: Work,
        function_call: Option<FunctionCall>,
        approval: Option<Approval>,
        call: F,
    ) where
        F: FnOnce(Invocation, Reporter) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let events = self.events_tx.clone();
        let operation = self
            .invoker
            .invoke(function_call, approval, &self.runtime, move |inv| call(inv, events));
        self.operations.insert(work, operation);
    }

    /// Cancels the work's callback through the host; the callback still settles.
    fn cancel(&mut self, work: Work) {
        if let Some(invocation) = self.operations.get(&work).and_then(|op| self.invoker.live(*op)) {
            self.setup.host.cancel(invocation);
        }
    }

    fn mint(&mut self) -> ResourceId {
        self.next_resource += 1;
        ResourceId(self.next_resource)
    }
}

/// The managed turn's verdict and, when `list_tools` and the run goes on, the step's tools.
async fn checkpoint_host(
    host: &dyn AgentCallbacks,
    cx: &telemetry::Cx,
    inv: &Invocation,
    list_tools: bool,
) -> (Result<bool, String>, Option<Vec<FunctionToolInfo>>) {
    let yielded = match host.checkpoint(cx, inv).await {
        Ok(yielded) => yielded,
        Err(error) => return (Err(error.to_string()), None),
    };
    if !list_tools || yielded {
        return (Ok(yielded), None);
    }
    match host.list_tools(cx, inv).await {
        Ok(tools) => (Ok(yielded), Some(tools)),
        Err(error) => (Err(error.to_string()), None),
    }
}

/// What preparing a call's arguments reports: the prepared value under `resource`, or why the
/// call was refused.
fn prepare_report(
    call: CallId,
    resource: ResourceId,
    prepared: Result<Result<PreparedArguments, ToolArgumentError>, HostError>,
) -> Report {
    let failure = match prepared {
        Ok(Ok(prepared)) => return Report::Prepared { call, resource, prepared },
        Ok(Err(e)) => PrepareFailure::Refused(e.message),
        Err(e) => {
            log::error!(target: "livekit.agents", "preparing a tool call failed: {e}");
            PrepareFailure::Internal
        }
    };
    Report::Core(AgentRunEvent::ToolPrepareFailed { call, failure })
}

fn log_repaired_arguments(call_id: &str, error: &str, repaired: Value) {
    let repaired = match repaired {
        Value::String(s) => s,
        other => other.to_string(),
    };
    log::warn!(
        target: "livekit.agents",
        call_id = call_id,
        (telemetry::log_keys::REPAIRED) = repaired.as_str(),
        (telemetry::log_keys::ERROR) = error;
        "repaired malformed function-call JSON arguments"
    );
}

/// One tool body run under its `function_tool` span.
struct ToolRun {
    host: Arc<dyn AgentCallbacks>,
    span: telemetry::ScopedSpan<telemetry::FunctionTool>,
    args: HostRef,
    function_call: FunctionCall,
    call: CallId,
    /// When the span started, in its clock's nanoseconds.
    started: u64,
}

impl ToolRun {
    /// Ends the span before reporting; the call's update precedes its settlement.
    async fn run(self, inv: Invocation, events: Reporter) {
        let Self { host, span, args, function_call, call, started } = self;
        let ran = host.run_tool(span.cx(), &inv, &args).await;
        match &ran {
            Ok((report, _)) => {
                crate::telemetry::end_function_tool(span, &function_call, started, &report.outcome)
            }
            Err(error) => telemetry::end(span, error),
        }
        let event = match ran {
            Ok((report, extra)) => {
                let _ =
                    events.send(Report::Core(AgentRunEvent::ToolCallUpdated { call, extra })).await;
                AgentRunEvent::ToolSettled { call, outcome: report.outcome }
            }
            Err(e) => AgentRunEvent::ToolLaunchFailed { call, error: e.to_string() },
        };
        let _ = events.send(Report::Core(event)).await;
    }
}

/// One model request: chunks reported as they arrive, then its end or failure. A cancel closes
/// the stream under the request's deadline and reports the request ended.
async fn stream_request(
    request: RequestId,
    req: ChatRequest,
    llm: Arc<dyn Llm>,
    clock: Arc<dyn Clock>,
    events: Reporter,
    mut cancelled: oneshot::Receiver<()>,
    mut trace: telemetry::LlmTrace,
) {
    let attempt = trace.attempt();
    let opened = tokio::select! {
        biased;
        _ = &mut cancelled => None,
        opened = llm.chat(trace.cx(), &req) => Some(opened),
    };
    let mut stream = match opened {
        None => {
            let _ = events
                .send(Report::Core(AgentRunEvent::LlmFailed { request, error: "cancelled".into() }))
                .await;
            return;
        }
        Some(Err(error)) => {
            trace.end_attempt(attempt, Some(&error));
            trace.fail(&error);
            let _ = events.send(Report::LlmFailed { request, error }).await;
            return;
        }
        Some(Ok(stream)) => stream,
    };
    loop {
        let next = tokio::select! {
            biased;
            _ = &mut cancelled => None,
            next = stream.next() => Some(next),
        };
        let report = match next {
            None => {
                let deadline = clock.now().saturating_add(Duration::from_secs(5));
                let _ = stream.close(deadline).await;
                Report::Core(AgentRunEvent::LlmFailed { request, error: "cancelled".into() })
            }
            Some(Ok(Some(chunk))) => {
                trace.observe(&chunk);
                if events
                    .send(Report::Core(AgentRunEvent::LlmChunk { request, chunk }))
                    .await
                    .is_err()
                {
                    return;
                }
                continue;
            }
            Some(Ok(None)) => {
                trace.end_attempt(attempt, None);
                trace.end(false, None);
                Report::Core(AgentRunEvent::LlmEnded { request })
            }
            Some(Err(error)) => {
                trace.end_attempt(attempt, Some(&error));
                trace.fail(&error);
                Report::LlmFailed { request, error }
            }
        };
        let _ = events.send(report).await;
        return;
    }
}
