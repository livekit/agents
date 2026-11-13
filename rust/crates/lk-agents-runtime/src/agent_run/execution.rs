use super::*;
use crate::host::HostInvoker;
use crate::telemetry;
use lk_agents_abi::agent::run::{RunInputRequirement, RunOutcome, RunResult};
use lk_agents_abi::agent::{FunctionToolInfo, OutputCandidate, ToolArgumentError, ToolTemplate};
use lk_agents_abi::errors::{ApiError, HostError};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::{ChatContext, SpawnerExt, ToolSpec};
use serde_json::{Map, Value};

#[derive(Clone)]
pub(crate) struct Reporter {
    pub id: String,
    pub tx: mpsc::Sender<(String, Report)>,
}

impl Reporter {
    /// Reports to the run's driver. A report nobody reads is dropped: the driver left with the
    /// run, and dropping the run cancels its requests and callbacks.
    async fn send(&self, report: Report) {
        drop(self.tx.send((self.id.clone(), report)).await);
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

impl Work {
    /// What the core is told when the work's callback ended without reporting.
    fn lost(self) -> Report {
        let error = || HostError::new("the host callback ended without reporting");
        let event = match self {
            Work::Checkpoint(checkpoint) => AgentRunEvent::HostCheckpointDone {
                checkpoint,
                should_yield: Err(error()),
                tools: None,
            },
            Work::Render(render) => {
                AgentRunEvent::InstructionsRendered { render, result: Err(error()) }
            }
            Work::Call(call) => AgentRunEvent::CallLost { call, outcome: ToolOutcome::Internal },
            Work::Validation(validation) => {
                let verdict = OutputVerdict::Failed(error().message);
                return Report::Validated { validation, verdict, value: None, serialized: None };
            }
        };
        Report::Core(event)
    }
}

/// One model request in flight: its stop, and the task streaming it.
pub(crate) struct Request {
    stop: tokio_util::sync::DropGuard,
    _task: lk_agents_abi::OwnedTask,
}

pub(crate) struct Execution {
    pub invoker: HostInvoker,
    /// The invocation's `agent_run` span, ended with its result; dropped (cancelled) with an
    /// abandoned invocation. Its tool calls nest under it.
    span: Option<telemetry::ScopedSpan<telemetry::AgentRun>>,
    /// The span's context, kept past its end for work still settling.
    cx: telemetry::Cx,
    pub(super) setup: RunSetup,
    spawner: Arc<dyn Spawner>,
    events_tx: Reporter,
    /// Each model request in flight; a request dropped from here stops too.
    pub requests: BTreeMap<RequestId, Request>,
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
        spawner: Arc<dyn Spawner>,
        events_tx: Reporter,
        span: telemetry::ScopedSpan<telemetry::AgentRun>,
    ) -> Self {
        Self {
            invoker: HostInvoker::new(setup.entry.clone()),
            cx: span.cx().clone(),
            span: Some(span),
            setup,
            spawner,
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
                if let Some(request) = self.requests.get(&request) {
                    request.stop.token().cancel();
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
                log::error!(target: lk_agents_abi::LOG_TARGET, "agent run cleanup missed its deadline: {}", pending.join(", "));
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
            let (should_yield, tools) = checkpoint_host(host.as_ref(), &cx, &inv, list_tools).await;
            let event = AgentRunEvent::HostCheckpointDone { checkpoint, should_yield, tools };
            events.send(Report::Core(event)).await;
        });
    }

    fn start_llm_request(
        &mut self,
        request: RequestId,
        chat_ctx: ChatContext,
        tools: Vec<ToolSpec>,
    ) {
        let cancelled = CancellationToken::new();
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
        let lost = events.clone();
        let stream = stream_request(request, req, llm, clock, events, cancelled.clone(), trace);
        // A stream whose task ended without its end or failure (it panicked) reports it failed.
        let task = self.spawner.spawn_settled(stream, move |outcome| async move {
            if !matches!(outcome, lk_agents_abi::TaskOutcome::Completed) {
                let error = ApiError::new("the model request ended without reporting").into();
                lost.send(Report::LlmFailed { request, error }).await;
            }
        });
        self.requests.insert(request, Request { stop: cancelled.drop_guard(), _task: task });
    }

    fn render_instructions(&mut self, render: RenderId) {
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        self.invoke(Work::Render(render), None, None, move |inv, events| async move {
            let result = host.render_instructions(&cx, &inv).await;
            let event = AgentRunEvent::InstructionsRendered { render, result };
            events.send(Report::Core(event)).await;
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
            events.send(prepare_report(call, resource, prepared)).await;
        });
    }

    /// Renders the tool's read-back; `None` when the prepared arguments were released.
    fn render_read_back(&mut self, call: CallId, prepared: ResourceId) -> Option<()> {
        let args = self.prepared.get(&prepared).cloned()?;
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        let function_call = self.calls.get(&call).cloned();
        self.invoke(Work::Call(call), function_call, None, move |inv, events| async move {
            let result = host.read_back(&cx, &inv, &args).await;
            let event = AgentRunEvent::ReadBackRendered { call, result };
            events.send(Report::Core(event)).await;
        });
        Some(())
    }

    fn render_tool_handling(&mut self, call: CallId, template: ToolTemplate) {
        let host = self.setup.host.clone();
        let cx = self.cx.clone();
        let function_call = self.calls.get(&call).cloned();
        self.invoke(Work::Call(call), function_call, None, move |inv, events| async move {
            let text = host.render_template(&cx, &inv, template.key(), template).await;
            let event = AgentRunEvent::ToolHandlingTemplateRendered { call, text };
            events.send(Report::Core(event)).await;
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
            events.send(Report::Validated { validation, verdict, value, serialized }).await;
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
                self.failure = Some(error.clone());
                (AgentRunEvent::LlmFailed { request, error }, None)
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

    /// One host callback through the invoker, under the work it serves. A callback that never
    /// reports (it panicked) reports its work lost, so the run settles it.
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
        let lost = self.events_tx.clone();
        let lost = async move {
            lost.send(work.lost()).await;
        };
        let operation = self.invoker.invoke(
            function_call,
            approval,
            &*self.spawner,
            move |inv| call(inv, events),
            lost,
        );
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
) -> (Result<bool, HostError>, Option<Vec<FunctionToolInfo>>) {
    let should_yield = match host.checkpoint(cx, inv).await {
        Ok(should_yield) => should_yield,
        Err(error) => return (Err(error), None),
    };
    if !list_tools || should_yield {
        return (Ok(should_yield), None);
    }
    match host.list_tools(cx, inv).await {
        Ok(tools) => (Ok(should_yield), Some(tools)),
        Err(error) => (Err(error), None),
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
        Err(e) => PrepareFailure::Host(e),
    };
    Report::Core(AgentRunEvent::ToolPrepareFailed { call, failure })
}

fn log_repaired_arguments(call_id: &str, error: &str, repaired: Value) {
    let repaired = match repaired {
        Value::String(s) => s,
        other => other.to_string(),
    };
    log::warn!(
        target: lk_agents_abi::LOG_TARGET,
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
            Ok(ran) => crate::telemetry::end_function_tool(
                span,
                &function_call,
                started,
                &ran.report.outcome,
            ),
            Err(error) => telemetry::end(span, error),
        }
        let event = match ran {
            Ok(lk_agents_abi::agent::ToolCompletion { report, extra, usage }) => {
                events.send(Report::Core(AgentRunEvent::ToolCallUpdated { call, extra })).await;
                if let Some(usage) = usage {
                    events.send(Report::Core(AgentRunEvent::DelegateUsage { usage })).await;
                }
                AgentRunEvent::ToolSettled { call, outcome: report.outcome }
            }
            Err(error) => AgentRunEvent::ToolLaunchFailed { call, error },
        };
        events.send(Report::Core(event)).await;
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
    cancelled: CancellationToken,
    mut trace: telemetry::LlmTrace,
) {
    let deadline = req
        .conn_options
        .total_timeout
        .map(|seconds| clock.now().saturating_add(Duration::from_secs_f64(seconds)));
    let attempt = trace.attempt();
    let opened = tokio::select! {
        biased;
        () = cancelled.cancelled() => None,
        opened = lk_agents_inference::llm::chat(llm.clone(), trace.cx(), &req, clock.clone(), None) => Some(opened),
    };
    let mut stream = match opened {
        None => {
            let error = ApiError::cancelled("the request was cancelled").into();
            events.send(Report::Core(AgentRunEvent::LlmFailed { request, error })).await;
            return;
        }
        Some(Err(error)) => {
            trace.end_attempt(attempt, Some(&error));
            trace.fail(&error);
            events.send(Report::LlmFailed { request, error }).await;
            return;
        }
        Some(Ok(stream)) => stream,
    };
    let report = loop {
        let next = tokio::select! {
            biased;
            () = cancelled.cancelled() => None,
            next = stream.next() => Some(next),
        };
        match next {
            None => {
                let deadline = deadline.unwrap_or_else(|| {
                    clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)
                });
                let closed = stream.close(deadline).await;
                if closed != lk_agents_abi::Closed::Clean {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, request = request.0, outcome:? = closed; "cancelled model request cleanup incomplete");
                }
                let error = ApiError::cancelled("the request was cancelled").into();
                break Report::Core(AgentRunEvent::LlmFailed { request, error });
            }
            Some(Ok(Some(chunk))) => {
                trace.observe(&chunk);
                events.send(Report::Core(AgentRunEvent::LlmChunk { request, chunk })).await;
                continue;
            }
            Some(Ok(None)) => {
                trace.end_attempt(attempt, None);
                trace.end(Default::default());
                break Report::Core(AgentRunEvent::LlmEnded { request });
            }
            Some(Err(error)) => {
                trace.end_attempt(attempt, Some(&error));
                trace.fail(&error);
                break Report::LlmFailed { request, error };
            }
        }
    };
    events.send(report).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{Closed, LlmChunk, LlmStream, StreamOutput, VirtualClock};

    struct Model {
        clock: Arc<VirtualClock>,
        stop: CancellationToken,
        closed_at: Arc<parking_lot::Mutex<Option<Ts>>>,
    }

    impl Llm for Model {
        fn label(&self) -> String {
            "test".into()
        }

        fn chat<'a>(
            &'a self,
            _: &telemetry::Cx,
            _: &'a ChatRequest,
        ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
            Box::pin(async move {
                Ok(Box::new(Waiting {
                    clock: self.clock.clone(),
                    stop: self.stop.clone(),
                    closed_at: self.closed_at.clone(),
                }) as LlmStream)
            })
        }
    }

    struct Waiting {
        clock: Arc<VirtualClock>,
        stop: CancellationToken,
        closed_at: Arc<parking_lot::Mutex<Option<Ts>>>,
    }

    impl StreamOutput<LlmChunk> for Waiting {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
            Box::pin(async move {
                self.clock.advance(Duration::from_secs(2));
                self.stop.cancel();
                std::future::pending().await
            })
        }

        fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
            Box::pin(async move {
                *self.closed_at.lock() = Some(deadline);
                Closed::Clean
            })
        }
    }

    impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Waiting {}

    #[tokio::test]
    async fn cancellation_preserves_the_request_deadline_or_uses_the_shared_cleanup_allowance() {
        for total_timeout in [Some(17.0), None] {
            let clock = Arc::new(VirtualClock::new());
            let stop = CancellationToken::new();
            let closed_at = Arc::new(parking_lot::Mutex::new(None));
            let model = Arc::new(Model {
                clock: clock.clone(),
                stop: stop.clone(),
                closed_at: closed_at.clone(),
            });
            let req = ChatRequest {
                conn_options: lk_agents_abi::ApiConnectOptions {
                    total_timeout,
                    ..Default::default()
                },
                ..Default::default()
            };
            let trace = telemetry::LlmTrace::start(&telemetry::Cx::NONE, model.as_ref(), &req);
            let (tx, mut rx) = mpsc::channel(1);
            let events = Reporter { id: "run".into(), tx };
            stream_request(RequestId(1), req, model, clock.clone(), events, stop, trace).await;
            let expected = total_timeout
                .map(|seconds| Ts::ZERO.saturating_add(Duration::from_secs_f64(seconds)))
                .unwrap_or_else(|| {
                    clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)
                });
            assert_eq!(*closed_at.lock(), Some(expected));
            assert!(matches!(
                rx.recv().await,
                Some((_, Report::Core(AgentRunEvent::LlmFailed { .. })))
            ));
        }
    }
}
