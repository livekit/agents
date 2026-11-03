//! I/O for one invocation, shared by the direct runner and TextSession. Coordinators own
//! decisions and update delivery; this executor owns host work and provider requests.
use super::*;
use crate::host::HostInvoker;
use lk_agents_abi::agent::run::RunOutcome;
use lk_agents_abi::telemetry::spans::text::{AgentRunEnd, RunStatus};
use lk_agents_inference::telemetry::LlmTrace;
use lk_agents_net::telemetry::Failure;

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
    span: Option<ScopedSpan<AgentRunSpan>>,
    /// The span's context, kept past its end for work still settling.
    cx: lk_agents_net::telemetry::Cx,
    pub(super) setup: RunSetup,
    runtime: tokio::runtime::Handle,
    events_tx: Reporter,
    pub requests: BTreeMap<RequestId, Option<oneshot::Sender<()>>>,
    operations: BTreeMap<Work, u64>,
    calls: BTreeMap<CallId, FunctionCall>,
    prepared: BTreeMap<ResourceId, HostRef>,
    pub outputs: BTreeMap<ValidationId, HostRef>,
    next_resource: u64,
    /// The provider error of the last failed model request.
    pub failure: Option<ProviderError>,
}

impl Execution {
    pub fn new(
        mut setup: RunSetup,
        runtime: tokio::runtime::Handle,
        events_tx: Reporter,
        span: ScopedSpan<AgentRunSpan>,
    ) -> Self {
        setup.entry.span = span.cx().span();
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
                let host = self.setup.host.clone();
                self.invoke(
                    Work::Checkpoint(checkpoint),
                    None,
                    None,
                    move |inv, events| async move {
                        let mut tools = None;
                        let result = async {
                            let yielded = host.checkpoint(&inv).await?;
                            if list_tools && !yielded {
                                tools = Some(host.list_tools(&inv).await?);
                            }
                            Ok::<_, lk_agents_abi::errors::HostError>(yielded)
                        }
                        .await
                        .map_err(|error| error.to_string());
                        let _ = events
                            .send(Report::Core(AgentRunEvent::HostCheckpointDone {
                                checkpoint,
                                result,
                                tools,
                            }))
                            .await;
                    },
                );
            }
            AgentRunEffect::HostCheckpointCancel { checkpoint } => {
                self.cancel(Work::Checkpoint(checkpoint))
            }
            AgentRunEffect::LlmRequestStart { request, chat_ctx, tools, .. } => {
                let (cancel, cancelled) = oneshot::channel();
                self.requests.insert(request, Some(cancel));
                let mut req = ChatRequest {
                    tools_host: self.setup.tools_host.clone(),
                    chat_ctx,
                    tools,
                    conn_options: self.setup.conn_options,
                    ..Default::default()
                };
                let llm = self.setup.llm.clone();
                let clock = self.setup.clock.clone();
                let events = self.events_tx.clone();
                let trace = LlmTrace::start(&self.cx, llm.as_ref(), &mut req);
                self.runtime
                    .spawn(stream_request(request, req, llm, clock, events, cancelled, trace));
            }
            AgentRunEffect::LlmRequestCancel { request } => {
                if let Some(cancel) = self.requests.get_mut(&request).and_then(Option::take) {
                    let _ = cancel.send(());
                }
            }
            AgentRunEffect::RenderInstructions { render } => {
                let host = self.setup.host.clone();
                self.invoke(Work::Render(render), None, None, move |inv, events| async move {
                    let result = host.render_instructions(&inv).await.map_err(|e| e.to_string());
                    let _ = events
                        .send(Report::Core(AgentRunEvent::InstructionsRendered { render, result }))
                        .await;
                });
            }
            AgentRunEffect::ToolPrepare { call, arguments, function_call } => {
                self.calls.insert(call, function_call.clone());
                let host = self.setup.host.clone();
                let resource = self.mint();
                self.invoke(Work::Call(call), Some(function_call), None, move |inv, events| async move {
                    let report = match host.prepare_tool(&inv, arguments).await {
                        Ok(Ok(prepared)) => Report::Prepared { call, resource, prepared },
                        Ok(Err(e)) => {
                            let failure = PrepareFailure::Refused(e.message);
                            Report::Core(AgentRunEvent::ToolPrepareFailed { call, failure })
                        }
                        Err(e) => {
                            log::error!(target: "livekit.agents", "preparing a tool call failed: {e}");
                            let failure = PrepareFailure::Internal;
                            Report::Core(AgentRunEvent::ToolPrepareFailed { call, failure })
                        }
                    };
                    let _ = events.send(report).await;
                });
            }
            AgentRunEffect::ReadBackRender { call, prepared, .. } => {
                let args = self.prepared.get(&prepared).cloned()?;
                let host = self.setup.host.clone();
                let function_call = self.calls.get(&call).cloned();
                self.invoke(Work::Call(call), function_call, None, move |inv, events| async move {
                    let result = host.read_back(&inv, &args).await.map_err(|e| e.to_string());
                    let _ = events
                        .send(Report::Core(AgentRunEvent::ReadBackRendered { call, result }))
                        .await;
                });
            }
            AgentRunEffect::RenderToolHandlingTemplate { call, template } => {
                let host = self.setup.host.clone();
                let function_call = self.calls.get(&call).cloned();
                self.invoke(
                    Work::Call(call),
                    function_call,
                    None,
                    move |inv, events| async move {
                        let text = host.render_template(&inv, template).await.unwrap_or_else(|e| {
                            log::error!(target: "livekit.agents", "a tool_handling template failed: {e}");
                            None
                        });
                        let _ = events
                            .send(Report::Core(AgentRunEvent::ToolHandlingTemplateRendered { call, text }))
                            .await;
                    },
                );
            }
            AgentRunEffect::ToolExecute { call, prepared, function_call, approval } => {
                let args = self.prepared.remove(&prepared)?;
                let host = self.setup.host.clone();
                let span = crate::telemetry::function_tool(
                    &self.cx,
                    &function_call,
                    &self.setup.entry.agent_name,
                    None,
                );
                let started = span.cx().tracer().now_nanos();
                self.invoke(
                    Work::Call(call),
                    Some(function_call.clone()),
                    approval,
                    move |mut inv, events| async move {
                        inv.span = span.cx().span();
                        let ran = host.run_tool(&inv, &args).await;
                        if let Ok((report, _)) = &ran {
                            crate::telemetry::end_function_tool(
                                span,
                                &function_call,
                                started,
                                &report.outcome,
                            );
                        } else if let Err(error) = &ran {
                            span.fail(error);
                        }
                        let event = match ran {
                            Ok((report, extra)) => {
                                let _ = events
                                    .send(Report::Core(AgentRunEvent::ToolCallUpdated {
                                        call,
                                        extra,
                                    }))
                                    .await;
                                AgentRunEvent::ToolSettled { call, outcome: report.outcome }
                            }
                            Err(e) => {
                                AgentRunEvent::ToolLaunchFailed { call, error: e.to_string() }
                            }
                        };
                        let _ = events.send(Report::Core(event)).await;
                    },
                );
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
                let host = self.setup.host.clone();
                self.invoke(
                    Work::Validation(validation),
                    None,
                    None,
                    move |inv, events| async move {
                        let (verdict, value, serialized) = crate::output_validation::project(
                            host.validate_output(&inv, candidate).await,
                        );
                        let _ = events
                            .send(Report::Validated { validation, verdict, value, serialized })
                            .await;
                    },
                );
            }
            AgentRunEffect::Emit(event) => {
                return Some(RunUpdate::Stream(event));
            }
            AgentRunEffect::ArgumentsRepaired { call_id, error, repaired } => {
                let repaired = match repaired {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                };
                log::warn!(
                    target: "livekit.agents",
                    call_id = call_id.as_str(),
                    "lk.pii.repaired" = repaired.as_str(),
                    "lk.pii.error" = error.as_str();
                    "repaired malformed function-call JSON arguments"
                );
            }
            AgentRunEffect::IncompleteCleanup { pending } => {
                log::error!(target: "livekit.agents", "agent run cleanup missed its deadline: {}", pending.join(", "));
                return Some(RunUpdate::IncompleteCleanup(pending));
            }
            AgentRunEffect::Finished { result, input_requirements } => {
                let settled = (*result).map(|result| {
                    let output = match result.outcome {
                        lk_agents_abi::agent::run::RunOutcome::Completed {
                            output: Output::Validated(v),
                        } => self.outputs.remove(&v),
                        _ => None,
                    };
                    RunSettled { result, output }
                });
                self.outputs.clear();
                self.prepared.clear();
                self.end_span(&settled);
                let update = match settled {
                    Ok(settled) => RunUpdate::Finished(Box::new(settled)),
                    Err(error) => {
                        let provider_error = if matches!(error, RunError::Model(_)) {
                            self.failure.take()
                        } else {
                            None
                        };
                        RunUpdate::Failed { error, input_requirements, provider_error }
                    }
                };
                return Some(update);
            }
        }
        None
    }

    /// Ends the invocation's span with its result.
    fn end_span(&mut self, settled: &Result<RunSettled, RunError>) {
        let Some(span) = self.span.take() else { return };
        let status = match settled {
            Ok(settled) => match settled.result.outcome {
                RunOutcome::InputRequired { .. } => RunStatus::InputRequired,
                _ => RunStatus::Completed,
            },
            Err(error) => {
                span.fail(&Failure { kind: "run_error", message: error.to_string() });
                RunStatus::Failed
            }
        };
        span.end(AgentRunEnd { status });
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

/// One model request: chunks reported as they arrive, then its end or failure. A cancel closes
/// the stream under the request's deadline and reports the request ended.
async fn stream_request(
    request: RequestId,
    req: ChatRequest,
    llm: Arc<dyn Llm>,
    clock: Arc<dyn Clock>,
    events: Reporter,
    mut cancelled: oneshot::Receiver<()>,
    mut trace: LlmTrace,
) {
    let attempt = trace.attempt();
    let opened = tokio::select! {
        biased;
        _ = &mut cancelled => None,
        opened = llm.chat(&req) => Some(opened),
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
