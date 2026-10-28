//! One agent invocation executed: the core's `AgentRun` stepped one event at a time, its model
//! requests run against the agent's `Llm`, its host callbacks through [`HostInvoker`], its
//! deadline on the clock. Updates (stream events, then the settled result) reach the consumer
//! through a bounded queue the driver waits on; nothing is dropped.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::agent::{AgentHost, Invocation, PreparedArguments, ValidatedOutput};
use lk_agents_abi::llm::tool_executor::Approval;
use lk_agents_abi::llm::tool_executor::INTERNAL_ERROR_OUTPUT;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{ApiConnectOptions, ChatRequest, Clock, FunctionCall, Llm, Ts};
use lk_agents_abi::agent::run::{Output, RunSettled, RunUpdate, ValidationId};
use lk_agents_core::agent_run::{CallId, Effect, Event, OutputVerdict, RenderId, RequestId};
use lk_agents_abi::agent::AgentDefinition;
use lk_agents_core::{AgentError, AgentRun, RunInput, approval};
use tokio::sync::{mpsc, oneshot};

use crate::host::HostInvoker;

/// Events the work in flight may report ahead of the driver.
const EVENT_QUEUE: usize = 64;

/// What a run executes against.
#[derive(Clone)]
pub struct Ports {
    pub llm: Arc<dyn Llm>,
    pub host: Arc<dyn AgentHost>,
    pub clock: Arc<dyn Clock>,
    pub conn_options: ApiConnectOptions,
    /// Captured at the `run()` entry.
    pub entry: Invocation,
}

/// The running invocation. Dropping it cancels the run; its work still settles.
pub struct AgentRunHandle {
    cancel: Option<oneshot::Sender<()>>,
}

impl AgentRunHandle {
    /// Requests cancellation; idempotent. The run settles once its work reported back.
    pub fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

impl Drop for AgentRunHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Starts one invocation on `runtime`; its updates arrive on the receiver.
pub fn start(
    runtime: &tokio::runtime::Handle,
    agent: AgentDefinition,
    input: RunInput,
    ports: Ports,
    updates: usize,
) -> Result<(AgentRunHandle, mpsc::Receiver<RunUpdate>), AgentError> {
    let run = AgentRun::new(agent, input)?;
    let (cancel, cancelled) = oneshot::channel();
    let (update_tx, update_rx) = mpsc::channel(updates.max(1));
    let (events_tx, events) = mpsc::channel(EVENT_QUEUE);
    let driver = Driver {
        run,
        invoker: HostInvoker::new(ports.entry.clone()),
        ports,
        runtime: runtime.clone(),
        events_tx,
        updates: update_tx,
        requests: BTreeMap::new(),
        operations: BTreeMap::new(),
        calls: BTreeMap::new(),
        prepared: BTreeMap::new(),
        outputs: BTreeMap::new(),
        next_resource: 0,
        finished: false,
    };
    runtime.spawn(driver.drive(events, cancelled));
    Ok((AgentRunHandle { cancel: Some(cancel) }, update_rx))
}

/// What work in flight reports to the driver: a core event, with the host value it returned when
/// the runtime keeps one for the core.
enum Report {
    Core(Event),
    Prepared { call: CallId, resource: ResourceId, prepared: PreparedArguments },
    Validated { validation: ValidationId, verdict: OutputVerdict, value: Option<HostRef> },
}

/// The host callback in flight for a core correlation id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Work {
    Call(CallId),
    Render(RenderId),
    Validation(ValidationId),
}

struct Driver {
    run: AgentRun,
    invoker: HostInvoker,
    ports: Ports,
    runtime: tokio::runtime::Handle,
    events_tx: mpsc::Sender<Report>,
    updates: mpsc::Sender<RunUpdate>,
    /// Model requests in flight, with their cancel signal until it is sent.
    requests: BTreeMap<RequestId, Option<oneshot::Sender<()>>>,
    /// The host callback in flight for each piece of work.
    operations: BTreeMap<Work, u64>,
    /// Each tool call as the model made it, for the invocations of its later callbacks.
    calls: BTreeMap<CallId, FunctionCall>,
    /// Prepared arguments the core names by id.
    prepared: BTreeMap<ResourceId, HostRef>,
    /// Validated output values the core names by validation.
    outputs: BTreeMap<ValidationId, HostRef>,
    next_resource: u64,
    finished: bool,
}

impl Driver {
    async fn drive(
        mut self,
        mut events: mpsc::Receiver<Report>,
        mut cancelled: oneshot::Receiver<()>,
    ) {
        let mut cancel_seen = false;
        let mut effects = self.run.handle(Event::Start, self.ports.clock.now());
        loop {
            for effect in effects.drain(..) {
                self.execute(effect).await;
            }
            if self.finished && self.invoker.is_idle() && self.requests.is_empty() {
                return;
            }
            let wake = self.run.wake_at();
            let clock = self.ports.clock.clone();
            let event = tokio::select! {
                biased;
                // A due wake precedes another input at the same instant.
                () = wait(clock.as_ref(), wake) => Event::Wake,
                _ = &mut cancelled, if !cancel_seen => {
                    cancel_seen = true;
                    Event::Cancel
                }
                Some(report) = events.recv() => self.adopt(report),
                _ = self.invoker.settled() => continue,
            };
            if let Event::LlmEnded { request } | Event::LlmFailed { request, .. } = &event {
                self.requests.remove(request);
            }
            effects = self.run.handle(event, self.ports.clock.now());
        }
    }

    async fn execute(&mut self, effect: Effect) {
        match effect {
            Effect::LlmRequestStart { request, chat_ctx, tools, .. } => {
                let (cancel, cancelled) = oneshot::channel();
                self.requests.insert(request, Some(cancel));
                let req = ChatRequest {
                    chat_ctx,
                    tools,
                    conn_options: self.ports.conn_options,
                    ..Default::default()
                };
                let llm = self.ports.llm.clone();
                let clock = self.ports.clock.clone();
                let events = self.events_tx.clone();
                self.runtime.spawn(stream_request(request, req, llm, clock, events, cancelled));
            }
            Effect::LlmRequestCancel { request } => {
                if let Some(cancel) = self.requests.get_mut(&request).and_then(Option::take) {
                    let _ = cancel.send(());
                }
            }
            Effect::InstructionsRender { render } => {
                let host = self.ports.host.clone();
                self.invoke(Work::Render(render), None, None, move |inv, events| async move {
                    let result = host.render_instructions(&inv).await.map_err(|e| e.to_string());
                    let _ = events
                        .send(Report::Core(Event::InstructionsRendered { render, result }))
                        .await;
                });
            }
            Effect::ToolPrepare { call, arguments, function_call } => {
                self.calls.insert(call, function_call.clone());
                let host = self.ports.host.clone();
                let resource = self.mint();
                self.invoke(Work::Call(call), Some(function_call), None, move |inv, events| async move {
                    let report = match host.prepare_tool(&inv, arguments).await {
                        Ok(Ok(prepared)) => Report::Prepared { call, resource, prepared },
                        Ok(Err(e)) => Report::Core(Event::ToolPrepareFailed { call, error: e.message }),
                        Err(e) => {
                            log::error!(target: "livekit.agents", "preparing a tool call failed: {e}");
                            let error = INTERNAL_ERROR_OUTPUT.to_owned();
                            Report::Core(Event::ToolPrepareFailed { call, error })
                        }
                    };
                    let _ = events.send(report).await;
                });
            }
            Effect::ReadBackRender { call, prepared, .. } => {
                let Some(args) = self.prepared.get(&prepared).cloned() else { return };
                let host = self.ports.host.clone();
                let function_call = self.calls.get(&call).cloned();
                self.invoke(Work::Call(call), function_call, None, move |inv, events| async move {
                    let result = host.read_back(&inv, &args).await.map_err(|e| e.to_string());
                    let _ =
                        events.send(Report::Core(Event::ReadBackRendered { call, result })).await;
                });
            }
            Effect::ToolExecute { call, prepared, function_call } => {
                let Some(args) = self.prepared.remove(&prepared) else { return };
                let host = self.ports.host.clone();
                let approval = approval::mark(&function_call);
                self.invoke(
                    Work::Call(call),
                    Some(function_call),
                    approval,
                    move |inv, events| async move {
                        let event = match host.run_tool(&inv, &args).await {
                            Ok(report) => Event::ToolSettled { call, outcome: report.outcome },
                            Err(e) => Event::ToolLaunchFailed { call, error: e.to_string() },
                        };
                        let _ = events.send(Report::Core(event)).await;
                    },
                );
            }
            Effect::ToolCancel { call } => self.cancel(Work::Call(call)),
            Effect::InstructionsCancel { render } => self.cancel(Work::Render(render)),
            Effect::OutputValidateCancel { validation } => {
                self.cancel(Work::Validation(validation))
            }
            Effect::ToolArgumentsRelease { prepared } => {
                self.prepared.remove(&prepared);
            }
            Effect::OutputValidate { validation, candidate } => {
                let host = self.ports.host.clone();
                self.invoke(
                    Work::Validation(validation),
                    None,
                    None,
                    move |inv, events| async move {
                        let (verdict, value) = match host.validate_output(&inv, candidate).await {
                            Ok(ValidatedOutput::Accepted { text, value }) => {
                                (OutputVerdict::Accepted { text }, value)
                            }
                            Ok(ValidatedOutput::Retry(reason)) => {
                                (OutputVerdict::Retry(reason), None)
                            }
                            Ok(ValidatedOutput::Failed(error)) => {
                                (OutputVerdict::Failed(error), None)
                            }
                            Err(e) => (OutputVerdict::Failed(e.to_string()), None),
                        };
                        let _ = events.send(Report::Validated { validation, verdict, value }).await;
                    },
                );
            }
            Effect::Emit(event) => {
                let _ = self.updates.send(RunUpdate::Stream(event)).await;
            }
            Effect::ArgumentsRepaired { call_id, error, repaired } => {
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
            Effect::IncompleteCleanup { pending } => {
                log::error!(target: "livekit.agents", "agent run cleanup missed its deadline: {}", pending.join(", "));
                let _ = self.updates.send(RunUpdate::IncompleteCleanup(pending)).await;
            }
            Effect::Finished(result) => {
                self.finished = true;
                let settled = result.map(|result| {
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
                let _ = self.updates.send(RunUpdate::Finished(settled)).await;
            }
        }
    }

    /// Keeps the host value a report carries and hands the core its event.
    fn adopt(&mut self, report: Report) -> Event {
        match report {
            Report::Core(event) => event,
            Report::Prepared { call, resource, prepared } => {
                self.prepared.insert(resource, prepared.prepared);
                Event::ToolPrepared { call, prepared: resource, identity: prepared.identity }
            }
            Report::Validated { validation, verdict, value } => {
                if let Some(value) = value {
                    self.outputs.insert(validation, value);
                }
                Event::OutputValidated { validation, verdict }
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
        F: FnOnce(Invocation, mpsc::Sender<Report>) -> Fut,
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
            self.ports.host.cancel(invocation);
        }
    }

    fn mint(&mut self) -> ResourceId {
        self.next_resource += 1;
        ResourceId(self.next_resource)
    }
}

/// Waits for the core's deadline; pending without one.
async fn wait(clock: &dyn Clock, at: Option<Ts>) {
    match at {
        Some(at) => clock.wait_until(at).await,
        None => std::future::pending().await,
    }
}

/// One model request: chunks reported as they arrive, then its end or failure. A cancel closes
/// the stream under the request's deadline and reports the request ended.
async fn stream_request(
    request: RequestId,
    req: ChatRequest,
    llm: Arc<dyn Llm>,
    clock: Arc<dyn Clock>,
    events: mpsc::Sender<Report>,
    mut cancelled: oneshot::Receiver<()>,
) {
    let opened = tokio::select! {
        biased;
        _ = &mut cancelled => None,
        opened = llm.chat(&req) => Some(opened),
    };
    let mut stream = match opened {
        None => {
            let _ = events
                .send(Report::Core(Event::LlmFailed { request, error: "cancelled".into() }))
                .await;
            return;
        }
        Some(Err(e)) => {
            let _ =
                events.send(Report::Core(Event::LlmFailed { request, error: e.to_string() })).await;
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
        let event = match next {
            None => {
                let deadline = clock.now().saturating_add(Duration::from_secs(5));
                let _ = stream.close(deadline).await;
                Event::LlmFailed { request, error: "cancelled".into() }
            }
            Some(Ok(Some(chunk))) => {
                if events.send(Report::Core(Event::LlmChunk { request, chunk })).await.is_err() {
                    return;
                }
                continue;
            }
            Some(Ok(None)) => Event::LlmEnded { request },
            Some(Err(e)) => Event::LlmFailed { request, error: e.to_string() },
        };
        let _ = events.send(Report::Core(event)).await;
        return;
    }
}

mod runner;
#[cfg(test)]
mod tests;

pub use runner::{Runner, Updates};
