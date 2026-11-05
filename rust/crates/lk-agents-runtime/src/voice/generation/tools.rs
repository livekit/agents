//! `perform_tool_executions`: a step's calls, run as they arrive and answered in the model's order.

use super::ReplyExecution;
use crate::telemetry;
use crate::voice::session::SessionShared;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::FunctionCall;
use lk_agents_core::tool_calls::PrepareFailure;
use lk_agents_voice::agent_session::Answer;
use lk_agents_voice::generation::{GenEvent, ReplySource};
use lk_agents_voice::tool_executor::{
    HostedTool, MadeCall, StepEnd, StepNext, ToolExecution, ToolGo, ToolOutcome, ToolReport, answer,
};
use std::sync::Arc;

/// Calls go to the driver as they arrive (latency hides behind the generation's tail), which
/// starts or drops each. **Order is the contract**: the step pairs by index.
pub(crate) struct ToolStep {
    calls: Vec<Dispatched>,
    /// Each call is the speech's task: a call outlives the reply until the speech is cancelled.
    speeches: crate::voice::SpeechTasks,
}

struct Dispatched {
    call: FunctionCall,
    /// Answers `None` when the driver dropped the call unrun.
    rx: tokio::sync::oneshot::Receiver<Option<ToolExecution>>,
}

impl ToolStep {
    pub(crate) fn new(speeches: crate::voice::SpeechTasks) -> Self {
        ToolStep { calls: Vec::new(), speeches }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Hands the call to the driver and runs it as told, as one of the speech's tasks.
    pub(crate) fn dispatch(
        &mut self,
        shared: &Arc<SessionShared>,
        handle: &Arc<SpeechHandle>,
        ctx: &ReplyExecution,
        call: FunctionCall,
    ) {
        let (speech, step) = (handle.id(), ctx.plan.step);
        let spec = ctx.plan.request.tools.iter().find(|t| t.name == call.name).cloned();
        let description = spec.as_ref().map(|t| t.description.clone());
        let made = MadeCall {
            call: call.clone(),
            spec,
            forbidden: ctx.plan.tool_choice == Some(lk_agents_abi::ToolChoice::Forbidden),
            answering: ctx.plan.answering.clone(),
            agent: ctx.resources.toolset.agent.clone(),
        };
        let (go, first) = shared.open_call(Answer::tool_go);
        let made = GenEvent::ToolCallMade { speech, step, made: Box::new(made), go: go.id };
        shared.report_awaited(made, go);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let shared = Arc::clone(shared);
        let handle = Arc::clone(handle);
        let task = ctx.plan.task;
        let bodies = self.speeches.clone();
        let turn = ctx.span;
        self.speeches.spawn(speech, "tool call", async move {
            // Dropped unrun, or a loop gone before the start: owed nothing.
            let Ok(ToolGo::Start { call: id, item, host, first }) = first.await else {
                let _ = tx.send(None);
                return;
            };
            // The callbacks run with the host's object for the speech.
            // The callbacks run in the speech's turn; the body in its own `function_tool`.
            let host = host.map(|mut host| {
                host.invocation.speech = Some(handle.object());
                host
            });
            let host = host.map(|host| *host);
            let call = HostedCall { shared: &shared, speech, host, bodies, description, turn };
            let (execution, outcome) = call.run(*item, *first, &handle, task).await;
            // Recorded as it arrives; the step answers the calls in the model's order.
            let message = execution.ended_message();
            let (call_id, end) = (execution.call.call_id.clone(), execution.end);
            let call = Some(id);
            let ended = GenEvent::ToolCallEnded { speech, call, call_id, end, message, outcome };
            shared.report(ended);
            let _ = tx.send(Some(execution));
        });
        self.calls.push(Dispatched { call, rx });
    }
}

/// One call's host work through its agent's `AgentCallbacks`, each move decided by its reply's run
/// in the loop.
struct HostedCall<'a> {
    shared: &'a Arc<SessionShared>,
    speech: lk_agents_voice::agent_activity::scheduling::SpeechId,
    host: Option<HostedTool>,
    bodies: crate::voice::SpeechTasks,
    /// The tool's description, for its `function_tool`.
    description: Option<String>,
    /// The speech's turn: the callbacks run under it.
    turn: Option<telemetry::SpanContext>,
}

impl HostedCall<'_> {
    /// The context the callbacks run under: the speech's turn, else the session.
    fn turn_cx(&self) -> telemetry::Cx {
        self.shared.turn_cx(self.turn)
    }

    /// Prepares, reads back or runs the call as told, from its `first` move until it is
    /// answered: the execution and the outcome the run records.
    async fn run(
        &self,
        mut one: FunctionCall,
        first: ToolGo,
        handle: &SpeechHandle,
        task: bool,
    ) -> (ToolExecution, ToolOutcome) {
        let wall = || self.shared.time.wall();
        let mut prepared = None;
        let mut go = Ok(first);
        loop {
            match go {
                // A loop gone answers nothing: the call never ran.
                Err(_) | Ok(ToolGo::Gone) => {
                    let outcome = ToolOutcome::Cancelled;
                    return (answer(&one, outcome.clone().into(), wall()), outcome);
                }
                // Started once; a repeat is no move.
                Ok(ToolGo::Start { .. }) => {
                    let outcome = ToolOutcome::Internal;
                    return (answer(&one, outcome.clone().into(), wall()), outcome);
                }
                Ok(ToolGo::Prepare { call, arguments }) => {
                    let identity = self.prepare(arguments, &mut prepared).await;
                    let speech = self.speech;
                    go = self.ask(|go| GenEvent::ToolPrepared { speech, call, identity, go }).await;
                }
                Ok(ToolGo::ReadBack { call }) => {
                    let answer = self.read_back(prepared.take()).await;
                    let speech = self.speech;
                    go = self
                        .ask(|go| GenEvent::ReadBackRendered { speech, call, answer, go })
                        .await;
                }
                Ok(ToolGo::Execute { item }) => {
                    one = *item;
                    let (Some(h), Some(args)) = (&self.host, prepared.take()) else {
                        let outcome = ToolOutcome::Internal;
                        return (answer(&one, outcome.clone().into(), wall()), outcome);
                    };
                    // The speech waits on the body until it answers or detaches: awaiting the
                    // speech from inside it is refused meanwhile.
                    handle.tool_blocking(&h.invocation, &one.call_id, &one.name);
                    let report = run_body(self, h, &args, &one, task).await;
                    handle.tool_returned(&h.invocation, &one.call_id);
                    let outcome = report.outcome.clone();
                    return (answer(&one, report, wall()), outcome);
                }
                Ok(ToolGo::Settled { item, outcome }) => {
                    return (answer(&item, outcome.clone().into(), wall()), outcome);
                }
            }
        }
    }

    /// Binds and validates the call's arguments through the host, keeping the prepared value in
    /// `prepared`: the identity the approval keys on, or why it can't be prepared.
    async fn prepare(
        &self,
        arguments: serde_json::Map<String, serde_json::Value>,
        prepared: &mut Option<lk_agents_abi::HostRef>,
    ) -> Result<lk_agents_abi::agent::ArgumentIdentity, PrepareFailure> {
        let Some(h) = &self.host else { return Err(PrepareFailure::Internal) };
        match h.host.prepare_tool(&self.turn_cx(), &h.invocation, arguments).await {
            Ok(Ok(args)) => {
                *prepared = Some(args.prepared);
                Ok(args.identity)
            }
            Ok(Err(refused)) => Err(PrepareFailure::Refused(refused.message)),
            Err(e) => {
                log::error!(target: "livekit.agents", "preparing a tool call failed: {e}");
                Err(PrepareFailure::Internal)
            }
        }
    }

    /// The host's read-back of the prepared arguments, for the person to approve.
    async fn read_back(
        &self,
        prepared: Option<lk_agents_abi::HostRef>,
    ) -> Result<lk_agents_abi::agent::ReadBackAnswer, String> {
        let (Some(h), Some(args)) = (&self.host, prepared) else {
            return Err("no prepared arguments".to_owned());
        };
        h.host.read_back(&self.turn_cx(), &h.invocation, &args).await.map_err(|e| e.to_string())
    }

    /// Reports the event built for the answer's request id, and awaits the call's next move.
    async fn ask(
        &self,
        event: impl FnOnce(lk_agents_voice::agent_session::RequestId) -> GenEvent,
    ) -> Result<ToolGo, crate::voice::session::Unanswered> {
        let (call, next) = self.shared.open_call(Answer::tool_go);
        self.shared.report_awaited(event(call.id), call);
        next.await
    }
}

/// Runs the body on the prepared arguments, `call` carrying its approval marks. Resolves when the
/// tool returns, raises, is cancelled, or releases the step with its first `ctx.update()`; a host
/// that fails outside the tool's own error handling is answered as an internal error. Only a
/// `VoiceTask`'s reply (`task`) ends a task: an ordinary tool's `Completed` is its call's output,
/// whatever value the host handed with it.
async fn run_body(
    site: &HostedCall<'_>,
    h: &HostedTool,
    prepared: &lk_agents_abi::HostRef,
    call: &FunctionCall,
    task: bool,
) -> ToolReport {
    let (shared, speech) = (site.shared, site.speech);
    let description = site.description.as_deref();
    // The body runs in its `function_tool`, under the speech's turn.
    let span = site.turn.map(|turn| {
        let cx = shared.trace.clone().under::<telemetry::AgentTurn>(turn);
        crate::telemetry::function_tool(&cx, call, &h.invocation.agent_name, description)
    });
    let started = shared.tracer.now_nanos();
    let invocation = lk_agents_abi::agent::Invocation {
        function_call: Some(call.clone()),
        approval: lk_agents_core::approval::mark(call),
        ..h.invocation.clone()
    };
    let cx = span.as_ref().map_or(telemetry::Cx::NONE, |span| span.cx().clone());
    let progress = shared.tool_bodies.tool_body(&invocation);
    let (tx, rx) = tokio::sync::oneshot::channel();
    let guard = BodyGuard { progress: shared.tool_bodies.clone(), invocation: invocation.clone() };
    let body = ToolBody {
        host: h.host.clone(),
        invocation,
        prepared: prepared.clone(),
        cx,
        shared: shared.clone(),
        call: call.clone(),
        started,
        task,
        _guard: guard,
    };
    site.bodies.spawn(speech, "host tool body", body.run(span, progress, tx));
    rx.await.unwrap_or_else(|_| ToolOutcome::Cancelled.into())
}

/// Ends the body's progress tracking once the body has finished, on every path.
struct BodyGuard {
    progress: crate::voice::session::calls::ToolBodies,
    invocation: lk_agents_abi::agent::Invocation,
}

impl Drop for BodyGuard {
    fn drop(&mut self) {
        self.progress.end_tool_body(&self.invocation);
    }
}

/// A tool body running on the host, owned by the speech's tasks.
struct ToolBody {
    host: Arc<dyn lk_agents_abi::agent::AgentCallbacks>,
    invocation: lk_agents_abi::agent::Invocation,
    prepared: lk_agents_abi::HostRef,
    /// The body's `function_tool`, or no span.
    cx: telemetry::Cx,
    shared: Arc<SessionShared>,
    call: FunctionCall,
    started: u64,
    task: bool,
    /// Last, so it drops after everything else.
    _guard: BodyGuard,
}

impl ToolBody {
    /// Answers the step with the body's report, or with its first progress update; then the
    /// body's own report reaches the core as a tool update.
    async fn run(
        self,
        span: Option<telemetry::ScopedSpan<telemetry::FunctionTool>>,
        mut progress: tokio::sync::oneshot::Receiver<ToolReport>,
        tx: tokio::sync::oneshot::Sender<ToolReport>,
    ) {
        let execute = self.execute(span);
        tokio::pin!(execute);
        let first = tokio::select! { biased;
            report = &mut execute => Ok(report),
            update = &mut progress => Err(update),
        };
        match first {
            Ok(report) => {
                let _ = tx.send(report);
            }
            Err(Ok(update)) => {
                let _ = tx.send(update);
                let report = execute.await;
                self.send_update(report);
            }
            Err(Err(_)) => {
                let _ = tx.send(execute.await);
            }
        }
    }

    /// The host body owns its cleanup even after a semantic update retired the core's call.
    /// Cancellation remains cooperative: a refusing host still runs to the existing session
    /// cleanup bound rather than being silently dropped here.
    async fn execute(
        &self,
        span: Option<telemetry::ScopedSpan<telemetry::FunctionTool>>,
    ) -> ToolReport {
        let work = self.complete(span);
        tokio::pin!(work);
        let mut closing = self.shared.closing.clone();
        tokio::select! { biased;
            report = &mut work => report,
            // The watch guard is dropped inside: it must not live across the await below.
            _ = async { let _ = closing.wait_for(|closing| *closing).await; } => {
                if self.shared.tool_bodies.claim_cancellation(&self.invocation) {
                    self.host.cancel(&self.invocation);
                }
                work.await
            }
        }
    }

    /// The body's run, its completion validated for a task, and its span ended.
    async fn complete(
        &self,
        span: Option<telemetry::ScopedSpan<telemetry::FunctionTool>>,
    ) -> ToolReport {
        let (host, invocation, prepared) = (&self.host, &self.invocation, &self.prepared);
        let mut report = match host.run_tool(&self.cx, invocation, prepared).await {
            Ok((report, _)) => report,
            Err(error) => {
                log::error!("the tool's host failed outside its error handling: {error}");
                ToolOutcome::Internal.into()
            }
        };
        // Cancellation ends execution, including any later completion validation. A refusing
        // callable may eventually return successfully while it unwinds, but that must not start
        // a new callback under an operation already cancelled by the core.
        let tool_bodies = &self.shared.tool_bodies;
        if tool_bodies.cancellation_requested(invocation) || *self.shared.closing.borrow() {
            return ToolOutcome::Cancelled.into();
        }
        if lk_agents_voice::tool_executor::validates_completion(self.task, &report) {
            report = host
                .validate_tool_completion(&self.cx, invocation, prepared, report)
                .await
                .unwrap_or_else(|error| {
                    log::error!("the task's completion validator failed: {error}");
                    ToolOutcome::Internal.into()
                });
        }
        let report = lk_agents_voice::tool_executor::completion(self.task, report);
        if let Some(span) = span {
            crate::telemetry::end_function_tool(span, &self.call, self.started, &report.outcome);
        }
        report
    }

    /// Only core's captured Running owner decides whether this is a live expert outcome or a
    /// stale terminal receipt after a call ID was reused.
    fn send_update(&self, report: ToolReport) {
        use lk_agents_voice::agent_session::{SpeechCommand, ToolUpdate, VoiceCommandOrigin};
        let invocation = &self.invocation;
        let origin = VoiceCommandOrigin::Tool {
            invocation_id: invocation.invocation_id.clone(),
            operation: invocation.operation,
            call_id: self.call.call_id.clone(),
        };
        let update = SpeechCommand::ToolUpdate {
            invocation_id: invocation.invocation_id.clone(),
            operation: invocation.operation,
            call_id: self.call.call_id.clone(),
            update: ToolUpdate::from(report),
        };
        let _ = self.shared.queue.send_from(origin, update.into());
    }
}

/// Awaits the step's executions in the model's order and hands them to the driver, which decides
/// the step; a model reply's step waits for what its run does next when the task `continues`.
/// `None` when the task does not go on from it.
pub(crate) async fn run_tools(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyExecution,
    tools: ToolStep,
    forwarded_text_empty: bool,
    continues: bool,
) -> Option<StepNext> {
    // A never-adopted preemptive generation's calls never started and are owed nothing; a `say`
    // has no run.
    let model = matches!(ctx.plan.source, ReplySource::Llm | ReplySource::Realtime);
    if !handle.scheduled() || (tools.is_empty() && !model) {
        return None;
    }

    // Read **before** the executions are awaited; the read after is a different behaviour. The
    // speech's interruption, not the step's.
    let interrupted_before_tools = handle.interrupted();

    // Awaited in the model's order: they ran concurrently, so this is a join, not a sequence.
    let mut executions = Vec::with_capacity(tools.calls.len());
    for d in tools.calls {
        let execution = match d.rx.await {
            Ok(Some(out)) => out,
            // Dropped unrun: the reference returns before executing, so it is owed nothing.
            Ok(None) => continue,
            // Aborted or panicked: never a hole, since the step pairs by index.
            Err(_) => {
                let missing = ToolOutcome::ToolError(
                    lk_agents_voice::tool_executor::MISSING_TOOL_OUTPUT.into(),
                );
                let execution = answer(&d.call, missing.clone().into(), shared.time.wall());
                shared.report(GenEvent::ToolCallEnded {
                    speech: handle.id(),
                    call: None,
                    call_id: d.call.call_id.clone(),
                    end: execution.end,
                    message: execution.ended_message(),
                    outcome: missing,
                });
                execution
            }
        };
        executions.push(execution);
    }
    let end = Box::new(StepEnd {
        executions,
        interrupted_before_tools,
        interrupted: handle.interrupted(),
        forwarded_text_empty,
        max_tool_steps: ctx.plan.speech.max_tool_steps(),
        user_metrics: ctx.plan.user_metrics.clone(),
    });
    let (speech, step) = (handle.id(), ctx.plan.step);
    if !continues {
        shared.report(GenEvent::ToolsFinished { speech, step, end, answer: None });
        return None;
    }
    let (call, next) = shared.open_call(Answer::tool_step);
    let finished = GenEvent::ToolsFinished { speech, step, end, answer: Some(call.id) };
    shared.report_awaited(finished, call);
    // A loop gone answers nothing: the reply ends.
    next.await.ok()
}
