//! `perform_tool_executions`: a step's calls, run as they arrive and answered in the model's order.

use super::ReplyExecution;
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
        let made = MadeCall {
            call: call.clone(),
            spec: ctx.plan.request.tools.iter().find(|t| t.name == call.name).cloned(),
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
        self.speeches.spawn(speech, "tool call", async move {
            // Dropped unrun, or a loop gone before the start: owed nothing.
            let Ok(ToolGo::Start { call: id, item, host, first }) = first.await else {
                let _ = tx.send(None);
                return;
            };
            // The callbacks run with the host's object for the speech.
            let host = host.map(|mut host| {
                host.invocation.speech = Some(handle.object());
                host
            });
            let call = HostedCall { shared: &shared, speech, host: host.map(|host| *host), bodies };
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
}

impl HostedCall<'_> {
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
                    let identity = match &self.host {
                        Some(h) => match h.host.prepare_tool(&h.invocation, arguments).await {
                            Ok(Ok(args)) => {
                                prepared = Some(args.prepared);
                                Ok(args.identity)
                            }
                            Ok(Err(refused)) => Err(PrepareFailure::Refused(refused.message)),
                            Err(e) => {
                                log::error!(target: "livekit.agents", "preparing a tool call failed: {e}");
                                Err(PrepareFailure::Internal)
                            }
                        },
                        None => Err(PrepareFailure::Internal),
                    };
                    let speech = self.speech;
                    go = self.ask(|go| GenEvent::ToolPrepared { speech, call, identity, go }).await;
                }
                Ok(ToolGo::ReadBack { call }) => {
                    let answer = match (&self.host, prepared.take()) {
                        (Some(h), Some(args)) => {
                            h.host.read_back(&h.invocation, &args).await.map_err(|e| e.to_string())
                        }
                        _ => Err("no prepared arguments".to_owned()),
                    };
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
                    handle.tool_blocking(&one.call_id, &one.name);
                    let report =
                        run_body(h, &args, &one, task, self.shared, &self.bodies, self.speech)
                            .await;
                    handle.tool_returned(&one.call_id);
                    let outcome = report.outcome.clone();
                    return (answer(&one, report, wall()), outcome);
                }
                Ok(ToolGo::Settled { item, outcome }) => {
                    return (answer(&item, outcome.clone().into(), wall()), outcome);
                }
            }
        }
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
    h: &HostedTool,
    prepared: &lk_agents_abi::HostRef,
    call: &FunctionCall,
    task: bool,
    shared: &Arc<SessionShared>,
    bodies: &crate::voice::SpeechTasks,
    speech: lk_agents_voice::agent_activity::scheduling::SpeechId,
) -> ToolReport {
    use lk_agents_voice::agent_session::{Origin, SpeechCommand, ToolUpdate};
    let invocation = lk_agents_abi::agent::Invocation {
        function_call: Some(call.clone()),
        approval: lk_agents_core::approval::mark(call),
        ..h.invocation.clone()
    };
    let mut progress = shared.tool_progress.tool_body(call.call_id.clone());
    let (tx, rx) = tokio::sync::oneshot::channel();
    let (host, prepared, shared, call_id) =
        (h.host.clone(), prepared.clone(), shared.clone(), call.call_id.clone());
    struct Body {
        progress: crate::voice::session::calls::ToolProgress,
        call_id: String,
    }
    impl Drop for Body {
        fn drop(&mut self) {
            self.progress.end_tool_body(&self.call_id);
        }
    }
    let owned = Body { progress: shared.tool_progress.clone(), call_id: call_id.clone() };
    bodies.spawn(speech, "host tool body", async move {
        let _owned = owned;
        let execute = async {
            let mut report = match host.run_tool(&invocation, &prepared).await {
                Ok((report, _)) => report,
                Err(error) => {
                    log::error!("the tool's host failed outside its error handling: {error}");
                    ToolOutcome::Internal.into()
                }
            };
            if lk_agents_voice::tool_body::validates_completion(task, &report) {
                report = host
                    .validate_tool_completion(&invocation, &prepared, report)
                    .await
                    .unwrap_or_else(|error| {
                        log::error!("the task's completion validator failed: {error}");
                        ToolOutcome::Internal.into()
                    });
            }
            lk_agents_voice::tool_body::completion(task, report)
        };
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
                let origin = Origin::Tool { call_id: call_id.clone() };
                let _ = shared.queue.send_from(
                    origin,
                    SpeechCommand::ToolUpdate { call_id, update: ToolUpdate::from(report) }.into(),
                );
            }
            Err(Err(_)) => {
                let _ = tx.send(execute.await);
            }
        }
    });
    rx.await.unwrap_or_else(|_| ToolOutcome::Cancelled.into())
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
