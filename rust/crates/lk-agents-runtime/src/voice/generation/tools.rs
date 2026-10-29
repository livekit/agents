//! `perform_tool_executions`: a step's calls, run as they arrive and answered in the model's order.

use lk_agents_abi::FunctionCall;
use lk_agents_core::tool_calls::PrepareFailure;
use lk_agents_voice::agent_session::SessionShared;
use lk_agents_voice::generation::{GenEvent, ReplyContext};
use lk_agents_voice::speech_handle::SpeechHandle;
use lk_agents_voice::tool_executor::{
    CallId, MadeCall, StepEnd, StepNext, ToolExecution, ToolGo, ToolOutcome, answer,
};
use std::sync::Arc;

/// Calls go to the driver as they arrive (latency hides behind the generation's tail), which
/// starts or drops each. **Order is the contract**: the step pairs by index.
pub(crate) struct ToolStep {
    calls: Vec<Dispatched>,
}

struct Dispatched {
    call: FunctionCall,
    /// Answers `None` when the driver dropped the call unrun.
    rx: tokio::sync::oneshot::Receiver<Option<ToolExecution>>,
}

impl ToolStep {
    pub(crate) fn new() -> Self {
        ToolStep { calls: Vec::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Hands the call to the driver and runs it as told. The task is **not** registered on the
    /// speech handle: a tool may outlive the turn.
    pub(crate) fn dispatch(
        &mut self,
        shared: &Arc<SessionShared>,
        handle: &Arc<SpeechHandle>,
        ctx: &ReplyContext,
        call: FunctionCall,
    ) {
        let (speech, step) = (handle.id(), handle.num_steps());
        let made = MadeCall {
            call: call.clone(),
            spec: ctx.request.tools.iter().find(|t| t.name == call.name).cloned(),
            forbidden: ctx.tool_choice == Some(lk_agents_abi::ToolChoice::Forbidden),
            answering: ctx.answering.clone(),
            runner: ctx.providers.toolset.runner.clone(),
        };
        let (go, first) = shared.open_call(lk_agents_voice::agent_session::Answer::tool_go);
        let made = GenEvent::ToolCallMade { speech, step, made: Box::new(made), go: go.id };
        shared.report_awaited(made, go);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let runner = ctx.providers.toolset.runner.clone();
        let host_speech = handle.object();
        let spawner = Arc::clone(&shared.spawner);
        let shared = Arc::clone(shared);
        let task_owner = Arc::clone(handle);
        let handle = Arc::clone(handle);
        let task = ctx.task;
        let abort = lk_agents_abi::SpawnerExt::spawn_unboxed(&*spawner, async move {
            // Dropped unrun, or a loop gone before the start: owed nothing.
            let Ok(ToolGo::Start { call: id, item, hosted }) = first.await else {
                let _ = tx.send(None);
                return;
            };
            let runner = runner.filter(|_| hosted);
            let call =
                HostedCall { shared: &shared, speech, step, runner, host_speech: &host_speech };
            let (execution, outcome) = call.run(id, *item, &handle, task).await;
            // Recorded as it arrives; the step answers the calls in the model's order.
            let message = execution.ended_message();
            let (call_id, end) = (execution.call.call_id.clone(), execution.end);
            let call = Some(id);
            let ended =
                GenEvent::ToolCallEnded { speech, step, call, call_id, end, message, outcome };
            shared.report(ended);
            let _ = tx.send(Some(execution));
        });
        task_owner.add_task(abort);
        self.calls.push(Dispatched { call, rx });
    }
}

/// One call's host work, each move decided by the shared tool machine in the loop.
struct HostedCall<'a> {
    shared: &'a Arc<SessionShared>,
    speech: lk_agents_voice::agent_activity::scheduling::SpeechId,
    step: u32,
    runner: Option<Arc<dyn lk_agents_voice::tool_executor::HostToolRunner>>,
    host_speech: &'a lk_agents_abi::CoreObject,
}

impl HostedCall<'_> {
    /// Describes the call's tool, then prepares, reads back or runs the call as told, until it is
    /// answered: the execution and the outcome the machine records.
    async fn run(
        &self,
        call: CallId,
        mut one: FunctionCall,
        handle: &SpeechHandle,
        task: bool,
    ) -> (ToolExecution, ToolOutcome) {
        let wall = || self.shared.time.wall();
        let mut prepared = None;
        let tool = match &self.runner {
            Some(runner) => runner.describe(&one.name).await.ok().flatten(),
            None => None,
        };
        let tool = tool.map(Box::new);
        let (speech, step) = (self.speech, self.step);
        let mut go = self.ask(|go| GenEvent::ToolDescribed { speech, step, call, tool, go }).await;
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
                    let identity = match &self.runner {
                        Some(runner) => {
                            match runner.prepare(&one, arguments, self.host_speech).await {
                                Ok(Ok(args)) => {
                                    prepared = Some(args.prepared);
                                    Ok(args.identity)
                                }
                                Ok(Err(refused)) => Err(PrepareFailure::Refused(refused.message)),
                                Err(e) => {
                                    log::error!(target: "livekit.agents", "preparing a tool call failed: {e}");
                                    Err(PrepareFailure::Internal)
                                }
                            }
                        }
                        None => Err(PrepareFailure::Internal),
                    };
                    let (speech, step) = (self.speech, self.step);
                    go = self
                        .ask(|go| GenEvent::ToolPrepared { speech, step, call, identity, go })
                        .await;
                }
                Ok(ToolGo::ReadBack { call }) => {
                    let answer = match (&self.runner, prepared.take()) {
                        (Some(runner), Some(args)) => {
                            runner.read_back(&args).await.map_err(|e| e.to_string())
                        }
                        _ => Err("no prepared arguments".to_owned()),
                    };
                    let (speech, step) = (self.speech, self.step);
                    go = self
                        .ask(|go| GenEvent::ReadBackRendered { speech, step, call, answer, go })
                        .await;
                }
                Ok(ToolGo::Execute { item }) => {
                    one = *item;
                    let (Some(runner), Some(args)) = (&self.runner, prepared.take()) else {
                        let outcome = ToolOutcome::Internal;
                        return (answer(&one, outcome.clone().into(), wall()), outcome);
                    };
                    // The speech waits on the body until it answers or detaches: awaiting the
                    // speech from inside it is refused meanwhile.
                    handle.tool_blocking(&one.call_id, &one.name);
                    let report = runner.answer(&args, &one, task).await;
                    handle.tool_returned(&one.call_id);
                    let outcome = report.outcome.clone();
                    return (answer(&one, report, wall()), outcome);
                }
                Ok(ToolGo::Settled { item, outcome, retry }) => {
                    let mut execution = answer(&item, outcome.clone().into(), wall());
                    execution.retry = retry;
                    return (execution, outcome);
                }
            }
        }
    }

    /// Reports the event built for the answer's request id, and awaits the call's next move.
    async fn ask(
        &self,
        event: impl FnOnce(lk_agents_voice::agent_session::RequestId) -> GenEvent,
    ) -> Result<ToolGo, lk_agents_voice::agent_session::api::Unanswered> {
        let (call, next) = self.shared.open_call(lk_agents_voice::agent_session::Answer::tool_go);
        self.shared.report_awaited(event(call.id), call);
        next.await
    }
}

/// Awaits the step's executions in the model's order and hands them to the driver, which decides
/// the step. `None` when nothing ran.
pub(crate) async fn run_tools(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    tools: ToolStep,
    forwarded_text_empty: bool,
) -> Option<StepNext> {
    // A never-adopted preemptive generation's calls never started and are owed nothing.
    if tools.is_empty() || !handle.scheduled() {
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
                    step: handle.num_steps(),
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
        max_tool_steps: ctx.speech.max_tool_steps(),
        user_metrics: ctx.user_metrics.clone(),
    });
    let (speech, step) = (handle.id(), handle.num_steps());
    if end.executions.is_empty() {
        shared.report(GenEvent::ToolsFinished { speech, step, end, answer: None });
        return None;
    }
    let (call, next) = shared.open_call(lk_agents_voice::agent_session::Answer::tool_step);
    let finished = GenEvent::ToolsFinished { speech, step, end, answer: Some(call.id) };
    shared.report_awaited(finished, call);
    // A loop gone answers nothing: the reply ends.
    next.await.ok()
}
