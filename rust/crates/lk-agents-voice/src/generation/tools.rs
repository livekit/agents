//! `perform_tool_executions`: a step's calls, run as they arrive and answered in the model's order.

use crate::agent_session::SessionShared;
use crate::events::{DiagnosticEvent, SessionEvent};
use crate::generation::{GenEvent, ReplyContext};
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::{CallId, StepDecision, ToolExecution, ToolGo, ToolOutcome, answer};
use lk_agents_abi::FunctionCall;
use std::sync::Arc;

/// Calls go to the runner as they arrive (latency hides behind the generation's tail). **Order is
/// the contract**: `next_step` pairs by index.
pub(crate) struct ToolStep {
    calls: Vec<Dispatched>,
    /// The start signal of the last call deferred to adoption, so the next one starts after it.
    last_deferred_start: Option<tokio::sync::oneshot::Receiver<()>>,
}

struct Dispatched {
    call: FunctionCall,
    /// `None` when `tool_choice="none"` dropped the call; answers `None` when the speech was
    /// interrupted before the step could start it.
    rx: Option<tokio::sync::oneshot::Receiver<Option<ToolExecution>>>,
}

impl ToolStep {
    pub(crate) fn new() -> Self {
        ToolStep { calls: Vec::new(), last_deferred_start: None }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Run once the step is authorized to play (a call made while another speech plays, or by a
    /// preemptive generation, waits). The runner's task is **not** registered on the speech
    /// handle: a tool may outlive the turn.
    pub(crate) fn dispatch(
        &mut self,
        shared: &Arc<SessionShared>,
        handle: &Arc<SpeechHandle>,
        ctx: &ReplyContext,
        mut call: FunctionCall,
    ) {
        // Malformed arguments the model's template leaked into are repaired once, here, and the
        // history keeps the repaired form: the next request would otherwise carry invalid JSON
        // that a provider refuses.
        if let Some(repaired) = repaired_arguments(&call.arguments) {
            call.arguments = repaired;
        }
        // A reply answering an `InputRequired(tool=)`: that tool takes the person's words, as
        // they said them, and the history shows the call as it ran.
        if let Some(answering) = ctx.answering.as_ref().filter(|a| a.tool == call.name)
            && let Some(spec) = ctx.request.tools.iter().find(|t| t.name == call.name)
            && let Some(arguments) = crate::tool_executor::answer_arguments(spec, &answering.said)
        {
            call.arguments = arguments;
        }
        // Only offered calls reach the speech; the shared tool machine answers an invented one.
        let advertised = ctx.request.tools.iter().any(|t| t.name == call.name);
        let names = vec![call.name.clone()];
        shared.trace(crate::trace::ToolStep::Calls { speech: handle.id(), names });
        let (speech, step) = (handle.id(), handle.num_steps());
        // `tool_choice="none"` calls are started and dropped: told to the speech and the loop as
        // the model made them.
        if ctx.tool_choice == Some(lk_agents_abi::ToolChoice::Forbidden) {
            if advertised {
                handle.item_added(&lk_agents_abi::ChatItem::FunctionCall(call.clone()));
            }
            if handle.scheduled() {
                let call = Box::new(call.clone());
                let (runner, go) = (None, None);
                shared.report(GenEvent::ToolCallStarted { speech, step, call, runner, go });
            }
            self.calls.push(Dispatched { call, rx: None });
            return;
        }
        // Offered calls with a runner are hosted: the loop's running set has them.
        let runner = ctx.providers.toolset.runner.clone().filter(|_| advertised);
        // `_tool_execution_started_cb`: the call is stamped with its start (history is ordered by
        // `created_at`, and it was made before the step was authorized), then told; the shared
        // tool machine answers with the call's first move.
        let start =
            move |shared: &SessionShared, handle: &SpeechHandle, call: &mut FunctionCall| {
                call.created_at = shared.time.wall();
                if advertised {
                    handle.item_added(&lk_agents_abi::ChatItem::FunctionCall(call.clone()));
                }
                let (go, moved) = shared.open_call(crate::agent_session::Answer::tool_go);
                let started = GenEvent::ToolCallStarted {
                    speech,
                    step,
                    call: Box::new(call.clone()),
                    runner: runner.clone(),
                    go: Some(go.id),
                };
                shared.report_awaited(started, go);
                (moved, runner)
            };
        let mut one = call.clone();
        let (deferred, started) = if handle.step_authorized(step) {
            (None, Some(start(shared, handle, &mut one)))
        } else {
            (Some(start), None)
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = Arc::clone(handle);
        let host_speech = handle.object();
        let spawner = Arc::clone(&shared.spawner);
        let shared = Arc::clone(shared);
        // Deferred starts are reported in the model's order.
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let previous = self.last_deferred_start.replace(started_rx);
        let task_owner = Arc::clone(&handle);
        let task = ctx.task;
        let abort = lk_agents_abi::SpawnerExt::spawn_unboxed(&*spawner, async move {
            // A start already reported runs, as the reference's executor starts the body
            // right after `tool_execution_started_cb`, interrupted or not.
            let mut started = started;
            if let Some(start) = deferred {
                // Authorization first: a discarded preemptive generation, or a speech
                // interrupted while queued, drops the call unrun.
                let authorized = tool_authorization(&handle, step).await;
                if !authorized {
                    let _ = tx.send(None);
                    return;
                }
                if let Some(previous) = previous {
                    let _ = previous.await;
                }
                started = Some(start(&shared, &handle, &mut one));
            }
            let _ = started_tx.send(());
            let Some((first, runner)) = started else { return };
            let call =
                HostedCall { shared: &shared, speech, step, runner, host_speech: &host_speech };
            let (execution, id, outcome) = call.run(first, one, &handle, task).await;
            // Recorded as it arrives; the step answers the calls in the model's order.
            let message = execution.ended_message();
            let (call_id, end) = (execution.call.call_id.clone(), execution.end);
            let call = id;
            let ended =
                GenEvent::ToolCallEnded { speech, step, call, call_id, end, message, outcome };
            shared.report(ended);
            let _ = tx.send(Some(execution));
        });
        task_owner.add_task(abort);
        self.calls.push(Dispatched { call, rx: Some(rx) });
    }
}

async fn tool_authorization(handle: &SpeechHandle, step: u32) -> bool {
    let mut interrupted = std::pin::pin!(handle.wait_for_interrupted());
    let mut done = std::pin::pin!(handle.wait_for_done());
    let mut authorized = std::pin::pin!(handle.wait_for_step_authorized(step));
    std::future::poll_fn(|cx| {
        if interrupted.as_mut().poll(cx).is_ready() || done.as_mut().poll(cx).is_ready() {
            return std::task::Poll::Ready(false);
        }
        if authorized.as_mut().poll(cx).is_ready() {
            return std::task::Poll::Ready(true);
        }
        std::task::Poll::Pending
    })
    .await
}

#[cfg(test)]
mod authorization_tests {
    use super::*;
    use std::future::Future;
    use std::task::{Context, Poll, Waker};

    #[test]
    fn a_failed_speech_discards_a_tool_still_waiting_for_authorization() {
        let handle = SpeechHandle::detached("failed-before-authorization".into(), true);
        let mut waiting = std::pin::pin!(tool_authorization(&handle, 1));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(waiting.as_mut().poll(&mut cx).is_pending());
        handle.mark_done(Some(lk_agents_abi::errors::ApiError::new("output refused").into()));
        assert_eq!(waiting.as_mut().poll(&mut cx), Poll::Ready(false));
    }
}

/// One call's host work, each move decided by the shared tool machine in the loop.
struct HostedCall<'a> {
    shared: &'a Arc<SessionShared>,
    speech: crate::agent_activity::scheduling::SpeechId,
    step: u32,
    runner: Option<Arc<dyn crate::tool_executor::HostToolRunner>>,
    host_speech: &'a lk_agents_abi::CoreObject,
}

impl HostedCall<'_> {
    /// Prepares, reads back or runs the call as told, until it is answered: the execution, the
    /// call's machine id once it has one, and the outcome the machine records.
    async fn run(
        &self,
        first: crate::agent_session::api::Awaited<ToolGo>,
        mut one: FunctionCall,
        handle: &SpeechHandle,
        task: bool,
    ) -> (ToolExecution, Option<CallId>, ToolOutcome) {
        let wall = || self.shared.time.wall();
        let mut id = None;
        let mut prepared = None;
        let mut go = first.await;
        loop {
            match go {
                // A loop gone answers nothing: the call never ran.
                Err(_) | Ok(ToolGo::Gone) => {
                    let outcome = ToolOutcome::Cancelled;
                    return (answer(&one, outcome.clone().into(), wall()), id, outcome);
                }
                Ok(ToolGo::Describe { call }) => {
                    id = Some(call);
                    let tool = match &self.runner {
                        Some(runner) => runner.describe(&one.name).await.ok().flatten(),
                        None => None,
                    };
                    let tool = tool.map(Box::new);
                    let (speech, step) = (self.speech, self.step);
                    go = self
                        .ask(|go| GenEvent::ToolDescribed { speech, step, call, tool, go })
                        .await;
                }
                Ok(ToolGo::Prepare { call, arguments }) => {
                    let identity = match &self.runner {
                        Some(runner) => {
                            match runner.prepare(&one, arguments, self.host_speech).await {
                                Ok(Ok(args)) => {
                                    prepared = Some(args.prepared);
                                    Ok(args.identity)
                                }
                                Ok(Err(refused)) => Err(refused.message),
                                Err(e) => Err(e.to_string()),
                            }
                        }
                        None => Err(format!("no runner for tool `{}`", one.name)),
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
                        return (answer(&one, outcome.clone().into(), wall()), id, outcome);
                    };
                    // The speech waits on the body until it answers or detaches: awaiting the
                    // speech from inside it is refused meanwhile.
                    handle.tool_blocking(&one.call_id, &one.name);
                    let report = runner.answer(&args, &one, task).await;
                    handle.tool_returned(&one.call_id);
                    let outcome = report.outcome.clone();
                    return (answer(&one, report, wall()), id, outcome);
                }
                Ok(ToolGo::Settled { item, outcome, retry }) => {
                    let mut execution = answer(&item, outcome.clone().into(), wall());
                    execution.retry = retry;
                    return (execution, id, outcome);
                }
            }
        }
    }

    /// Reports the event built for the answer's request id, and awaits the call's next move.
    async fn ask(
        &self,
        event: impl FnOnce(crate::agent_session::RequestId) -> GenEvent,
    ) -> Result<ToolGo, crate::agent_session::api::Unanswered> {
        let (call, next) = self.shared.open_call(crate::agent_session::Answer::tool_go);
        self.shared.report_awaited(event(call.id), call);
        next.await
    }
}

/// Arguments that are not JSON as written, as canonical JSON once repaired; `None` for valid or
/// unrepairable ones (the runner reports the latter to the model).
fn repaired_arguments(raw: &str) -> Option<String> {
    if raw.trim().is_empty() || serde_json::from_str::<serde_json::Value>(raw).is_ok() {
        return None;
    }
    let parsed = lk_agents_core::llm::tool_arguments::parse(raw).ok()?;
    parsed.repair.as_ref()?;
    Some(serde_json::Value::Object(parsed.arguments).to_string())
}

#[cfg(test)]
mod tests {
    // Catches: a repaired call's arguments left malformed in history, or valid ones rewritten.
    #[test]
    fn malformed_arguments_are_kept_repaired() {
        let raw = r#"{"order_id": ["<|\"|\"O_WAAB70<|\"|\"]}"#;
        let repaired = super::repaired_arguments(raw).expect("repaired");
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok(), "{repaired}");
        let leaked = super::repaired_arguments(r#"{"label":"human<|im_end|>"#).expect("repaired");
        assert_eq!(leaked, r#"{"label":"human"}"#);
        assert_eq!(super::repaired_arguments(r#"{"a": 1}"#), None);
        assert_eq!(super::repaired_arguments(""), None);
    }
}

/// `None` when nothing ran. The decision is reported before it is acted on.
pub(crate) async fn run_tools(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    tools: ToolStep,
    forwarded_text_empty: bool,
    retries: &mut lk_agents_core::tool_outcome::ToolRetries,
) -> Option<StepDecision> {
    // A never-adopted preemptive generation's calls never started and are owed nothing.
    if tools.is_empty() || !handle.scheduled() {
        return None;
    }

    // `tool_choice="none"` calls are dropped (this bounds a tool loop), each reported.
    if ctx.tool_choice == Some(lk_agents_abi::ToolChoice::Forbidden) {
        for d in &tools.calls {
            let diagnostic = DiagnosticEvent::error(crate::events::diagnostics::TOOL_CHOICE_NONE)
                .with("function", d.call.name.clone())
                .with("speech_id", handle.id().0.to_string());
            shared.report(GenEvent::Emit(Box::new(SessionEvent::Diagnostic(diagnostic))));
        }
        return None;
    }
    if tools.calls.iter().all(|d| d.rx.is_none()) {
        return None;
    }

    // Read **before** the executions are awaited (selects `NextStep::Interrupted`); the read
    // after is a different behaviour. The speech's interruption, not the step's.
    let interrupted_before_tools = handle.interrupted();

    // Awaited in the model's order: they ran concurrently, so this is a join, not a sequence.
    let mut executions = Vec::with_capacity(tools.calls.len());
    for d in tools.calls {
        let Some(rx) = d.rx else { continue };
        let execution = match rx.await {
            Ok(Some(out)) => out,
            // Never started: the reference returns before executing, so it is owed nothing.
            Ok(None) => continue,
            // Aborted or panicked: never a hole, since `next_step` pairs by index.
            Err(_) => {
                let missing =
                    ToolOutcome::ToolError(crate::tool_executor::MISSING_TOOL_OUTPUT.into());
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
    if executions.is_empty() {
        return None;
    }

    let limit = lk_agents_abi::agent::Retries::default().tool;
    let retried = executions.iter().map(|e| (e.call.name.as_str(), e.retry));
    let retries_exceeded = retries.step(retried, limit);
    let mut decision = crate::tool_executor::next_step(crate::tool_executor::StepInputs {
        tool_output: &executions,
        interrupted_before_tools,
        // Re-read after: a interruption during a tool suppresses the reply, not the handoff, commit
        // or step increment.
        interrupted: handle.interrupted(),
        num_steps: handle.num_steps(),
        max_tool_steps: ctx.speech.max_tool_steps(),
        forwarded_text_empty,
        user_metrics: ctx.user_metrics.clone(),
        retries_exceeded,
    });

    // A host running hooks runs the event's handlers before the step goes on: they may cancel the
    // reply or the handoff. A loop gone answers nothing, and nothing
    // changed.
    let (tx, rx) = tokio::sync::oneshot::channel();
    let waits = shared.hook_host && decision.tools_executed.is_some();
    shared.report(GenEvent::ToolsFinished {
        speech: handle.id(),
        step: handle.num_steps(),
        tools: crate::tool_executor::StepTools {
            commit: decision.commit.clone(),
            executed: decision.tools_executed.clone().map(Box::new),
            warnings: decision.warnings.clone(),
            deliveries: decision.deliveries.clone(),
            retries_exceeded: decision.retries_exceeded.clone(),
        },
        verdict: waits.then_some(tx),
    });
    if waits && let Ok(verdict) = rx.await {
        decision = decision.settle(verdict.executed);
    }
    Some(decision)
}
