//! `perform_tool_executions`: a step's calls, run as they arrive and answered in the model's order.

use crate::agent_session::SessionShared;
use crate::events::{DiagnosticEvent, SessionEvent};
use crate::generation::{GenEvent, ReplyContext};
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::{StepDecision, ToolEnd, ToolExecution, ToolOutcome, answer};
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

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
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
        // Only advertised calls reach the speech; an invented tool is answered with an error.
        let advertised = ctx.request.tools.iter().any(|t| t.name == call.name);
        let names = vec![call.name.clone()];
        shared.trace(crate::trace::ToolStep::Calls { speech: handle.id(), names });
        // Authorized: starts now; otherwise once the step is. `tool_choice="none"` calls are
        // started and dropped; unadvertised calls and calls with no runner are answered, not
        // run. `Err`: answered with an error naming the runnable ones, never dropped, or the next
        // request makes it again.
        let forbidden = ctx.tool_choice == Some(lk_agents_abi::ToolChoice::Forbidden);
        let work = match (forbidden, advertised, ctx.providers.toolset.runner.clone()) {
            (true, ..) => None,
            (false, false, _) => {
                Some(Err(ctx.request.tools.iter().map(|t| t.name.clone()).collect()))
            }
            (false, true, None) => Some(Err(Vec::new())),
            (false, true, Some(runner)) => Some(Ok(runner)),
        };
        // Hosted calls enter the loop's running set; driver-answered ones never run.
        let runner = match &work {
            Some(Ok(runner)) => Some(Arc::clone(runner)),
            Some(Err(_)) | None => None,
        };
        let (speech, step) = (handle.id(), handle.num_steps());
        let Some(work) = work else {
            // Dropped unrun: told to the speech and the loop as the model made it.
            if advertised {
                handle.item_added(&lk_agents_abi::ChatItem::FunctionCall(call.clone()));
            }
            if handle.scheduled() {
                let call = Box::new(call.clone());
                let grant = None;
                shared.report(GenEvent::ToolCallStarted { speech, step, call, runner, grant });
            }
            self.calls.push(Dispatched { call, rx: None });
            return;
        };
        // `_tool_execution_started_cb`: the call is stamped with its start (history is ordered by
        // `created_at`, and it was made before the step was authorized), then told.
        let start =
            move |shared: &SessionShared, handle: &SpeechHandle, call: &mut FunctionCall| {
                call.created_at = shared.time.wall();
                if advertised {
                    handle.item_added(&lk_agents_abi::ChatItem::FunctionCall(call.clone()));
                }
                let call = Box::new(call.clone());
                // A hosted call's approval is the loop's to decide, from the record as the start
                // reaches it: the body waits for the answer.
                if runner.is_none() {
                    let grant = None;
                    shared.report(GenEvent::ToolCallStarted { speech, step, call, runner, grant });
                    return None;
                }
                let (grant, granted) = shared.calls.open(crate::agent_session::Answer::approval);
                let started =
                    GenEvent::ToolCallStarted { speech, step, call, runner, grant: Some(grant.id) };
                shared.report_awaited(started, grant);
                Some(granted)
            };
        let mut one = call.clone();
        let (deferred, granted) = if handle.step_authorized(step) {
            let granted = start(shared, handle, &mut one);
            (None, granted)
        } else {
            (Some(start), None)
        };
        let rx = {
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
                let mut granted = granted;
                if let Some(start) = deferred {
                    // Authorization first: a discarded preemptive generation, or a speech
                    // interrupted while queued, drops the call unrun.
                    let authorized = tokio::select! {
                        biased;
                        () = handle.wait_for_interrupted() => false,
                        () = handle.wait_for_step_authorized(step) => true,
                    };
                    if !authorized {
                        let _ = tx.send(None);
                        return;
                    }
                    if let Some(previous) = previous {
                        let _ = previous.await;
                    }
                    granted = start(&shared, &handle, &mut one);
                }
                let _ = started_tx.send(());
                // The host classifies, the driver decides (`answer`).
                let report = match work {
                    // The speech waits on the body until it answers or detaches: awaiting the
                    // speech from inside it is refused meanwhile.
                    Ok(runner) => {
                        handle.tool_blocking(&one.call_id, &one.name);
                        // Decided as the body starts: a call dropped unrun spends no answer.
                        let granted = match granted {
                            Some(rx) => rx.await.unwrap_or(false),
                            None => false,
                        };
                        let report = runner.answer(&one, &host_speech, granted, task).await;
                        handle.tool_returned(&one.call_id);
                        report
                    }
                    // Never run: answered in the reference executor's words so the model corrects.
                    Err(available) => ToolOutcome::ToolError(
                        crate::tool_executor::unknown_tool_message(&one.name, &available),
                    )
                    .into(),
                };
                let _ = tx.send(Some(answer(&one, report, shared.time.wall())));
            });
            task_owner.add_task(abort);
            Some(rx)
        };
        self.calls.push(Dispatched { call, rx });
    }
}

/// Arguments that are not JSON as written, as canonical JSON once repaired; `None` for valid or
/// unrepairable ones (the runner reports the latter to the model).
fn repaired_arguments(raw: &str) -> Option<String> {
    if raw.trim().is_empty() || serde_json::from_str::<serde_json::Value>(raw).is_ok() {
        return None;
    }
    let parsed = lk_agents_abi::llm::tool_arguments::parse(raw).ok()?;
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
                answer(&d.call, missing.into(), shared.time.wall())
            }
        };
        // The lifecycle event's text: nothing for a silent return, handoff or cancellation.
        let message = match execution.end {
            ToolEnd::Done => {
                (!execution.output.output.is_empty()).then(|| execution.output.output.clone())
            }
            ToolEnd::Error => Some(execution.output.output.clone()),
            ToolEnd::Detached => execution.raw_update.clone(),
            ToolEnd::Cancelled => None,
        };
        let (call_id, end) = (d.call.call_id.clone(), execution.end);
        shared.report(GenEvent::ToolCallEnded { speech: handle.id(), call_id, end, message });
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
