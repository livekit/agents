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
    /// `None` when `tool_choice="none"` dropped the call.
    rx: Option<tokio::sync::oneshot::Receiver<ToolExecution>>,
}

impl ToolStep {
    pub(crate) fn new() -> Self {
        ToolStep { calls: Vec::new(), last_deferred_start: None }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Run once the speech is *scheduled* (a preemptive generation's calls wait for adoption). The
    /// runner's task is **not** registered on the speech handle: a tool may outlive the turn.
    pub(crate) fn dispatch(
        &mut self,
        shared: &Arc<SessionShared>,
        handle: &Arc<SpeechHandle>,
        ctx: &ReplyContext,
        call: FunctionCall,
    ) {
        // Only advertised calls reach the speech; an invented tool is answered with an error.
        let advertised = ctx.request.tools.iter().any(|t| t.name == call.name);
        if advertised {
            handle.item_added(&lk_agents_abi::ChatItem::FunctionCall(call.clone()));
        }
        // Journalled only: the call was pushed to the handle above, before the executor runs.
        let names = vec![call.name.clone()];
        shared.trace(crate::trace::ToolStep::Calls { speech: handle.id(), names });
        // Scheduled: starts now; a preemptive generation: on adoption. `tool_choice="none"` calls
        // are started and dropped; unadvertised calls and calls with no runner are answered, not
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
            Some(Ok(runner)) => Some(crate::tool_executor::RunnerRef(Arc::clone(runner))),
            Some(Err(_)) | None => None,
        };
        let (speech, step) = (handle.id(), handle.num_steps());
        let started =
            GenEvent::ToolCallStarted { speech, step, call: Box::new(call.clone()), runner };
        let deferred = if handle.scheduled() {
            shared.report(started);
            None
        } else {
            Some(started)
        };
        let rx =
            work.map(|work: Result<Arc<dyn crate::tool_executor::ToolRunner>, Vec<String>>| {
                let (tx, rx) = tokio::sync::oneshot::channel();
                let one = call.clone();
                let handle = Arc::clone(handle);
                let spawner = Arc::clone(&shared.spawner);
                let shared = Arc::clone(shared);
                // Deferred starts are reported in the model's order.
                let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
                let previous = self.last_deferred_start.replace(started_rx);
                let task_owner = Arc::clone(&handle);
                let abort = lk_agents_abi::SpawnerExt::spawn_unboxed(&*spawner, async move {
                    // Adoption first: a discarded preemptive generation drops the call unrun.
                    let scheduled = tokio::select! {
                        biased;
                        () = handle.wait_for_interrupted() => false,
                        () = handle.wait_for_scheduled() => true,
                    };
                    if !scheduled {
                        return;
                    }
                    if let Some(started) = deferred {
                        if let Some(previous) = previous {
                            let _ = previous.await;
                        }
                        shared.report(started);
                    }
                    let _ = started_tx.send(());
                    // The host reports, the driver decides (`decide`).
                    let outcome = match work {
                        Ok(runner) => crate::tool_executor::decide(
                            runner.run(one.clone(), speech, step).await,
                        ),
                        // Never run: answered in the reference executor's words so the model corrects.
                        Err(available) => ToolOutcome::Error(
                            crate::tool_executor::unknown_tool_message(&one.name, &available),
                        ),
                    };
                    let _ = tx.send(answer(&one, outcome, shared.time.wall()));
                });
                task_owner.add_task(Arc::new(abort));
                rx
            });
        self.calls.push(Dispatched { call, rx });
    }
}

/// `None` when nothing ran. The decision is reported before it is acted on.
pub(crate) async fn run_tools(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    tools: ToolStep,
    forwarded_text_empty: bool,
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
    let interrupted_before_tools = handle.interrupted_speech();

    // Awaited in the model's order: they ran concurrently, so this is a join, not a sequence.
    let mut executions = Vec::with_capacity(tools.calls.len());
    for d in tools.calls {
        let Some(rx) = d.rx else { continue };
        let execution = match rx.await {
            Ok(out) => out,
            // Aborted or panicked: never a hole, since `next_step` pairs by index.
            Err(_) => answer(&d.call, ToolOutcome::Missing, shared.time.wall()),
        };
        // The lifecycle event's text: nothing for a silent return, handoff or cancellation.
        let message = match execution.end {
            ToolEnd::Done => (!execution.output.output.is_empty()
                && execution.agent_task.is_none())
            .then(|| execution.output.output.clone()),
            ToolEnd::Error => Some(execution.output.output.clone()),
            ToolEnd::Detached => execution.raw_update.clone(),
            ToolEnd::Cancelled => None,
        };
        let (call_id, end) = (d.call.call_id.clone(), execution.end);
        shared.report(GenEvent::ToolCallEnded { speech: handle.id(), call_id, end, message });
        executions.push(execution);
    }

    let mut decision = crate::tool_executor::next_step(crate::tool_executor::StepInputs {
        tool_output: &executions,
        interrupted_before_tools,
        // Re-read after: a interruption during a tool suppresses the reply, not the handoff, commit
        // or step increment.
        interrupted: handle.interrupted_speech(),
        num_steps: handle.num_steps(),
        max_tool_steps: ctx.speech.max_tool_steps,
        forwarded_text_empty,
        user_metrics: ctx.user_metrics.clone(),
    });

    // A host running hooks runs the event's handlers before the step goes on: they may cancel the
    // reply or the handoff. A loop gone answers nothing, and nothing
    // changed.
    let (tx, rx) = tokio::sync::oneshot::channel();
    let waits = shared.hook_host && decision.tools_executed.is_some();
    shared.report(GenEvent::ToolsFinished {
        speech: handle.id(),
        step: handle.num_steps(),
        commit: decision.commit.clone(),
        executed: decision.tools_executed.clone().map(Box::new),
        warnings: decision.warnings.clone(),
        verdict: waits.then_some(tx),
    });
    if waits && let Ok(verdict) = rx.await {
        decision = decision.settle(verdict.executed);
    }
    Some(decision)
}
