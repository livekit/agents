//! `_pipeline_reply_task`: the one task every speech runs, its steps and their follow-ups.

use super::forward::{Voice, voice_leg};
use super::llm::{SegEvent, produce_segments};
use super::realtime_generation::RealtimePlayer;
use super::tools::run_tools;
use crate::voice::generation::realtime_generation::realtime_generation_task;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{ChatItem, ChatMessage, Ts};
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::agent_session::SessionShared;
pub(crate) use lk_agents_voice::generation::pipeline::Finished;
use lk_agents_voice::generation::pipeline::{
    CompletionObservation, Execution, StartEffect, StartEvent,
};
use lk_agents_voice::generation::{FinishInputs, TurnAccumulator, TurnOutcome, TurnStats};
use lk_agents_voice::generation::{GenEvent, ReplyContext, ReplySource, fatal};
use lk_agents_voice::speech_handle::SpeechHandle;
use lk_agents_voice::tool_executor::NextStep;
use std::sync::Arc;
use tokio::sync::mpsc;

/// `_pipeline_reply_task`, and **the one task every speech runs** (`say` and realtime generations
/// too). Spawned through the injected [`Spawner`](lk_agents_abi::Spawner), never `tokio::spawn`.
/// `player`: a realtime speech's events, which the runtime routes to it from the socket.
pub async fn pipeline_reply_task(
    shared: Arc<SessionShared>,
    handle: Arc<SpeechHandle>,
    ctx: ReplyContext,
    player: Option<RealtimePlayer>,
) {
    let speech = handle.id();
    let error = match run(&shared, &handle, ctx, player).await {
        Ok(Finished { outcome, interrupted }) => {
            shared.report(GenEvent::Done { speech, outcome: Box::new(outcome), interrupted });
            return;
        }
        Err(DriverError::Provider(error)) => error,
        Err(DriverError::TextOutput(lk_agents_abi::voice::io::IoError::Host(error))) => {
            ProviderError::Host { error, api_kind: None, retryable: false, status_code: None }
        }
        // Reachable from the host (`Agent(llm=None)` then `generate_reply()`).
        Err(DriverError::Missing(what)) => {
            fatal(format!("cannot generate a reply: the session has no {what}"))
        }
        Err(other) => fatal(other.to_string()),
    };
    // **Moved, never formatted**: only the binding can release a `ProviderError::Host`'s side-table
    // entry. Not marked here: the loop's `Failed` arm moves the error onto the handle.
    shared.report(GenEvent::Failed { speech, error: Box::new(error) });
}

fn step_spoken(
    shared: &SessionShared,
    handle: &SpeechHandle,
    interrupted: bool,
    has_tool_output: bool,
    tools_running: bool,
    item: Option<ChatMessage>,
) {
    shared.report(GenEvent::StepSpoken {
        speech: handle.id(),
        step: handle.num_steps(),
        interrupted,
        has_tool_output,
        tools_running,
        item: item.map(Box::new),
    });
}

/// A tool reply is the same turn on the same handle, so a interruption during a tool call stops what is
/// still talking.
async fn run(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    mut ctx: ReplyContext,
    mut player: Option<RealtimePlayer>,
) -> Result<Finished, DriverError> {
    let mut startup = ctx.startup(StartEvent::Begin { speech: handle.id() }, shared.time.read());
    loop {
        match startup {
            StartEffect::RequestPlan(plan) => {
                let planned = request_plan(shared, plan).await;
                startup = ctx.startup(StartEvent::Planned(planned), shared.time.read());
            }
            StartEffect::Run => break,
            StartEffect::Finished(finished) => return Ok(finished),
        }
    }
    // The last step spoken on the realtime path, kept in case its follow-up never arrives.
    let mut last: Option<Finished> = None;
    loop {
        let outcome = match ctx.source.execution() {
            // Borrowed: the channel outlives the step, because the follow-up arrives on it.
            Execution::Realtime => {
                let events =
                    player.as_mut().expect("the runtime gives a realtime speech its player");
                let wake = &shared.queue;
                realtime_generation_task(shared, handle, &ctx, events, wake).await?
            }
            _ => pipeline_reply_task_impl(shared, handle, &ctx).await?,
        };
        match outcome {
            StepOutcome::Done(finished) => return Ok(finished),
            StepOutcome::Again { request, pending } => {
                // Read pre-increment by `next_step` for the budget.
                let spoken_step = handle.num_steps();
                handle.step();
                shared.report(GenEvent::ToolStep { speech: handle.id(), step: handle.num_steps() });
                let Some(next) =
                    follow_up(shared, handle, &ctx, spoken_step, request, pending).await
                else {
                    return Ok(Finished {
                        outcome: empty_outcome(shared, &ctx, true),
                        interrupted: true,
                    });
                };
                ctx = next;
            }
            StepOutcome::Next(finished) => last = Some(finished),
            StepOutcome::Released => {
                let outcome = || empty_outcome(shared, &ctx, true);
                return Ok(
                    last.unwrap_or_else(|| Finished { outcome: outcome(), interrupted: true })
                );
            }
        }
    }
}

/// What one step decided. A going-again step's `TurnOutcome` is dropped: one completion per turn.
pub(crate) enum StepOutcome {
    Done(Finished),
    /// `pending` is what this step produced, which the loop may not have committed yet.
    Again {
        request: lk_agents_voice::tool_executor::ReplyRequest,
        pending: Vec<ChatItem>,
    },
    Next(Finished),
    /// Nothing more will be spoken (interrupted while parked, or an empty follow-up).
    Released,
}

/// One step's words folded into its outcome, the generation marked done, and the step reported
/// spoken **before its tools run**: `wait_for_playout` from a tool waits for the words before it.
#[allow(clippy::too_many_arguments)]
fn finish_step(
    shared: &SessionShared,
    handle: &SpeechHandle,
    ctx: &ReplyContext,
    acc: TurnAccumulator,
    interrupted: bool,
    stats: TurnStats,
    has_tool_output: bool,
    tools_running: bool,
) -> Finished {
    finish_step_with(
        shared,
        handle,
        ctx.message_id.clone(),
        acc,
        interrupted,
        stats,
        has_tool_output,
        tools_running,
        None,
    )
}

/// [`finish_step`], with the provider's ids for the assistant message's metrics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_step_with(
    shared: &SessionShared,
    handle: &SpeechHandle,
    message_id: String,
    acc: TurnAccumulator,
    interrupted: bool,
    stats: TurnStats,
    has_tool_output: bool,
    tools_running: bool,
    provider_request_ids: Option<Vec<String>>,
) -> Finished {
    let inputs = FinishInputs {
        stopped_speaking_at: shared.time.wall_secs(),
        interrupted,
        stats,
        message_id,
    };
    let observed = CompletionObservation {
        speech: handle.id(),
        step: handle.num_steps(),
        adopted_user_metrics: handle.user_metrics(),
        provider_request_ids,
        has_tool_output,
        tools_running,
    };
    let (finished, spoken) =
        lk_agents_voice::generation::pipeline::finish_step(acc, inputs, observed);
    shared.report(spoken);
    finished
}

/// One step of `_pipeline_reply_task_impl` (or `_tts_task`): the words and the voice, two futures
/// on one task.
async fn pipeline_reply_task_impl(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
) -> Result<StepOutcome, DriverError> {
    // Taken, not borrowed: no lock across an await, and a second read finds `None`.
    let voice = match &ctx.source {
        ReplySource::Audio { rx, .. } => {
            rx.lock().expect("say audio poisoned").take().map_or(Voice::Silent, Voice::Frames)
        }
        _ => ctx.providers.tts.clone().map_or(Voice::Silent, Voice::Tts),
    };
    // The words start now (preemptive generation). `join!`, not `try_join!`: the voice's host TTS
    // stream must reach its `close`. Errors are read afterwards, words first.
    let (tx, rx) = mpsc::unbounded_channel::<SegEvent>();
    let ((words, tools), spoken) = tokio::join!(
        produce_segments(shared, handle, ctx, tx),
        voice_leg(shared, handle, ctx, voice, rx)
    );
    // **Every call the model made is answered**, whatever became of the step, or the next request
    // would make it again.
    let llm_gen_data = match words {
        Ok(data) => data,
        Err(e) => {
            // A step that took the floor gives it back before its tools are waited for, as
            // `_mark_generation_done` precedes `await exe_task`: a tool waiting for idle would
            // otherwise wait on its own speech.
            if matches!(spoken, Ok(Some(_))) {
                let has_tools = !tools.is_empty();
                let tools_running = has_tools && !handle.interrupted();
                step_spoken(shared, handle, handle.interrupted(), has_tools, tools_running, None);
            }
            let _ = run_tools(shared, handle, ctx, tools, true).await;
            return Err(e);
        }
    };
    let Some(spoken) = spoken? else {
        // Interrupted before its authorization: started tools answered as interrupted. An
        // authorization granted in that instant marks the step spoken first, so a tool awaiting its
        // own playout does not deadlock.
        if handle.authorized() {
            handle.clear_authorization();
            step_spoken(shared, handle, true, !tools.is_empty(), false, None);
        }
        let _ = run_tools(shared, handle, ctx, tools, true).await;
        let outcome = empty_outcome(shared, ctx, true);
        return Ok(StepOutcome::Done(Finished { outcome, interrupted: true }));
    };
    // The step's item rides with the step: built from what was heard, committed by the loop. The
    // speech's interruption decides `tools_running`: a step whose audio was cut still has tools.
    // A `say` (`_tts_task`) times only its voice: no model, no metadata.
    let stats = ctx.source.turn_stats(
        llm_gen_data,
        ctx.providers.llm.as_ref().map(|llm| llm.id().metrics_metadata()),
        ctx.providers.tts.as_ref().map(|tts| tts.id().metrics_metadata()),
    );
    let has_tools = !tools.is_empty();
    let tools_running = has_tools && !handle.interrupted();
    let finished = finish_step(
        shared,
        handle,
        ctx,
        spoken.acc,
        spoken.interrupted,
        stats,
        has_tools,
        tools_running,
    );
    let forwarded_text_empty = finished.outcome.forwarded_text.is_empty();
    let Some(decision) = run_tools(shared, handle, ctx, tools, forwarded_text_empty).await else {
        return Ok(StepOutcome::Done(finished));
    };
    // A drain close during the tools: outputs are committed and no follow-up is requested.
    let draining = *shared.draining.borrow();
    let follow_up = match decision.next {
        NextStep::Done => None,
        NextStep::Interrupted => {
            let outcome = finished.outcome;
            return Ok(StepOutcome::Done(Finished { outcome, interrupted: true }));
        }
        NextStep::Reply(request) => Some(request),
    };
    match follow_up {
        Some(request) if !draining => {
            // What this step produced, for the next request, in case the loop has not committed it.
            let message = finished.outcome.assistant_message.map(ChatItem::Message);
            let pending = message.into_iter().chain(decision.commit).collect();
            Ok(StepOutcome::Again { request, pending })
        }
        Some(_) | None => Ok(StepOutcome::Done(finished)),
    }
}

/// The follow-up step's context over the loop's published frame (inline-task merges and
/// `update_instructions` included); `pending` covers what is not yet committed. The same tools
/// are advertised (`tool_choice` stops loops) and a pinned toolset is kept.
async fn follow_up(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    prev: &ReplyContext,
    spoken_step: u32,
    request: lk_agents_voice::tool_executor::ReplyRequest,
    pending: Vec<ChatItem>,
) -> Option<ReplyContext> {
    let planned = request_plan(
        shared,
        lk_agents_voice::generation::ReplyPlan {
            speech: handle.id(),
            after: handle.id(),
            step: Some(spoken_step),
            spec: lk_agents_voice::generation::PlanSpec {
                record: None,
                new_message: None,
                instructions: None,
                turn_instructions: None,
                tools: prev.request.tools.clone(),
                on_enter: false,
                running_tool_calls: Vec::new(),
                audio_output_enabled: prev.request.tts_enabled,
                text_output_enabled: prev.request.text_output_enabled,
                expressive: prev.request.expressive.clone(),
            },
            pending,
            tool_choice: Some(request.tool_choice),
        },
    )
    .await?;
    Some(ReplyContext {
        preemptive_tts: prev.preemptive_tts,
        request: planned.request,
        replan: None,
        source: ReplySource::Llm,
        // Always concrete on a follow-up: `"none"` or `"auto"`, never `"required"`.
        tool_choice: planned.tool_choice,
        frame: prev.frame,
        toolset: prev.toolset.clone(),
        providers: prev.providers.clone(),
        // Carried over when this turn forwarded no text, for `e2e_latency`.
        user_metrics: request.previous_user_metrics.or_else(|| prev.user_metrics.clone()),
        // A fresh id: this step commits its own assistant message.
        message_id: shared.mint_item_id(),
        speech: prev.speech,
        answering: None,
        task: prev.task,
    })
}

/// The runtime owns the waiter; the core owns when the accepted record is ready.
async fn request_plan(
    shared: &SessionShared,
    plan: lk_agents_voice::generation::ReplyPlan,
) -> Option<lk_agents_voice::generation::PlannedReply> {
    let (call, result) = shared.open_call(lk_agents_voice::agent_session::Answer::reply_plan);
    shared.report_awaited(GenEvent::Plan { call: call.id, plan: Box::new(plan) }, call);
    result.await.ok().flatten().map(|planned| *planned)
}

pub(super) fn empty_outcome(
    shared: &Arc<SessionShared>,
    ctx: &ReplyContext,
    interrupted: bool,
) -> TurnOutcome {
    ctx.empty_outcome(shared.time.wall(), shared.time.wall_secs(), interrupted)
}

pub(super) fn teardown_deadline(shared: &SessionShared) -> Ts {
    shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)
}
