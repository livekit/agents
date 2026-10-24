//! `_pipeline_reply_task`: the one task every speech runs, its steps and their follow-ups.

use super::forward::{Voice, voice_leg};
use super::llm::{SegEvent, produce_segments};
use super::tools::run_tools;
use crate::agent_session::DriverError;
use crate::agent_session::SessionShared;
use crate::generation::realtime_generation::realtime_generation_task;
use crate::generation::{
    FinishInputs, RequestInputs, TurnAccumulator, TurnOutcome, TurnStats, plan_request,
};
use crate::generation::{GenEvent, ReplyContext, ReplySource};
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::NextStep;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{ChatItem, ChatMessage, Ts};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// How long a follow-up waits for the loop to commit this step's items before planning against
/// the frame plus `pending`.
const FRAME_SYNC: Duration = Duration::from_secs(1);

/// `_pipeline_reply_task`, and **the one task every speech runs** (`say` and realtime generations
/// too). Spawned through the injected [`Spawner`](lk_agents_abi::Spawner), never `tokio::spawn`.
pub async fn pipeline_reply_task(
    shared: Arc<SessionShared>,
    handle: Arc<SpeechHandle>,
    ctx: ReplyContext,
) {
    let speech = handle.id();
    let error = match run(&shared, &handle, ctx).await {
        Ok(Finished { outcome, interrupted }) => {
            shared.report(GenEvent::Done { speech, outcome: Box::new(outcome), interrupted });
            // Releases the scheduling loop's wait. A never-authorized reply has no cell to mark.
            mark_generation_done(&handle);
            handle.mark_done(None);
            return;
        }
        Err(DriverError::Provider(error)) => error,
        // Reachable from the host (`Agent(llm=None)` then `generate_reply()`).
        Err(DriverError::Missing(what)) => {
            fatal(format!("cannot generate a reply: the session has no {what}"))
        }
        Err(other) => {
            debug_assert!(false, "pipeline_reply_task: {other}");
            handle.mark_done(None);
            return;
        }
    };
    // **Moved, never formatted**: only the binding can release a `ProviderError::Host`'s side-table
    // entry. Not marked here: the loop's `Failed` arm moves the error onto the handle.
    shared.report(GenEvent::Failed { speech, error: Box::new(error) });
}

pub(crate) struct Finished {
    pub(crate) outcome: TurnOutcome,
    pub(crate) interrupted: bool,
}

/// A never-authorized reply has no cell to mark.
fn mark_generation_done(handle: &SpeechHandle) {
    match handle.mark_generation_done() {
        Ok(()) | Err(crate::speech_handle::SpeechError::NoGenerationToMark) => {}
        Err(e) => debug_assert!(false, "mark_generation_done: {e}"),
    }
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

/// A tool reply is the same turn on the same handle, so a barge-in during a tool call stops what is
/// still talking.
async fn run(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    mut ctx: ReplyContext,
) -> Result<Finished, DriverError> {
    if let Some(replan) = ctx.replan.take() {
        ctx.request = replan_after(shared, handle, replan).await;
    }
    // The last step spoken on the realtime path, kept in case its follow-up never arrives.
    let mut last: Option<Finished> = None;
    loop {
        let outcome = match &ctx.source {
            // Borrowed: the channel outlives the step, because the follow-up arrives on it.
            ReplySource::Realtime { events, wake } => {
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
                ctx = follow_up(shared, handle, &ctx, spoken_step, request, pending).await;
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
        request: crate::tool_executor::ReplyRequest,
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
        ctx,
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
    ctx: &ReplyContext,
    acc: TurnAccumulator,
    interrupted: bool,
    stats: TurnStats,
    has_tool_output: bool,
    tools_running: bool,
    provider_request_ids: Option<Vec<String>>,
) -> Finished {
    let mut outcome = acc.finish(FinishInputs {
        stopped_speaking_at: shared.time.wall_secs(),
        interrupted,
        stats,
        message_id: ctx.message_id.clone(),
    });
    if let Some(msg) = outcome.assistant_message.as_mut() {
        msg.metrics.provider_request_ids = provider_request_ids;
    }
    mark_generation_done(handle);
    let item = outcome.assistant_message.clone();
    step_spoken(shared, handle, interrupted, has_tool_output, tools_running, item);
    Finished { outcome, interrupted }
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
            mark_generation_done(handle);
            step_spoken(shared, handle, true, !tools.is_empty(), false, None);
        }
        let _ = run_tools(shared, handle, ctx, tools, true).await;
        let outcome = empty_outcome(shared, ctx, true);
        return Ok(StepOutcome::Done(Finished { outcome, interrupted: true }));
    };
    // The step's item rides with the step: built from what was heard, committed by the loop. The
    // speech's interruption decides `tools_running`: a step whose audio was cut still has tools.
    let stats = TurnStats {
        llm_ttft: llm_gen_data.metrics.ttft(),
        llm_tps: llm_gen_data.tps,
        llm_started_at: Some(llm_gen_data.started_at),
    };
    let has_tools = !tools.is_empty();
    let tools_running = has_tools && !handle.interrupted_speech();
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
        // The swap is the host's; the driver reports it, and then still runs `follow_up`.
        NextStep::Handoff { agent, follow_up } => {
            shared.report(GenEvent::Handoff { speech: handle.id(), agent });
            follow_up
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
    request: crate::tool_executor::ReplyRequest,
    pending: Vec<ChatItem>,
) -> ReplyContext {
    // Once the loop has performed this step's `ToolsFinished` (and every command queued before it).
    let key = (handle.id(), spoken_step);
    let frame =
        wait_frame(shared, handle, FRAME_SYNC, |f| f.tools_committed.contains_key(&key)).await;
    // A draining activity's reply offers no tools.
    let mut request = request;
    if frame.tools_committed.get(&key) == Some(&true) {
        request.tool_choice = lk_agents_abi::ToolChoice::Forbidden;
    }
    let instructions = frame.instructions.render(Some(lk_agents_abi::Modality::Audio));
    let planned = plan_request(RequestInputs {
        record: &frame.record,
        pending: &pending,
        new_message: None,
        instructions: &instructions,
        turn_instructions: None,
        tools: &prev.request.tools,
        // The on-enter exclusion was applied to `prev.request.tools` already.
        on_enter: false,
        running_tool_calls: &[],
        modality: lk_agents_abi::Modality::Audio,
        audio_output_enabled: prev.request.tts_enabled,
        text_output_enabled: prev.request.text_output_enabled,
        now: shared.time.wall(),
    });
    ReplyContext {
        preemptive_tts: prev.preemptive_tts,
        request: planned,
        replan: None,
        source: ReplySource::Llm,
        // Always concrete on a follow-up: `"none"` or `"auto"`, never `"required"`.
        tool_choice: Some(request.tool_choice),
        frame: prev.frame,
        toolset: prev.toolset.clone(),
        providers: prev.providers.clone(),
        // Carried over when this turn forwarded no text, for `e2e_latency`.
        user_metrics: request.previous_user_metrics.or_else(|| prev.user_metrics.clone()),
        // A fresh id: this step commits its own assistant message.
        message_id: shared.mint_item_id(),
        speech: prev.speech,
    }
}

/// The first request of a reply created behind a playing or cut speech, planned once its item is on
/// the record. Bounded, and cut short by interruption or closing.
async fn replan_after(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    replan: crate::generation::Replan,
) -> crate::generation::Request {
    let crate::generation::Replan { after, spec } = replan;
    let bound = crate::speech_handle::INTERRUPTION_TIMEOUT + Duration::from_secs(1);
    let frame = wait_frame(shared, handle, bound, |f| f.finished.contains(&after)).await;
    let instructions = frame.instructions.render(Some(lk_agents_abi::Modality::Audio));
    spec.plan(&frame.record, &instructions, shared.time.wall())
}

/// Cut short by interruption or closing, and by `bound` on the clock.
async fn wait_frame(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    bound: Duration,
    ready: impl Fn(&crate::agent_session::FrameView) -> bool,
) -> crate::agent_session::FrameView {
    let mut rx = shared.frame.subscribe();
    let mut closing = shared.closing.subscribe();
    let published = async {
        while !ready(&rx.borrow_and_update()) {
            if rx.changed().await.is_err() {
                return;
            }
        }
    };
    tokio::select! {
        biased;
        () = handle.wait_for_interrupted() => {}
        _ = closing.wait_for(|c| *c) => {}
        _ = shared.time.timeout(bound, published) => {}
    }
    shared.frame.borrow().clone()
}

pub(super) fn empty_outcome(
    shared: &Arc<SessionShared>,
    ctx: &ReplyContext,
    interrupted: bool,
) -> TurnOutcome {
    TurnAccumulator::new(shared.time.wall(), ctx.user_metrics.clone()).finish(FinishInputs {
        stopped_speaking_at: shared.time.wall_secs(),
        interrupted,
        stats: TurnStats::default(),
        message_id: ctx.message_id.clone(),
    })
}

pub(super) fn teardown_deadline(shared: &SessionShared) -> Ts {
    shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)
}

pub(crate) fn fatal(message: impl Into<String>) -> ProviderError {
    ProviderError::Local(ApiError::new(message).with_retryable(false))
}
