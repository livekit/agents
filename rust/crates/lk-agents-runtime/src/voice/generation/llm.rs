//! `perform_llm_inference`: the words half of a step, streamed as segments, with its retries.

use super::tools::ToolStep;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::{ChatRequest, FunctionCall, LlmStream};
use lk_agents_core::telemetry::provider::Context;
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::agent_session::SessionShared;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::Attempt;
pub(super) use lk_agents_voice::generation::llm::LlmGenerationData;
use lk_agents_voice::generation::llm::{LlmRead, WordEffect, before_read, retry_after};
use lk_agents_voice::generation::{GenEvent, ReplyContext, ReplySource};
use lk_agents_voice::speech_handle::SpeechHandle;
use lk_agents_voice::trace::{FailureClass, GenerationStep, ProviderKind};
use lk_agents_voice::transcription::Visible;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Chunks, so a streaming TTS starts on the first token: a segment is `Chunk`+ then `End`. `raw`
/// keeps the tags (TTS, transcript); `text` is for the item.
#[derive(Debug)]
pub(super) enum SegEvent {
    Chunk(Visible),
    End,
}

impl From<WordEffect> for SegEvent {
    fn from(effect: WordEffect) -> Self {
        match effect {
            WordEffect::Chunk(words) => Self::Chunk(words),
            WordEffect::End => Self::End,
        }
    }
}

pub(super) fn extend(acc: &mut Visible, more: &Visible) {
    acc.raw.push_str(&more.raw);
    acc.text.push_str(&more.text);
    acc.tags.extend(more.tags.iter().cloned());
}

/// `_produce_segments`. The dispatched tools come back on every path, failure included.
pub(super) async fn produce_segments(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    tx: mpsc::UnboundedSender<SegEvent>,
    speeches: &crate::voice::SpeechTasks,
) -> (Result<LlmGenerationData, DriverError>, ToolStep) {
    let mut tools = ToolStep::new(speeches.clone());
    // A closed receiver means the voice gave up; dropping the chunk is right.
    let one_segment = |text: String| {
        if !text.is_empty() {
            let _ = tx.send(SegEvent::Chunk(Visible::of(&text)));
            let _ = tx.send(SegEvent::End);
        }
    };
    match &ctx.source {
        ReplySource::Llm | ReplySource::Realtime => {
            let data = perform_llm_inference(shared, handle, ctx, &tx, &mut tools).await;
            return (data, tools);
        }
        ReplySource::Text(text) => one_segment(text.clone()),
        // The caller's frames are the segment, with words or without (`_tts_task` forwards
        // `audio` whatever the text).
        ReplySource::Audio { text, .. } => {
            let _ = tx.send(SegEvent::Chunk(Visible::of(text)));
            let _ = tx.send(SegEvent::End);
        }
        // Taken, not borrowed: no lock across an await, and a second read finds `None`.
        ReplySource::TextStream(rx) => {
            let rx = rx.lock().expect("say stream poisoned").take();
            if let Some(mut rx) = rx {
                loop {
                    let Some(chunk) = until_stopped(tx.closed(), rx.recv()).await.flatten() else {
                        break;
                    };
                    one_segment(chunk);
                }
            }
        }
    }
    (Ok(LlmGenerationData::new(shared.time.wall_secs())), tools)
}

/// `perform_llm_inference`, retried per the request's budget while nothing has been produced.
async fn perform_llm_inference(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    tx: &mpsc::UnboundedSender<SegEvent>,
    tools: &mut ToolStep,
) -> Result<LlmGenerationData, DriverError> {
    // Held for the step: a mid-reply handoff must not move the model under a request on the wire.
    let llm = ctx.providers.llm.clone().ok_or(DriverError::Missing("llm"))?;
    let mut chat_ctx = ctx.request.chat_ctx.clone();
    // Rendered per request, so a tool's state change reaches the step after it.
    if let Some(dynamic) = &ctx.providers.dynamic_instructions {
        let speech = handle.object();
        match dynamic.render(&speech).await {
            Ok(text) => lk_agents_voice::generation::append_instructions(&mut chat_ctx, &text),
            Err(e) => {
                log::error!(target: "livekit.agents", "the agent's instructions functions failed; the static instructions are sent: {e}")
            }
        }
    }
    let req = ChatRequest {
        chat_ctx,
        tool_choice: ctx.tool_choice.clone(),
        tools_host: ctx
            .toolset
            .as_ref()
            .map_or_else(|| ctx.providers.toolset.host.clone(), |t| t.host.clone()),
        tools_host_extra: ctx.providers.tools_host_extra.clone(),
        // `plan_request` already applied the on-enter exclusion: the list *this step* may call.
        tools: ctx.request.tools.clone(),
        conn_options: shared.conn.llm.unwrap_or_default(),
        ..ChatRequest::default()
    };
    let (speech, step) = (handle.id(), handle.num_steps());
    // The engine owns the budget; the request carries the same options.
    let mut attempt = Attempt::new(req.conn_options.clone());
    loop {
        shared.trace(GenerationStep::ChatRequested {
            speech,
            step,
            ctx_hash: lk_agents_voice::generation::stable_hash(&format!(
                "{:?}",
                req.chat_ctx.items
            )),
            tools: req.tools.iter().map(|t| t.name.clone()).collect(),
            instructions_hash: ctx.request.instructions_hash,
            ctx_ids: req.chat_ctx.items.iter().map(|it| it.id().to_owned()).collect(),
            tool_choice: req.tool_choice.clone(),
        });
        let started_at = shared.time.wall_secs();
        // One `ChatFinished` per `ChatRequested`, a request that never became a stream included.
        let (error, data) = match llm.chat(&req).await {
            Err(error) => {
                let class = Some(FailureClass::of(&error));
                shared.report(GenEvent::ChatFinished { speech, step, error: class });
                (error, None)
            }
            Ok(mut stream) => {
                let mut data = LlmGenerationData::new(started_at);
                // A retried attempt produced nothing, so it dispatched no call.
                let ended =
                    read_llm_stream(shared, handle, ctx, &mut stream, tx, tools, &mut data).await;
                let class = ended.as_ref().err().map(|(error, _)| FailureClass::of(error));
                shared.report(GenEvent::ChatFinished { speech, step, error: class });
                report_llm_metrics(&data, ended.is_err(), shared, llm.as_ref(), handle);
                // Always, on every path: dropping does not cancel a host stream, and `close`
                // consumes it.
                let _ = stream
                    .close(shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
                    .await;
                match ended {
                    Ok(()) => return Ok(data),
                    Err((error, produced)) => {
                        if produced {
                            attempt.produced_output();
                        }
                        (error, Some(data))
                    }
                }
            }
        };
        let failed = |e, r| llm.failed(e, r);
        match retry(shared, handle, &mut attempt, ProviderKind::Llm, &error, &failed).await {
            Retried::Again => continue,
            // What an interrupted retry reports: this stream's metrics, or none if it never opened.
            Retried::Interrupted => {
                return Ok(data.unwrap_or_else(|| LlmGenerationData::new(shared.time.wall_secs())));
            }
            Retried::GiveUp => return Err(DriverError::from(error)),
        }
    }
}

/// Tool calls are dispatched on arrival. `Err` carries whether output came first, which forbids a
/// retry.
async fn read_llm_stream(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    stream: &mut LlmStream,
    tx: &mpsc::UnboundedSender<SegEvent>,
    tools: &mut ToolStep,
    data: &mut LlmGenerationData,
) -> Result<(), (ProviderError, bool)> {
    let ended = loop {
        let next = match before_read(handle.interrupted(), tx.is_closed(), stream.cancel_safe()) {
            LlmRead::End => break Ok(()),
            LlmRead::CancellableProvider => {
                let polled = until_stopped(tx.closed(), until_interrupted(handle, stream.next()))
                    .await
                    .flatten();
                match polled {
                    Some(next) => next,
                    None => break Ok(()),
                }
            }
            LlmRead::Provider => stream.next().await,
        };
        let chunk = match next {
            Ok(None) => break Ok(()),
            Err(error) => break Err((error, data.produced)),
            Ok(Some(chunk)) => chunk,
        };
        if let Some(delta) = &chunk.delta {
            for raw in &delta.tool_calls {
                tools.dispatch(shared, handle, ctx, mint_call(shared, raw));
            }
        }
        for effect in data.chunk(&chunk, shared.time.wall_secs()) {
            let _ = tx.send(effect.into());
        }
    };
    for effect in data.finish_stream() {
        let _ = tx.send(effect.into());
    }
    ended
}

fn report_llm_metrics(
    data: &LlmGenerationData,
    failed: bool,
    shared: &SessionShared,
    llm: &dyn lk_agents_abi::Llm,
    handle: &SpeechHandle,
) {
    let (id, now) = (llm.id(), shared.time.wall_secs());
    let context = Context {
        timestamp: now,
        label: id.label.clone(),
        cancelled: handle.interrupted(),
        metadata: Some(id.metadata()),
    };
    if let Some(record) = data.collect_metrics(failed, now, context, handle.speech_id()) {
        report_metrics(shared, handle, AgentMetrics::Llm(record));
    }
}

/// Minted here, because the pure tier has no ids.
fn mint_call(shared: &Arc<SessionShared>, raw: &lk_agents_abi::FunctionToolCall) -> FunctionCall {
    FunctionCall {
        id: shared.mint_item_id(),
        call_id: raw.call_id.clone(),
        arguments: raw.arguments.clone(),
        name: raw.name.clone(),
        created_at: shared.time.wall(),
        extra: raw.extra.clone().unwrap_or_default(),
        group_id: None,
    }
}

pub(super) enum Retried {
    Again,
    /// The speech was interrupted during the wait: stop, and report what was heard.
    Interrupted,
    GiveUp,
}

pub(super) async fn retry(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    attempt: &mut Attempt,
    leg: ProviderKind,
    error: &ProviderError,
    provider: &(dyn Fn(ProviderError, bool) + Sync),
) -> Retried {
    let wait = retry_after(attempt, leg, error, shared.time.now());
    provider(error.clone(), wait.is_some());
    shared.report(GenEvent::ProviderFailed {
        speech: handle.id(),
        leg,
        recoverable: wait.is_some(),
        error: Box::new(error.clone()),
    });
    let Some(wait) = wait else { return Retried::GiveUp };
    let at = wait.at;
    shared.trace(GenerationStep::RetryArmed {
        speech: handle.id(),
        leg,
        attempt: wait.failed_attempt,
        at,
    });
    // Raced against the interruption; a Rust-native wait, safe to drop.
    match until_interrupted(handle, shared.time.wait_until(at)).await {
        Some(()) => Retried::Again,
        None => Retried::Interrupted,
    }
}

pub(crate) async fn until_interrupted<T>(
    handle: &SpeechHandle,
    fut: impl std::future::Future<Output = T>,
) -> Option<T> {
    until_stopped(handle.wait_for_interrupted(), fut).await
}

/// Cancellation-safe waits only: finishing or dropping this future joins no host operation.
async fn until_stopped<T>(
    stopped: impl std::future::Future<Output = ()>,
    work: impl std::future::Future<Output = T>,
) -> Option<T> {
    use std::future::poll_fn;
    use std::task::Poll;
    let (mut stopped, mut work) = (std::pin::pin!(stopped), std::pin::pin!(work));
    poll_fn(|cx| {
        if stopped.as_mut().poll(cx).is_ready() {
            return Poll::Ready(None);
        }
        work.as_mut().poll(cx).map(Some)
    })
    .await
}

pub(super) fn report_metrics(shared: &SessionShared, handle: &SpeechHandle, record: AgentMetrics) {
    let ev =
        SessionEvent::MetricsCollected { metrics: Box::new(record), speech: Some(handle.id()) };
    shared.report(GenEvent::Emit(Box::new(ev)));
}
