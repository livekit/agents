//! `perform_llm_inference`: the words half of a step, streamed as segments, with its retries.

use super::pipeline_reply::teardown_deadline;
use super::tools::ToolStep;
use crate::agent_session::DriverError;
use crate::agent_session::SessionShared;
use crate::events::SessionEvent;
use crate::generation::{Attempt, Retry};
use crate::generation::{GenEvent, ReplyContext, ReplySource};
use crate::speech_handle::SpeechHandle;
use crate::trace::{FailureClass, GenerationStep, ProviderKind};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::telemetry::provider::{Context, LlmMetricState};
use lk_agents_abi::{ChatRequest, FunctionCall, LlmStream};
use lk_agents_voice::transcription::{Visible, VisibleStream};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Chunks, so a streaming TTS starts on the first token: a segment is `Chunk`+ then `End`. `raw`
/// keeps the tags (TTS, transcript); `text` is for the item.
#[derive(Debug)]
pub(super) enum SegEvent {
    Chunk(Visible),
    End,
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
) -> (Result<LlmGenerationData, DriverError>, ToolStep) {
    let mut tools = ToolStep::new();
    // A closed receiver means the voice gave up; dropping the chunk is right.
    let one_segment = |text: String| {
        if !text.is_empty() {
            let _ = tx.send(SegEvent::Chunk(Visible::of(&text)));
            let _ = tx.send(SegEvent::End);
        }
    };
    match &ctx.source {
        ReplySource::Llm | ReplySource::Realtime { .. } => {
            let data = perform_llm_inference(shared, handle, ctx, &tx, &mut tools).await;
            return (data, tools);
        }
        ReplySource::Text(text) | ReplySource::Audio { text, .. } => one_segment(text.clone()),
        // Taken, not borrowed: no lock across an await, and a second read finds `None`.
        ReplySource::TextStream(rx) => {
            let rx = rx.lock().expect("say stream poisoned").take();
            if let Some(mut rx) = rx {
                while let Some(chunk) = rx.recv().await {
                    one_segment(chunk);
                }
            }
        }
    }
    (Ok(LlmGenerationData::new(shared)), tools)
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
    let req = ChatRequest {
        chat_ctx: ctx.request.chat_ctx.clone(),
        tool_choice: ctx.tool_choice.clone(),
        tools_host: ctx
            .toolset
            .as_ref()
            .map_or_else(|| ctx.providers.toolset.host.clone(), |t| t.host.clone()),
        tools_host_extra: ctx.providers.tools_host_extra.clone(),
        // `plan_request` already applied the on-enter exclusion: the list *this step* may call.
        tools: ctx.request.tools.clone(),
        ..ChatRequest::default()
    };
    let (speech, step) = (handle.id(), handle.num_steps());
    // The engine owns the budget; the request carries the same options.
    let mut attempt = Attempt::new(req.conn_options.clone());
    loop {
        shared.trace(GenerationStep::ChatRequested {
            speech,
            step,
            ctx_hash: crate::generation::stable_hash(&format!("{:?}", req.chat_ctx.items)),
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
                let mut data = LlmGenerationData {
                    started_at,
                    metrics: LlmMetricState::for_host(started_at),
                    ..LlmGenerationData::new(shared)
                };
                // A retried attempt produced nothing, so it dispatched no call.
                let ended =
                    read_llm_stream(shared, handle, ctx, &mut stream, tx, tools, &mut data).await;
                let class = ended.as_ref().err().map(|(error, _)| FailureClass::of(error));
                shared.report(GenEvent::ChatFinished { speech, step, error: class });
                if ended.is_ok() {
                    data.report(shared, llm.as_ref(), handle);
                }
                // Always, on every path: dropping does not cancel a host stream, and `close`
                // consumes it.
                let _ = stream.close(teardown_deadline(shared)).await;
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
        match retry(shared, handle, &mut attempt, ProviderKind::Llm, &error).await {
            Retried::Again => continue,
            // What an interrupted retry reports: this stream's metrics, or none if it never opened.
            Retried::Interrupted => {
                return Ok(data.unwrap_or_else(|| LlmGenerationData::new(shared)));
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
    // Segments are flush-delimited (`_produce_segments`). A segment opens on its first non-empty
    // delta, so no empty segment ever reaches the TTS.
    let mut open = false;
    let mut produced = false;
    let mut tokens = 0u64;
    // One splitter per segment: a tag straddling two deltas is held until it closes.
    let mut visible = VisibleStream::new();
    let end_segment = |open: &mut bool, visible: &mut VisibleStream| {
        if std::mem::replace(open, false) {
            let tail = std::mem::replace(visible, VisibleStream::new()).flush();
            if !tail.text.is_empty() {
                let _ = tx.send(SegEvent::Chunk(tail));
            }
            let _ = tx.send(SegEvent::End);
        }
    };
    let ended = loop {
        // Interruption is checked between chunks. A host stream's `next()` is never raced; a
        // native one's is, so a speech cut at `Closing` does not hold its stream open.
        if handle.interrupted() {
            break Ok(());
        }
        let next = if stream.cancel_safe() {
            match until_interrupted(handle, stream.next()).await {
                Some(next) => next,
                None => break Ok(()),
            }
        } else {
            stream.next().await
        };
        let chunk = match next {
            Ok(None) => break Ok(()),
            Err(error) => break Err((error, produced)),
            Ok(Some(chunk)) => chunk,
        };
        produced |= chunk.has_response();
        if let Some(delta) = &chunk.delta {
            // Tool calls before the content guard: such a chunk has no content.
            for raw in &delta.tool_calls {
                tools.dispatch(shared, handle, ctx, mint_call(shared, raw));
            }
            if let Some(text) = &delta.content {
                tokens += 1;
                if !text.is_empty() {
                    open = true;
                    let _ = tx.send(SegEvent::Chunk(visible.push(text)));
                }
            }
        }
        // After the calls are minted: their stamps come first.
        data.metrics.observe(&chunk, || shared.time.wall_secs());
        // `FlushSentinel` after the same chunk's content, which belongs to the closing segment.
        if chunk.flush {
            end_segment(&mut open, &mut visible);
        }
    };
    // A segment left open when the stream ends, for any reason, is still a segment.
    end_segment(&mut open, &mut visible);
    let elapsed = shared.time.wall_secs() - data.started_at;
    // `Clock::wall` is strictly increasing, so the guard only rules out "no interval".
    data.tps = (elapsed > 0.0 && tokens > 0).then(|| tokens as f64 / elapsed);
    ended
}

/// `_LLMGenerationData` plus `LLMStream._metrics_monitor_task`'s fold.
pub(super) struct LlmGenerationData {
    pub(super) tps: Option<f64>,
    pub(super) started_at: f64,
    pub(super) metrics: LlmMetricState,
}

impl LlmGenerationData {
    /// No time-to-first-token and no rate: `0.0` would skew averages.
    fn new(shared: &SessionShared) -> Self {
        let started_at = shared.time.wall_secs();
        LlmGenerationData { tps: None, started_at, metrics: LlmMetricState::for_host(started_at) }
    }

    /// One `LLMMetrics` per request that ended cleanly having yielded a chunk.
    fn report(&self, shared: &SessionShared, llm: &dyn lk_agents_abi::Llm, handle: &SpeechHandle) {
        let stamp = || {
            let (id, now) = (llm.id(), shared.time.wall_secs());
            let context = Context {
                timestamp: now,
                label: id.label.clone(),
                cancelled: handle.interrupted(),
                metadata: Some(id.metadata()),
            };
            (now, context)
        };
        if let Some(mut record) = self.metrics.finish(false, stamp) {
            record.speech_id = Some(handle.speech_id().to_owned());
            report_metrics(shared, handle, AgentMetrics::Llm(record));
        }
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
) -> Retried {
    let decision = attempt.failed(error.retryable());
    // Every failed attempt is the reference's `error` event: `recoverable` while retried, terminal
    // when the budget gives up.
    shared.report(GenEvent::ProviderFailed {
        speech: handle.id(),
        leg,
        recoverable: matches!(decision, Retry::After { .. }),
        error: Box::new(error.clone()),
    });
    let Retry::After { attempt: failed, after } = decision else { return Retried::GiveUp };
    let at = shared.time.now().saturating_add(after);
    shared.trace(GenerationStep::RetryArmed { speech: handle.id(), leg, attempt: failed, at });
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
    tokio::pin!(fut);
    tokio::select! {
        biased;
        () = handle.wait_for_interrupted() => None,
        v = &mut fut => Some(v),
    }
}

pub(super) fn report_metrics(shared: &SessionShared, handle: &SpeechHandle, record: AgentMetrics) {
    let ev =
        SessionEvent::MetricsCollected { metrics: Box::new(record), speech: Some(handle.id()) };
    shared.report(GenEvent::Emit(Box::new(ev)));
}
