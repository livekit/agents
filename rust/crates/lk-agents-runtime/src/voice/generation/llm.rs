//! `perform_llm_inference`: the words half of a step, streamed as segments, with its retries.

use super::ReplyExecution;
use super::tools::ToolStep;
use crate::telemetry;
use crate::voice::session::SessionShared;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::{ChatRequest, FunctionCall, LlmStream};
use lk_agents_core::attempts::{AttemptStep, Attempts};
use lk_agents_core::usage::ModelKind;
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::SpeechSegmentEvent;
pub(super) use lk_agents_voice::generation::machine::llm::LlmGenerationData;
use lk_agents_voice::generation::machine::llm::{LlmRead, before_read};
use lk_agents_voice::generation::{GenEvent, ReplySource};
use lk_agents_voice::trace::{FailureClass, GenerationStep, ProviderKind};
use lk_agents_voice::transcription::Visible;
use std::sync::Arc;
use tokio::sync::mpsc;

pub(super) fn extend(acc: &mut Visible, more: &Visible) {
    acc.raw.push_str(&more.raw);
    acc.text.push_str(&more.text);
    acc.tags.extend(more.tags.iter().cloned());
}

/// `_produce_segments`. The dispatched tools come back on every path, failure included.
pub(super) async fn produce_segments(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyExecution,
    tx: mpsc::UnboundedSender<SpeechSegmentEvent>,
    speeches: &crate::voice::SpeechTasks,
) -> (Result<LlmGenerationData, DriverError>, ToolStep) {
    let mut tools = ToolStep::new(speeches.clone());
    // A closed receiver means the voice gave up; dropping the chunk is right.
    let one_segment = |text: String| {
        if !text.is_empty() {
            let _ = tx.send(SpeechSegmentEvent::Chunk(Visible::of(&text)));
            let _ = tx.send(SpeechSegmentEvent::End);
        }
    };
    match &ctx.plan.source {
        ReplySource::Llm | ReplySource::Realtime => {
            let data = perform_llm_inference(shared, handle, ctx, &tx, &mut tools).await;
            return (data, tools);
        }
        ReplySource::Text(text) => one_segment(text.clone()),
        // The caller's frames are the segment, with words or without (`_tts_task` forwards
        // `audio` whatever the text).
        ReplySource::Audio { text, .. } => {
            let _ = tx.send(SpeechSegmentEvent::Chunk(Visible::of(text)));
            let _ = tx.send(SpeechSegmentEvent::End);
        }
        // Taken off the host's handle: a second read finds `None`. Read until
        // the voice gives up or the speech is interrupted (`_tts_task` cancels its forwarding).
        ReplySource::TextStream => {
            if let Some(mut rx) = handle.take_text_stream() {
                loop {
                    let stopped = async {
                        tokio::select! {
                            biased;
                            () = handle.wait_for_interrupted() => {}
                            () = tx.closed() => {}
                        }
                    };
                    let Some(chunk) = until_stopped(stopped, rx.recv()).await.flatten() else {
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
    ctx: &ReplyExecution,
    tx: &mpsc::UnboundedSender<SpeechSegmentEvent>,
    tools: &mut ToolStep,
) -> Result<LlmGenerationData, DriverError> {
    // Held for the step: a mid-reply handoff must not move the model under a request on the wire.
    let llm = ctx.resources.llm.clone().ok_or(DriverError::Missing("llm"))?;
    let mut chat_ctx = ctx.plan.request.chat_ctx.clone();
    // Rendered per request, so a tool's state change reaches the step after it.
    if let Some(dynamic) = &ctx.resources.dynamic_instructions {
        let speech = handle.object();
        match dynamic.render(speech).await {
            Ok(text) => lk_agents_voice::generation::append_instructions(&mut chat_ctx, &text),
            Err(e) => {
                log::error!(target: "livekit.agents", "the agent's instructions functions failed; the static instructions are sent: {e}")
            }
        }
    }
    let mut req = ChatRequest {
        chat_ctx,
        tool_choice: ctx.plan.tool_choice.clone(),
        tools_host: ctx
            .plan
            .toolset
            .as_ref()
            .map_or_else(|| ctx.resources.toolset.host.clone(), |t| t.host.clone()),
        tools_host_extra: ctx.resources.tools_host_extra.clone(),
        // `plan_request` already applied the on-enter exclusion: the list *this step* may call.
        tools: ctx.plan.request.tools.clone(),
        conn_options: shared.conn.llm.unwrap_or_default(),
        ..ChatRequest::default()
    };
    let (speech, step) = (handle.id(), ctx.plan.step);
    // The engine owns the budget; the request carries the same options.
    let mut attempts = Attempts::retrying(&req.conn_options, shared.time.now());
    // The step's `llm_node`, one `llm_request` in it, its attempts each an `llm_request_run`. A
    // registered `llm_node` runs under the node span and makes its own request, if any.
    let node = LlmNodeTrace::start(shared, ctx.span, &req);
    let mut trace = telemetry::LlmTrace::start(node.cx(shared), llm.as_ref(), &mut req);
    loop {
        let attempt = trace.attempt();
        shared.trace(GenerationStep::ChatRequested {
            speech,
            step,
            ctx_hash: lk_agents_voice::generation::stable_hash(&format!(
                "{:?}",
                req.chat_ctx.items
            )),
            tools: req.tools.iter().map(|t| t.name.clone()).collect(),
            instructions_hash: ctx.plan.request.instructions_hash,
            ctx_ids: req.chat_ctx.items.iter().map(|it| it.id().to_owned()).collect(),
            tool_choice: req.tool_choice.clone(),
        });
        let started_at = shared.time.wall_secs();
        let id = llm.id();
        shared.report(GenEvent::RequestStarted {
            speech,
            kind: ModelKind::Llm,
            provider: id.provider,
            model: id.model,
        });
        // One `LlmRequestFinished` per `ChatRequested`, a request that never became a stream included.
        let (error, data) = match llm.chat(&req).await {
            Err(error) => {
                trace.end_attempt(attempt, Some(&error));
                let class = Some(FailureClass::of(&error));
                shared.report(GenEvent::LlmRequestFinished { speech, step, error: class });
                (error, None)
            }
            Ok(mut stream) => {
                let mut data = LlmGenerationData::new(started_at);
                // A retried attempt produced nothing, so it dispatched no call.
                let ended = read_llm_stream(
                    shared,
                    handle,
                    ctx,
                    &mut stream,
                    tx,
                    tools,
                    &mut data,
                    &mut trace,
                )
                .await;
                trace.end_attempt(attempt, ended.as_ref().err().map(|(error, _)| error));
                let class = ended.as_ref().err().map(|(error, _)| FailureClass::of(error));
                shared.report(GenEvent::LlmRequestFinished { speech, step, error: class });
                report_llm_metrics(&data, ended.is_err(), shared, llm.as_ref(), handle);
                // Always, on every path: dropping does not cancel a host stream, and `close`
                // consumes it.
                let _ = stream
                    .close(shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
                    .await;
                match ended {
                    Ok(()) => {
                        node.end(&trace, llm.as_ref(), None);
                        trace.end(handle.interrupted(), Some(handle.speech_id()));
                        return Ok(data);
                    }
                    Err((error, produced)) => {
                        if produced {
                            attempts.delivered();
                        }
                        (error, Some(data))
                    }
                }
            }
        };
        let failed = |e, r| llm.failed(e, r);
        match retry(shared, handle, &mut attempts, ProviderKind::Llm, &error, &failed).await {
            Retried::Again => continue,
            // What an interrupted retry reports: this stream's metrics, or none if it never opened.
            Retried::Interrupted => {
                node.end(&trace, llm.as_ref(), None);
                trace.end(true, Some(handle.speech_id()));
                return Ok(data.unwrap_or_else(|| LlmGenerationData::new(shared.time.wall_secs())));
            }
            Retried::GiveUp => {
                trace.fail(&error);
                node.end(&trace, llm.as_ref(), Some(&error));
                return Err(DriverError::from(error));
            }
        }
    }
}

/// Tool calls are dispatched on arrival. `Err` carries whether output came first, which forbids a
/// retry.
#[allow(clippy::too_many_arguments)]
async fn read_llm_stream(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyExecution,
    stream: &mut LlmStream,
    tx: &mpsc::UnboundedSender<SpeechSegmentEvent>,
    tools: &mut ToolStep,
    data: &mut LlmGenerationData,
    trace: &mut telemetry::LlmTrace,
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
        trace.observe(&chunk);
        if let Some(delta) = &chunk.delta {
            for raw in &delta.tool_calls {
                tools.dispatch(shared, handle, ctx, mint_call(shared, raw));
            }
        }
        for effect in data.chunk(&chunk, shared.time.wall_secs()) {
            let _ = tx.send(effect);
        }
    };
    for effect in data.finish_stream() {
        let _ = tx.send(effect);
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
    let context = telemetry::Context {
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

/// A failed attempt of a speech leg, decided by the shared attempt policy: on a retry, waits out
/// its backoff, raced against the interruption.
pub(super) async fn retry(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    attempts: &mut Attempts,
    leg: ProviderKind,
    error: &ProviderError,
    provider: &(dyn Fn(ProviderError, bool) + Sync),
) -> Retried {
    let (now, mut steps) = (shared.time.now(), Vec::new());
    if let Some(attempt) = attempts.current() {
        attempts.failed_error(attempt, error.attempt_error(), now, &mut steps);
        // The failed request is over: nothing of it is left to release.
        attempts.closed(attempt, now, &mut steps);
    }
    let failed = steps.iter().find_map(|step| match step {
        AttemptStep::Retry { attempt, .. } => Some(*attempt),
        _ => None,
    });
    let at = failed.and(attempts.wake_at(None));
    provider(error.clone(), at.is_some());
    shared.report(GenEvent::ProviderFailed {
        speech: handle.id(),
        leg,
        recoverable: at.is_some(),
        error: Box::new(error.clone()),
    });
    let (Some(failed), Some(at)) = (failed, at) else { return Retried::GiveUp };
    shared.trace(GenerationStep::RetryArmed { speech: handle.id(), leg, attempt: failed + 1, at });
    // Raced against the interruption; a Rust-native wait, safe to drop.
    match until_interrupted(handle, shared.time.wait_until(at)).await {
        Some(()) => {
            attempts.wake(at, None, &mut steps);
            Retried::Again
        }
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
    let ev = SessionEvent::MetricsCollected {
        metrics: Box::new(record),
        speech: Some(handle.id()),
        counted: false,
    };
    shared.report(GenEvent::Emit(Box::new(ev)));
}

/// A step's `llm_node`, under the speech's `agent_turn`: the context sent, the tools offered, and
/// the response with its time to first token measured from the node's start. The model and
/// provider are its request's: the LLM's labels only its time to first token measurement.
struct LlmNodeTrace {
    span: Option<telemetry::ScopedSpan<telemetry::LlmNode>>,
    started: u64,
}

impl LlmNodeTrace {
    fn start(shared: &SessionShared, turn: Option<telemetry::SpanRef>, req: &ChatRequest) -> Self {
        let span = turn.map(|turn| {
            let (provider, function): (Vec<_>, Vec<_>) =
                req.tools.iter().partition(|tool| tool.provider_tool.is_some());
            let names = |tools: Vec<&lk_agents_abi::ToolSpec>| {
                tools.into_iter().map(|tool| tool.name.clone()).collect()
            };
            telemetry::start(
                &shared.trace.clone().under::<telemetry::AgentTurn>(turn),
                telemetry::LlmNode {
                    chat_ctx: telemetry::Content(telemetry::Json(telemetry::gen_ai::chat_ctx(
                        &req.chat_ctx.items,
                    ))),
                    function_tools: names(function),
                    provider_tools: names(provider),
                    tool_sets: Vec::new(),
                    input: None,
                },
            )
        });
        LlmNodeTrace { span, started: shared.tracer.now_nanos() }
    }

    /// What the request runs under: the node, or the session outside a turn.
    fn cx<'a>(&'a self, shared: &'a SessionShared) -> &'a telemetry::Cx {
        self.span.as_ref().map_or(&shared.trace, |span| span.cx())
    }

    fn end(
        self,
        trace: &telemetry::LlmTrace,
        llm: &dyn lk_agents_abi::Llm,
        failed: Option<&ProviderError>,
    ) {
        let Some(span) = self.span else { return };
        if let Some(error) = failed {
            telemetry::fail(&span, error);
        }
        let (text, calls) = trace.response();
        let ttft = trace.first_chunk_at().map(|at| at.saturating_sub(self.started) as f64 / 1e9);
        if let Some(ttft) = ttft {
            let model = telemetry::model_labels(&llm.id());
            telemetry::measure(&span.cx(), &telemetry::TurnLlmTtft { model }, ttft);
        }
        telemetry::end_with(
            span,
            telemetry::LlmNodeEnd {
                response_text: telemetry::Content(text.to_owned()),
                function_calls: telemetry::Content(telemetry::Json(serde_json::Value::Array(
                    calls,
                ))),
                ttft: telemetry::Measured(ttft.map(telemetry::Seconds)),
            },
        );
    }
}
