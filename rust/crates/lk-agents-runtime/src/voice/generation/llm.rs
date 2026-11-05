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
    match &ctx.plan.source {
        ReplySource::Llm | ReplySource::Realtime => {
            let inference = LlmInference { shared, handle, ctx, tx: &tx, tools: &mut tools };
            let data = inference.run().await;
            return (data, tools);
        }
        ReplySource::Text(text) => {
            if !text.is_empty() {
                send_segment(&tx, text);
            }
        }
        // The caller's frames are the segment, with words or without (`_tts_task` forwards
        // `audio` whatever the text).
        ReplySource::Audio { text, .. } => send_segment(&tx, text),
        ReplySource::TextStream => forward_text_stream(handle, &tx).await,
    }
    (Ok(LlmGenerationData::new(shared.time.wall_secs(), String::new(), None)), tools)
}

/// One whole segment. A closed receiver means the voice gave up; dropping the chunk is right.
fn send_segment(tx: &mpsc::UnboundedSender<SpeechSegmentEvent>, text: &str) {
    let _ = tx.send(SpeechSegmentEvent::Chunk(Visible::of(text)));
    let _ = tx.send(SpeechSegmentEvent::End);
}

/// The host's text stream, each chunk a segment. Taken off the host's handle: a second read finds
/// `None`. Read until the voice gives up or the speech is interrupted (`_tts_task` cancels its
/// forwarding).
async fn forward_text_stream(
    handle: &SpeechHandle,
    tx: &mpsc::UnboundedSender<SpeechSegmentEvent>,
) {
    let Some(mut rx) = handle.take_text_stream() else { return };
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
        if !chunk.is_empty() {
            send_segment(tx, &chunk);
        }
    }
}

/// The step's request with the agent's instructions rendered for it.
async fn chat_request(
    shared: &SessionShared,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyExecution,
) -> ChatRequest {
    let mut chat_ctx = ctx.plan.request.chat_ctx.clone();
    // Rendered per request, so a tool's state change reaches the step after it.
    if let Some(dynamic) = &ctx.resources.dynamic_instructions {
        let speech = handle.object();
        match dynamic.render(&ctx.turn_cx(shared), speech).await {
            Ok(text) => lk_agents_voice::generation::append_instructions(&mut chat_ctx, &text),
            Err(e) => {
                log::error!(target: "livekit.agents", "the agent's instructions functions failed; the static instructions are sent: {e}")
            }
        }
    }
    ChatRequest {
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
    }
}

/// `perform_llm_inference`: one step's words and tool calls from the model, retried per the
/// request's budget while nothing has been produced.
struct LlmInference<'a> {
    shared: &'a Arc<SessionShared>,
    handle: &'a Arc<SpeechHandle>,
    ctx: &'a ReplyExecution,
    tx: &'a mpsc::UnboundedSender<SpeechSegmentEvent>,
    tools: &'a mut ToolStep,
}

impl LlmInference<'_> {
    async fn run(mut self) -> Result<LlmGenerationData, DriverError> {
        let (shared, handle) = (self.shared, self.handle);
        // Held for the step: a mid-reply handoff must not move the model under a request on the
        // wire.
        let llm = self.ctx.resources.llm.clone().ok_or(DriverError::Missing("llm"))?;
        let req = chat_request(shared, handle, self.ctx).await;
        // The engine owns the budget; the request carries the same options.
        let mut attempts = Attempts::retrying(&req.conn_options, shared.time.now());
        // The step's `llm_node`, one `llm_request` in it, its attempts each an `llm_request_run`.
        // A registered `llm_node` runs under the node span and makes its own request, if any.
        let node = LlmNodeTrace::start(shared, self.ctx.span, &req);
        let mut trace = telemetry::LlmTrace::start(node.cx(shared), llm.as_ref(), &req);
        loop {
            let attempt = trace.attempt();
            let mut data = self.announce_request(&req, llm.as_ref());
            // One `LlmRequestFinished` per `ChatRequested`, a request that never became a stream
            // included.
            let (error, data) = match llm.chat(trace.cx(), &req).await {
                Err(error) => {
                    trace.end_attempt(attempt, Some(&error));
                    self.request_finished(Some(&error));
                    (error, None)
                }
                Ok(mut stream) => {
                    let ended = self.read_stream(&mut stream, &mut data, &mut trace).await;
                    trace.end_attempt(attempt, ended.as_ref().err().map(|(error, _)| error));
                    let ended = self.settle_attempt(stream, &data, ended).await;
                    let Err((error, produced)) = ended else {
                        node.end(&trace, llm.as_ref(), None);
                        trace.end(handle.interrupted(), Some(handle.speech_id()));
                        return Ok(data);
                    };
                    if produced {
                        attempts.delivered();
                    }
                    (error, Some(data))
                }
            };
            let failed = |e, r| llm.failed(e, r);
            match retry(shared, handle, &mut attempts, ProviderKind::Llm, &error, &failed).await {
                Retried::Again => continue,
                // What an interrupted retry reports: this stream's metrics, or none if it never
                // opened.
                Retried::Interrupted => {
                    node.end(&trace, llm.as_ref(), None);
                    trace.end(true, Some(handle.speech_id()));
                    let empty =
                        || LlmGenerationData::new(shared.time.wall_secs(), String::new(), None);
                    return Ok(data.unwrap_or_else(empty));
                }
                Retried::GiveUp => {
                    trace.fail(&error);
                    node.end(&trace, llm.as_ref(), Some(&error));
                    return Err(DriverError::from(error));
                }
            }
        }
    }

    /// Journals the request and reports its start: the attempt's data, starting now.
    fn announce_request(
        &self,
        req: &ChatRequest,
        llm: &dyn lk_agents_abi::Llm,
    ) -> LlmGenerationData {
        let (shared, speech, step) = (self.shared, self.handle.id(), self.ctx.plan.step);
        shared.trace(GenerationStep::ChatRequested {
            speech,
            step,
            ctx_hash: lk_agents_voice::generation::stable_hash(&format!(
                "{:?}",
                req.chat_ctx.items
            )),
            tools: req.tools.iter().map(|t| t.name.clone()).collect(),
            instructions_hash: self.ctx.plan.request.instructions_hash,
            ctx_ids: req.chat_ctx.items.iter().map(|it| it.id().to_owned()).collect(),
            tool_choice: req.tool_choice.clone(),
        });
        let started_at = shared.time.wall_secs();
        let id = llm.id();
        let (label, metadata) = (id.label.clone(), Some(id.metadata()));
        shared.report(GenEvent::RequestStarted {
            speech,
            kind: ModelKind::Llm,
            provider: id.provider,
            model: id.model,
        });
        LlmGenerationData::new(started_at, label, metadata)
    }

    fn request_finished(&self, error: Option<&ProviderError>) {
        let (speech, step) = (self.handle.id(), self.ctx.plan.step);
        let error = error.map(FailureClass::of);
        self.shared.report(GenEvent::LlmRequestFinished { speech, step, error });
    }

    /// A read attempt reported and its stream closed. A retried attempt produced nothing, so it
    /// dispatched no call.
    async fn settle_attempt(
        &self,
        stream: LlmStream,
        data: &LlmGenerationData,
        ended: Result<(), (ProviderError, bool)>,
    ) -> Result<(), (ProviderError, bool)> {
        self.request_finished(ended.as_ref().err().map(|(error, _)| error));
        report_llm_metrics(data, ended.is_err(), self.shared, self.handle);
        // Always, on every path: dropping does not cancel a host stream, and `close` consumes it.
        let deadline = self.shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let _ = stream.close(deadline).await;
        ended
    }

    /// Tool calls are dispatched on arrival. `Err` carries whether output came first, which
    /// forbids a retry.
    async fn read_stream(
        &mut self,
        stream: &mut LlmStream,
        data: &mut LlmGenerationData,
        trace: &mut telemetry::LlmTrace,
    ) -> Result<(), (ProviderError, bool)> {
        let (shared, handle, tx) = (self.shared, self.handle, self.tx);
        let ended = loop {
            let read = before_read(handle.interrupted(), tx.is_closed(), stream.cancel_safe());
            let next = match read {
                LlmRead::End => break Ok(()),
                LlmRead::CancellableProvider => {
                    let polled =
                        until_stopped(tx.closed(), until_interrupted(handle, stream.next())).await;
                    let Some(next) = polled.flatten() else { break Ok(()) };
                    next
                }
                LlmRead::Provider => stream.next().await,
            };
            let chunk = match next {
                Ok(None) => break Ok(()),
                Err(error) => break Err((error, data.produced)),
                Ok(Some(chunk)) => chunk,
            };
            trace.observe(&chunk);
            for raw in chunk.delta.iter().flat_map(|delta| &delta.tool_calls) {
                self.tools.dispatch(shared, handle, self.ctx, mint_call(shared, raw));
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
}

fn report_llm_metrics(
    data: &LlmGenerationData,
    failed: bool,
    shared: &SessionShared,
    handle: &SpeechHandle,
) {
    let (now, cut) = (shared.time.wall_secs(), handle.interrupted());
    if let Some(record) = data.collect_metrics(failed, now, cut, handle.speech_id()) {
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
    fn start(
        shared: &SessionShared,
        turn: Option<telemetry::SpanContext>,
        req: &ChatRequest,
    ) -> Self {
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
                        &lk_agents_abi::ToDictOptions {
                            exclude_metrics: true,
                            ..Default::default()
                        },
                    ))),
                    function_tools: names(function),
                    provider_tools: names(provider),
                    tool_sets: Vec::new(),
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
        let (text, calls) = (trace.text(), trace.calls());
        let ttft = trace.first_chunk_at().map(|at| at.saturating_sub(self.started) as f64 / 1e9);
        if let Some(ttft) = ttft {
            let model = telemetry::model_labels(&llm.id());
            telemetry::measure(span.cx(), &telemetry::TurnLlmTtft { model }, ttft);
        }
        let end = telemetry::LlmNodeEnd {
            response_text: telemetry::Content(text.to_owned()),
            function_calls: telemetry::Content(telemetry::Json(calls.to_vec())),
            ttft: telemetry::Measured(ttft.map(telemetry::Seconds)),
        };
        match failed {
            Some(error) => {
                telemetry::end(span, telemetry::Failed { end, error });
            }
            None => {
                telemetry::end(span, end);
            }
        }
    }
}
