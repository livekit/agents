use super::ReplyExecution;
use super::tools::ToolStep;
use crate::telemetry;
use crate::voice::session::SessionShared;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::{ChatRequest, FunctionCall, LlmStream};
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
        // The caller's frames are the segment, with words or without.
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
                log::error!(target: lk_agents_abi::LOG_TARGET, "the agent's instructions functions failed; the static instructions are sent: {e}")
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
        // Already filtered by `plan_request` to the tools this step may call.
        tools: ctx.plan.request.tools.clone(),
        conn_options: shared.conn.llm.unwrap_or_default(),
        ..ChatRequest::default()
    }
}

/// One step's words and tool calls, retried per the request's budget while nothing was produced.
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
        // Held for the step: a mid-reply handoff must not swap the model under a live request.
        let llm = self.ctx.resources.llm.clone().ok_or(DriverError::Missing("llm"))?;
        let req = chat_request(shared, handle, self.ctx).await;
        let node = LlmNodeTrace::start(shared, self.ctx.span, &req);
        let mut trace = telemetry::LlmTrace::start(node.cx(shared), llm.as_ref(), &req);
        let attempt = trace.attempt();
        let mut data = self.announce_request(&req, llm.as_ref());
        let observer = give_up_observer(shared, handle, ProviderKind::Llm);
        let opened = lk_agents_inference::llm::chat(
            llm.clone(),
            trace.cx(),
            &req,
            shared.time.clock(),
            Some(observer),
        )
        .await;
        let ended = match opened {
            Err(error) => {
                self.request_finished(Some(&error));
                Err(error)
            }
            Ok(mut stream) => {
                report_retries(shared, handle, ProviderKind::Llm, &mut stream);
                let ended = self.read_stream(&mut stream, &mut data, &mut trace).await;
                self.settle_attempt(stream, &data, ended).await
            }
        };
        trace.end_attempt(attempt, ended.as_ref().err());
        match ended {
            Ok(()) => {
                node.end(&trace, llm.as_ref(), None);
                trace.end(telemetry::LlmEnd {
                    interrupted: handle.interrupted(),
                    speech_id: Some(handle.speech_id()),
                });
                Ok(data)
            }
            Err(error) => {
                trace.fail(&error);
                node.end(&trace, llm.as_ref(), Some(&error));
                Err(DriverError::from(error))
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
        ended: Result<(), ProviderError>,
    ) -> Result<(), ProviderError> {
        self.request_finished(ended.as_ref().err());
        report_llm_metrics(data, ended.is_err(), self.shared, self.handle);
        // Dropping does not cancel a host stream; `close` must run on every path.
        let deadline = self.shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let _ = stream.close(deadline).await;
        ended
    }

    /// Tool calls are dispatched on arrival; the inference stream guards against replay.
    async fn read_stream(
        &mut self,
        stream: &mut LlmStream,
        data: &mut LlmGenerationData,
        trace: &mut telemetry::LlmTrace,
    ) -> Result<(), ProviderError> {
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
                Err(error) => break Err(error),
                Ok(Some(chunk)) => chunk,
            };
            trace.observe(&chunk);
            for raw in chunk.delta.iter().flat_map(|delta| &delta.tool_calls) {
                self.tools.dispatch(shared, handle, self.ctx, mint_call(shared, raw));
            }
            let at = shared.time.now().as_secs_f64();
            for effect in data.chunk(&chunk, at, || shared.time.wall_secs()) {
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

/// The request gave up: the session's unrecoverable error. Its retries reach the session through
/// [`report_retries`].
pub(super) fn give_up_observer(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    leg: ProviderKind,
) -> lk_agents_inference::retry::Observer {
    let (shared, handle) = (shared.clone(), handle.clone());
    Arc::new(move |error, retry, _| {
        if matches!(retry, lk_agents_core::request::Retry::GiveUp(_)) {
            shared.report(GenEvent::ProviderFailed {
                speech: handle.id(),
                leg,
                recoverable: false,
                error: Box::new(error.clone()),
            });
        }
    })
}

/// Every retry of a leg's stream, its own or a voice's that retries its sentences itself, is the
/// session's recoverable error and a retry the journal waits out, whichever reply opened it.
pub(super) fn report_retries<T>(
    shared: &Arc<SessionShared>,
    handle: &SpeechHandle,
    leg: ProviderKind,
    source: &mut Box<dyn lk_agents_abi::ProviderStream<T>>,
) {
    let (on, speech) = (Arc::clone(shared), handle.id());
    source.report_retries(Arc::new(move |error, attempt, wait| {
        on.report(GenEvent::ProviderFailed {
            speech,
            leg,
            recoverable: true,
            error: Box::new(error),
        });
        let at = on.time.now().saturating_add(wait);
        on.trace(GenerationStep::RetryArmed { speech, leg, attempt, at });
    }));
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
