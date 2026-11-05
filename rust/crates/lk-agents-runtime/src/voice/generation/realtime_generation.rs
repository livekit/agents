//! `_realtime_generation_task`: one step of a realtime reply, the generation's events played as
//! segments through the active audio output.

use super::ReplyExecution;
use crate::voice::generation::forward::{TextLeg, first_text_started_speaking};
use crate::voice::generation::pipeline_reply::{Finished, StepOutcome, finish_step_with};
use crate::voice::generation::playout::{
    Next, Segment, end_unvoiced, first_frame, perform_audio_forwarding,
};
use crate::voice::generation::tools::{ToolStep, run_tools};
use crate::voice::session::SessionShared;
use crate::voice::session::Tx;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::llm::realtime as rt;
use lk_agents_abi::llm::realtime::Event;
use lk_agents_abi::voice::io::TextOutput;
use lk_agents_voice::agent_activity::realtime::RealtimeWake;
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::generation::effects::RealtimeGenerationEffect;
use lk_agents_voice::generation::machine::realtime_generation::{
    RealtimeGeneration, RealtimeGenerationControl, RealtimeGenerationObservation, unheard_messages,
};
use lk_agents_voice::generation::{GenEvent, TurnAccumulator, TurnStats};
use lk_agents_voice::tool_executor::StepNext;
use lk_agents_voice::trace::GenerationStep;
use std::sync::Arc;
use tokio::sync::mpsc;

/// A realtime speech's end of the socket's events, which the runtime routes to it by speech
/// (`Effect::RealtimePlayer`). The step that plays a generation owns it; it outlives the step,
/// because a tool follow-up arrives on it.
pub(crate) type RealtimePlayer = mpsc::Receiver<Event>;

pub use lk_agents_voice::generation::machine::realtime_generation::PLAYER_CAPACITY;

use crate::telemetry;
use lk_agents_abi::metrics::base::RealtimeModelMetrics;

/// `_realtime_generation_task`: one step of a realtime reply. Each `MessageDone` is a boundary;
/// the speech ends when its last frame has played.
pub(crate) async fn realtime_generation_task(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyExecution,
    events: &mut RealtimePlayer,
    wake: &Tx,
    speeches: &crate::voice::SpeechTasks,
) -> Result<StepOutcome, DriverError> {
    // The authorization, per step: the server's deltas queue meanwhile. It stops a server turn
    // talking over a `say()`, and a tool follow-up over what became current while tools ran.
    let authorized = tokio::select! {
        biased;
        () = handle.wait_for_interrupted() => false,
        () = handle.wait_for_step_authorized(ctx.plan.step) => true,
    };
    if !authorized {
        // Interrupted while parked. Nothing of it was heard: what the server already said of it is deleted.
        let unheard = unheard_messages(std::iter::from_fn(|| events.try_recv().ok()));
        if !unheard.is_empty() {
            let _ = wake.send(RealtimeWake::Heard { speech: handle.id(), cuts: unheard });
        }
        return Ok(StepOutcome::Released);
    }
    // The follow-up step waits for its own grant.

    let span = shared.realtime_spans.start(shared, ctx.span);
    let mut acc = TurnAccumulator::new(shared.time.wall(), ctx.plan.user_metrics.clone());
    // The calls run as they arrive, alongside the playout.
    let mut tools = ToolStep::new(speeches.clone());
    let mut generation = Generation {
        shared,
        handle,
        ctx,
        acc: &mut acc,
        events,
        state: RealtimeGeneration::default(),
        tools: &mut tools,
        text_out: None,
        // A text message is voiced by the TTS when one plays into the audio output
        // (`"audio" not in msg_modalities and self.tts`).
        tts: ctx.resources.tts.clone(),
        voiced: None,
        backlog: std::collections::VecDeque::new(),
        leg: None,
    };
    loop {
        let nodes = &ctx.resources.output_nodes;
        let Some(mut binding) =
            super::output::OutputBinding::acquire(shared, handle, nodes).await?
        else {
            generation.state.binding_refused();
            break;
        };
        if !binding.activate().await? {
            generation.state.binding_refused();
            break;
        }
        let audio_out = binding.audio();
        generation.text_out = binding.text();
        generation.state.bind(audio_out.is_some());
        let played =
            perform_audio_forwarding(shared, handle, audio_out.as_deref(), &mut generation).await;
        generation.unvoice().await;
        generation.end_message();
        if !generation.state.carried_message()
            && let Some(text) = &generation.text_out
        {
            text.flush();
        }
        let text_result = match generation.text_out.as_deref() {
            Some(text) => super::forward::finish_text(text, handle, played.is_err()).await,
            None => Ok(true),
        };
        binding.text_drained();
        let part = match played {
            Ok(part) => part,
            Err(failed) => {
                if let Some(part) = failed.heard {
                    generation.state.played(part);
                }
                return Err(heard_until(&mut generation, wake, handle, failed.error));
            }
        };
        let delivered = matches!(text_result, Ok(true));
        if generation.state.end_unvoiced(&part, generation.text_out.is_some(), delivered) {
            let output = audio_out.as_deref().expect("the binding has an audio output");
            end_unvoiced(shared, handle.id(), output);
        }
        generation.state.played(part);
        if let Err(error) = text_result {
            return Err(heard_until(&mut generation, wake, handle, error));
        }
        // Its final report has settled before selection can activate the following segment.
        drop(binding);
        if generation.state.stop_playout(handle.interrupted()) {
            break;
        }
        // A binding is for a message: the server may end the generation instead of sending one.
        {
            let mut interrupted = std::pin::pin!(handle.wait_for_interrupted());
            let mut next = std::pin::pin!(generation.playout_controls());
            std::future::poll_fn(|cx| {
                if interrupted.as_mut().poll(cx).is_ready() {
                    return std::task::Poll::Ready(());
                }
                next.as_mut().poll(cx)
            })
            .await;
        }
        if generation.state.stop_playout(handle.interrupted()) {
            break;
        }
    }
    let interrupted = generation.state.interrupted(handle.interrupted());
    if interrupted {
        // What the server holds of its reply is cut to what the caller heard; a message they heard
        // none of is gone (`truncate`, after `_realtime_generation_task`'s playout).
        generation.drain_started();
        let cuts = generation.state.cuts();
        let _ = wake.send(RealtimeWake::Heard { speech: handle.id(), cuts });
    }
    let Generation { state, .. } = generation;
    let lk_agents_voice::generation::machine::realtime_generation::RealtimeTranscript {
        heard,
        segments,
        received,
        response_id,
        message_id,
    } = state.into_transcript();
    if let Some(span) = span {
        shared.realtime_spans.end(span, response_id.as_deref());
    }
    // An empty follow-up round: the reply ends on the step already spoken.
    if !received {
        return Ok(StepOutcome::Released);
    }
    // The transcript *is* the provider's: the words committed are the ones the voice said, or,
    // cut, what the synchroniser says was heard of them (`_ForwardOutput.forwarded_text`).
    if interrupted {
        acc.record(&heard.text);
    } else {
        for text in &segments {
            acc.record(text);
        }
    }
    let has_calls = !tools.is_empty();
    // No model timings on the message, as the reference's realtime message carries none: the
    // response's own metrics hold its ttft.
    let stats = TurnStats::default();
    // The speech's interruption decides it, as the pipeline's step does: a reply whose audio the
    // output cut still has its tools running.
    let tools_running = has_calls && !handle.interrupted();
    // `assistant_metrics["provider_request_ids"] = [generation_ev.response_id]` (`:4168-4169`).
    let request_ids = response_id.map(|id| vec![id]);
    // The server's id, as `_realtime_generation_task` commits it: it fills the placeholder.
    let message_id = message_id.unwrap_or_else(|| ctx.plan.message_id.clone());
    let finished = finish_step_with(
        shared,
        handle,
        ctx.plan.step,
        message_id,
        acc,
        interrupted,
        stats,
        has_calls,
        tools_running,
        request_ids,
    );
    let forwarded_text_empty = finished.outcome.forwarded_text.is_empty();
    let next = run_tools(shared, handle, ctx, tools, forwarded_text_empty, true).await;
    Ok(deliver_realtime(handle, ctx.plan.step, wake, next, has_calls, finished))
}

/// The step failed: the server keeps only what the person heard of it before the failure.
fn heard_until(
    generation: &mut Generation<'_>,
    wake: &Tx,
    handle: &SpeechHandle,
    error: DriverError,
) -> DriverError {
    generation.drain_started();
    let cuts = generation.state.heard_until();
    let _ = wake.send(RealtimeWake::Heard { speech: handle.id(), cuts });
    error
}

/// The run's next step is the server's to generate: asked once the commit pushed what it reads.
fn deliver_realtime(
    handle: &SpeechHandle,
    step: u32,
    wake: &Tx,
    next: Option<StepNext>,
    has_calls: bool,
    finished: Finished,
) -> StepOutcome {
    let speech = handle.id();
    // The outputs went to the server with the commit (`finish_tools`).
    let (ask, tool_choice) = match next {
        Some(StepNext::Reply(next)) => (true, next.planned.tool_choice),
        // The user barged in over the tool call: the answer is committed and not spoken.
        None | Some(StepNext::Done | StepNext::Interrupted) => (false, None),
    };
    if !ask && !has_calls {
        return StepOutcome::Done(finished);
    }
    let step = if ask { step + 1 } else { step };
    if ask {
        let _ = wake.send(GenEvent::ToolStep { speech, step });
    }
    let sent = wake.send(RealtimeWake::ToolsDone { speech, step, ask, tool_choice }).is_ok();
    if ask && sent { StepOutcome::Next(finished) } else { StepOutcome::Done(finished) }
}

/// The open generation's events, as one segment's frames.
struct Generation<'a> {
    shared: &'a Arc<SessionShared>,
    handle: &'a Arc<SpeechHandle>,
    ctx: &'a ReplyExecution,
    acc: &'a mut TurnAccumulator,
    events: &'a mut RealtimePlayer,
    state: RealtimeGeneration,
    /// The calls, dispatched as they arrive.
    tools: &'a mut ToolStep,
    text_out: Option<Arc<dyn TextOutput>>,
    /// The TTS a text message is spoken with, when an audio output plays it.
    tts: Option<Arc<dyn lk_agents_abi::Tts>>,
    /// The text message the TTS is speaking now.
    voiced: Option<Voiced>,
    /// Events that came while a voiced message's audio was awaited, in order.
    backlog: std::collections::VecDeque<Event>,
    /// The open message's transcript leg, flushed at its `MessageDone`.
    leg: Option<TextLeg<'a>>,
}

/// A text message spoken by the TTS: its text goes in as it streams, its audio is the message's.
struct Voiced {
    sink: lk_agents_abi::TtsSink,
    source: lk_agents_abi::TtsSource,
    transforms: Option<lk_agents_voice::transcription::Pipeline>,
    /// The message's text is all in.
    input_done: bool,
    /// Its last frame came.
    ended: bool,
}

impl Generation<'_> {
    fn dispatch_call(&mut self, ready: rt::FunctionCallReady) {
        let mut call = *ready.call;
        call.created_at = self.shared.time.wall();
        self.tools.dispatch(self.shared, self.handle, self.ctx, call);
    }

    fn end_message(&mut self) {
        if let Some(leg) = self.leg.take() {
            leg.flush();
        }
    }

    /// Drain already received events into the core before asking what the server may retain.
    fn drain_started(&mut self) {
        for event in self.backlog.drain(..) {
            self.state.unplayed(event);
        }
        while let Ok(event) = self.events.try_recv() {
            self.state.unplayed(event);
        }
    }

    /// The current message turned out to be text: the TTS speaks it, from the words so far.
    async fn voice_message(&mut self, idx: usize, words: String) -> Result<(), DriverError> {
        let tts = self.tts.clone().expect("the core admitted a configured TTS");
        let text = super::tts::transformed(self.shared, &words);
        let segment = idx as u32;
        self.shared.trace(GenerationStep::TtsRequested { speech: self.handle.id(), segment, text });
        // Opened as the pipeline's synthesis is: a failed open is retried on the same budget.
        let mut attempts = lk_agents_core::attempts::Attempts::retrying(
            &self.shared.conn.tts.unwrap_or_default(),
            self.shared.time.now(),
        );
        let (sink, mut source) = loop {
            let id = tts.id();
            self.shared.report(GenEvent::RequestStarted {
                speech: self.handle.id(),
                kind: lk_agents_core::usage::ModelKind::Tts,
                provider: id.provider,
                model: id.model,
            });
            let expressive = self.ctx.plan.request.expressive.is_some();
            let error = match tts.stream(&telemetry::Cx::NONE, expressive).await {
                Ok(stream) => break stream.split(),
                Err(error) => error,
            };
            let kind = lk_agents_voice::trace::ProviderKind::Tts;
            let failed = |e, r| tts.failed(e, r);
            match super::llm::retry(self.shared, self.handle, &mut attempts, kind, &error, &failed)
                .await
            {
                super::llm::Retried::Again => {}
                super::llm::Retried::Interrupted => return Ok(()),
                super::llm::Retried::GiveUp => return Err(DriverError::from(error)),
            }
        };
        super::tts::report_retries(self.shared, self.handle, &mut source);
        let shared = self.shared;
        let mut transforms = (!shared.tts_transforms.is_empty())
            .then(|| lk_agents_voice::transcription::Pipeline::new(&shared.tts_transforms));
        super::tts::push_text(&sink, &mut transforms, &words)?;
        self.voiced = Some(Voiced { sink, source, transforms, input_done: false, ended: false });
        Ok(())
    }

    /// The voiced message's next frame, its text fed meanwhile; `None` once its audio is over.
    async fn voice(&mut self) -> Option<Result<Next, DriverError>> {
        let v = self.voiced.as_mut()?;
        if v.ended {
            self.unvoice().await;
            return None;
        }
        let got = {
            let mut pending = Box::pin(v.source.next());
            loop {
                if v.input_done {
                    break pending.await;
                }
                tokio::select! {
                    biased;
                    ev = self.events.recv() => match ev {
                        Some(Event::TextDelta(d)) => {
                            self.state.synthesized_text(&d.delta);
                            if !self.handle.interrupted() && let Some(leg) = &self.leg {
                                leg.capture(&d.delta).await;
                            }
                            if let Err(e) = super::tts::push_text(&v.sink, &mut v.transforms, &d.delta) {
                                break Err(e);
                            }
                        }
                        // Its text is all in; the boundary is the audio's end.
                        Some(Event::MessageDone(m)) => {
                            super::tts::end_input(&v.sink, &mut v.transforms);
                            v.input_done = true;
                            self.backlog.push_back(Event::MessageDone(m));
                        }
                        Some(ev) => self.backlog.push_back(ev),
                        None => {
                            super::tts::end_input(&v.sink, &mut v.transforms);
                            v.input_done = true;
                        }
                    },
                    got = &mut pending => break got,
                }
            }
        };
        match got {
            Ok(Some(chunk)) => {
                v.ended = chunk.is_final && v.input_done;
                self.state.add_audio(&chunk.frame);
                Some(Ok(Next::Frame(chunk.frame)))
            }
            Ok(None) => {
                self.unvoice().await;
                None
            }
            // The voice is lost: the session's error, as a pipeline reply's spent synthesis is.
            Err(error) => {
                if let Some(tts) = &self.tts {
                    tts.failed(error.clone(), false);
                }
                self.shared.report(GenEvent::ProviderFailed {
                    speech: self.handle.id(),
                    leg: lk_agents_voice::trace::ProviderKind::Tts,
                    recoverable: false,
                    error: Box::new(error.clone()),
                });
                Some(Err(DriverError::from(error)))
            }
        }
    }

    /// The voiced message's synthesis is over: its input ended and its stream closed.
    async fn unvoice(&mut self) {
        let Some(v) = self.voiced.take() else { return };
        v.sink.end_input();
        let _ = v
            .source
            .close(self.shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
            .await;
    }
}

impl Segment for Generation<'_> {
    fn stop_at_boundary(&self) -> bool {
        true
    }

    /// Ends once the generation is over or its next message's first event is held back.
    async fn playout_controls(&mut self) {
        // Service one cancellation-safe control event per poll, so even a continuously ready
        // provider cannot starve the playback receipt/deadline polled beside this future.
        // The first next-media event stays in the existing backlog until its binding is admitted.
        let mut deferred_media = false;
        std::future::poll_fn(|cx| {
            if self.state.finished() || deferred_media {
                return std::task::Poll::Ready(());
            }
            let event = match self.backlog.pop_front() {
                Some(event) => event,
                None => match self.events.poll_recv(cx) {
                    std::task::Poll::Ready(Some(event)) => event,
                    std::task::Poll::Ready(None) => {
                        self.state.closed();
                        return std::task::Poll::Ready(());
                    }
                    std::task::Poll::Pending => return std::task::Poll::Pending,
                },
            };
            match self.state.control(event) {
                Some(RealtimeGenerationControl::ToolCall(call)) => self.dispatch_call(call),
                Some(RealtimeGenerationControl::End) => self.end_message(),
                Some(RealtimeGenerationControl::Defer(event)) => {
                    self.backlog.push_front(event);
                    deferred_media = true;
                }
                None => {}
            }
            if self.state.finished() || deferred_media {
                return std::task::Poll::Ready(());
            }
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        })
        .await;
    }

    fn started_speaking(&mut self, at: f64) {
        first_frame(self.shared, self.acc, at);
    }

    async fn next(&mut self) -> Result<Next, DriverError> {
        loop {
            if self.voiced.is_some() {
                match self.voice().await {
                    Some(next) => return next,
                    None => continue,
                }
            }
            let ev = match self.backlog.pop_front() {
                Some(ev) => ev,
                None => match self.events.recv().await {
                    Some(ev) => ev,
                    // The channel closed: the session ended, or the loop released this speech.
                    None => {
                        self.end_message();
                        self.state.closed();
                        return Ok(Next::End);
                    }
                },
            };
            let observed = RealtimeGenerationObservation {
                interrupted: self.handle.interrupted(),
                synthesizing: self.voiced.is_some(),
                tts_available: self.tts.is_some(),
            };
            match self.state.handle(ev, observed) {
                Some(RealtimeGenerationEffect::OpenText) => {
                    self.end_message();
                    let speech = self.handle.id();
                    self.leg = Some(TextLeg::new(self.text_out.clone(), None, self.shared, speech));
                }
                Some(RealtimeGenerationEffect::Text(delta)) => {
                    if let Some(leg) = &self.leg {
                        leg.capture(&delta).await;
                    }
                    first_text_started_speaking(
                        self.shared,
                        self.handle,
                        self.acc,
                        lk_agents_voice::generation::machine::forward::TurnAccumulatorOutputEvent::TextDelta {
                            delta: &delta,
                            has_audio_output: self.state.has_audio_output(),
                            interrupted: self.handle.interrupted(),
                        },
                    );
                }
                Some(RealtimeGenerationEffect::Audio(frame)) => return Ok(Next::Frame(frame)),
                Some(RealtimeGenerationEffect::VoiceText { segment, text }) => {
                    self.voice_message(segment, text).await?
                }
                Some(RealtimeGenerationEffect::Boundary) => {
                    self.end_message();
                    return Ok(Next::Boundary);
                }
                Some(RealtimeGenerationEffect::End) => {
                    self.end_message();
                    return Ok(Next::End);
                }
                Some(RealtimeGenerationEffect::ToolCall(call)) => self.dispatch_call(call),
                Some(RealtimeGenerationEffect::Failed(message)) => {
                    return Err(DriverError::Provider(lk_agents_voice::generation::fatal(message)));
                }
                None => {}
            }
        }
    }

    fn text(&self) -> String {
        self.state.text()
    }
}

/// The session's `realtime_inference` spans: each response's metrics land on its span when they
/// arrive before it ends, else on a `realtime_metrics` child of the ended span.
#[derive(Default)]
pub(crate) struct RealtimeSpans(parking_lot::Mutex<RealtimeSpansState>);

#[derive(Default)]
struct RealtimeSpansState {
    model: Option<lk_agents_abi::ProviderId>,
    /// Metrics that arrived while their response's span was open, by response id.
    early: std::collections::VecDeque<RealtimeModelMetrics>,
    /// Ended responses still waiting for their metrics, by response id.
    ended: std::collections::VecDeque<(String, telemetry::SpanContext)>,
}

/// Bounds the responses remembered for metrics that never come.
const REMEMBERED: usize = 16;

impl RealtimeSpans {
    pub(crate) fn set_model(&self, model: Option<lk_agents_abi::ProviderId>) {
        self.0.lock().model = model;
    }

    /// Opens a response's span under the speech's turn.
    pub(crate) fn start(
        &self,
        shared: &SessionShared,
        turn: Option<telemetry::SpanContext>,
    ) -> Option<RealtimeSpan> {
        let model = self.0.lock().model.clone()?;
        let span = telemetry::start(
            &shared.trace.clone().under::<telemetry::AgentTurn>(turn?),
            telemetry::RealtimeInference {
                operation: telemetry::GenAiOperation::GenerateContent,
                provider: telemetry::provider_name(&model.provider),
                model: model.model,
                stream: true,
                output_type: if shared.audio_output().is_some() {
                    telemetry::OutputType::Speech
                } else {
                    telemetry::OutputType::Text
                },
            },
        );
        Some(RealtimeSpan(span))
    }

    /// Ends a response's span with its metrics when they came, else leaves it to them.
    pub(crate) fn end(&self, span: RealtimeSpan, response_id: Option<&str>) {
        let mut state = self.0.lock();
        let early = response_id.and_then(|id| {
            let at = state.early.iter().position(|m| m.request_id == id)?;
            state.early.remove(at)
        });
        let model = state.model.clone();
        match (early, response_id) {
            (Some(metrics), _) => {
                telemetry::end(span.0, Some(response(&metrics, model.as_ref())));
            }
            (None, Some(id)) => {
                if let Some(context) = span.0.cx().span() {
                    remember(&mut state.ended, (id.to_owned(), context));
                }
                telemetry::end(span.0, None);
            }
            (None, None) => {
                telemetry::end(span.0, None);
            }
        }
    }

    /// A response's metrics: kept for its open span, or recorded as a late `realtime_metrics`.
    pub(crate) fn metrics(&self, session: &telemetry::Cx, m: &RealtimeModelMetrics) {
        let mut state = self.0.lock();
        let model = state.model.clone();
        match state.ended.iter().position(|(id, _)| *id == m.request_id) {
            Some(at) => {
                let (_, ended) = state.ended.remove(at).expect("the position is in range");
                drop(state);
                let provider = model.as_ref().map(|p| p.provider.as_str()).unwrap_or_default();
                let late = telemetry::start(
                    &session.clone().under::<telemetry::RealtimeInference>(ended),
                    telemetry::RealtimeMetrics {
                        operation: telemetry::GenAiOperation::GenerateContent,
                        provider: telemetry::provider_name(provider),
                        model: model.as_ref().map(|p| p.model.clone()).unwrap_or_default(),
                        response: response(m, model.as_ref()),
                    },
                );
                telemetry::end(late, ());
            }
            None => remember(&mut state.early, m.clone()),
        }
    }
}

fn remember<T>(queue: &mut std::collections::VecDeque<T>, item: T) {
    if queue.len() == REMEMBERED {
        queue.pop_front();
    }
    queue.push_back(item);
}

/// An open `realtime_inference`.
pub(crate) struct RealtimeSpan(telemetry::ScopedSpan<telemetry::RealtimeInference>);

/// A response's metrics as span attributes: its usage splits, timing and the 1.8.10 blob.
fn response(
    m: &RealtimeModelMetrics,
    model: Option<&lk_agents_abi::ProviderId>,
) -> telemetry::RealtimeResponse {
    let t = &m.tokens;
    let n = |v: Option<u64>| v.unwrap_or(0);
    let ttft = m.ttft.filter(|t| *t >= 0.0);
    telemetry::RealtimeResponse {
        metrics: telemetry::Json(m.into()),
        response_model: model.map(|p| p.model.clone()),
        response_id: Some(m.request_id.clone()).filter(|id| !id.is_empty()),
        usage: telemetry::RealtimeUsage {
            input_tokens: t.input_tokens,
            output_tokens: t.output_tokens,
            text_input_tokens: n(t.input_text_tokens),
            audio_input_tokens: n(t.input_audio_tokens),
            image_input_tokens: n(t.input_image_tokens),
            text_output_tokens: n(t.output_text_tokens),
            audio_output_tokens: n(t.output_audio_tokens),
            cache_read_input_tokens: n(t.cache_read_tokens),
            text_cache_read_input_tokens: n(t.cache_read_text_tokens),
            audio_cache_read_input_tokens: n(t.cache_read_audio_tokens),
            image_cache_read_input_tokens: n(t.cache_read_image_tokens),
        },
        time_to_first_chunk: telemetry::Measured(ttft.map(telemetry::Seconds)),
        completion_start_time: ttft.map(|ttft| {
            telemetry::Json(telemetry::UtcInstant { unix_seconds: m.timestamp + ttft })
        }),
    }
}
