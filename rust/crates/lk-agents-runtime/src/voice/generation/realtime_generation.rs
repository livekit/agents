//! `_realtime_generation_task`: one step of a realtime reply, the generation's events played as
//! segments, through the realtime audio node when the agent has one.

use crate::voice::generation::forward::{TextLeg, first_text_started_speaking};
use crate::voice::generation::pipeline_reply::{Finished, StepOutcome, finish_step_with};
use crate::voice::generation::playout::{
    Next, Segment, end_unvoiced, first_frame, perform_audio_forwarding,
};
use crate::voice::generation::tools::{ToolStep, run_tools};
use crate::voice::session::SessionShared;
use crate::voice::session::Tx;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::llm::realtime::Event;
use lk_agents_abi::voice::io::{AudioOutput, PlaybackTracker, TextOutput};
use lk_agents_abi::voice::nodes::RealtimeAudioNode;
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, llm::realtime as rt};
use lk_agents_voice::agent_activity::realtime::RealtimeWake;
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::generation::realtime::{
    RealtimeControl, RealtimeEffect, RealtimeObservation, RealtimePlayout, unheard_messages,
};
use lk_agents_voice::generation::{GenEvent, ReplyContext, TurnAccumulator, TurnStats};
use lk_agents_voice::tool_executor::StepNext;
use lk_agents_voice::trace::GenerationStep;
use std::sync::Arc;
use tokio::sync::mpsc;

/// A realtime speech's end of the socket's events, which the runtime routes to it by speech
/// (`Effect::RealtimePlayer`). The step that plays a generation owns it; it outlives the step,
/// because a tool follow-up arrives on it.
pub(crate) type RealtimePlayer = mpsc::Receiver<Event>;

pub use lk_agents_voice::generation::realtime::PLAYER_CAPACITY;

/// `_realtime_generation_task`: one step of a realtime reply. Each `MessageDone` is a boundary;
/// the speech ends when its last frame has played.
pub(crate) async fn realtime_generation_task(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    events: &mut RealtimePlayer,
    wake: &Tx,
    speeches: &crate::voice::SpeechTasks,
) -> Result<StepOutcome, DriverError> {
    // The authorization, per step: the server's deltas queue meanwhile. It stops a server turn
    // talking over a `say()`, and a tool follow-up over what became current while tools ran.
    let authorized = tokio::select! {
        biased;
        () = handle.wait_for_interrupted() => false,
        () = handle.wait_for_step_authorized(ctx.step) => true,
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

    let mut acc = TurnAccumulator::new(shared.time.wall(), ctx.user_metrics.clone());
    // The calls run as they arrive, alongside the playout.
    let mut tools = ToolStep::new(speeches.clone());
    let mut generation = Generation {
        shared,
        handle,
        ctx,
        acc: &mut acc,
        events,
        state: RealtimePlayout::default(),
        tools: &mut tools,
        text_out: None,
        // A text message is voiced by the TTS when one plays into the audio output
        // (`"audio" not in msg_modalities and self.tts`).
        tts: ctx.providers.tts.clone(),
        voiced: None,
        backlog: std::collections::VecDeque::new(),
        leg: None,
    };
    loop {
        let nodes = &ctx.providers.output_nodes;
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
        // A node invocation belongs to this binding. Its configured transforms survive a tail
        // replacement; a pending invocation is ended before the next binding is acquired.
        let node =
            ctx.providers.output_nodes.realtime_audio.as_ref().zip(audio_out.as_ref()).and_then(
                |(nodes, output)| {
                    nodes
                        .audio_node(output.clone())
                        .inspect_err(|e| {
                            log::error!(target: "livekit.agents", error = e.to_string().as_str();
                    "failed to open the realtime_audio_output_node");
                        })
                        .ok()
                },
            );
        let node_out =
            node.as_deref().zip(audio_out.as_deref()).map(|(node, out)| NodeOutput { node, out });
        let output: Option<&dyn AudioOutput> = match &node_out {
            Some(node) => Some(node),
            None => audio_out.as_deref(),
        };
        let played = perform_audio_forwarding(shared, handle, output, &mut generation).await;
        if let Some(node) = &node {
            node.end();
        }
        generation.unvoice().await;
        generation.end_message();
        if !generation.state.carried_message()
            && let Some(text) = &generation.text_out
        {
            text.flush();
        }
        let text_result = match generation.text_out.as_deref() {
            Some(text) => super::forward::finish_text(text, handle, played.is_err()).await,
            None => Ok(()),
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
        if generation.state.end_unvoiced(&part, generation.text_out.is_some()) {
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
    let lk_agents_voice::generation::realtime::RealtimeTranscript {
        heard,
        segments,
        received,
        response_id,
        message_id,
    } = state.into_transcript();
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
    let message_id = message_id.unwrap_or_else(|| ctx.message_id.clone());
    let finished = finish_step_with(
        shared,
        handle,
        ctx.step,
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
    Ok(deliver_realtime(handle, ctx.step, wake, next, has_calls, finished))
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
    ctx: &'a ReplyContext,
    acc: &'a mut TurnAccumulator,
    events: &'a mut RealtimePlayer,
    state: RealtimePlayout,
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
        let mut attempt =
            lk_agents_voice::generation::Attempt::new(self.shared.conn.tts.unwrap_or_default());
        let (sink, source) = loop {
            let error = match tts.stream(self.ctx.request.expressive.is_some()).await {
                Ok(pair) => break pair,
                Err(error) => error,
            };
            let kind = lk_agents_voice::trace::ProviderKind::Tts;
            let failed = |e, r| tts.failed(e, r);
            match super::llm::retry(self.shared, self.handle, &mut attempt, kind, &error, &failed)
                .await
            {
                super::llm::Retried::Again => {}
                super::llm::Retried::Interrupted => return Ok(()),
                super::llm::Retried::GiveUp => return Err(DriverError::from(error)),
            }
        };
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
            Err(e) => Some(Err(DriverError::from(e))),
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
                Some(RealtimeControl::ToolCall(call)) => self.dispatch_call(call),
                Some(RealtimeControl::End) => self.end_message(),
                Some(RealtimeControl::Defer(event)) => {
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
            let observed = RealtimeObservation {
                interrupted: self.handle.interrupted(),
                synthesizing: self.voiced.is_some(),
                tts_available: self.tts.is_some(),
            };
            match self.state.handle(ev, observed) {
                Some(RealtimeEffect::OpenText) => {
                    self.end_message();
                    let speech = self.handle.id();
                    self.leg = Some(TextLeg::new(self.text_out.clone(), None, self.shared, speech));
                }
                Some(RealtimeEffect::Text(delta)) => {
                    if let Some(leg) = &self.leg {
                        leg.capture(&delta).await;
                    }
                    first_text_started_speaking(
                        self.shared,
                        self.handle,
                        self.acc,
                        lk_agents_voice::generation::forward::ForwardEvent::TextDelta {
                            delta: &delta,
                            has_audio_output: self.state.has_audio_output(),
                            interrupted: self.handle.interrupted(),
                        },
                    );
                }
                Some(RealtimeEffect::Audio(frame)) => return Ok(Next::Frame(frame)),
                Some(RealtimeEffect::VoiceText { segment, text }) => {
                    self.voice_message(segment, text).await?
                }
                Some(RealtimeEffect::Boundary) => {
                    self.end_message();
                    return Ok(Next::Boundary);
                }
                Some(RealtimeEffect::End) => {
                    self.end_message();
                    return Ok(Next::End);
                }
                Some(RealtimeEffect::ToolCall(call)) => self.dispatch_call(call),
                Some(RealtimeEffect::Failed(message)) => {
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

/// The node in front of the step's sink: a segment's boundary reaches `out` directly, an
/// interruption cuts both.
pub(crate) struct NodeOutput<'a> {
    pub(crate) node: &'a dyn RealtimeAudioNode,
    pub(crate) out: &'a dyn AudioOutput,
}

impl AudioOutput for NodeOutput<'_> {
    fn label(&self) -> &str {
        "RealtimeAudioNode"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        self.out.tracker()
    }

    fn begin_segment(
        &self,
        key: lk_agents_abi::voice::io::PlaybackSegment,
    ) -> lk_agents_abi::voice::io::Result<()> {
        self.out.begin_segment(key)
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        self.out.capabilities()
    }

    fn sample_rate(&self) -> Option<u32> {
        self.out.sample_rate()
    }

    fn capture<'a>(
        &'a self,
        frame: &'a AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async move {
            self.node.push(frame).await;
            Ok(())
        })
    }

    /// The tracker is `out`'s, and `AudioOutputExt::flush` on this node already moved it.
    fn flush_segment(&self) {
        self.out.flush_segment();
    }

    /// What the node still holds is cut with what `out` holds.
    fn clear_buffer(&self) {
        self.node.clear();
        self.out.clear_buffer();
    }

    fn pause(&self) {
        self.out.pause();
    }

    fn resume(&self) {
        self.out.resume();
    }
}
