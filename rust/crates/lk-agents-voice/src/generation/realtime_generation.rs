//! `_realtime_generation_task`: one step of a realtime reply, the generation's events played as
//! segments, through the realtime audio node when the agent has one.

use crate::agent_activity::realtime::{RealtimeWake, realtime_state};
use crate::agent_session::{DriverError, SessionShared, Tx};
use crate::generation::forward::{TextLeg, first_text_started_speaking};
use crate::generation::pipeline_reply::{Finished, StepOutcome, finish_step_with};
use crate::generation::playout::{
    Next, Segment, end_unvoiced, first_frame, perform_audio_forwarding,
};
use crate::generation::tools::{ToolStep, run_tools};
use crate::generation::{GenEvent, ReplyContext, TurnAccumulator, TurnStats};
use crate::io::RealtimeAudioNode;
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::{NextStep, StepDecision};
use crate::trace::GenerationStep;
use lk_agents_abi::llm::realtime::{Event, Modalities};
use lk_agents_abi::voice::io::{AudioOutput, PlaybackTracker, TextOutput};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, llm::realtime as rt};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
#[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
/// `_realtime_generation_task`: one step of a realtime reply. Each `MessageDone` is a boundary;
/// the speech ends when its last frame has played.
pub(crate) async fn realtime_generation_task(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    events: &tokio::sync::Mutex<mpsc::UnboundedReceiver<Event>>,
    wake: &Tx,
    retries: &mut lk_agents_core::tool_outcome::ToolRetries,
) -> Result<StepOutcome, DriverError> {
    // The authorization, per step: the server's deltas queue meanwhile. It stops a server turn
    // talking over a `say()`, and a tool follow-up over what became current while tools ran.
    let authorized = tokio::select! {
        biased;
        () = handle.wait_for_interrupted() => false,
        _ = handle.wait_for_authorization() => true,
    };
    if !authorized {
        // Interrupted while parked. Nothing of it was heard: what the server already said of it is deleted.
        let mut unheard = Vec::new();
        while let Ok(ev) = events.lock().await.try_recv() {
            if let Event::MessageStarted(m) = ev {
                unheard.push(rt::Truncate {
                    message_id: m.message_id,
                    has_audio: true,
                    audio_end_ms: 0,
                    audio_transcript: None,
                });
            }
        }
        if !unheard.is_empty() {
            let _ = wake.send(RealtimeWake::Heard { speech: handle.id(), cuts: unheard });
        }
        return Ok(StepOutcome::Released);
    }
    // Consumed: the follow-up step waits for its own grant.
    handle.clear_authorization();

    let mut acc = TurnAccumulator::new(shared.time.wall(), ctx.user_metrics.clone());
    // Locked for the step by the one task that ever reads it; the channel outlives the step.
    let mut rx = events.lock().await;
    let audio_out = shared.audio_output();
    // `text_output`, read once for the step as the reference does: each message's words go down it.
    let text_out = shared.transcription_enabled().then(|| shared.transcription_output());
    let text_out = ctx.providers.output_nodes.text_output(text_out.flatten());
    // `realtime_audio_output_node`: frames go to a node opened for this step over its output,
    // which emits into that output itself. A node that fails to open plays nothing of the step.
    let node = ctx.providers.output_nodes.realtime_audio.as_ref().zip(audio_out.as_ref()).and_then(
        |(nodes, out)| {
            nodes
                .audio_node(Arc::clone(out))
                .inspect_err(|e| {
                    log::error!(target: "livekit.agents", error = e.to_string().as_str();
                    "failed to open the realtime_audio_output_node");
                })
                .ok()
        },
    );
    let node_out =
        node.as_deref().zip(audio_out.as_deref()).map(|(node, out)| NodeOutput { node, out });
    let out: Option<&dyn AudioOutput> = match &node_out {
        Some(n) => Some(n),
        None => audio_out.as_deref(),
    };
    // The calls run as they arrive, alongside the playout.
    let mut tools = ToolStep::new();
    let mut generation = Generation {
        shared,
        handle,
        ctx,
        acc: &mut acc,
        events: &mut rx,
        current: None,
        segments: Vec::new(),
        tools: &mut tools,
        response_id: None,
        message_id: None,
        messages: Vec::new(),
        audio: Duration::ZERO,
        received: false,
        text_out: text_out.as_deref(),
        speaks_text: out.is_none(),
        plays_audio: out.is_some(),
        // A text message is voiced by the TTS when one plays into the audio output
        // (`"audio" not in msg_modalities and self.tts`).
        tts: out.and(ctx.providers.tts.clone()),
        voiced: None,
        backlog: std::collections::VecDeque::new(),
        leg: None,
    };
    let played = perform_audio_forwarding(shared, handle, out, &mut generation).await;
    // The node's invocation ends with the generation, so cut-off frames never reach the next one.
    if let Some(node) = &node {
        node.end();
    }
    // `Generation::next` fails only on the speech's own request failing: nothing was generated.
    generation.unvoice().await;
    let heard = played.map_err(|failed| failed.error)?;
    if heard.frames == 0
        && heard.complete
        && text_out.is_some()
        && !generation.segments.is_empty()
        && let Some(out) = audio_out.as_deref()
    {
        end_unvoiced(shared, handle.id(), out);
    }
    let interrupted = !heard.complete || handle.interrupted();
    if interrupted {
        // What the server holds of its reply is cut to what the caller heard; a message they heard
        // none of is gone (`truncate`, after `_realtime_generation_task`'s playout).
        generation.drain_started();
        let cuts = generation.cuts(&heard);
        let _ = wake.send(RealtimeWake::Heard { speech: handle.id(), cuts });
    }
    let Generation { segments, received, response_id, message_id, .. } = generation;
    drop(rx);
    // An empty follow-up round: the reply ends on the step already spoken.
    if !received && handle.num_steps() > 0 {
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
        message_id,
        acc,
        interrupted,
        stats,
        has_calls,
        tools_running,
        request_ids,
    );
    let forwarded_text_empty = finished.outcome.forwarded_text.is_empty();
    let Some(decision) = run_tools(shared, handle, ctx, tools, forwarded_text_empty, retries).await
    else {
        return Ok(StepOutcome::Done(finished));
    };
    Ok(deliver_realtime(shared, handle, wake, decision, finished))
}

fn deliver_realtime(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    wake: &Tx,
    decision: StepDecision,
    finished: Finished,
) -> StepOutcome {
    let speech = handle.id();
    // The outputs went to the server with the commit (`finish_tools`).
    let (ask, tool_choice) = match decision.next {
        NextStep::Reply(req) => (true, Some(req.tool_choice)),
        // The user barged in over the tool call: the answer is committed and not spoken.
        NextStep::Done | NextStep::Interrupted => (false, None),
    };
    if ask {
        handle.step();
        shared.report(GenEvent::ToolStep { speech, step: handle.num_steps() });
    }
    let step = handle.num_steps();
    let sent = wake.send(RealtimeWake::ToolsDone { speech, step, ask, tool_choice }).is_ok();
    if ask && sent { StepOutcome::Next(finished) } else { StepOutcome::Done(finished) }
}

/// The open generation's events, as one segment's frames.
struct Generation<'a> {
    shared: &'a Arc<SessionShared>,
    handle: &'a Arc<SpeechHandle>,
    ctx: &'a ReplyContext,
    acc: &'a mut TurnAccumulator,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    events: &'a mut mpsc::UnboundedReceiver<Event>,
    /// The message the text deltas append to: a provider may interleave two.
    current: Option<usize>,
    segments: Vec<String>,
    /// The calls, dispatched as they arrive.
    tools: &'a mut ToolStep,
    /// The generation's `response_id`, from its first event that names one.
    response_id: Option<String>,
    /// The server's id of the generation's first message.
    message_id: Option<String>,
    /// Every message, in order: where its audio starts in the generation's, and how long it is.
    messages: Vec<Message>,
    /// The generation's audio so far.
    audio: Duration,
    /// An empty round means released or interrupted.
    received: bool,
    /// The transcription output, and whether the words are what starts the agent speaking (no
    /// audio output).
    text_out: Option<&'a dyn TextOutput>,
    speaks_text: bool,
    /// An audio output plays the generation: a message whose modality is unsettled is audio.
    plays_audio: bool,
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

/// One message of the generation, as the server holds it.
struct Message {
    id: String,
    /// Where its audio starts in the generation's audio, and how much it has.
    start: Duration,
    audio: Duration,
    /// Its modality, once the server settled it.
    modalities: Option<Modalities>,
    /// `MessageDone` came: the server will add no more to it.
    done: bool,
}

impl Generation<'_> {
    fn end_message(&mut self) {
        if let Some(leg) = self.leg.take() {
            leg.flush();
        }
    }

    fn message_started(&mut self, id: String) {
        let start = self.audio;
        self.messages.push(Message {
            id,
            start,
            audio: Duration::ZERO,
            modalities: None,
            done: false,
        });
    }

    /// What was already sent to this speech when it was cut: the server holds it too.
    fn drain_started(&mut self) {
        let mut events: Vec<Event> = self.backlog.drain(..).collect();
        while let Ok(ev) = self.events.try_recv() {
            events.push(ev);
        }
        for ev in events {
            match ev {
                Event::MessageStarted(m) => self.message_started(m.message_id),
                Event::MessageModalities(m) => self.settle(&m.message_id, m.modalities),
                Event::AudioDelta(d) => self.add_audio(&d.frame),
                Event::MessageDone(m) => self.message_done(&m.message_id),
                _ => {}
            }
        }
    }

    fn add_audio(&mut self, frame: &lk_agents_abi::AudioFrame) {
        let len = lk_agents_rtc::frame_duration(frame);
        self.audio += len;
        if let Some(m) = self.messages.last_mut() {
            m.audio += len;
        }
    }

    fn message_done(&mut self, id: &str) {
        if let Some(m) = self.messages.iter_mut().find(|m| m.id == id) {
            m.done = true;
        }
    }

    fn settle(&mut self, id: &str, modalities: Modalities) {
        if let Some(m) = self.messages.iter_mut().find(|m| m.id == id) {
            m.modalities = Some(modalities);
        }
    }

    /// The current message turned out to be text: the TTS speaks it, from the words so far.
    async fn voice_message(&mut self) -> Result<(), DriverError> {
        let (Some(tts), Some(idx)) = (self.tts.clone(), self.current) else { return Ok(()) };
        if self.handle.interrupted() || self.voiced.is_some() {
            return Ok(());
        }
        let words = self.segments.get(idx).cloned().unwrap_or_default();
        let text = super::tts::transformed(self.shared, &words);
        let segment = idx as u32;
        self.shared.trace(GenerationStep::TtsRequested { speech: self.handle.id(), segment, text });
        // Opened as the pipeline's synthesis is: a failed open is retried on the same budget.
        let mut attempt = crate::generation::Attempt::new(self.shared.conn.tts);
        let (sink, source) = loop {
            let error = match tts.stream().await {
                Ok(pair) => break pair,
                Err(error) => error,
            };
            let kind = crate::trace::ProviderKind::Tts;
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

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
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
                            if let Some(seg) = self.current.and_then(|i| self.segments.get_mut(i)) {
                                seg.push_str(&d.delta);
                            }
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
                self.add_audio(&chunk.frame);
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
        let _ = v.source.close(super::pipeline_reply::teardown_deadline(self.shared)).await;
    }

    /// The truncations that leave the server holding what the caller heard (`heard.played` of the
    /// generation's audio): a message heard in part is cut there, one heard none of is deleted
    /// (cut at 0), one heard whole is left. A message with no audio is cut to the words heard.
    fn cuts(&self, heard: &crate::transcription::Heard) -> Vec<rt::Truncate> {
        let mut words = heard.text.as_str();
        let mut cuts = Vec::new();
        for (i, m) in self.messages.iter().enumerate() {
            let text = self.segments.get(i).map_or("", String::as_str);
            let (said, rest) = split_chars(words, text.chars().count());
            words = rest;
            let has_audio =
                m.modalities.map_or(m.audio > Duration::ZERO || self.plays_audio, |modalities| {
                    modalities == Modalities::AudioAndText
                });
            if has_audio {
                let ms = |d: Duration| u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
                let heard_ms = ms(heard.played.saturating_sub(m.start).min(m.audio));
                // Complete, and heard to within a millisecond of its end: heard whole.
                if m.done && m.audio > Duration::ZERO && heard_ms + 1 >= ms(m.audio) {
                    continue;
                }
                let cut = rt::Truncate {
                    message_id: m.id.clone(),
                    has_audio: true,
                    audio_end_ms: heard_ms,
                    audio_transcript: None,
                };
                cuts.push(cut);
            } else if said.len() < text.len() {
                let cut = rt::Truncate {
                    message_id: m.id.clone(),
                    has_audio: false,
                    audio_end_ms: 0,
                    audio_transcript: Some(said.to_owned()),
                };
                cuts.push(cut);
            }
        }
        cuts
    }
}

/// The first `n` characters of `s`, and the rest.
fn split_chars(s: &str, n: usize) -> (&str, &str) {
    let at = s.char_indices().nth(n).map_or(s.len(), |(i, _)| i);
    s.split_at(at)
}

impl Segment for Generation<'_> {
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
                        return Ok(Next::End);
                    }
                },
            };
            self.received = true;
            if self.response_id.is_none()
                && let Some(id) = realtime_state::response_id_of(&ev)
            {
                self.response_id = Some(id.to_owned());
            }
            match ev {
                Event::MessageStarted(m) => {
                    self.message_started(m.message_id.clone());
                    self.message_id.get_or_insert(m.message_id);
                    self.end_message();
                    self.current = Some(self.segments.len());
                    self.segments.push(String::new());
                    let speech = self.handle.id();
                    self.leg = Some(TextLeg::new(self.text_out, None, self.shared, speech));
                }
                Event::TextDelta(rt::TextDelta { delta, .. }) => {
                    let Some(seg) = self.current.and_then(|idx| self.segments.get_mut(idx)) else {
                        continue;
                    };
                    seg.push_str(&delta);
                    if self.handle.interrupted() {
                        continue;
                    }
                    if let Some(leg) = &self.leg {
                        leg.capture(&delta).await;
                    }
                    if self.speaks_text && self.text_out.is_some() && !delta.is_empty() {
                        first_text_started_speaking(self.shared, self.handle, self.acc);
                    }
                }
                Event::AudioDelta(rt::AudioDelta { frame, .. }) => {
                    self.add_audio(&frame);
                    return Ok(Next::Frame(frame));
                }
                Event::MessageModalities(m) => {
                    self.settle(&m.message_id, m.modalities);
                    if m.modalities == Modalities::TextOnly {
                        self.voice_message().await?;
                    }
                }
                // A boundary, so `wait_for_playout` can finish this message while the next arrives.
                Event::MessageDone(m) => {
                    self.message_done(&m.message_id);
                    self.current = None;
                    self.end_message();
                    return Ok(Next::Boundary);
                }
                Event::FunctionCall(ready) => {
                    // Stamped at its execution start: the call arrives during the generation,
                    // before the message it belongs to is placed.
                    let mut call = *ready.call;
                    call.created_at = self.shared.time.wall();
                    self.tools.dispatch(self.shared, self.handle, self.ctx, call);
                }
                Event::GenerationDone(_) => {
                    self.end_message();
                    return Ok(Next::End);
                }
                // The request this speech was announced for failed (`_mark_done(error=e)`).
                Event::GenerateReplyFailed(failed) => {
                    let error = crate::generation::pipeline_reply::fatal(failed.error.message);
                    return Err(DriverError::Provider(error));
                }
                // The loop routes only a generation's own events here; the rest are the loop's.
                _ => {}
            }
        }
    }

    fn text(&self) -> String {
        self.segments.concat()
    }
}

/// The node in front of the step's sink: a segment's boundary reaches `out` directly, an
/// interruption cuts both.
pub(crate) struct NodeOutput<'a> {
    pub(crate) node: &'a dyn RealtimeAudioNode,
    pub(crate) out: &'a dyn AudioOutput,
}

#[async_trait::async_trait]
impl AudioOutput for NodeOutput<'_> {
    fn label(&self) -> &str {
        "RealtimeAudioNode"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        self.out.tracker()
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        self.out.capabilities()
    }

    fn sample_rate(&self) -> Option<u32> {
        self.out.sample_rate()
    }

    async fn capture(&self, frame: &AudioFrame) -> lk_agents_abi::voice::io::Result<()> {
        self.node.push(frame).await;
        Ok(())
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
