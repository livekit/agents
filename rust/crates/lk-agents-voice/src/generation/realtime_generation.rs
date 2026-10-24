//! `_realtime_generation_task`: one step of a realtime reply, the generation's events played as
//! segments, through the realtime audio node when the agent has one.

use crate::agent_activity::realtime::{RealtimeWake, realtime_state};
use crate::agent_session::{DriverError, SessionShared, Tx};
use crate::generation::pipeline_reply::{Finished, StepOutcome, finish_step_with};
use crate::generation::playout::{Next, Segment, first_frame, perform_audio_forwarding};
use crate::generation::tools::{ToolStep, run_tools};
use crate::generation::{GenEvent, ReplyContext, TurnAccumulator, TurnStats};
use crate::io::RealtimeAudioNode;
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::{NextStep, StepDecision};
use lk_agents_abi::llm::realtime::Event;
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, ChatItem, llm::realtime as rt};
use lk_agents_voice::io::{AudioOutput, PlaybackTracker};
use std::sync::Arc;
use tokio::sync::mpsc;

/// `_realtime_generation_task`: one step of a realtime reply. Each `MessageDone` is a boundary;
/// the speech ends when its last frame has played.
pub(crate) async fn realtime_generation_task(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    events: &tokio::sync::Mutex<mpsc::UnboundedReceiver<Event>>,
    wake: &Tx<RealtimeWake>,
) -> Result<StepOutcome, DriverError> {
    // The authorization, per step: the server's deltas queue meanwhile. It stops a server turn
    // talking over a `say()`, and a tool follow-up over what became current while tools ran.
    let authorized = tokio::select! {
        biased;
        () = handle.wait_for_interrupted() => false,
        _ = handle.wait_for_authorization() => true,
    };
    if !authorized {
        // Interrupted while parked. A node that never got a first step's frames is ended here.
        if handle.num_steps() == 0
            && let Some(node) = &shared.realtime_audio
        {
            node.end();
        }
        return Ok(StepOutcome::Released);
    }
    // Consumed: the follow-up step waits for its own grant.
    handle.clear_authorization();

    let started_at = shared.time.wall_secs();
    let mut acc = TurnAccumulator::new(shared.time.wall(), ctx.user_metrics.clone());
    // Locked for the step by the one task that ever reads it; the channel outlives the step.
    let mut rx = events.lock().await;
    let audio_out = shared.output.audio();
    // `realtime_audio_output_node`: frames go to the node, which emits into the sink itself.
    let node = shared.realtime_audio.clone();
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
        first_token_at: None,
        segments: Vec::new(),
        tools: &mut tools,
        response_id: None,
        received: false,
    };
    // `Generation::next` never fails; a failure would be a bug in this file.
    let played = perform_audio_forwarding(shared, handle, out, &mut generation).await;
    // The node's invocation ends with the generation, so cut-off frames never reach the next one.
    if let Some(node) = &node {
        node.end();
    }
    let Generation { segments, received, first_token_at, response_id, .. } = generation;
    let interrupted = played.is_ok_and(|heard| !heard.complete) || handle.interrupted();
    // The metrics' `llm_started_at` anchor, on the driver's clock. `None` when nothing arrived.
    let ttft = first_token_at.map(|at| (at - started_at).max(0.0));
    drop(rx);
    // An empty follow-up round: the reply ends on the step already spoken.
    if !received && handle.num_steps() > 0 {
        return Ok(StepOutcome::Released);
    }
    // The transcript *is* the provider's: the words committed are the ones the voice said.
    for text in &segments {
        acc.record(text);
    }
    let has_calls = !tools.is_empty();
    // No token rate: it would measure the server's audio pacing, not generation speed.
    let stats = TurnStats { llm_ttft: ttft, llm_tps: None, llm_started_at: Some(started_at) };
    let tools_running = has_calls && !interrupted;
    // `assistant_metrics["provider_request_ids"] = [generation_ev.response_id]` (`:4168-4169`).
    let request_ids = response_id.map(|id| vec![id]);
    let finished = finish_step_with(
        shared,
        handle,
        ctx,
        acc,
        interrupted,
        stats,
        has_calls,
        tools_running,
        request_ids,
    );
    let forwarded_text_empty = finished.outcome.forwarded_text.is_empty();
    let Some(decision) = run_tools(shared, handle, ctx, tools, forwarded_text_empty).await else {
        return Ok(StepOutcome::Done(finished));
    };
    Ok(deliver_realtime(shared, handle, wake, decision, finished))
}

fn deliver_realtime(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    wake: &Tx<RealtimeWake>,
    decision: StepDecision,
    finished: Finished,
) -> StepOutcome {
    let speech = handle.id();
    // The outputs go to the server (the model's context lives there); the call is the server's own.
    let outputs: Vec<ChatItem> = decision
        .commit
        .into_iter()
        .filter(|item| matches!(item, ChatItem::FunctionCallOutput(_)))
        .collect();
    let ask = match decision.next {
        NextStep::Handoff { agent, follow_up } => {
            shared.report(GenEvent::Handoff { speech, agent });
            follow_up.is_some()
        }
        NextStep::Reply(_) => true,
        // The user barged in over the tool call: the answer is committed and not spoken.
        NextStep::Done | NextStep::Interrupted => false,
    };
    if ask {
        handle.step();
        shared.report(GenEvent::ToolStep { speech, step: handle.num_steps() });
    }
    let step = handle.num_steps();
    let sent = wake.send(RealtimeWake::ToolsDone { speech, step, outputs, ask }).is_ok();
    if ask && sent { StepOutcome::Next(finished) } else { StepOutcome::Done(finished) }
}

/// The open generation's events, as one segment's frames.
struct Generation<'a> {
    shared: &'a Arc<SessionShared>,
    handle: &'a Arc<SpeechHandle>,
    ctx: &'a ReplyContext,
    acc: &'a mut TurnAccumulator,
    events: &'a mut mpsc::UnboundedReceiver<Event>,
    /// The message the text deltas append to: a provider may interleave two.
    current: Option<usize>,
    first_token_at: Option<f64>,
    segments: Vec<String>,
    /// The calls, dispatched as they arrive.
    tools: &'a mut ToolStep,
    /// The generation's `response_id`, from its first event that names one.
    response_id: Option<String>,
    /// An empty round means released or interrupted.
    received: bool,
}

impl Segment for Generation<'_> {
    fn started_speaking(&mut self, at: f64) {
        first_frame(self.shared, self.acc, at);
    }

    async fn next(&mut self) -> Result<Next, DriverError> {
        loop {
            // The channel closed: the session ended, or the loop released this speech.
            let Some(ev) = self.events.recv().await else { return Ok(Next::End) };
            self.received = true;
            if self.response_id.is_none()
                && let Some(id) = realtime_state::response_id_of(&ev)
            {
                self.response_id = Some(id.to_owned());
            }
            let token = matches!(&ev, Event::AudioDelta(_))
                || matches!(&ev, Event::TextDelta(d) if !d.delta.is_empty());
            if token && self.first_token_at.is_none() {
                self.first_token_at = Some(self.shared.time.wall_secs());
            }
            match ev {
                Event::MessageStarted(_) => {
                    self.current = Some(self.segments.len());
                    self.segments.push(String::new());
                }
                Event::TextDelta(rt::TextDelta { delta, .. }) => {
                    if let Some(seg) = self.current.and_then(|idx| self.segments.get_mut(idx)) {
                        seg.push_str(&delta);
                    }
                }
                Event::AudioDelta(rt::AudioDelta { frame, .. }) => return Ok(Next::Frame(frame)),
                // A boundary, so `wait_for_playout` can finish this message while the next arrives.
                Event::MessageDone(_) => {
                    self.current = None;
                    return Ok(Next::Boundary);
                }
                Event::FunctionCall(ready) => {
                    // Stamped at its execution start: the call arrives during the generation,
                    // before the message it belongs to is placed.
                    let mut call = *ready.call;
                    call.created_at = self.shared.time.wall();
                    self.tools.dispatch(self.shared, self.handle, self.ctx, call);
                }
                Event::GenerationDone(_) => return Ok(Next::End),
                // The loop routes only a generation's own events here; the rest are the loop's.
                _ => {}
            }
        }
    }

    fn text(&self) -> String {
        self.segments.concat()
    }
}

/// The node in front of the session's sink; boundaries and interruption still reach `out` directly.
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

    async fn capture(&self, frame: &AudioFrame) -> lk_agents_voice::io::Result<()> {
        self.node.push(frame).await;
        Ok(())
    }

    fn flush_segment(&self) {
        // The tracker is `out`'s, and `AudioOutputExt::flush` on this node already moved it.
        self.out.flush_segment();
    }

    fn clear_buffer(&self) {
        self.out.clear_buffer();
    }

    fn pause(&self) {
        self.out.pause();
    }

    fn resume(&self) {
        self.out.resume();
    }
}
