//! `_DuplexRealtimeSession`'s decisions, synchronous: a duplex model's continuous output cut into
//! generations at the gate's edges, its transcript fragments attached to the sound that carries
//! them, and the conversation the session sees. Every input returns the events it produced.

use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

use lk_agents_abi::llm::realtime::{
    AudioDelta, Event, FunctionCallReady, GenerateReplyFailed, GenerationCreated, GenerationDone,
    GenerationStatus, MessageDone, MessageModalities, MessageStarted, Modalities, RealtimeError,
    RealtimeModelError, SessionReconnected, TextDelta, TokenUsage,
};
use lk_agents_abi::{AudioFrame, ChatContext, ChatItem, ChatMessage, CopyOptions, FunctionCall, Role, Ts};

use crate::duplex::gate::AudioGate;

/// Transcript no sound ever opens a burst for is emitted as a text-only generation this much audio
/// later, rather than lost. Audio is never held for it.
pub const UNCLAIMED_TRANSCRIPT_MS: i64 = 3000;

/// How long a requested reply waits for the model to start speaking before it counts as declined.
pub const REPLY_TIMEOUT: Duration = Duration::from_secs(10);

/// A provider may stop sending rather than stream its own silence; waiting out the frame in hand
/// plus this long is the silence the gate never received.
pub const AUDIO_TIMEOUT: Duration = Duration::from_millis(800);

/// How far ahead of its sound a fragment is handed over: span labels sit about a frame early
/// against the audible onset, and text a little early reads better than late.
pub const ATTACH_LEAD_MS: i64 = 300;

/// The message the framework keeps its task instructions under (`voice/tool_output.py`).
const INSTRUCTIONS_MESSAGE_ID: &str = "lk.agent_task.instructions";

/// A fragment of the model's transcript of its own speech, on the model's span clock.
#[derive(Debug, Clone, PartialEq)]
pub struct Fragment {
    pub text: String,
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Speech,
    Text,
    Tool,
}

/// One stretch of audible output, presented as a generation.
#[derive(Debug)]
struct Burst {
    id: String,
    has_message: bool,
    /// Where on the audio clock the burst opened.
    audio_start_ms: i64,
    /// The audio clock after the last frame the burst forwarded.
    heard_ms: i64,
    /// Span clock minus audio clock, fixed by the first fragment.
    anchor_ms: Option<i64>,
    opened_at: SystemTime,
    transcript: String,
    last_annotation: f64,
}

/// What the previous anchored burst left behind: its anchor tells text for audio it already played
/// from the words of the next utterance.
#[derive(Debug, Clone)]
struct LastBurst {
    id: String,
    anchor_ms: Option<i64>,
    heard_ms: i64,
}

pub struct Segmenter {
    gate: Box<dyn AudioGate>,
    /// The model's label, on the error a failed gate reports.
    label: String,
    audio_timeout: Duration,
    ids: Box<dyn FnMut() -> String + Send>,
    burst: Option<Burst>,
    last: Option<LastBurst>,
    /// The adapter's clock: output audio heard so far, which arrives at playback pace.
    audio_ms: i64,
    fragments: VecDeque<Fragment>,
    waiting_since_ms: i64,
    /// The reply asked for: the next speech to open answers it.
    pending_reply: Option<(String, Ts)>,
    /// The end of the frame in hand plus `audio_timeout`, while a burst is open.
    idle_deadline: Option<Ts>,
    /// The conversation as the framework sees it: what a context update diffs against.
    chat_ctx: ChatContext,
    failed: bool,
}

impl Segmenter {
    pub fn new(
        gate: Box<dyn AudioGate>,
        label: impl Into<String>,
        audio_timeout: Duration,
        ids: Box<dyn FnMut() -> String + Send>,
    ) -> Self {
        Segmenter {
            gate,
            label: label.into(),
            audio_timeout,
            ids,
            burst: None,
            last: None,
            audio_ms: 0,
            fragments: VecDeque::new(),
            waiting_since_ms: 0,
            pending_reply: None,
            idle_deadline: None,
            chat_ctx: ChatContext::default(),
            failed: false,
        }
    }

    pub fn chat_ctx(&self) -> &ChatContext {
        &self.chat_ctx
    }

    pub fn audio_ms(&self) -> i64 {
        self.audio_ms
    }

    pub fn burst_open(&self) -> bool {
        self.burst.is_some()
    }

    pub fn waiting_fragments(&self) -> usize {
        self.fragments.len()
    }

    pub fn next_deadline(&self) -> Option<Ts> {
        [self.idle_deadline, self.pending_reply.as_ref().map(|(_, at)| *at)].into_iter().flatten().min()
    }

    /// One frame of the model's output. A gate that fails ends the output for good.
    pub fn on_audio(&mut self, now: Ts, frame: AudioFrame, start_ms: Option<i64>) -> Vec<Event> {
        let mut out = Vec::new();
        if self.failed {
            return out;
        }
        let duration_ms = (frame.duration() * 1000.0).round() as i64;
        if let Some(at) = start_ms {
            self.audio_ms = at;
        }
        // the gate is the one boundary: a burst is open exactly while the model is audibly
        // producing output, its own pauses inside an utterance included
        let open = match self.gate.update(&frame) {
            Ok(open) => open,
            Err(message) => {
                self.failed = true;
                log::error!(target: "livekit.agents", "duplex audio stream failed: {message}");
                out.push(Event::Error(RealtimeModelError {
                    timestamp: now,
                    label: self.label.clone(),
                    message,
                    recoverable: false,
                    req_id: None,
                    code: None,
                }));
                self.close_burst(&mut out);
                return out;
            }
        };
        let frame_duration = Duration::from_secs_f64(frame.duration());
        if open {
            if self.burst.is_none() {
                let burst = self.open_burst(Kind::Speech, &mut out);
                self.burst = Some(burst);
            }
            let burst = self.burst.as_mut().expect("just opened");
            out.push(Event::AudioDelta(AudioDelta {
                message_id: burst.id.clone(),
                response_id: Some(burst.id.clone()),
                frame,
            }));
            self.audio_ms += duration_ms;
            burst.heard_ms = self.audio_ms;
            self.attach_fragments(&mut out);
            // from the end of the audio just handed over, not from its arrival
            self.idle_deadline = Some(now.saturating_add(self.audio_timeout + frame_duration));
            return out;
        }

        self.audio_ms += duration_ms;
        if self.burst.is_some() {
            self.close_burst(&mut out);
        } else if !self.fragments.is_empty()
            && self.audio_ms - self.waiting_since_ms >= UNCLAIMED_TRANSCRIPT_MS
        {
            log::error!(target: "livekit.agents", "duplex transcript outlived the audio it describes");
            let mut burst = self.open_burst(Kind::Text, &mut out);
            while let Some(f) = self.fragments.pop_front() {
                Self::attach(&mut burst, f, &mut out);
            }
            self.finish(burst, &mut out);
        }
        out
    }

    /// The output stream ended (or failed, with `error`): the burst in flight ends with it.
    pub fn on_output_ended(&mut self, now: Ts, error: Option<String>) -> Vec<Event> {
        let mut out = Vec::new();
        if let Some(message) = error {
            out.push(Event::Error(RealtimeModelError {
                timestamp: now,
                label: self.label.clone(),
                message,
                recoverable: false,
                req_id: None,
                code: None,
            }));
        }
        self.close_burst(&mut out);
        out
    }

    pub fn on_transcript(&mut self, fragment: Fragment) -> Vec<Event> {
        let mut out = Vec::new();
        if self.fragments.is_empty() {
            self.waiting_since_ms = self.audio_ms;
        }
        self.fragments.push_back(fragment);
        // words for sound the last burst already played go out at once, as text on their own,
        // rather than wait for the next sound and be taken for its words
        let Some(last) = self.last.clone() else { return out };
        let Some(anchor) = last.anchor_ms else { return out };
        if self.burst.as_ref().is_some_and(|b| b.id == last.id) {
            return out;
        }
        let late = self
            .fragments
            .iter()
            .take_while(|f| f.start_ms.is_some_and(|s| s - anchor <= last.heard_ms))
            .count();
        if late > 0 {
            let mut burst = self.open_burst(Kind::Text, &mut out);
            for _ in 0..late {
                let f = self.fragments.pop_front().expect("counted");
                Self::attach(&mut burst, f, &mut out);
            }
            self.finish(burst, &mut out);
        }
        out
    }

    /// A call joins the burst in flight so the tool runs while the model talks; alone, it is a
    /// generation of its own, over as soon as it is delivered.
    pub fn on_function_call(&mut self, call: FunctionCall) -> Vec<Event> {
        let mut out = Vec::new();
        self.chat_ctx.insert(ChatItem::FunctionCall(call.clone()));
        let ready = Event::FunctionCall(FunctionCallReady { call: Box::new(call) });
        if self.burst.is_some() {
            out.push(ready);
            return out;
        }
        let burst = self.open_burst(Kind::Tool, &mut out);
        out.push(ready);
        self.finish(burst, &mut out);
        out
    }

    /// A final transcription of the caller enters the conversation under the framework's id.
    pub fn on_input_transcription(
        &mut self,
        item_id: &str,
        transcript: &str,
        is_final: bool,
        confidence: Option<f64>,
        started_at: SystemTime,
    ) {
        if !is_final {
            return;
        }
        let mut msg = ChatMessage::text(item_id, Role::User, transcript, started_at);
        msg.transcript_confidence = Some(confidence.unwrap_or(1.0));
        self.chat_ctx.insert(ChatItem::Message(msg));
    }

    /// A dropped connection never delivers the rest of a burst, the sound its waiting words
    /// describe, or the reply it was asked for.
    pub fn on_reconnected(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        self.fragments.clear();
        self.close_burst(&mut out);
        self.last = None;
        self.fail_pending_reply("the session reconnected before the model replied", &mut out);
        out.push(Event::SessionReconnected(SessionReconnected {}));
        out
    }

    /// The reply is the next speech to open; a newer ask supersedes the one waiting.
    pub fn generate_reply(&mut self, now: Ts, req_id: String) -> Vec<Event> {
        let mut out = Vec::new();
        self.fail_pending_reply("a newer ask superseded this one", &mut out);
        self.pending_reply = Some((req_id, now.saturating_add(REPLY_TIMEOUT)));
        out
    }

    /// The model refused the ask (or the caller dropped it): nothing waits for it any more.
    pub fn forget_reply(&mut self, req_id: &str) {
        if self.pending_reply.as_ref().is_some_and(|(id, _)| id == req_id) {
            self.pending_reply = None;
        }
    }

    pub fn tick(&mut self, now: Ts) -> Vec<Event> {
        let mut out = Vec::new();
        if self.pending_reply.as_ref().is_some_and(|(_, at)| *at <= now) {
            self.fail_pending_reply("the model did not start speaking when asked", &mut out);
        }
        if self.idle_deadline.is_some_and(|at| at <= now) {
            // the provider stopped sending, and that absence is silence too
            self.close_burst(&mut out);
        }
        out
    }

    pub fn close(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        self.close_burst(&mut out);
        self.fail_pending_reply("the session closed before the model replied", &mut out);
        out
    }

    /// The startup configuration's history: the adapter's record and the model's both start from
    /// it. Returns the items to hand the model.
    pub fn seed(&mut self, chat_ctx: &ChatContext) -> Vec<ChatItem> {
        self.chat_ctx = chat_ctx.copy(&exclusions());
        self.chat_ctx.items.clone()
    }

    /// A context update, diffed against the adapter's own record: only what is new reaches the
    /// model. The framework records the model's speech as it was played, which may be less than
    /// what it said; anything else edited or removed is logged, since the model keeps it.
    pub fn update_chat_ctx(&mut self, chat_ctx: &ChatContext) -> Vec<ChatItem> {
        let mut chat_ctx = chat_ctx.copy(&exclusions());
        chat_ctx.items.retain(|i| i.id() != INSTRUCTIONS_MESSAGE_ID);
        let diff = lk_agents_abi::compute_chat_ctx_diff(&self.chat_ctx, &chat_ctx);
        let edited: Vec<&String> = diff
            .to_remove
            .iter()
            .chain(diff.to_update.iter().map(|(_, id)| id))
            .filter(|id| {
                !matches!(self.chat_ctx.get_by_id(id), Some(ChatItem::Message(m)) if m.role == Role::Assistant)
            })
            .collect();
        if !edited.is_empty() {
            log::warn!(target: "livekit.agents", item_ids:? = edited;
                "duplex context is append-only; the model keeps what it has been told");
        }
        let new_items: Vec<ChatItem> =
            diff.to_create.iter().filter_map(|(_, id)| chat_ctx.get_by_id(id).cloned()).collect();
        self.chat_ctx = chat_ctx;
        new_items
    }

    /// A single item the session appends (an image of the caller's video, say).
    pub fn push_item(&mut self, item: ChatItem) {
        self.chat_ctx.insert(item);
    }

    fn fail_pending_reply(&mut self, reason: &str, out: &mut Vec<Event>) {
        if let Some((req_id, _)) = self.pending_reply.take() {
            out.push(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error: RealtimeError::new(reason) }));
        }
    }

    fn open_burst(&mut self, kind: Kind, out: &mut Vec<Event>) -> Burst {
        let id = (self.ids)();
        // the model answers on the one stream it has, so speech opening is the reply asked for;
        // a call on its own is not, nor is text for sound already played
        let req_id = if kind == Kind::Speech { self.pending_reply.take().map(|(id, _)| id) } else { None };
        out.push(Event::GenerationCreated(GenerationCreated { response_id: id.clone(), req_id }));
        let has_message = kind != Kind::Tool;
        if has_message {
            out.push(Event::MessageStarted(MessageStarted { message_id: id.clone(), response_id: Some(id.clone()) }));
            out.push(Event::MessageModalities(MessageModalities {
                message_id: id.clone(),
                response_id: Some(id.clone()),
                modalities: Modalities::AudioAndText,
            }));
        }
        Burst {
            id,
            has_message,
            audio_start_ms: self.audio_ms,
            heard_ms: self.audio_ms,
            anchor_ms: None,
            opened_at: SystemTime::now(),
            transcript: String::new(),
            last_annotation: 0.0,
        }
    }

    /// The first fragment anchors the span clock to the audio clock; a later one is due when the
    /// audio reaches its span, and one this burst never reaches waits for the next.
    fn attach_fragments(&mut self, out: &mut Vec<Event>) {
        let Some(burst) = self.burst.as_mut() else { return };
        while let Some(f) = self.fragments.front() {
            if let Some(start) = f.start_ms {
                let anchor = *burst.anchor_ms.get_or_insert(start - burst.audio_start_ms);
                if start - anchor > burst.heard_ms + ATTACH_LEAD_MS {
                    break;
                }
            }
            let f = self.fragments.pop_front().expect("peeked");
            Self::attach(burst, f, out);
        }
        if burst.anchor_ms.is_some() {
            self.last = Some(LastBurst { id: burst.id.clone(), anchor_ms: burst.anchor_ms, heard_ms: burst.heard_ms });
        }
    }

    /// Placed on the forwarded audio so the synchroniser paces against real speech; the
    /// annotations never go backwards, since it indexes them by time.
    fn attach(burst: &mut Burst, f: Fragment, out: &mut Vec<Event>) {
        burst.transcript.push_str(&f.text);
        let start_time = match (f.start_ms, burst.anchor_ms) {
            (Some(s), Some(anchor)) => {
                let offset = anchor + burst.audio_start_ms;
                let start = burst.last_annotation.max((s - offset) as f64 / 1000.0);
                let end = f.end_ms.map(|e| start.max((e - offset) as f64 / 1000.0));
                burst.last_annotation = end.unwrap_or(start);
                Some(start)
            }
            _ => None,
        };
        out.push(Event::TextDelta(TextDelta {
            message_id: burst.id.clone(),
            response_id: Some(burst.id.clone()),
            delta: f.text,
            start_time,
        }));
    }

    fn close_burst(&mut self, out: &mut Vec<Event>) {
        self.idle_deadline = None;
        // the gate never stays open past the burst it opened, so the next opens on sound again
        self.gate.deactivate();
        if self.burst.is_some() {
            // words that arrived after the last frame still describe its sound
            self.attach_fragments(out);
            let burst = self.burst.take().expect("checked");
            self.finish(burst, out);
        }
        self.waiting_since_ms = self.audio_ms;
    }

    /// End a generation; what it said enters the conversation under the id and time the framework
    /// uses for it, so a context update matches.
    fn finish(&mut self, burst: Burst, out: &mut Vec<Event>) {
        if let Some(last) = self.last.as_mut()
            && last.id == burst.id
        {
            last.heard_ms = burst.heard_ms;
        }
        if burst.has_message {
            out.push(Event::MessageDone(MessageDone { message_id: burst.id.clone(), response_id: Some(burst.id.clone()) }));
        }
        out.push(Event::GenerationDone(GenerationDone {
            response_id: burst.id.clone(),
            status: GenerationStatus::Completed,
            usage: TokenUsage::default(),
        }));
        if !burst.transcript.is_empty() {
            self.chat_ctx.insert(ChatItem::Message(ChatMessage::text(
                burst.id,
                Role::Assistant,
                burst.transcript,
                burst.opened_at,
            )));
        }
    }
}

fn exclusions() -> CopyOptions {
    CopyOptions { exclude_handoff: true, exclude_config_update: true, ..CopyOptions::default() }
}
