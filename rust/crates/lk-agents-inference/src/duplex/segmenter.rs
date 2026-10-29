//! `_DuplexRealtimeSession`'s decisions, synchronous: a duplex model's continuous output cut into
//! generations at the gate's edges, its transcript fragments attached to the sound that carries
//! them, and the conversation the session sees. Every input returns the events it produced.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, SystemTime};

use lk_agents_abi::llm::realtime::{
    AudioDelta, Event, FunctionCallReady, GenerateReplyFailed, GenerationCreated, GenerationDone,
    GenerationStatus, MessageDone, MessageModalities, MessageStarted, Modalities, RealtimeError,
    RealtimeModelError, SessionReconnected, TextDelta, TokenUsage, TranscriptAttribution,
};
use lk_agents_abi::{
    AudioFrame, ChatContext, ChatItem, ChatMessage, CopyOptions, FunctionCall, Role, Ts,
};

/// Transcript no sound ever opens a burst for is emitted as a text-only generation this much audio
/// later, rather than lost. Audio is never held for it.
pub const UNCLAIMED_TRANSCRIPT_MS: i64 = 3000;

/// Buffered future transcript, shared across fragments; audio is never delayed for these words.
pub const MAX_TRANSCRIPT_FRAGMENTS: usize = 1024;
pub const MAX_TRANSCRIPT_BYTES: usize = 256 * 1024;

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
    kind: Kind,
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

/// The source of transcript fragments delivered after their speech generation completed.
#[derive(Debug)]
struct CompletedBurst {
    id: String,
    span_end_ms: i64,
}

/// Runtime actions produced by one segmenter transition.
#[derive(Debug, Default)]
pub struct Step {
    pub events: Vec<Event>,
    pub deactivate_gate: bool,
}

impl From<Vec<Event>> for Step {
    fn from(events: Vec<Event>) -> Self {
        Self { events, deactivate_gate: false }
    }
}

pub struct Segmenter {
    /// The model's label, on the error a failed gate reports.
    label: String,
    audio_timeout: Duration,
    id_prefix: String,
    next_id: u64,
    burst: Option<Burst>,
    completed: BTreeMap<i64, CompletedBurst>,
    /// The adapter's clock: output audio heard so far, which arrives at playback pace.
    audio_ms: i64,
    fragments: VecDeque<Fragment>,
    fragment_bytes: usize,
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
    pub fn new(label: impl Into<String>, audio_timeout: Duration, id_prefix: String) -> Self {
        Segmenter {
            label: label.into(),
            audio_timeout,
            id_prefix,
            next_id: 0,
            burst: None,
            completed: BTreeMap::new(),
            audio_ms: 0,
            fragments: VecDeque::new(),
            fragment_bytes: 0,
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
        [self.idle_deadline, self.pending_reply.as_ref().map(|(_, at)| *at)]
            .into_iter()
            .flatten()
            .min()
    }

    /// One frame of the model's output. A gate that fails ends the output for good.
    pub fn on_audio(
        &mut self,
        now: Ts,
        wall: SystemTime,
        frame: AudioFrame,
        start_ms: Option<i64>,
        gate_open: Result<bool, String>,
    ) -> Step {
        let mut out = Step::default();
        if self.failed {
            return out;
        }
        let duration_ms = (frame.duration() * 1000.0).round() as i64;
        if let Some(at) = start_ms {
            self.audio_ms = at;
        }
        // the gate is the one boundary: a burst is open exactly while the model is audibly
        // producing output, its own pauses inside an utterance included
        let open = match gate_open {
            Ok(open) => open,
            Err(message) => {
                self.failed = true;
                log::error!(target: "livekit.agents", "duplex audio stream failed: {message}");
                out.events.push(Event::Error(RealtimeModelError {
                    timestamp: now,
                    label: self.label.clone(),
                    message,
                    recoverable: false,
                    req_id: None,
                    code: None,
                    provider_error: None,
                }));
                self.close_burst(&mut out);
                return out;
            }
        };
        let frame_duration = Duration::from_secs_f64(frame.duration());
        if open {
            if self.burst.is_none() {
                let burst = self.open_burst(wall, Kind::Speech, &mut out.events);
                self.burst = Some(burst);
            }
            let burst = self.burst.as_mut().expect("just opened");
            out.events.push(Event::AudioDelta(AudioDelta {
                message_id: burst.id.clone(),
                response_id: Some(burst.id.clone()),
                frame,
            }));
            self.audio_ms += duration_ms;
            burst.heard_ms = self.audio_ms;
            self.attach_fragments(&mut out.events);
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
            self.flush_unclaimed(wall, &mut out.events);
        }
        out
    }

    /// The output stream ended (or failed, with `error`): the burst in flight ends with it.
    pub fn on_output_ended(&mut self, now: Ts, wall: SystemTime, error: Option<String>) -> Step {
        let mut out = Step::default();
        if let Some(message) = error {
            out.events.push(Event::Error(RealtimeModelError {
                timestamp: now,
                label: self.label.clone(),
                message,
                recoverable: false,
                req_id: None,
                code: None,
                provider_error: None,
            }));
        }
        self.close_burst(&mut out);
        self.flush_unclaimed(wall, &mut out.events);
        out
    }

    pub fn on_transcript(&mut self, now: Ts, wall: SystemTime, fragment: Fragment) -> Step {
        let mut out = Vec::new();
        if self.failed {
            return out.into();
        }
        let source = fragment.start_ms.and_then(|start| {
            self.completed
                .range(..=start)
                .next_back()
                .and_then(|(_, burst)| (start <= burst.span_end_ms).then(|| burst.id.clone()))
        });
        if let Some(source_response_id) = source {
            // Completed speech stays completed. A separate transcript generation names its source.
            let mut burst = self.open_burst(wall, Kind::Text, &mut out);
            out.insert(
                0,
                Event::TranscriptAttribution(TranscriptAttribution {
                    response_id: burst.id.clone(),
                    source_response_id,
                }),
            );
            Self::attach(&mut burst, fragment, &mut out);
            self.finish(burst, &mut out);
            return out.into();
        }
        if self.fragments.len() >= MAX_TRANSCRIPT_FRAGMENTS
            || fragment.text.capacity() > MAX_TRANSCRIPT_BYTES.saturating_sub(self.fragment_bytes)
        {
            self.failed = true;
            self.fragments.clear();
            self.fragment_bytes = 0;
            self.fail_pending_reply("duplex transcript backlog exceeded", &mut out);
            out.push(Event::Error(RealtimeModelError {
                timestamp: now,
                label: self.label.clone(),
                message: "duplex transcript backlog exceeded".into(),
                recoverable: false,
                req_id: None,
                code: Some("duplex_transcript_backlog_exceeded".into()),
                provider_error: Some(
                    lk_agents_abi::errors::ApiError::new("duplex transcript backlog exceeded")
                        .with_retryable(false),
                ),
            }));
            let mut step = out.into();
            self.close_burst(&mut step);
            return step;
        }
        if self.fragments.is_empty() {
            self.waiting_since_ms = self.audio_ms;
        }
        self.fragment_bytes += fragment.text.capacity();
        self.fragments.push_back(fragment);
        out.into()
    }

    /// A call joins the burst in flight so the tool runs while the model talks; alone, it is a
    /// generation of its own, over as soon as it is delivered.
    pub fn on_function_call(&mut self, call: FunctionCall) -> Vec<Event> {
        let mut out = Vec::new();
        if self.failed {
            return out;
        }
        self.chat_ctx.insert(ChatItem::FunctionCall(call.clone()));
        let wall = call.created_at;
        let ready = Event::FunctionCall(FunctionCallReady { call: Box::new(call) });
        if self.burst.is_some() {
            out.push(ready);
            return out;
        }
        let burst = self.open_burst(wall, Kind::Tool, &mut out);
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
    pub fn on_reconnected(&mut self) -> Step {
        let mut out = Step::default();
        self.fragments.clear();
        self.fragment_bytes = 0;
        self.close_burst(&mut out);
        self.completed.clear();
        self.fail_pending_reply(
            "the session reconnected before the model replied",
            &mut out.events,
        );
        out.events.push(Event::SessionReconnected(SessionReconnected {}));
        out
    }

    /// The reply is the next speech to open; a newer ask supersedes the one waiting.
    pub fn generate_reply(&mut self, now: Ts, req_id: String) -> Vec<Event> {
        let mut out = Vec::new();
        if self.failed {
            out.push(Event::GenerateReplyFailed(GenerateReplyFailed {
                req_id,
                error: RealtimeError::new("duplex output already ended"),
            }));
            return out;
        }
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

    pub fn tick(&mut self, now: Ts) -> Step {
        let mut out = Step::default();
        if self.pending_reply.as_ref().is_some_and(|(_, at)| *at <= now) {
            self.fail_pending_reply("the model did not start speaking when asked", &mut out.events);
        }
        if self.idle_deadline.is_some_and(|at| at <= now) {
            // the provider stopped sending, and that absence is silence too
            self.close_burst(&mut out);
        }
        out
    }

    pub fn close(&mut self) -> Step {
        let mut out = Step::default();
        self.failed = true;
        self.close_burst(&mut out);
        self.fragments.clear();
        self.fragment_bytes = 0;
        self.fail_pending_reply("the session closed before the model replied", &mut out.events);
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
        let diff = lk_agents_core::llm::ctx_diff::compute_chat_ctx_diff(&self.chat_ctx, &chat_ctx);
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
            out.push(Event::GenerateReplyFailed(GenerateReplyFailed {
                req_id,
                error: RealtimeError::new(reason),
            }));
        }
    }

    fn open_burst(&mut self, wall: SystemTime, kind: Kind, out: &mut Vec<Event>) -> Burst {
        self.next_id += 1;
        let id = format!("{}{}", self.id_prefix, self.next_id);
        // the model answers on the one stream it has, so speech opening is the reply asked for;
        // a call on its own is not, nor is text for sound already played
        let req_id =
            if kind == Kind::Speech { self.pending_reply.take().map(|(id, _)| id) } else { None };
        out.push(Event::GenerationCreated(GenerationCreated { response_id: id.clone(), req_id }));
        let has_message = kind != Kind::Tool;
        if has_message {
            out.push(Event::MessageStarted(MessageStarted {
                message_id: id.clone(),
                response_id: Some(id.clone()),
            }));
            out.push(Event::MessageModalities(MessageModalities {
                message_id: id.clone(),
                response_id: Some(id.clone()),
                modalities: Modalities::AudioAndText,
            }));
        }
        Burst {
            id,
            kind,
            audio_start_ms: self.audio_ms,
            heard_ms: self.audio_ms,
            anchor_ms: None,
            opened_at: wall,
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
            self.fragment_bytes -= f.text.capacity();
            Self::attach(burst, f, out);
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

    fn flush_unclaimed(&mut self, wall: SystemTime, out: &mut Vec<Event>) {
        if self.fragments.is_empty() {
            return;
        }
        let mut burst = self.open_burst(wall, Kind::Text, out);
        while let Some(fragment) = self.fragments.pop_front() {
            self.fragment_bytes -= fragment.text.capacity();
            Self::attach(&mut burst, fragment, out);
        }
        self.finish(burst, out);
    }

    fn close_burst(&mut self, out: &mut Step) {
        self.idle_deadline = None;
        out.deactivate_gate = true;
        if self.burst.is_some() {
            self.attach_fragments(&mut out.events);
            let burst = self.burst.take().expect("checked");
            self.finish(burst, &mut out.events);
        }
        self.waiting_since_ms = self.audio_ms;
    }

    /// End a generation; what it said enters the conversation under the id and time the framework
    /// uses for it, so a context update matches.
    fn finish(&mut self, burst: Burst, out: &mut Vec<Event>) {
        if burst.kind == Kind::Speech
            && let Some(anchor) = burst.anchor_ms
        {
            self.completed.insert(
                anchor + burst.audio_start_ms,
                CompletedBurst { id: burst.id.clone(), span_end_ms: anchor + burst.heard_ms },
            );
        }
        if burst.kind != Kind::Tool {
            out.push(Event::MessageDone(MessageDone {
                message_id: burst.id.clone(),
                response_id: Some(burst.id.clone()),
            }));
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
