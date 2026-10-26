//! `RealtimeSession`'s body as a synchronous state machine. A caller `req_id` goes out as
//! `metadata.client_event_id`; a rejection, timeout or cancel must leave a late `response.created` recognisable as stale.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

pub(crate) use lk_agents_abi::RealtimeCommand as Command;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{AudioByteStream, Maybe, Ts};
use lk_agents_abi::{ChatContext, ChatItem, compute_chat_ctx_diff};
use lk_agents_abi::{ToolChoice, ToolSpec};
use serde_json::{Map, Value, json};

use crate::realtime::dialect::{
    DEFAULT_MAX_OUTPUT_TOKENS, Dialect, NUM_CHANNELS, SAMPLE_RATE, TurnDetection,
};
use lk_agents_media::{FrameResampler, OnChange, Quality};
use crate::realtime::remote::RemoteChatContext;
use crate::realtime::wire::{self, IdGen, SessionPatch, TranscriptionCompleted};
use lk_agents_abi::llm::realtime::{
    Event, GenerateReplyFailed, GenerationDone, GenerationStatus, InputTranscription, MessageDone,
    MessageModalities, Modalities, RealtimeCapabilities, RealtimeError, RealtimeModelError,
    TokenUsage,
};

mod incoming;
#[cfg(test)]
mod tests;

/// `asyncio.get_event_loop().call_later(10.0, _on_timeout)` in `generate_reply`.
pub const GENERATE_REPLY_TIMEOUT: Duration = Duration::from_secs(10);
/// `asyncio.wait_for(..., timeout=5.0)` in `update_chat_ctx`.
pub const UPDATE_CHAT_CTX_TIMEOUT: Duration = Duration::from_secs(5);
/// `_schedule_stale_say_cleanup`'s `call_later(10.0, ...)`.
pub const STALE_SAY_CLEANUP: Duration = Duration::from_secs(10);

pub struct RealtimeMachine {
    dialect: Dialect,
    opts: SessionOptions,
    caps: RealtimeCapabilities,
    ids: Arc<dyn IdGen>,
    images: Arc<dyn ImageResolver + Send + Sync>,
    label: String,
    /// `Metadata(model_name=model, model_provider=provider)` on the metrics this session reports.
    provider: String,
    /// The response in flight: when it was created, and when its first output arrived.
    response_created_at: Option<Ts>,
    first_token_at: Option<Ts>,

    remote: RemoteChatContext,
    tools: Vec<ToolSpec>,

    generation: Generation,
    /// `_response_created_futures`, reduced to its deadline.
    pending_replies: HashMap<String, Ts>,
    /// `_discarded_event_ids`: requests abandoned before the server created their response. The
    /// response is cancelled by id and its trailing events skipped when it finally arrives.
    discarded: HashSet<String>,
    /// xAI: `_pending_say_event_ids`. `force_message` gets a `response.created` with no
    /// `client_event_id`, so the oldest un-answered say claims it.
    say_tags: VecDeque<String>,
    /// xAI: `_schedule_stale_say_cleanup`. Drops a tag the server never answered, so it cannot
    /// steal a later unrelated reply.
    stale_say: HashMap<String, Ts>,

    ctx_update: Option<CtxUpdate>,

    // Per-connection input turn state, discarded on reconnect.
    transcript_acc: HashMap<String, HashMap<usize, String>>,
    /// Per `item_id`: a late transcript would otherwise consume the *next* turn's start time.
    speech_started_at: HashMap<String, Ts>,
    /// xAI: the transcript being held.
    held: Option<TranscriptionCompleted>,
    /// xAI: whether the response in flight has produced any output at all.
    response_spoke: bool,

    bstream: AudioByteStream,
    /// `_resample_audio`: built for the first frame that is not 24 kHz mono.
    resampler: FrameResampler,
    pushed: Duration,
    /// The `response.create` of every pending reply, sent again by a reconnection.
    reply_frames: HashMap<String, Value>,

    session_deadline: Option<Ts>,
    closing: bool,
}

impl RealtimeMachine {
    pub fn new(
        dialect: Dialect,
        opts: SessionOptions,
        caps: RealtimeCapabilities,
        ids: Arc<dyn IdGen>,
        images: Arc<dyn ImageResolver + Send + Sync>,
    ) -> Self {
        let label = format!("lk_agents_realtime.{}", dialect.provider_label);
        RealtimeMachine {
            dialect,
            opts,
            caps,
            ids,
            images,
            label,
            provider: String::new(),
            response_created_at: None,
            first_token_at: None,
            remote: RemoteChatContext::new(),
            tools: Vec::new(),
            generation: Generation::None,
            pending_replies: HashMap::new(),
            discarded: HashSet::new(),
            say_tags: VecDeque::new(),
            stale_say: HashMap::new(),
            ctx_update: None,
            transcript_acc: HashMap::new(),
            speech_started_at: HashMap::new(),
            held: None,
            response_spoke: false,
            // 100 ms chunks: `samples_per_channel=None` is the reference's `SAMPLE_RATE // 10`.
            bstream: AudioByteStream::new(SAMPLE_RATE, NUM_CHANNELS, None, false),
            resampler: FrameResampler::new(SAMPLE_RATE, Quality::Medium, OnChange::Discard),
            pushed: Duration::ZERO,
            reply_frames: HashMap::new(),
            session_deadline: None,
            closing: false,
        }
    }

    /// The host model's `label`, on every error this session reports.
    pub fn with_label(mut self, label: String) -> Self {
        self.label = label;
        self
    }

    /// The host model's `provider`, for the metrics' metadata.
    pub fn with_provider(mut self, provider: String) -> Self {
        self.provider = provider;
        self
    }

    fn metadata(&self, model: &str) -> lk_agents_abi::metrics::base::Metadata {
        lk_agents_abi::metrics::base::Metadata {
            model_name: Some(model.to_owned()),
            model_provider: Some(self.provider.clone()),
        }
    }

    /// `_report_connection_acquired`: the connection's timing as a `RealtimeModelMetrics` with zero
    /// usage.
    pub fn connection_metrics(&self, now: Ts, acquire_time: Duration) -> Event {
        let metrics = lk_agents_abi::metrics::base::RealtimeModelMetrics {
            label: String::new(),
            request_id: String::new(),
            timestamp: secs(now),
            duration: 0.0,
            session_duration: 0.0,
            ttft: lk_agents_abi::metrics::base::RealtimeModelMetrics::NO_TTFT,
            cancelled: false,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            tokens_per_second: 0.0,
            input_token_details: Default::default(),
            output_token_details: Default::default(),
            acquire_time: acquire_time.as_secs_f64(),
            connection_reused: false,
            metadata: Some(self.metadata(&self.opts.model)),
        };
        Event::Metrics(lk_agents_abi::llm::realtime::RealtimeMetrics {
            metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::RealtimeModel(metrics)),
        })
    }

    /// The finished response's `RealtimeModelMetrics`, as `_handle_response_done` computes them.
    fn response_metrics(&mut self, now: Ts, response_id: &str, status: GenerationStatus, usage: &TokenUsage) -> Option<Event> {
        use lk_agents_abi::metrics::base::{CachedTokenDetails, InputTokenDetails, OutputTokenDetails, RealtimeModelMetrics};
        let created = self.response_created_at.take()?;
        let first = self.first_token_at.take();
        let duration = secs(now) - secs(created);
        let tokens = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        let metrics = RealtimeModelMetrics {
            label: self.label.clone(),
            request_id: response_id.to_owned(),
            timestamp: secs(created),
            duration,
            session_duration: 0.0,
            ttft: first.map_or(RealtimeModelMetrics::NO_TTFT, |f| secs(f) - secs(created)),
            cancelled: status == GenerationStatus::Cancelled,
            input_tokens: tokens(usage.input_tokens),
            output_tokens: tokens(usage.output_tokens),
            total_tokens: tokens(usage.total_tokens),
            tokens_per_second: if duration > 0.0 { usage.output_tokens as f64 / duration } else { 0.0 },
            input_token_details: InputTokenDetails {
                audio_tokens: tokens(usage.input_audio_tokens),
                text_tokens: tokens(usage.input_text_tokens),
                image_tokens: tokens(usage.input_image_tokens),
                cached_tokens: tokens(usage.input_cached_tokens),
                cached_tokens_details: Some(CachedTokenDetails {
                    audio_tokens: tokens(usage.cached_audio_tokens),
                    text_tokens: tokens(usage.cached_text_tokens),
                    image_tokens: tokens(usage.cached_image_tokens),
                }),
            },
            output_token_details: OutputTokenDetails {
                text_tokens: tokens(usage.output_text_tokens),
                audio_tokens: tokens(usage.output_audio_tokens),
                image_tokens: tokens(usage.output_image_tokens),
            },
            acquire_time: 0.0,
            connection_reused: false,
            metadata: Some(self.metadata(&self.opts.model)),
        };
        Some(Event::Metrics(lk_agents_abi::llm::realtime::RealtimeMetrics {
            metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::RealtimeModel(metrics)),
        }))
    }

    pub fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    /// `self.chat_ctx` -- the *mirror* (what the server believes), not the agent's context.
    pub fn chat_ctx(&self) -> ChatContext {
        self.remote.to_chat_ctx()
    }

    /// `has_active_generation`. A requested-but-not-created reply counts, so `interrupt()` works
    /// before `response.created`.
    pub fn has_active_generation(&self) -> bool {
        !matches!(self.generation, Generation::None) || !self.pending_replies.is_empty()
    }

    pub fn open(&mut self, now: Ts) -> Step {
        self.session_deadline = self.opts.max_session_duration.map(|d| now.saturating_add(d));
        Step::frame(self.session_update_frame())
    }

    /// `_reconnect()`. Reset turn state first (it settles into the mirror), then empty the mirror
    /// so the diff produces a create for every item.
    pub fn reconnect(&mut self, now: Ts) -> Step {
        let mut step = Step::default();
        self.session_deadline = self.opts.max_session_duration.map(|d| now.saturating_add(d));
        step.frames.push(self.session_update_frame());
        if !self.tools.is_empty() {
            step.frames.push(self.tools_update_frame());
        }

        step.merge(self.reset_input_turn_state());

        let replay = self.remote.to_chat_ctx();
        self.remote = RemoteChatContext::new();
        let (frames, _) = self.chat_ctx_events(&replay);
        step.frames.extend(frames);
        // The replay holds only confirmed items: an item event the lost connection never
        // confirmed goes again after it, and its confirmation settles the update.
        if let Some(update) = &self.ctx_update {
            step.frames.extend(
                update.frames.iter().filter(|(id, _)| update.by_event_id.contains_key(id)).map(|(_, f)| f.clone()),
            );
        }

        self.discarded.clear();
        self.say_tags.clear();
        self.stale_say.clear();
        step.merge(self.close_generation());
        // A reply asked for and not yet created is asked for again, now that the conversation it
        // answers is back; a `say()` (no `response.create` to resend) is failed.
        let mut pending: Vec<String> = self.pending_replies.keys().cloned().collect();
        pending.sort();
        for req_id in pending {
            match self.reply_frames.get(&req_id) {
                Some(frame) => {
                    self.pending_replies.insert(req_id, now.saturating_add(GENERATE_REPLY_TIMEOUT));
                    step.frames.push(frame.clone());
                }
                None => {
                    self.pending_replies.remove(&req_id);
                    step.events.push(Event::GenerateReplyFailed(GenerateReplyFailed {
                        req_id,
                        error: RealtimeError::new(
                            "pending response discarded due to session reconnection",
                        ),
                    }));
                }
            }
        }
        step.events.push(Event::SessionReconnected(lk_agents_abi::llm::realtime::SessionReconnected {}));
        step
    }

    pub fn close(&mut self, now: Ts) -> Step {
        self.closing = true;
        let mut step = self.flush_held_transcript(now);
        step.merge(self.close_generation());
        for req_id in std::mem::take(&mut self.pending_replies).into_keys() {
            step.events.push(Event::GenerateReplyFailed (GenerateReplyFailed {
                req_id,
                error: RealtimeError::new("realtime session closed"),
            }));
        }
        if let Some(ctx) = self.ctx_update.take() {
            step.events.push(Event::ChatCtxUpdated { req_id: ctx.req_id, rejected: ctx.rejected });
        }
        step
    }

    /// `_reset_input_turn_state`. On xAI this flushes the held transcript first, as the override
    /// does; discarding it would drop the last user turn of every recycled session.
    fn reset_input_turn_state(&mut self) -> Step {
        let step = if self.dialect.defer_transcript_finalization {
            self.flush_held_transcript(Ts::ZERO)
        } else {
            Step::default()
        };
        self.transcript_acc.clear();
        self.speech_started_at.clear();
        self.response_spoke = false;
        step
    }

    pub fn next_deadline(&self) -> Option<Ts> {
        self.pending_replies
            .values()
            .chain(self.stale_say.values())
            .chain(self.ctx_update.iter().map(|c| &c.deadline))
            .chain(self.session_deadline.iter())
            .copied()
            .min()
    }

    pub fn tick(&mut self, now: Ts) -> Step {
        let mut step = Step::default();

        let timed_out: Vec<String> = self
            .pending_replies
            .iter()
            .filter(|&(_, &at)| at <= now)
            .map(|(k, _)| k.clone())
            .collect();
        for req_id in timed_out {
            self.pending_replies.remove(&req_id);
            // `response.create` is already on the wire; remember the id so a late response is
            // cancelled and skipped rather than surfacing as an unsolicited turn.
            self.discarded.insert(req_id.clone());
            if self.dialect.force_message_say {
                self.stale_say.insert(req_id.clone(), now.saturating_add(STALE_SAY_CLEANUP));
            }
            step.events.push(Event::GenerateReplyFailed (GenerateReplyFailed {
                req_id,
                error: RealtimeError::new("generate_reply timed out."),
            }));
        }

        let stale: Vec<String> =
            self.stale_say.iter().filter(|&(_, &at)| at <= now).map(|(k, _)| k.clone()).collect();
        for req_id in stale {
            self.stale_say.remove(&req_id);
            // Only if still discarded: an answered tag was consumed, and removing it here would
            // drop a live correlation.
            if self.discarded.remove(&req_id) {
                self.say_tags.retain(|t| t != &req_id);
            }
        }

        if self.ctx_update.as_ref().is_some_and(|c| c.deadline <= now) {
            let ctx = self.ctx_update.take().expect("just checked");
            step.events.push(Event::ChatCtxUpdated {
                req_id: ctx.req_id,
                rejected: vec!["update_chat_ctx timed out.".to_owned()],
            });
        }

        if self.session_deadline.is_some_and(|at| at <= now) {
            // Recycle between turns, never mid-sentence: the deadline stays armed and re-checks
            // on the next tick (the reference awaits `_done_fut`).
            if matches!(self.generation, Generation::Active { .. }) {
                step.recycle = false;
            } else {
                self.session_deadline = None;
                step.recycle = true;
            }
        }

        step
    }

    pub fn command(&mut self, now: Ts, cmd: Command) -> Step {
        match cmd {
            Command::UpdateInstructions(lk_agents_abi::llm::realtime::UpdateInstructions { instructions }) => {
                let mut session = Map::new();
                session.insert("type".into(), json!("realtime"));
                session.insert("instructions".into(), json!(instructions));
                self.opts.instructions = Some(instructions);
                self.dialect.rewrite_session(&mut session);
                Step::frame(wire::session_update(&self.ids.short("instructions_update_"), session))
            }

            Command::UpdateTools(lk_agents_abi::llm::realtime::UpdateTools { tools }) => {
                self.tools = tools;
                Step::frame(self.tools_update_frame())
            }

            Command::UpdateChatCtx(lk_agents_abi::llm::realtime::UpdateChatCtx { req_id, chat_ctx }) => {
                let (frames, mut update) = self.chat_ctx_events(&chat_ctx);
                if !update.pending() {
                    // Empty `futs`: the reference returns immediately rather than waiting 5 s.
                    return Step {
                        frames,
                        events: vec![Event::ChatCtxUpdated { req_id, rejected: vec![] }],
                        ..Default::default()
                    };
                }
                update.req_id = req_id;
                update.deadline = now.saturating_add(UPDATE_CHAT_CTX_TIMEOUT);
                self.ctx_update = Some(update);
                Step { frames, ..Default::default() }
            }

            Command::PushAudio(lk_agents_abi::llm::realtime::PushAudio { frame }) => {
                let mut step = Step::default();
                let frames = match self.resampler.push(&frame) {
                    Ok(frames) => frames,
                    Err(e) => return Step::event(Event::Error(self.error(now, e.to_string(), true))),
                };
                for frame in frames {
                    for f in self.bstream.push(&frame.data) {
                        step.frames.push(wire::input_audio_buffer_append(&f.data));
                        self.pushed += Duration::from_secs_f64(
                            f.samples_per_channel as f64 / SAMPLE_RATE as f64,
                        );
                    }
                }
                step
            }

            Command::PushItem(lk_agents_abi::llm::realtime::PushItem { item, .. }) => match wire::to_wire_item(&item, self.images.as_ref()) {
                Ok(v) => {
                    Step::frame(wire::conversation_item_create(&self.ids.short("video_"), v, None))
                }
                Err(e) => Step::event(Event::Error(self.error(now, e, true))),
            },

            Command::CommitAudio(_) => {
                // "OpenAI requires at least 100ms of audio"; less gets
                // `input_audio_buffer_commit_empty` back.
                if self.pushed > Duration::from_millis(100) {
                    self.pushed = Duration::ZERO;
                    Step::frame(wire::input_audio_buffer_commit())
                } else {
                    Step::default()
                }
            }

            Command::ClearAudio(_) => {
                self.pushed = Duration::ZERO;
                Step::frame(wire::input_audio_buffer_clear())
            }

            Command::GenerateReply(lk_agents_abi::llm::realtime::GenerateReply { req_id, instructions, tool_choice }) => {
                self.pending_replies
                    .insert(req_id.clone(), now.saturating_add(GENERATE_REPLY_TIMEOUT));
                // OpenAI replaces session instructions with per-response ones, so concatenate;
                // sending the extra alone drops the system prompt for that turn.
                let merged = match (&self.opts.instructions, instructions) {
                    (Some(base), Some(extra)) => Some(format!("{base}\n{extra}")),
                    (_, other) => other,
                };
                let frame = wire::response_create(
                    &req_id,
                    merged.as_deref(),
                    // Without `per_response_tool_choice` the server would apply it session-wide.
                    tool_choice
                        .filter(|_| self.caps.per_response_tool_choice)
                        .map(|tc| wire::tool_choice(Some(&tc))),
                );
                self.reply_frames.retain(|id, _| self.pending_replies.contains_key(id));
                self.reply_frames.insert(req_id, frame.clone());
                Step::frame(frame)
            }

            Command::CancelReply(lk_agents_abi::llm::realtime::CancelReply { req_id }) => {
                if self.pending_replies.remove(&req_id).is_none() {
                    return Step::default();
                }
                let mut step = Step::frame(wire::response_cancel(None));
                // The cancel is a no-op if the response was never created; the id discards it
                // when it arrives anyway.
                self.discarded.insert(req_id.clone());
                if self.dialect.force_message_say {
                    self.stale_say.insert(req_id.clone(), now.saturating_add(STALE_SAY_CLEANUP));
                    if !self.say_tags.contains(&req_id) {
                        self.say_tags.push_back(req_id.clone());
                    }
                }
                step.events.push(Event::GenerateReplyFailed (GenerateReplyFailed {
                    req_id,
                    error: RealtimeError::new("generate_reply cancelled"),
                }));
                step
            }

            Command::Say(lk_agents_abi::llm::realtime::Say { req_id, text, .. }) => {
                if !self.dialect.force_message_say {
                    return Step::event(Event::GenerateReplyFailed(GenerateReplyFailed {
                        req_id,
                        error: RealtimeError::new(format!(
                            "{} does not implement say(). use a TTS model instead",
                            self.dialect.provider_label
                        )),
                    }));
                }
                self.pending_replies.insert(req_id.clone(), now.saturating_add(GENERATE_REPLY_TIMEOUT));
                let frame = wire::force_message(&req_id, &text);
                // Tag only after the frame is sent, so a say whose send failed cannot claim the
                // next unrelated response.
                if !self.say_tags.contains(&req_id) {
                    self.say_tags.push_back(req_id);
                }
                Step::frame(frame)
            }

            Command::Interrupt(_) => {
                if !self.has_active_generation() {
                    return Step::default();
                }
                let mut step = Step::frame(wire::response_cancel(None));
                if self.dialect.discard_silent_responses {
                    step.merge(self.discard_abandoned());
                }
                step
            }

            Command::Truncate(lk_agents_abi::llm::realtime::Truncate { message_id, has_audio, audio_end_ms, audio_transcript }) => {
                if has_audio {
                    // `audio_end_ms == 0`: interrupted before any audio played, delete the item.
                    return Step::frame(if audio_end_ms > 0 {
                        wire::conversation_item_truncate(&message_id, audio_end_ms)
                    } else {
                        wire::conversation_item_delete(
                            &self.ids.short("chat_ctx_delete_"),
                            &message_id,
                        )
                    });
                }
                let Some(transcript) = audio_transcript else { return Step::default() };
                // Text-only truncation is a chat-context edit: rewrite the item and let the diff
                // do the rest.
                let mut ctx = self.remote.to_chat_ctx();
                let Some(idx) = ctx.index_by_id(&message_id) else { return Step::default() };
                if let Some(m) = ctx.items[idx].as_message_mut() {
                    m.content = vec![lk_agents_abi::ChatContent::Text(transcript)];
                }
                let (frames, _) = self.chat_ctx_events(&ctx);
                Step { frames, ..Default::default() }
            }
        }
    }

    /// `_create_session_update_event`.
    fn session_update_frame(&self) -> Value {
        let mut p = SessionPatch::default();
        p.top.insert("model".into(), json!(self.opts.model));
        // The reference: "it'll respond in audio + transcript" -- ONE modality, not the list.
        p.top.insert(
            "output_modalities".into(),
            json!([if self.opts.audio_output { "audio" } else { "text" }]),
        );
        p.audio_in.insert("format".into(), wire::audio_format());
        // Explicit nulls (see `wire`): "off" is said, not left to the server's default.
        p.audio_in.insert(
            "noise_reduction".into(),
            self.opts.noise_reduction.clone().unwrap_or(Value::Null),
        );
        p.audio_in.insert(
            "transcription".into(),
            self.opts.transcription.clone().unwrap_or(Value::Null),
        );
        p.audio_in.insert(
            "turn_detection".into(),
            self.opts.turn_detection.as_ref().map_or(Value::Null, TurnDetection::to_json),
        );
        p.audio_out.insert("format".into(), wire::audio_format());
        p.audio_out.insert("speed".into(), json!(self.opts.speed));
        p.audio_out.insert("voice".into(), json!(self.opts.voice));
        p.top.insert("max_output_tokens".into(), self.opts.max_output_tokens.clone());
        p.top.insert("tool_choice".into(), wire::tool_choice(self.opts.tool_choice.as_ref()));
        p.top.insert("tracing".into(), self.opts.tracing.clone().unwrap_or(Value::Null));

        let mut session = p.into_session();
        if let Some(i) = &self.opts.instructions {
            session.insert("instructions".into(), json!(i));
        }
        if let Some(t) = &self.opts.truncation {
            session.insert("truncation".into(), t.clone());
        }
        if let Some(r) = &self.opts.reasoning {
            session.insert("reasoning".into(), r.clone());
        }
        self.dialect.rewrite_session(&mut session);
        wire::session_update(&self.ids.short("session_update_"), session)
    }

    /// `RealtimeSession.update_options`: what changed goes out as one `options_update_`
    /// `session.update`, the audio halves together; the derived capabilities follow.
    pub fn update_options(&mut self, p: OptionsPatch) -> Step {
        let mut patch = SessionPatch::default();
        if let Maybe::Given(tc) = p.tool_choice {
            let (current, next) =
                (wire::tool_choice(self.opts.tool_choice.as_ref()), wire::tool_choice(tc.as_ref()));
            self.opts.tool_choice = tc;
            if current != next {
                patch.top.insert("tool_choice".into(), next);
            }
        }
        if let Maybe::Given(v) = p.max_output_tokens {
            if self.opts.max_output_tokens != v {
                patch.top.insert("max_output_tokens".into(), v.clone());
            }
            self.opts.max_output_tokens = v;
        }
        for (key, given, slot) in [
            ("tracing", p.tracing, &mut self.opts.tracing),
            ("truncation", p.truncation, &mut self.opts.truncation),
            // `None` clears it server-side
            ("reasoning", p.reasoning, &mut self.opts.reasoning),
        ] {
            if let Maybe::Given(v) = given {
                if *slot != v {
                    patch.top.insert(key.into(), v.clone().unwrap_or(Value::Null));
                }
                *slot = v;
            }
        }
        if let Maybe::Given(v) = p.voice {
            if self.opts.voice != v {
                patch.audio_out.insert("voice".into(), json!(v));
            }
            self.opts.voice = v;
        }
        if let Maybe::Given(td) = p.turn_detection {
            if self.opts.turn_detection != td {
                patch.audio_in.insert(
                    "turn_detection".into(),
                    td.as_ref().map_or(Value::Null, TurnDetection::to_json),
                );
            }
            let server = TurnDetection::server_turn_taking(td.as_ref());
            self.caps.turn_detection = server;
            self.caps.auto_turn_reply_generation = server;
            self.opts.turn_detection = td;
        }
        if let Maybe::Given(v) = p.transcription {
            if self.opts.transcription != v {
                patch.audio_in.insert("transcription".into(), v.clone().unwrap_or(Value::Null));
            }
            self.caps.user_transcription = v.is_some();
            self.opts.transcription = v;
        }
        if let Maybe::Given(v) = p.noise_reduction {
            if self.opts.noise_reduction != v {
                patch.audio_in.insert("noise_reduction".into(), v.clone().unwrap_or(Value::Null));
            }
            self.opts.noise_reduction = v;
        }
        if let Maybe::Given(v) = p.speed {
            if self.opts.speed != v {
                patch.audio_out.insert("speed".into(), json!(v));
            }
            self.opts.speed = v;
        }
        if patch.top.is_empty() && patch.audio_in.is_empty() && patch.audio_out.is_empty() {
            return Step::default();
        }
        let mut session = patch.into_session();
        self.dialect.rewrite_session(&mut session);
        Step::frame(wire::session_update(&self.ids.short("options_update_"), session))
    }

    /// `_create_tools_update_event`.
    fn tools_update_frame(&self) -> Value {
        let tools: Vec<Value> = self.tools.iter().map(wire::tool_spec).collect();
        let mut session = Map::new();
        session.insert("type".into(), json!("realtime"));
        session.insert("model".into(), json!(self.opts.model));
        session.insert("tools".into(), Value::Array(tools));
        self.dialect.rewrite_session(&mut session);
        wire::session_update(&self.ids.short("tools_update_"), session)
    }

    /// `_create_update_chat_ctx_events`: the mirror diffed against the agent's context.
    fn chat_ctx_events(&mut self, chat_ctx: &ChatContext) -> (Vec<Value>, CtxUpdate) {
        let mut chat_ctx = chat_ctx.clone();

        // xAI: the held transcript is not in the agent's context yet; a plain diff would delete
        // the user's turn. Re-insert it at the mirror's position.
        if let Some(held) = &self.held
            && let Some(pos) = self.remote.items().iter().position(|i| i.id() == held.item_id)
            && chat_ctx.get_by_id(&held.item_id).is_none()
        {
            let at = self.remote.items()[..pos]
                .iter()
                .rev()
                .find_map(|prev| chat_ctx.index_by_id(prev.id()).map(|i| i + 1))
                .unwrap_or(0);
            chat_ctx.items.insert(at, self.remote.items()[pos].clone());
        }

        let remote_ctx = self.remote.to_chat_ctx();
        // Empty content can be a local placeholder or a remote item with non-text content; keep
        // the remote ones, or every committed audio turn is deleted on the next update.
        let remote_ids: HashSet<&str> = remote_ctx.items.iter().map(ChatItem::id).collect();
        let filtered = ChatContext::new(
            chat_ctx
                .items
                .iter()
                .filter(|i| {
                    i.kind() != lk_agents_abi::ItemKind::Message
                        || i.as_message().is_some_and(|m| !m.content.is_empty())
                        || remote_ids.contains(i.id())
                })
                .cloned()
                .collect(),
        );

        let diff = compute_chat_ctx_diff(&remote_ctx, &filtered);
        let mut frames = Vec::new();
        let mut update = CtxUpdate {
            req_id: String::new(),
            deadline: Ts::ZERO,
            awaiting_create: HashSet::new(),
            awaiting_delete: HashSet::new(),
            by_event_id: HashMap::new(),
            frames: Vec::new(),
            rejected: Vec::new(),
        };

        let delete = |frames: &mut Vec<Value>, update: &mut CtxUpdate, id: &str| {
            let event_id = self.ids.short("chat_ctx_delete_");
            let frame = wire::conversation_item_delete(&event_id, id);
            frames.push(frame.clone());
            update.frames.push((event_id.clone(), frame));
            update.awaiting_delete.insert(id.to_owned());
            update.by_event_id.insert(event_id, (true, id.to_owned()));
        };

        for id in &diff.to_remove {
            delete(&mut frames, &mut update, id);
        }

        let create =
            |frames: &mut Vec<Value>, update: &mut CtxUpdate, prev: &Option<String>, id: &str| {
                let Some(item) = filtered.get_by_id(id) else { return };
                let Ok(wire_item) = wire::to_wire_item(item, self.images.as_ref()) else { return };
                let event_id = self.ids.short("chat_ctx_create_");
                let frame = wire::conversation_item_create(
                    &event_id,
                    wire_item,
                    // `"root"` when there is no predecessor -- a literal, not an omitted key.
                    Some(prev.as_deref().unwrap_or("root")),
                );
                frames.push(frame.clone());
                update.frames.push((event_id.clone(), frame));
                update.awaiting_create.insert(id.to_owned());
                update.by_event_id.insert(event_id, (false, id.to_owned()));
            };

        for (prev, id) in &diff.to_create {
            create(&mut frames, &mut update, prev, id);
        }

        for (prev, id) in &diff.to_update {
            // Empty content means "not synced down"; recreating would replace a server-side audio
            // turn with an empty text message.
            let remote_is_empty = remote_ctx
                .get_by_id(id)
                .and_then(ChatItem::as_message)
                .is_some_and(|m| m.content.is_empty());
            if remote_is_empty {
                continue;
            }
            delete(&mut frames, &mut update, id);
            create(&mut frames, &mut update, prev, id);
        }

        (frames, update)
    }
    /// `_handle_conversion_item_input_audio_transcription_completed`, shared by the immediate path
    /// and xAI's deferred one.
    fn finalize_transcript(&mut self, t: TranscriptionCompleted) -> Step {
        self.take_accumulated(&t.item_id, t.content_index);
        if let Some(ChatItem::Message(m)) = self.remote.get_mut(&t.item_id) {
            m.content.push(lk_agents_abi::ChatContent::Text(t.transcript.clone()));
            m.transcript_confidence = t.confidence;
        }
        Step::event(Event::InputTranscription(InputTranscription {
            item_id: t.item_id.clone(),
            transcript: t.transcript,
            is_final: true,
            confidence: t.confidence,
            turn_started_at: self.speech_started_at.remove(&t.item_id),
        }))
    }

    /// xAI's `_flush_input_transcription`.
    fn flush_held_transcript(&mut self, _now: Ts) -> Step {
        let Some(t) = self.held.take() else { return Step::default() };
        // xAI re-sends the whole transcript on every commit and the finaliser appends: strip the
        // mirror's text first or the item holds one copy per commit.
        if let Some(ChatItem::Message(m)) = self.remote.get_mut(&t.item_id) {
            m.content.retain(|c| !matches!(c, lk_agents_abi::ChatContent::Text(_)));
        }
        self.finalize_transcript(t)
    }

    /// `_discard_abandoned_response`: the server created a response and then said nothing (xAI).
    /// Discarded on the next `response.created` so it does not hold the generation slot.
    fn discard_abandoned(&mut self) -> Step {
        if self.response_spoke || !matches!(self.generation, Generation::Active { .. }) {
            return Step::default();
        }
        let step = self.close_generation();
        self.generation = Generation::Discarded;
        step
    }

    /// `_close_current_generation`: end every open message and settle the turn, so no consumer
    /// hangs when a reconnect or close takes the turn away.
    fn close_generation(&mut self) -> Step {
        let Generation::Active { messages, order, response_id } =
            std::mem::replace(&mut self.generation, Generation::None)
        else {
            // A discarded generation has no channels or consumer.
            return Step::default();
        };
        let default_modalities = self.default_modalities();
        let mut step = Step::default();
        for id in &order {
            let Some(m) = messages.get(id) else { continue };
            if m.modalities.is_none() {
                step.events.push(Event::MessageModalities (MessageModalities {
                    message_id: id.clone(),
                    response_id: Some(response_id.clone()),
                    modalities: default_modalities,
                }));
            }
            step.events.push(Event::MessageDone(MessageDone { message_id: id.clone(), response_id: Some(response_id.clone()) }));
        }
        step.events.push(Event::GenerationDone (GenerationDone {
            response_id,
            status: GenerationStatus::Cancelled,
            usage: TokenUsage::default(),
        }));
        step
    }

    /// The response being produced (`GenerationCreatedEvent.response_id`). `None` between
    /// generations: the driver then routes by the open slot alone.
    fn response_id(&self) -> Option<String> {
        match &self.generation {
            Generation::Active { response_id, .. } => Some(response_id.clone()),
            Generation::None | Generation::Discarded => None,
        }
    }

    fn default_modalities(&self) -> Modalities {
        if self.opts.audio_output { Modalities::AudioAndText } else { Modalities::TextOnly }
    }

    fn settle_modalities(&mut self, item_id: &str, modalities: Modalities) -> Step {
        let Generation::Active { response_id, messages, .. } = &mut self.generation else {
            return Step::default();
        };
        let response_id = Some(response_id.clone());
        let Some(m) = messages.get_mut(item_id) else { return Step::default() };
        // `contextlib.suppress(asyncio.InvalidStateError)`: settling twice is a normal race; the
        // first answer wins.
        if m.modalities.is_some() {
            return Step::default();
        }
        m.modalities = Some(modalities);
        Step::event(Event::MessageModalities (MessageModalities { message_id: item_id.to_owned(), response_id, modalities }))
    }

    /// The server acked a create or a delete. Emits [`Event::ChatCtxUpdated`] once the last one
    /// lands (the `asyncio.gather` in `update_chat_ctx`).
    fn settle_ctx_item(&mut self, is_delete: bool, item_id: &str) -> Step {
        let Some(update) = &mut self.ctx_update else { return Step::default() };
        let set = if is_delete { &mut update.awaiting_delete } else { &mut update.awaiting_create };
        if !set.remove(item_id) {
            return Step::default();
        }
        update.by_event_id.retain(|_, (d, id)| !(*d == is_delete && id == item_id));
        if update.pending() {
            return Step::default();
        }
        let update = self.ctx_update.take().expect("just checked");
        Step::event(Event::ChatCtxUpdated { req_id: update.req_id, rejected: update.rejected })
    }

    fn take_accumulated(&mut self, item_id: &str, content_index: usize) -> Option<String> {
        let by_index = self.transcript_acc.get_mut(item_id)?;
        let partial = by_index.remove(&content_index);
        if by_index.is_empty() {
            self.transcript_acc.remove(item_id);
        }
        partial
    }

    fn error(&self, now: Ts, message: impl Into<String>, recoverable: bool) -> RealtimeModelError {
        RealtimeModelError {
            timestamp: now,
            label: self.label.clone(),
            message: message.into(),
            recoverable,
            req_id: None,
            code: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Step {
    pub frames: Vec<Value>,
    pub events: Vec<Event>,
    /// Recycle the socket; the driver reconnects and calls [`RealtimeMachine::reconnect`].
    pub recycle: bool,
    /// A terminal error: the driver must stop reconnecting.
    pub fatal: Option<ApiError>,
}

impl Step {
    fn frame(f: Value) -> Step {
        Step { frames: vec![f], ..Default::default() }
    }
    fn event(e: Event) -> Step {
        Step { events: vec![e], ..Default::default() }
    }
    fn merge(&mut self, other: Step) {
        self.frames.extend(other.frames);
        self.events.extend(other.events);
        self.recycle |= other.recycle;
        self.fatal = self.fatal.take().or(other.fatal);
    }
}

#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub model: String,
    pub voice: String,
    pub instructions: Option<String>,
    pub tool_choice: Option<ToolChoice>,
    pub turn_detection: Option<TurnDetection>,
    /// The `AudioTranscription` object as given. `None` is off (`null` on the wire), which also
    /// makes `capabilities.user_transcription` false.
    pub transcription: Option<Value>,
    /// The `NoiseReduction` object; `None` is off.
    pub noise_reduction: Option<Value>,
    pub audio_output: bool,
    pub speed: f64,
    /// `max_response_output_tokens`: `"inf"`, a count, or `null`.
    pub max_output_tokens: Value,
    /// Always on the wire, `null` when `None`.
    pub tracing: Option<Value>,
    /// On the wire only when set.
    pub truncation: Option<Value>,
    /// On the wire only when set.
    pub reasoning: Option<Value>,
    pub max_session_duration: Option<Duration>,
}

impl SessionOptions {
    /// The dialect's defaults, with `turn_detection_disabled` applied when the dialect honours it.
    pub fn from_dialect(d: &Dialect, turn_detection_disabled: bool) -> SessionOptions {
        let disable = turn_detection_disabled && d.supports_turn_detection_disabled;
        SessionOptions {
            model: d.default_model.to_owned(),
            voice: d.default_voice.to_owned(),
            instructions: None,
            tool_choice: None,
            turn_detection: if disable { None } else { d.default_turn_detection.clone() },
            transcription: d.default_transcription_model.map(|m| json!({"model": m})),
            noise_reduction: None,
            // `"audio" in modalities`, and the dialect may not let the caller choose.
            audio_output: d.forced_modalities.is_none_or(|m| m.contains(&"audio")),
            speed: 1.0,
            max_output_tokens: json!(DEFAULT_MAX_OUTPUT_TOKENS),
            tracing: None,
            truncation: None,
            reasoning: None,
            max_session_duration: d.default_max_session_duration,
        }
    }

    /// The model's half of `update_options`: the values are taken, the sessions are told.
    pub fn apply(&mut self, p: &OptionsPatch) {
        if let Maybe::Given(v) = &p.tool_choice {
            self.tool_choice = v.clone();
        }
        if let Maybe::Given(v) = &p.voice {
            self.voice = v.clone();
        }
        if let Maybe::Given(v) = &p.turn_detection {
            self.turn_detection = v.clone();
        }
        if let Maybe::Given(v) = &p.max_output_tokens {
            self.max_output_tokens = v.clone();
        }
        if let Maybe::Given(v) = &p.transcription {
            self.transcription = v.clone();
        }
        if let Maybe::Given(v) = &p.noise_reduction {
            self.noise_reduction = v.clone();
        }
        if let Maybe::Given(v) = p.speed {
            self.speed = v;
        }
        if let Maybe::Given(v) = &p.tracing {
            self.tracing = v.clone();
        }
        if let Maybe::Given(v) = &p.truncation {
            self.truncation = v.clone();
        }
        if let Maybe::Given(v) = &p.reasoning {
            self.reasoning = v.clone();
        }
    }
}

/// `update_options(..)`'s keywords; each `Given` is compared against the session's own value and
/// only a change goes on the wire.
#[derive(Clone, Default)]
pub struct OptionsPatch {
    pub tool_choice: Maybe<Option<ToolChoice>>,
    pub voice: Maybe<String>,
    pub turn_detection: Maybe<Option<TurnDetection>>,
    pub max_output_tokens: Maybe<Value>,
    pub transcription: Maybe<Option<Value>>,
    pub noise_reduction: Maybe<Option<Value>>,
    pub speed: Maybe<f64>,
    pub tracing: Maybe<Option<Value>>,
    pub truncation: Maybe<Option<Value>>,
    pub reasoning: Maybe<Option<Value>>,
}

#[derive(Debug, Default)]
struct MessageState {
    modalities: Option<Modalities>,
    /// Accumulated so `response.done` can append it to the mirror's copy of the item.
    transcript: String,
    open: bool,
}

#[derive(Debug)]
enum Generation {
    None,
    Active {
        response_id: String,
        messages: HashMap<String, MessageState>,
        /// Insertion order, so the flattened event stream is deterministic on close.
        order: Vec<String>,
    },
    /// `_DiscardedGeneration`: cancelled before it surfaced; its trailing events are skipped.
    Discarded,
}

#[derive(Debug)]
struct CtxUpdate {
    req_id: String,
    deadline: Ts,
    /// `_item_create_future` / `_item_delete_future`, keyed by item id.
    awaiting_create: HashSet<String>,
    awaiting_delete: HashSet<String>,
    /// `_chat_ctx_event_futures`, keyed by client `event_id`: an update sends a delete and a
    /// create under the same item id, so only the event id says which one a rejection answers.
    by_event_id: HashMap<String, (bool, String)>,
    /// Every frame of the update, by `event_id`: a reconnection sends the unconfirmed ones again.
    frames: Vec<(String, Value)>,
    rejected: Vec<String>,
}

impl CtxUpdate {
    fn pending(&self) -> bool {
        !self.awaiting_create.is_empty() || !self.awaiting_delete.is_empty()
    }
}

/// A session-clock stamp in `time.time()` seconds: the native session runs on the wall clock.
fn secs(t: Ts) -> f64 {
    t.as_nanos() as f64 / 1e9
}
