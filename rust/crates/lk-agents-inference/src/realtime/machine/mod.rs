//! `RealtimeSession`'s body as a synchronous state machine. A caller `req_id` goes out as
//! `metadata.client_event_id`; a rejection, timeout or cancel must leave a late `response.created` recognisable as stale.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, SystemTime};

pub use lk_agents_abi::RealtimeCommand as Command;
use lk_agents_abi::ToolSpec;
use lk_agents_abi::{ChatContext, ChatItem};
use lk_agents_abi::{Maybe, Ts};

use crate::realtime::dialect::{Dialect, TurnDetection};
use crate::realtime::protocol::{self as wire, TranscriptionCompleted};
use crate::realtime::remote::RemoteChatContext;
use lk_agents_abi::llm::realtime::{
    Event, GenerateReplyFailed, GenerationDone, GenerationStatus, InputTranscription, MessageDone,
    MessageModalities, Modalities, RealtimeCapabilities, RealtimeError, RealtimeModelError,
    TokenUsage,
};
use std::cell::Cell;
use std::sync::Arc;

mod context;
use context::{CtxUpdate, DeferredCommand, Retired, RetirementQueue};
mod effects;
mod incoming;
pub use context::ContextRetirement;
pub use effects::Outgoing;
mod options;
pub use options::{OptionsPatch, SessionOptions};
#[cfg(test)]
mod contract_tests;

/// How many generated messages are remembered for truncation.
pub(crate) const SPOKEN_REMEMBERED: usize = 32;
/// How many calls a context update deleted are remembered.
const DELETED_CALLS_REMEMBERED: usize = 64;
/// Requests whose creation can be awaiting a provider response at once.
const PENDING_RESPONSES: usize = 32;
/// Completed timeout/cancel identities retained until the socket is recycled.
const DISCARDED_RESPONSES: usize = 64;
/// `_schedule_stale_say_cleanup`'s `call_later(10.0, ...)`.
pub const STALE_SAY_CLEANUP: Duration = Duration::from_secs(10);

pub struct RealtimeMachine {
    dialect: Dialect,
    opts: SessionOptions,
    caps: RealtimeCapabilities,
    ids: EventIds,
    connection_generation: u64,
    response_timeout: Option<Duration>,
    reply_deadlines: BTreeMap<String, Ts>,
    label: String,
    /// `Metadata(model_name=model, model_provider=provider)` on the metrics this session reports.
    provider: String,
    /// The response in flight: when it was created, and when its first output arrived.
    response_created_at: Option<Ts>,
    first_token_at: Option<Ts>,

    remote: RemoteChatContext,
    desired: RemoteChatContext,
    desired_epoch: u64,
    tools: Arc<Vec<ToolSpec>>,

    generation: Generation,
    /// The messages of the generation the last `response.created` closed: the cancelled
    /// response's tail still streams after it (`corpus/probes/probe-cancel_then_create.json`),
    /// and is not the next generation's to play.
    closed_messages: BTreeSet<String>,
    /// The calls a context update deleted from the conversation: an output for one names a call
    /// the server no longer holds, and is refused (`invalid_tool_call_id`).
    /// The last few: an output answers a call the model just made.
    deleted_calls: VecDeque<String>,
    /// The last messages this connection's server generated: only those hold audio to cut (one
    /// a replay recreated holds its text).
    spoken: VecDeque<String>,
    /// `_response_created_futures`: the requests whose response is not created yet. How long one
    /// may wait is the driver's to say (it times out with `CancelReply`).
    pending_replies: BTreeSet<String>,
    /// `_discarded_event_ids`: requests abandoned before the server created their response. The
    /// response is cancelled by id and its trailing events skipped when it finally arrives.
    discarded: BTreeSet<String>,
    settled_discards: BTreeSet<String>,
    /// Discarded responses created while another generated: their `response.done` is skipped.
    discarded_responses: BTreeSet<String>,
    /// xAI: `_pending_say_event_ids`. `force_message` gets a `response.created` with no
    /// `client_event_id`, so the oldest un-answered say claims it.
    say_tags: VecDeque<String>,
    /// xAI: `_schedule_stale_say_cleanup`. Drops a tag the server never answered, so it cannot
    /// steal a later unrelated reply.
    stale_say: BTreeMap<String, Ts>,

    ctx_update: Option<CtxUpdate>,
    retired: RetirementQueue,
    discarded_context_items: Vec<ChatItem>,
    pending_context: Option<ChatContext>,

    // Per-connection input turn state, discarded on reconnect.
    transcript_acc: BTreeMap<String, BTreeMap<usize, String>>,
    /// Per `item_id`: a late transcript would otherwise consume the *next* turn's start time.
    speech_started_at: BTreeMap<String, Ts>,
    /// xAI: the transcript being held.
    held: Option<TranscriptionCompleted>,
    /// xAI: whether the response in flight has produced any output at all.
    response_spoke: bool,
    /// The server's VAD heard the person start and not yet stop.
    user_speaking: bool,

    audio_remainder: Duration,
    pushed: Duration,
    /// The `response.create` of every pending reply, sent again by a reconnection.
    reply_frames: BTreeMap<String, Outgoing>,
    deferred_replies: BTreeSet<String>,
    deferred_commands: VecDeque<DeferredCommand>,
    deferred_media_count: usize,
    deferred_media_bytes: usize,
    say_frames: BTreeMap<String, Outgoing>,

    session_deadline: Option<Ts>,
    closing: bool,
    /// The wall clock at `Ts(0)` of the session's clock: what `time.time()` stamps are read from.
    wall_anchor: SystemTime,
}

impl RealtimeMachine {
    pub fn new(
        dialect: Dialect,
        opts: SessionOptions,
        caps: RealtimeCapabilities,
        id_prefix: String,
    ) -> Self {
        let label = format!("lk_agents_realtime.{}", dialect.provider_label);
        RealtimeMachine {
            dialect,
            opts,
            caps,
            ids: EventIds { prefix: id_prefix, next: Cell::new(0) },
            connection_generation: 0,
            response_timeout: Some(Duration::from_secs(5)),
            reply_deadlines: BTreeMap::new(),
            label,
            provider: String::new(),
            response_created_at: None,
            first_token_at: None,
            remote: RemoteChatContext::new(),
            desired: RemoteChatContext::new(),
            desired_epoch: 0,
            tools: Arc::new(Vec::new()),
            generation: Generation::None,
            closed_messages: BTreeSet::new(),
            deleted_calls: VecDeque::new(),
            spoken: VecDeque::new(),
            pending_replies: BTreeSet::new(),
            discarded: BTreeSet::new(),
            settled_discards: BTreeSet::new(),
            discarded_responses: BTreeSet::new(),
            say_tags: VecDeque::new(),
            stale_say: BTreeMap::new(),
            ctx_update: None,
            retired: RetirementQueue::default(),
            discarded_context_items: Vec::new(),
            pending_context: None,
            transcript_acc: BTreeMap::new(),
            speech_started_at: BTreeMap::new(),
            held: None,
            response_spoke: false,
            user_speaking: false,
            audio_remainder: Duration::ZERO,
            pushed: Duration::ZERO,
            reply_frames: BTreeMap::new(),
            deferred_replies: BTreeSet::new(),
            deferred_commands: VecDeque::new(),
            deferred_media_count: 0,
            deferred_media_bytes: 0,
            say_frames: BTreeMap::new(),
            session_deadline: None,
            closing: false,
            wall_anchor: SystemTime::UNIX_EPOCH,
        }
    }

    /// The session clock's wall time at `Ts(0)`, so its stamps are `time.time()`'s.
    pub fn with_wall_anchor(mut self, wall_anchor: SystemTime) -> Self {
        self.wall_anchor = wall_anchor;
        self
    }

    /// The wall time a session instant fell at.
    pub(crate) fn wall_at(&self, t: Ts) -> SystemTime {
        self.wall_anchor + Duration::from_nanos(t.as_nanos())
    }

    /// `time.time()` at a session instant.
    pub(crate) fn wall_secs(&self, t: Ts) -> f64 {
        self.wall_at(t).duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
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
            timestamp: self.wall_secs(now),
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

    /// `_emit_transcription_metrics`: the input transcription's usage as `STTMetrics`, named by the
    /// transcription model.
    fn transcription_metrics(&self, now: Ts, t: &TranscriptionCompleted) -> Option<Event> {
        use lk_agents_abi::metrics::base::SttMetrics;
        let usage = t.usage.as_ref()?;
        let model = self
            .opts
            .transcription
            .as_ref()
            .and_then(|v| v.get("model"))
            .and_then(serde_json::Value::as_str);
        let (audio_duration, input_tokens, output_tokens, total_tokens, input_audio_tokens) =
            match *usage {
                wire::TranscriptionUsage::Tokens { input, output, total, input_audio } => {
                    (0.0, input, output, total, input_audio)
                }
                wire::TranscriptionUsage::Duration { seconds } => (seconds, 0, 0, 0, 0),
            };
        let metrics = SttMetrics {
            label: self.label.clone(),
            request_id: t.event_id.clone(),
            timestamp: self.wall_secs(now),
            duration: 0.0,
            audio_duration,
            streamed: true,
            input_tokens,
            output_tokens,
            total_tokens,
            input_audio_tokens,
            acquire_time: 0.0,
            connection_reused: false,
            metadata: Some(lk_agents_abi::metrics::base::Metadata {
                model_name: model.map(str::to_owned),
                model_provider: Some(self.provider.clone()),
            }),
        };
        Some(Event::Metrics(lk_agents_abi::llm::realtime::RealtimeMetrics {
            metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::Stt(metrics)),
        }))
    }

    /// The finished response's `RealtimeModelMetrics`, as `_handle_response_done` computes them.
    fn response_metrics(
        &mut self,
        now: Ts,
        response_id: &str,
        status: GenerationStatus,
        usage: &TokenUsage,
    ) -> Option<Event> {
        use lk_agents_abi::metrics::base::{
            CachedTokenDetails, InputTokenDetails, OutputTokenDetails, RealtimeModelMetrics,
        };
        let created = self.response_created_at.take()?;
        let first = self.first_token_at.take();
        let duration = secs(now) - secs(created);
        let tokens = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        let metrics = RealtimeModelMetrics {
            label: self.label.clone(),
            request_id: response_id.to_owned(),
            timestamp: self.wall_secs(created),
            duration,
            session_duration: 0.0,
            ttft: first.map_or(RealtimeModelMetrics::NO_TTFT, |f| secs(f) - secs(created)),
            cancelled: status == GenerationStatus::Cancelled,
            input_tokens: tokens(usage.input_tokens),
            output_tokens: tokens(usage.output_tokens),
            total_tokens: tokens(usage.total_tokens),
            tokens_per_second: if duration > 0.0 {
                usage.output_tokens as f64 / duration
            } else {
                0.0
            },
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

    pub fn connection_generation(&self) -> u64 {
        self.connection_generation
    }

    pub fn with_response_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.response_timeout = timeout;
        self
    }

    pub fn incoming_on(&mut self, generation: u64, now: Ts, msg: wire::Incoming) -> Step {
        if generation != self.connection_generation {
            return Step::default();
        }
        let mut step = self.tick(now);
        if step.recycle {
            return step;
        }
        step.merge(self.incoming(now, msg));
        self.enforce_retirement_capacity(now, &mut step);
        step
    }

    /// The runtime acknowledges a flushed response command before delivering its responses.
    pub fn response_sent(&mut self, generation: u64, now: Ts, req_id: &str, is_say: bool) {
        if self.closing || generation != self.connection_generation {
            return;
        }
        if self.pending_replies.contains(req_id) {
            if let Some(timeout) = self.response_timeout {
                self.reply_deadlines
                    .entry(req_id.to_owned())
                    .or_insert(now.saturating_add(timeout));
            }
        }
        if is_say && (self.pending_replies.contains(req_id) || self.discarded.contains(req_id)) {
            if !self.say_tags.iter().any(|id| id == req_id) {
                self.say_tags.push_back(req_id.to_owned());
            }
            if self.discarded.contains(req_id) {
                self.stale_say.insert(req_id.to_owned(), now.saturating_add(STALE_SAY_CLEANUP));
            }
        }
    }

    pub fn open(&mut self, now: Ts) -> Step {
        self.connection_generation += 1;
        self.session_deadline = self.opts.max_session_duration.map(|d| now.saturating_add(d));
        Step::frame(self.session_update_frame())
    }

    /// `_reconnect()`. Reset turn state first (it settles into the mirror), then empty the mirror
    /// so the diff produces a create for every item.
    pub fn reconnect(&mut self, now: Ts) -> Step {
        self.connection_generation += 1;
        let mut step = Step::default();
        self.session_deadline = self.opts.max_session_duration.map(|d| now.saturating_add(d));
        step.frames.push(self.session_update_frame());
        if !self.tools.is_empty() {
            step.frames.push(self.tools_update_frame());
        }

        step.merge(self.reset_input_turn_state());
        // The new connection's VAD heard no onset: the person's speech the old one opened is
        // over as far as the session can tell, or nothing would ever end it.
        if std::mem::take(&mut self.user_speaking) {
            step.events.push(Event::InputSpeechStopped(
                lk_agents_abi::llm::realtime::InputSpeechStopped {
                    user_transcription_enabled: self.opts.transcription.is_some(),
                },
            ));
        }
        // The new connection's buffer holds none of the audio sent to the old one: a commit now
        // would name nothing and be refused (`input_audio_buffer_commit_empty`).
        self.pushed = Duration::ZERO;
        self.audio_remainder = Duration::ZERO;

        // The conversation the new connection gets: the one the server is being brought to, a
        // context update the lost connection had not confirmed included. Resending that update's
        // frames after the replay re-created what the replay had just restored (an item it
        // deletes and creates again under the same id), and the server refused it.
        self.reply_deadlines.clear();
        self.replay_context();

        // A cancelled request's response will never come now.
        let mut discarded: Vec<String> = std::mem::take(&mut self.discarded).into_iter().collect();
        discarded.sort();
        for req_id in discarded {
            if self.settled_discards.remove(&req_id) {
                continue;
            }
            let error = RealtimeError::new("generate_reply cancelled");
            step.events.push(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }));
        }
        self.say_tags.clear();
        self.discarded_responses.clear();
        self.spoken.clear();
        self.stale_say.clear();
        step.merge(self.close_generation());
        // A reply asked for and not yet created is asked for again, now that the conversation it
        // answers is back; a `say()` (no `response.create` to resend) is failed.
        let mut pending: Vec<String> = self.pending_replies.iter().cloned().collect();
        pending.sort();
        for req_id in pending.into_iter().rev() {
            match self.reply_frames.get(&req_id) {
                Some(frame) => {
                    if self.context_pages_pending() {
                        self.defer_reply(req_id, true);
                    } else {
                        step.frames.push(frame.clone());
                    }
                }
                None => {
                    self.pending_replies.remove(&req_id);
                    self.reply_deadlines.remove(&req_id);
                    step.events.push(Event::GenerateReplyFailed(GenerateReplyFailed {
                        req_id,
                        error: RealtimeError::new(
                            "pending response discarded due to session reconnection",
                        ),
                    }));
                }
            }
        }
        step.merge(self.continue_context(now));
        step.events
            .push(Event::SessionReconnected(lk_agents_abi::llm::realtime::SessionReconnected {}));
        step
    }

    pub fn close(&mut self, now: Ts) -> Step {
        self.closing = true;
        let mut step = self.flush_held_transcript(now);
        step.merge(self.close_generation());
        for req_id in std::mem::take(&mut self.pending_replies) {
            step.events.push(Event::GenerateReplyFailed(GenerateReplyFailed {
                req_id,
                error: RealtimeError::new("realtime session closed"),
            }));
        }
        if let Some(update) = self.ctx_update.take() {
            self.retired.push_back(Retired::Update(update));
        }
        if let Some(target) = self.pending_context.take() {
            self.retired.push_back(Retired::Context(target));
        }
        if self.desired.revision() != 0 {
            let desired = self.take_desired();
            self.retired.push_back(Retired::Remote(desired));
        }
        if self.remote.revision() != 0 {
            self.retired.push_back(Retired::Remote(std::mem::take(&mut self.remote)));
        }
        self.reply_deadlines.clear();
        self.reply_frames.clear();
        self.deferred_replies.clear();
        if !self.deferred_commands.is_empty() {
            self.retired.push_back(Retired::Deferred(std::mem::take(&mut self.deferred_commands)));
        }
        self.deferred_media_bytes = 0;
        self.deferred_media_count = 0;
        self.say_frames.clear();
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
        self.stale_say
            .values()
            .chain(
                self.session_deadline
                    .iter()
                    .filter(|_| !matches!(self.generation, Generation::Active { .. })),
            )
            .chain(self.reply_deadlines.values())
            .copied()
            .min()
    }

    pub fn tick(&mut self, now: Ts) -> Step {
        if self.closing {
            return Step::default();
        }
        let mut step = Step::default();
        let expired: Vec<String> = self
            .reply_deadlines
            .iter()
            .filter(|(_, at)| **at <= now)
            .map(|(id, _)| id.clone())
            .collect();
        for req_id in expired {
            self.reply_deadlines.remove(&req_id);
            if self.pending_replies.remove(&req_id) {
                self.reply_frames.remove(&req_id);
                self.say_frames.remove(&req_id);
                self.forget_deferred_reply(&req_id);
                self.discarded.insert(req_id.clone());
                self.settled_discards.insert(req_id.clone());
                if self.say_tags.contains(&req_id) {
                    self.stale_say.insert(req_id.clone(), now.saturating_add(STALE_SAY_CLEANUP));
                }
                if self.discarded.len() >= DISCARDED_RESPONSES {
                    step.recycle = true;
                }
                step.events.push(Event::GenerateReplyFailed(GenerateReplyFailed {
                    req_id,
                    error: RealtimeError::new("response creation timed out"),
                }));
            }
        }

        let stale: Vec<String> =
            self.stale_say.iter().filter(|&(_, &at)| at <= now).map(|(k, _)| k.clone()).collect();
        for req_id in stale {
            self.stale_say.remove(&req_id);
            // Untagged late Say responses cannot be distinguished from spontaneous responses.
            // Quarantine its identity until the old socket is replaced.
            if self.discarded.contains(&req_id) {
                step.recycle = true;
            }
        }

        if self.session_deadline.is_some_and(|at| at <= now) {
            // Recycle between turns, never mid-sentence: the deadline stays armed and re-checks
            // on the next tick (the reference awaits `_done_fut`).
            if matches!(self.generation, Generation::Active { .. }) {
                // Suspend this wake until the active response finishes.
            } else {
                self.session_deadline = None;
                step.recycle = true;
            }
        }

        step
    }

    pub fn command(&mut self, now: Ts, cmd: Command) -> Step {
        if !self.closing
            && self.context_pages_pending()
            && matches!(
                &cmd,
                Command::PushAudio(_)
                    | Command::CommitAudio(_)
                    | Command::ClearAudio(_)
                    | Command::Truncate(_)
            )
        {
            return self.queue_context_media(now, cmd);
        }
        let mut step = self.apply_command(now, cmd, context::CONTEXT_PAGE);
        self.enforce_retirement_capacity(now, &mut step);
        step
    }

    fn apply_command(&mut self, now: Ts, cmd: Command, context_budget: usize) -> Step {
        if self.closing {
            return match cmd {
                Command::GenerateReply(lk_agents_abi::llm::realtime::GenerateReply {
                    req_id,
                    ..
                })
                | Command::Say(lk_agents_abi::llm::realtime::Say { req_id, .. }) => {
                    Step::event(Event::GenerateReplyFailed(GenerateReplyFailed {
                        req_id,
                        error: RealtimeError::new("realtime session closed"),
                    }))
                }
                _ => Step::default(),
            };
        }
        match cmd {
            Command::UpdateInstructions(lk_agents_abi::llm::realtime::UpdateInstructions {
                instructions,
            }) => {
                self.opts.instructions = Some(instructions.clone());
                Step::frame(Outgoing::InstructionsUpdate {
                    event_id: self.ids.short("instructions_update_"),
                    instructions,
                    dialect: self.dialect.clone(),
                })
            }

            Command::UpdateTools(lk_agents_abi::llm::realtime::UpdateTools { tools }) => {
                self.tools = Arc::new(tools);
                Step::frame(self.tools_update_frame())
            }

            Command::UpdateChatCtx(lk_agents_abi::llm::realtime::UpdateChatCtx { chat_ctx }) => {
                self.update_ctx(now, chat_ctx)
            }

            Command::PushAudio(lk_agents_abi::llm::realtime::PushAudio { frame }) => {
                if frame.sample_rate == 0 {
                    return Step::event(Event::Error(self.error(
                        now,
                        "invalid audio sample rate",
                        false,
                    )));
                }
                self.audio_remainder += Duration::from_secs_f64(
                    frame.samples_per_channel as f64 / frame.sample_rate as f64,
                );
                let chunks =
                    self.audio_remainder.as_nanos() / Duration::from_millis(100).as_nanos();
                let complete = Duration::from_millis((chunks as u64).saturating_mul(100));
                self.audio_remainder = self.audio_remainder.saturating_sub(complete);
                self.pushed += complete;
                Step::frame(Outgoing::AudioAppend { frame })
            }

            // The output of a call the server does not hold (a context update took it out, or it
            // was made on a session this one replaced) answers nothing: not sent, as the server
            // would refuse it (`invalid_tool_call_id`). The next context update carries the pair.
            Command::PushItem(lk_agents_abi::llm::realtime::PushItem { item, .. }) if matches!(&*item, ChatItem::FunctionCallOutput(o) if !self.holds_call(&o.call_id)) => {
                Step::default()
            }
            Command::PushItem(lk_agents_abi::llm::realtime::PushItem { item, .. }) => {
                if self.context_pages_pending() {
                    self.queue_context_item(now, *item)
                } else {
                    self.submit_context_item(*item)
                }
            }

            Command::CommitAudio(_) => {
                // "OpenAI requires at least 100ms of audio"; less gets
                // `input_audio_buffer_commit_empty` back.
                if self.pushed > Duration::from_millis(100) {
                    self.pushed = Duration::ZERO;
                    Step::frame(Outgoing::AudioCommit)
                } else {
                    Step::default()
                }
            }

            Command::ClearAudio(_) => {
                self.pushed = Duration::ZERO;
                self.audio_remainder = Duration::ZERO;
                Step::frame(Outgoing::AudioClear)
            }

            Command::GenerateReply(lk_agents_abi::llm::realtime::GenerateReply {
                req_id,
                instructions,
                tool_choice,
            }) => {
                if let Some(step) = self.admit_reply(&req_id) {
                    return step;
                }
                self.pending_replies.insert(req_id.clone());
                // OpenAI replaces session instructions with per-response ones, so concatenate;
                // sending the extra alone drops the system prompt for that turn.
                let merged = match (&self.opts.instructions, instructions) {
                    (Some(base), Some(extra)) => Some(format!("{base}\n{extra}")),
                    (_, other) => other,
                };
                let frame = Outgoing::ResponseCreate {
                    req_id: req_id.clone(),
                    instructions: merged,
                    tool_choice: tool_choice.filter(|_| self.caps.per_response_tool_choice),
                };
                self.reply_frames.retain(|id, _| self.pending_replies.contains(id));
                self.reply_frames.insert(req_id.clone(), frame.clone());
                if self.context_pages_pending() {
                    self.defer_reply(req_id, false);
                    Step::default()
                } else {
                    Step::frame(frame)
                }
            }

            Command::CancelReply(lk_agents_abi::llm::realtime::CancelReply { req_id }) => {
                if !self.pending_replies.remove(&req_id) {
                    return Step::default();
                }
                // Nothing is cancelled now: an untargeted cancel hits whatever is in flight, a
                // reply the server opened itself among them. The id discards the response when it
                // is created, and cancels it by id; the caller hears of the end then, the server
                // being busy with it until it is created.
                self.reply_deadlines.remove(&req_id);
                self.reply_frames.remove(&req_id);
                self.say_frames.remove(&req_id);
                self.forget_deferred_reply(&req_id);
                self.discarded.insert(req_id.clone());
                self.say_frames.remove(&req_id);
                if self.say_tags.contains(&req_id) {
                    self.stale_say.insert(req_id.clone(), now.saturating_add(STALE_SAY_CLEANUP));
                }
                Step { recycle: self.discarded.len() >= DISCARDED_RESPONSES, ..Default::default() }
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
                if let Some(step) = self.admit_reply(&req_id) {
                    return step;
                }
                self.pending_replies.insert(req_id.clone());
                let frame = Outgoing::ForceMessage { req_id: req_id.clone(), text };
                self.say_frames.insert(req_id.clone(), frame.clone());
                if self.context_pages_pending() {
                    self.defer_reply(req_id, false);
                    Step::default()
                } else {
                    Step::frame(frame)
                }
            }

            Command::Interrupt(lk_agents_abi::llm::realtime::Interrupt { response_id }) => {
                if !self.has_active_generation() {
                    return Step::default();
                }
                // Named, only the response it names, which may be one this session has moved past
                // (a reply the server opened while another was being created).
                let mut step = Step::frame(Outgoing::ResponseCancel { response_id });
                if self.dialect.discard_silent_responses {
                    step.merge(self.discard_abandoned());
                }
                step
            }

            Command::Truncate(lk_agents_abi::llm::realtime::Truncate {
                message_id,
                has_audio,
                audio_end_ms,
                audio_transcript,
            }) => {
                // A message a context update took out of the conversation has nothing left to
                // cut: the server would refuse it (`item_truncate_invalid_item_id`).
                if self.desired.get(&message_id).is_none() {
                    return Step::default();
                }
                if has_audio && audio_end_ms > 0 {
                    // A message the server holds as text (a replay recreated it) has no audio.
                    if !self.spoken.contains(&message_id) {
                        return Step::default();
                    }
                    return Step::frame(Outgoing::ItemTruncate {
                        item_id: message_id,
                        audio_end_ms,
                    });
                }
                if !has_audio && audio_transcript.is_none() {
                    return Step::default();
                }
                self.truncate_ctx(
                    now,
                    lk_agents_abi::llm::realtime::Truncate {
                        message_id,
                        has_audio,
                        audio_end_ms,
                        audio_transcript,
                    },
                    context_budget,
                )
            }
        }
    }

    fn admit_reply(&self, req_id: &str) -> Option<Step> {
        if self.pending_replies.contains(req_id) || self.discarded.contains(req_id) {
            return Some(Step::default());
        }
        (self.pending_replies.len() >= PENDING_RESPONSES
            || self.discarded.len() >= DISCARDED_RESPONSES)
            .then(|| {
                Step::event(Event::GenerateReplyFailed(GenerateReplyFailed {
                    req_id: req_id.to_owned(),
                    error: RealtimeError::new("realtime pending response capacity exceeded"),
                }))
            })
    }

    fn session_update_frame(&self) -> Outgoing {
        Outgoing::SessionUpdate {
            event_id: self.ids.short("session_update_"),
            opts: self.opts.clone(),
            dialect: self.dialect.clone(),
        }
    }

    pub fn update_options(&mut self, mut patch: OptionsPatch) -> Step {
        macro_rules! changed {
            ($($field:ident),*) => {$(if let Maybe::Given(ref value) = patch.$field {
                if self.opts.$field == *value { patch.$field = Maybe::NotGiven; }
            })*};
        }
        changed!(
            tool_choice,
            voice,
            turn_detection,
            max_output_tokens,
            transcription,
            noise_reduction,
            speed,
            tracing,
            truncation,
            reasoning
        );
        if patch.is_empty() {
            return Step::default();
        }
        self.opts.apply(&patch);
        let server = TurnDetection::server_turn_taking(self.opts.turn_detection.as_ref());
        self.caps.turn_detection = server;
        self.caps.auto_turn_reply_generation = server;
        self.caps.user_transcription = self.opts.transcription.is_some();
        Step::frame(Outgoing::OptionsUpdate {
            event_id: self.ids.short("options_update_"),
            patch,
            dialect: self.dialect.clone(),
        })
    }

    fn tools_update_frame(&self) -> Outgoing {
        Outgoing::ToolsUpdate {
            event_id: self.ids.short("tools_update_"),
            tools: self.tools.clone(),
            model: self.opts.model.clone(),
            dialect: self.dialect.clone(),
        }
    }

    /// `_handle_conversion_item_input_audio_transcription_completed`, shared by the immediate path
    /// and xAI's deferred one.
    fn finalize_transcript(&mut self, t: TranscriptionCompleted) -> Step {
        self.take_accumulated(&t.item_id, t.content_index);
        if let Some(ChatItem::Message(m)) = self.remote.get_mut(&t.item_id) {
            m.content.push(lk_agents_abi::ChatContent::Text(t.transcript.clone()));
            m.transcript_confidence = t.confidence;
        }
        if !self.ctx_update.as_ref().is_some_and(|update| update.touched.contains(&t.item_id))
            && let Some(ChatItem::Message(message)) = self.desired.get_mut(&t.item_id)
        {
            message.content.push(lk_agents_abi::ChatContent::Text(t.transcript.clone()));
            message.transcript_confidence = t.confidence;
        }
        Step::event(Event::InputTranscription(InputTranscription {
            item_id: t.item_id.clone(),
            transcript: t.transcript,
            is_final: true,
            confidence: t.confidence,
            // Wall seconds: this session's clock is the model's, not the driver's.
            turn_started_at: self.speech_started_at.remove(&t.item_id).map(|at| self.wall_secs(at)),
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
        if let Some(ChatItem::Message(message)) = self.desired.get_mut(&t.item_id) {
            message
                .content
                .retain(|content| !matches!(content, lk_agents_abi::ChatContent::Text(_)));
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
                step.events.push(Event::MessageModalities(MessageModalities {
                    message_id: id.clone(),
                    response_id: Some(response_id.clone()),
                    modalities: default_modalities,
                }));
            }
            step.events.push(Event::MessageDone(MessageDone {
                message_id: id.clone(),
                response_id: Some(response_id.clone()),
            }));
        }
        step.events.push(Event::GenerationDone(GenerationDone {
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
        Step::event(Event::MessageModalities(MessageModalities {
            message_id: item_id.to_owned(),
            response_id,
            modalities,
        }))
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
            provider_error: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Step {
    pub frames: Vec<Outgoing>,
    pub events: Vec<Event>,
    /// Recycle the socket; the driver reconnects and calls [`RealtimeMachine::reconnect`].
    pub recycle: bool,
    /// A terminal error, reported once: the driver must stop reconnecting.
    pub fatal: Option<RealtimeModelError>,
}

impl Step {
    fn frame(f: Outgoing) -> Step {
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
        messages: BTreeMap<String, MessageState>,
        /// Insertion order, so the flattened event stream is deterministic on close.
        order: Vec<String>,
    },
    /// `_DiscardedGeneration`: cancelled before it surfaced; its trailing events are skipped.
    Discarded,
}

/// A session-clock instant in seconds, for durations between two of them.
fn secs(t: Ts) -> f64 {
    t.as_nanos() as f64 / 1e9
}

struct EventIds {
    prefix: String,
    next: Cell<u64>,
}
impl EventIds {
    fn short(&self, kind: &str) -> String {
        let next = self.next.get() + 1;
        self.next.set(next);
        format!("{kind}{}{next}", self.prefix)
    }
}
