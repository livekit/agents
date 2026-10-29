//! `voice/generation.py`: the pure planning of a step (request, turn metrics, retries), what a
//! reply takes and reports ([`ReplyContext`], [`GenEvent`]), and the pipeline, one file per leg.

pub mod forward;
pub mod llm;
pub mod pipeline;
pub mod playout;
pub mod realtime;
pub mod tts;

use crate::tts::markup::MarkupExt;

use crate::agent_activity::scheduling::SpeechId;
use crate::tool_executor::Toolset;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::AgentState;
pub use lk_agents_abi::voice::session_options::Instructions;
use lk_agents_abi::{
    ApiConnectOptions, ChatContext, ChatItem, ChatMessage, CopyOptions, Extra, FrameId,
    FunctionCall, FunctionCallOutput, Llm, MetricsReport, Role, ToolSpec, Tts,
};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// The output of an injected in-progress tool call, so the model waits instead of re-issuing it.
pub const RUNNING_TOOL_PLACEHOLDER: &str = "The tool call is still in progress.";

/// `extra` flag marking an injected pair. Must not change: it travels in a chat item's `extra`.
pub const RUNNING_PLACEHOLDER_KEY: &str = "__lk_running_placeholder__";

/// The id of the rendered prompt in every request. Must not change (persisted contexts carry it).
pub const INSTRUCTIONS_MESSAGE_ID: &str = lk_agents_core::request::INSTRUCTIONS_MESSAGE_ID;

/// `remove_instructions(chat_ctx)` as a plan: the positions of every
/// [`INSTRUCTIONS_MESSAGE_ID`] item, ascending, given the items' ids.
#[lk_agents_abi::host(export(name = "remove_instructions", pid = false))]
pub fn instructions_positions(item_ids: impl IntoIterator<Item = impl Into<String>>) -> Vec<usize> {
    crate::tts::markup::positions_of(item_ids.into_iter().map(Into::into), INSTRUCTIONS_MESSAGE_ID)
}

/// Python truthiness for an `f64`: zero is "absent".
fn truthy(v: f64) -> bool {
    v != 0.0
}

/// An epoch float back to the wall clock, for the one stamp the metrics fold owns.
fn wall_from_secs(secs: f64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs_f64(secs.max(0.0))
}

/// The first segment's synthesis, as the metrics block reads it (`_TTSGenerationData`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TtsStats {
    /// Time to first audio byte.
    pub ttfb: Option<f64>,
    /// `perf_counter` when this segment's first text reached the TTS; `None` when a custom
    /// `tts_node` publishes no stamp (so `llm_node_ttfs` is optional).
    pub synthesis_started_at: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnStats {
    pub llm_ttft: Option<f64>,
    pub llm_tps: Option<f64>,
    /// `llm_gen_data.started_at`, the anchor for `_time_to_first_sentence`.
    pub llm_started_at: Option<f64>,
    /// `self.llm.metrics_metadata` / `self.tts.metrics_metadata`: a pipeline reply's providers.
    pub llm_metadata: Option<lk_agents_abi::MetricsMetadata>,
    pub tts_metadata: Option<lk_agents_abi::MetricsMetadata>,
    /// `llm_gen_data.generated_extra`, the assistant message's `extra`.
    pub llm_extra: Extra,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FinishInputs {
    /// `time.time()` at `._pipeline_reply_task_impl`, taken the moment the forwarding loop exits.
    pub stopped_speaking_at: f64,
    /// `speech_handle.interrupted`, which lands on the assistant message.
    pub interrupted: bool,
    pub stats: TurnStats,
    /// `llm_gen_data.id`, minted host-side.
    pub message_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnOutcome {
    /// `"".join(out.forwarded_text for out in segment_outputs)`.
    pub forwarded_text: String,
    /// The message to commit, carrying `assistant_metrics`; `None` when `forwarded_text` is empty
    /// (a tool-only turn, which also lets `_previous_user_metrics` carry over).
    pub assistant_message: Option<ChatMessage>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnAccumulator {
    reply_started_at: SystemTime,
    user_metrics: Option<MetricsReport>,
    started_speaking_at: Option<f64>,
    first_tts_segment: Option<usize>,
    first_tts: TtsStats,
    forwarded_text: String,
    /// A segment's synthesis failed for good, or the output cut one under it: the item reads as
    /// cut short.
    cut_short: bool,
}

impl TurnAccumulator {
    /// `reply_started_at` (after authorization) is only the assistant message's `created_at`; the
    /// user metrics' `stopped_speaking_at` anchors `e2e_latency`.
    pub fn new(reply_started_at: SystemTime, user_metrics: Option<MetricsReport>) -> Self {
        TurnAccumulator {
            reply_started_at,
            user_metrics,
            started_speaking_at: None,
            first_tts_segment: None,
            first_tts: TtsStats::default(),
            forwarded_text: String::new(),
            cut_short: false,
        }
    }

    /// Synthesis may start early; message chronology starts when playback is authorized.
    pub fn set_reply_started_at(&mut self, at: SystemTime) {
        self.reply_started_at = at;
    }

    /// `if first_tts_gen_data is None: first_tts_gen_data = segment.tts`: only the first segment
    /// with a synthesis feeds the turn's TTS metrics; later segments' calls are dropped.
    pub fn record_tts(&mut self, segment: usize, stats: TtsStats) {
        match self.first_tts_segment {
            None => {
                self.first_tts_segment = Some(segment);
                self.first_tts = stats;
            }
            Some(first) if first == segment => self.first_tts = stats,
            Some(_) => {}
        }
    }

    /// Adoption supplies the completed user turn's metrics to a preemptive reply.
    pub fn set_user_metrics(&mut self, metrics: MetricsReport) {
        self.user_metrics = Some(metrics);
    }

    pub fn started_speaking_at(&self) -> Option<f64> {
        self.started_speaking_at
    }

    /// `_on_first_frame`: stamps the turn's first audible instant, `at` (the capture instant for
    /// audio, nothing for text) or `now` (`fut.result() or time.time()`). `false` if not the turn's
    /// first frame (it fires per segment and from both outputs).
    pub fn on_first_frame(&mut self, at: Option<f64>, now: f64) -> bool {
        if self.started_speaking_at.is_some() {
            return false;
        }
        // Truthiness, not presence.
        self.started_speaking_at = Some(at.filter(|at| truthy(*at)).unwrap_or(now));
        true
    }

    /// `out.tts_failed`: a reply cut short by a failed TTS (or by its output) reads as
    /// interrupted to the LLM.
    pub fn cut_short(&mut self) {
        self.cut_short = true;
    }

    /// One forwarded segment's text (`segment_outputs.append(out)`).
    pub fn record(&mut self, forwarded_text: &str) {
        self.forwarded_text.push_str(forwarded_text);
    }

    /// The metrics block and the assistant message (`._pipeline_reply_task_impl`).
    pub fn finish(self, inputs: FinishInputs) -> TurnOutcome {
        let FinishInputs { stopped_speaking_at, interrupted, stats, message_id } = inputs;

        let mut metrics = MetricsReport {
            llm_node_ttft: stats.llm_ttft,
            llm_node_tps: stats.llm_tps,
            // `_time_to_first_sentence`; `None` unless both anchors exist.
            llm_node_ttfs: match (stats.llm_started_at, self.first_tts.synthesis_started_at) {
                (Some(llm_started), Some(synth_started)) => Some(synth_started - llm_started),
                (Some(_), None) | (None, Some(_)) | (None, None) => None,
            },
            tts_node_ttfb: self.first_tts.ttfb,
            llm_metadata: stats.llm_metadata,
            tts_metadata: stats.tts_metadata,
            ..Default::default()
        };

        // `if stopped_speaking_at and started_speaking_at:` guards the whole speaking block.
        if let Some(started_speaking_at) = self.started_speaking_at.filter(|v| truthy(*v))
            && truthy(stopped_speaking_at)
        {
            metrics.started_speaking_at = Some(started_speaking_at);
            metrics.stopped_speaking_at = Some(stopped_speaking_at);
            // Forwarding starts at the speaking instant: no sink reports it apart.
            metrics.playback_latency = Some(0.0);
            metrics.e2e_latency = self
                .user_metrics
                .as_ref()
                .and_then(|m| m.stopped_speaking_at)
                .map(|stopped| started_speaking_at - stopped);
        }

        // Acoustic stamp: an interrupted item sorts before the user turn that cut it; a turn never
        // audible keeps the authorization instant.
        let created_at = self
            .started_speaking_at
            .filter(|v| truthy(*v))
            .map(wall_from_secs)
            .unwrap_or(self.reply_started_at);
        let assistant_message = (!self.forwarded_text.is_empty()).then(|| {
            let mut msg =
                ChatMessage::text(message_id, Role::Assistant, &self.forwarded_text, created_at);
            msg.interrupted = interrupted || self.cut_short;
            msg.metrics = metrics;
            msg.extra = stats.llm_extra;
            msg
        });

        TurnOutcome { forwarded_text: self.forwarded_text, assistant_message }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EndOfSpeech {
    /// `_update_agent_state(..)`, thinking or listening; `None` leaves the state alone.
    pub state: Option<AgentState>,
    /// `_on_end_of_agent_speech(ignore_user_transcript_until=time.time())`.
    pub end_of_agent_speech: bool,
    /// `_restore_interruption_by_audio_activity()`.
    pub restore_interruption_by_audio_activity: bool,
}

/// The state transition at the end of the speech. `tools_running`: "a running tool keeps the agent
/// busy; 'listening' would arm the away timer".
pub fn end_of_speech(
    interrupted: bool,
    has_tool_output: bool,
    tools_running: bool,
    agent_state_is_speaking: bool,
    interruption_enabled: bool,
) -> EndOfSpeech {
    let state = if !interrupted && has_tool_output {
        Some(AgentState::Thinking)
    } else if agent_state_is_speaking {
        Some(if !interrupted && tools_running {
            AgentState::Thinking
        } else {
            AgentState::Listening
        })
    } else {
        return EndOfSpeech::default();
    };

    EndOfSpeech {
        state,
        end_of_agent_speech: true,
        restore_interruption_by_audio_activity: interruption_enabled,
    }
}

/// A tool call still running from a previous turn, with the caller-minted id for its placeholder output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunningToolCall {
    pub call: FunctionCall,
    /// `id` for the `FunctionCallOutput` carrying [`RUNNING_TOOL_PLACEHOLDER`].
    pub output_id: String,
}

#[derive(Debug, Clone)]
pub struct RequestInputs<'a> {
    /// Read, never written.
    pub record: &'a ChatContext,
    /// Merged by id.
    pub pending: &'a [ChatItem],
    pub new_message: Option<&'a ChatMessage>,
    pub instructions: &'a str,
    /// `generate_reply(instructions=..)`: a system message for this request alone.
    pub turn_instructions: Option<&'a str>,
    pub tools: &'a [ToolSpec],
    /// This reply runs inside the agent's `on_enter`: the `IGNORE_ON_ENTER` tools are withheld.
    pub on_enter: bool,
    pub running_tool_calls: &'a [RunningToolCall],
    /// `session.output.audio_enabled`; exactly [`Request::tts_enabled`].
    pub audio_output_enabled: bool,
    pub text_output_enabled: bool,
    /// `Clock::wall`: the stamp of every message this function mints.
    pub now: SystemTime,
    /// This turn's markup guide ([`ExpressiveLayers::guide`]); `None` is expressive off.
    pub expressive: Option<&'a str>,
}

/// One request, ready to send. Its trace step `ChatRequested` is derived from it.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub chat_ctx: ChatContext,
    /// After the on-enter exclusion.
    pub tools: Vec<ToolSpec>,
    /// Whether this step synthesises audio. Drives the segmenter and the per-segment TTS.
    pub tts_enabled: bool,
    pub text_output_enabled: bool,
    pub instructions_hash: u64,
    /// The markup guide it was planned with: the TTS is told, and a follow-up step plans with it.
    pub expressive: Option<String>,
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// Assemble one request: pending and new items by `created_at`, the expressive guide and the
/// running placeholders, then the core's plan (prompt first, per-turn instructions last).
pub fn plan_request(inputs: RequestInputs<'_>) -> Request {
    let RequestInputs {
        record,
        pending,
        new_message,
        instructions,
        turn_instructions,
        tools,
        on_enter,
        running_tool_calls,
        audio_output_enabled,
        text_output_enabled,
        now,
        expressive,
    } = inputs;

    let mut ctx = record.copy(&CopyOptions::default());

    let known: std::collections::HashSet<String> =
        ctx.items.iter().map(|it| it.id().to_owned()).collect();
    ctx.insert_all(pending.iter().filter(|it| !known.contains(it.id())).cloned());
    if let Some(msg) = new_message
        && !known.contains(msg.id.as_str())
    {
        if msg.role == Role::User {
            lk_agents_core::request::answer_pending_input(
                &mut ctx.items,
                &msg.raw_text_content().unwrap_or_default(),
            );
        }
        ctx.insert(msg.clone());
    }
    // On, the guide replaces last turn's instead of stacking; off, it goes and past turns lose
    // their markup.
    match expressive {
        Some(guide) => {
            if !guide.trim().is_empty() {
                upsert_expressive_instructions(&mut ctx, guide, now);
            }
        }
        None => {
            crate::tts::markup::remove_expressive_instructions(&mut ctx);
            crate::tts::markup::strip_assistant_markup(&mut ctx);
        }
    }

    inject_running_tool_calls(&mut ctx, running_tool_calls);

    let planned = lk_agents_core::request::plan_request(lk_agents_core::request::RequestSnapshot {
        history: &ctx.items,
        instructions,
        tools,
        on_enter,
        turn_instructions,
        created_at: now,
    });
    Request {
        chat_ctx: planned.chat_ctx,
        tools: planned.tools,
        tts_enabled: audio_output_enabled,
        text_output_enabled,
        instructions_hash: stable_hash(instructions),
        expressive: expressive.map(str::to_owned),
    }
}

/// `update_expressive_instructions`: the keyed system message, replaced in place (its stamp kept)
/// or appended.
fn upsert_expressive_instructions(ctx: &mut ChatContext, text: &str, now: SystemTime) {
    use crate::tts::markup::EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID as ID;
    match ctx.items.iter().position(|it| it.id() == ID) {
        Some(idx) => {
            let created_at = ctx.items[idx].created_at();
            ctx.items[idx] = ChatMessage::text(ID, Role::System, text, created_at).into();
        }
        None => ctx.push(ChatMessage::text(ID, Role::System, text, now)),
    }
}

/// `"You can control how you speak..."`: `DEFAULT_EXPRESSIVE_OPTIONS`' template.
pub const DEFAULT_TTS_INSTRUCTIONS_TEMPLATE: &str = "You can control how you speak using the \
following formatting tags. Use them when appropriate to make your speech more expressive and \
natural:\n\n{tts.markup.llm_instructions}";

/// The session's `expressive=` and the active agent's; the agent's wins when it said one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpressiveLayers {
    pub session: lk_agents_abi::voice::session_options::Expressive,
    /// `None`: the agent left it unsaid (`NOT_GIVEN`).
    pub agent: Option<lk_agents_abi::voice::session_options::Expressive>,
}

impl ExpressiveLayers {
    /// `_resolve_expressive_options` and `_inject_expressive_instructions`: this turn's markup
    /// guide, `None` when expressive is off. Off without a TTS, without a markup dialect, and for
    /// a natively streaming TTS nothing lowers markup for; `Some("")` is on with nothing to inject.
    pub fn guide(
        &self,
        tts: Option<&lk_agents_abi::resources::TtsRef>,
        modality: lk_agents_abi::Modality,
    ) -> Option<String> {
        use crate::tts::markup::SpeechSteering;
        use lk_agents_abi::voice::session_options::Instructions;
        let tts = tts?;
        let markup = tts.markup;
        markup.llm_instructions(None)?;
        if tts.capabilities.streaming && !tts.lowers_markup {
            return None;
        }
        let cfg = self.agent.as_ref().unwrap_or(&self.session).0.as_ref()?;
        // `resolve_expressive_options`: the user's steering over `DEFAULT_SPEECH_STEERING_OPTIONS`.
        let user = cfg.speech_steering.clone().unwrap_or_default();
        let steering = SpeechSteering { disfluencies: user.disfluencies.or(Some(true)), ..user };
        let mut template = cfg
            .tts_instructions_template
            .clone()
            .unwrap_or_else(|| Instructions::plain(DEFAULT_TTS_INSTRUCTIONS_TEMPLATE));
        // `_append_instructions`: onto the raw common text, so placeholders survive to the render.
        let fragment = markup.steering_instructions(&steering);
        for extra in [Some(fragment.as_str()), cfg.tts_instructions_append.as_deref()] {
            if let Some(extra) = extra.filter(|e| !e.is_empty()) {
                template.common = format!("{}\n\n{extra}", template.common);
            }
        }
        let llm_instructions = markup.llm_instructions(Some(&steering)).unwrap_or_default();
        if llm_instructions.is_empty() {
            return Some(String::new());
        }
        let data =
            serde_json::json!({ "tts": { "markup": { "llm_instructions": llm_instructions } } });
        Some(safe_render(&template.render(Some(modality)), &data).text)
    }
}

/// A stable hash of a rendered prompt, for `ChatRequested{instructions_hash}`.
pub fn stable_hash(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// Inject a flagged in-progress pair for each tool still running, so the model doesn't re-issue it.
fn inject_running_tool_calls(ctx: &mut ChatContext, running: &[RunningToolCall]) {
    // Owned, since the loop inserts into `ctx`.
    let mut existing: std::collections::HashSet<String> = ctx
        .items
        .iter()
        .filter_map(|it| match it {
            ChatItem::FunctionCall(c) => Some(c.call_id.clone()),
            ChatItem::FunctionCallOutput(o) => Some(o.call_id.clone()),
            ChatItem::Message(_) | ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => {
                None
            }
        })
        .collect();

    for entry in running {
        let call_id = entry.call.call_id.clone();
        if !existing.insert(call_id.clone()) {
            continue;
        }

        let mut call = entry.call.clone();
        let mut extra: Extra = call.extra.clone();
        extra.insert(RUNNING_PLACEHOLDER_KEY.to_string(), serde_json::Value::Bool(true));
        call.extra = extra;

        let mut out = FunctionCallOutput::new(
            &entry.output_id,
            &call_id,
            RUNNING_TOOL_PLACEHOLDER,
            false,
            // The call's time, so the pair sits next to the call.
            entry.call.created_at,
        );
        out.name = entry.call.name.clone();

        ctx.insert_all([ChatItem::from(call), ChatItem::from(out)]);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    /// Placeholder paths, in the order they appear, that `data` had no value for.
    pub missing: Vec<String>,
}

/// Fill `{placeholder}` and `{dotted.path}` holes from `data`, with `{{`/`}}` escapes. Unsupported
/// forms and missing values are reported in [`Rendered::missing`].
pub fn safe_render(template: &str, data: &serde_json::Value) -> Rendered {
    let mut out = String::with_capacity(template.len());
    let mut missing = Vec::new();
    let mut chars = template.char_indices().peekable();

    while let Some((_, c)) = chars.next() {
        match c {
            '{' if chars.peek().map(|(_, c)| *c) == Some('{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek().map(|(_, c)| *c) == Some('}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut field = String::new();
                let mut closed = false;
                for (_, c) in chars.by_ref() {
                    if c == '}' {
                        closed = true;
                        break;
                    }
                    field.push(c);
                }
                if !closed {
                    // An unterminated hole is left verbatim and reported missing.
                    out.push('{');
                    out.push_str(&field);
                    missing.push(field);
                    continue;
                }
                match lookup(data, &field) {
                    Some(v) => out.push_str(&v),
                    None => missing.push(field),
                }
            }
            other => out.push(other),
        }
    }

    Rendered { text: out, missing }
}

/// Resolve `a.b.c` against `data`. `None` is "no value"; a JSON `null` resolves to `""`.
fn lookup(data: &serde_json::Value, field: &str) -> Option<String> {
    let mut cur = data;
    for part in field.split('.') {
        cur = cur.as_object()?.get(part)?;
    }
    Some(match cur {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attempt {
    budget: ApiConnectOptions,
    /// Attempts made so far, the current one included. Starts at one.
    attempt: u32,
    /// This attempt has produced output, so no failure of it may be retried.
    produced: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Retry {
    /// Wait `after`, then attempt again. `attempt` is the number of the attempt that failed.
    After { attempt: u32, after: Duration },
    /// Terminal: budget spent, not retryable, or output already produced.
    Never,
}

impl Attempt {
    pub fn new(budget: ApiConnectOptions) -> Self {
        Attempt { budget, attempt: 1, produced: false }
    }

    /// Every later failure is terminal.
    pub fn produced_output(&mut self) {
        self.produced = true;
    }

    /// The attempt failed; on [`Retry::After`] the counter has advanced. A client cancellation
    /// (499) of a completion or synthesis, output already produced, a non-retryable error or a
    /// spent allowance ends it; the delay is the connection options' schedule.
    pub fn failed(
        &mut self,
        error: &lk_agents_abi::errors::ProviderError,
        kind: lk_agents_abi::types::retry::RequestKind,
    ) -> Retry {
        use lk_agents_abi::types::retry::RequestKind::{Llm, TtsBatch, TtsStream};
        let retries_so_far = self.attempt - 1;
        let cancelled =
            matches!(kind, Llm | TtsBatch | TtsStream) && error.status_code() == Some(499);
        if self.produced
            || cancelled
            || !error.retryable()
            || retries_so_far >= self.budget.max_retry
        {
            return Retry::Never;
        }
        let delay = self.budget.interval_for_retry(retries_so_far);
        let failed = self.attempt;
        self.attempt += 1;
        Retry::After { attempt: failed, after: Duration::from_secs_f64(delay.max(0.0)) }
    }
}

/// `say()` and `generate_reply()` share one pipeline, which keeps `say()` interruptible, scheduled
/// and metered like every speech.
// Not `Clone`: the streamed variant owns a receiver.
#[derive(Debug)]
pub enum ReplySource {
    Llm,
    /// Not re-segmented.
    Text(String),
    /// **One segment per chunk**, so the caller's chunk boundaries set the prosody. Taken once
    /// (`say` has no follow-up step).
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    TextStream(std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<String>>>),
    /// **Pre-synthesised** audio, played instead of asking the TTS; `text` is still committed.
    Audio {
        text: String,
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        rx: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<lk_agents_abi::AudioFrame>>>,
    },
    /// Runtime installs this speech's player after admission.
    Realtime,
}

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
pub struct ReplyContext {
    /// Resolved from the active agent at creation.
    pub preemptive_tts: bool,
    /// Built by `plan_request`, the only planner.
    pub request: crate::generation::Request,
    /// The first request is re-planned once the speech playing at creation is on the record.
    pub replan: Option<Replan>,
    pub source: ReplySource,
    /// Every host call inherits it.
    pub frame: FrameId,
    /// Fixed before the first frame; the anchor for `e2e_latency`.
    pub user_metrics: Option<lk_agents_abi::MetricsReport>,
    /// Minted by the driver: the pure tier has no RNG.
    pub message_id: String,
    /// `None` on a first step (`NOT_GIVEN`, not `"none"`); on a follow-up, `next_step`'s decision,
    /// which is what stops a tool loop.
    pub tool_choice: Option<lk_agents_abi::ToolChoice>,
    /// `None` reads the frame's live toolset per step, so `update_tools` mid-reply applies next
    /// step.
    pub toolset: Option<Arc<Toolset>>,
    pub providers: Providers,
    /// The speech options as of the reply's creation: a later `update_options` reaches the next
    /// reply, not this one.
    pub speech: crate::agent_activity::options::SpeechOptions,
    /// This reply answers an `InputRequired(tool=)`: its call of that tool takes the person's words.
    pub answering: Option<crate::tool_executor::Answering>,
    /// Made while a `VoiceTask`'s agent held the floor: a `Completed` from its tools ends the task.
    pub task: bool,
}

impl std::fmt::Debug for ReplyContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplyContext")
            .field("frame", &self.frame)
            .field("message_id", &self.message_id)
            .finish_non_exhaustive()
    }
}

pub use lk_agents_abi::agent::HostInstructions;

/// Append dynamic instructions to the request's static prompt.
pub fn append_instructions(ctx: &mut ChatContext, dynamic: &str) {
    if dynamic.is_empty() {
        return;
    }
    let Some(ChatItem::Message(prompt)) =
        ctx.items.iter_mut().find(|item| item.id() == INSTRUCTIONS_MESSAGE_ID)
    else {
        return;
    };
    let text = match prompt.raw_text_content() {
        Some(base) if !base.is_empty() => format!("{base}\n\n{dynamic}"),
        _ => dynamic.to_owned(),
    };
    prompt.content = vec![lk_agents_abi::ChatContent::Text(text)];
}

/// Snapshotted at creation: a reply in flight keeps the providers it started with.
#[derive(Clone)]
pub struct Providers {
    /// The frame's instructions functions, rendered before each model request; `None` when the
    /// agent has none.
    pub dynamic_instructions: Option<Arc<dyn HostInstructions>>,
    pub llm: Option<Arc<dyn Llm>>,
    /// A provider with no duplex path arrives adapted by the executor's `TtsAdapter`.
    pub tts: Option<Arc<dyn Tts>>,
    /// A per-turn override rides on [`ReplyContext::toolset`] instead.
    pub toolset: Arc<Toolset>,
    pub tools_host_extra: Vec<lk_agents_abi::HostRef>,
    /// The `transcription_node` and `realtime_audio_output_node` of the agent the reply was
    /// created under.
    pub output_nodes: crate::agent_session::nodes::OutputNodes,
}

/// Built once `after` is on the record, so the interrupted item precedes it. Only `ChatRequested`
/// waits.
#[derive(Debug, Clone)]
pub struct Replan {
    pub after: SpeechId,
    pub spec: PlanSpec,
}

#[derive(Debug, Clone)]
pub struct PlanSpec {
    /// `None` plans against the frame's record.
    pub record: Option<lk_agents_abi::ChatContext>,
    pub new_message: Option<lk_agents_abi::ChatMessage>,
    /// `None` renders the frame's instructions.
    pub instructions: Option<String>,
    pub turn_instructions: Option<String>,
    pub tools: Vec<lk_agents_abi::ToolSpec>,
    pub on_enter: bool,
    pub running_tool_calls: Vec<crate::generation::RunningToolCall>,
    pub audio_output_enabled: bool,
    pub text_output_enabled: bool,
    /// This turn's markup guide; `None` is expressive off.
    pub expressive: Option<String>,
}

impl PlanSpec {
    pub fn plan(
        &self,
        record: &lk_agents_abi::ChatContext,
        instructions: &str,
        now: std::time::SystemTime,
    ) -> crate::generation::Request {
        self.plan_pending(record, instructions, &[], now)
    }

    pub(crate) fn plan_pending(
        &self,
        record: &ChatContext,
        instructions: &str,
        pending: &[ChatItem],
        now: SystemTime,
    ) -> Request {
        crate::generation::plan_request(crate::generation::RequestInputs {
            record: self.record.as_ref().unwrap_or(record),
            pending,
            new_message: self.new_message.as_ref(),
            instructions: self.instructions.as_deref().unwrap_or(instructions),
            turn_instructions: self.turn_instructions.as_deref(),
            tools: &self.tools,
            on_enter: self.on_enter,
            running_tool_calls: &self.running_tool_calls,
            audio_output_enabled: self.audio_output_enabled,
            text_output_enabled: self.text_output_enabled,
            now,
            expressive: self.expressive.as_deref(),
        })
    }
}

/// A reply's first request, planned against the core's accepted record and instructions once
/// `after` is on that record: finished, or its tools committed.
#[derive(Debug)]
pub struct ReplyPlan {
    pub speech: SpeechId,
    pub after: SpeechId,
    pub spec: PlanSpec,
    pub tool_choice: Option<lk_agents_abi::ToolChoice>,
}

/// A planned model step: the core's answer to a [`ReplyPlan`], or a reply's next step.
#[derive(Debug)]
pub struct PlannedReply {
    pub request: Request,
    pub tool_choice: Option<lk_agents_abi::ToolChoice>,
}

/// Chunks never travel here; only the moments the session machines must hear about, each a
/// cross-machine write the task may not make itself.
#[derive(Debug)]
pub enum GenEvent {
    BindOutput {
        call: crate::agent_session::RequestId,
        speech: SpeechId,
    },
    ActivateOutput {
        call: crate::agent_session::RequestId,
        binding: crate::agent_session::RequestId,
        speech: SpeechId,
    },
    ReleaseOutput {
        binding: crate::agent_session::RequestId,
        speech: SpeechId,
    },
    OutputUnsettled {
        binding: crate::agent_session::RequestId,
        speech: SpeechId,
    },
    /// Answered with [`Answer::ReplyPlan`](crate::agent_session::Answer::ReplyPlan) under `call`.
    Plan {
        call: crate::agent_session::RequestId,
        plan: Box<ReplyPlan>,
    },
    /// With an audio output the edge is [`GenEvent::PlaybackStarted`]'s. Between them the only
    /// setters of `_agent_speaking`.
    FirstFrame {
        speech: SpeechId,
    },

    Done {
        speech: SpeechId,
        outcome: Box<TurnOutcome>,
        interrupted: bool,
    },

    /// Forwarded to the bus, ordered against every other `GenEvent`.
    Emit(Box<crate::events::SessionEvent>),

    /// `None` when no call, failure or timeout.
    KeytermAnswer {
        req: u64,
        arguments: Option<String>,
        failed: bool,
        metrics: Option<Box<lk_agents_abi::metrics::base::LlmMetrics>>,
    },

    /// A model step's calls are over (it may have made none). Awaited (`answer`) with the reply's
    /// [`StepNext`](crate::tool_executor::StepNext) when the task goes on from it.
    ToolsFinished {
        speech: SpeechId,
        /// `Source::{ToolCall, ToolOutput}` carry it.
        step: u32,
        end: Box<crate::tool_executor::StepEnd>,
        answer: Option<crate::agent_session::RequestId>,
    },

    ToolStep {
        speech: SpeechId,
        step: u32,
    },

    /// Reported **before** the step's tools run, so the loop can end the current speech per step.
    StepSpoken {
        speech: SpeechId,
        step: u32,
        interrupted: bool,
        has_tool_output: bool,
        tools_running: bool,
        /// Up to the played position; `None` for a tool-only step.
        item: Option<Box<lk_agents_abi::ChatMessage>>,
    },

    /// The error is **moved**, never formatted: a [`ProviderError::Host`] owns a side-table entry
    /// only the binding can release.
    Failed {
        speech: SpeechId,
        error: Box<ProviderError>,
    },

    Trace(crate::trace::Step),
    /// As the sink reported it: the loop's `speaking` edge (`_on_first_frame`).
    PlaybackStarted {
        speech: SpeechId,
    },
    /// Not reported when the hold ended with the speech's interruption.
    LaunchReleased {
        speech: SpeechId,
    },
    ChatFinished {
        speech: SpeechId,
        step: u32,
        error: Option<crate::trace::FailureClass>,
    },
    /// The model made a call. Awaited (`go`) with the call's start, or with `Gone` when the
    /// driver drops it unrun.
    ToolCallMade {
        speech: SpeechId,
        step: u32,
        made: Box<crate::tool_executor::MadeCall>,
        go: crate::agent_session::RequestId,
    },
    /// The host bound and validated the call's arguments, or did not prepare them. Awaited with
    /// the call's next move.
    ToolPrepared {
        speech: SpeechId,
        call: crate::tool_executor::CallId,
        identity: Result<
            lk_agents_abi::agent::ArgumentIdentity,
            lk_agents_core::tool_calls::PrepareFailure,
        >,
        go: crate::agent_session::RequestId,
    },
    /// The tool's read-back function answered. Awaited with the call's settlement.
    ReadBackRendered {
        speech: SpeechId,
        call: crate::tool_executor::CallId,
        answer: Result<lk_agents_abi::agent::ReadBackAnswer, String>,
        go: crate::agent_session::RequestId,
    },
    /// As the call's answer arrives; `message` is `None` when there is nothing to say. `call`:
    /// its id in the reply's run, once it started.
    ToolCallEnded {
        speech: SpeechId,
        call: Option<crate::tool_executor::CallId>,
        call_id: String,
        end: crate::tool_executor::ToolEnd,
        message: Option<String>,
        outcome: crate::tool_executor::ToolOutcome,
    },
    /// The reply's agent validated an output candidate; `value` is what it accepted.
    OutputValidated {
        speech: SpeechId,
        validation: lk_agents_abi::agent::run::ValidationId,
        verdict: lk_agents_core::agent_run::OutputVerdict,
        value: Option<lk_agents_abi::HostRef>,
    },
    /// The host rendered a `tool_handling` template; `None` keeps the framework's text.
    TemplateRendered {
        rendering: crate::tool_executor::Rendering,
        text: Option<String>,
    },
    /// `recoverable` when the budget retries it. Then the close policy.
    ProviderFailed {
        speech: SpeechId,
        leg: crate::trace::ProviderKind,
        recoverable: bool,
        error: Box<ProviderError>,
    },
}

#[cfg(test)]
mod expressive_tests {
    use super::*;
    use crate::tts::markup::{EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID, Markup, SpeechSteering};
    use lk_agents_abi::Modality;
    use lk_agents_abi::resources::{ResourceId, TtsRef};
    use lk_agents_abi::voice::session_options::{Expressive, ExpressiveConfig};
    use lk_agents_abi::{ChatContext, TtsCapabilities};

    fn tts(provider: &str, streaming: bool, lowers_markup: bool) -> TtsRef {
        TtsRef {
            id: ResourceId(1),
            provider: lk_agents_abi::ProviderId::of_label("t".to_owned()),
            capabilities: TtsCapabilities { streaming, aligned_transcript: false },
            sample_rate: 24_000,
            num_channels: 1,
            markup: Markup::from_provider_key(provider),
            lowers_markup,
        }
    }

    fn on(cfg: ExpressiveConfig) -> Expressive {
        Expressive(Some(cfg))
    }

    // Catches: the agent's `expressive` not winning over the session's, in either direction.
    #[test]
    fn the_agents_setting_overrides_the_sessions() {
        let t = tts("cartesia", false, true);
        let mut layers = ExpressiveLayers { session: on(Default::default()), agent: None };
        assert!(layers.guide(Some(&t), Modality::Audio).is_some());
        layers.agent = Some(Expressive(None));
        assert_eq!(layers.guide(Some(&t), Modality::Audio), None, "Agent(expressive=False)");
        layers.session = Expressive(None);
        layers.agent = Some(on(Default::default()));
        assert!(layers.guide(Some(&t), Modality::Audio).is_some(), "Agent(expressive=True)");
    }

    // Catches: markup taught to the LLM when no TTS can speak it: no dialect, no TTS, or a
    // natively streaming plugin nothing lowers markup for.
    #[test]
    fn it_is_off_without_a_dialect_or_a_lowering_path() {
        let layers = ExpressiveLayers { session: on(Default::default()), agent: None };
        assert_eq!(layers.guide(None, Modality::Audio), None);
        assert_eq!(layers.guide(Some(&tts("", false, true)), Modality::Audio), None);
        assert_eq!(layers.guide(Some(&tts("cartesia", true, false)), Modality::Audio), None);
        assert!(layers.guide(Some(&tts("cartesia", true, true)), Modality::Audio).is_some());
    }

    // Catches: `{}` read as off (the reference's `not expr and not isinstance(expr, dict)`), the
    // template's placeholder left unfilled, the default disfluencies line missing, and the user's
    // append not landing last.
    #[test]
    fn an_empty_dict_is_on_and_the_guide_is_rendered_from_the_template() {
        let t = tts("cartesia", false, true);
        let layers = ExpressiveLayers {
            session: on(ExpressiveConfig {
                tts_instructions_append: Some("Be brief.".into()),
                ..Default::default()
            }),
            agent: None,
        };
        let guide = layers.guide(Some(&t), Modality::Audio).unwrap();
        assert!(guide.starts_with("You can control how you speak"), "{guide}");
        assert!(!guide.contains("{tts.markup.llm_instructions}"), "{guide}");
        let steering = SpeechSteering { disfluencies: Some(true), ..Default::default() };
        let taught = Markup::from_provider_key("cartesia").llm_instructions(Some(&steering));
        assert!(guide.contains(&taught.unwrap()), "{guide}");
        assert!(guide.contains("natural fillers"), "{guide}");
        assert!(guide.ends_with("Be brief."), "{guide}");
    }

    fn inputs<'a>(
        record: &'a ChatContext,
        expressive: Option<&'a str>,
        now: SystemTime,
    ) -> RequestInputs<'a> {
        RequestInputs {
            record,
            pending: &[],
            new_message: None,
            instructions: "be nice",
            turn_instructions: None,
            tools: &[],
            on_enter: false,
            running_tool_calls: &[],
            audio_output_enabled: true,
            text_output_enabled: true,
            now,
            expressive,
        }
    }

    fn texts(ctx: &ChatContext) -> String {
        format!("{ctx:?}")
    }

    // Catches: the guide stacking a copy per turn instead of replacing the last one in place, and
    // an expressive-off turn keeping the guide or the past markup.
    #[test]
    fn the_guide_is_upserted_when_on_and_removed_with_the_markup_when_off() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(5);
        let mut record = ChatContext::new(Vec::new());
        record.push(ChatMessage::text(EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID, Role::System, "old", t0));
        let said = "hi <expr type=\"emotion\" label=\"happy\"/>there";
        record.push(ChatMessage::text("a", Role::Assistant, said, t0));

        let on = plan_request(inputs(&record, Some("new guide"), t1));
        let guides: Vec<_> = on
            .chat_ctx
            .items
            .iter()
            .filter(|it| it.id() == EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID)
            .collect();
        assert_eq!(guides.len(), 1);
        assert_eq!(guides[0].created_at(), t0, "replaced in place, stamp kept");
        assert_eq!(on.expressive.as_deref(), Some("new guide"));
        assert!(texts(&on.chat_ctx).contains("new guide"));
        assert!(texts(&on.chat_ctx).contains("<expr"), "markup kept while on");

        let off = plan_request(inputs(&record, None, t1));
        assert!(off.chat_ctx.items.iter().all(|it| it.id() != EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID));
        assert!(!texts(&off.chat_ctx).contains("<expr"), "markup stripped when off");
    }
}

/// A local generation failure that cannot be retried.
pub fn fatal(message: impl Into<String>) -> ProviderError {
    ProviderError::Local(lk_agents_abi::errors::ApiError::new(message).with_retryable(false))
}
