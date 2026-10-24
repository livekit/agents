//! `voice/generation.py`: the pure planning of a step (request, turn metrics, retries), what a
//! reply takes and reports ([`ReplyContext`], [`GenEvent`]), and the pipeline, one file per leg.

mod forward;
mod llm;
pub(crate) mod pipeline_reply;
pub(crate) mod playout;
mod realtime_generation;
pub(crate) mod tools;
mod tts;

use crate::agent_activity::scheduling::SpeechId;
use crate::tool_executor::Toolset;
use lk_agents_abi::Modality;
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
#[lk_agents_abi::host(export(pid = false, name = "instructions_message_id"))]
pub const INSTRUCTIONS_MESSAGE_ID: &str = "lk.agent_task.instructions";

/// `remove_instructions(chat_ctx)` as a plan: the positions of every
/// [`INSTRUCTIONS_MESSAGE_ID`] item, ascending, given the items' ids.
#[lk_agents_abi::host(export(name = "remove_instructions", pid = false))]
pub fn instructions_positions(item_ids: impl IntoIterator<Item = impl Into<String>>) -> Vec<usize> {
    lk_agents_abi::tts::markup::positions_of(
        item_ids.into_iter().map(Into::into),
        INSTRUCTIONS_MESSAGE_ID,
    )
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
        }
    }

    /// Synthesis may start early; message chronology starts when playback is authorized.
    pub(crate) fn set_reply_started_at(&mut self, at: SystemTime) {
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
            msg.interrupted = interrupted;
            msg.metrics = metrics;
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

/// The `instructions=` argument of one turn, with the id for the system message it becomes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnInstructions {
    pub instructions: Instructions,
    /// Minted by the caller. An ordinary system message, not [`INSTRUCTIONS_MESSAGE_ID`].
    pub message_id: String,
}

#[derive(Debug, Clone)]
pub struct RequestInputs<'a> {
    /// Read, never written.
    pub record: &'a ChatContext,
    /// Merged by id.
    pub pending: &'a [ChatItem],
    pub new_message: Option<&'a ChatMessage>,
    pub instructions: &'a str,
    /// `generate_reply(instructions=..)`.
    pub turn_instructions: Option<&'a TurnInstructions>,
    pub tools: &'a [ToolSpec],
    /// This reply runs inside the agent's `on_enter`: the `IGNORE_ON_ENTER` tools are withheld.
    pub on_enter: bool,
    pub running_tool_calls: &'a [RunningToolCall],
    pub modality: Modality,
    /// `session.output.audio_enabled`; exactly [`Request::tts_enabled`].
    pub audio_output_enabled: bool,
    pub text_output_enabled: bool,
    /// `Clock::wall`: the stamp of every message this function mints.
    pub now: SystemTime,
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
}

/// Assemble one request. Order matters: stale prompts removed, pending and new items by
/// `created_at`, prompt first, per-turn instructions, expressive guide, placeholders last.
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
        modality,
        audio_output_enabled,
        text_output_enabled,
        now,
    } = inputs;

    let mut ctx = record.copy(&CopyOptions::default());
    ctx.items.retain(|item| item.id() != INSTRUCTIONS_MESSAGE_ID);

    let tools: Vec<ToolSpec> =
        tools.iter().filter(|t| !(on_enter && t.flags.ignore_on_enter)).cloned().collect();

    let known: std::collections::HashSet<String> =
        ctx.items.iter().map(|it| it.id().to_owned()).collect();
    ctx.insert_all(pending.iter().filter(|it| !known.contains(it.id())).cloned());
    if let Some(msg) = new_message
        && !known.contains(msg.id.as_str())
    {
        ctx.insert(msg.clone());
    }

    ctx.items.insert(
        0,
        ChatMessage::text(INSTRUCTIONS_MESSAGE_ID, Role::System, instructions, now).into(),
    );
    if let Some(turn) = turn_instructions {
        ctx.push(ChatMessage::text(
            &turn.message_id,
            Role::System,
            turn.instructions.render(Some(modality)),
            now,
        ));
    }

    // Expressive is never resolved for a request: the guide goes and past turns lose their markup,
    // in the request's copy.
    lk_agents_abi::tts::markup::remove_expressive_instructions(&mut ctx);
    lk_agents_abi::tts::markup::strip_assistant_markup(&mut ctx);

    inject_running_tool_calls(&mut ctx, running_tool_calls);

    Request {
        chat_ctx: ctx,
        tools,
        tts_enabled: audio_output_enabled,
        text_output_enabled,
        instructions_hash: stable_hash(instructions),
    }
}

/// A stable hash of a rendered prompt, for `ChatRequested{instructions_hash}`.
pub fn stable_hash(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

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

    /// The attempt failed; on [`Retry::After`] the counter has advanced.
    pub fn failed(&mut self, retryable: bool) -> Retry {
        let retries_so_far = self.attempt - 1;
        if self.produced || !retryable || retries_so_far >= self.budget.max_retry {
            return Retry::Never;
        }
        let after =
            Duration::from_secs_f64(self.budget.interval_for_retry(retries_so_far).max(0.0));
        let failed = self.attempt;
        self.attempt += 1;
        Retry::After { attempt: failed, after }
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
    TextStream(std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<String>>>),
    /// **Pre-synthesised** audio, played instead of asking the TTS; `text` is still committed.
    Audio {
        text: String,
        rx: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<lk_agents_abi::AudioFrame>>>,
    },
    /// Held for the speech's life: a tool follow-up waits for the next generation on the same
    /// channel. `wake` is the loop's queue, ordered against the socket's events.
    Realtime {
        events: tokio::sync::Mutex<
            tokio::sync::mpsc::UnboundedReceiver<lk_agents_abi::llm::realtime::Event>,
        >,
        wake: crate::agent_session::Tx<crate::agent_activity::realtime::RealtimeWake>,
    },
}

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
    pub speech: crate::agent_activity::options::SpeechPolicy,
}

/// Snapshotted at creation: a reply in flight keeps the providers it started with.
#[derive(Clone)]
pub struct Providers {
    pub llm: Option<Arc<dyn Llm>>,
    /// A provider with no duplex path arrives wrapped in a
    /// [`StreamAdapter`](lk_agents_abi::StreamAdapter).
    pub tts: Option<Arc<dyn Tts>>,
    /// A per-turn override rides on [`ReplyContext::toolset`] instead.
    pub toolset: Arc<Toolset>,
    pub tools_host_extra: Vec<lk_agents_abi::HostRef>,
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
    pub turn_instructions: Option<crate::generation::TurnInstructions>,
    pub tools: Vec<lk_agents_abi::ToolSpec>,
    pub on_enter: bool,
    pub running_tool_calls: Vec<crate::generation::RunningToolCall>,
    pub audio_output_enabled: bool,
    pub text_output_enabled: bool,
}

impl PlanSpec {
    pub fn plan(
        &self,
        record: &lk_agents_abi::ChatContext,
        instructions: &str,
        now: std::time::SystemTime,
    ) -> crate::generation::Request {
        crate::generation::plan_request(crate::generation::RequestInputs {
            record: self.record.as_ref().unwrap_or(record),
            pending: &[],
            new_message: self.new_message.as_ref(),
            instructions: self.instructions.as_deref().unwrap_or(instructions),
            turn_instructions: self.turn_instructions.as_ref(),
            tools: &self.tools,
            on_enter: self.on_enter,
            running_tool_calls: &self.running_tool_calls,
            modality: lk_agents_abi::Modality::Audio,
            audio_output_enabled: self.audio_output_enabled,
            text_output_enabled: self.text_output_enabled,
            now,
        })
    }
}

/// Chunks never travel here; only the moments the session machines must hear about, each a
/// cross-machine write the task may not make itself.
#[derive(Debug)]
pub enum GenEvent {
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

    ToolsFinished {
        speech: SpeechId,
        /// `Source::{ToolCall, ToolOutput}` carry it.
        step: u32,
        commit: Vec<lk_agents_abi::ChatItem>,
        executed: Option<Box<crate::tool_executor::ToolsExecutedEvent>>,
        warnings: Vec<crate::tool_executor::StepWarning>,
        /// With `executed`: the step waits here for the event's handlers (a host round trip).
        verdict: Option<tokio::sync::oneshot::Sender<crate::tool_executor::ToolsVerdict>>,
    },

    ToolStep {
        speech: SpeechId,
        step: u32,
    },

    Handoff {
        speech: SpeechId,
        agent: lk_agents_abi::HostRef,
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
    /// `hosted`, or answered by the driver itself (an unadvertised name).
    ToolCallStarted {
        speech: SpeechId,
        step: u32,
        call: Box<lk_agents_abi::FunctionCall>,
        hosted: bool,
    },
    /// `None` when there is nothing to say.
    ToolCallEnded {
        speech: SpeechId,
        call_id: String,
        end: crate::tool_executor::ToolEnd,
        message: Option<String>,
    },
    /// `recoverable` when the budget retries it. Then the close policy.
    ProviderFailed {
        speech: SpeechId,
        leg: crate::trace::ProviderLeg,
        recoverable: bool,
        error: Box<ProviderError>,
    },
}
