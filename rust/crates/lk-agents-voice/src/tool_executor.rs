use crate::agent::{Source, Visibility};
use crate::agent_session::cycle::Cycle;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{Driver, FillerSource, HookId, Origin};
use crate::events::{DiagnosticEvent, SessionEvent, ToolEndStatus, ToolLifecycle};
use crate::filler_scheduler::{FillerAction, FillerId, FillerScheduler};
use crate::generation::RunningToolCall;
use crate::trace::{CoreStep, ToolStep};
use lk_agents_abi::{
    ChatItem, FunctionCall, FunctionCallOutput, HostRef, MetricsReport, ToolChoice, Ts,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::SystemTime;

/// The reference's `REPLY_INSTRUCTIONS_AT_TAIL`, used when the host renders nothing.
pub const REPLY_AT_TAIL: &str = "New results arrived from background tool calls (call_ids: {call_ids}).\nSummarize the results naturally. Do NOT repeat information you have already told the user.";

/// The reference's `REPLY_INSTRUCTIONS_MAYBE_COVERED`.
pub const REPLY_MAYBE_COVERED: &str = "New results arrived from background tool calls (call_ids: {call_ids}).\nYou may have already mentioned them in your most recent replies.\nIf you already told the user everything in these results, reply with an empty response (no text at all).\nOtherwise, summarize only what you have not said yet, with a natural transition.\nNever repeat information you have already told the user.";

/// The answer to a call no runnable tool has (`generation._execute_tools_task`), in the reference's
/// words so the model corrects itself.
#[lk_agents_abi::host(export)]
pub fn unknown_tool_message(name: &str, available: &[String]) -> String {
    format!("Unknown function: {name} - available tools: {}", available.join(", "))
}

/// The reference's `RuntimeError` for an `Agent` returned after `ctx.update()`.
pub const HANDOFF_AFTER_UPDATE: &str = "agent handoff after a progress update is not supported";

/// The replacement output for a handoff that never landed because the turn was interrupted.
pub const INTERRUPTED_HANDOFF_OUTPUT: &str = "the agent handoff was interrupted and did not happen";

pub const MISSING_TOOL_OUTPUT: &str = "the tool executor did not report a result";

/// `make_function_call_output`'s text for an unexpected exception; says nothing about it.
pub const INTERNAL_ERROR_OUTPUT: &str = "An internal error occurred";

/// `make_function_call_output`'s text for a return value the model cannot be told about.
pub const INVALID_OUTPUT: &str = "the tool returned an invalid output";

/// `make_tool_output`'s text when two `Agent`s came back from one call.
pub const MULTIPLE_AGENTS_OUTPUT: &str = "the tool returned more than one agent";

pub(crate) use tools::Tools;

/// The tool state, behind its methods: the driver reaches none of its fields.
mod tools {
    use super::*;

    /// `_RunningTasks` and the deferred reply (`AgentActivity`'s background tool calls), the open
    /// filler scopes, and the steps whose `function_tools_executed` the host is handling.
    #[derive(Debug, Default)]
    pub(crate) struct Tools {
        running: Vec<Running>,
        pending: Vec<Pending>,
        reply: Option<Reply>,
        rendering: Option<Rendering>,
        fillers: BTreeMap<FillerId, FillerRun>,
        handling: BTreeMap<HookId, Handling>,
    }

    impl Tools {
        /// `_RunningTasks` as the planner reads it: every hosted call not yet ended, in start order.
        pub(crate) fn running_tool_calls(&self) -> Vec<RunningToolCall> {
            self.running
                .iter()
                .map(|r| RunningToolCall { call: r.call.clone(), output_id: r.output_id.clone() })
                .collect()
        }

        /// The speech a running call belongs to (`Origin::Tool{call_id}` -> its speech).
        pub(crate) fn speech_of(&self, call_id: &str) -> Option<SpeechId> {
            self.running.iter().find(|r| r.call.call_id == call_id).map(|r| r.speech)
        }

        /// The first running call named `call_id`, out of the set.
        pub(crate) fn take(&mut self, call_id: &str) -> Option<Running> {
            let i = self.running.iter().position(|r| r.call.call_id == call_id)?;
            Some(self.running.remove(i))
        }

        pub(crate) fn live_blocking(&self, speech: SpeechId) -> u32 {
            u32::try_from(self.running.iter().filter(|r| r.speech == speech && r.updates == 0).count())
                .unwrap_or(u32::MAX)
        }

        /// A hosted call enters the running set.
        pub(crate) fn start(
            &mut self,
            call: FunctionCall,
            output_id: String,
            speech: SpeechId,
            step: u32,
        ) {
            self.running.push(Running { call, output_id, speech, step, updates: 0 });
        }

        /// `speech`'s call `call_id` ended: detached, it stays with its first update counted; otherwise
        /// it leaves the set. `false` when no such call runs. By speech as well as id: a model may reuse
        /// call ids.
        pub(crate) fn end(&mut self, speech: SpeechId, call_id: &str, detached: bool) -> bool {
            let Some(i) =
                self.running.iter().position(|r| r.speech == speech && r.call.call_id == call_id)
            else {
                return false;
            };
            if detached {
                self.running[i].updates = 1;
            } else {
                self.running.remove(i);
            }
            true
        }

        /// `speech`'s blocking calls still running, in start order.
        pub(crate) fn live(&self, speech: SpeechId) -> Vec<Running> {
            self.running.iter().filter(|r| r.speech == speech && r.updates == 0).cloned().collect()
        }

        /// A detached call named `call_id` runs: it may send updates.
        pub(crate) fn is_detached(&self, call_id: &str) -> bool {
            self.running.iter().any(|r| r.call.call_id == call_id && r.updates > 0)
        }

        /// The detached call's next progress update: its index, and the call as it stands.
        pub(crate) fn next_update(&mut self, call_id: &str) -> Option<(u32, Running)> {
            let r = self.running.iter_mut().find(|r| r.call.call_id == call_id && r.updates > 0)?;
            let n = r.updates;
            r.updates += 1;
            Some((n, r.clone()))
        }

        /// An update's pair, batched for the next deferred reply.
        pub(crate) fn queue_update(&mut self, agent: String, items: Vec<ChatItem>, update_id: String) {
            self.pending.push(Pending { agent, items, update_id });
        }

        /// Updates wait, and no reply or render is in flight.
        pub(crate) fn reply_due(&self) -> bool {
            !self.pending.is_empty() && self.reply.is_none() && self.rendering.is_none()
        }

        pub(crate) fn take_batch(&mut self) -> Batch {
            Batch(std::mem::take(&mut self.pending))
        }

        pub(crate) fn rendering(&mut self, hook: HookId, batch: Batch, at_tail: bool) {
            self.rendering = Some(Rendering { hook, batch, at_tail });
        }

        /// The render `hook` answered: its batch and tail fact.
        pub(crate) fn rendered(&mut self, hook: HookId) -> Option<(Batch, bool)> {
            let Rendering { batch, at_tail, .. } = self.rendering.take_if(|r| r.hook == hook)?;
            Some((batch, at_tail))
        }

        pub(crate) fn replying(&mut self, speech: SpeechId, update_ids: Vec<String>) {
            self.reply = Some(Reply { speech, update_ids });
        }

        /// `speech` was the deferred reply: the updates it answered.
        pub(crate) fn reply_done(&mut self, speech: SpeechId) -> Option<Vec<String>> {
            self.reply.take_if(|r| r.speech == speech).map(|r| r.update_ids)
        }

        /// The close: nothing pending is delivered, a render in flight is dropped.
        pub(crate) fn abandon_replies(&mut self) {
            self.pending.clear();
            self.rendering = None;
        }

        /// The host runs `function_tools_executed`'s handlers for the step under `hook`.
        pub(crate) fn handle_executed(&mut self, hook: HookId, handling: Handling) {
            self.handling.insert(hook, handling);
        }

        /// The handlers under `hook` are done (or will not answer).
        pub(crate) fn executed_handled(&mut self, hook: HookId) -> Option<Handling> {
            self.handling.remove(&hook)
        }

        // --- fillers ---

        pub(crate) fn has_filler(&self, id: FillerId) -> bool {
            self.fillers.contains_key(&id)
        }

        pub(crate) fn has_fillers(&self) -> bool {
            !self.fillers.is_empty()
        }

        pub(crate) fn open_filler(
            &mut self,
            id: FillerId,
            sched: FillerScheduler,
            call_id: String,
            speech: SpeechId,
            source: FillerSource,
        ) {
            self.fillers
                .insert(id, FillerRun { sched, call_id, speech, source, asking: None, armed: None });
        }

        /// The scope closed: whether its timer was armed, and the source hook it was asking.
        pub(crate) fn close_filler(&mut self, id: FillerId) -> Option<(bool, Option<HookId>)> {
            let run = self.fillers.remove(&id)?;
            Some((run.armed.is_some(), run.asking))
        }

        pub(crate) fn filler_ids(&self) -> Vec<FillerId> {
            self.fillers.keys().copied().collect()
        }

        pub(crate) fn fillers_of_speech(&self, speech: SpeechId) -> Vec<FillerId> {
            self.fillers.iter().filter(|(_, r)| r.speech == speech).map(|(id, _)| *id).collect()
        }

        pub(crate) fn fillers_of_call(&self, call_id: &str) -> Vec<FillerId> {
            self.fillers.iter().filter(|(_, r)| r.call_id == call_id).map(|(id, _)| *id).collect()
        }

        /// `reset_dwell`: somebody is busy.
        pub(crate) fn filler_busy(&mut self, id: FillerId, now: Ts) -> Option<FillerAction> {
            self.fillers.get_mut(&id).map(|r| r.sched.on_busy(now))
        }

        /// A wake's move for one scope: the dwell starts at idle and restarts when somebody speaks.
        pub(crate) fn filler_wake(
            &mut self,
            id: FillerId,
            idle: bool,
            busy: bool,
            now: Ts,
        ) -> Option<FillerAction> {
            let run = self.fillers.get_mut(&id)?;
            match run.sched.waiting() {
                FillerAction::WaitForIdle if idle && !busy && run.asking.is_none() => {
                    Some(run.sched.on_idle(now))
                }
                FillerAction::Dwell(_) if busy => Some(run.sched.on_busy(now)),
                FillerAction::WaitForIdle
                | FillerAction::Dwell(_)
                | FillerAction::CoolDown(_)
                | FillerAction::Fire { .. }
                | FillerAction::Done => None,
            }
        }

        /// The scope's deadline passed; one in the same wake as a speaking edge counts as busy.
        pub(crate) fn filler_timer(
            &mut self,
            id: FillerId,
            busy: bool,
            now: Ts,
        ) -> Option<FillerAction> {
            let run = self.fillers.get_mut(&id)?;
            run.armed = None;
            Some(match run.sched.waiting() {
                FillerAction::Dwell(_) if busy => run.sched.on_busy(now),
                FillerAction::WaitForIdle
                | FillerAction::Dwell(_)
                | FillerAction::CoolDown(_)
                | FillerAction::Fire { .. }
                | FillerAction::Done => run.sched.tick(now),
            })
        }

        /// The host answered the source `hook`: its scope and the step it asked for.
        pub(crate) fn filler_answered(&mut self, hook: HookId) -> Option<(FillerId, u32)> {
            let (id, run) = self.fillers.iter_mut().find(|(_, r)| r.asking == Some(hook))?;
            run.asking = None;
            Some((*id, run.sched.steps()))
        }

        pub(crate) fn filler_fired(
            &mut self,
            id: FillerId,
            now: Ts,
            produced: bool,
        ) -> Option<FillerAction> {
            self.fillers.get_mut(&id).map(|r| r.sched.fired(now, produced))
        }

        /// The timer a wait needs: `WaitForIdle` disarms, a dwell or cooldown (re)arms at its deadline.
        /// `None` when the scope is gone.
        pub(crate) fn filler_wait(&mut self, id: FillerId, until: Option<Ts>) -> Option<FillerTimer> {
            let run = self.fillers.get_mut(&id)?;
            Some(match until {
                None => FillerTimer { cancel: run.armed.take().is_some(), arm: None },
                Some(at) if run.armed == Some(at) => FillerTimer { cancel: false, arm: None },
                Some(at) => {
                    let cancel = run.armed.is_some();
                    run.armed = Some(at);
                    FillerTimer { cancel, arm: Some(at) }
                }
            })
        }

        pub(crate) fn filler_source(&self, id: FillerId) -> Option<FillerSource> {
            self.fillers.get(&id).map(|r| r.source.clone())
        }

        pub(crate) fn filler_asking(&mut self, id: FillerId, hook: HookId) {
            if let Some(run) = self.fillers.get_mut(&id) {
                run.asking = Some(hook);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Running {
    call: FunctionCall,
    /// The placeholder output id, minted once so it keeps one identity across plans.
    output_id: String,
    speech: SpeechId,
    step: u32,
    /// Progress updates sent so far, the first (the step's answer) included.
    updates: u32,
}

impl Running {
    pub(crate) fn call(&self) -> &FunctionCall {
        &self.call
    }

    pub(crate) fn speech(&self) -> SpeechId {
        self.speech
    }

    pub(crate) fn step(&self) -> u32 {
        self.step
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Pending {
    /// The frame whose record got the pair; a different frame at delivery re-inserts it.
    agent: String,
    items: Vec<ChatItem>,
    update_id: String,
}

/// The pending updates one deferred reply answers, in arrival order.
#[derive(Debug, Clone)]
pub(crate) struct Batch(Vec<Pending>);

impl Batch {
    /// Every update's id, in order (the reply's `{call_ids}`).
    pub(crate) fn update_ids(&self) -> Vec<String> {
        self.0.iter().map(|p| p.update_id.clone()).collect()
    }

    /// The id of the last item the batch committed.
    pub(crate) fn last_item_id(&self) -> Option<String> {
        self.0.last().and_then(|p| p.items.last()).map(|i| i.id().to_owned())
    }

    /// The items another frame's record got, to re-insert for this request only.
    pub(crate) fn foreign_items(&self, agent: &str) -> Vec<ChatItem> {
        self.0.iter().filter(|p| p.agent != agent).flat_map(|p| p.items.iter().cloned()).collect()
    }
}

#[derive(Debug, Clone)]
struct Reply {
    speech: SpeechId,
    update_ids: Vec<String>,
}

/// A batch handed to the host for rendering, kept until the answer or the bound.
#[derive(Debug, Clone)]
struct Rendering {
    hook: HookId,
    batch: Batch,
    at_tail: bool,
}

/// One open `with_filler` scope (`voice/filler_scheduler.py`).
#[derive(Debug)]
struct FillerRun {
    sched: FillerScheduler,
    call_id: String,
    speech: SpeechId,
    source: FillerSource,
    asking: Option<HookId>,
    armed: Option<Ts>,
}

/// A timer change a filler's wait asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FillerTimer {
    /// The armed deadline goes.
    pub(crate) cancel: bool,
    /// A new deadline.
    pub(crate) arm: Option<Ts>,
}

/// A step's `function_tools_executed` with the host, kept until its handlers answer.
#[derive(Debug)]
pub(crate) struct Handling {
    pub(crate) speech: SpeechId,
    pub(crate) step: u32,
    pub(crate) event: ToolsExecutedEvent,
    pub(crate) warnings: Vec<StepWarning>,
    pub(crate) verdict: tokio::sync::oneshot::Sender<ToolsVerdict>,
}

/// `{call_ids}` as Python's `str.format` spells a list: `['a', 'b']`.
fn python_list(ids: &[String]) -> String {
    let quoted: Vec<String> = ids.iter().map(|id| format!("'{id}'")).collect();
    format!("[{}]", quoted.join(", "))
}

pub fn default_reply_instructions(call_ids: &[String], at_tail: bool) -> String {
    let template = if at_tail { REPLY_AT_TAIL } else { REPLY_MAYBE_COVERED };
    template.replace("{call_ids}", &python_list(call_ids))
}

/// `_make_update_pair`: a `FunctionCall` under `id` copied from the running call, and its output.
pub(crate) fn pair_for(
    running: &Running,
    id: &str,
    outcome: ToolOutcome,
    wall: std::time::SystemTime,
    mut mint: impl FnMut() -> String,
) -> (FunctionCall, FunctionCallOutput) {
    let call = FunctionCall {
        id: mint(),
        call_id: id.to_owned(),
        name: running.call.name.clone(),
        arguments: running.call.arguments.clone(),
        created_at: wall,
        extra: running.call.extra.clone(),
        group_id: None,
    };
    let execution = answer(&call, outcome, wall);
    let mut output = execution.output;
    output.id = mint();
    (call, output)
}

// --- fillers: `RunContext.with_filler` on the driver's clock ---

#[derive(Debug, Clone, PartialEq)]
pub struct ToolExecution {
    pub call: FunctionCall,
    pub output: FunctionCallOutput,
    /// The `Agent` a tool returned, opaque to the core. `Some` is a requested handoff.
    pub agent_task: Option<HostRef>,
    /// How the run ended; a detached call is still running after its output is committed.
    pub end: ToolEnd,
    /// The first update as the tool wrote it (`ToolCallUpdated.message`), for a detached call.
    pub raw_update: Option<String>,
}

/// How a tool's run ended (`ToolCallEnded{status}`), plus `Detached`, which is not an end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEnd {
    /// A value, a silent return, a handoff: `status: "done"`.
    Done,
    /// An error output the model reads: `status: "error"`.
    Error,
    /// The host cancelled the tool: `status: "cancelled"`, an empty output, no reply.
    Cancelled,
    /// `ctx.update()`: the step is answered and the tool keeps running.
    Detached,
}

/// What the host's run of one tool came back with, before any decision; [`decide`] makes them.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolAnswer {
    /// The tool returned `value` as JSON, with `Agent`s set aside in `agents`.
    Returned { value: Option<serde_json::Value>, agents: Vec<HostRef> },
    Raised(ToolRaise),
    /// A value the model can't be told about (`_is_valid_function_output`).
    Invalid,
    Cancelled,
    /// `ctx.update(message)`: rendered `message`, `raw` as written; the tool runs on.
    Detached { message: String, raw: String },
}

/// The three exception kinds the reference tells apart (`llm/utils.py`, `make_function_call_output`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolRaise {
    /// `ToolError(message)`: the author's words, for the model.
    ToolError(String),
    /// `StopResponse`: the call is answered and nothing is said.
    StopResponse,
    /// Anything else. The detail went to the host's log; the model is told nothing about it.
    Internal,
}

/// `REPLY_INSTRUCTIONS_AT_TAIL` and `REPLY_INSTRUCTIONS_MAYBE_COVERED`, the deferred-reply templates.
/// Pure constants, read at import, so no fork guard.
#[lk_agents_abi::host(export(pid = false))]
pub fn tool_reply_templates() -> (&'static str, &'static str) {
    (REPLY_AT_TAIL, REPLY_MAYBE_COVERED)
}

/// A detached tool's raw answer (`tool_update(kind="final", answer=..)`): the marshal's dict; a
/// malformed one is the `ValueError` naming what is wrong.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for ToolAnswer {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let raw: lk_agents_abi::llm::tool_executor::RawToolAnswer = lk_agents_abi::python::extract_generated(&obj.to_owned())?;
        ToolAnswer::try_from(raw).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

impl TryFrom<lk_agents_abi::llm::tool_executor::RawToolAnswer> for ToolAnswer {
    type Error = String;

    /// The marshal's dict as a [`ToolAnswer`]. An unknown kind, or one missing its key, is an
    /// error, never a guess.
    fn try_from(raw: lk_agents_abi::llm::tool_executor::RawToolAnswer) -> Result<Self, String> {
        use lk_agents_abi::llm::tool_executor::{ToolAnswerKind, ToolRaiseKind};
        let kind = ToolAnswerKind::from_wire(&raw.kind).ok_or_else(|| format!("unknown tool answer kind {:?}", raw.kind))?;
        let message = |m: Option<String>| m.ok_or_else(|| format!("a `{}` answer carries a `message`", raw.kind));
        Ok(match kind {
            ToolAnswerKind::Returned => ToolAnswer::Returned { value: raw.value, agents: raw.agents },
            ToolAnswerKind::Raised => {
                let raised = raw.error.as_deref().and_then(ToolRaiseKind::from_wire);
                ToolAnswer::Raised(match raised.ok_or_else(|| format!("unknown raised kind {:?}", raw.error))? {
                    ToolRaiseKind::ToolError => ToolRaise::ToolError(message(raw.message)?),
                    ToolRaiseKind::StopResponse => ToolRaise::StopResponse,
                    ToolRaiseKind::Internal => ToolRaise::Internal,
                })
            }
            ToolAnswerKind::Invalid => ToolAnswer::Invalid,
            ToolAnswerKind::Cancelled => ToolAnswer::Cancelled,
            ToolAnswerKind::Detached => ToolAnswer::Detached {
                message: message(raw.message)?,
                raw: raw.raw.ok_or("a `detached` answer carries `raw`")?,
            },
        })
    }
}

/// Constructed only as a whole.
#[derive(Default, Clone)]
pub struct Toolset {
    /// Empty makes a missing `runner` unreachable.
    pub specs: Vec<lk_agents_abi::ToolSpec>,
    /// Required when `specs` is non-empty and the LLM is host-backed: `PyLlm` cannot rebuild a
    /// Python `Tool` from a `ToolSpec`.
    pub host: Option<lk_agents_abi::HostRef>,
    pub runner: Option<Arc<dyn ToolRunner>>,
}

impl std::fmt::Debug for Toolset {
    /// Hand-written: `runner` and `host` cannot derive it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Toolset")
            .field("tools", &self.specs.len())
            .field("host", &self.host.is_some())
            .field("runner", &self.runner.is_some())
            .finish()
    }
}

/// `_start_ivr_detection`'s `send_dtmf_events`. The frame's own tools win on a name clash; kept
/// across handoffs.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SessionTools {
    pub specs: Vec<lk_agents_abi::ToolSpec>,
    pub host: Option<lk_agents_abi::HostRef>,
}

/// The only port on the generation path; what the core decides around it is pure (`next_step`).
#[async_trait::async_trait]
pub trait ToolRunner: Send + Sync {
    /// Resolves when the tool returns, raises, is cancelled, or releases the step with its first
    /// `ctx.update()`.
    async fn run(
        &self,
        call: lk_agents_abi::FunctionCall,
        speech: SpeechId,
        step: u32,
    ) -> crate::tool_executor::ToolAnswer;
}

/// `voice/tool_executor.py`'s `_ToolExecutor`: runs one call and reports the marshal's dict. A
/// report, not a decision; [`decide`] owns the output.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.tool_executor._ToolExecutor"))]
#[async_trait::async_trait]
pub trait HostToolRunner: Send + Sync {
    async fn run(
        &self,
        call: &FunctionCall,
        speech_id: u64,
        step: u32,
    ) -> Result<lk_agents_abi::llm::tool_executor::RawToolAnswer, lk_agents_abi::errors::HostError>;
}

/// The driver's [`ToolRunner`](crate::tool_executor::ToolRunner) over a host executor. Every
/// call gets exactly one answer: a marshal that raises or reports an unreadable dict is answered
/// as the reference's internal error, and logged.
pub struct HostTools(pub std::sync::Arc<dyn HostToolRunner>);

#[async_trait::async_trait]
impl crate::tool_executor::ToolRunner for HostTools {
    async fn run(&self, call: FunctionCall, speech: SpeechId, step: u32) -> ToolAnswer {
        let answer = match self.0.run(&call, speech.0, step).await {
            Ok(raw) => ToolAnswer::try_from(raw),
            Err(e) => Err(e.to_string()),
        };
        answer.unwrap_or_else(|e| {
            log::error!(target: "livekit.agents", function = call.name.as_str(), call_id = call.call_id.as_str();
                "the tool executor failed outside the tool's own error handling: {e}");
            ToolAnswer::Raised(ToolRaise::Internal)
        })
    }
}

/// What one tool produced, decided; [`answer`] turns it into exactly one output.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolOutcome {
    Reply(String),
    /// Empty output, no reply: a `None` return or `StopResponse`.
    Silent,
    Error(String),
    /// A handoff, with its text; a bare handoff asks for no reply.
    Handoff { agent: HostRef, text: Option<String> },
    /// A progress update: answered like a reply, and the call is marked non-blocking.
    Detached { message: String, raw: String },
    /// The host cancelled the tool: an empty output, no reply, not an error.
    Cancelled,
    Missing,
}

/// What the model reads: a string verbatim, anything else as canonical JSON.
pub fn render(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The decision: text, error, whether a reply is asked for (`make_tool_output` +
/// `make_function_call_output`).
pub fn decide(answer: ToolAnswer) -> ToolOutcome {
    match answer {
        ToolAnswer::Returned { value, mut agents } => match agents.len() {
            0 => match value {
                None => ToolOutcome::Silent,
                Some(v) => ToolOutcome::Reply(render(&v)),
            },
            1 => {
                let agent = agents.pop().expect("one agent");
                // None is a bare handoff, one is that item, several stay a list.
                let text = match value {
                    None => None,
                    Some(serde_json::Value::Array(items)) => match items.len() {
                        0 => None,
                        1 => Some(render(&items[0])),
                        _ => Some(render(&serde_json::Value::Array(items))),
                    },
                    Some(other) => Some(render(&other)),
                };
                ToolOutcome::Handoff { agent, text }
            }
            // Two agents from one call: the handoff is refused entirely rather than one picked.
            _ => ToolOutcome::Error(MULTIPLE_AGENTS_OUTPUT.to_owned()),
        },
        ToolAnswer::Raised(ToolRaise::ToolError(message)) => ToolOutcome::Error(message),
        ToolAnswer::Raised(ToolRaise::StopResponse) => ToolOutcome::Silent,
        ToolAnswer::Raised(ToolRaise::Internal) => {
            ToolOutcome::Error(INTERNAL_ERROR_OUTPUT.to_owned())
        }
        ToolAnswer::Invalid => ToolOutcome::Error(INVALID_OUTPUT.to_owned()),
        ToolAnswer::Cancelled => ToolOutcome::Cancelled,
        ToolAnswer::Detached { message, raw } => ToolOutcome::Detached { message, raw },
    }
}

/// The one formatter: every committed `FunctionCallOutput` comes through here.
pub fn answer(call: &FunctionCall, outcome: ToolOutcome, now: SystemTime) -> ToolExecution {
    let output = |text: String, is_error: bool, reply_required: bool| {
        let mut out = FunctionCallOutput::new(
            format!("{}_out", call.id),
            call.call_id.clone(),
            text,
            is_error,
            now,
        );
        out.name = call.name.clone();
        out.reply_required = reply_required;
        out
    };
    let mut call = call.clone();
    let mut raw_update = None;
    let (output, agent_task, end) = match outcome {
        ToolOutcome::Reply(text) => (output(text, false, true), None, ToolEnd::Done),
        ToolOutcome::Silent => (output(String::new(), false, false), None, ToolEnd::Done),
        ToolOutcome::Error(text) => (output(text, true, true), None, ToolEnd::Error),
        ToolOutcome::Handoff { agent, text } => {
            let reply_required = text.is_some();
            (output(text.unwrap_or_default(), false, reply_required), Some(agent), ToolEnd::Done)
        }
        ToolOutcome::Detached { message, raw } => {
            // `RunContext.update`'s marker: a non-blocking tool may wait on its own speech.
            call.extra.insert(
                crate::speech_handle::TOOL_NON_BLOCKING_EXTRA_KEY.to_owned(),
                serde_json::Value::Bool(true),
            );
            raw_update = Some(raw);
            (output(message, false, true), None, ToolEnd::Detached)
        }
        ToolOutcome::Cancelled => (output(String::new(), false, false), None, ToolEnd::Cancelled),
        ToolOutcome::Missing => {
            (output(MISSING_TOOL_OUTPUT.to_owned(), true, true), None, ToolEnd::Error)
        }
    };
    ToolExecution { call, output, agent_task, end, raw_update }
}

pub use lk_agents_abi::voice::events::ToolsExecutedEvent;

/// The follow-up step on the *same* speech. Its context is `plan_request`'s to build.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplyRequest {
    /// `tool_choice` for the follow-up: `"none"` when out of budget, handing off or draining; else
    /// `"auto"`. Never forcing, or the model loops into another call.
    pub tool_choice: ToolChoice,
    /// Carried over only when this turn forwarded no text, so `e2e_latency` can still be computed.
    pub previous_user_metrics: Option<MetricsReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NextStep {
    /// The turn is over: no tool output, or tool output that asked for no reply.
    Done,
    /// Barge-in; finished tools are still in [`StepDecision::commit`].
    Interrupted,
    /// A tool returned an `Agent`: update and drain, then still run `follow_up` if any.
    Handoff {
        agent: HostRef,
        /// `Some` when a reply is owed and the handle wasn't interrupted; `tool_choice` is `"none"`.
        follow_up: Option<ReplyRequest>,
    },
    Reply(ReplyRequest),
}

/// Something the driver should log; a core crate has no logger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepWarning {
    /// The step budget is spent; reported even when no reply follows.
    MaxToolStepsReached,
    /// Two tools returned an agent: the handoff is cancelled rather than one picked.
    MultipleAgentTasks,
}

/// `function_tools_executed` as its handlers left it, for the step waiting on them
/// ([`StepDecision::settle`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolsVerdict {
    pub executed: ToolsExecutedEvent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepDecision {
    /// Decided as if the host's `function_tools_executed` handlers changed nothing; [`settle`]
    /// decides it again from what they left.
    ///
    /// [`settle`]: StepDecision::settle
    pub next: NextStep,
    /// `function_tools_executed`; `None` with nothing to emit.
    pub tools_executed: Option<ToolsExecutedEvent>,
    /// Items committed on arrival, so results survive a reply that never runs. Calls, then outputs.
    pub commit: Vec<ChatItem>,
    pub warnings: Vec<StepWarning>,
    /// The `Agent` the step hands off to, before the event's `handoff_required` is read.
    agent_task: Option<HostRef>,
    max_steps_reached: bool,
    /// The second interruption read.
    interrupted: bool,
    previous_user_metrics: Option<MetricsReport>,
}

impl StepDecision {
    /// `function_tools_executed` was handled: the next step from what the handlers left
    /// (`cancel_tool_reply()`, `cancel_agent_handoff()`; agent_activity.py:3657-3670,
    /// :4338-4345). An interrupted step stays interrupted.
    pub fn settle(mut self, event: ToolsExecutedEvent) -> StepDecision {
        self.commit = commit_of(&event.function_calls, &event.function_call_outputs);
        if self.next != NextStep::Interrupted {
            let handoff = self.agent_task.clone().filter(|_| event.handoff_required);
            self.next = next_of(
                handoff,
                event.has_tool_reply() && !self.interrupted,
                self.max_steps_reached,
                self.previous_user_metrics.clone(),
            );
        }
        self.tools_executed = Some(event);
        self
    }
}

/// The step after the tools: a handoff drains, which forbids tools in its reply.
fn next_of(
    handoff: Option<HostRef>,
    reply: bool,
    forbid_tools: bool,
    previous_user_metrics: Option<MetricsReport>,
) -> NextStep {
    let follow_up = reply.then(|| ReplyRequest {
        tool_choice: if forbid_tools || handoff.is_some() {
            ToolChoice::Forbidden
        } else {
            ToolChoice::Auto
        },
        previous_user_metrics,
    });
    match handoff {
        Some(agent) => NextStep::Handoff { agent, follow_up },
        None => match follow_up {
            Some(req) => NextStep::Reply(req),
            None => NextStep::Done,
        },
    }
}

#[derive(Debug, Clone)]
pub struct StepInputs<'a> {
    /// The executions, in the model's order.
    pub tool_output: &'a [ToolExecution],
    /// Interrupted before the executions were awaited: [`NextStep::Interrupted`].
    pub interrupted_before_tools: bool,
    /// Re-read after the executions: a barge-in during a tool suppresses the reply but not the
    /// handoff, commit or step increment.
    pub interrupted: bool,
    /// The step count before increment; starts at 1.
    pub num_steps: u32,
    pub max_tool_steps: u32,
    /// This step forwarded no text; see [`ReplyRequest::previous_user_metrics`].
    pub forwarded_text_empty: bool,
    pub user_metrics: Option<MetricsReport>,
}

pub fn next_step(inputs: StepInputs<'_>) -> StepDecision {
    if inputs.interrupted_before_tools {
        return interrupted(inputs.tool_output);
    }

    // No tool output: nothing at all -- no event, no commit, no increment.
    if inputs.tool_output.is_empty() {
        return quiet(NextStep::Done);
    }

    let mut warnings = Vec::new();

    // `>= max + 1` on purpose: the `+ 1` is `num_steps` starting at 1.
    #[allow(clippy::int_plus_one)]
    let max_steps_reached = inputs.num_steps >= inputs.max_tool_steps + 1;
    if max_steps_reached {
        warnings.push(StepWarning::MaxToolStepsReached);
    }

    let mut calls = Vec::with_capacity(inputs.tool_output.len());
    let mut outputs = Vec::with_capacity(inputs.tool_output.len());
    let mut agent_task: Option<HostRef> = None;
    let mut ignore_task_switch = false;

    for out in inputs.tool_output {
        calls.push(out.call.clone());
        outputs.push(out.output.clone());

        if agent_task.is_some() && out.agent_task.is_some() {
            ignore_task_switch = true;
            warnings.push(StepWarning::MultipleAgentTasks);
        }
        // The last value wins, `None` included.
        agent_task = out.agent_task.clone();
    }

    let handoff = if ignore_task_switch { None } else { agent_task };
    let event = ToolsExecutedEvent {
        function_calls: calls,
        function_call_outputs: outputs,
        handoff_required: handoff.is_some(),
    };

    // The *second* interruption read gates the reply.
    StepDecision {
        next: NextStep::Done,
        tools_executed: None,
        commit: Vec::new(),
        warnings,
        agent_task: handoff,
        max_steps_reached,
        interrupted: inputs.interrupted,
        previous_user_metrics: inputs
            .forwarded_text_empty
            .then(|| inputs.user_metrics.clone())
            .flatten(),
    }
    .settle(event)
}

pub(crate) fn commit_of(calls: &[FunctionCall], outputs: &[FunctionCallOutput]) -> Vec<ChatItem> {
    let calls = calls.iter().cloned().map(ChatItem::from);
    calls.chain(outputs.iter().cloned().map(ChatItem::from)).collect()
}

pub(crate) fn emit_ended(
    cycle: &mut Cycle,
    id: String,
    call_id: String,
    message: Option<String>,
    status: ToolEndStatus,
) {
    let ended = ToolLifecycle::Ended { id, call_id, message, status };
    cycle.machine().emit(SessionEvent::ToolExecutionUpdated(ended));
}

fn quiet(next: NextStep) -> StepDecision {
    StepDecision {
        next,
        tools_executed: None,
        commit: Vec::new(),
        warnings: Vec::new(),
        agent_task: None,
        max_steps_reached: false,
        interrupted: false,
        previous_user_metrics: None,
    }
}

fn interrupted(tool_output: &[ToolExecution]) -> StepDecision {
    // Nothing to commit, no event: an interrupted turn whose tools all got cancelled is silent.
    if tool_output.is_empty() {
        return quiet(NextStep::Interrupted);
    }
    let calls: Vec<FunctionCall> = tool_output.iter().map(|out| out.call.clone()).collect();
    let outputs: Vec<FunctionCallOutput> =
        tool_output.iter().map(interrupted_tool_output).collect();
    let commit = commit_of(&calls, &outputs);
    StepDecision {
        tools_executed: Some(ToolsExecutedEvent {
            function_calls: calls,
            function_call_outputs: outputs,
            // An interruption cancels the handoff, as the replacement output tells the model.
            handoff_required: false,
        }),
        commit,
        ..quiet(NextStep::Interrupted)
    }
}

/// What to record for a tool that finished on an interrupted turn: a handoff answers as a failure,
/// and nothing asks for a reply.
fn interrupted_tool_output(out: &ToolExecution) -> FunctionCallOutput {
    let mut fnc_call_out = out.output.clone();
    if out.agent_task.is_some() {
        fnc_call_out.output = INTERRUPTED_HANDOFF_OUTPUT.to_string();
        fnc_call_out.is_error = true;
    }
    fnc_call_out.reply_required = false;
    fnc_call_out
}

impl Driver {
    /// The host's `function_tools_executed` handlers answered under `hook` (`left`: each output's
    /// `reply_required` and `handoff_required`), or will not (`None`): the step goes on.
    pub(crate) fn on_tools_executed_handled(
        &mut self,
        hook: crate::agent_session::HookId,
        left: Option<(Vec<bool>, bool)>,
        cycle: &mut Cycle,
    ) {
        let Some(Handling { speech, step, mut event, warnings, verdict }) =
            self.activity.tools_mut().executed_handled(hook)
        else {
            return;
        };
        if let Some((reply_required, handoff_required)) = left {
            if reply_required.len() == event.function_call_outputs.len() {
                for (out, required) in event.function_call_outputs.iter_mut().zip(reply_required) {
                    out.reply_required = required;
                }
            } else {
                cycle.trace(CoreStep::PortFailure(
                    "function_tools_executed answered for a different number of outputs",
                ));
            }
            // A handler can cancel the handoff, not ask for one.
            event.handoff_required &= handoff_required;
        }
        let executed = Some(Box::new(event));
        self.finish_tools(speech, step, Vec::new(), executed, warnings, Some(verdict), true, cycle);
    }

    /// A step's tools are over and `function_tools_executed` handled: the results committed with
    /// whether the activity drains, and the waiting step (`verdict`) told what the handlers left.
    /// `hosted`: the host emitted the event itself.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish_tools(
        &mut self,
        speech: SpeechId,
        step: u32,
        commit: Vec<ChatItem>,
        executed: Option<Box<ToolsExecutedEvent>>,
        warnings: Vec<StepWarning>,
        verdict: Option<tokio::sync::oneshot::Sender<ToolsVerdict>>,
        hosted: bool,
        cycle: &mut Cycle,
    ) {
        // What the handlers left is what is committed: calls, then outputs.
        let commit = match &executed {
            Some(e) => crate::tool_executor::commit_of(&e.function_calls, &e.function_call_outputs),
            None => commit,
        };
        // **Commit first**, so results survive even if the reply speech never runs. The call's
        // stamp is its dispatch, the output's its answer.
        let n = commit.len();
        for item in commit {
            let source = match &item {
                ChatItem::FunctionCall(_) => Source::ToolCall { speech, step },
                ChatItem::FunctionCallOutput(_)
                | ChatItem::Message(_)
                | ChatItem::AgentHandoff(_)
                | ChatItem::AgentConfigUpdate(_) => Source::ToolOutput { speech, step },
            };
            // A call the request never advertised is answered by the driver and reaches the
            // handle only here. Once, by id.
            if let ChatItem::FunctionCall(_) = &item
                && let Some(h) = self.activity.scheduling_task().handle(speech)
                && !h.chat_items().iter().any(|i| i.id() == item.id())
            {
                h.item_added(&item);
            }
            let _ = self.commit_item(item, source, Visibility::Both, cycle);
        }
        // `self.scheduling_paused` (agent_activity.py:3659): the activity that made the speech
        // drains -- paused, left behind by a handoff, or a task that has its result.
        let draining = self.activity.scheduling_task().is_paused()
            || self.activity.speech_frame(speech).is_some_and(|f| f != self.activity.agent())
            || self.activity.tasks().driving_settled();
        // The step's follow-up plans against the frame once this is published.
        self.activity.mark_tools_committed(speech, step, draining);
        self.activity.set_tools_in_flight(speech, false);
        self.publish_frame();
        for w in &warnings {
            // A core crate has no logger, so the host says these (and the trace records them).
            // Only the spent step budget has a sentence in the reference.
            if matches!(w, StepWarning::MaxToolStepsReached) {
                let d = DiagnosticEvent::warning(crate::events::diagnostics::MAX_TOOL_STEPS)
                    .with("speech_id", speech.0.to_string());
                cycle.machine().emit(SessionEvent::Diagnostic(d));
            }
            cycle.trace(CoreStep::PortFailure(match w {
                StepWarning::MaxToolStepsReached => {
                    "max_tool_steps reached: the reply is forced to tool_choice=none"
                }
                StepWarning::MultipleAgentTasks => {
                    "two tools returned an agent: the handoff is cancelled entirely"
                }
            }));
        }
        cycle.trace(ToolStep::Finished {
            speech,
            committed: n,
            handoff: executed.as_ref().is_some_and(|e| e.handoff_required),
        });
        let Some(executed) = executed else { return };
        // Not the host's to emit: after the commit, as before any hook existed for it.
        if !hosted {
            cycle.machine().emit(SessionEvent::ToolsExecuted(executed.clone()));
        }
        if let Some(verdict) = verdict {
            let _ = verdict.send(ToolsVerdict { executed: *executed });
        }
    }

    /// Another live speech is in its tools.
    pub(crate) fn tools_running_elsewhere(&self, speech: SpeechId) -> bool {
        let scheduling = self.activity.scheduling_task();
        self.activity.speeches_in_tools().any(|id| id != speech && scheduling.contains(id))
    }

    /// The speech whose tool sent the command (`Origin::Tool`).
    pub(crate) fn tool_speech(&self, origin: &Origin) -> Option<SpeechId> {
        match origin {
            Origin::Tool { call_id } => self.activity.tools().speech_of(call_id),
            Origin::Host | Origin::Hook | Origin::Task | Origin::Amd | Origin::Ivr => None,
        }
    }
}

#[cfg(test)]
mod host_answer_tests {
    use super::{ToolAnswer, ToolRaise};
    use lk_agents_abi::llm::tool_executor::RawToolAnswer;

    fn raw(kind: &str, error: Option<&str>, message: Option<&str>, detached_raw: Option<&str>) -> RawToolAnswer {
        RawToolAnswer {
            kind: kind.to_owned(),
            value: None,
            agents: Vec::new(),
            error: error.map(str::to_owned),
            message: message.map(str::to_owned),
            raw: detached_raw.map(str::to_owned),
        }
    }

    // Catches: a kind that reaches this reader and stops there, answered with a default.
    #[test]
    fn every_kind_of_answer_lands_with_its_payload() {
        let mut returned = raw("returned", None, None, None);
        returned.value = Some(serde_json::json!({"temp": 12}));
        assert_eq!(
            ToolAnswer::try_from(returned),
            Ok(ToolAnswer::Returned { value: Some(serde_json::json!({"temp": 12})), agents: Vec::new() })
        );
        assert_eq!(
            ToolAnswer::try_from(raw("raised", Some("tool_error"), Some("nope"), None)),
            Ok(ToolAnswer::Raised(ToolRaise::ToolError("nope".to_owned())))
        );
        assert_eq!(
            ToolAnswer::try_from(raw("raised", Some("stop_response"), None, None)),
            Ok(ToolAnswer::Raised(ToolRaise::StopResponse))
        );
        assert_eq!(ToolAnswer::try_from(raw("raised", Some("internal"), None, None)), Ok(ToolAnswer::Raised(ToolRaise::Internal)));
        assert_eq!(ToolAnswer::try_from(raw("invalid", None, None, None)), Ok(ToolAnswer::Invalid));
        assert_eq!(ToolAnswer::try_from(raw("cancelled", None, None, None)), Ok(ToolAnswer::Cancelled));
        assert_eq!(
            ToolAnswer::try_from(raw("detached", None, Some("The tool has updated: working"), Some("working"))),
            Ok(ToolAnswer::Detached { message: "The tool has updated: working".to_owned(), raw: "working".to_owned() })
        );
    }

    // Catches: a malformed answer read as a plausible one instead of refused.
    #[test]
    fn a_malformed_answer_is_refused_rather_than_guessed() {
        assert!(ToolAnswer::try_from(raw("done", None, None, None)).is_err(), "an unknown kind");
        assert!(ToolAnswer::try_from(raw("raised", None, None, None)).is_err(), "a raise without its kind");
        assert!(ToolAnswer::try_from(raw("raised", Some("tool_error"), None, None)).is_err(), "a ToolError without its message");
        assert!(ToolAnswer::try_from(raw("detached", None, None, None)).is_err(), "an update without its message");
        assert!(ToolAnswer::try_from(raw("detached", None, Some("m"), None)).is_err(), "an update without its raw text");
    }
}
