use crate::agent::{Source, Visibility};
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::Out;
use crate::agent_session::{Driver, FillerSource, HookId, Origin};
use crate::events::{DiagnosticEvent, SessionEvent, ToolEndStatus, ToolLifecycle};
use crate::filler_scheduler::{FillerAction, FillerId, FillerScheduler};
use crate::generation::RunningToolCall;
use crate::trace::{CoreStep, ToolStep};
use lk_agents_abi::{ChatItem, FunctionCall, FunctionCallOutput, MetricsReport, ToolChoice, Ts};
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

pub const MISSING_TOOL_OUTPUT: &str = "the tool executor did not report a result";

/// `make_function_call_output`'s text for an unexpected exception; says nothing about it.
pub const INTERNAL_ERROR_OUTPUT: &str = "An internal error occurred";

/// `make_function_call_output`'s text for a return value the model cannot be told about.
pub const INVALID_OUTPUT: &str = "the tool returned an invalid output";

/// `_RunningTasks` and the deferred reply (`AgentActivity`'s background tool calls), the open
/// filler scopes, and the steps whose `function_tools_executed` the host is handling.
#[derive(Debug, Default)]
pub(crate) struct Tools {
    pub(crate) running: Vec<Running>,
    /// Update pairs batched for the next deferred reply.
    pub(crate) pending: Vec<Pending>,
    pub(crate) reply: Option<Reply>,
    /// A batch handed to the host for rendering, kept until the answer or the bound.
    pub(crate) rendering: Option<Rendering>,
    pub(crate) fillers: BTreeMap<FillerId, FillerRun>,
    /// The steps whose `function_tools_executed` handlers the host runs, until they answer.
    pub(crate) handling: BTreeMap<HookId, Handling>,
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

    /// `speech`'s blocking calls still running, in start order.
    pub(crate) fn live(&self, speech: SpeechId) -> impl Iterator<Item = &Running> {
        self.running.iter().filter(move |r| r.speech == speech && !r.detached())
    }

    /// `speech`'s call `call_id` ended: detached, it stays with its first update counted;
    /// otherwise it leaves the set. `false` when no such call runs. By speech as well as id: a
    /// model may reuse call ids.
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

    /// The detached call named `call_id`, which may send updates.
    pub(crate) fn detached(&mut self, call_id: &str) -> Option<&mut Running> {
        self.running.iter_mut().find(|r| r.call.call_id == call_id && r.detached())
    }

    pub(crate) fn fillers_of(&self, pick: impl Fn(&FillerRun) -> bool) -> Vec<FillerId> {
        self.fillers.iter().filter(|(_, r)| pick(r)).map(|(id, _)| *id).collect()
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
}

#[derive(Debug, Clone)]
pub(crate) struct Running {
    pub(crate) call: FunctionCall,
    /// The placeholder output id, minted once so it keeps one identity across plans.
    pub(crate) output_id: String,
    pub(crate) speech: SpeechId,
    pub(crate) step: u32,
    /// Progress updates sent so far, the first (the step's answer) included.
    pub(crate) updates: u32,
    /// What runs the body: its cancel goes there.
    pub(crate) runner: RunnerRef,
}

impl Running {
    pub(crate) fn detached(&self) -> bool {
        self.updates > 0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Pending {
    /// The agent whose record got the pair; a different agent at delivery re-inserts it.
    pub(crate) agent: String,
    pub(crate) items: Vec<ChatItem>,
    pub(crate) update_id: String,
}

/// The pending updates one deferred reply answers, in arrival order.
#[derive(Debug, Clone)]
pub(crate) struct Batch(pub(crate) Vec<Pending>);

impl Batch {
    /// Every update's id, in order (the reply's `{call_ids}`).
    pub(crate) fn update_ids(&self) -> Vec<String> {
        self.0.iter().map(|p| p.update_id.clone()).collect()
    }

    /// The id of the last item the batch committed.
    pub(crate) fn last_item_id(&self) -> Option<String> {
        self.0.last().and_then(|p| p.items.last()).map(|i| i.id().to_owned())
    }

    /// The items another agent's record got, to re-insert for this request only.
    pub(crate) fn foreign_items(&self, agent: &str) -> Vec<ChatItem> {
        self.0.iter().filter(|p| p.agent != agent).flat_map(|p| p.items.iter().cloned()).collect()
    }
}

/// The deferred reply in flight: its speech and the updates it answers.
#[derive(Debug, Clone)]
pub(crate) struct Reply {
    pub(crate) speech: SpeechId,
    pub(crate) update_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct Rendering {
    pub(crate) hook: HookId,
    pub(crate) batch: Batch,
    pub(crate) at_tail: bool,
}

/// One open `with_filler` scope (`voice/filler_scheduler.py`).
#[derive(Debug)]
pub(crate) struct FillerRun {
    pub(crate) sched: FillerScheduler,
    pub(crate) call_id: String,
    pub(crate) speech: SpeechId,
    pub(crate) source: FillerSource,
    /// The source hook asked and not yet answered.
    pub(crate) asking: Option<HookId>,
    pub(crate) armed: Option<Ts>,
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
    /// The tool returned `value` as JSON.
    Returned {
        value: Option<serde_json::Value>,
    },
    Raised(ToolRaise),
    /// A value the model can't be told about (`_is_valid_function_output`).
    Invalid,
    Cancelled,
    /// `ctx.update(message)`: rendered `message`, `raw` as written; the tool runs on.
    Detached {
        message: String,
        raw: String,
    },
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
        let raw: lk_agents_abi::llm::tool_executor::RawToolAnswer =
            lk_agents_abi::python::extract_generated(&obj.to_owned())?;
        ToolAnswer::try_from(raw).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

impl TryFrom<lk_agents_abi::llm::tool_executor::RawToolAnswer> for ToolAnswer {
    type Error = String;

    /// The marshal's dict as a [`ToolAnswer`]. An unknown kind, or one missing its key, is an
    /// error, never a guess.
    fn try_from(raw: lk_agents_abi::llm::tool_executor::RawToolAnswer) -> Result<Self, String> {
        use lk_agents_abi::llm::tool_executor::{ToolAnswerKind, ToolRaiseKind};
        let kind = ToolAnswerKind::from_wire(&raw.kind)
            .ok_or_else(|| format!("unknown tool answer kind {:?}", raw.kind))?;
        let message = |m: Option<String>| {
            m.ok_or_else(|| format!("a `{}` answer carries a `message`", raw.kind))
        };
        Ok(match kind {
            ToolAnswerKind::Returned => ToolAnswer::Returned { value: raw.value },
            ToolAnswerKind::Raised => {
                let raised = raw.error.as_deref().and_then(ToolRaiseKind::from_wire);
                ToolAnswer::Raised(
                    match raised.ok_or_else(|| format!("unknown raised kind {:?}", raw.error))? {
                        ToolRaiseKind::ToolError => ToolRaise::ToolError(message(raw.message)?),
                        ToolRaiseKind::StopResponse => ToolRaise::StopResponse,
                        ToolRaiseKind::Internal => ToolRaise::Internal,
                    },
                )
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

/// `_start_ivr_detection`'s `send_dtmf_events`. The agent's own tools win on a name clash; kept
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

    /// `_ToolExecutor.cancel`, unconditionally as its `aclose()` does: the body running under
    /// `call_id` is cancelled. Fire-and-forget: the driver has already told the call's end.
    fn cancel(&self, call_id: &str);
}

/// A running call's [`ToolRunner`], kept so its cancel reaches the runner that ran it. Compared by
/// identity.
#[derive(Clone)]
pub struct RunnerRef(pub Arc<dyn ToolRunner>);

impl PartialEq for RunnerRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for RunnerRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToolRunner")
    }
}

/// `voice/tool_executor.py`'s `ToolRunner`: runs one call and reports the marshal's dict. A
/// report, not a decision; [`decide`] owns the output.
#[lk_agents_abi::host(proxy)]
#[async_trait::async_trait]
pub trait HostToolRunner: Send + Sync {
    async fn run(
        &self,
        call: &FunctionCall,
        speech_id: u64,
        step: u32,
    ) -> Result<lk_agents_abi::llm::tool_executor::RawToolAnswer, lk_agents_abi::errors::HostError>;

    fn cancel(&self, call_id: &str);
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

    fn cancel(&self, call_id: &str) {
        self.0.cancel(call_id);
    }
}

/// What one tool produced, decided; [`answer`] turns it into exactly one output.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolOutcome {
    Reply(String),
    /// Empty output, no reply: a `None` return or `StopResponse`.
    Silent,
    Error(String),
    /// A progress update: answered like a reply, and the call is marked non-blocking.
    Detached {
        message: String,
        raw: String,
    },
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
        ToolAnswer::Returned { value } => match value {
            None => ToolOutcome::Silent,
            Some(v) => ToolOutcome::Reply(render(&v)),
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
    let (output, end) = match outcome {
        ToolOutcome::Reply(text) => (output(text, false, true), ToolEnd::Done),
        ToolOutcome::Silent => (output(String::new(), false, false), ToolEnd::Done),
        ToolOutcome::Error(text) => (output(text, true, true), ToolEnd::Error),
        ToolOutcome::Detached { message, raw } => {
            // `RunContext.update`'s marker: a non-blocking tool may wait on its own speech.
            call.extra.insert(
                crate::speech_handle::TOOL_NON_BLOCKING_EXTRA_KEY.to_owned(),
                serde_json::Value::Bool(true),
            );
            raw_update = Some(raw);
            (output(message, false, true), ToolEnd::Detached)
        }
        ToolOutcome::Cancelled => (output(String::new(), false, false), ToolEnd::Cancelled),
        ToolOutcome::Missing => {
            (output(MISSING_TOOL_OUTPUT.to_owned(), true, true), ToolEnd::Error)
        }
    };
    ToolExecution { call, output, end, raw_update }
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
    /// InterruptByAudioActivity-in; finished tools are still in [`StepDecision::commit`].
    Interrupted,
    Reply(ReplyRequest),
}

/// Something the driver should log; a core crate has no logger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepWarning {
    /// The step budget is spent; reported even when no reply follows.
    MaxToolStepsReached,
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
    max_steps_reached: bool,
    /// The second interruption read.
    interrupted: bool,
    previous_user_metrics: Option<MetricsReport>,
}

impl StepDecision {
    /// `function_tools_executed` was handled: the next step from what the handlers left
    /// (`cancel_tool_reply()`, `cancel_agent_handoff()`). An interrupted step stays interrupted.
    pub fn settle(mut self, event: ToolsExecutedEvent) -> StepDecision {
        self.commit = commit_of(&event.function_calls, &event.function_call_outputs);
        if self.next != NextStep::Interrupted {
            self.next = next_of(
                event.has_tool_reply() && !self.interrupted,
                self.max_steps_reached,
                self.previous_user_metrics.clone(),
            );
        }
        self.tools_executed = Some(event);
        self
    }
}

fn next_of(
    reply: bool,
    forbid_tools: bool,
    previous_user_metrics: Option<MetricsReport>,
) -> NextStep {
    if reply {
        NextStep::Reply(ReplyRequest {
            tool_choice: if forbid_tools { ToolChoice::Forbidden } else { ToolChoice::Auto },
            previous_user_metrics,
        })
    } else {
        NextStep::Done
    }
}

#[derive(Debug, Clone)]
pub struct StepInputs<'a> {
    /// The executions, in the model's order.
    pub tool_output: &'a [ToolExecution],
    /// Interrupted before the executions were awaited: [`NextStep::Interrupted`].
    pub interrupted_before_tools: bool,
    /// Re-read after the executions: a interruption during a tool suppresses the reply but not the
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

    // No tool output: nothing at all: no event, no commit, no increment.
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

    for out in inputs.tool_output {
        calls.push(out.call.clone());
        outputs.push(out.output.clone());
    }

    let event = ToolsExecutedEvent {
        function_calls: calls,
        function_call_outputs: outputs,
        handoff_required: false,
    };

    // The *second* interruption read gates the reply.
    StepDecision {
        next: NextStep::Done,
        tools_executed: None,
        commit: Vec::new(),
        warnings,
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
    out: &mut Out,
    id: String,
    call_id: String,
    message: Option<String>,
    status: ToolEndStatus,
) {
    let ended = ToolLifecycle::Ended { id, call_id, message, status };
    out.machine().emit(SessionEvent::ToolExecutionUpdated(ended));
}

fn quiet(next: NextStep) -> StepDecision {
    StepDecision {
        next,
        tools_executed: None,
        commit: Vec::new(),
        warnings: Vec::new(),
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
        out: &mut Out,
    ) {
        let Some(Handling { speech, step, mut event, warnings, verdict }) =
            self.activity.tools.handling.remove(&hook)
        else {
            return;
        };
        if let Some((reply_required, handoff_required)) = left {
            if reply_required.len() == event.function_call_outputs.len() {
                for (out, required) in event.function_call_outputs.iter_mut().zip(reply_required) {
                    out.reply_required = required;
                }
            } else {
                out.trace(CoreStep::PortFailure(
                    "function_tools_executed answered for a different number of outputs",
                ));
            }
            // A handler can cancel the handoff, not ask for one.
            event.handoff_required &= handoff_required;
        }
        let executed = Some(Box::new(event));
        self.finish_tools(speech, step, Vec::new(), executed, warnings, Some(verdict), true, out);
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
        out: &mut Out,
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
                && let Some(h) = self.activity.speech_scheduler.handle(speech)
                && !h.chat_items().iter().any(|i| i.id() == item.id())
            {
                h.item_added(&item);
            }
            self.commit_item(item, source, Visibility::Both, out);
        }
        // `self.scheduling_paused`: the activity that made the speech drains: paused, or left
        // behind by a handoff.
        let draining = self.activity.speech_scheduler.is_paused()
            || self.activity.speech_agent(speech).is_some_and(|f| *f != self.activity.agent);
        // The step's follow-up plans against the activity once this is published.
        self.activity.tools_committed.insert((speech, step), draining);
        self.activity.set_tools_in_flight(speech, false);
        self.publish_activity();
        for w in &warnings {
            // A core crate has no logger, so the host says these (and the trace records them).
            // Only the spent step budget has a sentence in the reference.
            if matches!(w, StepWarning::MaxToolStepsReached) {
                let d = DiagnosticEvent::warning(crate::events::diagnostics::MAX_TOOL_STEPS)
                    .with("speech_id", speech.0.to_string());
                out.machine().emit(SessionEvent::Diagnostic(d));
            }
            out.trace(CoreStep::PortFailure(match w {
                StepWarning::MaxToolStepsReached => {
                    "max_tool_steps reached: the reply is forced to tool_choice=none"
                }
            }));
        }
        out.trace(ToolStep::Finished {
            speech,
            committed: n,
            handoff: executed.as_ref().is_some_and(|e| e.handoff_required),
        });
        let Some(executed) = executed else { return };
        // Not the host's to emit: after the commit, as before any hook existed for it.
        if !hosted {
            out.machine().emit(SessionEvent::FunctionToolsExecuted(executed.clone()));
        }
        if let Some(verdict) = verdict {
            let _ = verdict.send(ToolsVerdict { executed: *executed });
        }
    }

    /// Another live speech is in its tools.
    pub(crate) fn tools_running_elsewhere(&self, speech: SpeechId) -> bool {
        let scheduling = &self.activity.speech_scheduler;
        self.activity
            .tools_in_flight
            .iter()
            .copied()
            .any(|id| id != speech && scheduling.contains(id))
    }

    /// The speech whose tool sent the command (`Origin::Tool`).
    pub(crate) fn tool_speech(&self, origin: &Origin) -> Option<SpeechId> {
        match origin {
            Origin::Tool { call_id } => self.activity.tools.speech_of(call_id),
            Origin::Host | Origin::Hook | Origin::Amd | Origin::Ivr => None,
        }
    }
}

#[cfg(test)]
mod host_answer_tests {
    use super::{ToolAnswer, ToolRaise};
    use lk_agents_abi::llm::tool_executor::RawToolAnswer;

    fn raw(
        kind: &str,
        error: Option<&str>,
        message: Option<&str>,
        detached_raw: Option<&str>,
    ) -> RawToolAnswer {
        RawToolAnswer {
            kind: kind.to_owned(),
            value: None,
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
            Ok(ToolAnswer::Returned { value: Some(serde_json::json!({"temp": 12})) })
        );
        assert_eq!(
            ToolAnswer::try_from(raw("raised", Some("tool_error"), Some("nope"), None)),
            Ok(ToolAnswer::Raised(ToolRaise::ToolError("nope".to_owned())))
        );
        assert_eq!(
            ToolAnswer::try_from(raw("raised", Some("stop_response"), None, None)),
            Ok(ToolAnswer::Raised(ToolRaise::StopResponse))
        );
        assert_eq!(
            ToolAnswer::try_from(raw("raised", Some("internal"), None, None)),
            Ok(ToolAnswer::Raised(ToolRaise::Internal))
        );
        assert_eq!(ToolAnswer::try_from(raw("invalid", None, None, None)), Ok(ToolAnswer::Invalid));
        assert_eq!(
            ToolAnswer::try_from(raw("cancelled", None, None, None)),
            Ok(ToolAnswer::Cancelled)
        );
        assert_eq!(
            ToolAnswer::try_from(raw(
                "detached",
                None,
                Some("The tool has updated: working"),
                Some("working")
            )),
            Ok(ToolAnswer::Detached {
                message: "The tool has updated: working".to_owned(),
                raw: "working".to_owned()
            })
        );
    }

    // Catches: a malformed answer read as a plausible one instead of refused.
    #[test]
    fn a_malformed_answer_is_refused_rather_than_guessed() {
        assert!(ToolAnswer::try_from(raw("done", None, None, None)).is_err(), "an unknown kind");
        assert!(
            ToolAnswer::try_from(raw("raised", None, None, None)).is_err(),
            "a raise without its kind"
        );
        assert!(
            ToolAnswer::try_from(raw("raised", Some("tool_error"), None, None)).is_err(),
            "a ToolError without its message"
        );
        assert!(
            ToolAnswer::try_from(raw("detached", None, None, None)).is_err(),
            "an update without its message"
        );
        assert!(
            ToolAnswer::try_from(raw("detached", None, Some("m"), None)).is_err(),
            "an update without its raw text"
        );
    }
}
