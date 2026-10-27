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
    /// The call an `InputRequired` left open: the person's next message is its output.
    pub(crate) pending_input: Option<Ask>,
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
    pub(crate) runner: Arc<dyn HostToolRunner>,
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
    /// When it last fired.
    pub(crate) fired_at: Option<Ts>,
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
    /// With its `executed` event, as issued.
    pub(crate) tools: StepTools,
    pub(crate) verdict: tokio::sync::oneshot::Sender<ToolsVerdict>,
}

/// A finished tool step, as the driver commits it.
#[derive(Debug)]
pub struct StepTools {
    /// Used when there is no `executed` event; otherwise the event's calls and outputs are.
    pub commit: Vec<ChatItem>,
    /// `function_tools_executed`.
    pub executed: Option<Box<ToolsExecutedEvent>>,
    pub warnings: Vec<StepWarning>,
    pub deliveries: Vec<Delivery>,
}

/// `FunctionCall.extra` key on a call of a tool that requires approval: `asked` when it returned
/// the read-back, `granted` when it ran on the person's answer.
pub const APPROVAL_KEY: &str = "lk.approval";

/// The output of an answered `InputRequired`: the person's words, or, for an approval's
/// read-back, that the call has not run and how to run it (`APPROVAL_ANSWERED`).
pub(crate) fn answered_text(
    record: &crate::agent::SessionChatCtx,
    ask: &Ask,
    text: &str,
) -> String {
    let asked = record.items().any(|item| match item {
        ChatItem::FunctionCall(call) => {
            call.call_id == ask.call_id
                && call.extra.get(APPROVAL_KEY).and_then(|v| v.as_str()) == Some("asked")
        }
        _ => false,
    });
    if !asked {
        return text.to_owned();
    }
    let name = record
        .items()
        .find_map(|item| match item {
            ChatItem::FunctionCall(call) if call.call_id == ask.call_id => Some(call.name.clone()),
            _ => None,
        })
        .unwrap_or_default();
    format!(
        "not run yet: the person answered {} to its read-back. If they agreed, call {name} again \
         with exactly the same arguments to run it; if not, do not, and go on from what they said.",
        python_repr(text)
    )
}

/// `approval_granted`: whether `call` repeats the latest read-back of its tool, with the same
/// arguments, after the person answered it, and no earlier repeat used that answer. A reading of
/// the record by the calls' [`APPROVAL_KEY`] marks: a call that asked something else, such as a
/// missing detail, is neither a read-back nor a run.
pub fn approval_granted(items: &[ChatItem], call: &FunctionCall) -> bool {
    let answered = |call_id: &str| {
        items.iter().any(|i| match i {
            ChatItem::FunctionCallOutput(o) => {
                o.call_id == call_id
                    && !o.output.starts_with(lk_agents_abi::llm::tool_executor::WAITING_PREFIX)
            }
            _ => false,
        })
    };
    let key = canonical_arguments(&call.arguments);
    // The arguments of the latest answered, unused read-back.
    let mut asked: Option<String> = None;
    for item in items {
        let ChatItem::FunctionCall(c) = item else { continue };
        if c.name != call.name {
            continue;
        }
        if c.call_id == call.call_id {
            break;
        }
        match c.extra.get(APPROVAL_KEY).and_then(|v| v.as_str()) {
            Some("granted") if asked.as_deref() == Some(&canonical_arguments(&c.arguments)) => {
                asked = None;
            }
            Some("asked") => {
                asked = answered(&c.call_id).then(|| canonical_arguments(&c.arguments));
            }
            _ => {}
        }
    }
    asked == Some(key)
}

/// Whether `call` runs on an answered read-back ([`approval_granted`]), decided as the model makes
/// it, off the session's approval record: one answer grants one repeat, also to a call made before
/// the granted one is committed.
pub(crate) fn grant(shared: &crate::agent_session::SessionShared, call: &FunctionCall) -> bool {
    let mut record = shared.approvals.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if !approval_granted(&record, call) {
        return false;
    }
    let mut marked = call.clone();
    let mark = serde_json::Value::String(Approval::Granted.as_str().to_owned());
    marked.extra.insert(APPROVAL_KEY.to_owned(), mark);
    record.push(ChatItem::FunctionCall(marked));
    true
}

/// A call is committed with the grant it ran under, however it ended (a cut one included), as
/// `approval_scope` marks it: that answer is spent.
pub(crate) fn mark_granted(record: &[ChatItem], item: &mut ChatItem) {
    let ChatItem::FunctionCall(call) = item else { return };
    if call.extra.contains_key(APPROVAL_KEY) {
        return;
    }
    let granted = record.iter().any(|i| matches!(i, ChatItem::FunctionCall(g) if g.id == call.id));
    if granted {
        let mark = serde_json::Value::String(Approval::Granted.as_str().to_owned());
        call.extra.insert(APPROVAL_KEY.to_owned(), mark);
    }
}

/// The session's approval record (`SessionShared::approvals`): every agent's calls marked
/// `lk.approval`, in order, with their outputs as last committed. What [`grant`] reads, in place of
/// the whole history.
pub(crate) fn note_approval(record: &mut Vec<ChatItem>, item: &ChatItem) {
    let tracked = match item {
        ChatItem::FunctionCall(c) => c.extra.contains_key(APPROVAL_KEY),
        ChatItem::FunctionCallOutput(o) => record.iter().any(|i| match i {
            ChatItem::FunctionCall(c) => c.call_id == o.call_id,
            _ => false,
        }),
        _ => false,
    };
    if !tracked {
        return;
    }
    match record.iter_mut().find(|i| i.id() == item.id()) {
        Some(known) => *known = item.clone(),
        None => record.push(item.clone()),
    }
}

/// Arguments compared as JSON with sorted keys; a string that is not JSON compares as written.
fn canonical_arguments(arguments: &str) -> String {
    fn sorted(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let sorted: std::collections::BTreeMap<_, _> =
                    map.into_iter().map(|(k, v)| (k, sorted(v))).collect();
                serde_json::Value::Object(sorted.into_iter().collect())
            }
            serde_json::Value::Array(a) => {
                serde_json::Value::Array(a.into_iter().map(sorted).collect())
            }
            other => other,
        }
    }
    let arguments = if arguments.is_empty() { "{}" } else { arguments };
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(v) => sorted(v).to_string(),
        Err(_) => arguments.to_owned(),
    }
}

/// `repr(str)`, as the reference's `{answer!r}` writes it.
fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
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
    report: ToolReport,
    wall: std::time::SystemTime,
    mut mint: impl FnMut() -> String,
) -> ToolExecution {
    let call = FunctionCall {
        id: mint(),
        call_id: id.to_owned(),
        name: running.call.name.clone(),
        arguments: running.call.arguments.clone(),
        created_at: wall,
        extra: running.call.extra.clone(),
        group_id: None,
    };
    let mut execution = answer(&call, report, wall);
    execution.output.id = mint();
    if let Some(Delivery::Ask(ask)) = &mut execution.delivery {
        ask.output_id = execution.output.id.clone();
    }
    execution
}

pub use lk_agents_abi::llm::tool_executor::{Approval, ToolOutcome, ToolReport};

/// `make_function_call_output`'s answer for a classified outcome, `(output, is_error,
/// reply_required)`: the table a call reads too.
#[lk_agents_abi::host(export(pid = false))]
pub fn tool_call_output(outcome: ToolOutcome) -> (String, bool, bool) {
    let lk_agents_abi::llm::tool_executor::ToolShape { text, is_error, reply_required } =
        outcome.shape();
    (text, is_error, reply_required)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolExecution {
    pub call: FunctionCall,
    pub output: FunctionCallOutput,
    /// How the run ended; a detached call is still running after its output is committed.
    pub end: ToolEnd,
    /// The first update as the tool wrote it (`ToolCallUpdated.message`), for a detached call.
    pub raw_update: Option<String>,
    /// What the person gets because of this call, once its output is committed.
    pub delivery: Option<Delivery>,
}

/// What reaches the person because of a call's outcome (the reference's verbatim `say()` and
/// `_apply_transition`), once the call's output is committed.
#[derive(Debug, Clone, PartialEq)]
pub enum Delivery {
    /// A verbatim result: said as written, the model reads that it was delivered.
    Say(String),
    /// `InputRequired`: the question asked, and the person's next message written as the output
    /// `output_id` of the call `call_id` (the approval's read-back included).
    Ask(Ask),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ask {
    pub call_id: String,
    pub output_id: String,
    pub question: String,
    pub verbatim: bool,
    /// `InputRequired(tool=)`: the tool the answer goes to.
    pub tool: Option<String>,
    /// The `InputRequired` object, the output of a run ending on the asking speech.
    pub transition: Option<lk_agents_abi::HostRef>,
}

/// How a tool's run ended (`ToolCallEnded{status}`), plus `Detached`, which is not an end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEnd {
    /// A value, a transition, a silent return: `status: "done"`.
    Done,
    /// An exception the model reads: `status: "error"`.
    Error,
    /// The host cancelled the tool: `status: "cancelled"`, an empty output, no reply.
    Cancelled,
    /// `ctx.update()`: the step is answered and the tool keeps running.
    Detached,
}

impl ToolEnd {
    fn of(outcome: &ToolOutcome) -> ToolEnd {
        match outcome {
            ToolOutcome::Update { .. } => ToolEnd::Detached,
            ToolOutcome::Cancelled => ToolEnd::Cancelled,
            ToolOutcome::ToolError(_) | ToolOutcome::Internal | ToolOutcome::Invalid => {
                ToolEnd::Error
            }
            ToolOutcome::Value { .. }
            | ToolOutcome::Nothing
            | ToolOutcome::StopResponse
            | ToolOutcome::Completed { .. }
            | ToolOutcome::Failed(_)
            | ToolOutcome::Rejected(_)
            | ToolOutcome::InputRequired { .. }
            | ToolOutcome::Working(_) => ToolEnd::Done,
        }
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
    /// `None` on the session's own additions: the agent's runner runs them.
    pub runner: Option<Arc<dyn HostToolRunner>>,
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

/// `voice/tool_executor.py`'s `ToolRunner`, the only port on the generation path: runs one call
/// and classifies what the body returned or raised. A report, not a decision; [`answer`] owns
/// the output.
#[lk_agents_abi::host(proxy)]
#[async_trait::async_trait]
pub trait HostToolRunner: Send + Sync {
    /// `speech` is the call's speech as the host's `SpeechHandle` sees it (`_core.SpeechHandle`),
    /// whose `host` is the host's own object once it made one. `granted`: the call repeats a
    /// read-back the person answered ([`approval_granted`]), so a tool that requires approval runs.
    async fn run(
        &self,
        call: &FunctionCall,
        speech: &lk_agents_abi::CoreObject,
        granted: bool,
    ) -> Result<ToolReport, lk_agents_abi::errors::HostError>;

    /// `_ToolExecutor.cancel`, unconditionally as its `aclose()` does: the body running under
    /// `call_id` is cancelled. Fire-and-forget: the driver has already told the call's end.
    fn cancel(&self, call_id: &str);
}

impl dyn HostToolRunner {
    /// Resolves when the tool returns, raises, is cancelled, or releases the step with its first
    /// `ctx.update()`. Every call gets exactly one answer: a runner that raises or reports an
    /// unreadable outcome is answered as the reference's internal error, and logged.
    pub async fn answer(
        &self,
        call: &FunctionCall,
        speech: &lk_agents_abi::CoreObject,
        granted: bool,
    ) -> ToolReport {
        self.run(call, speech, granted).await.unwrap_or_else(|e| {
            log::error!(target: "livekit.agents", function = call.name.as_str(), call_id = call.call_id.as_str();
                "the tool executor failed outside the tool's own error handling: {e}");
            ToolOutcome::Internal.into()
        })
    }
}

impl std::fmt::Debug for dyn HostToolRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HostToolRunner")
    }
}

/// The one formatter: every committed `FunctionCallOutput` comes through here, from the one
/// decision table (`ToolOutcome::shape`), a verbatim result rewritten as delivered.
pub fn answer(call: &FunctionCall, report: ToolReport, now: SystemTime) -> ToolExecution {
    let ToolReport { outcome, approval, transition } = report;
    let shape = outcome.shape();
    let mut output = FunctionCallOutput::new(
        format!("{}_out", call.id),
        call.call_id.clone(),
        outcome.delivered().unwrap_or(shape.text),
        shape.is_error,
        now,
    );
    output.name = call.name.clone();
    output.reply_required = shape.reply_required;
    let mut call = call.clone();
    if let Some(approval) = approval {
        let mark = serde_json::Value::String(approval.as_str().to_owned());
        call.extra.insert(APPROVAL_KEY.to_owned(), mark);
    }
    let end = ToolEnd::of(&outcome);
    let delivery = match outcome {
        ToolOutcome::Update { raw, .. } => {
            // `RunContext.update`'s marker: a non-blocking tool may wait on its own speech.
            call.extra.insert(
                crate::speech_handle::TOOL_NON_BLOCKING_EXTRA_KEY.to_owned(),
                serde_json::Value::Bool(true),
            );
            return ToolExecution { call, output, end, raw_update: Some(raw), delivery: None };
        }
        ToolOutcome::InputRequired { question, verbatim, tool } => Some(Delivery::Ask(Ask {
            call_id: call.call_id.clone(),
            output_id: output.id.clone(),
            question,
            verbatim,
            tool,
            transition,
        })),
        other => other.verbatim().map(|text| Delivery::Say(text.to_owned())),
    };
    ToolExecution { call, output, end, raw_update: None, delivery }
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
    /// What the calls owe the person, delivered once the commit is made.
    pub deliveries: Vec<Delivery>,
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
        deliveries: inputs.tool_output.iter().filter_map(|o| o.delivery.clone()).collect(),
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
        deliveries: Vec::new(),
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
        // A verbatim result was the person's as the call returned; a question is not asked of
        // a person who has moved on.
        deliveries: tool_output
            .iter()
            .filter_map(|o| o.delivery.clone())
            .filter(|d| matches!(d, Delivery::Say(_)))
            .collect(),
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
        let Some(Handling { speech, step, mut tools, verdict }) =
            self.activity.tools.handling.remove(&hook)
        else {
            return;
        };
        if let (Some((reply_required, handoff_required)), Some(event)) = (left, &mut tools.executed)
        {
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
        self.finish_tools(speech, step, tools, Some(verdict), true, out);
    }

    /// A step's tools are over and `function_tools_executed` handled: the results committed with
    /// whether the activity drains, what the calls owe the person delivered, and the waiting step
    /// (`verdict`) told what the handlers left. `hosted`: the host emitted the event itself.
    pub(crate) fn finish_tools(
        &mut self,
        speech: SpeechId,
        step: u32,
        tools: StepTools,
        verdict: Option<tokio::sync::oneshot::Sender<ToolsVerdict>>,
        hosted: bool,
        out: &mut Out,
    ) {
        let StepTools { commit, executed, warnings, deliveries } = tools;
        // What the handlers left is what is committed: calls, then outputs.
        let commit = match &executed {
            Some(e) => crate::tool_executor::commit_of(&e.function_calls, &e.function_call_outputs),
            None => commit,
        };
        // On a realtime session the outputs go to the server with the commit (the model's context
        // lives there; the call is the server's own), before anything this commit lets go -- a
        // reply that waited on these tools, the follow-up -- asks the server for a reply that must
        // read them. Pushed while closing too (the server holds the call open until answered);
        // only an ended socket takes nothing.
        if let Some(rt) = &self.activity.rt_session
            && !self.activity.realtime_state.ended()
        {
            let auto = rt.capabilities.auto_tool_reply_generation;
            for item in commit.iter().filter(|i| matches!(i, ChatItem::FunctionCallOutput(_))) {
                let push = lk_agents_abi::llm::realtime::PushItem {
                    item: Box::new(item.clone()),
                    reply_required: !auto,
                };
                self.rt_send(lk_agents_abi::RealtimeCommand::PushItem(push), out);
            }
        }
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
        // Before the follow-up is let go, so a verbatim result is heard before the reply to it.
        for delivery in deliveries {
            self.deliver_outcome(self.step.now, speech, delivery, out);
        }
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
