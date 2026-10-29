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

pub const MISSING_TOOL_OUTPUT: &str = "the tool executor did not report a result";

/// `_RunningTasks` and the deferred reply (`AgentActivity`'s background tool calls), the open
/// filler scopes, and the steps whose `function_tools_executed` the host is handling.
#[derive(Debug, Default)]
pub(crate) struct Tools {
    pub(crate) running: Vec<Running>,
    /// Update pairs batched for the next deferred reply.
    pub(crate) pending: Vec<Pending>,
    pub(crate) reply: Option<Reply>,
    pub(crate) fillers: BTreeMap<FillerId, FillerRun>,
    /// The steps whose `function_tools_executed` handlers the host runs, until they answer.
    pub(crate) handling: BTreeMap<HookId, Handling>,
    /// `InputRequired(tool=)`: the person's next turn is a call of this tool.
    pub(crate) claim: Option<Claim>,
    /// Each live model reply's `AgentRun`, which decides its steps.
    pub(crate) replies: BTreeMap<SpeechId, crate::agent_activity::reply_run::ReplyRun>,
    /// The waiting calls the person's latest message answered: the only read-backs a reply
    /// started since can consume.
    pub(crate) approval_answers: std::collections::BTreeSet<String>,
    /// Mints the hosted callbacks' operations, unique in the session.
    pub(crate) next_operation: u64,
    /// Calls made before their step was authorized to play, in the model's order: started once
    /// it is and their run decided their first move, dropped unrun if the speech is interrupted
    /// or done first.
    pub(crate) deferred: Vec<Deferred>,
    /// The calls `tool_choice="none"` dropped, by step, reported when the step's tools finish.
    pub(crate) dropped: BTreeMap<(SpeechId, u32), Vec<String>>,
}

/// A call the model made, as the reply task saw it: what the step offered and how it may call.
#[derive(Debug)]
pub struct MadeCall {
    pub call: FunctionCall,
    /// The step's offered tool of the call's name.
    pub spec: Option<lk_agents_abi::ToolSpec>,
    /// The step forbade tools (`tool_choice="none"`).
    pub forbidden: bool,
    /// The step answers an `InputRequired(tool=)`.
    pub answering: Option<Answering>,
    /// The reply's agent, which hosts its offered tools.
    pub agent: Option<Arc<dyn AgentFrame>>,
}

/// An admitted call waiting for its step's authorization.
#[derive(Debug)]
pub(crate) struct Deferred {
    pub(crate) speech: SpeechId,
    pub(crate) step: u32,
    /// Its id in the reply's run.
    pub(crate) id: CallId,
    pub(crate) call: FunctionCall,
    pub(crate) advertised: bool,
    pub(crate) agent: Option<Arc<dyn AgentFrame>>,
    pub(crate) go: crate::agent_session::RequestId,
}

pub use lk_agents_core::agent_run::CallId;

/// The next move of a call's task, decided by its reply's run.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolGo {
    /// The call starts as `item` (stamped and admitted) with its `first` move; hosted by the
    /// reply's agent unless it names a tool the request did not offer.
    Start { call: CallId, item: Box<FunctionCall>, host: Option<HostedTool>, first: Box<ToolGo> },
    /// Bind and validate these arguments (the framework's parse, with the schema's defaults).
    Prepare { call: CallId, arguments: serde_json::Map<String, serde_json::Value> },
    /// Render the read-back of the prepared arguments.
    ReadBack { call: CallId },
    /// Run the body; `item` carries the call's approval marks.
    Execute { item: Box<FunctionCall> },
    /// Settled without its body: `item` as recorded.
    Settled { item: Box<FunctionCall>, outcome: ToolOutcome },
    /// The call does not run: dropped unrun before it started (`tool_choice="none"`, a speech
    /// interrupted or done before its step was authorized), or the step's calls are over.
    Gone,
}

/// The tool an `InputRequired(tool=)` claimed the person's next turn for; once `after` (the speech
/// asking in the agent's own words) is done, so the claim does not steer that reply.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Claim {
    pub(crate) tool: String,
    pub(crate) after: Option<SpeechId>,
}

/// A reply that answers an `InputRequired(tool=)`: its call of `tool` carries what the person
/// said, as they said it, in place of the model's arguments (`_answer_arguments`).
#[derive(Debug, Clone, PartialEq)]
pub struct Answering {
    pub tool: String,
    pub said: String,
}

impl Tools {
    /// A new hosted callback's operation.
    pub(crate) fn mint_operation(&mut self) -> u64 {
        self.next_operation += 1;
        self.next_operation
    }

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

    /// The running call named `call_id` not yet heard to detach: its updates wait for it.
    pub(crate) fn undetached(&mut self, call_id: &str) -> Option<&mut Running> {
        self.running.iter_mut().find(|r| r.call.call_id == call_id && !r.detached())
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
    /// Updates sent before the step's join heard the call detach (`ctx.update()` reaches the
    /// driver directly, the detach in the model's order): applied, in order, once it has.
    pub(crate) early: Vec<crate::agent_session::ToolUpdate>,
    /// What runs the body: its cancel goes there.
    pub(crate) hosted: HostedTool,
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
    /// With its `tools_executed` event, as issued.
    pub(crate) decision: StepDecision,
}

/// A step's tools as the reply task awaited them: the executions in the model's order and the
/// speech's reads around them. The driver decides the step from it ([`next_step`]).
#[derive(Debug)]
pub struct StepEnd {
    pub executions: Vec<ToolExecution>,
    /// Read before the executions were awaited.
    pub interrupted_before_tools: bool,
    /// Read after them.
    pub interrupted: bool,
    pub forwarded_text_empty: bool,
    pub max_tool_steps: u32,
    pub user_metrics: Option<MetricsReport>,
}

/// What a model step leaves the reply task, as the reply's run decided it.
#[derive(Debug)]
pub enum StepNext {
    /// The reply is over.
    Done,
    /// The reply is over, cut by the person.
    Interrupted,
    /// The reply goes on with this step, its words a new assistant message (`message_id`).
    Reply(Box<NextReply>),
}

/// A reply's next model step.
#[derive(Debug)]
pub struct NextReply {
    pub planned: crate::generation::PlannedReply,
    pub message_id: String,
    /// Carried over when the step before forwarded no text, so `e2e_latency` can still be
    /// computed.
    pub previous_user_metrics: Option<MetricsReport>,
}

/// `FunctionCall.extra` key on a call of a tool that requires approval: `asked` when it returned
/// the read-back, `granted` when it ran on the person's answer.
pub use lk_agents_core::approval::APPROVAL_KEY;

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
    execution
}

pub use lk_agents_abi::llm::tool_executor::{ToolOutcome, ToolReport};

/// `make_function_call_output`'s answer for a classified outcome, `(output, is_error,
/// reply_required)`: the table a call reads too.
#[lk_agents_abi::host(export(pid = false))]
pub fn tool_call_output(outcome: ToolOutcome) -> (String, bool, bool) {
    let lk_agents_core::tool_outcome::ToolShape { text, is_error, reply_required } =
        lk_agents_core::tool_outcome::shape(&outcome, lk_agents_core::tool_outcome::Mode::Voice);
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

impl ToolExecution {
    /// The lifecycle event's text: nothing for a silent return, handoff or cancellation.
    pub fn ended_message(&self) -> Option<String> {
        match self.end {
            ToolEnd::Done => (!self.output.output.is_empty()).then(|| self.output.output.clone()),
            ToolEnd::Error => Some(self.output.output.clone()),
            ToolEnd::Detached => self.raw_update.clone(),
            ToolEnd::Cancelled => None,
        }
    }
}

/// What reaches the person because of a call's outcome (the reference's verbatim `say()` and
/// `_apply_transition`), once the call's output is committed.
#[derive(Debug, Clone, PartialEq)]
pub enum Delivery {
    /// A verbatim result: said as written, the model reads that it was delivered.
    Say(String),
    /// `InputRequired`: the question asked; the person's next message is written as the output
    /// of every call left waiting (the approval's read-back included).
    Ask(Ask),
    /// A VoiceTask tool's validated `Completed(value)`: the task ends with the value, and its
    /// agent says nothing more about it.
    Complete(lk_agents_abi::HostRef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ask {
    pub question: String,
    pub verbatim: bool,
    /// `InputRequired(tool=)`: the tool the answer goes to.
    pub tool: Option<String>,
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
            ToolOutcome::ToolError(_)
            | ToolOutcome::ModelRetry(_)
            | ToolOutcome::Internal
            | ToolOutcome::Invalid => ToolEnd::Error,
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
    /// The agent hosting the tools; `None` on the session's own additions, which the agent's
    /// host runs.
    pub agent: Option<Arc<dyn AgentFrame>>,
}

impl std::fmt::Debug for Toolset {
    /// Hand-written: `host` cannot derive it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Toolset")
            .field("tools", &self.specs.len())
            .field("host", &self.host.is_some())
            .field("agent", &self.agent)
            .finish()
    }
}

pub use lk_agents_abi::agent::{AgentFrame, FunctionToolInfo};

/// The [`AgentFrame`] a host builds (`_core.VoiceAgent`): the parts as given.
pub struct VoiceAgentFrame {
    definition: lk_agents_abi::agent::AgentDefinition,
    host: Arc<dyn lk_agents_abi::agent::AgentHost>,
    host_context: lk_agents_abi::HostRef,
}

impl AgentFrame for VoiceAgentFrame {
    fn definition(&self) -> &lk_agents_abi::agent::AgentDefinition {
        &self.definition
    }

    fn host(&self) -> &Arc<dyn lk_agents_abi::agent::AgentHost> {
        &self.host
    }

    fn host_context(&self) -> &lk_agents_abi::HostRef {
        &self.host_context
    }
}

impl lk_agents_abi::agent::VoiceAgent for VoiceAgentFrame {
    fn new(
        definition: lk_agents_abi::agent::AgentDefinition,
        host: Arc<dyn lk_agents_abi::agent::AgentHost>,
        host_context: lk_agents_abi::HostRef,
    ) -> Self {
        VoiceAgentFrame { definition, host, host_context }
    }
}

/// A hosted call: the host running its callbacks and the invocation they run in, which its cancel
/// names.
#[derive(Clone)]
pub struct HostedTool {
    pub host: Arc<dyn lk_agents_abi::agent::AgentHost>,
    pub invocation: lk_agents_abi::agent::Invocation,
}

impl HostedTool {
    /// A callback of `speech` (for `call`, or on its output), the `operation`th of its session,
    /// hosted by `agent`: the invocation of one callback on a voice reply, which
    /// `{session}/{speech}` names.
    pub fn of(
        agent: &Arc<dyn AgentFrame>,
        session: &str,
        speech: &Arc<crate::speech_handle::SpeechHandle>,
        operation: u64,
        call: Option<&FunctionCall>,
    ) -> HostedTool {
        let invocation = lk_agents_abi::agent::Invocation {
            operation,
            invocation_id: format!("{session}/{}", speech.speech_id()),
            agent_name: agent.definition().name.clone(),
            execution_id: None,
            frame_id: None,
            context_scope: None,
            application_context: None,
            history: Vec::new(),
            function_call: call.cloned(),
            approval: call.and_then(lk_agents_core::approval::mark),
            speech: Some(speech.object()),
            host_context: Some(agent.host_context().clone()),
        };
        HostedTool { host: Arc::clone(agent.host()), invocation }
    }

    /// Cancels the body running under this invocation.
    pub fn cancel(&self) {
        self.host.cancel(&self.invocation);
    }
}

impl std::fmt::Debug for HostedTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostedTool")
            .field("invocation_id", &self.invocation.invocation_id)
            .field("operation", &self.invocation.operation)
            .finish()
    }
}

/// One invocation: the same reply's same callback.
impl PartialEq for HostedTool {
    fn eq(&self, other: &Self) -> bool {
        self.invocation.invocation_id == other.invocation.invocation_id
            && self.invocation.operation == other.invocation.operation
    }
}

/// The one formatter: every committed `FunctionCallOutput` comes through here, from the one
/// decision table (`lk_agents_core::tool_outcome`), a verbatim result rewritten as delivered.
pub fn answer(call: &FunctionCall, report: ToolReport, now: SystemTime) -> ToolExecution {
    let ToolReport { outcome, transition } = report;
    let shape =
        lk_agents_core::tool_outcome::shape(&outcome, lk_agents_core::tool_outcome::Mode::Voice);
    let mut output = FunctionCallOutput::new(
        format!("{}_out", call.id),
        call.call_id.clone(),
        lk_agents_core::tool_outcome::delivered(&outcome).unwrap_or(shape.text),
        shape.is_error,
        now,
    );
    output.name = call.name.clone();
    output.reply_required = shape.reply_required;
    let mut call = call.clone();
    let end = ToolEnd::of(&outcome);
    // A completion the host validated for a VoiceTask asks for no reply: the parent gets the floor.
    if let (ToolOutcome::Completed { .. }, Some(value)) = (&outcome, &transition) {
        output.reply_required = false;
        let delivery = Some(Delivery::Complete(value.clone()));
        return ToolExecution { call, output, end, raw_update: None, delivery };
    }
    let delivery = match outcome {
        ToolOutcome::Update { raw, .. } => {
            // `RunContext.update`'s marker: a non-blocking tool may wait on its own speech.
            call.extra.insert(
                crate::speech_handle::TOOL_NON_BLOCKING_EXTRA_KEY.to_owned(),
                serde_json::Value::Bool(true),
            );
            let raw_update = Some(raw);
            return ToolExecution { call, output, end, raw_update, delivery: None };
        }
        ToolOutcome::InputRequired { question, verbatim, tool } => {
            Some(Delivery::Ask(Ask { question, verbatim, tool }))
        }
        other => lk_agents_core::tool_outcome::verbatim(&other)
            .map(|text| Delivery::Say(text.to_owned())),
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
    pub(crate) interrupted: bool,
    /// A call completed a VoiceTask: the step has no follow-up, whatever the outputs ask.
    completes: bool,
    previous_user_metrics: Option<MetricsReport>,
}

impl StepDecision {
    /// `function_tools_executed` was handled: the next step from what the handlers left
    /// (`cancel_tool_reply()`, `cancel_agent_handoff()`). An interrupted step stays interrupted.
    pub fn settle(mut self, event: ToolsExecutedEvent) -> StepDecision {
        self.commit = commit_of(&event.function_calls, &event.function_call_outputs);
        if self.next != NextStep::Interrupted {
            self.next = next_of(
                event.has_tool_reply() && !self.interrupted && !self.completes,
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
        completes: (inputs.tool_output.iter())
            .any(|o| matches!(o.delivery, Some(Delivery::Complete(_)))),
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
        completes: false,
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
        // A verbatim result was the person's as the call returned, as is a VoiceTask's validated
        // completion; a question is not asked of a person who has moved on.
        deliveries: tool_output
            .iter()
            .filter_map(|o| o.delivery.clone())
            .filter(|d| matches!(d, Delivery::Say(_) | Delivery::Complete(_)))
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
    /// A model step's calls are over: the step is decided on what they returned, after the
    /// host's `function_tools_executed` handlers when it runs any, and the reply's run decides
    /// what follows. Calls `tool_choice="none"` dropped are reported. `answer`: the reply task
    /// waiting for its next step.
    pub(crate) fn on_tools_finished(
        &mut self,
        now: Ts,
        (speech, step): (SpeechId, u32),
        end: StepEnd,
        answer: Option<crate::agent_session::RequestId>,
        out: &mut Out,
    ) {
        for name in self.activity.tools.dropped.remove(&(speech, step)).unwrap_or_default() {
            let diagnostic = DiagnosticEvent::error(crate::events::diagnostics::TOOL_CHOICE_NONE)
                .with("function", name)
                .with("speech_id", speech.0.to_string());
            out.machine().emit(SessionEvent::Diagnostic(diagnostic));
        }
        if !self.reply_step_ended(speech, answer, out) {
            return;
        }
        let decision = next_step(StepInputs {
            tool_output: &end.executions,
            interrupted_before_tools: end.interrupted_before_tools,
            interrupted: end.interrupted,
            num_steps: step,
            max_tool_steps: end.max_tool_steps,
            forwarded_text_empty: end.forwarded_text_empty,
            user_metrics: end.user_metrics,
        });
        // With a host running hooks, `function_tools_executed` is its to emit, and the step waits
        // for what the handlers leave.
        if let Some(event) = &decision.tools_executed
            && self.activity.hook_bound.is_some()
        {
            let id = self.activity.hooks.mint();
            let event = Box::new(event.clone());
            let call = crate::agent_session::HookCall::FunctionToolsExecuted { id, speech, event };
            let wait = self.hook_wait();
            if self.issue_hook(now, call, Some(wait), out) {
                let handling = Handling { speech, step, decision };
                self.activity.tools.handling.insert(id, handling);
                return;
            }
        }
        self.finish_tools(speech, step, decision, false, out);
    }

    /// The host's `function_tools_executed` handlers answered under `hook` (`left`: each output's
    /// `reply_required`), or will not (`None`): the step goes on from what they left
    /// (`cancel_tool_reply()`, `cancel_agent_handoff()`).
    pub(crate) fn on_tools_executed_handled(
        &mut self,
        hook: crate::agent_session::HookId,
        left: Option<Vec<bool>>,
        out: &mut Out,
    ) {
        let Some(Handling { speech, step, mut decision }) =
            self.activity.tools.handling.remove(&hook)
        else {
            return;
        };
        if let (Some(reply_required), Some(mut event)) = (left, decision.tools_executed.clone()) {
            if reply_required.len() == event.function_call_outputs.len() {
                for (out, required) in event.function_call_outputs.iter_mut().zip(reply_required) {
                    out.reply_required = required;
                }
                decision = decision.settle(event);
            } else {
                out.trace(CoreStep::PortFailure(
                    "function_tools_executed answered for a different number of outputs",
                ));
            }
        }
        self.finish_tools(speech, step, decision, true, out);
    }

    /// A step's tools are decided and `function_tools_executed` handled: the results committed,
    /// what the calls owe the person delivered, and the reply's run given the floor's verdict.
    /// `hosted`: the host emitted the event itself.
    fn finish_tools(
        &mut self,
        speech: SpeechId,
        step: u32,
        decision: StepDecision,
        hosted: bool,
        out: &mut Out,
    ) {
        let StepDecision {
            next,
            tools_executed: executed,
            commit,
            warnings,
            deliveries,
            interrupted,
            ..
        } = decision;
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
        self.activity.tools_committed.insert(speech);
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
        out.trace(ToolStep::Finished { speech, committed: n });
        // Not the host's to emit: after the commit, as before any hook existed for it.
        if let Some(executed) = executed
            && !hosted
        {
            out.machine().emit(SessionEvent::FunctionToolsExecuted(Box::new(executed)));
        }
        // The follow-up plans against the activity as published here.
        self.reply_step_decided(speech, next, interrupted, out);
    }

    /// Another live speech is in its tools. One awaiting a VoiceTask waits on the person, through
    /// the task: the agent listens, not thinks.
    pub(crate) fn tools_running_elsewhere(&self, speech: SpeechId) -> bool {
        let scheduling = &self.activity.speech_scheduler;
        let tasks = &self.activity.voice_tasks;
        self.activity
            .tools_in_flight
            .iter()
            .copied()
            .any(|id| id != speech && scheduling.contains(id) && !tasks.awaits(id))
    }

    /// The speech whose tool sent the command (`Origin::Tool`).
    pub(crate) fn tool_speech(&self, origin: &Origin) -> Option<SpeechId> {
        match origin {
            Origin::Tool { call_id } => self.activity.tools.speech_of(call_id),
            Origin::Host | Origin::Hook | Origin::Amd | Origin::Ivr => None,
        }
    }
}
