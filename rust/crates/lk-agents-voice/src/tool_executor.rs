use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{FillerSource, HookId};
use crate::events::{SessionEvent, ToolEndStatus, ToolLifecycle};
use crate::filler_scheduler::{FillerAction, FillerId, FillerScheduler};
use crate::generation::RunningToolCall;
use lk_agents_abi::{ChatItem, FunctionCall, FunctionCallOutput, MetricsReport, Ts};
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
    pub(crate) reply: Option<ToolReply>,
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
    /// Calls made before their step was authorized to play, in the model's order: joined to
    /// their reply's run and started once it is, dropped unrun if the speech is interrupted or
    /// done first.
    pub(crate) deferred: Vec<Deferred>,
    /// The calls `tool_choice="none"` dropped, by step, reported when the step's tools finish.
    pub(crate) dropped: BTreeMap<(SpeechId, u32), Vec<String>>,
    /// The agent turns of speeches that ended while calls they started still run: the last of
    /// those calls to end ends the turn.
    pub(crate) turns: Vec<crate::telemetry::AgentTurnSpan>,
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
    /// Its id in the reply's run, once it joined it.
    pub(crate) id: Option<CallId>,
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
    Start {
        call: CallId,
        item: Box<FunctionCall>,
        host: Option<Box<HostedTool>>,
        first: Box<ToolGo>,
    },
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

    /// The speech a running call belongs to (`VoiceCommandOrigin::Tool{call_id}` -> its speech).
    pub(crate) fn speech_of(
        &self,
        invocation_id: &str,
        operation: u64,
        call_id: &str,
    ) -> Option<SpeechId> {
        self.running
            .iter()
            .find(|running| running.owns_update(invocation_id, operation, call_id))
            .map(|running| running.speech)
    }

    /// `speech`'s blocking calls still running, in start order.
    pub(crate) fn live(&self, speech: SpeechId) -> impl Iterator<Item = &Running> {
        self.running.iter().filter(move |r| r.speech == speech && !r.detached())
    }

    /// This speech's admitted call ended: detached, it stays with its first update counted;
    /// otherwise it leaves the set. Model IDs never select the execution that is removed.
    pub(crate) fn end(&mut self, speech: SpeechId, call: Option<CallId>, detached: bool) -> bool {
        let Some(call) = call else { return false };
        let Some(i) =
            self.running.iter().position(|r| r.speech == speech && r.run_call == Some(call))
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
    /// The call in this speech's AgentRun, retained after that run enters cleanup. Model call
    /// IDs may be reused; cancellation must address this admitted execution instead.
    pub(crate) run_call: Option<CallId>,
    /// Validated argument identity used by the shared duplicate policy across live replies.
    pub(crate) identity: Option<lk_agents_abi::agent::ArgumentIdentity>,
    /// The shared policy admitted the body, rather than only preparing arguments/read-back.
    pub(crate) executing: bool,
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

impl Tools {
    pub(crate) fn update_owner(
        &mut self,
        invocation_id: &str,
        operation: u64,
        call_id: &str,
    ) -> Option<&mut Running> {
        self.running
            .iter_mut()
            .find(|running| running.owns_update(invocation_id, operation, call_id))
    }

    pub(crate) fn take_update_owner(
        &mut self,
        invocation_id: &str,
        operation: u64,
        call_id: &str,
    ) -> Option<Running> {
        let index = self
            .running
            .iter()
            .position(|running| running.owns_update(invocation_id, operation, call_id))?;
        Some(self.running.remove(index))
    }
}

impl Running {
    fn owns_update(&self, invocation_id: &str, operation: u64, call_id: &str) -> bool {
        self.call.call_id == call_id
            && self.hosted.invocation.invocation_id == invocation_id
            && self.hosted.invocation.operation == operation
    }

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
    /// The call's host: the deferred reply's instructions are its templates.
    pub(crate) hosted: HostedTool,
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

/// The deferred reply in flight.
#[derive(Debug, Clone)]
pub(crate) enum ToolReply {
    /// The host renders its instructions (`render` names the rendering).
    Rendering { render: u64, batch: Batch, at_tail: bool },
    /// Spoken by `speech`, answering the updates `update_ids`.
    Speaking { speech: SpeechId, update_ids: Vec<String> },
}

/// What a `tool_handling` template the host renders is for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rendering {
    /// The refusal of a duplicate call of `speech`'s run.
    Refusal { speech: SpeechId, call: CallId },
    /// The deferred reply's instructions.
    ToolReply { render: u64 },
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
    pub(crate) end: StepEnd,
    /// Canonical shared-run history, in the order supplied by AgentRun.
    pub(crate) items: Vec<ChatItem>,
    pub(crate) terminal_call: Option<String>,
    pub(crate) event: ToolsExecutedEvent,
}

/// A step's tools as the reply task awaited them: the executions in the model's order and the
/// speech's reads around them. AgentRun owns the outcome; voice supplies its constraints.
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
    /// A VoiceTask tool's `Failed(reason)`: the task ends, its awaiter raising the reason.
    Fail(String),
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
    /// The installed agent provides both execution and policy metadata. Required when any
    /// tools are offered; setup installs discovered tool metadata into this frame.
    pub agent: Option<Arc<dyn AgentFrame>>,
}

impl Toolset {
    /// A session advertising a tool with nothing to run it is refused at its start, not at the
    /// first call.
    pub fn refuse_unrunnable(&self) -> Result<(), crate::agent_session::DriverError> {
        if !self.specs.is_empty() && self.agent.is_none() {
            return Err(crate::agent_session::DriverError::Missing("voice_agent"));
        }
        if let Some(agent) = &self.agent
            && self.specs.iter().filter(|spec| spec.provider_tool.is_none()).any(|spec| {
                !agent.definition().tools.iter().any(|tool| tool.spec.name == spec.name)
            })
        {
            return Err(crate::agent_session::DriverError::Missing(
                "policy metadata for an advertised tool",
            ));
        }
        Ok(())
    }

    /// The tools as the host's setup reported them, every toolset's listed: the agent's frame
    /// takes their policies (approval, duplicates, retries, expert identity), so its definition
    /// is the one built after setup.
    pub fn set_up(
        tools: Vec<FunctionToolInfo>,
        host: Option<lk_agents_abi::HostRef>,
        agent: Option<Arc<dyn AgentFrame>>,
    ) -> Toolset {
        let specs = tools.iter().map(|t| t.spec.clone()).collect();
        let agent = agent.map(|frame| {
            let mut definition = frame.definition().clone();
            definition.tools =
                tools.into_iter().filter(|t| t.spec.provider_tool.is_none()).collect();
            Arc::new(VoiceAgentFrame {
                definition,
                host: Arc::clone(frame.host()),
                host_context: frame.host_context().clone(),
                toolset_scope: frame.toolset_scope(),
            }) as Arc<dyn AgentFrame>
        });
        Toolset { specs, host, agent }
    }
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
    toolset_scope: u64,
    definition: lk_agents_abi::agent::AgentDefinition,
    host: Arc<dyn lk_agents_abi::agent::AgentCallbacks>,
    host_context: lk_agents_abi::HostRef,
}

impl AgentFrame for VoiceAgentFrame {
    fn toolset_scope(&self) -> u64 {
        self.toolset_scope
    }
    fn definition(&self) -> &lk_agents_abi::agent::AgentDefinition {
        &self.definition
    }

    fn host(&self) -> &Arc<dyn lk_agents_abi::agent::AgentCallbacks> {
        &self.host
    }

    fn host_context(&self) -> &lk_agents_abi::HostRef {
        &self.host_context
    }
}

impl lk_agents_abi::agent::VoiceAgent for VoiceAgentFrame {
    fn new(
        definition: lk_agents_abi::agent::AgentDefinition,
        host: Arc<dyn lk_agents_abi::agent::AgentCallbacks>,
        host_context: lk_agents_abi::HostRef,
        toolset_scope: u64,
    ) -> Self {
        VoiceAgentFrame { definition, host, host_context, toolset_scope }
    }
}

/// A hosted call: the host running its callbacks and the invocation they run in, which its cancel
/// names. The invocation's `speech` is the runtime's to fill: it holds the host's objects.
#[derive(Clone)]
pub struct HostedTool {
    pub host: Arc<dyn lk_agents_abi::agent::AgentCallbacks>,
    pub speech: SpeechId,
    pub invocation: lk_agents_abi::agent::Invocation,
}

impl HostedTool {
    /// A callback of `speech` (for `call`, or on its output), the `operation`th of its session,
    /// hosted by `agent`: the invocation of one callback on a voice reply, which
    /// `{session}/{speech}` names.
    pub fn of(
        agent: &Arc<dyn AgentFrame>,
        session: &str,
        speech: SpeechId,
        operation: u64,
        call: Option<&FunctionCall>,
    ) -> HostedTool {
        let invocation = lk_agents_abi::agent::Invocation {
            operation,
            invocation_id: format!("{session}/{speech}"),
            agent_name: agent.definition().name.clone(),
            execution_id: None,
            frame_id: None,
            context_scope: None,
            application_context: None,
            history: Vec::new(),
            function_call: call.cloned(),
            approval: call.and_then(lk_agents_core::approval::mark),
            speech: None,
            host_context: Some(agent.host_context().clone()),
            span: None,
        };
        HostedTool { host: Arc::clone(agent.host()), speech, invocation }
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
    // A transition the host accepted for a VoiceTask asks for no reply: the parent gets the floor.
    let task_end = match (&outcome, &transition) {
        (ToolOutcome::Completed { .. }, Some(value)) => Some(Delivery::Complete(value.clone())),
        (ToolOutcome::Failed(reason), Some(_)) => Some(Delivery::Fail(reason.clone())),
        _ => None,
    };
    if let Some(delivery) = task_end {
        output.reply_required = false;
        return ToolExecution { call, output, end, raw_update: None, delivery: Some(delivery) };
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

pub(crate) fn tool_ended(
    id: String,
    call_id: String,
    message: Option<String>,
    status: ToolEndStatus,
) -> SessionEvent {
    let ended = ToolLifecycle::Ended { id, call_id, message, status };
    SessionEvent::ToolExecutionUpdated(ended)
}

pub fn validates_completion(task: bool, report: &ToolReport) -> bool {
    task && matches!(report.outcome, ToolOutcome::Completed { .. })
}

pub fn completion(task: bool, mut report: ToolReport) -> ToolReport {
    if !task {
        report.transition = None;
    }
    report
}
