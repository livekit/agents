use crate::agent::{Source, Visibility};
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::Out;
use crate::agent_session::{Driver, FillerSource, HookId, Origin};
use crate::events::{DiagnosticEvent, SessionEvent, ToolEndStatus, ToolLifecycle};
use crate::filler_scheduler::{FillerAction, FillerId, FillerScheduler};
use crate::generation::RunningToolCall;
use crate::trace::{CoreStep, ToolStep};
use lk_agents_abi::agent::run::RunError;
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
    /// Each tool step's calls on the shared machine, until the step's tools are committed.
    pub(crate) steps: BTreeMap<(SpeechId, u32), StepCalls>,
    /// The waiting calls the person's latest message answered: the only read-backs a step started
    /// since can consume.
    pub(crate) approval_answers: std::collections::BTreeSet<String>,
    /// Mints the steps' [`CallId`]s.
    pub(crate) next_call: u64,
}

/// One step's calls as `lk_agents_core::tool_calls` decides them, and the call tasks waiting
/// for their next move.
#[derive(Debug)]
pub(crate) struct StepCalls {
    pub(crate) calls: lk_agents_core::tool_calls::ToolCalls,
    /// The tools the step's calls name, as their runners described them.
    pub(crate) tools: Vec<FunctionToolInfo>,
    /// Started calls not yet on the machine, in the model's order; each joins once it and every
    /// earlier one is described (`Some`).
    pub(crate) describing: Vec<(CallId, FunctionCall, Option<Option<FunctionToolInfo>>)>,
    pub(crate) waiting: BTreeMap<CallId, crate::agent_session::RequestId>,
}

pub use lk_agents_core::agent_run::CallId;

/// The next move of a call's task, decided by the shared tool machine.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolGo {
    /// Ask the runner for the tool's description: what the machine decides with.
    Describe { call: CallId },
    /// Bind and validate these arguments (the framework's parse, with the schema's defaults).
    Prepare { call: CallId, arguments: serde_json::Map<String, serde_json::Value> },
    /// Render the read-back of the prepared arguments.
    ReadBack { call: CallId },
    /// Run the body; `item` carries the call's approval marks.
    Execute { item: Box<FunctionCall> },
    /// Settled without its body: `item` as recorded, and whether it spends a retry.
    Settled { item: Box<FunctionCall>, outcome: ToolOutcome, retry: bool },
    /// The step's calls are over: the call does not run.
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

/// `answer_arguments`: the person's words as the tool's first string parameter; `None` for a tool
/// without one (the model reads the answer and calls the tool itself).
pub fn answer_arguments(spec: &lk_agents_abi::ToolSpec, said: &str) -> Option<String> {
    let properties = spec.json_schema.get("properties")?.as_object()?;
    let (name, _) = properties
        .iter()
        .find(|(_, p)| p.get("type").and_then(|t| t.as_str()) == Some("string"))?;
    let mut arguments = serde_json::Map::new();
    arguments.insert(name.clone(), serde_json::Value::String(said.to_owned()));
    Some(serde_json::Value::Object(arguments).to_string())
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
    pub(crate) verdict: crate::agent_session::RequestId,
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
    /// A tool spent its retry budget: the reply ends here and the run fails with it.
    pub retries_exceeded: Option<RunError>,
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
    /// Invalid or unknown arguments, or `ModelRetry`: spends the tool's retry budget.
    pub retry: bool,
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

pub use lk_agents_abi::agent::FunctionToolInfo;
pub use lk_agents_abi::llm::tool_executor::HostToolRunner;

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
    let retry = matches!(outcome, ToolOutcome::ModelRetry(_));
    // A completion the host validated for a VoiceTask asks for no reply: the parent gets the floor.
    if let (ToolOutcome::Completed { .. }, Some(value)) = (&outcome, &transition) {
        output.reply_required = false;
        let delivery = Some(Delivery::Complete(value.clone()));
        return ToolExecution { call, output, end, raw_update: None, delivery, retry };
    }
    let delivery = match outcome {
        ToolOutcome::Update { raw, .. } => {
            // `RunContext.update`'s marker: a non-blocking tool may wait on its own speech.
            call.extra.insert(
                crate::speech_handle::TOOL_NON_BLOCKING_EXTRA_KEY.to_owned(),
                serde_json::Value::Bool(true),
            );
            let raw_update = Some(raw);
            return ToolExecution { call, output, end, raw_update, delivery: None, retry };
        }
        ToolOutcome::InputRequired { question, verbatim, tool } => {
            Some(Delivery::Ask(Ask { question, verbatim, tool }))
        }
        other => lk_agents_core::tool_outcome::verbatim(&other)
            .map(|text| Delivery::Say(text.to_owned())),
    };
    ToolExecution { call, output, end, raw_update: None, delivery, retry }
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
    /// A call completed a VoiceTask: the step has no follow-up, whatever the outputs ask.
    completes: bool,
    /// A tool spent its retry budget: no follow-up either.
    pub retries_exceeded: Option<RunError>,
    previous_user_metrics: Option<MetricsReport>,
}

impl StepDecision {
    /// `function_tools_executed` was handled: the next step from what the handlers left
    /// (`cancel_tool_reply()`, `cancel_agent_handoff()`). An interrupted step stays interrupted.
    pub fn settle(mut self, event: ToolsExecutedEvent) -> StepDecision {
        self.commit = commit_of(&event.function_calls, &event.function_call_outputs);
        if self.next != NextStep::Interrupted {
            self.next = next_of(
                event.has_tool_reply()
                    && !self.interrupted
                    && !self.completes
                    && self.retries_exceeded.is_none(),
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
    /// The reply's [`ToolRetries`](lk_agents_core::tool_outcome::ToolRetries) after this step.
    pub retries_exceeded: Option<RunError>,
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
        retries_exceeded: inputs.retries_exceeded,
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
        retries_exceeded: None,
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
    /// The host's `function_tools_executed` handlers answered under `hook` (`left`: each output's
    /// `reply_required`), or will not (`None`): the step goes on.
    pub(crate) fn on_tools_executed_handled(
        &mut self,
        hook: crate::agent_session::HookId,
        left: Option<Vec<bool>>,
        out: &mut Out,
    ) {
        let Some(Handling { speech, step, mut tools, verdict }) =
            self.activity.tools.handling.remove(&hook)
        else {
            return;
        };
        if let (Some(reply_required), Some(event)) = (left, &mut tools.executed) {
            if reply_required.len() == event.function_call_outputs.len() {
                for (out, required) in event.function_call_outputs.iter_mut().zip(reply_required) {
                    out.reply_required = required;
                }
            } else {
                out.trace(CoreStep::PortFailure(
                    "function_tools_executed answered for a different number of outputs",
                ));
            }
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
        verdict: Option<crate::agent_session::RequestId>,
        hosted: bool,
        out: &mut Out,
    ) {
        let StepTools { commit, executed, warnings, deliveries, retries_exceeded } = tools;
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
        out.trace(ToolStep::Finished { speech, committed: n });
        if let Some(error) = retries_exceeded {
            self.tool_retries_exceeded(speech, error, out);
        }
        let Some(executed) = executed else { return };
        // Not the host's to emit: after the commit, as before any hook existed for it.
        if !hosted {
            out.machine().emit(SessionEvent::FunctionToolsExecuted(executed.clone()));
        }
        if let Some(verdict) = verdict {
            out.answer(
                Some(verdict),
                crate::agent_session::Answer::ToolsVerdict(ToolsVerdict { executed: *executed }),
            );
        }
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
