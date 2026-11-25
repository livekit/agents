use super::effects::AgentRunEffect;
use super::execution;
use super::planning::{RequestSnapshot, plan_request};
use crate::agent::{self, AgentError};
use crate::llm::history::ChatContextExt;
use crate::tool_calls::{self, FinalCandidate, PrepareFailure, Recorded, Settled, ToolCalls};
use crate::tool_outcome;
use crate::usage::{Ledger, ModelKind};
use cancellation::Owed;
use lk_agents_abi::agent::run::{
    Output, RunError, RunOutcome, RunQuestion, RunResult, StreamEvent, ValidationId,
};
use lk_agents_abi::agent::{AgentDefinition, OutputSpec};
use lk_agents_abi::agent::{ArgumentIdentity, OutputCandidate, ReadBackAnswer};
use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::llm::tool_executor::{ASK_IN_OWN_WORDS, ToolOutcome};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{
    ChatContext, ChatItem, ChatMessage, CopyOptions, FunctionCall, LlmChunk, Role, ToolSpec, Ts,
    WallAnchor,
};
use model_step::{ModelStep, Purpose};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime};
use step_outcome::Next;

mod cancellation;
mod checkpoint;
mod coordinated;
mod model_step;
mod presentation;
mod remote_resume;
mod step_outcome;
mod text_session;
mod tool_output;
mod tool_settlement;

#[cfg(test)]
#[path = "input_offers_tests.rs"]
mod input_offers_tests;

pub use coordinated::{AgentRunContinuation, Opening};
pub use text_session::DelegationCheckpoint;

#[cfg(test)]
#[path = "machine_tests.rs"]
mod tests;

pub use super::{CallId, CheckpointId, RenderId, RequestId, StepId};

/// The feedback a rejected text output gives the model.
const RETRY_PROMPT: &str = "Validation failed, fix the output and try again: ";

/// The feedback an empty response gives the model when a structured output is expected.
const EMPTY_RESPONSE_PROMPT: &str = "Please return text or call a tool.";

/// What the caller supplies to one invocation.
#[derive(Debug, Clone)]
pub struct RunInput {
    /// Names the invocation in its events and spans.
    pub invocation_id: String,
    /// The random value its caller drew for the chat items it mints (`types::ids::short`).
    pub id_seed: u64,
    pub opening: Opening,
    /// The conversation so far.
    pub history: Vec<ChatItem>,
    /// Another agent asked this run: questions pass up as written; the facing agent words them.
    pub delegated: bool,
    /// The coordinator reading the outcome table.
    pub mode: tool_outcome::Mode,
    /// The agent's model as the runtime resolved it; an invocation opening on a message cannot
    /// run a step without one (a coordinated one's coordinator starts its streams).
    pub llm: Option<lk_agents_abi::resources::LlmRef>,
    /// Maps supplied instants to wall stamps on recorded items.
    pub wall: WallAnchor,
    /// How long cancelled work may take to settle before it is reported incomplete.
    pub cleanup_timeout: Duration,
}

/// The host's verdict on an output candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputVerdict {
    /// Accepted; a `str` output's final text (a validator may replace it).
    Accepted { text: Option<String> },
    /// `ModelRetry` or a validation error: the model corrects its output.
    Retry(String),
    /// A validator or output function failed outright.
    Failed(String),
}

/// Inputs from the queue owner.
#[derive(Debug, Clone)]
pub enum AgentRunEvent {
    Start,
    HostCheckpointDone {
        checkpoint: CheckpointId,
        /// Whether the run yields at the checkpoint.
        should_yield: Result<bool, HostError>,
        tools: Option<Vec<lk_agents_abi::agent::FunctionToolInfo>>,
    },
    InstructionsRendered {
        render: RenderId,
        result: Result<String, HostError>,
    },
    LlmChunk {
        request: RequestId,
        chunk: LlmChunk,
    },
    LlmEnded {
        request: RequestId,
    },
    /// Coordinated: the coordinator admitted a call of `request`'s response as it streamed;
    /// `call` keeps the coordinator's id, stamp and repaired arguments.
    CallMade {
        request: RequestId,
        call: FunctionCall,
    },
    /// The request failed after the provider's own recovery, or was cancelled.
    LlmFailed {
        request: RequestId,
        error: ProviderError,
    },
    /// The host bound and validated the arguments; `prepared` holds them for the body.
    ToolPrepared {
        call: CallId,
        prepared: ResourceId,
        identity: ArgumentIdentity,
    },
    /// The host did not prepare the arguments.
    ToolPrepareFailed {
        call: CallId,
        failure: PrepareFailure,
    },
    ReadBackRendered {
        call: CallId,
        result: Result<ReadBackAnswer, HostError>,
    },
    /// The host rendered the call's template; `Ok(None)` keeps the framework's text.
    ToolHandlingTemplateRendered {
        call: CallId,
        text: Result<Option<String>, HostError>,
    },
    /// A captured tool operation requests an ordered exact output. Missing source is a lost
    /// operation, never permission to attach it to another current call.
    ToolOutputRequested {
        output: super::ToolOutputId,
        call: Option<CallId>,
        text: String,
        stream: bool,
    },
    ToolOutputDelivered {
        output: super::ToolOutputId,
        outcome: lk_agents_abi::agent::run::SayOutcome,
    },
    /// `ctx.update(text)`.
    ToolProgress {
        call: CallId,
        message: String,
        silent: bool,
    },
    ToolCallUpdated {
        call: CallId,
        extra: Map<String, Value>,
    },
    ToolSettled {
        call: CallId,
        report: lk_agents_abi::llm::tool_executor::ToolReport,
    },
    /// A delegate the call ran in process (an agent tool) consumed `usage`; the run counts it.
    DelegateUsage {
        usage: lk_agents_abi::usage::Usage,
    },
    ToolLaunchFailed {
        call: CallId,
        error: HostError,
    },
    /// The call's task ended without reporting: the call settles in whatever phase it waits, a
    /// running body on `outcome`.
    CallLost {
        call: CallId,
        outcome: ToolOutcome,
    },
    OutputValidated {
        validation: ValidationId,
        verdict: OutputVerdict,
    },
    /// The coordinator handled the canonical tool results and supplies continuation constraints.
    ToolResultsHandled {
        step: StepId,
        continuation: AgentRunContinuation,
        /// Host edits to reply requirements, keyed by the canonical function call ID.
        /// Unknown call IDs cannot change this step or another step's history.
        reply_required: BTreeMap<String, bool>,
    },
    /// Coordinated: the person's latest message answered these read-backs; a step that has not
    /// called yet reads them.
    Answered {
        approval_answers: BTreeSet<String>,
    },
    /// Coordinated: the coordinator recorded `item` outside this invocation (another reply's
    /// call): approval reads it as asked or granted.
    Recorded {
        item: Box<ChatItem>,
    },
    Cancel,
    /// The deadline from `wake_at()` is due.
    Wake,
}

impl From<tool_calls::ToolCallEffect> for AgentRunEffect {
    fn from(effect: tool_calls::ToolCallEffect) -> Self {
        match effect {
            tool_calls::ToolCallEffect::Prepare { call, arguments, function_call } => {
                AgentRunEffect::ToolPrepare { call, arguments, function_call }
            }
            tool_calls::ToolCallEffect::ReadBack { call, tool, prepared } => {
                AgentRunEffect::ReadBackRender { call, tool, prepared }
            }
            tool_calls::ToolCallEffect::Execute { call, prepared, function_call, approval } => {
                AgentRunEffect::ToolExecute { call, prepared, function_call, approval }
            }
            tool_calls::ToolCallEffect::Cancel { call } => AgentRunEffect::ToolCancel { call },
            tool_calls::ToolCallEffect::CancelExecution { execution } => {
                AgentRunEffect::CancelToolExecution { execution }
            }
            tool_calls::ToolCallEffect::ReleaseArguments { prepared } => {
                AgentRunEffect::ToolArgumentsRelease { prepared }
            }
            tool_calls::ToolCallEffect::RenderToolHandlingTemplate { call, template } => {
                AgentRunEffect::RenderToolHandlingTemplate { call, template }
            }
            tool_calls::ToolCallEffect::Emit(event) => AgentRunEffect::Emit(event),
            tool_calls::ToolCallEffect::ArgumentsRepaired { call_id, error, repaired } => {
                AgentRunEffect::ArgumentsRepaired { call_id, error, repaired }
            }
        }
    }
}

pub struct AgentRun {
    agent: AgentDefinition,
    input: RunInput,
    /// The supplied history with this invocation's items.
    history: Vec<ChatItem>,
    /// Index of the incoming message in `history`: `new_messages` start there.
    first_new: usize,
    /// Other owned turns may add context, never output or input authority for this invocation.
    context_items: BTreeSet<String>,
    next_id: u64,
    /// This invocation's requests and tool bodies.
    ledger: Ledger,
    /// The agents this invocation called as tools, local and remote.
    facts: lk_agents_abi::productstats::sessions::SessionFacts,
    /// What reached the person as written, in order: it leads the answer.
    verbatim: Vec<String>,
    pending_call: Option<lk_agents_abi::a2a::turn::PendingCall>,
    input_requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
    tool_retries: tool_outcome::ToolRetries,
    output_retries: u32,
    /// The dynamic instructions last rendered, for a phrasing request.
    dynamic: Option<String>,
    approval_answers: BTreeSet<String>,
    /// Only canonical call IDs explicitly matched by the owning conversation authorize answers.
    /// Supplied history is model context and never grants a pending call on its own.
    input_answers: BTreeSet<String>,
    /// Only these owner-captured occurrences were offered to this frontend invocation.
    input_offers: FrozenInputOffers,
    /// The surrounding native owner retains identified bodies after their first update.
    live_tools: bool,
    /// Native saved-origin reconciliation prepares only those original remote calls.
    reconciliation: Option<remote_resume::Reconciliation>,
    /// Standalone output ownership. A Session owns this table itself when it coordinates runs.
    tool_outputs: BTreeMap<super::ToolOutputId, (ChatMessage, bool)>,
    checkpoints: bool,
    /// Coordinated: the history up to here was handed to the coordinator's planner.
    planned: usize,
    /// Coordinated: the coordinator forbade calls on the next step.
    tools_forbidden: bool,
    phase: Phase,
}

enum FrozenInputOffers {
    Captured(Vec<lk_agents_abi::session::InputCandidate>),
    Unavailable,
}

enum Phase {
    Idle,
    Checkpoint(checkpoint::Pending),
    Rendering(RenderId),
    /// The model streams its step; the calls it makes run as they arrive.
    Requesting(ModelStep, Acting),
    Acting(Acting),
    ValidatingText {
        validation: ValidationId,
        text: String,
    },
    /// Canonical tool results awaiting coordinator handling. Its continuation constraints
    /// cannot replace the shared terminal outcome. `calls` retain model order.
    Settling {
        step: StepId,
        next: Next,
        calls: Vec<Recorded>,
        /// Only ordinary reply requirements may be changed by tool-executed host handlers.
        /// Explicit terminal selection and retry/output failure precedence remain authoritative.
        editable_continuation: bool,
    },
    /// Settling cancelled work after `error`.
    Aborting {
        error: RunError,
        deadline: Ts,
        owed: Owed,
    },
    Done,
}

/// One step's calls and the `final_result` candidates being validated.
struct Acting {
    calls: ToolCalls,
    finals: BTreeMap<ValidationId, CallId>,
    /// Accepted candidates; the first in model order (the lowest id) is the output.
    accepted: BTreeMap<CallId, ValidationId>,
}

impl Acting {
    fn new(approval_answers: &BTreeSet<String>) -> Self {
        Acting {
            calls: ToolCalls::new(approval_answers.clone()),
            finals: BTreeMap::new(),
            accepted: BTreeMap::new(),
        }
    }
}

impl Phase {
    /// The step's calls, while the model streams them or once it is done.
    fn acting(&mut self) -> Option<&mut Acting> {
        match self {
            Phase::Requesting(_, acting) | Phase::Acting(acting) => Some(acting),
            _ => None,
        }
    }
}

impl AgentRun {
    /// Accepted history plus a stable-id, explicitly partial message for a streaming response.
    pub fn progress_history(&self) -> Vec<ChatItem> {
        let mut history = self.history.clone();
        if let Phase::Requesting(step, _) = &self.phase
            && !step.text.is_empty()
        {
            let mut message = ChatMessage::text(
                step.message_id.clone(),
                Role::Assistant,
                &step.text,
                self.stamp(self.input.wall.at),
            );
            message.interrupted = true;
            history.push(ChatItem::Message(message));
        }
        if let Phase::Requesting(_, acting) | Phase::Acting(acting) = &self.phase {
            for item in
                acting.calls.checkpoint_items(self.input.mode, self.stamp(self.input.wall.at))
            {
                if self.reconciliation.is_some() {
                    lk_agents_abi::llm::chat_context::ordering::commit_by_instant(
                        &mut history,
                        item,
                    );
                } else if !history.iter().any(|existing| existing.id() == item.id()) {
                    history.push(item);
                }
            }
        }
        history
    }

    /// This invocation's accepted and in-flight items. Supplied model context may contain
    /// private reminders or rewritten history; it is not a new conversation revision.
    pub fn progress_new_items(&self) -> Vec<ChatItem> {
        self.progress_history()
            .into_iter()
            .enumerate()
            .filter(|(at, item)| self.recorded_here(*at, item))
            .map(|(_, item)| item)
            .collect()
    }

    pub fn new(agent: AgentDefinition, mut input: RunInput) -> Result<AgentRun, AgentError> {
        agent::validate(&agent)?;
        if let Opening::Message(message) = &mut input.opening {
            execution::bind(message, &input.history, None, &input.invocation_id);
        }
        Ok(AgentRun {
            agent,
            history: Vec::new(),
            first_new: 0,
            context_items: BTreeSet::new(),
            input,
            next_id: 0,
            ledger: Ledger::default(),
            facts: Default::default(),
            verbatim: Vec::new(),
            pending_call: None,
            input_requirements: Vec::new(),
            tool_retries: tool_outcome::ToolRetries::default(),
            output_retries: 0,
            dynamic: None,
            approval_answers: BTreeSet::new(),
            input_answers: BTreeSet::new(),
            input_offers: FrozenInputOffers::Captured(Vec::new()),
            live_tools: false,
            reconciliation: None,
            tool_outputs: BTreeMap::new(),
            checkpoints: false,
            planned: 0,
            tools_forbidden: false,
            phase: Phase::Idle,
        })
    }

    pub fn wake_at(&self) -> Option<Ts> {
        match &self.phase {
            Phase::Aborting { deadline, .. } => Some(*deadline),
            _ => None,
        }
    }

    pub fn with_input_answers(mut self, answered: BTreeSet<String>) -> Self {
        self.input_answers = answered;
        self
    }

    /// The current step's exact admitted origins. A later human answer cannot authorize
    /// a step that already began calling; its authority applies only to the following step.
    pub fn input_answers(&self) -> &BTreeSet<String> {
        match &self.phase {
            Phase::Requesting(_, acting) | Phase::Acting(acting) => acting.calls.input_answers(),
            _ => &self.approval_answers,
        }
    }

    pub fn with_input_offers(
        mut self,
        offered: Vec<lk_agents_abi::session::InputCandidate>,
    ) -> Self {
        self.input_offers = FrozenInputOffers::Captured(offered);
        self
    }

    /// Voice opens its run before the first frontend request is planned. Response admission
    /// fills its frozen offers before releasing that request to the provider.
    pub fn set_input_offers(&mut self, offered: Vec<lk_agents_abi::session::InputCandidate>) {
        self.input_offers = FrozenInputOffers::Captured(offered);
    }

    pub fn input_offers(&self) -> &[lk_agents_abi::session::InputCandidate] {
        match &self.input_offers {
            FrozenInputOffers::Captured(offered) => offered,
            FrozenInputOffers::Unavailable => &[],
        }
    }

    /// A provider ended without the human words needed for native response correlation.
    /// Remote calls receive feedback rather than guessing a response or waiting forever.
    pub fn input_offers_unavailable(&mut self) {
        self.input_offers = FrozenInputOffers::Unavailable;
    }

    pub fn handle(&mut self, event: AgentRunEvent, now: Ts) -> Vec<AgentRunEffect> {
        self.handle_with_running_tools(event, now, &[])
    }

    /// Drive an invocation with live bodies owned by its surrounding session. A detached
    /// body's first update settled its model step, not its execution or duplicate policy.
    /// Supply bodies outside this invocation's currently running call table only.
    pub fn handle_with_running_tools(
        &mut self,
        event: AgentRunEvent,
        now: Ts,
        running: &[crate::tool_calls::RunningToolExecution],
    ) -> Vec<AgentRunEffect> {
        let fx = self.step(event, now, running);
        // A tool body starts with its `ToolExecute`, whatever it then returns.
        for effect in &fx {
            let AgentRunEffect::ToolExecute { function_call, .. } = effect else { continue };
            self.ledger.tool_started();
            let tools = &self.agent.tools;
            if let Some(tool) = tools.iter().find(|tool| tool.spec.name == function_call.name) {
                self.facts.tool_started(tool);
            }
        }
        fx
    }

    fn step(
        &mut self,
        event: AgentRunEvent,
        now: Ts,
        running: &[crate::tool_calls::RunningToolExecution],
    ) -> Vec<AgentRunEffect> {
        let mut fx = Vec::new();
        match event {
            AgentRunEvent::ToolOutputRequested { output, call, text, stream } => {
                self.request_tool_output(output, call, text, stream, now, &mut fx);
            }
            AgentRunEvent::ToolOutputDelivered { output, outcome } => {
                self.delivered_tool_output(output, outcome, &mut fx);
            }
            AgentRunEvent::Start => self.start(now, &mut fx),
            AgentRunEvent::Cancel => self.abort(RunError::Cancelled, now, &mut fx),
            AgentRunEvent::Wake => self.wake(now, &mut fx),
            event if matches!(self.phase, Phase::Aborting { .. } | Phase::Done) => {
                self.settle_owed(event, &mut fx)
            }
            AgentRunEvent::HostCheckpointDone { checkpoint, should_yield, tools } => {
                self.checkpoint_done(checkpoint, should_yield, tools, now, &mut fx)
            }
            AgentRunEvent::InstructionsRendered { render, result } => match self.phase {
                Phase::Rendering(r) if r == render => match result {
                    Ok(dynamic) => {
                        self.dynamic = Some(dynamic.clone());
                        self.request(Purpose::Turn, Some(dynamic), now, &mut fx)
                    }
                    Err(e) => self.abort(RunError::Host(e.to_string()), now, &mut fx),
                },
                _ => {}
            },
            AgentRunEvent::LlmChunk { request, chunk } => self.chunk(request, chunk, now, &mut fx),
            AgentRunEvent::LlmEnded { request } => self.response(request, now, &mut fx),
            AgentRunEvent::CallMade { request, call } => {
                if matches!(&self.phase, Phase::Requesting(s, _) if s.request == request) {
                    self.admit(call, &mut fx);
                }
            }
            AgentRunEvent::LlmFailed { request, error } => {
                if matches!(&self.phase, Phase::Requesting(s, _) if s.request == request) {
                    self.ledger.finish(request.0, false);
                    // The provider already recovered what it could; the calls the response made
                    // so far are cancelled.
                    let Phase::Requesting(step, acting) =
                        std::mem::replace(&mut self.phase, Phase::Idle)
                    else {
                        unreachable!()
                    };
                    end_text(&step, true, &mut fx);
                    self.phase = Phase::Acting(acting);
                    let error = if error.cancelled() {
                        RunError::Cancelled
                    } else {
                        RunError::Model {
                            detail: error.to_string(),
                            kind: error.attempt_error().error_kind,
                        }
                    };
                    self.abort(error, now, &mut fx);
                }
            }
            AgentRunEvent::ToolPrepared { call, prepared, identity } => {
                let held = match self.phase.acting() {
                    Some(a) => a.calls.prepared(
                        call,
                        (prepared, identity),
                        &self.agent.tools,
                        self.history.iter(),
                        running,
                        &mut fx,
                    ),
                    None => false,
                };
                if !held {
                    fx.push(AgentRunEffect::ToolArgumentsRelease { prepared });
                }
                self.step_settled(now, &mut fx);
            }
            AgentRunEvent::ToolPrepareFailed { call, failure } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.prepare_failed(
                        call,
                        failure,
                        &self.agent.tools,
                        self.history.iter(),
                        running,
                        &mut fx,
                    );
                }
                self.step_settled(now, &mut fx);
            }
            AgentRunEvent::ReadBackRendered { call, result } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.read_back(
                        call,
                        result,
                        &self.agent.tools,
                        self.history.iter(),
                        running,
                        &mut fx,
                    );
                }
                self.step_settled(now, &mut fx);
            }
            AgentRunEvent::ToolHandlingTemplateRendered { call, text } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.template_rendered(call, text, &mut fx);
                }
                self.step_settled(now, &mut fx);
            }
            AgentRunEvent::ToolProgress { call, message, silent } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.progress(call, message.clone(), silent, &mut fx);
                }
                if self.live_tools {
                    self.tool_settled(
                        call,
                        ToolOutcome::Update {
                            raw: message.clone(),
                            message,
                            reply_required: !silent,
                        }
                        .into(),
                        now,
                        &mut fx,
                    );
                }
            }
            AgentRunEvent::ToolCallUpdated { call, extra } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.update_extra(call, extra);
                }
            }
            AgentRunEvent::DelegateUsage { usage } => self.ledger.add(&usage),
            AgentRunEvent::ToolSettled { call, report } => {
                self.tool_settled(call, report, now, &mut fx)
            }
            AgentRunEvent::CallLost { call, outcome } => {
                let body = match self.phase.acting() {
                    Some(a) => {
                        a.calls.lost(call, &self.agent.tools, self.history.iter(), running, &mut fx)
                    }
                    None => false,
                };
                if body {
                    self.tool_settled(call, outcome.into(), now, &mut fx);
                } else {
                    self.step_settled(now, &mut fx);
                }
            }
            AgentRunEvent::ToolLaunchFailed { call, error } => {
                self.tool_settled(call, tool_outcome::host_failure(&error).into(), now, &mut fx);
            }
            AgentRunEvent::OutputValidated { validation, verdict } => {
                self.validated(validation, verdict, now, &mut fx)
            }
            AgentRunEvent::ToolResultsHandled { step, continuation, reply_required } => {
                self.tool_results_handled(step, continuation, reply_required, now, &mut fx)
            }
            AgentRunEvent::Answered { approval_answers } => self.answered(approval_answers),
            AgentRunEvent::Recorded { item } => self.history.push(*item),
        }
        fx
    }

    /// `item_<12 hex>`, as v1: each step's message is its own provider group, and its calls,
    /// `{message}/fnc_i`, join it there.
    fn item_id(&mut self) -> String {
        let n = self.mint();
        lk_agents_abi::types::ids::short("item", self.input.id_seed, n)
    }

    fn stamp(&self, now: Ts) -> SystemTime {
        let secs = self.input.wall.seconds(now);
        lk_agents_abi::wall_from_secs(secs.max(0.0)).unwrap_or(SystemTime::UNIX_EPOCH)
    }

    fn start(&mut self, now: Ts, fx: &mut Vec<AgentRunEffect>) {
        if !matches!(self.phase, Phase::Idle) {
            return;
        }
        if self.input.llm.is_none() && !self.input.opening.coordinated() {
            return self.finish(Err(RunError::NoModel), fx);
        }
        self.history = std::mem::take(&mut self.input.history);
        let message = match &mut self.input.opening {
            Opening::Message(message) => (**message).clone(),
            Opening::Coordinated { approval_answers } => {
                self.approval_answers = std::mem::take(approval_answers);
                self.first_new = self.history.len();
                self.planned = self.history.len();
                return self.next_step(now, fx);
            }
        };
        self.approval_answers = std::mem::take(&mut self.input_answers);
        self.first_new = self.history.len();
        self.history.push(ChatItem::Message(message));
        self.next_step(now, fx);
    }

    /// The next model step: dynamic instructions first when there are any.
    /// Coordinated, the coordinator renders the instructions with the rest of the request.
    fn next_step(&mut self, now: Ts, fx: &mut Vec<AgentRunEffect>) {
        if self.agent.dynamic_instructions && !self.input.opening.coordinated() {
            let render = RenderId(self.mint());
            fx.push(AgentRunEffect::RenderInstructions { render });
            self.phase = Phase::Rendering(render);
        } else {
            self.request(Purpose::Turn, None, now, fx);
        }
    }

    fn request(
        &mut self,
        purpose: Purpose,
        dynamic: Option<String>,
        now: Ts,
        fx: &mut Vec<AgentRunEffect>,
    ) {
        if self.checkpoints {
            self.checkpoint_request(purpose, dynamic, fx);
        } else {
            self.start_request(purpose, dynamic, now, fx);
        }
    }

    fn start_request(
        &mut self,
        purpose: Purpose,
        dynamic: Option<String>,
        now: Ts,
        fx: &mut Vec<AgentRunEffect>,
    ) {
        let mut tools: Vec<ToolSpec> = Vec::new();
        let mut phrase = None;
        match &purpose {
            Purpose::Turn => {
                tools.extend(self.agent.tools.iter().map(|t| t.spec.clone()));
                tools.extend(agent::final_result_tool(&self.agent));
            }
            Purpose::Phrase { ask, results: false } => {
                phrase = Some(format!("{ASK_IN_OWN_WORDS}{}", ask.question))
            }
            Purpose::Phrase { ask, results: true } => {
                phrase = Some(format!("{}{}", presentation::REPORT_THEN_ASK, ask.question))
            }
            Purpose::Present { candidates, results } => {
                phrase = Some(presentation::prompt(candidates, *results));
                tools.push(presentation::tool());
            }
            Purpose::Report(_) => phrase = Some(presentation::REPORT.into()),
        }
        if self.input.opening.coordinated() {
            let request = RequestId(self.mint());
            let message_id = self.item_id();
            self.request_started(request);
            fx.push(AgentRunEffect::PlanRequest {
                request,
                message_id: message_id.clone(),
                pending: self.history[self.planned..].to_vec(),
                tools,
                tools_forbidden: std::mem::take(&mut self.tools_forbidden),
            });
            self.planned = self.history.len();
            let step = ModelStep::new(request, purpose, message_id);
            self.phase = Phase::Requesting(step, Acting::new(&self.approval_answers));
            return;
        }
        let Some(llm) = self.input.llm.as_ref().map(|l| l.id) else {
            return self.finish(Err(RunError::NoModel), fx);
        };
        let instructions =
            agent::instructions_with(&self.agent, dynamic.as_deref().unwrap_or_default());
        let planned = plan_request(RequestSnapshot {
            history: &self.history,
            instructions: &instructions,
            tools: &tools,
            on_enter: false,
            turn_instructions: phrase.as_deref(),
            created_at: self.stamp(now),
        });
        let request = RequestId(self.mint());
        let message_id = self.item_id();
        self.request_started(request);
        fx.push(AgentRunEffect::LlmRequestStart {
            request,
            llm,
            chat_ctx: planned.chat_ctx,
            tools: planned.tools,
        });
        let step = ModelStep::new(request, purpose, message_id);
        self.phase = Phase::Requesting(step, Acting::new(&self.approval_answers));
    }

    /// A chunk of the step's response: its text streams, and each call it makes starts at once
    /// (voice dispatches before the model's end of stream).
    fn chunk(
        &mut self,
        request: RequestId,
        chunk: LlmChunk,
        now: Ts,
        fx: &mut Vec<AgentRunEffect>,
    ) {
        let Phase::Requesting(step, _) = &mut self.phase else {
            return;
        };
        if step.request != request {
            return;
        }
        if let Some(usage) = chunk.usage.as_deref() {
            self.ledger.report(request.0, usage.into());
        }
        let added = step.chunk(chunk);
        let message_id = step.message_id.clone();
        if let Some(delta) = added.text {
            let message_id = message_id.clone();
            fx.push(AgentRunEffect::Emit(StreamEvent::TextDelta { message_id, delta }));
        }
        for (i, c) in added.calls {
            let call = FunctionCall {
                id: format!("{message_id}/fnc_{i}"),
                call_id: c.call_id,
                arguments: c.arguments,
                name: c.name,
                created_at: self.stamp(now),
                extra: c.extra.unwrap_or_default(),
                group_id: None,
            };
            self.admit(call, fx);
        }
    }

    /// Starts one call of the step: a `final_result` candidate goes to output validation.
    fn admit(&mut self, call: FunctionCall, fx: &mut Vec<AgentRunEffect>) {
        let id = CallId(self.mint());
        let offered = match &self.input_offers {
            FrozenInputOffers::Captured(offered) => Some(offered.as_slice()),
            FrozenInputOffers::Unavailable => None,
        };
        let Some(acting) = self.phase.acting() else {
            return;
        };
        let Some(FinalCandidate { call, value }) =
            acting.calls.add_step_call(id, call, &self.agent, offered, fx)
        else {
            return;
        };
        let value = match value {
            Ok(value) => value,
            Err(reason) => return acting.calls.final_settled(call, Some(reason.message), fx),
        };
        let validation = ValidationId(self.mint());
        let candidate = OutputCandidate::Value(value);
        fx.push(AgentRunEffect::ValidateOutput { validation, candidate });
        if let Some(acting) = self.phase.acting() {
            acting.finals.insert(validation, call);
        }
    }

    fn response(&mut self, request: RequestId, now: Ts, fx: &mut Vec<AgentRunEffect>) {
        if !matches!(&self.phase, Phase::Requesting(s, _) if s.request == request) {
            return;
        }
        let Phase::Requesting(step, acting) = std::mem::replace(&mut self.phase, Phase::Idle)
        else {
            unreachable!()
        };
        end_text(&step, false, fx);
        let (message_id, purpose) = (step.message_id.clone(), step.purpose.clone());
        let response = step.end();
        self.ledger.finish(request.0, true);
        if let Purpose::Present { candidates, .. } = &purpose {
            return self.presented(candidates, response.presentation, now, fx);
        }
        if let Purpose::Phrase { ask, .. } = purpose {
            let text = response.text.trim();
            let question = if text.is_empty() { ask.question.clone() } else { text.to_owned() };
            return self.asked(
                RunQuestion { question, ..ask },
                (!text.is_empty()).then_some(message_id),
                &[],
                None,
                now,
                fx,
            );
        }
        if let Purpose::Report(ask) = purpose {
            let report = response.text.trim().to_owned();
            let lead = (!report.is_empty()).then(|| {
                let at = self.stamp(now);
                self.record(ChatMessage::text(message_id, Role::Assistant, &report, at).into());
                report
            });
            return self.asked(ask, None, &[], lead, now, fx);
        }
        if !response.text.is_empty() {
            self.record(
                ChatMessage::text(
                    message_id.clone(),
                    Role::Assistant,
                    &response.text,
                    self.stamp(now),
                )
                .into(),
            );
        }
        if acting.calls.is_empty() {
            return self.text_output(response.text, now, fx);
        }
        self.phase = Phase::Acting(acting);
        self.step_settled(now, fx);
    }

    fn text_output(&mut self, text: String, now: Ts, fx: &mut Vec<AgentRunEffect>) {
        match &self.agent.output {
            OutputSpec::Structured { .. } if text.is_empty() => {
                self.retry_output(EMPTY_RESPONSE_PROMPT.to_owned(), now, fx)
            }
            // Conversational text where a typed output is expected asks the person.
            OutputSpec::Structured { .. } => {
                let ask = RunQuestion { question: text, verbatim: false };
                self.settle_ask(ask, None, fx)
            }
            OutputSpec::Text { validated: true } => {
                let validation = ValidationId(self.mint());
                fx.push(AgentRunEffect::ValidateOutput {
                    validation,
                    candidate: OutputCandidate::Text(text.clone()),
                });
                self.phase = Phase::ValidatingText { validation, text };
            }
            OutputSpec::Text { validated: false } => self.complete(Output::Text(text), fx),
        }
    }

    fn validated(
        &mut self,
        validation: ValidationId,
        verdict: OutputVerdict,
        now: Ts,
        fx: &mut Vec<AgentRunEffect>,
    ) {
        if let OutputVerdict::Failed(error) = verdict {
            let ours = match &self.phase {
                Phase::ValidatingText { validation: v, .. } => *v == validation,
                Phase::Requesting(_, a) | Phase::Acting(a) => a.finals.contains_key(&validation),
                _ => false,
            };
            if ours {
                if matches!(self.phase, Phase::ValidatingText { .. }) {
                    // This callback has already settled. Do not cancel and wait for the
                    // very validation whose failure is being processed.
                    self.phase = Phase::Idle;
                } else if let Some(a) = self.phase.acting() {
                    a.finals.remove(&validation);
                }
                self.abort(RunError::Host(error), now, fx);
            }
            return;
        }
        match &mut self.phase {
            Phase::ValidatingText { validation: v, text } if *v == validation => {
                let text = std::mem::take(text);
                match verdict {
                    OutputVerdict::Accepted { text: replaced } => {
                        self.complete(Output::Text(replaced.unwrap_or(text)), fx)
                    }
                    OutputVerdict::Retry(reason) => {
                        self.retry_output(format!("{RETRY_PROMPT}{reason}"), now, fx)
                    }
                    OutputVerdict::Failed(_) => unreachable!(),
                }
            }
            Phase::Requesting(_, acting) | Phase::Acting(acting) => {
                let Some(call) = acting.finals.remove(&validation) else {
                    return;
                };
                let rejection = match verdict {
                    OutputVerdict::Accepted { .. } => {
                        acting.accepted.insert(call, validation);
                        None
                    }
                    OutputVerdict::Retry(reason) => Some(reason),
                    OutputVerdict::Failed(_) => unreachable!(),
                };
                acting.calls.final_settled(call, rejection, fx);
                self.step_settled(now, fx);
            }
            _ => {}
        }
    }

    /// The current step's call the model made as `call_id` (the latest of that id).
    pub fn step_call_id(&self, call_id: &str) -> Option<CallId> {
        match &self.phase {
            Phase::Requesting(_, a) | Phase::Acting(a) => a.calls.find(call_id),
            _ => None,
        }
    }

    /// A call of the current step as recorded (its approval marks included), with its settlement
    /// once it has one; calls stay readable until the coordinator has handled their results.
    pub fn step_call(&self, call: CallId) -> Option<(&FunctionCall, Option<&Settled>)> {
        match &self.phase {
            Phase::Requesting(_, a) | Phase::Acting(a) => {
                Some((a.calls.item(call)?, a.calls.settled(call)))
            }
            Phase::Settling { calls, .. } => {
                let r = calls.iter().find(|r| r.id == call)?;
                Some((&r.item, Some(&r.settled)))
            }
            _ => None,
        }
    }

    fn retry_output(&mut self, feedback: String, now: Ts, fx: &mut Vec<AgentRunEffect>) {
        self.output_retries += 1;
        if self.output_retries > self.agent.retries.output {
            return self.finish(
                Err(RunError::OutputRetriesExceeded { retries: self.agent.retries.output }),
                fx,
            );
        }
        let id = self.item_id();
        self.record(ChatMessage::text(id, Role::User, feedback, self.stamp(now)).into());
        self.next_step(now, fx);
    }

    /// A tool's question: worded by the facing agent in one request without tools, unless it is
    /// verbatim or another agent asked this run; then it is said as written.
    /// Coordinated, the coordinator asks it.
    /// With `results`, the asking step also settled results the person has not heard: they
    /// are reported before the question, never dropped.
    fn ask_from_tool(
        &mut self,
        ask: RunQuestion,
        results: bool,
        now: Ts,
        fx: &mut Vec<AgentRunEffect>,
    ) {
        if self.input.opening.coordinated() {
            return self.settle_ask(ask, None, fx);
        }
        if !ask.verbatim && !self.input.delegated {
            let dynamic = self.dynamic.clone();
            let candidates: Vec<_> = self
                .input_requirements
                .iter()
                .filter(|requirement| !requirement.verbatim)
                .cloned()
                .collect();
            if candidates.len() > 1 {
                return self.request(Purpose::Present { candidates, results }, dynamic, now, fx);
            }
            return self.request(Purpose::Phrase { ask, results }, dynamic, now, fx);
        }
        if results {
            let dynamic = self.dynamic.clone();
            return self.request(Purpose::Report(ask), dynamic, now, fx);
        }
        self.asked(ask, None, &[], None, now, fx)
    }

    /// Records the question as the reply (`message_id` when a request streamed it) and settles;
    /// `lead` is the reported text the question follows.
    fn asked(
        &mut self,
        ask: RunQuestion,
        message_id: Option<String>,
        members: &[String],
        lead: Option<String>,
        now: Ts,
        fx: &mut Vec<AgentRunEffect>,
    ) {
        let streamed = message_id.is_some();
        let id = message_id.unwrap_or_else(|| self.item_id());
        if let Some(call) = &self.pending_call {
            for requirement in &mut self.input_requirements {
                if (requirement.call == *call || members.contains(&requirement.call.id))
                    && requirement.presentation_message_id.is_none()
                {
                    requirement.presentation_message_id = Some(id.clone());
                }
            }
        }
        self.record(
            ChatMessage::text(id.clone(), Role::Assistant, &ask.question, self.stamp(now)).into(),
        );
        if !streamed {
            // Exact tool questions use the same delivery boundary as model text. Merely
            // retaining a question in the result cannot establish that it was presented.
            fx.push(AgentRunEffect::Emit(StreamEvent::TextDelta {
                message_id: id.clone(),
                delta: ask.question.clone(),
            }));
            fx.push(AgentRunEffect::Emit(StreamEvent::TextEnd {
                message_id: id,
                interrupted: false,
            }));
        }
        self.settle_ask(ask, lead, fx)
    }

    fn settle_ask(&mut self, ask: RunQuestion, lead: Option<String>, fx: &mut Vec<AgentRunEffect>) {
        let exact = ask.verbatim.then(|| self.composed(Some(ask.question.clone())));
        let text = match lead {
            Some(lead) => format!("{lead}\n\n{}", ask.question),
            None => ask.question.clone(),
        };
        let question = self.composed(Some(text)).unwrap_or_default();
        let verbatim = exact.flatten().or_else(|| self.verbatim_text());
        let outcome = RunOutcome::InputRequired(RunQuestion { question, ..ask });
        self.finish(Ok(self.result(outcome, verbatim)), fx)
    }

    fn complete(&mut self, output: Output, fx: &mut Vec<AgentRunEffect>) {
        let output = match output {
            Output::Text(text) => Output::Text(self.composed(Some(text)).unwrap_or_default()),
            other => other,
        };
        let verbatim = self.verbatim_text();
        self.finish(Ok(self.result(RunOutcome::Completed { output }, verbatim)), fx)
    }

    /// The turn ended producing no output. A `str` agent whose text
    /// reached the person as written completes with it; anything else has no output.
    fn end_without_output(&mut self, fx: &mut Vec<AgentRunEffect>) {
        if matches!(self.agent.output, OutputSpec::Text { .. }) && !self.verbatim.is_empty() {
            return self.complete(Output::Text(String::new()), fx);
        }
        let verbatim = self.verbatim_text();
        self.finish(Ok(self.result(RunOutcome::NoOutput, verbatim)), fx)
    }

    /// Ends the invocation as cut by the person (a managed turn's verdict, a voice interruption).
    fn complete_interrupted(&mut self, fx: &mut Vec<AgentRunEffect>) {
        let outcome = RunOutcome::NoOutput;
        let mut result = self.result(outcome, self.verbatim_text());
        result.interrupted = true;
        self.finish_now(Ok(result), fx);
    }

    fn verbatim_text(&self) -> Option<String> {
        (!self.verbatim.is_empty()).then(|| self.verbatim.join("\n\n"))
    }

    /// The text that reached the person as written leads the answer.
    fn composed(&self, text: Option<String>) -> Option<String> {
        if self.verbatim.is_empty() {
            return text;
        }
        let mut parts = self.verbatim.clone();
        parts.extend(text.filter(|t| !t.is_empty()));
        Some(parts.join("\n\n"))
    }

    fn result(&self, outcome: RunOutcome, verbatim: Option<String>) -> RunResult {
        let mut history = self.history.clone();
        if let Some(ChatItem::Message(message)) = history.get_mut(self.first_new) {
            execution::settle(message, matches!(outcome, RunOutcome::InputRequired { .. }));
        }
        let all = ChatContext::new(history.clone())
            .copy(&CopyOptions { exclude_instructions: true, ..Default::default() });
        RunResult {
            input_requirements: self.input_requirements.clone(),
            pending_call: if matches!(outcome, RunOutcome::InputRequired { .. }) {
                self.pending_call.clone()
            } else {
                None
            },
            outcome,
            interrupted: false,
            new_messages: history
                .iter()
                .enumerate()
                .filter(|(at, item)| self.recorded_here(*at, item))
                .map(|(_, item)| item.clone())
                .collect(),
            all_messages: all.items,
            usage: self.ledger.snapshot(),
            verbatim,
        }
    }

    fn record(&mut self, item: ChatItem) {
        self.history.push(item);
    }

    fn finish(&mut self, result: Result<RunResult, RunError>, fx: &mut Vec<AgentRunEffect>) {
        if self.checkpoints
            && let Ok(result) = result
        {
            self.checkpoint_finish(result, fx);
        } else {
            self.finish_now(result, fx);
        }
    }

    fn finish_now(
        &mut self,
        mut result: Result<RunResult, RunError>,
        fx: &mut Vec<AgentRunEffect>,
    ) {
        self.ledger.settle();
        if let Ok(result) = &mut result {
            result.usage = self.ledger.snapshot();
        }
        fx.push(AgentRunEffect::Finished {
            result: Box::new(result),
            input_requirements: self.input_requirements.clone(),
        });
        self.phase = Phase::Done;
    }

    /// The latest snapshot of this invocation's usage.
    pub fn usage(&self) -> Usage {
        self.ledger.snapshot()
    }

    /// Whether this invocation called a local agent or a `RemoteAgent` as a tool, for its
    /// session's product statistics.
    pub fn session_facts(&self) -> lk_agents_abi::productstats::sessions::SessionFacts {
        self.facts
    }

    /// What the invocation consumed; it changes [`revision`](Ledger::revision) as it moves.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    fn request_started(&mut self, request: RequestId) {
        let (provider, model) = self
            .input
            .llm
            .as_ref()
            .map_or(("", ""), |l| (l.provider.provider.as_str(), l.provider.model.as_str()));
        self.ledger.start(request.0, ModelKind::Llm, provider, model);
    }
}

/// Closes the streamed text of a request, if it streamed any.
fn end_text(step: &ModelStep, interrupted: bool, fx: &mut Vec<AgentRunEffect>) {
    if step.has_text() {
        fx.push(AgentRunEffect::Emit(StreamEvent::TextEnd {
            message_id: step.message_id.clone(),
            interrupted,
        }));
    }
}
