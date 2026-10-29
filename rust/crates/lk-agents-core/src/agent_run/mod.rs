//! `AgentRun`: one agent invocation as a synchronous machine (rust-core "Agent execution"). It
//! composes a model-step child ([`model_step`]), the tool/approval child ([`crate::tool_calls`]) and the
//! output acceptance kept here, and decides when to request another step, ask the person, or
//! settle. Direct runs, TextSession and VoiceSession drive it with `handle(event, now)` and
//! `wake_at()`; the runtime executes its effects and returns their outcomes as events.
//!
//! - `mod.rs`: the invocation's events, effects and step decisions.
//! - [`result`]: the settled result, stream events and correlation ids.
//! - [`model_step`]: one model request.
//!
//! One step's calls are [`crate::tool_calls`]'s, the machine VoiceSession composes too.

mod checkpoint;
pub mod execution;
mod model_step;
pub mod result;
mod text_session;

pub use text_session::DelegationCheckpoint;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime};

use lk_agents_abi::llm::tool_executor::{ASK_IN_OWN_WORDS, Approval, ToolOutcome};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{
    ChatContext, ChatItem, ChatMessage, CopyOptions, FunctionCall, FunctionCallOutput, LlmChunk,
    Role, ToolSpec, Ts, WallAnchor,
};
use serde_json::{Map, Value};

use crate::agent::{self, AgentError, first_string_parameter};
use crate::request::{RequestSnapshot, plan_request};
use crate::tool_calls::{self, FinalCandidate, PrepareFailure, Recorded, ToolCalls};
use crate::tool_outcome;
use lk_agents_abi::agent::run::{
    Output, RunError, RunOutcome, RunResult, RunUsage, StreamEvent, ValidationId,
};
use lk_agents_abi::agent::{AgentDefinition, OutputSpec};
use lk_agents_abi::agent::{ArgumentIdentity, OutputCandidate, ReadBackAnswer};
use model_step::{ModelStep, Purpose};
pub use result::{CallId, CheckpointId, RenderId, RequestId};

/// The feedback a rejected text output gives the model.
const RETRY_PROMPT: &str = "Validation failed, fix the output and try again: ";

/// The feedback an empty response gives the model when a structured output is expected.
const EMPTY_RESPONSE_PROMPT: &str = "Please return text or call a tool.";

/// How long cancelled work may take to settle before it is reported incomplete, unless the
/// caller supplies another bound.
pub const CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

/// What the caller supplies to one invocation.
#[derive(Debug, Clone)]
pub struct RunInput {
    /// Prefixes every id the invocation mints.
    pub invocation_id: String,
    pub message: ChatMessage,
    /// The conversation so far.
    pub history: Vec<ChatItem>,
    /// The previous invocation asked with `InputRequired(tool=)`: this message is that tool's.
    pub answer_tool: Option<String>,
    /// Another agent asked this run: questions pass up as written; the facing agent words them.
    pub delegated: bool,
    /// The coordinator reading the outcome table.
    pub mode: tool_outcome::Mode,
    /// The agent's model as the runtime resolved it; an invocation without one cannot run a step.
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
pub enum Event {
    Start,
    HostCheckpointDone {
        checkpoint: CheckpointId,
        result: Result<bool, String>,
        tools: Option<Vec<lk_agents_abi::agent::FunctionToolInfo>>,
    },
    InstructionsRendered {
        render: RenderId,
        result: Result<String, String>,
    },
    LlmChunk {
        request: RequestId,
        chunk: LlmChunk,
    },
    LlmEnded {
        request: RequestId,
    },
    LlmFailed {
        request: RequestId,
        error: String,
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
        result: Result<ReadBackAnswer, String>,
    },
    /// `ctx.update(text)`.
    ToolProgress {
        call: CallId,
        message: String,
    },
    ToolCallUpdated {
        call: CallId,
        extra: Map<String, Value>,
    },
    ToolSettled {
        call: CallId,
        outcome: ToolOutcome,
    },
    ToolLaunchFailed {
        call: CallId,
        error: String,
    },
    OutputValidated {
        validation: ValidationId,
        verdict: OutputVerdict,
    },
    Cancel,
    /// The deadline from `wake_at()` is due.
    Wake,
}

/// Work for the runtime. Every outcome-bearing effect is answered by a correlated event.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    HostCheckpoint {
        checkpoint: CheckpointId,
        list_tools: bool,
    },
    HostCheckpointCancel {
        checkpoint: CheckpointId,
    },
    /// Render the `@agent.instructions` functions.
    InstructionsRender {
        render: RenderId,
    },
    InstructionsCancel {
        render: RenderId,
    },
    LlmRequestStart {
        request: RequestId,
        llm: ResourceId,
        chat_ctx: ChatContext,
        tools: Vec<ToolSpec>,
    },
    LlmRequestCancel {
        request: RequestId,
    },
    /// Bind and validate the arguments in the call's invocation scope.
    ToolPrepare {
        call: CallId,
        arguments: Map<String, Value>,
        function_call: FunctionCall,
    },
    /// Run the tool's read-back function on the prepared arguments.
    ReadBackRender {
        call: CallId,
        tool: String,
        prepared: ResourceId,
    },
    /// Run the body with the prepared arguments; `function_call` carries its approval marks and
    /// `approval` is the scope the body runs under.
    ToolExecute {
        call: CallId,
        prepared: ResourceId,
        function_call: FunctionCall,
        approval: Option<Approval>,
    },
    /// Cancel whatever host work the call has in flight.
    ToolCancel {
        call: CallId,
    },
    /// Prepared arguments no body will run with.
    ToolArgumentsRelease {
        prepared: ResourceId,
    },
    OutputValidate {
        validation: ValidationId,
        candidate: OutputCandidate,
    },
    OutputValidateCancel {
        validation: ValidationId,
    },
    Emit(StreamEvent),
    /// Diagnostic: malformed model arguments were repaired.
    ArgumentsRepaired {
        call_id: String,
        error: String,
        repaired: Value,
    },
    /// Cancelled work did not settle by its deadline; the runtime still owns it.
    IncompleteCleanup {
        pending: Vec<String>,
    },
    /// The invocation settled; no later effect follows.
    Finished(Result<RunResult, RunError>),
}

impl From<tool_calls::Effect> for Effect {
    fn from(effect: tool_calls::Effect) -> Self {
        match effect {
            tool_calls::Effect::Prepare { call, arguments, function_call } => {
                Effect::ToolPrepare { call, arguments, function_call }
            }
            tool_calls::Effect::ReadBack { call, tool, prepared } => {
                Effect::ReadBackRender { call, tool, prepared }
            }
            tool_calls::Effect::Execute { call, prepared, function_call, approval } => {
                Effect::ToolExecute { call, prepared, function_call, approval }
            }
            tool_calls::Effect::Cancel { call } => Effect::ToolCancel { call },
            tool_calls::Effect::ReleaseArguments { prepared } => {
                Effect::ToolArgumentsRelease { prepared }
            }
            tool_calls::Effect::Emit(event) => Effect::Emit(event),
            tool_calls::Effect::ArgumentsRepaired { call_id, error, repaired } => {
                Effect::ArgumentsRepaired { call_id, error, repaired }
            }
        }
    }
}

/// A question the invocation ends on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ask {
    question: String,
    verbatim: bool,
    answer_tool: Option<String>,
}

pub struct AgentRun {
    agent: AgentDefinition,
    input: RunInput,
    /// The supplied history with this invocation's items.
    history: Vec<ChatItem>,
    /// Index of the incoming message in `history`: `new_messages` start there.
    first_new: usize,
    next_id: u64,
    usage: RunUsage,
    /// What reached the person as written, in order: it leads the answer.
    verbatim: Vec<String>,
    tool_retries: tool_outcome::ToolRetries,
    output_retries: u32,
    /// The dynamic instructions last rendered, for a phrasing request.
    dynamic: Option<String>,
    approval_answers: BTreeSet<String>,
    checkpoints: bool,
    phase: Phase,
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

impl Acting {
    /// Cancels every call and validation still in flight; their settlement is owed.
    fn cancel(mut self, owed: &mut Owed, fx: &mut Vec<Effect>) {
        owed.calls.extend(self.calls.cancel(fx));
        for validation in self.finals.into_keys() {
            fx.push(Effect::OutputValidateCancel { validation });
            owed.validations.insert(validation);
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

/// Cancelled work whose settlement is still owed.
#[derive(Default)]
struct Owed {
    checkpoint: Option<CheckpointId>,
    request: Option<RequestId>,
    render: Option<RenderId>,
    calls: BTreeSet<CallId>,
    validations: BTreeSet<ValidationId>,
}

impl Owed {
    fn is_empty(&self) -> bool {
        self.checkpoint.is_none()
            && self.request.is_none()
            && self.render.is_none()
            && self.calls.is_empty()
            && self.validations.is_empty()
    }

    fn describe(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.extend(self.checkpoint.map(|c| format!("managed checkpoint {}", c.0)));
        out.extend(self.request.map(|r| format!("llm request {}", r.0)));
        out.extend(self.render.map(|r| format!("instructions render {}", r.0)));
        out.extend(self.calls.iter().map(|c| format!("tool call {}", c.0)));
        out.extend(self.validations.iter().map(|v| format!("output validation {}", v.0)));
        out
    }
}

impl AgentRun {
    pub fn new(agent: AgentDefinition, mut input: RunInput) -> Result<AgentRun, AgentError> {
        agent::validate(&agent)?;
        execution::bind(&mut input.message, &input.history, None, &input.invocation_id);
        Ok(AgentRun {
            agent,
            history: Vec::new(),
            first_new: 0,
            input,
            next_id: 0,
            usage: RunUsage::default(),
            verbatim: Vec::new(),
            tool_retries: tool_outcome::ToolRetries::default(),
            output_retries: 0,
            dynamic: None,
            approval_answers: BTreeSet::new(),
            checkpoints: false,
            phase: Phase::Idle,
        })
    }

    pub fn wake_at(&self) -> Option<Ts> {
        match &self.phase {
            Phase::Aborting { deadline, .. } => Some(*deadline),
            _ => None,
        }
    }

    pub fn handle(&mut self, event: Event, now: Ts) -> Vec<Effect> {
        let mut fx = Vec::new();
        match event {
            Event::Start => self.start(now, &mut fx),
            Event::Cancel => self.abort(RunError::Cancelled, now, &mut fx),
            Event::Wake => self.wake(now, &mut fx),
            event if matches!(self.phase, Phase::Aborting { .. } | Phase::Done) => {
                self.settle_owed(event, &mut fx)
            }
            Event::HostCheckpointDone { checkpoint, result, tools } => {
                self.checkpoint_done(checkpoint, result, tools, now, &mut fx)
            }
            Event::InstructionsRendered { render, result } => match self.phase {
                Phase::Rendering(r) if r == render => match result {
                    Ok(dynamic) => {
                        self.dynamic = Some(dynamic.clone());
                        self.request(Purpose::Turn, Some(dynamic), now, &mut fx)
                    }
                    Err(e) => self.abort(RunError::Host(e), now, &mut fx),
                },
                _ => {}
            },
            Event::LlmChunk { request, chunk } => self.chunk(request, chunk, now, &mut fx),
            Event::LlmEnded { request } => self.response(request, now, &mut fx),
            Event::LlmFailed { request, error } => {
                if matches!(&self.phase, Phase::Requesting(s, _) if s.request == request) {
                    // The provider already recovered what it could; the calls the response made
                    // so far are cancelled.
                    let Phase::Requesting(step, acting) =
                        std::mem::replace(&mut self.phase, Phase::Idle)
                    else {
                        unreachable!()
                    };
                    end_text(&step, true, &mut fx);
                    self.phase = Phase::Acting(acting);
                    self.abort(RunError::Model(error), now, &mut fx);
                }
            }
            Event::ToolPrepared { call, prepared, identity } => {
                let held = match self.phase.acting() {
                    Some(a) => a.calls.prepared(
                        call,
                        prepared,
                        identity,
                        &self.agent.tools,
                        self.history.iter(),
                        &mut fx,
                    ),
                    None => false,
                };
                if !held {
                    fx.push(Effect::ToolArgumentsRelease { prepared });
                }
                self.step_settled(now, &mut fx);
            }
            Event::ToolPrepareFailed { call, failure } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.prepare_failed(
                        call,
                        failure,
                        &self.agent.tools,
                        self.history.iter(),
                        &mut fx,
                    );
                }
                self.step_settled(now, &mut fx);
            }
            Event::ReadBackRendered { call, result } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.read_back(
                        call,
                        result,
                        &self.agent.tools,
                        self.history.iter(),
                        &mut fx,
                    );
                }
                self.step_settled(now, &mut fx);
            }
            Event::ToolProgress { call, message } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.progress(call, message, &mut fx);
                }
            }
            Event::ToolCallUpdated { call, extra } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.update_extra(call, extra);
                }
            }
            Event::ToolSettled { call, outcome } => {
                let cancelled = if let Some(a) = self.phase.acting() {
                    a.calls.body_settled(call, outcome, &mut fx)
                } else {
                    false
                };
                if cancelled {
                    self.abort(RunError::Cancelled, now, &mut fx);
                } else {
                    self.step_settled(now, &mut fx);
                }
            }
            Event::ToolLaunchFailed { call, .. } => {
                if let Some(a) = self.phase.acting() {
                    a.calls.body_settled(call, ToolOutcome::Internal, &mut fx);
                }
                self.step_settled(now, &mut fx);
            }
            Event::OutputValidated { validation, verdict } => {
                self.validated(validation, verdict, now, &mut fx)
            }
        }
        fx
    }

    fn mint(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn item_id(&mut self) -> String {
        let n = self.mint();
        format!("{}/item_{n}", self.input.invocation_id)
    }

    fn stamp(&self, now: Ts) -> SystemTime {
        let secs = self.input.wall.seconds(now);
        SystemTime::UNIX_EPOCH + Duration::try_from_secs_f64(secs.max(0.0)).unwrap_or_default()
    }

    fn start(&mut self, now: Ts, fx: &mut Vec<Effect>) {
        if !matches!(self.phase, Phase::Idle) {
            return;
        }
        let Some(_) = &self.input.llm else {
            return self.finish(Err(RunError::NoModel), fx);
        };
        self.history = std::mem::take(&mut self.input.history);
        let answer = self.input.message.raw_text_content().unwrap_or_default();
        self.approval_answers = crate::request::answer_pending_input(&mut self.history, &answer);
        // A transport (the A2A door) may already have written the answer: the read-backs of the
        // execution this message continues were still waiting, so this invocation answers them.
        if let Some((at, _)) = execution::pending_turn(&self.history) {
            for item in &self.history[at..] {
                if let ChatItem::FunctionCall(call) = item
                    && crate::approval::mark(call) == Some(Approval::Asked)
                {
                    self.approval_answers.insert(call.call_id.clone());
                }
            }
        }
        self.first_new = self.history.len();
        self.history.push(ChatItem::Message(self.input.message.clone()));
        // The answer is that tool's: it takes it as its first string parameter, no model turn.
        let handed = self.input.answer_tool.clone().and_then(|name| {
            let parameter =
                first_string_parameter(&agent::tool(&self.agent, &name)?.spec)?.to_owned();
            Some((name, parameter))
        });
        match handed {
            Some((name, parameter)) => {
                let id = self.item_id();
                let arguments = Value::Object(Map::from_iter([(parameter, Value::String(answer))]));
                let call = FunctionCall {
                    call_id: format!("{id}/answer"),
                    id,
                    arguments: arguments.to_string(),
                    name,
                    created_at: self.stamp(now),
                    extra: Default::default(),
                    group_id: None,
                };
                self.phase = Phase::Acting(Acting::new(&self.approval_answers));
                self.admit(call, fx);
                self.step_settled(now, fx);
            }
            None => self.next_step(now, fx),
        }
    }

    /// The next model step: dynamic instructions first when there are any.
    fn next_step(&mut self, now: Ts, fx: &mut Vec<Effect>) {
        if self.agent.dynamic_instructions {
            let render = RenderId(self.mint());
            fx.push(Effect::InstructionsRender { render });
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
        fx: &mut Vec<Effect>,
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
        fx: &mut Vec<Effect>,
    ) {
        let Some(llm) = self.input.llm.as_ref().map(|l| l.id) else {
            return self.finish(Err(RunError::NoModel), fx);
        };
        let instructions =
            agent::instructions_with(&self.agent, dynamic.as_deref().unwrap_or_default());
        let mut tools: Vec<ToolSpec> = Vec::new();
        let mut phrase = None;
        match &purpose {
            Purpose::Turn => {
                tools.extend(self.agent.tools.iter().map(|t| t.spec.clone()));
                tools.extend(agent::final_result_tool(&self.agent));
            }
            Purpose::Phrase(ask) => phrase = Some(format!("{ASK_IN_OWN_WORDS}{}", ask.question)),
        }
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
        self.usage.requests += 1;
        fx.push(Effect::LlmRequestStart {
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
    fn chunk(&mut self, request: RequestId, chunk: LlmChunk, now: Ts, fx: &mut Vec<Effect>) {
        let Phase::Requesting(step, _) = &mut self.phase else { return };
        if step.request != request {
            return;
        }
        let added = step.chunk(chunk);
        let message_id = step.message_id.clone();
        if let Some(delta) = added.text {
            let message_id = message_id.clone();
            fx.push(Effect::Emit(StreamEvent::TextDelta { message_id, delta }));
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
    fn admit(&mut self, call: FunctionCall, fx: &mut Vec<Effect>) {
        self.usage.tool_calls += 1;
        let id = CallId(self.mint());
        let Some(acting) = self.phase.acting() else { return };
        let Some(FinalCandidate { call, value }) =
            acting.calls.add_step_call(id, call, &self.agent, fx)
        else {
            return;
        };
        let value = match value {
            Ok(value) => value,
            Err(reason) => return acting.calls.final_settled(call, Some(reason), fx),
        };
        let validation = ValidationId(self.mint());
        let candidate = OutputCandidate::Value(value);
        fx.push(Effect::OutputValidate { validation, candidate });
        if let Some(acting) = self.phase.acting() {
            acting.finals.insert(validation, call);
        }
    }

    fn response(&mut self, request: RequestId, now: Ts, fx: &mut Vec<Effect>) {
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
        if let Some(usage) = &response.usage {
            self.usage.input_tokens += u64::from(usage.prompt_tokens);
            self.usage.output_tokens += u64::from(usage.completion_tokens);
        }
        if let Purpose::Phrase(ask) = purpose {
            let text = response.text.trim();
            let question = if text.is_empty() { ask.question.clone() } else { text.to_owned() };
            return self.asked(Ask { question, ..ask }, Some(message_id), now, fx);
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

    fn text_output(&mut self, text: String, now: Ts, fx: &mut Vec<Effect>) {
        match &self.agent.output {
            OutputSpec::Structured { .. } if text.is_empty() => {
                self.retry_output(EMPTY_RESPONSE_PROMPT.to_owned(), now, fx)
            }
            // Conversational text where a typed output is expected asks the person.
            OutputSpec::Structured { .. } => {
                self.settle_ask(Ask { question: text, verbatim: false, answer_tool: None }, fx)
            }
            OutputSpec::Text { validated: true } => {
                let validation = ValidationId(self.mint());
                fx.push(Effect::OutputValidate {
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
        fx: &mut Vec<Effect>,
    ) {
        if let OutputVerdict::Failed(error) = verdict {
            let ours = match &self.phase {
                Phase::ValidatingText { validation: v, .. } => *v == validation,
                Phase::Requesting(_, a) | Phase::Acting(a) => a.finals.contains_key(&validation),
                _ => false,
            };
            if ours {
                if let Some(a) = self.phase.acting() {
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
                let Some(call) = acting.finals.remove(&validation) else { return };
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

    /// Once every call of the step settled: record the calls and outputs in model order, then
    /// continue, ask, or settle.
    fn step_settled(&mut self, now: Ts, fx: &mut Vec<Effect>) {
        let settled =
            matches!(&self.phase, Phase::Acting(a) if a.finals.is_empty() && a.calls.all_settled());
        if !settled {
            return;
        }
        let Phase::Acting(acting) = std::mem::replace(&mut self.phase, Phase::Idle) else {
            unreachable!()
        };
        let recorded = acting.calls.finish();
        // Retry budgets: consecutive failures per tool, any success resets.
        let structured = matches!(self.agent.output, OutputSpec::Structured { .. });
        let is_final = |r: &&Recorded| structured && r.item.name == crate::agent::FINAL_RESULT_TOOL;
        let output_rejected = (recorded.iter().filter(is_final))
            .any(|r| matches!(r.settled.outcome, ToolOutcome::ToolError(_)));
        let tools = recorded.iter().filter(|r| !is_final(r));
        let mut exceeded = self.tool_retries.step(
            tools.map(|r| (r.item.name.as_str(), r.settled.retry)),
            &self.agent.tools,
            self.agent.retries.tools,
        );
        let accepted = acting.accepted.first_key_value().map(|(c, v)| (*c, *v));
        if output_rejected && accepted.is_none() {
            self.output_retries += 1;
            if self.output_retries > self.agent.retries.output {
                exceeded.get_or_insert(RunError::OutputRetriesExceeded {
                    retries: self.agent.retries.output,
                });
            }
        }
        // The first terminal outcome in model order decides; the others are recorded as answered.
        let mut terminal: Option<Terminal> = None;
        let mut replies = false;
        let mut items = Vec::with_capacity(recorded.len() * 2);
        for Recorded { id, item, settled } in recorded {
            let outcome = &settled.outcome;
            let shape = tool_outcome::shape(outcome, self.input.mode);
            let mut text = tool_outcome::delivered(outcome).unwrap_or(shape.text);
            let mut reply_required = shape.reply_required;
            if let Some(said) = tool_outcome::verbatim(outcome) {
                self.verbatim.push(said.to_owned());
            }
            let this = match outcome {
                _ if accepted.is_some_and(|(call, _)| call == id) => Some(Terminal::Output),
                ToolOutcome::InputRequired { question, verbatim, tool } => {
                    if terminal.is_some() {
                        // Not the question asked: its text is recorded as answered.
                        text = question.clone();
                    }
                    Some(Terminal::Ask(Ask {
                        question: question.clone(),
                        verbatim: *verbatim,
                        answer_tool: tool.clone(),
                    }))
                }
                ToolOutcome::StopResponse => Some(Terminal::Stop),
                _ if tool_outcome::verbatim(outcome).is_some() && !shape.reply_required => {
                    Some(Terminal::Stop)
                }
                _ => None,
            };
            if terminal.is_none() {
                terminal = this;
            } else if matches!(outcome, ToolOutcome::InputRequired { .. }) {
                reply_required = false;
            }
            replies |= reply_required;
            let mut output = FunctionCallOutput::new(
                format!("{}_out", item.id),
                item.call_id.clone(),
                text,
                shape.is_error,
                self.stamp(now),
            );
            output.name = item.name.clone();
            output.reply_required = reply_required;
            items.push(ChatItem::FunctionCall(item));
            items.push(ChatItem::FunctionCallOutput(output));
        }
        for item in items {
            self.record(item);
        }
        if let Some(error) = exceeded {
            return self.finish(Err(error), fx);
        }
        match terminal {
            Some(Terminal::Output) => {
                let (_, validation) = accepted.expect("an accepted candidate is the terminal");
                self.complete(Output::Validated(validation), fx)
            }
            Some(Terminal::Ask(ask)) => self.ask_from_tool(ask, now, fx),
            Some(Terminal::Stop) => self.complete(Output::Nothing, fx),
            // A step whose every output asks for no reply ends the turn there.
            None if !replies => self.complete(Output::Nothing, fx),
            None => self.next_step(now, fx),
        }
    }

    fn retry_output(&mut self, feedback: String, now: Ts, fx: &mut Vec<Effect>) {
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
    fn ask_from_tool(&mut self, ask: Ask, now: Ts, fx: &mut Vec<Effect>) {
        if !ask.verbatim && !self.input.delegated {
            let dynamic = self.dynamic.clone();
            return self.request(Purpose::Phrase(ask), dynamic, now, fx);
        }
        self.asked(ask, None, now, fx)
    }

    /// Records the question as the reply (`message_id` when a request streamed it) and settles.
    fn asked(&mut self, ask: Ask, message_id: Option<String>, now: Ts, fx: &mut Vec<Effect>) {
        let id = message_id.unwrap_or_else(|| self.item_id());
        self.record(ChatMessage::text(id, Role::Assistant, &ask.question, self.stamp(now)).into());
        self.settle_ask(ask, fx)
    }

    fn settle_ask(&mut self, ask: Ask, fx: &mut Vec<Effect>) {
        let question = self.composed(Some(ask.question.clone())).unwrap_or_default();
        let verbatim = if ask.verbatim { Some(question.clone()) } else { self.verbatim_text() };
        let outcome = RunOutcome::InputRequired {
            question,
            verbatim: ask.verbatim,
            answer_tool: ask.answer_tool,
        };
        self.finish(Ok(self.result(outcome, verbatim)), fx)
    }

    fn complete(&mut self, output: Output, fx: &mut Vec<Effect>) {
        let output = match output {
            Output::Text(text) => Output::Text(self.composed(Some(text)).unwrap_or_default()),
            Output::Nothing if !self.verbatim.is_empty() => {
                Output::Text(self.composed(None).unwrap_or_default())
            }
            other => other,
        };
        let verbatim = self.verbatim_text();
        self.finish(Ok(self.result(RunOutcome::Completed { output }, verbatim)), fx)
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
            outcome,
            interrupted: false,
            new_messages: history[self.first_new..].to_vec(),
            all_messages: all.items,
            usage: self.usage,
            verbatim,
        }
    }

    fn record(&mut self, item: ChatItem) {
        self.history.push(item);
    }

    fn finish(&mut self, result: Result<RunResult, RunError>, fx: &mut Vec<Effect>) {
        if self.checkpoints
            && let Ok(result) = result
        {
            self.checkpoint_finish(result, fx);
        } else {
            self.finish_now(result, fx);
        }
    }

    fn finish_now(&mut self, result: Result<RunResult, RunError>, fx: &mut Vec<Effect>) {
        fx.push(Effect::Finished(result));
        self.phase = Phase::Done;
    }

    /// Cancels everything in flight; settles once it all reported back, or at the deadline.
    fn abort(&mut self, error: RunError, now: Ts, fx: &mut Vec<Effect>) {
        let mut owed = Owed::default();
        match std::mem::replace(&mut self.phase, Phase::Idle) {
            Phase::Idle => {}
            Phase::Checkpoint(pending) => {
                fx.push(Effect::HostCheckpointCancel { checkpoint: pending.id });
                owed.checkpoint = Some(pending.id);
            }
            Phase::Done => {
                self.phase = Phase::Done;
                return;
            }
            phase @ Phase::Aborting { .. } => {
                self.phase = phase;
                return;
            }
            Phase::Rendering(render) => {
                fx.push(Effect::InstructionsCancel { render });
                owed.render = Some(render);
            }
            Phase::Requesting(step, acting) => {
                end_text(&step, true, fx);
                fx.push(Effect::LlmRequestCancel { request: step.request });
                owed.request = Some(step.request);
                acting.cancel(&mut owed, fx);
            }
            Phase::Acting(acting) => acting.cancel(&mut owed, fx),
            Phase::ValidatingText { validation, .. } => {
                fx.push(Effect::OutputValidateCancel { validation });
                owed.validations.insert(validation);
            }
        }
        if owed.is_empty() {
            return self.finish(Err(error), fx);
        }
        let deadline = now.saturating_add(self.input.cleanup_timeout);
        self.phase = Phase::Aborting { error, deadline, owed };
    }

    /// An event for work cancelled (or finished) earlier: it settles what is owed, and a resource
    /// it returns is released rather than adopted.
    fn settle_owed(&mut self, event: Event, fx: &mut Vec<Effect>) {
        if let Event::ToolPrepared { prepared, .. } = &event {
            fx.push(Effect::ToolArgumentsRelease { prepared: *prepared });
        }
        let Phase::Aborting { owed, .. } = &mut self.phase else { return };
        match event {
            Event::HostCheckpointDone { checkpoint, .. } if owed.checkpoint == Some(checkpoint) => {
                owed.checkpoint = None;
            }
            Event::InstructionsRendered { render, .. } if owed.render == Some(render) => {
                owed.render = None
            }
            Event::LlmEnded { request } | Event::LlmFailed { request, .. }
                if owed.request == Some(request) =>
            {
                owed.request = None
            }
            Event::ToolPrepared { call, .. }
            | Event::ToolPrepareFailed { call, .. }
            | Event::ReadBackRendered { call, .. }
            | Event::ToolSettled { call, .. }
            | Event::ToolLaunchFailed { call, .. } => {
                owed.calls.remove(&call);
            }
            Event::OutputValidated { validation, .. } => {
                owed.validations.remove(&validation);
            }
            _ => {}
        }
        if owed.is_empty() {
            let Phase::Aborting { error, .. } = std::mem::replace(&mut self.phase, Phase::Idle)
            else {
                unreachable!()
            };
            self.finish(Err(error), fx);
        }
    }

    fn wake(&mut self, now: Ts, fx: &mut Vec<Effect>) {
        let due = matches!(&self.phase, Phase::Aborting { deadline, .. } if *deadline <= now);
        if !due {
            return;
        }
        let Phase::Aborting { error, owed, .. } = std::mem::replace(&mut self.phase, Phase::Idle)
        else {
            unreachable!()
        };
        fx.push(Effect::IncompleteCleanup { pending: owed.describe() });
        self.finish(Err(error), fx);
    }
}

/// What a step ends the invocation with.
enum Terminal {
    Output,
    Ask(Ask),
    Stop,
}

/// Closes the streamed text of a request, if it streamed any.
fn end_text(step: &ModelStep, interrupted: bool, fx: &mut Vec<Effect>) {
    if step.has_text() {
        fx.push(Effect::Emit(StreamEvent::TextEnd {
            message_id: step.message_id.clone(),
            interrupted,
        }));
    }
}
