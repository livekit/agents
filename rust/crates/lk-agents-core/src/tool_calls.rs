//! The tool calls of one model step (rust-core "Shared tool execution"): parse and canonicalise
//! the model's arguments, have the host prepare them, bind approval to the validated identity,
//! admit duplicates per the tool's policy, run the body, and record each outcome as it arrives.
//! Results are assembled in model order by the coordinator; waiting on one call never holds back
//! another's lifecycle. `AgentRun` and VoiceSession compose the same machine: the coordinator
//! executes its [`Effect`]s and returns the outcomes.

use std::collections::BTreeSet;

use crate::llm::tool_arguments;
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::llm::tool_executor::{Approval, ToolOutcome};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ChatItem, FunctionCall, ToolSpec};
use serde_json::{Map, Value};

use crate::agent::{self, FINAL_RESULT_TOOL, WRAPPED_RESULT_KEY};
use crate::agent_run::CallId;
use crate::approval;
use crate::tool_outcome;
use lk_agents_abi::agent::run::{StreamEvent, ToolResultStatus};
use lk_agents_abi::agent::{AgentDefinition, FunctionToolInfo, OutputSpec, ReadBack};
use lk_agents_abi::agent::{ArgumentIdentity, ReadBackAnswer};

/// Host work for the coordinator. Every outcome-bearing effect is answered for its call.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    /// Bind and validate the arguments (framework-parsed, defaults applied) in the call's scope.
    Prepare {
        call: CallId,
        arguments: Map<String, Value>,
        function_call: FunctionCall,
    },
    /// Run the tool's read-back function on the prepared arguments.
    ReadBack {
        call: CallId,
        tool: String,
        prepared: ResourceId,
    },
    /// Run the body with the prepared arguments; `function_call` carries its approval marks and
    /// `approval` is the scope the body runs under.
    Execute {
        call: CallId,
        prepared: ResourceId,
        function_call: FunctionCall,
        approval: Option<Approval>,
    },
    /// Cancel whatever host work the call has in flight.
    Cancel {
        call: CallId,
    },
    /// Prepared arguments no body will run with.
    ReleaseArguments {
        prepared: ResourceId,
    },
    Emit(StreamEvent),
    /// Diagnostic: malformed model arguments were repaired.
    ArgumentsRepaired {
        call_id: String,
        error: String,
        repaired: Value,
    },
}

/// The parameter the model sets to re-assert a call of an `on_duplicate="confirm"` tool.
use lk_agents_abi::llm::provider_format::tools::CONFIRM_DUPLICATE_PARAM;

/// The output of a call refused because the same tool is still running.
fn duplicate_text(tool: &str, running: &[&FunctionCall], confirm: bool) -> String {
    let calls: Vec<String> = running
        .iter()
        .map(|c| format!("{{\"call_id\": {:?}, \"arguments\": {:?}}}", c.call_id, c.arguments))
        .collect();
    let next = if confirm {
        format!(
            "Call it again with `{CONFIRM_DUPLICATE_PARAM}` set to true to run a duplicate, only \
             when the user explicitly asks for it."
        )
    } else {
        "Do not call it again until it has finished.".to_owned()
    };
    format!("Same tool `{tool}` is already running:\n{}\n{next}", calls.join("\n"))
}

#[derive(Debug)]
pub struct ToolCalls {
    calls: Vec<Call>,
    /// The pending calls the invocation's incoming message answered: only these read-backs can
    /// grant a repeat.
    approval_answers: BTreeSet<String>,
}

#[derive(Debug)]
struct Call {
    id: CallId,
    item: FunctionCall,
    phase: Phase,
    cancellation_requested: bool,
}

#[derive(Debug)]
enum Phase {
    /// The host binds and validates the arguments.
    Preparing {
        confirmed: bool,
    },
    /// Prepared; waits until every earlier call of the same tool is decided.
    Admitting {
        prepared: ResourceId,
        identity: ArgumentIdentity,
        confirmed: bool,
    },
    /// The application's read-back function runs.
    ReadingBack {
        prepared: ResourceId,
        identity: ArgumentIdentity,
    },
    Running {
        identity: String,
    },
    /// A `final_result` candidate the parent validates.
    Validating,
    Settled(Settled),
}

/// A call's recorded end.
#[derive(Debug, Clone, PartialEq)]
pub struct Settled {
    pub outcome: ToolOutcome,
    /// Feedback that counts against the tool's retry budget: invalid or unknown arguments,
    /// `ModelRetry`.
    pub retry: bool,
}

/// What [`ToolCalls::begin`] leaves to the parent.
pub(crate) struct FinalCandidate {
    pub(crate) call: CallId,
    pub(crate) value: Result<Value, String>,
}

/// One settled call, in model order.
pub struct Recorded {
    pub id: CallId,
    /// The call as recorded: its approval marks included.
    pub item: FunctionCall,
    pub settled: Settled,
}

impl ToolCalls {
    /// Starts one call of an `AgentRun` step as the model makes it. A call that cannot run
    /// settles at once; a `final_result` call is returned as a candidate for the parent's output
    /// acceptance.
    pub(crate) fn add_step_call<E: From<Effect>>(
        &mut self,
        id: CallId,
        item: FunctionCall,
        agent: &AgentDefinition,
        fx: &mut Vec<E>,
    ) -> Option<FinalCandidate> {
        if item.name == FINAL_RESULT_TOOL && matches!(agent.output, OutputSpec::Structured { .. }) {
            let candidate = FinalCandidate { call: id, value: final_value(agent, &item.arguments) };
            let phase = Phase::Validating;
            self.calls.push(Call { id, item, phase, cancellation_requested: false });
            return Some(candidate);
        }
        self.add(id, item, &agent.tools, fx);
        None
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// A step whose calls may consume the read-backs answered by `approval_answers`.
    pub fn new(approval_answers: BTreeSet<String>) -> Self {
        ToolCalls { calls: Vec::new(), approval_answers }
    }

    /// The read-backs the step's later calls can consume, as the person's latest message left them.
    pub(crate) fn answered(&mut self, approval_answers: BTreeSet<String>) {
        self.approval_answers = approval_answers;
    }

    /// Starts one call as the model makes it, against the tools its request offered: an unknown
    /// tool or arguments that are not a JSON object settle at once as feedback.
    pub fn add<E: From<Effect>>(
        &mut self,
        id: CallId,
        item: FunctionCall,
        tools: &[FunctionToolInfo],
        fx: &mut Vec<E>,
    ) {
        let phase = start(id, &item, tools, fx);
        self.calls.push(Call { id, item, phase, cancellation_requested: false });
    }

    /// The host prepared the call's arguments; `history` is what approval reads, the calls
    /// decided before this step with their outputs. `false` when the call was not preparing.
    pub fn prepared<'h, E: From<Effect>>(
        &mut self,
        id: CallId,
        prepared: ResourceId,
        identity: ArgumentIdentity,
        tools: &[FunctionToolInfo],
        history: impl Iterator<Item = &'h ChatItem> + Clone,
        fx: &mut Vec<E>,
    ) -> bool {
        let Some(call) = self.call_mut(id) else { return false };
        let Phase::Preparing { confirmed } = call.phase else { return false };
        call.phase = Phase::Admitting { prepared, identity, confirmed };
        self.admit(tools, history, fx);
        true
    }

    /// Binding or validation failed: the error is the model's feedback and counts as a retry.
    pub fn prepare_failed<'h, E: From<Effect>>(
        &mut self,
        id: CallId,
        failure: PrepareFailure,
        tools: &[FunctionToolInfo],
        history: impl Iterator<Item = &'h ChatItem> + Clone,
        fx: &mut Vec<E>,
    ) {
        let Some(call) = self.call_mut(id) else { return };
        if !matches!(call.phase, Phase::Preparing { .. }) {
            return;
        }
        let outcome = match failure {
            PrepareFailure::Refused(feedback) => ToolOutcome::ToolError(feedback),
            PrepareFailure::Internal => ToolOutcome::Internal,
        };
        settle(call, outcome, true, fx);
        self.admit(tools, history, fx);
    }

    pub fn read_back<'h, E: From<Effect>>(
        &mut self,
        id: CallId,
        answer: Result<ReadBackAnswer, String>,
        tools: &[FunctionToolInfo],
        history: impl Iterator<Item = &'h ChatItem> + Clone,
        fx: &mut Vec<E>,
    ) {
        let Some(call) = self.call_mut(id) else { return };
        let Phase::ReadingBack { prepared, ref identity } = call.phase else { return };
        let canonical = approval::canonical(identity);
        fx.push(Effect::ReleaseArguments { prepared }.into());
        let outcome = match answer {
            Ok(ReadBackAnswer::Question(question)) => {
                approval::set_mark(&mut call.item, Approval::Asked, &canonical);
                ToolOutcome::InputRequired { question, verbatim: true, tool: None }
            }
            // Something is missing before there is anything to approve: not a read-back.
            Ok(ReadBackAnswer::InputRequired { question, verbatim }) => {
                ToolOutcome::InputRequired { question, verbatim, tool: None }
            }
            Err(_) => ToolOutcome::Internal,
        };
        settle(call, outcome, false, fx);
        self.admit(tools, history, fx);
    }

    pub fn progress<E: From<Effect>>(&mut self, id: CallId, message: String, fx: &mut Vec<E>) {
        if let Some(call) = self.call_mut(id).filter(|c| matches!(c.phase, Phase::Running { .. })) {
            fx.push(
                Effect::Emit(StreamEvent::Progress {
                    call_id: call.item.call_id.clone(),
                    tool_name: call.item.name.clone(),
                    message,
                })
                .into(),
            );
        }
    }

    pub fn update_extra(&mut self, id: CallId, extra: Map<String, Value>) {
        let Some(call) = self.call_mut(id) else { return };
        if matches!(call.phase, Phase::Running { .. }) {
            call.item.extra.extend(extra);
        }
    }

    /// The body's outcome; `true` when it reports a cancellation nobody asked for. An update
    /// is progress: the tool runs on.
    pub fn body_settled<E: From<Effect>>(
        &mut self,
        id: CallId,
        outcome: ToolOutcome,
        fx: &mut Vec<E>,
    ) -> bool {
        if let ToolOutcome::Update { raw, .. } = outcome {
            self.progress(id, raw, fx);
            return false;
        }
        let Some(call) = self.call_mut(id) else { return false };
        if !matches!(call.phase, Phase::Running { .. }) {
            return false;
        }
        let cancelled = matches!(outcome, ToolOutcome::Cancelled) && !call.cancellation_requested;
        let retry = matches!(outcome, ToolOutcome::ModelRetry(_));
        settle(call, outcome, retry, fx);
        cancelled
    }

    /// A body that released its step with its first update (`ctx.update()` on a call): the
    /// update answers the call, and the body runs on outside the step.
    pub fn detached<E: From<Effect>>(&mut self, id: CallId, update: ToolOutcome, fx: &mut Vec<E>) {
        let Some(call) = self.call_mut(id) else { return };
        if matches!(call.phase, Phase::Running { .. }) {
            settle(call, update, false, fx);
        }
    }

    /// The parent's verdict on a `final_result` candidate: `None` accepted.
    pub(crate) fn final_settled<E: From<Effect>>(
        &mut self,
        id: CallId,
        rejection: Option<String>,
        fx: &mut Vec<E>,
    ) {
        let Some(call) = self.call_mut(id) else { return };
        if !matches!(call.phase, Phase::Validating) {
            return;
        }
        let outcome = match rejection {
            None => value_outcome("final result recorded"),
            Some(reason) => {
                ToolOutcome::ToolError(format!("invalid {FINAL_RESULT_TOOL}: {reason}"))
            }
        };
        settle(call, outcome, false, fx);
    }

    /// Cancels every call still in flight and returns the ones whose settlement is still owed.
    pub fn cancel<E: From<Effect>>(&mut self, fx: &mut Vec<E>) -> Vec<CallId> {
        let mut owed = Vec::new();
        for call in &mut self.calls {
            match call.phase {
                Phase::Preparing { .. } | Phase::ReadingBack { .. } | Phase::Running { .. } => {
                    fx.push(Effect::Cancel { call: call.id }.into());
                    owed.push(call.id);
                }
                Phase::Admitting { prepared, .. } => {
                    fx.push(Effect::ReleaseArguments { prepared }.into());
                    settle(call, ToolOutcome::Cancelled, false, fx);
                }
                Phase::Validating | Phase::Settled(_) => {}
            }
        }
        owed
    }

    /// The latest call the model made as `call_id`.
    pub fn find(&self, call_id: &str) -> Option<CallId> {
        self.calls.iter().rev().find(|c| c.item.call_id == call_id).map(|c| c.id)
    }

    /// The call as recorded so far, its approval marks included.
    pub fn item(&self, id: CallId) -> Option<&FunctionCall> {
        self.calls.iter().find(|c| c.id == id).map(|c| &c.item)
    }

    /// The call's settlement, once it has one.
    pub fn settled(&self, id: CallId) -> Option<&Settled> {
        match &self.calls.iter().find(|c| c.id == id)?.phase {
            Phase::Settled(settled) => Some(settled),
            _ => None,
        }
    }

    pub fn all_settled(&self) -> bool {
        self.calls.iter().all(|c| matches!(c.phase, Phase::Settled(_)))
    }

    pub(crate) fn has_terminal_outcome(&self) -> bool {
        self.calls.iter().any(|call| {
            matches!(&call.phase,
            Phase::Settled(settled) if matches!(settled.outcome,
                ToolOutcome::InputRequired { .. } | ToolOutcome::StopResponse)
                || tool_outcome::verbatim(&settled.outcome).is_some())
        })
    }

    pub(crate) fn delegation_checkpoint(
        &self,
        delegated: &[CallId],
        mode: tool_outcome::Mode,
        created_at: std::time::SystemTime,
    ) -> Option<Vec<ChatItem>> {
        // A sibling with a terminal outcome still belongs to AgentRun's terminal arbitration.
        // The session must not discard it in order to start a child.
        if delegated.iter().any(|id| !self.calls.iter().any(|c| c.id == *id)) {
            return None;
        }
        let mut items = Vec::new();
        for call in &self.calls {
            items.push(ChatItem::FunctionCall(call.item.clone()));
            match &call.phase {
                Phase::Running { .. } if delegated.contains(&call.id) => {}
                Phase::Settled(settled) => {
                    let outcome = &settled.outcome;
                    let shape = tool_outcome::shape(outcome, mode);
                    let mut output = lk_agents_abi::FunctionCallOutput::new(
                        format!("{}_out", call.item.id),
                        &call.item.call_id,
                        shape.text,
                        shape.is_error,
                        created_at,
                    );
                    output.name = call.item.name.clone();
                    output.reply_required = shape.reply_required;
                    items.push(ChatItem::FunctionCallOutput(output));
                }
                _ => return None,
            }
        }
        Some(items)
    }

    /// The settled calls in model order.
    pub fn finish(self) -> Vec<Recorded> {
        self.calls
            .into_iter()
            .filter_map(|c| match c.phase {
                Phase::Settled(settled) => Some(Recorded { id: c.id, item: c.item, settled }),
                _ => None,
            })
            .collect()
    }

    fn call_mut(&mut self, id: CallId) -> Option<&mut Call> {
        self.calls.iter_mut().find(|c| c.id == id)
    }

    /// Decides every prepared call whose earlier calls of the same tool are decided, in model
    /// order: approval binds the validated identity, and a grant is consumed by the first repeat.
    fn admit<'h, E: From<Effect>>(
        &mut self,
        tools: &[FunctionToolInfo],
        history: impl Iterator<Item = &'h ChatItem> + Clone,
        fx: &mut Vec<E>,
    ) {
        for index in 0..self.calls.len() {
            if !matches!(self.calls[index].phase, Phase::Admitting { .. }) {
                continue;
            }
            let name = self.calls[index].item.name.clone();
            let undecided = self.calls[..index].iter().any(|c| {
                c.item.name == name
                    && matches!(
                        c.phase,
                        Phase::Preparing { .. }
                            | Phase::Admitting { .. }
                            | Phase::ReadingBack { .. }
                    )
            });
            if undecided {
                continue;
            }
            self.decide(index, tools, history.clone(), fx);
        }
    }

    fn decide<'h, E: From<Effect>>(
        &mut self,
        index: usize,
        tools: &[FunctionToolInfo],
        history: impl Iterator<Item = &'h ChatItem> + Clone,
        fx: &mut Vec<E>,
    ) {
        let Some(tool) = find(tools, &self.calls[index].item.name) else { return };
        let Phase::Admitting { prepared, ref identity, confirmed } = self.calls[index].phase else {
            return;
        };
        let identity = identity.clone();
        let canonical = approval::canonical(&identity);
        if let Some(read_back) = tool.needs_approval {
            // The history and this step's earlier decisions, the read-backs' answers included.
            let earlier: Vec<ChatItem> = (self.calls[..index].iter())
                .map(|c| ChatItem::FunctionCall(c.item.clone()))
                .collect();
            let call = &mut self.calls[index];
            let decided = history.map(|item| item).chain(&earlier);
            if approval::granted(decided, &self.approval_answers, &call.item.name, &canonical) {
                approval::set_mark(&mut call.item, Approval::Granted, &canonical);
            } else if read_back == ReadBack::Function {
                fx.push(
                    Effect::ReadBack { call: call.id, tool: call.item.name.clone(), prepared }
                        .into(),
                );
                call.phase = Phase::ReadingBack { prepared, identity };
                return;
            } else {
                approval::set_mark(&mut call.item, Approval::Asked, &canonical);
                fx.push(Effect::ReleaseArguments { prepared }.into());
                let question = approval::read_back(&tool.spec.description, &identity);
                let outcome = ToolOutcome::InputRequired { question, verbatim: true, tool: None };
                return settle(call, outcome, false, fx);
            }
        }
        let key = match tool.duplicate_scope {
            DuplicateScope::Name => None,
            DuplicateScope::NameAndArgs => Some(canonical.clone()),
        };
        let running: Vec<usize> = (0..self.calls.len())
            .filter(|&i| i != index && self.calls[i].item.name == tool.spec.name)
            .filter(|&i| match &self.calls[i].phase {
                Phase::Running { identity } => key.as_ref().is_none_or(|k| k == identity),
                _ => false,
            })
            .collect();
        let refusal = match tool.on_duplicate {
            _ if running.is_empty() => None,
            DuplicateMode::Allow => None,
            DuplicateMode::Reject => Some(false),
            DuplicateMode::Confirm if !confirmed => Some(true),
            DuplicateMode::Confirm => None,
            DuplicateMode::Replace => {
                for &i in &running {
                    self.calls[i].cancellation_requested = true;
                    fx.push(Effect::Cancel { call: self.calls[i].id }.into());
                }
                None
            }
        };
        if let Some(confirm) = refusal {
            let shown: Vec<&FunctionCall> = running.iter().map(|&i| &self.calls[i].item).collect();
            let text = duplicate_text(&tool.spec.name, &shown, confirm);
            fx.push(Effect::ReleaseArguments { prepared }.into());
            return settle(&mut self.calls[index], ToolOutcome::ToolError(text), false, fx);
        }
        let call = &mut self.calls[index];
        let arguments = match identity {
            ArgumentIdentity::Json(value) => value,
            ArgumentIdentity::Fingerprint(_) => Value::Null,
        };
        fx.push(
            Effect::Emit(StreamEvent::ToolCall {
                call_id: call.item.call_id.clone(),
                tool_name: call.item.name.clone(),
                arguments,
            })
            .into(),
        );
        fx.push(
            Effect::Execute {
                call: call.id,
                prepared,
                function_call: call.item.clone(),
                approval: approval::mark(&call.item),
            }
            .into(),
        );
        call.phase = Phase::Running { identity: canonical };
    }
}

/// How a step takes a call the model made, before the tool machine sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// An offered tool: the machine decides the call.
    Offered,
    /// A name the request did not offer: answered as unknown, never run.
    Unoffered,
    /// The step forbade tools (`tool_choice="none"`): recorded as made, never run.
    Forbidden { offered: bool },
}

impl Admission {
    /// The request offered the tool.
    pub fn offered(self) -> bool {
        matches!(self, Admission::Offered | Admission::Forbidden { offered: true })
    }
}

/// The model's call as a streaming step takes it: arguments a model template leaked into are
/// repaired in place (the history keeps the repaired form a provider accepts), and a call of the
/// tool an `InputRequired(tool=)` claimed (`answering`: the tool and the person's words) takes the
/// words as its first string parameter.
pub fn admit(
    call: &mut FunctionCall,
    offered: &[ToolSpec],
    forbidden: bool,
    answering: Option<(&str, &str)>,
) -> Admission {
    let malformed = !call.arguments.trim().is_empty()
        && serde_json::from_str::<Value>(&call.arguments).is_err();
    if malformed
        && let Ok(parsed) = tool_arguments::parse(&call.arguments)
        && parsed.repair.is_some()
    {
        call.arguments = Value::Object(parsed.arguments).to_string();
    }
    let spec = offered.iter().find(|t| t.name == call.name);
    if let (Some((tool, said)), Some(spec)) = (answering, spec)
        && tool == call.name
        && let Some(parameter) = crate::agent::first_string_parameter(spec)
    {
        let arguments = Map::from_iter([(parameter.to_owned(), Value::from(said))]);
        call.arguments = Value::Object(arguments).to_string();
    }
    match (forbidden, spec) {
        (true, spec) => Admission::Forbidden { offered: spec.is_some() },
        (false, Some(_)) => Admission::Offered,
        (false, None) => Admission::Unoffered,
    }
}

/// Why the host did not prepare a call's arguments. Either way the call spends a retry.
#[derive(Debug, Clone, PartialEq)]
pub enum PrepareFailure {
    /// The arguments did not bind or validate: the message is the model's feedback.
    Refused(String),
    /// The host failed; it logged the error, and the model is told nothing about it.
    Internal,
}

/// Parses a call's arguments and asks the host to prepare them; an unknown tool or arguments
/// that are not a JSON object settle at once as feedback.
fn start<E: From<Effect>>(
    id: CallId,
    item: &FunctionCall,
    tools: &[FunctionToolInfo],
    fx: &mut Vec<E>,
) -> Phase {
    let failed = |text: String, fx: &mut Vec<E>| {
        let outcome = ToolOutcome::ToolError(text);
        fx.push(result_event(item, &outcome).into());
        Phase::Settled(Settled { outcome, retry: true })
    };
    let Some(tool) = find(tools, &item.name) else {
        return failed(format!("unknown tool `{}`", item.name), fx);
    };
    let parsed = match tool_arguments::parse(&item.arguments) {
        Ok(parsed) => parsed,
        Err(error) => return failed(error, fx),
    };
    if let Some(repair) = parsed.repair {
        fx.push(
            Effect::ArgumentsRepaired {
                call_id: item.call_id.clone(),
                error: repair.error,
                repaired: repair.repaired,
            }
            .into(),
        );
    }
    let mut arguments = parsed.arguments;
    let confirmed = match tool.on_duplicate {
        DuplicateMode::Confirm => {
            arguments.remove(CONFIRM_DUPLICATE_PARAM).is_some_and(|v| truthy(&v))
        }
        _ => false,
    };
    if !tool.raw_schema {
        arguments = with_defaults(arguments, &tool.spec.json_schema);
    }
    fx.push(Effect::Prepare { call: id, arguments, function_call: item.clone() }.into());
    Phase::Preparing { confirmed }
}

fn find<'a>(tools: &'a [FunctionToolInfo], name: &str) -> Option<&'a FunctionToolInfo> {
    tools.iter().find(|t| t.spec.name == name)
}

/// The schema's defaults for omitted parameters: the framework's, applied before the host
/// validates, so a call that omits a default and one that spells it are the same call.
fn with_defaults(arguments: Map<String, Value>, schema: &Value) -> Map<String, Value> {
    let value = Value::Object(arguments);
    match crate::llm::schema::inject_schema_defaults(&value, schema, schema) {
        Ok(Value::Object(filled)) => filled,
        _ => match value {
            Value::Object(arguments) => arguments,
            _ => Map::new(),
        },
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::Null => false,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A `final_result` call's value: its arguments, unwrapped when the schema was wrapped.
fn final_value(agent: &AgentDefinition, raw: &str) -> Result<Value, String> {
    let arguments = tool_arguments::parse(raw)?.arguments;
    if !agent::output_wrapped(agent) {
        return Ok(Value::Object(arguments));
    }
    let mut arguments = arguments;
    arguments
        .remove(WRAPPED_RESULT_KEY)
        .ok_or_else(|| format!("missing required argument `{WRAPPED_RESULT_KEY}`"))
}

fn value_outcome(text: &str) -> ToolOutcome {
    ToolOutcome::Value { text: text.to_owned(), reply_required: true, verbatim: false, more: None }
}

fn settle<E: From<Effect>>(call: &mut Call, outcome: ToolOutcome, retry: bool, fx: &mut Vec<E>) {
    fx.push(result_event(&call.item, &outcome).into());
    call.phase = Phase::Settled(Settled { outcome, retry });
}

/// The call's result as it is recognised, before any model-order assembly.
fn result_event(item: &FunctionCall, outcome: &ToolOutcome) -> Effect {
    // The text and error flag: the same in every mode.
    let shape = tool_outcome::shape(outcome, tool_outcome::Mode::Text);
    let status = match outcome {
        ToolOutcome::Cancelled => ToolResultStatus::Cancelled,
        _ if shape.is_error => ToolResultStatus::Failed,
        _ => ToolResultStatus::Completed,
    };
    Effect::Emit(StreamEvent::ToolResult {
        call_id: item.call_id.clone(),
        tool_name: item.name.clone(),
        status,
        output: shape.text,
    })
}

#[cfg(test)]
mod admit_tests {
    use super::*;

    fn repaired(raw: &str) -> String {
        let mut call = FunctionCall {
            id: "c".into(),
            call_id: "c".into(),
            arguments: raw.into(),
            name: "lookup".into(),
            created_at: std::time::SystemTime::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        };
        admit(&mut call, &[], false, None);
        call.arguments
    }

    // Catches: a repaired call's arguments left malformed in history, or valid ones rewritten.
    #[test]
    fn malformed_arguments_are_kept_repaired() {
        let raw = r#"{"order_id": ["<|\"|\"O_WAAB70<|\"|\"]}"#;
        assert!(serde_json::from_str::<Value>(&repaired(raw)).is_ok(), "{}", repaired(raw));
        assert_eq!(repaired(r#"{"label":"human<|im_end|>"#), r#"{"label":"human"}"#);
        assert_eq!(repaired(r#"{"a": 1}"#), r#"{"a": 1}"#);
        assert_eq!(repaired(""), "");
    }
}
