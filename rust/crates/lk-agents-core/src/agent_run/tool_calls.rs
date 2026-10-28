//! The tool calls of one model step (rust-core "Shared tool execution"): parse and canonicalise
//! the model's arguments, have the host prepare them, bind approval to the validated identity,
//! admit duplicates per the tool's policy, run the body, and record each outcome as it arrives.
//! Results are assembled in model order by the parent; waiting on one call never holds back
//! another's lifecycle.

use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::llm::tool_arguments;
use lk_agents_abi::llm::tool_executor::{Approval, ToolOutcome};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ChatItem, FunctionCall};
use serde_json::{Map, Value};

use super::Effect;
use super::result::CallId;
use crate::agent::{self, FINAL_RESULT_TOOL, WRAPPED_RESULT_KEY};
use crate::approval;
use crate::tool_outcome;
use lk_agents_abi::agent::run::{StreamEvent, ToolResultStatus};
use lk_agents_abi::agent::{AgentDefinition, OutputSpec, ReadBack};
use lk_agents_abi::agent::{ArgumentIdentity, ReadBackAnswer};

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
pub(super) struct ToolCalls {
    calls: Vec<Call>,
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
#[derive(Debug, Clone)]
pub(super) struct Settled {
    pub(super) outcome: ToolOutcome,
    /// Feedback that counts against the tool's retry budget: invalid or unknown arguments,
    /// `ModelRetry`.
    pub(super) retry: bool,
}

/// What [`ToolCalls::begin`] leaves to the parent.
pub(super) struct FinalCandidate {
    pub(super) call: CallId,
    pub(super) value: Result<Value, String>,
}

/// One settled call, in model order.
pub(super) struct Recorded {
    pub(super) id: CallId,
    pub(super) item: FunctionCall,
    pub(super) settled: Settled,
}

impl ToolCalls {
    /// Starts every call of a step. A call that cannot run settles at once; a `final_result`
    /// call is returned as a candidate for the parent's output acceptance.
    pub(super) fn begin(
        calls: Vec<(CallId, FunctionCall)>,
        agent: &AgentDefinition,
        fx: &mut Vec<Effect>,
    ) -> (ToolCalls, Vec<FinalCandidate>) {
        let mut this = ToolCalls { calls: Vec::with_capacity(calls.len()) };
        let mut finals = Vec::new();
        for (id, item) in calls {
            let phase = if item.name == FINAL_RESULT_TOOL
                && matches!(agent.output, OutputSpec::Structured { .. })
            {
                finals
                    .push(FinalCandidate { call: id, value: final_value(agent, &item.arguments) });
                Phase::Validating
            } else {
                start(id, &item, agent, fx)
            };
            this.calls.push(Call { id, item, phase, cancellation_requested: false });
        }
        (this, finals)
    }

    pub(super) fn prepared(
        &mut self,
        id: CallId,
        prepared: ResourceId,
        identity: ArgumentIdentity,
        agent: &AgentDefinition,
        history: &[ChatItem],
        fx: &mut Vec<Effect>,
    ) -> bool {
        let Some(call) = self.call_mut(id) else { return false };
        let Phase::Preparing { confirmed } = call.phase else { return false };
        call.phase = Phase::Admitting { prepared, identity, confirmed };
        self.admit(agent, history, fx);
        true
    }

    pub(super) fn prepare_failed(
        &mut self,
        id: CallId,
        error: String,
        agent: &AgentDefinition,
        history: &[ChatItem],
        fx: &mut Vec<Effect>,
    ) {
        let Some(call) = self.call_mut(id) else { return };
        if !matches!(call.phase, Phase::Preparing { .. }) {
            return;
        }
        settle(call, ToolOutcome::ToolError(error), true, fx);
        self.admit(agent, history, fx);
    }

    pub(super) fn read_back(
        &mut self,
        id: CallId,
        answer: Result<ReadBackAnswer, String>,
        agent: &AgentDefinition,
        history: &[ChatItem],
        fx: &mut Vec<Effect>,
    ) {
        let Some(call) = self.call_mut(id) else { return };
        let Phase::ReadingBack { prepared, ref identity } = call.phase else { return };
        let canonical = approval::canonical(identity);
        fx.push(Effect::ToolArgumentsRelease { prepared });
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
        self.admit(agent, history, fx);
    }

    pub(super) fn progress(&mut self, id: CallId, message: String, fx: &mut Vec<Effect>) {
        if let Some(call) = self.call_mut(id).filter(|c| matches!(c.phase, Phase::Running { .. })) {
            fx.push(Effect::Emit(StreamEvent::Progress {
                call_id: call.item.call_id.clone(),
                tool_name: call.item.name.clone(),
                message,
            }));
        }
    }

    pub(super) fn update_extra(&mut self, id: CallId, extra: Map<String, Value>) {
        let Some(call) = self.call_mut(id) else { return };
        if matches!(call.phase, Phase::Running { .. }) {
            call.item.extra.extend(extra);
        }
    }

    /// The body's outcome. An update is progress: the tool runs on.
    pub(super) fn body_settled(
        &mut self,
        id: CallId,
        outcome: ToolOutcome,
        fx: &mut Vec<Effect>,
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

    /// The parent's verdict on a `final_result` candidate: `None` accepted.
    pub(super) fn final_settled(
        &mut self,
        id: CallId,
        rejection: Option<String>,
        fx: &mut Vec<Effect>,
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
    pub(super) fn cancel(&mut self, fx: &mut Vec<Effect>) -> Vec<CallId> {
        let mut owed = Vec::new();
        for call in &mut self.calls {
            match call.phase {
                Phase::Preparing { .. } | Phase::ReadingBack { .. } | Phase::Running { .. } => {
                    fx.push(Effect::ToolCancel { call: call.id });
                    owed.push(call.id);
                }
                Phase::Admitting { prepared, .. } => {
                    fx.push(Effect::ToolArgumentsRelease { prepared });
                    settle(call, ToolOutcome::Cancelled, false, fx);
                }
                Phase::Validating | Phase::Settled(_) => {}
            }
        }
        owed
    }

    pub(super) fn all_settled(&self) -> bool {
        self.calls.iter().all(|c| matches!(c.phase, Phase::Settled(_)))
    }

    pub(super) fn has_terminal_outcome(&self) -> bool {
        self.calls.iter().any(|call| {
            matches!(&call.phase,
            Phase::Settled(settled) if matches!(settled.outcome,
                ToolOutcome::InputRequired { .. } | ToolOutcome::StopResponse)
                || tool_outcome::verbatim(&settled.outcome).is_some())
        })
    }

    pub(super) fn delegation_checkpoint(
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
    pub(super) fn finish(self) -> Vec<Recorded> {
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
    fn admit(&mut self, agent: &AgentDefinition, history: &[ChatItem], fx: &mut Vec<Effect>) {
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
            self.decide(index, agent, history, fx);
        }
    }

    fn decide(
        &mut self,
        index: usize,
        agent: &AgentDefinition,
        history: &[ChatItem],
        fx: &mut Vec<Effect>,
    ) {
        let Some(tool) = agent::tool(agent, &self.calls[index].item.name) else { return };
        let Phase::Admitting { prepared, ref identity, confirmed } = self.calls[index].phase else {
            return;
        };
        let identity = identity.clone();
        let canonical = approval::canonical(&identity);
        if let Some(read_back) = tool.needs_approval {
            // The history and this step's earlier decisions, the read-backs' answers included.
            let mut decided: Vec<ChatItem> = history.to_vec();
            decided
                .extend(self.calls[..index].iter().map(|c| ChatItem::FunctionCall(c.item.clone())));
            let call = &mut self.calls[index];
            if approval::granted(&decided, &call.item.name, &canonical) {
                approval::set_mark(&mut call.item, Approval::Granted, &canonical);
            } else if read_back == ReadBack::Function {
                fx.push(Effect::ReadBackRender {
                    call: call.id,
                    tool: call.item.name.clone(),
                    prepared,
                });
                call.phase = Phase::ReadingBack { prepared, identity };
                return;
            } else {
                approval::set_mark(&mut call.item, Approval::Asked, &canonical);
                fx.push(Effect::ToolArgumentsRelease { prepared });
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
                    fx.push(Effect::ToolCancel { call: self.calls[i].id });
                }
                None
            }
        };
        if let Some(confirm) = refusal {
            let shown: Vec<&FunctionCall> = running.iter().map(|&i| &self.calls[i].item).collect();
            let text = duplicate_text(&tool.spec.name, &shown, confirm);
            fx.push(Effect::ToolArgumentsRelease { prepared });
            return settle(&mut self.calls[index], ToolOutcome::ToolError(text), false, fx);
        }
        let call = &mut self.calls[index];
        let arguments = match identity {
            ArgumentIdentity::Json(value) => value,
            ArgumentIdentity::Fingerprint(_) => Value::Null,
        };
        fx.push(Effect::Emit(StreamEvent::ToolCall {
            call_id: call.item.call_id.clone(),
            tool_name: call.item.name.clone(),
            arguments,
        }));
        fx.push(Effect::ToolExecute { call: call.id, prepared, function_call: call.item.clone() });
        call.phase = Phase::Running { identity: canonical };
    }
}

/// Parses a call's arguments and asks the host to prepare them; an unknown tool or arguments
/// that are not a JSON object settle at once as feedback.
fn start(id: CallId, item: &FunctionCall, agent: &AgentDefinition, fx: &mut Vec<Effect>) -> Phase {
    let failed = |text: String, fx: &mut Vec<Effect>| {
        let outcome = ToolOutcome::ToolError(text);
        fx.push(result_event(item, &outcome));
        Phase::Settled(Settled { outcome, retry: true })
    };
    let Some(tool) = agent::tool(agent, &item.name) else {
        return failed(format!("unknown tool `{}`", item.name), fx);
    };
    let parsed = match tool_arguments::parse(&item.arguments) {
        Ok(parsed) => parsed,
        Err(error) => return failed(error, fx),
    };
    if let Some(repair) = parsed.repair {
        fx.push(Effect::ArgumentsRepaired {
            call_id: item.call_id.clone(),
            error: repair.error,
            repaired: repair.repaired,
        });
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
    fx.push(Effect::ToolPrepare { call: id, arguments, function_call: item.clone() });
    Phase::Preparing { confirmed }
}

/// The schema's defaults for omitted parameters: the framework's, applied before the host
/// validates, so a call that omits a default and one that spells it are the same call.
fn with_defaults(arguments: Map<String, Value>, schema: &Value) -> Map<String, Value> {
    let value = Value::Object(arguments);
    match lk_agents_abi::llm::schema::inject_schema_defaults(&value, schema, schema) {
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

fn settle(call: &mut Call, outcome: ToolOutcome, retry: bool, fx: &mut Vec<Effect>) {
    fx.push(result_event(&call.item, &outcome));
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
