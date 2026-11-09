//! An agent called as a tool (tools.md "Agents as tools"): what its child starts from, what the
//! parent reads when it settles, and what a child waiting on the person keeps on its parent's call.
//! A TextSession runs the child as a frame; elsewhere the child is an in-process delegate.

use std::time::SystemTime;

use lk_agents_abi::agent::run::{Output, RunError, RunOutcome, RunResult};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::text_session::{ChatHistory, RoutingError};
use lk_agents_abi::{ChatContext, ChatItem, ChatMessage, CopyOptions, FunctionCall, Role};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llm::history::ChatContextExt;

/// A waiting child's own history, on the parent's call that it asked from.
pub const STATE_KEY: &str = "lk.agent_tool";

/// How deep delegates nest outside a TextSession, as a TextSession's default `max_depth`.
pub const MAX_DEPTH: usize = 8;

#[derive(Serialize, Deserialize)]
struct Waiting {
    agent: String,
    history: Vec<ChatItem>,
}

/// The child's opening history: nothing (`fresh`), or the parent's answered exchanges without
/// instructions, handoffs or configuration updates (`fork`).
pub fn opening(parent: &[ChatItem], mode: ChatHistory) -> Vec<ChatItem> {
    if mode == ChatHistory::Fresh {
        return Vec::new();
    }
    let answered: std::collections::BTreeSet<&str> = parent
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output) => Some(output.call_id.as_str()),
            _ => None,
        })
        .filter(|id| {
            parent.iter().any(|i| matches!(i, ChatItem::FunctionCall(c) if c.call_id == *id))
        })
        .collect();
    let complete = parent
        .iter()
        .filter(|item| match item {
            ChatItem::FunctionCall(call) => answered.contains(call.call_id.as_str()),
            ChatItem::FunctionCallOutput(output) => answered.contains(output.call_id.as_str()),
            _ => true,
        })
        .cloned()
        .collect();
    let options = CopyOptions {
        exclude_instructions: true,
        exclude_handoff: true,
        exclude_config_update: true,
        ..Default::default()
    };
    ChatContext::new(complete).copy(&options).items
}

/// The child's task: the parent's validated arguments, as its user message.
pub fn task(id: String, arguments: &Value, at: SystemTime) -> ChatMessage {
    ChatMessage::text(id, Role::User, arguments.to_string(), at)
}

/// A delegate nested past [`MAX_DEPTH`] is refused before it runs.
pub fn admit(depth: usize) -> Result<(), ToolOutcome> {
    if depth > MAX_DEPTH {
        let refused = RoutingError::MaxDepth { max_depth: MAX_DEPTH };
        return Err(ToolOutcome::ToolError(refused.to_string()));
    }
    Ok(())
}

/// The waiting child a call of the tool resumes: the own history kept on the tool's latest
/// earlier call, when that call ended on the child's question; empty otherwise.
pub fn retained(history: &[ChatItem], call: &FunctionCall, agent: &str) -> Vec<ChatItem> {
    let latest = history.iter().rev().find_map(|item| match item {
        ChatItem::FunctionCall(c) if c.name == call.name && c.call_id != call.call_id => Some(c),
        _ => None,
    });
    latest
        .and_then(|c| c.extra.get(STATE_KEY))
        .and_then(|state| serde_json::from_value::<Waiting>(state.clone()).ok())
        .filter(|waiting| waiting.agent == agent)
        .map(|waiting| waiting.history)
        .unwrap_or_default()
}

/// The call's retained state once the child settled: its own history while it waits on the
/// person, so the next call resumes it; nothing once it answered.
pub fn record(extra: &mut Map<String, Value>, agent: &str, waiting: Option<Vec<ChatItem>>) {
    match waiting {
        Some(history) => {
            let state = Waiting { agent: agent.to_owned(), history };
            let state = serde_json::to_value(state).expect("chat items serialize");
            extra.insert(STATE_KEY.into(), state);
        }
        None => {
            extra.remove(STATE_KEY);
        }
    }
}

/// What a settled child invocation is to its parent's call.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// The call's outcome.
    Done(ToolOutcome),
    /// The child waits on the person: on a call the question is put to them and the answer
    /// resumes the child; elsewhere the call ends on it ([`Step::outcome`]).
    Ask { question: String, verbatim: bool },
}

impl Step {
    /// The call's outcome off a call: a question ends it as `InputRequired`, passed up as asked.
    pub fn outcome(self) -> ToolOutcome {
        match self {
            Step::Done(outcome) => outcome,
            Step::Ask { question, verbatim } => {
                ToolOutcome::InputRequired { question, verbatim, tool: None }
            }
        }
    }
}

/// The parent's view of the child's settled invocation; `output` is the accepted value,
/// serialized, when the child's output is typed.
pub fn settle(result: &Result<RunResult, RunError>, output: Option<&Value>) -> Step {
    let result = match result {
        Ok(result) => result,
        Err(RunError::Cancelled) => return Step::Done(ToolOutcome::Cancelled),
        Err(error) => return Step::Done(ToolOutcome::Failed(error.to_string())),
    };
    match &result.outcome {
        RunOutcome::Completed { output: Output::Text(text) } => {
            Step::Done(crate::a2a::delegation::answer(text, result.verbatim.as_deref()))
        }
        RunOutcome::Completed { output: Output::Validated(_) } => Step::Done(match output {
            Some(Value::String(text)) => value(text.clone()),
            Some(value_) => value(value_.to_string()),
            None => ToolOutcome::Failed("the agent's output could not be serialized".into()),
        }),
        RunOutcome::InputRequired(ask) => {
            Step::Ask { question: ask.question.clone(), verbatim: ask.verbatim }
        }
        RunOutcome::NoOutput if result.interrupted => Step::Done(ToolOutcome::Cancelled),
        RunOutcome::NoOutput => {
            Step::Done(ToolOutcome::ToolError("the agent ended without an answer".into()))
        }
    }
}

fn value(text: String) -> ToolOutcome {
    ToolOutcome::Value { text, reply_required: true, verbatim: false, more: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::agent::run::RunQuestion;
    use serde_json::json;
    use std::time::UNIX_EPOCH;

    fn result(outcome: RunOutcome) -> Result<RunResult, RunError> {
        Ok(RunResult {
            outcome,
            interrupted: false,
            new_messages: Vec::new(),
            all_messages: Vec::new(),
            usage: Default::default(),
            verbatim: None,
            pending_call: None,
            input_requirements: Vec::new(),
        })
    }

    fn call(id: &str, extra: Map<String, Value>) -> ChatItem {
        ChatItem::FunctionCall(FunctionCall {
            id: id.into(),
            call_id: id.into(),
            name: "booking".into(),
            arguments: "{}".into(),
            created_at: UNIX_EPOCH,
            extra,
            group_id: None,
        })
    }

    #[test]
    fn a_settled_child_is_its_parents_tool_outcome() {
        let typed = result(RunOutcome::Completed {
            output: Output::Validated(lk_agents_abi::agent::run::ValidationId(1)),
        });
        assert_eq!(
            settle(&typed, Some(&json!({"room": 4}))),
            Step::Done(value("{\"room\":4}".into()))
        );
        assert!(matches!(settle(&typed, None), Step::Done(ToolOutcome::Failed(_))));
        let text = result(RunOutcome::Completed { output: Output::Text("Booked.".into()) });
        assert_eq!(settle(&text, None), Step::Done(value("Booked.".into())));
        let silent = result(RunOutcome::NoOutput);
        assert!(matches!(settle(&silent, None), Step::Done(ToolOutcome::ToolError(_))));
        let cancelled = Err(RunError::Cancelled);
        assert_eq!(settle(&cancelled, None), Step::Done(ToolOutcome::Cancelled));
    }

    #[test]
    fn a_childs_question_passes_up_as_asked() {
        let ask = RunQuestion { question: "Confirm?".into(), verbatim: true, answer_tool: None };
        let step = settle(&result(RunOutcome::InputRequired(ask)), None);
        assert_eq!(step, Step::Ask { question: "Confirm?".into(), verbatim: true });
        assert_eq!(
            step.outcome(),
            ToolOutcome::InputRequired { question: "Confirm?".into(), verbatim: true, tool: None }
        );
    }

    #[test]
    fn only_the_tools_latest_call_left_waiting_is_resumed() {
        let own = vec![ChatItem::Message(task("task".into(), &json!({}), UNIX_EPOCH))];
        let mut waiting = Map::new();
        record(&mut waiting, "booking", Some(own.clone()));
        let current = call("now", Map::new());
        let ChatItem::FunctionCall(current_call) = current.clone() else { unreachable!() };
        let history = vec![call("asked", waiting.clone()), current.clone()];
        assert_eq!(retained(&history, &current_call, "booking"), own);
        assert!(retained(&history, &current_call, "billing").is_empty());

        let mut answered = waiting;
        record(&mut answered, "booking", None);
        let later = vec![history[0].clone(), call("settled", answered), current];
        assert!(retained(&later, &current_call, "booking").is_empty());
    }

    #[test]
    fn delegates_past_the_depth_limit_are_refused() {
        assert!(admit(MAX_DEPTH).is_ok());
        assert!(matches!(admit(MAX_DEPTH + 1), Err(ToolOutcome::ToolError(_))));
    }
}
