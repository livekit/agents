//! An agent called as a tool: its opening context and the parent's view of its outcome.
//! A TextSession runs the child as a frame; elsewhere the child is an in-process delegate.

use std::time::SystemTime;

use lk_agents_abi::agent::run::{Output, RunError, RunOutcome, RunResult};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::text_session::{ChatHistory, RoutingError};
use lk_agents_abi::{ChatContext, ChatItem, ChatMessage, CopyOptions, Role};
use serde_json::Value;

use crate::llm::history::ChatContextExt;

/// How deep delegates nest outside a TextSession, as a TextSession's default `max_depth`.
pub const MAX_DEPTH: usize = 8;

/// The child's opening history: nothing (`fresh`), or the parent's answered exchanges without
/// instructions, handoffs or configuration updates (`fork`).
pub fn opening(parent: &[ChatItem], mode: ChatHistory) -> Vec<ChatItem> {
    if mode == ChatHistory::Fresh {
        return Vec::new();
    }
    let facts: Vec<_> = parent.iter().map(crate::llm::history::ItemFacts::of).collect();
    let answered: std::collections::BTreeSet<_> = crate::llm::history::tool_pairs(&facts)
        .into_iter()
        .flat_map(|(call, output)| [call, output])
        .collect();
    let complete = parent
        .iter()
        .enumerate()
        .filter(|(index, item)| match item {
            ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_) => answered.contains(index),
            _ => true,
        })
        .map(|(_, item)| item.clone())
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
            Step::Ask { question, verbatim } => ToolOutcome::InputRequired { question, verbatim },
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
            Step::Done(crate::delegation::remote::answer(text, result.verbatim.as_deref()))
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
    use lk_agents_abi::FunctionCall;
    use lk_agents_abi::agent::run::RunQuestion;
    use serde_json::{Map, json};
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
    fn a_fork_does_not_inherit_an_unanswered_call_with_a_reused_provider_id() {
        let first = call("provider", Default::default());
        let output = ChatItem::FunctionCallOutput(lk_agents_abi::FunctionCallOutput::new(
            "opaque-receipt",
            "provider",
            "first result",
            false,
            UNIX_EPOCH,
        ));
        let mut second = first.clone();
        let ChatItem::FunctionCall(current) = &mut second else { unreachable!() };
        current.id = "new-item".into();
        let history = [first, output, second];
        assert_eq!(opening(&history, ChatHistory::Fork), history[..2]);
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
        let ask = RunQuestion { question: "Confirm?".into(), verbatim: true };
        let step = settle(&result(RunOutcome::InputRequired(ask)), None);
        assert_eq!(step, Step::Ask { question: "Confirm?".into(), verbatim: true });
        assert_eq!(
            step.outcome(),
            ToolOutcome::InputRequired { question: "Confirm?".into(), verbatim: true }
        );
    }

    #[test]
    fn delegates_past_the_depth_limit_are_refused() {
        assert!(admit(MAX_DEPTH).is_ok());
        assert!(matches!(admit(MAX_DEPTH + 1), Err(ToolOutcome::ToolError(_))));
    }
}
