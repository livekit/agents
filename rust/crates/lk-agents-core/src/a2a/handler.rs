//! Handler return normalization and run journal projection, shared by language hosts.

use lk_agents_abi::a2a::{
    metadata::VERBATIM_KEY,
    turn::HandlerOutput,
    turn::{HandlerOutcome, HandlerRun},
};
use lk_agents_abi::storage::{
    StorageError, TaskState,
    journal::{ToolExecution, ToolPhase},
};
use lk_agents_abi::{ChatItem, llm::chat_context::protocol::encode_chat_item};
use serde_json::Value;

pub struct HandlerOutcomes;

impl lk_agents_abi::a2a::turn::HandlerOutcomes for HandlerOutcomes {
    fn normalize_handler_output(
        mut output: HandlerOutput,
        outcome: HandlerOutcome,
        runs: Vec<HandlerRun>,
    ) -> Result<HandlerOutput, StorageError> {
        match outcome {
            HandlerOutcome::Value { value } => output.output = Some(value),
            HandlerOutcome::Completed { value, verbatim } => {
                if verbatim && value.is_string() {
                    output.metadata.insert(VERBATIM_KEY.into(), value.clone());
                }
                output.output = Some(value);
            }
            HandlerOutcome::InputRequired { question } => {
                output.output = Some(Value::String(question.clone()));
                output.input_required = Some(question);
            }
            HandlerOutcome::Failed { error } => {
                output.error = Some(error);
                output.output = None;
            }
            HandlerOutcome::Rejected { reason } => {
                output.status = Some(TaskState::Rejected);
                output.output = Some(Value::String(reason));
            }
            HandlerOutcome::Working { message } => {
                output.status = Some(TaskState::Working);
                output.output = message.map(Value::String);
            }
        }
        if let Some(verbatim) = runs.last().and_then(|run| run.verbatim.clone()) {
            output.metadata.entry(VERBATIM_KEY).or_insert(Value::String(verbatim));
        }
        if let Some(asked) = runs.iter().rev().find(|run| run.question.is_some()) {
            output.input_required = asked.question.clone();
            output.pending_call = asked.pending_call.clone();
        }
        let mut events = Vec::new();
        for run in runs {
            output.usage.push(run.usage);
            for item in run.items {
                let event = match &item {
                    ChatItem::FunctionCall(call) => {
                        Some((ToolPhase::Started, &call.call_id, &call.name, None))
                    }
                    ChatItem::FunctionCallOutput(call) => {
                        Some((ToolPhase::Ended, &call.call_id, &call.name, Some(call.is_error)))
                    }
                    _ => None,
                };
                if let Some((phase, call_id, name, is_error)) = event {
                    events.push(ToolExecution {
                        phase,
                        call_id: Some(call_id.clone()),
                        name: Some(name.clone()),
                        is_error,
                        text: None,
                    });
                }
                output.new_items.push(
                    serde_json::to_value(encode_chat_item(&item))
                        .map_err(|error| StorageError::Invalid(error.to_string()))?,
                );
            }
        }
        events.append(&mut output.tool_executions);
        output.tool_executions = events;
        Ok(output)
    }
}
