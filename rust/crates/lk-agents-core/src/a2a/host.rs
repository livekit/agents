//! Conversion of native turn records at the generated host boundary. Protocol JSON is decoded
//! once here, before the runtime accepts handler output.

use lk_agents_abi::a2a::turn as host;
use lk_agents_abi::storage::StorageError;

use crate::a2a::{HandlerOutput, Message, TurnContext, TurnResult};

impl From<Message> for host::Message {
    fn from(message: Message) -> Self {
        let mut out = crate::a2a::client::message_from_wire(message.raw.clone());
        out.message_id = Some(message.message_id);
        out.parts = message.parts.into_iter().map(crate::a2a::client::part_from_wire).collect();
        out.channel = Some(message.channel);
        out.received_at = Some(message.received_at);
        out.task_id = message.task_id;
        out.context_id = message.context_id;
        out.raw = Some(serde_json::to_value(message.raw).expect("A2A message"));
        out
    }
}

impl TryFrom<TurnContext> for host::TurnContext {
    type Error = StorageError;

    fn try_from(ctx: TurnContext) -> Result<Self, Self::Error> {
        fn object(
            value: serde_json::Value,
            field: &str,
        ) -> Result<serde_json::Map<String, serde_json::Value>, StorageError> {
            match value {
                serde_json::Value::Object(value) => Ok(value),
                _ => Err(StorageError::Invalid(format!("{field} must be a chat context object"))),
            }
        }
        let message = object(serde_json::to_value(ctx.message).expect("chat item"), "message")?;
        let caller_history =
            ctx.caller_history.map(|value| object(value, "caller_history")).transpose()?;
        Ok(Self {
            thread_id: ctx.thread_id,
            lane: ctx.lane,
            message,
            batch: ctx.batch.into_iter().map(Into::into).collect(),
            history: ctx.history.into_iter().map(Into::into).collect(),
            state: ctx.state,
            context_scope_id: ctx.context_scope_id,
            context_scopes: ctx.context_scopes,
            read_only: ctx.read_only,
            task_id: ctx.task_id,
            tasks: ctx.tasks.into_iter().map(Into::into).collect(),
            caller_history,
            answer_tool: ctx.answer_tool,
            host_context: None,
        })
    }
}

impl TryFrom<host::HandlerOutput> for HandlerOutput {
    type Error = StorageError;

    fn try_from(output: host::HandlerOutput) -> Result<Self, Self::Error> {
        let new_items = output
            .new_items
            .into_iter()
            .map(|item| {
                serde_json::from_value(item)
                    .map_err(|e| StorageError::Invalid(format!("handler chat item: {e}")))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            output: output.output,
            new_items,
            state: output.state,
            scope_writes: output.scope_writes,
            app_writes: output.app_writes,
            usage: crate::usage::combined(&output.usage),
            input_required: output.input_required,
            pending_call: output.pending_call,
            interrupted: output.interrupted,
            metadata: output.metadata,
            error: output.error,
            tool_executions: output.tool_executions,
            status: output.status,
        })
    }
}

impl From<TurnResult> for host::TurnResult {
    fn from(result: TurnResult) -> Self {
        Self {
            task: result.task.into(),
            output: result.output,
            thread_id: result.thread_id,
            answered_by: result.answered_by,
            run_id: result.run_id,
        }
    }
}

impl From<HandlerOutput> for host::HandlerOutput {
    fn from(output: HandlerOutput) -> Self {
        Self {
            output: output.output,
            new_items: output
                .new_items
                .into_iter()
                .map(|item| serde_json::to_value(item).expect("chat item"))
                .collect(),
            state: output.state,
            scope_writes: output.scope_writes,
            app_writes: output.app_writes,
            usage: vec![output.usage],
            input_required: output.input_required,
            pending_call: output.pending_call,
            interrupted: output.interrupted,
            metadata: output.metadata,
            error: output.error,
            tool_executions: output.tool_executions,
            status: output.status,
        }
    }
}
