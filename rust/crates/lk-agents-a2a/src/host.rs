//! Conversion of native turn records at the generated host boundary. Protocol JSON is decoded
//! once here, before the runtime accepts handler output.

use lk_agents_abi::a2a::turn as host;
use lk_agents_storage::StorageError;

use crate::{HandlerOutput, HostEvent, Message, PendingCall, TurnContext, TurnResult};

impl From<Message> for host::Message {
    fn from(message: Message) -> Self {
        Self {
            message_id: message.message_id,
            parts: message
                .parts
                .into_iter()
                .map(|p| serde_json::to_value(p).expect("A2A part"))
                .collect(),
            metadata: message.metadata,
            channel: message.channel,
            received_at: message.received_at,
            task_id: message.task_id,
            context_id: message.context_id,
            raw: serde_json::to_value(message.raw).expect("A2A message"),
        }
    }
}

impl From<TurnContext> for host::TurnContext {
    fn from(ctx: TurnContext) -> Self {
        Self {
            thread_id: ctx.thread_id,
            lane: ctx.lane,
            message: serde_json::to_value(ctx.message).expect("chat item"),
            batch: ctx.batch.into_iter().map(Into::into).collect(),
            history: ctx.history.into_iter().map(Into::into).collect(),
            state: ctx.state,
            context_scope_id: ctx.context_scope_id,
            context_scopes: ctx.context_scopes,
            read_only: ctx.read_only,
            task_id: ctx.task_id,
            tasks: ctx.tasks.into_iter().map(Into::into).collect(),
            caller_history: ctx.caller_history,
            answer_tool: ctx.answer_tool,
        }
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
            usage: output.usage,
            input_required: output.input_required,
            pending_call: output.pending_call.map(|p| PendingCall {
                name: p.name,
                call_id: p.call_id,
                answer_tool: p.answer_tool,
            }),
            interrupted: output.interrupted,
            metadata: output.metadata,
            error: output.error,
            events: output
                .events
                .into_iter()
                .map(|e| HostEvent { r#type: e.r#type, payload: e.payload })
                .collect(),
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
            usage: output.usage,
            input_required: output.input_required,
            pending_call: output.pending_call.map(|p| host::PendingCall {
                name: p.name,
                call_id: p.call_id,
                answer_tool: p.answer_tool,
            }),
            interrupted: output.interrupted,
            metadata: output.metadata,
            error: output.error,
            events: output
                .events
                .into_iter()
                .map(|e| host::HostEvent { r#type: e.r#type, payload: e.payload })
                .collect(),
        }
    }
}
