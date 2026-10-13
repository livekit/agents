pub use crate::workflows::ctx::{
    Task, TaskCtx, TaskOptions, ToolError, ToolReply, Warning, args,
};
pub use crate::workflows::driver::{NativeTask, Outcome, TaskDriver};
pub use crate::workflows::handle::{
    HostTaskHandle, JoinError, TaskError, TaskHandle, TaskId, TaskPhase,
};
pub use crate::workflows::tools::{
    ConfirmSlot, ToolBox, confirmation_required, update_tool_flags,
};

mod ctx;
mod driver;
mod handle;
mod tools;

pub mod builtins;

/// The `chat_ctx.merge(...)` flags at an inline task's return. `exclude_instructions` is always
/// true (else the task's prompt lands in the parent's history).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextMerge {
    pub exclude_function_call: bool,
    pub exclude_instructions: bool,
}

impl ContextMerge {
    /// The merge an `AgentTask(preserve_function_call_history=..)` asks for on its way out.
    pub fn for_task(preserve_function_call_history: bool) -> Self {
        ContextMerge {
            exclude_function_call: !preserve_function_call_history,
            exclude_instructions: true,
        }
    }
}
