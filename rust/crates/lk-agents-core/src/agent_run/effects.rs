//! Work produced by [`super::machine::AgentRun`], executed by the runtime's agent-run driver.
//! Tool-call effects are converted into these invocation variants, not nested in another enum.

use super::{CallId, CheckpointId, RenderId, RequestId, StepId};
use lk_agents_abi::agent::run::{RunError, RunResult, StreamEvent, ValidationId};
use lk_agents_abi::agent::{OutputCandidate, ToolTemplate};
use lk_agents_abi::llm::tool_executor::Approval;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ChatContext, ChatItem, FunctionCall, ToolSpec};
use serde_json::{Map, Value};

/// Work for the runtime. Every outcome-bearing effect is answered by a correlated event.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentRunEffect {
    HostCheckpoint {
        checkpoint: CheckpointId,
        list_tools: bool,
    },
    HostCheckpointCancel {
        checkpoint: CheckpointId,
    },
    /// Render the `@agent.instructions` functions.
    RenderInstructions {
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
    /// Coordinated: plan and start the model request `request` (its chunks come back as
    /// [`super::AgentRunEvent::LlmChunk`]). `pending` are the items recorded since the last plan, merged by id;
    /// `message_id` names the step's assistant message; `tools` is what the step offers, which it
    /// may not call when `tools_forbidden`.
    PlanRequest {
        request: RequestId,
        message_id: String,
        pending: Vec<ChatItem>,
        tools: Vec<ToolSpec>,
        tools_forbidden: bool,
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
    /// Cancel a still-running body owned by the surrounding voice session. The supplied
    /// execution identity is session-scoped, unlike this invocation's `CallId`.
    CancelToolExecution {
        execution: u64,
    },
    /// Prepared arguments no body will run with.
    ToolArgumentsRelease {
        prepared: ResourceId,
    },
    /// Render the call's `tool_handling` template (`AgentCallbacks.render_template`), answered by
    /// [`super::AgentRunEvent::ToolHandlingTemplateRendered`].
    RenderToolHandlingTemplate {
        call: CallId,
        template: ToolTemplate,
    },
    ValidateOutput {
        validation: ValidationId,
        candidate: OutputCandidate,
    },
    OutputValidateCancel {
        validation: ValidationId,
    },
    /// Coordinated: handle canonical items (calls, then outputs, each in model order).
    /// `continues` when the run would request another step. Answered by
    /// [`super::AgentRunEvent::ToolResultsHandled`].
    HandleToolResults {
        step: StepId,
        /// Lossless correlated requests from this step; presentation belongs to the coordinator.
        input_requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
        items: Vec<ChatItem>,
        continues: bool,
        /// The one terminal outcome selected in model order, identified by function call ID.
        /// None when no call is terminal or a retry-budget failure takes precedence.
        terminal_call: Option<String>,
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
    Finished {
        result: Box<Result<RunResult, RunError>>,
        /// Retained even when failure has no RunResult. This does not turn failure into success.
        input_requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
    },
}
