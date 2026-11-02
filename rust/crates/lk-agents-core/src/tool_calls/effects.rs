//! Host work produced by [`super::machine::ToolCalls`], consumed by invocation coordination.
//! [`crate::effects::AgentRunEffect`] projects this protocol into canonical invocation work.

use crate::agent_run::CallId;
use lk_agents_abi::FunctionCall;
use lk_agents_abi::agent::ToolTemplate;
use lk_agents_abi::agent::run::StreamEvent;
use lk_agents_abi::llm::tool_executor::Approval;
use lk_agents_abi::resources::ResourceId;
use serde_json::{Map, Value};

/// Host work for the coordinator. Every outcome-bearing effect is answered for its call.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallEffect {
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
    /// Cancel a session-owned body outside this step, using its owner's stable operation ID.
    CancelExecution {
        execution: u64,
    },
    /// Prepared arguments no body will run with.
    ReleaseArguments {
        prepared: ResourceId,
    },
    /// Render the call's `tool_handling` template; answered with the text, or `None` for the
    /// framework's.
    RenderToolHandlingTemplate {
        call: CallId,
        template: ToolTemplate,
    },
    Emit(StreamEvent),
    /// Diagnostic: malformed model arguments were repaired.
    ArgumentsRepaired {
        call_id: String,
        error: String,
        repaired: Value,
    },
}
