//! Native report provenance distinguishes pending bodies from already-final tracked messages.

use lk_agents_abi::FunctionCall;
use lk_agents_abi::llm::tool_executor::{REMOTE_REPORT_KEY, RemoteReport, ToolOutcome};

pub(crate) fn remote_report(call: &FunctionCall) -> Option<RemoteReport> {
    serde_json::from_value(call.extra.get(REMOTE_REPORT_KEY)?.clone()).ok()
}

pub(crate) fn record_remote_report(call: &mut FunctionCall, outcome: &ToolOutcome) {
    let report = match outcome {
        ToolOutcome::Update { .. } | ToolOutcome::Working(_) => RemoteReport::Pending,
        ToolOutcome::InputRequired { .. } => RemoteReport::InputRequired,
        ToolOutcome::Cancelled => RemoteReport::Cancelled,
        _ if crate::tool_outcome::shape(outcome, crate::tool_outcome::Mode::Text).is_error => {
            RemoteReport::Failed
        }
        _ => RemoteReport::Completed,
    };
    call.extra.insert(
        REMOTE_REPORT_KEY.into(),
        serde_json::to_value(report).expect("a native report stage serializes"),
    );
}
