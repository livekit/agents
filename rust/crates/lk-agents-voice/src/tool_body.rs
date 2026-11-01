//! Host tool execution policy shared by the runtime's body and progress pumps.

use lk_agents_abi::llm::tool_executor::{ToolOutcome, ToolReport};

pub fn validates_completion(task: bool, report: &ToolReport) -> bool {
    task && matches!(report.outcome, ToolOutcome::Completed { .. })
}

pub fn completion(task: bool, mut report: ToolReport) -> ToolReport {
    if !task {
        report.transition = None;
    }
    report
}
