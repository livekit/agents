pub mod effects;
pub mod machine;
mod remote_report;
pub(crate) use remote_report::{record_remote_report, remote_report};

pub use effects::ToolCallEffect;
pub(crate) use machine::FinalCandidate;
pub use machine::{
    Admission, PrepareFailure, Recorded, RunningToolExecution, Settled, ToolCalls, admit,
    rendered_template,
};

/// Progress answers a model step, not the local business operation. Saved copies cannot restore
/// that operation and must refuse continuation until its owner records a real final report.
pub(crate) fn provisional(call: &lk_agents_abi::FunctionCall) -> bool {
    call.extra.contains_key(lk_agents_abi::llm::tool_executor::LIVE_LOCAL_PENDING_KEY)
}

impl Settled {
    /// One reported question, independent of step assembly or sibling cancellation.
    pub(crate) fn input_requirement(
        &self,
        invocation_id: &str,
        item: &lk_agents_abi::FunctionCall,
    ) -> Option<lk_agents_abi::agent::run::RunInputRequirement> {
        let lk_agents_abi::llm::tool_executor::ToolOutcome::InputRequired { question, verbatim } =
            &self.outcome
        else {
            return None;
        };
        Some(lk_agents_abi::agent::run::RunInputRequirement {
            invocation_id: invocation_id.to_owned(),
            call: lk_agents_abi::a2a::turn::PendingCall {
                id: item.id.clone(),
                name: item.name.clone(),
                call_id: item.call_id.clone(),
            },
            question: question.clone(),
            verbatim: *verbatim,
            input_request: self.input_request.clone(),
            presentation_message_id: None,
        })
    }
}
