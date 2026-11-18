use super::*;

impl AgentRun {
    pub(super) fn tool_settled(
        &mut self,
        call: CallId,
        report: lk_agents_abi::llm::tool_executor::ToolReport,
        now: Ts,
        fx: &mut Vec<AgentRunEffect>,
    ) {
        // On a call, the body's first update answers its step; the body runs on detached.
        let detaches = (self.live_tools || self.input.mode == tool_outcome::Mode::Voice)
            && matches!(report.outcome, ToolOutcome::Update { .. });
        let cancelled = match self.phase.acting() {
            Some(a) if detaches => {
                if a.calls.item(call).is_some_and(|item| {
                    self.agent
                        .tools
                        .iter()
                        .any(|tool| tool.spec.name == item.name && tool.remote.is_some())
                }) {
                    a.calls.remote_pending(call);
                }
                if self.live_tools
                    && a.calls.item(call).is_some_and(|item| {
                        self.agent
                            .tools
                            .iter()
                            .any(|tool| tool.spec.name == item.name && tool.remote.is_none())
                    })
                {
                    a.calls.update_extra(
                        call,
                        [(
                            lk_agents_abi::llm::tool_executor::LIVE_LOCAL_PENDING_KEY.into(),
                            serde_json::Value::Bool(true),
                        )]
                        .into_iter()
                        .collect(),
                    );
                }
                a.calls.detached(call, report.outcome, fx);
                false
            }
            Some(a) => a.calls.body_settled(call, report, fx),
            None => false,
        };
        if cancelled && self.reconciliation.is_none() {
            self.abort(RunError::Cancelled, now, fx);
        } else {
            self.step_settled(now, fx);
        }
    }

    pub(super) fn mint(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }
}
