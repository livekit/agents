//! Resume already-admitted remote origins through the normal tool executor, without a model.

use super::*;

pub(super) struct Reconciliation {
    items: BTreeSet<String>,
}

impl AgentRun {
    pub(crate) fn commit_remote_reports(
        &mut self,
        reports: &BTreeMap<String, lk_agents_abi::llm::tool_executor::RemoteReport>,
    ) {
        for item in &mut self.history {
            if let ChatItem::FunctionCall(call) = item
                && let Some(report) = reports.get(&call.id)
            {
                call.extra.insert(
                    lk_agents_abi::llm::tool_executor::REMOTE_REPORT_KEY.into(),
                    serde_json::to_value(report).expect("native report provenance serializes"),
                );
            }
        }
    }

    /// Only Session's journal-backed saved-origin plan may enter this path. Canonical call
    /// IDs and original arguments survive; no local coroutine or model-selected call is replayed.
    pub(crate) fn resume_remote(
        &mut self,
        calls: Vec<FunctionCall>,
        now: Ts,
    ) -> Vec<AgentRunEffect> {
        if !matches!(self.phase, Phase::Idle) {
            return Vec::new();
        }
        let mut effects = Vec::new();
        if calls.is_empty()
            || calls.iter().any(|call| {
                !self.agent.tools.iter().any(|tool| {
                    tool.spec.name == call.name
                        && tool.remote.is_some()
                        && tool.local_agent.is_none()
                })
            })
        {
            self.finish(
                Err(RunError::Host("native recovery requires configured remote calls".into())),
                &mut effects,
            );
            return effects;
        }
        self.history = std::mem::take(&mut self.input.history);
        self.first_new = self.history.len();
        self.reconciliation = Some(Reconciliation {
            items: calls.iter().flat_map(|call| [call.id.clone(), call.output_id()]).collect(),
        });
        // Progress does not release this private step: actual body outcomes and cleanup are
        // the barrier before an accepted child can reach its parent.
        self.live_tools = false;
        let mut acting = Acting::new(&BTreeSet::new());
        for call in calls {
            let id = CallId(self.mint());
            acting.calls.recover(id, call, &self.agent.tools, &mut effects);
        }
        self.phase = Phase::Acting(acting);
        self.step_settled(now, &mut effects);
        effects
    }

    pub(super) fn recorded_here(&self, at: usize, item: &ChatItem) -> bool {
        !self.context_items.contains(item.id())
            && (at >= self.first_new
                || self
                    .reconciliation
                    .as_ref()
                    .is_some_and(|resume| resume.items.contains(item.id())))
    }
}

#[cfg(test)]
#[path = "remote_resume_tests.rs"]
mod tests;
