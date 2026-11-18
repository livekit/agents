use super::*;

impl ToolCalls {
    pub(crate) fn remote_pending(&mut self, id: CallId) {
        if let Some(call) = self.call_mut(id) {
            call.item.extra.insert(
                lk_agents_abi::llm::tool_executor::REMOTE_REPORT_KEY.into(),
                serde_json::to_value(lk_agents_abi::llm::tool_executor::RemoteReport::Pending)
                    .expect("native report provenance serializes"),
            );
        }
    }

    /// Only Session's native saved-origin recovery uses this. It is the same already-admitted
    /// operation, so duplicate/read-back policy cannot turn a GetTask into another call.
    pub(crate) fn recover<E: From<ToolCallEffect>>(
        &mut self,
        id: CallId,
        item: FunctionCall,
        tools: &[FunctionToolInfo],
        effects: &mut Vec<E>,
    ) {
        let phase = prepare(id, &item, tools, effects);
        let phase =
            if matches!(phase, Phase::Preparing { .. }) { Phase::Recovering } else { phase };
        self.calls.push(Call {
            id,
            item,
            phase,
            cancellation_requested: false,
            reports_remote: false,
        });
    }
}
