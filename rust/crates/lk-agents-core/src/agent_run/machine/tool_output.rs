use super::*;
use crate::agent_run::ToolOutputId;
use lk_agents_abi::agent::run::SayOutcome;
#[cfg(test)]
#[path = "tool_output_tests.rs"]
mod tests;

impl AgentRun {
    /// Shared owners provide correlation; invocation startup must not consume another pending
    /// question simply because a new human message arrived. Approval still requires a repeat.
    /// Only a managed native owner that retains the body may release its model step.
    pub fn with_live_tools(mut self) -> Self {
        self.live_tools = true;
        self
    }

    /// A session-wide duplicate replacement requested cancellation of this run's body.
    /// Mark it before cancellation can return, so the expected outcome does not abort the run.
    pub fn tool_cancellation_requested(&mut self, call: CallId) {
        if let Some(acting) = self.phase.acting() {
            acting.calls.request_cancellation(call);
        }
    }

    /// Canonical output identity belongs to the source machine, not the transport or the
    /// latest speaking/model turn. Session uses this planner without duplicating ownership.
    pub(crate) fn plan_tool_output(
        &mut self,
        call: CallId,
        text: String,
        now: Ts,
    ) -> Option<ChatMessage> {
        let active = match &self.phase {
            Phase::Requesting(_, acting) | Phase::Acting(acting) => {
                acting.calls.running_call(call).is_some()
            }
            _ => false,
        };
        active.then(|| ChatMessage::text(self.item_id(), Role::Assistant, text, self.stamp(now)))
    }

    /// The surrounding source owner has accepted this same canonical message. This factual
    /// assistant output grants no human/approval authority and is not terminal verbatim output.
    pub(crate) fn record_tool_output(&mut self, message: ChatMessage) {
        if !self.history.iter().any(|item| item.id() == message.id) {
            self.history.push(ChatItem::Message(message.clone()));
        }
        if let Phase::Checkpoint(pending) = &mut self.phase {
            pending.record_tool_output(&message);
        }
    }

    pub(super) fn request_tool_output(
        &mut self,
        output: ToolOutputId,
        call: Option<CallId>,
        text: String,
        stream: bool,
        now: Ts,
        effects: &mut Vec<AgentRunEffect>,
    ) {
        if self.tool_outputs.contains_key(&output) {
            return;
        }
        // A lost tool operation, or one no longer running, has no live source to speak for.
        match call.and_then(|call| self.plan_tool_output(call, text, now)) {
            Some(message) => {
                self.tool_outputs.insert(output, (message.clone(), stream));
                effects.push(AgentRunEffect::DeliverToolOutput {
                    output,
                    message: Box::new(message.clone()),
                });
                if stream {
                    effects.push(AgentRunEffect::Emit(StreamEvent::TextDelta {
                        message_id: message.id.clone(),
                        delta: message.raw_text_content().unwrap_or_default(),
                    }));
                }
            }
            None => effects
                .push(AgentRunEffect::ToolOutputSettled { output, outcome: SayOutcome::Failed }),
        }
    }

    pub(super) fn delivered_tool_output(
        &mut self,
        output: ToolOutputId,
        outcome: SayOutcome,
        effects: &mut Vec<AgentRunEffect>,
    ) {
        let Some((message, stream)) = self.tool_outputs.remove(&output) else {
            return;
        };
        if stream {
            effects.push(AgentRunEffect::Emit(StreamEvent::TextEnd {
                message_id: message.id.clone(),
                interrupted: outcome != SayOutcome::Delivered,
            }));
        }
        // Words the ended run can no longer record were cut off by its close.
        let outcome = match outcome {
            SayOutcome::Delivered if matches!(self.phase, Phase::Done) => SayOutcome::Interrupted,
            SayOutcome::Delivered => {
                self.record_tool_output(message);
                SayOutcome::Delivered
            }
            undelivered => undelivered,
        };
        effects.push(AgentRunEffect::ToolOutputSettled { output, outcome });
    }
}
