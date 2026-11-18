//! Adapt native voice occurrence effects to the same serialized conversation records.

use super::{VoiceEvent, VoiceRuntime};
use lk_agents_abi::ChatMessage;
use lk_agents_abi::session::{
    AgentInputRequest, InputRequestRef, InputRequestState, SessionInputCommand, ToolCallRef,
};
use lk_agents_core::session::input;

#[cfg(test)]
#[path = "local_input_tests.rs"]
mod tests;

impl VoiceRuntime {
    pub(super) fn publish_local_input_candidates(&self) {
        if self.driver.stopped() {
            return;
        }
        match self.conversation_snapshot() {
            Ok(state) => {
                let requests = input::candidates(&state)
                    .into_iter()
                    .filter_map(|candidate| {
                        candidate.reference.remote().is_none().then_some(candidate.reference)
                    })
                    .collect();
                let _ = self.reports.send(VoiceEvent::LocalInputCandidates(requests));
            }
            Err(error) => {
                let _ = self.reports.send(VoiceEvent::ConversationStateFailed(error));
            }
        }
    }

    pub(super) fn retain_local_input(
        &mut self,
        request_id: String,
        call_id: String,
        source: ChatMessage,
    ) {
        let result = (|| {
            let mut state = self.conversation_snapshot()?;
            let agent_id = state.root_frame_id.value.clone();
            let reference = InputRequestRef::Agent {
                agent_id: agent_id.clone(),
                request_id: request_id.clone(),
            };
            input::apply(
                &mut state,
                vec![SessionInputCommand::Retain {
                    agent_id: agent_id.clone(),
                    request_id: request_id.clone(),
                    input: Box::new(AgentInputRequest {
                        state: InputRequestState::Open,
                        source_message: source,
                        response_message: None,
                        origin: Some(ToolCallRef { agent_id, call_id }),
                        presented_message_id: None,
                    }),
                }],
            )?;
            self.session_sink.as_ref().expect("snapshot has a binding").stage_voice(state)?;
            Ok(reference)
        })();
        let event = match result {
            Ok(reference) => VoiceEvent::LocalInputRetained { request_id, reference },
            Err(error) => VoiceEvent::ConversationStateFailed(error),
        };
        let _ = self.reports.send(event);
    }

    pub(super) fn close_local_inputs(&mut self, requests: Vec<InputRequestRef>) {
        let result = (|| {
            let mut state = self.conversation_snapshot()?;
            input::apply(&mut state, vec![SessionInputCommand::Close { requests }])?;
            self.session_sink.as_ref().expect("snapshot has a binding").stage_voice(state)
        })();
        if let Err(error) = result {
            let _ = self.reports.send(VoiceEvent::ConversationStateFailed(error));
        } else {
            self.publish_local_input_candidates();
            self.local_inputs_checkpoint_pending = true;
        }
    }
}
