use lk_agents_abi::session::{AgentCallState, AgentCallStatus, AgentState, ToolCallRef};
use lk_agents_abi::text_session::{ChildCallStatus, ParentCall, SessionFrame};

use super::{SessionState, State, StorageError, invalid};

fn origin(call: &ParentCall) -> ToolCallRef {
    ToolCallRef { agent_id: call.frame_id.value.clone(), call_id: call.call_id.clone() }
}

impl SessionState {
    pub(super) fn agent_view(&self, current: bool) -> Result<Option<AgentState>, StorageError> {
        let snapshot = self.0.lock().clone();
        let State::Conversation { state, .. } = snapshot.state else { return Ok(None) };
        let id = if current { &state.active_frame_id } else { &state.root_frame_id };
        let frame =
            state.frames.get(id).ok_or_else(|| invalid("session references a missing agent"))?;
        Ok(Some(frame_view(frame)?))
    }

    pub(super) fn agent_views(
        &self,
    ) -> Result<std::collections::BTreeMap<String, AgentState>, StorageError> {
        let snapshot = self.0.lock().clone();
        let State::Conversation { state, .. } = snapshot.state else {
            return Ok(Default::default());
        };
        state.frames.into_iter().map(|(id, frame)| Ok((id.value, frame_view(&frame)?))).collect()
    }
}

fn frame_view(frame: &SessionFrame) -> Result<AgentState, StorageError> {
    let calls = frame
        .pending_calls
        .iter()
        .map(|call| {
            let (status, agent_id) = match &call.status {
                ChildCallStatus::Approval => (AgentCallStatus::Approval, None),
                ChildCallStatus::Ready => (AgentCallStatus::Ready, None),
                ChildCallStatus::Running { frame_id } => {
                    (AgentCallStatus::Running, Some(frame_id.value.clone()))
                }
            };
            let state = AgentCallState {
                origin: origin(&call.parent),
                parent_invocation_id: call.invocation_id.clone(),
                parent_execution_id: call.execution_id.clone(),
                parent_tool_name: call.tool_name.clone(),
                agent_name: call.agent_name.clone(),
                validated_arguments: call
                    .validated_arguments
                    .as_object()
                    .ok_or_else(|| {
                        invalid("agent-as-tool arguments are not a validated JSON object")
                    })?
                    .clone(),
                identity: call.identity.clone(),
                status,
                agent_id,
            };
            Ok((call.parent.call_id.clone(), state))
        })
        .collect::<Result<_, StorageError>>()?;
    Ok(AgentState {
        agent_name: frame.agent_name.clone(),
        chat_ctx: frame.history.clone(),
        origin: frame.parent.as_ref().map(origin),
        calls,
        input_requests: frame.input_requests.clone(),
        completed: frame.completed,
        output: frame.output.clone(),
        error: frame.error.clone(),
    })
}
