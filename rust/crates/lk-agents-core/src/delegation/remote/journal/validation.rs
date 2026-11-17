use super::{RemoteError, invalid};
use lk_agents_abi::delegation::remote::*;
use lk_agents_abi::session::InputRequestState;

/// Restored data carries observations, never authority to replay an incoherent operation.
pub fn validate(state: &RemoteAgentState) -> Result<(), RemoteError> {
    if state.endpoint.is_empty() || state.default_context_id.as_deref() == Some("") {
        return Err(invalid("missing endpoint or empty default context"));
    }
    for (position, event) in state.events.iter().enumerate() {
        let request = state
            .requests
            .get(&event.request_id)
            .ok_or_else(|| invalid("journal entry has no request"))?;
        if event.sequence != position as u64 {
            return Err(invalid("journal positions changed"));
        }
        match (&event.kind, &event.payload) {
            (A2AJournalKind::RequestRecorded, A2AJournalPayload::Request(body)) => {
                if request.request_event != event.sequence
                    || request.method
                        != body.method(
                            request.method
                                == lk_agents_abi::a2a_protocol::methods::SEND_STREAMING_MESSAGE,
                        )
                {
                    return Err(invalid("request content or SDK method changed"));
                }
            }
            (A2AJournalKind::ResponseReceived, A2AJournalPayload::Response(_))
                if request.dispatch_started => {}
            // An earlier observer can fail before dispatch and be replaced. Historical
            // failures do not reconstruct the current stream's mutable lifetime.
            (A2AJournalKind::RequestError, A2AJournalPayload::Error(_)) => {}
            _ => return Err(invalid("journal kind disagrees with its content or dispatch facts")),
        }
    }
    for (id, request) in &state.requests {
        let entry = state
            .events
            .get(request.request_event as usize)
            .ok_or_else(|| invalid("request content is missing"))?;
        if &entry.request_id != id || entry.kind != A2AJournalKind::RequestRecorded {
            return Err(invalid("request refers to another operation's content"));
        }
        if let Some(reference) = &request.task {
            if reference.task_id.is_empty() || reference.context_id.is_empty() {
                return Err(invalid("a known task needs both IDs"));
            }
            if state
                .tasks
                .get(&reference.task_id)
                .is_some_and(|task| task.task.context_id != reference.context_id)
            {
                return Err(invalid("request task context changed"));
            }
        }
        if let Some(reference) = &request.input_request {
            let input = state
                .tasks
                .get(&reference.task.task_id)
                .filter(|task| task.task.context_id == reference.task.context_id)
                .and_then(|task| task.input_requests.get(&reference.occurrence_id))
                .ok_or_else(|| invalid("request refers to a missing input occurrence"))?;
            let A2AJournalPayload::Request(A2ARequest::Send(body)) = &entry.payload else {
                return Err(invalid("a control operation cannot respond to an input occurrence"));
            };
            if input.response_message_id.as_deref() != Some(body.message.message_id.as_str()) {
                return Err(invalid("reserved input response identity changed"));
            }
        }
    }
    for (id, message) in &state.messages {
        if id.is_empty() || id != &message.message_id {
            return Err(invalid("message index identity changed"));
        }
    }
    for (id, task) in &state.tasks {
        if id.is_empty() || id != &task.task.id || task.task.context_id.is_empty() {
            return Err(invalid("task index identity changed"));
        }
        for input in task.input_requests.values() {
            if input.source_message_id.as_ref().is_some_and(|id| !state.messages.contains_key(id)) {
                return Err(invalid("input source message is missing"));
            }
            let responding =
                matches!(input.state, InputRequestState::Responding | InputRequestState::Responded);
            if responding && input.response_message_id.is_none() {
                return Err(invalid("a selected input response has no message"));
            }
            if input.state == InputRequestState::Open && input.response_message_id.is_some() {
                return Err(invalid("an open input occurrence already has a response"));
            }
            if let Some(response) = &input.response_message_id {
                let response = state
                    .messages
                    .get(response)
                    .ok_or_else(|| invalid("reserved response message is missing"))?;
                if response.task_id.as_deref() != Some(id.as_str())
                    || response.context_id.as_deref() != Some(task.task.context_id.as_str())
                {
                    return Err(invalid("the input response addresses another task"));
                }
            }
        }
    }
    Ok(())
}
