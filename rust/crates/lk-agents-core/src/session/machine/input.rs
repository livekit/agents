use super::{Phase, Session};
use crate::session::InputCandidate;
use lk_agents_abi::session::{InputRequestRef, InputRequestState, SessionInputCommand};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::{ChatMessage, Role};
use std::collections::BTreeSet;

impl Session {
    /// Select every target against one snapshot, then commit them together. An invalid or stale
    /// member cannot reserve the rest of a grouped answer. Runtime checkpoints before replying.
    pub fn input(&mut self, command: SessionInputCommand) -> Result<(), StorageError> {
        if matches!(self.phase, Phase::Done) {
            return Err(invalid("the conversation owner has ended"));
        }
        if (self.closing || self.persistence_failure.is_some())
            && !matches!(command, SessionInputCommand::Presented { .. })
        {
            return Err(invalid("the conversation owner is closing"));
        }
        let mut next = self.state.clone();
        match command {
            SessionInputCommand::Presented { requests, message_id } => {
                canonical(&next.chat_history, &message_id, Role::Assistant)?;
                unique(&requests)?;
                for reference in requests {
                    match target(&mut next, &reference)? {
                        Target::Agent(input) => {
                            input.presented_message_id = Some(message_id.clone())
                        }
                        Target::Remote(input) => {
                            input.presented_message_id = Some(message_id.clone())
                        }
                    }
                }
            }
            SessionInputCommand::Respond { requests, message_id } => {
                let response = canonical(&next.chat_history, &message_id, Role::User)?.clone();
                if response.raw_text_content().unwrap_or_default().trim().is_empty() {
                    return Err(invalid("an empty message cannot answer an input request"));
                }
                unique(&requests)?;
                for reference in requests {
                    match target(&mut next, &reference)? {
                        Target::Agent(input) => {
                            if input
                                .response_message
                                .as_ref()
                                .is_some_and(|held| held.id == message_id)
                            {
                                continue;
                            }
                            available(input.state, input.presented_message_id.as_deref())?;
                            input.state = InputRequestState::Responding;
                            input.response_message = Some(response.clone());
                        }
                        Target::Remote(input) => {
                            if input.human_response_id.as_deref() == Some(message_id.as_str()) {
                                continue;
                            }
                            available(input.state, input.presented_message_id.as_deref())?;
                            input.state = InputRequestState::Responding;
                            input.human_response_id = Some(message_id.clone());
                        }
                    }
                }
            }
            SessionInputCommand::Correct { request, previous_message_id, message_id } => {
                let response = canonical(&next.chat_history, &message_id, Role::User)?.clone();
                if response.raw_text_content().unwrap_or_default().trim().is_empty() {
                    return Err(invalid("an empty message cannot replace an input response"));
                }
                // Once dispatch may have begun, preserve its exact message and reconcile it.
                if let InputRequestRef::RemoteTask { remote_id, .. } = &request {
                    let remote = next
                        .remote_agents
                        .get_mut(remote_id)
                        .ok_or_else(|| invalid("unknown remote input owner"))?;
                    let dispatched = remote.requests.values().any(|operation| {
                        operation.input_request.as_ref() == Some(&request)
                            && operation.dispatch_started
                    });
                    if dispatched {
                        let held = match target(&mut next, &request)? {
                            Target::Remote(input) => input.human_response_id.as_deref(),
                            Target::Agent(_) => None,
                        };
                        if held != Some(message_id.as_str()) {
                            return Err(invalid("the earlier response may already have been sent"));
                        }
                    } else {
                        for operation in remote.requests.values_mut() {
                            if operation.input_request.as_ref() == Some(&request) {
                                operation.ended = true;
                            }
                        }
                    }
                }
                match target(&mut next, &request)? {
                    Target::Agent(input) => {
                        if input.response_message.as_ref().is_some_and(|held| held.id == message_id)
                        {
                            return Ok(());
                        }
                        if input.state != InputRequestState::Responding
                            || input.response_message.as_ref().map(|held| held.id.as_str())
                                != Some(previous_message_id.as_str())
                        {
                            return Err(invalid(
                                "the selected local response changed or was consumed",
                            ));
                        }
                        input.response_message = Some(response);
                    }
                    Target::Remote(input) => {
                        if input.human_response_id.as_deref() == Some(message_id.as_str()) {
                            return Ok(());
                        }
                        if input.state != InputRequestState::Responding
                            || input.human_response_id.as_deref()
                                != Some(previous_message_id.as_str())
                        {
                            return Err(invalid(
                                "the selected remote response changed or was consumed",
                            ));
                        }
                        input.human_response_id = Some(message_id);
                        input.response_message_id = None;
                    }
                }
            }
        }
        self.state = next;
        Ok(())
    }

    /// Recognition includes presented questions from inactive agents. Dispatch eligibility is
    /// a later decision; lacking a tool cannot turn a clear earlier answer into unrelated input.
    pub fn input_candidates(&self) -> Vec<InputCandidate> {
        let mut candidates = Vec::new();
        for (agent_id, agent) in &self.state.frames {
            for (request_id, input) in &agent.input_requests {
                if input.state != InputRequestState::Open {
                    continue;
                }
                let Some(presentation) =
                    presented(&self.state.chat_history, input.presented_message_id.as_deref())
                else {
                    continue;
                };
                candidates.push(InputCandidate {
                    reference: InputRequestRef::Agent {
                        agent_id: agent_id.value.clone(),
                        request_id: request_id.clone(),
                    },
                    question: input.source_message.raw_text_content().unwrap_or_default(),
                    presentation: presentation.clone(),
                });
            }
        }
        for (remote_id, remote) in &self.state.remote_agents {
            for (task_id, task) in &remote.tasks {
                for (request_id, input) in &task.input_requests {
                    if input.state != InputRequestState::Open {
                        continue;
                    }
                    let Some(presentation) =
                        presented(&self.state.chat_history, input.presented_message_id.as_deref())
                    else {
                        continue;
                    };
                    let question = input
                        .source_message_id
                        .as_ref()
                        .and_then(|id| remote.messages.get(id))
                        .and_then(|message| message.text())
                        .unwrap_or_default()
                        .to_owned();
                    candidates.push(InputCandidate {
                        reference: InputRequestRef::RemoteTask {
                            remote_id: remote_id.clone(),
                            request_id: request_id.clone(),
                            task: lk_agents_abi::delegation::remote::RemoteTaskRef {
                                task_id: task_id.clone(),
                                context_id: task.task.context_id.clone(),
                            },
                        },
                        question,
                        presentation: presentation.clone(),
                    });
                }
            }
        }
        candidates
    }
}

enum Target<'a> {
    Agent(&'a mut lk_agents_abi::session::AgentInputRequest),
    Remote(&'a mut lk_agents_abi::delegation::remote::InputRequest),
}

fn target<'a>(
    state: &'a mut lk_agents_abi::text_session::SessionSnapshot,
    reference: &InputRequestRef,
) -> Result<Target<'a>, StorageError> {
    match reference {
        InputRequestRef::Agent { agent_id, request_id } => {
            if agent_id.is_empty() || request_id.is_empty() {
                return Err(invalid("empty local input reference"));
            }
            state
                .frames
                .get_mut(&lk_agents_abi::text_session::FrameId { value: agent_id.clone() })
                .and_then(|agent| agent.input_requests.get_mut(request_id))
                .map(Target::Agent)
                .ok_or_else(|| invalid("unknown local input occurrence"))
        }
        InputRequestRef::RemoteTask { remote_id, task, request_id } => {
            if remote_id.is_empty()
                || task.task_id.is_empty()
                || task.context_id.is_empty()
                || request_id.is_empty()
            {
                return Err(invalid("empty remote input reference"));
            }
            state
                .remote_agents
                .get_mut(remote_id)
                .and_then(|remote| remote.tasks.get_mut(&task.task_id))
                .filter(|observed| observed.task.context_id == task.context_id)
                .and_then(|task| task.input_requests.get_mut(request_id))
                .map(Target::Remote)
                .ok_or_else(|| invalid("unknown remote input occurrence or context"))
        }
    }
}

fn canonical<'a>(
    history: &'a [ChatMessage],
    id: &str,
    role: Role,
) -> Result<&'a ChatMessage, StorageError> {
    history
        .iter()
        .find(|message| message.id == id && message.role == role)
        .ok_or_else(|| invalid("the input or delivery is not in the canonical human conversation"))
}

fn presented<'a>(history: &'a [ChatMessage], id: Option<&str>) -> Option<&'a ChatMessage> {
    canonical(history, id?, Role::Assistant).ok()
}

fn unique(requests: &[InputRequestRef]) -> Result<(), StorageError> {
    if requests.is_empty() || requests.iter().collect::<BTreeSet<_>>().len() != requests.len() {
        return Err(invalid("input selection must contain distinct occurrences"));
    }
    Ok(())
}

fn available(state: InputRequestState, presented: Option<&str>) -> Result<(), StorageError> {
    if state != InputRequestState::Open || presented.is_none() {
        return Err(invalid("the input occurrence is not open and presented"));
    }
    Ok(())
}

fn invalid(message: &str) -> StorageError {
    StorageError::Invalid(message.into())
}
