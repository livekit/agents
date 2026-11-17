//! Shared input admission over conversation records. The execution owner supplies lifecycle,
//! actual delivery receipts and persistence; these rules contain no model or transport loop.

use super::{InputAdmission, InputCandidate, InputRoute, SessionSnapshot};
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::session::{InputRequestRef, InputRequestState, SessionInputCommand};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::{ChatMessage, Role};
use std::collections::{BTreeMap, BTreeSet};

/// Tool eligibility comes from the caller's configured destination, not the identity of the
/// agent that originally asked. No response is guessed from the current agent's private chat.
pub fn held_remote_input(
    remote_id: &str,
    remote: &lk_agents_abi::delegation::remote::RemoteAgentState,
) -> Result<Option<InputRequestRef>, StorageError> {
    let mut selected = None;
    for (task_id, task) in &remote.tasks {
        for (request_id, input) in &task.input_requests {
            if input.state != InputRequestState::Responding || input.human_response_id.is_none() {
                continue;
            }
            if input.response_message_id.is_some() {
                return Err(invalid(
                    "the held response already has an outgoing operation; reconcile it first",
                ));
            }
            if selected.is_some() {
                return Err(invalid(
                    "several held responses address this remote; select an explicit input request",
                ));
            }
            selected = Some(InputRequestRef::RemoteTask {
                remote_id: remote_id.into(),
                task: lk_agents_abi::delegation::remote::RemoteTaskRef {
                    context_id: task.task.context_id.clone(),
                    task_id: task_id.clone(),
                },
                request_id: request_id.clone(),
            });
        }
    }
    Ok(selected)
}

pub fn validate_dispatch(
    state: &SessionSnapshot,
    command: &lk_agents_abi::delegation::remote::RemoteCommand,
) -> Result<(), StorageError> {
    use lk_agents_abi::delegation::remote::{A2ARequest, RemoteInput};
    let RemoteInput::Prepare {
        request: A2ARequest::Send(send),
        input_request: Some(reference),
        ..
    } = &command.input
    else {
        return Ok(());
    };
    let InputRequestRef::RemoteTask { remote_id, task, request_id } = reference else {
        return Ok(());
    };
    let response = state
        .remote_agents
        .get(remote_id)
        .and_then(|remote| remote.tasks.get(&task.task_id))
        .filter(|saved| saved.task.context_id == task.context_id)
        .and_then(|task| task.input_requests.get(request_id))
        .and_then(|input| input.human_response_id.as_deref());
    if let Some(id) = response {
        let human = canonical(&state.chat_history, id, Role::User)?;
        if send.message.text() != human.raw_text_content().as_deref() {
            return Err(invalid(
                "the prepared response no longer matches the reserved human message",
            ));
        }
    }
    Ok(())
}

pub fn response(
    state: &SessionSnapshot,
    reference: &InputRequestRef,
) -> Result<ChatMessage, StorageError> {
    let id = match reference {
        InputRequestRef::Agent { agent_id, request_id } => state
            .frames
            .get(&lk_agents_abi::text_session::FrameId { value: agent_id.clone() })
            .and_then(|agent| agent.input_requests.get(request_id))
            .filter(|input| input.state == InputRequestState::Responding)
            .and_then(|input| input.response_message.as_ref())
            .map(|message| &message.id),
        InputRequestRef::RemoteTask { remote_id, task, request_id } => state
            .remote_agents
            .get(remote_id)
            .and_then(|remote| remote.tasks.get(&task.task_id))
            .filter(|saved| saved.task.context_id == task.context_id)
            .and_then(|task| task.input_requests.get(request_id))
            .filter(|input| {
                input.state == InputRequestState::Responding && input.response_message_id.is_none()
            })
            .and_then(|input| input.human_response_id.as_ref()),
    }
    .ok_or_else(|| invalid("the input request has no held response; do not guess an answer"))?;
    canonical(&state.chat_history, id, Role::User).cloned()
}

/// Apply a detector verdict against the exact offered occurrences and held-response revisions.
/// The owner checkpoints the result before allowing downstream model or transport work.
/// A revised human message invalidates its verdict; classify the current version again.
pub fn route(
    state: &mut SessionSnapshot,
    message: &ChatMessage,
    candidates: &[InputCandidate],
    result: Option<Result<InputRoute, lk_agents_abi::errors::ProviderError>>,
) -> Result<InputAdmission, Invalid> {
    if message.role == Role::User
        && !matches!(canonical(&state.chat_history, &message.id, Role::User), Ok(current) if current == message)
    {
        return Err(Invalid(
            "the human message changed while response routing was in progress".into(),
        ));
    }
    let message_id = message.id.clone();
    let offered: BTreeMap<_, _> = candidates
        .iter()
        .map(|candidate| {
            (candidate.reference.clone(), candidate.response.as_ref().map(|held| held.id.clone()))
        })
        .collect();
    let mut accepted = Vec::new();
    let observation = match result {
        Some(Ok(InputRoute::Matched(requests)))
            if unique(&requests).is_ok()
                && requests.iter().all(|reference| offered.contains_key(reference)) =>
        {
            let commands = requests
                .iter()
                .map(|reference| match &offered[reference] {
                    Some(previous_message_id) => SessionInputCommand::Correct {
                        request: reference.clone(),
                        previous_message_id: previous_message_id.clone(),
                        message_id: message_id.clone(),
                    },
                    None => SessionInputCommand::Respond {
                        requests: vec![reference.clone()],
                        message_id: message_id.clone(),
                    },
                })
                .collect();
            let selected = apply(state, commands);
            match selected {
                Ok(()) => {
                    let observation = format!(
                        "This human message responds to these presented input requests: {}. \
                     The responses are reserved; they are not proof of remote delivery. \
                     Continue handling the entire message, including any other topic. \
                     Recognition is not approval for a business operation.",
                        serde_json::to_string(&requests).expect("input references serialize"),
                    );
                    accepted = requests;
                    observation
                }
                Err(_) => "The selected input requests changed. Keep them pending and clarify; \
                    do not guess a destination or forward an answer automatically."
                    .into(),
            }
        }
        Some(Ok(InputRoute::NoMatch)) => "This message was not recognized as a response \
            to a presented input request. Handle it as the current agent. Leave those \
            requests pending; do not forward it automatically as an answer."
            .into(),
        _ => "Response routing was unclear or unavailable. Handle the message and clarify \
            its intended destination naturally. Keep the input requests pending and do \
            not guess an answer destination."
            .into(),
    };
    Ok(InputAdmission { observation, requests: accepted })
}

/// Validate the entire group against one snapshot, then publish it together. A stale member
/// cannot partly reserve a group or retire an otherwise valid outgoing operation.
pub fn apply(
    state: &mut SessionSnapshot,
    commands: Vec<SessionInputCommand>,
) -> Result<(), StorageError> {
    let mut next = state.clone();
    for command in commands {
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
                let repeated = match target(&mut next, &request)? {
                    Target::Agent(input) => {
                        input.response_message.as_ref().is_some_and(|held| held.id == message_id)
                    }
                    Target::Remote(input) => {
                        input.human_response_id.as_deref() == Some(message_id.as_str())
                    }
                };
                if repeated {
                    continue;
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
                        return Err(invalid("the earlier response may already have been sent"));
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
    }
    *state = next;
    Ok(())
}

/// Recognition includes presented questions from inactive agents. Dispatch eligibility is
/// a later decision; lacking a tool cannot turn a clear earlier answer into unrelated input.
pub fn candidates(state: &SessionSnapshot) -> Vec<InputCandidate> {
    let mut candidates = Vec::new();
    for (agent_id, agent) in &state.frames {
        for (request_id, input) in &agent.input_requests {
            let response = match input.state {
                InputRequestState::Open => None,
                InputRequestState::Responding => {
                    let Some(response) = &input.response_message else { continue };
                    let Ok(response) = canonical(&state.chat_history, &response.id, Role::User)
                    else {
                        continue;
                    };
                    Some(response.clone())
                }
                _ => continue,
            };
            let Some(presentation) =
                presented(&state.chat_history, input.presented_message_id.as_deref())
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
                response,
            });
        }
    }
    for (remote_id, remote) in &state.remote_agents {
        for (task_id, task) in &remote.tasks {
            for (request_id, input) in &task.input_requests {
                let reference = InputRequestRef::RemoteTask {
                    remote_id: remote_id.clone(),
                    request_id: request_id.clone(),
                    task: lk_agents_abi::delegation::remote::RemoteTaskRef {
                        task_id: task_id.clone(),
                        context_id: task.task.context_id.clone(),
                    },
                };
                let response = match input.state {
                    InputRequestState::Open => None,
                    InputRequestState::Responding => {
                        if remote.requests.values().any(|operation| {
                            operation.input_request.as_ref() == Some(&reference)
                                && operation.dispatch_started
                        }) {
                            continue;
                        }
                        let Some(id) = &input.human_response_id else { continue };
                        let Ok(response) = canonical(&state.chat_history, id, Role::User) else {
                            continue;
                        };
                        Some(response.clone())
                    }
                    _ => continue,
                };
                let Some(presentation) =
                    presented(&state.chat_history, input.presented_message_id.as_deref())
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
                    reference,
                    question,
                    presentation: presentation.clone(),
                    response,
                });
            }
        }
    }
    candidates
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
