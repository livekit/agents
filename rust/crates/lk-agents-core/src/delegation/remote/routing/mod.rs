//! Proposed-call routing is separate from recognizing a human response. A reserved answer
//! never redirects unrelated business arguments merely because it belongs to the same remote.

mod prompt;

use lk_agents_abi::delegation::remote::{
    A2AJournalPayload, A2ARequest, RemoteAgentState, RemoteTaskRef,
};
pub use lk_agents_abi::delegation::routing::{
    CallAdmission, CallCandidate, CallRoute, CallRoutingRequest,
};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::session::{InputCandidate, InputRequestRef, InputRequestState};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::text_session::SessionSnapshot;
pub use prompt::{INSTRUCTIONS, TOOL, parse, prompt, tool};
use std::collections::BTreeSet;

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;

pub const INPUT_OFFERS_KEY: &str = "lk.delegation.input_offers";

/// Disabled routing starts omitted-task work and continues an explicit selection. An enabled
/// classifier with no candidates has the same unambiguous new-work plan; classifier failures
/// never fall back to this policy.
pub fn without_classifier(request: &CallRoutingRequest) -> CallRoute {
    match &request.selected_task {
        Some(task) => CallRoute::Continue(task.clone()),
        None => CallRoute::New,
    }
}

/// Owner-authored model-turn metadata overwrites any provider-supplied value at admission.
pub fn capture_offers(call: &mut lk_agents_abi::FunctionCall, offered: &[InputCandidate]) {
    call.extra.insert(
        INPUT_OFFERS_KEY.into(),
        serde_json::to_value(offered).expect("native input candidates serialize"),
    );
}

pub fn captured_offers(
    call: &lk_agents_abi::FunctionCall,
) -> Result<Vec<InputCandidate>, StorageError> {
    match call.extra.get(INPUT_OFFERS_KEY) {
        Some(serde_json::Value::Null) => Err(invalid(
            "human response routing was unavailable; clarify or retry before forwarding remote work",
        )),
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|error| invalid(&format!("invalid captured input occurrences: {error}"))),
        None => Ok(Vec::new()),
    }
}

/// Capture at model-turn admission, including responses this human turn already forwarded.
/// A subsequent call can recognize that admission without treating all later calls as duplicates.
pub fn offered(state: &SessionSnapshot) -> Vec<InputCandidate> {
    let mut candidates = crate::session::input::candidates(state);
    let Some(human) =
        state.chat_history.iter().rev().find(|message| message.role == lk_agents_abi::Role::User)
    else {
        return candidates;
    };
    for (remote_id, remote) in &state.remote_agents {
        for (task_id, task) in &remote.tasks {
            for (request_id, input) in &task.input_requests {
                if !matches!(
                    input.state,
                    InputRequestState::Responding | InputRequestState::Responded
                ) || input.human_response_id.as_deref() != Some(human.id.as_str())
                    || input.response_message_id.is_none()
                {
                    continue;
                }
                let Some(presentation) = state.chat_history.iter().find(|message| {
                    message.role == lk_agents_abi::Role::Assistant
                        && Some(&message.id) == input.presented_message_id.as_ref()
                }) else {
                    continue;
                };
                let reference = InputRequestRef::RemoteTask {
                    remote_id: remote_id.clone(),
                    task: RemoteTaskRef {
                        task_id: task_id.clone(),
                        context_id: task.task.context_id.clone(),
                    },
                    request_id: request_id.clone(),
                };
                if candidates.iter().any(|candidate| candidate.reference == reference) {
                    continue;
                }
                candidates.push(InputCandidate {
                    reference,
                    question: input
                        .source_message_id
                        .as_ref()
                        .and_then(|id| remote.messages.get(id))
                        .and_then(|message| message.text())
                        .unwrap_or_default()
                        .into(),
                    presentation: presentation.clone(),
                    response: Some(human.clone()),
                });
            }
        }
    }
    candidates
}

/// Refresh reservations without admitting occurrences that appeared while classification
/// was pending. A replacement question cannot inherit an older human message's authority.
pub fn refresh_offered(
    state: &SessionSnapshot,
    captured: &[InputCandidate],
) -> Vec<InputCandidate> {
    let current = offered(state);
    captured
        .iter()
        .filter_map(|candidate| {
            current.iter().find(|current| current.reference == candidate.reference).cloned()
        })
        .collect()
}

/// `offered` was captured when the frontend model turn began. Looking up today's newest
/// question here would allow an older model turn to answer a different question occurrence.
pub fn request(
    state: &SessionSnapshot,
    remote_id: &str,
    endpoint: &str,
    proposed: serde_json::Value,
    selected_task: Option<&str>,
    offered: &[InputCandidate],
) -> Result<CallRoutingRequest, StorageError> {
    let remote = crate::session::remote::state(state, remote_id, endpoint)?;
    let selected_task = selected_task
        .map(|task_id| {
            let remote = remote.ok_or_else(|| invalid("this remote has no known tasks"))?;
            crate::session::remote::task(remote, task_id)
        })
        .transpose()?;
    let mut candidates = Vec::new();
    if let Some(remote) = remote {
        for (task_id, saved) in &remote.tasks {
            if selected_task.as_ref().is_some_and(|task| &task.task_id != task_id) {
                continue;
            }
            let task = RemoteTaskRef {
                task_id: task_id.clone(),
                context_id: saved.task.context_id.clone(),
            };
            let responses: Vec<_> = offered
                .iter()
                .filter(|candidate| {
                    candidate.response.is_some()
                        && candidate
                            .reference
                            .remote()
                            .is_some_and(|(owner, target, _)| owner == remote_id && target == &task)
                })
                .cloned()
                .collect();
            // A terminal task cannot accept new continuation, but a duplicate answer can
            // still refer to its original admission rather than silently starting work again.
            if saved.task.status.state.is_terminal() && responses.is_empty() {
                continue;
            }
            if task.task_id.is_empty() || task.context_id.is_empty() || saved.task.id != *task_id {
                return Err(invalid("the saved task identity is invalid"));
            }
            candidates.push(CallCandidate {
                messages: task_messages(remote, &task),
                task,
                state: saved.task.status.state.clone(),
                responses,
            });
        }
    }
    Ok(CallRoutingRequest {
        remote_id: remote_id.into(),
        endpoint: endpoint.into(),
        proposed,
        selected_task,
        history: state.chat_history.clone(),
        candidates,
    })
}

/// Validate model output against its offered tasks and exact reserved human message. This
/// does not consume a response; the serialized outgoing Prepare reserves it before any I/O.
pub fn admit(
    state: &SessionSnapshot,
    request: &CallRoutingRequest,
    verdict: Option<Result<CallRoute, ProviderError>>,
) -> Result<CallAdmission, StorageError> {
    let remote = crate::session::remote::state(state, &request.remote_id, &request.endpoint)?;
    match verdict {
        Some(Ok(CallRoute::New)) if request.selected_task.is_none() => Ok(CallAdmission::New),
        Some(Ok(CallRoute::Continue(task))) => {
            validate_task(remote, request, &task)?;
            Ok(CallAdmission::Continue(task))
        }
        Some(Ok(CallRoute::Respond(reference))) => {
            let Some((owner, task, _)) = reference.remote() else {
                return Err(invalid("a remote call cannot answer a local agent input request"));
            };
            if owner != request.remote_id {
                return Err(invalid("the response belongs to another remote"));
            }
            let offered = request
                .candidates
                .iter()
                .filter(|candidate| &candidate.task == task)
                .flat_map(|candidate| &candidate.responses)
                .find(|candidate| candidate.reference == reference)
                .and_then(|candidate| candidate.response.as_ref())
                .ok_or_else(|| invalid("the response was not offered to this model turn"))?;
            let saved = remote
                .filter(|_| request.selected_task.as_ref().is_none_or(|selected| selected == task))
                .and_then(|remote| remote.tasks.get(&task.task_id))
                .filter(|saved| saved.task.context_id == task.context_id)
                .and_then(|saved| {
                    saved.input_requests.get(reference.remote().expect("remote checked").2)
                })
                .filter(|input| {
                    matches!(
                        input.state,
                        InputRequestState::Responding | InputRequestState::Responded
                    ) && input.human_response_id.as_deref() == Some(offered.id.as_str())
                })
                .ok_or_else(|| invalid("the offered input occurrence changed or was consumed"))?;
            let current = state.chat_history.iter().find(|message| {
                message.id == offered.id && message.role == lk_agents_abi::Role::User
            });
            if !current
                .is_some_and(|current| crate::session::input::same_human_version(current, offered))
            {
                return Err(invalid(
                    "the reserved human response changed; route the current words again",
                ));
            }
            if saved.response_message_id.is_some() {
                return Ok(CallAdmission::AlreadyForwarded { task: task.clone() });
            }
            validate_task(remote, request, task)?;
            let message = crate::session::input::response(state, &reference)?;
            if !crate::session::input::same_human_version(&message, offered) {
                return Err(invalid(
                    "the reserved human response changed; route the current words again",
                ));
            }
            Ok(CallAdmission::Respond { request: reference, message: Box::new(message) })
        }
        Some(Ok(CallRoute::New)) => {
            Err(invalid("an explicit task_id selects existing work and cannot start another task"))
        }
        _ => Err(invalid(
            "the remote call's destination is unclear or routing was unavailable; clarify or retry without guessing a task",
        )),
    }
}

fn validate_task(
    remote: Option<&RemoteAgentState>,
    request: &CallRoutingRequest,
    task: &RemoteTaskRef,
) -> Result<(), StorageError> {
    if request.selected_task.as_ref().is_some_and(|selected| selected != task)
        || !request.candidates.iter().any(|candidate| &candidate.task == task)
    {
        return Err(invalid("the task was not offered for this remote call"));
    }
    let remote = remote.ok_or_else(|| invalid("this remote has no known tasks"))?;
    if crate::session::remote::task(remote, &task.task_id)? != *task {
        return Err(invalid("the selected task's context changed"));
    }
    Ok(())
}

fn task_messages(
    remote: &RemoteAgentState,
    task: &RemoteTaskRef,
) -> Vec<lk_agents_abi::a2a_protocol::Message> {
    let mut messages = Vec::new();
    let mut seen = BTreeSet::new();
    // Initial sends need not yet contain a taskId. Their observed request association keeps
    // the original business intent available when the peer's task omits optional history.
    for event in &remote.events {
        if remote.requests.get(&event.request_id).is_some_and(|request| {
            request.task.as_ref() == Some(task) && request.request_event == event.sequence
        }) && let A2AJournalPayload::Request(A2ARequest::Send(send)) = &event.payload
            && seen.insert(send.message.message_id.clone())
        {
            messages.push(send.message.clone());
        }
    }
    for message in remote.messages.values().filter(|message| {
        message.task_id.as_deref() == Some(task.task_id.as_str())
            && message.context_id.as_deref() == Some(task.context_id.as_str())
    }) {
        if seen.insert(message.message_id.clone()) {
            messages.push(message.clone());
        }
    }
    messages
}

fn invalid(message: &str) -> StorageError {
    StorageError::Invalid(message.into())
}
