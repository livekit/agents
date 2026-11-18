//! Communication observations and dispatch intent are committed by the conversation owner.

use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::*;
use lk_agents_abi::session::InputRequestState;

mod projection;
mod terminal;
pub use terminal::observed;
mod validation;
pub use validation::validate;
#[cfg(test)]
#[path = "journal_tests.rs"]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid remote communication state: {0}")]
pub struct RemoteError(String);

/// Apply one serialized input. Observed response content remains journaled even if its
/// projection is refused; malformed or conflicting content never overwrites accepted state.
pub fn apply(
    remote_id: &str,
    state: &mut RemoteAgentState,
    input: RemoteInput,
) -> Result<Option<InputRequestRef>, RemoteError> {
    match input {
        RemoteInput::Prepare {
            request_id,
            request,
            streaming,
            origin,
            input_request,
            control_scope,
        } => {
            let control_scope = control_scope.map(|scope| *scope);
            if request_id.is_empty() {
                return Err(invalid("request identity is empty"));
            }
            if let Some(existing) = state.requests.get(&request_id) {
                let original = state.events.get(existing.request_event as usize);
                let same_body = original.is_some_and(|entry| {
                    matches!(
                        &entry.payload,A2AJournalPayload::Request(body) if body == &request
                    )
                });
                if same_body
                    && existing.method == request.method(streaming)
                    && !existing.dispatch_started
                    && (existing.input_request.is_none() || response_reserved(state, existing))
                    && (input_request.is_none() || input_request == existing.input_request)
                    && (origin.is_none() || origin == existing.origin)
                    && (control_scope.is_none() || control_scope == existing.control_scope)
                {
                    // Before dispatch, a fresh observer may retry this exact prepared body.
                    // Retain its original content and errors rather than recording it twice.
                    request_mut(state, &request_id)?.ended = false;
                    return Ok(None);
                }
                return Err(invalid("request identity is already prepared or may have been sent"));
            }
            let mut next = state.clone();
            let method = request.method(streaming).to_owned();
            let task = request_task(&next, &request)?;
            if let Some(scope) = &control_scope {
                let A2ARequest::Send(send) = &request else {
                    return Err(invalid("only a verified send can establish control scope"));
                };
                if scope.project_origin.is_empty()
                    || scope.project_issuer.is_empty()
                    || scope.caller_identity.is_empty()
                    || scope.thread_id.is_empty()
                    || send.message.context_id.as_deref() != Some(scope.thread_id.as_str())
                {
                    return Err(invalid("the verified control scope does not address this send"));
                }
            }
            if let A2ARequest::Send(send) = &request {
                projection::message(&mut next.messages, send.message.clone())?;
            }
            if let Some(reference) = &input_request {
                let (selected_task, occurrence) = addressed_input(reference, remote_id)?;
                let A2ARequest::Send(send) = &request else {
                    return Err(invalid("only a send can respond to input"));
                };
                if task.as_ref() != Some(selected_task) {
                    return Err(invalid("the response does not address the selected task"));
                }
                let input = next
                    .tasks
                    .get_mut(&selected_task.task_id)
                    .and_then(|task| task.input_requests.get_mut(occurrence))
                    .ok_or_else(|| invalid("the input occurrence is unknown"))?;
                let held = input.state == InputRequestState::Responding
                    && input.human_response_id.is_some()
                    && input.response_message_id.is_none();
                if input.state != InputRequestState::Open && !held {
                    return Err(invalid("the input occurrence already has a response"));
                }
                input.state = InputRequestState::Responding;
                input.response_message_id = Some(send.message.message_id.clone());
            }
            if let A2ARequest::Cancel(cancel) = &request {
                let observed =
                    next.tasks.get_mut(&cancel.id).ok_or_else(|| invalid("unknown task"))?;
                observed.cancel_requested = true;
            }
            let request_event = append(
                &mut next,
                &request_id,
                A2AJournalKind::RequestRecorded,
                A2AJournalPayload::Request(request),
            );
            next.requests.insert(
                request_id,
                RemoteRequestState {
                    method,
                    request_event,
                    dispatch_started: false,
                    ended: false,
                    task,
                    origin,
                    input_request,
                    control_scope,
                },
            );
            *state = next;
        }
        RemoteInput::DispatchStarted { request_id } => {
            let request =
                state.requests.get(&request_id).ok_or_else(|| invalid("unknown request"))?;
            if request.ended || request.dispatch_started {
                return Err(invalid("this request may already have been sent; reconcile it"));
            }
            if request.input_request.is_some() && !response_reserved(state, request) {
                return Err(invalid("the input occurrence no longer accepts this response"));
            }
            request_mut(state, &request_id)?.dispatch_started = true;
        }
        RemoteInput::Response { request_id, response } => {
            let request = request_mut(state, &request_id)?.clone();
            if !request.dispatch_started || request.ended {
                return Err(invalid("response without an active dispatched request"));
            }
            // Preserve the original successful SDK value before interpreting its contents.
            append(
                state,
                &request_id,
                A2AJournalKind::ResponseReceived,
                A2AJournalPayload::Response(response.clone()),
            );
            let mut next = state.clone();
            // Admission resolves this occurrence before interpreting the peer's next question.
            // Projection errors discard this staged change along with the rest of `next`.
            if let Some(reference) = &request.input_request {
                let (selected_task, occurrence) = addressed_input(reference, remote_id)?;
                let input = next
                    .tasks
                    .get_mut(&selected_task.task_id)
                    .and_then(|task| task.input_requests.get_mut(occurrence))
                    .ok_or_else(|| invalid("the reserved input occurrence is missing"))?;
                input.state = InputRequestState::Responded;
            }
            let input_required = match response.as_ref() {
                A2AResponse::Task(task)
                | A2AResponse::Send(a2a::SendMessageResponse::Task(task))
                | A2AResponse::Stream(a2a::StreamResponse::Task(task)) => {
                    task.status.state == a2a::TaskState::InputRequired
                }
                A2AResponse::Stream(a2a::StreamResponse::StatusUpdate(update)) => {
                    update.status.state == a2a::TaskState::InputRequired
                }
                _ => false,
            };
            let task = projection::response(&mut next, &request, *response)?;
            let input_request = task.as_ref().filter(|_| input_required).and_then(|reference| {
                let task = next.tasks.get(&reference.task_id)?;
                let sequence = task.next_input.checked_sub(1)?;
                let occurrence_id = format!("{}/input_{sequence}", reference.task_id);
                task.input_requests
                    .get(&occurrence_id)
                    .filter(|input| input.state == InputRequestState::Open)?;
                Some(InputRequestRef::RemoteTask {
                    remote_id: remote_id.into(),
                    task: reference.clone(),
                    request_id: occurrence_id,
                })
            });
            if let Some(task) = task {
                next.requests.get_mut(&request_id).expect("request was validated").task =
                    Some(task);
            }
            *state = next;
            return Ok(input_request);
        }
        RemoteInput::Error { request_id, error } => {
            let request = request_mut(state, &request_id)?;
            if request.ended {
                return Err(invalid("the request already ended"));
            }
            request.ended = true;
            append(
                state,
                &request_id,
                A2AJournalKind::RequestError,
                A2AJournalPayload::Error(error),
            );
        }
        RemoteInput::Ended { request_id } => {
            request_mut(state, &request_id)?.ended = true;
        }
    }
    Ok(None)
}

/// Recovery never replays a send merely to discover the task it may have created.
pub fn recovery(state: &RemoteAgentState, request_id: &str) -> Result<RemoteRecovery, RemoteError> {
    let request = state.requests.get(request_id).ok_or_else(|| invalid("unknown request"))?;
    if let Some(target) = &request.task
        && let Some(response) = observed(state, target)?
    {
        return Ok(RemoteRecovery::Observed(Box::new(response)));
    }
    if !request.dispatch_started {
        if let Some(reference) = &request.input_request
            && !response_reserved(state, request)
        {
            let (_, task, _) = reference
                .remote()
                .ok_or_else(|| invalid("a remote operation selected a local input"))?;
            return Ok(RemoteRecovery::GetTask(task.clone()));
        }
        let entry = state
            .events
            .get(request.request_event as usize)
            .ok_or_else(|| invalid("missing request content"))?;
        let A2AJournalPayload::Request(body) = &entry.payload else {
            return Err(invalid("request position does not contain a request"));
        };
        return Ok(RemoteRecovery::Dispatch {
            request_id: request_id.into(),
            request: Box::new(body.clone()),
            streaming: matches!(
                request.method.as_str(),
                lk_agents_abi::a2a_protocol::methods::SEND_STREAMING_MESSAGE
                    | lk_agents_abi::a2a_protocol::methods::SUBSCRIBE_TO_TASK
            ),
            input_request: request.input_request.clone(),
        });
    }
    if let Some(task) = &request.task {
        return Ok(RemoteRecovery::GetTask(task.clone()));
    }
    // A received message-only reply is already proof of the exchange, even without a task.
    for event in state.events.iter().rev().filter(|event| event.request_id == request_id) {
        let A2AJournalPayload::Response(response) = &event.payload else { continue };
        let message = match (request.method.as_str(), response.as_ref()) {
            (
                lk_agents_abi::a2a_protocol::methods::SEND_MESSAGE,
                A2AResponse::Send(lk_agents_abi::a2a_protocol::SendMessageResponse::Message(
                    message,
                )),
            )
            | (
                lk_agents_abi::a2a_protocol::methods::SEND_STREAMING_MESSAGE,
                A2AResponse::Stream(lk_agents_abi::a2a_protocol::StreamResponse::Message(message)),
            ) => message,
            _ => continue,
        };
        // Rejected protocol evidence is still journaled. Only the accepted projection can
        // settle a restored call; merely observing a malformed response proves no outcome.
        if state.messages.get(&message.message_id) == Some(message) {
            return Ok(RemoteRecovery::Observed(Box::new(a2a::SendMessageResponse::Message(
                message.clone(),
            ))));
        }
    }
    Ok(RemoteRecovery::Uncertain)
}

fn response_reserved(state: &RemoteAgentState, request: &RemoteRequestState) -> bool {
    let Some(reference) = &request.input_request else { return false };
    let Some((_, selected_task, occurrence)) = reference.remote() else { return false };
    let Some(A2AJournalEntry {
        payload: A2AJournalPayload::Request(A2ARequest::Send(body)), ..
    }) = state.events.get(request.request_event as usize)
    else {
        return false;
    };
    state
        .tasks
        .get(&selected_task.task_id)
        .filter(|task| task.task.context_id == selected_task.context_id)
        .and_then(|task| task.input_requests.get(occurrence))
        .is_some_and(|input| {
            input.state == InputRequestState::Responding
                && input.response_message_id.as_deref() == Some(body.message.message_id.as_str())
        })
}

pub(super) fn addressed_input<'a>(
    reference: &'a InputRequestRef,
    remote_id: &str,
) -> Result<(&'a RemoteTaskRef, &'a str), RemoteError> {
    let Some((owner, task, occurrence)) = reference.remote() else {
        return Err(invalid("a remote operation cannot select a local input"));
    };
    if owner != remote_id || owner.is_empty() || occurrence.is_empty() {
        return Err(invalid("the input occurrence belongs to another remote owner"));
    }
    Ok((task, occurrence))
}

fn request_task(
    state: &RemoteAgentState,
    request: &A2ARequest,
) -> Result<Option<RemoteTaskRef>, RemoteError> {
    let id = match request {
        A2ARequest::Send(send) => {
            let Some(id) = &send.message.task_id else { return Ok(None) };
            if id.is_empty() {
                return Err(invalid("task ID is empty"));
            }
            let context_id = send
                .message
                .context_id
                .clone()
                .ok_or_else(|| invalid("a task response needs its context"))?;
            if let Some(task) = state.tasks.get(id)
                && task.task.context_id != context_id
            {
                return Err(conflict(id));
            }
            return Ok(Some(RemoteTaskRef { context_id, task_id: id.clone() }));
        }
        A2ARequest::Get(request) => &request.id,
        A2ARequest::Cancel(request) => &request.id,
        A2ARequest::Subscribe(request) => &request.id,
        A2ARequest::List(_) => return Ok(None),
    };
    if id.is_empty() {
        return Err(invalid("task ID is empty"));
    }
    Ok(state.tasks.get(id).map(|task| RemoteTaskRef {
        task_id: id.clone(),
        context_id: task.task.context_id.clone(),
    }))
}

fn append(
    state: &mut RemoteAgentState,
    request_id: &str,
    kind: A2AJournalKind,
    payload: A2AJournalPayload,
) -> u64 {
    let sequence = state.events.last().map_or(0, |event| event.sequence + 1);
    state.events.push(A2AJournalEntry { sequence, request_id: request_id.into(), kind, payload });
    sequence
}

fn request_mut<'a>(
    state: &'a mut RemoteAgentState,
    id: &str,
) -> Result<&'a mut RemoteRequestState, RemoteError> {
    state.requests.get_mut(id).ok_or_else(|| invalid("unknown request"))
}

fn invalid(message: &str) -> RemoteError {
    RemoteError(message.into())
}

fn conflict(id: &str) -> RemoteError {
    invalid(&format!("conflicting remote identity: {id}"))
}
