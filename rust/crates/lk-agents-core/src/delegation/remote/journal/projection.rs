use super::{RemoteError, conflict, invalid};
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::*;
use lk_agents_abi::session::InputRequestState;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn response(
    state: &mut RemoteAgentState,
    request: &RemoteRequestState,
    response: A2AResponse,
) -> Result<Option<RemoteTaskRef>, RemoteError> {
    use a2a::jsonrpc::methods;
    let event = match (request.method.as_str(), response) {
        (methods::SEND_MESSAGE, A2AResponse::Send(a2a::SendMessageResponse::Task(task))) => {
            a2a::StreamResponse::Task(task)
        }
        (methods::SEND_MESSAGE, A2AResponse::Send(a2a::SendMessageResponse::Message(message))) => {
            a2a::StreamResponse::Message(message)
        }
        (
            methods::SEND_STREAMING_MESSAGE | methods::SUBSCRIBE_TO_TASK,
            A2AResponse::Stream(event),
        ) => event,
        (methods::GET_TASK | methods::CANCEL_TASK, A2AResponse::Task(task)) => {
            a2a::StreamResponse::Task(task)
        }
        (methods::LIST_TASKS, A2AResponse::List(list)) => {
            for observed in list.tasks {
                task(state, observed)?;
            }
            return Ok(None);
        }
        _ => return Err(invalid("the response shape does not match its SDK operation")),
    };
    let reference = match &event {
        a2a::StreamResponse::Message(message) => message.task_id.as_ref().and_then(|id| {
            let context_id = message
                .context_id
                .clone()
                .or_else(|| state.tasks.get(id).map(|task| task.task.context_id.clone()))
                .or_else(|| {
                    request
                        .task
                        .as_ref()
                        .filter(|task| &task.task_id == id)
                        .map(|task| task.context_id.clone())
                })?;
            Some(RemoteTaskRef { task_id: id.clone(), context_id })
        }),
        a2a::StreamResponse::Task(task) => {
            Some(RemoteTaskRef { task_id: task.id.clone(), context_id: task.context_id.clone() })
        }
        a2a::StreamResponse::StatusUpdate(update) => Some(RemoteTaskRef {
            task_id: update.task_id.clone(),
            context_id: update.context_id.clone(),
        }),
        a2a::StreamResponse::ArtifactUpdate(update) => Some(RemoteTaskRef {
            task_id: update.task_id.clone(),
            context_id: update.context_id.clone(),
        }),
    };
    let entry = state
        .events
        .get(request.request_event as usize)
        .ok_or_else(|| invalid("missing request content"))?;
    if let A2AJournalPayload::Request(A2ARequest::Send(body)) = &entry.payload
        && let Some(expected) = &body.message.context_id
    {
        let observed = reference.as_ref().map(|task| &task.context_id).or(match &event {
            a2a::StreamResponse::Message(message) => message.context_id.as_ref(),
            _ => None,
        });
        if observed != Some(expected) {
            return Err(invalid("the response belongs to another conversation context"));
        }
    }
    if let Some(reference) = &reference {
        if reference.task_id.is_empty() || reference.context_id.is_empty() {
            return Err(invalid("an observed task requires nonempty task and context IDs"));
        }
        if let Some(expected) = &request.task
            && expected != reference
        {
            return Err(conflict(&reference.task_id));
        }
        // A direct task lookup still has addressing before its context is known locally.
        let entry = state
            .events
            .get(request.request_event as usize)
            .ok_or_else(|| invalid("missing request content"))?;
        if let A2AJournalPayload::Request(body) = &entry.payload {
            let id = match body {
                A2ARequest::Get(body) => Some(&body.id),
                A2ARequest::Cancel(body) => Some(&body.id),
                A2ARequest::Subscribe(body) => Some(&body.id),
                A2ARequest::Send(body) => body.message.task_id.as_ref(),
                A2ARequest::List(_) => None,
            };
            if id.is_some_and(|id| id != &reference.task_id) {
                return Err(conflict(&reference.task_id));
            }
        }
    }
    match event {
        a2a::StreamResponse::Message(mut observed) => {
            if observed.context_id.is_none() {
                observed.context_id = reference.as_ref().map(|task| task.context_id.clone());
            }
            message(&mut state.messages, observed)?;
        }
        a2a::StreamResponse::Task(observed) => task(state, observed)?,
        a2a::StreamResponse::StatusUpdate(update) => {
            let mut observed = current_task(state, &update.task_id, &update.context_id)?;
            observed.status = update.status;
            task(state, observed)?;
        }
        a2a::StreamResponse::ArtifactUpdate(update) => {
            let mut observed = current_task(state, &update.task_id, &update.context_id)?;
            let artifacts = observed.artifacts.get_or_insert_default();
            if let Some(existing) = artifacts
                .iter_mut()
                .find(|artifact| artifact.artifact_id == update.artifact.artifact_id)
            {
                if update.append == Some(true) {
                    crate::a2a::client::append_artifact(existing, update.artifact);
                } else {
                    *existing = update.artifact;
                }
            } else {
                artifacts.push(update.artifact);
            }
            task(state, observed)?;
        }
    }
    if state.default_context_id.is_none() {
        let context = reference
            .as_ref()
            .map(|task| task.context_id.clone())
            .or_else(|| state.messages.values().find_map(|message| message.context_id.clone()));
        state.default_context_id = context;
    }
    Ok(reference)
}

fn current_task(
    state: &RemoteAgentState,
    id: &str,
    context: &str,
) -> Result<a2a::Task, RemoteError> {
    if let Some(task) = state.tasks.get(id) {
        if task.task.context_id != context {
            return Err(conflict(id));
        }
        return Ok(task.task.clone());
    }
    // The stream may start with an artifact or status delta. Unspecified is the SDK's state
    // for a task whose status has not been observed; never manufacture a completed outcome.
    Ok(a2a::Task {
        id: id.into(),
        context_id: context.into(),
        status: a2a::TaskStatus {
            state: a2a::TaskState::Unspecified,
            message: None,
            timestamp: None,
        },
        artifacts: None,
        history: None,
        metadata: None,
    })
}

fn task(state: &mut RemoteAgentState, observed: a2a::Task) -> Result<(), RemoteError> {
    if observed.id.is_empty() || observed.context_id.is_empty() {
        return Err(invalid("a task needs its task and context IDs"));
    }
    for item in observed.history.iter().flatten().chain(observed.status.message.iter()) {
        let mut item = item.clone();
        enrich(&mut item.context_id, Some(observed.context_id.clone()), &item.message_id)?;
        enrich(&mut item.task_id, Some(observed.id.clone()), &item.message_id)?;
        message(&mut state.messages, item)?;
    }
    let dispatched: BTreeSet<_> = state
        .requests
        .values()
        .filter(|request| request.dispatch_started)
        .filter_map(|request| request.input_request.as_ref())
        .filter(|reference| reference.task.task_id == observed.id)
        .map(|reference| reference.occurrence_id.as_str())
        .collect();
    let saved = state.tasks.entry(observed.id.clone()).or_insert_with(|| RemoteTaskState {
        task: observed.clone(),
        input_requests: BTreeMap::new(),
        cancel_requested: false,
        next_input: 0,
    });
    if saved.task.context_id != observed.context_id {
        return Err(conflict(&observed.id));
    }
    for input in saved.input_requests.values_mut() {
        if input.state == InputRequestState::Responding
            && input.response_message_id.as_ref().is_some_and(|id| {
                observed.history.iter().flatten().any(|message| &message.message_id == id)
            })
        {
            input.state = InputRequestState::Responded;
        }
    }
    let source_id = observed.status.message.as_ref().map(|message| message.message_id.clone());
    if observed.status.state == a2a::TaskState::InputRequired {
        // Older uncertain responses can remain in the map. Only the latest occurrence can
        // keep this question open; a newly admitted response can lead to the same question.
        let latest = saved.next_input.checked_sub(1).and_then(|sequence| {
            saved.input_requests.get(&format!("{}/input_{sequence}", observed.id))
        });
        let same_occurrence = saved.task.status.state == a2a::TaskState::InputRequired
            && saved.task.status.message.as_ref().map(|message| &message.message_id)
                == source_id.as_ref()
            && latest.is_some_and(|input| {
                matches!(input.state, InputRequestState::Open | InputRequestState::Responding)
            });
        if !same_occurrence {
            retire_inputs(&mut saved.input_requests, InputRequestState::Superseded, &dispatched);
            let occurrence_id = format!("{}/input_{}", observed.id, saved.next_input);
            saved.next_input = saved
                .next_input
                .checked_add(1)
                .ok_or_else(|| invalid("input occurrence identity exhausted"))?;
            saved.input_requests.insert(
                occurrence_id,
                InputRequest {
                    state: InputRequestState::Open,
                    source_message_id: source_id,
                    response_message_id: None,
                },
            );
        }
    } else {
        retire_inputs(&mut saved.input_requests, InputRequestState::Closed, &dispatched);
    }
    // A limited getTask response may omit optional history/artifacts. Omission does not erase
    // earlier observations; original response appearances remain unchanged in the journal.
    let mut observed = observed;
    if observed.history.is_none() {
        observed.history = saved.task.history.clone();
    }
    if observed.artifacts.is_none() {
        observed.artifacts = saved.task.artifacts.clone();
    }
    saved.task = observed;
    Ok(())
}

/// A new question or status retires unsent selections. Once dispatch may have started, keep
/// the original response correlation until remote evidence resolves its admission.
fn retire_inputs(
    inputs: &mut BTreeMap<String, InputRequest>,
    disposition: InputRequestState,
    dispatched: &BTreeSet<&str>,
) {
    for (id, input) in inputs {
        if input.state == InputRequestState::Open
            || input.state == InputRequestState::Responding && !dispatched.contains(id.as_str())
        {
            input.state = disposition;
        }
    }
}

pub(super) fn message(
    messages: &mut BTreeMap<String, a2a::Message>,
    observed: a2a::Message,
) -> Result<(), RemoteError> {
    if observed.message_id.is_empty() {
        return Err(invalid("an observed message needs its messageId"));
    }
    let Some(existing) = messages.get_mut(&observed.message_id) else {
        messages.insert(observed.message_id.clone(), observed);
        return Ok(());
    };
    if existing.role != observed.role
        || existing.parts != observed.parts
        || existing.extensions != observed.extensions
        || existing.reference_task_ids != observed.reference_task_ids
    {
        return Err(conflict(&observed.message_id));
    }
    enrich(&mut existing.context_id, observed.context_id, &observed.message_id)?;
    enrich(&mut existing.task_id, observed.task_id, &observed.message_id)?;
    if let Some(metadata) = observed.metadata {
        let current = existing.metadata.get_or_insert_default();
        for (key, value) in metadata {
            if current.get(&key).is_some_and(|known| known != &value) {
                return Err(conflict(&observed.message_id));
            }
            current.insert(key, value);
        }
    }
    Ok(())
}

fn enrich(
    current: &mut Option<String>,
    observed: Option<String>,
    id: &str,
) -> Result<(), RemoteError> {
    if let Some(observed) = observed {
        if current.as_ref().is_some_and(|known| known != &observed) {
            return Err(conflict(id));
        }
        *current = Some(observed);
    }
    Ok(())
}
