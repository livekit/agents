//! Bounded retention. Unfinished work keeps everything it references; settled history is
//! removed oldest first, one whole operation and its settled tasks at a time.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::*;
use lk_agents_abi::session::{InputRequestState, ToolCallRef};

/// Settled journal entries kept for inspection beyond what unfinished work pins.
pub const RETAINED_SETTLED_EVENTS: usize = 256;

/// Pinned: a nonterminal task or one with an unanswered input occurrence, a request that has
/// not ended, may still be recovered (unresolved originating call, unprovable outcome), or
/// addresses a pinned task, and each pinned task's accepted terminal evidence. A removed
/// request takes the settled tasks it addressed or completed, and every request addressing
/// them, so recovery never finds half an operation. The newest entry always stays: journal
/// sequences continue from it.
pub fn compact(
    state: &mut RemoteAgentState,
    retained: usize,
    unresolved: impl Fn(&ToolCallRef) -> bool,
) {
    let Some(newest) = state.events.last().map(|event| event.request_id.clone()) else { return };
    let mut pinned_tasks: BTreeSet<&str> = state
        .tasks
        .iter()
        .filter(|(_, task)| {
            !task.task.status.state.is_terminal()
                || task.input_requests.values().any(|input| {
                    matches!(input.state, InputRequestState::Open | InputRequestState::Responding)
                })
        })
        .map(|(id, _)| id.as_str())
        .collect();
    let live: BTreeSet<&str> = state
        .requests
        .iter()
        .filter(|(id, request)| {
            !request.ended
                || request.origin.as_ref().is_some_and(&unresolved)
                || request.dispatch_started
                    && !matches!(
                        super::recovery(state, id),
                        Ok(RemoteRecovery::Observed(_) | RemoteRecovery::GetTask(_))
                    )
        })
        .map(|(id, _)| id.as_str())
        .collect();
    pinned_tasks.extend(
        live.iter()
            .filter_map(|id| state.requests[*id].task.as_ref())
            .map(|task| task.task_id.as_str()),
    );
    let terminal_holders: BTreeMap<u64, &str> = state
        .tasks
        .iter()
        .filter_map(|(id, task)| Some((task.terminal_event?, id.as_str())))
        .collect();
    let mut pinned: BTreeSet<&str> = live;
    for (sequence, task) in &terminal_holders {
        if pinned_tasks.contains(task)
            && let Some(event) = state.event(*sequence)
        {
            pinned.insert(event.request_id.as_str());
        }
    }
    let mut addressing: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (id, request) in &state.requests {
        if let Some(task) = &request.task {
            if pinned_tasks.contains(task.task_id.as_str()) {
                pinned.insert(id.as_str());
            }
            addressing.entry(task.task_id.as_str()).or_default().push(id.as_str());
        }
    }
    let mut sequences: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    for event in &state.events {
        sequences.entry(event.request_id.as_str()).or_default().push(event.sequence);
    }
    let mut settled =
        state.events.iter().filter(|event| !pinned.contains(event.request_id.as_str())).count();
    if settled <= retained {
        return;
    }
    let mut oldest: Vec<(&str, u64)> = state
        .requests
        .iter()
        .filter(|(id, _)| !pinned.contains(id.as_str()))
        .map(|(id, request)| (id.as_str(), request.request_event))
        .collect();
    oldest.sort_by_key(|(_, sequence)| *sequence);

    let mut dropped_requests: BTreeSet<&str> = BTreeSet::new();
    let mut dropped_tasks: BTreeSet<&str> = BTreeSet::new();
    for (start, _) in oldest {
        if settled <= retained {
            break;
        }
        if dropped_requests.contains(start) {
            continue;
        }
        let mut requests = BTreeSet::from([start]);
        let mut tasks = BTreeSet::new();
        let mut queue = vec![start];
        while let Some(id) = queue.pop() {
            let addressed = state.requests[id].task.as_ref().map(|task| task.task_id.as_str());
            let completed = sequences
                .get(id)
                .into_iter()
                .flatten()
                .filter_map(|sequence| terminal_holders.get(sequence).copied());
            for task in addressed.into_iter().chain(completed) {
                if state.tasks.contains_key(task) && tasks.insert(task) {
                    for request in addressing.get(task).into_iter().flatten().copied() {
                        if requests.insert(request) {
                            queue.push(request);
                        }
                    }
                }
            }
        }
        if requests.iter().any(|id| pinned.contains(id) || *id == newest)
            || tasks.iter().any(|task| pinned_tasks.contains(task))
        {
            continue;
        }
        settled -= requests.iter().map(|id| sequences.get(id).map_or(0, Vec::len)).sum::<usize>();
        dropped_requests.extend(requests);
        dropped_tasks.extend(tasks);
    }
    if dropped_requests.is_empty() {
        return;
    }
    let dropped_requests: BTreeSet<String> =
        dropped_requests.into_iter().map(str::to_owned).collect();
    let dropped_tasks: BTreeSet<String> = dropped_tasks.into_iter().map(str::to_owned).collect();
    state.events.retain(|event| !dropped_requests.contains(&event.request_id));
    state.requests.retain(|id, _| !dropped_requests.contains(id));
    state.tasks.retain(|id, _| !dropped_tasks.contains(id));
    let referenced = referenced_messages(state);
    state.messages.retain(|id, _| referenced.contains(id.as_str()));
}

/// Messages a retained observation still refers to: request and response content, retained
/// tasks' history, status and input occurrences, and messages addressed to a retained task.
fn referenced_messages(state: &RemoteAgentState) -> BTreeSet<String> {
    fn task(task: &a2a::Task, ids: &mut BTreeSet<String>) {
        for message in task.history.iter().flatten().chain(task.status.message.iter()) {
            ids.insert(message.message_id.clone());
        }
    }
    let mut ids = BTreeSet::new();
    for event in &state.events {
        match &event.payload {
            A2AJournalPayload::Request(A2ARequest::Send(send)) => {
                ids.insert(send.message.message_id.clone());
            }
            A2AJournalPayload::Response(response) => match response.as_ref() {
                A2AResponse::Send(a2a::SendMessageResponse::Message(message))
                | A2AResponse::Stream(a2a::StreamResponse::Message(message)) => {
                    ids.insert(message.message_id.clone());
                }
                A2AResponse::Send(a2a::SendMessageResponse::Task(observed))
                | A2AResponse::Stream(a2a::StreamResponse::Task(observed))
                | A2AResponse::Task(observed) => task(observed, &mut ids),
                A2AResponse::Stream(a2a::StreamResponse::StatusUpdate(update)) => {
                    ids.extend(update.status.message.iter().map(|m| m.message_id.clone()));
                }
                A2AResponse::List(list) => {
                    for observed in &list.tasks {
                        task(observed, &mut ids);
                    }
                }
                A2AResponse::Stream(a2a::StreamResponse::ArtifactUpdate(_)) => {}
            },
            _ => {}
        }
    }
    for saved in state.tasks.values() {
        task(&saved.task, &mut ids);
        for input in saved.input_requests.values() {
            ids.extend(input.source_message_id.iter().cloned());
            ids.extend(input.response_message_id.iter().cloned());
        }
    }
    for message in state.messages.values() {
        if message.task_id.as_ref().is_some_and(|id| state.tasks.contains_key(id)) {
            ids.insert(message.message_id.clone());
        }
    }
    ids
}
