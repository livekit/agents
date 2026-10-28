//! A run's records and decisions: the task states and their transitions ([`merge_write`]), the
//! run's outcome and usage, and what a run commits ([`run_commit`]): the changed and deleted
//! items, the journal entries, the task rows, the snapshot that changed and `run_ended`.

use std::collections::BTreeMap;

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::StorageError;
use crate::db::{Commit, EventWrite, TaskWrite};
use crate::schema::{Ask, EventRow, RunEnded, RunEvent, RunStarted, TaskInternal};
pub use crate::schema::{ItemRow, LaneRow, TaskRow};
use livekit_protocol::agent::chat_context::ChatItem;

/// a waiter that has not acquired after 60 s fails `lease-timeout`.
pub use lk_agents_abi::a2a::options::LEASE_WAIT_TIMEOUT;

pub use lk_agents_abi::storage::{RunUsage, TaskState};

/// Everything a run commits, as staged by its owner.
pub struct RunWrites<'a> {
    pub run_id: &'a str,
    pub task_id: &'a str,
    pub channel: Option<&'a str>,
    /// Under a lease; `false` is the `concurrent` policy's append-only write.
    pub exclusive: bool,
    pub history: &'a [ItemRow],
    /// What was loaded, by id, in order.
    pub loaded: &'a [(String, ChatItem)],
    pub state: Option<&'a Value>,
    pub loaded_state: Option<&'a Value>,
    /// The journal entries the run recorded, in order.
    pub events: &'a [EventWrite],
    pub tasks: &'a [TaskWrite],
    pub usage: RunUsage,
    pub outcome: RunOutcome,
    pub reason: Option<&'a str>,
    /// When the commit's own entries are stamped, Unix milliseconds.
    pub now_ms: i64,
    /// The last inbox row a drain consumed.
    pub drained_id: Option<i64>,
    /// The task the turn answers, read back with the commit.
    pub answer_task: Option<String>,
}

/// The run's one write: the changed and deleted items, their journal entries, the task rows,
/// usage, the snapshot when it changed, and `run_ended`. An append-only (`concurrent`) run writes
/// no state and may only add items: editing or deleting a loaded one is refused.
pub fn run_commit(w: RunWrites<'_>) -> Result<Commit, StorageError> {
    let (items, delete_items) = item_changes(w.history, w.loaded);
    if !w.exclusive {
        let edited = items.iter().find(|i| w.loaded.iter().any(|(id, _)| *id == i.id));
        if let Some(id) = edited.map(|i| &i.id).or(delete_items.first()) {
            return Err(StorageError::Invalid(format!(
                "a concurrent run only appends to the history; it edited or deleted item {id}"
            )));
        }
    }
    let mut events = Vec::new();
    if !w.exclusive {
        let started = RunStarted {
            task_id: Some(w.task_id.to_owned()),
            stream_id: Some(w.task_id.to_owned()),
            channel: w.channel.map(str::to_owned),
            exclusive: Some(false),
        };
        events.push(EventWrite::run_started(w.now_ms, &started));
    }
    events.extend(items.iter().filter(|i| !w.loaded.iter().any(|(id, _)| *id == i.id)).map(|i| {
        EventWrite {
            ts_ms: w.now_ms,
            r#type: "conversation_item_added".into(),
            payload: json!({"item": &i.payload}),
        }
    }));
    events.extend(w.events.iter().cloned());
    if w.usage.requests != 0 || w.usage.tool_calls != 0 {
        let payload = serde_json::to_value(w.usage).unwrap_or(Value::Null);
        events.push(EventWrite { ts_ms: w.now_ms, r#type: "usage".into(), payload });
    }
    Ok(Commit {
        run_id: w.run_id.to_owned(),
        delete_items,
        items,
        events,
        tasks: w.tasks.to_vec(),
        state: w.state.filter(|s| w.exclusive && Some(*s) != w.loaded_state).cloned(),
        loaded_state: w.loaded_state.cloned(),
        context_scopes: Vec::new(),
        scope_writes: Vec::new(),
        drained_id: w.drained_id.filter(|_| w.exclusive),
        run_ended: EventWrite::run_ended(
            w.now_ms,
            &RunEnded { outcome: w.outcome, reason: w.reason.map(str::to_owned) },
        ),
        read_task: w.answer_task,
    })
}

/// The stored items that differ from what was `loaded`, and the loaded ids no longer in the
/// history, in loaded order.
pub fn item_changes(
    history: &[ItemRow],
    loaded: &[(String, ChatItem)],
) -> (Vec<ItemRow>, Vec<String>) {
    let by_id: BTreeMap<&str, &ChatItem> = loaded.iter().map(|(id, i)| (id.as_str(), i)).collect();
    let stored: Vec<&ItemRow> = history.iter().filter(|i| i.stored()).collect();
    let changed = stored
        .iter()
        .filter(|i| by_id.get(i.id.as_str()) != Some(&&i.payload))
        .map(|i| (*i).clone())
        .collect();
    let kept: std::collections::BTreeSet<&str> = stored.iter().map(|i| i.id.as_str()).collect();
    let deleted =
        loaded.iter().filter(|(id, _)| !kept.contains(id.as_str())).map(|(id, _)| id.clone());
    (changed, deleted.collect())
}

pub use lk_agents_abi::a2a::metadata::{
    CHANNEL_KEY, EXTENSION_URI, INTERRUPTED_KEY, INTERRUPTION_REASON_KEY, KIND_KEY,
    MERGED_INTO_KEY, REASON_KEY,
};

/// The `failed` and `rejected` reasons.
pub const REASON_WORKER_LOST: &str = "worker-lost";
pub const REASON_LEASE_TIMEOUT: &str = "lease-timeout";
pub const REASON_REJECTED: &str = "rejected";
pub const REASON_THREAD_NOT_FOUND: &str = "thread-not-found";
pub const REASON_STORAGE_UNAVAILABLE: &str = "storage-unavailable";
pub const REASON_TIMEOUT: &str = "timeout";
pub const REASON_UNKNOWN_CLASS: &str = "unknown-class";
pub const REASON_INVALID_ARGUMENTS: &str = "invalid-arguments";

/// The channel of the append-only run that writes a task's `submitted` row before the lease.
pub const SUBMIT_CHANNEL: &str = "submit";

/// What `run_ended` records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Host)]
#[serde(rename_all = "kebab-case")]
#[python(literal, module = "storage_run")]
pub enum RunOutcome {
    #[host(rename = "completed")]
    Completed,
    #[host(rename = "input-required")]
    InputRequired,
    #[host(rename = "failed")]
    Failed,
    #[host(rename = "canceled")]
    Canceled,
    #[host(rename = "rejected")]
    Rejected,
}

/// One run as the journal tells it: `run_started` and, when it ended, `run_ended`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run")]
pub struct RunSummary {
    pub run_id: String,
    pub lane: String,
    pub task_id: Option<String>,
    pub channel: Option<String>,
    pub started_at_ms: i64,
    pub ended_at_ms: Option<i64>,
    pub outcome: Option<RunOutcome>,
    pub reason: Option<String>,
}

impl RunSummary {
    /// A `run_started` without a `run_ended`: the worker died mid-run.
    pub fn interrupted(&self) -> bool {
        self.ended_at_ms.is_none()
    }

    /// In the order the runs started; a `run_ended` whose run was not started is ignored, and a
    /// second `run_started` of a run (a holder re-entering its lease) replaces the first in place.
    pub fn from_events(events: &[EventRow]) -> Vec<RunSummary> {
        let mut runs: Vec<RunSummary> = Vec::new();
        for ev in events {
            match ev.run_event() {
                Some(RunEvent::Started(started)) => {
                    let summary = RunSummary {
                        run_id: ev.run_id.clone(),
                        lane: ev.lane.clone(),
                        task_id: started.task_id,
                        channel: started.channel,
                        started_at_ms: ev.ts_ms,
                        ended_at_ms: None,
                        outcome: None,
                        reason: None,
                    };
                    match runs.iter_mut().find(|r| r.run_id == ev.run_id) {
                        Some(existing) => *existing = summary,
                        None => runs.push(summary),
                    }
                }
                Some(RunEvent::Ended(ended)) => {
                    if let Some(run) = runs.iter_mut().find(|r| r.run_id == ev.run_id) {
                        run.ended_at_ms = Some(ev.ts_ms);
                        run.outcome = Some(ended.outcome);
                        run.reason = ended.reason;
                    }
                }
                None => {}
            }
        }
        runs
    }
}

/// How many rows a lookup by id or `messageId` scans.
pub const LOOKUP_LIMIT: i64 = 500;
/// How many `input-required` rows `pending_ask` scans.
pub const PENDING_ASK_LIMIT: i64 = 50;

/// What a write changes; everything left `None` keeps the existing row's value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskPatch {
    pub message_id: Option<String>,
    /// An optional application task label, the `name` column.
    pub name: Option<String>,
    /// Merged over the row's `data.metadata`; an empty map changes nothing.
    pub metadata: Option<Map<String, Value>>,
    pub ask: Option<Ask>,
    pub streams: Option<Vec<lk_agents_abi::storage::StreamInfo>>,
    pub status_message: Option<a2a::Message>,
    pub history: Option<Vec<a2a::Message>>,
    pub artifacts: Option<Vec<a2a::Artifact>>,
}

/// The pending transition a task waits on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WaitOptions {
    /// For a conversation tool's `InputRequired`, the pending call's tool.
    pub tool: Option<String>,
    /// For a conversation tool's `InputRequired`, the pending call.
    pub call_id: Option<String>,
    /// The tool the answer is handed to, when the `InputRequired` named one.
    pub answer_tool: Option<String>,
    /// The tool's parameter schema, added to the status message as a data part.
    pub schema: Option<Value>,
}

/// `input-required`'s patch: the question is the status message (`message_id`), `ask` the
/// pending transition, the tool's schema a data part when given.
pub fn wait_patch(
    task_id: &str,
    question: &str,
    opts: WaitOptions,
    message_id: String,
) -> TaskPatch {
    let ask = Ask {
        question: question.to_owned(),
        tool: opts.tool,
        call_id: opts.call_id,
        answer_tool: opts.answer_tool,
    };
    let mut parts = vec![a2a::Part::text(question)];
    if let Some(schema) = opts.schema {
        parts.push(a2a::Part::data(schema));
    }
    TaskPatch {
        ask: Some(ask),
        status_message: Some(agent_message(parts, task_id, message_id)),
        ..TaskPatch::default()
    }
}

/// A task write: `patch` over `base`, `data` and `internal` rebuilt from both. A terminal base is
/// immutable.
pub fn merge_write(
    task_id: &str,
    base: Option<&TaskWrite>,
    status: TaskState,
    patch: TaskPatch,
) -> Result<TaskWrite, StorageError> {
    if let Some(b) = base
        && b.status.is_terminal()
    {
        return Err(StorageError::Invalid(format!(
            "task {task_id} is {}: terminal rows are immutable",
            b.status.as_str()
        )));
    }
    let mut data = base.map(|b| b.data.clone()).unwrap_or_default();
    if let Some(metadata) = patch.metadata {
        data.metadata.extend(metadata);
    }
    if let Some(history) = patch.history {
        data.history = history;
    }
    if let Some(artifacts) = patch.artifacts {
        data.artifacts = Some(artifacts);
    }
    if let Some(message) = patch.status_message {
        data.status_message = Some(message);
    }

    let ask = if patch.ask.is_none() && (status.is_terminal() || status == TaskState::Working) {
        // a terminal or running row holds no pending question
        None
    } else {
        patch.ask.or_else(|| {
            base.and_then(|b| b.internal.as_ref().and_then(|internal| internal.ask.clone()))
        })
    };

    Ok(TaskWrite {
        task_id: task_id.to_owned(),
        status,
        message_id: patch.message_id.or_else(|| base.and_then(|b| b.message_id.clone())),
        name: patch.name.or_else(|| base.and_then(|b| b.name.clone())),
        data,
        internal: {
            let mut streams = base
                .and_then(|b| b.internal.as_ref())
                .map(|i| i.streams.clone())
                .unwrap_or_default();
            for stream in patch.streams.unwrap_or_default() {
                if !streams.iter().any(|existing| existing.stream_id == stream.stream_id) {
                    streams.push(stream);
                }
            }
            (ask.is_some() || !streams.is_empty()).then_some(TaskInternal { ask, streams })
        },
    })
}

/// An agent message `message_id` of `task_id`.
pub fn agent_message(parts: Vec<a2a::Part>, task_id: &str, message_id: String) -> a2a::Message {
    a2a::Message {
        message_id,
        task_id: Some(task_id.to_owned()),
        ..a2a::Message::new(a2a::Role::Agent, parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use livekit_protocol::agent::chat_context::chat_item;
    use serde_json::json;

    #[test]
    fn item_changes_skip_instructions_and_unchanged_items_and_list_deletions() {
        use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
        use livekit_protocol::agent::{AgentConfigUpdate, ChatMessage, ChatRole, FunctionCall};
        let msg = |id: &str, role: ChatRole, text: &str| {
            ItemRow::new(chat_item::Item::Message(ChatMessage {
                id: id.into(),
                role: role as i32,
                content: vec![ChatContent { payload: Some(Payload::Text(text.into())) }],
                ..Default::default()
            }))
        };
        let call = ItemRow::new(chat_item::Item::FunctionCall(FunctionCall {
            id: "i3".into(),
            name: "f".into(),
            ..Default::default()
        }));
        let loaded: Vec<(String, ChatItem)> =
            [msg("i1", ChatRole::User, "hi"), msg("i2", ChatRole::Assistant, "hello"), call]
                .into_iter()
                .map(|i| (i.id, i.payload))
                .collect();
        let mut history = vec![
            msg("sys", ChatRole::System, "be nice"),
            msg("dev", ChatRole::Developer, "x"),
            ItemRow::new(chat_item::Item::AgentConfigUpdate(AgentConfigUpdate {
                id: "cfg".into(),
                ..Default::default()
            })),
            msg("i1", ChatRole::User, "hi"),
            // edited
            msg("i2", ChatRole::Assistant, "hey"),
            // i3 removed; i4 new
            ItemRow { message_id: Some("m4".into()), ..msg("i4", ChatRole::User, "bye") },
        ];
        let (changed, deleted) = item_changes(&history, &loaded);
        assert_eq!(
            changed.iter().map(|w| (w.id.as_str(), w.message_id.as_deref())).collect::<Vec<_>>(),
            vec![("i2", None), ("i4", Some("m4"))]
        );
        assert_eq!(deleted, vec!["i3".to_owned()]);
        history.clear();
        let (changed, deleted) = item_changes(&history, &loaded);
        assert!(changed.is_empty());
        assert_eq!(deleted, vec!["i1", "i2", "i3"]);
    }

    #[test]
    fn run_summaries_follow_run_started_and_run_ended() {
        let ev = |id: i64, run_id: &str, ty: &str, payload: Value| EventRow {
            id,
            ts_ms: id * 10,
            lane: "front-desk".into(),
            run_id: run_id.into(),
            r#type: ty.into(),
            payload,
        };
        let events = vec![
            ev(1, "r1", "run_started", json!({"task_id": "AST_1", "channel": "a2a"})),
            ev(2, "r1", "run_ended", json!({"outcome": "completed", "reason": null})),
            ev(3, "r2", "run_started", json!({"task_id": "AST_2", "channel": null})),
            ev(4, "r9", "run_ended", json!({"outcome": "failed"})),
        ];
        let runs = RunSummary::from_events(&events);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].outcome, Some(RunOutcome::Completed));
        assert_eq!(runs[0].ended_at_ms, Some(20));
        assert!(!runs[0].interrupted());
        assert_eq!(runs[1].channel, None);
        assert!(runs[1].interrupted());
    }

    fn message(id: &str) -> a2a::Message {
        a2a::Message { message_id: id.into(), ..a2a::Message::new(a2a::Role::User, vec![]) }
    }

    #[test]
    fn a_later_write_keeps_the_row_and_clears_the_ask_when_terminal_or_working() {
        let base = merge_write(
            "AST_1",
            None,
            TaskState::InputRequired,
            TaskPatch {
                message_id: Some("m1".into()),
                ask: Some(Ask { question: "when?".into(), ..Ask::default() }),
                history: Some(vec![message("m1")]),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        let ask = |w: &TaskWrite| w.internal.as_ref().and_then(|internal| internal.ask.clone());
        assert_eq!(ask(&base).map(|a| a.question), Some("when?".into()));
        let msg = merge_write(
            "AST_1",
            Some(&base),
            TaskState::InputRequired,
            TaskPatch { history: Some(vec![message("m2"), message("m3")]), ..TaskPatch::default() },
        )
        .unwrap();
        assert_eq!(ask(&msg), ask(&base), "input-required keeps the pending transition");
        assert_eq!(msg.message_id.as_deref(), Some("m1"));
        let done = merge_write(
            "AST_1",
            Some(&base),
            TaskState::Completed,
            TaskPatch { artifacts: Some(vec![]), ..TaskPatch::default() },
        )
        .unwrap();
        assert_eq!(ask(&done), None);
        assert_eq!(done.data.artifacts, Some(vec![]));
        assert_eq!(done.data.history, vec![message("m1")]);
        let working =
            merge_write("AST_1", Some(&base), TaskState::Working, TaskPatch::default()).unwrap();
        assert_eq!(ask(&working), None);
        let again = merge_write("AST_1", Some(&done), TaskState::Working, TaskPatch::default());
        assert!(matches!(again, Err(StorageError::Invalid(m)) if m.contains("is completed")));
    }

    #[test]
    fn metadata_merges_over_the_stored_metadata() {
        let base = merge_write(
            "AST_1",
            None,
            TaskState::Submitted,
            TaskPatch {
                name: Some("T".into()),
                metadata: Some(json!({"a": 1, "b": 1}).as_object().cloned().unwrap()),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        let next = merge_write(
            "AST_1",
            Some(&base),
            TaskState::Canceled,
            TaskPatch {
                metadata: Some(json!({"b": 2, "c": 3}).as_object().cloned().unwrap()),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&next.data.metadata).unwrap(),
            json!({"a": 1, "b": 2, "c": 3})
        );
        assert_eq!(next.name.as_deref(), Some("T"));
    }
}
