//! The lane's task rows as one run writes them (plan 5.5): `Tasks`, the port of `storage/run.py`'s
//! `Tasks`. A write carries the whole row, so a change to an existing row loads it first; terminal
//! rows are immutable and a write to one is `StorageError::Invalid`. Every write lands with the
//! run's commit, as a `TaskWrite` of the `CommitRequest`.
//!
//! The JSON the rows hold is what the Python door reads back, so the shapes here are the Python's
//! byte for byte: the `data` column is the A2A `Task` fields, `metadata` merged in; a status message is `{messageId, role: "agent", parts, taskId}`; an artifact is
//! `{artifactId, parts: [part]}` with a text part for a string result and a data part otherwise.

use std::collections::HashMap;

use serde_json::{Map, Value};

use super::records::{REASON_KEY, Ref, TaskRecord, TaskState};
use super::shortuuid;
use crate::StorageError;
use crate::schema::{TaskInternal, TaskRow};
use crate::statements::{TaskWrite, ThreadDb};

/// How many rows a lookup by id or `messageId` scans (`Tasks._load`, `Tasks.by_message_id`).
pub(crate) const LOOKUP_LIMIT: i64 = 500;
/// How many `input-required` rows `pending_ask` scans.
const PENDING_ASK_LIMIT: i64 = 50;

/// What a write changes; everything left `None` keeps the existing row's value (`Tasks.write`'s
/// keyword arguments).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskPatch {
    pub message_id: Option<String>,
    /// The `AgentTask` class name, the `name` column.
    pub name: Option<String>,
    /// Its constructor arguments, `internal.args`.
    pub args: Option<Value>,
    /// Merged over the row's `data.metadata`; an empty map changes nothing.
    pub metadata: Option<Map<String, Value>>,
    pub deadline: Option<i64>,
    pub ask: Option<Value>,
    pub status_message: Option<Value>,
    pub history: Option<Vec<Value>>,
    pub artifacts: Option<Vec<Value>>,
    pub parent: Option<String>,
}

/// `Tasks.submit`'s keyword arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubmitOptions {
    /// The task id; minted as `AST_<shortuuid>` when absent.
    pub task_id: Option<String>,
    pub deadline: Option<i64>,
    pub parent: Option<String>,
}

/// `Tasks.wait`'s keyword arguments: the pending transition.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WaitOptions {
    /// For a conversation tool's `InputRequired`, the pending call's tool.
    pub tool: Option<String>,
    /// For a conversation tool's `InputRequired`, the pending call.
    pub call_id: Option<String>,
    /// The tool the answer is handed to, when the question carries a reference.
    pub answer_tool: Option<String>,
    /// What the tool that takes the answer continues with it, such as the expert's task.
    pub answer_ref: Option<String>,
    pub deadline: Option<i64>,
    /// The tool's parameter schema, added to the status message as a data part.
    pub schema: Option<Value>,
}

/// The lane's task rows as this run writes them (plan 5.5).
pub struct Tasks {
    db: ThreadDb,
    lane: String,
    /// In write order: `Tasks.writes` is the values of a dict keyed by task id.
    writes: Vec<TaskWrite>,
    rows: HashMap<String, TaskRecord>,
    /// The lane's newest rows as the acquire loaded them, newest first: what the run's lookups
    /// read instead of the database. Complete when shorter than [`LOOKUP_LIMIT`].
    loaded: Option<Vec<TaskRecord>>,
}

/// What a write starts from: a pending write of this run, or a row read from the lane.
enum Base<'a> {
    Write(&'a TaskWrite),
    Row(&'a TaskRecord),
}

impl Tasks {
    pub fn new(db: ThreadDb, lane: &str) -> Self {
        Self { db, lane: lane.to_owned(), writes: Vec::new(), rows: HashMap::new(), loaded: None }
    }

    /// Over the rows an acquire loaded, newest first.
    pub fn loaded(db: ThreadDb, lane: &str, rows: Vec<TaskRow>) -> Self {
        let loaded: Vec<TaskRecord> = rows.into_iter().map(TaskRecord::from_row).collect();
        let rows = loaded.iter().map(|r| (r.id.clone(), r.clone())).collect();
        Self { db, lane: lane.to_owned(), writes: Vec::new(), rows, loaded: Some(loaded) }
    }

    /// The loaded rows, when they are the whole lane.
    fn snapshot(&self) -> Option<&[TaskRecord]> {
        self.loaded.as_deref().filter(|rows| rows.len() < LOOKUP_LIMIT as usize)
    }

    /// The lane's newest rows, `limit` of them, as of the acquire.
    pub async fn recent(&self, limit: i64) -> Result<Vec<TaskRecord>, StorageError> {
        if let Some(rows) = self.loaded.as_deref() {
            return Ok(rows.iter().take(limit as usize).cloned().collect());
        }
        let rows = self.db.tasks(&self.lane, None, limit).await?;
        Ok(rows.into_iter().map(TaskRecord::from_row).collect())
    }

    /// The `working` rows, oldest first: a dead holder's, as of the acquire.
    pub async fn working(&self) -> Result<Vec<TaskRecord>, StorageError> {
        if let Some(rows) = self.snapshot() {
            let working = TaskState::Working.as_str();
            return Ok(rows.iter().rev().filter(|r| r.status == working).cloned().collect());
        }
        let rows = self.db.orphans(&self.lane).await?;
        Ok(rows.into_iter().map(TaskRecord::from_row).collect())
    }

    pub fn lane(&self) -> &str {
        &self.lane
    }

    /// The pending writes, in the order they were first made.
    pub fn writes(&self) -> &[TaskWrite] {
        &self.writes
    }

    /// `Tasks._load`: a pending write first, else the row, reading the lane's rows once an id is
    /// not cached.
    async fn load(&mut self, task_id: &str) -> Result<Option<Base<'_>>, StorageError> {
        if !self.writes.iter().any(|w| w.task_id == task_id) && !self.rows.contains_key(task_id) {
            for row in self.db.tasks(&self.lane, None, LOOKUP_LIMIT).await? {
                self.rows.entry(row.task_id.clone()).or_insert_with(|| TaskRecord::from_row(row));
            }
        }
        if let Some(w) = self.writes.iter().find(|w| w.task_id == task_id) {
            return Ok(Some(Base::Write(w)));
        }
        Ok(self.rows.get(task_id).map(Base::Row))
    }

    pub async fn get(&mut self, task_id: &str) -> Result<Option<TaskRecord>, StorageError> {
        let lane = self.lane.clone();
        Ok(match self.load(task_id).await? {
            Some(Base::Write(w)) => Some(record_from_write(w, &lane)),
            Some(Base::Row(r)) => Some(r.clone()),
            None => None,
        })
    }

    /// The task of an A2A `messageId`: a pending write first, then the lane's rows.
    pub async fn by_message_id(
        &self,
        message_id: &str,
    ) -> Result<Option<TaskRecord>, StorageError> {
        if let Some(w) = self.writes.iter().find(|w| w.message_id.as_deref() == Some(message_id)) {
            return Ok(Some(record_from_write(w, &self.lane)));
        }
        if let Some(rows) = self.snapshot() {
            return Ok(rows.iter().find(|r| r.message_id.as_deref() == Some(message_id)).cloned());
        }
        Ok(self
            .db
            .tasks(&self.lane, None, LOOKUP_LIMIT)
            .await?
            .into_iter()
            .find(|r| r.message_id.as_deref() == Some(message_id))
            .map(TaskRecord::from_row))
    }

    /// The newest `input-required` row holding a conversation tool's pending call.
    pub async fn pending_ask(&self) -> Result<Option<TaskRecord>, StorageError> {
        let input_required = TaskState::InputRequired.as_str();
        if let Some(rows) = self.snapshot() {
            return Ok(rows
                .iter()
                .find(|r| r.status == input_required && r.ask().is_some_and(truthy))
                .cloned());
        }
        Ok(self
            .db
            .tasks(&self.lane, Some(input_required), PENDING_ASK_LIMIT)
            .await?
            .into_iter()
            .map(TaskRecord::from_row)
            .find(|r| r.ask().is_some_and(truthy)))
    }

    /// Insert or update one row; what the patch leaves `None` keeps the existing row's value.
    pub async fn write(
        &mut self,
        task_id: &str,
        status: TaskState,
        patch: TaskPatch,
    ) -> Result<&TaskWrite, StorageError> {
        let base = match self.load(task_id).await? {
            Some(Base::Write(w)) => Some(w.clone()),
            Some(Base::Row(r)) => Some(write_from_record(r)),
            None => None,
        };
        let write = merge_write(task_id, base.as_ref(), status, patch)?;
        match self.writes.iter().position(|w| w.task_id == task_id) {
            Some(i) => {
                self.writes[i] = write;
                Ok(&self.writes[i])
            }
            None => {
                self.writes.push(write);
                Ok(self.writes.last().expect("just pushed"))
            }
        }
    }

    /// A `submitted` row for `ref`; the task id, minted when `opts.task_id` is absent.
    pub async fn submit(&mut self, r: &Ref, opts: SubmitOptions) -> Result<String, StorageError> {
        let task_id = opts.task_id.unwrap_or_else(|| shortuuid("AST_"));
        self.write(
            &task_id,
            TaskState::Submitted,
            TaskPatch {
                name: Some(r.name.clone()),
                args: r.args.clone(),
                deadline: opts.deadline,
                parent: opts.parent,
                ..TaskPatch::default()
            },
        )
        .await?;
        Ok(task_id)
    }

    /// `input-required`: the question is the status message, `ask` the pending transition.
    pub async fn wait(
        &mut self,
        task_id: &str,
        question: &str,
        opts: WaitOptions,
    ) -> Result<(), StorageError> {
        let mut ask = Map::new();
        ask.insert("question".into(), Value::String(question.to_owned()));
        if let Some(tool) = opts.tool {
            ask.insert("tool".into(), Value::String(tool));
        }
        if let Some(call_id) = opts.call_id {
            ask.insert("call_id".into(), Value::String(call_id));
        }
        if let Some(answer_tool) = opts.answer_tool {
            ask.insert("answer_tool".into(), Value::String(answer_tool));
        }
        if let Some(answer_ref) = opts.answer_ref {
            ask.insert("answer_ref".into(), Value::String(answer_ref));
        }
        let mut parts = vec![text_part(question)];
        if let Some(schema) = opts.schema {
            parts.push(serde_json::json!({"kind": "data", "data": schema}));
        }
        self.write(
            task_id,
            TaskState::InputRequired,
            TaskPatch {
                ask: Some(Value::Object(ask)),
                status_message: Some(agent_message(parts, task_id)),
                deadline: opts.deadline,
                ..TaskPatch::default()
            },
        )
        .await?;
        Ok(())
    }

    /// Appends `msg` to the task's history; the status stays, `submitted` for a row not yet there.
    pub async fn message(&mut self, task_id: &str, msg: Value) -> Result<(), StorageError> {
        let (status, mut history) = match self.load(task_id).await? {
            Some(Base::Write(w)) => (parse_status(task_id, &w.status)?, w.data.history.clone()),
            Some(Base::Row(r)) => (parse_status(task_id, &r.status)?, r.data.history.clone()),
            None => (TaskState::Submitted, Vec::new()),
        };
        history.push(msg);
        self.write(task_id, status, TaskPatch { history: Some(history), ..TaskPatch::default() })
            .await?;
        Ok(())
    }

    /// `completed`: the result is the artifact and, when it is text, the task's last message.
    pub async fn done(&mut self, task_id: &str, result: Option<Value>) -> Result<(), StorageError> {
        let result = result.filter(|r| !r.is_null());
        let artifacts = result.iter().map(artifact).collect();
        let status_message = result
            .as_ref()
            .and_then(Value::as_str)
            .map(|t| agent_message(vec![text_part(t)], task_id));
        self.write(
            task_id,
            TaskState::Completed,
            TaskPatch { artifacts: Some(artifacts), status_message, ..TaskPatch::default() },
        )
        .await?;
        Ok(())
    }

    /// `failed`: the error is the status message, with `livekit/reason` when a reason is given.
    pub async fn fail(
        &mut self,
        task_id: &str,
        error: &str,
        reason: Option<&str>,
    ) -> Result<(), StorageError> {
        let mut msg = agent_message(vec![text_part(error)], task_id);
        if let Some(reason) = reason.filter(|r| !r.is_empty()) {
            msg["metadata"] = serde_json::json!({REASON_KEY: reason});
        }
        self.write(
            task_id,
            TaskState::Failed,
            TaskPatch { status_message: Some(msg), ..TaskPatch::default() },
        )
        .await?;
        Ok(())
    }

    /// `canceled`, with the metadata the Live extension keys carry (`livekit/interrupted`).
    pub async fn cancel(
        &mut self,
        task_id: &str,
        metadata: Option<Map<String, Value>>,
    ) -> Result<(), StorageError> {
        self.write(task_id, TaskState::Canceled, TaskPatch { metadata, ..TaskPatch::default() })
            .await?;
        Ok(())
    }
}

/// `Tasks.write`'s row: `patch` over `base`, `data` and `internal` rebuilt from both.
fn merge_write(
    task_id: &str,
    base: Option<&TaskWrite>,
    status: TaskState,
    patch: TaskPatch,
) -> Result<TaskWrite, StorageError> {
    if let Some(b) = base
        && TaskState::parse(&b.status).is_some_and(TaskState::is_terminal)
    {
        return Err(StorageError::Invalid(format!(
            "task {task_id} is {}: terminal rows are immutable",
            b.status
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

    let was = base.and_then(|b| b.internal.clone()).unwrap_or_default();
    let ask = if patch.ask.is_none() && (status.is_terminal() || status == TaskState::Working) {
        // a terminal or running row holds no pending transition
        None
    } else {
        patch.ask.or(was.ask)
    };
    let internal = TaskInternal {
        args: patch.args.or(was.args),
        ask,
        deadline: patch.deadline.or(was.deadline),
        parent: patch.parent.or(was.parent),
    };

    Ok(TaskWrite {
        task_id: task_id.to_owned(),
        status: status.as_str().to_owned(),
        message_id: patch.message_id.or_else(|| base.and_then(|b| b.message_id.clone())),
        name: patch.name.or_else(|| base.and_then(|b| b.name.clone())),
        status_message: patch
            .status_message
            .or_else(|| base.and_then(|b| b.status_message.clone())),
        data,
        internal: (!internal.is_empty()).then_some(internal),
    })
}

/// `{"messageId": "msg_..", "role": "agent", "parts": parts, "taskId": task_id}`.
pub fn agent_message(parts: Vec<Value>, task_id: &str) -> Value {
    serde_json::json!({
        "messageId": shortuuid("msg_"),
        "role": "agent",
        "parts": parts,
        "taskId": task_id,
    })
}

/// `{"artifactId": "art_..", "parts": [part]}`: a text part for a string, a data part otherwise.
pub fn artifact(value: &Value) -> Value {
    let part = match value.as_str() {
        Some(text) => text_part(text),
        None => serde_json::json!({"kind": "data", "data": value}),
    };
    serde_json::json!({"artifactId": shortuuid("art_"), "parts": [part]})
}

fn text_part(text: &str) -> Value {
    serde_json::json!({"kind": "text", "text": text})
}

/// A write as `TaskRecord`: the lane, `seq` 0 and `updated_at` 0, since the row is not stored yet.
pub fn record_from_write(w: &TaskWrite, lane: &str) -> TaskRecord {
    TaskRecord {
        id: w.task_id.clone(),
        lane: lane.to_owned(),
        seq: 0,
        name: w.name.clone(),
        message_id: w.message_id.clone(),
        status: w.status.clone(),
        status_message: w.status_message.clone(),
        updated_at: 0,
        data: w.data.clone(),
        internal: w.internal.clone(),
    }
}

fn write_from_record(r: &TaskRecord) -> TaskWrite {
    TaskWrite {
        task_id: r.id.clone(),
        status: r.status.clone(),
        message_id: r.message_id.clone(),
        name: r.name.clone(),
        status_message: r.status_message.clone(),
        data: r.data.clone(),
        internal: r.internal.clone(),
    }
}

fn parse_status(task_id: &str, status: &str) -> Result<TaskState, StorageError> {
    if status.is_empty() {
        return Ok(TaskState::Submitted);
    }
    TaskState::parse(status)
        .ok_or_else(|| StorageError::Decode(format!("task {task_id}: unknown status {status}")))
}

/// Python truthiness of a JSON value: `if row.get("ask")`.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `Tasks.write(task_id, status="submitted", name=..., args=...)` on no row.
    #[test]
    fn a_fresh_submitted_write_is_the_row() {
        let w = merge_write(
            "AST_1",
            None,
            TaskState::Submitted,
            TaskPatch {
                name: Some("Followup".into()),
                args: Some(json!({"n": 1})),
                deadline: Some(5),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&w).unwrap(),
            json!({
                "task_id": "AST_1",
                "status": "submitted",
                "message_id": null,
                "name": "Followup",
                "status_message": null,
                "data": {},
                "internal": {"args": {"n": 1}, "deadline": 5},
            })
        );
        let plain = merge_write("AST_2", None, TaskState::Submitted, TaskPatch::default()).unwrap();
        assert_eq!(
            (plain.name, plain.internal),
            (None, None),
            "a turn has no name and no internal"
        );
        let with_meta = merge_write(
            "AST_3",
            None,
            TaskState::Canceled,
            TaskPatch {
                metadata: Some(json!({"livekit/interrupted": true}).as_object().cloned().unwrap()),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&with_meta.data).unwrap(),
            r#"{"metadata":{"livekit/interrupted":true}}"#
        );
    }

    #[test]
    fn a_later_write_keeps_the_row_and_clears_the_ask_when_terminal_or_working() {
        let base = merge_write(
            "AST_1",
            None,
            TaskState::InputRequired,
            TaskPatch {
                message_id: Some("m1".into()),
                ask: Some(json!({"question": "when?"})),
                deadline: Some(9),
                history: Some(vec![json!({"messageId": "m1"})]),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        let ask = |w: &TaskWrite| w.internal.as_ref().and_then(|i| i.ask.clone());
        assert_eq!(ask(&base), Some(json!({"question": "when?"})));
        let msg = merge_write(
            "AST_1",
            Some(&base),
            TaskState::InputRequired,
            TaskPatch { history: Some(vec![json!(1), json!(2)]), ..TaskPatch::default() },
        )
        .unwrap();
        assert_eq!(ask(&msg), ask(&base), "input-required keeps the pending transition");
        assert_eq!(msg.message_id.as_deref(), Some("m1"));
        assert_eq!(msg.internal.as_ref().and_then(|i| i.deadline), Some(9));
        let done = merge_write(
            "AST_1",
            Some(&base),
            TaskState::Completed,
            TaskPatch { artifacts: Some(vec![]), ..TaskPatch::default() },
        )
        .unwrap();
        assert_eq!(ask(&done), None);
        assert_eq!(done.data.artifacts, Some(vec![]));
        assert_eq!(done.data.history, vec![json!({"messageId": "m1"})]);
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

    #[test]
    fn messages_and_artifacts_have_the_a2a_shape() {
        let m = agent_message(vec![text_part("hi")], "AST_1");
        let o = m.as_object().unwrap();
        assert_eq!(o.keys().collect::<Vec<_>>(), vec!["messageId", "role", "parts", "taskId"]);
        assert!(o["messageId"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(o["messageId"].as_str().unwrap().len(), 16);
        assert_eq!(m["role"], "agent");
        assert_eq!(m["parts"], json!([{"kind": "text", "text": "hi"}]));
        assert_eq!(m["taskId"], "AST_1");
        let a = artifact(&json!("done"));
        assert!(a["artifactId"].as_str().unwrap().starts_with("art_"));
        assert_eq!(a["parts"], json!([{"kind": "text", "text": "done"}]));
        let d = artifact(&json!({"n": 1}));
        assert_eq!(d["parts"], json!([{"kind": "data", "data": {"n": 1}}]));
        assert!(truthy(&json!({"question": "q"})) && !truthy(&json!({})) && !truthy(&json!(null)));
    }
}
