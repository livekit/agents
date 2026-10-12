//! The records a run reads and writes, as `storage/protocol.py` defines them: the task states and
//! the run outcomes (plan 5.5, 5.1), `Ref` (rule 9 of 5.3), `TaskRecord`, `RunSummary` and
//! `LaneInfo`. JSON shapes are the protocol's, byte for byte: `Ref` is `{name, args}`, a task row's
//! `data` column the A2A `Task` fields and its `internal` column the `AgentTask`'s own.

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::StorageError;
use crate::schema::{EventRow, LaneRow, TaskData, TaskInternal, TaskRow};

/// The Live extension's metadata keys (plan 5.5, 5.12), written `livekit/<name>`.
pub const INTERRUPTED_KEY: &str = "livekit/interrupted";
pub const INTERRUPTION_REASON_KEY: &str = "livekit/interruptionReason";
pub const MERGED_INTO_KEY: &str = "livekit/mergedInto";
pub const CHANNEL_KEY: &str = "livekit/channel";
pub const KIND_KEY: &str = "livekit/kind";
pub const KEY_KEY: &str = "livekit/key";
/// The key `Tasks::fail` writes into the status message's metadata.
pub const REASON_KEY: &str = "livekit/reason";

/// Plan 5.12: the `failed` and `rejected` reasons a client and a dashboard read.
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

/// The A2A task states (plan 5.5); the last four are terminal and immutable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Host)]
#[serde(rename_all = "kebab-case")]
#[python(literal, module = "storage_run")]
pub enum TaskState {
    #[host(rename = "submitted")]
    Submitted,
    #[host(rename = "working")]
    Working,
    #[host(rename = "input-required")]
    InputRequired,
    #[host(rename = "completed")]
    Completed,
    #[host(rename = "failed")]
    Failed,
    #[host(rename = "canceled")]
    Canceled,
    #[host(rename = "rejected")]
    Rejected,
}

impl TaskState {
    /// `None` for a status no state spells (the derive's `from_wire`).
    pub fn parse(s: &str) -> Option<TaskState> {
        TaskState::from_wire(s)
    }

    /// `completed`, `failed`, `canceled`, `rejected`: immutable once written.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Failed | TaskState::Canceled | TaskState::Rejected
        )
    }
}

/// What `run_ended` records (plan 5.1).
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

/// One rebuild rule for everything durable that names code (plan 5.2, rule 9 of 5.3): `name` is an
/// `Agent`'s name or an `AgentTask` class name; `args` the constructor arguments, JSON-shaped,
/// `None` for a plain `Agent`. On the agent stack a task entry also carries `state`, the task's
/// own state as it waits, and `call_id`, the call of the agent below that awaits it. Stored as
/// `{"name": .., "args": .., "state": .., "call_id": ..}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run", config = "frozen=True")]
pub struct Ref {
    pub name: String,
    #[serde(default)]
    #[python(default = "None")]
    pub args: Option<Value>,
    #[serde(default)]
    #[python(default = "None")]
    pub state: Option<Value>,
    #[serde(default)]
    #[python(default = "None")]
    pub call_id: Option<String>,
}

impl Ref {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), args: None, state: None, call_id: None }
    }

    pub fn with_args(name: impl Into<String>, args: Value) -> Self {
        Self { name: name.into(), args: Some(args), state: None, call_id: None }
    }

    /// `{"name", "args"}`, plus `state` and `call_id` when the entry has them.
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("name".into(), Value::String(self.name.clone()));
        m.insert("args".into(), self.args.clone().unwrap_or(Value::Null));
        if let Some(state) = &self.state {
            m.insert("state".into(), state.clone());
        }
        if let Some(call_id) = &self.call_id {
            m.insert("call_id".into(), Value::String(call_id.clone()));
        }
        Value::Object(m)
    }

    /// `Ref.from_json`: `name` as a string, the rest as stored (`null` is `None`).
    pub fn from_json(data: &Value) -> Result<Self, StorageError> {
        let name = data.get("name").ok_or_else(|| StorageError::Decode("ref: no name".into()))?;
        let name = match name {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let field = |key: &str| data.get(key).filter(|a| !a.is_null()).cloned();
        let call_id = field("call_id").and_then(|c| c.as_str().map(str::to_owned));
        Ok(Self { name, args: field("args"), state: field("state"), call_id })
    }

    /// The agent stack column as `Ref`s; anything that is not a list is an empty stack.
    pub fn stack_from_json(stack: Option<&Value>) -> Result<Vec<Ref>, StorageError> {
        match stack.and_then(Value::as_array) {
            Some(refs) => refs.iter().map(Ref::from_json).collect(),
            None => Ok(Vec::new()),
        }
    }
}

/// One task row as the framework and a dashboard read it (plan 5.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run", config = "frozen=True")]
pub struct TaskRecord {
    pub id: String,
    pub lane: String,
    pub seq: i64,
    /// The `AgentTask` class; `None` for a turn.
    pub name: Option<String>,
    pub message_id: Option<String>,
    /// A `TaskState`, as the row spells it.
    pub status: String,
    pub status_message: Option<Value>,
    pub updated_at: i64,
    /// The A2A `Task` fields: `history`, `artifacts`, `metadata`.
    pub data: TaskData,
    /// The `AgentTask`'s arguments and pending question.
    pub internal: Option<TaskInternal>,
}

impl TaskRecord {
    pub fn state(&self) -> Option<TaskState> {
        TaskState::parse(&self.status)
    }

    pub fn terminal(&self) -> bool {
        self.state().is_some_and(TaskState::is_terminal)
    }

    /// The pending transition, when the task waits on one.
    pub fn ask(&self) -> Option<&Value> {
        self.internal.as_ref().and_then(|i| i.ask.as_ref())
    }

    /// A `data` or `internal` column that does not parse reads as empty.
    pub fn from_row(row: TaskRow) -> Self {
        Self {
            id: row.task_id,
            lane: row.lane,
            seq: row.seq,
            name: row.name,
            message_id: row.message_id,
            status: row.status,
            status_message: row.status_message,
            updated_at: row.updated_at,
            data: serde_json::from_value(row.data).unwrap_or_default(),
            internal: row.internal.and_then(|i| serde_json::from_value(i).ok()),
        }
    }
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
    pub outcome: Option<String>,
    pub reason: Option<String>,
}

impl RunSummary {
    /// A `run_started` without a `run_ended`: the worker died mid-run (plan 5.1).
    pub fn interrupted(&self) -> bool {
        self.ended_at_ms.is_none()
    }

    /// In the order the runs started; a `run_ended` whose run was not started is ignored, and a
    /// second `run_started` of a run (a holder re-entering its lease) replaces the first in place.
    pub fn from_events(events: &[EventRow]) -> Vec<RunSummary> {
        let mut runs: Vec<RunSummary> = Vec::new();
        for ev in events {
            let payload = ev.payload.as_object();
            let text = |key: &str| {
                payload.and_then(|p| p.get(key)).and_then(Value::as_str).map(str::to_owned)
            };
            if ev.r#type == "run_started" {
                let summary = RunSummary {
                    run_id: ev.run_id.clone(),
                    lane: ev.lane.clone(),
                    task_id: text("task_id"),
                    channel: text("channel"),
                    started_at_ms: ev.ts_ms,
                    ended_at_ms: None,
                    outcome: None,
                    reason: None,
                };
                match runs.iter_mut().find(|r| r.run_id == ev.run_id) {
                    Some(existing) => *existing = summary,
                    None => runs.push(summary),
                }
            } else if ev.r#type == "run_ended"
                && let Some(run) = runs.iter_mut().find(|r| r.run_id == ev.run_id)
            {
                run.ended_at_ms = Some(ev.ts_ms);
                run.outcome = text("outcome");
                run.reason = text("reason");
            }
        }
        runs
    }
}

/// One lane row, read from outside a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run", config = "frozen=True")]
pub struct LaneInfo {
    pub name: String,
    pub version: i64,
    pub agent_stack: Vec<Ref>,
    pub lease_holder: Option<String>,
    pub lease_until: i64,
    pub waiter: Option<String>,
    pub drained_id: i64,
}

impl LaneInfo {
    pub fn from_row(row: LaneRow) -> Result<Self, StorageError> {
        Ok(Self {
            name: row.lane,
            version: row.version,
            agent_stack: Ref::stack_from_json(row.agent_stack.as_ref())?,
            lease_holder: row.lease_holder,
            lease_until: row.lease_until,
            waiter: row.waiter,
            drained_id: row.drained_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn states_and_outcomes_spell_as_the_protocol() {
        assert_eq!(TaskState::InputRequired.as_str(), "input-required");
        assert_eq!(TaskState::parse("canceled"), Some(TaskState::Canceled));
        assert_eq!(TaskState::parse("cancelled"), None);
        assert!(TaskState::Rejected.is_terminal());
        assert!(!TaskState::Working.is_terminal());
        assert_eq!(serde_json::to_value(RunOutcome::InputRequired).unwrap(), "input-required");
        assert_eq!(serde_json::to_value(TaskState::Submitted).unwrap(), "submitted");
    }

    #[test]
    fn a_ref_round_trips_as_name_and_args() {
        let r = Ref::with_args("Followup", json!({"n": 1}));
        assert_eq!(r.to_json().to_string(), r#"{"name":"Followup","args":{"n":1}}"#);
        assert_eq!(
            Ref::new("front-desk").to_json().to_string(),
            r#"{"name":"front-desk","args":null}"#
        );
        let task = Ref {
            name: "BookRoom".into(),
            args: Some(json!({"guests": 2})),
            state: Some(json!({"room": "suite"})),
            call_id: Some("call_1".into()),
        };
        assert_eq!(Ref::from_json(&task.to_json()).unwrap(), task);
        assert_eq!(Ref::from_json(&json!({"name": "x"})).unwrap(), Ref::new("x"));
        assert_eq!(Ref::from_json(&json!({"name": "x", "args": null})).unwrap(), Ref::new("x"));
        assert_eq!(Ref::stack_from_json(Some(&json!({}))).unwrap(), vec![]);
        assert_eq!(Ref::stack_from_json(None).unwrap(), vec![]);
        assert!(Ref::from_json(&json!({"args": {}})).is_err());
    }

    #[test]
    fn a_task_record_reads_data_and_internal() {
        let row = TaskRow {
            task_id: "AST_1".into(),
            lane: "front-desk".into(),
            seq: 2,
            name: Some("Followup".into()),
            message_id: Some("m1".into()),
            status: "completed".into(),
            status_message: None,
            updated_at: 9,
            data: json!({"history": [{"messageId": "m1"}], "artifacts": [], "metadata": {"k": "v"}}),
            internal: Some(json!({"args": {"n": 1}, "deadline": 5})),
        };
        let rec = TaskRecord::from_row(row);
        assert_eq!(rec.name.as_deref(), Some("Followup"));
        let internal = rec.internal.clone().unwrap();
        assert_eq!(
            (internal.args, internal.deadline, internal.ask),
            (Some(json!({"n": 1})), Some(5), None)
        );
        assert_eq!(rec.data.metadata.get("k"), Some(&json!("v")));
        assert_eq!(rec.data.history.len(), 1);
        assert_eq!(rec.data.artifacts, Some(vec![]));
        assert!(rec.terminal());
        let empty: TaskData = serde_json::from_value(json!({})).unwrap();
        assert_eq!(serde_json::to_string(&empty).unwrap(), "{}");
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
        assert_eq!(runs[0].outcome.as_deref(), Some("completed"));
        assert_eq!(runs[0].ended_at_ms, Some(20));
        assert!(!runs[0].interrupted());
        assert_eq!(runs[1].channel, None);
        assert!(runs[1].interrupted());
    }
}
