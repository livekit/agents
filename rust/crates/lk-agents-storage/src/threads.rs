//! The threads service's client: `ThreadService` of the Public API, the durable conversation
//! (`AT_`) and the database created with it. The thread, item and task messages are
//! `livekit-protocol`'s `livekit.agent` package. A thread belongs to a project, not an agent: any
//! agent of the project picks it up, and a listing is by scope. Times cross the host as unix
//! seconds and TTLs as seconds; `0` is never. The client is `livekit-api`'s `ThreadClient`.
//!
//! A thread the service does not know is [`StorageError::ThreadNotFound`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use livekit_api::services::threads::ThreadClient;
pub use livekit_protocol::agent::agent_task::{GetTasksRequest, Status as TaskStatus, Task};
pub use livekit_protocol::agent::agent_thread::{
    CreateRequest, GetItemsRequest, Item, ListRequest, Thread, UpdateRequest,
};
use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::StorageError;
use crate::agentstore::error;

#[derive(Clone)]
pub struct Threads {
    client: Arc<ThreadClient>,
}

/// A `livekit-api` failure, with a missing thread as [`StorageError::ThreadNotFound`].
fn thread_error(e: livekit_api::services::ServiceError) -> StorageError {
    match error(e) {
        StorageError::NotFound(m) => StorageError::ThreadNotFound(m),
        other => other,
    }
}

impl Threads {
    pub(crate) fn new(client: ThreadClient) -> Self {
        Self { client: Arc::new(client) }
    }

    /// A new thread with its database.
    pub async fn create_thread(
        &self,
        scope: &str,
        attributes: HashMap<String, String>,
        idle_ttl_seconds: i64,
    ) -> Result<Thread, StorageError> {
        let req = CreateRequest {
            scope: scope.to_owned(),
            attributes,
            idle_ttl: duration_of(idle_ttl_seconds),
        };
        self.client.create(req).await.map_err(thread_error)
    }

    pub async fn get_thread(&self, thread_id: &str) -> Result<Thread, StorageError> {
        self.client.get(thread_id).await.map_err(thread_error)
    }

    /// One page of a scope's threads, newest first; `before` is the cursor, `limit` 0 the
    /// server's default.
    pub async fn list_threads(
        &self,
        scope: &str,
        before: &str,
        limit: i32,
    ) -> Result<ThreadPage, StorageError> {
        let req = ListRequest { scope: scope.to_owned(), page_token: before.to_owned(), limit };
        let r = self.client.list(req).await.map_err(thread_error)?;
        Ok(ThreadPage {
            threads: r.threads.into_iter().map(ThreadInfo::from).collect(),
            before: r.next_page_token,
        })
    }

    /// Merges `attributes` (an empty value deletes the entry) and sets the idle TTL when given.
    pub async fn update_thread(
        &self,
        thread_id: &str,
        attributes: HashMap<String, String>,
        idle_ttl_seconds: Option<i64>,
    ) -> Result<Thread, StorageError> {
        let req = UpdateRequest {
            thread_id: thread_id.to_owned(),
            attributes,
            idle_ttl: idle_ttl_seconds.and_then(duration_of),
        };
        self.client.update(req).await.map_err(thread_error)
    }

    pub async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        self.client.delete(thread_id).await.map_err(thread_error)
    }

    /// One page of the thread's chat items, oldest first; `agent` keeps one agent's.
    pub async fn get_thread_items(
        &self,
        thread_id: &str,
        agent: &str,
        after: &str,
        limit: i32,
    ) -> Result<ThreadItemPage, StorageError> {
        let req = GetItemsRequest {
            thread_id: thread_id.to_owned(),
            agent: agent.to_owned(),
            page_token: after.to_owned(),
            limit,
            ..Default::default()
        };
        let r = self.client.items(req).await.map_err(thread_error)?;
        let items = r.items.into_iter().map(ThreadItem::try_from).collect::<Result<_, _>>()?;
        Ok(ThreadItemPage { items, after: r.next_page_token })
    }

    /// Every background task of the thread, of one status when given.
    pub async fn get_thread_tasks(
        &self,
        thread_id: &str,
        status: Option<TaskStatus>,
    ) -> Result<Vec<Task>, StorageError> {
        let mut tasks = Vec::new();
        let mut page_token = String::new();
        loop {
            let req = GetTasksRequest {
                thread_id: thread_id.to_owned(),
                status: status.map(i32::from),
                page_token,
                ..Default::default()
            };
            let r = self.client.tasks(req).await.map_err(thread_error)?;
            tasks.extend(r.tasks);
            if r.next_page_token.is_empty() {
                return Ok(tasks);
            }
            page_token = r.next_page_token;
        }
    }
}

/// A `Timestamp` as unix seconds; `None` is `0`, never.
pub fn seconds_of(t: Option<pbjson_types::Timestamp>) -> i64 {
    t.map_or(0, |t| t.seconds)
}

/// A `Duration` as whole seconds; `None` is `0`.
pub fn duration_seconds(d: Option<pbjson_types::Duration>) -> i64 {
    d.map_or(0, |d| d.seconds)
}

/// Seconds as a `Duration` for the wire; `0` sends none, the server's default.
pub fn duration_of(seconds: i64) -> Option<pbjson_types::Duration> {
    (seconds > 0).then(|| pbjson_types::Duration::from(Duration::from_secs(seconds as u64)))
}

/// A thread as the host reads it (plan 4.3): the proto's fields, times as unix seconds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadInfo {
    pub thread_id: String,
    /// The subject the thread belongs to, the one a grant is scoped by.
    pub scope: String,
    pub attributes: HashMap<String, String>,
    pub created_at: i64,
    pub last_active: i64,
    /// Unix seconds; `0` for never.
    pub expires_at: i64,
    pub idle_ttl_seconds: i64,
    /// The thread's database on agent-db, `DB_..`, created with the thread.
    pub database_id: String,
}

impl From<Thread> for ThreadInfo {
    fn from(t: Thread) -> Self {
        Self {
            thread_id: t.thread_id,
            scope: t.scope,
            attributes: t.attributes,
            created_at: seconds_of(t.created_at),
            last_active: seconds_of(t.last_active),
            expires_at: seconds_of(t.expires_at),
            idle_ttl_seconds: duration_seconds(t.idle_ttl),
            database_id: t.database_id,
        }
    }
}

/// One page of `ListThreads`, newest first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadPage {
    pub threads: Vec<ThreadInfo>,
    /// The cursor for the next page; empty at the end.
    pub before: String,
}

/// One chat item of a thread as the host reads it; `item` is the `ChatContext.ChatItem` as
/// protobuf JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadItem {
    pub agent: String,
    /// Position in the agent's history.
    pub seq: i64,
    pub message_id: String,
    pub item: Value,
}

impl TryFrom<Item> for ThreadItem {
    type Error = StorageError;

    fn try_from(i: Item) -> Result<Self, StorageError> {
        let item = serde_json::to_value(i.item).map_err(|e| StorageError::Decode(e.to_string()))?;
        Ok(Self { agent: i.agent, seq: i.seq, message_id: i.message_id, item })
    }
}

/// One page of `GetThreadItems`, oldest first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadItemPage {
    pub items: Vec<ThreadItem>,
    /// The cursor for the next page; empty at the end.
    pub after: String,
}

/// One background task of a thread, as `GetThreadTasks` answers it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadTask {
    pub task_id: String,
    pub thread_id: String,
    pub name: String,
    /// `running`, `completed`, `failed` or `canceled`.
    pub status: String,
    pub updated_at: i64,
}

impl From<Task> for ThreadTask {
    fn from(t: Task) -> Self {
        let status = TaskStatus::try_from(t.status).unwrap_or_default();
        Self {
            task_id: t.task_id,
            thread_id: t.thread_id,
            name: t.name,
            status: status.as_str_name().to_lowercase(),
            updated_at: seconds_of(t.updated_at),
        }
    }
}

/// A task status from the host, `running` and so on; `None` for every status.
pub fn task_status(status: Option<&str>) -> Result<Option<TaskStatus>, StorageError> {
    status
        .map(|s| {
            TaskStatus::from_str_name(&s.to_uppercase())
                .ok_or_else(|| StorageError::Invalid(format!("unknown task status {s}")))
        })
        .transpose()
}

/// `attributes` from the host: a string map.
pub fn attributes_from_json(v: Option<&Value>) -> Result<HashMap<String, String>, StorageError> {
    let Some(v) = v else { return Ok(HashMap::new()) };
    let obj = v
        .as_object()
        .ok_or_else(|| StorageError::Invalid("attributes must be an object of strings".into()))?;
    obj.iter()
        .map(|(k, v)| {
            v.as_str()
                .map(|s| (k.clone(), s.to_owned()))
                .ok_or_else(|| StorageError::Invalid(format!("attribute {k} is not a string")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thread_renders_with_the_proto_field_names() {
        let t =
            Thread { thread_id: "AT_1".into(), database_id: "DB_1".into(), ..Default::default() };
        let v = serde_json::to_value(ThreadInfo::from(t)).unwrap();
        assert_eq!(v["thread_id"], "AT_1");
        assert_eq!(v["expires_at"], 0, "no timestamp is never");
        assert_eq!(seconds_of(Some(pbjson_types::Timestamp { seconds: 7, nanos: 0 })), 7);
        assert_eq!(duration_of(0), None);
        assert_eq!(duration_seconds(duration_of(90)), 90);
        assert_eq!(v["database_id"], "DB_1");
        assert!(attributes_from_json(Some(&serde_json::json!({"a": 1}))).is_err());
        let task = Task { status: TaskStatus::Running.into(), ..Default::default() };
        assert_eq!(ThreadTask::from(task).status, "running");
        assert_eq!(task_status(Some("completed")).unwrap(), Some(TaskStatus::Completed));
        assert!(task_status(Some("nope")).is_err());
    }

    #[test]
    fn a_thread_item_keeps_its_chat_item() {
        use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
        let message = chat_item::Item::Message(Default::default());
        let item =
            Item { seq: 3, item: Some(ChatItem { item: Some(message) }), ..Default::default() };
        let v = ThreadItem::try_from(item).unwrap();
        assert_eq!(v.seq, 3);
        assert!(v.item.get("message").is_some(), "the item crosses, not null: {}", v.item);
    }
}
