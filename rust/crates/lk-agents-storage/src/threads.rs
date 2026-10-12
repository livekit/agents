//! The threads service's client: `AgentThreads` of cloud-protocol, the durable conversation
//! (`AT_`) and the database created with it. The message types are `AgentThread.*` of
//! `livekit-protocol`'s `livekit.agent` package. A thread belongs to a project, not an agent:
//! any agent of the project picks it up, and a listing is by scope. Times cross the host as
//! unix seconds and TTLs as seconds; `0` is never.
//!
//! Errors keep gRPC's code names: `invalid_argument` for an oversized field,
//! `permission_denied` for a grant out of scope, `not_found` otherwise, which is
//! [`StorageError::ThreadNotFound`] here.

use std::collections::HashMap;
use std::time::Duration;

pub use livekit_protocol::agent::agent_task::{
    GetTasksRequest, GetTasksResponse, Status as TaskStatus, Task,
};
pub use livekit_protocol::agent::agent_thread::{
    CreateRequest, DeleteRequest, DeleteResponse, GetItemsRequest, GetItemsResponse, GetRequest,
    Item, ListRequest, ListResponse, Thread, UpdateRequest,
};
use lk_agents_host_derive::Host;
use prost::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::StorageError;
use crate::grpc::Client;
use crate::token::TokenMinter;

#[derive(Clone)]
pub struct Threads {
    client: Client,
}

impl Threads {
    pub fn new(url: &str, minter: TokenMinter) -> Result<Self, StorageError> {
        Ok(Self { client: Client::new(url, minter)? })
    }

    async fn call<Req: Message + 'static, Resp: Message + Default + 'static>(
        &self,
        method: &'static str,
        req: Req,
    ) -> Result<Resp, StorageError> {
        self.client.unary(method, req).await.map_err(|e| match e {
            StorageError::NotFound(m) => StorageError::ThreadNotFound(m),
            other => other,
        })
    }

    /// A new thread with its database.
    pub async fn create_thread(&self, req: CreateRequest) -> Result<Thread, StorageError> {
        self.call("/cloud_protocol.AgentThreads/CreateThread", req).await
    }

    pub async fn get_thread(&self, thread_id: &str) -> Result<Thread, StorageError> {
        self.call(
            "/cloud_protocol.AgentThreads/GetThread",
            GetRequest { thread_id: thread_id.to_owned() },
        )
        .await
    }

    pub async fn list_threads(&self, req: ListRequest) -> Result<ListResponse, StorageError> {
        self.call("/cloud_protocol.AgentThreads/ListThreads", req).await
    }

    pub async fn update_thread(&self, req: UpdateRequest) -> Result<Thread, StorageError> {
        self.call("/cloud_protocol.AgentThreads/UpdateThread", req).await
    }

    pub async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        self.call::<_, DeleteResponse>(
            "/cloud_protocol.AgentThreads/DeleteThread",
            DeleteRequest { thread_id: thread_id.to_owned() },
        )
        .await
        .map(|_| ())
    }

    pub async fn get_thread_items(
        &self,
        req: GetItemsRequest,
    ) -> Result<GetItemsResponse, StorageError> {
        self.call("/cloud_protocol.AgentThreads/GetThreadItems", req).await
    }

    pub async fn get_thread_tasks(
        &self,
        req: GetTasksRequest,
    ) -> Result<GetTasksResponse, StorageError> {
        self.call("/cloud_protocol.AgentThreads/GetThreadTasks", req).await
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

impl From<ListResponse> for ThreadPage {
    fn from(r: ListResponse) -> Self {
        Self {
            threads: r.threads.into_iter().map(ThreadInfo::from).collect(),
            before: r.next_page_token,
        }
    }
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

impl TryFrom<GetItemsResponse> for ThreadItemPage {
    type Error = StorageError;

    fn try_from(r: GetItemsResponse) -> Result<Self, StorageError> {
        let items = r.items.into_iter().map(ThreadItem::try_from).collect::<Result<_, _>>()?;
        Ok(Self { items, after: r.next_page_token })
    }
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
pub fn task_status(status: Option<&str>) -> Result<Option<i32>, StorageError> {
    status
        .map(|s| {
            TaskStatus::from_str_name(&s.to_uppercase())
                .map(i32::from)
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
        assert_eq!(task_status(Some("completed")).unwrap(), Some(TaskStatus::Completed.into()));
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
