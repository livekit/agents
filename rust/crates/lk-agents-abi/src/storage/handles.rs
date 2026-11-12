use super::context::{ContextScope, ContextScopeWrite};
use super::{
    EventRow, InboxRow, ItemRow, PartitionRow, StorageError, TaskPage, TaskRecord, TaskState,
    ThreadItemPage,
};
use serde_json::{Map, Value};

#[crate::host(class(name = "Thread"))]
pub trait Thread: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(class(name = "Run"))]
    type Run: Send + Sync + 'static;

    #[host(getter)]
    fn id(&self) -> String;
    #[host(await)]
    async fn partitions(&self) -> Result<Vec<PartitionRow>, StorageError>;
    #[host(await)]
    async fn journal(
        &self,
        #[host(default = 0)] after_id: i64,
        #[host(default = 100)] limit: i64,
    ) -> Result<Vec<EventRow>, StorageError>;
    #[host(await)]
    async fn history(&self, partition_key: String) -> Result<Vec<ItemRow>, StorageError>;
    #[host(await)]
    async fn tasks(
        &self,
        partition_key: String,
        #[host(default = 50)] limit: i64,
    ) -> Result<Vec<TaskRecord>, StorageError>;
    #[host(await)]
    async fn task(&self, task_id: String) -> Result<Option<TaskRecord>, StorageError>;
    #[host(await)]
    async fn list_tasks(
        &self,
        #[host(default = None)] partition_key: Option<String>,
        #[host(default = None)] status: Option<TaskState>,
        #[host(default = "updated")] order: crate::storage::thread::TaskOrder,
        #[host(default = 50)] page_size: i64,
        #[host(default = None)] page_token: Option<String>,
    ) -> Result<TaskPage, StorageError>;
    #[host(await)]
    async fn list_items(
        &self,
        #[host(default = None)] partition_key: Option<String>,
        #[host(default = None)] after: Option<String>,
        #[host(default = 100)] limit: i64,
    ) -> Result<ThreadItemPage, StorageError>;
    #[host(await)]
    async fn partition_info(
        &self,
        partition_key: String,
    ) -> Result<Option<PartitionRow>, StorageError>;
    #[host(await)]
    async fn inbox(
        &self,
        partition_key: String,
        #[host(default = false)] pending: bool,
    ) -> Result<Vec<InboxRow>, StorageError>;
    #[host(await)]
    async fn cancel_task(
        &self,
        partition_key: String,
        task_id: String,
        metadata: Map<String, Value>,
    ) -> Result<Option<TaskRecord>, StorageError>;
    #[host(await)]
    async fn open_run(
        &self,
        partition_key: String,
        run_id: String,
        #[host(kw, default = None)] channel: Option<String>,
        #[host(default = None)] wait_timeout: Option<std::time::Duration>,
    ) -> Result<Self::Run, StorageError>;
    /// Schedule connection release without blocking the caller.
    #[host(call)]
    fn close(&self, env: &Self::Env);
}

#[crate::host(class(name = "Run"))]
pub trait Run: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn version(&self) -> i64;
    #[host(getter)]
    fn closed(&self) -> bool;
    #[host(getter)]
    fn history(&self) -> Result<Vec<ItemRow>, StorageError>;
    #[host(getter)]
    fn state(&self) -> Option<Value>;
    #[host(getter)]
    fn context_scopes(&self) -> Vec<ContextScope>;
    #[host(await)]
    async fn commit(
        &self,
        history: Vec<ItemRow>,
        #[host(default = None)] state: Option<Value>,
        #[host(kw, default = None)] scope_writes: Option<Vec<ContextScopeWrite>>,
    ) -> Result<i64, StorageError>;
    #[host(await)]
    async fn abandon(
        &self,
        outcome: crate::storage::thread::RunOutcome,
        reason: String,
    ) -> Result<(), StorageError>;
}

/// `ctx.thread` (docs/threads.md): one thread of a storage namespace, as the partition_key `partition_key` reads
/// it. Each read opens the thread's database for the call and releases it after; there is
/// nothing to open or close.
#[crate::host(class(name = "_core_ThreadHandle"))]
pub trait ThreadHandle: Send + Sync + Sized + 'static {
    #[host(class(name = "Storage"))]
    type Namespace: Send + Sync + 'static;

    /// Reads take no lease.
    #[host(new)]
    fn new(
        namespace: std::sync::Arc<Self::Namespace>,
        thread_id: String,
        partition_key: String,
    ) -> Self;

    /// The thread id.
    #[host(getter)]
    fn id(&self) -> String;
    /// The partition_key whose history `chat_history()` reads by default.
    #[host(getter)]
    fn partition_key(&self) -> String;
    /// The namespace the thread lives in.
    #[host(getter)]
    fn namespace(&self) -> std::sync::Arc<Self::Namespace>;

    /// The platform's record: scope, attributes, created, last active, expiry.
    #[host(await)]
    async fn info(&self) -> Result<super::ThreadInfo, StorageError>;
    /// Change the attributes a listing returns, or the idle period.
    #[host(await)]
    async fn update(
        &self,
        #[host(kw, default = None)] attributes: Option<std::collections::BTreeMap<String, String>>,
        #[host(default = None)] idle_ttl_seconds: Option<i64>,
    ) -> Result<super::ThreadInfo, StorageError>;
    /// One page of the thread's committed chat items in insertion order, every partition's or
    /// `partition_key`'s; pass the page's `after` to read the next (empty on the last page). The same
    /// records Cloud's threads API answers, for an application's own read routes.
    #[host(await)]
    async fn list_items(
        &self,
        #[host(kw, default = None)] partition_key: Option<String>,
        #[host(default = None)] after: Option<String>,
        #[host(default = 100)] limit: i64,
    ) -> Result<ThreadItemPage, StorageError>;
    /// One page of the thread's A2A task records: in insertion order (`order="inserted"`), or
    /// last updated first (`"updated"`) as A2A `tasks/list` lists them; of `partition_key`'s partition and
    /// `status` when given. Pass the page's `next_page_token` to read the next; a token is bound
    /// to its listing.
    #[host(await)]
    async fn list_tasks(
        &self,
        #[host(kw, default = None)] partition_key: Option<String>,
        #[host(default = None)] status: Option<TaskState>,
        #[host(default = "inserted")] order: crate::storage::thread::TaskOrder,
        #[host(default = None)] page_token: Option<String>,
        #[host(default = 50)] limit: i64,
    ) -> Result<TaskPage, StorageError>;
    /// One task record, `None` when this thread has none of that id.
    #[host(await)]
    async fn task(&self, task_id: String) -> Result<Option<TaskRecord>, StorageError>;
    /// One page of the journal (docs/internals/storage.md#journal), every partition, in order; pass
    /// the last `id` to read the next.
    #[host(await)]
    async fn journal(
        &self,
        #[host(default = 0)] after_id: i64,
        #[host(default = 100)] limit: i64,
    ) -> Result<Vec<EventRow>, StorageError>;
    /// `partition_key`'s partition history, in order.
    #[host(await)]
    async fn history(&self, partition_key: String) -> Result<Vec<ItemRow>, StorageError>;
    /// `partition_key`'s partition task records, most recent first; only those in `status` when given.
    #[host(await)]
    async fn tasks(
        &self,
        partition_key: String,
        #[host(kw, default = None)] status: Option<TaskState>,
        #[host(default = 50)] limit: i64,
    ) -> Result<Vec<TaskRecord>, StorageError>;
}

/// `ctx.thread`'s reads (docs/threads.md): each opens the thread's database for the call.
#[crate::host(facade(
    path = "livekit.agents.storage",
    extended,
    name = "Thread",
    native = ThreadHandle
))]
pub trait ThreadFacade {
    /// The thread id.
    #[host(getter)]
    fn id(&self) -> String;
    /// The platform's record: scope, attributes, created, last active, expiry.
    async fn info(&self) -> Result<super::ThreadInfo, StorageError>;
    /// Change the attributes a listing returns, or the idle period.
    async fn update(
        &self,
        #[host(kw, default = None)] attributes: Option<std::collections::BTreeMap<String, String>>,
        #[host(default = None)] idle_ttl_seconds: Option<i64>,
    ) -> Result<super::ThreadInfo, StorageError>;
    /// One page of the thread's committed chat items in insertion order, every partition's or
    /// `partition_key`'s; pass the page's `after` to read the next (empty on the last page). The same
    /// records Cloud's threads API answers, for an application's own read routes.
    async fn list_items(
        &self,
        #[host(kw, default = None)] partition_key: Option<String>,
        #[host(default = None)] after: Option<String>,
        #[host(default = 100)] limit: i64,
    ) -> Result<ThreadItemPage, StorageError>;
    /// One page of the thread's A2A task records: in insertion order (`order="inserted"`), or
    /// last updated first (`"updated"`) as A2A `tasks/list` lists them; of `partition_key`'s partition and
    /// `status` when given. Pass the page's `next_page_token` to read the next; a token is bound
    /// to its listing.
    async fn list_tasks(
        &self,
        #[host(kw, default = None)] partition_key: Option<String>,
        #[host(default = None)] status: Option<TaskState>,
        #[host(default = "inserted")] order: crate::storage::thread::TaskOrder,
        #[host(default = None)] page_token: Option<String>,
        #[host(default = 50)] limit: i64,
    ) -> Result<TaskPage, StorageError>;
    /// One task record, `None` when this thread has none of that id.
    async fn task(&self, task_id: String) -> Result<Option<TaskRecord>, StorageError>;
    /// One page of the journal (docs/internals/storage.md#journal), every partition, in order; pass
    /// the last `id` to read the next.
    async fn journal(
        &self,
        #[host(default = 0)] after_id: i64,
        #[host(default = 100)] limit: i64,
    ) -> Result<Vec<EventRow>, StorageError>;
}

/// `thread.partition(partition_key)`: one partition_key's partition of a thread (docs/threads.md), its history and its
/// tasks.
#[crate::host(facade(path = "livekit.agents.storage", native = ThreadHandle, from_native))]
pub trait Partition {
    #[host(new)]
    fn new(partition_key: String) -> Self;
    /// The partition's whole history, in order.
    #[host(bind(partition_key = "_partition_key"), returns = "..llm.ChatContext", each = ItemRow::payload)]
    async fn history(&self) -> Result<Vec<ItemRow>, StorageError>;
    /// The partition's A2A task records, most recent first; only those in `status` when given.
    #[host(bind(partition_key = "_partition_key"))]
    async fn tasks(
        &self,
        #[host(kw, default = None)] status: Option<TaskState>,
        #[host(default = 50)] limit: i64,
    ) -> Result<Vec<TaskRecord>, StorageError>;
}
