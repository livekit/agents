//! Open thread and lane-run handles. Storage owns their resources; bindings only translate values.

use super::context::{ContextScope, ContextScopeWrite};
use super::{
    EventRow, InboxRow, ItemRow, LaneRow, StorageError, TaskPage, TaskRecord, TaskState,
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
    #[host(getter)]
    fn holder(&self) -> String;
    #[host(await)]
    async fn lanes(&self) -> Result<Vec<LaneRow>, StorageError>;
    #[host(await)]
    async fn journal(
        &self,
        #[host(default = 0)] after_id: i64,
        #[host(default = 100)] limit: i64,
    ) -> Result<Vec<EventRow>, StorageError>;
    #[host(await)]
    async fn history(&self, lane: String) -> Result<Vec<ItemRow>, StorageError>;
    #[host(await)]
    async fn tasks(
        &self,
        lane: String,
        #[host(default = 50)] limit: i64,
    ) -> Result<Vec<TaskRecord>, StorageError>;
    #[host(await)]
    async fn task(&self, task_id: String) -> Result<Option<TaskRecord>, StorageError>;
    #[host(await)]
    async fn list_tasks(
        &self,
        #[host(default = None)] lane: Option<String>,
        #[host(default = None)] status: Option<TaskState>,
        #[host(default = "updated")] order: String,
        #[host(default = 50)] page_size: i64,
        #[host(default = None)] page_token: Option<String>,
    ) -> Result<TaskPage, StorageError>;
    #[host(await)]
    async fn list_items(
        &self,
        #[host(default = None)] lane: Option<String>,
        #[host(default = None)] after: Option<String>,
        #[host(default = 100)] limit: i64,
    ) -> Result<ThreadItemPage, StorageError>;
    #[host(await)]
    async fn lane_info(&self, lane: String) -> Result<Option<LaneRow>, StorageError>;
    #[host(await)]
    async fn inbox(
        &self,
        lane: String,
        #[host(default = false)] pending: bool,
    ) -> Result<Vec<InboxRow>, StorageError>;
    #[host(await)]
    async fn cancel_task(
        &self,
        lane: String,
        task_id: String,
        metadata: Map<String, Value>,
    ) -> Result<Option<TaskRecord>, StorageError>;
    #[host(await)]
    async fn open_run(
        &self,
        lane: String,
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
    fn history(&self) -> Vec<ItemRow>;
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
    async fn abandon(&self, outcome: String, reason: String) -> Result<(), StorageError>;
}
