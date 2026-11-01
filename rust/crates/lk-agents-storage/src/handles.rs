//! ABI views of open threads and explicitly committed lane runs.

use crate::{CommitOptions, ItemRow, Run, RunOptions, RunOutcome, StorageError, Thread};
use lk_agents_abi::SpawnerExt;
use lk_agents_abi::storage as host;
use lk_agents_abi::storage::context::{ContextScope, ContextScopeWrite};
use lk_agents_net::env::ProcessEnv;
use serde_json::{Map, Value};
use std::sync::Arc;

impl host::handles::Thread for Thread {
    type Env = ProcessEnv;
    type Run = RunHandle;

    fn id(&self) -> String {
        self.id().to_owned()
    }
    fn holder(&self) -> String {
        self.holder().to_owned()
    }
    async fn lanes(&self) -> Result<Vec<host::LaneRow>, StorageError> {
        self.lanes().await
    }
    async fn journal(
        &self,
        after_id: i64,
        limit: i64,
    ) -> Result<Vec<host::EventRow>, StorageError> {
        self.journal(after_id, limit).await
    }
    async fn history(&self, lane: String) -> Result<Vec<host::ItemRow>, StorageError> {
        Ok(self.lane(&lane).history().await?.into_iter().map(Into::into).collect())
    }
    async fn tasks(&self, lane: String, limit: i64) -> Result<Vec<host::TaskRecord>, StorageError> {
        Ok(self.lane(&lane).tasks(None, limit).await?.into_iter().map(Into::into).collect())
    }
    async fn task(&self, task_id: String) -> Result<Option<host::TaskRecord>, StorageError> {
        Ok(self.task(&task_id).await?.map(Into::into))
    }
    async fn list_tasks(
        &self,
        lane: Option<String>,
        status: Option<host::TaskState>,
        order: String,
        page_size: i64,
        page_token: Option<String>,
    ) -> Result<host::TaskPage, StorageError> {
        let order = match order.as_str() {
            "updated" => crate::db::TaskOrder::Updated,
            "inserted" => crate::db::TaskOrder::Inserted,
            _ => return Err(StorageError::Invalid(format!("unknown task order {order:?}"))),
        };
        let query =
            crate::TaskQuery { lane, status, order, page_size, page_token, updated_after: 0 };
        Ok(self.list_tasks(&query).await?.into())
    }
    async fn list_items(
        &self,
        lane: Option<String>,
        after: Option<String>,
        limit: i64,
    ) -> Result<host::ThreadItemPage, StorageError> {
        self.list_items(lane.as_deref(), after.as_deref().filter(|a| !a.is_empty()), limit)
            .await
            .and_then(crate::namespace::thread_item_page)
    }
    async fn lane_info(&self, lane: String) -> Result<Option<host::LaneRow>, StorageError> {
        self.lane(&lane).info().await
    }
    async fn inbox(
        &self,
        lane: String,
        pending: bool,
    ) -> Result<Vec<host::InboxRow>, StorageError> {
        Ok(self.lane(&lane).inbox(pending).await?.into_iter().map(Into::into).collect())
    }
    async fn cancel_task(
        &self,
        lane: String,
        task_id: String,
        metadata: Map<String, Value>,
    ) -> Result<Option<host::TaskRecord>, StorageError> {
        Ok(self.lane(&lane).cancel_task(&task_id, metadata).await?.map(Into::into))
    }
    async fn open_run(
        &self,
        lane: String,
        run_id: String,
        channel: Option<String>,
        wait_timeout: Option<std::time::Duration>,
    ) -> Result<RunHandle, StorageError> {
        let mut options = RunOptions { channel, ..RunOptions::new(run_id.clone(), run_id) };
        if let Some(timeout) = wait_timeout {
            options.wait_timeout = timeout;
        }
        Ok(RunHandle::new(self.lane(&lane).open(options).await?))
    }
    fn close(&self, env: &ProcessEnv) {
        let thread = self.clone();
        match (env.spawner)() {
            Ok(spawner) => spawner.spawn_detached(async move { thread.close().await }),
            Err(error) => log::warn!("the database connection could not close: {error}"),
        }
    }
}

/// The loaded snapshot remains stable; commit and abandon serialize access to the owned run.
pub struct RunHandle {
    run: Arc<tokio::sync::Mutex<Run>>,
    version: i64,
    closed: bool,
    history: Vec<ItemRow>,
    state: Option<Value>,
    context_scopes: Vec<ContextScope>,
}

impl RunHandle {
    fn new(run: Run) -> Self {
        Self {
            version: run.version(),
            closed: run.closed,
            history: run.history.clone(),
            state: run.state.clone(),
            context_scopes: run.context_scopes.clone(),
            run: Arc::new(tokio::sync::Mutex::new(run)),
        }
    }
}

impl host::handles::Run for RunHandle {
    fn version(&self) -> i64 {
        self.version
    }
    fn closed(&self) -> bool {
        self.closed
    }
    fn history(&self) -> Vec<host::ItemRow> {
        self.history.iter().cloned().map(Into::into).collect()
    }
    fn state(&self) -> Option<Value> {
        self.state.clone()
    }
    fn context_scopes(&self) -> Vec<ContextScope> {
        self.context_scopes.clone()
    }
    async fn commit(
        &self,
        history: Vec<host::ItemRow>,
        state: Option<Value>,
        scope_writes: Option<Vec<ContextScopeWrite>>,
    ) -> Result<i64, StorageError> {
        let history = history.into_iter().map(ItemRow::try_from).collect::<Result<_, _>>()?;
        let mut run = self.run.lock().await;
        run.history = history;
        run.scope_writes = scope_writes.unwrap_or_default();
        if state.is_some() {
            run.state = state;
        }
        Ok(run.commit(CommitOptions::default()).await?.version)
    }
    async fn abandon(&self, outcome: String, reason: String) -> Result<(), StorageError> {
        let outcome = RunOutcome::parse(&outcome)
            .ok_or_else(|| StorageError::Invalid(format!("unknown outcome {outcome}")))?;
        self.run.lock().await.abandon(outcome, reason).await
    }
}
