pub mod agentdb;
pub mod sqlite;

use crate::sql::{BatchResult, Row, Statement};
use crate::{StorageError, Thread};
use async_trait::async_trait;
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};

/// One thread database's execution: typed reads and atomic ordered batches. Statements and
/// results are backend-neutral; an `Err` from `batch` never claims a rollback it did not see.
#[async_trait]
pub trait ThreadDbConnection: Send + Sync {
    /// A read's rows; changes nothing.
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError>;
    /// The statements in one transaction, in order, each answered; success means committed.
    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError>;
    /// Releases the connection; repeat-safe.
    async fn close(&self);
}

/// Thread lifecycle and opening a thread, the same contract on every backend.
#[async_trait]
pub trait Storage: Send + Sync {
    /// A new `AT_` thread, returned only once its metadata and database exist.
    async fn create_thread(&self, request: CreateRequest) -> Result<ThreadRecord, StorageError>;
    /// Unknown, deleted or expired threads fail [`StorageError::NotFound`].
    async fn get_thread(&self, thread_id: &str) -> Result<ThreadRecord, StorageError>;
    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError>;
    /// Merges attributes (an empty value deletes the key) and moves expiry: the activity
    /// boundary.
    async fn update_thread(&self, request: UpdateRequest) -> Result<ThreadRecord, StorageError>;
    /// Makes the thread unavailable and reclaims it; repeat-safe.
    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError>;
    /// The shared engine's handle on an existing thread. Opening takes no lease.
    async fn open(&self, thread_id: &str) -> Result<Thread, StorageError>;
    /// Namespace reads use the backend's read interface; the local default borrows a thread.
    async fn get_thread_items(
        &self,
        id: &str,
        partition_key: Option<&str>,
        after: Option<&str>,
        limit: i64,
    ) -> Result<lk_agents_abi::storage::ThreadItemPage, StorageError> {
        with_thread(self, id, async |thread| {
            crate::storage::thread_item_page(thread.list_items(partition_key, after, limit).await?)
        })
        .await
    }
    async fn get_thread_tasks(
        &self,
        id: &str,
        status: Option<crate::TaskState>,
    ) -> Result<Vec<lk_agents_abi::storage::ThreadTask>, StorageError> {
        with_thread(self, id, async |thread| crate::storage::thread_tasks(thread, id, status).await)
            .await
    }
    /// Releases backend resources; repeat-safe.
    async fn close(&self);
}

/// Executes one operation on a borrowed thread and releases its connection on either result.
pub async fn with_thread<T, E: From<StorageError>>(
    storage: &(impl Storage + ?Sized),
    id: &str,
    f: impl AsyncFnOnce(&Thread) -> Result<T, E>,
) -> Result<T, E> {
    let thread = storage.open(id).await?;
    let result = f(&thread).await;
    thread.close().await;
    result
}

#[cfg(test)]
#[path = "backends_tests.rs"]
mod tests;
