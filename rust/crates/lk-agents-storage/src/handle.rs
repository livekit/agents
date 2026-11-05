//! `ctx.thread`: a thread of a namespace, its database opened for each read and released after.

use std::collections::BTreeMap;
use std::sync::Arc;

use lk_agents_abi::storage as host;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::handles::Thread as Reads;

use crate::Thread;
use crate::namespace::Namespace;

pub struct ThreadHandle {
    namespace: Arc<Namespace>,
    thread_id: String,
    holder: String,
    agent: String,
}

impl ThreadHandle {
    /// `read` on the thread's open database, which is released after it, failed or not.
    async fn read<T>(
        &self,
        read: impl AsyncFnOnce(&Thread) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let thread = self.namespace.storage().await?.open(&self.thread_id, &self.holder).await?;
        let out = read(&thread).await;
        thread.close().await;
        out
    }
}

impl host::handles::ThreadHandle for ThreadHandle {
    type Namespace = Namespace;

    fn new(namespace: Arc<Namespace>, thread_id: String, holder: String, agent: String) -> Self {
        Self { namespace, thread_id, holder, agent }
    }
    fn id(&self) -> String {
        self.thread_id.clone()
    }
    fn agent(&self) -> String {
        self.agent.clone()
    }
    fn namespace(&self) -> Arc<Namespace> {
        self.namespace.clone()
    }
    async fn info(&self) -> Result<host::ThreadInfo, StorageError> {
        host::Storage::get_thread(&*self.namespace, self.thread_id.clone()).await
    }
    async fn update(
        &self,
        attributes: Option<BTreeMap<String, String>>,
        idle_ttl_seconds: Option<i64>,
    ) -> Result<host::ThreadInfo, StorageError> {
        let id = self.thread_id.clone();
        host::Storage::update_thread(&*self.namespace, id, attributes, idle_ttl_seconds).await
    }
    async fn list_items(
        &self,
        agent: Option<String>,
        after: Option<String>,
        limit: i64,
    ) -> Result<host::ThreadItemPage, StorageError> {
        self.read(async |t| Reads::list_items(t, agent, after, limit).await).await
    }
    async fn list_tasks(
        &self,
        agent: Option<String>,
        status: Option<host::TaskState>,
        order: String,
        page_token: Option<String>,
        limit: i64,
    ) -> Result<host::TaskPage, StorageError> {
        self.read(async |t| Reads::list_tasks(t, agent, status, order, limit, page_token).await)
            .await
    }
    async fn task(&self, task_id: String) -> Result<Option<host::TaskRecord>, StorageError> {
        self.read(async |t| Reads::task(t, task_id).await).await
    }
    async fn journal(
        &self,
        after_id: i64,
        limit: i64,
    ) -> Result<Vec<host::EventRow>, StorageError> {
        self.read(async |t| Reads::journal(t, after_id, limit).await).await
    }
    async fn history(&self, agent: String) -> Result<Vec<host::ItemRow>, StorageError> {
        self.read(async |t| Reads::history(t, agent).await).await
    }
    async fn tasks(
        &self,
        agent: String,
        status: Option<host::TaskState>,
        limit: i64,
    ) -> Result<Vec<host::TaskRecord>, StorageError> {
        match status {
            None => self.read(async |t| Reads::tasks(t, agent, limit).await).await,
            Some(status) => {
                let page = self
                    .read(async |t| {
                        let (agent, order) = (Some(agent), "updated".to_owned());
                        Reads::list_tasks(t, agent, Some(status), order, limit, None).await
                    })
                    .await?;
                Ok(page.tasks)
            }
        }
    }
}
