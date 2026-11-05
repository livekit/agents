//! The host's `Storage` class (`lk_agents_abi::storage::Storage`): one backend, chosen by its
//! configuration and opened on first use, its records converted to the ABI's; and the
//! storage-owned thread rows as the host reads them.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use livekit_protocol::agent::agent_task::{GetTasksRequest, Task};
use livekit_protocol::agent::agent_thread::{
    CreateRequest, GetItemsRequest, Item, ListRequest, ListResponse, Thread as ThreadRecord,
    UpdateRequest,
};
use lk_agents_abi::storage as host;
use lk_agents_abi::storage::{
    Storage as HostStorageClass, StorageError, TaskState, ThreadInfo, ThreadItem, ThreadItemPage,
    ThreadPage, ThreadTask,
};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TaskSet;
use serde_json::Value;
use tokio::sync::OnceCell;

use crate::backends::agentdb::{AgentDbOptions, AgentDbStorage, TokenMinter, service_error};
use crate::backends::sqlite::SqliteStorage;
use crate::context::store::ContextStore;
use crate::thread::LeaseTiming;
use crate::thread::db::{ItemPage, TaskPage, TaskQuery};
use crate::voice::VoiceStore;
use crate::{CommitOptions, ItemRow, Run, RunOptions, RunOutcome, Storage, Thread};
use lk_agents_abi::SpawnerExt;
use lk_agents_abi::storage::context::ContextScope;
use lk_agents_abi::storage::context::ContextScopeWrite;
use serde_json::Map;

/// What the lazily opened backend is built from.
enum Config {
    Local { directory: PathBuf, lease: LeaseTiming, project_id: String },
    Cloud { opts: AgentDbOptions, minter: TokenMinter, env: ProcessEnv },
}

/// The project a local store without options belongs to.
const DEFAULT_PROJECT_ID: &str = "default";
/// The bare project-id prefix, refused as an id of its own.
const PROJECT_ID_PREFIX: &str = "p_";
/// The longest project id accepted.
const MAX_PROJECT_ID_LEN: usize = 128;

impl Config {
    fn local(
        options: Option<host::LocalStorageOptions>,
        lease: LeaseTiming,
    ) -> Result<Self, StorageError> {
        let host::LocalStorageOptions { directory, project_id } =
            options.unwrap_or(host::LocalStorageOptions {
                directory: None,
                project_id: DEFAULT_PROJECT_ID.into(),
            });
        check_project_id(&project_id)?;
        let directory = match directory {
            Some(directory) => PathBuf::from(directory),
            None => default_directory(&project_id)?,
        };
        let directory = std::path::absolute(&directory)
            .map_err(|e| StorageError::Invalid(format!("storage directory: {e}")))?;
        Ok(Config::Local { directory, lease, project_id })
    }

    fn cloud(
        url: &str,
        key: String,
        secret: String,
        lease: LeaseTiming,
        env: &ProcessEnv,
    ) -> Result<Self, StorageError> {
        let mut opts = AgentDbOptions::new(url);
        opts.lease = lease;
        if !opts.url.starts_with("http://") && !opts.url.starts_with("https://") {
            return Err(StorageError::Invalid(format!(
                "agentstore url {} is not http(s)",
                opts.url
            )));
        }
        Ok(Config::Cloud { opts, minter: TokenMinter::new(key, secret)?, env: env.clone() })
    }
}

fn check_project_id(project_id: &str) -> Result<(), StorageError> {
    let allowed = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'-';
    if project_id.is_empty()
        || project_id == PROJECT_ID_PREFIX
        || project_id.len() > MAX_PROJECT_ID_LEN
        || !project_id.bytes().all(allowed)
    {
        return Err(StorageError::Invalid(
            "project_id must contain letters, digits, - or _".into(),
        ));
    }
    Ok(())
}

/// `~/.livekit/agent-threads/<project_id>`.
fn default_directory(project_id: &str) -> Result<PathBuf, StorageError> {
    let home = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    let home = std::env::var_os(home).filter(|value| !value.is_empty()).ok_or_else(|| {
        StorageError::Invalid("home directory is unavailable; configure directory=".into())
    })?;
    Ok(PathBuf::from(home).join(".livekit").join("agent-threads").join(project_id))
}

enum Backend {
    Local(Arc<SqliteStorage>),
    Cloud(Arc<AgentDbStorage>),
}

impl Backend {
    fn storage(&self) -> Arc<dyn Storage> {
        match self {
            Backend::Local(s) => s.clone(),
            Backend::Cloud(s) => s.clone(),
        }
    }
}

/// A storage namespace: a local directory, or the Cloud project of a key pair.
pub struct Namespace {
    config: Config,
    backend: OnceCell<Backend>,
    a2a_tasks: Arc<TaskSet>,
    clock: Arc<dyn lk_agents_abi::Clock>,
}

/// The holder of a read-only handle: it never takes a lease.
const READER: &str = "reader";
/// A local items page when the caller names no limit.
const LOCAL_PAGE: i64 = 100;

impl Namespace {
    async fn backend(&self) -> Result<&Backend, StorageError> {
        self.backend
            .get_or_try_init(|| async {
                Ok(match &self.config {
                    Config::Local { directory, lease, .. } => {
                        let storage = SqliteStorage::open(directory, *lease).await?;
                        log::info!("local thread storage in {}", directory.display());
                        Backend::Local(Arc::new(storage))
                    }
                    Config::Cloud { opts, minter, env } => {
                        let (opts, minter, env) = (opts.clone(), minter.clone(), env.clone());
                        let runtime = (env.runtime)().map_err(|e| {
                            StorageError::Unavailable(format!("storage runtime: {e}"))
                        })?;
                        // The API client loads the system trust store. Keep that blocking
                        // work off the request executor as well as the module import path.
                        let storage = lk_agents_net::blocking_io::blocking_io()
                            .run((), move || {
                                let _guard = runtime.enter();
                                AgentDbStorage::new(opts, minter, &env)
                            })
                            .await
                            .map_err(|e| StorageError::Unavailable(e.to_string()))??;
                        Backend::Cloud(Arc::new(storage))
                    }
                })
            })
            .await
    }

    /// The namespace's backend, opened on first use.
    pub async fn storage(&self) -> Result<Arc<dyn Storage>, StorageError> {
        Ok(self.backend().await?.storage())
    }

    /// The A2A work accepted over this namespace, which [`aclose`](HostStorageClass::aclose)
    /// settles before closing.
    pub fn a2a_tasks(&self) -> Arc<TaskSet> {
        Arc::clone(&self.a2a_tasks)
    }
}

impl host::StorageNamespace for Namespace {}

fn seconds(value: Option<f64>, default: Duration, name: &str) -> Result<Duration, StorageError> {
    match value {
        Some(s) => Duration::try_from_secs_f64(s)
            .map_err(|e| StorageError::Invalid(format!("{name}: {e}"))),
        None => Ok(default),
    }
}

/// Whole seconds for the wire; `0` sends none, the default.
fn ttl(seconds: i64) -> Option<pbjson_types::Duration> {
    (seconds > 0).then_some(pbjson_types::Duration { seconds, nanos: 0 })
}

/// agent-db's page size: `0` lets the server choose.
fn page_limit(limit: Option<u32>) -> Result<i32, StorageError> {
    limit.map_or(Ok(0), |l| i32::try_from(l).map_err(|_| StorageError::Invalid("limit".into())))
}

fn seconds_of(t: Option<pbjson_types::Timestamp>) -> i64 {
    t.map_or(0, |t| t.seconds)
}

fn thread_info(t: ThreadRecord) -> ThreadInfo {
    ThreadInfo {
        thread_id: t.thread_id,
        scope: t.scope,
        attributes: t.attributes.into_iter().collect(),
        created_at: seconds_of(t.created_at),
        last_active: seconds_of(t.last_active),
        expires_at: seconds_of(t.expires_at),
        idle_ttl_seconds: t.idle_ttl.map_or(0, |d| d.seconds),
        database_id: t.database_id,
    }
}

fn thread_page(r: ListResponse) -> ThreadPage {
    ThreadPage {
        threads: r.threads.into_iter().map(thread_info).collect(),
        before: r.next_page_token,
    }
}

fn json<T: serde::Serialize>(v: &T) -> Result<Value, StorageError> {
    serde_json::to_value(v).map_err(|e| StorageError::Invalid(e.to_string()))
}

/// A thread handle's item page as the host record: the same shape Cloud's `GetThreadItems`
/// answers.
pub fn thread_item_page(page: ItemPage) -> Result<ThreadItemPage, StorageError> {
    let items = page
        .items
        .into_iter()
        .map(|p| {
            Ok(ThreadItem {
                agent: p.lane,
                seq: p.seq,
                message_id: p.item.message_id.unwrap_or_default(),
                item: json(&p.item.payload)?,
            })
        })
        .collect::<Result<_, StorageError>>()?;
    Ok(ThreadItemPage { items, after: page.next.unwrap_or_default() })
}

fn cloud_item(i: Item) -> Result<ThreadItem, StorageError> {
    Ok(ThreadItem {
        agent: i.agent,
        seq: i.seq,
        message_id: i.message_id,
        item: json(&i.item.unwrap_or_default())?,
    })
}

/// Preserve the framework's A2A state and the presence of an application label.
fn cloud_task(t: Task) -> Result<ThreadTask, StorageError> {
    let status = serde_json::from_value::<crate::http::generated::TaskStatus>(Value::String(
        t.status.clone(),
    ))
    .ok()
    .and_then(crate::http::task_filter)
    .ok_or_else(|| StorageError::Unavailable(format!("unknown task state {}", t.status)))?;
    Ok(ThreadTask {
        task_id: t.task_id,
        thread_id: t.thread_id,
        name: t.name,
        status,
        updated_at: seconds_of(t.updated_at),
    })
}

/// Every task of a local thread, of `status` when given.
async fn local_tasks(
    thread: &Thread,
    thread_id: &str,
    status: Option<TaskState>,
) -> Result<Vec<ThreadTask>, StorageError> {
    let mut tasks = Vec::new();
    let mut page_token = None;
    loop {
        // the host's state names the row state of the same spelling
        let status = status.and_then(|s| crate::TaskState::parse(s.as_str()));
        let query = TaskQuery { status, page_size: 100, page_token, ..TaskQuery::default() };
        let page = thread.list_tasks(&query).await?;
        for row in page.tasks {
            tasks.push(ThreadTask {
                task_id: row.task_id,
                thread_id: thread_id.to_owned(),
                name: row.name,
                status: row.status,
                updated_at: row.updated_at / 1000,
            });
        }
        if page.next_page_token.is_empty() {
            return Ok(tasks);
        }
        page_token = Some(page.next_page_token);
    }
}

impl HostStorageClass for Namespace {
    type Thread = Thread;
    type ContextActivation = ContextStore;
    type VoiceStore = VoiceStore;
    type ThreadServer = crate::http::ThreadServer;
    type Env = ProcessEnv;

    async fn _core_serve_threads(
        &self,
        api_key: String,
        api_secret: String,
        port: u16,
    ) -> Result<Self::ThreadServer, StorageError> {
        if !matches!(self.config, Config::Local { .. }) {
            return Err(StorageError::Invalid(
                "the local ThreadService requires SQLiteStorage".into(),
            ));
        }
        crate::http::ThreadServer::start(
            self.storage().await?,
            self.project_id(),
            api_key,
            api_secret,
            port,
        )
        .await
    }

    async fn _core_voice_store(
        &self,
        thread_id: String,
        lane: String,
        initial_history: Option<Vec<lk_agents_abi::ChatItem>>,
    ) -> Result<Self::VoiceStore, StorageError> {
        VoiceStore::bound(self.storage().await?, thread_id, lane, initial_history).await
    }

    fn new(
        env: &ProcessEnv,
        local: Option<host::LocalStorageOptions>,
        url: Option<String>,
        api_key: Option<String>,
        api_secret: Option<String>,
        lease_ttl: Option<f64>,
        renew_interval: Option<f64>,
    ) -> Result<Self, StorageError> {
        let default = LeaseTiming::default();
        let lease = LeaseTiming::new(
            seconds(lease_ttl, default.ttl, "lease_ttl")?,
            seconds(renew_interval, default.renew_interval, "renew_interval")?,
        )?;
        let config = match (local, url, api_key, api_secret) {
            (local, None, None, None) => Config::local(local, lease)?,
            (None, Some(url), Some(key), Some(secret)) => {
                Config::cloud(&url, key, secret, lease, env)?
            }
            _ => {
                return Err(StorageError::Invalid(
                    "storage is a local directory, or a project's url, api_key and api_secret"
                        .into(),
                ));
            }
        };
        let (backend, a2a_tasks, clock) = (OnceCell::new(), Arc::default(), env.clock.clone());
        Ok(Self { config, backend, a2a_tasks, clock })
    }

    fn location(&self) -> String {
        match &self.config {
            Config::Local { directory, .. } => directory.display().to_string(),
            Config::Cloud { opts, .. } => opts.url.clone(),
        }
    }

    fn project_id(&self) -> String {
        match &self.config {
            Config::Local { project_id, .. } => project_id.clone(),
            Config::Cloud { .. } => String::new(),
        }
    }

    async fn create_thread(
        &self,
        scope: Option<String>,
        attributes: Option<BTreeMap<String, String>>,
        idle_ttl_seconds: i64,
    ) -> Result<ThreadInfo, StorageError> {
        let request = CreateRequest {
            scope: scope.unwrap_or_default(),
            attributes: attributes.unwrap_or_default().into_iter().collect(),
            idle_ttl: ttl(idle_ttl_seconds),
        };
        Ok(thread_info(self.storage().await?.create_thread(request).await?))
    }

    async fn get_thread(&self, thread_id: String) -> Result<ThreadInfo, StorageError> {
        Ok(thread_info(self.storage().await?.get_thread(&thread_id).await?))
    }

    async fn list_threads(
        &self,
        scope: String,
        before: Option<String>,
        limit: Option<u32>,
    ) -> Result<ThreadPage, StorageError> {
        let request = ListRequest {
            scope,
            page_token: before.unwrap_or_default(),
            limit: page_limit(limit)?,
        };
        Ok(thread_page(self.storage().await?.list_threads(request).await?))
    }

    async fn update_thread(
        &self,
        thread_id: String,
        attributes: Option<BTreeMap<String, String>>,
        idle_ttl_seconds: Option<i64>,
    ) -> Result<ThreadInfo, StorageError> {
        let attributes = attributes.unwrap_or_default().into_iter().collect();
        let idle_ttl = idle_ttl_seconds.map(|s| pbjson_types::Duration { seconds: s, nanos: 0 });
        let request = UpdateRequest { thread_id, attributes, idle_ttl };
        Ok(thread_info(self.storage().await?.update_thread(request).await?))
    }

    async fn delete_thread(&self, thread_id: String) -> Result<(), StorageError> {
        self.storage().await?.delete_thread(&thread_id).await
    }

    async fn get_thread_items(
        &self,
        thread_id: String,
        agent: Option<String>,
        after: Option<String>,
        limit: Option<u32>,
    ) -> Result<ThreadItemPage, StorageError> {
        match self.backend().await? {
            Backend::Cloud(cloud) => {
                let request = GetItemsRequest {
                    thread_id,
                    agent: agent.unwrap_or_default(),
                    page_token: after.unwrap_or_default(),
                    limit: page_limit(limit)?,
                    ..Default::default()
                };
                let page = cloud.threads().items(request).await.map_err(service_error)?;
                let items = page.items.into_iter().map(cloud_item).collect::<Result<_, _>>()?;
                Ok(ThreadItemPage { items, after: page.next_page_token })
            }
            Backend::Local(local) => {
                let thread = local.open(&thread_id, READER).await?;
                let limit = limit.map_or(LOCAL_PAGE, i64::from);
                let page = thread.list_items(agent.as_deref(), after.as_deref(), limit).await;
                thread.close().await;
                thread_item_page(page?)
            }
        }
    }

    async fn get_thread_tasks(
        &self,
        thread_id: String,
        status: Option<TaskState>,
    ) -> Result<Vec<ThreadTask>, StorageError> {
        match self.backend().await? {
            Backend::Cloud(cloud) => {
                let status =
                    status.map(|status| crate::http::task_status(status).as_str().to_owned());
                let mut tasks = Vec::new();
                let mut page_token = String::new();
                loop {
                    let request = GetTasksRequest {
                        thread_id: thread_id.clone(),
                        status: status.clone(),
                        page_token,
                        ..Default::default()
                    };
                    let page = cloud.threads().tasks(request).await.map_err(service_error)?;
                    tasks.extend(
                        page.tasks.into_iter().map(cloud_task).collect::<Result<Vec<_>, _>>()?,
                    );
                    if page.next_page_token.is_empty() {
                        return Ok(tasks);
                    }
                    page_token = page.next_page_token;
                }
            }
            Backend::Local(local) => {
                let thread = local.open(&thread_id, READER).await?;
                let tasks = local_tasks(&thread, &thread_id, status).await;
                thread.close().await;
                tasks
            }
        }
    }

    async fn open(&self, thread_id: String, holder: String) -> Result<Thread, StorageError> {
        self.storage().await?.open(&thread_id, &holder).await
    }

    async fn _core_context_store(
        &self,
        thread_id: String,
        lane: String,
    ) -> Result<Self::ContextActivation, StorageError> {
        ContextStore::bound(self.storage().await?, thread_id, lane).await
    }

    async fn _core_turn_store(
        &self,
        thread_id: String,
        lane: String,
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
        read_only: bool,
    ) -> Result<Self::ContextActivation, StorageError> {
        let storage = self.storage().await?;
        ContextStore::turn(storage, thread_id, lane, scopes, routing_state, read_only).await
    }

    async fn aclose(&self) {
        let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        if !self.a2a_tasks.wait_empty(&*self.clock, Some(deadline)).await {
            log::error!(target: "livekit.agents", "storage is closing while accepted A2A work settles");
        }

        if let Some(backend) = self.backend.get() {
            backend.storage().close().await;
        }
    }
}

impl From<TaskPage> for host::TaskPage {
    fn from(page: TaskPage) -> Self {
        host::TaskPage {
            tasks: page.tasks.into_iter().map(Into::into).collect(),
            next_page_token: page.next_page_token,
            page_size: page.page_size,
            total_size: page.total_size,
        }
    }
}

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
            "updated" => crate::thread::db::TaskOrder::Updated,
            "inserted" => crate::thread::db::TaskOrder::Inserted,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn cloud_storage_starts_nothing_until_first_use_and_shares_its_backend() {
        static RUNTIME_READS: AtomicUsize = AtomicUsize::new(0);
        fn runtime() -> Result<tokio::runtime::Handle, String> {
            RUNTIME_READS.fetch_add(1, Ordering::Relaxed);
            tokio::runtime::Handle::try_current().map_err(|e| e.to_string())
        }
        let mut env = ProcessEnv::current_runtime();
        env.runtime = runtime;
        let storage = Namespace::new(
            &env,
            None,
            Some("wss://project.livekit.cloud".into()),
            Some("key".into()),
            Some("secret".into()),
            None,
            None,
        )
        .unwrap();
        assert!(!storage.location().is_empty());
        storage.aclose().await;
        assert_eq!(RUNTIME_READS.load(Ordering::Relaxed), 0);
        assert!(storage.backend.get().is_none());

        let (first, second) = tokio::join!(storage.storage(), storage.storage());
        let (first, second) = (first.unwrap(), second.unwrap());
        assert!(Arc::ptr_eq(&first, &second));
        let reads = RUNTIME_READS.load(Ordering::Relaxed);
        assert!(reads > 0);
        assert!(Arc::ptr_eq(&first, &storage.storage().await.unwrap()));
        assert_eq!(RUNTIME_READS.load(Ordering::Relaxed), reads);
        storage.aclose().await;
    }
}
