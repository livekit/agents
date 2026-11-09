use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};
use lk_agents_abi::storage as host;
use lk_agents_abi::storage::handles::Thread as Reads;
use lk_agents_abi::storage::{
    Storage as HostStorageClass, StorageError, TaskState, ThreadInfo, ThreadItem, ThreadItemPage,
    ThreadPage, ThreadTask,
};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TaskSet;
use serde_json::Value;
use tokio::sync::OnceCell;

use crate::backends::agentdb::{AgentDbOptions, AgentDbStorage, TokenMinter};
use crate::backends::sqlite::SqliteStorage;
use crate::backends::with_thread;
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
pub const DEFAULT_PROJECT_ID: &str = "default";
/// Where `SQLiteStorage()` keeps its project directories instead of `~/.livekit/agent-threads`.
const THREADS_DIR_ENV: &str = "LIVEKIT_AGENT_THREADS_DIR";
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

/// `$LIVEKIT_AGENT_THREADS_DIR/<project_id>`, else `~/.livekit/agent-threads/<project_id>`.
fn default_directory(project_id: &str) -> Result<PathBuf, StorageError> {
    if let Some(threads) = std::env::var_os(THREADS_DIR_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(threads).join(project_id));
    }
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
    /// The lease holder id this process opens threads under.
    holder: String,
}

/// A local items page when the caller names no limit.
const LOCAL_PAGE: i64 = 100;

impl Namespace {
    async fn backend(&self) -> Result<&Backend, StorageError> {
        self.backend
            .get_or_try_init(|| async {
                Ok(match &self.config {
                    Config::Local { directory, lease, .. } => {
                        let storage = SqliteStorage::open(directory, *lease).await?;
                        log::info!(target: lk_agents_abi::LOG_TARGET, "local thread storage in {}", directory.display());
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
                            .try_run((), move || {
                                let _guard = runtime.enter();
                                AgentDbStorage::new(opts, minter, &env)
                            })
                            .await?;
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

/// `value`, else the environment variable `name` as float seconds.
fn env_seconds(value: Option<Duration>, name: &str) -> Result<Option<Duration>, StorageError> {
    if value.is_some() {
        return Ok(value);
    }
    match std::env::var(name) {
        Ok(raw) if !raw.is_empty() => {
            let invalid =
                |e: &dyn std::fmt::Display| StorageError::Invalid(format!("{name}={raw:?}: {e}"));
            let seconds: f64 = raw.parse().map_err(|e| invalid(&e))?;
            Duration::try_from_secs_f64(seconds).map(Some).map_err(|e| invalid(&e))
        }
        _ => Ok(None),
    }
}

/// The `thread_id` a dispatch's metadata names, if any.
fn call_thread_id(metadata: &str) -> Option<String> {
    let metadata: serde_json::Value = serde_json::from_str(metadata).ok()?;
    let id = metadata.get("thread_id")?.as_str()?;
    (!id.is_empty()).then(|| id.to_owned())
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
                item: chat_item(p.item.payload)?,
            })
        })
        .collect::<Result<_, StorageError>>()?;
    Ok(ThreadItemPage { items, after: page.next.unwrap_or_default() })
}

fn chat_item(
    item: lk_agents_abi::livekit_protocol::agent::chat_context::ChatItem,
) -> Result<lk_agents_abi::ChatItem, StorageError> {
    lk_agents_abi::llm::chat_context::protocol::decode_chat_item(item).map_err(StorageError::from)
}

/// Every task of a thread, of `status` when given.
pub(crate) async fn thread_tasks(
    thread: &Thread,
    thread_id: &str,
    status: Option<TaskState>,
) -> Result<Vec<ThreadTask>, StorageError> {
    let mut tasks = Vec::new();
    let mut page_token = None;
    loop {
        // the host's state names the row state of the same spelling
        let status = status.and_then(|s| crate::TaskState::from_wire(s.as_str()));
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
        lease_ttl: Option<Duration>,
        renew_interval: Option<Duration>,
        holder: Option<String>,
    ) -> Result<Self, StorageError> {
        let default = LeaseTiming::default();
        let lease = |ttl: Option<Duration>, renew_interval: Option<Duration>| {
            LeaseTiming::new(
                ttl.unwrap_or(default.ttl),
                renew_interval.unwrap_or(default.renew_interval),
            )
        };
        let config = match local {
            Some(local) => {
                if url.is_some() || api_key.is_some() || api_secret.is_some() {
                    return Err(StorageError::Invalid(
                        "local storage takes no project url or key pair".into(),
                    ));
                }
                Config::local(Some(local), lease(lease_ttl, renew_interval)?)?
            }
            None => {
                let setting = |value: Option<String>, name: &str| {
                    value.or_else(|| std::env::var(name).ok()).filter(|value| !value.is_empty())
                };
                let (Some(url), Some(key), Some(secret)) = (
                    setting(url, "LIVEKIT_URL"),
                    setting(api_key, "LIVEKIT_API_KEY"),
                    setting(api_secret, "LIVEKIT_API_SECRET"),
                ) else {
                    return Err(StorageError::Invalid(
                        "AgentDBStorage needs the project's LIVEKIT_URL, LIVEKIT_API_KEY and \
                         LIVEKIT_API_SECRET"
                            .into(),
                    ));
                };
                let lease = lease(
                    env_seconds(lease_ttl, "LIVEKIT_AGENTDB_LEASE_TTL")?,
                    env_seconds(renew_interval, "LIVEKIT_AGENTDB_RENEW_INTERVAL")?,
                )?;
                Config::cloud(&url, key, secret, lease, env)?
            }
        };
        let (backend, a2a_tasks, clock) = (OnceCell::new(), Arc::default(), env.clock.clone());
        let holder = holder.filter(|h| !h.is_empty()).unwrap_or_else(|| env.lease_holder());
        Ok(Self { config, backend, a2a_tasks, clock, holder })
    }

    fn holder(&self) -> String {
        self.holder.clone()
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

    async fn _core_call_thread(&self, metadata: String) -> Result<ThreadInfo, StorageError> {
        match call_thread_id(&metadata) {
            Some(thread_id) => self.get_thread(thread_id).await,
            None => self.create_thread(None, None, 0).await,
        }
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
        self.storage()
            .await?
            .get_thread_items(
                &thread_id,
                agent.as_deref(),
                after.as_deref(),
                limit.map_or(LOCAL_PAGE, i64::from),
            )
            .await
    }

    async fn get_thread_tasks(
        &self,
        thread_id: String,
        status: Option<TaskState>,
    ) -> Result<Vec<ThreadTask>, StorageError> {
        self.storage().await?.get_thread_tasks(&thread_id, status).await
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
            log::error!(target: lk_agents_abi::LOG_TARGET, "storage is closing while accepted A2A work settles");
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
        self.lane(&lane).history().await?.into_iter().map(TryInto::try_into).collect()
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
        order: lk_agents_abi::storage::thread::TaskOrder,
        page_size: i64,
        page_token: Option<String>,
    ) -> Result<host::TaskPage, StorageError> {
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
            .and_then(crate::storage::thread_item_page)
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
            Err(error) => {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "the database connection could not close: {error}")
            }
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
            closed: run.closed(),
            history: run.history().to_vec(),
            state: run.state().cloned(),
            context_scopes: run.context_scopes().to_vec(),
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
    fn history(&self) -> Result<Vec<host::ItemRow>, StorageError> {
        self.history.iter().cloned().map(TryInto::try_into).collect()
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
        let history = history.into_iter().map(ItemRow::from).collect();
        let mut run = self.run.lock().await;
        *run.history_mut() = history;
        run.stage_scopes(scope_writes.unwrap_or_default());
        if let Some(state) = state {
            run.set_state(state);
        }
        Ok(run.commit(CommitOptions::default()).await?.version)
    }
    async fn abandon(&self, outcome: RunOutcome, reason: String) -> Result<(), StorageError> {
        self.run.lock().await.abandon(outcome, reason).await
    }
}

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
        let storage = self.namespace.storage().await?;
        with_thread(&*storage, &self.thread_id, &self.holder, read).await
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
        order: lk_agents_abi::storage::thread::TaskOrder,
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
                        let (agent, order) =
                            (Some(agent), lk_agents_abi::storage::thread::TaskOrder::Updated);
                        Reads::list_tasks(t, agent, Some(status), order, limit, None).await
                    })
                    .await?;
                Ok(page.tasks)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Catches: an empty or non-string id taken as named, or malformed metadata failing the call.
    #[test]
    fn only_a_nonempty_string_thread_id_names_the_calls_thread() {
        assert_eq!(call_thread_id(r#"{"thread_id": "AT_1"}"#).as_deref(), Some("AT_1"));
        for metadata in ["", "not json", "[1]", r#"{"thread_id": ""}"#, r#"{"thread_id": 1}"#, "{}"]
        {
            assert_eq!(call_thread_id(metadata), None, "{metadata}");
        }
    }

    #[tokio::test]
    async fn cloud_storage_starts_nothing_until_first_use_and_shares_its_backend() {
        static RUNTIME_READS: AtomicUsize = AtomicUsize::new(0);
        fn runtime() -> Result<tokio::runtime::Handle, lk_agents_net::env::ProcessError> {
            RUNTIME_READS.fetch_add(1, Ordering::Relaxed);
            tokio::runtime::Handle::try_current()
                .map_err(|e| lk_agents_net::env::ProcessError::NoRuntime(e.to_string()))
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
