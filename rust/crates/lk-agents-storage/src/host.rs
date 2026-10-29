//! The host's `Storage` class (`lk_agents_abi::storage::Storage`): one backend, chosen by its
//! configuration and opened on first use, its records converted to the ABI's; and the
//! storage-owned thread rows as the host reads them.

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
use serde_json::{Map, Value};
use tokio::sync::OnceCell;
use tokio_util::task::TaskTracker;

use crate::agentdb::{AgentDbOptions, AgentDbStorage, TokenMinter, service_error};
use crate::context::store::ContextStore;
use crate::db::{ItemPage, TaskPage, TaskQuery};
use crate::engine::LeaseTiming;
use crate::sqlite::SqliteStorage;
use crate::voice::VoiceStore;
use crate::{QueuedStreams, Storage, Thread};
use lk_agents_abi::storage::context::ContextScope;

/// What the lazily opened backend is built from.
enum Config {
    Memory { lease: LeaseTiming },
    Local { directory: PathBuf, lease: LeaseTiming },
    Cloud(AgentDbStorage),
}

enum Backend {
    Local(Arc<SqliteStorage>),
    Cloud(Arc<AgentDbStorage>),
}

impl Backend {
    fn storage(&self) -> &dyn Storage {
        match self {
            Backend::Local(s) => s.as_ref(),
            Backend::Cloud(s) => s.as_ref(),
        }
    }

    fn shared_storage(&self) -> Arc<dyn Storage> {
        match self {
            Backend::Local(s) => s.clone(),
            Backend::Cloud(s) => s.clone(),
        }
    }
}

/// A storage namespace: a local directory, or the Cloud project of a key pair.
pub struct HostStorage {
    config: Config,
    backend: OnceCell<Backend>,
    streams: OnceCell<Arc<QueuedStreams>>,
    a2a_tasks: TaskTracker,
}

/// The holder of a read-only handle: it never takes a lease.
const READER: &str = "reader";
/// A local items page when the caller names no limit.
const LOCAL_PAGE: i64 = 100;

impl HostStorage {
    async fn backend(&self) -> Result<&Backend, StorageError> {
        self.backend
            .get_or_try_init(|| async {
                Ok(match &self.config {
                    Config::Memory { lease } => {
                        Backend::Local(Arc::new(SqliteStorage::in_memory(*lease).await?))
                    }
                    Config::Local { directory, lease } => {
                        let storage = SqliteStorage::open(directory, *lease).await?;
                        log::info!("local thread storage in {}", directory.display());
                        Backend::Local(Arc::new(storage))
                    }
                    Config::Cloud(storage) => Backend::Cloud(Arc::new(storage.clone())),
                })
            })
            .await
    }

    async fn storage(&self) -> Result<&dyn Storage, StorageError> {
        Ok(self.backend().await?.storage())
    }

    /// The namespace's backend, opened on first use.
    pub async fn shared_storage(&self) -> Result<Arc<dyn Storage>, StorageError> {
        Ok(self.backend().await?.shared_storage())
    }

    /// The A2A work accepted over this namespace, which [`aclose`](HostStorageClass::aclose)
    /// settles before closing.
    pub fn a2a_tasks(&self) -> TaskTracker {
        self.a2a_tasks.clone()
    }
}

impl host::StorageNamespace for HostStorage {}

fn seconds(value: Option<f64>, default: Duration, name: &str) -> Result<Duration, StorageError> {
    match value {
        Some(s) => Duration::try_from_secs_f64(s)
            .map_err(|e| StorageError::Invalid(format!("{name}: {e}"))),
        None => Ok(default),
    }
}

fn attributes(
    v: Option<Map<String, Value>>,
) -> Result<std::collections::HashMap<String, String>, StorageError> {
    v.unwrap_or_default()
        .into_iter()
        .map(|(k, v)| match v {
            Value::String(s) => Ok((k, s)),
            _ => Err(StorageError::Invalid(format!("attribute {k} is not a string"))),
        })
        .collect()
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
        attributes: t.attributes.into_iter().map(|(k, v)| (k, Value::String(v))).collect(),
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
    let status = TaskState::parse(&t.status)
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
                status: row.status.into(),
                updated_at: row.updated_at / 1000,
            });
        }
        if page.next_page_token.is_empty() {
            return Ok(tasks);
        }
        page_token = Some(page.next_page_token);
    }
}

impl HostStorageClass for HostStorage {
    type Thread = Thread;
    type Streams = Arc<QueuedStreams>;
    type ContextActivation = ContextStore;
    type VoiceStore = VoiceStore;
    type Env = ProcessEnv;

    async fn _core_voice_store(
        &self,
        thread_id: String,
        lane: String,
        initial_history: Option<Vec<lk_agents_abi::ChatItem>>,
    ) -> Result<Self::VoiceStore, StorageError> {
        VoiceStore::bound(self.shared_storage().await?.clone(), thread_id, lane, initial_history)
            .await
    }

    fn new(
        env: &ProcessEnv,
        directory: Option<String>,
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
        let config = match (directory, url, api_key, api_secret) {
            (None, None, None, None) => Config::Memory { lease },
            (Some(directory), None, None, None) => {
                let directory = std::path::absolute(&directory)
                    .map_err(|e| StorageError::Invalid(format!("storage directory: {e}")))?;
                Config::Local { directory, lease }
            }
            (None, Some(url), Some(key), Some(secret)) => {
                let mut opts = AgentDbOptions::new(&url);
                opts.lease = lease;
                {
                    let runtime = (env.runtime)()
                        .map_err(|e| StorageError::Unavailable(format!("storage runtime: {e}")))?;
                    Config::Cloud(AgentDbStorage::new(
                        opts,
                        TokenMinter::new(key, secret)?,
                        runtime,
                    )?)
                }
            }
            _ => {
                return Err(StorageError::Invalid(
                    "storage is a local directory, or a project's url, api_key and api_secret"
                        .into(),
                ));
            }
        };
        Ok(Self {
            config,
            backend: OnceCell::new(),
            streams: OnceCell::new(),
            a2a_tasks: TaskTracker::new(),
        })
    }

    fn location(&self) -> String {
        match &self.config {
            Config::Memory { .. } => ":memory:".into(),
            Config::Local { directory, .. } => directory.display().to_string(),
            Config::Cloud(storage) => storage.options().url.clone(),
        }
    }

    async fn create_thread(
        &self,
        scope: Option<String>,
        attributes: Option<Map<String, Value>>,
        idle_ttl_seconds: i64,
    ) -> Result<ThreadInfo, StorageError> {
        let attributes = self::attributes(attributes)?;
        let request = CreateRequest {
            scope: scope.unwrap_or_default(),
            attributes,
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
        attributes: Option<Map<String, Value>>,
        idle_ttl_seconds: Option<i64>,
    ) -> Result<ThreadInfo, StorageError> {
        let attributes = self::attributes(attributes)?;
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
                let status = status.map(|status| status.as_str().to_owned());
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
        ContextStore::bound(self.backend().await?.shared_storage(), thread_id, lane).await
    }

    async fn _core_turn_store(
        &self,
        thread_id: String,
        lane: String,
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
        read_only: bool,
    ) -> Result<Self::ContextActivation, StorageError> {
        let storage = self.backend().await?.shared_storage();
        ContextStore::turn(storage, thread_id, lane, scopes, routing_state, read_only).await
    }

    async fn streams(&self) -> Result<Arc<QueuedStreams>, StorageError> {
        let storage = self.storage().await?;
        Ok(self
            .streams
            .get_or_init(|| async { Arc::new(QueuedStreams::new(storage.response_streams())) })
            .await
            .clone())
    }

    async fn aclose(&self) {
        self.a2a_tasks.close();
        if tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, self.a2a_tasks.wait())
            .await
            .is_err()
        {
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
