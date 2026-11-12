use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use livekit_protocol::agent::agent_thread::CreateRequest;

use crate::StorageError;
use lk_agents_abi::storage::Storage as _;
use lk_agents_abi::storage::Unacknowledged;
use lk_agents_net::env::ProcessEnv;

use crate::backends::sqlite::{FileConnection, SqliteStorage, Workers};
use crate::sql::{BatchResult, Bind as _, Row, SqlValue, Statement, StatementKind, statement};
use crate::storage::Namespace;
use crate::thread::{LeaseTiming, SchemaCell};
use crate::{Storage, Thread, ThreadDbConnection, shortuuid};

// AgentDB cases need a LiveKit project (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) and are
// #[ignore]d; run them with --ignored, where missing credentials fail.

#[derive(Debug, Clone, Copy)]
pub enum Backend {
    Sqlite,
    AgentDb,
}

/// One thread on one backend, deleted (with its directory) at `cleanup`.
pub struct Fixture {
    pub storage: Arc<dyn Storage>,
    pub thread_id: String,
    lease: LeaseTiming,
    pub dir: Option<PathBuf>,
}

/// A backend with no thread yet, as the host's `Storage` namespace configures it, removed (with
/// its directory) at `close`.
pub struct Store {
    pub namespace: Arc<Namespace>,
    pub storage: Arc<dyn Storage>,
    pub dir: Option<PathBuf>,
}

impl Store {
    pub async fn close(self) {
        self.namespace.aclose().await;
        if let Some(dir) = self.dir {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

impl Backend {
    pub async fn store(self, lease: LeaseTiming) -> Store {
        let env = ProcessEnv::current_runtime();
        let (ttl, renew) = (Some(lease.ttl), Some(lease.renew_interval));
        let (namespace, dir) = match self {
            Backend::Sqlite => {
                let dir = std::env::temp_dir().join(shortuuid("lk-agents-storage-"));
                let local = Some(lk_agents_abi::storage::LocalStorageOptions {
                    directory: Some(dir.display().to_string()),
                    project_id: "default".into(),
                });
                (Namespace::new(&env, local, None, None, None, ttl, renew, None), Some(dir))
            }
            Backend::AgentDb => {
                let var = |name: &str| {
                    let value = std::env::var(name);
                    Some(value.unwrap_or_else(|_| panic!("the AgentDB cases need {name}")))
                };
                let (url, key, secret) =
                    (var("LIVEKIT_URL"), var("LIVEKIT_API_KEY"), var("LIVEKIT_API_SECRET"));
                (Namespace::new(&env, None, url, key, secret, ttl, renew, None), None)
            }
        };
        let namespace = Arc::new(namespace.unwrap());
        let storage = namespace.storage().await.unwrap();
        Store { namespace, storage, dir }
    }

    /// A fresh thread; `lease` overrides the default lease timing.
    pub async fn fixture(self, lease: Option<(Duration, Duration)>) -> Fixture {
        let lease = match lease {
            Some((ttl, interval)) => LeaseTiming::new(ttl, interval).unwrap(),
            None => LeaseTiming::default(),
        };
        let Store { storage, dir, .. } = self.store(lease).await;
        let thread = storage.create_thread(CreateRequest::default()).await.unwrap();
        Fixture { storage, thread_id: thread.thread_id, lease, dir }
    }
}

impl Fixture {
    /// One handle over the thread, as one worker holds it: its own connection.
    pub async fn thread(&self) -> Thread {
        self.storage.open(&self.thread_id).await.unwrap()
    }

    /// The SQLite thread's database file.
    fn file(&self) -> PathBuf {
        let dir = self.dir.as_ref().expect("a SQLite thread");
        dir.join("threads").join(format!("{}.sqlite", self.thread_id))
    }

    /// Hands `partition_key`'s lease to `holder` under a token no handle has, for a minute, as a dead
    /// worker or a lost cleanup leaves it.
    pub async fn orphan_lease(&self, partition_key: &str, holder: &str) {
        let conn = FileConnection::open(self.file(), false, threads()).await.unwrap();
        conn.batch(vec![OrphanLease { partition_key, holder }.bind()]).await.unwrap();
        conn.close().await;
    }

    pub async fn cleanup(self, threads: &[&Thread]) {
        for t in threads {
            t.close().await;
        }
        self.storage.delete_thread(&self.thread_id).await.unwrap();
        self.storage.close().await;
        if let Some(dir) = self.dir {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// Registers each case on both backends: `<case>_sqlite` and an ignored `<case>_agentdb`.
#[macro_export]
macro_rules! contract {
    ($($case:ident),* $(,)?) => {
        mod sqlite {
            $(
                #[::tokio::test(flavor = "multi_thread", worker_threads = 4)]
                async fn $case() {
                    super::$case($crate::testing::Backend::Sqlite).await
                }
            )*
        }
        mod agentdb {
            $(
                #[::tokio::test(flavor = "multi_thread", worker_threads = 4)]
                #[ignore = "needs a LiveKit project: run with --ignored and its credentials"]
                async fn $case() {
                    super::$case($crate::testing::Backend::AgentDb).await
                }
            )*
        }
    };
}

statement! {
    /// Hands the partition's lease to a holder under a token nobody has.
    OrphanLease<'a> { partition_key: &'a str, holder: &'a str } => r#"
UPDATE partitions SET lease_holder = :holder, lease_token = 'orphan', lease_expires_at_ms = unixepoch() + 60
WHERE partition_key = :partition_key
"#
}

/// The lease holding `partition_key` as `(token, until)`, `until` in the database's seconds.
pub async fn lessee(conn: &dyn ThreadDbConnection, partition_key: &str) -> Option<(String, i64)> {
    let rows =
        conn.query(crate::thread::schema::SelectFence { partition_key }.bind()).await.ok()?;
    match rows.first()?.as_slice() {
        [_, SqlValue::Text(token), SqlValue::Int(until)] => Some((token.clone(), *until)),
        _ => None,
    }
}

/// How the next write fails: its acknowledgment lost before or after it applied, applied with
/// no answer ever, or refused outright, by a disk I/O error or as too large.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteFault {
    LoseAckBefore,
    LoseAckAfter,
    /// Applied, then pending forever: the request outlives whoever awaits it.
    StallAfter,
    Refuse,
    RefuseTooLarge,
}

/// A thread database connection with faults a real connection shows: a commit's answer lost
/// (`Uncertain`, applied or not), a refused commit (`Unavailable`, or `TooLarge` as an AgentDB
/// frame over its limit is), and an unreachable database for a while.
pub struct Faulty {
    inner: Box<dyn ThreadDbConnection>,
    faults: Arc<Faults>,
}

/// The faults armed on a [`Faulty`] connection and those sharing them.
#[derive(Default)]
pub struct Faults {
    commit: parking_lot::Mutex<Option<WriteFault>>,
    take: parking_lot::Mutex<Option<WriteFault>>,
    write: parking_lot::Mutex<Option<(&'static str, WriteFault)>>,
    probes_until: parking_lot::Mutex<Option<tokio::time::Instant>>,
    unreachable_for: parking_lot::Mutex<Option<Duration>>,
    unreachable_until: parking_lot::Mutex<Option<tokio::time::Instant>>,
}

impl std::ops::Deref for Faulty {
    type Target = Faults;
    fn deref(&self) -> &Faults {
        &self.faults
    }
}

impl Faulty {
    pub fn new(inner: Box<dyn ThreadDbConnection>) -> Self {
        Arc::<Faults>::default().over(inner)
    }
}

impl Faults {
    /// `inner` under these faults, armed for every connection sharing them.
    pub fn over(self: &Arc<Self>, inner: Box<dyn ThreadDbConnection>) -> Faulty {
        Faulty { inner, faults: self.clone() }
    }

    /// Every request fails `Unavailable` for `d` from now.
    pub fn unreachable_for(&self, d: Duration) {
        *self.unreachable_until.lock() = Some(tokio::time::Instant::now() + d);
    }

    pub fn fail_next_commit(&self, fault: WriteFault) {
        *self.commit.lock() = Some(fault);
    }

    /// A lost commit's resolution probes fail `Unavailable` for `d` from now; other requests
    /// still reach the database.
    pub fn probes_unavailable_for(&self, d: Duration) {
        *self.probes_until.lock() = Some(tokio::time::Instant::now() + d);
    }

    /// The next batch with a statement whose SQL contains `sql`: any statement kind.
    pub fn fail_next_write(&self, sql: &'static str, fault: WriteFault) {
        *self.write.lock() = Some((sql, fault));
    }

    /// The next batch that takes a lease (an acquire or an inbox receive).
    pub fn fail_next_lease_take(&self, fault: WriteFault) {
        *self.take.lock() = Some(fault);
    }

    /// Every request after the next faulted batch fails `Unavailable` for `d`.
    pub fn unreachable_after_commit_for(&self, d: Duration) {
        *self.unreachable_for.lock() = Some(d);
    }

    fn reachable(&self) -> Result<(), StorageError> {
        let now = tokio::time::Instant::now();
        match *self.unreachable_until.lock() {
            Some(until) if now < until => Err(StorageError::Unavailable("unreachable".into())),
            _ => Ok(()),
        }
    }
}

#[async_trait::async_trait]
impl ThreadDbConnection for Faulty {
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        self.reachable()?;
        self.inner.query(statement).await
    }

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        self.reachable()?;
        let has = |kind| statements.iter().any(|s| s.kind() == Some(kind));
        let is_commit = has(StatementKind::Commit);
        // only a commit carrying staged writes can outgrow the database's frame: a run closed
        // with nothing of its own writes the partition and its journal, and leaves the fault armed
        let staged = statements
            .iter()
            .any(|s| !matches!(s.kind(), Some(StatementKind::Commit | StatementKind::Bookkeeping)));
        if has(StatementKind::FenceRead)
            && self.probes_until.lock().is_some_and(|until| tokio::time::Instant::now() < until)
        {
            return Err(StorageError::Unavailable("probe unreachable".into()));
        }
        let takes = has(StatementKind::LeaseTake);
        let fault = match *self.commit.lock() {
            Some(WriteFault::RefuseTooLarge) if !staged => None,
            ref mut armed if is_commit => armed.take(),
            _ => None,
        };
        let fault = fault.or_else(|| takes.then(|| self.take.lock().take()).flatten());
        let fault = fault.or_else(|| {
            let mut write = self.write.lock();
            let armed =
                write.is_some_and(|(sql, _)| statements.iter().any(|s| s.sql().contains(sql)));
            armed.then(|| write.take().map(|(_, fault)| fault)).flatten()
        });
        if fault.is_some()
            && let Some(d) = self.unreachable_for.lock().take()
        {
            *self.unreachable_until.lock() = Some(tokio::time::Instant::now() + d);
        }
        match fault {
            None => self.inner.batch(statements).await,
            Some(WriteFault::LoseAckBefore) => {
                Err(StorageError::Uncertain(Unacknowledged::sent("lost before")))
            }
            Some(WriteFault::LoseAckAfter) => {
                self.inner.batch(statements).await?;
                Err(StorageError::Uncertain(Unacknowledged::sent("lost after")))
            }
            Some(WriteFault::StallAfter) => {
                self.inner.batch(statements).await?;
                std::future::pending().await
            }
            Some(WriteFault::Refuse) => {
                let io = rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR);
                Err(crate::backends::sqlite::sqlite_error(rusqlite::Error::SqliteFailure(io, None)))
            }
            Some(WriteFault::RefuseTooLarge) => {
                Err(StorageError::TooLarge { what: "the commit".into(), limit: 0 })
            }
        }
    }

    async fn close(&self) {
        self.inner.close().await;
    }
}

impl Fixture {
    /// A handle over a SQLite thread whose connection takes injected faults.
    pub async fn faulty_thread(&self) -> (Thread, Arc<Faulty>) {
        let conn = FileConnection::open(self.file(), false, threads()).await.unwrap();
        let faulty = Arc::new(Faulty::new(Box::new(conn)));
        let thread =
            Thread::new(faulty.clone(), &self.thread_id, self.lease, SchemaCell::default())
                .await
                .unwrap();
        (thread, faulty)
    }
}

/// A directory of its own under the system's temp directory, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(prefix: &str) -> Self {
        let dir = std::env::temp_dir().join(shortuuid(prefix));
        std::fs::create_dir_all(&dir).expect("a temp directory");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Production's workers: a thread each, on the OS wall clock.
pub fn threads() -> Workers {
    Workers::threads(Arc::new(lk_agents_net::clock::RealClock::new()))
}

/// Workers as tasks of the current runtime, on the OS wall clock: a paused-time test's waits
/// stay virtual.
pub fn tasks() -> Workers {
    let spawner = Arc::new(lk_agents_net::task::TokioSpawner::current());
    Workers { spawner, clock: Arc::new(lk_agents_net::clock::RealClock::new()) }
}

/// A thread database file of its own, its connections' workers [`tasks`]. Removed on drop.
pub struct TempDatabase {
    dir: TempDir,
    workers: Workers,
}

impl TempDatabase {
    pub fn new() -> Self {
        Self { dir: TempDir::new("lk-agents-db-"), workers: tasks() }
    }

    /// Another worker's connection to the database.
    pub async fn connect(&self) -> FileConnection {
        let file = self.dir.0.join("thread.sqlite");
        FileConnection::open(file, true, self.workers.clone()).await.expect("a thread database")
    }
}

impl Default for TempDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// A spawner whose tasks die, dropped at their next poll once armed: a database worker that
/// crashes with requests queued.
pub struct Dying {
    spawner: Arc<dyn lk_agents_abi::Spawner>,
    armed: Arc<std::sync::atomic::AtomicBool>,
    deaths: Arc<std::sync::atomic::AtomicU64>,
}

impl Dying {
    pub fn new(spawner: Arc<dyn lk_agents_abi::Spawner>) -> Self {
        Self { spawner, armed: Arc::default(), deaths: Arc::default() }
    }

    /// How many tasks died.
    pub fn deaths(&self) -> u64 {
        self.deaths.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The next task polled dies instead.
    pub fn die_next_poll(&self) {
        self.armed.store(true, std::sync::atomic::Ordering::Release);
    }
}

impl lk_agents_abi::Spawner for Dying {
    fn launch(
        &self,
        mut task: lk_agents_abi::Launch,
    ) -> lk_agents_abi::OwnedTask<Box<dyn std::any::Any + Send>> {
        let (armed, deaths) = (self.armed.clone(), self.deaths.clone());
        let mut work = task.work;
        task.work = Box::pin(std::future::poll_fn(move |cx| {
            if armed.swap(false, std::sync::atomic::Ordering::AcqRel) {
                deaths.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                return std::task::Poll::Ready(Box::new(()) as Box<dyn std::any::Any + Send>);
            }
            work.as_mut().poll(cx)
        }));
        self.spawner.launch(task)
    }
}

/// The SQLite backend with each thread connection it opens decorated by `wrap`: how a test or a
/// simulation puts faults under the production partition. Everything else is the backend's own.
pub struct Wrapped<W> {
    pub storage: Arc<SqliteStorage>,
    pub wrap: W,
}

#[async_trait::async_trait]
impl<W> Storage for Wrapped<W>
where
    W: Fn(FileConnection) -> Arc<dyn ThreadDbConnection> + Send + Sync,
{
    async fn create_thread(
        &self,
        request: CreateRequest,
    ) -> Result<livekit_protocol::agent::agent_thread::Thread, StorageError> {
        self.storage.create_thread(request).await
    }

    async fn get_thread(
        &self,
        thread_id: &str,
    ) -> Result<livekit_protocol::agent::agent_thread::Thread, StorageError> {
        self.storage.get_thread(thread_id).await
    }

    async fn list_threads(
        &self,
        request: livekit_protocol::agent::agent_thread::ListRequest,
    ) -> Result<livekit_protocol::agent::agent_thread::ListResponse, StorageError> {
        self.storage.list_threads(request).await
    }

    async fn update_thread(
        &self,
        request: livekit_protocol::agent::agent_thread::UpdateRequest,
    ) -> Result<livekit_protocol::agent::agent_thread::Thread, StorageError> {
        self.storage.update_thread(request).await
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        self.storage.delete_thread(thread_id).await
    }

    async fn open(&self, thread_id: &str) -> Result<Thread, StorageError> {
        let connection = (self.wrap)(self.storage.connect(thread_id).await?);
        Thread::new(connection, thread_id, self.storage.lease, SchemaCell::default()).await
    }

    async fn close(&self) {
        self.storage.close().await;
    }
}
