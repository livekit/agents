//! Test tooling (feature `testing`): the backends every storage and A2A contract case runs
//! against, and a connection with injected faults. Each case is written once,
//! `async fn case(backend: Backend)`, and [`contract!`](crate::contract) registers it per
//! backend: local SQLite in a fresh directory always runs; AgentDB needs a LiveKit project
//! (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`) and is `#[ignore]`d, so a plain run
//! reports it skipped. The credentialed job runs `--ignored`, where missing credentials fail.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use livekit_protocol::agent::agent_thread::CreateRequest;

use crate::StorageError;
use crate::agentdb::{AgentDbOptions, AgentDbStorage, TokenMinter};
use crate::engine::{LeaseTiming, SchemaCell};
use crate::sql::{BatchResult, Row, Statement};
use crate::sqlite::{FileConnection, SqliteStorage};
use crate::{SqliteConnection, Storage, Thread, ThreadDb, shortuuid};

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

/// A backend with no thread yet, removed (with its directory) at `close`.
pub struct Store {
    pub storage: Arc<dyn Storage>,
    pub dir: Option<PathBuf>,
}

impl Store {
    pub async fn close(self) {
        self.storage.close().await;
        if let Some(dir) = self.dir {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

impl Backend {
    pub async fn store(self, lease: LeaseTiming) -> Store {
        match self {
            Backend::Sqlite => {
                let dir = std::env::temp_dir().join(shortuuid("lk-agents-storage-"));
                let storage = Arc::new(SqliteStorage::open(&dir, lease).await.unwrap());
                Store { storage, dir: Some(dir) }
            }
            Backend::AgentDb => {
                let env = |name: &str| {
                    std::env::var(name).unwrap_or_else(|_| panic!("the AgentDB cases need {name}"))
                };
                let mut opts = AgentDbOptions::new(&env("LIVEKIT_URL"));
                opts.lease = lease;
                let minter =
                    TokenMinter::new(env("LIVEKIT_API_KEY"), env("LIVEKIT_API_SECRET")).unwrap();
                Store { storage: Arc::new(AgentDbStorage::new(opts, minter).unwrap()), dir: None }
            }
        }
    }

    /// A fresh thread; `lease` overrides the default lease timing.
    pub async fn fixture(self, lease: Option<(Duration, Duration)>) -> Fixture {
        let lease = match lease {
            Some((ttl, interval)) => LeaseTiming::new(ttl, interval).unwrap(),
            None => LeaseTiming::default(),
        };
        let Store { storage, dir } = self.store(lease).await;
        let thread = storage.create_thread(CreateRequest::default()).await.unwrap();
        Fixture { storage, thread_id: thread.thread_id, lease, dir }
    }
}

impl Fixture {
    /// One handle over the thread, as one worker holds it: its own connection.
    pub async fn thread(&self, holder: &str) -> Thread {
        self.storage.open(&self.thread_id, holder).await.unwrap()
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

/// How the next commit batch fails: its acknowledgment lost before or after it applied, or
/// refused outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitFault {
    LoseAckBefore,
    LoseAckAfter,
    Refuse,
}

/// A thread database connection with faults a real connection shows: a commit's answer lost
/// (`Uncertain`, applied or not), a refused commit, and an unreachable database for a while.
pub struct Faulty {
    inner: Box<dyn SqliteConnection>,
    commit: std::sync::Mutex<Option<CommitFault>>,
    unreachable_for: std::sync::Mutex<Option<Duration>>,
    unreachable_until: std::sync::Mutex<Option<tokio::time::Instant>>,
}

impl Faulty {
    pub fn fail_next_commit(&self, fault: CommitFault) {
        *self.commit.lock().unwrap() = Some(fault);
    }

    /// Every request after the next faulted commit fails `Unavailable` for `d`.
    pub fn unreachable_after_commit_for(&self, d: Duration) {
        *self.unreachable_for.lock().unwrap() = Some(d);
    }

    fn reachable(&self) -> Result<(), StorageError> {
        match *self.unreachable_until.lock().unwrap() {
            Some(until) if tokio::time::Instant::now() < until => {
                Err(StorageError::Unavailable("unreachable".into()))
            }
            _ => Ok(()),
        }
    }
}

#[async_trait::async_trait]
impl SqliteConnection for Faulty {
    async fn query(
        &self,
        statement: Statement,
    ) -> Result<Vec<Row>, StorageError> {
        self.reachable()?;
        self.inner.query(statement).await
    }

    async fn batch(
        &self,
        statements: Vec<Statement>,
    ) -> Result<BatchResult, StorageError> {
        self.reachable()?;
        let is_commit = statements.iter().any(|s| s.sql.contains("version = version + 1"));
        let fault = if is_commit { self.commit.lock().unwrap().take() } else { None };
        if fault.is_some()
            && let Some(d) = self.unreachable_for.lock().unwrap().take()
        {
            *self.unreachable_until.lock().unwrap() = Some(tokio::time::Instant::now() + d);
        }
        match fault {
            None => self.inner.batch(statements).await,
            Some(CommitFault::LoseAckBefore) => Err(StorageError::Uncertain("lost before".into())),
            Some(CommitFault::LoseAckAfter) => {
                self.inner.batch(statements).await?;
                Err(StorageError::Uncertain("lost after".into()))
            }
            Some(CommitFault::Refuse) => Err(StorageError::Server {
                code: "sqlite".into(),
                message: "disk I/O error".into(),
            }),
        }
    }

    async fn close(&self) {
        self.inner.close().await;
    }
}

impl Fixture {
    /// A handle over a SQLite thread whose connection takes injected faults.
    pub async fn faulty_thread(&self, holder: &str) -> (Thread, Arc<Faulty>) {
        let dir = self.dir.as_ref().expect("faults are injected on the SQLite backend");
        let file = dir.join("threads").join(format!("{}.sqlite", self.thread_id));
        let faulty = Arc::new(Faulty {
            inner: Box::new(FileConnection::open(file, false).await.unwrap()),
            commit: std::sync::Mutex::new(None),
            unreachable_for: std::sync::Mutex::new(None),
            unreachable_until: std::sync::Mutex::new(None),
        });
        let db = ThreadDb::new(faulty.clone(), &self.thread_id, self.lease, SchemaCell::default())
            .await
            .unwrap();
        (Thread::new(db, holder), faulty)
    }
}
