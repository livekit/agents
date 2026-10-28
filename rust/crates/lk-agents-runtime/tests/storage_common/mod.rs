//! The backends every storage and A2A contract case runs against. Each case is written once,
//! `async fn case(backend: Backend)`, and `contract!` registers it per backend: local SQLite in a
//! fresh directory always runs; AgentDB needs a LiveKit project (`LIVEKIT_URL`,
//! `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`) and is `#[ignore]`d, so a plain run reports it
//! skipped. The credentialed job runs `--ignored`, where missing credentials fail.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use livekit_protocol::agent::agent_thread::CreateRequest;
use lk_agents_runtime::storage::agentdb::{AgentDbOptions, AgentDbStorage, TokenMinter};
use lk_agents_runtime::storage::engine::LeaseTiming;
use lk_agents_runtime::storage::sqlite::SqliteStorage;
use lk_agents_runtime::storage::{Storage, Thread, shortuuid};

#[derive(Debug, Clone, Copy)]
pub enum Backend {
    Sqlite,
    AgentDb,
}

/// One thread on one backend, deleted (with its directory) at `cleanup`.
pub struct Fixture {
    pub storage: Arc<dyn Storage>,
    pub thread_id: String,
    dir: Option<PathBuf>,
}

impl Backend {
    /// A fresh thread; `lease` overrides the default lease timing.
    pub async fn fixture(self, lease: Option<(Duration, Duration)>) -> Fixture {
        let lease = match lease {
            Some((ttl, interval)) => LeaseTiming::new(ttl, interval).unwrap(),
            None => LeaseTiming::default(),
        };
        let (storage, dir): (Arc<dyn Storage>, _) = match self {
            Backend::Sqlite => {
                let dir = std::env::temp_dir().join(shortuuid("lk-agents-storage-"));
                (Arc::new(SqliteStorage::open(&dir, lease).await.unwrap()), Some(dir))
            }
            Backend::AgentDb => {
                let env = |name: &str| {
                    std::env::var(name).unwrap_or_else(|_| panic!("the AgentDB cases need {name}"))
                };
                let mut opts = AgentDbOptions::new(&env("LIVEKIT_URL"));
                opts.lease = lease;
                let minter =
                    TokenMinter::new(env("LIVEKIT_API_KEY"), env("LIVEKIT_API_SECRET")).unwrap();
                (Arc::new(AgentDbStorage::new(opts, minter).unwrap()), None)
            }
        };
        let thread = storage.create_thread(CreateRequest::default()).await.unwrap();
        Fixture { storage, thread_id: thread.thread_id, dir }
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
                #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
                async fn $case() {
                    super::$case($crate::storage_common::Backend::Sqlite).await
                }
            )*
        }
        mod agentdb {
            $(
                #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
                #[ignore = "needs a LiveKit project: run with --ignored and its credentials"]
                async fn $case() {
                    super::$case($crate::storage_common::Backend::AgentDb).await
                }
            )*
        }
    };
}
