use std::collections::HashMap;
use std::time::Duration;

use crate::StorageError;
use crate::backends::sqlite::SqliteStorage;
use crate::testing::Backend;
use crate::thread::LeaseTiming;
use crate::{CommitOptions, RunOptions, Storage};
use livekit_protocol::agent::agent_thread::{CreateRequest, ListRequest, UpdateRequest};

fn ttl(seconds: i64) -> Option<pbjson_types::Duration> {
    Some(pbjson_types::Duration { seconds, nanos: 0 })
}

fn attributes(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn not_found<T: std::fmt::Debug>(r: Result<T, StorageError>) -> bool {
    matches!(r, Err(StorageError::NotFound(_)))
}

async fn a_thread_is_created_read_listed_and_updated(backend: Backend) {
    let store = backend.store(LeaseTiming::default()).await;
    let s = &store.storage;
    let scope = crate::shortuuid("user-");
    let create = |a: &[(&str, &str)]| CreateRequest {
        scope: scope.clone(),
        attributes: attributes(a),
        idle_ttl: ttl(3600),
    };
    let first = s.create_thread(create(&[("topic", "billing")])).await.unwrap();
    let second = s.create_thread(create(&[])).await.unwrap();
    assert!(first.thread_id.starts_with("AT_") && first.thread_id != second.thread_id);
    assert_eq!(
        s.get_thread(&first.thread_id).await.unwrap().attributes,
        attributes(&[("topic", "billing")])
    );

    let page = s.list_threads(ListRequest { scope: scope.clone(), limit: 1, ..Default::default() });
    let page = page.await.unwrap();
    assert_eq!(page.threads.len(), 1);
    let rest = ListRequest { scope: scope.clone(), page_token: page.next_page_token, limit: 1 };
    let rest = s.list_threads(rest).await.unwrap();
    let mut listed: Vec<String> =
        page.threads.iter().chain(&rest.threads).map(|t| t.thread_id.clone()).collect();
    listed.sort();
    let mut created = vec![first.thread_id.clone(), second.thread_id.clone()];
    created.sort();
    assert_eq!(listed, created, "two pages, no gap or duplicate");

    // an empty value deletes its key only; an explicit zero TTL is kept, never read as "unset"
    let update = UpdateRequest {
        thread_id: first.thread_id.clone(),
        attributes: attributes(&[("topic", ""), ("tier", "gold")]),
        idle_ttl: ttl(0),
    };
    let updated = s.update_thread(update).await.unwrap();
    assert_eq!(updated.attributes, attributes(&[("tier", "gold")]));
    assert_eq!(updated.idle_ttl.map(|d| d.seconds), Some(0));
    let reread = s.get_thread(&first.thread_id).await.unwrap();
    assert_eq!(
        (reread.attributes, reread.idle_ttl.map(|d| d.seconds)),
        (attributes(&[("tier", "gold")]), Some(0))
    );

    for t in [&first, &second] {
        s.delete_thread(&t.thread_id).await.unwrap();
    }
    store.close().await;
}

async fn a_deleted_thread_is_gone_and_deleting_again_is_safe(backend: Backend) {
    let store = backend.store(LeaseTiming::default()).await;
    let s = &store.storage;
    let t = s.create_thread(CreateRequest::default()).await.unwrap();
    s.delete_thread(&t.thread_id).await.unwrap();
    s.delete_thread(&t.thread_id).await.unwrap();
    assert!(not_found(s.get_thread(&t.thread_id).await));
    assert!(not_found(s.open(&t.thread_id, "w1").await.map(|_| ())));
    store.close().await;
}

async fn an_expired_thread_reads_as_not_found(backend: Backend) {
    let store = backend.store(LeaseTiming::default()).await;
    let s = &store.storage;
    let scope = crate::shortuuid("user-");
    let req = CreateRequest { scope: scope.clone(), idle_ttl: ttl(1), ..Default::default() };
    let t = s.create_thread(req).await.unwrap();
    tokio::time::sleep(Duration::from_millis(2100)).await;
    assert!(not_found(s.get_thread(&t.thread_id).await));
    let listed = s.list_threads(ListRequest { scope, ..Default::default() }).await.unwrap();
    assert!(listed.threads.is_empty());
    store.close().await;
}

/// Only `update_thread` moves expiry: a managed turn's commit does not touch the thread.
async fn a_commit_does_not_extend_expiry(backend: Backend) {
    let store = backend.store(LeaseTiming::default()).await;
    let s = &store.storage;
    let t = s.create_thread(CreateRequest { idle_ttl: ttl(3600), ..Default::default() }).await;
    let t = t.unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let thread = s.open(&t.thread_id, "w1").await.unwrap();
    let mut run = thread
        .lane("front-desk")
        .open(RunOptions::new("r1", "AST_1"), &crate::Holder::new("test"))
        .await
        .unwrap();
    run.set_state(serde_json::json!({"n": 1}));
    run.commit(CommitOptions::default()).await.unwrap();
    thread.close().await;
    assert_eq!(s.get_thread(&t.thread_id).await.unwrap().expires_at, t.expires_at);
    s.delete_thread(&t.thread_id).await.unwrap();
    store.close().await;
}

crate::contract!(
    a_thread_is_created_read_listed_and_updated,
    a_deleted_thread_is_gone_and_deleting_again_is_safe,
    an_expired_thread_reads_as_not_found,
    a_commit_does_not_extend_expiry,
);

/// Local ids name files: anything but a plain `AT_` id is refused before a path is built, and a
/// database file the registry does not list (a create that crashed before its registry row) is
/// not a thread.
#[tokio::test]
async fn sqlite_refuses_unknown_and_unlisted_ids() {
    let store = Backend::Sqlite.store(LeaseTiming::default()).await;
    let s = &store.storage;
    for id in ["AT_../../registry", "AT_a/b", "DB_x", "", "AT_unknown"] {
        assert!(not_found(s.get_thread(id).await), "{id}");
        assert!(not_found(s.open(id, "w1").await.map(|_| ())), "{id}");
    }
    let dir = store.dir.clone().unwrap();
    std::fs::write(dir.join("threads").join("AT_orphan.sqlite"), b"").unwrap();
    assert!(not_found(s.open("AT_orphan", "w1").await.map(|_| ())));
    store.close().await;
}

#[tokio::test]
async fn sqlite_metadata_updates_merge_across_storage_instances() {
    let store = Backend::Sqlite.store(LeaseTiming::default()).await;
    let other =
        SqliteStorage::open(store.dir.as_ref().unwrap(), LeaseTiming::default()).await.unwrap();
    let thread = store
        .storage
        .create_thread(CreateRequest { idle_ttl: ttl(3600), ..Default::default() })
        .await
        .unwrap();
    let updates = (0..20).map(|index| {
        let storage: &dyn Storage = if index % 2 == 0 { &*store.storage } else { &other };
        let request = UpdateRequest {
            thread_id: thread.thread_id.clone(),
            attributes: attributes(&[(&format!("key.{index}"), "value")]),
            idle_ttl: (index == 0).then(|| ttl(0).unwrap()),
        };
        storage.update_thread(request)
    });
    for result in futures_util::future::join_all(updates).await {
        result.unwrap();
    }
    let updated = other.get_thread(&thread.thread_id).await.unwrap();
    assert_eq!(updated.attributes.len(), 20);
    assert_eq!(updated.idle_ttl.unwrap().seconds, 0, "an omitted TTL must keep the current value");
    let updated = other
        .update_thread(UpdateRequest {
            thread_id: thread.thread_id.clone(),
            attributes: attributes(&[("key.0", ""), ("key.1", "changed")]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(updated.attributes.len(), 19);
    assert!(!updated.attributes.contains_key("key.0"));
    assert_eq!(updated.attributes["key.1"], "changed");
    other.close().await;
    store.storage.delete_thread(&thread.thread_id).await.unwrap();
    store.close().await;
}

#[tokio::test]
async fn sqlite_delete_and_expiry_invalidate_open_handles() {
    let store = Backend::Sqlite.store(LeaseTiming::default()).await;
    let storage = &store.storage;
    for expired in [false, true] {
        let record = storage
            .create_thread(CreateRequest {
                idle_ttl: expired.then(|| ttl(1).unwrap()),
                ..Default::default()
            })
            .await
            .unwrap();
        let thread = storage.open(&record.thread_id, "reader").await.unwrap();
        let lane = thread.lane("desk");
        let mut run =
            lane.open(RunOptions::new("held", "task"), &crate::Holder::new("test")).await.unwrap();
        if expired {
            tokio::time::sleep(Duration::from_millis(2100)).await;
        } else {
            storage.delete_thread(&record.thread_id).await.unwrap();
        }
        assert!(not_found(lane.history().await));
        assert!(not_found(run.commit(CommitOptions::default()).await));
        thread.close().await;
        storage.delete_thread(&record.thread_id).await.unwrap();
    }
    store.close().await;
}

// Real process takeover: a stopped owner cannot overwrite its replacement's durable state.
#[cfg(unix)]
mod process {
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command};
    use std::sync::Arc;
    use std::time::Duration;

    use crate::backends::sqlite::SqliteStorage;
    use crate::testing::{Backend, Fixture};
    use crate::thread::LeaseTiming;
    use crate::{CommitOptions, RunOptions, Storage, shortuuid};
    use crate::{RunOutcome, StorageError};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    const PREFIX: &str = "LK_STORAGE_PROCESS_";
    // Eight seconds for lease admission, two ten-second network phases, five for cleanup.
    const PROCESS_WAIT: Duration = Duration::from_secs(33);

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    enum Outcome {
        Committed,
        Restored,
        Stale,
        Deleted,
    }

    struct Process(Child);

    impl Drop for Process {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    struct Control(PathBuf);

    impl Drop for Control {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn timing() -> LeaseTiming {
        LeaseTiming::new(Duration::from_secs(2), Duration::from_millis(200)).unwrap()
    }

    fn input(name: &str) -> String {
        std::env::var(format!("{PREFIX}{name}")).expect("process fixture input")
    }

    async fn worker() -> TestResult<()> {
        let control = PathBuf::from(input("CONTROL"));
        let role = input("ROLE");
        let storage: Arc<dyn Storage> = match input("BACKEND").as_str() {
            "sqlite" => Arc::new(SqliteStorage::open(input("DIRECTORY"), timing()).await?),
            "agentdb" => Backend::AgentDb.store(timing()).await.storage,
            other => return Err(format!("unknown process backend {other}").into()),
        };
        let thread = storage.open(&input("THREAD"), &role).await?;
        let mut run = thread
            .lane("desk")
            .open(
                RunOptions {
                    wait_timeout: Duration::from_secs(8),
                    ..RunOptions::new(format!("run-{role}"), format!("AST_{role}"))
                },
                &crate::Holder::new("test"),
            )
            .await?;
        let outcome = match role.as_str() {
            "deleted" => {
                std::fs::write(control.join("ready"), b"")?;
                while !control.join("resume").exists() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                if !matches!(thread.lane("desk").history().await, Err(StorageError::NotFound(_))) {
                    return Err("a deleted thread remained readable in another process".into());
                }
                if !matches!(
                    run.commit(CommitOptions::default()).await,
                    Err(StorageError::NotFound(_))
                ) {
                    return Err("a deleted thread accepted a commit in another process".into());
                }
                Outcome::Deleted
            }
            "old" => {
                run.set_state(json!({"owner": "stale"}));
                std::fs::write(control.join("ready"), b"")?;
                while !control.join("resume").exists() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                match run.commit(CommitOptions::default()).await {
                    Err(StorageError::VersionConflict { .. }) => Outcome::Stale,
                    other => return Err(format!("stale owner's commit: {other:?}").into()),
                }
            }
            "replacement" => {
                run.set_state(json!({"owner": "replacement"}));
                run.commit(CommitOptions::default()).await?;
                Outcome::Committed
            }
            "reader" => {
                if run.state() != Some(&json!({"owner": "replacement"})) {
                    return Err(format!("restart loaded {:?}", run.state()).into());
                }
                run.abandon(RunOutcome::Completed, "restart verified").await?;
                Outcome::Restored
            }
            other => return Err(format!("unknown process role {other}").into()),
        };
        thread.close().await;
        storage.close().await;
        std::fs::write(control.join(&role), serde_json::to_vec(&outcome)?)?;
        Ok(())
    }

    #[test]
    #[ignore = "subprocess fixture invoked by the backend takeover contract"]
    fn process_worker() {
        tokio::runtime::Runtime::new().unwrap().block_on(worker()).unwrap();
    }

    fn spawn(
        backend: Backend,
        fixture: &Fixture,
        control: &Path,
        role: &str,
    ) -> TestResult<Process> {
        let mut command = Command::new(std::env::current_exe()?);
        command.args(["--exact", &test_name("process_worker"), "--ignored", "--nocapture"]);
        for (name, value) in [
            (
                "BACKEND",
                match backend {
                    Backend::Sqlite => "sqlite",
                    Backend::AgentDb => "agentdb",
                },
            ),
            ("THREAD", fixture.thread_id.as_str()),
            ("ROLE", role),
        ] {
            command.env(format!("{PREFIX}{name}"), value);
        }
        command.env(format!("{PREFIX}CONTROL"), control);
        if let Some(directory) = &fixture.dir {
            command.env(format!("{PREFIX}DIRECTORY"), directory);
        }
        Ok(Process(command.spawn()?))
    }

    async fn wait(process: &mut Process, outcome: &Path) -> TestResult<Outcome> {
        let deadline = tokio::time::Instant::now() + PROCESS_WAIT;
        loop {
            if let Some(status) = process.0.try_wait()? {
                if !status.success() {
                    return Err(format!("process {} exited with {status}", process.0.id()).into());
                }
                return Ok(serde_json::from_slice(&std::fs::read(outcome)?)?);
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(format!("process {} did not settle", process.0.id()).into());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    fn signal(process: &Process, signal: &str) -> TestResult<()> {
        let status = Command::new("kill").args([signal, &process.0.id().to_string()]).status()?;
        if status.success() { Ok(()) } else { Err(format!("kill {signal}: {status}").into()) }
    }

    async fn takeover(backend: Backend, fixture: &Fixture, control: &Path) -> TestResult<()> {
        let mut old = spawn(backend, fixture, control, "old")?;
        let deadline = tokio::time::Instant::now() + PROCESS_WAIT;
        while !control.join("ready").exists() {
            if let Some(status) = old.0.try_wait()? {
                return Err(format!("owner exited before acquiring: {status}").into());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err("owner never acquired the lease".into());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        signal(&old, "-STOP")?;
        let mut replacement = spawn(backend, fixture, control, "replacement")?;
        let committed = wait(&mut replacement, &control.join("replacement")).await?;
        if committed != Outcome::Committed {
            return Err(format!("replacement: {committed:?}").into());
        }
        std::fs::write(control.join("resume"), b"")?;
        signal(&old, "-CONT")?;
        let stale = wait(&mut old, &control.join("old")).await?;
        if stale != Outcome::Stale {
            return Err(format!("old owner: {stale:?}").into());
        }
        let mut reader = spawn(backend, fixture, control, "reader")?;
        let restored = wait(&mut reader, &control.join("reader")).await?;
        if restored != Outcome::Restored {
            return Err(format!("reader: {restored:?}").into());
        }
        Ok(())
    }

    async fn stopped_owner_is_fenced_after_process_takeover_and_restart(backend: Backend) {
        let fixture = backend.fixture(Some((timing().ttl, timing().renew_interval))).await;
        let control = Control(std::env::temp_dir().join(shortuuid("lk-storage-process-")));
        std::fs::create_dir_all(&control.0).unwrap();
        let result = takeover(backend, &fixture, &control.0).await;
        fixture.cleanup(&[]).await;
        result.unwrap();
    }

    crate::contract!(stopped_owner_is_fenced_after_process_takeover_and_restart);

    #[tokio::test]
    async fn sqlite_delete_invalidates_another_process_open_handle() -> TestResult<()> {
        let fixture = Backend::Sqlite.fixture(None).await;
        let control = Control(std::env::temp_dir().join(shortuuid("lk-storage-delete-")));
        std::fs::create_dir_all(&control.0)?;
        let mut worker = spawn(Backend::Sqlite, &fixture, &control.0, "deleted")?;
        let deadline = tokio::time::Instant::now() + PROCESS_WAIT;
        while !control.0.join("ready").exists() {
            if tokio::time::Instant::now() >= deadline || worker.0.try_wait()?.is_some() {
                return Err("reader never opened the thread".into());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        fixture.storage.delete_thread(&fixture.thread_id).await?;
        std::fs::write(control.0.join("resume"), b"")?;
        assert_eq!(wait(&mut worker, &control.0.join("deleted")).await?, Outcome::Deleted);
        fixture.cleanup(&[]).await;
        Ok(())
    }

    /// `name`'s full path in this test binary, for re-running it alone in a child process.
    fn test_name(name: &str) -> String {
        let module = module_path!().split_once("::").map_or("", |(_, module)| module);
        if module.is_empty() { name.to_owned() } else { format!("{module}::{name}") }
    }
}
