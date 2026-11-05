//! Real process takeover: a stopped owner cannot overwrite its replacement's durable state.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::Duration;

use lk_agents_storage::backends::sqlite::SqliteStorage;
use lk_agents_storage::testing::{Backend, Fixture};
use lk_agents_storage::thread::LeaseTiming;
use lk_agents_storage::{CommitOptions, RunOptions, Storage, shortuuid};
use lk_agents_storage::{RunOutcome, StorageError};
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
        .open(RunOptions {
            wait_timeout: Duration::from_secs(8),
            ..RunOptions::new(format!("run-{role}"), format!("AST_{role}"))
        })
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
            if !matches!(run.commit(CommitOptions::default()).await, Err(StorageError::NotFound(_)))
            {
                return Err("a deleted thread accepted a commit in another process".into());
            }
            Outcome::Deleted
        }
        "old" => {
            run.state = Some(json!({"owner": "stale"}));
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
            run.state = Some(json!({"owner": "replacement"}));
            run.commit(CommitOptions::default()).await?;
            Outcome::Committed
        }
        "reader" => {
            if run.state != Some(json!({"owner": "replacement"})) {
                return Err(format!("restart loaded {:?}", run.state).into());
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

fn spawn(backend: Backend, fixture: &Fixture, control: &Path, role: &str) -> TestResult<Process> {
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

lk_agents_storage::contract!(stopped_owner_is_fenced_after_process_takeover_and_restart);

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

/// `name`'s full path in this integration binary, for re-running it alone in a child process.
fn test_name(name: &str) -> String {
    let module = module_path!().split_once("::").map_or("", |(_, module)| module);
    if module.is_empty() { name.to_owned() } else { format!("{module}::{name}") }
}
