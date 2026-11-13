//! A worker killed with SIGKILL leaves its SQLite thread in a state the next worker recovers
//! from. The test binary re-executes itself as the dying worker (`crash_child`, its role in
//! `LK_CRASH_ROLE`); the parent kills it at a chosen point, then takes the partition as another worker.
#![cfg(unix)]
#![allow(
    clippy::disallowed_methods,
    reason = "the test kills a real worker process and waits on it in real time"
)]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use livekit_protocol::agent::agent_thread::CreateRequest;
use livekit_protocol::agent::chat_context::chat_item;
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{ChatMessage, ChatRole};
use lk_agents_abi::storage::thread::TaskPatch;
use lk_agents_storage::backends::sqlite::SqliteStorage;
use lk_agents_storage::testing::{TempDir, threads};
use lk_agents_storage::thread::LeaseTiming;
use lk_agents_storage::thread::run::RunSummary;
use lk_agents_storage::thread::schema::{ItemRow, TaskData};
use lk_agents_storage::{
    CommitOptions, Holder, LeaseAfterCommit, PartitionAccess, RunOptions, Storage, StorageError,
    TaskPickup, TaskState, TaskWrite, Thread, WhenHeld,
};

const PARTITION_KEY: &str = "front-desk";
const TTL: Duration = Duration::from_secs(2);
const RENEW: Duration = Duration::from_millis(400);
/// From the kill to the next worker holding the partition: the database counts whole seconds (a
/// lease taken at `t` lapses once `floor(now) > floor(t) + ttl`), then the acquire's backoff
/// polls at most once a second.
const TAKEOVER: Duration = Duration::from_secs(6);
/// The dying worker's history: one exchange, committed together or not at all.
const ITEMS: [&str; 2] = ["item_user", "item_agent"];

/// What the dying worker does before it is killed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Takes the partition, stages its items, never commits.
    Hold,
    /// Commits keeping the lease, then idles: the partition stays held without renewal.
    CommitKeep,
    /// Takes the partition and idles while its renewal keeps it.
    Idle,
    /// Commits releasing the lease as soon as it holds the partition; killed somewhere around it.
    CommitRelease,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Role::Hold => "hold",
            Role::CommitKeep => "commit-keep",
            Role::Idle => "idle",
            Role::CommitRelease => "commit-release",
        }
    }

    fn parse(name: &str) -> Self {
        [Role::Hold, Role::CommitKeep, Role::Idle, Role::CommitRelease]
            .into_iter()
            .find(|r| r.name() == name)
            .expect("a known role")
    }
}

fn timing() -> LeaseTiming {
    LeaseTiming::new(TTL, RENEW).expect("a valid lease timing")
}

async fn storage(dir: &Path) -> SqliteStorage {
    SqliteStorage::open(dir, timing(), threads()).await.expect("the local storage")
}

fn message(id: &str, role: ChatRole) -> ItemRow {
    ItemRow::new(chat_item::Item::Message(ChatMessage {
        id: id.into(),
        role: role as i32,
        content: vec![ChatContent { payload: Some(Payload::Text(id.into())) }],
        ..Default::default()
    }))
}

fn submitted(task_id: &str) -> TaskWrite {
    TaskWrite {
        task_id: task_id.into(),
        status: TaskState::Submitted,
        message_id: Some(format!("m_{task_id}")),
        name: None,
        data: TaskData::default(),
        internal: None,
    }
}

fn claim(run_id: &str, task_id: &str) -> RunOptions {
    RunOptions { pickup: TaskPickup::Claim, ..RunOptions::new(run_id, task_id) }
}

fn say(line: &str) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    writeln!(out, "LK_CRASH {line}").expect("the parent's pipe");
    out.flush().expect("the parent's pipe");
}

/// The dying worker. Ignored, so it runs only when the parent re-executes the binary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "the crash cases' child process"]
async fn crash_child() {
    let Ok(role) = std::env::var("LK_CRASH_ROLE") else { return };
    let role = Role::parse(&role);
    let dir = PathBuf::from(std::env::var("LK_CRASH_DIR").unwrap());
    let thread_id = std::env::var("LK_CRASH_THREAD").unwrap();
    let storage = storage(&dir).await;
    let thread = storage.open(&thread_id).await.unwrap();
    let partition_key = thread.partition(PARTITION_KEY);
    assert!(partition_key.submit(&submitted("AST_dead")).await.unwrap());
    let mut run =
        partition_key.open(claim("run_dead", "AST_dead"), &Holder::new("dying")).await.unwrap();
    assert!(run.picked_up());
    run.history_mut()
        .extend([message(ITEMS[0], ChatRole::User), message(ITEMS[1], ChatRole::Assistant)]);
    run.tasks().write("AST_dead", TaskState::Completed, TaskPatch::default()).await.unwrap();
    say("acquired");
    match role {
        Role::Hold | Role::Idle => {}
        Role::CommitKeep | Role::CommitRelease => {
            let lease = match role {
                Role::CommitKeep => LeaseAfterCommit::Keep,
                _ => LeaseAfterCommit::Release,
            };
            run.commit(CommitOptions { lease, ..Default::default() }).await.unwrap();
            say("committed");
        }
    }
    // killed here
    tokio::time::sleep(Duration::from_secs(3600)).await;
    drop(run);
}

/// The dying worker, its stdout read line by line.
struct Dying {
    child: Child,
    lines: mpsc::Receiver<String>,
}

impl Dying {
    fn spawn(role: Role, dir: &Path, thread_id: &str) -> Self {
        let mut child = Command::new(std::env::current_exe().expect("the test binary"))
            .args(["--exact", "crash_child", "--ignored", "--nocapture", "--test-threads=1"])
            .env("LK_CRASH_ROLE", role.name())
            .env("LK_CRASH_DIR", dir)
            .env("LK_CRASH_THREAD", thread_id)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("the child worker");
        let stdout = child.stdout.take().expect("the child's stdout");
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                // libtest may print `test crash_child ... ` ahead of the first line
                if let Some((_, said)) = line.split_once("LK_CRASH ") {
                    let _ = tx.send(said.to_owned());
                }
            }
        });
        Self { child, lines }
    }

    fn wait_for(&self, said: &str) {
        let line = self.lines.recv_timeout(Duration::from_secs(20));
        assert_eq!(line.as_deref(), Ok(said), "the child worker's progress");
    }

    /// SIGKILL: no destructor, no release, no flush.
    fn kill(&mut self) -> Instant {
        self.child.kill().expect("SIGKILL the child");
        let killed = Instant::now();
        let status = self.child.wait().expect("reap the child");
        assert_eq!(std::os::unix::process::ExitStatusExt::signal(&status), Some(9), "{status}");
        killed
    }
}

/// A failed case never leaves its child running.
impl Drop for Dying {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A fresh storage directory and thread, with the storage the parent reopens it through.
async fn thread() -> (TempDir, String) {
    let dir = TempDir::new("lk-crash-");
    let storage = storage(&dir.0).await;
    let thread_id =
        storage.create_thread(CreateRequest::default()).await.expect("a new thread").thread_id;
    storage.close().await;
    (dir, thread_id)
}

/// What the next worker found after the kill.
struct Recovered {
    committed: bool,
    takeover: Duration,
}

/// The next worker takes the partition and checks it against the dead run: its items committed exactly
/// once or not at all, its task completed by that commit or failed as an orphan, its run closed
/// when it committed.
async fn recover(dir: &Path, thread_id: &str, killed: Instant) -> Recovered {
    let storage = storage(dir).await;
    let thread = storage.open(thread_id).await.expect("the thread opens");
    let partition_key = thread.partition(PARTITION_KEY);
    let holder = Holder::new("next");
    assert!(partition_key.submit(&submitted("AST_next")).await.expect("the submit"));
    let mut run = partition_key
        .open(RunOptions { wait_timeout: TAKEOVER * 2, ..claim("run_next", "AST_next") }, &holder)
        .await
        .expect("the next worker takes the partition_key once the dead lease lapses");
    let takeover = killed.elapsed();
    assert!(takeover < TAKEOVER, "took over {takeover:?} after the kill");

    let ids: Vec<&str> = run.history().iter().map(|i| i.id.as_str()).collect();
    let committed = match ids.as_slice() {
        [] => false,
        ids if ids == ITEMS => true,
        ids => panic!("the dead run's history is partial or duplicated: {ids:?}"),
    };
    let summary = dead_run(&thread).await;
    // `working` rows other than the one this run just claimed belonged to a dead holder
    let orphans: Vec<String> = run
        .tasks()
        .working()
        .await
        .expect("the working rows")
        .into_iter()
        .map(|t| t.task_id)
        .filter(|t| t != "AST_next")
        .collect();
    if committed {
        assert_eq!(run.version(), 1, "one commit before this run");
        assert_eq!(summary.outcome, Some(lk_agents_storage::RunOutcome::Completed));
        assert!(orphans.is_empty(), "a committed task is no orphan: {orphans:?}");
        assert_eq!(task(&thread, "AST_dead").await, TaskState::Completed);
    } else {
        assert_eq!(run.version(), 0, "nothing committed before this run");
        assert!(summary.interrupted(), "the dead run started and never ended: {summary:?}");
        assert_eq!(orphans, ["AST_dead"], "the dead run's task is found working");
        let failed: a2a::Message = serde_json::from_value(serde_json::json!({
            "messageId": "msg_worker_lost", "role": "ROLE_AGENT", "parts": [{"text": "worker lost"}]
        }))
        .expect("a message");
        assert!(partition_key.fail_orphan("AST_dead", &failed).await.expect("the orphan fails"));
        assert_eq!(task(&thread, "AST_dead").await, TaskState::Failed);
    }
    run.tasks()
        .write("AST_next", TaskState::Completed, TaskPatch::default())
        .await
        .expect("the task write");
    let after = run.commit(CommitOptions::default()).await.expect("the commit").version;
    assert_eq!(after, run.version() + 1);

    // The dead run again, as a retry of its request would: closed when it committed, writing
    // nothing either way.
    let mut retry = partition_key
        .open(claim("run_dead", "AST_dead"), &holder)
        .await
        .expect("the retry takes the free partition_key");
    assert_eq!(retry.closed(), committed, "a committed run id is closed");
    assert!(!retry.picked_up(), "a terminal task is never picked up again");
    let ids: Vec<String> = retry.history().iter().map(|i| i.id.clone()).collect();
    if committed {
        assert_eq!(
            retry.commit(CommitOptions::default()).await.expect("the closed commit").version,
            after
        );
    } else {
        retry
            .abandon(lk_agents_storage::RunOutcome::Canceled, "retried")
            .await
            .expect("the abandon");
    }
    assert_eq!(
        partition_key.history().await.expect("the history").len(),
        ids.len(),
        "the retry wrote no items"
    );
    let partition_row =
        partition_key.info().await.expect("the partition_key row").expect("the partition_key");
    assert_eq!(partition_row.version, after, "the retry moved no version");
    thread.close().await;
    storage.close().await;
    Recovered { committed, takeover }
}

async fn dead_run(thread: &Thread) -> RunSummary {
    let journal = thread.journal(0, 100).await.expect("the journal");
    let runs = RunSummary::from_events(&journal).expect("well-formed runs");
    let mut dead = runs.into_iter().filter(|r| r.run_id == "run_dead");
    let run = dead.next().expect("the dead run's run_started");
    assert!(dead.next().is_none(), "one dead run");
    run
}

async fn task(thread: &Thread, task_id: &str) -> TaskState {
    thread.task(task_id).await.expect("the task read").expect("the task row").status
}

/// One case end to end: the child runs `role`, the parent kills it once `then` holds.
fn case(role: Role, then: impl FnOnce(&Dying, &Path, &str)) -> Recovered {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("a runtime");
    let (dir, thread_id) = rt.block_on(thread());
    let mut child = Dying::spawn(role, &dir.0, &thread_id);
    then(&child, &dir.0, &thread_id);
    let killed = child.kill();
    rt.block_on(recover(&dir.0, &thread_id, killed))
}

/// (a) Killed holding the lease with staged, uncommitted items: none appear, the task fails as
/// an orphan once the lease lapses.
#[test]
fn killed_before_commit_leaves_nothing_but_an_orphan() {
    let out = case(Role::Hold, |child, _, _| child.wait_for("acquired"));
    assert!(!out.committed);
}

/// (b) Killed right after a commit that kept the lease: the commit appears once, the partition is
/// taken over after the lapse, and a retry of the run is closed.
#[test]
fn killed_after_commit_keeps_the_commit_exactly_once() {
    let out = case(Role::CommitKeep, |child, _, _| {
        child.wait_for("acquired");
        child.wait_for("committed");
    });
    assert!(out.committed);
}

/// (c) Killed idle while renewing: the renewal kept the partition across several TTLs, and the partition
/// is free once the last renewal lapses.
#[test]
fn killed_while_renewing_releases_the_partition_after_one_ttl() {
    let out = case(Role::Idle, |child, dir, thread_id| {
        child.wait_for("acquired");
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let storage = storage(dir).await;
            let thread = storage.open(thread_id).await.unwrap();
            let reject = RunOptions {
                access: PartitionAccess::Exclusive(WhenHeld::Reject),
                ..RunOptions::new("run_early", "")
            };
            // past two TTLs, the partition is still the dying worker's
            for _ in 0..2 {
                tokio::time::sleep(TTL).await;
                let refused = thread
                    .partition(PARTITION_KEY)
                    .open(reject.clone(), &Holder::new("early"))
                    .await;
                assert!(
                    matches!(refused, Err(StorageError::Rejected { .. })),
                    "a renewed lease is not taken over"
                );
            }
            let row =
                thread.partition(PARTITION_KEY).info().await.unwrap().expect("the partition_key");
            assert!(row.lease_holder.as_deref().is_some_and(|h| h.starts_with("dying/")));
            thread.close().await;
            storage.close().await;
        });
    });
    assert!(!out.committed);
}

/// Killed at seeded delays around a commit that releases the partition: the commit is all or nothing.
#[test]
fn killed_around_a_commit_is_all_or_nothing() {
    let mut seed: u64 = 0x5eed_c0de;
    let delays: Vec<Duration> = (0..6)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Duration::from_micros((seed >> 33) % 4000)
        })
        .collect();
    let handles: Vec<_> = delays
        .iter()
        .map(|&delay| {
            std::thread::spawn(move || {
                case(Role::CommitRelease, |child, _, _| {
                    child.wait_for("acquired");
                    std::thread::sleep(delay);
                })
            })
        })
        .collect();
    for (delay, handle) in delays.iter().zip(handles) {
        let out = handle.join().expect("the case");
        eprintln!(
            "killed {delay:?} after the acquire: committed={} takeover={:?}",
            out.committed, out.takeover
        );
    }
}
