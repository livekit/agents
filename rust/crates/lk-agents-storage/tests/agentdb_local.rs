//! Against agentstore through public-api-server: a staging project's `LIVEKIT_URL` and its key
//! pair. Skipped when they are not set, or when the project
//! cannot create a database.

use std::time::Duration;

use livekit_protocol::agent::chat_context::chat_item;
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{ChatMessage, ChatRole};
use lk_agents_storage::db::*;
use lk_agents_storage::run::TaskState;
use lk_agents_storage::schema::{ItemRow, RunStarted, TaskData};
mod common;

use lk_agents_storage::{Storage, StorageError, Thread};
use serde_json::json;

fn user_message(message_id: &str, text: &str) -> a2a::Message {
    a2a::Message {
        message_id: message_id.into(),
        ..a2a::Message::new(a2a::Role::User, vec![a2a::Part::text(text)])
    }
}

fn storage() -> Option<Storage> {
    let (mut opts, minter) = common::project()?;
    opts.lease_ttl = Duration::from_secs(5);
    opts.renew_interval = Duration::from_millis(100);
    Some(Storage::new(opts, minter).unwrap())
}

fn event(ty: &str, payload: serde_json::Value) -> EventWrite {
    EventWrite { ts_ms: 1_700_000_000_000, r#type: ty.into(), payload }
}

fn acquire<'a>(holder: &'a str, run_id: &'a str) -> Acquire<'a> {
    Acquire {
        holder,
        run_id,
        task_id: "",
        run_started: RunStarted {
            task_id: Some("AST_1".into()),
            channel: Some("a2a".into()),
            ..RunStarted::default()
        },
        submit: None,
        pickup: false,
        tasks_limit: 0,
    }
}

/// One connection to the database scoped to `lane`, and the thread to close it with.
async fn open(storage: &Storage, database_id: &str, lane: &str) -> (Thread, LaneDb) {
    let thread = storage.open(database_id, "test").await.unwrap();
    let db = thread.lane(lane).db().clone();
    (thread, db)
}

fn message_item(id: &str, role: ChatRole, text: &str, message_id: Option<&str>) -> ItemRow {
    let item = ItemRow::new(chat_item::Item::Message(ChatMessage {
        id: id.into(),
        role: role as i32,
        content: vec![ChatContent { payload: Some(Payload::Text(text.into())) }],
        ..Default::default()
    }));
    ItemRow { message_id: message_id.map(str::to_owned), ..item }
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
/// Whether a renewal of `lease` fails within a few intervals: a lease that is no longer its own.
async fn renewal_fails(storage: &Storage, lease: &Lease) -> bool {
    lease.keep_alive();
    let lost = tokio::time::timeout(storage.options().renew_interval * 5, lease.lost()).await;
    lease.stop();
    lost.is_ok()
}

#[tokio::test]
async fn waiters_take_the_lane_in_the_order_their_rows_were_submitted() {
    let Some(storage) = storage() else { return };
    let created = common::create_database(&storage).await.unwrap();
    let (thread, db) = open(&storage, &created.database_id, "ordered").await;
    // the lane exists before any row does
    let lease = db.acquire(&acquire("w0", "r0")).await.unwrap().expect("acquired").lease;
    lease.release(None).await.unwrap();
    for (task_id, message_id) in [("AST_first", "m1"), ("AST_second", "m2")] {
        assert!(db.submit(&submitted(task_id, message_id)).await.unwrap());
    }

    // the second message's run cannot take the free lane while the first still waits
    let second = Acquire { task_id: "AST_second", ..acquire("w2", "r2") };
    assert!(db.acquire(&second).await.unwrap().is_none(), "the older waiter goes first");
    let first = Acquire { task_id: "AST_first", ..acquire("w1", "r1") };
    let lease = db.acquire(&first).await.unwrap().expect("acquired").lease;
    assert_eq!(db.pickup("AST_first").await.unwrap(), Some(TaskState::Submitted));
    lease.release(None).await.unwrap();

    // the first row is working now: the second is the oldest waiter and takes the lane
    assert!(db.acquire(&second).await.unwrap().is_some());
    thread.close();
    storage.databases().delete(&created.database_id).await.unwrap();
}

fn submitted(task_id: &str, message_id: &str) -> TaskWrite {
    TaskWrite {
        task_id: task_id.into(),
        status: TaskState::Submitted,
        message_id: Some(message_id.into()),
        name: None,
        data: TaskData { history: vec![user_message(message_id, "")], ..TaskData::default() },
        ask: None,
    }
}

/// The lease's remaining TTL, read from the lane row against the local clock (one second of
/// slack for the skew between them).
async fn assert_lease_ttl(db: &LaneDb, ttl: Duration) {
    let info = db.info().await.unwrap().expect("a lane row");
    let remaining = info.lease_until - unix_now();
    let seconds = ttl.as_secs() as i64;
    assert!((seconds - 2..=seconds + 1).contains(&remaining), "remaining TTL: {remaining}");
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
#[tokio::test]
async fn a_second_holder_acquires_after_a_short_lease_expires_without_renewal() {
    let Some((mut opts, minter)) = common::project() else { return };
    opts.lease_ttl = Duration::from_secs(1);
    let ttl = opts.lease_ttl;
    let storage = Storage::new(opts, minter).unwrap();
    let created = common::create_database(&storage).await.unwrap();
    let lane = "short-lease";
    let (first_thread, first) = open(&storage, &created.database_id, lane).await;
    let started = tokio::time::Instant::now();
    let lease = first.acquire(&acquire("w1", "r1")).await.unwrap().expect("acquired").lease;
    let (second_thread, second) = open(&storage, &created.database_id, lane).await;
    assert_lease_ttl(&first, ttl).await;
    assert!(second.acquire(&acquire("w2", "r2")).await.unwrap().is_none());

    // No renewal task is started. The integer clock and strict expiry comparison can add
    // one second to the TTL; allow another second for scheduling and database requests.
    let loaded = tokio::time::timeout(ttl + Duration::from_secs(2), async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if let Some(loaded) = second.acquire(&acquire("w2", "r2")).await.unwrap() {
                break loaded;
            }
        }
    })
    .await
    .expect("the short lease expired");
    assert!(started.elapsed() >= ttl);
    assert_eq!(loaded.lane.lease_holder.as_deref(), Some("w2"));
    assert!(renewal_fails(&storage, &lease).await, "the first holder's lease is gone");
    first_thread.close();
    second_thread.close();
    storage.databases().delete(&created.database_id).await.unwrap();
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
#[tokio::test]
async fn a_lapsed_lease_nobody_took_renews_and_a_released_one_does_not() {
    let Some((mut opts, minter)) = common::project() else { return };
    opts.lease_ttl = Duration::from_secs(1);
    opts.renew_interval = Duration::from_millis(200);
    let storage = Storage::new(opts, minter).unwrap();
    let created = common::create_database(&storage).await.unwrap();
    let (thread, db) = open(&storage, &created.database_id, "stalled-holder").await;
    let lease = db.acquire(&acquire("w1", "r1")).await.unwrap().expect("acquired").lease;

    // The holder stalled past the TTL (a slow database, a paused process) and nobody took the
    // lane: its token still stands, so the late renewal keeps the lane.
    tokio::time::sleep(Duration::from_millis(2100)).await;
    let row = db.info().await.unwrap().unwrap();
    assert!(row.lease_until <= unix_now(), "the lease lapsed");
    lease.keep_alive();
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert!(!lease.is_lost(), "renewed on the standing token");
    let row = db.info().await.unwrap().unwrap();
    assert!(row.lease_until > unix_now(), "the lease is live again");
    lease.stop();

    // A release clears the token: nothing renews a released lane.
    lease.release(None).await.unwrap();
    assert!(renewal_fails(&storage, &lease).await);
    assert_eq!(db.info().await.unwrap().unwrap().lease_until, 0);
    thread.close();
    storage.databases().delete(&created.database_id).await.unwrap();
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after the epoch")
        .as_secs() as i64
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
#[tokio::test]
async fn a_lane_runs_through_the_lease_the_commit_and_the_inbox() {
    let Some(storage) = storage() else { return };
    // A server that cannot hand out a database (a stale dev instance whose Scylla session died)
    // is not the agent-db this test is for; say so and skip rather than fail.
    let created = match common::create_database(&storage).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: the project cannot create a database: {e}");
            return;
        }
    };
    let ttl = storage.options().lease_ttl;
    let (thread, db) = open(&storage, &created.database_id, "front-desk").await;

    // A database with no schema reads as empty.
    assert!(db.history().await.unwrap().is_empty());
    assert!(thread.lanes().await.unwrap().is_empty());

    // First acquire: the DDL, the lane row, the lease; nothing loaded yet.
    let loaded = db.acquire(&acquire("w1", "r1")).await.unwrap().expect("acquired");
    assert_lease_ttl(&db, ttl).await;
    assert_eq!(loaded.lease.version(), 0);
    assert!(!loaded.already_committed);
    assert!(loaded.items.is_empty());
    assert_eq!(loaded.lane.lease_holder.as_deref(), Some("w1"));
    let journal = db.events(0).await.unwrap();
    assert_eq!(journal.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(), vec!["run_started"]);

    // A second holder fails the lease and records itself as waiting; the holder sees it.
    assert!(db.acquire(&acquire("w2", "r2")).await.unwrap().is_none());
    assert!(db.set_waiter("w2").await.unwrap());
    assert_eq!(db.info().await.unwrap().unwrap().waiter.as_deref(), Some("w2"));
    // The same holder re-enters its own lease, under a new token.
    let lease = db.acquire(&acquire("w1", "r1")).await.unwrap().expect("re-entered").lease;
    lease.keep_alive();
    assert!(renewal_fails(&storage, &loaded.lease).await, "the old token is gone");
    assert!(!lease.is_lost(), "the new token renews");
    assert_lease_ttl(&db, ttl).await;

    // The early submitted row, before the lease would be tried under the other
    // policies: once for the task, nothing for a retry of it or of its messageId.
    let early = TaskWrite {
        task_id: "AST_1".into(),
        status: TaskState::Submitted,
        message_id: Some("m0".into()),
        name: None,
        data: TaskData { history: vec![user_message("m0", "")], ..TaskData::default() },
        ask: None,
    };
    assert!(db.submit(&early).await.unwrap());
    assert!(!db.submit(&early).await.unwrap());
    assert!(
        !db.submit(&TaskWrite { task_id: "AST_1b".into(), ..early.clone() }).await.unwrap(),
        "a retried messageId has its row already"
    );
    let submitted = db.tasks(Some(TaskState::Submitted), 50).await.unwrap();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].seq, 1);
    assert_eq!(submitted[0].data, early.data);
    assert_eq!(submitted[0].internal, None);

    // The fenced commit: items, an event, the task row to its terminal state, the snapshots;
    // release.
    let artifacts = TaskData {
        artifacts: Some(vec![a2a::Artifact {
            artifact_id: "art_1".into(),
            name: None,
            description: None,
            parts: vec![a2a::Part::text("hello")],
            metadata: None,
            extensions: None,
        }]),
        ..TaskData::default()
    };
    let commit = Commit {
        run_id: "r1".into(),
        items: vec![
            message_item("i1", ChatRole::User, "hi", Some("m0")),
            message_item("i2", ChatRole::Assistant, "hello", None),
        ],
        events: vec![event("agent_state_changed", json!({"state": "thinking"}))],
        tasks: vec![TaskWrite {
            status: TaskState::Completed,
            data: artifacts.clone(),
            ..early.clone()
        }],
        state: Some(json!({"patient": "Ann"})),
        run_ended: event("run_ended", json!({"outcome": "completed"})),
        ..Commit::default()
    };
    assert_eq!(lease.commit(&commit, false).await.unwrap().version, 1);
    let done = db.tasks(None, 50).await.unwrap();
    assert_eq!(done.len(), 1, "the upsert landed on the early row");
    assert_eq!((done[0].status.as_str(), done[0].seq), ("completed", 1));
    assert_eq!(done[0].data, artifacts);
    let lost = tokio::time::timeout(storage.options().renew_interval * 5, lease.lost()).await;
    assert!(lost.is_ok(), "released: nothing to renew");
    lease.stop();
    // The same commit again is a conflict: the version moved.
    assert!(matches!(
        lease.commit(&commit, false).await,
        Err(StorageError::VersionConflict { expected: 0, .. })
    ));

    // The next holder loads what was committed; a retried run_id is already committed.
    let loaded = db.acquire(&acquire("w2", "r2")).await.unwrap().expect("the lease was released");
    assert_eq!(loaded.lease.version(), 1);
    assert_eq!(loaded.items.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), vec!["i1", "i2"]);
    assert_eq!(loaded.items[0].message_id.as_deref(), Some("m0"));
    assert_eq!(loaded.lane.state, Some(json!({"patient": "Ann"})));
    assert_eq!(loaded.lane.waiter, None, "the waiter that acquired cleared itself");
    let journal = db.events(0).await.unwrap();
    assert!(journal.iter().any(|e| e.r#type == "run_ended" && e.run_id == "r1"));
    let ended = event("run_ended", json!({"outcome": "failed", "reason": "storage-unavailable"}));
    loaded.lease.release(Some(("r2", &ended))).await.unwrap();
    let retried = db.acquire(&acquire("w1", "r1")).await.unwrap().expect("acquired");
    assert!(retried.already_committed);
    // Two run_started for r1: the first acquire and the re-entry above; none for this retry.
    let journal = db.events(0).await.unwrap();
    assert_eq!(journal.iter().filter(|e| e.run_id == "r1" && e.r#type == "run_started").count(), 2);
    assert!(journal.iter().any(|e| e.run_id == "r2" && e.r#type == "run_ended"));
    retried.lease.release(None).await.unwrap();

    // The inbox: receive takes the lease when it is free, a retry is a duplicate.
    let message = user_message("m1", "are you open?");
    let first = db.receive("w3", "AST_2", "m1", &message, &TaskData::default()).await.unwrap();
    assert_lease_ttl(&db, ttl).await;
    let Receive::Received(Some(inbox_held)) = first else { panic!("not received under a lease") };
    assert_eq!(inbox_held.version(), 1);
    // A retried message/send: a new stream, hence a new task id, the same messageId.
    let retry = db.receive("w6", "AST_2b", "m1", &message, &TaskData::default()).await.unwrap();
    let Receive::Duplicate(task_id) = retry else { panic!("not a duplicate") };
    assert_eq!(task_id.as_deref(), Some("AST_2"));
    let pending = db.inbox(true).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].task_id, "AST_2");
    assert_eq!(pending[0].message, message);
    let id = pending[0].id;
    assert!(!inbox_held.release_if_drained().await.unwrap(), "a row is pending");
    let drained = Commit {
        run_id: "r4".into(),
        delete_items: vec!["i2".into()],
        drained_id: Some(id),
        run_ended: event("run_ended", json!({"outcome": "completed"})),
        ..Commit::default()
    };
    assert_eq!(inbox_held.commit(&drained, true).await.unwrap().version, 2);
    assert_lease_ttl(&db, ttl).await;
    inbox_held.keep_alive();
    tokio::time::sleep(storage.options().renew_interval * 3).await;
    assert!(!inbox_held.is_lost(), "the renew variant kept the lease");
    inbox_held.stop();
    assert!(db.inbox(true).await.unwrap().is_empty());
    assert!(inbox_held.release_if_drained().await.unwrap(), "drained: released");
    assert_eq!(db.info().await.unwrap().unwrap().lease_until, 0);
    assert_eq!(db.history().await.unwrap().len(), 1, "i2 deleted");

    // Tasks: the submitted row from the receive, the completed one from the commit; an orphan.
    let tasks = db.tasks(None, 50).await.unwrap();
    assert_eq!(
        tasks.iter().map(|t| t.task_id.as_str()).collect::<Vec<_>>(),
        vec!["AST_2", "AST_1"]
    );
    assert_eq!(db.tasks(Some(TaskState::Submitted), 50).await.unwrap().len(), 1);
    assert_eq!(db.pickup("AST_2").await.unwrap(), Some(TaskState::Submitted));
    assert_eq!(db.pickup("AST_2").await.unwrap(), Some(TaskState::Working));
    assert_eq!(db.tasks(Some(TaskState::Working), 50).await.unwrap().len(), 1);
    assert!(db.fail_orphan("AST_2", &user_message("m9", "worker-lost")).await.unwrap());
    assert!(db.tasks(Some(TaskState::Working), 50).await.unwrap().is_empty());
    assert_eq!(db.task("AST_2").await.unwrap().unwrap().status, TaskState::Failed);

    // The journal, whole, and the lanes.
    let events = thread.events(0, 100).await.unwrap();
    assert!(events.len() >= 5);
    assert_eq!(db.events(events[2].id).await.unwrap().len(), events.len() - 3);
    assert_eq!(thread.lanes().await.unwrap().len(), 1);

    // A renewal timer over a lease that is released underneath it reports the loss.
    let lease = db.acquire(&acquire("w5", "r5")).await.unwrap().expect("acquired").lease;
    assert_eq!(lease.version(), 2);
    lease.keep_alive();
    lease.release(None).await.unwrap();
    let lost = tokio::time::timeout(storage.options().renew_interval * 5, lease.lost())
        .await
        .expect("the timer noticed");
    assert!(matches!(lost, StorageError::Unavailable(_)), "{lost}");
    assert!(lease.is_lost());
    lease.stop();

    // Reopening reads the schema version; a closed connection is unavailable.
    let again = storage.open(&created.database_id, "again").await.unwrap();
    assert_eq!(again.lane("front-desk").history().await.unwrap().len(), 1);
    again.close();
    assert!(matches!(again.lane("front-desk").history().await, Err(StorageError::Unavailable(_))));
    thread.close();
    storage.databases().delete(&created.database_id).await.unwrap();
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
#[tokio::test]
async fn a_stream_is_created_appended_read_and_closed() {
    let Some(storage) = storage() else { return };
    let streams = storage.streams();
    let info = streams.create(None, None).await.unwrap();
    assert!(info.stream_id.contains("ST_"), "{}", info.stream_id);
    assert!(info.expires_at > 0, "a fixed TTL, the default day");
    streams.append(&info.stream_id, br#"{"n":1}"#.to_vec());
    streams.append(&info.stream_id, br#"{"n":2}"#.to_vec());
    streams.flush(&info.stream_id).await;
    let mut reader = streams.read(&info.stream_id, None, false);
    let mut got = Vec::new();
    while let Some(r) = reader.next().await.unwrap() {
        got.push(String::from_utf8(r).unwrap());
    }
    assert_eq!(got, vec![r#"{"n":1}"#, r#"{"n":2}"#]);
    let resume_from = reader.next_offset().to_owned();

    // A live reader sees the tail and the close.
    let streams2 = streams.clone();
    let id = info.stream_id.clone();
    let tail = tokio::spawn(async move {
        let mut live = streams2.read(&id, Some(&resume_from), true);
        let mut got = Vec::new();
        while let Some(r) = live.next().await.unwrap() {
            got.push(String::from_utf8(r).unwrap());
        }
        (got, live.is_closed())
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    streams.append_now(&info.stream_id, br#"{"n":3}"#).await.unwrap();
    streams.close(&info.stream_id).await.unwrap();
    let (got, closed) = tokio::time::timeout(Duration::from_secs(10), tail).await.unwrap().unwrap();
    assert_eq!(got, vec![r#"{"n":3}"#]);
    assert!(closed);
    assert!(matches!(
        streams.append_now(&info.stream_id, b"{}").await,
        Err(StorageError::Http { status: 409, .. })
    ));
    streams.delete(&info.stream_id).await.unwrap();
    assert!(matches!(
        streams.append_now(&info.stream_id, b"{}").await,
        Err(StorageError::NotFound(_))
    ));
}

#[tokio::test]
async fn the_threads_service_creates_and_deletes_a_thread() {
    let Some(storage) = storage() else { return };
    let threads = storage.threads();
    let thread = threads.create(Default::default()).await.unwrap();
    assert!(thread.thread_id.starts_with("AT_"));
    assert!(thread.database_id.starts_with("DB_"));
    let again = threads.get(&thread.thread_id).await.unwrap();
    assert_eq!(again.database_id, thread.database_id);
    threads.delete(&thread.thread_id).await.unwrap();
    assert!(matches!(
        threads.get(&thread.thread_id).await.map_err(lk_agents_storage::thread_error),
        Err(StorageError::ThreadNotFound(_))
    ));
}

#[tokio::test]
async fn two_handles_opened_on_a_fresh_database_before_any_write_both_commit() {
    let Some(storage) = storage() else { return };
    let created = common::create_database(&storage).await.unwrap();
    // both connections open before either writes: both find no schema
    let first = storage.open(&created.database_id, "w1").await.unwrap();
    let second = storage.open(&created.database_id, "w2").await.unwrap();
    // the first creates the schema in its acquire batch; the second's batch would have tried to
    // create it again
    let desk = first.lane("desk").db().acquire(&acquire("w1", "r1")).await.unwrap();
    let desk = desk.expect("first acquired").lease;
    let expert = second.lane("expert").db().acquire(&acquire("w2", "r2")).await.unwrap();
    let expert = expert.expect("second acquired").lease;
    let commit = |lane: &str, run: &str, item: &str| Commit {
        run_id: run.into(),
        items: vec![message_item(item, ChatRole::Assistant, lane, None)],
        run_ended: event("run_ended", json!({"outcome": "completed"})),
        ..Commit::default()
    };
    desk.commit(&commit("desk", "r1", "i1"), false).await.unwrap();
    expert.commit(&commit("expert", "r2", "i2"), false).await.unwrap();
    // each reads the other's lane through its own connection
    let desk = second.lane("desk").history().await.unwrap();
    let expert = first.lane("expert").history().await.unwrap();
    assert_eq!(desk.len(), 1, "the second handle reads the first's lane");
    assert_eq!(expert.len(), 1, "the first handle reads the second's lane");
    first.close();
    second.close();
}
