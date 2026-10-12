//! Against agentstore through public-api-server: a staging project's `LIVEKIT_URL` and its key
//! pair. Skipped when they are not set, or when the project
//! cannot create a database.

use std::time::Duration;

use lk_agents_storage::schema::TaskData;
use lk_agents_storage::statements::*;
use lk_agents_storage::{Storage, StorageError, StorageOptions, TokenMinter};
use serde_json::json;

/// agentstore through public-api-server, derived from `LIVEKIT_URL`, and the project's `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`. `None`, and the
/// test skips, when they are not set.
fn options() -> Option<StorageOptions> {
    match (StorageOptions::from_env(), TokenMinter::from_env()) {
        (Ok(opts), Ok(_)) => Some(opts),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("skipped: no agentstore project in the environment ({e})");
            None
        }
    }
}

fn minter() -> TokenMinter {
    TokenMinter::from_env().unwrap()
}

fn storage() -> Option<Storage> {
    let mut opts = options()?;
    opts.lease_ttl = Duration::from_secs(5);
    opts.renew_interval = Duration::from_millis(100);
    Some(Storage::new(opts, minter()).unwrap())
}

fn event(ty: &str, payload: serde_json::Value) -> EventWrite {
    EventWrite { ts_ms: 1_700_000_000_000, r#type: ty.into(), payload }
}

fn acquire(lane: &str, holder: &str, token: &str, run_id: &str) -> AcquireRequest {
    AcquireRequest {
        lane: lane.into(),
        holder: holder.into(),
        lease_token: token.into(),
        run_id: run_id.into(),
        ts_ms: 1_700_000_000_000,
        task_id: String::new(),
        run_started: json!({"task_id": "AST_1", "channel": "a2a"}),
        events_after: 0,
    }
}

#[tokio::test]
async fn waiters_take_the_lane_in_the_order_their_rows_were_submitted() {
    let Some(storage) = storage() else { return };
    let created = storage.databases().create_database(3600).await.unwrap();
    let db = storage.open(&created.database_id).await.unwrap();
    let lane = "ordered";
    // the lane exists before any row does
    db.acquire(&acquire(lane, "w0", "t0", "r0")).await.unwrap().expect("acquired");
    assert!(db.release(lane, "t0").await.unwrap());
    for (task_id, message_id) in [("AST_first", "m1"), ("AST_second", "m2")] {
        db.insert_submitted_task(&submitted(lane, task_id, message_id)).await.unwrap();
    }

    // the second message's run cannot take the free lane while the first still waits
    let second = AcquireRequest { task_id: "AST_second".into(), ..acquire(lane, "w2", "t2", "r2") };
    assert!(db.acquire(&second).await.unwrap().is_none(), "the older waiter goes first");
    let first = AcquireRequest { task_id: "AST_first".into(), ..acquire(lane, "w1", "t1", "r1") };
    assert!(db.acquire(&first).await.unwrap().is_some());
    assert_eq!(db.pickup_task(lane, "AST_first").await.unwrap().as_deref(), Some("submitted"));
    assert!(db.release(lane, "t1").await.unwrap());

    // the first row is working now: the second is the oldest waiter and takes the lane
    assert!(db.acquire(&second).await.unwrap().is_some());
    db.close();
    storage.databases().delete_database(&created.database_id).await.unwrap();
}

fn submitted(lane: &str, task_id: &str, message_id: &str) -> SubmittedTask {
    SubmittedTask {
        lane: lane.into(),
        task_id: task_id.into(),
        message_id: Some(message_id.into()),
        name: None,
        data: TaskData {
            history: vec![json!({"messageId": message_id, "role": "user", "parts": []})],
            ..TaskData::default()
        },
        internal: None,
    }
}

async fn assert_lease_ttl(db: &ThreadDb, lane: &str, ttl: Duration) {
    let rows = db
        .conn()
        .query_all("SELECT lease_until - unixepoch() FROM lanes WHERE lane = ?", vec![lane.into()])
        .await
        .unwrap();
    let remaining = rows.rows[0][0].as_i64().unwrap();
    let seconds = ttl.as_secs() as i64;
    assert!((seconds - 1..=seconds).contains(&remaining), "remaining TTL: {remaining}");
}

#[tokio::test]
async fn a_second_holder_acquires_after_a_short_lease_expires_without_renewal() {
    let Some(mut opts) = options() else { return };
    opts.lease_ttl = Duration::from_secs(1);
    let ttl = opts.lease_ttl;
    let storage = Storage::new(opts, minter()).unwrap();
    let created = storage.databases().create_database(3600).await.unwrap();
    let first = storage.open(&created.database_id).await.unwrap();
    let lane = "short-lease";
    let started = tokio::time::Instant::now();
    first.acquire(&acquire(lane, "w1", "t1", "r1")).await.unwrap().expect("acquired");
    let second = storage.open(&created.database_id).await.unwrap();
    assert_lease_ttl(&first, lane, ttl).await;
    assert!(second.acquire(&acquire(lane, "w2", "t2", "r2")).await.unwrap().is_none());

    // No renewal task is started. The integer clock and strict expiry comparison can add
    // one second to the TTL; allow another second for scheduling and database requests.
    let loaded = tokio::time::timeout(ttl + Duration::from_secs(2), async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if let Some(loaded) = second.acquire(&acquire(lane, "w2", "t2", "r2")).await.unwrap() {
                break loaded;
            }
        }
    })
    .await
    .expect("the short lease expired");
    assert!(started.elapsed() >= ttl);
    assert_eq!(loaded.lane.lease_holder.as_deref(), Some("w2"));
    assert!(!first.renew(lane, "w1", "t1").await.unwrap());
    first.close();
    second.close();
    storage.databases().delete_database(&created.database_id).await.unwrap();
}

#[tokio::test]
async fn a_lapsed_lease_nobody_took_renews_and_a_released_one_does_not() {
    let Some(mut opts) = options() else { return };
    opts.lease_ttl = Duration::from_secs(1);
    let storage = Storage::new(opts, minter()).unwrap();
    let created = storage.databases().create_database(3600).await.unwrap();
    let db = storage.open(&created.database_id).await.unwrap();
    let lane = "stalled-holder";
    db.acquire(&acquire(lane, "w1", "t1", "r1")).await.unwrap().expect("acquired");

    // The holder stalled past the TTL (a slow database, a paused process) and nobody took the
    // lane: its token still stands, so the late renewal keeps the lane.
    tokio::time::sleep(Duration::from_millis(2100)).await;
    let row = db.lane(lane).await.unwrap().unwrap();
    assert!(row.lease_until <= unix_now(), "the lease lapsed");
    assert!(db.renew(lane, "w1", "t1").await.unwrap(), "renewed on the standing token");
    let row = db.lane(lane).await.unwrap().unwrap();
    assert!(row.lease_until > unix_now(), "the lease is live again");

    // A release clears the token: nothing renews a released lane.
    assert!(db.release(lane, "t1").await.unwrap());
    assert!(!db.renew(lane, "w1", "t1").await.unwrap());
    assert_eq!(db.lane(lane).await.unwrap().unwrap().lease_token, None);
    db.close();
    storage.databases().delete_database(&created.database_id).await.unwrap();
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after the epoch")
        .as_secs() as i64
}

#[tokio::test]
async fn a_lane_runs_through_the_lease_the_commit_and_the_inbox() {
    let Some(storage) = storage() else { return };
    // A server that cannot hand out a database (a stale dev instance whose Scylla session died)
    // is not the agent-db this test is for; say so and skip rather than fail.
    let created = match storage.databases().create_database(3600).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipped: the project cannot create a database: {e}");
            return;
        }
    };
    let db = storage.open(&created.database_id).await.unwrap();
    let lane = "front-desk";

    // A database with no schema reads as empty.
    assert!(db.history(lane).await.unwrap().is_empty());
    assert!(db.lanes().await.unwrap().is_empty());

    // First acquire: the DDL, the lane row, the lease; nothing loaded yet.
    let loaded = db.acquire(&acquire(lane, "w1", "t1", "r1")).await.unwrap().expect("acquired");
    assert_lease_ttl(&db, lane, storage.options().lease_ttl).await;
    assert_eq!(loaded.version, 0);
    assert!(!loaded.already_committed);
    assert!(loaded.items.is_empty());
    assert_eq!(loaded.lane.lease_holder.as_deref(), Some("w1"));
    assert_eq!(loaded.events.len(), 1, "run_started");
    assert_eq!(loaded.events[0].r#type, "run_started");

    // A second holder fails the lease and records itself as waiting; the holder sees it.
    assert!(db.acquire(&acquire(lane, "w2", "t2", "r2")).await.unwrap().is_none());
    assert!(db.set_waiter(lane, "w2").await.unwrap());
    assert_eq!(db.waiter(lane).await.unwrap().as_deref(), Some("w2"));
    // The same holder re-enters its own lease.
    assert!(db.acquire(&acquire(lane, "w1", "t1b", "r1")).await.unwrap().is_some());
    assert!(db.renew(lane, "w1", "t1b").await.unwrap());
    assert_lease_ttl(&db, lane, storage.options().lease_ttl).await;
    assert!(!db.renew(lane, "w1", "t1").await.unwrap(), "the old token is gone");

    // The early submitted row (plan 3.3.4), before the lease would be tried under the other
    // policies: once for the task, nothing for a retry of it or of its messageId.
    let early = SubmittedTask {
        lane: lane.into(),
        task_id: "AST_1".into(),
        message_id: Some("m0".into()),
        name: None,
        data: TaskData { history: vec![json!({"messageId": "m0"})], ..TaskData::default() },
        internal: None,
    };
    assert!(db.insert_submitted_task(&early).await.unwrap());
    assert!(!db.insert_submitted_task(&early).await.unwrap());
    assert!(
        !db.insert_submitted_task(&SubmittedTask { task_id: "AST_1b".into(), ..early.clone() })
            .await
            .unwrap(),
        "a retried messageId has its row already"
    );
    let submitted = db.tasks(lane, Some("submitted"), 50).await.unwrap();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].seq, 1);
    assert_eq!(submitted[0].data, json!({"history": [{"messageId": "m0"}]}));
    assert_eq!(submitted[0].internal, None);

    // The fenced commit: items, an event, the task row to its terminal state, the snapshots;
    // release.
    let commit = CommitRequest {
        lane: lane.into(),
        run_id: "r1".into(),
        fence: Some(Fence { version: 0, lease_token: "t1b".into() }),
        items: vec![
            ItemWrite {
                id: "i1".into(),
                message_id: Some("m0".into()),
                payload: json!({"type": "message", "role": "user", "content": ["hi"]}),
            },
            ItemWrite {
                id: "i2".into(),
                message_id: None,
                payload: json!({"type": "message", "role": "assistant", "content": ["hello"]}),
            },
        ],
        delete_items: vec![],
        events: vec![event("agent_state_changed", json!({"state": "thinking"}))],
        tasks: vec![TaskWrite {
            task_id: "AST_1".into(),
            status: "completed".into(),
            message_id: Some("m0".into()),
            name: None,
            status_message: None,
            data: TaskData {
                artifacts: Some(vec![json!({"parts": [{"text": "hello"}]})]),
                ..TaskData::default()
            },
            internal: None,
        }],
        state: Some(json!({"patient": "Ann"})),
        agent_stack: Some(json!([{"name": "front-desk", "args": null}])),
        drained_id: None,
        run_ended: event("run_ended", json!({"outcome": "completed"})),
        after: LeaseAfter::Release,
    };
    let r = db.commit(&commit).await.unwrap();
    assert_eq!(r.version, 1);
    let done = db.tasks(lane, None, 50).await.unwrap();
    assert_eq!(done.len(), 1, "the upsert landed on the early row");
    assert_eq!((done[0].status.as_str(), done[0].seq), ("completed", 1));
    assert_eq!(done[0].data, json!({"artifacts": [{"parts": [{"text": "hello"}]}]}));
    assert!(!db.renew(lane, "w1", "t1b").await.unwrap(), "released: nothing to renew");
    // The same commit again is a conflict: the version moved.
    assert!(matches!(
        db.commit(&commit).await,
        Err(StorageError::VersionConflict { expected: 0, .. })
    ));

    // The next holder loads what was committed; a retried run_id is already committed.
    let loaded = db
        .acquire(&acquire(lane, "w2", "t2", "r2"))
        .await
        .unwrap()
        .expect("the lease was released");
    assert_eq!(loaded.version, 1);
    assert_eq!(
        loaded.items.iter().map(|i| (i.seq, i.id.as_str())).collect::<Vec<_>>(),
        vec![(1, "i1"), (2, "i2")]
    );
    assert_eq!(loaded.lane.state, Some(json!({"patient": "Ann"})));
    assert_eq!(loaded.lane.waiter, None, "the waiter that acquired cleared itself");
    assert!(loaded.events.iter().any(|e| e.r#type == "run_ended" && e.run_id == "r1"));
    db.run_ended(&RunEndedRequest {
        lane: lane.into(),
        run_id: "r2".into(),
        lease_token: "t2".into(),
        ts_ms: 1,
        payload: json!({"outcome": "failed", "reason": "storage-unavailable"}),
    })
    .await
    .unwrap();
    let retried = db.acquire(&acquire(lane, "w1", "t3", "r1")).await.unwrap().expect("acquired");
    assert!(retried.already_committed);
    // Two run_started for r1: the first acquire and the re-entry above; none for this retry.
    assert_eq!(
        retried.events.iter().filter(|e| e.run_id == "r1" && e.r#type == "run_started").count(),
        2
    );
    assert!(db.release(lane, "t3").await.unwrap());

    // The inbox: receive takes the lease when it is free, a retry is a duplicate.
    let receive = ReceiveRequest {
        lane: lane.into(),
        holder: "w3".into(),
        lease_token: "t4".into(),
        task_id: "AST_2".into(),
        message_id: "m1".into(),
        message: json!({"messageId": "m1", "parts": [{"text": "are you open?"}]}),
        name: None,
        data: TaskData::default(),
        internal: None,
    };
    let first = db.receive(&receive).await.unwrap();
    assert_lease_ttl(&db, lane, storage.options().lease_ttl).await;
    let Receive::Received(Received { id, acquired: Some(1) }) = first else { panic!("{first:?}") };
    // A retried message/send: a new stream, hence a new task id, the same messageId.
    let retry = ReceiveRequest {
        task_id: "AST_2b".into(),
        holder: "w6".into(),
        lease_token: "t6".into(),
        ..receive.clone()
    };
    assert_eq!(
        db.receive(&retry).await.unwrap(),
        Receive::Duplicate(Duplicate { task_id: Some("AST_2".into()) })
    );
    let pending = db.inbox_pending(lane).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, id);
    assert_eq!(pending[0].message["messageId"], "m1");
    let check = db.drain_check(lane, "t4").await.unwrap();
    assert_eq!(check, DrainCheck { pending: 1, released: false });
    let drained = CommitRequest {
        lane: lane.into(),
        run_id: "r4".into(),
        fence: Some(Fence { version: 1, lease_token: "t4".into() }),
        items: vec![],
        delete_items: vec!["i2".into()],
        events: vec![],
        tasks: vec![],
        state: None,
        agent_stack: None,
        drained_id: Some(id),
        run_ended: event("run_ended", json!({"outcome": "completed"})),
        after: LeaseAfter::Renew,
    };
    assert_eq!(db.commit(&drained).await.unwrap().version, 2);
    assert_lease_ttl(&db, lane, storage.options().lease_ttl).await;
    assert!(db.renew(lane, "w3", "t4").await.unwrap(), "the renew variant kept the lease");
    assert_eq!(
        db.drain_check(lane, "t4").await.unwrap(),
        DrainCheck { pending: 0, released: true }
    );
    assert_eq!(db.history(lane).await.unwrap().len(), 1, "i2 deleted");

    // Tasks: the submitted row from the receive, the completed one from the commit; an orphan.
    let tasks = db.tasks(lane, None, 50).await.unwrap();
    assert_eq!(
        tasks.iter().map(|t| t.task_id.as_str()).collect::<Vec<_>>(),
        vec!["AST_2", "AST_1"]
    );
    assert_eq!(db.tasks(lane, Some("submitted"), 50).await.unwrap().len(), 1);
    assert_eq!(db.pickup_task(lane, "AST_2").await.unwrap().as_deref(), Some("submitted"));
    assert_eq!(db.pickup_task(lane, "AST_2").await.unwrap().as_deref(), Some("working"));
    assert_eq!(db.orphans(lane).await.unwrap().len(), 1);
    assert!(db.mark_orphan_failed(lane, "AST_2", &json!({"reason": "worker-lost"})).await.unwrap());
    assert!(db.orphans(lane).await.unwrap().is_empty());

    // The journal, whole, and the lanes.
    let events = db.events(0, 100).await.unwrap();
    assert!(events.len() >= 5);
    assert_eq!(db.lane_events(lane, events[2].id).await.unwrap().len(), events.len() - 3);
    assert_eq!(db.lanes().await.unwrap().len(), 1);

    // A renewal timer over a lease that is released underneath it reports the loss.
    let held = db.acquire(&acquire(lane, "w5", "t5", "r5")).await.unwrap().expect("acquired");
    assert_eq!(held.version, 2);
    let lease = db.start_renewal(lane, "w5", "t5");
    assert!(db.release(lane, "t5").await.unwrap());
    let lost = tokio::time::timeout(storage.options().renew_interval * 5, lease.lost())
        .await
        .expect("the timer noticed");
    assert!(matches!(lost, StorageError::Unavailable(_)), "{lost}");
    lease.stop();

    // Reopening reads the schema version; a closed connection is unavailable.
    let again = storage.open(&created.database_id).await.unwrap();
    assert_eq!(again.history(lane).await.unwrap().len(), 1);
    again.close();
    assert!(matches!(again.history(lane).await, Err(StorageError::Unavailable(_))));
    db.close();
    storage.databases().delete_database(&created.database_id).await.unwrap();
}

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
    let thread = threads.create_thread("", Default::default(), 0).await.unwrap();
    assert!(thread.thread_id.starts_with("AT_"));
    assert!(thread.database_id.starts_with("DB_"));
    let again = threads.get_thread(&thread.thread_id).await.unwrap();
    assert_eq!(again.database_id, thread.database_id);
    threads.delete_thread(&thread.thread_id).await.unwrap();
    assert!(matches!(
        threads.get_thread(&thread.thread_id).await,
        Err(StorageError::ThreadNotFound(_))
    ));
}

#[tokio::test]
async fn two_handles_opened_on_a_fresh_database_before_any_write_both_commit() {
    let Some(storage) = storage() else { return };
    let created = storage.databases().create_database(3600).await.unwrap();
    // both connections open before either writes: both find no schema
    let first = storage.open(&created.database_id).await.unwrap();
    let second = storage.open(&created.database_id).await.unwrap();
    // the first creates the schema in its acquire batch; the second's batch would have tried to
    // create it again
    first.acquire(&acquire("desk", "w1", "t1", "r1")).await.unwrap().expect("first acquired");
    second.acquire(&acquire("expert", "w2", "t2", "r2")).await.unwrap().expect("second acquired");
    let commit = |lane: &str, run: &str, token: &str, item: &str| CommitRequest {
        lane: lane.into(),
        run_id: run.into(),
        fence: Some(Fence { version: 0, lease_token: token.into() }),
        items: vec![ItemWrite {
            id: item.into(),
            message_id: None,
            payload: json!({"type": "message", "role": "assistant", "content": [lane]}),
        }],
        delete_items: vec![],
        events: vec![],
        tasks: vec![],
        state: None,
        agent_stack: None,
        drained_id: None,
        run_ended: event("run_ended", json!({"outcome": "completed"})),
        after: LeaseAfter::Release,
    };
    first.commit(&commit("desk", "r1", "t1", "i1")).await.unwrap();
    second.commit(&commit("expert", "r2", "t2", "i2")).await.unwrap();
    // each reads the other's lane through its own connection
    let desk = second.history("desk").await.unwrap();
    let expert = first.history("expert").await.unwrap();
    assert_eq!(desk.len(), 1, "the second handle reads the first's lane");
    assert_eq!(expert.len(), 1, "the first handle reads the second's lane");
    first.close();
    second.close();
}
