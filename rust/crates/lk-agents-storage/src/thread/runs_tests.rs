//! The run, the shared engine's unit of work, against both storage backends (`storage_common`):
//! lane leases and waiters, the one fenced commit, abandon, closed runs, version conflicts and the
//! append-only concurrent runs.

use std::time::{Duration, Instant};

use lk_agents_abi::storage::journal::JournalEvent;
use lk_agents_storage::TaskWrite;
use lk_agents_storage::testing::{Backend, Fixture};
use lk_agents_storage::thread::run::{ItemRow, RunOutcome, TaskPatch, TaskState};
use lk_agents_storage::thread::schema::TaskData;
use lk_agents_storage::{CommitOptions, RunOptions, Thread};

use livekit_protocol::agent::chat_context::chat_item;
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{ChatMessage, ChatRole};
use lk_agents_storage::StorageError;
use serde_json::json;

async fn waiter(s: &Thread) -> Option<String> {
    s.lane(LANE).info().await.unwrap().expect("a lane row").waiter
}

fn message(id: &str, role: ChatRole, text: &str) -> ItemRow {
    ItemRow::new(chat_item::Item::Message(ChatMessage {
        id: id.into(),
        role: role as i32,
        content: vec![ChatContent { payload: Some(Payload::Text(text.into())) }],
        ..Default::default()
    }))
}

const LANE: &str = "front-desk";

async fn the_second_lane_waits_while_the_first_holds_and_gets_it_after_the_commit(
    backend: Backend,
) {
    let fx = backend.fixture(None).await;
    let s1 = fx.thread("w1").await;
    assert!(s1.holder().starts_with("w1/"));
    assert_eq!(s1.id(), fx.thread_id);

    let mut run1 = s1.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    // The first acquire applies the DDL; the second handle dials once the first run exists, as a
    // worker reached later would.
    let s2 = fx.thread("w2").await;
    assert_ne!(s1.holder(), s2.holder());
    assert_eq!(run1.version(), 0);
    assert!(!run1.closed && run1.lease().is_some());
    assert!(run1.history.is_empty());
    assert_eq!(run1.state, None);
    let journal = s1.journal(0, 100).await.unwrap();
    assert_eq!(journal.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(), vec!["run_started"]);
    assert!(run1.lease().is_some_and(|h| !h.is_lost()));
    assert_eq!(waiter(&s1).await, None);

    // A second holder: refused at once without the wait, and timed out with a short one.
    let lane2 = s2.lane(LANE);
    let refused = lane2
        .open(RunOptions { wait: false, ..RunOptions::new("r2", "AST_2") })
        .await
        .err()
        .expect("lease");
    assert!(matches!(refused, StorageError::Rejected { .. }), "{refused}");
    let started = Instant::now();
    let timed_out = lane2
        .open(RunOptions {
            wait_timeout: Duration::from_millis(250),
            ..RunOptions::new("r2", "AST_2")
        })
        .await
        .err()
        .expect("lease");
    assert!(matches!(timed_out, StorageError::Rejected { .. }), "{timed_out}");
    assert!(started.elapsed() >= Duration::from_millis(250));

    // The waiter under `interrupt`: the holder sees it at its boundary.
    let waiting = tokio::spawn(async move {
        let run = lane2
            .open(RunOptions {
                interrupt: true,
                channel: Some("a2a".into()),
                wait_timeout: Duration::from_secs(10),
                ..RunOptions::new("r2", "AST_2")
            })
            .await
            .unwrap();
        (run, Instant::now())
    });
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(waiter(&s1).await.is_some_and(|holder| holder.starts_with(s2.holder())));

    // The first run writes everything a run writes, and releases.
    run1.history.push(message("i1", ChatRole::User, "hi"));
    run1.history.last_mut().unwrap().message_id = Some("m1".into());
    run1.history.push(message("i2", ChatRole::Assistant, "hello"));
    run1.history.push(message("sys", ChatRole::System, "be nice"));
    run1.state = Some(json!({"patient": "Ann"}));
    run1.events.append(JournalEvent::Error { error: "boom".into() });
    run1.usage.requests = 1;
    run1.usage.tokens.input_tokens = 10;
    let request = run1.writes(CommitOptions::default()).unwrap();
    assert_eq!(request.items.len(), 2, "instructions are not stored");
    assert_eq!(request.items[0].message_id.as_deref(), Some("m1"));
    assert_eq!(
        request.events.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(),
        vec!["conversation_item_added", "conversation_item_added", "error", "usage"]
    );
    assert_eq!(request.events[3].payload["tokens"]["input_tokens"], 10);
    assert_eq!(request.state, Some(json!({"patient": "Ann"})));
    assert_eq!(request.run_ended.payload, json!({"outcome": "completed", "reason": null}));
    let released_at = Instant::now();
    assert_eq!(run1.commit(CommitOptions::default()).await.unwrap().version, 1);
    assert!(matches!(run1.commit(CommitOptions::default()).await, Err(StorageError::Invalid(_))));

    // The waiter acquires within a poll, at the committed version, with everything loaded.
    let (mut run2, acquired_at) = waiting.await.unwrap();
    assert!(acquired_at >= released_at);
    assert!(acquired_at - released_at < Duration::from_millis(1500), "a poll interval at most");
    assert_eq!(run2.version(), 1);
    assert_eq!(run2.history.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), vec!["i1", "i2"]);
    let first = ItemRow { message_id: Some("m1".into()), ..message("i1", ChatRole::User, "hi") };
    assert_eq!(run2.history[0], first);
    assert_eq!(run2.state, Some(json!({"patient": "Ann"})));
    assert_eq!(waiter(&s2).await, None, "the waiter that won cleared its own mark");
    assert!(s2.journal(0, 100).await.unwrap().iter().any(|e| e.r#type == "usage"));

    // An unchanged history writes no item; a changed state is written.
    assert!(run2.writes(CommitOptions::default()).unwrap().items.is_empty());
    let unchanged = run2.writes(CommitOptions::default()).unwrap().state;
    assert_eq!(unchanged, None, "unchanged state is not rewritten");
    run2.history.retain(|i| i.id != "i2");
    run2.state = Some(json!({"patient": "Ann", "seen": true}));
    let request = run2.writes(CommitOptions::default()).unwrap();
    assert_eq!(request.delete_items, vec!["i2".to_owned()]);
    assert_eq!(run2.commit(CommitOptions::default()).await.unwrap().version, 2);

    // The reads outside a run.
    let lane = s1.lane(LANE);
    assert_eq!(lane.history().await.unwrap().len(), 1);
    let runs = lane.runs(10).await.unwrap();
    assert_eq!(
        runs.iter().map(|r| (r.run_id.as_str(), r.outcome)).collect::<Vec<_>>(),
        vec![("r2", Some(RunOutcome::Completed)), ("r1", Some(RunOutcome::Completed))]
    );
    assert_eq!(runs[0].channel.as_deref(), Some("a2a"));
    assert_eq!(runs[0].task_id.as_deref(), Some("AST_2"));
    assert_eq!(lane.runs(1).await.unwrap().len(), 1);
    let info = lane.info().await.unwrap().expect("a lane row");
    assert_eq!((info.version, info.drained_id, info.lease_until), (2, 0, 0));
    assert_eq!(s2.lanes().await.unwrap().len(), 1);
    assert!(s2.journal(0, 100).await.unwrap().len() >= 5);
    assert!(lane.inbox(true).await.unwrap().is_empty());
    fx.cleanup(&[&s1, &s2]).await;
}

async fn a_run_that_aborts_writes_run_ended_and_nothing_else(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.history.push(message("i1", ChatRole::User, "hi"));
    run.state = Some(json!({"x": 1}));
    run.events.append(JournalEvent::Error { error: "boom".into() });
    run.tasks
        .write(
            "AST_t",
            TaskState::Submitted,
            TaskPatch { name: Some("Followup".into()), ..TaskPatch::default() },
        )
        .await
        .unwrap();
    run.abandon(RunOutcome::Failed, "RuntimeError: boom").await.unwrap();

    assert!(lane.history().await.unwrap().is_empty());
    assert!(lane.tasks(None, 50).await.unwrap().is_empty());
    let info = lane.info().await.unwrap().unwrap();
    assert_eq!((info.version, info.lease_until), (0, 0), "released, not bumped");
    let events = s.journal(0, 100).await.unwrap();
    assert_eq!(
        events.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(),
        vec!["run_started", "run_ended"]
    );
    assert_eq!(events[1].payload, json!({"outcome": "failed", "reason": "RuntimeError: boom"}));
    let runs = lane.runs(10).await.unwrap();
    assert_eq!(runs[0].reason.as_deref(), Some("RuntimeError: boom"));

    // The lane is free at once, and the next run sees a clean lane.
    let next =
        lane.open(RunOptions { wait: false, ..RunOptions::new("r2", "AST_2") }).await.unwrap();
    assert!(next.history.is_empty() && next.state.is_none());
    assert_eq!(next.version(), 0);
    drop(next);
    fx.cleanup(&[&s]).await;
}

async fn a_committed_run_id_yields_a_closed_run(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.history.push(message("i1", ChatRole::User, "hi"));
    run.commit(CommitOptions::default()).await.unwrap();

    let mut again = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    assert!(again.closed);
    assert_eq!(again.version(), 1);
    assert_eq!(again.history.len(), 1, "loaded, so a retry can answer from it");
    again.history.push(message("i2", ChatRole::Assistant, "hello"));
    let committed = again.commit(CommitOptions::default()).await.unwrap();
    assert_eq!(committed.version, 1, "no bump");
    assert_eq!(lane.info().await.unwrap().unwrap().version, 1);
    assert_eq!(lane.history().await.unwrap().len(), 1, "a closed run writes nothing");
    let events = s.journal(0, 100).await.unwrap();
    assert_eq!(events.iter().filter(|e| e.r#type == "run_started").count(), 1, "not again");
    assert_eq!(events.iter().filter(|e| e.r#type == "run_ended").count(), 1);
    // Released: a third open takes the lane without waiting.
    let free =
        lane.open(RunOptions { wait: false, ..RunOptions::new("r2", "AST_2") }).await.unwrap();
    assert!(!free.closed);
    drop(free);
    fx.cleanup(&[&s]).await;
}

async fn a_stale_commit_is_a_version_conflict(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s1 = fx.thread("w1").await;
    let mut stale = s1.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    let s2 = fx.thread("w2").await;
    // The lease lapses under the stalled worker (here: released by hand), and another run moves
    // the lane on.
    stale.lease().unwrap().release(None).await.unwrap();
    assert_eq!(s1.lane(LANE).info().await.unwrap().unwrap().lease_until, 0);
    let mut fresh = s2
        .lane(LANE)
        .open(RunOptions { wait: false, ..RunOptions::new("r2", "AST_2") })
        .await
        .unwrap();
    fresh.history.push(message("i1", ChatRole::User, "hi"));
    assert_eq!(fresh.commit(CommitOptions::default()).await.unwrap().version, 1);

    stale.history.push(message("i9", ChatRole::Assistant, "late"));
    let err = stale.commit(CommitOptions::default()).await.expect_err("fenced out");
    assert!(matches!(err, StorageError::VersionConflict { expected: 0, .. }), "{err}");
    assert_eq!(s1.lane(LANE).history().await.unwrap().len(), 1, "the loser wrote nothing");
    fx.cleanup(&[&s1, &s2]).await;
}

async fn concurrent_runs_take_no_lease_and_append(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s1 = fx.thread("w1").await;
    // The schema and the lane row come from a first exclusive run; the concurrent runs then share
    // a lane at version 1 and never move it.
    let mut first = s1.lane(LANE).open(RunOptions::new("r0", "AST_0")).await.unwrap();
    first.commit(CommitOptions::default()).await.unwrap();
    let s2 = fx.thread("w2").await;
    let open = |s: &Thread, run_id: &str, task_id: &str| {
        let lane = s.lane(LANE);
        let opts = RunOptions {
            exclusive: false,
            channel: Some("a2a".into()),
            ..RunOptions::new(run_id, task_id)
        };
        async move { lane.open(opts).await.unwrap() }
    };
    let mut a = open(&s1, "r1", "AST_1").await;
    let mut b = open(&s2, "r2", "AST_2").await;
    assert!(a.lease().is_none());
    a.history.push(message("a1", ChatRole::User, "one"));
    a.state = Some(json!({"ignored": true}));
    b.history.push(message("b1", ChatRole::User, "two"));
    let drained = CommitOptions { drained_id: Some(3), ..CommitOptions::default() };
    let request = a.writes(drained.clone()).unwrap();
    assert_eq!(request.state, None);
    assert_eq!(request.drained_id, None);
    assert_eq!(request.events[0].r#type, "run_started");
    assert_eq!(
        request.events[0].payload,
        json!({"task_id": "AST_1", "stream_id": "AST_1", "channel": "a2a", "exclusive": false})
    );
    let b_version = b.commit(CommitOptions::default()).await.unwrap().version;
    let a_version = a.commit(drained).await.unwrap().version;
    assert_eq!((a_version, b_version), (1, 1));

    let lane = s1.lane(LANE);
    assert_eq!(
        lane.history().await.unwrap().iter().map(|i| i.id.as_str()).collect::<Vec<_>>(),
        vec!["b1", "a1"]
    );
    let info = lane.info().await.unwrap().unwrap();
    assert_eq!((info.version, info.lease_until, info.drained_id), (1, 0, 0));
    let runs = lane.runs(10).await.unwrap();
    assert_eq!(runs.len(), 3);
    assert!(runs.iter().all(|r| r.outcome == Some(RunOutcome::Completed)));
    // A concurrent run has nothing to abandon.
    let mut c = open(&s1, "r3", "AST_3").await;
    c.abandon(RunOutcome::Canceled, "CancelledError").await.unwrap();
    assert_eq!(s1.journal(0, 100).await.unwrap().len(), 8, "two items entered the history");
    fx.cleanup(&[&s1, &s2]).await;
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

fn metadata(key: &str) -> serde_json::Map<String, serde_json::Value> {
    json!({ key: true }).as_object().cloned().unwrap()
}

/// I-6: a terminal row is immutable. A cancel arriving after the commit leaves the completed
/// task as it was.
async fn a_cancel_never_overwrites_a_terminal_task(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.tasks.write("AST_1", TaskState::Completed, TaskPatch::default()).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();

    let after = lane.cancel_task("AST_1", metadata("late")).await.unwrap().expect("the row");
    assert_eq!(after.status, TaskState::Completed);
    assert!(after.data.metadata.is_empty(), "nothing merged into a terminal row");
    assert_eq!(lane.task("AST_1").await.unwrap().unwrap().status, TaskState::Completed);
    fx.cleanup(&[&s]).await;
}

/// I-6: a client's cancel during the run wins: the run's commit lands its history but cannot
/// move the canceled row, in either race order.
async fn a_commit_never_overwrites_a_canceled_task(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s1 = fx.thread("w1").await;
    let s2 = fx.thread("w2").await;
    let lane = s1.lane(LANE);
    lane.submit(&submitted("AST_1")).await.unwrap();
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    let canceled = s2.lane(LANE).cancel_task("AST_1", metadata("client")).await.unwrap();
    assert_eq!(canceled.unwrap().status, TaskState::Canceled);

    run.history.push(message("i1", ChatRole::User, "hi"));
    run.tasks.write("AST_1", TaskState::Completed, TaskPatch::default()).await.unwrap();
    let committed = run
        .commit(CommitOptions { answer_task: Some("AST_1".into()), ..CommitOptions::default() })
        .await
        .unwrap();
    let row = committed.task.expect("read back");
    assert_eq!(row.status, TaskState::Canceled);
    assert_eq!(row.data.metadata.get("client"), Some(&json!(true)));
    assert_eq!(lane.history().await.unwrap().len(), 1, "the rest of the commit landed");
    fx.cleanup(&[&s1, &s2]).await;
}

/// A2A_SEED=3497 (no_failure_nobody_observed, Working): a row a turn answered `working` (its
/// handler's `Working` outcome) is no run's; a later pickup took it as a dead holder's and the
/// framework failed it `worker-lost`. A pickup leaves it as the turn answered it.
async fn a_pickup_leaves_a_row_answered_working(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    lane.submit(&submitted("AST_1")).await.unwrap();
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    let answered = TaskPatch { answered: true, ..TaskPatch::default() };
    run.tasks.write("AST_1", TaskState::Working, answered).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();

    assert_eq!(lane.pickup("AST_1").await.unwrap(), None, "no dead holder's row");
    let row = lane.task("AST_1").await.unwrap().unwrap();
    assert_eq!(row.status, TaskState::Working);
    assert!(row.internal.is_some_and(|i| i.answered));
    fx.cleanup(&[&s]).await;
}

/// A2A_SEED=50 60 267 (answer_is_an_outcome, Concurrent): a concurrent run reads its rows live, so
/// a client's cancel mid-turn is the base of its write. The write keeps the client's row, as the
/// commit would, instead of refusing the whole turn.
async fn a_concurrent_write_keeps_a_row_canceled_mid_turn(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s1 = fx.thread("w1").await;
    let s2 = fx.thread("w2").await;
    let lane = s1.lane(LANE);
    lane.open(RunOptions::new("r0", "AST_0"))
        .await
        .unwrap()
        .commit(Default::default())
        .await
        .unwrap();
    lane.submit(&submitted("AST_1")).await.unwrap();
    let opts = RunOptions { exclusive: false, pickup: true, ..RunOptions::new("r1", "AST_1") };
    let mut run = lane.open(opts).await.unwrap();
    assert!(run.picked_up);
    s2.lane(LANE).cancel_task("AST_1", metadata("client")).await.unwrap();

    run.history.push(message("i1", ChatRole::User, "hi"));
    run.tasks.write("AST_1", TaskState::Completed, TaskPatch::default()).await.unwrap();
    let committed = run
        .commit(CommitOptions { answer_task: Some("AST_1".into()), ..CommitOptions::default() })
        .await
        .unwrap();
    assert_eq!(committed.task.expect("read back").status, TaskState::Canceled);
    assert_eq!(lane.history().await.unwrap().len(), 1, "the rest of the turn landed");
    fx.cleanup(&[&s1, &s2]).await;
}

/// I-5: a run dropped without commit or abandon stops renewing: its lease lapses within the TTL
/// and another holder takes the lane.
async fn a_dropped_run_stops_renewing_and_its_lease_lapses(backend: Backend) {
    let fx = backend.fixture(Some((Duration::from_secs(1), Duration::from_millis(300)))).await;
    let s1 = fx.thread("w1").await;
    let s2 = fx.thread("w2").await;
    let run = s1.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    drop(run);
    tokio::time::sleep(Duration::from_millis(2600)).await;
    let lane2 = s2.lane(LANE);
    let next = lane2.open(RunOptions { wait: false, ..RunOptions::new("r2", "AST_2") }).await;
    let next = next.expect("the dropped run's lease lapsed");
    drop(next);
    fx.cleanup(&[&s1, &s2]).await;
}

/// Waiters run in the order their messages' rows were submitted: a later message cannot take a
/// free lane while an older one still waits for it.
async fn waiters_take_the_lane_in_the_order_their_rows_were_submitted(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    lane.open(RunOptions::new("r0", ""))
        .await
        .unwrap()
        .commit(CommitOptions::default())
        .await
        .unwrap();
    for task_id in ["AST_first", "AST_second"] {
        assert!(lane.submit(&submitted(task_id)).await.unwrap());
    }
    let second = || RunOptions { wait: false, pickup: true, ..RunOptions::new("r2", "AST_second") };
    let refused = lane.open(second()).await.err().expect("the older waiter goes first");
    assert!(matches!(refused, StorageError::Rejected { .. }), "{refused}");
    let first = RunOptions { wait: false, pickup: true, ..RunOptions::new("r1", "AST_first") };
    let mut run = lane.open(first).await.unwrap();
    assert!(run.picked_up);
    run.commit(CommitOptions::default()).await.unwrap();
    // the first row is working now: the second is the oldest waiter and takes the lane
    lane.open(second()).await.unwrap().commit(CommitOptions::default()).await.unwrap();
    fx.cleanup(&[&s]).await;
}

/// A lost handle cannot restart renewal. Takeover replaces its token even before the new
/// owner commits, so the old owner's unchanged version is not enough to authorize a write.
async fn an_expired_owner_cannot_restart_or_commit_after_takeover(backend: Backend) {
    let fx = backend.fixture(Some((Duration::from_secs(1), Duration::from_millis(200)))).await;
    let first = fx.thread("w1").await;
    let second = fx.thread("w2").await;
    let mut stale = first.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    let old_lease = stale.lease().unwrap().clone();
    old_lease.stop();
    tokio::time::sleep(Duration::from_millis(2100)).await;
    assert!(old_lease.is_lost());
    old_lease.keep_alive();
    assert!(old_lease.is_lost(), "restarting renewal cannot revive an expired handle");

    let owner = second.lane(LANE).open(RunOptions::new("r2", "AST_2")).await.unwrap();
    assert_eq!(stale.version(), owner.version(), "takeover alone does not change the version");
    assert_ne!(old_lease.fence().token, owner.lease().unwrap().fence().token);
    stale.state = Some(json!({"stale": true}));
    stale.history.push(message("stale", ChatRole::Assistant, "old owner's answer"));
    let error = stale.commit(CommitOptions::default()).await.unwrap_err();
    assert!(matches!(error, StorageError::VersionConflict { .. }), "{error}");
    assert!(second.lane(LANE).history().await.unwrap().is_empty());
    assert!(second.lane(LANE).info().await.unwrap().unwrap().state.is_none());

    owner.lease().unwrap().stop();
    owner.lease().unwrap().release(None).await.unwrap();
    owner.lease().unwrap().keep_alive();
    let lost = tokio::time::timeout(Duration::from_secs(2), owner.lease().unwrap().lost()).await;
    assert!(matches!(lost, Ok(StorageError::LeaseLost(_))));
    // The deliberately released owner has no work to commit or abandon.
    drop(owner);
    fx.cleanup(&[&first, &second]).await;
}

async fn an_item_id_of_another_lane_fails_the_commit_without_mutation(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let mut a = s.lane("lane-a").open(RunOptions::new("ra", "AST_a")).await.unwrap();
    a.history.push(message("shared", ChatRole::User, "a's"));
    a.commit(CommitOptions::default()).await.unwrap();

    let mut b = s.lane("lane-b").open(RunOptions::new("rb", "AST_b")).await.unwrap();
    b.history.push(message("b1", ChatRole::User, "b's own"));
    b.history.push(message("shared", ChatRole::User, "b's"));
    assert!(b.commit(CommitOptions::default()).await.is_err());
    let a_items = s.lane("lane-a").history().await.unwrap();
    assert_eq!(a_items, vec![message("shared", ChatRole::User, "a's")]);
    assert!(s.lane("lane-b").history().await.unwrap().is_empty(), "atomic: nothing of b landed");
    fx.cleanup(&[&s]).await;
}

/// I-18: a `concurrent` run appends only: editing or deleting a loaded item is refused and
/// nothing is written.
async fn a_concurrent_run_refuses_to_edit_or_delete_loaded_history(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut first = lane.open(RunOptions::new("r0", "AST_0")).await.unwrap();
    first.history.push(message("i1", ChatRole::User, "hi"));
    first.commit(CommitOptions::default()).await.unwrap();
    let concurrent =
        |run_id: &str| RunOptions { exclusive: false, ..RunOptions::new(run_id, "AST_1") };

    let mut deleting = lane.open(concurrent("r1")).await.unwrap();
    deleting.history.clear();
    deleting.history.push(message("i2", ChatRole::User, "new"));
    assert!(matches!(
        deleting.commit(CommitOptions::default()).await,
        Err(StorageError::Invalid(_))
    ));
    let mut editing = lane.open(concurrent("r2")).await.unwrap();
    editing.history[0] = message("i1", ChatRole::User, "rewritten");
    assert!(matches!(
        editing.commit(CommitOptions::default()).await,
        Err(StorageError::Invalid(_))
    ));
    assert_eq!(lane.history().await.unwrap(), vec![message("i1", ChatRole::User, "hi")]);
    fx.cleanup(&[&s]).await;
}

/// I-19: the acquire loads the whole history in the lease's own batch, past any page size.
async fn a_run_loads_its_whole_history_in_its_acquire(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    for i in 0..600 {
        run.history.push(message(&format!("i{i}"), ChatRole::User, "hi"));
    }
    run.commit(CommitOptions::default()).await.unwrap();
    let again = lane.open(RunOptions::new("r2", "AST_2")).await.unwrap();
    assert_eq!(again.history.len(), 600);
    assert_eq!(again.history.last().map(|i| i.id.as_str()), Some("i599"));
    drop(again);
    fx.cleanup(&[&s]).await;
}

/// I-24: items page by insertion position with an optional lane filter; the cursor is opaque,
/// bound to its filter, and pages neither skip nor repeat.
async fn items_page_by_position_with_a_filter_bound_cursor(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    for (lane, n) in [("lane-a", 0), ("lane-b", 1), ("lane-a", 2)] {
        let mut run = s.lane(lane).open(RunOptions::new(format!("r{n}"), "")).await.unwrap();
        run.history.push(message(&format!("i{n}"), ChatRole::User, "hi"));
        run.commit(CommitOptions::default()).await.unwrap();
    }
    let ids = |p: &lk_agents_storage::thread::db::ItemPage| -> Vec<String> {
        p.items.iter().map(|i| format!("{}/{}", i.lane, i.item.id)).collect()
    };
    let first = s.list_items(None, None, 2).await.unwrap();
    assert_eq!(ids(&first), vec!["lane-a/i0", "lane-b/i1"]);
    let rest = s.list_items(None, first.next.as_deref(), 2).await.unwrap();
    assert_eq!((ids(&rest), rest.next), (vec!["lane-a/i2".to_owned()], None));
    let a = s.list_items(Some("lane-a"), None, 10).await.unwrap();
    assert_eq!(ids(&a), vec!["lane-a/i0", "lane-a/i2"]);
    let crossed = s.list_items(Some("lane-a"), first.next.as_deref(), 2).await;
    assert!(matches!(crossed, Err(StorageError::Invalid(_))), "a cursor of another filter");
    fx.cleanup(&[&s]).await;
}

/// I-22/I-24: A2A ListTasks pages by last update (ties by task id) without skipping or
/// repeating, filters by status, and refuses a page token made under other filters.
async fn tasks_page_by_last_update_with_a_filter_bound_token(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    for id in ["AST_1", "AST_2", "AST_3", "AST_4", "AST_5"] {
        assert!(lane.submit(&submitted(id)).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(5)).await;
    lane.cancel_task("AST_2", Default::default()).await.unwrap();
    let query = |status, page_token| lk_agents_storage::thread::db::TaskQuery {
        status,
        page_size: 2,
        page_token,
        ..Default::default()
    };
    let (mut seen, mut token) = (Vec::new(), None);
    loop {
        let page = s.list_tasks(&query(None, token)).await.unwrap();
        assert_eq!(page.total_size, 5);
        seen.extend(page.tasks.iter().map(|t| (t.updated_at, t.task_id.clone())));
        if page.next_page_token.is_empty() {
            break;
        }
        token = Some(page.next_page_token);
    }
    let mut ordered = seen.clone();
    ordered.sort_by(|a, b| b.cmp(a));
    ordered.dedup();
    assert_eq!(seen, ordered, "last update first, ties by id, nothing repeated");
    assert_eq!(seen.len(), 5);
    assert_eq!(seen[0].1, "AST_2", "the canceled task was updated last");

    let canceled = s.list_tasks(&query(Some(TaskState::Canceled), None)).await.unwrap();
    assert_eq!(canceled.tasks.iter().map(|t| t.task_id.as_str()).collect::<Vec<_>>(), ["AST_2"]);
    let submitted = s.list_tasks(&query(Some(TaskState::Submitted), None)).await.unwrap();
    assert_eq!(submitted.total_size, 4);
    let crossed = s.list_tasks(&query(None, Some(submitted.next_page_token))).await;
    assert!(matches!(crossed, Err(StorageError::Invalid(_))), "a token of another filter");
    fx.cleanup(&[&s]).await;
}

/// A thread's tasks browse in insertion order, of one lane when asked, whatever was updated
/// since; the token is bound to that order and lane.
async fn tasks_browse_in_insertion_order_by_lane(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    for (lane, id) in [(LANE, "AST_1"), ("expert", "AST_2"), (LANE, "AST_3"), (LANE, "AST_4")] {
        assert!(s.lane(lane).submit(&submitted(id)).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(5)).await;
    s.lane(LANE).cancel_task("AST_1", Default::default()).await.unwrap();
    let query = |page_token| lk_agents_storage::thread::db::TaskQuery {
        lane: Some(LANE.into()),
        order: lk_agents_storage::thread::db::TaskOrder::Inserted,
        page_size: 2,
        page_token,
        ..Default::default()
    };
    let (mut seen, mut token) = (Vec::new(), None);
    loop {
        let page = s.list_tasks(&query(token)).await.unwrap();
        assert_eq!(page.total_size, 3);
        seen.extend(page.tasks.into_iter().map(|t| t.task_id));
        if page.next_page_token.is_empty() {
            break;
        }
        token = Some(page.next_page_token);
    }
    assert_eq!(seen, ["AST_1", "AST_3", "AST_4"], "oldest first, the lane's only");
    let first = s.list_tasks(&query(None)).await.unwrap();
    let crossed = s
        .list_tasks(&lk_agents_storage::thread::db::TaskQuery {
            page_token: Some(first.next_page_token),
            ..Default::default()
        })
        .await;
    assert!(matches!(crossed, Err(StorageError::Invalid(_))), "a token of another listing");
    fx.cleanup(&[&s]).await;
}

lk_agents_storage::contract!(
    tasks_page_by_last_update_with_a_filter_bound_token,
    tasks_browse_in_insertion_order_by_lane,
    items_page_by_position_with_a_filter_bound_cursor,
    a_run_loads_its_whole_history_in_its_acquire,
    a_concurrent_run_refuses_to_edit_or_delete_loaded_history,
    an_item_id_of_another_lane_fails_the_commit_without_mutation,
    waiters_take_the_lane_in_the_order_their_rows_were_submitted,
    an_expired_owner_cannot_restart_or_commit_after_takeover,
    a_dropped_run_stops_renewing_and_its_lease_lapses,
    a_cancel_never_overwrites_a_terminal_task,
    a_pickup_leaves_a_row_answered_working,
    a_commit_never_overwrites_a_canceled_task,
    a_concurrent_write_keeps_a_row_canceled_mid_turn,
    the_second_lane_waits_while_the_first_holds_and_gets_it_after_the_commit,
    a_run_that_aborts_writes_run_ended_and_nothing_else,
    a_committed_run_id_yields_a_closed_run,
    a_stale_commit_is_a_version_conflict,
    concurrent_runs_take_no_lease_and_append,
);

// -- I-7: a lost commit acknowledgment resolves to the original run (SQLite, injected faults) --

/// A run that writes one item and completes `AST_1`, committed over a faulty connection.
async fn commit_through(
    fault: lk_agents_storage::testing::CommitFault,
) -> (Fixture, Thread, Result<i64, StorageError>) {
    let fx = Backend::Sqlite.fixture(None).await;
    let (s, faults) = fx.faulty_thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.history.push(message("i1", ChatRole::User, "hi"));
    run.tasks.write("AST_1", TaskState::Completed, TaskPatch::default()).await.unwrap();
    faults.fail_next_commit(fault);
    let committed = run.commit(CommitOptions::default()).await.map(|c| c.version);
    (fx, s, committed)
}

async fn run_ended_count(s: &Thread) -> usize {
    s.journal(0, 100).await.unwrap().iter().filter(|e| e.r#type == "run_ended").count()
}

#[tokio::test]
async fn a_commit_whose_ack_was_lost_after_it_applied_resolves_to_that_commit() {
    let (fx, s, committed) =
        commit_through(lk_agents_storage::testing::CommitFault::LoseAckAfter).await;
    assert_eq!(committed.unwrap(), 1, "the original run's commit, not a failure");
    assert_eq!(s.lane(LANE).history().await.unwrap().len(), 1);
    assert_eq!(s.lane(LANE).task("AST_1").await.unwrap().unwrap().status, TaskState::Completed);
    assert_eq!(run_ended_count(&s).await, 1, "written once");
    fx.cleanup(&[&s]).await;
}

#[tokio::test]
async fn a_commit_whose_ack_was_lost_before_it_applied_is_sent_again_under_its_fence() {
    let (fx, s, committed) =
        commit_through(lk_agents_storage::testing::CommitFault::LoseAckBefore).await;
    assert_eq!(committed.unwrap(), 1);
    assert_eq!(s.lane(LANE).history().await.unwrap().len(), 1);
    assert_eq!(run_ended_count(&s).await, 1);
    fx.cleanup(&[&s]).await;
}

#[tokio::test]
async fn a_lost_ack_after_the_lane_moved_on_is_a_rejection_with_nothing_written() {
    let fx = Backend::Sqlite.fixture(None).await;
    let (s1, faults) = fx.faulty_thread("w1").await;
    let s2 = fx.thread("w2").await;
    let mut stale = s1.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    stale.lease().unwrap().release(None).await.unwrap();
    let mut fresh = s2.lane(LANE).open(RunOptions::new("r2", "AST_2")).await.unwrap();
    fresh.commit(CommitOptions::default()).await.unwrap();

    stale.history.push(message("i9", ChatRole::Assistant, "late"));
    faults.fail_next_commit(lk_agents_storage::testing::CommitFault::LoseAckBefore);
    let err = stale.commit(CommitOptions::default()).await.expect_err("rejected");
    assert!(matches!(err, StorageError::VersionConflict { .. }), "{err}");
    assert!(s1.lane(LANE).history().await.unwrap().is_empty());
    fx.cleanup(&[&s1, &s2]).await;
}

/// Failed renewals retry within the confirmed ownership deadline, then stop exactly at it.
#[tokio::test(start_paused = true)]
async fn a_failing_renewal_retries_within_the_lease_and_is_lost_after_it() {
    use lk_agents_storage::testing::{Faulty, MemoryConnection};
    use lk_agents_storage::thread::{LeaseTiming, SchemaCell};
    use std::sync::Arc;

    let clock = Arc::new(lk_agents_net::clock::RealClock::new());
    let faulty = Arc::new(Faulty::new(Box::new(MemoryConnection::new(clock).unwrap())));
    let timing = LeaseTiming::new(Duration::from_secs(10), Duration::from_secs(3)).unwrap();
    let thread = Thread::new(faulty.clone(), "AT_renewal", timing, SchemaCell::default(), "w1")
        .await
        .unwrap();
    let run = thread.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    let lease = run.lease().unwrap().clone();
    let start = tokio::time::Instant::now();

    // The extend at 3 s fails; those at 6 s and 9 s confirm ownership.
    faulty.unreachable_for(Duration::from_secs(5));
    tokio::time::sleep(Duration::from_millis(9500)).await;
    assert!(!lease.is_lost(), "a failure within the window is retried");

    // Extends at 12 s, 15 s and 18 s fail; the last confirmation expires at 19 s.
    faulty.unreachable_for(Duration::from_secs(20));
    tokio::time::sleep(Duration::from_millis(8000)).await;
    assert!(!lease.is_lost(), "still within the deadline at 17.5 s");
    let lost = tokio::time::timeout(Duration::from_secs(5), lease.lost()).await;
    assert!(matches!(lost, Ok(StorageError::LeaseLost(_))), "{lost:?}");
    assert_eq!(start.elapsed().as_secs(), 19);
}
