//! The run unit of work (`run` module) against agentstore, the same way `agentdb_local.rs` drives
//! the statements: through public-api-server with a staging project's `LIVEKIT_URL` and
//! key pair; skipped when they are not set or when the project cannot
//! create a database. Every test creates its own database and deletes it at the end.

use std::time::{Duration, Instant};

use lk_agents_storage::run::{
    CommitOptions, ItemRow, Ref, RunOptions, RunOutcome, TaskPatch, TaskState, Thread,
};
mod common;

use livekit_protocol::agent::chat_context::chat_item;
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{ChatMessage, ChatRole};
use lk_agents_storage::{Storage, StorageError};
use serde_json::json;

/// The storage and a fresh database, or `None` when there is no agent-db to test against.
async fn fresh_database() -> Option<(Storage, String)> {
    let (opts, minter) = common::project()?;
    let storage = Storage::new(opts, minter).unwrap();
    match common::create_database(&storage).await {
        Ok(c) => Some((storage, c.database_id)),
        Err(e) => {
            eprintln!("skipped: the project cannot create a database: {e}");
            None
        }
    }
}

/// One handle over the database, as one worker would hold it.
async fn thread(storage: &Storage, database_id: &str, holder: &str) -> Thread {
    storage.open(database_id, holder).await.unwrap()
}

async fn cleanup(storage: &Storage, database_id: &str, sessions: &[&Thread]) {
    for s in sessions {
        s.close();
    }
    storage.databases().delete(database_id).await.unwrap();
}

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

#[tokio::test]
async fn the_second_lane_waits_while_the_first_holds_and_gets_it_after_the_commit() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s1 = thread(&storage, &database_id, "w1").await;
    assert!(s1.holder().starts_with("w1/"));
    assert_eq!(s1.id(), database_id);

    let mut run1 = s1.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    // The first acquire applies the DDL; the second handle dials once the first run exists, as a
    // worker reached later would.
    let s2 = thread(&storage, &database_id, "w2").await;
    assert_ne!(s1.holder(), s2.holder());
    assert_eq!(run1.version(), 0);
    assert!(!run1.closed && run1.lease().is_some());
    assert!(run1.history.is_empty());
    assert_eq!(run1.state, None);
    assert!(run1.agents.is_empty());
    let journal = s1.events(0, 100).await.unwrap();
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
    assert!(matches!(refused, StorageError::LeaseHeld(_)), "{refused}");
    let started = Instant::now();
    let timed_out = lane2
        .open(RunOptions {
            wait_timeout: Duration::from_millis(250),
            ..RunOptions::new("r2", "AST_2")
        })
        .await
        .err()
        .expect("lease");
    assert!(matches!(timed_out, StorageError::LeaseTimeout(_)), "{timed_out}");
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
    assert_eq!(waiter(&s1).await.as_deref(), Some(s2.holder()));

    // The first run writes everything a run writes, and releases.
    run1.history.push(message("i1", ChatRole::User, "hi"));
    run1.history.last_mut().unwrap().message_id = Some("m1".into());
    run1.history.push(message("i2", ChatRole::Assistant, "hello"));
    run1.history.push(message("sys", ChatRole::System, "be nice"));
    run1.state = Some(json!({"patient": "Ann"}));
    run1.agents.push(Ref::with_args("Followup", json!({"n": 1})));
    run1.events.append("agent_state_changed", json!({"state": "thinking"}));
    run1.usage.requests = 1;
    run1.usage.input_tokens = 10;
    let request = run1.writes(CommitOptions::default());
    assert_eq!(request.items.len(), 2, "instructions are not stored");
    assert_eq!(request.items[0].message_id.as_deref(), Some("m1"));
    assert_eq!(
        request.events.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(),
        vec!["conversation_item_added", "conversation_item_added", "agent_state_changed", "usage"]
    );
    assert_eq!(request.events[3].payload["input_tokens"], 10);
    assert_eq!(request.state, Some(json!({"patient": "Ann"})));
    assert_eq!(request.agent_stack, Some(json!([{"name": "Followup", "args": {"n": 1}}])));
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
    assert_eq!(run2.history[0], message("i1", ChatRole::User, "hi"));
    assert_eq!(run2.state, Some(json!({"patient": "Ann"})));
    assert_eq!(run2.agents.current().map(|r| r.name.as_str()), Some("Followup"));
    assert!(!run2.agents.is_dirty());
    assert_eq!(waiter(&s2).await, None, "the waiter that won cleared its own mark");
    assert!(s2.events(0, 100).await.unwrap().iter().any(|e| e.r#type == "usage"));

    // An unchanged history writes no item; a popped stack and a changed state are written.
    assert!(run2.writes(CommitOptions::default()).items.is_empty());
    let unchanged = run2.writes(CommitOptions::default()).state;
    assert_eq!(unchanged, None, "unchanged state is not rewritten");
    run2.history.retain(|i| i.id != "i2");
    run2.agents.pop();
    run2.state = Some(json!({"patient": "Ann", "seen": true}));
    let request = run2.writes(CommitOptions::default());
    assert_eq!(request.delete_items, vec!["i2".to_owned()]);
    assert_eq!(request.agent_stack, Some(json!([])));
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
    assert!(info.agent_stack.is_none_or(|s| s.is_empty()));
    assert_eq!(s2.lanes().await.unwrap().len(), 1);
    assert!(s2.events(0, 100).await.unwrap().len() >= 5);
    assert!(lane.db().inbox(true).await.unwrap().is_empty());
    cleanup(&storage, &database_id, &[&s1, &s2]).await;
}

#[tokio::test]
async fn a_run_that_aborts_writes_run_ended_and_nothing_else() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s = thread(&storage, &database_id, "w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.history.push(message("i1", ChatRole::User, "hi"));
    run.state = Some(json!({"x": 1}));
    run.agents.push(Ref::new("Other"));
    run.events.append("agent_state_changed", json!({"state": "thinking"}));
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
    assert_eq!(info.agent_stack, Some(vec![]));
    let events = s.events(0, 100).await.unwrap();
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
    cleanup(&storage, &database_id, &[&s]).await;
}

#[tokio::test]
async fn a_committed_run_id_yields_a_closed_run() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s = thread(&storage, &database_id, "w1").await;
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
    let events = s.events(0, 100).await.unwrap();
    assert_eq!(events.iter().filter(|e| e.r#type == "run_started").count(), 1, "not again");
    assert_eq!(events.iter().filter(|e| e.r#type == "run_ended").count(), 1);
    // Released: a third open takes the lane without waiting.
    let free =
        lane.open(RunOptions { wait: false, ..RunOptions::new("r2", "AST_2") }).await.unwrap();
    assert!(!free.closed);
    drop(free);
    cleanup(&storage, &database_id, &[&s]).await;
}

#[tokio::test]
async fn a_stale_commit_is_a_version_conflict() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s1 = thread(&storage, &database_id, "w1").await;
    let mut stale = s1.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    let s2 = thread(&storage, &database_id, "w2").await;
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
    let err = stale.commit(CommitOptions::default()).await.err().expect("fenced out");
    assert!(matches!(err, StorageError::VersionConflict { expected: 0, .. }), "{err}");
    assert_eq!(s1.lane(LANE).history().await.unwrap().len(), 1, "the loser wrote nothing");
    cleanup(&storage, &database_id, &[&s1, &s2]).await;
}

#[tokio::test]
async fn concurrent_runs_take_no_lease_and_append() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s1 = thread(&storage, &database_id, "w1").await;
    // The schema and the lane row come from a first exclusive run; the concurrent runs then share
    // a lane at version 1 and never move it.
    let mut first = s1.lane(LANE).open(RunOptions::new("r0", "AST_0")).await.unwrap();
    first.commit(CommitOptions::default()).await.unwrap();
    let s2 = thread(&storage, &database_id, "w2").await;
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
    a.agents.push(Ref::new("Nope"));
    b.history.push(message("b1", ChatRole::User, "two"));
    let drained = CommitOptions { drained_id: Some(3), ..CommitOptions::default() };
    let request = a.writes(drained.clone());
    assert_eq!(request.state, None);
    assert_eq!(request.agent_stack, None);
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
    assert_eq!(info.agent_stack, Some(vec![]));
    let runs = lane.runs(10).await.unwrap();
    assert_eq!(runs.len(), 3);
    assert!(runs.iter().all(|r| r.outcome == Some(RunOutcome::Completed)));
    // A concurrent run has nothing to abandon.
    let mut c = open(&s1, "r3", "AST_3").await;
    c.abandon(RunOutcome::Canceled, "CancelledError").await.unwrap();
    assert_eq!(s1.events(0, 100).await.unwrap().len(), 8, "two items entered the history");
    cleanup(&storage, &database_id, &[&s1, &s2]).await;
}
