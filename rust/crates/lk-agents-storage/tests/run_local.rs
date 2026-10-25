//! The run unit of work (`run` module) against agentstore, the same way `agentdb_local.rs` drives
//! the statements: through public-api-server with a staging project's `LIVEKIT_URL` and
//! key pair; skipped when they are not set or when the project cannot
//! create a database. Every test creates its own database and deletes it at the end.

use std::time::{Duration, Instant};

use lk_agents_storage::run::{
    Item, Ref, RunOptions, RunOutcome, SubmitOptions, TaskState, Thread, WaitOptions,
};
use lk_agents_storage::{Storage, StorageError, StorageOptions, TokenMinter};
use serde_json::{Value, json};

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

/// The storage and a fresh database, or `None` when there is no agent-db to test against.
async fn fresh_database() -> Option<(Storage, String)> {
    let opts = options()?;
    let storage = Storage::new(opts, minter()).unwrap();
    match storage.databases().create_database(3600).await {
        Ok(c) => Some((storage, c.database_id)),
        Err(e) => {
            eprintln!("skipped: the project cannot create a database: {e}");
            None
        }
    }
}

/// One handle over the database, as one worker would hold it.
async fn thread(storage: &Storage, database_id: &str, holder: &str) -> Thread {
    Thread::new(storage.open(database_id).await.unwrap(), holder)
}

async fn cleanup(storage: &Storage, database_id: &str, sessions: &[&Thread]) {
    for s in sessions {
        s.close();
    }
    storage.databases().delete_database(database_id).await.unwrap();
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

fn message(id: &str, role: &str, text: &str) -> Item {
    Item::new(json!({"id": id, "type": "message", "role": role, "content": [text]})).unwrap()
}

const LANE: &str = "front-desk";

#[tokio::test]
async fn the_second_lane_waits_while_the_first_holds_and_gets_it_after_the_commit() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s1 = thread(&storage, &database_id, "w1").await;
    assert!(s1.holder().starts_with("w1/"));
    assert_eq!(s1.id(), database_id);

    let mut run1 = s1.lane(LANE).open(RunOptions::new("r1", "AST_1")).await.unwrap();
    // A `ThreadDb` reads the schema version at open and the first acquire applies the DDL, so
    // the second handle dials once the first run exists, as a worker reached later would.
    let s2 = thread(&storage, &database_id, "w2").await;
    assert_ne!(s1.holder(), s2.holder());
    assert_eq!(run1.version, 0);
    assert!(!run1.closed && run1.exclusive);
    assert!(run1.history.is_empty());
    assert_eq!(run1.state, None);
    assert!(run1.agents.is_empty());
    assert_eq!(run1.events.loaded.len(), 1, "run_started");
    assert!(run1.lease_token().is_some_and(|t| t.starts_with("lease_")));
    assert!(!run1.lease_lost());
    assert_eq!(run1.waiter().await.unwrap(), None);

    // A second holder: refused at once without the wait, and timed out with a short one.
    let lane2 = s2.lane(LANE);
    let refused = lane2
        .open(RunOptions { wait: false, ..RunOptions::new("r2", "AST_2") })
        .await
        .err()
        .expect("held");
    assert!(matches!(refused, StorageError::LeaseHeld(_)), "{refused}");
    let started = Instant::now();
    let timed_out = lane2
        .open(RunOptions {
            wait_timeout: Duration::from_millis(250),
            ..RunOptions::new("r2", "AST_2")
        })
        .await
        .err()
        .expect("held");
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
    assert_eq!(run1.waiter().await.unwrap().as_deref(), Some(s2.holder()));

    // The first run writes everything a run writes, and releases.
    run1.history.push(message("i1", "user", "hi"));
    run1.mark_message("i1", "m1");
    run1.history.push(message("i2", "assistant", "hello"));
    run1.history.push(message("sys", "system", "be nice"));
    run1.state = Some(json!({"patient": "Ann"}));
    run1.agents.push(Ref::with_args("Followup", json!({"n": 1})));
    run1.events.append("agent_state_changed", json!({"state": "thinking"}));
    run1.usage.requests = 1;
    run1.usage.input_tokens = 10;
    let request = run1.commit_request(1);
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
    run1.commit(now_ms()).await.unwrap();
    assert_eq!(run1.committed_version(), Some(1));
    assert!(matches!(run1.commit(now_ms()).await, Err(StorageError::Invalid(_))));

    // The waiter acquires within a poll, at the committed version, with everything loaded.
    let (mut run2, acquired_at) = waiting.await.unwrap();
    assert!(acquired_at >= released_at);
    assert!(acquired_at - released_at < Duration::from_millis(1500), "a poll interval at most");
    assert_eq!(run2.version, 1);
    assert_eq!(run2.history.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), vec!["i1", "i2"]);
    assert_eq!(run2.history[0].payload["content"], json!(["hi"]));
    assert_eq!(run2.state, Some(json!({"patient": "Ann"})));
    assert_eq!(run2.agents.current().map(|r| r.name.as_str()), Some("Followup"));
    assert!(!run2.agents.is_dirty());
    assert_eq!(run2.waiter().await.unwrap(), None, "the waiter that won cleared its own mark");
    assert!(run2.events.loaded.iter().any(|e| e.r#type == "usage"));

    // An unchanged history writes no item; a popped stack and a changed state are written.
    assert!(run2.commit_request(2).items.is_empty());
    assert_eq!(run2.commit_request(2).state, None, "unchanged state is not rewritten");
    run2.history.retain(|i| i.id != "i2");
    run2.agents.pop();
    run2.state = Some(json!({"patient": "Ann", "seen": true}));
    let request = run2.commit_request(2);
    assert_eq!(request.delete_items, vec!["i2".to_owned()]);
    assert_eq!(request.agent_stack, Some(json!([])));
    run2.commit(now_ms()).await.unwrap();
    assert_eq!(run2.committed_version(), Some(2));

    // The reads outside a run.
    let lane = s1.lane(LANE);
    assert_eq!(lane.history().await.unwrap().len(), 1);
    let runs = lane.runs(10).await.unwrap();
    assert_eq!(
        runs.iter().map(|r| (r.run_id.as_str(), r.outcome.as_deref())).collect::<Vec<_>>(),
        vec![("r2", Some("completed")), ("r1", Some("completed"))]
    );
    assert_eq!(runs[0].channel.as_deref(), Some("a2a"));
    assert_eq!(runs[0].task_id.as_deref(), Some("AST_2"));
    assert_eq!(lane.runs(1).await.unwrap().len(), 1);
    let info = lane.info().await.unwrap().expect("a lane row");
    assert_eq!((info.version, info.drained_id, info.lease_until), (2, 0, 0));
    assert!(info.agent_stack.is_empty());
    assert_eq!(s2.lanes().await.unwrap().len(), 1);
    assert!(s2.events(0, 100).await.unwrap().len() >= 5);
    assert!(lane.inbox_pending().await.unwrap().is_empty());
    cleanup(&storage, &database_id, &[&s1, &s2]).await;
}

#[tokio::test]
async fn a_run_that_aborts_writes_run_ended_and_nothing_else() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s = thread(&storage, &database_id, "w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.history.push(message("i1", "user", "hi"));
    run.state = Some(json!({"x": 1}));
    run.agents.push(Ref::new("Other"));
    run.events.append("agent_state_changed", json!({"state": "thinking"}));
    run.tasks
        .submit(
            &Ref::new("Followup"),
            SubmitOptions { task_id: Some("AST_t".into()), ..Default::default() },
        )
        .await
        .unwrap();
    run.abandon(now_ms(), RunOutcome::Failed, "RuntimeError: boom").await.unwrap();

    assert!(lane.history().await.unwrap().is_empty());
    assert!(lane.tasks(None, 50).await.unwrap().is_empty());
    let info = lane.info().await.unwrap().unwrap();
    assert_eq!((info.version, info.lease_until), (0, 0), "released, not bumped");
    assert_eq!(info.agent_stack, vec![]);
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
    assert_eq!(next.version, 0);
    drop(next);
    cleanup(&storage, &database_id, &[&s]).await;
}

#[tokio::test]
async fn a_committed_run_id_yields_a_closed_run() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s = thread(&storage, &database_id, "w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.history.push(message("i1", "user", "hi"));
    run.commit(now_ms()).await.unwrap();

    let mut again = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    assert!(again.closed);
    assert_eq!(again.version, 1);
    assert_eq!(again.history.len(), 1, "loaded, so a retry can answer from it");
    assert!(again.lease().is_none(), "no renewal for a closed run");
    again.history.push(message("i2", "assistant", "hello"));
    again.commit(now_ms()).await.unwrap();
    assert_eq!(again.committed_version(), None);
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
    assert!(s1.db().release(LANE, stale.lease_token().unwrap()).await.unwrap());
    let mut fresh = s2
        .lane(LANE)
        .open(RunOptions { wait: false, ..RunOptions::new("r2", "AST_2") })
        .await
        .unwrap();
    fresh.history.push(message("i1", "user", "hi"));
    fresh.commit(now_ms()).await.unwrap();
    assert_eq!(fresh.committed_version(), Some(1));

    stale.history.push(message("i9", "assistant", "late"));
    let err = stale.commit(now_ms()).await.expect_err("fenced out");
    assert!(matches!(err, StorageError::VersionConflict { expected: 0, .. }), "{err}");
    assert_eq!(s1.lane(LANE).history().await.unwrap().len(), 1, "the loser wrote nothing");
    cleanup(&storage, &database_id, &[&s1, &s2]).await;
}

/// Replaces a minted id (`msg_`, `art_` plus twelve hex) with a placeholder after checking it.
fn placeholder_id(v: &mut Value, key: &str, prefix: &str) {
    let id = v[key].as_str().expect("an id").to_owned();
    assert!(id.starts_with(prefix) && id.len() == prefix.len() + 12, "{id}");
    assert!(id[prefix.len()..].chars().all(|c| c.is_ascii_hexdigit()), "{id}");
    v[key] = Value::String(format!("{prefix}X"));
}

#[tokio::test]
async fn task_rows_read_back_as_the_python_tasks_write_would_have_produced() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s = thread(&storage, &database_id, "w1").await;
    let lane = s.lane(LANE);

    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    let id = run
        .tasks
        .submit(
            &Ref::with_args("Followup", json!({"n": 1})),
            SubmitOptions {
                task_id: Some("AST_t1".into()),
                deadline: Some(5),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(id, "AST_t1");
    let minted = run.tasks.submit(&Ref::new("Other"), SubmitOptions::default()).await.unwrap();
    assert!(minted.starts_with("AST_") && minted.len() == 16);
    run.tasks
        .wait(
            "AST_t2",
            "when?",
            WaitOptions {
                tool: Some("book".into()),
                call_id: Some("c1".into()),
                deadline: Some(99),
                schema: Some(json!({"type": "object"})),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    run.tasks.fail("AST_t3", "boom", Some("lease-timeout")).await.unwrap();
    run.tasks
        .cancel("AST_t4", Some(json!({"livekit/interrupted": true}).as_object().cloned().unwrap()))
        .await
        .unwrap();
    // A pending write reads back before the commit; by messageId too.
    let pending = run.tasks.get("AST_t1").await.unwrap().expect("pending");
    assert_eq!((pending.seq, pending.status.as_str()), (0, "submitted"));
    run.tasks
        .write(
            "AST_t5",
            TaskState::Working,
            lk_agents_storage::run::TaskPatch {
                message_id: Some("m5".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(run.tasks.by_message_id("m5").await.unwrap().unwrap().id, "AST_t5");
    assert_eq!(run.tasks.writes().len(), 6);
    run.commit(now_ms()).await.unwrap();

    // Hand-written from `Tasks.write`, `Tasks.submit`, `Tasks.wait`, `Tasks.fail`, `Tasks.cancel`.
    let rows = lane.tasks(None, 50).await.unwrap();
    let row = |id: &str| rows.iter().find(|r| r.id == id).unwrap_or_else(|| panic!("{id}")).clone();
    let t1 = row("AST_t1");
    assert_eq!((t1.status.as_str(), t1.seq), ("submitted", 1));
    assert_eq!(t1.name.as_deref(), Some("Followup"));
    let internal = t1.internal.clone().unwrap();
    assert_eq!((internal.args, internal.deadline), (Some(json!({"n": 1})), Some(5)));
    assert!(t1.data.history.is_empty() && t1.ask().is_none() && t1.data.artifacts.is_none());
    let t2 = row("AST_t2");
    assert_eq!(t2.status, "input-required");
    assert_eq!(t2.internal.as_ref().and_then(|i| i.deadline), Some(99));
    assert_eq!(
        t2.ask().unwrap().to_string(),
        r#"{"question":"when?","tool":"book","call_id":"c1"}"#
    );
    let mut sm = t2.status_message.clone().unwrap();
    placeholder_id(&mut sm, "messageId", "msg_");
    assert_eq!(
        sm.to_string(),
        r#"{"messageId":"msg_X","role":"agent","parts":[{"kind":"text","text":"when?"},{"kind":"data","data":{"type":"object"}}],"taskId":"AST_t2"}"#
    );
    assert_eq!(t2.name, None, "a row written without a class has no name");
    let t3 = row("AST_t3");
    let mut sm = t3.status_message.clone().unwrap();
    placeholder_id(&mut sm, "messageId", "msg_");
    assert_eq!(
        sm.to_string(),
        r#"{"messageId":"msg_X","role":"agent","parts":[{"kind":"text","text":"boom"}],"taskId":"AST_t3","metadata":{"livekit/reason":"lease-timeout"}}"#
    );
    let t4 = row("AST_t4");
    assert_eq!(t4.status, "canceled");
    assert_eq!(
        t4.data.metadata,
        json!({"livekit/interrupted": true}).as_object().cloned().unwrap()
    );
    let t5 = row("AST_t5");
    assert_eq!((t5.status.as_str(), t5.message_id.as_deref()), ("working", Some("m5")));

    // The next run: the open question is found, a message appends to the history, a
    // completion writes the artifact and the last message, and a terminal row refuses a write.
    let mut run = lane.open(RunOptions::new("r2", "AST_9")).await.unwrap();
    let waiting = run.tasks.waiting().await.unwrap();
    assert_eq!(waiting.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["AST_t2"]);
    let stored = run.tasks.get("AST_t1").await.unwrap().expect("stored");
    assert_eq!((stored.seq, stored.status.as_str()), (1, "submitted"));
    assert!(stored.updated_at > 0);
    run.tasks.message("AST_t2", json!({"messageId": "m2", "role": "user"})).await.unwrap();
    run.tasks.done("AST_t1", Some(json!("all set"))).await.unwrap();
    run.tasks.done("AST_t5", Some(json!({"booked": true}))).await.unwrap();
    run.tasks.done(&minted, None).await.unwrap();
    let refused = run.tasks.done("AST_t3", None).await.expect_err("terminal");
    assert!(matches!(refused, StorageError::Invalid(_)), "{refused}");
    run.commit(now_ms()).await.unwrap();

    let rows = lane.tasks(None, 50).await.unwrap();
    let row = |id: &str| rows.iter().find(|r| r.id == id).unwrap_or_else(|| panic!("{id}")).clone();
    let t2 = row("AST_t2");
    assert_eq!(t2.status, "input-required");
    assert_eq!(t2.data.history, vec![json!({"messageId": "m2", "role": "user"})]);
    assert!(t2.ask().is_some(), "input-required keeps its pending transition");
    let t1 = row("AST_t1");
    assert_eq!((t1.status.as_str(), t1.seq), ("completed", 1));
    assert_eq!(t1.internal.as_ref().and_then(|i| i.deadline), Some(5));
    assert_eq!(t1.name.as_deref(), Some("Followup"), "the name is kept");
    let mut artifacts = t1.data.artifacts.clone().unwrap();
    placeholder_id(&mut artifacts[0], "artifactId", "art_");
    assert_eq!(
        Value::Array(artifacts).to_string(),
        r#"[{"artifactId":"art_X","parts":[{"kind":"text","text":"all set"}]}]"#
    );
    let mut sm = t1.status_message.clone().unwrap();
    placeholder_id(&mut sm, "messageId", "msg_");
    assert_eq!(
        sm.to_string(),
        r#"{"messageId":"msg_X","role":"agent","parts":[{"kind":"text","text":"all set"}],"taskId":"AST_t1"}"#
    );
    let t5 = row("AST_t5");
    let mut artifacts = t5.data.artifacts.clone().unwrap();
    placeholder_id(&mut artifacts[0], "artifactId", "art_");
    assert_eq!(
        Value::Array(artifacts).to_string(),
        r#"[{"artifactId":"art_X","parts":[{"kind":"data","data":{"booked":true}}]}]"#
    );
    assert_eq!(t5.status_message, None, "a data result is no message to the person");
    assert_eq!(t5.message_id.as_deref(), Some("m5"));
    let none = row(&minted);
    assert_eq!(none.data.artifacts, Some(vec![]));
    assert_eq!(none.status_message, None);
    assert_eq!(lane.tasks(Some(TaskState::Completed), 50).await.unwrap().len(), 3);
    assert_eq!(lane.task("AST_t4").await.unwrap().unwrap().status, "canceled");
    cleanup(&storage, &database_id, &[&s]).await;
}

#[tokio::test]
async fn concurrent_runs_take_no_lease_and_append() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s1 = thread(&storage, &database_id, "w1").await;
    // The schema and the lane row come from a first exclusive run; the concurrent runs then share
    // a lane at version 1 and never move it.
    s1.lane(LANE)
        .open(RunOptions::new("r0", "AST_0"))
        .await
        .unwrap()
        .commit(now_ms())
        .await
        .unwrap();
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
    assert!(a.lease_token().is_none() && !a.exclusive);
    assert_eq!(a.waiter().await.unwrap(), None);
    a.history.push(message("a1", "user", "one"));
    a.state = Some(json!({"ignored": true}));
    a.agents.push(Ref::new("Nope"));
    a.drained_id = Some(3);
    b.history.push(message("b1", "user", "two"));
    let request = a.commit_request(7);
    assert!(request.fence.is_none());
    assert_eq!(request.state, None);
    assert_eq!(request.agent_stack, None);
    assert_eq!(request.drained_id, None);
    assert_eq!(request.events[0].r#type, "run_started");
    assert_eq!(
        request.events[0].payload,
        json!({"task_id": "AST_1", "stream_id": "AST_1", "channel": "a2a", "exclusive": false})
    );
    b.commit(now_ms()).await.unwrap();
    a.commit(now_ms()).await.unwrap();
    assert_eq!((a.committed_version(), b.committed_version()), (Some(1), Some(1)));

    let lane = s1.lane(LANE);
    assert_eq!(
        lane.history().await.unwrap().iter().map(|i| i.id.as_str()).collect::<Vec<_>>(),
        vec!["b1", "a1"]
    );
    let info = lane.info().await.unwrap().unwrap();
    assert_eq!((info.version, info.lease_until, info.drained_id), (1, 0, 0));
    assert_eq!(info.agent_stack, vec![]);
    let runs = lane.runs(10).await.unwrap();
    assert_eq!(runs.len(), 3);
    assert!(runs.iter().all(|r| r.outcome.as_deref() == Some("completed")));
    // A concurrent run has nothing to abandon.
    let mut c = open(&s1, "r3", "AST_3").await;
    c.abandon(now_ms(), RunOutcome::Canceled, "CancelledError").await.unwrap();
    assert_eq!(s1.events(0, 100).await.unwrap().len(), 8, "two items entered the history");
    cleanup(&storage, &database_id, &[&s1, &s2]).await;
}
