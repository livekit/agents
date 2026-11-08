use std::time::{Duration, Instant};

use livekit_protocol::agent::ChatRole;
use livekit_protocol::agent::chat_context::chat_item;
use lk_agents_abi::storage::journal::JournalEvent;
use serde_json::json;

use super::*;
use crate::testing::{Backend, Fixture};
use crate::thread::tests::{LANE, message, submitted};
use crate::{CommitOptions, LaneAccess, RunOptions, StorageError, TaskPickup, Thread, WhenHeld};

async fn waiter(s: &Thread) -> Option<String> {
    s.lane(LANE).info().await.unwrap().expect("a lane row").waiter
}

#[test]
fn item_changes_skip_instructions_and_unchanged_items_and_list_deletions() {
    use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
    use livekit_protocol::agent::{AgentConfigUpdate, ChatMessage, ChatRole, FunctionCall};
    let msg = |id: &str, role: ChatRole, text: &str| {
        ItemRow::new(chat_item::Item::Message(ChatMessage {
            id: id.into(),
            role: role as i32,
            content: vec![ChatContent { payload: Some(Payload::Text(text.into())) }],
            ..Default::default()
        }))
    };
    let call = ItemRow::new(chat_item::Item::FunctionCall(FunctionCall {
        id: "i3".into(),
        name: "f".into(),
        ..Default::default()
    }));
    let loaded: Vec<(String, ChatItem)> =
        [msg("i1", ChatRole::User, "hi"), msg("i2", ChatRole::Assistant, "hello"), call]
            .into_iter()
            .map(|i| (i.id, i.payload))
            .collect();
    let mut history = vec![
        msg("sys", ChatRole::System, "be nice"),
        msg("dev", ChatRole::Developer, "x"),
        ItemRow::new(chat_item::Item::AgentConfigUpdate(AgentConfigUpdate {
            id: "cfg".into(),
            ..Default::default()
        })),
        msg("i1", ChatRole::User, "hi"),
        // edited
        msg("i2", ChatRole::Assistant, "hey"),
        // i3 removed; i4 new
        ItemRow { message_id: Some("m4".into()), ..msg("i4", ChatRole::User, "bye") },
    ];
    let (changed, deleted) = item_changes(&history, &loaded);
    assert_eq!(
        changed.iter().map(|w| (w.id.as_str(), w.message_id.as_deref())).collect::<Vec<_>>(),
        vec![("i2", None), ("i4", Some("m4"))]
    );
    assert_eq!(deleted, vec!["i3".to_owned()]);
    history.clear();
    let (changed, deleted) = item_changes(&history, &loaded);
    assert!(changed.is_empty());
    assert_eq!(deleted, vec!["i1", "i2", "i3"]);
}

#[test]
fn run_summaries_follow_run_started_and_run_ended() {
    let ev = |id: i64, run_id: &str, ty: &str, payload: Value| EventRow {
        id,
        ts_ms: id * 10,
        lane: "front-desk".into(),
        run_id: run_id.into(),
        r#type: ty.into(),
        payload,
    };
    let events = vec![
        ev(1, "r1", "run_started", json!({"task_id": "AST_1", "channel": "a2a"})),
        ev(2, "r1", "run_ended", json!({"outcome": "completed", "reason": null})),
        ev(3, "r2", "run_started", json!({"task_id": "AST_2", "channel": null})),
        ev(4, "r9", "run_ended", json!({"outcome": "failed"})),
    ];
    let runs = RunSummary::from_events(&events).unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].outcome, Some(RunOutcome::Completed));
    assert_eq!(runs[0].ended_at_ms, Some(20));
    assert!(!runs[0].interrupted());
    assert_eq!(runs[1].channel, None);
    assert!(runs[1].interrupted());
    // A malformed entry is refused, not skipped.
    let corrupt = [ev(5, "r3", "run_ended", json!({"outcome": "maybe"}))];
    assert!(matches!(RunSummary::from_events(&corrupt), Err(StorageError::Invalid(_))));
}

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
    assert!(!run1.closed() && run1.lease().is_some());
    assert!(run1.history().is_empty());
    assert_eq!(run1.state(), None);
    let journal = s1.journal(0, 100).await.unwrap();
    assert_eq!(journal.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(), vec!["run_started"]);
    assert!(run1.lease().is_some_and(|h| !h.is_lost()));
    assert_eq!(waiter(&s1).await, None);

    // A second holder: refused at once without the wait, and timed out with a short one.
    let lane2 = s2.lane(LANE);
    let refused = lane2
        .open(RunOptions {
            access: LaneAccess::Exclusive(WhenHeld::Reject),
            ..RunOptions::new("r2", "AST_2")
        })
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
                access: LaneAccess::Exclusive(WhenHeld::Interrupt),
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
    run1.history_mut().push(message("i1", ChatRole::User, "hi"));
    run1.history_mut().last_mut().unwrap().message_id = Some("m1".into());
    run1.history_mut().push(message("i2", ChatRole::Assistant, "hello"));
    run1.history_mut().push(message("sys", ChatRole::System, "be nice"));
    run1.set_state(json!({"patient": "Ann"}));
    run1.events().append(JournalEvent::Error { error: "boom".into() });
    let mut usage = lk_agents_abi::usage::Usage::default();
    usage.requests = 1;
    usage.tokens.input_tokens = 10;
    run1.set_usage(usage);
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
    assert_eq!(run2.history().iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), vec!["i1", "i2"]);
    let first = ItemRow { message_id: Some("m1".into()), ..message("i1", ChatRole::User, "hi") };
    assert_eq!(run2.history()[0], first);
    assert_eq!(run2.state(), Some(&json!({"patient": "Ann"})));
    assert_eq!(waiter(&s2).await, None, "the waiter that won cleared its own mark");
    assert!(s2.journal(0, 100).await.unwrap().iter().any(|e| e.r#type == "usage"));

    // An unchanged history writes no item; a changed state is written.
    assert!(run2.writes(CommitOptions::default()).unwrap().items.is_empty());
    let unchanged = run2.writes(CommitOptions::default()).unwrap().state;
    assert_eq!(unchanged, None, "unchanged state is not rewritten");
    run2.history_mut().retain(|i| i.id != "i2");
    run2.set_state(json!({"patient": "Ann", "seen": true}));
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
    run.history_mut().push(message("i1", ChatRole::User, "hi"));
    run.set_state(json!({"x": 1}));
    run.events().append(JournalEvent::Error { error: "boom".into() });
    run.tasks()
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
    let next = lane
        .open(RunOptions {
            access: LaneAccess::Exclusive(WhenHeld::Reject),
            ..RunOptions::new("r2", "AST_2")
        })
        .await
        .unwrap();
    assert!(next.history().is_empty() && next.state().is_none());
    assert_eq!(next.version(), 0);
    drop(next);
    fx.cleanup(&[&s]).await;
}

async fn a_committed_run_id_yields_a_closed_run(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.history_mut().push(message("i1", ChatRole::User, "hi"));
    run.commit(CommitOptions::default()).await.unwrap();

    let mut again = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    assert!(again.closed());
    assert_eq!(again.version(), 1);
    assert_eq!(again.history().len(), 1, "loaded, so a retry can answer from it");
    again.history_mut().push(message("i2", ChatRole::Assistant, "hello"));
    let committed = again.commit(CommitOptions::default()).await.unwrap();
    assert_eq!(committed.version, 1, "no bump");
    assert_eq!(lane.info().await.unwrap().unwrap().version, 1);
    assert_eq!(lane.history().await.unwrap().len(), 1, "a closed run writes nothing");
    let events = s.journal(0, 100).await.unwrap();
    assert_eq!(events.iter().filter(|e| e.r#type == "run_started").count(), 1, "not again");
    assert_eq!(events.iter().filter(|e| e.r#type == "run_ended").count(), 1);
    // Released: a third open takes the lane without waiting.
    let free = lane
        .open(RunOptions {
            access: LaneAccess::Exclusive(WhenHeld::Reject),
            ..RunOptions::new("r2", "AST_2")
        })
        .await
        .unwrap();
    assert!(!free.closed());
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
        .open(RunOptions {
            access: LaneAccess::Exclusive(WhenHeld::Reject),
            ..RunOptions::new("r2", "AST_2")
        })
        .await
        .unwrap();
    fresh.history_mut().push(message("i1", ChatRole::User, "hi"));
    assert_eq!(fresh.commit(CommitOptions::default()).await.unwrap().version, 1);

    stale.history_mut().push(message("i9", ChatRole::Assistant, "late"));
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
            access: LaneAccess::Concurrent,
            channel: Some("a2a".into()),
            ..RunOptions::new(run_id, task_id)
        };
        async move { lane.open(opts).await.unwrap() }
    };
    let mut a = open(&s1, "r1", "AST_1").await;
    let mut b = open(&s2, "r2", "AST_2").await;
    assert!(a.lease().is_none());
    a.history_mut().push(message("a1", ChatRole::User, "one"));
    a.set_state(json!({"ignored": true}));
    b.history_mut().push(message("b1", ChatRole::User, "two"));
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

fn metadata(key: &str) -> serde_json::Map<String, serde_json::Value> {
    json!({ key: true }).as_object().cloned().unwrap()
}

/// A terminal row is immutable. A cancel arriving after the commit leaves the completed
/// task as it was.
async fn a_cancel_never_overwrites_a_terminal_task(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.tasks().write("AST_1", TaskState::Completed, TaskPatch::default()).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();

    let after = lane.cancel_task("AST_1", metadata("late")).await.unwrap().expect("the row");
    assert_eq!(after.status, TaskState::Completed);
    assert!(after.data.metadata.is_empty(), "nothing merged into a terminal row");
    assert_eq!(lane.task("AST_1").await.unwrap().unwrap().status, TaskState::Completed);
    fx.cleanup(&[&s]).await;
}

/// A client's cancel during the run wins: the run's commit lands its history but cannot
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

    run.history_mut().push(message("i1", ChatRole::User, "hi"));
    run.tasks().write("AST_1", TaskState::Completed, TaskPatch::default()).await.unwrap();
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

/// A row a turn answered `working` (its handler's `Working` outcome) is no run's: a pickup leaves
/// it as the turn answered it rather than failing it as a dead holder's.
async fn a_pickup_leaves_a_row_answered_working(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    lane.submit(&submitted("AST_1")).await.unwrap();
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    let answered = TaskPatch { answered: true, ..TaskPatch::default() };
    run.tasks().write("AST_1", TaskState::Working, answered).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();

    assert_eq!(lane.pickup("AST_1").await.unwrap(), None, "no dead holder's row");
    let row = lane.task("AST_1").await.unwrap().unwrap();
    assert_eq!(row.status, TaskState::Working);
    assert!(row.internal.is_some_and(|i| i.answered));
    fx.cleanup(&[&s]).await;
}

/// A concurrent run reads its rows live, so a client's cancel mid-turn is the base of its write.
/// The write keeps the client's row, as the commit would, instead of refusing the whole turn.
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
    let opts = RunOptions {
        access: LaneAccess::Concurrent,
        pickup: TaskPickup::Claim,
        ..RunOptions::new("r1", "AST_1")
    };
    let mut run = lane.open(opts).await.unwrap();
    assert!(run.picked_up());
    s2.lane(LANE).cancel_task("AST_1", metadata("client")).await.unwrap();

    run.history_mut().push(message("i1", ChatRole::User, "hi"));
    run.tasks().write("AST_1", TaskState::Completed, TaskPatch::default()).await.unwrap();
    let committed = run
        .commit(CommitOptions { answer_task: Some("AST_1".into()), ..CommitOptions::default() })
        .await
        .unwrap();
    assert_eq!(committed.task.expect("read back").status, TaskState::Canceled);
    assert_eq!(lane.history().await.unwrap().len(), 1, "the rest of the turn landed");
    fx.cleanup(&[&s1, &s2]).await;
}

async fn an_item_id_of_another_lane_fails_the_commit_without_mutation(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let mut a = s.lane("lane-a").open(RunOptions::new("ra", "AST_a")).await.unwrap();
    a.history_mut().push(message("shared", ChatRole::User, "a's"));
    a.commit(CommitOptions::default()).await.unwrap();

    let mut b = s.lane("lane-b").open(RunOptions::new("rb", "AST_b")).await.unwrap();
    b.history_mut().push(message("b1", ChatRole::User, "b's own"));
    b.history_mut().push(message("shared", ChatRole::User, "b's"));
    assert!(b.commit(CommitOptions::default()).await.is_err());
    let a_items = s.lane("lane-a").history().await.unwrap();
    assert_eq!(a_items, vec![message("shared", ChatRole::User, "a's")]);
    assert!(s.lane("lane-b").history().await.unwrap().is_empty(), "atomic: nothing of b landed");
    fx.cleanup(&[&s]).await;
}

/// A `concurrent` run appends only: editing or deleting a loaded item is refused and
/// nothing is written.
async fn a_concurrent_run_refuses_to_edit_or_delete_loaded_history(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut first = lane.open(RunOptions::new("r0", "AST_0")).await.unwrap();
    first.history_mut().push(message("i1", ChatRole::User, "hi"));
    first.commit(CommitOptions::default()).await.unwrap();
    let concurrent = |run_id: &str| RunOptions {
        access: LaneAccess::Concurrent,
        ..RunOptions::new(run_id, "AST_1")
    };

    let mut deleting = lane.open(concurrent("r1")).await.unwrap();
    deleting.history_mut().clear();
    deleting.history_mut().push(message("i2", ChatRole::User, "new"));
    assert!(matches!(
        deleting.commit(CommitOptions::default()).await,
        Err(StorageError::Invalid(_))
    ));
    let mut editing = lane.open(concurrent("r2")).await.unwrap();
    editing.history_mut()[0] = message("i1", ChatRole::User, "rewritten");
    assert!(matches!(
        editing.commit(CommitOptions::default()).await,
        Err(StorageError::Invalid(_))
    ));
    assert_eq!(lane.history().await.unwrap(), vec![message("i1", ChatRole::User, "hi")]);
    fx.cleanup(&[&s]).await;
}

/// The acquire loads the whole history in the lease's own batch, past any page size.
async fn a_run_loads_its_whole_history_in_its_acquire(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    for i in 0..600 {
        run.history_mut().push(message(&format!("i{i}"), ChatRole::User, "hi"));
    }
    run.commit(CommitOptions::default()).await.unwrap();
    let again = lane.open(RunOptions::new("r2", "AST_2")).await.unwrap();
    assert_eq!(again.history().len(), 600);
    assert_eq!(again.history().last().map(|i| i.id.as_str()), Some("i599"));
    drop(again);
    fx.cleanup(&[&s]).await;
}

crate::contract!(
    a_run_loads_its_whole_history_in_its_acquire,
    a_concurrent_run_refuses_to_edit_or_delete_loaded_history,
    an_item_id_of_another_lane_fails_the_commit_without_mutation,
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

// A lost commit acknowledgment resolves to the original run (SQLite, injected faults).

/// A run that writes one item and completes `AST_1`, committed over a faulty connection.
async fn commit_through(
    fault: crate::testing::CommitFault,
) -> (Fixture, Thread, Result<i64, StorageError>) {
    let fx = Backend::Sqlite.fixture(None).await;
    let (s, faults) = fx.faulty_thread("w1").await;
    let lane = s.lane(LANE);
    let mut run = lane.open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.history_mut().push(message("i1", ChatRole::User, "hi"));
    run.tasks().write("AST_1", TaskState::Completed, TaskPatch::default()).await.unwrap();
    faults.fail_next_commit(fault);
    let committed = run.commit(CommitOptions::default()).await.map(|c| c.version);
    (fx, s, committed)
}

async fn run_ended_count(s: &Thread) -> usize {
    s.journal(0, 100).await.unwrap().iter().filter(|e| e.r#type == "run_ended").count()
}

#[tokio::test]
async fn a_commit_whose_ack_was_lost_after_it_applied_resolves_to_that_commit() {
    let (fx, s, committed) = commit_through(crate::testing::CommitFault::LoseAckAfter).await;
    assert_eq!(committed.unwrap(), 1, "the original run's commit, not a failure");
    assert_eq!(s.lane(LANE).history().await.unwrap().len(), 1);
    assert_eq!(s.lane(LANE).task("AST_1").await.unwrap().unwrap().status, TaskState::Completed);
    assert_eq!(run_ended_count(&s).await, 1, "written once");
    fx.cleanup(&[&s]).await;
}

#[tokio::test]
async fn a_commit_whose_ack_was_lost_before_it_applied_is_sent_again_under_its_fence() {
    let (fx, s, committed) = commit_through(crate::testing::CommitFault::LoseAckBefore).await;
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

    stale.history_mut().push(message("i9", ChatRole::Assistant, "late"));
    faults.fail_next_commit(crate::testing::CommitFault::LoseAckBefore);
    let err = stale.commit(CommitOptions::default()).await.expect_err("rejected");
    assert!(matches!(err, StorageError::VersionConflict { .. }), "{err}");
    assert!(s1.lane(LANE).history().await.unwrap().is_empty());
    fx.cleanup(&[&s1, &s2]).await;
}
