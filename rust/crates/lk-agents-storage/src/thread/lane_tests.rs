use std::sync::Arc;
use std::time::Duration;

use crate::WhenHeld;
use crate::testing::{Faulty, TempDatabase, WriteFault};
use crate::thread::db::{Receive, ReceiveInput};
use crate::thread::schema::TaskData;
use crate::thread::tests::LANE;
use crate::thread::{LeaseTiming, SchemaCell};
use crate::{CommitOptions, Holder, LaneAccess, RunOptions, RunOutcome, StorageError, Thread};

// One database file on paused time: its workers are tasks, so a wait is virtual.

fn reject(run_id: &str) -> RunOptions {
    RunOptions { access: LaneAccess::Exclusive(WhenHeld::Reject), ..RunOptions::new(run_id, "") }
}

fn database() -> TempDatabase {
    TempDatabase::new()
}

/// One worker's handle on the database.
async fn handle(db: &TempDatabase) -> Thread {
    let timing = LeaseTiming::default();
    Thread::new(Arc::new(db.connect().await), "AT_lane", timing, SchemaCell::default())
        .await
        .unwrap()
}

/// A faulty handle on a database whose schema exists, so a fault never meets the bootstrap.
async fn faulty(db: &TempDatabase, owner: &Holder) -> (Thread, Arc<Faulty>) {
    let faulty = Arc::new(Faulty::new(Box::new(db.connect().await)));
    let timing = LeaseTiming::default();
    let thread =
        Thread::new(faulty.clone(), "AT_lane", timing, SchemaCell::default()).await.unwrap();
    let mut run = thread.lane(LANE).open(RunOptions::new("r0", ""), owner).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();
    (thread, faulty)
}

async fn outcome(thread: &Thread, run_id: &str) -> Option<RunOutcome> {
    let runs = thread.lane(LANE).runs(100).await.unwrap();
    runs.into_iter().find(|r| r.run_id == run_id).and_then(|r| r.outcome)
}

/// Polls `check` every 50 ms, up to two seconds, until it answers: what a detached cleanup
/// leaves shows up shortly.
async fn eventually<T, F: Future<Output = Option<T>>>(mut check: impl FnMut() -> F) -> Option<T> {
    for _ in 0..40 {
        if let Some(found) = check().await {
            return Some(found);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    None
}

/// An acquire whose future is dropped after its batch committed releases the lease and ends
/// its run `canceled`.
#[tokio::test(start_paused = true)]
async fn an_acquire_dropped_after_its_write_releases_and_ends_its_run() {
    let db = database();
    let owner = Holder::new("voice");
    let (thread, faulty) = faulty(&db, &owner).await;
    let lane = thread.lane(LANE);
    faulty.fail_next_lease_take(WriteFault::StallAfter);
    let open = lane.open(RunOptions::new("r1", ""), &owner);
    assert!(tokio::time::timeout(Duration::from_millis(200), open).await.is_err());
    let ended = eventually(|| outcome(&thread, "r1")).await;
    assert_eq!(ended, Some(RunOutcome::Canceled));
    let other = handle(&db).await;
    let mut next = other.lane(LANE).open(reject("r2"), &Holder::new("other")).await.unwrap();
    next.commit(CommitOptions::default()).await.unwrap();
}

/// When the dropped acquire's cleanup cannot reach the database, the orphaned lease still
/// belongs to its owner: the owner re-enters it at once, from another handle too, while another
/// holder is refused.
#[tokio::test(start_paused = true)]
async fn an_orphaned_lease_is_reentered_by_its_owner_and_fenced_from_others() {
    let db = database();
    let owner = Holder::new("voice");
    let (thread, faulty) = faulty(&db, &owner).await;
    faulty.fail_next_lease_take(WriteFault::StallAfter);
    faulty.unreachable_after_commit_for(Duration::from_millis(300));
    let lane = thread.lane(LANE);
    let open = lane.open(RunOptions::new("r1", ""), &owner);
    assert!(tokio::time::timeout(Duration::from_millis(100), open).await.is_err());
    tokio::time::sleep(Duration::from_millis(300)).await;
    let other = handle(&db).await;
    let refused = other.lane(LANE).open(reject("r2"), &Holder::new("other")).await;
    assert!(matches!(refused, Err(StorageError::Rejected { .. })), "fenced");
    let reopened = handle(&db).await;
    let mut next = reopened.lane(LANE).open(reject("r3"), &owner).await.expect("re-entered");
    next.commit(CommitOptions::default()).await.unwrap();
}

/// A run dropped without commit or abandon releases its lease: no holder waits for the TTL.
#[tokio::test(start_paused = true)]
async fn a_dropped_run_releases_its_lease() {
    let db = database();
    let thread = handle(&db).await;
    let (owner, lane) = (Holder::new("voice"), thread.lane(LANE));
    drop(lane.open(RunOptions::new("r1", ""), &owner).await.unwrap());
    let mut again = lane.open(reject("r2"), &owner).await.expect("the owner takes it at once");
    again.commit(CommitOptions::default()).await.unwrap();
    drop(lane.open(RunOptions::new("r3", ""), &owner).await.unwrap());
    let (other, other_owner) = (handle(&db).await.lane(LANE), Holder::new("other"));
    let (other, other_owner) = (&other, &other_owner);
    let taken = eventually(|| async move {
        match other.open(reject("r4"), other_owner).await {
            Ok(run) => Some(run),
            Err(StorageError::Rejected { .. }) => None,
            Err(error) => panic!("{error}"),
        }
    });
    let mut taken = taken.await.expect("the dropped run's lease was released");
    taken.commit(CommitOptions::default()).await.unwrap();
    // r1's end is fenced out: its owner took the lane again before the release ran
    assert_eq!(outcome(&thread, "r3").await, Some(RunOutcome::Canceled), "the dropped run ended");
}

/// A waiting acquire notices another worker's release, over another connection, within one
/// poll pause.
#[tokio::test(start_paused = true)]
async fn a_waiting_acquire_takes_a_lane_another_worker_released_within_a_poll() {
    let db = database();
    let (holder, waiter) = (handle(&db).await, handle(&db).await);
    let mut held =
        holder.lane(LANE).open(RunOptions::new("r1", ""), &Holder::new("a")).await.unwrap();
    let lane = waiter.lane(LANE);
    let owner = Holder::new("b");
    let waiting = tokio::spawn(async move {
        let run = lane.open(RunOptions::new("r2", ""), &owner).await;
        (run.map(drop), tokio::time::Instant::now())
    });
    tokio::time::sleep(Duration::from_secs(4)).await;
    held.commit(CommitOptions::default()).await.unwrap();
    let released = tokio::time::Instant::now();
    let (taken, at) = waiting.await.unwrap();
    taken.expect("the waiter takes the lane");
    assert!(at - released <= Duration::from_millis(1_100), "taken {:?} after", at - released);
}

/// An acquire whose acknowledgment was lost, applied or not, is retried with its token.
#[tokio::test(start_paused = true)]
async fn an_uncertain_acquire_is_retried() {
    for fault in [WriteFault::LoseAckBefore, WriteFault::LoseAckAfter] {
        let db = database();
        let owner = Holder::new("voice");
        let (thread, faulty) = faulty(&db, &owner).await;
        faulty.fail_next_lease_take(fault);
        let mut run = thread.lane(LANE).open(reject("r1"), &owner).await.expect("retried");
        run.commit(CommitOptions::default()).await.unwrap();
        assert_eq!(outcome(&thread, "r1").await, Some(RunOutcome::Completed), "{fault:?}");
        let other = handle(&db).await;
        let mut next = other.lane(LANE).open(reject("r2"), &Holder::new("other")).await.unwrap();
        next.commit(CommitOptions::default()).await.unwrap();
    }
}

/// Another holder's dropped acquire never releases the lease the owner holds.
#[tokio::test(start_paused = true)]
async fn another_holders_dropped_acquire_leaves_the_owners_lease() {
    let db = database();
    let thread = handle(&db).await;
    let mut held =
        thread.lane(LANE).open(RunOptions::new("r1", ""), &Holder::new("voice")).await.unwrap();
    let other = handle(&db).await;
    let other_owner = Holder::new("other");
    let other_lane = other.lane(LANE);
    let waiting = other_lane.open(RunOptions::new("r2", ""), &other_owner);
    assert!(tokio::time::timeout(Duration::from_millis(300), waiting).await.is_err());
    tokio::time::sleep(Duration::from_millis(200)).await;
    held.commit(CommitOptions::default()).await.expect("the owner kept its lease");
}

/// An inbox receive whose acknowledgment was lost is retried: the lease it took is returned.
#[tokio::test(start_paused = true)]
async fn an_uncertain_receive_is_retried() {
    for fault in [WriteFault::LoseAckBefore, WriteFault::LoseAckAfter] {
        let db = database();
        let owner = Holder::new("request");
        let (thread, faulty) = faulty(&db, &owner).await;
        let message: a2a::Message = serde_json::from_value(serde_json::json!({
            "messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hello"}]
        }))
        .unwrap();
        let data = TaskData { history: vec![message.clone()], ..Default::default() };
        let input =
            ReceiveInput { task_id: "AST_1", message: &message, data: &data, internal: None };
        faulty.fail_next_lease_take(fault);
        let (received, lease) = thread.lane(LANE).receive(&owner, input).await.expect("retried");
        assert!(matches!(received, Receive::Received(Some(_))), "{fault:?}: {received:?}");
        let lease = lease.expect("the receive took the free lane");
        let other = handle(&db).await;
        let refused = other.lane(LANE).open(reject("r2"), &Holder::new("other")).await;
        assert!(matches!(refused, Err(StorageError::Rejected { .. })), "{fault:?}: fenced");
        lease.release(None).await.unwrap();
    }
}

/// Each lane write whose acknowledgment is lost after it applied is sent again under its key and
/// answers as applied, never `Uncertain` and never "already there".
#[tokio::test(start_paused = true)]
async fn lane_writes_whose_answer_was_lost_answer_as_applied() {
    let db = database();
    let owner = Holder::new("voice");
    let (thread, faulty) = faulty(&db, &owner).await;
    let lane = thread.lane(LANE);

    faulty.fail_next_write("INSERT OR IGNORE INTO tasks", WriteFault::LoseAckAfter);
    let submitted = lane.submit(&crate::thread::tests::submitted("AST_1")).await;
    assert!(submitted.unwrap(), "the row the first send inserted is this submit's");
    lane.pickup("AST_1").await.unwrap();

    let failed = crate::thread::run::agent_message(Vec::new(), "AST_1", "msg_failed".into());
    faulty.fail_next_write("SET status = 'failed'", WriteFault::LoseAckAfter);
    assert!(lane.fail_orphan("AST_1", &failed).await.unwrap(), "failed by this write");

    let held = lane.open(RunOptions::new("r1", ""), &owner).await.unwrap();
    faulty.fail_next_write("SET waiter = :holder", WriteFault::LoseAckAfter);
    assert!(lane.set_waiter("other").await.unwrap());

    faulty.fail_next_write("lease_until = 0", WriteFault::LoseAckAfter);
    held.lease().unwrap().release(None).await.expect("a lost release is sent again");
}
