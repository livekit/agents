use std::sync::Arc;
use std::time::Duration;

use crate::WhenHeld;
use crate::testing::{Backend, CommitFault, Faulty, Fixture};
use crate::thread::db::{Receive, ReceiveInput};
use crate::thread::schema::TaskData;
use crate::thread::tests::LANE;
use crate::{CommitOptions, Holder, LaneAccess, RunOptions, RunOutcome, StorageError, Thread};

fn reject(run_id: &str) -> RunOptions {
    RunOptions { access: LaneAccess::Exclusive(WhenHeld::Reject), ..RunOptions::new(run_id, "") }
}

/// A faulty handle on a thread whose schema exists, so a fault never meets the bootstrap.
async fn faulty(fx: &Fixture, owner: &Holder) -> (Thread, Arc<Faulty>) {
    let (thread, faulty) = fx.faulty_thread("w1").await;
    let mut run = thread.lane(LANE).open(RunOptions::new("r0", ""), owner).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();
    (thread, faulty)
}

async fn outcome(thread: &Thread, run_id: &str) -> Option<RunOutcome> {
    let runs = thread.lane(LANE).runs(100).await.unwrap();
    runs.into_iter().find(|r| r.run_id == run_id).and_then(|r| r.outcome)
}

/// Polls until `run_id` ended, as a detached cleanup writes it.
async fn ended(thread: &Thread, run_id: &str) -> Option<RunOutcome> {
    for _ in 0..40 {
        if let Some(outcome) = outcome(thread, run_id).await {
            return Some(outcome);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    None
}

/// An acquire whose future is dropped after its batch committed releases the lease and ends
/// its run `canceled`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_acquire_dropped_after_its_write_releases_and_ends_its_run() {
    let fx = Backend::Sqlite.fixture(None).await;
    let owner = Holder::new("voice");
    let (thread, faulty) = faulty(&fx, &owner).await;
    let lane = thread.lane(LANE);
    faulty.fail_next_lease_take(CommitFault::StallAfter);
    let open = lane.open(RunOptions::new("r1", ""), &owner);
    assert!(tokio::time::timeout(Duration::from_millis(200), open).await.is_err());
    assert_eq!(ended(&thread, "r1").await, Some(RunOutcome::Canceled));
    let other = fx.thread("w2").await;
    let mut next = other.lane(LANE).open(reject("r2"), &Holder::new("other")).await.unwrap();
    next.commit(CommitOptions::default()).await.unwrap();
    fx.cleanup(&[&thread, &other]).await;
}

/// When the dropped acquire's cleanup cannot reach the database, the orphaned lease still
/// belongs to its owner: the owner re-enters it at once, from another handle too, while another
/// holder is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_orphaned_lease_is_reentered_by_its_owner_and_fenced_from_others() {
    let fx = Backend::Sqlite.fixture(None).await;
    let owner = Holder::new("voice");
    let (thread, faulty) = faulty(&fx, &owner).await;
    faulty.fail_next_lease_take(CommitFault::StallAfter);
    faulty.unreachable_after_commit_for(Duration::from_millis(300));
    let lane = thread.lane(LANE);
    let open = lane.open(RunOptions::new("r1", ""), &owner);
    assert!(tokio::time::timeout(Duration::from_millis(100), open).await.is_err());
    tokio::time::sleep(Duration::from_millis(300)).await;
    let other = fx.thread("w2").await;
    let refused = other.lane(LANE).open(reject("r2"), &Holder::new("other")).await;
    assert!(matches!(refused, Err(StorageError::Rejected { .. })), "fenced");
    let reopened = fx.thread("w1").await;
    let mut next = reopened.lane(LANE).open(reject("r3"), &owner).await.expect("re-entered");
    next.commit(CommitOptions::default()).await.unwrap();
    fx.cleanup(&[&thread, &other, &reopened]).await;
}

/// A run dropped without commit or abandon releases its lease: no holder waits for the TTL.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_run_releases_its_lease() {
    let fx = Backend::Sqlite.fixture(None).await;
    let thread = fx.thread("w1").await;
    let (owner, lane) = (Holder::new("voice"), thread.lane(LANE));
    drop(lane.open(RunOptions::new("r1", ""), &owner).await.unwrap());
    let mut again = lane.open(reject("r2"), &owner).await.expect("the owner takes it at once");
    again.commit(CommitOptions::default()).await.unwrap();
    drop(lane.open(RunOptions::new("r3", ""), &owner).await.unwrap());
    let other = fx.thread("w2").await;
    let other_owner = Holder::new("other");
    let mut taken = None;
    for _ in 0..40 {
        match other.lane(LANE).open(reject("r4"), &other_owner).await {
            Ok(run) => {
                taken = Some(run);
                break;
            }
            Err(StorageError::Rejected { .. }) => {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            Err(error) => panic!("{error}"),
        }
    }
    let mut taken = taken.expect("the dropped run's lease was released");
    taken.commit(CommitOptions::default()).await.unwrap();
    fx.cleanup(&[&thread, &other]).await;
}

/// An acquire whose acknowledgment was lost, applied or not, is retried with its token.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_uncertain_acquire_is_retried() {
    for fault in [CommitFault::LoseAckBefore, CommitFault::LoseAckAfter] {
        let fx = Backend::Sqlite.fixture(None).await;
        let owner = Holder::new("voice");
        let (thread, faulty) = faulty(&fx, &owner).await;
        faulty.fail_next_lease_take(fault);
        let mut run = thread.lane(LANE).open(reject("r1"), &owner).await.expect("retried");
        run.commit(CommitOptions::default()).await.unwrap();
        assert_eq!(outcome(&thread, "r1").await, Some(RunOutcome::Completed), "{fault:?}");
        let other = fx.thread("w2").await;
        let mut next = other.lane(LANE).open(reject("r2"), &Holder::new("other")).await.unwrap();
        next.commit(CommitOptions::default()).await.unwrap();
        fx.cleanup(&[&thread, &other]).await;
    }
}

/// Another holder's dropped acquire never releases the lease the owner holds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn another_holders_dropped_acquire_leaves_the_owners_lease() {
    let fx = Backend::Sqlite.fixture(None).await;
    let thread = fx.thread("w1").await;
    let lane = thread.lane(LANE);
    let mut held = lane.open(RunOptions::new("r1", ""), &Holder::new("voice")).await.unwrap();
    let other = fx.thread("w2").await;
    let other_lane = other.lane(LANE);
    let other_owner = Holder::new("other");
    let waiting = other_lane.open(RunOptions::new("r2", ""), &other_owner);
    assert!(tokio::time::timeout(Duration::from_millis(300), waiting).await.is_err());
    tokio::time::sleep(Duration::from_millis(200)).await;
    held.commit(CommitOptions::default()).await.expect("the owner kept its lease");
    fx.cleanup(&[&thread, &other]).await;
}

/// An inbox receive whose acknowledgment was lost is retried: the lease it took is returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_uncertain_receive_is_retried() {
    for fault in [CommitFault::LoseAckBefore, CommitFault::LoseAckAfter] {
        let fx = Backend::Sqlite.fixture(None).await;
        let owner = Holder::new("request");
        let (thread, faulty) = faulty(&fx, &owner).await;
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
        let other = fx.thread("w2").await;
        let refused = other.lane(LANE).open(reject("r2"), &Holder::new("other")).await;
        assert!(matches!(refused, Err(StorageError::Rejected { .. })), "{fault:?}: fenced");
        lease.release(None).await.unwrap();
        fx.cleanup(&[&thread, &other]).await;
    }
}
