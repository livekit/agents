use std::time::Duration;

use livekit_protocol::agent::ChatRole;
use serde_json::json;

use super::*;
use crate::testing::Backend;
use crate::thread::tests::{LANE, message, submitted};
use crate::{CommitOptions, LaneAccess, LeaseAfterCommit, RunOptions, RunOutcome, StorageError};
use crate::{TaskPickup, Thread, WhenHeld};
use lk_agents_abi::storage::thread::Backoff;

#[test]
fn the_poll_backs_off_to_a_second_and_stops_at_its_deadline() {
    let t0 = Duration::from_secs(100);
    let ms = Duration::from_millis;
    let mut p = Backoff::new(t0, ms(2000));
    assert_eq!(p.next(t0), Some(t0 + ms(100)));
    assert_eq!(p.next(t0 + ms(100)), Some(t0 + ms(300)));
    assert_eq!(p.next(t0 + ms(300)), Some(t0 + ms(700)));
    assert_eq!(p.next(t0 + ms(700)), Some(t0 + ms(1500)));
    assert_eq!(p.next(t0 + ms(1500)), Some(t0 + ms(2500)));
    assert_eq!(p.next(t0 + ms(2500)), None);
    assert_eq!(LeaseTiming::new(ms(1500), ms(1)).unwrap().secs(), 2);
}

/// A run dropped without commit or abandon stops renewing: its lease lapses within the TTL
/// and another holder takes the lane.
async fn a_dropped_run_stops_renewing_and_its_lease_lapses(backend: Backend) {
    let fx = backend.fixture(Some((Duration::from_secs(1), Duration::from_millis(300)))).await;
    let s1 = fx.thread().await;
    let s2 = fx.thread().await;
    let run = s1
        .lane(LANE)
        .open(RunOptions::new("r1", "AST_1"), &crate::Holder::new("test"))
        .await
        .unwrap();
    drop(run);
    tokio::time::sleep(Duration::from_millis(2600)).await;
    let lane2 = s2.lane(LANE);
    let next = lane2
        .open(
            RunOptions {
                access: LaneAccess::Exclusive(WhenHeld::Reject),
                ..RunOptions::new("r2", "AST_2")
            },
            &crate::Holder::new("test"),
        )
        .await;
    let next = next.expect("the dropped run's lease lapsed");
    drop(next);
    fx.cleanup(&[&s1, &s2]).await;
}

/// Waiters run in the order their messages' rows were submitted: a later message cannot take a
/// free lane while an older one still waits for it.
async fn waiters_take_the_lane_in_the_order_their_rows_were_submitted(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread().await;
    let lane = s.lane(LANE);
    lane.open(RunOptions::new("r0", ""), &crate::Holder::new("test"))
        .await
        .unwrap()
        .commit(CommitOptions::default())
        .await
        .unwrap();
    for task_id in ["AST_first", "AST_second"] {
        assert!(lane.submit(&submitted(task_id)).await.unwrap());
    }
    let second = || RunOptions {
        access: LaneAccess::Exclusive(WhenHeld::Reject),
        pickup: TaskPickup::Claim,
        ..RunOptions::new("r2", "AST_second")
    };
    let refused = lane
        .open(second(), &crate::Holder::new("test"))
        .await
        .err()
        .expect("the older waiter goes first");
    assert!(matches!(refused, StorageError::Rejected { .. }), "{refused}");
    let first = RunOptions {
        access: LaneAccess::Exclusive(WhenHeld::Reject),
        pickup: TaskPickup::Claim,
        ..RunOptions::new("r1", "AST_first")
    };
    let mut run = lane.open(first, &crate::Holder::new("test")).await.unwrap();
    assert!(run.picked_up());
    run.commit(CommitOptions::default()).await.unwrap();
    // the first row is working now: the second is the oldest waiter and takes the lane
    lane.open(second(), &crate::Holder::new("test"))
        .await
        .unwrap()
        .commit(CommitOptions::default())
        .await
        .unwrap();
    fx.cleanup(&[&s]).await;
}

/// A lost handle cannot restart renewal. Takeover replaces its token even before the new
/// owner commits, so the old owner's unchanged version is not enough to authorize a write.
async fn an_expired_owner_cannot_restart_or_commit_after_takeover(backend: Backend) {
    let fx = backend.fixture(Some((Duration::from_secs(1), Duration::from_millis(200)))).await;
    let first = fx.thread().await;
    let second = fx.thread().await;
    let mut stale = first
        .lane(LANE)
        .open(RunOptions::new("r1", "AST_1"), &crate::Holder::new("test"))
        .await
        .unwrap();
    let old_lease = stale.lease().unwrap().clone();
    old_lease.stop();
    tokio::time::sleep(Duration::from_millis(2100)).await;
    assert!(old_lease.is_lost());
    old_lease.keep_alive();
    assert!(old_lease.is_lost(), "restarting renewal cannot revive an expired handle");

    let owner = second
        .lane(LANE)
        .open(RunOptions::new("r2", "AST_2"), &crate::Holder::new("test"))
        .await
        .unwrap();
    assert_eq!(stale.version(), owner.version(), "takeover alone does not change the version");
    assert_ne!(old_lease.fence().token, owner.lease().unwrap().fence().token);
    stale.set_state(json!({"stale": true}));
    stale.history_mut().push(message("stale", ChatRole::Assistant, "old owner's answer"));
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

async fn cloned_handles_do_not_replace_an_active_runs_lease(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let thread = fixture.thread().await;
    let cloned = thread.clone();
    let lane = thread.lane("front-desk");
    let mut first = lane
        .open(RunOptions::new("first", "AST_first"), &crate::Holder::new("test"))
        .await
        .unwrap();
    let result = cloned
        .lane("front-desk")
        .open(
            RunOptions {
                access: LaneAccess::Exclusive(WhenHeld::Reject),
                ..RunOptions::new("second", "AST_second")
            },
            &crate::Holder::new("test"),
        )
        .await;
    let refused = match result {
        Err(StorageError::Rejected { .. }) => true,
        Ok(mut stolen) => {
            stolen.abandon(RunOutcome::Failed, "test cleanup").await.unwrap();
            false
        }
        Err(error) => panic!("unexpected second-run result: {error}"),
    };
    let original = first.commit(CommitOptions::default()).await;
    fixture.cleanup(&[&thread]).await;
    assert!(refused, "a clone replaced the active run's lease");
    assert!(original.is_ok(), "the original run lost its lease: {original:?}");
}

async fn a_dead_waiter_does_not_interrupt_a_new_owner(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let thread = fixture.thread().await;
    let lane = thread.lane("desk");
    let mut owner =
        lane.open(RunOptions::new("first", "task1"), &crate::Holder::new("test")).await.unwrap();
    assert!(lane.set_waiter("dead-worker").await.unwrap());
    owner.commit(CommitOptions::default()).await.unwrap();
    let mut next =
        lane.open(RunOptions::new("second", "task2"), &crate::Holder::new("test")).await.unwrap();
    assert_eq!(lane.checkpoint("task2").await.unwrap().0, None);
    next.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

async fn canceled_and_timed_out_acquires_withdraw_their_marks(backend: Backend) {
    use std::time::Duration;
    let fixture = backend.fixture(None).await;
    let thread = fixture.thread().await;
    let lane = thread.lane("desk");
    let mut owner =
        lane.open(RunOptions::new("owner", "task"), &crate::Holder::new("test")).await.unwrap();
    let result = lane
        .open(
            RunOptions {
                access: LaneAccess::Exclusive(WhenHeld::Interrupt),
                wait_timeout: Duration::ZERO,
                ..RunOptions::new("timeout", "timeout-task")
            },
            &crate::Holder::new("test"),
        )
        .await;
    assert!(matches!(result, Err(StorageError::Rejected { .. })));
    tokio::time::timeout(Duration::from_secs(5), async {
        while lane.checkpoint("task").await.unwrap().0.is_some() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("timed-out waiter's mark was not withdrawn");

    let waiting_lane = lane.clone();
    let waiting = tokio::spawn(async move {
        waiting_lane
            .open(
                RunOptions {
                    access: LaneAccess::Exclusive(WhenHeld::Interrupt),
                    ..RunOptions::new("canceled", "canceled-task")
                },
                &crate::Holder::new("test"),
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while lane.checkpoint("task").await.unwrap().0.is_none() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("waiter never marked the lane");
    waiting.abort();
    assert!(waiting.await.err().unwrap().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), async {
        while lane.checkpoint("task").await.unwrap().0.is_some() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("canceled waiter's mark was not withdrawn");
    owner.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

async fn a_resumed_drain_keeps_another_waiters_mark(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let thread = fixture.thread().await;
    let lane = thread.lane("desk");
    let mut owner =
        lane.open(RunOptions::new("first", "task1"), &crate::Holder::new("test")).await.unwrap();
    let lease = owner.lease().unwrap().clone();
    assert!(lane.set_waiter("other-worker").await.unwrap());
    owner
        .commit(CommitOptions { lease: LeaseAfterCommit::Keep, ..Default::default() })
        .await
        .unwrap();
    let mut resumed = lane.resume(RunOptions::new("second", "task2"), lease).await.unwrap();
    assert_eq!(lane.checkpoint("task2").await.unwrap().0.as_deref(), Some("other-worker"));
    resumed.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

/// A run's cleanup still ends it after its own holder re-entered the lane under a new token (a
/// canceled acquire's release applied after the holder's next turn took the lane): the newer
/// lease stays held, and resending the cleanup writes nothing more.
async fn a_runs_cleanup_ends_it_after_its_holder_re_entered(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let thread = fixture.thread().await;
    let lane = thread.lane(LANE);
    let holder = crate::Holder::new("test");
    let mut first = lane.open(RunOptions::new("first", "AST_first"), &holder).await.unwrap();
    let mut next = lane.open(RunOptions::new("next", "AST_next"), &holder).await.unwrap();
    assert_ne!(first.lease().unwrap().fence().token, next.lease().unwrap().fence().token);
    first.abandon(RunOutcome::Canceled, "acquire-canceled").await.unwrap();
    let ended = |runs: &[crate::thread::run::RunSummary], id: &str| {
        runs.iter().filter(|r| r.run_id == id && !r.interrupted()).count()
    };
    let runs = lane.runs(usize::MAX).await.unwrap();
    assert_eq!(ended(&runs, "first"), 1, "the re-entered holder's run never ended: {runs:?}");
    let conn = thread.conn().clone();
    let ended_events = || async {
        let count = "SELECT COUNT(*) FROM events WHERE run_id = 'first' AND type = 'run_ended'";
        conn.query(crate::sql::Statement::raw(count)).await.unwrap()[0][0].as_i64()
    };
    assert_eq!(ended_events().await, Some(1));
    // A resend of the applied cleanup writes no second `run_ended`, and leaves the newer lease.
    let lease = next.lease().unwrap().clone();
    let ended = RunEnded { outcome: RunOutcome::Canceled, reason: None };
    let event = EventWrite::new(0, JournalEvent::RunEnded(ended));
    let token = first.lease().unwrap().fence().token.clone();
    conn.batch(release(LANE, &token, holder.as_str(), Some(("first", &event)))).await.unwrap();
    assert_eq!(ended_events().await, Some(1));
    assert!(!lease.is_lost());
    next.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

crate::contract!(
    waiters_take_the_lane_in_the_order_their_rows_were_submitted,
    an_expired_owner_cannot_restart_or_commit_after_takeover,
    a_dropped_run_stops_renewing_and_its_lease_lapses,
    cloned_handles_do_not_replace_an_active_runs_lease,
    a_dead_waiter_does_not_interrupt_a_new_owner,
    canceled_and_timed_out_acquires_withdraw_their_marks,
    a_resumed_drain_keeps_another_waiters_mark,
    a_runs_cleanup_ends_it_after_its_holder_re_entered,
);

#[test]
fn renewal_timing_reserves_a_full_request_interval_before_expiry() {
    use crate::thread::LeaseTiming;
    use std::time::Duration;

    let ttl = Duration::from_secs(10);
    for interval in [Duration::ZERO, Duration::from_secs(5), ttl, Duration::from_secs(11)] {
        assert!(LeaseTiming::new(ttl, interval).is_err(), "interval {interval:?}");
    }
    assert!(LeaseTiming::new(ttl, Duration::from_secs(3)).is_ok());
}

/// Applies a database request immediately, but can delay its acknowledgment or stall renewals.
/// This models the difference between a durable write and knowing it succeeded.
struct SlowReplies {
    database: crate::backends::sqlite::FileConnection,
    _file: crate::testing::TempDatabase,
    acquire_delay: std::time::Duration,
    renewal_delay: std::time::Duration,
    stall_after_first: bool,
    attempts: parking_lot::Mutex<Vec<tokio::time::Instant>>,
}

#[async_trait::async_trait]
impl crate::ThreadDbConnection for SlowReplies {
    async fn query(
        &self,
        statement: crate::sql::Statement,
    ) -> Result<Vec<crate::sql::Row>, StorageError> {
        self.database.query(statement).await
    }

    async fn batch(
        &self,
        statements: Vec<crate::sql::Statement>,
    ) -> Result<crate::sql::BatchResult, StorageError> {
        let renewal =
            statements.iter().any(|s| s.sql().contains("SET lease_until = unixepoch() +"));
        let acquire = statements.iter().any(|s| s.sql().contains("RETURNING version"));
        if renewal {
            let attempt = {
                let mut attempts = self.attempts.lock();
                attempts.push(tokio::time::Instant::now());
                attempts.len()
            };
            if self.stall_after_first && attempt > 1 {
                return std::future::pending().await;
            }
        }
        let result = self.database.batch(statements).await?;
        let delay = if renewal {
            self.renewal_delay
        } else if acquire {
            self.acquire_delay
        } else {
            std::time::Duration::ZERO
        };
        tokio::time::sleep(delay).await;
        Ok(result)
    }

    async fn close(&self) {
        self.database.close().await;
    }
}

async fn slow_replies(
    acquire_delay: u64,
    renewal_delay: u64,
    stall_after_first: bool,
) -> std::sync::Arc<SlowReplies> {
    let file = crate::testing::TempDatabase::new();
    std::sync::Arc::new(SlowReplies {
        database: file.connect().await,
        _file: file,
        acquire_delay: std::time::Duration::from_secs(acquire_delay),
        renewal_delay: std::time::Duration::from_secs(renewal_delay),
        stall_after_first,
        attempts: parking_lot::Mutex::new(Vec::new()),
    })
}

async fn slow_thread(connection: std::sync::Arc<SlowReplies>) -> crate::Thread {
    crate::Thread::new(connection, "AT_timing", Default::default(), Default::default())
        .await
        .unwrap()
}

#[tokio::test(start_paused = true)]
async fn slow_acknowledgments_do_not_shift_the_renewal_cadence() {
    use std::time::Duration;
    let connection = slow_replies(0, 2, false).await;
    let thread = slow_thread(connection.clone()).await;
    let started = tokio::time::Instant::now();
    let run = thread
        .lane("desk")
        .open(RunOptions::new("run", "task"), &crate::Holder::new("test"))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(9500)).await;
    let attempts: Vec<_> =
        connection.attempts.lock().iter().map(|at| (*at - started).as_secs()).collect();
    assert_eq!(attempts, [3, 6, 9]);
    assert!(!run.lease().unwrap().is_lost());
}

#[tokio::test(start_paused = true)]
async fn acknowledgment_latency_and_stalled_renewals_cannot_extend_the_ownership_budget() {
    use std::time::Duration;
    let connection = slow_replies(0, 2, true).await;
    let thread = slow_thread(connection.clone()).await;
    let started = tokio::time::Instant::now();
    let run = thread
        .lane("desk")
        .open(RunOptions::new("run", "task"), &crate::Holder::new("test"))
        .await
        .unwrap();
    let lease = run.lease().unwrap();
    let error = lease.lost().await;
    assert!(matches!(error, StorageError::LeaseLost(_)));
    // The renewal starts at 3 s, is acknowledged at 5 s, and confirms ownership only until 13 s.
    assert_eq!(started.elapsed(), Duration::from_secs(13));
    let attempts = connection.attempts.lock().len();
    lease.keep_alive();
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert_eq!(connection.attempts.lock().len(), attempts, "loss is terminal");
}

#[tokio::test(start_paused = true)]
async fn a_slow_acquisition_cannot_start_work_with_an_expired_ownership_budget() {
    let thread = slow_thread(slow_replies(11, 0, false).await).await;
    let error = thread
        .lane("desk")
        .open(RunOptions::new("run", "task"), &crate::Holder::new("test"))
        .await
        .err()
        .expect("expired claim");
    assert!(matches!(error, StorageError::LeaseLost(_)), "{error}");
}

#[tokio::test(start_paused = true)]
async fn an_inbox_claim_keeps_renewing_while_acceptance_or_debounce_waits() {
    use std::time::Duration;
    let connection = slow_replies(0, 0, false).await;
    let thread = slow_thread(connection.clone()).await;
    let message: a2a::Message = serde_json::from_value(serde_json::json!({
        "messageId": "first", "role": "ROLE_USER", "parts": [{"text": "hello"}]
    }))
    .unwrap();
    let data =
        crate::thread::schema::TaskData { history: vec![message.clone()], ..Default::default() };
    let (_, lease) = thread
        .lane("desk")
        .receive(
            &crate::Holder::new("worker"),
            crate::thread::db::ReceiveInput {
                task_id: "task",
                message: &message,
                data: &data,
                internal: None,
            },
        )
        .await
        .unwrap();
    let lease = lease.expect("the free lane was claimed");
    tokio::time::sleep(Duration::from_millis(12500)).await;
    assert!(!lease.is_lost());
    assert_eq!(connection.attempts.lock().len(), 4);
}

/// Failed renewals retry within the confirmed ownership deadline, then stop exactly at it.
#[tokio::test(start_paused = true)]
async fn a_failing_renewal_retries_within_the_lease_and_is_lost_after_it() {
    use crate::testing::{Faulty, TempDatabase};
    use crate::thread::{LeaseTiming, SchemaCell};
    use std::sync::Arc;

    let file = TempDatabase::new();
    let faulty = Arc::new(Faulty::new(Box::new(file.connect().await)));
    let timing = LeaseTiming::new(Duration::from_secs(10), Duration::from_secs(3)).unwrap();
    let thread =
        Thread::new(faulty.clone(), "AT_renewal", timing, SchemaCell::default()).await.unwrap();
    let run = thread
        .lane(LANE)
        .open(RunOptions::new("r1", "AST_1"), &crate::Holder::new("test"))
        .await
        .unwrap();
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
