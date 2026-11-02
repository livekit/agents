//! A cloned thread handle cannot replace an active run's lease.

use lk_agents_storage::{CommitOptions, RunOptions};
use lk_agents_storage::{RunOutcome, StorageError};

use lk_agents_storage::testing::Backend;

async fn cloned_handles_do_not_replace_an_active_runs_lease(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let thread = fixture.thread("worker-a").await;
    let cloned = thread.clone();
    let lane = thread.lane("front-desk");
    let mut first = lane.open(RunOptions::new("first", "AST_first")).await.unwrap();
    let result = cloned
        .lane("front-desk")
        .open(RunOptions { wait: false, ..RunOptions::new("second", "AST_second") })
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
    let thread = fixture.thread("worker").await;
    let lane = thread.lane("desk");
    let mut owner = lane.open(RunOptions::new("first", "task1")).await.unwrap();
    assert!(lane.set_waiter("dead-worker").await.unwrap());
    owner.commit(CommitOptions::default()).await.unwrap();
    let mut next = lane.open(RunOptions::new("second", "task2")).await.unwrap();
    assert_eq!(lane.checkpoint("task2").await.unwrap().0, None);
    next.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

async fn canceled_and_timed_out_acquires_withdraw_their_marks(backend: Backend) {
    use std::time::Duration;
    let fixture = backend.fixture(None).await;
    let thread = fixture.thread("worker").await;
    let lane = thread.lane("desk");
    let mut owner = lane.open(RunOptions::new("owner", "task")).await.unwrap();
    let result = lane
        .open(RunOptions {
            interrupt: true,
            wait_timeout: Duration::ZERO,
            ..RunOptions::new("timeout", "timeout-task")
        })
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
            .open(RunOptions { interrupt: true, ..RunOptions::new("canceled", "canceled-task") })
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
    let thread = fixture.thread("worker").await;
    let lane = thread.lane("desk");
    let mut owner = lane.open(RunOptions::new("first", "task1")).await.unwrap();
    let lease = owner.lease().unwrap().clone();
    assert!(lane.set_waiter("other-worker").await.unwrap());
    owner.commit(CommitOptions { keep_lease: true, ..Default::default() }).await.unwrap();
    let mut resumed = lane.resume(RunOptions::new("second", "task2"), lease).await.unwrap();
    assert_eq!(lane.checkpoint("task2").await.unwrap().0.as_deref(), Some("other-worker"));
    resumed.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

lk_agents_storage::contract!(
    cloned_handles_do_not_replace_an_active_runs_lease,
    a_dead_waiter_does_not_interrupt_a_new_owner,
    canceled_and_timed_out_acquires_withdraw_their_marks,
    a_resumed_drain_keeps_another_waiters_mark,
);

#[test]
fn renewal_timing_reserves_a_full_request_interval_before_expiry() {
    use lk_agents_storage::thread::LeaseTiming;
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
    database: lk_agents_storage::testing::MemoryConnection,
    acquire_delay: std::time::Duration,
    renewal_delay: std::time::Duration,
    stall_after_first: bool,
    attempts: parking_lot::Mutex<Vec<tokio::time::Instant>>,
}

#[async_trait::async_trait]
impl lk_agents_storage::ThreadDbConnection for SlowReplies {
    async fn query(
        &self,
        statement: lk_agents_storage::sql::Statement,
    ) -> Result<Vec<lk_agents_storage::sql::Row>, StorageError> {
        self.database.query(statement).await
    }

    async fn batch(
        &self,
        statements: Vec<lk_agents_storage::sql::Statement>,
    ) -> Result<lk_agents_storage::sql::BatchResult, StorageError> {
        let renewal = statements.iter().any(|s| s.sql.contains("SET lease_until = unixepoch() +"));
        let acquire = statements.iter().any(|s| s.sql.contains("RETURNING version"));
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

fn slow_replies(
    acquire_delay: u64,
    renewal_delay: u64,
    stall_after_first: bool,
) -> std::sync::Arc<SlowReplies> {
    std::sync::Arc::new(SlowReplies {
        database: lk_agents_storage::testing::MemoryConnection::new(std::sync::Arc::new(
            lk_agents_net::clock::RealClock::new(),
        ))
        .unwrap(),
        acquire_delay: std::time::Duration::from_secs(acquire_delay),
        renewal_delay: std::time::Duration::from_secs(renewal_delay),
        stall_after_first,
        attempts: parking_lot::Mutex::new(Vec::new()),
    })
}

async fn slow_thread(connection: std::sync::Arc<SlowReplies>) -> lk_agents_storage::Thread {
    lk_agents_storage::Thread::new(
        connection,
        "AT_timing",
        Default::default(),
        Default::default(),
        "worker",
    )
    .await
    .unwrap()
}

#[tokio::test(start_paused = true)]
async fn slow_acknowledgments_do_not_shift_the_renewal_cadence() {
    use std::time::Duration;
    let connection = slow_replies(0, 2, false);
    let thread = slow_thread(connection.clone()).await;
    let started = tokio::time::Instant::now();
    let run = thread.lane("desk").open(RunOptions::new("run", "task")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(9500)).await;
    let attempts: Vec<_> =
        connection.attempts.lock().iter().map(|at| (*at - started).as_secs()).collect();
    assert_eq!(attempts, [3, 6, 9]);
    assert!(!run.lease().unwrap().is_lost());
}

#[tokio::test(start_paused = true)]
async fn acknowledgment_latency_and_stalled_renewals_cannot_extend_the_ownership_budget() {
    use std::time::Duration;
    let connection = slow_replies(0, 2, true);
    let thread = slow_thread(connection.clone()).await;
    let started = tokio::time::Instant::now();
    let run = thread.lane("desk").open(RunOptions::new("run", "task")).await.unwrap();
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
    let thread = slow_thread(slow_replies(11, 0, false)).await;
    let error = thread
        .lane("desk")
        .open(RunOptions::new("run", "task"))
        .await
        .err()
        .expect("expired claim");
    assert!(matches!(error, StorageError::LeaseLost(_)), "{error}");
}

#[tokio::test(start_paused = true)]
async fn an_inbox_claim_keeps_renewing_while_acceptance_or_debounce_waits() {
    use std::time::Duration;
    let connection = slow_replies(0, 0, false);
    let thread = slow_thread(connection.clone()).await;
    let message: a2a::Message = serde_json::from_value(serde_json::json!({
        "messageId": "first", "role": "ROLE_USER", "parts": [{"text": "hello"}]
    }))
    .unwrap();
    let data = lk_agents_storage::thread::schema::TaskData {
        history: vec![message.clone()],
        ..Default::default()
    };
    let (_, lease) = thread
        .lane("desk")
        .receive(
            "worker",
            lk_agents_storage::thread::db::ReceiveInput {
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
