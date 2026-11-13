use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::VirtualClock;
use lk_agents_abi::telemetry::decl::Value;
use lk_agents_abi::telemetry::record::{Measurement, Number, Record, RecordSink};
use lk_agents_telemetry::Tracer;

use super::*;
use crate::sql::{Bind as _, statement};
use crate::testing::{Faulty, TempDatabase, WriteFault};
use crate::thread::tests::PARTITION_KEY;
use crate::thread::{LeaseTiming, SchemaCell};
use crate::{CommitOptions, Holder, PartitionAccess, RunOptions, StorageError, Thread, WhenHeld};

#[derive(Default)]
struct Collect(Mutex<Vec<Measurement>>);

impl RecordSink for Collect {
    fn record(&self, record: Record) {
        if let Record::Measurement(m) = record {
            self.0.lock().push(m);
        }
    }
}

impl Collect {
    /// Each measurement of `instrument`: its value and its one label.
    fn of(&self, instrument: &str) -> Vec<(f64, Value)> {
        let value = |n: &Number| match n {
            Number::Int(i) => *i as f64,
            Number::Double(d) => *d,
        };
        let measurements = self.0.lock();
        let of = measurements.iter().filter(|m| m.instrument == instrument);
        of.map(|m| (value(&m.value), m.labels[0].1.clone())).collect()
    }
}

fn recording() -> (Cx, Arc<Collect>) {
    let sink = Arc::new(Collect::default());
    let tracer = Tracer::new(sink.clone(), Arc::new(VirtualClock::new()), 7);
    (tracer.root(), sink)
}

fn label(value: &str) -> Value {
    Value::String(value.into())
}

#[test]
fn each_event_records_its_instrument() {
    let (cx, sink) = recording();
    let clock = VirtualClock::new();
    acquire_waited(
        &cx,
        &clock,
        PartitionKind::Voice,
        PARTITION_KEY,
        Duration::from_secs(20),
        Some(Duration::from_secs(4)),
    );
    let short = Duration::from_millis(5);
    acquire_waited(&cx, &clock, PartitionKind::Text, PARTITION_KEY, short, None);
    lease_lapsed(&cx, &clock, PartitionKind::A2a, PARTITION_KEY, "dead");
    teardown_overrun(&cx, &clock, TeardownStage::LeaseRelease, PARTITION_KEY);
    write_uncertain(&cx, &clock, StorageWrite::Commit);
    let waits = sink.of("lk.agents.partition.acquire_wait");
    assert_eq!(waits, vec![(20.0, label("voice")), (0.005, label("text"))]);
    assert_eq!(sink.of("lk.agents.partition.ownerless_wait"), vec![(4.0, label("voice"))]);
    assert_eq!(sink.of("lk.agents.partition.lease_lapsed"), vec![(1.0, label("a2a"))]);
    assert_eq!(sink.of("lk.agents.teardown.overrun"), vec![(1.0, label("lease_release"))]);
    assert_eq!(sink.of("lk.agents.storage.write_uncertain"), vec![(1.0, label("commit"))]);
}

#[test]
fn a_throttle_admits_one_log_per_interval_and_counts_the_rest() {
    let throttle = Throttle::new();
    let at = lk_agents_abi::simulated_epoch();
    assert_eq!(throttle.admit(at), Some(0));
    assert_eq!(throttle.admit(at), None);
    assert_eq!(throttle.admit(at + HEALTH_LOG_INTERVAL / 2), None);
    assert_eq!(throttle.admit(at + HEALTH_LOG_INTERVAL), Some(2));
}

statement! {
    /// Hands the partition to `holder` under a lease ending `until_ms` milliseconds from now.
    SetLease<'a> { partition_key: &'a str, holder: &'a str, until_ms: i64 } => r#"
UPDATE partitions SET lease_holder = :holder, lease_token = 'dead', lease_expires_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER) + :until_ms
WHERE partition_key = :partition_key
"#
}

async fn handle(db: &TempDatabase, cx: &Cx) -> Thread {
    let conn = Arc::new(db.connect().await);
    let (timing, schema) = (LeaseTiming::default(), SchemaCell::default());
    let thread = Thread::new(conn, "AT_health", timing, schema, crate::testing::background());
    thread.await.unwrap().with_telemetry(cx.clone())
}

fn reject(run_id: &str) -> RunOptions {
    RunOptions {
        access: PartitionAccess::Exclusive(WhenHeld::Reject),
        ..RunOptions::new(run_id, "")
    }
}

/// Leaves the partition's lease to a dead holder, ending `until` seconds from now.
async fn dead_holder(thread: &Thread, until: i64) {
    let mut run =
        thread.partition(PARTITION_KEY).open(reject("r0"), &Holder::new("seed")).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();
    let lease =
        SetLease { partition_key: PARTITION_KEY, holder: "dead", until_ms: until * 1000 }.bind();
    thread.conn().batch(vec![lease]).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn an_acquire_refused_by_a_lease_nobody_renews_records_an_ownerless_wait() {
    let (cx, sink) = recording();
    let db = TempDatabase::new();
    let thread = handle(&db, &cx).await;
    // a 10 s lease renewed every 3 s, last renewed 6 s ago
    dead_holder(&thread, 4).await;
    let refused = thread.partition(PARTITION_KEY).open(reject("r1"), &Holder::new("other")).await;
    assert!(matches!(refused, Err(StorageError::Rejected { .. })), "{:?}", refused.err());
    assert_eq!(sink.of("lk.agents.partition.ownerless_wait").len(), 1);
    assert_eq!(sink.of("lk.agents.partition.acquire_wait").len(), 2, "the seed's and this one");
}

#[tokio::test(start_paused = true)]
async fn an_acquire_refused_by_a_live_lease_records_no_ownerless_wait() {
    let (cx, sink) = recording();
    let db = TempDatabase::new();
    let thread = handle(&db, &cx).await;
    dead_holder(&thread, 9).await;
    let refused = thread.partition(PARTITION_KEY).open(reject("r1"), &Holder::new("other")).await;
    assert!(matches!(refused, Err(StorageError::Rejected { .. })));
    assert!(sink.of("lk.agents.partition.ownerless_wait").is_empty());
}

#[tokio::test(start_paused = true)]
async fn taking_over_a_lapsed_lease_counts_it() {
    let (cx, sink) = recording();
    let db = TempDatabase::new();
    let thread = handle(&db, &cx).await;
    dead_holder(&thread, -1).await;
    let mut run =
        thread.partition(PARTITION_KEY).open(reject("r1"), &Holder::new("other")).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();
    assert_eq!(sink.of("lk.agents.partition.lease_lapsed"), vec![(1.0, label("a2a"))]);
    // a released lease is not a lapse
    let mut run =
        thread.partition(PARTITION_KEY).open(reject("r2"), &Holder::new("third")).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();
    assert_eq!(sink.of("lk.agents.partition.lease_lapsed").len(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_write_left_uncertain_counts_it_by_kind() {
    let (cx, sink) = recording();
    let db = TempDatabase::new();
    let faulty = Arc::new(Faulty::new(Box::new(db.connect().await)));
    let (timing, schema) = (LeaseTiming::default(), SchemaCell::default());
    let thread =
        Thread::new(faulty.clone(), "AT_health", timing, schema, crate::testing::background());
    let thread = thread.await.unwrap().with_telemetry(cx);
    let mut run =
        thread.partition(PARTITION_KEY).open(reject("r0"), &Holder::new("seed")).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();
    faulty.fail_next_lease_take(WriteFault::LoseAckBefore);
    faulty.unreachable_after_commit_for(Duration::from_secs(5));
    let options = RunOptions { wait_timeout: Duration::from_secs(1), ..reject("r1") };
    let lost = thread.partition(PARTITION_KEY).open(options, &Holder::new("other")).await;
    assert!(matches!(lost, Err(StorageError::Uncertain(_))), "{:?}", lost.err());
    assert_eq!(sink.of("lk.agents.storage.write_uncertain"), vec![(1.0, label("acquire"))]);
}
