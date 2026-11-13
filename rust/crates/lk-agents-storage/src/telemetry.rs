//! Storage health: each event is a measurement and a warn log, the log at most once per
//! [`HEALTH_LOG_INTERVAL`] per event kind, on the wall time of the caller's clock.

use std::time::{Duration, SystemTime};

pub(crate) use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::telemetry::health::{
    HEALTH_LOG_INTERVAL, PartitionAcquireWait, PartitionLeaseLapsed, PartitionOwnerlessWait,
    STUCK_PARTITION_WAIT, StorageWriteUncertain, TeardownOverrun,
};
pub(crate) use lk_agents_abi::telemetry::health::{PartitionKind, StorageWrite, TeardownStage};
use lk_agents_abi::{Clock, LOG_TARGET};
use parking_lot::Mutex;

/// One event kind's last log and the logs suppressed since.
struct Throttle(Mutex<(Option<SystemTime>, u64)>);

impl Throttle {
    const fn new() -> Self {
        Self(parking_lot::const_mutex((None, 0)))
    }

    /// The count suppressed since the last log when one at `now` may log, else `None`.
    fn admit(&self, now: SystemTime) -> Option<u64> {
        let mut state = self.0.lock();
        let recent = |at: SystemTime| now.duration_since(at).is_ok_and(|d| d < HEALTH_LOG_INTERVAL);
        if state.0.is_some_and(recent) {
            state.1 += 1;
            return None;
        }
        state.0 = Some(now);
        Some(std::mem::take(&mut state.1))
    }
}

static STUCK: Throttle = Throttle::new();
static OWNERLESS: Throttle = Throttle::new();
static LAPSED: Throttle = Throttle::new();
static OVERRUN: Throttle = Throttle::new();
static UNCERTAIN: Throttle = Throttle::new();

/// An exclusive acquire ended, taken or refused, after `waited`; `ownerless` of it was spent on
/// a holder that had stopped renewing.
pub(crate) fn acquire_waited(
    cx: &Cx,
    clock: &dyn Clock,
    kind: PartitionKind,
    partition_key: &str,
    waited: Duration,
    ownerless: Option<Duration>,
) {
    lk_agents_telemetry::measure(cx, &PartitionAcquireWait { kind }, waited.as_secs_f64());
    if let Some(ownerless) = ownerless {
        lk_agents_telemetry::measure(cx, &PartitionOwnerlessWait { kind }, ownerless.as_secs_f64());
        if let Some(suppressed) = OWNERLESS.admit(clock.wall()) {
            log::warn!(target: LOG_TARGET, partition_key, partition_kind:? = kind, wait_s = ownerless.as_secs_f64(), suppressed; "partition_key held by a lease its holder stopped renewing");
        }
    }
    if waited >= STUCK_PARTITION_WAIT
        && let Some(suppressed) = STUCK.admit(clock.wall())
    {
        log::warn!(target: LOG_TARGET, partition_key, partition_kind:? = kind, wait_s = waited.as_secs_f64(), suppressed; "partition_key acquire waited past the stuck threshold");
    }
}

/// An acquire took over a lease that lapsed instead of being released.
pub(crate) fn lease_lapsed(
    cx: &Cx,
    clock: &dyn Clock,
    kind: PartitionKind,
    partition_key: &str,
    holder: &str,
) {
    lk_agents_telemetry::measure(cx, &PartitionLeaseLapsed { kind }, 1u64);
    if let Some(suppressed) = LAPSED.admit(clock.wall()) {
        log::warn!(target: LOG_TARGET, partition_key, partition_kind:? = kind, holder, suppressed; "took over a lease that lapsed instead of being released");
    }
}

/// A teardown ran past its bound.
pub(crate) fn teardown_overrun(
    cx: &Cx,
    clock: &dyn Clock,
    stage: TeardownStage,
    partition_key: &str,
) {
    lk_agents_telemetry::measure(cx, &TeardownOverrun { stage }, 1u64);
    if let Some(suppressed) = OVERRUN.admit(clock.wall()) {
        log::warn!(target: LOG_TARGET, partition_key, stage:? = stage, suppressed; "teardown ran past its bound");
    }
}

/// A write's resends ran out with its outcome unknown.
pub(crate) fn write_uncertain(cx: &Cx, clock: &dyn Clock, write: StorageWrite) {
    lk_agents_telemetry::measure(cx, &StorageWriteUncertain { write }, 1u64);
    if let Some(suppressed) = UNCERTAIN.admit(clock.wall()) {
        log::warn!(target: LOG_TARGET, write:? = write, suppressed; "write outcome still unknown after its resends");
    }
}

#[cfg(test)]
#[path = "telemetry_tests.rs"]
mod tests;
