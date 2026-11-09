//! Production health: lanes held by a dead owner, stuck lane waits, teardown past its bound and
//! writes whose outcome stayed unknown. Each one is also a warn log.

use std::time::Duration;

use super::decl::{Attrs, SpanValue, metric};
use super::instruments::DURATION_BOUNDS;

/// An acquire that waited at least this long is logged as stuck.
pub const STUCK_LANE_WAIT: Duration = Duration::from_secs(15);

/// A holder that has not renewed within its renew interval plus this slack is presumed dead:
/// the database counts whole seconds and a renewal takes a round trip.
pub const RENEW_SLACK: Duration = Duration::from_secs(2);

/// At most one warn log per event kind in this window; the others are counted into the next.
pub const HEALTH_LOG_INTERVAL: Duration = Duration::from_secs(10);

/// The run that took, or waited for, a lane.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneKind {
    /// A voice session's turn.
    Voice,
    /// A text run opened through a thread handle.
    Text,
    /// An A2A request.
    A2a,
    /// A direct context save.
    Context,
}

/// A storage write sent again after its acknowledgment was lost.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageWrite {
    /// A lane acquire.
    Acquire,
    /// An A2A inbox receive.
    Receive,
    /// A lease release.
    Release,
    /// A lease extension.
    Extend,
    /// A run's commit.
    Commit,
    /// A task's `submitted` row.
    Submit,
    /// A waiter's mark, set or withdrawn.
    Waiter,
    /// A dead holder's task failed.
    FailOrphan,
    /// A task cancel.
    Cancel,
    /// The thread schema.
    Schema,
    /// An application context transaction.
    Context,
    /// A thread record's update or delete.
    Thread,
}

/// Teardown that ran past its bound.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeardownStage {
    /// A voice session's close watchdog.
    SessionClose,
    /// A dropped lease's release.
    LeaseRelease,
}

/// Time an exclusive lane acquire waited, taken or refused.
#[metric(name = "lk.agents.lane.acquire_wait", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct LaneAcquireWait {
    /// The run that took, or waited for, the lane.
    #[attr("lane_kind")]
    pub kind: LaneKind,
}

/// Time an acquire waited on a lease whose holder stopped renewing it.
#[metric(name = "lk.agents.lane.ownerless_wait", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct LaneOwnerlessWait {
    #[attr("lane_kind")]
    pub kind: LaneKind,
}

/// Leases taken over after they lapsed instead of being released.
#[metric(name = "lk.agents.lane.lease_lapsed", kind = Counter, unit = "{lease}")]
#[derive(Attrs)]
pub struct LaneLeaseLapsed {
    #[attr("lane_kind")]
    pub kind: LaneKind,
}

/// Teardowns that ran past their bound.
#[metric(name = "lk.agents.teardown.overrun", kind = Counter, unit = "{teardown}")]
#[derive(Attrs)]
pub struct TeardownOverrun {
    /// The teardown that overran.
    #[attr("stage")]
    pub stage: TeardownStage,
}

/// Writes whose outcome was still unknown when their resends ran out.
#[metric(name = "lk.agents.storage.write_uncertain", kind = Counter, unit = "{write}")]
#[derive(Attrs)]
pub struct StorageWriteUncertain {
    /// The write left unresolved.
    #[attr("write")]
    pub write: StorageWrite,
}
