#![allow(
    clippy::disallowed_macros,
    reason = "Clippy checks macro bans at crate scope; exact legacy macro sites are ratcheted."
)]

//! `lk-agents-storage`: the synchronous storage core of Agents 2.0. It decides; the runtime
//! (`lk_agents_runtime::storage`) executes. No I/O, clock, entropy or task lives here.
//!
//! * [`sql`]: the backend-neutral SQL vocabulary: values, statements, results, typed row reads.
//! * [`schema`]: the thread tables as records, and every statement.
//! * [`db`]: each lane operation as a plan (one batch or read) and its typed decoding.
//! * [`run`]: task states and transitions, the run's outcome and the commit it writes.
//! * [`lease`]: lease renewal and lane polling decisions on supplied time.
//! * [`streams`]: the answer streams' records.
//! * `sqlite` (feature): the statements executed on a SQLite connection the caller owns.

pub mod context;
pub mod db;
pub mod host;
pub mod lease;
pub mod run;
pub mod schema;
pub mod sql;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod streams;

pub use db::{Commit, Committed, EventWrite, Fence, TaskWrite};
pub use run::{RunOutcome, TaskState};

pub use lk_agents_abi::storage::StorageError;

/// Unix seconds as a `Timestamp`, to the nearest nanosecond, so the float reads back unchanged.
pub fn timestamp(t: f64) -> pbjson_types::Timestamp {
    let seconds = t.floor();
    let nanos = ((t - seconds) * 1e9).round() as i32;
    if nanos >= 1_000_000_000 {
        pbjson_types::Timestamp { seconds: seconds as i64 + 1, nanos: 0 }
    } else {
        pbjson_types::Timestamp { seconds: seconds as i64, nanos }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The float reads back unchanged, and a rounding to a whole second carries.
    #[test]
    fn a_timestamp_keeps_the_seconds_float() {
        let t = timestamp(1791760302.4035509);
        assert_eq!(t.seconds as f64 + f64::from(t.nanos) / 1e9, 1791760302.4035509);
        assert_eq!(timestamp(0.9999999999), pbjson_types::Timestamp { seconds: 1, nanos: 0 });
    }
}
