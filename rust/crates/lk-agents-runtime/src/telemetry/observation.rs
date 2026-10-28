//! Optional observation ingress is a bounded ring. Taking a batch releases its lock before any
//! observer runs. Required protocol completions never travel through this lossy channel.

use std::collections::VecDeque;
use std::sync::Mutex;

use lk_agents_abi::telemetry::observation::Observation;

struct Pending {
    records: VecDeque<(Observation, usize)>,
    bytes: usize,
    dropped_records: u64,
    dropped_bytes: u64,
}

pub struct ObservationBuffer {
    pending: Mutex<Pending>,
    max_records: usize,
    max_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationDrops {
    pub records: u64,
    pub bytes: u64,
}

impl ObservationBuffer {
    pub fn new(max_records: usize, max_bytes: usize) -> Self {
        Self {
            pending: Mutex::new(Pending {
                records: VecDeque::new(),
                bytes: 0,
                dropped_records: 0,
                dropped_bytes: 0,
            }),
            max_records,
            max_bytes,
        }
    }

    /// Optional export refuses the newest record at capacity, in O(1) queue work. It never
    /// shifts a full vector or loops over old records trying to make a large item fit.
    pub fn try_push(&self, observation: Observation) -> bool {
        let bytes = observation.retained_bytes();
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if pending.records.len() >= self.max_records
            || bytes > self.max_bytes.saturating_sub(pending.bytes)
        {
            pending.dropped_records = pending.dropped_records.saturating_add(1);
            pending.dropped_bytes = pending.dropped_bytes.saturating_add(bytes as u64);
            return false;
        }
        pending.bytes += bytes;
        pending.records.push_back((observation, bytes));
        true
    }

    pub fn drain(&self, max_records: usize) -> Vec<Observation> {
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        let count = pending.records.len().min(max_records);
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            let (observation, bytes) = pending.records.pop_front().expect("count bounded above");
            pending.bytes -= bytes;
            batch.push(observation);
        }
        batch
    }

    pub fn drops(&self) -> ObservationDrops {
        let pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        ObservationDrops { records: pending.dropped_records, bytes: pending.dropped_bytes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::telemetry::observation::{
        ObservationIdentity, ObservationKind, ObservationTime,
    };

    fn record(job: &str, operation: &str, at: u64, parent: Option<&str>) -> Observation {
        Observation {
            identity: ObservationIdentity {
                thread_id: format!("thread/{job}"),
                lane: "backoffice".into(),
                job_id: job.into(),
                operation_id: operation.into(),
            },
            at: ObservationTime { wall: (123, at as u32), monotonic_ns: at },
            capture_content: false,
            kind: ObservationKind::SpanStarted {
                name: "tool".into(),
                parent_operation: parent.map(str::to_owned),
                trace_parent: None,
                attributes: vec![],
            },
        }
    }

    #[test]
    fn observation_identity_time_survive_interleaved_jobs_and_slow_drain() {
        let buffer = ObservationBuffer::new(4, 8192);
        let first = record("one", "tool/1", 10, Some("run/1"));
        let second = record("two", "tool/2", 20, Some("run/2"));
        assert!(buffer.try_push(first.clone()));
        assert!(buffer.try_push(second.clone()));
        let batch = buffer.drain(1);
        // The observer holding its batch cannot stall decisions or overwrite its captured ids.
        assert!(buffer.try_push(record("three", "tool/3", 30, None)));
        assert_eq!(batch, [first]);
        assert_eq!(buffer.drain(1), [second]);
    }

    #[test]
    fn telemetry_buffer_full_refuses_newest_and_reports_item_and_byte_loss() {
        let sample = record("one", "op", 1, None);
        let bytes = sample.retained_bytes();
        let buffer = ObservationBuffer::new(1, bytes);
        assert!(buffer.try_push(sample.clone()));
        let started = std::time::Instant::now();
        for _ in 0..8192 {
            assert!(!buffer.try_push(sample.clone()));
        }
        eprintln!("telemetry_buffer_full inserts=8192 elapsed={:?}", started.elapsed());
        assert_eq!(buffer.drops(), ObservationDrops { records: 8192, bytes: 8192 * bytes as u64 });
        assert_eq!(buffer.drain(2), [sample.clone()]);
        assert!(buffer.try_push(sample.clone()));
        let disabled = ObservationBuffer::new(0, 0);
        assert!(!disabled.try_push(sample));
        assert!(disabled.drain(1).is_empty());
    }
}
