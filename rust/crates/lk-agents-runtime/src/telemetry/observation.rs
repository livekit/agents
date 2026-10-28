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

/// Session report snapshots expire after one hour. These budgets apply before the host copies
/// an event, independently of SDK exporter retention.
const EVENT_ITEMS: usize = 1024;
const EVENT_BYTES: usize = 1024 * 1024;
const EVENT_RECORD_BYTES: usize = 64 * 1024;
const EVENT_RETENTION_NS: u64 = 3600 * 1_000_000_000;

#[derive(Default)]
struct RetainedEvents {
    job_id: Option<String>,
    entries: VecDeque<(u64, usize)>,
    bytes: usize,
    now_ns: u64,
    dropped: u64,
    expired: u64,
}

impl RetainedEvents {
    fn expire(&mut self, now_ns: u64) -> usize {
        // Admission time, rather than the event's wall timestamp, orders the expiry queue.
        self.now_ns = self.now_ns.max(now_ns);
        let mut expired = 0;
        while self
            .entries
            .front()
            .is_some_and(|(at, _)| self.now_ns.saturating_sub(*at) >= EVENT_RETENTION_NS)
        {
            let (_, bytes) = self.entries.pop_front().expect("front exists");
            self.bytes -= bytes;
            expired += 1;
        }
        self.expired = self.expired.saturating_add(expired as u64);
        expired
    }

    fn admit(&mut self, bytes: usize, now_ns: u64) -> (usize, bool) {
        let expired = self.expire(now_ns);
        if bytes > EVENT_RECORD_BYTES
            || self.entries.len() >= EVENT_ITEMS
            || bytes > EVENT_BYTES.saturating_sub(self.bytes)
        {
            self.dropped = self.dropped.saturating_add(1);
            return (expired, false);
        }
        self.entries.push_back((self.now_ns, bytes));
        self.bytes += bytes;
        (expired, true)
    }
}

pub struct EventRetention {
    capture_content: bool,
    retained: Mutex<RetainedEvents>,
}

impl lk_agents_abi::telemetry::observation::EventRetention for EventRetention {
    fn new(capture_content: bool) -> Self {
        Self { capture_content, retained: Mutex::new(RetainedEvents::default()) }
    }

    fn claim(&self, job_id: String) {
        self.retained.lock().unwrap_or_else(|p| p.into_inner()).job_id.get_or_insert(job_id);
    }

    fn allows_read(&self, job_id: String) -> bool {
        self.retained
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .job_id
            .as_ref()
            .is_none_or(|owner| owner == &job_id)
    }

    fn accepts(&self, kind: String, redaction: bool) -> bool {
        match kind.as_str() {
            "user_state_changed"
            | "agent_state_changed"
            | "user_transcription_timeout"
            | "session_usage_updated" => true,
            "user_input_transcribed"
            | "conversation_item_added"
            | "function_tools_executed"
            | "tool_execution_updated"
            | "agent_false_interruption"
            | "error"
            | "close" => self.capture_content && !redaction,
            // SpeechCreated holds a live SpeechHandle; overlapping speech carries numpy arrays.
            _ => false,
        }
    }

    fn admit(&self, retained_bytes: usize, now_ns: u64) -> (usize, bool) {
        self.retained.lock().unwrap_or_else(|p| p.into_inner()).admit(retained_bytes, now_ns)
    }

    fn expire(&self, now_ns: u64) -> usize {
        self.retained.lock().unwrap_or_else(|p| p.into_inner()).expire(now_ns)
    }

    fn diagnostics(&self) -> Vec<(String, u64)> {
        let retained = self.retained.lock().unwrap_or_else(|p| p.into_inner());
        vec![
            ("retained_events".into(), retained.entries.len() as u64),
            ("retained_bytes".into(), retained.bytes as u64),
            ("dropped_events".into(), retained.dropped),
            ("expired_events".into(), retained.expired),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::telemetry::observation::{
        ObservationIdentity, ObservationKind, ObservationTime,
    };

    #[test]
    fn event_retention_expires_charged_snapshots_and_refuses_live_or_denied_content() {
        use lk_agents_abi::telemetry::observation::EventRetention as _;
        let recording = EventRetention::new(false);
        assert!(recording.allows_read("outside".into()));
        recording.claim("owner".into());
        recording.claim("another".into());
        assert!(recording.allows_read("owner".into()));
        assert!(!recording.allows_read("another".into()));
        assert!(!recording.accepts("user_input_transcribed".into(), false));
        assert!(!recording.accepts("speech_created".into(), false));
        assert!(!recording.accepts("unknown".into(), false));
        assert!(recording.accepts("agent_state_changed".into(), false));
        let opted_in = EventRetention::new(true);
        assert!(opted_in.accepts("user_input_transcribed".into(), false));
        assert!(!opted_in.accepts("user_input_transcribed".into(), true));
        for _ in 0..EVENT_BYTES / EVENT_RECORD_BYTES {
            assert_eq!(recording.admit(EVENT_RECORD_BYTES, 1), (0, true));
        }
        assert_eq!(recording.admit(1, 2), (0, false));
        assert_eq!(recording.expire(EVENT_RETENTION_NS), 0);
        assert_eq!(recording.expire(EVENT_RETENTION_NS + 1), 16);
        assert_eq!(recording.admit(EVENT_RECORD_BYTES + 1, EVENT_RETENTION_NS + 2), (0, false));
        for _ in 0..EVENT_ITEMS {
            assert_eq!(recording.admit(1, EVENT_RETENTION_NS + 2), (0, true));
        }
        assert_eq!(recording.admit(0, EVENT_RETENTION_NS + 2), (0, false));
        assert_eq!(recording.expire(0), 0, "a backwards clock cannot corrupt expiry order");
        assert_eq!(recording.expire(2 * EVENT_RETENTION_NS + 2), EVENT_ITEMS);
        let counters = recording.diagnostics();
        assert!(counters.contains(&("retained_bytes".into(), 0)));
        assert!(counters.contains(&("dropped_events".into(), 3)));
    }

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
