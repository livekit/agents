//! The session report's event retention: per-session snapshot budgets under the process ceiling
//! of its [`ReportAdmission`].

use std::collections::VecDeque;
use std::sync::Arc;

use lk_agents_abi::telemetry::upload::UploadError;

use super::admission::{EventQuota, ReportAdmission};
use crate::env::ProcessEnv;

/// Session report snapshots expire after one hour. These budgets apply before the host copies
/// an event, independently of SDK exporter retention.
const EVENT_ITEMS: usize = lk_agents_abi::telemetry::policy::REPORT_EVENT_ITEMS;
const EVENT_BYTES: usize = lk_agents_abi::telemetry::policy::REPORT_EVENT_BYTES;
const EVENT_RECORD_BYTES: usize = lk_agents_abi::telemetry::policy::REPORT_EVENT_RECORD_BYTES;
const EVENT_RETENTION_NS: u64 = 3600 * 1_000_000_000;

struct RetainedEvents {
    quota: Arc<EventQuota>,
    job_id: Option<String>,
    entries: VecDeque<(u64, usize)>,
    bytes: usize,
    now_ns: u64,
    dropped: u64,
    expired: u64,
}

impl RetainedEvents {
    fn new(quota: Arc<EventQuota>) -> Self {
        Self {
            quota,
            job_id: None,
            entries: VecDeque::new(),
            bytes: 0,
            now_ns: 0,
            dropped: 0,
            expired: 0,
        }
    }

    fn expire(&mut self, now_ns: u64) -> usize {
        // Admission time, rather than the event's wall timestamp, orders the expiry queue.
        self.now_ns = self.now_ns.max(now_ns);
        let mut expired = 0;
        let mut released_bytes = 0;
        while self
            .entries
            .front()
            .is_some_and(|(at, _)| self.now_ns.saturating_sub(*at) >= EVENT_RETENTION_NS)
        {
            let (_, bytes) = self.entries.pop_front().expect("front exists");
            self.bytes -= bytes;
            released_bytes += bytes;
            expired += 1;
        }
        self.quota.release(expired, released_bytes);
        self.expired = self.expired.saturating_add(expired as u64);
        expired
    }

    fn admit(&mut self, bytes: usize, now_ns: u64) -> (usize, bool) {
        let expired = self.expire(now_ns);
        if bytes > EVENT_RECORD_BYTES
            || self.entries.len() >= EVENT_ITEMS
            || bytes > EVENT_BYTES.saturating_sub(self.bytes)
            || !self.quota.admit(bytes)
        {
            self.dropped = self.dropped.saturating_add(1);
            return (expired, false);
        }
        self.entries.push_back((self.now_ns, bytes));
        self.bytes += bytes;
        (expired, true)
    }
}

impl Drop for RetainedEvents {
    fn drop(&mut self) {
        self.quota.release(self.entries.len(), self.bytes);
    }
}

/// A session's retained events, charged to the process's report admission.
pub struct EventRetention {
    capture_content: bool,
    retained: RetainedEvents,
}

impl EventRetention {
    fn with_admission(capture_content: bool, admission: &ReportAdmission) -> Self {
        Self { capture_content, retained: RetainedEvents::new(admission.events.clone()) }
    }
}

impl lk_agents_abi::telemetry::observation::EventRetention for EventRetention {
    type Env = ProcessEnv;

    fn new(_env: &ProcessEnv, capture_content: bool) -> Result<Self, UploadError> {
        Ok(Self::with_admission(capture_content, &ReportAdmission::process()))
    }

    fn claim(&mut self, job_id: String) {
        self.retained.job_id.get_or_insert(job_id);
    }

    fn allows_read(&self, job_id: String) -> bool {
        self.retained.job_id.as_ref().is_none_or(|owner| owner == &job_id)
    }

    fn capture_content(&self) -> bool {
        self.capture_content
    }

    fn accepts(&self, kind: String, redaction: bool) -> bool {
        lk_agents_core::telemetry::report::event_allowed(&kind, self.capture_content, redaction)
    }

    fn admit(&mut self, retained_bytes: usize, now_ns: u64) -> (usize, bool) {
        self.retained.admit(retained_bytes, now_ns)
    }

    fn expire(&mut self, now_ns: u64) -> usize {
        self.retained.expire(now_ns)
    }

    fn diagnostics(&self) -> Vec<(String, u64)> {
        let retained = &self.retained;
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
    use super::super::admission::{PROCESS_EVENT_BYTES, PROCESS_EVENT_ITEMS};
    use super::*;

    type Retention = EventRetention;

    #[test]
    fn event_retention_expires_charged_snapshots_and_refuses_live_or_denied_content() {
        use lk_agents_abi::telemetry::observation::EventRetention as _;
        let admission = ReportAdmission::default();
        let mut recording = Retention::with_admission(false, &admission);
        assert!(recording.allows_read("outside".into()));
        recording.claim("owner".into());
        recording.claim("another".into());
        assert!(recording.allows_read("owner".into()));
        assert!(!recording.allows_read("another".into()));
        assert!(!recording.accepts("user_input_transcribed".into(), false));
        assert!(!recording.accepts("speech_created".into(), false));
        assert!(!recording.accepts("unknown".into(), false));
        assert!(recording.accepts("agent_state_changed".into(), false));
        let opted_in = Retention::with_admission(true, &admission);
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

    #[test]
    fn session_event_stores_share_capacity_until_expiry_or_drop() {
        use lk_agents_abi::telemetry::observation::EventRetention as _;

        let admission = ReportAdmission::default();
        let store = || Retention::with_admission(false, &admission);
        let records_per_store = EVENT_BYTES / EVENT_RECORD_BYTES;
        let mut stores: Vec<_> = (0..PROCESS_EVENT_BYTES / EVENT_BYTES).map(|_| store()).collect();
        // Concurrent sessions consume the process's full byte ceiling before any is released.
        std::thread::scope(|scope| {
            for session in &mut stores {
                scope.spawn(move || {
                    for _ in 0..records_per_store {
                        assert_eq!(session.admit(EVENT_RECORD_BYTES, 1), (0, true));
                    }
                });
            }
        });
        let mut overflow = store();
        assert_eq!(overflow.admit(EVENT_RECORD_BYTES, 1), (0, false));
        assert_eq!(stores[0].expire(EVENT_RETENTION_NS + 1), records_per_store);
        assert_eq!(overflow.admit(EVENT_RECORD_BYTES, 1), (0, true));
        drop(stores.pop());
        for _ in 1..records_per_store {
            assert_eq!(overflow.admit(EVENT_RECORD_BYTES, 1), (0, true));
        }
        assert_eq!(overflow.admit(1, 1), (0, false), "session ceiling still applies");
        drop(stores);
        drop(overflow);

        // Tiny events exhaust the process's item ceiling before its byte ceiling.
        let mut stores: Vec<_> = (0..PROCESS_EVENT_ITEMS / EVENT_ITEMS).map(|_| store()).collect();
        for session in &mut stores {
            for _ in 0..EVENT_ITEMS {
                assert_eq!(session.admit(1, 1), (0, true));
            }
        }
        let mut overflow = store();
        assert_eq!(overflow.admit(1, 1), (0, false));
        drop(stores);
        assert_eq!(overflow.admit(1, 1), (0, true));
    }
}
