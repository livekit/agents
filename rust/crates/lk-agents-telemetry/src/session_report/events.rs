//! The session report's event retention: every event a session accepts is kept until the report
//! is built. A session, or the process, retaining an unusual number of them logs one warning.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{ReportAdmission, RetainedEvents};
use lk_agents_abi::telemetry::report::ReportEventKind;

/// A session retaining more events than this logs a warning, once.
const SESSION_EVENTS_WARNING: usize = 10_000;
/// Every session of the process retaining more events than this logs a warning, once per process.
pub(super) const PROCESS_EVENTS_WARNING: usize = 100_000;

/// A session's retained events, counted in the process's report admission.
pub struct EventRetention {
    job_id: Option<String>,
    retained: usize,
    process: Arc<RetainedEvents>,
}

impl EventRetention {
    pub fn with_admission(admission: &ReportAdmission) -> Self {
        Self { job_id: None, retained: 0, process: admission.events.clone() }
    }

    /// Counted in the process's report admission.
    pub fn process() -> Self {
        Self::with_admission(&ReportAdmission::process())
    }

    /// Binds report access to the first job that starts the session; ownership never transfers.
    pub fn claim(&mut self, job_id: String) {
        self.job_id.get_or_insert(job_id);
    }

    /// A session started outside a job keeps its application-owned access.
    pub fn allows_read(&self, job_id: &str) -> bool {
        self.job_id.as_deref().is_none_or(|owner| owner == job_id)
    }

    /// Refuses a redacted session's content, unknown kinds and live-object events.
    pub fn accepts(&self, kind: &str, redaction: bool) -> bool {
        ReportEventKind::from_wire(kind)
            .is_some_and(|kind| lk_agents_core::telemetry::report::event_allowed(kind, redaction))
    }

    /// Counts one retained event; every accepted event is kept.
    pub fn admit(&mut self) {
        self.retained += 1;
        if self.retained == SESSION_EVENTS_WARNING + 1 {
            log::warn!(
                target: "livekit.agents",
                "the session report retains more than {SESSION_EVENTS_WARNING} events"
            );
        }
        let process = self.process.items.fetch_add(1, Ordering::Relaxed) + 1;
        if process > PROCESS_EVENTS_WARNING && !self.process.warned.swap(true, Ordering::Relaxed) {
            log::warn!(
                target: "livekit.agents",
                "session reports retain {process} events across the process"
            );
        }
    }
}

impl Drop for EventRetention {
    fn drop(&mut self) {
        self.process.items.fetch_sub(self.retained, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_retention_keeps_every_event_and_refuses_live_or_redacted_content() {
        let admission = ReportAdmission::default();
        let mut recording = EventRetention::with_admission(&admission);
        assert!(recording.allows_read("outside"));
        recording.claim("owner".into());
        recording.claim("another".into());
        assert!(recording.allows_read("owner"));
        assert!(!recording.allows_read("another"));
        assert!(recording.accepts("user_input_transcribed", false));
        assert!(!recording.accepts("user_input_transcribed", true));
        assert!(!recording.accepts("speech_created", false));
        assert!(!recording.accepts("unknown", false));
        assert!(recording.accepts("agent_state_changed", false));
        assert!(recording.accepts("error", true) && recording.accepts("close", true));

        for _ in 0..PROCESS_EVENTS_WARNING + 1 {
            recording.admit();
        }
        assert_eq!(recording.retained, PROCESS_EVENTS_WARNING + 1);
        assert!(admission.events.warned.load(Ordering::Relaxed));
        drop(recording);
        assert_eq!(admission.events.items.load(Ordering::Relaxed), 0);
    }
}
