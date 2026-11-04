//! The session report's event retention: every event a session accepts is kept until the report
//! is built. A session, or the process, retaining an unusual number of them logs one warning.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use lk_agents_abi::telemetry::policy::TelemetryValue;

use super::{ReportAdmission, RetainedEvents};
use crate::env::ProcessEnv;

/// A session retaining more events than this logs a warning, once.
const SESSION_EVENTS_WARNING: usize = 10_000;
/// Every session of the process retaining more events than this logs a warning, once per process.
pub(super) const PROCESS_EVENTS_WARNING: usize = 100_000;

/// A session's retained events, counted in the process's report admission.
pub struct EventRetention {
    capture_content: bool,
    job_id: Option<String>,
    retained: usize,
    process: Arc<RetainedEvents>,
}

impl EventRetention {
    pub fn with_admission(capture_content: bool, admission: &ReportAdmission) -> Self {
        Self { capture_content, job_id: None, retained: 0, process: admission.events.clone() }
    }

    fn count(&mut self) {
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

impl lk_agents_abi::telemetry::report::EventRetention for EventRetention {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv) -> Self {
        Self::with_admission(env.tracer.capture_content(), &ReportAdmission::process())
    }

    fn claim(&mut self, job_id: String) {
        self.job_id.get_or_insert(job_id);
    }

    fn allows_read(&self, job_id: String) -> bool {
        self.job_id.as_ref().is_none_or(|owner| owner == &job_id)
    }

    fn accepts(&self, kind: String, redaction: bool) -> bool {
        lk_agents_core::telemetry::report::event_allowed(&kind, self.capture_content, redaction)
    }

    fn admit(&mut self, _event: TelemetryValue, _now_ns: u64) -> (usize, bool) {
        self.count();
        (0, true)
    }

    fn expire(&mut self, _now_ns: u64) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::telemetry::report::EventRetention as _;

    use super::*;

    #[test]
    fn event_retention_keeps_every_event_and_refuses_live_or_denied_content() {
        let admission = ReportAdmission::default();
        let mut recording = EventRetention::with_admission(false, &admission);
        assert!(recording.allows_read("outside".into()));
        recording.claim("owner".into());
        recording.claim("another".into());
        assert!(recording.allows_read("owner".into()));
        assert!(!recording.allows_read("another".into()));
        assert!(!recording.accepts("user_input_transcribed".into(), false));
        assert!(!recording.accepts("speech_created".into(), false));
        assert!(!recording.accepts("unknown".into(), false));
        assert!(recording.accepts("agent_state_changed".into(), false));
        assert!(recording.accepts("error".into(), true) && recording.accepts("close".into(), true));
        let opted_in = EventRetention::with_admission(true, &admission);
        assert!(opted_in.accepts("user_input_transcribed".into(), false));
        assert!(!opted_in.accepts("user_input_transcribed".into(), true));

        for _ in 0..PROCESS_EVENTS_WARNING + 1 {
            assert_eq!(recording.admit(TelemetryValue::None, 1), (0, true));
        }
        assert_eq!(recording.expire(u64::MAX), 0);
        assert!(admission.events.warned.load(Ordering::Relaxed));
        drop(recording);
        assert_eq!(admission.events.items.load(Ordering::Relaxed), 0);
    }
}
