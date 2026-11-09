use lk_agents_abi::telemetry::decl::Value;
use lk_agents_abi::telemetry::record::{
    JobKey, Measurement, Record, RecordSink, SpanContext, SpanRecord,
};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Default)]
pub(crate) struct Records(Mutex<Vec<Record>>);

impl RecordSink for Records {
    fn record(&self, record: Record) {
        self.0.lock().push(record);
    }
}

impl Records {
    pub(crate) fn spans(&self) -> Vec<SpanRecord> {
        let all = self.0.lock();
        all.iter()
            .filter_map(|r| if let Record::Span(s) = r { Some(s.clone()) } else { None })
            .collect()
    }

    pub(crate) fn measurements(&self, instrument: &str) -> Vec<Measurement> {
        let all = self.0.lock();
        let found = all.iter().filter_map(|r| match r {
            Record::Measurement(m) if m.instrument == instrument => Some(m.clone()),
            _ => None,
        });
        found.collect()
    }
}

pub(crate) const ENTRYPOINT: SpanContext = SpanContext { trace_id: 3, span_id: 4, sampled: true };

pub(crate) fn labels(m: &Measurement) -> Vec<(String, Value)> {
    m.labels.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

// Catches: stalls over the rate limit producing spans, the suppressed count lost, or the
// measurement skipped for a suppressed stall.
#[test]
fn event_loop_stalls_are_rate_limited() {
    let records = Arc::new(Records::default());
    let tracer = lk_agents_telemetry::Tracer::new(
        records.clone(),
        Arc::new(lk_agents_abi::VirtualClock::new()),
        1,
    );
    let cx = tracer.root().continuing(Some(ENTRYPOINT));
    let blocked = crate::telemetry::Blocked::default();
    for _ in 0..32 {
        crate::telemetry::event_loop_blocked(
            &cx,
            &blocked,
            std::time::Duration::from_millis(200),
            None,
            None,
            lk_agents_abi::telemetry::record::StallState::Ended,
        );
    }
    crate::telemetry::event_loop_blocked(
        &cx,
        &blocked,
        std::time::Duration::from_millis(600),
        Some(vec![lk_agents_abi::telemetry::record::StackFrame {
            function: "busy".into(),
            file: "app.py".into(),
            line: 7,
        }]),
        None,
        lk_agents_abi::telemetry::record::StallState::Ended,
    );
    // a loop still stuck is reported past the limit
    crate::telemetry::event_loop_blocked(
        &cx,
        &blocked,
        std::time::Duration::from_secs(5),
        None,
        None,
        lk_agents_abi::telemetry::record::StallState::Ongoing,
    );
    let spans = records.spans();
    assert_eq!(spans.len(), 31, "30 a minute, and the still-stuck loop past them");
    assert!(spans.iter().all(|s| s.parent == Some(4)));
    let durations = records.measurements("lk.agents.event_loop.blocked_duration");
    assert_eq!(durations.len(), 34);
    // the 33rd is over the rate limit: its stack rides no span
    assert_eq!(labels(&durations[32]), [("severity".to_owned(), Value::String("error".into()))]);
}

#[test]
fn a_stall_span_keeps_its_formatted_stack_and_threshold() {
    let records = Arc::new(Records::default());
    let tracer = lk_agents_telemetry::Tracer::new(
        records.clone(),
        Arc::new(lk_agents_abi::VirtualClock::new()),
        1,
    );
    crate::telemetry::event_loop_blocked(
        &tracer.root(),
        &crate::telemetry::Blocked::default(),
        std::time::Duration::from_millis(500),
        Some(vec![
            lk_agents_abi::telemetry::record::StackFrame {
                function: "busy".into(),
                file: "/app/tools.py".into(),
                line: 7,
            },
            lk_agents_abi::telemetry::record::StackFrame {
                function: "<module>".into(),
                file: "/app/main.py".into(),
                line: 1,
            },
        ]),
        Some("/app/tools.py".into()),
        lk_agents_abi::telemetry::record::StallState::Ended,
    );
    let spans = records.spans();
    let span = &spans[0];
    assert_eq!(span.name, "event_loop_blocked");
    for (key, expected) in [
        ("lk.blocking.duration", Value::Double(0.5)),
        ("lk.blocking.threshold", Value::Double(0.1)),
        ("lk.blocking.severity", Value::String("error".into())),
        (
            "lk.blocking.stack",
            Value::String("busy (/app/tools.py:7)\n<module> (/app/main.py:1)".into()),
        ),
        ("lk.blocking.import", Value::String("/app/tools.py".into())),
    ] {
        assert_eq!(
            span.attributes.iter().find(|(name, _)| name == key).map(|(_, value)| value),
            Some(&expected),
            "{key}"
        );
    }
}

// Catches: a stall of the job's loop missing its sessions, or reaching another job's.
#[test]
fn a_stall_reaches_the_sessions_of_its_job() {
    let heard: Arc<Mutex<Vec<(&str, std::time::Duration)>>> = Arc::default();
    let watch = |name: &'static str, job: &str| {
        let heard = heard.clone();
        let session = SpanContext { trace_id: 1, span_id: 2, sampled: true };
        crate::telemetry::watch_stalls(
            Some(JobKey(job.into())),
            session,
            Arc::new(move |d| heard.lock().push((name, d))),
        )
    };
    let stall = || crate::telemetry::LoopStall {
        duration: std::time::Duration::from_millis(250),
        parent: None,
        job: Some("AJ_stall".into()),
        stack: None,
        import: None,
        state: lk_agents_abi::telemetry::record::StallState::Ended,
    };
    let (ours, theirs) = (watch("ours", "AJ_stall"), watch("theirs", "AJ_other"));
    crate::telemetry::loop_stalled(stall());
    drop(ours);
    crate::telemetry::loop_stalled(stall());
    drop(theirs);
    assert_eq!(*heard.lock(), [("ours", std::time::Duration::from_millis(250))]);
}
