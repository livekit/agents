//! The session's runtime spans and measurements: its trace and job, usage counters, late
//! realtime metrics and event-loop stalls.

use super::*;
use crate::voice::session::{DriverBuilder, Timebase};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::telemetry::decl::Value;
use lk_agents_abi::telemetry::record::{
    JobKey, Measurement, Record, RecordSink, SpanId, SpanRecord, SpanRef, TraceContext, TraceId,
};
use lk_agents_voice::events::SessionEvent;
use parking_lot::Mutex;

#[derive(Default)]
struct Records(Mutex<Vec<Record>>);

impl RecordSink for Records {
    fn record(&self, record: Record) {
        self.0.lock().push(record);
    }
}

impl Records {
    fn spans(&self) -> Vec<SpanRecord> {
        let all = self.0.lock();
        all.iter()
            .filter_map(|r| if let Record::Span(s) = r { Some(s.clone()) } else { None })
            .collect()
    }

    fn measurements(&self, instrument: &str) -> Vec<Measurement> {
        let all = self.0.lock();
        let found = all.iter().filter_map(|r| match r {
            Record::Measurement(m) if m.instrument == instrument => Some(m.clone()),
            _ => None,
        });
        found.collect()
    }
}

const JOB: &str = "AJ_1";
const ENTRYPOINT: SpanRef = SpanRef { trace_id: TraceId(3), span_id: SpanId(4), sampled: true };

async fn traced(name: &str) -> (VoiceRuntime, Arc<Records>) {
    let records = Arc::new(Records::default());
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let time = Timebase::from_virtual(clock.clone());
    let config = lk_agents_voice::agent_activity::DriverConfig::new(
        Default::default(),
        lk_agents_voice::agent_activity::options::SessionLegs::none(),
    );
    let spawner = Arc::new(lk_agents_net::task::TokioSpawner::current());
    let mut builder = DriverBuilder::new("agent".into(), time, spawner, config);
    builder.session_id = Some(name.into());
    builder.tracer = lk_agents_net::telemetry::Tracer::new(records.clone(), clock, 5);
    builder.trace =
        TraceContext { span: Some(ENTRYPOINT), job: Some(JobKey(JOB.into())), conversation: None };
    let mut executor = Executor::new(builder.time.clone(), builder.spawner.clone());
    let (cores, _handle) = builder.start(&mut executor).await.unwrap();
    (executor.into_runtime(cores), records)
}

fn labels(m: &Measurement) -> Vec<(String, Value)> {
    m.labels.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

// Catches: the session's spans leaving the job's trace or its Cloud scope.
#[tokio::test(start_paused = true)]
async fn the_session_joins_the_jobs_trace() {
    let (runtime, _records) = traced("joined").await;
    let start = runtime.driver.session_start_span().expect("session_start is open");
    assert_eq!(start.trace_id, TraceId(3), "the session continues the entrypoint's trace");
    assert_eq!(runtime.shared.trace.job(), Some(&JobKey(JOB.into())));
}

// Catches: usage counted under GenAI labels instead of 1.8.10's model labels, a zero counted, or
// the connection time missing its reuse label.
#[tokio::test(start_paused = true)]
async fn usage_counters_carry_the_model_labels() {
    let (mut runtime, records) = traced("usage").await;
    let mut tts = lk_agents_abi::metrics::testkit::tts();
    tts.metadata = lk_agents_abi::metrics::testkit::meta("cartesia", "sonic");
    tts.characters_count = 12;
    tts.audio_duration = 0.0;
    tts.acquire_time = Some(0.2);
    tts.connection_reused = true;
    runtime.deliver_session_event(SessionEvent::MetricsCollected {
        metrics: Box::new(AgentMetrics::Tts(tts)),
        speech: None,
        counted: false,
    });
    let characters = records.measurements("lk.agents.usage.tts_characters");
    assert_eq!(characters.len(), 1);
    assert_eq!(
        labels(&characters[0]),
        [
            ("model_provider".to_owned(), Value::String("cartesia".into())),
            ("model_name".to_owned(), Value::String("sonic".into())),
        ]
    );
    assert_eq!(characters[0].job, Some(JobKey(JOB.into())));
    assert!(records.measurements("lk.agents.usage.tts_audio_duration").is_empty());
    let acquire = records.measurements("lk.agents.connection.acquire_time");
    assert!(
        labels(&acquire[0]).contains(&("connection_reused".into(), Value::String("true".into())))
    );
}

// Catches: a response's metrics arriving after its span ended being dropped, or recorded on a
// span outside the ended inference.
#[tokio::test(start_paused = true)]
async fn late_realtime_metrics_become_a_child_of_the_ended_inference() {
    use crate::voice::generation::realtime_generation::RealtimeSpans;
    let (runtime, records) = traced("realtime").await;
    let spans = RealtimeSpans::default();
    spans.set_model(Some(lk_agents_abi::ProviderId {
        label: "rt".into(),
        model: "gpt-realtime".into(),
        provider: "openai".into(),
    }));
    let turn = SpanRef { trace_id: TraceId(3), span_id: SpanId(9), sampled: true };
    let early = spans.start(&runtime.shared, Some(turn)).expect("a model");
    let mut m = lk_agents_abi::metrics::testkit::realtime();
    m.request_id = "resp_early".into();
    m.ttft = None;
    spans.metrics(&runtime.shared.trace, &m);
    spans.end(early, Some("resp_early"));
    let late = spans.start(&runtime.shared, Some(turn)).expect("a model");
    spans.end(late, Some("resp_late"));
    m.request_id = "resp_late".into();
    spans.metrics(&runtime.shared.trace, &m);

    let spans = records.spans();
    let inferences: Vec<_> = spans.iter().filter(|s| s.name == "realtime_inference").collect();
    assert_eq!(inferences.len(), 2);
    let blob = |s: &SpanRecord| s.attributes.iter().any(|(k, _)| *k == "lk.realtime_model_metrics");
    assert!(blob(inferences[0]), "early metrics land on the span: {:?}", inferences[0]);
    let ttft = inferences[0].attributes.iter().find(|(k, _)| *k == "lk.realtime_model_metrics");
    let Some((_, Value::String(json))) = ttft else { panic!() };
    assert!(json.contains("\"ttft\":-1.0"), "{json}");
    let late = spans.iter().find(|s| s.name == "realtime_metrics").expect("{spans:?}");
    assert_eq!(late.parent, Some(inferences[1].context.span_id));
    assert!(blob(late));
}

// Catches: stalls over the rate limit producing spans, the suppressed count lost, or the
// measurement skipped for a suppressed stall.
#[test]
fn event_loop_stalls_are_rate_limited() {
    let records = Arc::new(Records::default());
    let tracer = lk_agents_net::telemetry::Tracer::new(
        records.clone(),
        Arc::new(lk_agents_abi::VirtualClock::new()),
        1,
    );
    let cx = tracer.root().continuing(Some(ENTRYPOINT));
    let blocked = crate::telemetry::Blocked::default();
    for _ in 0..32 {
        crate::telemetry::event_loop_blocked(&cx, &blocked, 0.2, None);
    }
    crate::telemetry::event_loop_blocked(&cx, &blocked, 0.6, None);
    let spans = records.spans();
    assert_eq!(spans.len(), 30, "30 a minute");
    assert!(spans.iter().all(|s| s.parent == Some(SpanId(4))));
    let durations = records.measurements("lk.agents.event_loop.blocked_duration");
    assert_eq!(durations.len(), 33);
    assert_eq!(labels(&durations[32]), [("severity".to_owned(), Value::String("error".into()))]);
}

/// A job whose room connection records the span it ran under.
#[derive(Default)]
struct Job(Mutex<Vec<Option<SpanRef>>>);

impl lk_agents_abi::worker::job::SessionJob for Job {
    fn connect(
        &self,
        trace: TraceContext,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), lk_agents_abi::errors::HostError>> {
        self.0.lock().push(trace.span);
        Box::pin(async { Ok(()) })
    }
}

// Catches: the room connecting before `session_start` exists, so `room_connect` parents to the
// job instead of the session's start; or the driver drawing spans other than those `start()`
// reserved, which the startup work already ran under.
#[tokio::test(start_paused = true)]
async fn startup_runs_under_the_spans_start_reserved() {
    use lk_agents_abi::telemetry::spans::session::{AgentSession, SessionStart};
    let records = Arc::new(Records::default());
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let tracer = lk_agents_net::telemetry::Tracer::new(records.clone(), clock.clone(), 5);
    let env = lk_agents_net::env::ProcessEnv {
        tracer: tracer.clone(),
        ..lk_agents_net::env::ProcessEnv::current_runtime()
    };
    let time = Timebase::from_virtual(clock);
    let room = crate::room_io::FixedRoom { rtc: None, time, pool: None, tracer: tracer.clone() };
    let job = Arc::new(Job::default());
    let args = lk_agents_abi::voice::session::VoiceSessionArgs {
        agent: "agent".into(),
        job: Some(job.clone()),
        ..Default::default()
    };
    let session = <VoiceSession as lk_agents_abi::voice::session::VoiceSession>::new(
        &env,
        Arc::new(room),
        args,
        None,
    )
    .unwrap();
    let trace =
        TraceContext { span: Some(ENTRYPOINT), job: Some(JobKey(JOB.into())), conversation: None };
    let cx = tracer.resume(&trace);
    let reserved = cx.reserve::<AgentSession>();
    let start = cx.under::<AgentSession>(reserved.context).reserve::<SessionStart>();
    let opening = lk_agents_voice::telemetry::Opening { session: reserved, start, activity: None };
    session.trace_under(trace, opening);
    use lk_agents_abi::voice::session::VoiceSession as _;
    session.start().unwrap().await.unwrap();
    assert_eq!(*job.0.lock(), [Some(start.context)], "the room connects under session_start");
    session.close_session(lk_agents_abi::voice::events::CloseReason::UserInitiated, false).await;
    let spans = records.spans();
    let named = |name: &str| spans.iter().find(|s| s.name == name).expect(name).clone();
    let root = named("agent_session");
    assert_eq!((root.context, root.parent), (reserved.context, Some(ENTRYPOINT.span_id)));
    assert_eq!(root.start_unix_nanos, reserved.start, "the session starts at start()");
    let started = named("session_start");
    assert_eq!((started.context, started.parent), (start.context, Some(root.context.span_id)));
}

// Catches: a stall of the job's loop missing its sessions, or reaching another job's.
#[test]
fn a_stall_reaches_the_sessions_of_its_job() {
    use lk_agents_abi::telemetry::host::Telemetry as _;
    let heard: Arc<Mutex<Vec<(&str, std::time::Duration)>>> = Arc::default();
    let watch = |name: &'static str, job: &str| {
        let heard = heard.clone();
        let session = SpanRef { trace_id: TraceId(1), span_id: SpanId(2), sampled: true };
        crate::telemetry::watch_stalls(
            Some(JobKey(job.into())),
            session,
            Arc::new(move |d| heard.lock().push((name, d))),
        )
    };
    let (ours, theirs) = (watch("ours", "AJ_stall"), watch("theirs", "AJ_other"));
    crate::telemetry::Telemetry::_core_event_loop_blocked(
        0.25,
        None,
        Some("AJ_stall".into()),
        None,
    );
    drop(ours);
    crate::telemetry::Telemetry::_core_event_loop_blocked(
        0.25,
        None,
        Some("AJ_stall".into()),
        None,
    );
    drop(theirs);
    assert_eq!(*heard.lock(), [("ours", std::time::Duration::from_millis(250))]);
}
