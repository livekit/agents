use super::*;
use lk_agents_abi::VirtualClock;
use lk_agents_abi::errors::{ApiError, HostError};
use lk_agents_abi::telemetry::decl::Value;
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_abi::telemetry::spans::inference::{
    Attempt, AttemptEnd, LlmRequestRun, TtsRequest, TtsTarget,
};
use lk_agents_abi::telemetry::spans::job::{
    EntrypointStarted, JobEntrypoint, JobShutdown, OnSessionEnd,
};
use lk_agents_abi::telemetry::spans::text::{
    AgentRun, AgentRunEnd, JudgeEvaluation, JudgeEvaluationEnd, RunStatus,
};

#[derive(Default)]
struct Collect(Mutex<Vec<Record>>);

impl RecordSink for Collect {
    fn record(&self, record: Record) {
        self.0.lock().push(record);
    }
}

impl Collect {
    fn spans(&self) -> Vec<SpanRecord> {
        let records = self.0.lock();
        records
            .iter()
            .filter_map(|r| if let Record::Span(s) = r { Some(s.clone()) } else { None })
            .collect()
    }
}

fn tracer() -> (Tracer, Arc<Collect>) {
    let sink = Arc::new(Collect::default());
    (Tracer::new(sink.clone(), Arc::new(VirtualClock::new()), 7), sink)
}

fn attr<'a>(span: &'a SpanRecord, key: &str) -> Option<&'a Value> {
    span.attributes.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn job() -> JobEntrypoint {
    JobEntrypoint {
        dispatch_id: None,
        worker_id: None,
        agent_id: None,
        accept_latency: None,
        assignment_latency: None,
        launch_latency: None,
        entrypoint_latency: None,
        dispatch_latency: None,
    }
}

fn run() -> AgentRun {
    AgentRun { operation: GenAiOperation::InvokeAgent, agent_name: "front".into() }
}

// Catches: a dropped span never recorded, or recorded without `lk.cancelled`, or a child not
// parented under the span whose context opened it.
#[test]
fn a_dropped_span_ends_cancelled_under_its_parent() {
    let (tracer, sink) = tracer();
    let root = tracer.root();
    {
        let span = start(&root, job());
        let child = start(span.cx(), JobShutdown { reason: "done".into(), user_initiated: false });
        drop(child);
        event(&span, EntrypointStarted);
    }
    let spans = sink.spans();
    let [shutdown, entry] = spans.as_slice() else { panic!("{spans:?}") };
    assert_eq!(shutdown.parent, Some(entry.context.span_id));
    assert_eq!(shutdown.context.trace_id, entry.context.trace_id);
    assert_eq!(entry.parent, None);
    for span in [shutdown, entry] {
        assert_eq!(attr(span, "lk.cancelled"), Some(&Value::Bool(true)), "{span:?}");
        assert_eq!(span.status, Status::Unset);
    }
    assert_eq!(entry.events[0].name, "entrypoint_started");
}

// Catches: `check` losing the error's type, message or traceback, an `Err` recorded as
// cancelled, or `end` attributes missing.
#[test]
fn check_records_the_exception_and_end_its_attributes() {
    let (tracer, sink) = tracer();
    let cx = tracer.root().with_job(Some(JobKey("J".into()))).with_conversation(Some("RM_1"));
    let failed: Result<(), HostError> =
        Err(HostError::new("boom").with_traceback(Some("Traceback ...".into())));
    {
        let span = start(&cx, run());
        let _ = check(&span, failed);
    }
    let span = start(&cx, run());
    let ok: Result<u8, ApiError> = Ok(1);
    assert_eq!(check(&span, ok), Ok(1));
    end_with(span, AgentRunEnd { status: RunStatus::Completed });

    let spans = sink.spans();
    let [failed, ended] = spans.as_slice() else { panic!("{spans:?}") };
    assert_eq!(failed.status, Status::Error("boom".into()));
    assert_eq!(attr(failed, "error.type"), Some(&Value::String("HostError".into())));
    assert_eq!(attr(failed, "lk.cancelled"), None, "an error is never a cancellation");
    let exception = &failed.events[0];
    assert_eq!(exception.name, "exception");
    assert!(
        exception
            .attributes
            .contains(&("exception.stacktrace".into(), Value::String("Traceback ...".into())))
    );
    assert_eq!(attr(ended, "lk.run.status"), Some(&Value::String("completed".into())));
    assert_eq!(attr(ended, "gen_ai.conversation.id"), Some(&Value::String("RM_1".into())));
    assert_eq!(ended.job, Some(JobKey("J".into())));
    assert_eq!(attr(ended, "lk.cancelled"), None);
}

// Catches: a `Current` span refusing a foreign parent, an `External` span refusing none, or a
// span keeping a foreign parent's trace.
#[test]
fn spans_continue_foreign_contexts() {
    let (tracer, sink) = tracer();
    let foreign = SpanRef { trace_id: TraceId(42), span_id: SpanId(9), sampled: true };
    let cx = tracer.root().continuing(Some(foreign));
    drop(start(&cx, JudgeEvaluation { name: "j".into() }));
    drop(start(&cx, run()));
    drop(start(&tracer.root().continuing(None), job()));
    for span in &sink.spans()[..2] {
        assert_eq!((span.context.trace_id, span.parent), (TraceId(42), Some(SpanId(9))));
    }
    let under = tracer.root().under::<JobEntrypoint>(foreign);
    let shutdown = start(&under, JobShutdown { reason: "r".into(), user_initiated: true });
    end_with(start(shutdown.cx(), OnSessionEnd { callback: "cb".into() }), ());
    let _ = JudgeEvaluationEnd { verdict: None, explanation: None };
}

// Catches: a span opened under an undeclared parent going unnoticed in debug builds.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "llm_request_run opened under")]
fn an_undeclared_parent_fails_the_debug_assertion() {
    let (tracer, _) = tracer();
    let attempt = Attempt { provider: "p".into(), model: "m".into(), retry_count: 0 };
    let span = start(
        &tracer.root(),
        TtsRequest {
            target: TtsTarget {
                streaming: false,
                label: "l".into(),
                provider: "p".into(),
                model: "m".into(),
            },
            recovery: None,
        },
    );
    end_with(start(span.cx(), LlmRequestRun { attempt }), AttemptEnd::new(vec![]));
}

// Catches: ids repeating or depending on anything but the seed (the DST replays them).
#[test]
fn ids_follow_the_seed() {
    let ids = |seed| {
        let tracer = Tracer::disabled(Arc::new(VirtualClock::new()));
        let tracer = Tracer::new(tracer.0.sink.clone(), tracer.0.clock.clone(), seed);
        let span = start(&tracer.root(), job());
        span.cx().span().unwrap()
    };
    assert_eq!(ids(3), ids(3));
    assert_ne!(ids(3).span_id, ids(4).span_id);
}

struct Response(u8);

impl SpanEnd<AgentRun> for Response {
    fn end(&self) -> AgentRunEnd {
        AgentRunEnd { status: if self.0 == 0 { RunStatus::Completed } else { RunStatus::Failed } }
    }
}

#[span(AgentRun { operation: GenAiOperation::InvokeAgent, agent_name: name.into() })]
async fn traced(cx: &Cx, name: &str, fail: bool) -> Result<Response, ApiError> {
    if fail {
        return Err(ApiError::timeout(None));
    }
    let inner = start(cx, JudgeEvaluation { name: name.into() });
    drop(inner);
    Ok(Response(0))
}

// Catches: the `#[span]` attribute not opening exactly one span, its body not running under it,
// an early `return` escaping `check`, or `Ok` not ended through `SpanEnd`.
#[tokio::test]
async fn span_functions_are_one_span() {
    let (tracer, sink) = tracer();
    assert!(traced(&tracer.root(), "a", false).await.is_ok());
    assert!(traced(&tracer.root(), "b", true).await.is_err());
    let spans = sink.spans();
    let [judge, ok, failed] = spans.as_slice() else { panic!("{spans:?}") };
    assert_eq!(judge.parent, Some(ok.context.span_id));
    assert_eq!(attr(ok, "lk.run.status"), Some(&Value::String("completed".into())));
    assert_eq!(attr(failed, "error.type"), Some(&Value::String("timeout".into())));
    assert_eq!(attr(failed, "lk.cancelled"), None);
}
