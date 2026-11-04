use std::borrow::Cow;
use std::io::Read as _;
use std::sync::Arc;
use std::time::Duration;

use super::proto::collector::logs::v1::ExportLogsServiceRequest;
use super::proto::collector::metrics::v1::ExportMetricsServiceRequest;
use super::proto::collector::trace::v1::ExportTraceServiceRequest;
use super::proto::common::v1::{KeyValue, any_value::Value as Any};
use super::proto::metrics::v1 as proto_metrics;
use lk_agents_abi::telemetry::policy::REDACTED_EXCEPTION;
use lk_agents_abi::telemetry::record::{
    EventRecord, InstrumentKind, JobKey, KeyValues, LogRecord, Measurement, Number, Record, SpanId,
    SpanKind, SpanRecord, SpanRef, Status, TraceId, Value,
};
use prost::Message as _;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use super::*;

const DISABLED: &[u8] = b"project data recording is disabled by owner";

struct Received {
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Received {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    fn bytes(&self) -> Vec<u8> {
        if self.header("content-encoding") != Some("gzip") {
            return self.body.clone();
        }
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(self.body.as_slice()).read_to_end(&mut out).unwrap();
        out
    }

    fn decode<M: prost::Message + Default>(&self) -> M {
        assert_eq!(self.header("content-type"), Some("application/x-protobuf"));
        M::decode(self.bytes().as_slice()).unwrap()
    }
}

/// A local OTLP/HTTP endpoint: `reply` answers each path with a status and body, or never.
async fn endpoint(
    reply: impl Fn(&str) -> Option<(u16, &'static [u8])> + Send + Sync + 'static,
) -> (String, mpsc::UnboundedReceiver<Received>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::unbounded_channel();
    let reply = Arc::new(reply);
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let (tx, reply) = (tx.clone(), reply.clone());
            tokio::spawn(async move {
                let mut data = Vec::new();
                let end = loop {
                    let mut chunk = [0; 8192];
                    let n = socket.read(&mut chunk).await.unwrap();
                    if n == 0 {
                        return;
                    }
                    data.extend_from_slice(&chunk[..n]);
                    if let Some(end) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                        break end + 4;
                    }
                };
                let head = String::from_utf8(data[..end].to_vec()).unwrap();
                let mut lines = head.lines();
                let path = lines.next().unwrap().split(' ').nth(1).unwrap().to_owned();
                let headers: Vec<(String, String)> = lines
                    .filter_map(|l| l.split_once(':'))
                    .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
                    .collect();
                let len: usize = headers
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                    .map_or(0, |(_, v)| v.parse().unwrap());
                while data.len() - end < len {
                    let mut chunk = [0; 8192];
                    let n = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(n, 0);
                    data.extend_from_slice(&chunk[..n]);
                }
                let answer = reply(&path);
                let _ = tx.send(Received { path, headers, body: data[end..].to_vec() });
                let Some((status, body)) = answer else {
                    std::future::pending::<()>().await;
                    return;
                };
                let head = format!(
                    "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
            });
        }
    });
    (url, rx)
}

fn ok(_: &str) -> Option<(u16, &'static [u8])> {
    Some((200, b""))
}

fn received(rx: &mut mpsc::UnboundedReceiver<Received>) -> Vec<Received> {
    std::iter::from_fn(|| rx.try_recv().ok()).collect()
}

fn process() -> ProcessIdentity {
    ProcessIdentity {
        sdk_name: SdkName::Python,
        sdk_version: "2.0.0".into(),
        cloud_agent_id: Some("CA_agent".into()),
        deployment_id: None,
    }
}

fn exporter(limits: Limits) -> Exporter {
    Exporter::start(
        &crate::task::TokioSpawner::current(),
        Arc::new(crate::clock::RealClock::new()),
        process(),
        limits,
    )
}

fn cloud(url: &str) -> CloudEndpoint {
    CloudEndpoint { base_url: url.to_owned(), api_key: "key".into(), api_secret: "secret".into() }
}

fn key(job: &str) -> JobKey {
    JobKey(job.to_owned())
}

fn identity(job: &str) -> JobIdentity {
    JobIdentity {
        room_id: format!("RM_{job}"),
        job_id: job.to_owned(),
        room_name: "private-room".into(),
        agent_name: Some("front-desk".into()),
    }
}

fn recording() -> CloudRecording {
    CloudRecording { traces: true, logs: true, metrics: true, ..Default::default() }
}

fn span(name: &'static str, job: Option<&str>, attributes: KeyValues) -> Record {
    Record::Span(SpanRecord {
        context: SpanRef { trace_id: TraceId(7), span_id: SpanId(9), sampled: true },
        parent: Some(SpanId(3)),
        name: Cow::Borrowed(name),
        kind: SpanKind::Internal,
        start_unix_nanos: 1,
        end_unix_nanos: 2,
        attributes,
        events: Vec::new(),
        links: Vec::new(),
        status: Status::Unset,
        scope: Cow::Borrowed("livekit-agents"),
        job: job.map(key),
    })
}

fn counter(job: Option<&str>, value: i64) -> Record {
    Record::Measurement(Measurement {
        instrument: "lk.agents.usage.llm_input_tokens",
        kind: InstrumentKind::Counter,
        unit: "tokens",
        bounds: &[],
        value: Number::Int(value),
        labels: vec![("model_name".into(), "gpt".into())],
        job: job.map(key),
    })
}

fn attr<'a>(attributes: &'a [KeyValue], key: &str) -> Option<&'a Any> {
    attributes.iter().find(|kv| kv.key == key).and_then(|kv| kv.value.as_ref()?.value.as_ref())
}

fn string(value: &str) -> Option<Any> {
    Some(Any::StringValue(value.to_owned()))
}

const LIMIT: Duration = Duration::from_secs(5);

// Catches: a request mixing jobs, a record or resource missing the identity stamp or the native
// session flags, the PII room name on a resource, or a request without the observability token.
#[tokio::test]
async fn cloud_requests_carry_one_job_and_its_identity() {
    let (url, mut rx) = endpoint(ok).await;
    let exporter = exporter(Limits::default());
    exporter.configure_cloud(cloud(&url));
    for job in ["AJ_1", "AJ_2"] {
        exporter.open_job(key(job), identity(job));
    }
    exporter.register_job(&key("AJ_1"), recording());
    let simulation = Simulation { run_id: Some("SR_1".into()), job_id: None };
    exporter
        .register_job(&key("AJ_2"), CloudRecording { simulation: Some(simulation), ..recording() });
    for job in ["AJ_1", "AJ_2", "AJ_1", "AJ_2"] {
        exporter.record(span("agent_turn", Some(job), vec![("room_id".into(), "forged".into())]));
    }
    exporter.record(span("worker", None, vec![]));
    exporter.record(Record::Log(LogRecord {
        scope: "chat_history".into(),
        body: "chat item".into(),
        time_unix_nanos: 5,
        severity_number: 9,
        severity_text: "INFO".into(),
        attributes: vec![("chat.item".into(), Value::KvList(vec![]))],
        span: None,
        job: Some(key("AJ_1")),
    }));
    assert!(exporter.flush(LIMIT).await);
    let requests = received(&mut rx);
    assert_eq!(requests.len(), 3, "two trace requests, one log request");
    for request in &requests {
        assert_eq!(request.header("content-type"), Some("application/x-protobuf"));
        assert_eq!(request.header("content-encoding"), Some("gzip"));
        let token = request.header("authorization").unwrap().strip_prefix("Bearer ").unwrap();
        let claims =
            livekit_token::TokenVerifier::with_api_key("key", "secret").verify(token).unwrap();
        assert!(claims.observability.write);
    }
    let mut jobs = Vec::new();
    for request in requests.iter().filter(|r| r.path == "/observability/traces/otlp/v0") {
        let traces: ExportTraceServiceRequest = request.decode();
        let resource = traces.resource_spans[0].resource.as_ref().unwrap();
        let job = match attr(&resource.attributes, "job_id") {
            Some(Any::StringValue(job)) => job.clone(),
            other => panic!("{other:?}"),
        };
        for (k, v) in [
            ("service.name", "livekit-agents"),
            ("lk.sdk.name", "PYTHON"),
            ("lk.sdk.version", "2.0.0"),
            ("lk.job_id", job.as_str()),
            ("room_id", &format!("RM_{job}")),
            ("lk.room_id", &format!("RM_{job}")),
            ("lk.agent_name", "front-desk"),
            ("lk.cloud_agent_id", "CA_agent"),
        ] {
            assert_eq!(attr(&resource.attributes, k).cloned(), string(v), "{k}");
        }
        assert_eq!(attr(&resource.attributes, "lk.pii.room_name"), None);
        assert_eq!(attr(&resource.attributes, "lk.deployment_id"), None);
        let spans = &traces.resource_spans[0].scope_spans[0].spans;
        assert_eq!(spans.len(), 2);
        let simulated = job == "AJ_2";
        for span in spans {
            let a = &span.attributes;
            assert_eq!(attr(a, "job_id").cloned(), string(&job));
            assert_eq!(attr(a, "room_id").cloned(), string(&format!("RM_{job}")));
            assert_eq!(attr(a, "lk.pii.room_name").cloned(), string("private-room"));
            assert_eq!(attr(a, "lk.redaction.enabled"), Some(&Any::BoolValue(false)));
            assert_eq!(attr(a, "lk.simulation.enabled"), Some(&Any::BoolValue(simulated)));
            assert_eq!(
                attr(a, "lk.simulation.run_id").cloned(),
                simulated.then(|| Any::StringValue("SR_1".into()))
            );
            assert_eq!(span.trace_id, 7u128.to_be_bytes());
            assert_eq!(span.parent_span_id, 3u64.to_be_bytes());
        }
        jobs.push(job);
    }
    jobs.sort();
    assert_eq!(jobs, ["AJ_1", "AJ_2"]);
    let log = requests.iter().find(|r| r.path == "/observability/logs/otlp/v0").unwrap();
    let logs: ExportLogsServiceRequest = log.decode();
    let scope_logs = &logs.resource_logs[0].scope_logs[0];
    assert_eq!(scope_logs.scope.as_ref().unwrap().name, "chat_history");
    let record = &scope_logs.log_records[0];
    assert_eq!(attr(&record.attributes, "job_id").cloned(), string("AJ_1"));
    assert_eq!(attr(&record.attributes, "lk.redaction.enabled"), Some(&Any::BoolValue(false)));
}

// Catches: spans of an undecided job exported, lost on release, kept for a job that records no
// traces, or held without bound.
#[tokio::test]
async fn spans_are_held_until_the_recording_decision() {
    let (url, mut rx) = endpoint(ok).await;
    let exporter = exporter(Limits::default());
    exporter.configure_cloud(cloud(&url));
    exporter.open_job(key("AJ_1"), identity("AJ_1"));
    exporter.open_job(key("AJ_2"), identity("AJ_2"));
    exporter.open_job(key("AJ_3"), identity("AJ_3"));
    exporter.record(span("job_entrypoint", Some("AJ_1"), vec![]));
    exporter.record(span("job_entrypoint", Some("AJ_2"), vec![]));
    let held = super::destination::HELD_SPANS;
    for _ in 0..=held {
        exporter.record(span("job_entrypoint", Some("AJ_3"), vec![]));
    }
    assert!(exporter.flush(LIMIT).await);
    assert!(received(&mut rx).is_empty(), "nothing is sent before the decision");
    assert_eq!(exporter.diagnostics().dropped, 1, "the hold is bounded");

    exporter.register_job(&key("AJ_1"), recording());
    exporter.register_job(&key("AJ_2"), CloudRecording { traces: false, ..recording() });
    exporter.close_job(&key("AJ_3"), LIMIT).await;
    assert!(exporter.flush(LIMIT).await);
    let requests = received(&mut rx);
    assert_eq!(requests.len(), 1);
    let traces: ExportTraceServiceRequest = requests[0].decode();
    let spans = &traces.resource_spans[0].scope_spans[0].spans;
    assert_eq!((spans.len(), spans[0].name.as_str()), (1, "job_entrypoint"));
    assert_eq!(attr(&spans[0].attributes, "job_id").cloned(), string("AJ_1"));
    assert_eq!(exporter.diagnostics(), Diagnostics { dropped: 1, ..Default::default() });
}

fn sensitive() -> Record {
    let Record::Span(mut record) = span(
        "llm_request",
        Some("AJ_1"),
        vec![
            ("gen_ai.input.messages".into(), "secret prompt".into()),
            ("lk.pii.user_transcript".into(), "secret words".into()),
            (
                "lk.options".into(),
                Value::KvList(vec![
                    ("lk.pii.keyterms".into(), "secret".into()),
                    ("model".into(), "gpt".into()),
                ]),
            ),
            ("gen_ai.request.model".into(), "gpt".into()),
        ],
    ) else {
        unreachable!()
    };
    record.events.push(EventRecord {
        name: "exception".into(),
        time_unix_nanos: 2,
        attributes: vec![
            ("exception.type".into(), "ValueError".into()),
            ("exception.message".into(), "secret failure".into()),
            ("exception.stacktrace".into(), "secret trace".into()),
        ],
    });
    record.status = Status::Error("secret failure".into());
    Record::Span(record)
}

fn only_span(request: &Received) -> super::proto::trace::v1::Span {
    let traces: ExportTraceServiceRequest = request.decode();
    traces.resource_spans[0].scope_spans[0].spans[0].clone()
}

// Catches: Cloud stripping `lk.pii.*` itself (ingest does), Cloud keeping GenAI content or
// exception text for a redacted session, or the application endpoint receiving PII without
// `allow_pii`, at any depth.
#[tokio::test]
async fn each_destination_applies_its_own_content_policy() {
    for (redaction, allow_pii) in [(true, false), (false, true)] {
        let (cloud_url, mut cloud_rx) = endpoint(ok).await;
        let (app_url, mut app_rx) = endpoint(ok).await;
        let exporter = exporter(Limits::default());
        exporter.configure_cloud(cloud(&cloud_url));
        let mut app = AppConfig::with_endpoint(&app_url);
        app.allow_pii = allow_pii;
        app.metadata = vec![("deployment.environment".into(), "prod".into())];
        exporter.configure_app(Some(app));
        exporter.open_job(key("AJ_1"), identity("AJ_1"));
        exporter.register_job(&key("AJ_1"), CloudRecording { redaction, ..recording() });
        exporter.record(sensitive());
        assert!(exporter.flush(LIMIT).await);

        let cloud = only_span(&received(&mut cloud_rx)[0]);
        let a = &cloud.attributes;
        assert_eq!(attr(a, "lk.pii.user_transcript").cloned(), string("secret words"));
        assert_eq!(attr(a, "lk.redaction.enabled"), Some(&Any::BoolValue(redaction)));
        assert_eq!(attr(a, "gen_ai.input.messages").is_some(), !redaction);
        assert_eq!(attr(a, "gen_ai.request.model").cloned(), string("gpt"));
        let event = &cloud.events[0].attributes;
        let message = if redaction { REDACTED_EXCEPTION } else { "secret failure" };
        assert_eq!(attr(event, "exception.message").cloned(), string(message));
        assert_eq!(attr(event, "exception.stacktrace").is_some(), !redaction);
        assert_eq!(cloud.status.as_ref().unwrap().message, message);

        let request = received(&mut app_rx);
        assert_eq!(request[0].path, "/v1/traces");
        let app = only_span(&request[0]);
        let a = &app.attributes;
        assert_eq!(attr(a, "lk.pii.user_transcript").is_some(), allow_pii);
        assert_eq!(attr(a, "lk.pii.room_name").is_some(), allow_pii);
        assert_eq!(attr(a, "gen_ai.input.messages").is_some(), allow_pii);
        let Some(Any::KvlistValue(options)) = attr(a, "lk.options") else { panic!() };
        assert_eq!(attr(&options.values, "lk.pii.keyterms").is_some(), allow_pii);
        assert_eq!(attr(&options.values, "model").cloned(), string("gpt"));
        assert_eq!(attr(a, "job_id").cloned(), string("AJ_1"));
        assert_eq!(attr(a, "deployment.environment").cloned(), string("prod"));
        assert_eq!(attr(a, "lk.redaction.enabled"), None, "session flags are Cloud's");
        let message = if allow_pii { "secret failure" } else { REDACTED_EXCEPTION };
        assert_eq!(attr(&app.events[0].attributes, "exception.message").cloned(), string(message));
        assert_eq!(app.status.as_ref().unwrap().message, message);
    }
}

fn sums(request: &Received) -> (i32, Vec<(i64, Vec<KeyValue>)>) {
    let metrics: ExportMetricsServiceRequest = request.decode();
    let metric = &metrics.resource_metrics[0].scope_metrics[0].metrics[0];
    let Some(proto_metrics::metric::Data::Sum(sum)) = &metric.data else { panic!() };
    let points = sum
        .data_points
        .iter()
        .map(|p| {
            let Some(proto_metrics::number_data_point::Value::AsInt(v)) = p.value else { panic!() };
            (v, p.attributes.clone())
        })
        .collect();
    (sum.aggregation_temporality, points)
}

// Catches: cumulative sums sent to Cloud, a Cloud point without the job's identity, the
// application's points carrying job identity, or a job's last interval lost at close.
#[tokio::test]
async fn cloud_metrics_are_delta_per_job_and_application_metrics_cumulative() {
    let (cloud_url, mut cloud_rx) = endpoint(ok).await;
    let (app_url, mut app_rx) = endpoint(ok).await;
    let exporter = exporter(Limits::default());
    exporter.configure_cloud(cloud(&cloud_url));
    exporter.configure_app(Some(AppConfig::with_endpoint(&app_url)));
    exporter.open_job(key("AJ_1"), identity("AJ_1"));
    exporter.record(counter(Some("AJ_1"), 2));
    exporter.register_job(&key("AJ_1"), recording());
    exporter.record(counter(Some("AJ_1"), 3));
    assert!(exporter.flush(LIMIT).await);
    exporter.record(counter(Some("AJ_1"), 4));
    exporter.close_job(&key("AJ_1"), LIMIT).await;
    exporter.record(counter(Some("AJ_1"), 100));
    assert!(exporter.flush(LIMIT).await);

    let cloud = received(&mut cloud_rx);
    assert!(cloud.iter().all(|r| r.path == "/observability/metrics/otlp/v0"));
    let cloud: Vec<_> = cloud.iter().map(sums).collect();
    let delta = proto_metrics::AggregationTemporality::Delta as i32;
    assert_eq!(
        cloud.iter().map(|(t, p)| (*t, p[0].0)).collect::<Vec<_>>(),
        [(delta, 5), (delta, 4)]
    );
    let labels = &cloud[0].1[0].1;
    assert_eq!(attr(labels, "model_name").cloned(), string("gpt"));
    assert_eq!(attr(labels, "job_id").cloned(), string("AJ_1"));
    assert_eq!(attr(labels, "room_id").cloned(), string("RM_AJ_1"));

    let app: Vec<_> = received(&mut app_rx).iter().map(sums).collect();
    let cumulative = proto_metrics::AggregationTemporality::Cumulative as i32;
    assert_eq!(
        app.iter().map(|(t, p)| (*t, p[0].0)).collect::<Vec<_>>(),
        [(cumulative, 5), (cumulative, 109)]
    );
    assert_eq!(app[0].1[0].1.len(), 1, "only declared labels");
}

// Catches: a full queue blocking or growing instead of dropping and counting.
#[tokio::test]
async fn a_full_queue_drops_and_counts() {
    let (url, mut rx) = endpoint(ok).await;
    let exporter = exporter(Limits { records: 3, bytes: usize::MAX });
    exporter.configure_app(Some(AppConfig::with_endpoint(&url)));
    for _ in 0..5 {
        exporter.record(span("tts_request", None, vec![]));
    }
    assert_eq!(exporter.diagnostics().dropped, 2);
    assert!(exporter.flush(LIMIT).await);
    let traces: ExportTraceServiceRequest = received(&mut rx)[0].decode();
    assert_eq!(traces.resource_spans[0].scope_spans[0].spans.len(), 3);

    let small = self::exporter(Limits { records: 100, bytes: 600 });
    small.configure_app(Some(AppConfig::with_endpoint(&url)));
    small.record(span("tts_request", None, vec![]));
    small.record(span("tts_request", None, vec![("lk.pii.text".into(), "x".repeat(400).into())]));
    assert_eq!(small.diagnostics().dropped, 1, "the byte bound applies too");
}

// Catches: export continuing after the project disabled recording, or the refusal counted as a
// failure.
#[tokio::test]
async fn recording_disabled_stops_that_jobs_export() {
    let refused = std::sync::atomic::AtomicBool::new(false);
    let (url, mut rx) = endpoint(move |path| {
        let first = path.contains("traces") && !refused.swap(true, Ordering::Relaxed);
        Some(if first { (401, DISABLED) } else { (200, b"" as &[u8]) })
    })
    .await;
    let exporter = exporter(Limits::default());
    exporter.configure_cloud(cloud(&url));
    for job in ["AJ_1", "AJ_2"] {
        exporter.open_job(key(job), identity(job));
        exporter.register_job(&key(job), recording());
    }
    exporter.record(span("agent_turn", Some("AJ_1"), vec![]));
    assert!(exporter.flush(LIMIT).await);
    assert_eq!(received(&mut rx).len(), 1);
    exporter.record(span("agent_turn", Some("AJ_1"), vec![]));
    exporter.record(counter(Some("AJ_1"), 1));
    exporter.record(span("agent_turn", Some("AJ_2"), vec![]));
    assert!(exporter.flush(LIMIT).await);
    let requests = received(&mut rx);
    assert_eq!(requests.len(), 1, "only the other job still exports");
    let traces: ExportTraceServiceRequest = requests[0].decode();
    let span = &traces.resource_spans[0].scope_spans[0].spans[0];
    assert_eq!(attr(&span.attributes, "job_id").cloned(), string("AJ_2"));
    let diagnostics = exporter.diagnostics();
    assert_eq!((diagnostics.recording_disabled, diagnostics.failed), (1, 0));
}

// Catches: headers not sent, or a new configuration applied to what the previous one already
// queued.
#[tokio::test]
async fn the_application_endpoint_speaks_protobuf_with_its_headers() {
    let (first_url, mut first_rx) = endpoint(ok).await;
    let (proto_url, mut proto_rx) = endpoint(|_| Some((500, b"" as &[u8]))).await;
    let exporter = exporter(Limits::default());
    let mut first = AppConfig::with_endpoint(&first_url);
    first.headers = vec![("x-api-key".into(), "abc".into())];
    exporter.configure_app(Some(first));
    exporter.record(span("first", None, vec![("n".into(), Value::Int(3))]));
    let mut proto = AppConfig::with_endpoint(&proto_url);
    proto.gzip = true;
    exporter.configure_app(Some(proto));
    exporter.record(span("second", None, vec![]));
    assert!(exporter.flush(LIMIT).await);

    let first = received(&mut first_rx);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].header("content-type"), Some("application/x-protobuf"));
    assert_eq!(first[0].header("content-encoding"), None);
    assert_eq!(first[0].header("x-api-key"), Some("abc"));
    assert_eq!(only_span(&first[0]).name, "first");

    let proto = received(&mut proto_rx);
    assert_eq!(proto[0].header("content-type"), Some("application/x-protobuf"));
    assert_eq!(proto[0].header("content-encoding"), Some("gzip"));
    assert_eq!(only_span(&proto[0]).name, "second");
    assert_eq!(exporter.diagnostics().failed, 1, "the 500 is counted");
}

// Catches: shutdown waiting on an endpoint that never answers, or accepting records after it.
#[tokio::test]
async fn shutdown_is_bounded_by_its_deadline() {
    let (url, mut rx) = endpoint(|_| None).await;
    let exporter = exporter(Limits::default());
    exporter.configure_app(Some(AppConfig::with_endpoint(&url)));
    exporter.record(span("agent_turn", None, vec![]));
    exporter.record(counter(None, 1));
    let started = std::time::Instant::now();
    exporter.shutdown(Duration::from_millis(300)).await;
    assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
    assert_eq!(received(&mut rx).len(), 1, "the hung request; metrics abandoned");
    let diagnostics = exporter.diagnostics();
    assert_eq!(diagnostics.incomplete_cleanup, 1);
    assert_eq!(diagnostics.failed + diagnostics.dropped, 2);
    exporter.record(span("late", None, vec![]));
    exporter.shutdown(LIMIT).await;
    assert!(received(&mut rx).is_empty());
}

#[test]
fn a_protobuf_round_trip_keeps_values() {
    let request = super::encode::traces_request(
        super::encode::app_resource(&process(), &[]),
        &[
            match span(
                "s",
                None,
                vec![("n".into(), Value::Int(3)), ("b".into(), Value::Bytes(vec![1]))],
            ) {
                Record::Span(s) => s,
                _ => unreachable!(),
            },
        ],
    );
    let decoded = ExportTraceServiceRequest::decode(request.encode_to_vec().as_slice()).unwrap();
    let a = &decoded.resource_spans[0].scope_spans[0].spans[0].attributes;
    assert_eq!(attr(a, "n"), Some(&Any::IntValue(3)));
    assert_eq!(attr(a, "b"), Some(&Any::BytesValue(vec![1])));
}
