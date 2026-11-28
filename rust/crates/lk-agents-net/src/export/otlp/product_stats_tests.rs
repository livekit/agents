use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::productstats::runner::{Runner, RunnerCmd, RunnerKind};
use lk_agents_abi::productstats::testkit::{TestCounts, TestDuration};
use lk_agents_abi::telemetry::decl::{self, MetricDecl};
use lk_agents_abi::telemetry::record::{Number, RecordSink as _};

use super::super::proto::collector::metrics::v1::ExportMetricsServiceRequest;
use super::super::proto::common::v1::{KeyValue, any_value::Value as Any};
use super::super::proto::metrics::v1::{AggregationTemporality, metric::Data};
use super::super::tests::{
    DISABLED, Received, cloud, counter, endpoint, identity, key, ok, process, received, recording,
};
use super::super::{CloudRecording, Exporter, Limits};
use super::*;

const LIMIT: Duration = Duration::from_secs(5);

/// An exporter whose product statistics are on (`on`) or opted out of.
fn exporter(on: bool) -> Exporter {
    let resource = ProductStatsResource {
        runner: Runner { cmd: RunnerCmd::Start, kind: RunnerKind::Production },
        ..super::resource("2.0.0", RuntimeName::Python, "3.12.4", RunnerCmd::Start)
    };
    Exporter::start(
        &crate::task::TokioSpawner::current(),
        Arc::new(crate::clock::RealClock::new()),
        process(),
        Limits::default(),
        on.then_some(resource),
    )
}

/// The test's collector, as a Cloud project's.
fn collector(url: &str) -> Collector {
    Collector { base_url: url.to_owned(), api_key: "key".into(), api_secret: "secret".into() }
}

/// Points the exporter's product statistics at `collector`; `None` is off LiveKit Cloud.
fn configure(exporter: &Exporter, collector: Option<Collector>) {
    let mut state = exporter.shared.state.lock();
    state.product_stats.as_mut().expect("product statistics are on").configure(collector);
}

fn counts(label: &str) -> TestCounts {
    TestCounts { label: Label(label.to_owned()) }
}

/// One measurement, as its declared audience routes it.
fn measure<M: MetricDecl>(exporter: &Exporter, labels: &M, value: impl Into<Number>) {
    exporter.record(decl::record_of(labels, value.into(), None));
}

/// The requests the collector's route received, each decoded and checked for its bearer.
fn collected(requests: &[Received]) -> Vec<ExportMetricsServiceRequest> {
    requests.iter().filter(|r| r.path == PATH).map(decode).collect()
}

fn decode(request: &Received) -> ExportMetricsServiceRequest {
    assert_eq!(request.path, PATH);
    assert_eq!(request.header("content-encoding"), Some("gzip"));
    let token = request.header("authorization").unwrap().strip_prefix("Bearer ").unwrap();
    let claims = livekit_token::TokenVerifier::with_api_key("key", "secret").verify(token).unwrap();
    assert!(claims.observability.write);
    request.decode()
}

fn string(value: &Option<super::super::proto::common::v1::AnyValue>) -> &str {
    match value.as_ref().and_then(|v| v.value.as_ref()) {
        Some(Any::StringValue(s)) => s,
        other => panic!("{other:?}"),
    }
}

/// One data point: its instrument, its labels, and its value (whole; a histogram's count).
type Point = (String, Vec<(String, String)>, i64);

/// Every data point of the request.
fn values(request: &ExportMetricsServiceRequest) -> Vec<Point> {
    let metrics = &request.resource_metrics[0].scope_metrics[0].metrics;
    let mut out = Vec::new();
    for metric in metrics {
        let points: Vec<(Vec<KeyValue>, i64)> = match &metric.data {
            Some(Data::Sum(sum)) => {
                assert_eq!(sum.aggregation_temporality, AggregationTemporality::Delta as i32);
                sum.data_points
                    .iter()
                    .map(|p| {
                        use super::super::proto::metrics::v1::number_data_point::Value;
                        let v = match p.value {
                            Some(Value::AsInt(v)) => v,
                            Some(Value::AsDouble(v)) if v.fract() == 0.0 => v as i64,
                            other => panic!("{other:?}"),
                        };
                        (p.attributes.clone(), v)
                    })
                    .collect()
            }
            Some(Data::Histogram(h)) => {
                assert_eq!(h.aggregation_temporality, AggregationTemporality::Delta as i32);
                h.data_points.iter().map(|p| (p.attributes.clone(), p.count as i64)).collect()
            }
            other => panic!("{other:?}"),
        };
        for (attributes, v) in points {
            let labels = attributes
                .iter()
                .map(|kv| (kv.key.clone(), string(&kv.value).to_owned()))
                .collect();
            out.push((metric.name.clone(), labels, v));
        }
    }
    out
}

// Catches: a product statistics export carrying a key it does not declare (an `lk.pii.*` key, a
// job's identity, GenAI content) or the customer's resource instead of its own, a resource key
// missing, renamed or left at its default (the storage the process opened), cumulative
// temporality, a window sent twice, or a request off the collector's route or without the
// observability bearer.
#[tokio::test]
async fn an_export_carries_its_own_resource_and_only_declared_product_statistics_keys() {
    let (url, mut rx) = endpoint(ok).await;
    let stats = exporter(true);
    configure(&stats, Some(collector(&url)));
    stats.storage_opened(ThreadStorage::AgentDb);
    measure(&stats, &counts("a"), 2u64);
    measure(&stats, &counts("b"), 1u64);
    measure(&stats, &TestDuration, 0.3);
    assert!(stats.flush(None, LIMIT).await);
    let requests = received(&mut rx);
    assert_eq!(requests.len(), 1);
    let request = decode(&requests[0]);
    let resource = request.resource_metrics[0].resource.as_ref().unwrap();
    let mut resource: Vec<(&str, &str)> =
        resource.attributes.iter().map(|kv| (kv.key.as_str(), string(&kv.value))).collect();
    resource.sort();
    assert_eq!(
        resource,
        [
            ("lk.product.agents.arch", std::env::consts::ARCH),
            ("lk.product.agents.framework.version", "2.0.0"),
            ("lk.product.agents.os", std::env::consts::OS),
            ("lk.product.agents.runner.cmd", "start"),
            ("lk.product.agents.runner.kind", "production"),
            ("lk.product.agents.runtime.name", "python"),
            ("lk.product.agents.runtime.version", "3.12.4"),
            ("lk.product.agents.storage", "agent_db"),
            ("service.name", "livekit-agents"),
        ]
    );
    let label = |v: &str| vec![("lk.product.agents.test.label".to_owned(), v.to_owned())];
    let mut sorted = values(&request);
    sorted.sort();
    assert_eq!(
        sorted,
        [
            ("lk.product.agents.test.counts".to_owned(), label("a"), 2),
            ("lk.product.agents.test.counts".to_owned(), label("b"), 1),
            ("lk.product.agents.test.duration".to_owned(), vec![], 1),
        ]
    );
    assert!(stats.flush(None, LIMIT).await);
    assert!(received(&mut rx).is_empty(), "a delta window is sent once");
}

// Catches: a model statistic exported under another name, without one of its labels, as the
// wrong instrument (a latency summed, a count bucketed) or with its unit lost.
#[tokio::test]
async fn model_statistics_reach_the_collector_as_declared() {
    use lk_agents_abi::productstats::models::{
        EndpointedBy, ModelAudioDuration, ModelKind, ModelLabels, ModelRequests,
        ModelTranscriptionDelay,
    };
    let (url, mut rx) = endpoint(ok).await;
    let stats = exporter(true);
    configure(&stats, Some(collector(&url)));
    let id = lk_agents_abi::ProviderId {
        label: "livekit.plugins.deepgram.stt.STT".into(),
        model: "nova-3".into(),
        provider: "deepgram".into(),
    };
    let model = ModelLabels::of(ModelKind::Stt, &id);
    measure(&stats, &ModelRequests { model: model.clone() }, 1u64);
    measure(&stats, &ModelAudioDuration { model: model.clone() }, 2.0);
    let delay = ModelTranscriptionDelay { model, endpointing: EndpointedBy::Vad };
    measure(&stats, &delay, 0.3);
    assert!(stats.flush(None, LIMIT).await);
    let request = decode(&received(&mut rx)[0]);
    let metrics = &request.resource_metrics[0].scope_metrics[0].metrics;
    let mut shapes: Vec<(&str, &str, bool)> = metrics
        .iter()
        .map(|m| (m.name.as_str(), m.unit.as_str(), matches!(m.data, Some(Data::Histogram(_)))))
        .collect();
    shapes.sort();
    assert_eq!(
        shapes,
        [
            ("lk.product.agents.model.audio_duration", "s", false),
            ("lk.product.agents.model.requests", "{request}", false),
            ("lk.product.agents.model.transcription_delay", "s", true),
        ]
    );
    let labels = |more: &[(&str, &str)]| {
        let model = [
            ("lk.product.agents.model.kind", "stt"),
            ("lk.product.agents.model.plugin", "deepgram"),
            ("lk.product.agents.model.provider", "deepgram"),
            ("lk.product.agents.model.name", "nova-3"),
        ];
        let all = model.iter().chain(more).map(|(k, v)| ((*k).to_owned(), (*v).to_owned()));
        all.collect::<Vec<_>>()
    };
    let endpointing = [("lk.product.agents.model.transcription_delay.endpointing", "vad")];
    let mut sorted = values(&request);
    sorted.sort();
    assert_eq!(
        sorted,
        [
            ("lk.product.agents.model.audio_duration".to_owned(), labels(&[]), 2),
            ("lk.product.agents.model.requests".to_owned(), labels(&[]), 1),
            ("lk.product.agents.model.transcription_delay".to_owned(), labels(&endpointing), 1),
        ]
    );
}

// Catches: product statistics sent to a self-hosted server, an `.example.com` host taken for
// Cloud by a substring, a non-Cloud URL that still sends what was held, or a process that opted
// out sending anything.
#[tokio::test]
async fn only_livekit_cloud_receives_product_statistics_and_an_opted_out_process_sends_none() {
    assert_eq!(
        Collector::cloud("wss://p.livekit.cloud", "key", "secret"),
        Some(collector("https://p.livekit.cloud"))
    );
    for url in ["ws://localhost:7880", "wss://sfu.example.com", "wss://x.com/p.livekit.cloud"] {
        assert_eq!(Collector::cloud(url, "key", "secret"), None, "{url}");
    }
    assert_eq!(Collector::cloud("wss://p.livekit.cloud", "", ""), None);
    let (url, mut rx) = endpoint(ok).await;
    let stats = exporter(true);
    measure(&stats, &counts("a"), 1u64);
    assert!(stats.flush(None, LIMIT).await);
    stats.configure_product_stats("wss://sfu.example.com", "key", "secret");
    measure(&stats, &counts("a"), 1u64);
    assert!(stats.flush(None, LIMIT).await);
    configure(&stats, Some(collector(&url)));
    assert!(stats.flush(None, LIMIT).await);
    assert!(received(&mut rx).is_empty());

    // `LIVEKIT_PRODUCT_STATS=0` or `DO_NOT_TRACK=1`: the exporter is built without them.
    let opted_out = exporter(false);
    opted_out.configure_product_stats("wss://p.livekit.cloud", "key", "secret");
    measure(&opted_out, &counts("a"), 1u64);
    assert!(opted_out.flush(None, LIMIT).await);
    assert_eq!(opted_out.runner_cmd(), RunnerCmd::NoCommand);
    assert!(received(&mut rx).is_empty());
}

// Catches: a process with nothing recorded posting an empty export (its resource alone), or the
// window recorded before the credential arrived never sent, or more than one window held.
#[tokio::test]
async fn an_idle_process_sends_nothing_and_a_waiting_window_is_sent_once_configured() {
    let (url, mut rx) = endpoint(ok).await;
    let stats = exporter(true);
    configure(&stats, Some(collector(&url)));
    assert!(stats.flush(None, LIMIT).await);
    stats.shutdown(LIMIT).await;
    assert!(received(&mut rx).is_empty(), "no data point, no request");

    let stats = exporter(true);
    for label in ["replaced", "held"] {
        measure(&stats, &counts(label), 1u64);
        assert!(stats.flush(None, LIMIT).await, "a window waits for a credential");
    }
    configure(&stats, Some(collector(&url)));
    assert!(stats.flush(None, LIMIT).await);
    let requests = received(&mut rx);
    assert_eq!(requests.len(), 1);
    let labels: Vec<_> = values(&decode(&requests[0])).into_iter().map(|(_, l, _)| l).collect();
    assert_eq!(labels, [[("lk.product.agents.test.label".to_owned(), "held".to_owned())]]);
}

// Catches: the two products, sharing one exporter, sharing a fate: the customer's observability
// refusal (its project's recording disabled, its job's export stopped by the 401) stopping
// product statistics, or the collector's refusal disabling the customer's export.
#[tokio::test]
async fn neither_products_refusal_stops_the_other() {
    for refused in ["customer", "collector"] {
        let reply = move |path: &str| {
            let customer = path.starts_with("/observability/") && path != PATH;
            let refuse = if refused == "customer" { customer } else { path == PATH };
            Some(if refuse { (401, DISABLED) } else { (200, &b""[..]) })
        };
        let (url, mut rx) = endpoint(reply).await;
        let exporter = exporter(true);
        exporter.configure_cloud(cloud(&url));
        configure(&exporter, Some(collector(&url)));
        exporter.open_job(key("AJ_1"), identity("AJ_1"));
        exporter.register_job(&key("AJ_1"), CloudRecording { ..recording() });
        for _ in 0..2 {
            exporter.record(counter(Some("AJ_1"), 1));
            measure(&exporter, &counts("a"), 1u64);
            assert!(exporter.flush(None, LIMIT).await);
        }
        assert_eq!(exporter.job_disabled(&key("AJ_1")), refused == "customer");
        let paths: Vec<String> = received(&mut rx).into_iter().map(|r| r.path).collect();
        let (stats, customer): (Vec<_>, Vec<_>) = paths.iter().partition(|p| *p == PATH);
        assert_eq!(stats.len(), 2, "{refused}: {paths:?}");
        assert_eq!(customer.len(), if refused == "customer" { 1 } else { 2 }, "{paths:?}");
    }
}

// Catches: a product statistic reaching a customer destination (their Cloud project's queue, an
// undecided job's hold, their application endpoint), a customer measurement reaching LiveKit's
// collector, or a product statistic waiting on a job's recording decision or dropped with a job
// that records nothing.
#[tokio::test]
async fn a_product_statistic_reaches_its_destination_only_and_no_job_gates_it() {
    use super::super::config::{AppConfig, Target};
    let (url, mut rx) = endpoint(ok).await;
    let exporter = exporter(true);
    exporter.configure_cloud(cloud(&url));
    configure(&exporter, Some(collector(&url)));
    let target = Target { url: format!("{url}/app/v1/metrics"), headers: Vec::new(), gzip: false };
    exporter.configure_app(Some(AppConfig { metrics: Some(target), ..AppConfig::default() }));
    // One job records nothing, another is still undecided.
    let nothing = CloudRecording { traces: false, logs: false, metrics: false, ..recording() };
    exporter.open_job(key("AJ_off"), identity("AJ_off"));
    exporter.register_job(&key("AJ_off"), nothing);
    exporter.open_job(key("AJ_undecided"), identity("AJ_undecided"));
    measure(&exporter, &counts("a"), 1u64);
    assert!(exporter.flush(None, LIMIT).await);
    let requests = received(&mut rx);
    let paths: Vec<&str> = requests.iter().map(|r| r.path.as_str()).collect();
    assert_eq!(paths, [PATH], "a product statistic went elsewhere, or nowhere");

    // A customer's measurement goes to the customer's destinations and not to the collector.
    exporter.register_job(&key("AJ_undecided"), recording());
    exporter.record(counter(Some("AJ_undecided"), 1));
    assert!(exporter.flush(None, LIMIT).await);
    let requests = received(&mut rx);
    assert!(!requests.is_empty());
    assert!(collected(&requests).is_empty(), "a customer measurement reached the collector");
}

// Catches: product statistics riding the exporter's one-second span tick or its 30 s metrics
// tick instead of their own minute.
#[tokio::test]
async fn a_window_leaves_on_its_own_minute_only() {
    use super::super::destination::{Batch, Take};
    assert_eq!(INTERVAL, Duration::from_secs(60));
    let stats = exporter(true);
    configure(&stats, Some(collector("https://p.livekit.cloud")));
    measure(&stats, &counts("a"), 1u64);
    let windows = |product_stats: bool| {
        let take = Take { job: None, metrics: true, product_stats, all: true };
        let batches = stats.shared.state.lock().take(take, 1);
        batches.iter().filter(|batch| matches!(batch, Batch::ProductStats(_))).count()
    };
    assert_eq!(windows(false), 0, "a span or metrics tick took the window");
    assert_eq!(windows(true), 1);
}

// Catches: a statistic declared that the documented list does not have, or a documented one not
// declared: the registry generated from the declarations names exactly the documented keys.
#[test]
fn the_declared_keys_are_exactly_the_documented_list() {
    const DOCUMENTED: &[&str] = &[
        "arch",
        "feature.agent_as_tool",
        "feature.background_audio",
        "feature.handoff",
        "feature.language_switch",
        "feature.mcp",
        "feature.model_switch",
        "feature.model_switch.from.kind",
        "feature.model_switch.to.kind",
        "feature.override_options",
        "feature.remote_agent",
        "feature.update_options",
        "framework.version",
        "model.audio_duration",
        "model.characters",
        "model.input_tokens",
        "model.kind",
        "model.name",
        "model.output_tokens",
        "model.plugin",
        "model.provider",
        "model.requests",
        "model.transcription_delay",
        "model.transcription_delay.endpointing",
        "model.ttfb",
        "model.ttft",
        "os",
        "runner.cmd",
        "runner.kind",
        "runtime.name",
        "runtime.version",
        "session.entrypoint",
        "sessions",
        "storage",
        "voice.false_interruption",
        "voice.interruption",
        "voice.pipeline",
        "voice.sessions",
        "voice.turn_detection",
    ];
    let registry = [
        include_str!("../../../../../otel/registry/attributes.yaml"),
        include_str!("../../../../../otel/registry/metrics.yaml"),
    ];
    let mut declared = std::collections::BTreeSet::new();
    for file in registry {
        for (at, _) in file.match_indices("lk.product.agents.") {
            let key = &file[at + "lk.product.agents.".len()..];
            let end = key.find(|c: char| !(c.is_ascii_lowercase() || c == '.' || c == '_'));
            declared.insert(&key[..end.unwrap_or(key.len())]);
        }
    }
    assert_eq!(declared.into_iter().collect::<Vec<_>>(), DOCUMENTED);
}
