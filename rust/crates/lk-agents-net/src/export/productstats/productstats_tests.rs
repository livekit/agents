use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::productstats::runner::{Runner, RunnerCmd, RunnerKind};
use lk_agents_abi::productstats::testkit::{TestCounts, TestDuration};
use lk_agents_abi::telemetry::decl::{self, MetricDecl};
use lk_agents_abi::telemetry::record::{Number, RecordSink as _};
use lk_agents_telemetry::productstats::Sink as _;

use super::super::otlp::proto::collector::metrics::v1::ExportMetricsServiceRequest;
use super::super::otlp::proto::common::v1::{KeyValue, any_value::Value as Any};
use super::super::otlp::proto::metrics::v1::{AggregationTemporality, metric::Data};
use super::super::otlp::tests::{
    DISABLED, Received, cloud, counter, endpoint, exporter, identity, key, ok, received, recording,
};
use super::super::otlp::{CloudRecording, Limits};
use super::*;

const LIMIT: Duration = Duration::from_secs(5);

fn stats() -> ProductStats {
    let resource = ProductStatsResource {
        runner: Runner { cmd: RunnerCmd::Start, kind: RunnerKind::Production },
        ..super::resource("2.0.0", RuntimeName::Python, "3.12.4")
    };
    ProductStats::start(
        &crate::task::TokioSpawner::current(),
        Arc::new(crate::clock::RealClock::new()),
        resource,
    )
}

fn destination(url: &str) -> Destination {
    Destination { base_url: url.to_owned(), api_key: "key".into(), api_secret: "secret".into() }
}

fn counts(label: &str) -> TestCounts {
    TestCounts { label: Label(label.to_owned()) }
}

fn measure<M: MetricDecl<Audience = decl::ProductStats>>(
    stats: &ProductStats,
    labels: &M,
    value: impl Into<Number>,
) {
    let mut out = KeyValues::new();
    labels.write(&mut out);
    stats.record(Measurement {
        instrument: M::NAME,
        kind: M::KIND,
        unit: M::UNIT,
        bounds: M::BOUNDS,
        value: value.into(),
        labels: out,
        job: None,
    });
}

fn string(value: &Option<super::super::otlp::proto::common::v1::AnyValue>) -> &str {
    match value.as_ref().and_then(|v| v.value.as_ref()) {
        Some(Any::StringValue(s)) => s,
        other => panic!("{other:?}"),
    }
}

fn decode(request: &Received) -> ExportMetricsServiceRequest {
    assert_eq!(request.path, PATH);
    assert_eq!(request.header("content-encoding"), Some("gzip"));
    let token = request.header("authorization").unwrap().strip_prefix("Bearer ").unwrap();
    let claims = livekit_token::TokenVerifier::with_api_key("key", "secret").verify(token).unwrap();
    assert!(claims.observability.write);
    request.decode()
}

/// One data point: its instrument, its labels, and its value (a histogram's count).
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
                        let v = match p.value {
                            Some(super::super::otlp::proto::metrics::v1::number_data_point::Value::AsInt(v)) => v,
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
// job's identity, GenAI content), a resource key missing, renamed or left at its default (the
// storage the process opened after the installation), cumulative temporality, or a request off
// the collector's route or without the observability bearer.
#[tokio::test]
async fn an_export_carries_only_declared_product_statistics_keys() {
    let (url, mut rx) = endpoint(ok).await;
    let stats = stats();
    stats.configure(Some(destination(&url)));
    thread_storage(ThreadStorage::AgentDb);
    measure(&stats, &counts("a"), 2u64);
    measure(&stats, &counts("b"), 1u64);
    measure(&stats, &TestDuration, 0.3);
    assert!(stats.flush(LIMIT).await);
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
    let values = values(&request);
    for (name, labels, _) in &values {
        assert!(name.starts_with("lk.product.agents."), "{name}");
        for (key, _) in labels {
            assert!(key.starts_with("lk.product.agents.") && !key.starts_with("lk.pii."), "{key}");
        }
    }
    let label = |v: &str| vec![("lk.product.agents.test.label".to_owned(), v.to_owned())];
    let mut sorted = values;
    sorted.sort();
    assert_eq!(
        sorted,
        [
            ("lk.product.agents.test.counts".to_owned(), label("a"), 2),
            ("lk.product.agents.test.counts".to_owned(), label("b"), 1),
            ("lk.product.agents.test.duration".to_owned(), vec![], 1),
        ]
    );
    assert!(stats.flush(LIMIT).await);
    assert!(received(&mut rx).is_empty(), "a delta window is sent once");
}

// Catches: product statistics sent to a self-hosted server, an `.example.com` host taken for
// Cloud by a substring, or a non-Cloud URL that still sends what was held.
#[tokio::test]
async fn only_livekit_cloud_receives_product_statistics() {
    assert_eq!(
        Destination::cloud("wss://p.livekit.cloud", "key", "secret"),
        Some(destination("https://p.livekit.cloud"))
    );
    for url in ["ws://localhost:7880", "wss://sfu.example.com", "wss://x.com/p.livekit.cloud"] {
        assert_eq!(Destination::cloud(url, "key", "secret"), None, "{url}");
    }
    assert_eq!(Destination::cloud("wss://p.livekit.cloud", "", ""), None);
    let (url, mut rx) = endpoint(ok).await;
    let stats = stats();
    measure(&stats, &counts("a"), 1u64);
    assert!(stats.flush(LIMIT).await);
    stats.configure(None);
    measure(&stats, &counts("a"), 1u64);
    assert!(stats.flush(LIMIT).await);
    stats.configure(Some(destination(&url)));
    assert!(stats.flush(LIMIT).await);
    assert!(received(&mut rx).is_empty());
}

// Catches: a process with nothing recorded posting an empty export (its resource alone), or the
// window recorded before the credential arrived never sent, or more than one window held.
#[tokio::test]
async fn an_idle_process_sends_nothing_and_a_waiting_window_is_sent_once_configured() {
    let (url, mut rx) = endpoint(ok).await;
    let stats = stats();
    stats.configure(Some(destination(&url)));
    assert!(stats.flush(LIMIT).await);
    stats.shutdown(LIMIT).await;
    assert!(received(&mut rx).is_empty(), "no data point, no request");

    let stats = self::stats();
    for label in ["replaced", "held"] {
        measure(&stats, &counts(label), 1u64);
        assert!(stats.flush(LIMIT).await, "a window waits for a credential");
    }
    stats.configure(Some(destination(&url)));
    assert!(stats.flush(LIMIT).await);
    let requests = received(&mut rx);
    assert_eq!(requests.len(), 1);
    let labels: Vec<_> = values(&decode(&requests[0])).into_iter().map(|(_, l, _)| l).collect();
    assert_eq!(labels, [[("lk.product.agents.test.label".to_owned(), "held".to_owned())]]);
}

// Catches: the customer's observability refusal (or its project's recording disabled) stopping
// product statistics, which have their own destination and failures.
#[tokio::test]
async fn a_customer_observability_refusal_leaves_product_statistics_on() {
    let reply = |path: &str| {
        Some(if path.starts_with("/observability/") { (401, DISABLED) } else { (200, &b""[..]) })
    };
    let (url, mut rx) = endpoint(reply).await;
    let customer = exporter(Limits::default());
    customer.configure_cloud(cloud(&url));
    customer.open_job(key("AJ_1"), identity("AJ_1"));
    customer.register_job(&key("AJ_1"), CloudRecording { ..recording() });
    let stats = stats();
    stats.configure(Some(destination(&url)));
    for _ in 0..2 {
        customer.record(counter(Some("AJ_1"), 1));
        measure(&stats, &counts("a"), 1u64);
        customer.flush(None, LIMIT).await;
        assert!(stats.flush(LIMIT).await);
    }
    assert!(customer.job_disabled(&key("AJ_1")));
    let paths: Vec<String> = received(&mut rx).into_iter().map(|r| r.path).collect();
    assert_eq!(paths.iter().filter(|p| *p == PATH).count(), 2, "{paths:?}");
}
