use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::productstats::{
    EventLoopStalls, FailureType, Operation, ProviderFailures, ProviderLabels, ProviderRequests,
    ProviderTtft,
};
use lk_agents_abi::telemetry::record::RecordSink as _;
use lk_agents_abi::telemetry::runner::{Launch, Runner, RunnerKind};
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
        plugins: Vec::new(),
        runner: Runner { launch: Launch::Start, kind: RunnerKind::Production },
        ..super::resource("2.0.0", Some("3.12.4"))
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

fn llm() -> ProviderLabels {
    lk_agents_telemetry::productstats::provider(Operation::Llm, "openai", "gpt-4o")
}

fn record<M: MetricDecl<Audience = decl::ProductStats>>(
    stats: &ProductStats,
    labels: &M,
    n: impl Into<lk_agents_abi::telemetry::record::Number>,
) {
    let mut m = measurement(labels, 0);
    m.value = n.into();
    stats.record(m);
}

fn string(value: &Option<super::super::otlp::proto::common::v1::AnyValue>) -> &str {
    match value.as_ref().and_then(|v| v.value.as_ref()) {
        Some(Any::StringValue(s)) => s,
        other => panic!("{other:?}"),
    }
}

fn keys(attributes: &[KeyValue]) -> Vec<&str> {
    attributes.iter().map(|kv| kv.key.as_str()).collect()
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
// job's identity, GenAI content), cumulative temporality, or a request off the collector's route
// or without the observability bearer.
#[tokio::test]
async fn an_export_carries_only_declared_product_statistics_keys() {
    let (url, mut rx) = endpoint(ok).await;
    let stats = stats();
    plugin("livekit.plugins.openai", "1.2.3");
    plugin("my_company.secret_plugin", "0.1");
    stats.configure(Some(destination(&url)));
    record(&stats, &ProviderRequests { provider: llm() }, 2u64);
    record(
        &stats,
        &ProviderFailures { provider: llm(), error_type: FailureType::ApiStatusError },
        1u64,
    );
    record(&stats, &ProviderTtft { provider: llm() }, 0.3);
    let stack = [lk_agents_abi::telemetry::record::StackFrame {
        function: "requests_get".into(),
        file: "/home/someone/app.py".into(),
        line: 9,
    }];
    let signature = lk_agents_telemetry::productstats::stall_signature(Some(&stack));
    record(&stats, &EventLoopStalls { signature }, 1u64);
    assert!(stats.flush(LIMIT).await);
    let requests = received(&mut rx);
    assert_eq!(requests.len(), 1);
    let request = decode(&requests[0]);
    let resource = request.resource_metrics[0].resource.as_ref().unwrap();
    let mut resource_keys = keys(&resource.attributes);
    resource_keys.sort();
    assert_eq!(
        resource_keys,
        [
            "lk.productstats.arch",
            "lk.productstats.framework.version",
            "lk.productstats.os",
            "lk.productstats.plugins",
            "lk.productstats.python.version",
            "lk.productstats.runner.kind",
            "lk.productstats.runner.launch",
            "service.name",
        ]
    );
    let plugins = resource.attributes.iter().find(|kv| kv.key == "lk.productstats.plugins");
    let plugins = match plugins.and_then(|kv| kv.value.as_ref()?.value.as_ref()) {
        Some(Any::ArrayValue(a)) => a.values.iter().map(|v| string(&Some(v.clone())).to_owned()),
        other => panic!("{other:?}"),
    };
    assert_eq!(plugins.collect::<Vec<_>>(), ["livekit.plugins.openai==1.2.3"]);
    let values = values(&request);
    for (name, labels, _) in &values {
        assert!(name.starts_with("lk.productstats."), "{name}");
        for (key, _) in labels {
            assert!(key.starts_with("lk.productstats.") && !key.starts_with("lk.pii."), "{key}");
        }
    }
    let provider = |extra: &[(&str, &str)]| {
        let mut labels: Vec<(String, String)> = [
            ("lk.productstats.operation", "llm"),
            ("lk.productstats.provider", "openai"),
            ("lk.productstats.model", "gpt-4o"),
        ]
        .iter()
        .chain(extra)
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
        labels.sort();
        labels
    };
    let mut sorted: Vec<_> = values
        .into_iter()
        .map(|(n, mut l, v)| {
            l.sort();
            (n, l, v)
        })
        .collect();
    sorted.sort();
    assert_eq!(
        sorted,
        [
            (
                "lk.productstats.event_loop.stalls".to_owned(),
                vec![("lk.productstats.stall.signature".to_owned(), "requests_get".to_owned())],
                1
            ),
            (
                "lk.productstats.provider.failures".to_owned(),
                provider(&[("lk.productstats.error.type", "api_status_error")]),
                1
            ),
            ("lk.productstats.provider.requests".to_owned(), provider(&[]), 2),
            ("lk.productstats.provider.ttft".to_owned(), provider(&[]), 1),
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
    record(&stats, &ProviderRequests { provider: llm() }, 1u64);
    assert!(stats.flush(LIMIT).await);
    stats.configure(None);
    record(&stats, &ProviderRequests { provider: llm() }, 1u64);
    assert!(stats.flush(LIMIT).await);
    stats.configure(Some(destination(&url)));
    assert!(stats.flush(LIMIT).await);
    assert!(received(&mut rx).is_empty());
}

// Catches: a window lost silently while no credential exists, more than one window held, or a
// refused export's points not reported on the next export.
#[tokio::test]
async fn drops_are_counted_and_exported_next() {
    let refuse = |_: &str| Some((503, &b"busy"[..]));
    let (refusing, mut refused_rx) = endpoint(refuse).await;
    let (url, mut rx) = endpoint(ok).await;
    let stats = stats();
    for _ in 0..2 {
        record(&stats, &ProviderRequests { provider: llm() }, 1u64);
        assert!(stats.flush(LIMIT).await, "a window waits for a credential");
    }
    stats.configure(Some(destination(&refusing)));
    assert!(stats.flush(LIMIT).await);
    assert_eq!(received(&mut refused_rx).len(), 1);
    stats.configure(Some(destination(&url)));
    assert!(stats.flush(LIMIT).await);
    let requests = received(&mut rx);
    assert_eq!(requests.len(), 1);
    let dropped: Vec<_> = values(&decode(&requests[0]))
        .into_iter()
        .filter(|(name, ..)| name == ExportDropped::NAME)
        .map(|(_, labels, v)| (labels[0].1.clone(), v))
        .collect();
    // The first window was replaced while waiting; the second (with that drop) was refused.
    assert_eq!(dropped, [("refused".to_owned(), 2)]);
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
        record(&stats, &ProviderRequests { provider: llm() }, 1u64);
        customer.flush(None, LIMIT).await;
        assert!(stats.flush(LIMIT).await);
    }
    assert!(customer.job_disabled(&key("AJ_1")));
    let paths: Vec<String> = received(&mut rx).into_iter().map(|r| r.path).collect();
    assert_eq!(paths.iter().filter(|p| *p == PATH).count(), 2, "{paths:?}");
}
