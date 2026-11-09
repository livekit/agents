use crate::voice::tests::traced;
use lk_agents_abi::telemetry::decl::Value;
use lk_agents_abi::telemetry::record::{SpanContext, SpanRecord};

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
    let turn = SpanContext { trace_id: 3, span_id: 9, sampled: true };
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
    assert!(!json.contains("\"ttft\""), "an unmeasured ttft is omitted: {json}");
    let late = spans.iter().find(|s| s.name == "realtime_metrics").expect("{spans:?}");
    assert_eq!(late.parent, Some(inferences[1].context.span_id));
    assert!(blob(late));
}
