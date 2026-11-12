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

/// A realtime response's token usage lands once, on its own `realtime_inference` span. https://github.com/livekit/agents/issues/7304 https://github.com/livekit/agents/issues/3087
#[tokio::test(start_paused = true)]
async fn realtime_inference_span_carries_its_usage() {
    use crate::voice::generation::realtime_generation::RealtimeSpans;
    use lk_agents_abi::telemetry::semconv::otel;
    let (runtime, records) = traced("realtime").await;
    let spans = RealtimeSpans::default();
    spans.set_model(Some(lk_agents_abi::ProviderId {
        label: "rt".into(),
        model: "gpt-realtime".into(),
        provider: "openai".into(),
    }));
    let turn = SpanContext { trace_id: 3, span_id: 9, sampled: true };
    let open = spans.start(&runtime.shared, Some(turn)).expect("a model");
    let mut m = lk_agents_abi::metrics::testkit::realtime();
    m.request_id = "resp_1".into();
    m.tokens.input_tokens = 120;
    m.tokens.output_tokens = 45;
    m.tokens.input_audio_tokens = Some(100);
    m.tokens.output_audio_tokens = Some(40);
    spans.metrics(&runtime.shared.trace, &m);
    spans.end(open, Some("resp_1"));

    let spans = records.spans();
    let inference = spans.iter().find(|s| s.name == "realtime_inference").expect("{spans:?}");
    let attr = |key: &str| inference.attributes.iter().find(|(k, _)| *k == key).map(|(_, v)| v);
    assert_eq!(attr(otel::GEN_AI_USAGE_INPUT_TOKENS), Some(&Value::Int(120)), "{inference:?}");
    assert_eq!(attr(otel::GEN_AI_USAGE_OUTPUT_TOKENS), Some(&Value::Int(45)));
    assert_eq!(attr(otel::GEN_AI_REQUEST_MODEL), Some(&Value::String("gpt-realtime".into())));
    assert_eq!(inference.parent, Some(turn.span_id), "under the speech's turn");
    let usage_spans = spans
        .iter()
        .filter(|s| s.attributes.iter().any(|(k, _)| *k == otel::GEN_AI_USAGE_INPUT_TOKENS))
        .count();
    assert_eq!(usage_spans, 1, "usage recorded once: {spans:?}");
}
