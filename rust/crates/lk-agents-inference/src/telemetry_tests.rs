use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::telemetry::decl::Value;
use lk_agents_abi::telemetry::record::{
    Measurement, Record, RecordSink, SpanId, SpanRecord, SpanRef, TraceId,
};
use lk_agents_abi::{
    ChatContext, ChatMessage, ChatRequest, ChoiceDelta, Closed, Llm, LlmChunk, LlmStream, Role,
    StreamOutput, Ts, VirtualClock,
};
use lk_agents_net::telemetry::{Cx, Tracer};
use parking_lot::Mutex;

use super::{FALLBACK_LLM_LABEL, LlmTrace};
use crate::fallback::{ChainPolicy, FallbackLlm};
use crate::testkit::fallback::{boom, parts};

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

    fn measurements(&self) -> Vec<Measurement> {
        let all = self.0.lock();
        let measured = all.iter().filter_map(|r| match r {
            Record::Measurement(m) => Some(m.clone()),
            _ => None,
        });
        measured.collect()
    }
}

fn attr<'a>(span: &'a SpanRecord, key: &str) -> Option<&'a Value> {
    span.attributes.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
}

fn named<'a>(spans: &'a [SpanRecord], name: &str) -> Vec<&'a SpanRecord> {
    spans.iter().filter(|s| s.name == name).collect()
}

const PARENT: SpanRef = SpanRef { trace_id: TraceId(7), span_id: SpanId(8), sampled: true };

fn traced(capture: bool) -> (Arc<Records>, Cx) {
    let (records, cx, _) = traced_by(capture);
    (records, cx)
}

/// [`traced`], and the tracer a fallback adapter records with.
fn traced_by(capture: bool) -> (Arc<Records>, Cx, Tracer) {
    let records = Arc::new(Records::default());
    let tracer = Tracer::new(records.clone(), Arc::new(VirtualClock::new()), 11);
    tracer.set_capture_content(capture);
    let cx = tracer.root().continuing(Some(PARENT));
    (records, cx, tracer)
}

struct Script {
    label: &'static str,
    serves: bool,
}

struct Chunks(Vec<LlmChunk>);

impl StreamOutput<LlmChunk> for Chunks {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move { Ok(self.0.pop()) })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Chunks {}

fn chunk(text: &str) -> LlmChunk {
    let delta = ChoiceDelta { content: Some(text.into()), ..Default::default() };
    LlmChunk { id: "resp_1".into(), delta: Some(delta), ..Default::default() }
}

impl Llm for Script {
    fn label(&self) -> String {
        self.label.into()
    }
    fn model(&self) -> String {
        format!("{}-model", self.label)
    }
    fn chat<'a>(
        &'a self,
        _req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            if !self.serves {
                return Err(boom("down"));
            }
            Ok(Box::new(Chunks(vec![chunk("hi")])) as LlmStream)
        })
    }
}

fn request() -> ChatRequest {
    let at = std::time::SystemTime::UNIX_EPOCH;
    ChatRequest {
        chat_ctx: ChatContext::new(vec![
            ChatMessage::text(
                lk_agents_core::request::INSTRUCTIONS_MESSAGE_ID,
                Role::System,
                "be brief",
                at,
            )
            .into(),
            ChatMessage::text("u", Role::User, "hello", at).into(),
        ]),
        ..ChatRequest::default()
    }
}

// Catches: an attempt not nested in its request, the 1.8.10 metrics blob or the Langfuse first
// chunk missing, the response's finish reason lost, or messages recorded without capture.
#[test]
fn a_request_records_its_attempt_metrics_and_messages() {
    for capture in [true, false] {
        let (records, cx) = traced(capture);
        let llm = Script { label: "openai.LLM", serves: true };
        let mut req = request();
        let mut trace = LlmTrace::start(&cx, &llm, &mut req);
        let attempt = trace.attempt();
        trace.observe(&chunk("hi"));
        trace.end_attempt(attempt, None);
        trace.end(false, Some("SP_1"));

        let spans = records.spans();
        let request = named(&spans, "llm_request")[0];
        let run = named(&spans, "llm_request_run")[0];
        assert_eq!(request.parent, Some(SpanId(8)));
        assert_eq!(run.parent, Some(request.context.span_id));
        assert_eq!(
            attr(run, "lk.provider_request_ids"),
            Some(&Value::Array(vec![Value::String("resp_1".into())]))
        );
        let Some(Value::String(blob)) = attr(request, "lk.llm_metrics") else {
            panic!("{request:?}")
        };
        let blob: serde_json::Value = serde_json::from_str(blob).unwrap();
        assert_eq!(
            (blob["type"].as_str(), blob["speech_id"].as_str()),
            (Some("llm_metrics"), Some("SP_1"))
        );
        let Some(Value::String(start)) =
            attr(request, "langfuse.observation.completion_start_time")
        else {
            panic!("{request:?}")
        };
        assert!(start.starts_with('"') && start.ends_with('"'), "{start}");
        assert_eq!(
            attr(request, "gen_ai.response.finish_reasons"),
            Some(&Value::Array(vec![Value::String("stop".into())]))
        );
        assert_eq!(attr(request, "gen_ai.system_instructions").is_some(), capture);
        assert_eq!(attr(request, "gen_ai.input.messages").is_some(), capture);
        assert_eq!(attr(request, "gen_ai.output.messages").is_some(), capture);
    }
}

// Catches: the GenAI request metrics labelled with LiveKit's model labels, or the reverse.
#[test]
fn request_metrics_carry_the_genai_labels() {
    let (records, cx) = traced(false);
    let llm = Script { label: "openai.LLM", serves: true };
    let mut req = request();
    let trace = LlmTrace::start(&cx, &llm, &mut req);
    trace.end(false, None);
    let measured = records.measurements();
    let duration = measured
        .iter()
        .find(|m| m.instrument == "gen_ai.client.operation.duration")
        .expect("the request's duration");
    let keys: Vec<_> = duration.labels.iter().map(|(k, _)| k.to_string()).collect();
    assert!(keys.contains(&"gen_ai.request.model".to_owned()), "{keys:?}");
    assert!(!keys.iter().any(|k| k.starts_with("model_")), "{keys:?}");
}

/// A session's registered `llm_node` over `Script`.
struct Node(Script);

impl Llm for Node {
    fn label(&self) -> String {
        self.0.label()
    }
    fn is_node(&self) -> bool {
        true
    }
    fn chat<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        self.0.chat(req)
    }
}

// Catches: a request span opened around a registered `llm_node` (the node's own output recorded
// as a provider request, or a delegating node's request doubled), or the node not handed the
// context its default request opens under.
#[test]
fn a_registered_node_makes_its_own_request() {
    let (records, cx) = traced(true);
    let mut req = request();
    let mut trace =
        LlmTrace::start(&cx, &Node(Script { label: "openai.LLM", serves: true }), &mut req);
    let attempt = trace.attempt();
    assert!(attempt.is_none());
    trace.observe(&chunk("hi"));
    trace.end_attempt(attempt, None);
    assert_eq!(trace.response().0, "hi", "the node span still reads the response");
    trace.end(false, None);
    assert!(records.spans().is_empty() && records.measurements().is_empty());
    assert_eq!(req.trace.and_then(|t| t.span), Some(PARENT));
}

// Catches: a fallback adapter's caller opening an `llm_request` of its own, the adapter span not
// parenting each entry's request, or the served entry not recorded on it.
#[tokio::test]
async fn a_fallback_adapter_records_one_request_per_entry_under_its_span() {
    let (records, cx, tracer) = traced_by(false);
    let (clock, spawner, obs) = parts();
    let entries: Vec<Arc<dyn Llm>> = vec![
        Arc::new(Script { label: "primary", serves: false }),
        Arc::new(Script { label: "backup", serves: true }),
    ];
    let chain = FallbackLlm::new(entries, ChainPolicy::llm_defaults(), clock, spawner, obs, tracer)
        .expect("entries");
    assert_eq!(chain.label(), FALLBACK_LLM_LABEL);
    let mut req = request();
    let trace = LlmTrace::start(&cx, &chain, &mut req);
    assert!(req.trace.is_some(), "the adapter gets the caller's context");
    let mut stream = chain.chat(&req).await.expect("opens");
    while stream.next().await.expect("serves").is_some() {}
    stream.close(Ts::ZERO).await;
    trace.end(false, None);
    for _ in 0..50 {
        if !named(&records.spans(), "llm_fallback_adapter").is_empty() {
            break;
        }
        tokio::task::yield_now().await;
    }

    let spans = records.spans();
    let adapter = named(&spans, "llm_fallback_adapter");
    assert_eq!(adapter.len(), 1, "{spans:?}");
    let adapter = adapter[0];
    assert_eq!(adapter.parent, Some(SpanId(8)));
    assert_eq!(attr(adapter, "lk.fallback.index"), Some(&Value::Int(1)));
    assert_eq!(attr(adapter, "lk.fallback.label"), Some(&Value::String("backup".into())));
    let models: Vec<_> =
        adapter.attributes.iter().filter(|(k, _)| *k == "gen_ai.provider.name").collect();
    assert_eq!(models.len(), 1, "an end attribute replaces its start value: {adapter:?}");
    // The failed primary's recovery probe is a request of its own, marked as one.
    let (probes, requests): (Vec<_>, Vec<_>) = named(&spans, "llm_request")
        .into_iter()
        .partition(|r| attr(r, "lk.fallback.recovery") == Some(&Value::Bool(true)));
    assert_eq!(requests.len(), 2, "one per entry tried: {spans:?}");
    assert!(probes.len() <= 1, "{probes:?}");
    assert!(requests.iter().all(|r| r.parent == Some(adapter.context.span_id)));
}

/// A streaming TTS whose stream opens, or fails to.
struct Speaker {
    label: &'static str,
    serves: bool,
}

struct Discard;

impl lk_agents_abi::StreamInput<String> for Discard {
    fn push(&self, _: String) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl lk_agents_abi::Tts for Speaker {
    fn label(&self) -> String {
        self.label.into()
    }
    fn capabilities(&self) -> lk_agents_abi::TtsCapabilities {
        lk_agents_abi::TtsCapabilities { streaming: true, aligned_transcript: false }
    }
    fn sample_rate(&self) -> u32 {
        24_000
    }
    fn num_channels(&self) -> u16 {
        1
    }
    fn synthesize<'a>(
        &'a self,
        _: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<lk_agents_abi::TtsSource, ProviderError>> {
        Box::pin(async move { Err(boom("one-shot")) })
    }
    fn stream(
        &self,
        _: bool,
        _: lk_agents_abi::telemetry::record::TraceContext,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::DuplexStream<lk_agents_abi::TtsSink, lk_agents_abi::TtsSource>,
            ProviderError,
        >,
    > {
        Box::pin(async move {
            if !self.serves {
                return Err(boom("down"));
            }
            let frames = crate::testkit::fallback::Frames {
                left: 2,
                rate: 24_000,
                channels: 1,
                dies: false,
            };
            Ok(lk_agents_abi::DuplexStream::new(
                Box::new(Discard) as lk_agents_abi::TtsSink,
                Box::new(frames) as lk_agents_abi::TtsSource,
            ))
        })
    }
}

// Catches: a fallback TTS's request recorded as a plain `tts_request` with nothing inside, as
// 1.8.10 never did: its request is a `tts_fallback_adapter`, and each instance tried is a
// `tts_request` under the adapter's attempt, the failed one with its error.
#[tokio::test]
async fn a_tts_fallback_adapter_records_each_instance_under_its_attempt() {
    use crate::fallback::FallbackTts;
    use lk_agents_abi::Tts;
    use lk_agents_abi::telemetry::record::Status;
    let (records, cx, tracer) = traced_by(false);
    let (clock, spawner, obs) = parts();
    let entries: Vec<Arc<dyn Tts>> = vec![
        Arc::new(Speaker { label: "primary", serves: false }),
        Arc::new(Speaker { label: "backup", serves: true }),
    ];
    let policy = ChainPolicy { max_retry_per_backend: 0, ..ChainPolicy::tts_defaults() };
    let chain =
        FallbackTts::new(entries, None, policy, clock, spawner, obs, tracer).expect("entries");
    let mut trace = super::TtsTrace::start(&cx, &chain, true);
    let attempt = trace.attempt();
    let (sink, mut source) =
        chain.stream(false, attempt.cx().trace_context()).await.expect("opens").split();
    sink.push("hello".into()).expect("pushed");
    sink.end_input();
    while source.next().await.expect("serves").is_some() {}
    source.close(Ts::ZERO).await;
    trace.end_attempt(attempt, None);
    trace.end("hello".into(), None);
    chain.aclose();

    let spans = records.spans();
    let adapter = named(&spans, "tts_fallback_adapter");
    assert_eq!(adapter.len(), 1, "{spans:?}");
    assert!(named(&spans, "tts_request").iter().all(|r| r.parent != Some(SpanId(8))));
    let run = named(&spans, "tts_request_run")
        .into_iter()
        .find(|r| r.parent == Some(adapter[0].context.span_id))
        .expect("the adapter's attempt");
    // An attempt whose provider named no request carries no ids, not an empty list (1.8.10).
    assert_eq!(attr(run, "lk.provider_request_ids"), None, "{run:?}");
    let requests: Vec<_> = named(&spans, "tts_request")
        .into_iter()
        .filter(|r| r.parent == Some(run.context.span_id))
        .collect();
    assert_eq!(requests.len(), 2, "one per instance tried: {spans:?}");
    let label = |r: &SpanRecord| attr(r, "lk.tts.label").cloned();
    assert_eq!(label(requests[0]), Some(Value::String("primary".into())));
    assert!(matches!(requests[0].status, Status::Error(_)), "{:?}", requests[0]);
    assert_eq!(label(requests[1]), Some(Value::String("backup".into())));
    assert_eq!(attr(requests[1], "lk.pii.input_text"), Some(&Value::String("hello".into())));
}
