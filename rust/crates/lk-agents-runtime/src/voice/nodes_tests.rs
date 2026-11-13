use super::*;
use crate::telemetry::tests::labels;
use crate::voice::tests::traced;
use lk_agents_abi::ProviderStream as _;
use lk_agents_abi::telemetry::decl::Value;
use lk_agents_voice::events::SessionEvent;

#[test]
fn node_input_refuses_overflow_and_stops_accepting_after_either_half_closes() {
    let (sink, source) = channel();
    for item in 0..256 {
        sink.push(item).unwrap();
    }
    assert!(sink.push(256).is_err());
    drop(source);
    assert!(sink.push(257).is_err());

    let (sink, _source) = channel();
    sink.end_input();
    assert!(sink.push(0).is_err());
}

// Catches: a node's output metered as one call when its default node metered its provider's
// calls, or those records read twice; a node that ran no default node metered at all; the call
// in flight when the node is cut not ended.
#[test]
fn a_tts_node_reports_the_calls_its_default_node_made() {
    let calls = Arc::new(NodeCalls::default());
    let (_, input) = channel::<SynthesizedAudio>();
    let mut node = Mapped { source: Box::new(input), map: |a| a, calls: Some(calls.clone()) };
    assert!(node.take_call_metrics(false).is_none(), "no default node ran yet");
    assert_eq!(node.take_call_metrics(true), Some(vec![]), "its own audio is no call");
    let record = AgentMetrics::Tts(lk_agents_abi::metrics::testkit::tts());
    let ledger = Arc::new(Mutex::new(vec![record.clone(), record]));
    let cut = Arc::new(AtomicBool::new(false));
    let take: CallLedger = {
        let (ledger, cut) = (ledger.clone(), cut.clone());
        Arc::new(move |in_flight| {
            cut.fetch_or(in_flight, Ordering::Relaxed);
            std::mem::take(&mut *ledger.lock())
        })
    };
    calls.record(Some(take));
    assert_eq!(node.take_call_metrics(false).map(|r| r.len()), Some(2));
    assert_eq!(node.take_call_metrics(true).map(|r| r.len()), Some(0));
    assert!(cut.load(Ordering::Relaxed), "the call in flight is ended");
    let streamed = Arc::new(NodeCalls::default());
    streamed.record(None);
    let mut node =
        Mapped { source: Box::new(channel::<()>().1), map: |a| a, calls: Some(streamed) };
    assert!(node.take_call_metrics(true).is_none(), "a one-call stream: the node meters");
}

// Catches: a `stt_node` that reads slower than the call failing its input, or replaying more
// than its bound once it catches up.
#[tokio::test]
async fn a_slow_stt_node_keeps_the_newest_ten_seconds_and_never_refuses() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let (sink, mut source) = audio_channel(clock, telemetry::Cx::NONE);
    for n in 0..1200u16 {
        sink.push(AudioFrame::new(n.to_le_bytes().repeat(160), 16_000, 1, 160)).unwrap();
    }
    sink.end_input();
    let mut heard = Duration::ZERO;
    let mut last = Vec::new();
    while let Some(frame) = source.next().await.unwrap() {
        heard += lk_agents_abi::voice::rtc::frame_duration(&frame);
        last = frame.data.to_vec();
    }
    assert_eq!(heard, Duration::from_secs(10));
    assert!(last.ends_with(&1199u16.to_le_bytes()), "the newest audio is kept");
    assert!(sink.push(AudioFrame::new(vec![0; 320], 16_000, 1, 160)).is_err());
}

struct Below;

impl Llm for Below {
    fn label(&self) -> String {
        "below".to_owned()
    }

    fn chat<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        _req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        unreachable!("the node makes the request")
    }
}

/// A node whose provider gave up after its own retries.
struct GaveUp;

impl NodeFactory for GaveUp {
    fn text_node(
        &self,
        _below: Arc<dyn TranscriptionOutput>,
    ) -> Result<Arc<dyn lk_agents_abi::voice::nodes::TextNode>, HostError> {
        unreachable!()
    }

    fn llm_node<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        _req: &'a ChatRequest,
        _calls: Arc<dyn lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            Err(lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::Server,
                "down",
            )
            .into())
        })
    }

    fn tts_node(
        &self,
        _cx: &telemetry::Cx,
        _calls: Arc<dyn lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<lk_agents_abi::TextSink, TtsSource>, ProviderError>,
    > {
        Box::pin(async move {
            Err(lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::Server,
                "down",
            )
            .into())
        })
    }

    fn stt_node(
        &self,
        _cx: &telemetry::Cx,
        _calls: Arc<dyn lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<SttSink, SttSource>, ProviderError>,
    > {
        Box::pin(async move {
            Err(lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::Server,
                "down",
            )
            .into())
        })
    }
}

// Catches: a node the engine's attempt loop retries (the node's code would run again on top
// of the retries its provider already made), or a node that hides its provider from metrics.
#[tokio::test]
async fn a_node_speaks_as_its_provider_and_its_failure_is_final() {
    let nodes = Nodes { llm: Some(Arc::new(GaveUp)), ..Nodes::default() };
    let llm = nodes.llm(Arc::new(Below));
    assert_eq!(llm.label(), "below");
    let error = llm
        .chat(&telemetry::Cx::NONE, &ChatRequest::default())
        .await
        .err()
        .expect("the node failed");
    assert!(!error.retryable(), "the engine would run the node again");
}

/// The host's `_SessionNodes` over the default `tts_node` alone: its frames cross back as bare
/// audio, as the host's do, so nothing but the node's calls carries what the provider acquired.
struct DefaultTtsNode(Arc<dyn lk_agents_abi::Tts>);

struct HostFrames(crate::voice::default_nodes::NodeOutput<lk_agents_abi::voice::nodes::NodeFrame>);

impl lk_agents_abi::StreamOutput<lk_agents_abi::SynthesizedAudio> for HostFrames {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::SynthesizedAudio>, lk_agents_abi::errors::ProviderError>,
    > {
        use lk_agents_abi::voice::nodes::TtsNodeFrames as _;
        Box::pin(async move {
            let frame = self.0.next().await?;
            Ok(frame.map(|f| lk_agents_abi::SynthesizedAudio::new(f.frame, "")))
        })
    }

    fn close(
        self: Box<Self>,
        _: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async { lk_agents_abi::Closed::Clean })
    }
}

impl lk_agents_abi::ProviderStream<lk_agents_abi::SynthesizedAudio> for HostFrames {}

impl lk_agents_abi::voice::nodes::SessionNodes for DefaultTtsNode {
    fn llm<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::ChatRequest,
        _: Arc<dyn lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Box<dyn lk_agents_abi::ProviderStream<lk_agents_abi::voice::nodes::NodeChunk>>,
            lk_agents_abi::errors::ProviderError,
        >,
    > {
        unreachable!("only the tts_node runs")
    }

    fn tts<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        text: Box<dyn lk_agents_abi::StreamOutput<String>>,
        calls: Arc<dyn lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Box<dyn lk_agents_abi::ProviderStream<lk_agents_abi::SynthesizedAudio>>,
            lk_agents_abi::errors::ProviderError,
        >,
    > {
        use lk_agents_abi::voice::nodes::DefaultNodes as _;
        let env = lk_agents_net::env::ProcessEnv::current_runtime();
        let frames = crate::voice::default_nodes::DefaultNodes::tts_node(
            &env,
            self.0.clone(),
            text,
            None,
            Some(calls),
            lk_agents_abi::telemetry::Cx::NONE,
        )
        .expect("default options");
        Box::pin(async move { Ok(Box::new(HostFrames(frames)) as Box<_>) })
    }

    fn stt<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Box<dyn lk_agents_abi::StreamOutput<lk_agents_abi::AudioFrame>>,
        _: Arc<dyn lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Box<dyn lk_agents_abi::ProviderStream<lk_agents_abi::SpeechEvent>>,
            lk_agents_abi::errors::ProviderError,
        >,
    > {
        unreachable!("only the tts_node runs")
    }

    fn transcription<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Box<dyn lk_agents_abi::StreamOutput<lk_agents_abi::TranscriptChunk>>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Box<dyn lk_agents_abi::StreamOutput<lk_agents_abi::TranscriptChunk>>,
            lk_agents_abi::errors::ProviderError,
        >,
    > {
        unreachable!("only the tts_node runs")
    }
}

// Catches: the connection a gateway TTS acquired lost on its way through the session's
// `tts_node`, left out of the reply's metrics, or never measured.
#[tokio::test]
async fn a_tts_connection_reaches_the_sink_through_the_node() {
    use lk_agents_abi::voice::nodes::{AgentNodes as _, NodeFactory as _};
    use lk_agents_abi::{StreamInput as _, StreamOutput as _};
    use lk_agents_inference::testkit::{FakeWs, text};
    let (mut runtime, records) = traced("connection").await;
    let pcm = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, [1u8; 4]);
    let ws = Arc::new(
        FakeWs::new(vec![
            text(serde_json::json!({"type": "output_audio", "audio": pcm})),
            text(serde_json::json!({"type": "done"})),
        ])
        .after_flush(),
    );
    let opts = lk_agents_inference::tts::TtsOptions {
        model: "cartesia/sonic-3".into(),
        base_url: "https://gw/v1".into(),
        api_key: "k".into(),
        api_secret: "s".into(),
        ..Default::default()
    };
    let gateway = lk_agents_inference::tts::GatewayTts::new(
        opts,
        ws,
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        Arc::new(lk_agents_net::clock::RealClock::new()),
    );
    let tts: Arc<dyn lk_agents_abi::Tts> = Arc::new(gateway);
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    let nodes = crate::voice::nodes::AgentNodes::new(&env, Arc::new(DefaultTtsNode(tts.clone())));
    let calls = Arc::new(crate::voice::nodes::NodeCalls::default());
    let (sink, mut source) =
        nodes.tts_node(&lk_agents_abi::telemetry::Cx::NONE, calls).await.unwrap().split();
    sink.push("Hello there.".to_owned()).unwrap();
    sink.end_input();
    let mut fold =
        lk_agents_voice::generation::machine::tts::SynthesisMetrics::new(tts.id(), 0, 0.0, 0.0);
    while let Some(frame) = source.next().await.unwrap() {
        fold.note(&frame, || 1.0);
    }
    let metrics = fold.finish("Hello there.", 2.0, false, "speech_1").expect("a record");
    let AgentMetrics::Tts(record) = &metrics else { panic!("a TTS record") };
    assert!(!record.connection_reused && record.acquire_time.is_some_and(|t| t > 0.0));
    runtime.deliver_session_event(SessionEvent::MetricsCollected {
        metrics: Box::new(metrics),
        speech: None,
        counted: false,
    });
    let acquire = records.measurements("lk.agents.connection.acquire_time");
    assert_eq!(acquire.len(), 1);
    assert!(
        labels(&acquire[0]).contains(&("connection_reused".into(), Value::String("false".into())))
    );
}
