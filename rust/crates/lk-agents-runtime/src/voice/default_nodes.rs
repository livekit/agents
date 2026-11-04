//! `session.default`'s `llm_node`, `tts_node` and `stt_node`: the session's provider over a
//! registered node's input. The LLM and TTS nodes are where a node's provider request is made, so
//! they open its `llm_request` or `tts_request`, under the span current at the call.

use std::any::Any;
use std::sync::Arc;

use crate::telemetry;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::types::connect::ApiConnectOptionsError;
use lk_agents_abi::voice::nodes::{NodeFrame, ProviderCalls};
use lk_agents_abi::{
    ApiConnectOptions, AudioFrame, ChatContext, ChatItem, ChatRequest, Clock,
    HOST_TEARDOWN_TIMEOUT, HostRef, Llm, LlmChunk, SpawnerExt as _, SpeechEvent, StreamOutput, Stt,
    SynthesizedAudio, ToolChoice, ToolSpec, Tts, Vad,
};
use lk_agents_core::attempts::{AttemptStep, Attempts};
use lk_agents_core::metrics::provider::TtsMetricState;
use lk_agents_net::env::ProcessEnv;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::nodes::{NodeCalls, Opened};

pub struct DefaultNodes;

/// What a default node pumps into: read by the host as an async iterator.
pub struct NodeOutput<T>(tokio::sync::Mutex<mpsc::Receiver<Result<T, ProviderError>>>);

impl<T> NodeOutput<T> {
    async fn recv(&self) -> Result<Option<T>, ProviderError> {
        self.0.lock().await.recv().await.transpose()
    }
}

impl lk_agents_abi::voice::nodes::LlmNodeChunks for NodeOutput<LlmChunk> {
    async fn next(&self) -> Result<Option<LlmChunk>, ProviderError> {
        self.recv().await
    }
}

impl lk_agents_abi::voice::nodes::TtsNodeFrames for NodeOutput<NodeFrame> {
    async fn next(&self) -> Result<Option<NodeFrame>, ProviderError> {
        self.recv().await
    }
}

impl lk_agents_abi::voice::nodes::SttNodeEvents for NodeOutput<SpeechEvent> {
    async fn next(&self) -> Result<Option<SpeechEvent>, ProviderError> {
        self.recv().await
    }
}

fn node_calls(calls: Option<Arc<dyn ProviderCalls>>) -> Option<Arc<NodeCalls>> {
    calls.and_then(|calls| (calls as Arc<dyn Any + Send + Sync>).downcast::<NodeCalls>().ok())
}

/// Where a default node's request records: under the span current at the call (the node's, or an
/// application span inside it), for the job and conversation of the step whose node runs it.
fn request_cx(
    env: &ProcessEnv,
    calls: Option<&NodeCalls>,
    caller: &telemetry::Cx,
) -> telemetry::Cx {
    let step = calls.map(|calls| calls.trace.clone()).unwrap_or_default();
    env.tracer.resume(&telemetry::TraceContext { span: caller.span().or(step.span), ..step })
}

impl lk_agents_abi::voice::nodes::DefaultNodes for DefaultNodes {
    type Env = ProcessEnv;
    type Chunks = NodeOutput<LlmChunk>;
    type Frames = NodeOutput<NodeFrame>;
    type Events = NodeOutput<SpeechEvent>;

    fn llm_node(
        env: &ProcessEnv,
        llm: Arc<dyn Llm>,
        chat_ctx: Vec<ChatItem>,
        chat_ctx_host: HostRef,
        tools: Vec<ToolSpec>,
        tools_host: Option<HostRef>,
        tool_choice: Option<ToolChoice>,
        conn_options: Option<ApiConnectOptions>,
        calls: Option<Arc<dyn ProviderCalls>>,
        cx: telemetry::Cx,
    ) -> Result<NodeOutput<LlmChunk>, ApiConnectOptionsError> {
        let conn_options = policy(conn_options, None)?;
        let req = ChatRequest {
            chat_ctx: ChatContext { items: chat_ctx },
            chat_ctx_host: Some(chat_ctx_host),
            tools,
            tools_host,
            tool_choice,
            conn_options,
            ..ChatRequest::default()
        };
        let cx = request_cx(env, node_calls(calls).as_deref(), &cx);
        // One attempt per call: the node retries the request, as a reply's leg does.
        let llm = crate::voice::session::llm_leg(llm, Some(conn_options));
        let (tx, rx) = mpsc::channel(1);
        match (env.spawner)() {
            // Detached: dropping the output closes `tx`, which ends the request.
            Ok(spawner) => spawner.spawn_detached(complete(llm, req, cx, env.clock.clone(), tx)),
            Err(error) => {
                let _ = tx.try_send(Err(lk_agents_abi::errors::ApiError::new(error).into()));
            }
        }
        Ok(NodeOutput(tokio::sync::Mutex::new(rx)))
    }

    fn tts_node(
        env: &ProcessEnv,
        tts: Arc<dyn Tts>,
        text: Box<dyn StreamOutput<String>>,
        conn_options: Option<ApiConnectOptions>,
        calls: Option<Arc<dyn ProviderCalls>>,
        cx: telemetry::Cx,
    ) -> Result<NodeOutput<NodeFrame>, ApiConnectOptionsError> {
        let conn_options = policy(conn_options, tts.conn_options())?;
        let tts = tts.with_conn_options(&conn_options).unwrap_or(tts);
        let clock = Arc::clone(&env.clock);
        let tts = lk_agents_inference::tts::sentence_adapter::StreamAdapter::adapted_for(
            tts,
            Arc::clone(&clock),
            conn_options,
        );
        let calls = node_calls(calls);
        let expressive = calls.as_ref().is_some_and(|calls| calls.expressive);
        let cx = request_cx(env, calls.as_deref(), &cx);
        Ok(pumped(env, async move {
            // One `tts_request`: the provider's own retries are its attempts, below it.
            let synthesis = Synthesis::start(&cx, tts.as_ref());
            let opened = tts.stream(synthesis.attempt_cx(), expressive).await;
            if let (Ok(stream), Some(calls)) = (&opened, &calls) {
                calls.record(stream.output().call_ledger());
            }
            let map = move |audio: SynthesizedAudio| {
                if let Some(calls) = &calls {
                    calls.connected(audio.connection);
                }
                NodeFrame {
                    frame: audio.frame,
                    timed_transcript: audio.timed_transcript,
                    started_time: audio.started_time,
                }
            };
            (opened, text, clock, map, synthesis)
        }))
    }

    fn stt_node(
        env: &ProcessEnv,
        stt: Arc<dyn Stt>,
        audio: Box<dyn StreamOutput<AudioFrame>>,
        conn_options: Option<ApiConnectOptions>,
        vad: Option<Arc<dyn Vad>>,
        calls: Option<Arc<dyn ProviderCalls>>,
    ) -> Result<NodeOutput<SpeechEvent>, ApiConnectOptionsError> {
        let conn_options = policy(conn_options, stt.conn_options())?;
        let stt = stt.with_conn_options(&conn_options).unwrap_or(stt);
        let clock = Arc::clone(&env.clock);
        let stt = lk_agents_inference::stt::stream_adapter::StreamAdapter::adapted(
            stt,
            vad,
            Arc::clone(&clock),
        );
        let calls = node_calls(calls);
        Ok(pumped(env, async move {
            let label = stt.label();
            let opened = stt.stream(&telemetry::Cx::NONE, None, None).await.and_then(|opened| {
                opened.ok_or_else(|| {
                    lk_agents_abi::errors::ApiError::new(format!(
                        "the STT ({label}) does not stream: add a VAD, or wrap it in an stt.StreamAdapter"
                    ))
                    .into()
                })
            });
            let map = move |event: SpeechEvent| {
                if let (Some(calls), Some(usage)) = (&calls, &event.recognition_usage) {
                    calls.connected(usage.connection);
                }
                event
            };
            (opened, audio, clock, map, ())
        }))
    }
}

/// The node's `conn_options`, else the provider's own policy (models.md), else the defaults.
fn policy(
    given: Option<ApiConnectOptions>,
    provider: Option<ApiConnectOptions>,
) -> Result<ApiConnectOptions, ApiConnectOptionsError> {
    let Some(given) = given else { return Ok(provider.unwrap_or_default()) };
    given.validate()?;
    Ok(given)
}

/// The default `llm_node`'s request: the provider's completion, retried while it has produced
/// nothing, its chunks handed to the node until the node stops reading.
async fn complete(
    llm: Arc<dyn Llm>,
    mut req: ChatRequest,
    cx: telemetry::Cx,
    clock: Arc<dyn Clock>,
    tx: mpsc::Sender<Result<LlmChunk, ProviderError>>,
) {
    let mut trace = telemetry::LlmTrace::start(&cx, llm.as_ref(), &mut req);
    let mut attempts = Attempts::retrying(&req.conn_options, clock.now());
    loop {
        let attempt = trace.attempt();
        let (error, produced) = match llm.chat(&telemetry::Cx::NONE, &req).await {
            Err(error) => (error, false),
            Ok(mut stream) => {
                let mut produced = false;
                let ended = loop {
                    let next = tokio::select! {
                        biased;
                        () = tx.closed() => break None,
                        next = stream.next() => next,
                    };
                    match next {
                        Ok(Some(chunk)) => {
                            trace.observe(&chunk);
                            produced |= chunk.has_response();
                            if tx.send(Ok(chunk)).await.is_err() {
                                break None;
                            }
                        }
                        Ok(None) => break Some(Ok(())),
                        Err(error) => break Some(Err(error)),
                    }
                };
                let _ = stream.close(clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
                match ended {
                    Some(Err(error)) => (error, produced),
                    // `None`: the node stopped reading, which cuts the request.
                    ended => {
                        trace.end_attempt(attempt, None);
                        trace.end(ended.is_none(), None);
                        return;
                    }
                }
            }
        };
        trace.end_attempt(attempt, Some(&error));
        if produced {
            attempts.delivered();
        }
        let (now, mut steps) = (clock.now(), Vec::new());
        if let Some(attempt) = attempts.current() {
            attempts.failed_error(attempt, error.attempt_error(), now, &mut steps);
            attempts.closed(attempt, now, &mut steps);
        }
        let retried = steps.iter().any(|step| matches!(step, AttemptStep::Retry { .. }));
        let at = retried.then(|| attempts.wake_at(None)).flatten();
        llm.failed(error.clone(), at.is_some());
        let Some(at) = at else {
            trace.fail(&error);
            trace.end(false, None);
            let _ = tx.send(Err(error)).await;
            return;
        };
        tokio::select! {
            biased;
            () = tx.closed() => {
                trace.end(true, None);
                return;
            }
            () = clock.wait_until(at) => attempts.wake(at, None, &mut steps),
        }
    }
}

/// What a default node's request records as its stream runs.
trait Request<I, O>: Send + 'static {
    fn fed(&mut self, _item: &I) {}
    fn produced(&mut self, _item: &O) {}
    /// The stream ended: with the error that ended it, or `cut` when its reader went first.
    fn ended(self, _failed: Option<&ProviderError>, _cut: bool)
    where
        Self: Sized,
    {
    }
}

impl<I, O> Request<I, O> for () {}

/// The default `tts_node`'s `tts_request`: the text fed, and its 1.8.10 metrics.
struct Synthesis {
    trace: telemetry::TtsTrace,
    attempt: telemetry::ScopedSpan<telemetry::TtsRequestRun>,
    started: f64,
    text: String,
    metrics: TtsMetricState,
}

impl Synthesis {
    fn start(cx: &telemetry::Cx, tts: &dyn Tts) -> Self {
        let mut trace = telemetry::TtsTrace::start(cx, tts, true);
        let attempt = trace.attempt();
        let started = Synthesis::now(cx);
        let id = tts.id();
        let metrics = TtsMetricState::new(true, id.label.clone(), Some(id.metadata()));
        Synthesis { trace, attempt, started, text: String::new(), metrics }
    }

    /// What the provider's stream runs under: an adapter's own requests open there.
    fn attempt_cx(&self) -> &telemetry::Cx {
        self.attempt.cx()
    }

    fn now(cx: &telemetry::Cx) -> f64 {
        cx.now_nanos() as f64 / 1e9
    }
}

impl Request<String, SynthesizedAudio> for Synthesis {
    fn fed(&mut self, text: &String) {
        self.text.push_str(text);
    }

    fn produced(&mut self, audio: &SynthesizedAudio) {
        self.trace.observe(audio);
        let now = Synthesis::now(self.trace.cx());
        self.metrics.observe(audio, || now - self.started);
    }

    fn ended(self, failed: Option<&ProviderError>, cut: bool) {
        let Synthesis { mut trace, attempt, started, text, mut metrics } = self;
        trace.end_attempt(attempt, failed);
        if let Some(error) = failed {
            trace.fail(error);
        }
        let now = Synthesis::now(trace.cx());
        let metrics = metrics.finish(now, started, &text, now, cut);
        trace.end(text, metrics.as_ref());
    }
}

/// The provider's stream, once `open` opens it, fed from the caller's input and drained into the
/// returned output until either side stops, its request recording both. Detached: the pump ends
/// once the caller's input ends or its reader is gone.
fn pumped<I, O, T, F, M, R>(env: &ProcessEnv, open: F) -> NodeOutput<T>
where
    I: Send + 'static,
    O: Send + 'static,
    T: Send + 'static,
    F: std::future::Future<Output = (Opened<I, O>, Box<dyn StreamOutput<I>>, Arc<dyn Clock>, M, R)>
        + Send
        + 'static,
    M: Fn(O) -> T + Send + Sync + 'static,
    R: Request<I, O>,
{
    let (tx, rx) = mpsc::channel(1);
    let spawner = match (env.spawner)() {
        Ok(spawner) => spawner,
        Err(error) => {
            let _ = tx.try_send(Err(lk_agents_abi::errors::ApiError::new(error).into()));
            return NodeOutput(tokio::sync::Mutex::new(rx));
        }
    };
    let pump = async move {
        let (opened, mut input, clock, map, request) = open.await;
        let (sink, mut source) = match opened {
            Ok(stream) => stream.split(),
            Err(e) => {
                request.ended(Some(&e), false);
                let _ = tx.send(Err(e)).await;
                return;
            }
        };
        // Locked only between awaits, by the feed and the drain of this one task.
        let request = Mutex::new(request);
        let feed_clock = Arc::clone(&clock);
        let feed = async {
            while let Ok(Some(item)) = input.next().await {
                request.lock().fed(&item);
                if sink.push(item).is_err() {
                    break;
                }
            }
            sink.end_input();
            input.close(feed_clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
            // fed: what remains is the provider's
            std::future::pending::<()>().await
        };
        let drain = async {
            loop {
                let item = match source.next().await {
                    Ok(Some(item)) => {
                        request.lock().produced(&item);
                        Ok(map(item))
                    }
                    Ok(None) => return Some(None),
                    Err(e) => Err(e),
                };
                let failed = item.as_ref().err().cloned();
                if tx.send(item).await.is_err() {
                    return None;
                }
                if failed.is_some() {
                    return Some(failed);
                }
            }
        };
        // `None`: the reader went first.
        let ended = tokio::select! {
            biased;
            () = tx.closed() => None,
            ended = drain => ended,
            () = feed => None,
        };
        source.close(clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
        let failed = ended.as_ref().and_then(Option::as_ref);
        request.into_inner().ended(failed, ended.is_none());
    };
    // Detached: the returned output owns the pump; dropping it closes `tx`, which ends the pump.
    spawner.spawn_detached(pump);
    NodeOutput(tokio::sync::Mutex::new(rx))
}
