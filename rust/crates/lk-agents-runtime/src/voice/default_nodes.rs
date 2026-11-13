use std::any::Any;
use std::sync::Arc;

use crate::telemetry;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::types::connect::ApiConnectOptionsError;
use lk_agents_abi::voice::nodes::{NodeFrame, ProviderCalls};
use lk_agents_abi::{
    ApiConnectOptions, AudioFrame, ChatContext, ChatItem, ChatRequest, Clock,
    HOST_TEARDOWN_TIMEOUT, HostRef, Llm, LlmChunk, SpeechEvent, StreamOutput, Stt,
    SynthesizedAudio, ToolChoice, ToolSpec, Tts, Vad,
};
use lk_agents_core::metrics::provider::TtsMetricState;
use lk_agents_net::env::{ProcessEnv, ProcessError};
use lk_agents_net::outbox::{Outbox, outbox};
use lk_agents_net::owned::{Background, Owned};
use lk_agents_net::signal;
use parking_lot::Mutex;

use super::nodes::{NodeCalls, Opened};

pub struct DefaultNodes;

/// What a default node pumps into: read by the host as an async iterator. Its request runs as a
/// task of the session (else the process) until the reader stops: dropped, or closed.
pub struct NodeOutput<T> {
    items: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<Result<T, ProviderError>>>,
    /// Ends once the request's task is gone, its streams closed.
    done: signal::Cell<Option<()>>,
    clock: Arc<dyn Clock>,
}

impl<T: Send + 'static> NodeOutput<T> {
    /// `request` run on `owner`, which also owns its streams, fed the output's sender; refused,
    /// the output yields the refusal.
    fn spawn<F>(
        owner: Result<Background, ProcessError>,
        clock: Arc<dyn Clock>,
        name: &str,
        request: impl FnOnce(Background, Outbox<Result<T, ProviderError>>) -> F,
    ) -> Self
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = outbox(1);
        let (finished, done) = signal::once();
        let refused = tx.clone();
        let spawned = match owner {
            Ok(owner) => {
                let request = request(owner.clone(), tx);
                // The producer ends with the task, however it ends.
                let task = async move {
                    request.await;
                    drop(finished);
                };
                owner.tasks.try_spawn(owner.spawner.as_ref(), name, task).then_some(()).ok_or_else(
                    || {
                        lk_agents_abi::errors::ApiError::new(
                            lk_agents_abi::errors::ErrorKind::Connection,
                            "the session running the node closed",
                        )
                    },
                )
            }
            Err(error) => Err(lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::InvalidRequest,
                error.to_string(),
            )),
        };
        if let Err(error) = spawned {
            let sent = refused.try_send(Err(error.into()));
            debug_assert!(sent, "a request that never ran sent nothing");
        }
        NodeOutput { items: tokio::sync::Mutex::new(rx), done, clock }
    }

    async fn recv(&self) -> Result<Option<T>, ProviderError> {
        self.items.lock().await.recv().await.transpose()
    }

    /// Stops reading, then waits for the request to end, bounded by the teardown limit.
    async fn close(&self) {
        let closed = async {
            self.items.lock().await.close();
            self.done.until(Option::is_some).await;
        };
        if lk_agents_abi::timeout(&*self.clock, HOST_TEARDOWN_TIMEOUT, closed).await.is_err() {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "a default node's request did not close in time");
        }
    }
}

impl lk_agents_abi::voice::nodes::LlmNodeChunks for NodeOutput<LlmChunk> {
    async fn next(&self) -> Result<Option<LlmChunk>, ProviderError> {
        self.recv().await
    }

    async fn aclose(&self) {
        self.close().await;
    }
}

impl lk_agents_abi::voice::nodes::TtsNodeFrames for NodeOutput<NodeFrame> {
    async fn next(&self) -> Result<Option<NodeFrame>, ProviderError> {
        self.recv().await
    }

    async fn aclose(&self) {
        self.close().await;
    }
}

impl lk_agents_abi::voice::nodes::SttNodeEvents for NodeOutput<SpeechEvent> {
    async fn next(&self) -> Result<Option<SpeechEvent>, ProviderError> {
        self.recv().await
    }

    async fn aclose(&self) {
        self.close().await;
    }
}

fn node_calls(calls: Option<Arc<dyn ProviderCalls>>) -> Option<Arc<NodeCalls>> {
    calls.and_then(|calls| (calls as Arc<dyn Any + Send + Sync>).downcast::<NodeCalls>().ok())
}

/// Where a default node's request runs: the session's tasks when a session's node runs it, else
/// the process's (a node the host calls outside a session).
fn owner(env: &ProcessEnv, calls: Option<&NodeCalls>) -> Result<Background, ProcessError> {
    match calls.and_then(|calls| calls.owner.clone()) {
        Some(owner) => Ok(owner),
        None => env.background(),
    }
}

/// Where a default node's request records: under the span current at the call (the node's, or an
/// application span inside it), for the job and conversation of the step whose node runs it.
fn request_cx(
    env: &ProcessEnv,
    calls: Option<&NodeCalls>,
    caller: &telemetry::Cx,
) -> telemetry::Cx {
    let step = calls.and_then(|calls| calls.step.clone()).unwrap_or_else(|| env.tracer.root());
    let span = caller.span().or(step.span());
    step.continuing(span)
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
        let calls = node_calls(calls);
        let cx = request_cx(env, calls.as_deref(), &cx);
        // One attempt per call: the node retries the request, as a reply's leg does.
        let llm = crate::voice::session::llm_leg(llm, Some(conn_options));
        let owner = owner(env, calls.as_deref());
        // Dropping or closing the output closes `tx`, which ends the request.
        Ok(NodeOutput::spawn(
            owner,
            env.clock.clone(),
            "default llm node",
            |owner, tx| async move {
                complete(llm, req, cx, &owner, tx).await;
            },
        ))
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
        let tts = lk_agents_inference::tts::sentence_adapter::StreamAdapter::adapted(
            tts,
            Arc::clone(&clock),
            conn_options,
        );
        let calls = node_calls(calls);
        let cx = request_cx(env, calls.as_deref(), &cx);
        let owner = owner(env, calls.as_deref());
        Ok(pumped(owner, &env.clock, text, async move {
            // One `tts_request`: the provider's own retries are its attempts, below it.
            let synthesis = Synthesis::start(&cx, tts.as_ref());
            let opened = lk_agents_inference::tts::stream(
                tts.clone(),
                synthesis.attempt_cx(),
                conn_options,
                clock.clone(),
                None,
            )
            .await;
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
            (opened, map, synthesis)
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
        let owner = owner(env, calls.as_deref());
        Ok(pumped(owner, &env.clock, audio, async move {
            let label = stt.label();
            let opened = lk_agents_inference::stt::stream(stt.clone(), conn_options, clock.clone(), &telemetry::Cx::NONE, None, None).await.and_then(|opened| {
                opened.ok_or_else(|| {
                    lk_agents_abi::errors::ApiError::new(lk_agents_abi::errors::ErrorKind::InvalidRequest, format!(
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
            (opened, map, ())
        }))
    }
}

/// The node's `conn_options`, else the provider's own policy, else the defaults.
fn policy(
    given: Option<ApiConnectOptions>,
    provider: Option<ApiConnectOptions>,
) -> Result<ApiConnectOptions, ApiConnectOptionsError> {
    let Some(given) = given else { return Ok(provider.unwrap_or_default()) };
    given.validate()?;
    Ok(given)
}

/// The default `llm_node`'s request: the provider's completion, retried while it has produced
/// nothing, its chunks handed to the node until the node stops reading. Its stream is `owner`'s
/// from the start, so a cancelled request still closes it.
async fn complete(
    llm: Arc<dyn Llm>,
    req: ChatRequest,
    cx: telemetry::Cx,
    owner: &Background,
    tx: Outbox<Result<LlmChunk, ProviderError>>,
) {
    let mut trace = telemetry::LlmTrace::start(&cx, llm.as_ref(), &req);
    let attempt = trace.attempt();
    let clock = owner.clock.clone();
    let opened = lk_agents_inference::llm::chat(llm, trace.cx(), &req, clock, None).await;
    let result = match opened {
        Err(error) => Err(error),
        Ok(stream) => forward_chunks(Owned::stream(owner, stream), &tx, &mut trace).await,
    };
    trace.end_attempt(attempt, result.as_ref().err());
    match result {
        Ok(end) => {
            trace.end(telemetry::LlmEnd {
                interrupted: matches!(end, CompletionEnd::Cut),
                ..Default::default()
            });
        }
        Err(error) => {
            trace.fail(&error);
            trace.end(Default::default());
            tx.send(Err(error)).await;
        }
    }
}

/// How an attempt's stream ended without failing.
enum CompletionEnd {
    /// The provider finished its completion.
    Finished,
    /// The node stopped reading, which cuts the request.
    Cut,
}

/// One attempt's chunks handed to the node, then the stream closed. On failure, the error with
/// whether the attempt produced output.
async fn forward_chunks(
    mut stream: Owned<lk_agents_abi::LlmStream>,
    tx: &Outbox<Result<LlmChunk, ProviderError>>,
    trace: &mut telemetry::LlmTrace,
) -> Result<CompletionEnd, ProviderError> {
    let ended = loop {
        let next = tokio::select! {
            biased;
            () = tx.closed() => break Ok(CompletionEnd::Cut),
            next = stream.next() => next,
        };
        match next {
            Ok(Some(chunk)) => {
                trace.observe(&chunk);
                // A reader that left is seen by the `closed` branch.
                tx.send(Ok(chunk)).await;
            }
            Ok(None) => break Ok(CompletionEnd::Finished),
            Err(error) => break Err(error),
        }
    };
    stream.close().await;
    ended
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

/// The default `tts_node`'s `tts_request`: the text fed, and its metrics.
struct Synthesis {
    trace: telemetry::TtsTrace,
    attempt: Option<telemetry::ScopedSpan<telemetry::TtsRequestRun>>,
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
        self.attempt.as_ref().map_or(self.trace.cx(), |a| a.cx())
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

/// The provider's stream, once `open` opens it, fed from the caller's `input` and drained into
/// the returned output until either side stops, its request recording both. `owner` runs it: the
/// pump ends once the caller's input ends or its reader is gone. Both streams are owned from the
/// start, so however the pump ends, each is closed.
fn pumped<I, O, T, F, M, R>(
    owner: Result<Background, ProcessError>,
    clock: &Arc<dyn Clock>,
    input: Box<dyn StreamOutput<I>>,
    open: F,
) -> NodeOutput<T>
where
    I: Send + 'static,
    O: Send + 'static,
    T: Send + 'static,
    F: std::future::Future<Output = (Opened<I, O>, M, R)> + Send + 'static,
    M: Fn(O) -> T + Send + Sync + 'static,
    R: Request<I, O>,
{
    NodeOutput::spawn(owner, Arc::clone(clock), "default node pump", move |owner, tx| {
        let input = Owned::stream(&owner, input);
        pump(owner, input, open, tx)
    })
}

/// [`pumped`]'s task: the provider's stream opened, fed and drained, then both closed.
async fn pump<I, O, T, F, M, R>(
    owner: Background,
    input: Owned<Box<dyn StreamOutput<I>>>,
    open: F,
    tx: Outbox<Result<T, ProviderError>>,
) where
    I: Send + 'static,
    O: Send + 'static,
    T: Send + 'static,
    F: std::future::Future<Output = (Opened<I, O>, M, R)> + Send + 'static,
    M: Fn(O) -> T + Send + Sync + 'static,
    R: Request<I, O>,
{
    let (opened, map, request) = open.await;
    let (sink, source) = match opened {
        Ok(stream) => stream.split(),
        Err(e) => {
            request.ended(Some(&e), false);
            tx.send(Err(e)).await;
            return;
        }
    };
    let mut source = Owned::stream(&owner, source);
    // Locked only between awaits, by the feed and the drain of this one task.
    let request = Mutex::new(request);
    // `None`: the reader went first.
    let ended = tokio::select! {
        biased;
        () = tx.closed() => None,
        failed = drain_provider(&mut **source, &tx, &request, &map) => Some(failed),
        () = feed_provider(input, &*sink, &request) => None,
    };
    source.close().await;
    let failed = ended.as_ref().and_then(Option::as_ref);
    request.into_inner().ended(failed, ended.is_none());
}

/// The caller's input pushed to the provider, then its input ended and closed. Never completes:
/// once fed, what remains is the provider's.
async fn feed_provider<I: 'static, O, R: Request<I, O>>(
    mut input: Owned<Box<dyn StreamOutput<I>>>,
    sink: &dyn lk_agents_abi::StreamInput<I>,
    request: &Mutex<R>,
) {
    while let Ok(Some(item)) = input.next().await {
        request.lock().fed(&item);
        if sink.push(item).is_err() {
            break;
        }
    }
    sink.end_input();
    input.close().await;
    std::future::pending::<()>().await
}

/// The provider's output handed to the reader, until it ends: with the error that ended it, if
/// any.
async fn drain_provider<I, O, T, R: Request<I, O>>(
    source: &mut dyn lk_agents_abi::ProviderStream<O>,
    tx: &Outbox<Result<T, ProviderError>>,
    request: &Mutex<R>,
    map: &impl Fn(O) -> T,
) -> Option<ProviderError> {
    loop {
        let item = match source.next().await {
            Ok(Some(item)) => {
                request.lock().produced(&item);
                Ok(map(item))
            }
            Ok(None) => return None,
            Err(e) => Err(e),
        };
        let failed = item.as_ref().err().cloned();
        // A reader that left is seen by the caller's `closed` branch.
        tx.send(item).await;
        if failed.is_some() {
            return failed;
        }
    }
}
