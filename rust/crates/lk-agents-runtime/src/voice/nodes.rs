//! The session's registered nodes over the host's `_SessionNodes` (`_core.AgentNodes`), and the
//! default `tts_node` and `stt_node` those nodes may run. The core feeds each node its input,
//! drains its output and calls nothing but the node. [`Nodes`]: a registered `llm_node`, `tts_node` or
//! `stt_node` as the leg the session runs, speaking as the provider below it.

use parking_lot::Mutex;
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::types::connect::ApiConnectOptionsError;
use lk_agents_abi::utils::stream::CallLedger;
use lk_agents_abi::voice::io::{AudioOutput, AudioOutputExt as _, IoError, TextOutput};
use lk_agents_abi::voice::nodes::{
    NodeChunk, NodeFactory, NodeFrame, ProviderCalls, RealtimeAudioNode, SessionNodes, TextNode,
};
use lk_agents_abi::{
    ApiConnectOptions, AudioFrame, ChatRequest, ChoiceDelta, Clock, Closed, HOST_TEARDOWN_TIMEOUT,
    LanguageCode, Llm, LlmChunk, LlmStream, Role, Spawner, SpawnerExt as _, SpeechEvent,
    StreamSink, StreamSource, Stt, SttCapabilities, SttSink, SttSource, SynthesizedAudio,
    TimedString, Ts, Tts, TtsCapabilities, TtsSink, TtsSource, Vad,
};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TaskSet;
use tokio::sync::{mpsc, watch};

/// The numbers in a node's chunk and request ids; nothing reads them but a trace.
static NODE_IDS: AtomicU64 = AtomicU64::new(0);

fn node_id(prefix: &str) -> String {
    format!("{prefix}{}", NODE_IDS.fetch_add(1, Ordering::Relaxed))
}

pub struct AgentNodes {
    host: Arc<dyn SessionNodes>,
    clock: Arc<dyn Clock>,
    spawner: fn() -> Result<Arc<dyn Spawner>, String>,
}

impl lk_agents_abi::voice::nodes::AgentNodes for AgentNodes {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv, host: Arc<dyn SessionNodes>) -> Self {
        AgentNodes { host, clock: Arc::clone(&env.clock), spawner: env.spawner }
    }
}

impl NodeFactory for AgentNodes {
    fn text_node(&self, below: Arc<dyn TextOutput>) -> Result<Arc<dyn TextNode>, HostError> {
        Ok(Arc::new(TranscriptionNode {
            opener: self.opener()?,
            below,
            segment: Mutex::new(None),
            completion: watch::channel(None).0,
            retained: Arc::new(Mutex::new(None)),
            started: AtomicBool::new(false),
            cancelled: watch::channel(false).0,
            tasks: Arc::new(TaskSet::default()),
        }))
    }

    fn audio_node(
        &self,
        below: Arc<dyn AudioOutput>,
    ) -> Result<Arc<dyn RealtimeAudioNode>, HostError> {
        Ok(Arc::new(AudioOutputNode {
            opener: self.opener()?,
            below,
            generation: Mutex::new(None),
            cut: Arc::new(AtomicBool::new(false)),
        }))
    }

    fn llm_node<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            let out = self.host.llm(req).await?;
            Ok(Box::new(Mapped { source: out, map: llm_chunk, calls: None }) as LlmStream)
        })
    }

    fn tts_node(
        &self,
        expressive: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move {
            let (sink, input) = channel();
            let calls = Arc::new(NodeCalls { state: Mutex::default(), expressive });
            let out = self.host.tts(Box::new(input), calls.clone()).await?;
            let request_id = node_id("tts_node_");
            let map = move |audio: SynthesizedAudio| SynthesizedAudio {
                request_id: request_id.clone(),
                ..audio
            };
            let out = Mapped { source: out, map, calls: Some(calls) };
            Ok((Box::new(sink) as TtsSink, Box::new(out) as TtsSource))
        })
    }

    fn stt_node(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(SttSink, SttSource), ProviderError>> {
        Box::pin(async move {
            let (sink, input) = channel();
            let out = self.host.stt(Box::new(input)).await?;
            Ok((Box::new(sink) as SttSink, out))
        })
    }
}

impl AgentNodes {
    fn opener(&self) -> Result<Opener, HostError> {
        let spawner = (self.spawner)().map_err(HostError::new)?;
        Ok(Opener { host: Arc::clone(&self.host), clock: Arc::clone(&self.clock), spawner })
    }
}

/// What an `llm_node` yielded, as the core reads a model's chunks.
fn llm_chunk(chunk: NodeChunk) -> LlmChunk {
    match chunk {
        NodeChunk::Text(text) => LlmChunk {
            id: node_id("chunk_"),
            delta: Some(ChoiceDelta {
                role: Some(Role::Assistant),
                content: Some(text),
                ..ChoiceDelta::default()
            }),
            node_text: true,
            ..LlmChunk::default()
        },
        NodeChunk::Flush => LlmChunk { flush: true, ..LlmChunk::default() },
        NodeChunk::Chunk(chunk) => chunk,
    }
}

// ----------------------------------------------------------------------------- streamed nodes

/// Opens the node invocations of one transcription or realtime audio leg.
struct Opener {
    host: Arc<dyn SessionNodes>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
}

impl Opener {
    /// One invocation of a node that reads a stream and emits into the leg below: `deliver` takes
    /// each item it yields until the leg refuses one, `done` runs once it has ended. Cancelled, it
    /// stops reading and closes the node's output under the teardown deadline.
    fn open<I, T, O, D, F>(
        &self,
        name: &'static str,
        call: O,
        deliver: D,
        done: impl FnOnce(Result<(), IoError>) + Send + 'static,
        tasks: Option<&Arc<TaskSet>>,
        mut cancelled: Option<watch::Receiver<bool>>,
    ) -> ChannelSink<I>
    where
        I: Send + 'static,
        T: Send + 'static,
        O: FnOnce(Arc<dyn SessionNodes>, ChannelSource<I>) -> OpenNode<T> + Send + 'static,
        D: Fn(T) -> F + Send + 'static,
        F: std::future::Future<Output = Result<bool, IoError>> + Send,
    {
        let (sink, input) = channel();
        let (host, clock) = (Arc::clone(&self.host), Arc::clone(&self.clock));
        let deadline = move || clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        let drain = async move {
            let opened = tokio::select! {
                biased;
                () = node_cancelled(&mut cancelled) => return done(Ok(())),
                opened = call(host, input) => opened,
            };
            let mut out = match opened {
                Ok(out) => out,
                Err(error) => return done(Err(error.into())),
            };
            let outcome = loop {
                let next = tokio::select! {
                    biased;
                    () = node_cancelled(&mut cancelled) => break Ok(()),
                    next = out.next() => next,
                };
                let item = match next {
                    Ok(Some(item)) => item,
                    Ok(None) => break Ok(()),
                    Err(e) => break Err(IoError::from(e)),
                };
                let delivered = tokio::select! {
                    biased;
                    () = node_cancelled(&mut cancelled) => break Ok(()),
                    delivered = deliver(item) => delivered,
                };
                match delivered {
                    Ok(true) => {}
                    Ok(false) => break Ok(()),
                    Err(error) => break Err(error),
                }
            };
            let cleanup = out.close(deadline()).await;
            done(outcome.and_then(|()| match cleanup {
                Closed::Clean => Ok(()),
                _ => Err(IoError::Rtc("node output cleanup did not settle".into())),
            }));
        };
        match tasks {
            Some(tasks) => {
                tasks.spawn(self.spawner.as_ref(), name, drain);
            }
            // Detached: no segment owns this node; the drain ends with its output.
            None => self.spawner.spawn_detached(drain),
        }
        sink
    }
}

type OpenNode<T> =
    lk_agents_abi::BoxFuture<'static, Result<Box<dyn StreamSource<T>>, ProviderError>>;

async fn node_cancelled(cancelled: &mut Option<watch::Receiver<bool>>) {
    match cancelled {
        Some(cancelled) => {
            let _ = cancelled.wait_for(|value| *value).await;
        }
        None => std::future::pending().await,
    }
}

/// `Agent.transcription_node`: one invocation per segment, opened by its first text and ended by
/// its flush. What the node yields goes down the leg below, flushed once the node has ended.
struct TranscriptionNode {
    opener: Opener,
    below: Arc<dyn TextOutput>,
    /// The open segment's input, while the node reads it.
    segment: Mutex<Option<ChannelSink<TimedString>>>,
    completion: watch::Sender<Option<Result<(), IoError>>>,
    retained: Arc<Mutex<Option<Box<dyn Send>>>>,
    started: AtomicBool,
    cancelled: watch::Sender<bool>,
    tasks: Arc<TaskSet>,
}

impl TranscriptionNode {
    fn push(&self, text: TimedString) {
        let mut segment = self.segment.lock();
        if segment.is_none() {
            let (below, flushed) = (Arc::clone(&self.below), Arc::clone(&self.below));
            self.started.store(true, Ordering::SeqCst);
            let done = self.completion.clone();
            let retained = self.retained.clone();
            *segment = Some(self.opener.open(
                "transcription",
                |host, input| {
                    Box::pin(async move { host.transcription(Box::new(input)).await })
                        as OpenNode<TimedString>
                },
                move |t: TimedString| {
                    let below = Arc::clone(&below);
                    async move {
                        let timed = t.start_time.is_some() || t.end_time.is_some();
                        let sent = if timed {
                            below.capture_timed(&t).await
                        } else {
                            below.capture_text(&t.text).await
                        };
                        sent.map(|()| true)
                    }
                },
                move |outcome| {
                    flushed.flush();
                    let owner = {
                        let mut owner = retained.lock();
                        done.send_replace(Some(outcome));
                        owner.take()
                    };
                    drop(owner);
                },
                Some(&self.tasks),
                Some(self.cancelled.subscribe()),
            ));
        }
        if let Some(sink) = segment.as_ref() {
            let _ = sink.push(text);
        }
    }
}

impl TextNode for TranscriptionNode {
    fn capture_text(&self, text: &str) {
        self.push(TimedString::new(text));
    }

    fn capture_timed(&self, text: &TimedString) {
        self.push(text.clone());
    }

    fn retain_until_drained(&self, owner: Box<dyn Send>) {
        let mut retained = self.retained.lock();
        if self.started.load(Ordering::SeqCst) && self.completion.borrow().is_none() {
            assert!(retained.is_none(), "a segment has one binding owner");
            *retained = Some(owner);
        } else {
            drop(retained);
            drop(owner);
        }
    }

    fn cancel(&self) {
        self.cancelled.send_replace(true);
    }

    fn drain(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), IoError>> {
        let mut completion = self.completion.subscribe();
        Box::pin(async move {
            if !self.started.load(Ordering::SeqCst) {
                return Ok(());
            }
            let result = completion
                .wait_for(Option::is_some)
                .await
                .map_err(|_| IoError::Rtc("transcription node completion was lost".into()))?;
            result.as_ref().expect("completed node").clone()
        })
    }

    /// A segment the node never saw a word of ends at once.
    fn flush(&self) {
        match self.segment.lock().take() {
            Some(sink) => sink.end_input(),
            None => self.below.flush(),
        }
    }
}

impl Drop for TranscriptionNode {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// `Agent.realtime_audio_output_node`: one invocation, opened by the generation's first frame and
/// fed until the generation ends (`end`); what it yields goes to the leg below, whose segment
/// boundaries the core sets. Once cut (`clear`) nothing it yields goes down.
struct AudioOutputNode {
    opener: Opener,
    below: Arc<dyn AudioOutput>,
    /// The open invocation's input, while the node reads it.
    generation: Mutex<Option<ChannelSink<AudioFrame>>>,
    cut: Arc<AtomicBool>,
}

impl RealtimeAudioNode for AudioOutputNode {
    fn push<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, ()> {
        Box::pin(async move {
            let mut generation = self.generation.lock();
            if generation.is_none() {
                let (below, cut) = (Arc::clone(&self.below), Arc::clone(&self.cut));
                *generation = Some(self.opener.open(
                    "realtime_audio",
                    |host, input| {
                        Box::pin(async move { host.realtime_audio(Box::new(input)).await })
                            as OpenNode<AudioFrame>
                    },
                    move |frame: AudioFrame| {
                        let (below, cut) = (Arc::clone(&below), Arc::clone(&cut));
                        async move {
                            if cut.load(Ordering::SeqCst) {
                                Ok(false)
                            } else {
                                below.capture_frame(&frame).await.map(|()| true)
                            }
                        }
                    },
                    |outcome| {
                        if let Err(error) = outcome {
                            log::error!(target: "livekit.agents", "error in audio node: {error}");
                        }
                    },
                    None,
                    None,
                ));
            }
            if let Some(sink) = generation.as_ref() {
                let _ = sink.push(frame.clone());
            }
        })
    }

    fn clear(&self) {
        self.cut.store(true, Ordering::SeqCst);
        self.end();
    }

    fn end(&self) {
        if let Some(sink) = self.generation.lock().take() {
            sink.end_input();
        }
    }
}

// ----------------------------------------------------------------------------- streams

/// What the core hands a node as its input: the node reads what was pushed until the input ends.
/// Unbounded because the producer bounds it: one segment's text, or one realtime generation's
/// audio paced by the model; a node that stops reading drops the receiver and pushes are discarded.
/// A node that stopped reading drops the rest, as a node that returned early ignores its input.
struct ChannelSink<I>(Mutex<Option<mpsc::UnboundedSender<I>>>);

struct ChannelSource<I>(mpsc::UnboundedReceiver<I>);

fn channel<I>() -> (ChannelSink<I>, ChannelSource<I>) {
    let (tx, rx) = mpsc::unbounded_channel();
    (ChannelSink(Mutex::new(Some(tx))), ChannelSource(rx))
}

impl<I: Send> StreamSink<I> for ChannelSink<I> {
    fn push(&self, item: I) -> Result<(), ProviderError> {
        if let Some(tx) = self.0.lock().as_ref() {
            let _ = tx.send(item);
        }
        Ok(())
    }

    /// The reference's node input has no segment marker.
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        self.0.lock().take();
    }
}

impl<I: Send + 'static> StreamSource<I> for ChannelSource<I> {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<I>, ProviderError>> {
        Box::pin(async move { Ok(self.0.recv().await) })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

/// A node's output, each item read as the core reads it; a `tts_node`'s metered by the calls of
/// the default node it runs.
struct Mapped<A, M> {
    source: Box<dyn StreamSource<A>>,
    map: M,
    calls: Option<Arc<NodeCalls>>,
}

impl<A: Send + 'static, T, M: FnMut(A) -> T + Send> StreamSource<T> for Mapped<A, M> {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move { Ok(self.source.next().await?.map(&mut self.map)) })
    }

    fn cancel_safe(&self) -> bool {
        self.source.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        self.source.close(deadline)
    }

    fn started_time(&self) -> Option<f64> {
        self.source.started_time()
    }

    fn take_call_metrics(&mut self, in_flight: bool) -> Option<Vec<AgentMetrics>> {
        self.calls.as_ref()?.take(in_flight)
    }
}

/// The provider calls of the default nodes a `tts_node` runs, as the reference meters a TTS's
/// streams and not the node's output.
#[derive(Default)]
pub struct NodeCalls {
    state: Mutex<NodeCallState>,
    /// The synthesis carries expressive markup; the default node opens its provider with it.
    expressive: bool,
}

#[derive(Default)]
enum NodeCallState {
    /// No default node: the node makes its own audio, which no provider call stands behind.
    #[default]
    Unused,
    /// A provider stream that is one call: the node's output stands for it.
    Streamed,
    /// Providers that meter their own calls (a `StreamAdapter`'s sentences).
    Metered(Vec<CallLedger>),
}

impl NodeCalls {
    fn record(&self, ledger: Option<CallLedger>) {
        let mut state = self.state.lock();
        match (ledger, &mut *state) {
            (_, NodeCallState::Streamed) => {}
            (None, _) => *state = NodeCallState::Streamed,
            (Some(ledger), NodeCallState::Metered(ledgers)) => ledgers.push(ledger),
            (Some(ledger), NodeCallState::Unused) => *state = NodeCallState::Metered(vec![ledger]),
        }
    }

    fn take(&self, in_flight: bool) -> Option<Vec<AgentMetrics>> {
        match &*self.state.lock() {
            NodeCallState::Unused => in_flight.then(Vec::new),
            NodeCallState::Streamed => None,
            NodeCallState::Metered(ledgers) => {
                Some(ledgers.iter().flat_map(|take| take(in_flight)).collect())
            }
        }
    }
}

impl ProviderCalls for NodeCalls {}

impl lk_agents_abi::voice::nodes::NodeCalls for NodeCalls {}

// ----------------------------------------------------------------------------- the default nodes

pub struct DefaultNodes;

/// What a default node pumps into: read by the host as an async iterator.
pub struct NodeOutput<T>(tokio::sync::Mutex<mpsc::Receiver<Result<T, ProviderError>>>);

impl<T> NodeOutput<T> {
    async fn recv(&self) -> Result<Option<T>, ProviderError> {
        self.0.lock().await.recv().await.transpose()
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

impl lk_agents_abi::voice::nodes::DefaultNodes for DefaultNodes {
    type Env = ProcessEnv;
    type Frames = NodeOutput<NodeFrame>;
    type Events = NodeOutput<SpeechEvent>;

    fn tts_node(
        env: &ProcessEnv,
        tts: Arc<dyn Tts>,
        text: Box<dyn StreamSource<String>>,
        conn_options: Option<ApiConnectOptions>,
        calls: Option<Arc<dyn ProviderCalls>>,
    ) -> Result<NodeOutput<NodeFrame>, ApiConnectOptionsError> {
        let conn_options = policy(conn_options, tts.conn_options())?;
        let tts = tts.with_conn_options(&conn_options).unwrap_or(tts);
        let clock = Arc::clone(&env.clock);
        let tts = lk_agents_inference::tts::sentence_adapter::StreamAdapter::adapted_for(
            tts,
            Arc::clone(&clock),
            conn_options,
        );
        let calls = calls
            .and_then(|calls| (calls as Arc<dyn Any + Send + Sync>).downcast::<NodeCalls>().ok());
        let expressive = calls.as_ref().is_some_and(|calls| calls.expressive);
        Ok(pumped(env, async move {
            let opened = tts.stream(expressive).await;
            if let (Ok((_, source)), Some(calls)) = (&opened, &calls) {
                calls.record(source.call_ledger());
            }
            (opened, text, clock, |audio: SynthesizedAudio| NodeFrame {
                frame: audio.frame,
                timed_transcript: audio.timed_transcript,
                started_time: audio.started_time,
            })
        }))
    }

    fn stt_node(
        env: &ProcessEnv,
        stt: Arc<dyn Stt>,
        audio: Box<dyn StreamSource<AudioFrame>>,
        conn_options: Option<ApiConnectOptions>,
        vad: Option<Arc<dyn Vad>>,
    ) -> Result<NodeOutput<SpeechEvent>, ApiConnectOptionsError> {
        let conn_options = policy(conn_options, stt.conn_options())?;
        let stt = stt.with_conn_options(&conn_options).unwrap_or(stt);
        let clock = Arc::clone(&env.clock);
        let stt = lk_agents_inference::stt::stream_adapter::StreamAdapter::adapted(
            stt,
            vad,
            Arc::clone(&clock),
        );
        Ok(pumped(env, async move {
            let label = stt.label();
            let opened = stt.stream(None, None).await.and_then(|opened| {
                opened.ok_or_else(|| {
                    lk_agents_abi::errors::ApiError::new(format!(
                        "the STT ({label}) does not stream: add a VAD, or wrap it in an stt.StreamAdapter"
                    ))
                    .into()
                })
            });
            (opened, audio, clock, |event: SpeechEvent| event)
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

type Opened<I, O> = Result<(Box<dyn StreamSink<I>>, Box<dyn StreamSource<O>>), ProviderError>;

/// The provider's stream, once `open` opens it, fed from the caller's input and drained into the
/// returned output until either side stops. Detached: the pump ends once the caller's input ends
/// or its reader is gone.
fn pumped<I, O, T, F, M>(env: &ProcessEnv, open: F) -> NodeOutput<T>
where
    I: Send + 'static,
    O: Send + 'static,
    T: Send + 'static,
    F: std::future::Future<Output = (Opened<I, O>, Box<dyn StreamSource<I>>, Arc<dyn Clock>, M)>
        + Send
        + 'static,
    M: Fn(O) -> T + Send + Sync + 'static,
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
        let (opened, mut input, clock, map) = open.await;
        let (sink, mut source) = match opened {
            Ok(halves) => halves,
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        };
        let feed_clock = Arc::clone(&clock);
        let feed = async move {
            while let Ok(Some(item)) = input.next().await {
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
                let next = source.next().await;
                let end = !matches!(next, Ok(Some(_)));
                let item = match next {
                    Ok(Some(item)) => Ok(map(item)),
                    Ok(None) => return,
                    Err(e) => Err(e),
                };
                if tx.send(item).await.is_err() || end {
                    return;
                }
            }
        };
        tokio::select! {
            biased;
            () = tx.closed() => {}
            () = drain => {}
            () = feed => {}
        }
        source.close(clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
    };
    // Detached: the returned output owns the pump; dropping it closes `tx`, which ends the pump.
    spawner.spawn_detached(pump);
    NodeOutput(tokio::sync::Mutex::new(rx))
}

/// The nodes the session registers; a leg without one is the provider itself.
#[derive(Default, Clone)]
pub struct Nodes {
    pub llm: Option<Arc<dyn NodeFactory>>,
    pub tts: Option<Arc<dyn NodeFactory>>,
    pub stt: Option<Arc<dyn NodeFactory>>,
}

impl Nodes {
    pub fn llm(&self, below: Arc<dyn Llm>) -> Arc<dyn Llm> {
        match &self.llm {
            Some(host) => Arc::new(Node { below, host: Arc::clone(host) }),
            None => below,
        }
    }

    pub fn tts(&self, below: Arc<dyn Tts>) -> Arc<dyn Tts> {
        match &self.tts {
            Some(host) => Arc::new(Node { below, host: Arc::clone(host) }),
            None => below,
        }
    }

    pub fn stt(&self, below: Arc<dyn Stt>) -> Arc<dyn Stt> {
        match &self.stt {
            Some(host) => Arc::new(Node { below, host: Arc::clone(host) }),
            None => below,
        }
    }
}

struct Node<P: ?Sized> {
    below: Arc<P>,
    host: Arc<dyn NodeFactory>,
}

/// The engine does not retry a node: the provider the node calls retries below it, and the node's
/// own code runs once per request, as in the reference.
fn final_error(error: ProviderError) -> ProviderError {
    error.into_final()
}

struct Final<T>(Box<dyn StreamSource<T>>);

impl<T: Send + 'static> StreamSource<T> for Final<T> {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move { self.0.next().await.map_err(final_error) })
    }

    fn cancel_safe(&self) -> bool {
        self.0.cancel_safe()
    }

    fn close(
        self: Box<Self>,
        deadline: Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        self.0.close(deadline)
    }

    fn started_time(&self) -> Option<f64> {
        self.0.started_time()
    }

    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<lk_agents_abi::metrics::base::AgentMetrics>> {
        self.0.take_call_metrics(in_flight)
    }

    fn call_ledger(&self) -> Option<lk_agents_abi::utils::stream::CallLedger> {
        self.0.call_ledger()
    }
}

fn duplex<I, T: Send + 'static>(
    opened: Result<lk_agents_abi::StreamPair<I, T>, ProviderError>,
) -> Result<lk_agents_abi::StreamPair<I, T>, ProviderError> {
    let (sink, source) = opened.map_err(final_error)?;
    Ok((sink, Box::new(Final(source))))
}

impl Llm for Node<dyn Llm> {
    fn label(&self) -> String {
        self.below.label()
    }

    fn model(&self) -> String {
        self.below.model()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.below.id()
    }

    fn prewarm(&self) {
        self.below.prewarm();
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.below.failed(error, recoverable);
    }

    fn chat<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            let stream = self.host.llm_node(req).await.map_err(final_error)?;
            Ok(Box::new(Final(stream)) as LlmStream)
        })
    }
}

impl Tts for Node<dyn Tts> {
    fn label(&self) -> String {
        self.below.label()
    }

    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, ..self.below.capabilities() }
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.below.id()
    }

    fn prewarm(&self) {
        self.below.prewarm();
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.below.failed(error, recoverable);
    }

    fn sample_rate(&self) -> u32 {
        self.below.sample_rate()
    }

    fn num_channels(&self) -> u16 {
        self.below.num_channels()
    }

    fn markup_provider(&self) -> String {
        self.below.markup_provider()
    }

    fn lowers_markup(&self) -> bool {
        self.below.lowers_markup()
    }

    /// One text, through the node all the same.
    fn synthesize<'a>(
        &'a self,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        Box::pin(async move {
            let (sink, source) = self.stream(false).await?;
            sink.push(text.to_owned()).map_err(final_error)?;
            sink.end_input();
            Ok(source)
        })
    }

    fn stream(
        &self,
        expressive: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move { duplex(self.host.tts_node(expressive).await) })
    }
}

impl Stt for Node<dyn Stt> {
    fn label(&self) -> String {
        self.below.label()
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..self.below.capabilities() }
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.below.id()
    }

    fn prewarm(&self) {
        self.below.prewarm();
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.below.failed(error, recoverable);
    }

    fn update_keyterms(&self, keyterms: &[String]) {
        self.below.update_keyterms(keyterms);
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        self.below.push_conversation_item(role, text);
    }

    /// `stt_node` is a stream; a batch recognition is the provider's.
    fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        self.below.recognize(buffer, language)
    }

    fn stream(
        &self,
        _language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>> {
        Box::pin(async move { duplex(self.host.stt_node().await).map(Some) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a node's output metered as one call when the default node it ran metered its
    // provider's (a non-streaming TTS's sentences, as the reference's `ChunkedStream`s report
    // theirs), or those records read twice; a node that ran no default node metered at all; the
    // call in flight when the node is cut not ended.
    #[test]
    fn a_tts_node_reports_the_calls_its_default_node_made() {
        let calls = Arc::new(NodeCalls { state: Mutex::default(), expressive: false });
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
        let streamed = Arc::new(NodeCalls { state: Mutex::default(), expressive: false });
        streamed.record(None);
        let mut node =
            Mapped { source: Box::new(channel::<()>().1), map: |a| a, calls: Some(streamed) };
        assert!(node.take_call_metrics(true).is_none(), "a one-call stream: the node meters");
    }

    struct Below;

    impl Llm for Below {
        fn label(&self) -> String {
            "below".to_owned()
        }

        fn chat<'a>(
            &'a self,
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
            _below: Arc<dyn TextOutput>,
        ) -> Result<Arc<dyn lk_agents_abi::voice::nodes::TextNode>, HostError> {
            unreachable!()
        }

        fn audio_node(
            &self,
            _below: Arc<dyn lk_agents_abi::voice::io::AudioOutput>,
        ) -> Result<Arc<dyn lk_agents_abi::voice::nodes::RealtimeAudioNode>, HostError> {
            unreachable!()
        }

        fn llm_node<'a>(
            &'a self,
            _req: &'a ChatRequest,
        ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
            Box::pin(async move { Err(lk_agents_abi::errors::ApiError::new("down").into()) })
        }

        fn tts_node(
            &self,
            _: bool,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
            Box::pin(async move { Err(lk_agents_abi::errors::ApiError::new("down").into()) })
        }

        fn stt_node(
            &self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(SttSink, SttSource), ProviderError>> {
            Box::pin(async move { Err(lk_agents_abi::errors::ApiError::new("down").into()) })
        }
    }

    // Catches: a node the engine's attempt loop retries (the node's code would run again on top
    // of the retries its provider already made), or a node that hides its provider from metrics.
    #[tokio::test]
    async fn a_node_speaks_as_its_provider_and_its_failure_is_final() {
        let nodes = Nodes { llm: Some(Arc::new(GaveUp)), ..Nodes::default() };
        let llm = nodes.llm(Arc::new(Below));
        assert_eq!(llm.label(), "below");
        let error = llm.chat(&ChatRequest::default()).await.err().expect("the node failed");
        assert!(!error.retryable(), "the engine would run the node again");
    }
}
