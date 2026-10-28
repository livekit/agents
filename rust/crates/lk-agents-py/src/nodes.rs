//! The Python face of an agent's nodes, over the one host-iterable bridge
//! (`lk_agents_abi::python::{HostSource, ItemStream}`): an agent's `llm_node`, `tts_node`,
//! `stt_node`, `transcription_node` and `realtime_audio_output_node` called in the agent's binding
//! (`_core.AgentNodes`), and the default `tts_node` / `stt_node` (`_core.tts_node`,
//! `_core.stt_node`). The voice core sees a `NodeFactory` and Rust streams of records; this side
//! feeds the node its input, drains its output, and calls nothing but the node.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::python::{
    Ambient, Convert, HostHandle, HostProxy, HostSource, ItemSink, ItemStream, NodeCallState,
    NodeCalls, captured, check_pid, item_stream, pyo3_async_runtimes,
};
use lk_agents_abi::voice::io::{AudioOutput, AudioOutputExt as _, TextOutput};
use lk_agents_abi::voice::nodes::NodeFactory;
use lk_agents_abi::voice::nodes::{RealtimeAudioNode, TextNode};
use lk_agents_abi::{
    AudioFrame, ChatRequest, ChoiceDelta, LlmChunk, LlmStream, Role, SpawnerExt as _, SpeechEvent,
    StreamSink as _, StreamSource as _, Stt, SttSink, SttSource, SynthesizedAudio, TimedString,
    Tts, TtsSink, TtsSource,
};
use pyo3::IntoPyObjectExt as _;
use pyo3::prelude::*;

const LOG: &str = "livekit.agents";

/// The numbers in a node's chunk and request ids; nothing reads them but a trace.
static NODE_IDS: AtomicU64 = AtomicU64::new(0);

fn node_id(prefix: &str) -> String {
    format!("{prefix}{}", NODE_IDS.fetch_add(1, Ordering::Relaxed))
}

/// The calls of the agent's `tts_node` running in this context, as the default `tts_node` it
/// calls finds them: its provider's sentences are metered for the speech (the reference's
/// `ChunkedStream`s report theirs), not the node's output.
#[pyclass(frozen)]
struct NodeCallsCell(NodeCalls);

fn node_calls_var(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static VAR: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
    let var = VAR.get_or_try_init(py, || {
        let contextvars = py.import("contextvars")?;
        Ok::<_, PyErr>(contextvars.getattr("ContextVar")?.call1(("lk_node_calls",))?.unbind())
    })?;
    Ok(var.bind(py))
}

// ----------------------------------------------------------------------------- an agent's nodes

/// `_core.AgentNodes(context, host)`: `host.llm(req)`, `host.tts(text)`, `host.stt(audio)`,
/// `host.transcription(text)` and `host.realtime_audio(audio)` call the agent's own nodes; each
/// call runs in a copy of `context`, which holds the agent's binding, so the node's code sees its
/// session as hooks and tools do.
#[pyclass(frozen, name = "AgentNodes", module = "livekit.agents._core")]
pub struct PyAgentNodes {
    inner: Arc<AgentNodes>,
}

lk_agents_abi::register_host!(class PyAgentNodes);

lk_agents_abi::python::inventory::submit! {
    lk_agents_abi::python::HostNative {
        trait_id: || std::any::TypeId::of::<dyn NodeFactory>(),
        read: |obj| {
            let nodes = obj.extract::<PyRef<'_, PyAgentNodes>>().ok()?;
            let nodes: Arc<dyn NodeFactory> = Arc::clone(&nodes.inner) as Arc<dyn NodeFactory>;
            Some(Box::new(nodes) as Box<dyn std::any::Any>)
        },
    }
}

#[pymethods]
impl PyAgentNodes {
    #[new]
    fn new(py: Python<'_>, context: Py<PyAny>, host: Py<PyAny>) -> PyResult<Self> {
        check_pid()?;
        let locals = pyo3_async_runtimes::TaskLocals::with_running_loop(py)?;
        let clock = <Arc<dyn lk_agents_abi::Clock> as Ambient>::ambient();
        let spawner = <Arc<dyn lk_agents_abi::Spawner> as Ambient>::ambient();
        Ok(PyAgentNodes { inner: Arc::new(AgentNodes { context, host, locals, clock, spawner }) })
    }
}

struct AgentNodes {
    context: Py<PyAny>,
    host: Py<PyAny>,
    /// The session's loop, which every node call runs on.
    locals: pyo3_async_runtimes::TaskLocals,
    clock: Arc<dyn lk_agents_abi::Clock>,
    /// Runs a transcription or audio node's drain, which outlives the push that opened it.
    spawner: Arc<dyn lk_agents_abi::Spawner>,
}

impl AgentNodes {
    /// The same nodes, for a drain that outlives the call that opened it.
    fn share(&self) -> AgentNodes {
        Python::attach(|py| AgentNodes {
            context: self.context.clone_ref(py),
            host: self.host.clone_ref(py),
            locals: self.locals.clone(),
            clock: Arc::clone(&self.clock),
            spawner: Arc::clone(&self.spawner),
        })
    }

    /// `host.<node>(arg)` in a fresh copy of the binding's context, on the loop; its output is
    /// drained in that same copy.
    async fn call<T: Send + 'static>(
        &self,
        node: &'static str,
        arg: impl FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send + 'static,
        convert: Convert<T>,
    ) -> Result<HostSource<T>, ProviderError> {
        let (context, host) =
            Python::attach(|py| (self.context.clone_ref(py), self.host.clone_ref(py)));
        let on_loop = Python::attach(|py| HostHandle::from_parts(py.None(), self.locals.clone()));
        let calls = (node == "tts").then(NodeCalls::default);
        let cell = calls.clone();
        let (context, out) = on_loop
            .run_on_loop(move |py| {
                let context = context.bind(py).call_method0("copy")?;
                if let Some(calls) = cell {
                    let set = node_calls_var(py)?.getattr("set")?;
                    context.call_method1("run", (set, Py::new(py, NodeCallsCell(calls))?))?;
                }
                let out = context.call_method1("run", (host.bind(py).getattr(node)?, arg(py)?))?;
                Ok((context.unbind(), out.unbind()))
            })
            .await?;
        let locals = Python::attach(|py| self.locals.clone().with_context(context.into_bound(py)));
        let source = HostSource::returned(out, locals, convert, Arc::clone(&self.clock));
        Ok(match calls {
            Some(calls) => source.with_calls(calls),
            None => source,
        })
    }

    /// A node with input: the core's half of its `ItemStream`, and the node's output.
    async fn duplex<I: Send + Sync + 'static, T: Send + 'static>(
        &self,
        node: &'static str,
        to_py: fn(Python<'_>, &I) -> PyResult<Py<PyAny>>,
        convert: Convert<T>,
    ) -> Result<(ItemSink<I>, HostSource<T>), ProviderError> {
        let (sink, input) = Python::attach(|py| item_stream(py, to_py)).map_err(captured)?;
        let source = self.call(node, move |_| Ok(input.into_any()), convert).await?;
        Ok((sink, source))
    }

    /// One invocation of a node that reads a stream and emits into the leg below: `deliver` takes
    /// each item it yields until the leg refuses one, `done` runs once it has ended.
    fn open<I, T, D, F>(
        &self,
        node: &'static str,
        to_py: fn(Python<'_>, &I) -> PyResult<Py<PyAny>>,
        convert: Convert<T>,
        deliver: D,
        done: impl FnOnce() + Send + 'static,
    ) -> Option<ItemSink<I>>
    where
        I: Send + Sync + 'static,
        T: Send + 'static,
        D: Fn(T) -> F + Send + 'static,
        F: std::future::Future<Output = bool> + Send,
    {
        let (sink, input) = match Python::attach(|py| item_stream(py, to_py)) {
            Ok(halves) => halves,
            Err(e) => {
                lk_agents_abi::python::log::error!(target: LOG, node = node, error = e.to_string().as_str(); "failed to open a node");
                return None;
            }
        };
        let nodes = self.share();
        self.spawner.spawn_unboxed(async move {
            match nodes.call(node, move |_| Ok(input.into_any()), convert).await {
                Ok(mut out) => {
                    loop {
                        let item = match out.next().await {
                            Ok(Some(item)) => item,
                            Ok(None) => break,
                            Err(e) => {
                                lk_agents_abi::python::log::error!(target: LOG, node = node, error = e.to_string().as_str(); "error in a node");
                                break;
                            }
                        };
                        if !deliver(item).await {
                            break;
                        }
                    }
                    let deadline = nodes.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
                    Box::new(out).close(deadline).await;
                }
                Err(e) => lk_agents_abi::python::log::error!(target: LOG, node = node, error = e.to_string().as_str(); "error in a node"),
            }
            done();
        });
        Some(sink)
    }
}

impl NodeFactory for AgentNodes {
    fn text_node(&self, below: Arc<dyn TextOutput>) -> Result<Arc<dyn TextNode>, HostError> {
        let node = TranscriptionNode { nodes: self.share(), below, segment: Mutex::new(None) };
        Ok(Arc::new(node))
    }

    fn audio_node(
        &self,
        below: Arc<dyn AudioOutput>,
    ) -> Result<Arc<dyn RealtimeAudioNode>, HostError> {
        let cut = Arc::new(AtomicBool::new(false));
        let node =
            AudioOutputNode { nodes: self.share(), below, generation: Mutex::new(None), cut };
        Ok(Arc::new(node))
    }

    fn llm_node<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            let req = req.clone();
            Ok(
                Box::new(
                    self.call("llm", move |py| req.into_py_any(py), Box::new(llm_chunk)).await?,
                ) as LlmStream,
            )
        })
    }

    fn tts_node(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move {
            let request_id = node_id("tts_node_");
            let convert: Convert<SynthesizedAudio> =
                Box::new(move |frame| synthesized(frame, &request_id));
            let (sink, source) =
                self.duplex("tts", |py, text: &String| text.into_py_any(py), convert).await?;
            Ok((Box::new(sink) as TtsSink, Box::new(source) as TtsSource))
        })
    }

    fn stt_node(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(SttSink, SttSource), ProviderError>> {
        Box::pin(async move {
            let convert: Convert<SpeechEvent> =
                Box::new(|ev| ev.extract::<SpeechEvent>().map_err(Into::into));
            let (sink, source) =
                self.duplex("stt", |py, frame: &AudioFrame| frame.into_py_any(py), convert).await?;
            Ok((Box::new(sink) as SttSink, Box::new(source) as SttSource))
        })
    }
}

/// `Agent.transcription_node`: one invocation per segment, opened by its first text and ended by
/// its flush. What the node yields goes down the leg below, flushed once the node has ended.
struct TranscriptionNode {
    nodes: AgentNodes,
    below: Arc<dyn TextOutput>,
    /// The open segment's input, while the node reads it.
    segment: Mutex<Option<ItemSink<TimedString>>>,
}

impl TranscriptionNode {
    fn push(&self, text: TimedString) {
        let mut segment = self.segment.lock().unwrap_or_else(|p| p.into_inner());
        if segment.is_none() {
            let (below, flushed) = (Arc::clone(&self.below), Arc::clone(&self.below));
            *segment = self.nodes.open(
                "transcription",
                timed_to_py,
                Box::new(|t| t.extract::<TimedString>().map_err(Into::into)),
                move |t: TimedString| {
                    let below = Arc::clone(&below);
                    async move {
                        let timed = t.start_time.is_some() || t.end_time.is_some();
                        let sent = if timed {
                            below.capture_timed(&t).await
                        } else {
                            below.capture_text(&t.text).await
                        };
                        sent.is_ok()
                    }
                },
                move || flushed.flush(),
            );
        }
        if let Some(sink) = segment.as_ref() {
            sink.send(&text);
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

    /// A segment the node never saw a word of ends at once.
    fn flush(&self) {
        match self.segment.lock().unwrap_or_else(|p| p.into_inner()).take() {
            Some(sink) => sink.end_input(),
            None => self.below.flush(),
        }
    }
}

/// A transcript delta as the node reads it: a `str`, or a `TimedString` when it carries timing.
fn timed_to_py(py: Python<'_>, text: &TimedString) -> PyResult<Py<PyAny>> {
    if text.start_time.is_none() && text.end_time.is_none() {
        return text.text.as_str().into_py_any(py);
    }
    text.into_py_any(py)
}

/// `Agent.realtime_audio_output_node`: one invocation, opened by the generation's first frame and
/// fed until the generation ends (`end`); what it yields goes to the leg below, whose segment
/// boundaries the core sets. Once cut (`clear`) nothing it yields goes down.
struct AudioOutputNode {
    nodes: AgentNodes,
    below: Arc<dyn AudioOutput>,
    /// The open invocation's input, while the node reads it.
    generation: Mutex<Option<ItemSink<AudioFrame>>>,
    cut: Arc<AtomicBool>,
}

impl RealtimeAudioNode for AudioOutputNode {
    fn push<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, ()> {
        Box::pin(async move {
            let mut generation = self.generation.lock().unwrap_or_else(|p| p.into_inner());
            if generation.is_none() {
                let (below, cut) = (Arc::clone(&self.below), Arc::clone(&self.cut));
                *generation = self.nodes.open(
                    "realtime_audio",
                    |py, f: &AudioFrame| f.into_py_any(py),
                    Box::new(|f| f.extract::<AudioFrame>().map_err(Into::into)),
                    move |frame: AudioFrame| {
                        let (below, cut) = (Arc::clone(&below), Arc::clone(&cut));
                        async move {
                            !cut.load(Ordering::SeqCst) && below.capture_frame(&frame).await.is_ok()
                        }
                    },
                    || {},
                );
            }
            if let Some(sink) = generation.as_ref() {
                sink.send(frame);
            }
        })
    }

    fn clear(&self) {
        self.cut.store(true, Ordering::SeqCst);
        self.end();
    }

    fn end(&self) {
        if let Some(sink) = self.generation.lock().unwrap_or_else(|p| p.into_inner()).take() {
            sink.end_input();
        }
    }
}

// ----------------------------------------------------------------------------- what nodes yield

/// What an `llm_node` yields: a `str` the node answered itself (spoken, not a model call), a
/// `FlushSentinel` (the spoken segment ends here), or a model's `ChatChunk`.
fn llm_chunk(item: &Bound<'_, PyAny>) -> PyResult<LlmChunk> {
    if let Ok(text) = item.extract::<String>() {
        let delta = ChoiceDelta {
            role: Some(Role::Assistant),
            content: Some(text),
            ..ChoiceDelta::default()
        };
        return Ok(LlmChunk {
            id: node_id("chunk_"),
            delta: Some(delta),
            node_text: true,
            ..LlmChunk::default()
        });
    }
    let flush = item.py().import("livekit.agents.types")?.getattr("FlushSentinel")?;
    if item.is_instance(&flush)? {
        return Ok(LlmChunk { flush: true, ..LlmChunk::default() });
    }
    item.extract::<LlmChunk>().map_err(Into::into)
}

/// A frame a `tts_node` yields, as the reference wraps it (`tts.SynthesizedAudio`): its userdata
/// carries the words it covers and when its synthesis started.
fn synthesized(frame: &Bound<'_, PyAny>, request_id: &str) -> PyResult<SynthesizedAudio> {
    let class = frame.py().import("livekit.agents.tts")?.getattr("SynthesizedAudio")?;
    let kwargs = pyo3::types::PyDict::new(frame.py());
    kwargs.set_item("frame", frame)?;
    kwargs.set_item("request_id", request_id)?;
    class.call((), Some(&kwargs))?.extract::<SynthesizedAudio>()
}

/// A frame as the reference's default `tts_node` yields it (`ev.frame`), its userdata as the
/// provider stamped it.
fn audio_frame(py: Python<'_>, audio: &SynthesizedAudio) -> PyResult<Py<PyAny>> {
    let frame = (&audio.frame).into_pyobject(py)?;
    let types = py.import("livekit.agents.types")?;
    let userdata = frame.getattr("userdata")?;
    if !audio.timed_transcript.is_empty() {
        userdata.set_item(types.getattr("USERDATA_TIMED_TRANSCRIPT")?, &audio.timed_transcript)?;
    }
    if let Some(started) = audio.started_time {
        // the reference stamps `perf_counter()`; the core keeps wall-clock seconds
        let time = py.import("time")?;
        let wall: f64 = time.call_method0("time")?.extract()?;
        let perf: f64 = time.call_method0("perf_counter")?.extract()?;
        userdata.set_item(types.getattr("USERDATA_TTS_STARTED_TIME")?, perf - (wall - started))?;
    }
    Ok(frame.into_any().unbind())
}

/// A speech event as the reference's `stt.SpeechEvent`.
fn speech_event(py: Python<'_>, ev: &SpeechEvent) -> PyResult<Py<PyAny>> {
    let convert = py.import("livekit.agents.stt")?.getattr("_speech_event_from_core")?;
    Ok(convert.call1((ev.clone().into_pyobject(py)?,))?.unbind())
}

// ----------------------------------------------------------------------------- the default nodes

fn conn_options(options: &Bound<'_, PyAny>) -> PyResult<lk_agents_abi::ApiConnectOptions> {
    let options: lk_agents_abi::ApiConnectOptions = options.extract()?;
    options.validate().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(options)
}

/// `input` (any async iterable), drained in the caller's context: the node calling a default node
/// hands it its own iterator.
fn caller_input<T: Send + 'static>(
    py: Python<'_>,
    input: Py<PyAny>,
    convert: Convert<T>,
) -> PyResult<HostSource<T>> {
    let locals = pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?;
    let clock = <Arc<dyn lk_agents_abi::Clock> as Ambient>::ambient();
    Ok(HostSource::returned(input, locals, convert, clock))
}

/// The provider's stream fed from `input` and drained into `output`, until either side stops.
/// Its calls are read through `calls` by the node's reader.
async fn pump<I: Send + 'static, O: Send + 'static>(
    opened: Result<
        (Box<dyn lk_agents_abi::StreamSink<I>>, Box<dyn lk_agents_abi::StreamSource<O>>),
        ProviderError,
    >,
    mut input: HostSource<I>,
    output: ItemSink<O>,
    clock: Arc<dyn lk_agents_abi::Clock>,
    calls: Option<NodeCalls>,
) {
    let (sink, mut source) = match opened {
        Ok(halves) => halves,
        Err(e) => return output.fail(e),
    };
    if let Some(calls) = &calls {
        let mut calls = calls.lock().unwrap_or_else(|p| p.into_inner());
        match (source.call_ledger(), &mut *calls) {
            (_, NodeCallState::Streamed) => {}
            (None, _) => *calls = NodeCallState::Streamed,
            (Some(ledger), NodeCallState::Metered(ledgers)) => ledgers.push(ledger),
            (Some(ledger), NodeCallState::Unused) => *calls = NodeCallState::Metered(vec![ledger]),
        }
    }
    let feed_clock = Arc::clone(&clock);
    let feed = async move {
        while let Ok(Some(item)) = input.next().await {
            if sink.push(item).is_err() {
                break;
            }
        }
        sink.end_input();
        let deadline = feed_clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        Box::new(input).close(deadline).await;
        // fed: what remains is the provider's
        std::future::pending::<()>().await
    };
    let drain = async {
        loop {
            match source.next().await {
                Ok(Some(item)) => {
                    if !output.send(&item) {
                        return;
                    }
                }
                Ok(None) => return,
                Err(e) => return output.fail(e),
            }
        }
    };
    tokio::select! {
        () = feed => {}
        () = drain => {}
        () = output.closed() => {}
    }
    let deadline = clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    source.close(deadline).await;
}

/// `Agent.default.tts_node`: `text` synthesised by `tts` (sentence by sentence when it does not
/// stream), its frames as an async iterator.
#[pyfunction]
#[pyo3(signature = (tts, text, *, conn_options))]
fn tts_node(
    py: Python<'_>,
    tts: &Bound<'_, PyAny>,
    text: Py<PyAny>,
    conn_options: &Bound<'_, PyAny>,
) -> PyResult<Py<ItemStream>> {
    check_pid()?;
    let options = self::conn_options(conn_options)?;
    let clock = <Arc<dyn lk_agents_abi::Clock> as Ambient>::ambient();
    let tts = <dyn Tts as HostProxy>::from_host(tts, None)?;
    let tts = tts.with_conn_options(&options).unwrap_or(tts);
    let tts = lk_agents_abi::StreamAdapter::adapted(
        tts,
        lk_agents_runtime::cores::SentenceBuffer::factory(),
        lk_agents_runtime::speech_text::markup::text_processor_factory(),
        Arc::clone(&clock),
        options,
    );
    let input = caller_input(py, text, Box::new(|t| t.extract::<String>().map_err(Into::into)))?;
    let (output, frames) = item_stream(py, audio_frame)?;
    let calls = node_calls_var(py)?.call_method1("get", (py.None(),))?;
    let calls = calls.cast::<NodeCallsCell>().ok().map(|cell| Arc::clone(&cell.get().0));
    let spawner = <Arc<dyn lk_agents_abi::Spawner> as Ambient>::ambient();
    spawner.spawn_unboxed(async move {
        pump(tts.stream().await, input, output, clock, calls).await;
    });
    Ok(frames)
}

/// `Agent.default.stt_node`: `audio` recognised by `stt` (over `vad` when it does not stream),
/// its speech events as an async iterator.
#[pyfunction]
#[pyo3(signature = (stt, audio, *, conn_options, vad = None))]
fn stt_node(
    py: Python<'_>,
    stt: &Bound<'_, PyAny>,
    audio: Py<PyAny>,
    conn_options: &Bound<'_, PyAny>,
    vad: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<ItemStream>> {
    check_pid()?;
    let options = self::conn_options(conn_options)?;
    let clock = <Arc<dyn lk_agents_abi::Clock> as Ambient>::ambient();
    let stt = <dyn Stt as HostProxy>::from_host(stt, None)?;
    let vad = vad.map(|v| <dyn lk_agents_abi::Vad as HostProxy>::from_host(v, None)).transpose()?;
    let stt = stt.with_conn_options(&options).unwrap_or(stt);
    let stt =
        lk_agents_abi::stt::stream_adapter::SttStreamAdapter::adapted(stt, vad, Arc::clone(&clock));
    let input =
        caller_input(py, audio, Box::new(|f| f.extract::<AudioFrame>().map_err(Into::into)))?;
    let (output, events) = item_stream(py, speech_event)?;
    let spawner = <Arc<dyn lk_agents_abi::Spawner> as Ambient>::ambient();
    spawner.spawn_unboxed(async move {
        let label = stt.label();
        let opened = stt.stream(None, None).await.and_then(|opened| {
            opened.ok_or_else(|| {
                lk_agents_abi::errors::ApiError::new(format!(
                    "the STT ({label}) does not stream: add a VAD, or wrap it in an stt.StreamAdapter"
                ))
                .into()
            })
        });
        pump(opened, input, output, clock, None).await;
    });
    Ok(events)
}

lk_agents_abi::register_host!(fn tts_node);
lk_agents_abi::register_host!(fn stt_node);
