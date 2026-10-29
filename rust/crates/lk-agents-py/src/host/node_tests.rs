//! The runtime's node legs over the generated `_AgentNodes` proxy and a live interpreter.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use lk_agents_abi::voice::io::{IoError, TextOutput};
use lk_agents_abi::voice::nodes::{AgentNodes as _, NodeFactory};
use lk_agents_abi::{AudioFrame, HostRef};
use lk_agents_runtime::net::env::ProcessEnv;
use lk_agents_runtime::voice::nodes::AgentNodes;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::generated::voice::SessionNodesProxy;

struct Destination;

impl TextOutput for Destination {
    fn label(&self) -> &str {
        "destination"
    }
    fn capture_text<'a>(&'a self, _: &'a str) -> lk_agents_abi::BoxFuture<'a, Result<(), IoError>> {
        Box::pin(async { Ok(()) })
    }
    fn flush(&self) {}
}

struct BindingOwner(Arc<AtomicBool>);

impl Drop for BindingOwner {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// The host classes the node items convert to, standing in for the `livekit` packages.
const HOST_TYPES: &std::ffi::CStr = c"
import sys, types
class TimedString(str):
    def __new__(cls, text, **times):
        self = super().__new__(cls, text)
        self.__dict__.update(times)
        return self
class AudioFrame:
    def __init__(self, **fields):
        self.__dict__.update(fields)
for name, attrs in {
    'livekit': {}, 'livekit.agents': {},
    'livekit.agents.types': {'TimedString': TimedString}, 'livekit.rtc': {'AudioFrame': AudioFrame},
}.items():
    sys.modules.setdefault(name, types.ModuleType(name)).__dict__.update(attrs)
";

/// The agent nodes over the host object `processing`, defined by `source` on the test's loop.
fn nodes(source: &std::ffi::CStr) -> (Py<PyDict>, AgentNodes) {
    Python::initialize();
    let (g, captured): (Py<PyDict>, HostRef) = Python::attach(super::stream::tests::setup);
    let host = Python::attach(|py| {
        py.run(HOST_TYPES, None, None).unwrap();
        py.run(source, Some(g.bind(py)), None).unwrap();
        let processing = g.bind(py).get_item("processing").unwrap().unwrap();
        SessionNodesProxy::in_context(&processing, &captured).unwrap()
    });
    (g, AgentNodes::new(&ProcessEnv::current_runtime(), host))
}

async fn flag(g: &Py<PyDict>, name: &str) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while !Python::attach(|py| {
            let event = g.bind(py).get_item(name).unwrap().unwrap();
            event.call_method0("is_set").unwrap().extract::<bool>().unwrap()
        }) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the Python node did not reach the expected boundary");
}

// Catches: a cancelled transcription node dropping its binding owner, or settling its drain,
// before the node's `finally` block has run to its end.
#[tokio::test]
async fn text_node_cancellation_retains_its_binding_until_python_finally_finishes() {
    let (g, nodes) = nodes(
        c"
entered = threading.Event()
closing = threading.Event()
cleaned = threading.Event()
close_gate = asyncio.Event()
class Processing:
    async def transcription(self, text):
        try:
            while (word := await text.recv()) is not None:
                yield word
            entered.set()
            await asyncio.Future()
        finally:
            closing.set()
            await close_gate.wait()
            cleaned.set()
processing = Processing()
",
    );
    let node = nodes.text_node(Arc::new(Destination)).unwrap();
    node.capture_text("queued words");
    node.flush();
    flag(&g, "entered").await;
    let released = Arc::new(AtomicBool::new(false));
    node.retain_until_drained(Box::new(BindingOwner(released.clone())));
    node.cancel();
    flag(&g, "closing").await;
    let mut receipt = node.drain();
    let was_pending =
        tokio::time::timeout(Duration::from_millis(30), receipt.as_mut()).await.is_err();
    let was_retained = !released.load(Ordering::SeqCst);
    // Always release the gate, a regression failure included, so the host loop ends clean.
    Python::attach(|py| {
        let g = g.bind(py);
        let set = g.get_item("close_gate").unwrap().unwrap().getattr("set").unwrap();
        let event_loop = g.get_item("loop").unwrap().unwrap();
        event_loop.call_method1("call_soon_threadsafe", (set,)).unwrap();
    });
    if was_pending {
        tokio::time::timeout(Duration::from_secs(2), receipt)
            .await
            .expect("the text receipt did not settle after the node's cleanup")
            .unwrap();
    }
    flag(&g, "cleaned").await;
    assert!(was_pending, "the node completed while its Python finally block was still open");
    assert!(was_retained, "the binding owner was dropped before the node's cleanup");
    assert!(released.load(Ordering::SeqCst), "the completed node retained its binding");
}

// Catches: the leg awaiting the node's output before handing back its input sink, which
// deadlocks a coroutine node (`_node_output` awaits it on the first pull) that reads its whole
// input before it returns.
#[tokio::test]
async fn coroutine_node_receives_its_input_before_returning() {
    let (g, nodes) = nodes(
        c"
consumed = []
async def node(audio):
    while (frame := await audio.recv()) is not None:
        consumed.append(frame.samples_per_channel)
class Processing:
    async def stt(self, audio):
        if (await node(audio)) is not None:
            yield
processing = Processing()
",
    );
    let (sink, mut output) = tokio::time::timeout(Duration::from_secs(2), nodes.stt_node())
        .await
        .expect("the leg awaited the node's output before returning its input")
        .unwrap();
    sink.push(AudioFrame::new(vec![0u8; 320], 16_000, 1, 160)).unwrap();
    sink.end_input();
    let end = tokio::time::timeout(Duration::from_secs(2), output.next())
        .await
        .expect("the node did not return once its input ended")
        .unwrap();
    assert!(end.is_none(), "a coroutine returning None yields nothing");
    let consumed = Python::attach(|py| {
        g.bind(py).get_item("consumed").unwrap().unwrap().extract::<Vec<u32>>().unwrap()
    });
    assert_eq!(consumed, [160], "the node read the input pushed before it returned");
}
