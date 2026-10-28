//! A host async iterator consumed as a [`StreamSource`] (docs/internals/rust-core.md "Stream
//! lifecycle"): the host half of the one stream primitive.
//!
//! The stream owns one copy of the context it was opened in; every pull, the close and the
//! cancellation run inside that same copy on its loop, so a generator that opens a scope before a
//! yield closes it in the context it opened it in, and the consumer's context is never entered.
//! A pull is demand-driven (one `__anext__` per [`next`](StreamSource::next)) and is kept when its
//! caller stops awaiting: the next call resumes it instead of advancing the iterator again. The
//! host's `CancelledError` is a failure, never the end. Closing cancels a pull in flight, waits
//! for it to settle, then awaits `aclose()` where the iterator has one, all under the deadline.
//!
//! A duplex host stream ([`HostStream::duplex`]) adds its input half, a [`HostSink`] in the same
//! context: input is queued owned and bounded, and one drain scheduled on the loop at a time
//! converts and hands the queue to the host's `_core_push`/`_core_flush`/`_core_end_input` in
//! order. The caller never converts or waits; a full queue or a failed delivery refuses the next
//! push.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::python::captured;
use lk_agents_abi::utils::clock::{Clock, Ts};
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{Closed, StreamSink, StreamSource};
use pyo3::prelude::*;

use super::python::{Started, captured_pair, start_in};

/// Reads one host item as `T`.
pub type Convert<T> = fn(&Bound<'_, PyAny>) -> PyResult<T>;

/// Writes one input item as the host's.
pub type ToHost<I> = fn(Python<'_>, I) -> PyResult<Bound<'_, PyAny>>;

/// Input entries a host stream holds before its drain hands them over: past this, a push is
/// refused rather than queued.
pub const SINK_QUEUE: usize = 256;

pub struct HostStream<T> {
    iterator: Py<PyAny>,
    event_loop: Py<PyAny>,
    /// The stream's own context: a copy taken once, at open.
    ctx: Py<PyAny>,
    convert: Convert<T>,
    clock: Arc<dyn Clock>,
    /// The pull whose caller stopped awaiting it: the next one resumes it.
    pending: Option<Started>,
    ended: bool,
}

impl<T> HostStream<T> {
    /// The stream of `iterable` (its `__aiter__()`, taken inside the stream's context), opened in
    /// a copy of `context`.
    pub fn open(
        py: Python<'_>,
        iterable: &Bound<'_, PyAny>,
        context: &HostRef,
        convert: Convert<T>,
        clock: Arc<dyn Clock>,
    ) -> PyResult<Self> {
        let (event_loop, ctx) = captured_pair(py, context)?;
        let ctx = ctx.call_method0("copy")?;
        let iterator = ctx.call_method1("run", (iterable.getattr("__aiter__")?,))?;
        Ok(HostStream {
            iterator: iterator.unbind(),
            event_loop: event_loop.unbind(),
            ctx: ctx.unbind(),
            convert,
            clock,
            pending: None,
            ended: false,
        })
    }

    /// The two halves of the duplex stream `obj`: input through its `_core_push`, `_core_flush`
    /// and `_core_end_input`, output by iterating it, both in one copy of `context`.
    pub fn duplex<I>(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        context: &HostRef,
        to_host: ToHost<I>,
        convert: Convert<T>,
        clock: Arc<dyn Clock>,
    ) -> PyResult<(HostSink<I>, Self)> {
        let source = Self::open(py, obj, context, convert, clock)?;
        let sink = HostSink {
            shared: Arc::new(SinkShared {
                target: obj.clone().unbind(),
                event_loop: source.event_loop.clone_ref(py),
                ctx: source.ctx.clone_ref(py),
                to_host,
                queue: Mutex::new(SinkQueue {
                    entries: VecDeque::new(),
                    scheduled: false,
                    failed: None,
                }),
            }),
        };
        Ok((sink, source))
    }

    fn start(&self, method: &'static str) -> Result<Started, ProviderError> {
        Python::attach(|py| {
            let iterator = self.iterator.clone_ref(py);
            start_in(py, self.event_loop.bind(py), self.ctx.bind(py), move |py| {
                iterator.bind(py).call_method0(method)
            })
        })
        .map_err(captured)
    }
}

#[async_trait]
impl<T: Send + 'static> StreamSource<T> for HostStream<T> {
    async fn next(&mut self) -> Result<Option<T>, ProviderError> {
        if self.ended {
            return Ok(None);
        }
        let pull = match &mut self.pending {
            Some(pull) => pull,
            None => self.pending.insert(self.start("__anext__")?),
        };
        let pulled = pull.settled().await;
        self.pending = None;
        let item = Python::attach(|py| match pulled {
            Ok(item) => (self.convert)(item.bind(py)).map(Some),
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) => Ok(None),
            Err(e) => Err(e),
        });
        match item {
            Ok(Some(item)) => Ok(Some(item)),
            Ok(None) => {
                self.ended = true;
                Ok(None)
            }
            Err(e) => {
                self.ended = true;
                Err(captured(e))
            }
        }
    }

    /// A dropped pull is kept and resumed, never repeated.
    fn cancel_safe(&self) -> bool {
        true
    }

    async fn close(mut self: Box<Self>, deadline: Ts) -> Closed {
        let clock = self.clock.clone();
        let pending = self.pending.take();
        let has_aclose = Python::attach(|py| self.iterator.bind(py).hasattr("aclose"));
        let mut failed = false;
        let teardown = async {
            if let Some(mut pull) = pending {
                pull.cancel();
                let _ = pull.settled().await;
            }
            if !matches!(has_aclose, Ok(true)) {
                return;
            }
            let closed = match self.start("aclose") {
                Ok(mut closing) => closing.settled().await.map_err(captured),
                Err(e) => Err(e),
            };
            if let Err(e) = closed {
                log::warn!(target: "livekit.agents", "closing a host stream failed: {e}");
                failed = true;
            }
        };
        let closed = Closed::bounded(&*clock, deadline, teardown).await;
        if failed { closed.and(Closed::Failed) } else { closed }
    }
}

/// The input half of a duplex host stream.
pub struct HostSink<I> {
    shared: Arc<SinkShared<I>>,
}

struct SinkShared<I> {
    target: Py<PyAny>,
    event_loop: Py<PyAny>,
    ctx: Py<PyAny>,
    to_host: ToHost<I>,
    queue: Mutex<SinkQueue<I>>,
}

struct SinkQueue<I> {
    entries: VecDeque<Entry<I>>,
    /// A drain is scheduled on the loop and has not emptied the queue yet.
    scheduled: bool,
    /// The first failed delivery: every later push reports it.
    failed: Option<ProviderError>,
}

enum Entry<I> {
    Item(I),
    Flush,
    EndInput,
}

impl<I: Send + 'static> HostSink<I> {
    fn admit(&self, entry: Entry<I>) -> Result<(), ProviderError> {
        let schedule = {
            let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(failed) = &queue.failed {
                return Err(failed.clone());
            }
            if queue.entries.len() >= SINK_QUEUE {
                return Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(format!(
                    "the host stream is not taking its input: {SINK_QUEUE} entries are waiting"
                ))));
            }
            queue.entries.push_back(entry);
            !std::mem::replace(&mut queue.scheduled, true)
        };
        if schedule && let Err(e) = Python::attach(|py| self.schedule(py)) {
            let e = captured(e);
            self.shared.queue.lock().unwrap_or_else(|p| p.into_inner()).failed = Some(e.clone());
            return Err(e);
        }
        Ok(())
    }

    fn schedule(&self, py: Python<'_>) -> PyResult<()> {
        let shared = self.shared.clone();
        let drain = pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _| {
            drain(args.py(), &shared);
            PyResult::Ok(())
        })?;
        let s = &self.shared;
        s.event_loop
            .bind(py)
            .call_method1("call_soon_threadsafe", (s.ctx.bind(py).getattr("run")?, drain))?;
        Ok(())
    }
}

/// On the loop, inside the stream's context: everything queued, in order, until the queue is
/// empty; the first failure stops delivery and is kept for the next push.
fn drain<I>(py: Python<'_>, shared: &SinkShared<I>) {
    loop {
        let batch: Vec<Entry<I>> = {
            let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            if queue.entries.is_empty() {
                queue.scheduled = false;
                return;
            }
            queue.entries.drain(..).collect()
        };
        for entry in batch {
            let target = shared.target.bind(py);
            let delivered = match entry {
                Entry::Item(item) => {
                    (shared.to_host)(py, item).and_then(|v| target.call_method1("_core_push", (v,)))
                }
                Entry::Flush => target.call_method0("_core_flush"),
                Entry::EndInput => target.call_method0("_core_end_input"),
            };
            if let Err(e) = delivered {
                let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
                queue.failed = Some(lk_agents_abi::python::capture(py, &e));
                queue.entries.clear();
                queue.scheduled = false;
                return;
            }
        }
    }
}

impl<I: Send + Sync + 'static> StreamSink<I> for HostSink<I> {
    fn push(&self, item: I) -> Result<(), ProviderError> {
        self.admit(Entry::Item(item))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.admit(Entry::Flush)
    }

    /// A refused end is the failure the source reports.
    fn end_input(&self) {
        let _ = self.admit(Entry::EndInput);
    }
}

#[cfg(test)]
mod tests {
    //! Against a live interpreter and a running loop.
    use std::time::Duration;

    use lk_agents_abi::RealClock;
    use pyo3::types::PyDict;

    use super::*;

    const SETUP: &str = r#"
import asyncio, contextvars, threading
var = contextvars.ContextVar('var', default='unset')
log = []
loop = asyncio.new_event_loop()
ready = threading.Event()
def run():
    asyncio.set_event_loop(loop); loop.call_soon(ready.set); loop.run_forever()
threading.Thread(target=run, daemon=True).start()
ready.wait()

def capture(value):
    var.set(value)
    return (loop, contextvars.copy_context())

gate = asyncio.Event()

async def scoped():
    token = var.set('inside')
    try:
        yield var.get()
        await gate.wait()
        yield var.get()
    finally:
        log.append(('closed in', var.get()))
        var.reset(token)

async def cancelled():
    yield 1
    raise asyncio.CancelledError()

class Unclosable:
    def __aiter__(self):
        return self
    async def __anext__(self):
        return 'x'
    async def aclose(self):
        raise RuntimeError('cannot close')

def unclosable():
    return Unclosable()

class Duplex:
    def __init__(self):
        self.got = []
        self.ended = asyncio.Event()
    def __aiter__(self):
        var.set('stream')
        return self.output()
    async def output(self):
        await self.ended.wait()
        yield str(self.got)
    def _core_push(self, v):
        if v == 'bad':
            raise ValueError('refused')
        self.got.append((v, var.get()))
    def _core_flush(self):
        self.got.append(('flush', var.get()))
    def _core_end_input(self):
        self.got.append(('end', var.get()))
        self.ended.set()
"#;

    fn setup(py: Python<'_>) -> (Py<PyDict>, HostRef) {
        let g = PyDict::new(py);
        py.run(&std::ffi::CString::new(SETUP).unwrap(), Some(&g), None).unwrap();
        let pair = g.get_item("capture").unwrap().unwrap().call1(("caller",)).unwrap();
        g.set_item("pair", &pair).unwrap();
        (g.unbind(), lk_agents_abi::python::intern_host(pair.unbind()))
    }

    fn open(g: &Py<PyDict>, context: &HostRef, generator: &str) -> HostStream<String> {
        Python::attach(|py| {
            let iterable = g.bind(py).get_item(generator).unwrap().unwrap().call0().unwrap();
            let convert: Convert<String> = |o| Ok(o.str()?.to_string());
            HostStream::open(py, &iterable, context, convert, Arc::new(RealClock::new())).unwrap()
        })
    }

    fn deadline() -> Ts {
        RealClock::new().now().saturating_add(Duration::from_secs(2))
    }

    // Catches: a fresh context per pull (a scope opened before a yield lost after it), a dropped
    // pull advancing the iterator twice, or close not settling the blocked pull before `aclose`.
    #[tokio::test]
    async fn one_context_spans_pulls_and_close_and_a_dropped_pull_is_resumed() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let mut stream = open(&g, &context, "scoped");
        assert_eq!(stream.next().await.unwrap().as_deref(), Some("inside"));
        // The second pull blocks on the gate; its caller gives up, then a new pull resumes it.
        let blocked = tokio::time::timeout(Duration::from_millis(30), stream.next()).await;
        assert!(blocked.is_err());
        Python::attach(|py| {
            let g = g.bind(py);
            let set = g.get_item("gate").unwrap().unwrap().getattr("set").unwrap();
            g.get_item("loop")
                .unwrap()
                .unwrap()
                .call_method1("call_soon_threadsafe", (set,))
                .unwrap();
        });
        assert_eq!(stream.next().await.unwrap().as_deref(), Some("inside"));
        assert_eq!(Box::new(stream).close(deadline()).await, Closed::Clean);
        Python::attach(|py| {
            let log = g.bind(py).get_item("log").unwrap().unwrap();
            assert_eq!(log.str().unwrap().to_string(), "[('closed in', 'inside')]");
            // The captured context is as the caller left it.
            let g = g.bind(py);
            let var = g.get_item("var").unwrap().unwrap();
            let ctx = g.get_item("pair").unwrap().unwrap().get_item(1).unwrap();
            assert_eq!(ctx.get_item(var).unwrap().str().unwrap().to_string(), "caller");
        });
    }

    // Catches: closing a stream whose pull is blocked waiting behind that pull instead of
    // cancelling it.
    #[tokio::test]
    async fn close_cancels_a_blocked_pull() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let mut stream = open(&g, &context, "scoped");
        stream.next().await.unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(30), stream.next()).await.is_err());
        assert_eq!(Box::new(stream).close(deadline()).await, Closed::Clean);
        Python::attach(|py| {
            let log = g.bind(py).get_item("log").unwrap().unwrap();
            assert_eq!(log.str().unwrap().to_string(), "[('closed in', 'inside')]");
        });
    }

    fn duplex(g: &Py<PyDict>, context: &HostRef) -> (HostSink<String>, HostStream<String>) {
        Python::attach(|py| {
            let obj = g.bind(py).get_item("Duplex").unwrap().unwrap().call0().unwrap();
            let to_host: ToHost<String> = |py, s| pyo3::IntoPyObjectExt::into_bound_py_any(s, py);
            let convert: Convert<String> = |o| Ok(o.str()?.to_string());
            HostStream::duplex(py, &obj, context, to_host, convert, Arc::new(RealClock::new()))
                .unwrap()
        })
    }

    // Catches: input delivered out of order, outside the stream's context, flush or end of input
    // lost among the items, or ending the input closing the output instead of letting it drain.
    #[tokio::test]
    async fn duplex_input_reaches_the_host_in_order_in_the_streams_context() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let (sink, mut source) = duplex(&g, &context);
        sink.push("a".into()).unwrap();
        sink.flush().unwrap();
        sink.push("b".into()).unwrap();
        sink.end_input();
        assert_eq!(
            source.next().await.unwrap().as_deref(),
            Some("[('a', 'stream'), ('flush', 'stream'), ('b', 'stream'), ('end', 'stream')]")
        );
    }

    // Catches: a delivery the host refused dropped silently, input still accepted after it.
    #[tokio::test]
    async fn a_refused_delivery_refuses_the_next_push() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let (sink, _source) = duplex(&g, &context);
        sink.push("bad".into()).unwrap();
        for _ in 0..200 {
            if sink.push("next".into()).is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("the refused delivery was never reported");
    }

    // Catches: the host's CancelledError read as the end of the stream.
    #[tokio::test]
    async fn a_host_cancellation_is_a_failure_not_the_end() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let mut stream = open(&g, &context, "cancelled");
        assert_eq!(stream.next().await.unwrap().as_deref(), Some("1"));
        assert!(stream.next().await.is_err());
    }

    // Catches: a host stream whose `aclose()` raised reported as closed cleanly (B-10).
    #[tokio::test]
    async fn a_failed_aclose_is_a_failed_close() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let mut stream = open(&g, &context, "unclosable");
        assert_eq!(stream.next().await.unwrap().as_deref(), Some("x"));
        assert_eq!(Box::new(stream).close(deadline()).await, Closed::Failed);
    }
}
