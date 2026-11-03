//! Python async iterators and native sources crossing the binding boundary.

/// Interpreter interface namespaces. Generated inherent methods perform the declared
/// imports and member access; they accept Python objects without asserting their runtime type.
pub mod interfaces {
    #[lk_agents_abi::host(interface(
        host = "collections.abc.AsyncIterable",
        attrs(aiter(name = "__aiter__"))
    ))]
    pub struct AsyncIterable;

    #[lk_agents_abi::host(interface(
        host = "collections.abc.AsyncIterator",
        attrs(anext(name = "__anext__"))
    ))]
    pub struct AsyncIterator;

    #[lk_agents_abi::host(interface(
        protocol = "supports aclose()",
        attrs(aclose),
        protocols(matches(required(aclose)))
    ))]
    pub struct AsyncClose;

    #[lk_agents_abi::host(interface(
        protocol = "framework host stream input",
        attrs(
            push(name = "_core_push"),
            flush(name = "_core_flush"),
            end_input(name = "_core_end_input")
        )
    ))]
    pub struct StreamInput;
}

use crate::generated::accessors as access;
use crate::py_calls::Started;
use crate::py_calls::captured_pair;
use crate::py_calls::start_in;
use crate::py_errors::captured;
use lk_agents_abi::Closed;
use lk_agents_abi::StreamInput;
use lk_agents_abi::StreamOutput;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::clock::Clock;
use lk_agents_abi::utils::clock::Ts;
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_runtime::net::source::Pulled;
use lk_agents_runtime::net::source::SharedStream;
use parking_lot::Mutex;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    input_stopped: Arc<AtomicBool>,
    failure: tokio::sync::watch::Sender<Option<ProviderError>>,
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
        let ctx = access::Context::copy(&ctx)?.call0()?;
        Self::open_in(iterable, &event_loop, &ctx, convert, clock)
    }

    /// An iterable returned by a call whose context must also own every pull and its close.
    pub(crate) fn open_in(
        iterable: &Bound<'_, PyAny>,
        event_loop: &Bound<'_, PyAny>,
        ctx: &Bound<'_, PyAny>,
        convert: Convert<T>,
        clock: Arc<dyn Clock>,
    ) -> PyResult<Self> {
        let iterator =
            access::Context::run(ctx)?.call1((access::AsyncIterable::aiter(iterable)?,))?;
        Ok(HostStream {
            iterator: iterator.unbind(),
            event_loop: event_loop.clone().unbind(),
            ctx: ctx.clone().unbind(),
            convert,
            clock,
            pending: None,
            ended: false,
            input_stopped: Arc::new(AtomicBool::new(false)),
            failure: tokio::sync::watch::channel(None).0,
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
    ) -> PyResult<lk_agents_abi::DuplexStream<HostSink<I>, Self>> {
        let source = Self::open(py, obj, context, convert, clock)?;
        let sink = HostSink {
            shared: Arc::new(SinkShared {
                target: obj.clone().unbind(),
                event_loop: source.event_loop.clone_ref(py),
                ctx: source.ctx.clone_ref(py),
                to_host,
                stopped: source.input_stopped.clone(),
                failure: source.failure.clone(),
                queue: Mutex::new(SinkQueue {
                    entries: VecDeque::new(),
                    scheduled: false,
                    ended: false,
                }),
            }),
        };
        Ok(lk_agents_abi::DuplexStream::new(sink, source))
    }

    fn start(&self, close: bool) -> Result<Started, ProviderError> {
        Python::attach(|py| {
            let iterator = self.iterator.clone_ref(py);
            start_in(py, self.event_loop.bind(py), self.ctx.bind(py), move |py| {
                let callable = if close {
                    access::AsyncClose::aclose(iterator.bind(py))?
                } else {
                    access::AsyncIterator::anext(iterator.bind(py))?
                };
                callable.call0()
            })
        })
        .map_err(captured)
    }
}

impl<T: Send + 'static> StreamOutput<T> for HostStream<T> {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move {
            if self.ended {
                return Ok(None);
            }
            let mut failure = self.failure.subscribe();
            if let Some(error) = failure.borrow().clone() {
                self.ended = true;
                self.input_stopped.store(true, Ordering::Release);
                return Err(error);
            }
            let pull = match &mut self.pending {
                Some(pull) => pull,
                None => self.pending.insert(self.start(false)?),
            };
            let pulled = tokio::select! {
                biased;
                failed = failure.wait_for(|error| error.is_some()) => {
                    let error = failed.expect("the stream owns its failure sender").clone().expect("waited for failure");
                    self.ended = true;
                    self.input_stopped.store(true, Ordering::Release);
                    return Err(error);
                },
                pulled = pull.settled() => pulled,
            };
            self.pending = None;
            let item = Python::attach(|py| match pulled {
                Ok(item) => (self.convert)(item.bind(py)).map(Some),
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) => {
                    Ok(None)
                }
                Err(e) => Err(e),
            });
            match item {
                Ok(Some(item)) => Ok(Some(item)),
                Ok(None) => {
                    self.ended = true;
                    self.input_stopped.store(true, Ordering::Release);
                    Ok(None)
                }
                Err(e) => {
                    self.ended = true;
                    self.input_stopped.store(true, Ordering::Release);
                    Err(captured(e))
                }
            }
        })
    }

    /// A dropped pull is kept and resumed, never repeated.
    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.input_stopped.store(true, Ordering::Release);
            let clock = self.clock.clone();
            let pending = self.pending.take();
            let has_aclose =
                Python::attach(|py| access::AsyncClose::matches(self.iterator.bind(py)));
            let mut failed = false;
            let teardown = async {
                if let Some(mut pull) = pending {
                    pull.cancel();
                    let _ = pull.settled().await;
                }
                match has_aclose {
                    Ok(true) => {}
                    Ok(false) => return,
                    Err(error) => {
                        log::warn!(target: "livekit.agents", "looking up host stream cleanup failed: {error}");
                        failed = true;
                        return;
                    }
                }
                let closed = match self.start(true) {
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
        })
    }
}

impl<T: Send + 'static> lk_agents_abi::utils::stream::ProviderStream<T> for HostStream<T> {}

/// The input half of a duplex host stream.
pub struct HostSink<I> {
    shared: Arc<SinkShared<I>>,
}

struct SinkShared<I> {
    target: Py<PyAny>,
    event_loop: Py<PyAny>,
    ctx: Py<PyAny>,
    to_host: ToHost<I>,
    stopped: Arc<AtomicBool>,
    failure: tokio::sync::watch::Sender<Option<ProviderError>>,
    queue: Mutex<SinkQueue<I>>,
}

struct SinkQueue<I> {
    entries: VecDeque<Entry<I>>,
    /// A drain is scheduled on the loop and has not emptied the queue yet.
    scheduled: bool,
    ended: bool,
}

enum Entry<I> {
    Item(I),
    Flush,
    EndInput,
}

impl<I: Send + 'static> HostSink<I> {
    fn admit(&self, entry: Entry<I>) -> Result<(), ProviderError> {
        let schedule = {
            let mut queue = self.shared.queue.lock();
            if let Some(failed) = self.shared.failure.borrow().clone() {
                return Err(failed);
            }
            if self.shared.stopped.load(Ordering::Acquire) || queue.ended {
                return Err(lk_agents_abi::errors::ApiError::new("stream input is closed").into());
            }
            let ending = matches!(entry, Entry::EndInput);
            if !ending && queue.entries.len() >= SINK_QUEUE {
                return Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(format!(
                    "the host stream is not taking its input: {SINK_QUEUE} entries are waiting"
                ))));
            }
            queue.ended |= ending;
            queue.entries.push_back(entry);
            !std::mem::replace(&mut queue.scheduled, true)
        };
        if schedule && let Err(e) = Python::attach(|py| self.schedule(py)) {
            let e = captured(e);
            self.shared.failure.send_replace(Some(e.clone()));
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
        access::EventLoop::call_soon_threadsafe(s.event_loop.bind(py))?
            .call1((access::Context::run(s.ctx.bind(py))?, drain))?;
        Ok(())
    }
}

/// On the loop, inside the stream's context: everything queued, in order, until the queue is
/// empty; the first failure stops delivery and is kept for the next push.
fn drain<I>(py: Python<'_>, shared: &SinkShared<I>) {
    loop {
        if shared.stopped.load(Ordering::Acquire) {
            let _dropped = {
                let mut queue = shared.queue.lock();
                queue.scheduled = false;
                std::mem::take(&mut queue.entries)
            };
            return;
        }
        let batch: Vec<Entry<I>> = {
            let mut queue = shared.queue.lock();
            if queue.entries.is_empty() {
                queue.scheduled = false;
                return;
            }
            queue.entries.drain(..).collect()
        };
        for entry in batch {
            let target = shared.target.bind(py);
            let delivered = match entry {
                Entry::Item(item) => (shared.to_host)(py, item)
                    .and_then(|v| access::StreamInput::push(target)?.call1((v,))),
                Entry::Flush => access::StreamInput::flush(target).and_then(|call| call.call0()),
                Entry::EndInput => {
                    access::StreamInput::end_input(target).and_then(|call| call.call0())
                }
            };
            if let Err(e) = delivered {
                // Capturing reads the exception's attributes: Python code, run before the lock.
                let failed = crate::py_errors::capture(py, &e);
                // Dropped after the guard: an item's host objects may run Python finalizers.
                shared.failure.send_replace(Some(failed));
                let _dropped = {
                    let mut queue = shared.queue.lock();
                    queue.scheduled = false;
                    std::mem::take(&mut queue.entries)
                };
                return;
            }
        }
    }
}

impl<I: Send + Sync + 'static> StreamInput<I> for HostSink<I> {
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
pub(crate) mod tests {
    //! Against a live interpreter and a running loop.
    use super::*;
    use std::time::Duration;

    use lk_agents_runtime::net::clock::RealClock;

    pub(crate) fn setup(py: Python<'_>) -> (Py<PyAny>, HostRef) {
        let g = crate::py_test_support::fixture(py, "streams");
        let pair = g.getattr("capture").unwrap().call1(("caller",)).unwrap();
        g.setattr("pair", &pair).unwrap();
        (g.unbind(), crate::py_objects::retain(pair.unbind()))
    }

    fn open(g: &Py<PyAny>, context: &HostRef, generator: &str) -> HostStream<String> {
        Python::attach(|py| {
            let iterable = g.bind(py).getattr(generator).unwrap().call0().unwrap();
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
            let set = g.getattr("gate").unwrap().getattr("set").unwrap();
            g.getattr("loop").unwrap().call_method1("call_soon_threadsafe", (set,)).unwrap();
        });
        assert_eq!(stream.next().await.unwrap().as_deref(), Some("inside"));
        assert_eq!(Box::new(stream).close(deadline()).await, Closed::Clean);
        Python::attach(|py| {
            let log = g.bind(py).getattr("log").unwrap();
            assert_eq!(log.str().unwrap().to_string(), "[('closed in', 'inside')]");
            // The captured context is as the caller left it.
            let g = g.bind(py);
            let var = g.getattr("var").unwrap();
            let ctx = g.getattr("pair").unwrap().get_item(1).unwrap();
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
            let log = g.bind(py).getattr("log").unwrap();
            assert_eq!(log.str().unwrap().to_string(), "[('closed in', 'inside')]");
        });
    }

    fn duplex(g: &Py<PyAny>, context: &HostRef) -> (HostSink<String>, HostStream<String>) {
        Python::attach(|py| {
            let obj = g.bind(py).getattr("Duplex").unwrap().call0().unwrap();
            let to_host: ToHost<String> = |py, s| pyo3::IntoPyObjectExt::into_bound_py_any(s, py);
            let convert: Convert<String> = |o| Ok(o.str()?.to_string());
            HostStream::duplex(py, &obj, context, to_host, convert, Arc::new(RealClock::new()))
                .unwrap()
                .split()
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
        assert!(sink.push("late".into()).is_err());
        assert!(sink.flush().is_err());
        assert_eq!(
            source.next().await.unwrap().as_deref(),
            Some("[('a', 'stream'), ('flush', 'stream'), ('b', 'stream'), ('end', 'stream')]")
        );
    }

    #[tokio::test]
    async fn closing_output_refuses_input_without_delivering_queued_items() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let (sink, source) = duplex(&g, &context);
        assert_eq!(Box::new(source).close(deadline()).await, Closed::Clean);
        assert!(sink.push("late".into()).is_err());
        assert!(sink.flush().is_err());
    }

    // Catches: a delivery the host refused dropped silently, input still accepted after it.
    #[tokio::test]
    async fn a_refused_delivery_wakes_the_output_and_refuses_the_next_push() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let (sink, mut source) = duplex(&g, &context);
        sink.push("bad".into()).unwrap();
        let error = tokio::time::timeout(Duration::from_secs(2), source.next())
            .await
            .expect("input failure must wake output")
            .unwrap_err();
        assert!(error.to_string().contains("refused"));
        assert!(sink.push("next".into()).is_err());
        assert_eq!(Box::new(source).close(deadline()).await, Closed::Clean);
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

    #[tokio::test]
    async fn a_failed_cleanup_lookup_is_a_failed_close() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let stream = open(&g, &context, "BrokenCleanup");
        assert_eq!(Box::new(stream).close(deadline()).await, Closed::Failed);
    }
}

pub fn recv_native<'py, T: Send + 'static>(
    source: &SharedStream<T>,
    py: Python<'py>,
    to_py: fn(Python<'_>, T) -> PyResult<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let source = source.clone();
    crate::py_calls::future_into_py(py, async move {
        match source.pull().await {
            Ok(Pulled::Item(item)) => Python::attach(|py| to_py(py, item)).map(Some),
            Ok(Pulled::End | Pulled::Cancelled) => Ok(None),
            Err(error) => Err(crate::py_errors::HostException::to_py_err(&error)),
        }
    })
}

pub fn close_native<'py, T: Send + 'static>(
    source: &SharedStream<T>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let source = source.clone();
    crate::py_calls::future_into_py(py, async move {
        if source.aclose().await == lk_agents_abi::Closed::TimedOut {
            log::error!(target: "livekit.agents", "a native stream did not release its resources in time");
        }
        Ok(())
    })
}

#[cfg(test)]
mod boundary_tests {
    //! Python node streams and binding behavior under GIL contention.

    mod nodes {
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

        use crate::generated::voice::SessionNodesProxy;

        struct Destination;

        impl TextOutput for Destination {
            fn label(&self) -> &str {
                "destination"
            }
            fn capture_text<'a>(
                &'a self,
                _: &'a str,
            ) -> lk_agents_abi::BoxFuture<'a, Result<(), IoError>> {
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

        /// The agent nodes over the host fixture's `processing` object, on this test's loop.
        fn nodes(scenario: &str) -> (Py<PyAny>, AgentNodes) {
            Python::initialize();
            let (g, captured): (Py<PyAny>, HostRef) =
                Python::attach(crate::py_streams::tests::setup);
            let host = Python::attach(|py| {
                crate::py_test_support::fixture(py, "install_node_types");
                crate::py_test_support::call(py, scenario, (g.bind(py),)).unwrap();
                let processing = g.bind(py).getattr("processing").unwrap();
                SessionNodesProxy::in_context(&processing, &captured).unwrap()
            });
            (g, AgentNodes::new(&ProcessEnv::current_runtime(), host))
        }

        async fn flag(g: &Py<PyAny>, name: &str) {
            tokio::time::timeout(Duration::from_secs(2), async {
                while !Python::attach(|py| {
                    let event = g.bind(py).getattr(name).unwrap();
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
            let (g, nodes) =
                nodes("text_node_cancellation_retains_its_binding_until_python_finally_finishes");
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
                let set = g.getattr("close_gate").unwrap().getattr("set").unwrap();
                let event_loop = g.getattr("loop").unwrap();
                event_loop.call_method1("call_soon_threadsafe", (set,)).unwrap();
            });
            if was_pending {
                tokio::time::timeout(Duration::from_secs(2), receipt)
                    .await
                    .expect("the text receipt did not settle after the node's cleanup")
                    .unwrap();
            }
            flag(&g, "cleaned").await;
            assert!(
                was_pending,
                "the node completed while its Python finally block was still open"
            );
            assert!(was_retained, "the binding owner was dropped before the node's cleanup");
            assert!(released.load(Ordering::SeqCst), "the completed node retained its binding");
        }

        // Catches: the leg awaiting the node's output before handing back its input sink, which
        // deadlocks a coroutine node (`_node_output` awaits it on the first pull) that reads its whole
        // input before it returns.
        #[tokio::test]
        async fn coroutine_node_receives_its_input_before_returning() {
            let (g, nodes) = nodes("coroutine_node_receives_its_input_before_returning");
            let (sink, mut output) = tokio::time::timeout(Duration::from_secs(2), nodes.stt_node())
                .await
                .expect("the leg awaited the node's output before returning its input")
                .unwrap()
                .split();
            sink.push(AudioFrame::new(vec![0u8; 320], 16_000, 1, 160)).unwrap();
            sink.end_input();
            let end = tokio::time::timeout(Duration::from_secs(2), output.next())
                .await
                .expect("the node did not return once its input ended")
                .unwrap();
            assert!(end.is_none(), "a coroutine returning None yields nothing");
            let consumed = Python::attach(|py| {
                g.bind(py).getattr("consumed").unwrap().extract::<Vec<u32>>().unwrap()
            });
            assert_eq!(consumed, [160], "the node read the input pushed before it returned");
        }
    }

    mod gil {
        //! The binding boundary under contention: a Python thread holding the GIL calls a generated
        //! synchronous entry whose Rust side waits for a lock, while the Rust task holding that lock waits
        //! for the GIL (review13's deadlock class, decision 129).

        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        use lk_agents_abi::errors::ProviderError;
        use lk_agents_abi::{
            BoxFuture, StreamInput, StreamOutput, SynthesizedAudio, Tts, TtsCapabilities,
        };
        use lk_agents_abi::{Closed, Ts, TtsSink, TtsSource};
        use lk_agents_runtime::inference::fallback::{ChainPolicy, FallbackTts, IgnoreEvents};
        use lk_agents_runtime::inference::tts::stream::TtsStream;
        use pyo3::prelude::*;

        /// A host TTS as the fallback chain sees it: its stream opens once `opened` fires, and its sink
        /// takes the GIL on every push, as a host sink scheduling its drain does.
        struct HostLikeTts {
            opened: tokio::sync::watch::Receiver<bool>,
            pushing: mpsc::SyncSender<()>,
        }

        struct HostLikeSink(mpsc::SyncSender<()>);

        impl StreamInput<String> for HostLikeSink {
            fn push(&self, _item: String) -> Result<(), ProviderError> {
                let _ = self.0.try_send(());
                Python::attach(|_| ());
                Ok(())
            }
            fn flush(&self) -> Result<(), ProviderError> {
                Ok(())
            }
            fn end_input(&self) {}
        }

        struct Silent;

        impl StreamOutput<SynthesizedAudio> for Silent {
            fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
                Box::pin(std::future::pending())
            }
            fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
                Box::pin(async { Closed::Clean })
            }
        }

        impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for Silent {}

        impl Tts for HostLikeTts {
            fn label(&self) -> String {
                "host-like".into()
            }
            fn capabilities(&self) -> TtsCapabilities {
                TtsCapabilities { streaming: true, aligned_transcript: false }
            }
            fn sample_rate(&self) -> u32 {
                24_000
            }
            fn num_channels(&self) -> u16 {
                1
            }
            fn synthesize<'a>(
                &'a self,
                _text: &'a str,
            ) -> BoxFuture<'a, Result<TtsSource, ProviderError>> {
                unreachable!("the test only streams")
            }
            fn stream(
                &self,
                _expressive: bool,
                _trace: lk_agents_abi::telemetry::record::TraceContext,
            ) -> BoxFuture<'_, Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>>
            {
                let mut opened = self.opened.clone();
                let pushing = self.pushing.clone();
                Box::pin(async move {
                    opened.wait_for(|opened| *opened).await.unwrap();
                    Ok(lk_agents_abi::DuplexStream::new(
                        Box::new(HostLikeSink(pushing)) as TtsSink,
                        Box::new(Silent) as TtsSource,
                    ))
                })
            }
        }

        // Catches: a generated sync entry (`TtsStream.push_text`) keeping the GIL while it waits for the
        // fallback's token log, whose holder (the chain pump replaying text into a host TTS) waits for
        // the GIL.
        #[test]
        fn pushing_text_while_the_fallback_replays_into_a_host_tts_does_not_deadlock() {
            Python::initialize();
            let (open, opened) = tokio::sync::watch::channel(false);
            let (pushing, pushed) = mpsc::sync_channel(1);
            let child: Arc<dyn Tts> = Arc::new(HostLikeTts { opened, pushing });
            let runtime = lk_agents_runtime::process::handle().unwrap();
            let fallback = FallbackTts::new(
                vec![child],
                None,
                ChainPolicy::tts_defaults(),
                lk_agents_runtime::process::host_clock(),
                lk_agents_runtime::process::host_spawner().unwrap(),
                Arc::new(IgnoreEvents),
                lk_agents_runtime::telemetry::tracer(),
            )
            .unwrap();
            let (sink, source) =
                runtime.block_on(fallback.stream(false, Default::default())).unwrap().split();
            let stream = Arc::new(TtsStream::new(
                Some(sink),
                source,
                lk_agents_runtime::process::host_clock(),
            ));
            lk_agents_abi::tts::streams::TtsStream::push_text(&*stream, "replayed".into()).unwrap();

            let (attached, attaching) = mpsc::channel();
            let (done, finished) = mpsc::channel();
            std::thread::spawn(move || {
                Python::attach(|py| {
                    let stream = Py::new(py, crate::generated::tts::TtsStream(stream)).unwrap();
                    attached.send(()).unwrap();
                    // The pump now holds the log, replaying into the host TTS, which waits for the GIL.
                    pushed.recv_timeout(Duration::from_secs(10)).expect("the pump replays");
                    stream.bind(py).call_method1("push_text", ("live",)).unwrap();
                });
                done.send(()).unwrap();
            });
            attaching.recv().unwrap();
            open.send_replace(true);
            assert!(finished.recv_timeout(Duration::from_secs(10)).is_ok(), "deadlocked");
        }
    }
}
