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

use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::python::captured;
use lk_agents_abi::utils::clock::{Clock, Ts};
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{Closed, StreamSource};
use pyo3::prelude::*;

use super::python::{Started, captured_pair, start_in};

/// Reads one host item as `T`.
pub type Convert<T> = fn(&Bound<'_, PyAny>) -> PyResult<T>;

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
        let teardown = async move {
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
            }
        };
        Closed::bounded(&*clock, deadline, teardown).await
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

    // Catches: the host's CancelledError read as the end of the stream.
    #[tokio::test]
    async fn a_host_cancellation_is_a_failure_not_the_end() {
        Python::initialize();
        let (g, context) = Python::attach(setup);
        let mut stream = open(&g, &context, "cancelled");
        assert_eq!(stream.next().await.unwrap().as_deref(), Some("1"));
        assert!(stream.next().await.is_err());
    }
}
