//! The one bridge between host iterables and the core's streams, both ways: a host async iterable,
//! awaitable or value drained by the core ([`HostSource`]: a node's output, a callable's answer, a
//! provider's stream), awaited in the context it was opened in so a node sees its agent's binding;
//! and a core channel read by the host as an async iterator ([`ItemStream`], fed through
//! [`ItemSink`]).

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use tokio::sync::mpsc;

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
#[rustfmt::skip]
use super::{
    AsyncMutex, CancellingHandle, HostBridge as _, capture, captured, provider_error_to_py,
};
use crate::errors::ProviderError;

type Item = Result<Py<PyAny>, ProviderError>;

/// `_core.ItemStream`: what the core hands a node (its text, its audio) or a default node hands
/// back (frames, speech events), as `async for` reads it. Ends when the core ends it; a failure
/// is raised where it happened.
#[pyclass(frozen, name = "ItemStream", module = "livekit.agents._core")]
pub struct ItemStream {
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    rx: Arc<AsyncMutex<Option<mpsc::UnboundedReceiver<Item>>>>,
}

crate::register_host!(class ItemStream);

#[pymethods]
impl ItemStream {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = Arc::clone(&self.rx);
        super::future_into_py(py, async move {
            let mut rx = rx.lock().await;
            let next = match rx.as_mut() {
                Some(rx) => rx.recv().await,
                None => None,
            };
            match next {
                Some(Ok(item)) => Ok(item),
                Some(Err(e)) => Err(provider_error_to_py(e)),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }

    /// Stop reading: what feeds it stops at its next item.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = Arc::clone(&self.rx);
        super::future_into_py(py, async move {
            rx.lock().await.take();
            Ok(())
        })
    }
}

/// The core's end of an [`ItemStream`]: each item converted to its host object as it is pushed.
pub struct ItemSink<I> {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    tx: Mutex<Option<mpsc::UnboundedSender<Item>>>,
    to_py: fn(Python<'_>, &I) -> PyResult<Py<PyAny>>,
}

impl<I> ItemSink<I> {
    /// Whether the host still reads.
    pub fn send(&self, item: &I) -> bool {
        self.try_send(item).expect("ItemSink state is poisoned")
    }

    fn try_send(&self, item: &I) -> Result<bool, ProviderError> {
        let tx = self
            .tx
            .lock()
            .map_err(|poison| {
                drop(poison);
                captured(pyo3::exceptions::PyRuntimeError::new_err("ItemSink state is poisoned"))
            })?
            .clone();
        let Some(tx) = tx else {
            return Ok(false);
        };
        let converted = Python::attach(|py| (self.to_py)(py, item)).map_err(captured);
        Ok(tx.send(converted).is_ok())
    }

    /// Raised on the host's next read; nothing follows it.
    pub fn fail(&self, error: ProviderError) {
        if let Some(tx) = self.tx.lock().expect("ItemSink state is poisoned").take() {
            let _ = tx.send(Err(error));
        }
    }

    /// Resolves once the host stopped reading (closed or dropped the stream).
    pub async fn closed(&self) {
        let tx = self.tx.lock().expect("ItemSink state is poisoned").clone();
        if let Some(tx) = tx {
            tx.closed().await;
        }
    }
}

impl<I: Send + Sync> crate::StreamSink<I> for ItemSink<I> {
    /// A host that stopped reading drops the rest, as a node that returned early ignores its input.
    fn push(&self, item: I) -> Result<(), ProviderError> {
        self.try_send(&item)?;
        Ok(())
    }

    /// The reference's node input has no segment marker.
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        self.tx.lock().expect("ItemSink state is poisoned").take();
    }
}

#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// A channel the core feeds and the host reads.
pub fn item_stream<I>(
    py: Python<'_>,
    to_py: fn(Python<'_>, &I) -> PyResult<Py<PyAny>>,
) -> PyResult<(ItemSink<I>, Py<ItemStream>)> {
    let (tx, rx) = mpsc::unbounded_channel();
    let stream = Py::new(py, ItemStream { rx: Arc::new(AsyncMutex::new(Some(rx))) })?;
    Ok((ItemSink { tx: Mutex::new(Some(tx)), to_py }, stream))
}

/// Turns one item of a node's output into the core's.
pub type Convert<T> = Box<dyn Fn(&Bound<'_, PyAny>) -> PyResult<T> + Send + Sync>;

#[allow(clippy::disallowed_types, reason = "S0: legacy host channel state")]
enum State {
    /// What the node returned, not yet looked at: an async iterable, an awaitable, one value or
    /// `None`.
    Returned(Py<PyAny>),
    Iterating(Py<PyAny>),
    /// Read to its end, kept for its `_started_time` and its teardown.
    Ended(Py<PyAny>),
    /// An `ItemStream` handed on as it is (a node passing its input to a default node): read
    /// from the core's channel directly.
    Channel(Arc<AsyncMutex<Option<mpsc::UnboundedReceiver<Item>>>>),
    Done,
}

/// The provider calls of the default nodes a node runs, as the reference meters a TTS's streams
/// and not the node's output.
#[derive(Clone, Default)]
pub enum NodeCallState {
    /// No default node: the node makes its own audio, which no provider call stands behind.
    #[default]
    Unused,
    /// A provider stream that is one call: the node's output stands for it.
    Streamed,
    /// Providers that meter their own calls (a `StreamAdapter`'s sentences).
    Metered(Vec<crate::utils::stream::CallLedger>),
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
pub type NodeCalls = Arc<Mutex<NodeCallState>>;

/// The one host-iterable bridge: what a host callable returned (a node's output, a callable's
/// answer) or a host stream (a provider's), drained by the core as a [`crate::StreamSource`], as
/// the reference's node task drains it: an awaitable is awaited first, an async iterable is read
/// to its end, a single value is the one item, `None` is none. Every await runs in `locals`'
/// context.
pub struct HostSource<T> {
    state: State,
    locals: pyo3_async_runtimes::TaskLocals,
    convert: Convert<T>,
    clock: Arc<dyn crate::Clock>,
    calls: Option<NodeCalls>,
}

impl<T> HostSource<T> {
    /// What a host callable returned, in `locals`' context.
    pub fn returned(
        returned: Py<PyAny>,
        locals: pyo3_async_runtimes::TaskLocals,
        convert: Convert<T>,
        clock: Arc<dyn crate::Clock>,
    ) -> Self {
        HostSource {
            state: State::Returned(returned),
            locals,
            convert,
            clock,
            calls: None,
        }
    }

    /// The calls the default nodes it runs make, reported in place of its own output.
    pub fn with_calls(mut self, calls: NodeCalls) -> Self {
        self.calls = Some(calls);
        self
    }

    /// Its awaits are cancelled when the core stops waiting.
    fn handle(&self, obj: &Py<PyAny>) -> CancellingHandle {
        Python::attach(|py| CancellingHandle::from_parts(obj.clone_ref(py), self.locals.clone()))
    }

    fn convert(&self, item: Py<PyAny>) -> Result<Option<T>, ProviderError> {
        Python::attach(|py| (self.convert)(item.bind(py)).map(Some).map_err(|e| capture(py, &e)))
    }
}

/// A host async iterable handed to the core as an argument (`say(AsyncIterable[str])`), drained on
/// the running loop in the caller's context.
impl<'a, 'py, T> FromPyObject<'a, 'py> for Box<dyn crate::StreamSource<T>>
where
    T: Send + 'static,
    T: for<'x, 'y> FromPyObject<'x, 'y>,
    for<'x, 'y> <T as FromPyObject<'x, 'y>>::Error: Into<PyErr>,
{
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        let locals = pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?;
        let clock = <Arc<dyn crate::Clock> as super::Ambient>::ambient();
        let convert: Convert<T> = Box::new(|i| i.extract::<T>().map_err(Into::into));
        Ok(Box::new(HostSource::returned(obj.to_owned().unbind(), locals, convert, clock)))
    }
}

#[allow(clippy::disallowed_types, reason = "S0: legacy host channel state")]
enum Look {
    Channel(Arc<AsyncMutex<Option<mpsc::UnboundedReceiver<Item>>>>),
    Iterate(Py<PyAny>),
    Await,
    One(Py<PyAny>),
    Nothing,
}

impl<T: Send + 'static> crate::StreamSource<T> for HostSource<T> {
    /// `StopAsyncIteration` and a cancelled await are the end (an end leaves no side-table entry);
    /// every other exception is captured, so the original can be re-raised.
    fn next(&mut self) -> crate::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move {
            loop {
                match std::mem::replace(&mut self.state, State::Done) {
                    State::Done => return Ok(None),
                    State::Channel(rx) => {
                        let next = match rx.lock().await.as_mut() {
                            Some(rx) => rx.recv().await,
                            None => None,
                        };
                        let Some(item) = next else { return Ok(None) };
                        self.state = State::Channel(rx);
                        return self.convert(item?);
                    }
                    State::Ended(iter) => {
                        self.state = State::Ended(iter);
                        return Ok(None);
                    }
                    // Kept in place across the await: a read dropped mid-await still leaves the
                    // iterator to `close`.
                    State::Iterating(iter) => {
                        let handle = self.handle(&iter);
                        self.state = State::Iterating(iter);
                        let next = handle.anext().await;
                        let ended = |s: &mut Self| {
                            if let State::Iterating(iter) =
                                std::mem::replace(&mut s.state, State::Done)
                            {
                                s.state = State::Ended(iter);
                            }
                        };
                        return match next {
                            Ok(Some(item)) => self.convert(item),
                            Ok(None) => {
                                ended(self);
                                Ok(None)
                            }
                            Err(e) => {
                                ended(self);
                                Err(captured(e))
                            }
                        };
                    }
                    State::Returned(out) => {
                        let look = Python::attach(|py| -> PyResult<Look> {
                            let obj = out.bind(py);
                            Ok(if obj.is_none() {
                                Look::Nothing
                            } else if let Ok(stream) = obj.cast::<ItemStream>() {
                                Look::Channel(Arc::clone(&stream.get().rx))
                            } else if obj.hasattr("__aiter__")? {
                                Look::Iterate(obj.call_method0("__aiter__")?.unbind())
                            } else if obj.hasattr("__await__")? {
                                Look::Await
                            } else {
                                Look::One(out.clone_ref(py))
                            })
                        })
                        .map_err(captured)?;
                        match look {
                            Look::Nothing => return Ok(None),
                            Look::Channel(rx) => self.state = State::Channel(rx),
                            Look::Iterate(iter) => self.state = State::Iterating(iter),
                            Look::One(item) => return self.convert(item),
                            Look::Await => match self.handle(&out).await_call(|o| Ok(o.clone())).await {
                                Ok(value) => self.state = State::Returned(value),
                                Err(e) => {
                                    let cancelled = Python::attach(|py| {
                                        e.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py)
                                    });
                                    return if cancelled { Ok(None) } else { Err(captured(e)) };
                                }
                            },
                        }
                    }
                }
            }
        })
    }

    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<crate::metrics::base::AgentMetrics>> {
        let calls = self.calls.as_ref()?;
        let calls = calls.lock().expect("NodeCallState is poisoned").clone();
        match calls {
            NodeCallState::Unused => in_flight.then(Vec::new),
            NodeCallState::Streamed => None,
            NodeCallState::Metered(ledgers) => {
                Some(ledgers.iter().flat_map(|take| take(in_flight)).collect())
            }
        }
    }

    /// `SynthesizeStream._started_time`, which is `0` until `_mark_started()`; an iterator without
    /// the attribute (not a synthesis stream) has none. A `perf_counter()` stamp, read as
    /// wall-clock seconds.
    fn started_time(&self) -> Option<f64> {
        let (State::Iterating(iter) | State::Ended(iter)) = &self.state else { return None };
        let read = Python::attach(|py| -> PyResult<(f64, f64)> {
            let started = iter.bind(py).getattr("_started_time")?.extract::<f64>()?;
            let time = py.import("time")?;
            let wall: f64 = time.call_method0("time")?.extract()?;
            let perf: f64 = time.call_method0("perf_counter")?.extract()?;
            Ok((started, wall - (perf - started)))
        });
        read.ok().filter(|(started, _)| *started != 0.0).map(|(_, wall)| wall)
    }

    /// An iterator is closed (`aclose()`) in its context, bounded (past `deadline` it is
    /// abandoned, not cancelled); a coroutine never awaited is closed so it is not reported as
    /// never awaited.
    fn close(self: Box<Self>, deadline: crate::Ts) -> crate::BoxFuture<'static, crate::Closed> {
        Box::pin(async move {
            match &self.state {
                // A host stream is closed even once read to its end: `aclose` is its teardown.
                State::Iterating(iter) => {
                    let handle = self.handle(iter);
                    crate::Closed::bounded(&*self.clock, deadline, handle.aclose()).await
                }
                State::Returned(out) => {
                    Python::attach(|py| {
                        let obj = out.bind(py);
                        if obj.hasattr("cr_frame").unwrap_or(false) {
                            let _ = obj.call_method0("close");
                        }
                    });
                    crate::Closed::Clean
                }
                State::Ended(_) | State::Channel(_) | State::Done => crate::Closed::Clean,
            }
        })
    }
}

#[cfg(test)]
mod poison_tests {
    use super::*;
    use crate::StreamSink;

    #[test]
    fn poisoned_sink_refuses_new_items() {
        Python::initialize();
        Python::attach(|py| {
            let (sink, _stream) =
                item_stream(py, |py, item: &u32| Ok(item.into_pyobject(py)?.into_any().unbind()))
                    .unwrap();
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = sink.tx.lock().unwrap();
                panic!("interrupted while changing the sink");
            }));
            assert!(sink.push(1).is_err(), "a poisoned sink must not resume delivery");
        });
    }
}
