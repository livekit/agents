//! What a node crosses (`Agent.llm_node`, `tts_node`, `stt_node` and the default nodes): a host
//! async iterable drained by the core ([`NodeSource`]), awaited in the context it was opened in so
//! the node sees its agent's binding, and a core channel read by the host as an async iterator
//! ([`ItemStream`], fed through [`ItemSink`]).

use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use tokio::sync::mpsc;

use super::{AsyncMutex, CancellingHandle, HostBridge as _, captured, provider_error_to_py};
use crate::errors::ProviderError;

type Item = Result<Py<PyAny>, ProviderError>;

/// `_core.ItemStream`: what the core hands a node (its text, its audio) or a default node hands
/// back (frames, speech events), as `async for` reads it. Ends when the core ends it; a failure
/// is raised where it happened.
#[pyclass(frozen, name = "ItemStream", module = "livekit.agents._core")]
pub struct ItemStream {
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
    tx: Mutex<Option<mpsc::UnboundedSender<Item>>>,
    to_py: fn(Python<'_>, &I) -> PyResult<Py<PyAny>>,
}

impl<I> ItemSink<I> {
    /// Whether the host still reads.
    pub fn send(&self, item: &I) -> bool {
        let Some(tx) = self.tx.lock().unwrap_or_else(|p| p.into_inner()).clone() else {
            return false;
        };
        let converted = Python::attach(|py| (self.to_py)(py, item)).map_err(captured);
        tx.send(converted).is_ok()
    }

    /// Raised on the host's next read; nothing follows it.
    pub fn fail(&self, error: ProviderError) {
        if let Some(tx) = self.tx.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = tx.send(Err(error));
        }
    }

    /// Resolves once the host stopped reading (closed or dropped the stream).
    pub async fn closed(&self) {
        let tx = self.tx.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if let Some(tx) = tx {
            tx.closed().await;
        }
    }
}

impl<I: Send + Sync> crate::StreamSink<I> for ItemSink<I> {
    /// A host that stopped reading drops the rest, as a node that returned early ignores its input.
    fn push(&self, item: I) -> Result<(), ProviderError> {
        self.send(&item);
        Ok(())
    }

    /// The reference's node input has no segment marker.
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        self.tx.lock().unwrap_or_else(|p| p.into_inner()).take();
    }
}

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

enum State {
    /// What the node returned, not yet looked at: an async iterable, an awaitable, one value or
    /// `None`.
    Returned(Py<PyAny>),
    Iterating(Py<PyAny>),
    /// An `ItemStream` handed on as it is (a node passing its input to a default node): read
    /// from the core's channel directly.
    Channel(Arc<AsyncMutex<Option<mpsc::UnboundedReceiver<Item>>>>),
    Done,
}

/// A node's output as the core drains it, as the reference's node task does: an awaitable is
/// awaited first, an async iterable is read to its end, a single value is the one item, `None` is
/// none. Every await runs in `locals`' context and is cancelled if the core stops waiting.
pub struct NodeSource<T> {
    state: State,
    locals: pyo3_async_runtimes::TaskLocals,
    convert: Convert<T>,
    clock: Arc<dyn crate::Clock>,
}

impl<T> NodeSource<T> {
    pub fn new(
        returned: Py<PyAny>,
        locals: pyo3_async_runtimes::TaskLocals,
        convert: Convert<T>,
        clock: Arc<dyn crate::Clock>,
    ) -> Self {
        NodeSource { state: State::Returned(returned), locals, convert, clock }
    }

    fn handle(&self, obj: &Py<PyAny>) -> CancellingHandle {
        Python::attach(|py| CancellingHandle::from_parts(obj.clone_ref(py), self.locals.clone()))
    }
}

enum Look {
    Channel(Arc<AsyncMutex<Option<mpsc::UnboundedReceiver<Item>>>>),
    Iterate(Py<PyAny>),
    Await,
    One(Py<PyAny>),
    Nothing,
}

#[async_trait::async_trait]
impl<T: Send> crate::StreamSource<T> for NodeSource<T> {
    /// A cancelled await is the end, as a cancelled host stream is.
    async fn next(&mut self) -> Result<Option<T>, ProviderError> {
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
                    let item = item?;
                    return Python::attach(|py| (self.convert)(item.bind(py)).map(Some))
                        .map_err(captured);
                }
                State::Iterating(iter) => {
                    let Some(item) = self.handle(&iter).anext().await.map_err(captured)? else {
                        return Ok(None);
                    };
                    self.state = State::Iterating(iter);
                    return Python::attach(|py| (self.convert)(item.bind(py)).map(Some))
                        .map_err(captured);
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
                        Look::One(item) => {
                            return Python::attach(|py| (self.convert)(item.bind(py)).map(Some))
                                .map_err(captured);
                        }
                        Look::Await => {
                            let awaited = self.handle(&out).await_call(|o| Ok(o.clone())).await;
                            match awaited {
                                Ok(value) => self.state = State::Returned(value),
                                Err(e) => {
                                    let cancelled = Python::attach(|py| {
                                        e.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py)
                                    });
                                    return if cancelled { Ok(None) } else { Err(captured(e)) };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// An iterator is closed (`aclose()`) in its context, bounded; a coroutine never awaited is
    /// closed so it is not reported as never awaited.
    async fn close(self: Box<Self>, deadline: crate::Ts) -> crate::Closed {
        match &self.state {
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
            State::Channel(_) | State::Done => crate::Closed::Clean,
        }
    }
}
