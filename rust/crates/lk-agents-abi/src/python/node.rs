//! A host async iterable, awaitable or value drained by the core ([`HostSource`]: an in-place
//! port's stream argument), awaited in the context it was opened in.

use std::sync::Arc;

use pyo3::prelude::*;

// The bridge's shared items, `AsyncMutex` included: each use site carries its own allowance.
use super::*;
use crate::errors::ProviderError;

/// Turns one item of a node's output into the core's.
pub type Convert<T> = Box<dyn Fn(&Bound<'_, PyAny>) -> PyResult<T> + Send + Sync>;

enum State {
    /// What the node returned, not yet looked at: an async iterable, an awaitable, one value or
    /// `None`.
    Returned(Py<PyAny>),
    Iterating(Py<PyAny>),
    /// Read to its end, kept for its `_started_time` and its teardown.
    Ended(Py<PyAny>),
    Done,
}

/// The one host-iterable bridge: what a host callable returned (a node's output, a callable's
/// answer) or a host stream (a provider's), drained by the core as a [`crate::StreamSource`], as
/// the reference's node task drains it: an awaitable is awaited first, an async iterable is read
/// to its end, a single value is the one item, `None` is none. Every await runs in `locals`'
/// context.
pub struct HostSource<T> {
    state: State,
    locals: pyo3_async_runtimes::TaskLocals,
    /// A returned value's awaits are cancelled when the core stops waiting; a host stream's are
    /// left to its own `aclose`, its only teardown.
    cancel: bool,
    convert: Convert<T>,
    clock: Arc<dyn crate::Clock>,
}

/// The awaits of one host object: cancelled with their Rust waiter, or not.
enum Handle {
    Cancelling(CancellingHandle),
    Host(HostHandle),
}

impl Handle {
    async fn anext(&self) -> PyResult<Option<Py<PyAny>>> {
        match self {
            Handle::Cancelling(h) => h.anext().await,
            Handle::Host(h) => h.anext().await,
        }
    }

    async fn awaited(&self) -> PyResult<Py<PyAny>> {
        match self {
            Handle::Cancelling(h) => h.await_call(|o| Ok(o.clone())).await,
            Handle::Host(h) => h.await_call(|o| Ok(o.clone())).await,
        }
    }

    async fn aclose(&self) {
        match self {
            Handle::Cancelling(h) => h.aclose().await,
            Handle::Host(h) => h.aclose().await,
        }
    }
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
            cancel: true,
            convert,
            clock,
        }
    }

    /// A host stream already iterating (`__anext__` on `handle`'s object).
    pub fn iterating(
        handle: HostHandle,
        convert: Convert<T>,
        clock: Arc<dyn crate::Clock>,
    ) -> Self {
        let (obj, locals) =
            Python::attach(|py| (handle.object().clone_ref(py), handle.locals().clone()));
        HostSource {
            state: State::Iterating(obj),
            locals,
            cancel: false,
            convert,
            clock,
        }
    }

    fn handle(&self, obj: &Py<PyAny>) -> Handle {
        Python::attach(|py| {
            let (obj, locals) = (obj.clone_ref(py), self.locals.clone());
            if self.cancel {
                Handle::Cancelling(CancellingHandle::from_parts(obj, locals))
            } else {
                Handle::Host(HostHandle::from_parts(obj, locals))
            }
        })
    }

    fn convert(&self, item: Py<PyAny>) -> Result<Option<T>, ProviderError> {
        Python::attach(|py| (self.convert)(item.bind(py)).map(Some).map_err(|e| capture(py, &e)))
    }
}

/// A host async iterable handed to the core as an argument (an in-place RTC stream port), drained on
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
        let convert: Convert<T> = Box::new(|i| i.extract::<T>().map_err(Into::into));
        let clock = super::loop_clock();
        Ok(Box::new(HostSource::returned(obj.to_owned().unbind(), locals, convert, clock)))
    }
}

#[allow(clippy::disallowed_types, reason = "S0: legacy host channel state")]
enum Look {
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
                            Look::Iterate(iter) => self.state = State::Iterating(iter),
                            Look::One(item) => return self.convert(item),
                            Look::Await => match self.handle(&out).awaited().await {
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
                State::Ended(iter) if !self.cancel => {
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
                State::Ended(_) | State::Done => crate::Closed::Clean,
            }
        })
    }
}
