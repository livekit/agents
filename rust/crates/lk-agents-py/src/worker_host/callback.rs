//! Captured Python worker callbacks entered through the shared HostInvoker port.

use std::sync::{Arc, Mutex};

use lk_agents_abi::HostRef;
use lk_agents_abi::agent::HostContext;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::python::{capture, intern_host, pyo3_async_runtimes};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyTuple};

pub(super) struct Callback {
    object: Arc<Py<PyAny>>,
    locals: pyo3_async_runtimes::TaskLocals,
    context: HostRef,
    notices: crate::host::Notifier,
}

impl HostContext for Callback {
    fn host_context(&self) -> Option<&HostRef> {
        Some(&self.context)
    }
}

impl Callback {
    pub(super) fn new(py: Python<'_>, object: Py<PyAny>) -> PyResult<Self> {
        let locals = pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?;
        Ok(Self::from_parts(object, locals))
    }

    pub(super) fn from_parts(object: Py<PyAny>, locals: pyo3_async_runtimes::TaskLocals) -> Self {
        let context = Python::attach(|py| {
            let pair = PyTuple::new(py, [locals.event_loop(py), locals.context(py)])
                .expect("two host references form a tuple");
            intern_host(pair.into_any().unbind())
        });
        Self {
            object: Arc::new(object),
            locals,
            context,
            notices: crate::host::Notifier::default(),
        }
    }

    pub(super) fn child(&self, object: Py<PyAny>) -> Self {
        Self {
            object: Arc::new(object),
            locals: self.locals.clone(),
            context: self.context.clone(),
            notices: crate::host::Notifier::default(),
        }
    }

    pub(super) fn object(&self) -> &Py<PyAny> {
        &self.object
    }

    pub(super) fn task_locals(&self) -> &pyo3_async_runtimes::TaskLocals {
        &self.locals
    }

    pub(super) fn await_call<F>(
        &self,
        call: F,
    ) -> impl Future<Output = PyResult<Py<PyAny>>> + Send + 'static
    where
        F: for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> + Send + 'static,
    {
        let object = self.object.clone();
        lk_agents_runtime::host::call(
            &crate::host::Entry,
            "worker callback",
            self,
            &[],
            thunk(move |py| call(object.bind(py)).map(Bound::unbind)),
        )
    }

    pub(super) fn call_now<F>(&self, call: F) -> PyResult<Py<PyAny>>
    where
        F: for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> + Send + 'static,
    {
        let object = self.object.clone();
        lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            thunk(move |py| call(object.bind(py)).map(Bound::unbind)),
        )
    }

    pub(super) fn call_soon<F>(&self, method: &'static str, args: F) -> PyResult<()>
    where
        F: for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyTuple>> + Send + 'static,
    {
        let object = self.object.clone();
        lk_agents_runtime::host::notify(
            &self.notices,
            method,
            self,
            &[],
            false,
            Box::new(move |py| Ok((object.bind(py).getattr(method)?, args(py)?))),
        );
        Ok(())
    }

    pub(super) async fn run_on_loop<T, F>(&self, call: F) -> Result<T, ProviderError>
    where
        F: FnOnce(Python<'_>) -> PyResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        lk_agents_runtime::host::call(
            &crate::host::Entry,
            "worker callback",
            self,
            &[],
            thunk(move |py| {
                let value = call(py)?;
                let _ = sender.send(value);
                Ok(py.None())
            }),
        )
        .await
        .map_err(|error| Python::attach(|py| capture(py, &error)))?;
        receiver.await.map_err(|_| {
            ProviderError::Local(lk_agents_abi::errors::ApiError::connection(Some(
                "worker callback dropped its result".into(),
            )))
        })
    }
}

/// Adapt a host-boundary conversion to Entry's callable-and-arguments shape.
fn thunk<F>(call: F) -> crate::host::HostCall
where
    F: FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send + 'static,
{
    Box::new(move |py| {
        let once = Mutex::new(Some(call));
        let callable = PyCFunction::new_closure(py, None, None, move |args, _| {
            let call =
                once.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take().ok_or_else(
                    || pyo3::exceptions::PyRuntimeError::new_err("worker callback reused"),
                )?;
            call(args.py())
        })?;
        Ok((callable.into_any(), PyTuple::empty(py)))
    })
}

pub(super) fn call_now<F>(call: F) -> PyResult<Py<PyAny>>
where
    F: FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send + 'static,
{
    lk_agents_runtime::host::call_now(&crate::host::Entry, thunk(call))
}
