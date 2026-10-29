//! The host face the generated stream classes forward to: `recv()` resolves to a native source's
//! next item (crossing by `to_py`, the item's generated conversion) or `None` at the end;
//! `aclose()` releases the source.

use lk_agents_runtime::net::source::{NativeSource, PullError, Pulled};
use pyo3::prelude::*;

pub fn recv<'py, T: Send + 'static>(
    source: &NativeSource<T>,
    py: Python<'py>,
    to_py: fn(Python<'_>, T) -> PyResult<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let source = source.clone();
    super::future_into_py(py, async move {
        match source.pull().await {
            Ok(Pulled::Item(item)) => Python::attach(|py| to_py(py, item)).map(Some),
            Ok(Pulled::End | Pulled::Cancelled) => Ok(None),
            Err(PullError::Overlap) => Err(lk_agents_abi::python::host_exception(
                "livekit.agents._exceptions",
                "StreamStateError",
                PullError::Overlap.to_string(),
            )),
            Err(PullError::Failed(err)) => Err(lk_agents_abi::python::provider_error_to_py(err)),
        }
    })
}

pub fn aclose<'py, T: Send + 'static>(
    source: &NativeSource<T>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let source = source.clone();
    super::future_into_py(py, async move {
        if source.close(lk_agents_abi::python::NATIVE_CLOSE_DEADLINE).await
            == lk_agents_abi::Closed::TimedOut
        {
            log::error!(target: "livekit.agents", "a native stream did not release its resources in time");
        }
        Ok(())
    })
}
