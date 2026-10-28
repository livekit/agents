//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `agent` proxy bridges and in-place records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static RUN_ERROR_NO_MODEL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_CANCELLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_MODEL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_HOST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_TOOL_RETRIES_EXCEEDED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_OUTPUT_RETRIES_EXCEEDED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::RunError {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::NoModel => RUN_ERROR_NO_MODEL.import(py, "livekit.agents._generated.agent_run", "RunErrorNoModel")?.call0(),
            Self::Cancelled => RUN_ERROR_CANCELLED.import(py, "livekit.agents._generated.agent_run", "RunErrorCancelled")?.call0(),
            Self::Model(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                RUN_ERROR_MODEL.import(py, "livekit.agents._generated.agent_run", "RunErrorModel")?.call((), Some(&kwargs))
            },
            Self::Host(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                RUN_ERROR_HOST.import(py, "livekit.agents._generated.agent_run", "RunErrorHost")?.call((), Some(&kwargs))
            },
            Self::ToolRetriesExceeded { tool, retries } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("tool", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool, py)?)?;
                kwargs.set_item("retries", ::pyo3::IntoPyObjectExt::into_bound_py_any(retries, py)?)?;
                RUN_ERROR_TOOL_RETRIES_EXCEEDED.import(py, "livekit.agents._generated.agent_run", "RunErrorToolRetriesExceeded")?.call((), Some(&kwargs))
            },
            Self::OutputRetriesExceeded { retries } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("retries", ::pyo3::IntoPyObjectExt::into_bound_py_any(retries, py)?)?;
                RUN_ERROR_OUTPUT_RETRIES_EXCEEDED.import(py, "livekit.agents._generated.agent_run", "RunErrorOutputRetriesExceeded")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::agent::AgentHost {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::agent::AgentHost>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`AgentHost` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::agent::HostInstructions {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::agent::HostInstructions>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`HostInstructions` is a declared proxy: it is taken from the host object, not a view"))
    }
}
