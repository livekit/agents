//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `worker` proxy bridges and in-place records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint.rs.
static ENDPOINT_REQUEST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::worker::endpoint::EndpointRequest {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(ENDPOINT_REQUEST.import(py, "livekit.agents._generated.worker_endpoint", "EndpointRequest")?)? {
            return Err(super::not_a(obj, "EndpointRequest"));
        }
        Ok(Self {
            request_id: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "request_id", |v| v.extract::<::std::string::String>())?,
            authorized: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "authorized", |v| v.extract::<bool>())?,
            route: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "route", |v| v.extract::<::std::string::String>())?,
            kind: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "kind", |v| v.extract::<::std::string::String>())?,
            client_addr: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "client_addr", |v| v.extract::<::std::string::String>())?,
            scheme: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "scheme", |v| v.extract::<::std::string::String>())?,
            timeout_ms: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "timeout_ms", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "EndpointRequest.timeout_ms"))?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::worker::endpoint::EndpointRequest {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { request_id, authorized, route, kind, client_addr, scheme, timeout_ms } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("request_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id, py)?)?;
        kwargs.set_item("authorized", ::pyo3::IntoPyObjectExt::into_bound_py_any(authorized, py)?)?;
        kwargs.set_item("route", ::pyo3::IntoPyObjectExt::into_bound_py_any(route, py)?)?;
        kwargs.set_item("kind", ::pyo3::IntoPyObjectExt::into_bound_py_any(kind, py)?)?;
        kwargs.set_item("client_addr", ::pyo3::IntoPyObjectExt::into_bound_py_any(client_addr, py)?)?;
        kwargs.set_item("scheme", ::pyo3::IntoPyObjectExt::into_bound_py_any(scheme, py)?)?;
        kwargs.set_item("timeout_ms", ::pyo3::IntoPyObjectExt::into_bound_py_any(timeout_ms, py)?)?;
        ENDPOINT_REQUEST.import(py, "livekit.agents._generated.worker_endpoint", "EndpointRequest")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc.rs.
static IPC_FAILURE_IO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static IPC_FAILURE_MALFORMED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static IPC_FAILURE_TOO_LARGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static IPC_FAILURE_UNEXPECTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static IPC_FAILURE_VERSION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::worker::ipc::IpcFailure {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Io { kind, os_error, message } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("kind", ::pyo3::IntoPyObjectExt::into_bound_py_any(kind, py)?)?;
                kwargs.set_item("os_error", match os_error { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                IPC_FAILURE_IO.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureIo")?.call((), Some(&kwargs))
            },
            Self::Malformed { message } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                IPC_FAILURE_MALFORMED.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureMalformed")?.call((), Some(&kwargs))
            },
            Self::TooLarge { size } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("size", ::pyo3::IntoPyObjectExt::into_bound_py_any(size, py)?)?;
                IPC_FAILURE_TOO_LARGE.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureTooLarge")?.call((), Some(&kwargs))
            },
            Self::Unexpected { message } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                IPC_FAILURE_UNEXPECTED.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureUnexpected")?.call((), Some(&kwargs))
            },
            Self::Version { received, expected } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("received", ::pyo3::IntoPyObjectExt::into_bound_py_any(received, py)?)?;
                kwargs.set_item("expected", ::pyo3::IntoPyObjectExt::into_bound_py_any(expected, py)?)?;
                IPC_FAILURE_VERSION.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureVersion")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc.rs.
static PROCESS_COMPLETION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::worker::ipc::ProcessCompletion {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { exit_code, failure } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("exit_code", match exit_code { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("failure", match failure { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        PROCESS_COMPLETION.import(py, "livekit.agents._generated.worker_ipc", "ProcessCompletion")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
static REGISTERED_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::worker::worker::RegisteredInfo {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { worker_id, region, protocol, deployment, agent_name } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("worker_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(worker_id, py)?)?;
        kwargs.set_item("region", ::pyo3::IntoPyObjectExt::into_bound_py_any(region, py)?)?;
        kwargs.set_item("protocol", ::pyo3::IntoPyObjectExt::into_bound_py_any(protocol, py)?)?;
        kwargs.set_item("deployment", ::pyo3::IntoPyObjectExt::into_bound_py_any(deployment, py)?)?;
        kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
        REGISTERED_INFO.import(py, "livekit.agents._generated.worker_server", "RegisteredInfo")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/job.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::worker::job::SessionJob {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::worker::job::SessionJob>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/remote_session.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::worker::remote_session::SessionPort {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::worker::remote_session::SessionPort>(obj)
    }
}
