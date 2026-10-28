//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `worker` records.

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

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint.rs.
static ENDPOINT_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::worker::endpoint::EndpointOptions {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(ENDPOINT_OPTIONS.import(py, "livekit.agents._generated.worker_endpoint", "EndpointOptions")?)? {
            return Err(super::not_a(obj, "EndpointOptions"));
        }
        Ok(Self {
            public: ::lk_agents_abi::python::attr_with(obj, "EndpointOptions", "public", |v| v.extract::<bool>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::worker::endpoint::EndpointOptions {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { public } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("public", ::pyo3::IntoPyObjectExt::into_bound_py_any(public, py)?)?;
        ENDPOINT_OPTIONS.import(py, "livekit.agents._generated.worker_endpoint", "EndpointOptions")?.call((), Some(&kwargs))
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

// Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.
static CLI_ARGS_START: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static CLI_ARGS_CONSOLE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static CLI_ARGS_DOWNLOAD_FILES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::worker::cli::CliArgs {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(CLI_ARGS_START.import(py, "livekit.agents._generated.worker_cli", "CliArgsStart")?)? {
            return Ok(Self::Start { entrypoint: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "entrypoint", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, log_level: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "log_level", |v| v.extract::<::std::string::String>())?, log_format: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "log_format", |v| v.extract::<::std::string::String>())?, log_style: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "log_style", |v| v.extract::<::std::string::String>())?, dev: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "dev", |v| v.extract::<bool>())?, cli_addr: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "cli_addr", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, simulation: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "simulation", |v| v.extract::<bool>())?, ws_url: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "ws_url", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, api_key: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "api_key", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, api_secret: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "api_secret", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
        }
        if obj.is_instance(CLI_ARGS_CONSOLE.import(py, "livekit.agents._generated.worker_cli", "CliArgsConsole")?)? {
            return Ok(Self::Console { entrypoint: ::lk_agents_abi::python::attr_with(obj, "CliArgsConsole", "entrypoint", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, connect_addr: ::lk_agents_abi::python::attr_with(obj, "CliArgsConsole", "connect_addr", |v| v.extract::<::std::string::String>())?, record: ::lk_agents_abi::python::attr_with(obj, "CliArgsConsole", "record", |v| v.extract::<bool>())? });
        }
        if obj.is_instance(CLI_ARGS_DOWNLOAD_FILES.import(py, "livekit.agents._generated.worker_cli", "CliArgsDownloadFiles")?)? {
            return Ok(Self::DownloadFiles);
        }
        Err(super::not_a(obj, "CliArgsStart | CliArgsConsole | CliArgsDownloadFiles"))
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::worker::cli::CliArgs {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Start { entrypoint, log_level, log_format, log_style, dev, cli_addr, simulation, ws_url, api_key, api_secret } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("entrypoint", match entrypoint { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("log_level", ::pyo3::IntoPyObjectExt::into_bound_py_any(log_level, py)?)?;
                kwargs.set_item("log_format", ::pyo3::IntoPyObjectExt::into_bound_py_any(log_format, py)?)?;
                kwargs.set_item("log_style", ::pyo3::IntoPyObjectExt::into_bound_py_any(log_style, py)?)?;
                kwargs.set_item("dev", ::pyo3::IntoPyObjectExt::into_bound_py_any(dev, py)?)?;
                kwargs.set_item("cli_addr", match cli_addr { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("simulation", ::pyo3::IntoPyObjectExt::into_bound_py_any(simulation, py)?)?;
                kwargs.set_item("ws_url", match ws_url { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("api_key", match api_key { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("api_secret", match api_secret { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                CLI_ARGS_START.import(py, "livekit.agents._generated.worker_cli", "CliArgsStart")?.call((), Some(&kwargs))
            },
            Self::Console { entrypoint, connect_addr, record } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("entrypoint", match entrypoint { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("connect_addr", ::pyo3::IntoPyObjectExt::into_bound_py_any(connect_addr, py)?)?;
                kwargs.set_item("record", ::pyo3::IntoPyObjectExt::into_bound_py_any(record, py)?)?;
                CLI_ARGS_CONSOLE.import(py, "livekit.agents._generated.worker_cli", "CliArgsConsole")?.call((), Some(&kwargs))
            },
            Self::DownloadFiles => CLI_ARGS_DOWNLOAD_FILES.import(py, "livekit.agents._generated.worker_cli", "CliArgsDownloadFiles")?.call0(),
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/job.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::worker::job::SessionJob {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::worker::job::SessionJob>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`SessionJob` is a declared proxy: it is taken from the host object, not a view"))
    }
}
