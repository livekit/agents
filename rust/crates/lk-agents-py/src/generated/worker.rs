//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `worker` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
type EndpointStreamImpl = <crate::Selected as super::Implementations>::EndpointStream;

fn _extract_endpoint_stream_read_max_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "max_bytes")
}

fn _extract_endpoint_stream_write_data(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<u8>> {
    { if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&obj, "data") } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "data") }).collect::<PyResult<Vec<_>>>()) } }
}

fn _extract_endpoint_stream_reset_code(obj: &Bound<'_, PyAny>) -> PyResult<i32> {
    ::lk_agents_abi::python::number_from_py::<i32>(&obj, "code")
}

fn _extract_endpoint_stream_stop_code(obj: &Bound<'_, PyAny>) -> PyResult<i32> {
    ::lk_agents_abi::python::number_from_py::<i32>(&obj, "code")
}

/// An exchange after its preamble, with independent reads and writes.
#[pyclass(frozen, name = "EndpointStream", module = "livekit.agents._core")]
pub struct EndpointStream(pub(crate) ::std::sync::Arc<EndpointStreamImpl>);

#[pymethods]
impl EndpointStream {
    #[pyo3(signature = (max_bytes = 65536))]
    fn read(&self, #[pyo3(from_py_with = _extract_endpoint_stream_read_max_bytes)] max_bytes: usize) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::read(&self.0, max_bytes).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (data))]
    fn write(&self, #[pyo3(from_py_with = _extract_endpoint_stream_write_data)] data: ::std::vec::Vec<u8>) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::write(&self.0, data).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn finish(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::finish(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (code))]
    fn reset(&self, #[pyo3(from_py_with = _extract_endpoint_stream_reset_code)] code: i32) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::reset(&self.0, code).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (code))]
    fn stop(&self, #[pyo3(from_py_with = _extract_endpoint_stream_stop_code)] code: i32) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::stop(&self.0, code).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
type EndpointServerImpl = <crate::Selected as super::Implementations>::EndpointServer;

fn _extract_endpoint_server_close_websockets_code(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    ::lk_agents_abi::python::number_from_py::<u16>(&obj, "code")
}

/// The host's ASGI application bridge and its active WebSockets.
#[pyclass(frozen, name = "EndpointServer", module = "livekit.agents._core")]
pub struct EndpointServer(pub(crate) ::std::sync::Arc<EndpointServerImpl>);

#[pymethods]
impl EndpointServer {
    #[new]
    fn new() -> PyResult<EndpointServer> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <EndpointServerImpl as ::lk_agents_abi::worker::endpoint_host::EndpointServer>::new();
        Ok(EndpointServer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (stream, app, request))]
    fn serve(&self, stream: ::lk_agents_abi::HostRef, app: ::lk_agents_abi::HostRef, request: ::lk_agents_abi::HostRef) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <EndpointServerImpl as ::lk_agents_abi::worker::endpoint_host::EndpointServer>::serve(&self.0, stream, app, request).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (code = 1001, reason = "worker draining".to_owned()))]
    fn close_websockets(&self, #[pyo3(from_py_with = _extract_endpoint_server_close_websockets_code)] code: u16, reason: ::std::string::String) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <EndpointServerImpl as ::lk_agents_abi::worker::endpoint_host::EndpointServer>::close_websockets(&self.0, code, reason).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
type _core_AsgiReceiveImpl = <crate::Selected as super::Implementations>::AsgiReceive;

/// The native ASGI receive callback, bound to one endpoint exchange.
#[pyclass(frozen, name = "_core_AsgiReceive", module = "livekit.agents._core")]
pub struct _core_AsgiReceive(pub(crate) ::std::sync::Arc<_core_AsgiReceiveImpl>);

#[pymethods]
impl _core_AsgiReceive {
    fn __call__(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_AsgiReceiveImpl as ::lk_agents_abi::worker::endpoint_host::AsgiReceive>::__call__(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
type _core_AsgiSendImpl = <crate::Selected as super::Implementations>::AsgiSend;

/// The native ASGI send callback, bound to one endpoint exchange.
#[pyclass(frozen, name = "_core_AsgiSend", module = "livekit.agents._core")]
pub struct _core_AsgiSend(pub(crate) ::std::sync::Arc<_core_AsgiSendImpl>);

#[pymethods]
impl _core_AsgiSend {
    #[pyo3(signature = (message))]
    fn __call__(&self, message: ::lk_agents_abi::HostRef) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_AsgiSendImpl as ::lk_agents_abi::worker::endpoint_host::AsgiSend>::__call__(&self.0, message).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/job.rs.
type JobTraceImpl = <crate::Selected as super::Implementations>::JobTrace;

fn _extract_job_trace_job_timeline_received_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "received_at")
}

fn _extract_job_trace_job_timeline_accepted_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "accepted_at")
}

fn _extract_job_trace_job_timeline_assigned_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "assigned_at")
}

fn _extract_job_trace_job_timeline_launched_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "launched_at")
}

fn _extract_job_trace_job_timeline_entrypoint_started_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "entrypoint_started_at")
}

fn _extract_job_trace_job_timeline_server_started_at(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "server_started_at")
}

/// A job's dispatch timeline, for its root span: one event per known stage instant (unix
/// nanoseconds) and the seconds between adjacent stages as attributes (they sum to the dispatch
/// latency). A zero instant is unknown (a simulation, the console, a resumed job) and skipped.
/// `server_started_at` is the server's own start (`JobState.started_at`, unix nanoseconds, or
/// milliseconds or seconds should that change).
#[pyfunction]
#[pyo3(signature = (received_at, accepted_at, assigned_at, launched_at, entrypoint_started_at, server_started_at))]
pub fn job_timeline(#[pyo3(from_py_with = _extract_job_trace_job_timeline_received_at)] received_at: f64, #[pyo3(from_py_with = _extract_job_trace_job_timeline_accepted_at)] accepted_at: f64, #[pyo3(from_py_with = _extract_job_trace_job_timeline_assigned_at)] assigned_at: f64, #[pyo3(from_py_with = _extract_job_trace_job_timeline_launched_at)] launched_at: f64, #[pyo3(from_py_with = _extract_job_trace_job_timeline_entrypoint_started_at)] entrypoint_started_at: f64, #[pyo3(from_py_with = _extract_job_trace_job_timeline_server_started_at)] server_started_at: i64) -> PyResult<(::std::vec::Vec<(::std::string::String, i64)>, ::std::vec::Vec<(::std::string::String, f64)>)> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <JobTraceImpl as ::lk_agents_abi::worker::job::JobTrace>::job_timeline(received_at, accepted_at, assigned_at, launched_at, entrypoint_started_at, server_started_at);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
type ServerEnvironmentImpl = <crate::Selected as super::Implementations>::ServerEnvironment;

/// Logical CPUs, fractional under a cgroup quota. Reads, starts nothing.
#[pyfunction]
pub fn cpu_count() -> PyResult<f64> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <ServerEnvironmentImpl as ::lk_agents_abi::worker::worker::ServerEnvironment>::cpu_count();
    Ok(out)
}

/// `https://<project>/agents/<agent>/<deployment>` (`AgentServer._endpoint_url_base`), when
/// the URL and the agent are known; a worker registered without a deployment is `default`.
#[pyfunction]
#[pyo3(signature = (ws_url, agent_name, deployment))]
pub fn endpoint_url_base(ws_url: ::std::string::String, agent_name: ::std::string::String, deployment: ::std::string::String) -> PyResult<::std::option::Option<::std::string::String>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <ServerEnvironmentImpl as ::lk_agents_abi::worker::worker::ServerEnvironment>::endpoint_url_base(ws_url, agent_name, deployment);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.
type WorkerCliImpl = <crate::Selected as super::Implementations>::WorkerCli;

#[pyfunction]
#[pyo3(signature = (argv))]
pub fn parse_cli<'py>(py: Python<'py>, argv: ::std::vec::Vec<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::parse_cli(argv).map_err(|e: ::lk_agents_abi::worker::cli::CliError| <::lk_agents_abi::worker::cli::CliError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(crate::generated::records::worker_cli::cli_args_to_py(py, out)?)
}

#[pyfunction]
#[pyo3(signature = (path))]
pub fn entrypoint_module(path: ::std::option::Option<::std::string::String>) -> PyResult<(::std::string::String, ::std::string::String)> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::entrypoint_module(path).map_err(|e: ::lk_agents_abi::worker::cli::CliError| <::lk_agents_abi::worker::cli::CliError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// Format one host logging record using its boundary adapter.
#[pyfunction]
#[pyo3(signature = (style, record, message, asctime, exception, stack, ensure_ascii))]
pub fn format_log_record(style: ::std::string::String, record: ::lk_agents_abi::HostRef, message: ::std::string::String, asctime: ::std::string::String, exception: ::std::option::Option<::std::string::String>, stack: ::std::option::Option<::std::string::String>, ensure_ascii: bool) -> PyResult<::std::string::String> {
    let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::format_log_record(style, record, message, asctime, exception, stack, ensure_ascii).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// The development launcher connection.
#[pyfunction]
#[pyo3(signature = (cli_addr, agent_name, url, api_key, api_secret, server))]
pub fn dev_channel<'py>(py: Python<'py>, cli_addr: ::std::string::String, agent_name: ::std::string::String, url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, server: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let server = crate::generated::worker::DevServerProxy::from_host(&server)?;
    crate::host::future_into_py(py, async move {
        let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::dev_channel(cli_addr, agent_name, url, api_key, api_secret, server).await;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

/// Create the control endpoints for process bootstrap.
#[pyfunction]
pub fn ipc_pair() -> PyResult<(i64, i64)> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::ipc_pair().map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.
type ExitSignalsImpl = <crate::Selected as super::Implementations>::CliSignals;

/// The process launcher's signal and final event-loop cleanup owner.
#[pyclass(frozen, name = "ExitSignals", module = "livekit.agents._core")]
pub struct ExitSignals(pub(crate) ::std::sync::Arc<ExitSignalsImpl>);

#[pymethods]
impl ExitSignals {
    #[new]
    #[pyo3(signature = (escalate = true))]
    fn new(escalate: bool) -> PyResult<ExitSignals> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::new(escalate);
        Ok(ExitSignals(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (exit_served))]
    fn signal(&self, exit_served: bool) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::signal(&self.0, exit_served);
        Ok(out)
    }

    fn served(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::served(&self.0);
        Ok(())
    }

    fn draining(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::draining(&self.0);
        Ok(())
    }

    /// Arm the native deadline before the host cancels tasks or closes its executor.
    fn closing(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::closing(&self.0);
        Ok(())
    }

    /// Disarm the deadline only after the host event loop has actually closed.
    fn closed(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::closed(&self.0);
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type ProcSupervisorImpl = <crate::Selected as super::Implementations>::WorkerProcSupervisor;

fn _extract_worker_proc_supervisor_new_fd(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "fd")
}

fn _extract_worker_proc_supervisor_new_pid(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "pid")
}

fn _extract_worker_proc_supervisor_new_initialize_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "initialize_timeout")
}

fn _extract_worker_proc_supervisor_new_close_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "close_timeout")
}

fn _extract_worker_proc_supervisor_new_memory_warn_mb(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "memory_warn_mb")
}

fn _extract_worker_proc_supervisor_new_memory_limit_mb(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "memory_limit_mb")
}

fn _extract_worker_proc_supervisor_new_ping_interval(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "ping_interval")
}

fn _extract_worker_proc_supervisor_new_ping_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "ping_timeout")
}

fn _extract_worker_proc_supervisor_new_high_ping_threshold(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "high_ping_threshold")
}

fn _extract_worker_proc_supervisor__core_launched_process_pid(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "pid")
}

fn _extract_worker_proc_supervisor__core_launched_process_sentinel(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "sentinel")
}

fn _extract_worker_proc_supervisor__core_launched_process_owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<i64>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<i64>(&item, "owned") }).collect::<PyResult<Vec<_>>>())
}

#[pyclass(frozen, name = "ProcSupervisor", module = "livekit.agents._core")]
pub struct ProcSupervisor(pub(crate) ::std::sync::Arc<ProcSupervisorImpl>);

#[pymethods]
impl ProcSupervisor {
    #[new]
    #[pyo3(signature = (fd, host, *, pid, initialize_timeout, close_timeout, memory_warn_mb, memory_limit_mb, ping_interval, ping_timeout, high_ping_threshold, killable = true))]
    fn new(#[pyo3(from_py_with = _extract_worker_proc_supervisor_new_fd)] fd: i64, host: ::lk_agents_abi::HostRef, #[pyo3(from_py_with = _extract_worker_proc_supervisor_new_pid)] pid: u32, #[pyo3(from_py_with = _extract_worker_proc_supervisor_new_initialize_timeout)] initialize_timeout: f64, #[pyo3(from_py_with = _extract_worker_proc_supervisor_new_close_timeout)] close_timeout: f64, #[pyo3(from_py_with = _extract_worker_proc_supervisor_new_memory_warn_mb)] memory_warn_mb: f64, #[pyo3(from_py_with = _extract_worker_proc_supervisor_new_memory_limit_mb)] memory_limit_mb: f64, #[pyo3(from_py_with = _extract_worker_proc_supervisor_new_ping_interval)] ping_interval: f64, #[pyo3(from_py_with = _extract_worker_proc_supervisor_new_ping_timeout)] ping_timeout: f64, #[pyo3(from_py_with = _extract_worker_proc_supervisor_new_high_ping_threshold)] high_ping_threshold: f64, killable: bool) -> PyResult<ProcSupervisor> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::new(fd, host, pid, initialize_timeout, close_timeout, memory_warn_mb, memory_limit_mb, ping_interval, ping_timeout, high_ping_threshold, killable).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ProcSupervisor(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (*, asyncio_debug, http_proxy))]
    fn initialize(&self, asyncio_debug: bool, http_proxy: ::std::option::Option<::std::string::String>) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::initialize(&self.0, asyncio_debug, http_proxy).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[getter]
    fn initialized(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::initialized(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (pid, sentinel, method, owned))]
    fn _core_launched_process(&self, #[pyo3(from_py_with = _extract_worker_proc_supervisor__core_launched_process_pid)] pid: u32, #[pyo3(from_py_with = _extract_worker_proc_supervisor__core_launched_process_sentinel)] sentinel: i64, method: ::std::string::String, #[pyo3(from_py_with = _extract_worker_proc_supervisor__core_launched_process_owned)] owned: ::std::vec::Vec<i64>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::_core_launched_process(&self.0, pid, sentinel, method, owned).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (joined, host))]
    fn run(&self, joined: ::lk_agents_abi::HostRef, host: ::lk_agents_abi::HostRef) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::run(&self.0, joined, host).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn _core_wait_exit(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::_core_wait_exit(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn _core_join(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::_core_join(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn aclose(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::aclose(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn kill(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::kill(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (fields))]
    fn start_job(&self, fields: ::lk_agents_abi::worker::job::StartJobFields) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::start_job(&self.0, fields).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[getter]
    fn killed(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcSupervisor>::killed(&self.0);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type IpcClientImpl = <crate::Selected as super::Implementations>::WorkerIpcClient;

fn _extract_worker_ipc_client_new_fd(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "fd")
}

#[pyclass(frozen, name = "IpcClient", module = "livekit.agents._core")]
pub struct IpcClient(pub(crate) ::std::sync::Arc<IpcClientImpl>);

#[pymethods]
impl IpcClient {
    #[new]
    #[pyo3(signature = (fd, initialize, *, process = false))]
    fn new(#[pyo3(from_py_with = _extract_worker_ipc_client_new_fd)] fd: i64, initialize: ::lk_agents_abi::HostRef, process: bool) -> PyResult<IpcClient> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::WorkerIpcClient>::new(fd, initialize, process).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(IpcClient(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn asyncio_debug(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::WorkerIpcClient>::asyncio_debug(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (host))]
    fn serve(&self, host: ::lk_agents_abi::HostRef) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::WorkerIpcClient>::serve(&self.0, host).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn stop(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::WorkerIpcClient>::stop(&self.0);
        Ok(())
    }

    fn aclose(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::WorkerIpcClient>::aclose(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type ProcPoolImpl = <crate::Selected as super::Implementations>::WorkerProcPool;

fn _extract_worker_proc_pool_new_num_idle_processes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "num_idle_processes")
}

fn _extract_worker_proc_pool_new_initialize_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "initialize_timeout")
}

fn _extract_worker_proc_pool_new_close_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "close_timeout")
}

fn _extract_worker_proc_pool_set_target_idle_processes_n(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "n")
}

#[pyclass(frozen, name = "ProcPool", module = "livekit.agents._core")]
pub struct ProcPool(pub(crate) ::std::sync::Arc<ProcPoolImpl>);

#[pymethods]
impl ProcPool {
    #[new]
    #[pyo3(signature = (host, *, num_idle_processes, initialize_timeout, close_timeout))]
    fn new(host: ::lk_agents_abi::HostRef, #[pyo3(from_py_with = _extract_worker_proc_pool_new_num_idle_processes)] num_idle_processes: usize, #[pyo3(from_py_with = _extract_worker_proc_pool_new_initialize_timeout)] initialize_timeout: f64, #[pyo3(from_py_with = _extract_worker_proc_pool_new_close_timeout)] close_timeout: f64) -> PyResult<ProcPool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::new(host, num_idle_processes, initialize_timeout, close_timeout).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ProcPool(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn processes(&self) -> PyResult<::std::vec::Vec<::lk_agents_abi::HostRef>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::processes(&self.0);
        Ok(out)
    }

    #[getter]
    fn jobs_waiting(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::jobs_waiting(&self.0);
        Ok(out)
    }

    #[getter]
    fn target_idle_processes(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::target_idle_processes(&self.0);
        Ok(out)
    }

    #[setter(target_idle_processes)]
    fn set_target_idle_processes(&self, #[pyo3(from_py_with = _extract_worker_proc_pool_set_target_idle_processes_n)] n: usize) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::set_target_idle_processes(&self.0, n);
        Ok(())
    }

    fn start(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::start(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (info))]
    fn launch_job(&self, info: ::lk_agents_abi::HostRef) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::launch_job(&self.0, info).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn join_jobs(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::join_jobs(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn aclose(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::WorkerProcPool>::aclose(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type JobRunImpl = <crate::Selected as super::Implementations>::WorkerJobRun;

fn _extract_worker_job_run_new_session_end_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "session_end_timeout")
}

fn _extract_worker_job_run_new_session_aclose_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "session_aclose_timeout")
}

fn _extract_worker_job_run_add_participant_entrypoint_kinds(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<i32>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<i32>(&item, "kinds") }).collect::<PyResult<Vec<_>>>())
}

fn _extract_worker_job_run_participant_joined_kind(obj: &Bound<'_, PyAny>) -> PyResult<i32> {
    ::lk_agents_abi::python::number_from_py::<i32>(&obj, "kind")
}

#[pyclass(frozen, name = "JobRun", module = "livekit.agents._core")]
pub struct JobRun(pub(crate) ::std::sync::Arc<JobRunImpl>);

#[pymethods]
impl JobRun {
    #[new]
    #[pyo3(signature = (*, fake_job, session_end_timeout, redaction, url = "".to_owned(), recording = false, session_end = None, session_aclose_timeout = 60.0))]
    fn new(fake_job: bool, #[pyo3(from_py_with = _extract_worker_job_run_new_session_end_timeout)] session_end_timeout: f64, redaction: bool, url: ::std::string::String, recording: bool, session_end: ::std::option::Option<::lk_agents_abi::HostRef>, #[pyo3(from_py_with = _extract_worker_job_run_new_session_aclose_timeout)] session_aclose_timeout: f64) -> PyResult<JobRun> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::new(fake_job, session_end_timeout, redaction, url, recording, session_end, session_aclose_timeout).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(JobRun(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (callback))]
    fn add_shutdown_callback(&self, callback: ::lk_agents_abi::HostRef) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::add_shutdown_callback(&self.0, callback).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn observability_url(&self) -> PyResult<::std::option::Option<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::observability_url(&self.0);
        Ok(out)
    }

    #[getter]
    fn buffer_logs(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::buffer_logs(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (reason, user_initiated))]
    fn shutdown(&self, reason: ::std::string::String, user_initiated: bool) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::shutdown(&self.0, reason, user_initiated);
        Ok(())
    }

    fn connected(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::connected(&self.0);
        Ok(())
    }

    #[pyo3(signature = (connect))]
    fn connect(&self, connect: ::lk_agents_abi::HostRef) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::connect(&self.0, connect).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (entrypoint, kinds))]
    fn add_participant_entrypoint(&self, entrypoint: ::lk_agents_abi::HostRef, #[pyo3(from_py_with = _extract_worker_job_run_add_participant_entrypoint_kinds)] kinds: ::std::vec::Vec<i32>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::add_participant_entrypoint(&self.0, entrypoint, kinds).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (ctx, participant, identity, kind))]
    fn participant_joined(&self, ctx: ::lk_agents_abi::HostRef, participant: ::lk_agents_abi::HostRef, identity: ::std::string::String, #[pyo3(from_py_with = _extract_worker_job_run_participant_joined_kind)] kind: i32) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::participant_joined(&self.0, ctx, participant, identity, kind).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (attributes))]
    fn participant_left(&self, attributes: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::participant_left(&self.0, attributes);
        Ok(())
    }

    #[pyo3(signature = (call, name, console_result))]
    fn api_call(&self, call: ::lk_agents_abi::HostRef, name: ::std::string::String, console_result: ::lk_agents_abi::HostRef) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::api_call(&self.0, call, name, console_result).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn close_tasks(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::close_tasks(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (options))]
    fn init_recording(&self, options: ::std::collections::BTreeMap<::std::string::String, bool>) -> PyResult<::std::option::Option<bool>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::init_recording(&self.0, options).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (session))]
    fn claim_session(&self, session: ::lk_agents_abi::HostRef) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::claim_session(&self.0, session).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[getter]
    fn primary_session(&self) -> PyResult<::std::option::Option<::lk_agents_abi::HostRef>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::primary_session(&self.0);
        Ok(out)
    }

    #[getter]
    fn recording_initialized(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::recording_initialized(&self.0);
        Ok(out)
    }

    #[getter]
    fn redaction(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::redaction(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (host))]
    fn run(&self, host: ::lk_agents_abi::HostRef) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::WorkerJobRun>::run(&self.0, host).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/server_host.rs.
type ServerConfigImpl = <crate::Selected as super::Implementations>::ServerConfig;

#[pyclass(frozen, name = "ServerConfig", module = "livekit.agents._core")]
pub struct ServerConfig(pub(crate) ::std::sync::Arc<ServerConfigImpl>);

#[pymethods]
impl ServerConfig {
    #[new]
    #[pyo3(signature = (options))]
    fn new(options: ::lk_agents_abi::worker::worker::ServerOptions) -> PyResult<ServerConfig> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::new(options).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ServerConfig(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (options))]
    fn update(&self, options: ::lk_agents_abi::worker::worker::ServerOptions) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::update(&self.0, options).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn ws_url(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::ws_url(&self.0);
        Ok(out)
    }

    #[getter]
    fn api_key(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::api_key(&self.0);
        Ok(out)
    }

    #[getter]
    fn api_secret(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::api_secret(&self.0);
        Ok(out)
    }

    #[getter]
    fn agent_name(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::agent_name(&self.0);
        Ok(out)
    }

    #[getter]
    fn deployment(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::deployment(&self.0);
        Ok(out)
    }

    #[getter]
    fn cloud(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::cloud(&self.0);
        Ok(out)
    }

    #[getter]
    fn drain_timeout(&self) -> PyResult<f64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::drain_timeout(&self.0);
        Ok(out)
    }

    #[getter]
    fn simulation(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::simulation(&self.0);
        Ok(out)
    }

    #[setter(simulation)]
    fn set_simulation(&self, value: bool) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::set_simulation(&self.0, value);
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::server_host::ServerConfiguration`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::server_host::ServerConfiguration>(),
        read: |obj| {
            let this = obj.cast::<ServerConfig>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::server_host::ServerConfiguration> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/server_host.rs.
type ServerRuntimeImpl = <crate::Selected as super::Implementations>::ServerRuntime;

fn _extract_server_runtime_drain_timeout(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "timeout")).map(Some) }
}

fn _extract_server_runtime_simulate_job_room_info(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<::std::vec::Vec<u8>>> {
    if obj.is_none() { Ok(None) } else { ({ if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&obj, "room_info") } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "room_info") }).collect::<PyResult<Vec<_>>>()) } }).map(Some) }
}

/// Returned host references are futures owned by the bootstrap loop.
#[pyclass(frozen, name = "ServerRuntime", module = "livekit.agents._core")]
pub struct ServerRuntime(pub(crate) ::std::sync::Arc<ServerRuntimeImpl>);

#[pymethods]
impl ServerRuntime {
    #[new]
    #[pyo3(signature = (config, host))]
    fn new(config: ::lk_agents_abi::HostRef, host: ::lk_agents_abi::HostRef) -> PyResult<ServerRuntime> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::new(config, host).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ServerRuntime(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn worker_id(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::worker_id(&self.0);
        Ok(out)
    }

    #[getter]
    fn draining(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::draining(&self.0);
        Ok(out)
    }

    #[getter]
    fn http_port(&self) -> PyResult<u16> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::http_port(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (*, devmode, unregistered, has_rtc, has_text_doors, publisher, routes, has_load_fnc, has_on_request))]
    fn run(&self, devmode: bool, unregistered: bool, has_rtc: bool, has_text_doors: bool, publisher: bool, routes: ::lk_agents_abi::HostRef, has_load_fnc: bool, has_on_request: bool) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::run(&self.0, devmode, unregistered, has_rtc, has_text_doors, publisher, routes, has_load_fnc, has_on_request).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (timeout = None))]
    fn drain(&self, #[pyo3(from_py_with = _extract_server_runtime_drain_timeout)] timeout: ::std::option::Option<f64>) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::drain(&self.0, timeout).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn aclose(&self) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::aclose(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (job_id, status))]
    fn job_status(&self, job_id: ::std::string::String, status: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::job_status(&self.0, job_id, status);
        Ok(())
    }

    #[pyo3(signature = (room, *, fake_job, agent_identity = None, room_info = None, token = None))]
    fn simulate_job(&self, room: ::std::string::String, fake_job: bool, agent_identity: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_server_runtime_simulate_job_room_info)] room_info: ::std::option::Option<::std::vec::Vec<u8>>, token: ::std::option::Option<::std::string::String>) -> PyResult<::lk_agents_abi::HostRef> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::simulate_job(&self.0, room, fake_job, agent_identity, room_info, token).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/job.rs.

/// [`::lk_agents_abi::worker::job::SessionJob`] over a host `livekit.agents.job.JobContext`.
#[derive(Clone)]
pub struct SessionJobProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl SessionJobProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::job::SessionJob>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::job::SessionJob>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::job::SessionJob>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(SessionJobProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::worker::job::SessionJob>(|obj| SessionJobProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::worker::job::SessionJob for SessionJobProxy {
    fn connect(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionJob._core_connect", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_connect")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/remote_session.rs.

/// [`::lk_agents_abi::worker::remote_session::SessionPort`] over a host `livekit.agents.voice.remote_session.SessionPort`.
#[derive(Clone)]
pub struct SessionPortProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    framework_info: (::std::string::String, ::std::string::String),
}

impl SessionPortProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::remote_session::SessionPort>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::remote_session::SessionPort>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::remote_session::SessionPort>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let framework_info = {
            let v = crate::host::read(obj, "_core_framework_info")?;
            (|| -> ::pyo3::PyResult<_> { let tuple = v.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_framework_info"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?,)) })()?
        };
        Ok(SessionPortProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, framework_info })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::worker::remote_session::SessionPort>(|obj| SessionPortProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::worker::remote_session::SessionPort for SessionPortProxy {
    async fn chat_history(&self) -> ::std::result::Result<::std::vec::Vec<::lk_agents_abi::ChatItem>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_chat_history", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_chat_history")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ChatItem>().map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn agent_info(&self) -> ::std::result::Result<(::std::string::String, ::std::option::Option<::std::string::String>, ::std::vec::Vec<::std::string::String>, ::std::vec::Vec<::lk_agents_abi::ChatItem>), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_agent_info", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_agent_info")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 4 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 4 items", "SessionPort._core_agent_info"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; Ok(((item_0.extract::<::std::string::String>())?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?, (item_2.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()))?, (item_3.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ChatItem>().map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>()))?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn run_input(&self, text: &str) -> ::std::result::Result<(::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>, ::std::option::Option<::std::string::String>), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_run_input", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_run_input")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_run_input"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((if item_0.is_none() { Ok(None) } else { (item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ChatItem>().map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>())).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn session_state(&self) -> ::std::result::Result<(::std::string::String, ::std::string::String, ::std::string::String, ::std::collections::BTreeMap<::std::string::String, ::std::string::String>, ::std::option::Option<f64>), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_session_state", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_session_state")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "SessionPort._core_session_state"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<::std::string::String>())?, (item_3.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| dict.iter().map(|(key, item)| Ok((key.extract::<::std::string::String>()?, item.extract::<::std::string::String>()?))).collect::<PyResult<::std::collections::BTreeMap<_, _>>>()))?, (if item_4.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&item_4, "SessionPort._core_session_state")).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn rtc_stats(&self) -> ::std::result::Result<::std::option::Option<(::std::vec::Vec<::serde_json::Value>, ::std::vec::Vec<::serde_json::Value>)>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_rtc_stats", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_rtc_stats")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_rtc_stats"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::value_from_py(&item, "SessionPort._core_rtc_stats") }).collect::<PyResult<Vec<_>>>()))?, (item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::value_from_py(&item, "SessionPort._core_rtc_stats") }).collect::<PyResult<Vec<_>>>()))?,)) })()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn session_usage(&self) -> ::std::result::Result<::std::vec::Vec<::lk_agents_abi::metrics::ModelUsage>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_session_usage", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_session_usage")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::metrics::ModelUsage>() }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn framework_info(&self) -> (::std::string::String, ::std::string::String) {
        ::std::clone::Clone::clone(&self.framework_info)
    }

    async fn update_io(&self, update: (::std::option::Option<bool>, ::std::option::Option<bool>, ::std::option::Option<bool>, ::std::option::Option<bool>, ::std::option::Option<bool>)) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let update_arg = update;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_update_io", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(update_arg, py)?];
                Ok((host.bind(py).getattr("_core_update_io")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn finalize_simulation(&self, provisional: (bool, ::std::string::String)) -> ::std::result::Result<(::std::option::Option<(bool, ::std::string::String)>, ::std::option::Option<::std::string::String>), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let provisional_arg = provisional;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_finalize_simulation", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(provisional_arg, py)?];
                Ok((host.bind(py).getattr("_core_finalize_simulation")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((if item_0.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = item_0.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<bool>())?, (item_1.extract::<::std::string::String>())?,)) })()).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.

/// [`::lk_agents_abi::worker::cli::DevServer`] over a host `livekit.agents.cli._DevServer`.
#[derive(Clone)]
pub struct DevServerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl DevServerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::cli::DevServer>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::cli::DevServer>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::cli::DevServer>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(DevServerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::worker::cli::DevServer>(|obj| DevServerProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::worker::cli::DevServer for DevServerProxy {
    fn running_jobs(&self) -> ::std::result::Result<::std::vec::Vec<(::std::vec::Vec<u8>, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, bool)>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_running_jobs")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; (|| -> ::pyo3::PyResult<_> { let tuple = item.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 8 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 8 items", "DevServer._core_running_jobs"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; let item_5 = tuple.get_item(5)?; let item_6 = tuple.get_item(6)?; let item_7 = tuple.get_item(7)?; Ok((({ if item_0.is_instance_of::<::pyo3::types::PyBytes>() || item_0.is_instance_of::<::pyo3::types::PyByteArray>() || item_0.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&item_0, "DevServer._core_running_jobs") } else { item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "DevServer._core_running_jobs") }).collect::<PyResult<Vec<_>>>()) } })?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<::std::string::String>())?, (item_3.extract::<::std::string::String>())?, (item_4.extract::<::std::string::String>())?, (item_5.extract::<::std::string::String>())?, (item_6.extract::<::std::string::String>())?, (item_7.extract::<bool>())?,)) })() }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn reload_jobs(&self, jobs: ::std::vec::Vec<(::std::vec::Vec<u8>, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, bool)>) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let jobs_arg = jobs;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DevServer._core_reload_jobs", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(jobs_arg, py)?];
                Ok((host.bind(py).getattr("_core_reload_jobs")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
