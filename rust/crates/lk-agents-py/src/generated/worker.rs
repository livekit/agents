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

#[pyclass(frozen, name = "EndpointStream", module = "livekit.agents._core")]
pub struct EndpointStream(pub(crate) ::std::sync::Arc<EndpointStreamImpl>);

#[pymethods]
impl EndpointStream {
    #[pyo3(signature = (max_bytes = 65536))]
    fn read<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_endpoint_stream_read_max_bytes)] max_bytes: usize) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::read(&this, max_bytes)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (data))]
    fn write<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_endpoint_stream_write_data)] data: ::std::vec::Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::write(&this, data)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Idempotent; a stream already reset raises.
    fn finish(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::finish(&self.0).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (code))]
    fn reset(&self, #[pyo3(from_py_with = _extract_endpoint_stream_reset_code)] code: i32) -> PyResult<()> {
        crate::host::check_pid()?;
        <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::reset(&self.0, code).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (code))]
    fn stop(&self, #[pyo3(from_py_with = _extract_endpoint_stream_stop_code)] code: i32) -> PyResult<()> {
        crate::host::check_pid()?;
        <EndpointStreamImpl as ::lk_agents_abi::worker::endpoint_host::EndpointStream>::stop(&self.0, code).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::endpoint_host::ExchangeStream`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::endpoint_host::ExchangeStream>(),
        read: |obj| {
            let this = obj.cast::<EndpointStream>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::ExchangeStream> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
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
        crate::host::check_pid()?;
        let out = <EndpointServerImpl as ::lk_agents_abi::worker::endpoint_host::EndpointServer>::new();
        Ok(EndpointServer(::std::sync::Arc::new(out)))
    }

    /// Serve `stream` with `app` to completion; `request` is its preamble.
    #[pyo3(signature = (stream, app, request))]
    fn serve<'py>(&self, py: Python<'py>, stream: Bound<'_, PyAny>, app: Bound<'_, PyAny>, request: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let stream = crate::generated::worker::ExchangeStreamProxy::from_host(&stream)?;
        let app = crate::generated::worker::AsgiApplicationProxy::from_host(&app)?;
        let request = crate::generated::records::worker_endpoint::endpoint_request_from_py(&request)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <EndpointServerImpl as ::lk_agents_abi::worker::endpoint_host::EndpointServer>::serve(&this, stream, app, request)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// A close frame to every open WebSocket, so clients reconnect to another worker.
    #[pyo3(signature = (code = 1001, reason = "worker draining".to_owned()))]
    fn close_websockets<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_endpoint_server_close_websockets_code)] code: u16, reason: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <EndpointServerImpl as ::lk_agents_abi::worker::endpoint_host::EndpointServer>::close_websockets(&this, code, reason)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::endpoint_host::EndpointService`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::endpoint_host::EndpointService>(),
        read: |obj| {
            let this = obj.cast::<EndpointServer>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::EndpointService> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
type _core_AsgiReceiveImpl = <crate::Selected as super::Implementations>::AsgiReceive;

#[pyclass(frozen, name = "_core_AsgiReceive", module = "livekit.agents._core")]
pub struct _core_AsgiReceive(pub(crate) ::std::sync::Arc<_core_AsgiReceiveImpl>);

#[pymethods]
impl _core_AsgiReceive {
    fn __call__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_AsgiReceiveImpl as ::lk_agents_abi::worker::endpoint_host::AsgiReceive>::__call__(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::worker_endpoint::asgi_received_to_py(py, out)?, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::endpoint_host::AsgiReceiver`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::endpoint_host::AsgiReceiver>(),
        read: |obj| {
            let this = obj.cast::<_core_AsgiReceive>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::AsgiReceiver> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
type _core_AsgiSendImpl = <crate::Selected as super::Implementations>::AsgiSend;

#[pyclass(frozen, name = "_core_AsgiSend", module = "livekit.agents._core")]
pub struct _core_AsgiSend(pub(crate) ::std::sync::Arc<_core_AsgiSendImpl>);

#[pymethods]
impl _core_AsgiSend {
    #[pyo3(signature = (message))]
    fn __call__<'py>(&self, py: Python<'py>, message: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let message = crate::generated::records::worker_endpoint::asgi_sent_from_py(&message)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_AsgiSendImpl as ::lk_agents_abi::worker::endpoint_host::AsgiSend>::__call__(&this, message)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::endpoint_host::AsgiSender`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::endpoint_host::AsgiSender>(),
        read: |obj| {
            let this = obj.cast::<_core_AsgiSend>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::AsgiSender> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
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
    crate::host::check_pid()?;
    let out = <JobTraceImpl as ::lk_agents_abi::worker::job::JobTrace>::job_timeline(received_at, accepted_at, assigned_at, launched_at, entrypoint_started_at, server_started_at);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
type ServerEnvironmentImpl = <crate::Selected as super::Implementations>::ServerEnvironment;

/// Logical CPUs, fractional under a cgroup quota. Reads, starts nothing.
#[pyfunction]
pub fn cpu_count() -> PyResult<f64> {
    crate::host::check_pid()?;
    let out = <ServerEnvironmentImpl as ::lk_agents_abi::worker::worker::ServerEnvironment>::cpu_count();
    Ok(out)
}

/// `https://<project>/agents/<agent>/<deployment>` (`AgentServer._endpoint_url_base`), when
/// the URL and the agent are known; a worker registered without a deployment is `default`.
#[pyfunction]
#[pyo3(signature = (ws_url, agent_name, deployment))]
pub fn endpoint_url_base(ws_url: ::std::string::String, agent_name: ::std::string::String, deployment: ::std::string::String) -> PyResult<::std::option::Option<::std::string::String>> {
    crate::host::check_pid()?;
    let out = <ServerEnvironmentImpl as ::lk_agents_abi::worker::worker::ServerEnvironment>::endpoint_url_base(ws_url, agent_name, deployment);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.
type WorkerCliImpl = <crate::Selected as super::Implementations>::WorkerCli;

#[pyfunction]
#[pyo3(signature = (argv))]
pub fn parse_cli<'py>(py: Python<'py>, argv: ::std::vec::Vec<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::parse_cli(argv).map_err(|e: ::lk_agents_abi::worker::cli::CliError| <::lk_agents_abi::worker::cli::CliError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(crate::generated::records::worker_cli::cli_args_to_py(py, out)?)
}

#[pyfunction]
#[pyo3(signature = (path))]
pub fn entrypoint_module(path: ::std::option::Option<::std::string::String>) -> PyResult<(::std::string::String, ::std::string::String)> {
    crate::host::check_pid()?;
    let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::entrypoint_module(path).map_err(|e: ::lk_agents_abi::worker::cli::CliError| <::lk_agents_abi::worker::cli::CliError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// One log line in `style`: `message` is the formatted message, `asctime` the record's time
/// as the host formats it, `exception` and `stack` the tracebacks.
#[pyfunction]
#[pyo3(signature = (style, record, message, asctime, exception, stack, ensure_ascii))]
pub fn format_log_record(style: ::std::string::String, record: Bound<'_, PyAny>, message: ::std::string::String, asctime: ::std::string::String, exception: ::std::option::Option<::std::string::String>, stack: ::std::option::Option<::std::string::String>, ensure_ascii: bool) -> PyResult<::std::string::String> {
    let record = crate::generated::records::worker_cli::log_fields_from_py(&record)?;
    let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::format_log_record(style, record, message, asctime, exception, stack, ensure_ascii);
    Ok(out)
}

/// The development launcher connection.
#[pyfunction]
#[pyo3(signature = (cli_addr, agent_name, url, api_key, api_secret, server))]
pub fn dev_channel<'py>(py: Python<'py>, cli_addr: ::std::string::String, agent_name: ::std::string::String, url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, server: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let server = crate::generated::worker::DevServerProxy::from_host(&server)?;
    crate::host::future_into_py(py, async move {
        let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::dev_channel(cli_addr, agent_name, url, api_key, api_secret, server).await;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

/// Create the control endpoints for process bootstrap.
#[pyfunction]
pub fn ipc_pair() -> PyResult<(i64, i64)> {
    crate::host::check_pid()?;
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
        crate::host::check_pid()?;
        let out = <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::new(escalate);
        Ok(ExitSignals(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (exit_served))]
    fn signal(&self, exit_served: bool) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::signal(&self.0, exit_served);
        Ok(out)
    }

    fn served(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::served(&self.0);
        Ok(())
    }

    fn draining(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::draining(&self.0);
        Ok(())
    }

    /// Arm the native deadline before the host cancels tasks or closes its executor.
    fn closing(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::closing(&self.0);
        Ok(())
    }

    /// Disarm the deadline only after the host event loop has actually closed.
    fn closed(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <ExitSignalsImpl as ::lk_agents_abi::worker::cli::CliSignals>::closed(&self.0);
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type ProcSupervisorImpl = <crate::Selected as super::Implementations>::ProcessSupervisor;

fn _extract_process_supervisor_new_fd(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "fd")
}

fn _extract_process_supervisor_new_pid(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "pid")
}

fn _extract_process_supervisor_new_initialize_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "initialize_timeout")
}

fn _extract_process_supervisor_new_close_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "close_timeout")
}

fn _extract_process_supervisor_new_memory_warn_mb(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "memory_warn_mb")
}

fn _extract_process_supervisor_new_memory_limit_mb(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "memory_limit_mb")
}

fn _extract_process_supervisor_new_ping_interval(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "ping_interval")
}

fn _extract_process_supervisor_new_ping_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "ping_timeout")
}

fn _extract_process_supervisor_new_high_ping_threshold(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "high_ping_threshold")
}

fn _extract_process_supervisor__core_launched_process_pid(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "pid")
}

fn _extract_process_supervisor__core_launched_process_sentinel(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "sentinel")
}

fn _extract_process_supervisor__core_launched_process_owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<i64>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<i64>(&item, "owned") }).collect::<PyResult<Vec<_>>>())
}

#[pyclass(frozen, name = "ProcSupervisor", module = "livekit.agents._core")]
pub struct ProcSupervisor(pub(crate) ::std::sync::Arc<ProcSupervisorImpl>);

#[pymethods]
impl ProcSupervisor {
    #[new]
    #[pyo3(signature = (fd, host, *, pid, initialize_timeout, close_timeout, memory_warn_mb, memory_limit_mb, ping_interval, ping_timeout, high_ping_threshold, killable = true))]
    fn new(#[pyo3(from_py_with = _extract_process_supervisor_new_fd)] fd: i64, host: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_process_supervisor_new_pid)] pid: u32, #[pyo3(from_py_with = _extract_process_supervisor_new_initialize_timeout)] initialize_timeout: f64, #[pyo3(from_py_with = _extract_process_supervisor_new_close_timeout)] close_timeout: f64, #[pyo3(from_py_with = _extract_process_supervisor_new_memory_warn_mb)] memory_warn_mb: f64, #[pyo3(from_py_with = _extract_process_supervisor_new_memory_limit_mb)] memory_limit_mb: f64, #[pyo3(from_py_with = _extract_process_supervisor_new_ping_interval)] ping_interval: f64, #[pyo3(from_py_with = _extract_process_supervisor_new_ping_timeout)] ping_timeout: f64, #[pyo3(from_py_with = _extract_process_supervisor_new_high_ping_threshold)] high_ping_threshold: f64, killable: bool) -> PyResult<ProcSupervisor> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::worker::JobExecutorProxy::from_host(&host)?;
        let context = crate::host::entered();
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::new(&env, fd, host, pid, initialize_timeout, close_timeout, memory_warn_mb, memory_limit_mb, ping_interval, ping_timeout, high_ping_threshold, killable, context).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ProcSupervisor(::std::sync::Arc::new(out)))
    }

    /// The seconds the process took to be ready.
    #[pyo3(signature = (*, asyncio_debug, http_proxy))]
    fn initialize<'py>(&self, py: Python<'py>, asyncio_debug: bool, http_proxy: ::std::option::Option<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::initialize(&this, asyncio_debug, http_proxy)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[getter]
    fn initialized(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::initialized(&self.0);
        Ok(out)
    }

    /// The launch's receipt, reported on the launch worker right after the OS created the child.
    #[pyo3(signature = (pid, sentinel, method, owned))]
    fn _core_launched_process(&self, #[pyo3(from_py_with = _extract_process_supervisor__core_launched_process_pid)] pid: u32, #[pyo3(from_py_with = _extract_process_supervisor__core_launched_process_sentinel)] sentinel: i64, method: ::std::string::String, #[pyo3(from_py_with = _extract_process_supervisor__core_launched_process_owned)] owned: ::std::vec::Vec<i64>) -> PyResult<()> {
        crate::host::check_pid()?;
        <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::_core_launched_process(&self.0, pid, sentinel, method, owned).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// Launch, supervise, observe the exit and reap; whether the process ran (an OS child was
    /// reaped). Cancelling the wait leaves the process owned.
    fn run<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::run(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn _core_wait_exit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::_core_wait_exit(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn _core_join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::_core_join(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn kill<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::kill(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (fields))]
    fn start_job<'py>(&self, py: Python<'py>, fields: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let fields = crate::generated::records::worker_ipc::start_job_fields_from_py(&fields)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::start_job(&this, fields)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[getter]
    fn killed(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <ProcSupervisorImpl as ::lk_agents_abi::worker::ipc_host::ProcessSupervisor>::killed(&self.0);
        Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::ipc_host::SupervisedProcess`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::ipc_host::SupervisedProcess>(),
        read: |obj| {
            let this = obj.cast::<ProcSupervisor>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::SupervisedProcess> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type IpcClientImpl = <crate::Selected as super::Implementations>::IpcClient;

fn _extract_ipc_client_new_fd(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "fd")
}

#[pyclass(frozen, name = "IpcClient", module = "livekit.agents._core")]
pub struct IpcClient(pub(crate) ::std::sync::Arc<IpcClientImpl>);

#[pymethods]
impl IpcClient {
    /// The handshake on the parent's socket, blocking before the process's loop exists.
    #[new]
    #[pyo3(signature = (fd, initialize, *, process = false))]
    fn new<'py>(py: Python<'py>, #[pyo3(from_py_with = _extract_ipc_client_new_fd)] fd: i64, initialize: Bound<'_, PyAny>, process: bool) -> PyResult<IpcClient> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let initialize = crate::generated::worker::ChildInitializeProxy::from_host(&initialize)?;
        let out = py.detach(|| <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::new(&env, fd, initialize, process)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(IpcClient(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn asyncio_debug(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::asyncio_debug(&self.0);
        Ok(out)
    }

    /// Serve the parent's messages until it goes, its shutdown ends the process, or `stop()`.
    #[pyo3(signature = (host))]
    fn serve<'py>(&self, py: Python<'py>, host: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let host = crate::generated::worker::JobProcessProxy::from_host(&host)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::serve(&this, host)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn stop(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::stop(&self.0);
        Ok(())
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type ProcPoolImpl = <crate::Selected as super::Implementations>::ProcessPool;

fn _extract_process_pool_new_num_idle_processes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "num_idle_processes")
}

fn _extract_process_pool_new_initialize_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "initialize_timeout")
}

fn _extract_process_pool_new_close_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "close_timeout")
}

fn _extract_process_pool_set_target_idle_processes_n(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "n")
}

#[pyclass(frozen, name = "ProcPool", module = "livekit.agents._core")]
pub struct ProcPool(pub(crate) ::std::sync::Arc<ProcPoolImpl>);

#[pymethods]
impl ProcPool {
    #[new]
    #[pyo3(signature = (host, *, num_idle_processes, initialize_timeout, close_timeout))]
    fn new(host: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_process_pool_new_num_idle_processes)] num_idle_processes: usize, #[pyo3(from_py_with = _extract_process_pool_new_initialize_timeout)] initialize_timeout: f64, #[pyo3(from_py_with = _extract_process_pool_new_close_timeout)] close_timeout: f64) -> PyResult<ProcPool> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::worker::PoolCallbacksProxy::from_host(&host)?;
        let context = crate::host::entered();
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::new(&env, host, num_idle_processes, initialize_timeout, close_timeout, context).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ProcPool(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn processes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::processes(&self.0);
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(crate::generated::worker::JobExecutorProxy::to_host(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    #[getter]
    fn jobs_waiting(&self) -> PyResult<usize> {
        crate::host::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::jobs_waiting(&self.0);
        Ok(out)
    }

    #[getter]
    fn target_idle_processes(&self) -> PyResult<usize> {
        crate::host::check_pid()?;
        let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::target_idle_processes(&self.0);
        Ok(out)
    }

    #[setter(target_idle_processes)]
    fn set_target_idle_processes(&self, #[pyo3(from_py_with = _extract_process_pool_set_target_idle_processes_n)] n: usize) -> PyResult<()> {
        crate::host::check_pid()?;
        <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::set_target_idle_processes(&self.0, n);
        Ok(())
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (job_id, info))]
    fn launch_job<'py>(&self, py: Python<'py>, job_id: ::std::string::String, info: ::lk_agents_abi::HostRef) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::launch_job(&this, job_id, info)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Waits until every process a job was launched on exited (the drain).
    fn join_jobs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::join_jobs(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type JobRunImpl = <crate::Selected as super::Implementations>::JobRun;

fn _extract_job_run_new_session_end_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "session_end_timeout")
}

fn _extract_job_run_new_session_aclose_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "session_aclose_timeout")
}

fn _extract_job_run_add_participant_entrypoint_kinds(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<i32>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<i32>(&item, "kinds") }).collect::<PyResult<Vec<_>>>())
}

fn _extract_job_run_participant_joined_kind(obj: &Bound<'_, PyAny>) -> PyResult<i32> {
    ::lk_agents_abi::python::number_from_py::<i32>(&obj, "kind")
}

/// A job's run: what ends the job (the first reason wins), the entrypoint's supervision, the
/// order and bounds of the shutdown, the participant entrypoints, the room connected once, the
/// application calls in flight and the recording's setup.
#[pyclass(frozen, name = "JobRun", module = "livekit.agents._core")]
pub struct JobRun(pub(crate) ::std::sync::Arc<JobRunImpl>);

#[pymethods]
impl JobRun {
    /// `session_end` names the door's `on_session_end`, when it has one.
    #[new]
    #[pyo3(signature = (*, fake_job, session_end_timeout, redaction, url = "".to_owned(), recording = false, session_end = None, session_aclose_timeout = 60.0))]
    fn new(fake_job: bool, #[pyo3(from_py_with = _extract_job_run_new_session_end_timeout)] session_end_timeout: f64, redaction: bool, url: ::std::string::String, recording: bool, session_end: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_job_run_new_session_aclose_timeout)] session_aclose_timeout: f64) -> PyResult<JobRun> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::new(&env, fake_job, session_end_timeout, redaction, url, recording, session_end, session_aclose_timeout).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(JobRun(::std::sync::Arc::new(out)))
    }

    /// `callback(reason)` when it takes the reason, else `callback()`, as the job shuts down; a
    /// framework callback runs in the shutdown's span, any other in its own span named `name`.
    #[pyo3(signature = (callback, takes_reason, framework, name))]
    fn add_shutdown_callback(&self, callback: ::lk_agents_abi::HostRef, takes_reason: bool, framework: bool, name: ::std::string::String) -> PyResult<()> {
        crate::host::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::add_shutdown_callback(&self.0, callback, takes_reason, framework, name);
        Ok(())
    }

    #[getter]
    fn observability_url(&self) -> PyResult<::std::option::Option<::std::string::String>> {
        crate::host::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::observability_url(&self.0);
        Ok(out)
    }

    #[getter]
    fn buffer_logs(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::buffer_logs(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (reason, user_initiated))]
    fn shutdown(&self, reason: ::std::string::String, user_initiated: bool) -> PyResult<()> {
        crate::host::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::shutdown(&self.0, reason, user_initiated);
        Ok(())
    }

    fn connected(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::connected(&self.0);
        Ok(())
    }

    /// Awaits `connect()` once: a call while it runs waits for it, a call after it connected
    /// returns at once, a failed one leaves the next call to try again.
    #[pyo3(signature = (connect))]
    fn connect<'py>(&self, py: Python<'py>, connect: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let connect = crate::generated::worker::JobCallProxy::from_host(&connect)?;
        let context = crate::host::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::connect(&this, connect, context)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `entrypoint` (named `name`) runs for each joining participant of one of `kinds`.
    #[pyo3(signature = (entrypoint, kinds, name))]
    fn add_participant_entrypoint(&self, entrypoint: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_job_run_add_participant_entrypoint_kinds)] kinds: ::std::vec::Vec<i32>, name: ::std::string::String) -> PyResult<()> {
        crate::host::check_pid()?;
        let entrypoint = crate::generated::worker::ParticipantEntrypointProxy::from_host(&entrypoint)?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::add_participant_entrypoint(&self.0, entrypoint, kinds, name);
        Ok(())
    }

    /// A participant joined, or was in the room when it connected; `job` is the `JobContext`.
    #[pyo3(signature = (job, participant, identity, kind))]
    fn participant_joined(&self, job: ::lk_agents_abi::HostRef, participant: ::lk_agents_abi::HostRef, identity: ::std::string::String, #[pyo3(from_py_with = _extract_job_run_participant_joined_kind)] kind: i32) -> PyResult<()> {
        crate::host::check_pid()?;
        let context = crate::host::entered();
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::participant_joined(&self.0, job, participant, identity, kind, context);
        Ok(())
    }

    #[pyo3(signature = (attributes))]
    fn participant_left(&self, attributes: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>) -> PyResult<()> {
        crate::host::check_pid()?;
        <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::participant_left(&self.0, attributes);
        Ok(())
    }

    /// An application API call (`delete_room`, ...): its failure is logged whether or not the
    /// caller awaits it, and it is cancelled once the job ends. A console job makes no call: it
    /// resolves to `console_result`.
    #[pyo3(signature = (call, name, console_result))]
    fn api_call<'py>(&self, py: Python<'py>, call: Bound<'_, PyAny>, name: ::std::string::String, console_result: ::lk_agents_abi::HostRef) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = crate::generated::worker::JobCallProxy::from_host(&call)?;
        let context = crate::host::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::api_call(&this, call, name, console_result, context)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Cancels the API calls still running and waits for them.
    fn close_tasks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::close_tasks(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// The redaction to record with when the cloud upload is to be set up now, `None` when not.
    #[pyo3(signature = (options))]
    fn init_recording(&self, options: ::std::collections::BTreeMap<::std::string::String, bool>) -> PyResult<::std::option::Option<bool>> {
        crate::host::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::init_recording(&self.0, options).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[getter]
    fn recording_initialized(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::recording_initialized(&self.0);
        Ok(out)
    }

    #[getter]
    fn redaction(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::redaction(&self.0);
        Ok(out)
    }

    /// The entrypoint, then the shutdown sequence once the job ends; `host` runs each step in the
    /// context this call captured.
    #[pyo3(signature = (host))]
    fn run<'py>(&self, py: Python<'py>, host: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let host = crate::generated::worker::JobProcessProxy::from_host(&host)?;
        let context = crate::host::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::run(&this, host, context)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::ipc_host::JobExecution`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::ipc_host::JobExecution>(),
        read: |obj| {
            let this = obj.cast::<JobRun>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecution> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/server_host.rs.
type ServerConfigImpl = <crate::Selected as super::Implementations>::ServerConfig;

#[pyclass(frozen, name = "ServerConfig", module = "livekit.agents._core")]
pub struct ServerConfig(pub(crate) ::std::sync::Arc<ServerConfigImpl>);

#[pymethods]
impl ServerConfig {
    /// `options` left empty read the environment (`http_proxy`: `HTTPS_PROXY`, else
    /// `HTTP_PROXY`).
    #[new]
    #[pyo3(signature = (options))]
    fn new(options: ::lk_agents_abi::worker::worker::ServerOptions) -> PyResult<ServerConfig> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::new(options).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ServerConfig(::std::sync::Arc::new(out)))
    }

    /// Before the server starts only.
    #[pyo3(signature = (options))]
    fn update(&self, options: ::lk_agents_abi::worker::worker::ServerOptions) -> PyResult<()> {
        crate::host::check_pid()?;
        <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::update(&self.0, options).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn ws_url(&self) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::ws_url(&self.0);
        Ok(out)
    }

    #[getter]
    fn api_key(&self) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::api_key(&self.0);
        Ok(out)
    }

    #[getter]
    fn api_secret(&self) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::api_secret(&self.0);
        Ok(out)
    }

    #[getter]
    fn agent_name(&self) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::agent_name(&self.0);
        Ok(out)
    }

    #[getter]
    fn deployment(&self) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::deployment(&self.0);
        Ok(out)
    }

    /// Hosted on Cloud (`LIVEKIT_WORKER_TOKEN` set).
    #[getter]
    fn cloud(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::cloud(&self.0);
        Ok(out)
    }

    #[getter]
    fn drain_timeout(&self) -> PyResult<f64> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::drain_timeout(&self.0);
        Ok(out)
    }

    #[getter]
    fn simulation(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::simulation(&self.0);
        Ok(out)
    }

    #[setter(simulation)]
    fn set_simulation(&self, value: bool) -> PyResult<()> {
        crate::host::check_pid()?;
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

#[pyclass(frozen, name = "ServerRuntime", module = "livekit.agents._core")]
pub struct ServerRuntime(pub(crate) ::std::sync::Arc<ServerRuntimeImpl>);

#[pymethods]
impl ServerRuntime {
    #[new]
    #[pyo3(signature = (config, host))]
    fn new(config: Bound<'_, PyAny>, host: Bound<'_, PyAny>) -> PyResult<ServerRuntime> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let config = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::server_host::ServerConfiguration>(&config).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("config", ": expected a native ServerConfig")))?;
        let host = crate::generated::worker::ServerHostProxy::from_host(&host)?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::new(&env, config, host).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ServerRuntime(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn worker_id(&self) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::worker_id(&self.0);
        Ok(out)
    }

    #[getter]
    fn draining(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::draining(&self.0);
        Ok(out)
    }

    #[getter]
    fn http_port(&self) -> PyResult<u16> {
        crate::host::check_pid()?;
        let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::http_port(&self.0);
        Ok(out)
    }

    /// Start the job processes and the health listener and, unless `unregistered`, register;
    /// return once closed. `endpoints` serves `routes` with `app`.
    #[pyo3(signature = (*, devmode, unregistered, has_rtc, has_text_doors, routes, has_load_fnc, endpoints, app))]
    fn run<'py>(&self, py: Python<'py>, devmode: bool, unregistered: bool, has_rtc: bool, has_text_doors: bool, routes: Bound<'_, PyAny>, has_load_fnc: bool, endpoints: Bound<'_, PyAny>, app: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let routes = routes.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::worker_server::endpoint_route_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let endpoints = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::endpoint_host::EndpointService>(&endpoints).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("endpoints", ": expected a native EndpointServer")))?;
        let app = crate::generated::worker::AsgiApplicationProxy::from_host(&app)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::run(&this, devmode, unregistered, has_rtc, has_text_doors, routes, has_load_fnc, endpoints, app)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `None` (or `0`) waits without a deadline; past it raises `TimeoutError`.
    #[pyo3(signature = (timeout = None))]
    fn drain<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_server_runtime_drain_timeout)] timeout: ::std::option::Option<f64>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::drain(&this, timeout)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `"running"`, `"failed"` or `"success"`.
    #[pyo3(signature = (job_id, status))]
    fn job_status(&self, job_id: ::std::string::String, status: ::std::string::String) -> PyResult<()> {
        crate::host::check_pid()?;
        <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::job_status(&self.0, job_id, status);
        Ok(())
    }

    /// A job in `room` with no dispatch: the console's, a notebook's.
    #[pyo3(signature = (room, *, fake_job, agent_identity = None, room_info = None, token = None))]
    fn simulate_job<'py>(&self, py: Python<'py>, room: ::std::string::String, fake_job: bool, agent_identity: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_server_runtime_simulate_job_room_info)] room_info: ::std::option::Option<::std::vec::Vec<u8>>, token: ::std::option::Option<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::simulate_job(&this, room, fake_job, agent_identity, room_info, token)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.

/// [`::lk_agents_abi::worker::endpoint_host::ExchangeStream`] over a host `livekit.agents.http.ExchangeStream`.
#[derive(Clone)]
pub struct ExchangeStreamProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl ExchangeStreamProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::ExchangeStream>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::endpoint_host::ExchangeStream>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::ExchangeStream>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::endpoint_host::ExchangeStream>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::endpoint_host::ExchangeStream>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(ExchangeStreamProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
    }
}

impl ::lk_agents_abi::worker::endpoint_host::ExchangeStream for ExchangeStreamProxy {
    fn read(&self, max_bytes: usize) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::vec::Vec<u8>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let max_bytes_arg = max_bytes;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ExchangeStream.read", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(max_bytes_arg, py)?];
                Ok((host.bind(py).getattr("read")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ if o.is_instance_of::<::pyo3::types::PyBytes>() || o.is_instance_of::<::pyo3::types::PyByteArray>() || o.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&o, "ExchangeStream.read") } else { o.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "ExchangeStream.read") }).collect::<PyResult<Vec<_>>>()) } }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn write(&self, data: ::std::vec::Vec<u8>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let data_arg = data;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ExchangeStream.write", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(data_arg, py)?];
                Ok((host.bind(py).getattr("write")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn finish(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "ExchangeStream.finish", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("finish")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn reset(&self, code: i32) {
        let host = ::std::sync::Arc::clone(&self.host);
        let code_arg = code;
        ::lk_agents_runtime::host::notify(&self.notifier, "ExchangeStream.reset", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(code_arg, py)?];
                Ok((host.bind(py).getattr("reset")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn stop(&self, code: i32) {
        let host = ::std::sync::Arc::clone(&self.host);
        let code_arg = code;
        ::lk_agents_runtime::host::notify(&self.notifier, "ExchangeStream.stop", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(code_arg, py)?];
                Ok((host.bind(py).getattr("stop")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.

/// [`::lk_agents_abi::worker::endpoint_host::AsgiApplication`] over a host `livekit.agents.http.AsgiApplication`.
#[derive(Clone)]
pub struct AsgiApplicationProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl AsgiApplicationProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::AsgiApplication>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::endpoint_host::AsgiApplication>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::AsgiApplication>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::endpoint_host::AsgiApplication>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::endpoint_host::AsgiApplication>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AsgiApplicationProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::worker::endpoint_host::AsgiApplication for AsgiApplicationProxy {
    fn call(&self, scope: ::lk_agents_abi::worker::endpoint_host::AsgiScope, request: ::lk_agents_abi::worker::endpoint::EndpointRequest, receive: ::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::AsgiReceiver>, send: ::std::sync::Arc<dyn ::lk_agents_abi::worker::endpoint_host::AsgiSender>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::worker::endpoint_host::AsgiFailure>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let scope_arg = scope;
        let request_arg = request;
        let receive_arg = receive;
        let send_arg = send;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AsgiApplication._core_call", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [crate::generated::records::worker_endpoint::asgi_scope_to_py(py, scope_arg)?, crate::generated::records::worker_endpoint::endpoint_request_to_py(py, request_arg)?, <dyn ::lk_agents_abi::worker::endpoint_host::AsgiReceiver as super::NativeToHost>::to_host(py, receive_arg)?, <dyn ::lk_agents_abi::worker::endpoint_host::AsgiSender as super::NativeToHost>::to_host(py, send_arg)?];
                Ok((host.bind(py).getattr("_core_call")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::worker_endpoint::asgi_failure_from_py(&o)).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::job::SessionJob>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::job::SessionJob>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::job::SessionJob>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(SessionJobProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::remote_session::SessionPort>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::remote_session::SessionPort>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::remote_session::SessionPort>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let framework_info = {
            let v = crate::host::read(obj, "_core_framework_info")?;
            (|| -> ::pyo3::PyResult<_> { let tuple = v.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_framework_info"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?,)) })()?
        };
        Ok(SessionPortProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, framework_info })
    }
}

impl ::lk_agents_abi::worker::remote_session::SessionPort for SessionPortProxy {
    fn chat_history(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::vec::Vec<::lk_agents_abi::ChatItem>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_chat_history", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_chat_history")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ChatItem>().map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn agent_info(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(::std::string::String, ::std::option::Option<::std::string::String>, ::std::vec::Vec<::std::string::String>, ::std::vec::Vec<::lk_agents_abi::ChatItem>), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_agent_info", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_agent_info")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 4 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 4 items", "SessionPort._core_agent_info"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; Ok(((item_0.extract::<::std::string::String>())?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?, (item_2.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()))?, (item_3.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ChatItem>().map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>()))?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn run_input<'a>(&'a self, text: &str) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>, ::std::option::Option<::std::string::String>), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_run_input", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_run_input")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_run_input"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((if item_0.is_none() { Ok(None) } else { (item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ChatItem>().map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>())).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn session_state(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(::std::string::String, ::std::string::String, ::std::string::String, ::std::collections::BTreeMap<::std::string::String, ::std::string::String>, ::std::option::Option<f64>), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_session_state", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_session_state")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "SessionPort._core_session_state"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<::std::string::String>())?, (item_3.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| dict.iter().map(|(key, item)| Ok((key.extract::<::std::string::String>()?, item.extract::<::std::string::String>()?))).collect::<PyResult<::std::collections::BTreeMap<_, _>>>()))?, (if item_4.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&item_4, "SessionPort._core_session_state")).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn rtc_stats(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<(::std::vec::Vec<::serde_json::Value>, ::std::vec::Vec<::serde_json::Value>)>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_rtc_stats", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_rtc_stats")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (if o.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_rtc_stats"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::value_from_py(&item, "SessionPort._core_rtc_stats") }).collect::<PyResult<Vec<_>>>()))?, (item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::value_from_py(&item, "SessionPort._core_rtc_stats") }).collect::<PyResult<Vec<_>>>()))?,)) })()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn session_usage(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::vec::Vec<::lk_agents_abi::metrics::ModelUsage>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_session_usage", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_session_usage")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::usage::model_usage_from_py(&item) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn framework_info(&self) -> (::std::string::String, ::std::string::String) {
        ::std::clone::Clone::clone(&self.framework_info)
    }

    fn update_io(&self, update: (::std::option::Option<bool>, ::std::option::Option<bool>, ::std::option::Option<bool>, ::std::option::Option<bool>, ::std::option::Option<bool>)) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let update_arg = update;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_update_io", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(update_arg, py)?];
                Ok((host.bind(py).getattr("_core_update_io")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn finalize_simulation(&self, provisional: (bool, ::std::string::String)) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(::std::option::Option<(bool, ::std::string::String)>, ::std::option::Option<::std::string::String>), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let provisional_arg = provisional;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionPort._core_finalize_simulation", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(provisional_arg, py)?];
                Ok((host.bind(py).getattr("_core_finalize_simulation")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((if item_0.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = item_0.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<bool>())?, (item_1.extract::<::std::string::String>())?,)) })()).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::cli::DevServer>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::cli::DevServer>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::cli::DevServer>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(DevServerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

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

    fn reload_jobs(&self, jobs: ::std::vec::Vec<(::std::vec::Vec<u8>, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, bool)>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let jobs_arg = jobs;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DevServer._core_reload_jobs", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(jobs_arg, py)?];
                Ok((host.bind(py).getattr("_core_reload_jobs")?, ::pyo3::types::PyTuple::new(py, args)?))
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

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.

/// [`::lk_agents_abi::worker::ipc_host::JobExecutor`] over a host `livekit.agents.ipc.JobExecutor`.
#[derive(Clone)]
pub struct JobExecutorProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host object it was taken from, handed back as itself.
    source: ::std::sync::Arc<Py<PyAny>>,
}

impl JobExecutorProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let source = ::std::sync::Arc::new(obj.clone().unbind());
        let _ = context;
        Ok(JobExecutorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), source })
    }
}

impl JobExecutorProxy {
    /// The host object behind a proxy of `::lk_agents_abi::worker::ipc_host::JobExecutor`; a native implementation has none.
    pub fn to_host<'py>(py: Python<'py>, value: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>) -> PyResult<Bound<'py, PyAny>> {
        let any: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = value;
        match any.downcast::<Self>() {
            Ok(proxy) => Ok(proxy.source.bind(py).clone()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err("a native `JobExecutor` has no host object to hand back")),
        }
    }
}

impl ::lk_agents_abi::worker::ipc_host::JobExecutor for JobExecutorProxy {
    fn launch<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor._core_launch", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_launch")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn launched<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor._core_launched", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_launched")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn launch_failed<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, cause: ::std::string::String) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let cause_arg = cause;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor._core_launch_failed", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(cause_arg, py)?];
                Ok((host.bind(py).getattr("_core_launch_failed")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn completed<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, completion: ::lk_agents_abi::worker::ipc::ProcessCompletion, pid: u32) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let completion_arg = completion;
        let pid_arg = pid;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor._core_completed", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::worker_ipc::process_completion_to_py(py, completion_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(pid_arg, py)?];
                Ok((host.bind(py).getattr("_core_completed")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn log(&self, level: i32, message: ::std::string::String, extra: ::serde_json::Map<::std::string::String, ::serde_json::Value>, error: ::std::option::Option<::lk_agents_abi::errors::HostError>) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let level_arg = level;
        let message_arg = message;
        let extra_arg = extra;
        let error_arg = error;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [::pyo3::IntoPyObjectExt::into_bound_py_any(level_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(message_arg, py)?, ::lk_agents_abi::python::extra_to_py(py, &extra_arg)?.into_any(), match error_arg { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? }];
                Ok((host.bind(py).getattr("_core_log")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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

    fn start<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::SupervisedProcess>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor._core_start", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_start")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::SupervisedProcess>(&o).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("JobExecutor._core_start", ": expected a native ProcSupervisor")))).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn initialize<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor.initialize", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("initialize")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn launch_job<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, info: ::lk_agents_abi::HostRef, launched_at: f64) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let info_arg = info;
        let launched_at_arg = launched_at;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor._core_launch_job", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(info_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(launched_at_arg, py)?];
                Ok((host.bind(py).getattr("_core_launch_job")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn close<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor.aclose", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn join<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<bool, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobExecutor._core_join", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_join")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.

/// [`::lk_agents_abi::worker::ipc_host::ChildInitialize`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct ChildInitializeProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl ChildInitializeProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(ChildInitializeProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::worker::ipc_host::ChildInitialize for ChildInitializeProxy {
    fn initialize(&self, http_proxy: ::std::option::Option<::std::string::String>) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let http_proxy_arg = http_proxy;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(http_proxy_arg, py)?];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.

/// [`::lk_agents_abi::worker::ipc_host::JobProcess`] over a host `livekit.agents.ipc.child._JobProc`.
#[derive(Clone)]
pub struct JobProcessProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl JobProcessProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(JobProcessProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::worker::ipc_host::JobProcess for JobProcessProxy {
    fn ping(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_ping", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_ping")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn start_job(&self, fields: ::lk_agents_abi::worker::job::StartJobFields) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecution>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let fields_arg = fields;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_start_job", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::worker_ipc::start_job_fields_to_py(py, fields_arg)?];
                Ok((host.bind(py).getattr("_core_start_job")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::JobExecution>(&o).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("JobProcess._core_start_job", ": expected a native JobRun")))).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn dump_stacks(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_dump_stacks", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_dump_stacks")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn start_entrypoint<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_start_entrypoint", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_start_entrypoint")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.extract::<::lk_agents_abi::HostRef>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn wait<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, task: ::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let task_arg = task;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_wait", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(task_arg, py)?];
                Ok((host.bind(py).getattr("_core_wait")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn cancel<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, task: ::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let task_arg = task;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_cancel", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(task_arg, py)?];
                Ok((host.bind(py).getattr("_core_cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn shutdown_began<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, reason: ::std::string::String, user_initiated: bool) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let reason_arg = reason;
        let user_initiated_arg = user_initiated;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_shutdown_began", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(reason_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(user_initiated_arg, py)?];
                Ok((host.bind(py).getattr("_core_shutdown_began")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn close_session<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_close_session", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close_session")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn session_end<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, span: ::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let span_arg = span;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_session_end", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(span_arg, py)?];
                Ok((host.bind(py).getattr("_core_session_end")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn upload<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, span: ::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let span_arg = span;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_upload", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(span_arg, py)?];
                Ok((host.bind(py).getattr("_core_upload")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn disconnect<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, span: ::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let span_arg = span;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_disconnect", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(span_arg, py)?];
                Ok((host.bind(py).getattr("_core_disconnect")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn shutdown_callback<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, callback: ::lk_agents_abi::HostRef, reason: ::std::option::Option<::std::string::String>, span: ::std::option::Option<::lk_agents_abi::HostRef>) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let callback_arg = callback;
        let reason_arg = reason;
        let span_arg = span;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobProcess._core_shutdown_callback", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(callback_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(reason_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(span_arg, py)?];
                Ok((host.bind(py).getattr("_core_shutdown_callback")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn span(&self, name: ::std::string::String, callback: ::std::option::Option<::std::string::String>) -> ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let name_arg = name;
        let callback_arg = callback;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(name_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(callback_arg, py)?];
                Ok((host.bind(py).getattr("_core_span")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::HostRef>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn span_end(&self, span: ::lk_agents_abi::HostRef, error: ::std::option::Option<::lk_agents_abi::errors::HostError>, timed_out: bool) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let span_arg = span;
        let error_arg = error;
        let timed_out_arg = timed_out;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(span_arg, py)?, match error_arg { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? }, ::pyo3::IntoPyObjectExt::into_bound_py_any(timed_out_arg, py)?];
                Ok((host.bind(py).getattr("_core_span_end")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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

    fn log(&self, level: i32, message: ::std::string::String, extra: ::serde_json::Map<::std::string::String, ::serde_json::Value>, error: ::std::option::Option<::lk_agents_abi::errors::HostError>) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let level_arg = level;
        let message_arg = message;
        let extra_arg = extra;
        let error_arg = error;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [::pyo3::IntoPyObjectExt::into_bound_py_any(level_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(message_arg, py)?, ::lk_agents_abi::python::extra_to_py(py, &extra_arg)?.into_any(), match error_arg { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? }];
                Ok((host.bind(py).getattr("_core_log")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.

/// [`::lk_agents_abi::worker::ipc_host::JobCall`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct JobCallProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl JobCallProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobCall>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::JobCall>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobCall>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::JobCall>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::JobCall>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(JobCallProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::worker::ipc_host::JobCall for JobCallProxy {
    fn call<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "JobCall.__call__", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.extract::<::lk_agents_abi::HostRef>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.

/// [`::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct ParticipantEntrypointProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl ParticipantEntrypointProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(ParticipantEntrypointProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint for ParticipantEntrypointProxy {
    fn run<'a>(&'a self, context: &::lk_agents_abi::invoke::CallContext, job: ::lk_agents_abi::HostRef, participant: ::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let job_arg = job;
        let participant_arg = participant;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ParticipantEntrypoint.__call__", context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(job_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(participant_arg, py)?];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
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

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.

/// [`::lk_agents_abi::worker::ipc_host::PoolCallbacks`] over a host `livekit.agents.ipc.ProcPool`.
#[derive(Clone)]
pub struct PoolCallbacksProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl PoolCallbacksProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(PoolCallbacksProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
    }
}

impl ::lk_agents_abi::worker::ipc_host::PoolCallbacks for PoolCallbacksProxy {
    fn create(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "PoolCallbacks._core_create", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_create")?, ::pyo3::types::PyTuple::new(py, args)?))
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
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("PoolCallbacks._core_create: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (super::worker::JobExecutorProxy::in_context(o, context)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn emit(&self, event: ::std::string::String, process: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>) {
        let host = ::std::sync::Arc::clone(&self.host);
        let event_arg = event;
        let process_arg = process;
        ::lk_agents_runtime::host::notify(&self.notifier, "PoolCallbacks.emit", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(event_arg, py)?, super::worker::JobExecutorProxy::to_host(py, ::std::sync::Arc::clone(&process_arg))?];
                Ok((host.bind(py).getattr("emit")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/server_host.rs.

/// [`::lk_agents_abi::worker::server_host::ServerHost`] over a host `livekit.agents.worker._ServerHost`.
#[derive(Clone)]
pub struct ServerHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl ServerHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::server_host::ServerHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::server_host::ServerHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::server_host::ServerHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::worker::server_host::ServerHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::server_host::ServerHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(ServerHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::worker::server_host::ServerHost for ServerHostProxy {
    fn launch_job(&self, info: ::lk_agents_abi::worker::worker::RunningJobInfo) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let info_arg = info;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_launch_job", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::worker_server::running_job_info_to_py(py, info_arg)?];
                Ok((host.bind(py).getattr("_core_launch_job")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn terminate_job(&self, job_id: ::std::string::String) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let job_id_arg = job_id;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_terminate_job", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(job_id_arg, py)?];
                Ok((host.bind(py).getattr("_core_terminate_job")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn load(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<f64, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_load", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_load")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (::lk_agents_abi::python::number_from_py::<f64>(&o, "ServerHost._core_load")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn start(&self, options: ::lk_agents_abi::worker::worker::ProcessStartOptions) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let options_arg = options;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_start", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::worker_server::process_start_options_to_py(py, options_arg)?];
                Ok((host.bind(py).getattr("_core_start")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn started(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_started", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_started")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn idle_processes(&self, target: u32) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let target_arg = target;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_idle_processes", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(target_arg, py)?];
                Ok((host.bind(py).getattr("_core_idle_processes")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn worker_load(&self, load: f64) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let load_arg = load;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_worker_load", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(load_arg, py)?];
                Ok((host.bind(py).getattr("_core_worker_load")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn join_jobs(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_join_jobs", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_join_jobs")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn close(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_close", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn active_jobs(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<usize, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_health", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_health")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (::lk_agents_abi::python::number_from_py::<usize>(&o, "ServerHost._core_health")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn metrics(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(::std::string::String, ::std::string::String), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ServerHost._core_metrics", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_metrics")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "ServerHost._core_metrics"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}
