//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `worker` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/worker/mod.rs.
type JoinTokensImpl =
    <crate::generated::implementations::Selected as super::Implementations>::JoinTokens;

fn _extract_join_tokens_room_join_token_ttl(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JoinTokensRoomJoinTokenTtlInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "ttl")).map(Some)
    }
}
type JoinTokensRoomJoinTokenTtlInput = ::std::option::Option<f64>;

/// A token joining `room` as `identity`, publishing, subscribing and updating its own
/// metadata; `agent` joins as an agent. `ttl` in seconds; the signer's default when `None`.
#[pyfunction]
#[pyo3(signature = (api_key, api_secret, identity, room, *, agent = false, ttl = None))]
pub fn room_join_token(
    py: Python<'_>,
    api_key: ::std::string::String,
    api_secret: ::std::string::String,
    identity: ::std::string::String,
    room: ::std::string::String,
    agent: bool,
    #[pyo3(from_py_with = _extract_join_tokens_room_join_token_ttl)]
    ttl: JoinTokensRoomJoinTokenTtlInput,
) -> PyResult<::std::string::String> {
    crate::py_calls::check_pid()?;
    let out = py
        .detach(|| {
            <JoinTokensImpl as ::lk_agents_abi::worker::JoinTokens>::room_join_token(
                api_key, api_secret, identity, room, agent, ttl,
            )
        })
        .map_err(|e: ::lk_agents_abi::worker::TokenRefused| {
            <::lk_agents_abi::worker::TokenRefused as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/http.rs.
type ByteStreamImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ByteStream;

fn _extract_byte_stream_read_max_bytes(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ByteStreamReadMaxBytesInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "max_bytes")
}
type ByteStreamReadMaxBytesInput = usize;

fn _extract_byte_stream_write_data(obj: &Bound<'_, PyAny>) -> PyResult<ByteStreamWriteDataInput> {
    if obj.is_instance_of::<::pyo3::types::PyBytes>()
        || obj.is_instance_of::<::pyo3::types::PyByteArray>()
        || obj.is_instance_of::<::pyo3::types::PyMemoryView>()
    {
        crate::py_values::owned_bytes(obj.as_any(), "data")
    } else {
        obj.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::py_values::number_from_py::<u8>(item.as_any(), "data")
                })
                .collect::<PyResult<Vec<_>>>()
        })
    }
}
type ByteStreamWriteDataInput = ::std::vec::Vec<u8>;

fn _extract_byte_stream_reset_code(obj: &Bound<'_, PyAny>) -> PyResult<ByteStreamResetCodeInput> {
    crate::py_values::number_from_py::<i32>(obj.as_any(), "code")
}
type ByteStreamResetCodeInput = i32;

fn _extract_byte_stream_stop_code(obj: &Bound<'_, PyAny>) -> PyResult<ByteStreamStopCodeInput> {
    crate::py_values::number_from_py::<i32>(obj.as_any(), "code")
}
type ByteStreamStopCodeInput = i32;

#[pyclass(frozen, name = "ByteStream", module = "livekit.agents._core")]
pub struct ByteStream(pub(crate) ::std::sync::Arc<ByteStreamImpl>);

#[pymethods]
impl ByteStream {
    #[pyo3(signature = (max_bytes = 65536))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_byte_stream_read_max_bytes)]
        max_bytes: ByteStreamReadMaxBytesInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ByteStreamImpl as ::lk_agents_abi::worker::http::ByteStream>::read(&this, max_bytes)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (data))]
    fn write<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_byte_stream_write_data)] data: ByteStreamWriteDataInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ByteStreamImpl as ::lk_agents_abi::worker::http::ByteStream>::write(&this, data)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Idempotent; a stream already reset raises.
    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ByteStreamImpl as ::lk_agents_abi::worker::http::ByteStream>::finish(&self.0)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (code))]
    fn reset(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_byte_stream_reset_code)] code: ByteStreamResetCodeInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ByteStreamImpl as ::lk_agents_abi::worker::http::ByteStream>::reset(&self.0, code)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (code))]
    fn stop(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_byte_stream_stop_code)] code: ByteStreamStopCodeInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ByteStreamImpl as ::lk_agents_abi::worker::http::ByteStream>::stop(&self.0, code)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::http::ByteTransport`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::http::ByteTransport>(),
        read: |obj| {
            let this = obj.cast::<ByteStream>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::http::ByteTransport> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/job.rs.
type JobTraceImpl =
    <crate::generated::implementations::Selected as super::Implementations>::JobTrace;
type JobTraceJobTimelineOutput =
    (::std::vec::Vec<(::std::string::String, i64)>, ::std::vec::Vec<(::std::string::String, f64)>);

fn _extract_job_trace_job_timeline_received_at(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobTraceJobTimelineReceivedAtInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "received_at")
}
type JobTraceJobTimelineReceivedAtInput = f64;

fn _extract_job_trace_job_timeline_accepted_at(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobTraceJobTimelineAcceptedAtInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "accepted_at")
}
type JobTraceJobTimelineAcceptedAtInput = f64;

fn _extract_job_trace_job_timeline_assigned_at(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobTraceJobTimelineAssignedAtInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "assigned_at")
}
type JobTraceJobTimelineAssignedAtInput = f64;

fn _extract_job_trace_job_timeline_launched_at(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobTraceJobTimelineLaunchedAtInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "launched_at")
}
type JobTraceJobTimelineLaunchedAtInput = f64;

fn _extract_job_trace_job_timeline_entrypoint_started_at(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobTraceJobTimelineEntrypointStartedAtInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "entrypoint_started_at")
}
type JobTraceJobTimelineEntrypointStartedAtInput = f64;

fn _extract_job_trace_job_timeline_server_started_at(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobTraceJobTimelineServerStartedAtInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "server_started_at")
}
type JobTraceJobTimelineServerStartedAtInput = i64;

/// A job's dispatch timeline, for its root span: one event per known stage instant (unix
/// nanoseconds) and the seconds between adjacent stages as attributes (they sum to the dispatch
/// latency). A zero instant is unknown (a simulation, the console, a resumed job) and skipped.
/// `server_started_at` is the server's own start (`JobState.started_at`, unix nanoseconds, or
/// milliseconds or seconds should that change).
#[pyfunction]
#[pyo3(signature = (received_at, accepted_at, assigned_at, launched_at, entrypoint_started_at, server_started_at))]
pub fn job_timeline(
    py: Python<'_>,
    #[pyo3(from_py_with = _extract_job_trace_job_timeline_received_at)]
    received_at: JobTraceJobTimelineReceivedAtInput,
    #[pyo3(from_py_with = _extract_job_trace_job_timeline_accepted_at)]
    accepted_at: JobTraceJobTimelineAcceptedAtInput,
    #[pyo3(from_py_with = _extract_job_trace_job_timeline_assigned_at)]
    assigned_at: JobTraceJobTimelineAssignedAtInput,
    #[pyo3(from_py_with = _extract_job_trace_job_timeline_launched_at)]
    launched_at: JobTraceJobTimelineLaunchedAtInput,
    #[pyo3(from_py_with = _extract_job_trace_job_timeline_entrypoint_started_at)]
    entrypoint_started_at: JobTraceJobTimelineEntrypointStartedAtInput,
    #[pyo3(from_py_with = _extract_job_trace_job_timeline_server_started_at)]
    server_started_at: JobTraceJobTimelineServerStartedAtInput,
) -> PyResult<JobTraceJobTimelineOutput> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| {
        <JobTraceImpl as ::lk_agents_abi::worker::job::JobTrace>::job_timeline(
            received_at,
            accepted_at,
            assigned_at,
            launched_at,
            entrypoint_started_at,
            server_started_at,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
type ServerEnvironmentImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ServerEnvironment;
type ServerEnvironmentEndpointUrlBaseOutput = ::std::option::Option<::std::string::String>;

/// Logical CPUs, fractional under a cgroup quota. Reads, starts nothing.
#[pyfunction]
pub fn cpu_count(py: Python<'_>) -> PyResult<f64> {
    crate::py_calls::check_pid()?;
    let out = py.detach(
        <ServerEnvironmentImpl as ::lk_agents_abi::worker::types::ServerEnvironment>::cpu_count,
    );
    ::pyo3::PyResult::Ok(out)
}

/// `https://<project>/agents/<agent>/<deployment>` (`AgentServer._endpoint_url_base`), when
/// the URL and the agent are known; a worker registered without a deployment is `default`.
#[pyfunction]
#[pyo3(signature = (ws_url, agent_name, deployment))]
pub fn endpoint_url_base(
    py: Python<'_>,
    ws_url: ::std::string::String,
    agent_name: ::std::string::String,
    deployment: ::std::string::String,
) -> PyResult<ServerEnvironmentEndpointUrlBaseOutput> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <ServerEnvironmentImpl as ::lk_agents_abi::worker::types::ServerEnvironment>::endpoint_url_base(ws_url, agent_name, deployment));
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type JobSupervisorImpl =
    <crate::generated::implementations::Selected as super::Implementations>::JobSupervisor;

fn _extract_job_supervisor_new_initialize_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorNewInitializeTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "initialize_timeout")
}
type JobSupervisorNewInitializeTimeoutInput = f64;

fn _extract_job_supervisor_new_close_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorNewCloseTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "close_timeout")
}
type JobSupervisorNewCloseTimeoutInput = f64;

fn _extract_job_supervisor_new_memory_warn_mb(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorNewMemoryWarnMbInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "memory_warn_mb")
}
type JobSupervisorNewMemoryWarnMbInput = f64;

fn _extract_job_supervisor_new_memory_limit_mb(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorNewMemoryLimitMbInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "memory_limit_mb")
}
type JobSupervisorNewMemoryLimitMbInput = f64;

fn _extract_job_supervisor_new_ping_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorNewPingIntervalInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "ping_interval")
}
type JobSupervisorNewPingIntervalInput = f64;

fn _extract_job_supervisor_new_ping_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorNewPingTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "ping_timeout")
}
type JobSupervisorNewPingTimeoutInput = f64;

fn _extract_job_supervisor_new_high_ping_threshold(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorNewHighPingThresholdInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "high_ping_threshold")
}
type JobSupervisorNewHighPingThresholdInput = f64;

fn _extract_job_supervisor_core_launched_process_pid(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorCoreLaunchedProcessPidInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "pid")
}
type JobSupervisorCoreLaunchedProcessPidInput = u32;

fn _extract_job_supervisor_core_launched_process_sentinel(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorCoreLaunchedProcessSentinelInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "sentinel")
}
type JobSupervisorCoreLaunchedProcessSentinelInput = i64;

fn _extract_job_supervisor_core_launched_process_owned(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobSupervisorCoreLaunchedProcessOwnedInput> {
    obj.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::py_values::number_from_py::<i64>(item.as_any(), "owned")
            })
            .collect::<PyResult<Vec<_>>>()
    })
}
type JobSupervisorCoreLaunchedProcessOwnedInput = ::std::vec::Vec<i64>;

#[pyclass(frozen, name = "JobSupervisor", module = "livekit.agents._core")]
pub struct JobSupervisor(pub(crate) ::std::sync::Arc<JobSupervisorImpl>);

#[pymethods]
impl JobSupervisor {
    /// `thread`: the runner is a thread of this process, else a process of the host's start
    /// method.
    #[new]
    #[pyo3(signature = (host, *, thread, initialize_timeout, close_timeout, memory_warn_mb, memory_limit_mb, ping_interval, ping_timeout, high_ping_threshold))]
    fn new(
        py: Python<'_>,
        host: Bound<'_, PyAny>,
        thread: bool,
        #[pyo3(from_py_with = _extract_job_supervisor_new_initialize_timeout)]
        initialize_timeout: JobSupervisorNewInitializeTimeoutInput,
        #[pyo3(from_py_with = _extract_job_supervisor_new_close_timeout)]
        close_timeout: JobSupervisorNewCloseTimeoutInput,
        #[pyo3(from_py_with = _extract_job_supervisor_new_memory_warn_mb)]
        memory_warn_mb: JobSupervisorNewMemoryWarnMbInput,
        #[pyo3(from_py_with = _extract_job_supervisor_new_memory_limit_mb)]
        memory_limit_mb: JobSupervisorNewMemoryLimitMbInput,
        #[pyo3(from_py_with = _extract_job_supervisor_new_ping_interval)]
        ping_interval: JobSupervisorNewPingIntervalInput,
        #[pyo3(from_py_with = _extract_job_supervisor_new_ping_timeout)]
        ping_timeout: JobSupervisorNewPingTimeoutInput,
        #[pyo3(from_py_with = _extract_job_supervisor_new_high_ping_threshold)] high_ping_threshold: JobSupervisorNewHighPingThresholdInput,
    ) -> PyResult<JobSupervisor> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::worker::JobExecutorProxy::from_host(host.as_any())?;
        let context = crate::py_calls::entered();
        let out = py.detach(|| <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::new(&env, host, thread, initialize_timeout, close_timeout, memory_warn_mb, memory_limit_mb, ping_interval, ping_timeout, high_ping_threshold, context)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(JobSupervisor(::std::sync::Arc::new(out)))
    }

    /// The seconds the process took to be ready.
    #[pyo3(signature = (*, asyncio_debug, http_proxy))]
    fn initialize<'py>(
        &self,
        py: Python<'py>,
        asyncio_debug: bool,
        http_proxy: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::initialize(&this, asyncio_debug, http_proxy)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[getter]
    fn initialized(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::initialized(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The launch's receipt, reported on the launch worker right after the OS created the child.
    #[pyo3(signature = (pid, sentinel, method, owned))]
    fn _core_launched_process(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_job_supervisor_core_launched_process_pid)]
        pid: JobSupervisorCoreLaunchedProcessPidInput,
        #[pyo3(from_py_with = _extract_job_supervisor_core_launched_process_sentinel)]
        sentinel: JobSupervisorCoreLaunchedProcessSentinelInput,
        method: ::std::string::String,
        #[pyo3(from_py_with = _extract_job_supervisor_core_launched_process_owned)]
        owned: JobSupervisorCoreLaunchedProcessOwnedInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::_core_launched_process(&self.0, pid, sentinel, method, owned)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// A job thread returned, reported from that thread.
    fn _core_thread_exited(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::_core_thread_exited(&self.0)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// Launch, supervise and wait for the runner's end; whether one ran (a process reaped, a
    /// thread returned). Cancelling the wait leaves the runner owned.
    fn run<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out =
                <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::run(&this)
                    .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn _core_wait_exit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::_core_wait_exit(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn _core_join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::_core_join(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::aclose(
                    &this,
                )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn kill<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::kill(
                &this,
            )
            .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[pyo3(signature = (fields))]
    fn start_job<'py>(
        &self,
        py: Python<'py>,
        fields: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let fields =
            crate::generated::records::worker_ipc::start_job_fields_from_py(fields.as_any())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::start_job(&this, fields)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[getter]
    fn killed(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::killed(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::ipc_host::SupervisedJob`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::ipc_host::SupervisedJob>(),
        read: |obj| {
            let this = obj.cast::<JobSupervisor>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::SupervisedJob> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type IpcClientImpl =
    <crate::generated::implementations::Selected as super::Implementations>::IpcClient;

fn _extract_ipc_client_new_fd(obj: &Bound<'_, PyAny>) -> PyResult<IpcClientNewFdInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "fd")
}
type IpcClientNewFdInput = i64;

#[pyclass(frozen, name = "IpcClient", module = "livekit.agents._core")]
pub struct IpcClient(pub(crate) ::std::sync::Arc<IpcClientImpl>);

#[pymethods]
impl IpcClient {
    /// The handshake on the parent's socket, blocking before the process's loop exists.
    #[new]
    #[pyo3(signature = (fd, initialize, *, process = false))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_ipc_client_new_fd)] fd: IpcClientNewFdInput,
        initialize: Bound<'_, PyAny>,
        process: bool,
    ) -> PyResult<IpcClient> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let initialize =
            crate::generated::worker::ChildInitializeProxy::from_host(initialize.as_any())?;
        let out = py.detach(|| <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::new(&env, fd, initialize, process)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(IpcClient(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn asyncio_debug(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::asyncio_debug(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// Serve the parent's messages until it goes, its shutdown ends the process, or `stop()`.
    #[pyo3(signature = (host))]
    fn serve<'py>(&self, py: Python<'py>, host: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let host = crate::generated::worker::JobProcessProxy::from_host(host.as_any())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::serve(&this, host)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::stop(&self.0)
        });
        Ok(())
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <IpcClientImpl as ::lk_agents_abi::worker::ipc_host::IpcClient>::aclose(&this)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type ProcPoolImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ProcessPool;

fn _extract_process_pool_new_num_idle_processes(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProcessPoolNewNumIdleProcessesInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "num_idle_processes")
}
type ProcessPoolNewNumIdleProcessesInput = usize;

fn _extract_process_pool_new_initialize_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProcessPoolNewInitializeTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "initialize_timeout")
}
type ProcessPoolNewInitializeTimeoutInput = f64;

fn _extract_process_pool_new_close_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProcessPoolNewCloseTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "close_timeout")
}
type ProcessPoolNewCloseTimeoutInput = f64;

fn _extract_process_pool_set_target_idle_processes_n(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProcessPoolSetTargetIdleProcessesNInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "n")
}
type ProcessPoolSetTargetIdleProcessesNInput = usize;

#[pyclass(frozen, name = "ProcPool", module = "livekit.agents._core")]
pub struct ProcPool(pub(crate) ::std::sync::Arc<ProcPoolImpl>);

#[pymethods]
impl ProcPool {
    #[new]
    #[pyo3(signature = (host, *, num_idle_processes, initialize_timeout, close_timeout))]
    fn new(
        py: Python<'_>,
        host: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_process_pool_new_num_idle_processes)]
        num_idle_processes: ProcessPoolNewNumIdleProcessesInput,
        #[pyo3(from_py_with = _extract_process_pool_new_initialize_timeout)]
        initialize_timeout: ProcessPoolNewInitializeTimeoutInput,
        #[pyo3(from_py_with = _extract_process_pool_new_close_timeout)]
        close_timeout: ProcessPoolNewCloseTimeoutInput,
    ) -> PyResult<ProcPool> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::worker::PoolCallbacksProxy::from_host(host.as_any())?;
        let context = crate::py_calls::entered();
        let out = py.detach(|| <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::new(&env, host, num_idle_processes, initialize_timeout, close_timeout, context)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(ProcPool(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn processes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::processes(&self.0)
        });
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::worker::JobExecutorProxy::to_host(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    #[getter]
    fn jobs_waiting(&self, py: Python<'_>) -> PyResult<usize> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::jobs_waiting(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn target_idle_processes(&self, py: Python<'_>) -> PyResult<usize> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::target_idle_processes(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[setter(target_idle_processes)]
    fn set_target_idle_processes(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_process_pool_set_target_idle_processes_n)]
        n: ProcessPoolSetTargetIdleProcessesNInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::set_target_idle_processes(&self.0, n));
        Ok(())
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[pyo3(signature = (job_id, info))]
    fn launch_job<'py>(
        &self,
        py: Python<'py>,
        job_id: ::std::string::String,
        info: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let info = crate::py_objects::object_ref_from_py(info.as_any())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::launch_job(&this, job_id, info)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Waits until every process a job was launched on exited (the drain).
    fn join_jobs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::join_jobs(&this)
                    .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::aclose(&this)
                    .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type JobRunImpl = <crate::generated::implementations::Selected as super::Implementations>::JobRun;

fn _extract_job_run_new_session_end_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobRunNewSessionEndTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "session_end_timeout")
}
type JobRunNewSessionEndTimeoutInput = f64;

fn _extract_job_run_new_session_aclose_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobRunNewSessionAcloseTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "session_aclose_timeout")
}
type JobRunNewSessionAcloseTimeoutInput = f64;
type JobRunObservabilityUrlOutput = ::std::option::Option<::std::string::String>;

fn _extract_job_run_add_participant_entrypoint_kinds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobRunAddParticipantEntrypointKindsInput> {
    obj.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::py_values::number_from_py::<i32>(item.as_any(), "kinds")
            })
            .collect::<PyResult<Vec<_>>>()
    })
}
type JobRunAddParticipantEntrypointKindsInput = ::std::vec::Vec<i32>;

fn _extract_job_run_participant_joined_kind(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobRunParticipantJoinedKindInput> {
    crate::py_values::number_from_py::<i32>(obj.as_any(), "kind")
}
type JobRunParticipantJoinedKindInput = i32;
type JobRunInitRecordingOutput = ::std::option::Option<bool>;

/// A job's run: what ends the job (the first reason wins), the entrypoint's supervision, the
/// order and bounds of the shutdown, the participant entrypoints, the room connected once, the
/// application calls in flight and the recording's setup.
#[pyclass(frozen, name = "JobRun", module = "livekit.agents._core")]
pub struct JobRun(pub(crate) ::std::sync::Arc<JobRunImpl>);

#[pymethods]
impl JobRun {
    /// `session_end` names the door's `on_session_end`, when it has one. `observability_url`
    /// overrides where the job's telemetry and recording go (`LIVEKIT_OBSERVABILITY_URL`).
    #[new]
    #[pyo3(signature = (*, fake_job, session_end_timeout, redaction, url = "".to_owned(), recording = false, session_end = None, session_aclose_timeout = 60.0, observability_url = None))]
    fn new(
        py: Python<'_>,
        fake_job: bool,
        #[pyo3(from_py_with = _extract_job_run_new_session_end_timeout)]
        session_end_timeout: JobRunNewSessionEndTimeoutInput,
        redaction: bool,
        url: ::std::string::String,
        recording: bool,
        session_end: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_job_run_new_session_aclose_timeout)] session_aclose_timeout: JobRunNewSessionAcloseTimeoutInput,
        observability_url: ::std::option::Option<::std::string::String>,
    ) -> PyResult<JobRun> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::new(&env, fake_job, session_end_timeout, redaction, url, recording, session_end, session_aclose_timeout, observability_url)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(JobRun(::std::sync::Arc::new(out)))
    }

    /// `callback(reason)` when it takes the reason, else `callback()`, as the job shuts down; a
    /// framework callback runs in the shutdown's span, any other in its own span named `name`.
    #[pyo3(signature = (callback, takes_reason, framework, name))]
    fn add_shutdown_callback(
        &self,
        py: Python<'_>,
        callback: Bound<'_, PyAny>,
        takes_reason: bool,
        framework: bool,
        name: ::std::string::String,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let callback = crate::py_objects::object_ref_from_py(callback.as_any())?;
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::add_shutdown_callback(
                &self.0,
                callback,
                takes_reason,
                framework,
                name,
            )
        });
        Ok(())
    }

    #[getter]
    fn observability_url(&self, py: Python<'_>) -> PyResult<JobRunObservabilityUrlOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::observability_url(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn buffer_logs(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::buffer_logs(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (reason, user_initiated))]
    fn shutdown(
        &self,
        py: Python<'_>,
        reason: ::std::string::String,
        user_initiated: bool,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::shutdown(
                &self.0,
                reason,
                user_initiated,
            )
        });
        Ok(())
    }

    fn connected(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::connected(&self.0));
        Ok(())
    }

    /// Awaits `connect()` once: a call while it runs waits for it, a call after it connected
    /// returns at once, a failed one leaves the next call to try again.
    #[pyo3(signature = (connect))]
    fn connect<'py>(
        &self,
        py: Python<'py>,
        connect: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let connect = crate::generated::worker::JobCallProxy::from_host(connect.as_any())?;
        let context = crate::py_calls::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::connect(&this, connect, context)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `entrypoint` (named `name`) runs for each joining participant of one of `kinds`.
    #[pyo3(signature = (entrypoint, kinds, name))]
    fn add_participant_entrypoint(
        &self,
        py: Python<'_>,
        entrypoint: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_job_run_add_participant_entrypoint_kinds)]
        kinds: JobRunAddParticipantEntrypointKindsInput,
        name: ::std::string::String,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let entrypoint =
            crate::generated::worker::ParticipantEntrypointProxy::from_host(entrypoint.as_any())?;
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::add_participant_entrypoint(
                &self.0, entrypoint, kinds, name,
            )
        });
        Ok(())
    }

    /// A participant joined, or was in the room when it connected; `job` is the `JobContext`.
    #[pyo3(signature = (job, participant, identity, kind))]
    fn participant_joined(
        &self,
        py: Python<'_>,
        job: Bound<'_, PyAny>,
        participant: Bound<'_, PyAny>,
        identity: ::std::string::String,
        #[pyo3(from_py_with = _extract_job_run_participant_joined_kind)]
        kind: JobRunParticipantJoinedKindInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let job = crate::py_objects::object_ref_from_py(job.as_any())?;
        let participant = crate::py_objects::object_ref_from_py(participant.as_any())?;
        let context = crate::py_calls::entered();
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::participant_joined(
                &self.0,
                job,
                participant,
                identity,
                kind,
                context,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (attributes))]
    fn participant_left(
        &self,
        py: Python<'_>,
        attributes: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::participant_left(
                &self.0, attributes,
            )
        });
        Ok(())
    }

    /// An application API call (`delete_room`, ...): its failure is logged whether or not the
    /// caller awaits it, and it is cancelled once the job ends. A console job makes no call: it
    /// resolves to `console_result`.
    #[pyo3(signature = (call, name, console_result))]
    fn api_call<'py>(
        &self,
        py: Python<'py>,
        call: Bound<'_, PyAny>,
        name: ::std::string::String,
        console_result: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = crate::generated::worker::JobCallProxy::from_host(call.as_any())?;
        let console_result = crate::py_objects::object_ref_from_py(console_result.as_any())?;
        let context = crate::py_calls::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::api_call(&this, call, name, console_result, context)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::py_objects::object_ref_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// Cancels the API calls still running and waits for them.
    fn close_tasks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::close_tasks(&this).await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// The redaction to record with when the cloud upload is to be set up now, `None` when not.
    #[pyo3(signature = (options))]
    fn init_recording(
        &self,
        py: Python<'_>,
        options: ::std::collections::BTreeMap<::std::string::String, bool>,
    ) -> PyResult<JobRunInitRecordingOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::init_recording(&self.0, options)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn recording_initialized(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::recording_initialized(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn redaction(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::redaction(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The entrypoint, then the shutdown sequence once the job ends; `host` runs each step in the
    /// context this call captured.
    #[pyo3(signature = (host))]
    fn run<'py>(&self, py: Python<'py>, host: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let host = crate::generated::worker::JobProcessProxy::from_host(host.as_any())?;
        let context = crate::py_calls::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::run(&this, host, context)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::ipc_host::JobExecution`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::ipc_host::JobExecution>(),
        read: |obj| {
            let this = obj.cast::<JobRun>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecution> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type ProcessBootstrapImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ProcessBootstrap;
type ProcessBootstrapIpcPairOutput = (i64, i64);

#[pyfunction]
pub fn ipc_pair(py: Python<'_>) -> PyResult<ProcessBootstrapIpcPairOutput> {
    crate::py_calls::check_pid()?;
    let out = py
        .detach(
            <ProcessBootstrapImpl as ::lk_agents_abi::worker::ipc_host::ProcessBootstrap>::ipc_pair,
        )
        .map_err(|e: ::lk_agents_abi::errors::HostError| {
            <::lk_agents_abi::errors::HostError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type ProcessInitializationImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ProcessInitialization;

#[pyfunction]
#[pyo3(signature = (role))]
pub fn init(py: Python<'_>, role: Bound<'_, PyAny>) -> PyResult<()> {
    let role = crate::generated::native_enums::role_from_py(role.as_any())?;
    py.detach(|| <ProcessInitializationImpl as ::lk_agents_abi::worker::ipc_host::ProcessInitialization>::init(role));
    Ok(())
}

// Declared in rust/crates/lk-agents-abi/src/worker/server_host.rs.
type ServerConfigImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ServerConfig;

#[pyclass(frozen, name = "ServerConfig", module = "livekit.agents._core")]
pub struct ServerConfig(pub(crate) ::std::sync::Arc<ServerConfigImpl>);

#[pymethods]
impl ServerConfig {
    /// `options` left empty read the environment (`http_proxy`: `HTTPS_PROXY`, else
    /// `HTTP_PROXY`).
    #[new]
    #[pyo3(signature = (options))]
    fn new(py: Python<'_>, options: Bound<'_, PyAny>) -> PyResult<ServerConfig> {
        crate::py_calls::check_pid()?;
        let options =
            crate::generated::records::worker_options::server_options_from_py(options.as_any())?;
        let out = py.detach(|| <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::new(options)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(ServerConfig(::std::sync::Arc::new(out)))
    }

    /// Before the server starts only.
    #[pyo3(signature = (options))]
    fn update(&self, py: Python<'_>, options: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let options =
            crate::generated::records::worker_options::server_options_from_py(options.as_any())?;
        py.detach(|| <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::update(&self.0, options)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn ws_url(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::ws_url(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn api_key(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::api_key(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn api_secret(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::api_secret(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn agent_name(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::agent_name(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn deployment(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::deployment(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// Hosted on Cloud (`LIVEKIT_WORKER_TOKEN` set).
    #[getter]
    fn cloud(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::cloud(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn drain_timeout(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::drain_timeout(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn shutdown_timeout(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::shutdown_timeout(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn simulation(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::simulation(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[setter(simulation)]
    fn set_simulation(&self, py: Python<'_>, value: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::set_simulation(&self.0, value));
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::server_host::ServerConfiguration`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::server_host::ServerConfiguration>(),
        read: |obj| {
            let this = obj.cast::<ServerConfig>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::server_host::ServerConfiguration> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/server_host.rs.
type ServerRuntimeImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ServerRuntime;

fn _extract_server_runtime_drain_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ServerRuntimeDrainTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "timeout")).map(Some)
    }
}
type ServerRuntimeDrainTimeoutInput = ::std::option::Option<f64>;

fn _extract_server_runtime_simulate_job_room_info(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ServerRuntimeSimulateJobRoomInfoInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (if obj.is_instance_of::<::pyo3::types::PyBytes>()
            || obj.is_instance_of::<::pyo3::types::PyByteArray>()
            || obj.is_instance_of::<::pyo3::types::PyMemoryView>()
        {
            crate::py_values::owned_bytes(obj.as_any(), "room_info")
        } else {
            obj.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::py_values::number_from_py::<u8>(item.as_any(), "room_info")
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
        })
        .map(Some)
    }
}
type ServerRuntimeSimulateJobRoomInfoInput = ::std::option::Option<::std::vec::Vec<u8>>;

#[pyclass(frozen, name = "ServerRuntime", module = "livekit.agents._core")]
pub struct ServerRuntime(pub(crate) ::std::sync::Arc<ServerRuntimeImpl>);

#[pymethods]
impl ServerRuntime {
    #[new]
    #[pyo3(signature = (config, host))]
    fn new(
        py: Python<'_>,
        config: Bound<'_, PyAny>,
        host: Bound<'_, PyAny>,
    ) -> PyResult<ServerRuntime> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let config = crate::py_objects::native::<
            dyn ::lk_agents_abi::worker::server_host::ServerConfiguration,
        >(config.as_any())
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "config",
                ": expected a native ServerConfig"
            ))
        })?;
        let host = crate::generated::worker::ServerHostProxy::from_host(host.as_any())?;
        let out = py.detach(|| <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::new(&env, config, host)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(ServerRuntime(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn worker_id(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::worker_id(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn draining(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::draining(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn http_port(&self, py: Python<'_>) -> PyResult<u16> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::http_port(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// Start the job processes and the health listener and, unless `unregistered`, register;
    /// return once closed. `endpoints` receives each stream for `routes`.
    #[pyo3(signature = (*, devmode, unregistered, has_rtc, has_text_doors, routes, has_load_fnc, endpoints))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        routes: Bound<'_, PyAny>,
        has_load_fnc: bool,
        endpoints: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let routes = routes.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::worker_server::endpoint_route_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let endpoints =
            crate::generated::worker::EndpointHandlerProxy::from_host(endpoints.as_any())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::run(&this, devmode, unregistered, has_rtc, has_text_doors, routes, has_load_fnc, endpoints)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `None` (or `0`) waits without a deadline; past it raises `TimeoutError`.
    #[pyo3(signature = (timeout = None))]
    fn drain<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_server_runtime_drain_timeout)]
        timeout: ServerRuntimeDrainTimeoutInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::drain(&this, timeout)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `"running"`, `"failed"` or `"success"`.
    #[pyo3(signature = (job_id, status))]
    fn job_status(
        &self,
        py: Python<'_>,
        job_id: ::std::string::String,
        status: ::std::string::String,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::job_status(
                &self.0, job_id, status,
            )
        });
        Ok(())
    }

    /// A job in `room` with no dispatch: the console's, a notebook's.
    #[pyo3(signature = (room, *, fake_job, agent_identity = None, room_info = None, token = None))]
    fn simulate_job<'py>(
        &self,
        py: Python<'py>,
        room: ::std::string::String,
        fake_job: bool,
        agent_identity: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_server_runtime_simulate_job_room_info)]
        room_info: ServerRuntimeSimulateJobRoomInfoInput,
        token: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::simulate_job(&this, room, fake_job, agent_identity, room_info, token)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/http.rs.

/// [`::lk_agents_abi::worker::http::EndpointHandler`] over a host `livekit.agents.http._AsgiServer`.
#[derive(Clone)]
pub struct EndpointHandlerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl EndpointHandlerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::http::EndpointHandler>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::http::EndpointHandler>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::http::EndpointHandler>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::http::EndpointHandler>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::http::EndpointHandler>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(EndpointHandlerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::worker::http::EndpointHandler for EndpointHandlerProxy {
    fn handle(
        &self,
        stream: ::std::sync::Arc<dyn ::lk_agents_abi::worker::http::ByteTransport>,
        request: ::lk_agents_abi::worker::http::EndpointRequest,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let stream_arg = stream;
        let request_arg = request;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "EndpointHandler._core_handle",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [<dyn ::lk_agents_abi::worker::http::ByteTransport as super::NativeToHost>::to_host(py, stream_arg)?, crate::generated::records::worker_endpoint::endpoint_request_to_py(py, request_arg)?];
                Ok((host.bind(py).getattr("_core_handle")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn drain(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "EndpointHandler._core_drain",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_drain")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::job::SessionJob>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::job::SessionJob>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::job::SessionJob>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::job::SessionJob>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::job::SessionJob>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(SessionJobProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::worker::job::SessionJob for SessionJobProxy {
    fn connect(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionJob._core_connect",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_connect")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/remote_session.rs.

/// [`::lk_agents_abi::worker::remote_session::SessionPort`] over a host `livekit.agents.voice.remote_session.RemoteSessionAccess`.
#[derive(Clone)]
pub struct SessionPortProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    framework_info: (::std::string::String, ::std::string::String),
}

impl SessionPortProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::remote_session::SessionPort>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::worker::remote_session::SessionPort,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::remote_session::SessionPort>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::worker::remote_session::SessionPort,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::remote_session::SessionPort>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let framework_info = {
            let v = crate::py_calls::read(obj, "_core_framework_info")?;
            (|| -> ::pyo3::PyResult<_> {
                let tuple = v.cast::<::pyo3::types::PyTuple>()?;
                if tuple.len() != 2 {
                    return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                        "{}: expected a tuple of 2 items",
                        "SessionPort._core_framework_info"
                    )));
                }
                let item_0 = tuple.get_item(0)?;
                let item_1 = tuple.get_item(1)?;
                Ok((
                    (item_0.extract::<::std::string::String>())?,
                    (item_1.extract::<::std::string::String>())?,
                ))
            })()?
        };
        Ok(SessionPortProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            framework_info,
        })
    }
}

impl ::lk_agents_abi::worker::remote_session::SessionPort for SessionPortProxy {
    fn chat_history(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::ChatItem>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort._core_chat_history",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_chat_history")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn agent_info(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            (
                ::std::string::String,
                ::std::option::Option<::std::string::String>,
                ::std::vec::Vec<::std::string::String>,
                ::std::vec::Vec<::lk_agents_abi::ChatItem>,
            ),
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort._core_agent_info",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_agent_info")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 4 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 4 items", "SessionPort._core_agent_info"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; Ok(((item_0.extract::<::std::string::String>())?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?, (item_2.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()))?, (item_3.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>()))?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn run_input<'a>(
        &'a self,
        text: &str,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            (
                ::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>,
                ::std::option::Option<::std::string::String>,
            ),
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort._core_run_input",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_run_input")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_run_input"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((if item_0.is_none() { Ok(None) } else { (item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn session_state(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            (
                ::std::string::String,
                ::std::string::String,
                ::std::string::String,
                ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
                ::std::option::Option<f64>,
            ),
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort._core_session_state",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_session_state")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "SessionPort._core_session_state"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<::std::string::String>())?, (item_3.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| dict.iter().map(|(key, item)| Ok((key.extract::<::std::string::String>()?, item.extract::<::std::string::String>()?))).collect::<PyResult<::std::collections::BTreeMap<_, _>>>()))?, (if item_4.is_none() { Ok(None) } else { (crate::py_values::number_from_py::<f64>(item_4.as_any(), "SessionPort._core_session_state")).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn rtc_stats(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<(
                ::std::vec::Vec<::serde_json::Value>,
                ::std::vec::Vec<::serde_json::Value>,
            )>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort._core_rtc_stats",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_rtc_stats")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (if o.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_rtc_stats"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::py_values::value_from_py(item.as_any(), "SessionPort._core_rtc_stats") }).collect::<PyResult<Vec<_>>>()))?, (item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::py_values::value_from_py(item.as_any(), "SessionPort._core_rtc_stats") }).collect::<PyResult<Vec<_>>>()))?,)) })()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn session_usage(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<::lk_agents_abi::usage::Usage, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort._core_session_usage",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_session_usage")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::generated::records::usage::usage_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn framework_info(&self) -> (::std::string::String, ::std::string::String) {
        ::std::clone::Clone::clone(&self.framework_info)
    }

    fn update_io(
        &self,
        update: (
            ::std::option::Option<bool>,
            ::std::option::Option<bool>,
            ::std::option::Option<bool>,
            ::std::option::Option<bool>,
            ::std::option::Option<bool>,
        ),
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let update_arg = update;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort._core_update_io",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(update_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_update_io")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn finalize_simulation(
        &self,
        provisional: (bool, ::std::string::String),
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            (
                ::std::option::Option<(bool, ::std::string::String)>,
                ::std::option::Option<::std::string::String>,
            ),
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let provisional_arg = provisional;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort._core_finalize_simulation",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(provisional_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_finalize_simulation")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((if item_0.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = item_0.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<bool>())?, (item_1.extract::<::std::string::String>())?,)) })()).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let source = ::std::sync::Arc::new(obj.clone().unbind());
        let _ = context;
        Ok(JobExecutorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), source })
    }
}

impl JobExecutorProxy {
    /// The host object behind a proxy of `::lk_agents_abi::worker::ipc_host::JobExecutor`; a native implementation has none.
    pub fn to_host<'py>(
        py: Python<'py>,
        value: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let any: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = value;
        match any.downcast::<Self>() {
            Ok(proxy) => Ok(proxy.source.bind(py).clone()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `JobExecutor` has no host object to hand back",
            )),
        }
    }
}

impl ::lk_agents_abi::worker::ipc_host::JobExecutor for JobExecutorProxy {
    fn launch<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        endpoint: i64,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let endpoint_arg = endpoint;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobExecutor._core_launch",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(endpoint_arg, py)?];
                Ok((host.bind(py).getattr("_core_launch")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn completed<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        completion: ::lk_agents_abi::worker::ipc::JobCompletion,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let completion_arg = completion;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobExecutor._core_completed",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::worker_ipc::job_completion_to_py(
                        py,
                        completion_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_completed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn log(
        &self,
        level: i32,
        message: ::std::string::String,
        extra: ::serde_json::Map<::std::string::String, ::serde_json::Value>,
        error: ::std::option::Option<::lk_agents_abi::errors::HostError>,
    ) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let level_arg = level;
        let message_arg = message;
        let extra_arg = extra;
        let error_arg = error;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(level_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(message_arg, py)?,
                    crate::py_values::extra_to_py(py, &extra_arg)?.into_any(),
                    match error_arg {
                        None => py.None().into_bound(py),
                        Some(x) => crate::py_errors::ExceptionValue::to_exception(x, py)?,
                    },
                ];
                Ok((host.bind(py).getattr("_core_log")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn start<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::SupervisedJob>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobExecutor._core_start",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_start")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::SupervisedJob>(o.as_any()).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("JobExecutor._core_start", ": expected a native JobSupervisor")))).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn initialize<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobExecutor.initialize",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("initialize")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn launch_job<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        info: ::lk_agents_abi::HostRef,
        launched_at: f64,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let info_arg = info;
        let launched_at_arg = launched_at;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobExecutor._core_launch_job",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::py_objects::object_ref_to_py(py, info_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(launched_at_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_launch_job")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn close<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobExecutor.aclose",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn join<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<bool, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobExecutor._core_join",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_join")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::ChildInitialize>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(ChildInitializeProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::worker::ipc_host::ChildInitialize for ChildInitializeProxy {
    fn initialize(
        &self,
        http_proxy: ::std::option::Option<::std::string::String>,
    ) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let http_proxy_arg = http_proxy;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(http_proxy_arg, py)?];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::JobProcess>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(JobProcessProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::worker::ipc_host::JobProcess for JobProcessProxy {
    fn ping(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_ping",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_ping")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn start_job(
        &self,
        fields: ::lk_agents_abi::worker::job::StartJobFields,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecution>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let fields_arg = fields;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_start_job",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::worker_ipc::start_job_fields_to_py(
                        py, fields_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_start_job")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobExecution>(o.as_any()).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("JobProcess._core_start_job", ": expected a native JobRun")))).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn start_entrypoint<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_start_entrypoint",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_start_entrypoint")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::py_objects::object_ref_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn wait<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        task: ::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let task_arg = task;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_wait",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, task_arg)?];
                Ok((host.bind(py).getattr("_core_wait")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn cancel<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        task: ::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let task_arg = task;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_cancel",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, task_arg)?];
                Ok((host.bind(py).getattr("_core_cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn shutdown_began<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        reason: ::std::string::String,
        user_initiated: bool,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let reason_arg = reason;
        let user_initiated_arg = user_initiated;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_shutdown_began",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(reason_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(user_initiated_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_shutdown_began")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn close_session<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_close_session",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_close_session")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn session_end<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        span: ::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let span_arg = span;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_session_end",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, span_arg)?];
                Ok((
                    host.bind(py).getattr("_core_session_end")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn upload<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        span: ::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let span_arg = span;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_upload",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, span_arg)?];
                Ok((host.bind(py).getattr("_core_upload")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn disconnect<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        span: ::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let span_arg = span;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_disconnect",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, span_arg)?];
                Ok((
                    host.bind(py).getattr("_core_disconnect")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn shutdown_callback<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        callback: ::lk_agents_abi::HostRef,
        reason: ::std::option::Option<::std::string::String>,
        span: ::std::option::Option<::lk_agents_abi::HostRef>,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let callback_arg = callback;
        let reason_arg = reason;
        let span_arg = span;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_shutdown_callback",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    crate::py_objects::object_ref_to_py(py, callback_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(reason_arg, py)?,
                    match span_arg {
                        None => py.None().into_bound(py),
                        Some(x) => crate::py_objects::object_ref_to_py(py, x)?,
                    },
                ];
                Ok((
                    host.bind(py).getattr("_core_shutdown_callback")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn span(
        &self,
        name: ::std::string::String,
        callback: ::std::option::Option<::std::string::String>,
    ) -> ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let name_arg = name;
        let callback_arg = callback;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(name_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(callback_arg, py)?,
                ];
                Ok((host.bind(py).getattr("_core_span")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::py_objects::object_ref_from_py(o.as_any())).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn span_end(
        &self,
        span: ::lk_agents_abi::HostRef,
        error: ::std::option::Option<::lk_agents_abi::errors::HostError>,
        timed_out: bool,
    ) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let span_arg = span;
        let error_arg = error;
        let timed_out_arg = timed_out;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    crate::py_objects::object_ref_to_py(py, span_arg)?,
                    match error_arg {
                        None => py.None().into_bound(py),
                        Some(x) => crate::py_errors::ExceptionValue::to_exception(x, py)?,
                    },
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(timed_out_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_span_end")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn log(
        &self,
        level: i32,
        message: ::std::string::String,
        extra: ::serde_json::Map<::std::string::String, ::serde_json::Value>,
        error: ::std::option::Option<::lk_agents_abi::errors::HostError>,
    ) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let level_arg = level;
        let message_arg = message;
        let extra_arg = extra;
        let error_arg = error;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(level_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(message_arg, py)?,
                    crate::py_values::extra_to_py(py, &extra_arg)?.into_any(),
                    match error_arg {
                        None => py.None().into_bound(py),
                        Some(x) => crate::py_errors::ExceptionValue::to_exception(x, py)?,
                    },
                ];
                Ok((host.bind(py).getattr("_core_log")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobCall>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobCall>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobCall>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobCall>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::JobCall>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(JobCallProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::worker::ipc_host::JobCall for JobCallProxy {
    fn call<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobCall.__call__",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::py_objects::object_ref_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint>>
    {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(ParticipantEntrypointProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::worker::ipc_host::ParticipantEntrypoint for ParticipantEntrypointProxy {
    fn run<'a>(
        &'a self,
        context: &::lk_agents_abi::invoke::CallContext,
        job: ::lk_agents_abi::HostRef,
        participant: ::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let job_arg = job;
        let participant_arg = participant;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ParticipantEntrypoint.__call__",
            context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::py_objects::object_ref_to_py(py, job_arg)?,
                    crate::py_objects::object_ref_to_py(py, participant_arg)?,
                ];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
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
    notifier: crate::py_calls::Notifier,
}

impl PoolCallbacksProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::PoolCallbacks>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(PoolCallbacksProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::worker::ipc_host::PoolCallbacks for PoolCallbacksProxy {
    fn create(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "PoolCallbacks._core_create",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_create")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("PoolCallbacks._core_create: the call captured no host context")));
                };
                let o = out.bind(py);
                let out = (super::worker::JobExecutorProxy::in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn emit(
        &self,
        event: ::std::string::String,
        process: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobExecutor>,
    ) {
        let host = ::std::sync::Arc::clone(&self.host);
        let event_arg = event;
        let process_arg = process;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "PoolCallbacks.emit",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(event_arg, py)?,
                    super::worker::JobExecutorProxy::to_host(
                        py,
                        ::std::sync::Arc::clone(&process_arg),
                    )?,
                ];
                Ok((host.bind(py).getattr("emit")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/server_host.rs.

/// [`::lk_agents_abi::worker::server_host::ServerHost`] over a host `livekit.agents.worker._ServerCallbacks`.
#[derive(Clone)]
pub struct ServerHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl ServerHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::server_host::ServerHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::server_host::ServerHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::server_host::ServerHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::server_host::ServerHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::server_host::ServerHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(ServerHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::worker::server_host::ServerHost for ServerHostProxy {
    fn launch_job(
        &self,
        info: ::lk_agents_abi::worker::types::RunningJobInfo,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let info_arg = info;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_launch_job",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::worker_server::running_job_info_to_py(
                        py, info_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_launch_job")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn terminate_job(
        &self,
        job_id: ::std::string::String,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let job_id_arg = job_id;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_terminate_job",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(job_id_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_terminate_job")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn load(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<f64, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_load",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_load")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::py_values::number_from_py::<f64>(o.as_any(), "ServerHost._core_load")).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn start(
        &self,
        options: ::lk_agents_abi::worker::types::ProcessStartOptions,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let options_arg = options;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_start",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::worker_server::process_start_options_to_py(
                        py,
                        options_arg,
                    )?];
                Ok((host.bind(py).getattr("_core_start")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn started(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_started",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_started")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn idle_processes(
        &self,
        target: u32,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let target_arg = target;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_idle_processes",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(target_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_idle_processes")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn worker_load(
        &self,
        load: f64,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let load_arg = load;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_worker_load",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(load_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_worker_load")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn join_jobs(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_join_jobs",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_join_jobs")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn close(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_close",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn active_jobs(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<usize, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_health",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_health")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::py_values::number_from_py::<usize>(o.as_any(), "ServerHost._core_health")).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn metrics(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            (::std::string::String, ::std::string::String),
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_metrics",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_metrics")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "ServerHost._core_metrics"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::worker::TokenRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::worker::ipc_host::WorkerError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Invalid(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::TimedOut(..) => ::pyo3::exceptions::PyTimeoutError::new_err(
                ::std::string::ToString::to_string(self),
            ),
            Self::Closed => ::pyo3::exceptions::PyConnectionError::new_err(
                ::std::string::ToString::to_string(self),
            ),
            Self::Cancelled => crate::py_errors::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Host(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}
