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

// Declared in rust/crates/lk-agents-abi/src/worker/http/mod.rs.
type HttpExchangeImpl =
    <crate::generated::implementations::Selected as super::Implementations>::HttpExchange;

/// One request and its response, or one WebSocket connection. The worker owns parsing, framing,
/// limits, deadlines and closing; the binding reads events and sends messages.
#[pyclass(frozen, name = "HttpExchange", module = "livekit.agents._core")]
pub struct HttpExchange(pub(crate) ::std::sync::Arc<HttpExchangeImpl>);

#[pymethods]
impl HttpExchange {
    #[getter]
    fn head<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <HttpExchangeImpl as ::lk_agents_abi::worker::http::HttpExchange>::head(&self.0)
        });
        crate::generated::records::worker_endpoint::request_head_to_py(py, out)
    }

    /// The next request event. After the request body ends it waits for the response to
    /// complete, then reports `Disconnected`.
    fn receive<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <HttpExchangeImpl as ::lk_agents_abi::worker::http::HttpExchange>::receive(&this, )
                .await.map_err(|e: ::lk_agents_abi::worker::http::ExchangeError| <::lk_agents_abi::worker::http::ExchangeError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::worker_endpoint::exchange_event_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (message))]
    fn send<'py>(&self, py: Python<'py>, message: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let message =
            crate::generated::records::worker_endpoint::exchange_message_from_py(message.as_any())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <HttpExchangeImpl as ::lk_agents_abi::worker::http::HttpExchange>::send(&this, message)
                .await.map_err(|e: ::lk_agents_abi::worker::http::ExchangeError| <::lk_agents_abi::worker::http::ExchangeError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::http::EndpointExchange`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::http::EndpointExchange>(),
        read: |obj| {
            let this = obj.cast::<HttpExchange>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::http::EndpointExchange> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
type ServerEnvironmentImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ServerEnvironment;

/// Logical CPUs, fractional under a cgroup quota. Reads, starts nothing.
#[pyfunction]
pub fn cpu_count(py: Python<'_>) -> PyResult<f64> {
    crate::py_calls::check_pid()?;
    let out = py.detach(
        <ServerEnvironmentImpl as ::lk_agents_abi::worker::types::ServerEnvironment>::cpu_count,
    );
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type JobLaunchImpl =
    <crate::generated::implementations::Selected as super::Implementations>::JobLaunch;

fn _extract_job_launch_process_started_pid(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobLaunchProcessStartedPidInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "pid")
}
type JobLaunchProcessStartedPidInput = u32;

fn _extract_job_launch_process_started_sentinel(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobLaunchProcessStartedSentinelInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "sentinel")
}
type JobLaunchProcessStartedSentinelInput = i64;

fn _extract_job_launch_process_started_owned(
    obj: &Bound<'_, PyAny>,
) -> PyResult<JobLaunchProcessStartedOwnedInput> {
    obj.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::py_values::number_from_py::<i64>(item.as_any(), "owned")
            })
            .collect::<PyResult<Vec<_>>>()
    })
}
type JobLaunchProcessStartedOwnedInput = ::std::vec::Vec<i64>;

/// One launch of a job runner, as the launcher reports it.
#[pyclass(frozen, name = "JobLaunch", module = "livekit.agents._core")]
pub struct JobLaunch(pub(crate) ::std::sync::Arc<JobLaunchImpl>);

#[pymethods]
impl JobLaunch {
    /// The runner is a thread of this process, else a process of the launcher's start method.
    #[getter]
    fn thread(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobLaunchImpl as ::lk_agents_abi::worker::ipc_host::JobLaunch>::thread(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The process's receipt, reported on the launch worker right after the OS created it.
    #[pyo3(signature = (pid, sentinel, method, owned))]
    fn process_started(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_job_launch_process_started_pid)]
        pid: JobLaunchProcessStartedPidInput,
        #[pyo3(from_py_with = _extract_job_launch_process_started_sentinel)]
        sentinel: JobLaunchProcessStartedSentinelInput,
        method: ::std::string::String,
        #[pyo3(from_py_with = _extract_job_launch_process_started_owned)]
        owned: JobLaunchProcessStartedOwnedInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <JobLaunchImpl as ::lk_agents_abi::worker::ipc_host::JobLaunch>::process_started(&self.0, pid, sentinel, method, owned)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// The job thread returned, reported from that thread.
    fn thread_exited(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <JobLaunchImpl as ::lk_agents_abi::worker::ipc_host::JobLaunch>::thread_exited(&self.0)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::worker::ipc_host::LaunchReport`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::worker::ipc_host::LaunchReport>(),
        read: |obj| {
            let this = obj.cast::<JobLaunch>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::LaunchReport> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.
type JobSupervisorImpl =
    <crate::generated::implementations::Selected as super::Implementations>::JobSupervisor;
type JobSupervisorPidOutput = ::std::option::Option<u32>;
type JobSupervisorExitcodeOutput = ::std::option::Option<i32>;
type JobSupervisorInitializeTimeOutput = ::std::option::Option<f64>;

/// One job runner, a process or (`options.thread`) a thread of this process, and its one job:
/// the worker launches it through `launcher`, initializes it, hands it the job, pings it,
/// samples its memory, closes it and waits for its end, whether or not the host still awaits.
///
/// `status` is `None` until a job was launched, then `RUNNING` until the runner ended, then
/// `SUCCESS` (exit code 0 and a clean control channel) or `FAILED`. `running_job` is set once
/// the runner took its job and is kept after the runner ended. `aclose()` asks an initialized
/// runner to shut down and stops it when late (`close_timeout`); `kill()` stops it at once. A
/// thread cannot be stopped: both wait for it to return, and it has no pong deadline nor memory
/// limit.
#[pyclass(frozen, name = "JobSupervisor", module = "livekit.agents._core")]
pub struct JobSupervisor(pub(crate) ::std::sync::Arc<JobSupervisorImpl>);

#[pymethods]
impl JobSupervisor {
    #[new]
    #[pyo3(signature = (launcher, options))]
    fn new(
        py: Python<'_>,
        launcher: Bound<'_, PyAny>,
        options: Bound<'_, PyAny>,
    ) -> PyResult<JobSupervisor> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let launcher = crate::generated::worker::JobLauncherProxy::from_host(launcher.as_any())?;
        let options =
            crate::generated::records::worker_ipc::job_runner_options_from_py(options.as_any())?;
        let context = crate::py_calls::entered();
        let out = py.detach(|| <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::new(&env, launcher, options, context)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(JobSupervisor(::std::sync::Arc::new(out)))
    }

    /// `PCEXEC_...` for a process, `THEXEC_...` for a thread.
    #[getter]
    fn id(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::id(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn thread(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::thread(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// `start()` was called.
    #[getter]
    fn started(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::started(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The runner answered its initialization.
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

    /// `aclose()` or `kill()` was called: it starts no runner and takes no job.
    #[getter]
    fn closing(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::closing(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The runner was stopped by force (a process killed).
    #[getter]
    fn killed(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::killed(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The process's id, once the OS created it; `None` for a thread, and for a forkserver child
    /// whose id the forkserver has not reported.
    #[getter]
    fn pid(&self, py: Python<'_>) -> PyResult<JobSupervisorPidOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::pid(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The runner's exit code once its supervision is over (negative: the signal that ended a
    /// Unix process).
    #[getter]
    fn exitcode(&self, py: Python<'_>) -> PyResult<JobSupervisorExitcodeOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::exitcode(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::status(&self.0)
        });
        out.map(|value| crate::generated::records::worker_ipc::job_status_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn running_job<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::running_job(
                &self.0,
            )
        });
        out.map(|value| crate::generated::records::worker_server::running_job_info_to_py(py, value))
            .transpose()
    }

    /// The seconds the runner took to be ready, once it was.
    #[getter]
    fn initialize_time(&self, py: Python<'_>) -> PyResult<JobSupervisorInitializeTimeOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::initialize_time(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// Launch the runner; returns once it exists. Raises once closing, when already started, or
    /// with the launcher's own error when nothing was launched. A launch failing after the OS
    /// created the process is logged with its cause and the process is stopped.
    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Send the runner its `Hello` and wait for it to be ready: the seconds it took. A late runner
    /// is stopped (`TimeoutError`).
    fn initialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::initialize(&this, )
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Hand the ready runner its one job, stamped `launched_at` now; the job continues the
    /// caller's current span. Raises when it is not ready or already has a job.
    #[pyo3(signature = (info))]
    fn launch_job<'py>(
        &self,
        py: Python<'py>,
        info: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let info =
            crate::generated::records::worker_server::running_job_info_from_py(info.as_any())?;
        let context = crate::py_calls::entered();
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::launch_job(&this, info, context, cx)
                .await.map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Close it gracefully (see the class) and wait for its cleanup; at once when not started.
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

    /// Stop it at once and wait for its cleanup; at once when not started.
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

    /// Wait for the end of its supervision, its completion reported to the launcher: whether a
    /// runner ran. `False` at once when it was never started.
    fn join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out =
                <JobSupervisorImpl as ::lk_agents_abi::worker::ipc_host::JobSupervisor>::join(
                    &this,
                )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
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

fn _extract_process_pool_set_target_idle_processes_n(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProcessPoolSetTargetIdleProcessesNInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "n")
}
type ProcessPoolSetTargetIdleProcessesNInput = usize;

/// The warm job processes: the pool keeps `num_idle_processes` ready, hands each job the next one
/// and retries a failed spawn or launch. Each process is a `JobSupervisor` with `options`.
#[pyclass(frozen, name = "ProcPool", module = "livekit.agents._core")]
pub struct ProcPool(pub(crate) ::std::sync::Arc<ProcPoolImpl>);

#[pymethods]
impl ProcPool {
    #[new]
    #[pyo3(signature = (host, *, num_idle_processes, options))]
    fn new(
        py: Python<'_>,
        host: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_process_pool_new_num_idle_processes)]
        num_idle_processes: ProcessPoolNewNumIdleProcessesInput,
        options: Bound<'_, PyAny>,
    ) -> PyResult<ProcPool> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::worker::PoolCallbacksProxy::from_host(host.as_any())?;
        let options =
            crate::generated::records::worker_ipc::job_runner_options_from_py(options.as_any())?;
        let context = crate::py_calls::entered();
        let out = py.detach(|| <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::new(&env, host, num_idle_processes, options, context)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(ProcPool(::std::sync::Arc::new(out)))
    }

    /// The processes the pool owns, from their creation until their cleanup.
    #[getter]
    fn processes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::processes(&self.0)
        });
        ::pyo3::PyResult::Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| <dyn ::lk_agents_abi::worker::ipc_host::SupervisedJob as crate::generated::NativeToHost>::to_host(py, x)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    /// The jobs whose process runs them now (`status` `RUNNING`).
    #[getter]
    fn active_jobs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::active_jobs(&self.0)
        });
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| {
                        crate::generated::records::worker_server::running_job_info_to_py(py, x)
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// The pool's process that took the job `job_id`.
    #[pyo3(signature = (job_id))]
    fn get_by_job_id<'py>(
        &self,
        py: Python<'py>,
        job_id: ::std::string::String,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::get_by_job_id(
                &self.0, job_id,
            )
        });
        out.map(|value| <dyn ::lk_agents_abi::worker::ipc_host::SupervisedJob as crate::generated::NativeToHost>::to_host(py, value)).transpose()
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

    /// The job continues the caller's current span.
    #[pyo3(signature = (info))]
    fn launch_job<'py>(
        &self,
        py: Python<'py>,
        info: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let info =
            crate::generated::records::worker_server::running_job_info_from_py(info.as_any())?;
        let context = crate::py_calls::entered();
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ProcPoolImpl as ::lk_agents_abi::worker::ipc_host::ProcessPool>::launch_job(&this, info, context, cx)
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

/// A job's run: what ends the job (the first reason wins), the entrypoint's supervision, the
/// order and bounds of the shutdown, the participant entrypoints, the room connected once, the
/// application calls in flight and the recording's setup.
#[pyclass(frozen, name = "JobRun", module = "livekit.agents._core")]
pub struct JobRun(pub(crate) ::std::sync::Arc<JobRunImpl>);

#[pymethods]
impl JobRun {
    /// `session_end` names the door's `on_session_end`, when it has one. `observability_url`
    /// overrides where the job's telemetry and recording go (`LIVEKIT_OBSERVABILITY_URL`);
    /// `api_key` and `api_secret` sign its Cloud telemetry.
    #[new]
    #[pyo3(signature = (*, fake_job, session_end_timeout, redaction, url = "".to_owned(), session_end = None, session_aclose_timeout = 60.0, observability_url = None, api_key = "".to_owned(), api_secret = "".to_owned()))]
    fn new(
        py: Python<'_>,
        fake_job: bool,
        #[pyo3(from_py_with = _extract_job_run_new_session_end_timeout)]
        session_end_timeout: JobRunNewSessionEndTimeoutInput,
        redaction: bool,
        url: ::std::string::String,
        session_end: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_job_run_new_session_aclose_timeout)] session_aclose_timeout: JobRunNewSessionAcloseTimeoutInput,
        observability_url: ::std::option::Option<::std::string::String>,
        api_key: ::std::string::String,
        api_secret: ::std::string::String,
    ) -> PyResult<JobRun> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::new(&env, fake_job, session_end_timeout, redaction, url, session_end, session_aclose_timeout, observability_url, api_key, api_secret)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
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

    /// Opens the job's root span for the job the parent handed over, before its run starts.
    #[pyo3(signature = (fields))]
    fn start_trace(&self, py: Python<'_>, fields: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let fields =
            crate::generated::records::worker_ipc::start_job_fields_from_py(fields.as_any())?;
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::start_trace(&self.0, fields)
        });
        Ok(())
    }

    /// The application ends the job.
    #[pyo3(signature = (reason))]
    fn shutdown(&self, py: Python<'_>, reason: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::shutdown(&self.0, reason)
        });
        Ok(())
    }

    fn room_disconnected(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::room_disconnected(&self.0)
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
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::connect(&this, connect, context, cx)
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
        let cx = crate::py_calls::entry_cx();
        py.detach(|| {
            <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::participant_joined(
                &self.0,
                job,
                participant,
                identity,
                kind,
                context,
                cx,
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
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::api_call(&this, call, name, console_result, context, cx)
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

    /// The session's recording options: the job registers with the exporter's Cloud destination
    /// once, at the first call.
    #[pyo3(signature = (options))]
    fn init_recording(
        &self,
        py: Python<'_>,
        options: ::std::collections::BTreeMap<::std::string::String, bool>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <JobRunImpl as ::lk_agents_abi::worker::ipc_host::JobRun>::init_recording(&self.0, options)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
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
type ServerConfigEndpointUrlOutput = ::std::option::Option<::std::string::String>;

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

    /// The address `endpoint` publishes in its card: its Cloud URL when Cloud routes it; its
    /// path when this worker serves it, put on the address the request reached.
    #[pyo3(signature = (endpoint))]
    fn endpoint_url(
        &self,
        py: Python<'_>,
        endpoint: ::std::string::String,
    ) -> PyResult<ServerConfigEndpointUrlOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::endpoint_url(
                &self.0, endpoint,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// Where this worker delegates to its own `endpoint`: Cloud, in the worker's deployment, or
    /// the worker's public listener. Raises before the server knows which.
    #[pyo3(signature = (endpoint))]
    fn delegation_url(
        &self,
        py: Python<'_>,
        endpoint: ::std::string::String,
    ) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <ServerConfigImpl as ::lk_agents_abi::worker::server_host::ServerConfig>::delegation_url(&self.0, endpoint)).map_err(|e: ::lk_agents_abi::worker::ipc_host::WorkerError| <::lk_agents_abi::worker::ipc_host::WorkerError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
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
    /// return once closed. The worker serves each `text` route with its door from `text_doors`,
    /// whose handlers run in the context this call captured; `endpoints` receives every other
    /// stream for `routes`.
    #[pyo3(signature = (*, devmode, unregistered, has_rtc, has_text_doors, routes, has_load_fnc, endpoints, text_doors))]
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
        text_doors: Bound<'_, PyAny>,
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
        let text_doors = text_doors.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::py_objects::native::<dyn ::lk_agents_abi::a2a::door::TextEndpoint>(
                        item.as_any(),
                    )
                    .ok_or_else(|| {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "text_doors",
                            ": expected a native A2A door"
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let context = crate::py_calls::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ServerRuntimeImpl as ::lk_agents_abi::worker::server_host::ServerRuntime>::run(&this, devmode, unregistered, has_rtc, has_text_doors, routes, has_load_fnc, endpoints, text_doors, context)
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

// Declared in rust/crates/lk-agents-abi/src/worker/http/mod.rs.

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
        cx: &::lk_agents_abi::telemetry::Cx,
        exchange: ::std::sync::Arc<dyn ::lk_agents_abi::worker::http::EndpointExchange>,
        request: ::lk_agents_abi::worker::http::EndpointRequest,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let exchange_arg = exchange;
        let request_arg = request;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "EndpointHandler._core_handle",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [<dyn ::lk_agents_abi::worker::http::EndpointExchange as super::NativeToHost>::to_host(py, exchange_arg)?, crate::generated::records::worker_endpoint::endpoint_request_to_py(py, request_arg)?];
                Ok((
                    host.bind(py).getattr("_core_handle")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
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
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionJob._core_connect",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_connect")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn enable_recording(&self) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("job")?,
                    ::pyo3::types::PyString::new(py, "enable_recording").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            let o = out.value.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn claim_primary(
        &self,
        session: ::lk_agents_abi::HostRef,
    ) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let session_arg = session;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, session_arg)?];
                Ok((
                    host.bind(py).getattr("_core_claim_primary")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            let o = out.value.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn recording_dir(
        &self,
    ) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_recording_dir")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            let o = out.value.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn init_recording(
        &self,
        options: ::std::collections::BTreeMap<::std::string::String, bool>,
    ) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let options_arg = options;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(options_arg, py)?];
                Ok((
                    host.bind(py).getattr("init_recording")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            let o = out.value.bind(py);
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

    fn on_shutdown(
        &self,
        session: ::lk_agents_abi::HostRef,
    ) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let session_arg = session;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, session_arg)?];
                Ok((
                    host.bind(py).getattr("_core_on_shutdown")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            let o = out.value.bind(py);
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

// Declared in rust/crates/lk-agents-abi/src/worker/remote_session.rs.

/// [`::lk_agents_abi::worker::remote_session::SessionPort`] over a host `livekit.agents.voice._session_bindings.RemoteSessionAccess`.
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
        cx: &::lk_agents_abi::telemetry::Cx,
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
            "SessionPort.session.history.items",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("session")?.getattr("history")?,
                    ::pyo3::types::PyString::new(py, "items").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn agent_info(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_agent_info")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 4 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 4 items", "SessionPort._core_agent_info"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; Ok(((item_0.extract::<::std::string::String>())?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?, (item_2.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()))?, (item_3.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>()))?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn run_input<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_run_input")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_run_input"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((if item_0.is_none() { Ok(None) } else { (item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn session_state(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_session_state")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "SessionPort._core_session_state"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<::std::string::String>())?, (item_3.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| dict.iter().map(|(key, item)| Ok((key.extract::<::std::string::String>()?, item.extract::<::std::string::String>()?))).collect::<PyResult<::std::collections::BTreeMap<_, _>>>()))?, (if item_4.is_none() { Ok(None) } else { (crate::py_values::number_from_py::<f64>(item_4.as_any(), "SessionPort._core_session_state")).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn rtc_stats(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_rtc_stats")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = (if o.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_rtc_stats"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::py_values::value_from_py(item.as_any(), "SessionPort._core_rtc_stats") }).collect::<PyResult<Vec<_>>>()))?, (item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::py_values::value_from_py(item.as_any(), "SessionPort._core_rtc_stats") }).collect::<PyResult<Vec<_>>>()))?,)) })()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn session_usage(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<::lk_agents_abi::usage::Usage, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionPort.session.usage",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("session")?,
                    ::pyo3::types::PyString::new(py, "usage").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
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
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(update_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_update_io")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn finalize_simulation(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(provisional_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_finalize_simulation")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((if item_0.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = item_0.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<bool>())?, (item_1.extract::<::std::string::String>())?,)) })()).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.extract::<::std::string::String>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/ipc_host.rs.

/// [`::lk_agents_abi::worker::ipc_host::JobLauncher`] over a host `livekit.agents.ipc.JobLauncher`.
#[derive(Clone)]
pub struct JobLauncherProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl JobLauncherProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobLauncher>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobLauncher>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobLauncher>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobLauncher>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::worker::ipc_host::JobLauncher>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(JobLauncherProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::worker::ipc_host::JobLauncher for JobLauncherProxy {
    fn launch<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
        endpoint: i64,
        launch: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::LaunchReport>,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let endpoint_arg = endpoint;
        let launch_arg = launch;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobLauncher._core_launch",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(endpoint_arg, py)?, <dyn ::lk_agents_abi::worker::ipc_host::LaunchReport as super::NativeToHost>::to_host(py, launch_arg)?];
                Ok((
                    host.bind(py).getattr("_core_launch")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn completed<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
        completion: ::lk_agents_abi::worker::ipc::JobCompletion,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let completion_arg = completion;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobLauncher._core_completed",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::worker_ipc::job_completion_to_py(
                        py,
                        completion_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_completed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
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
                Ok((
                    host.bind(py).getattr("_core_log")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            let o = out.value.bind(py);
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
                Ok((
                    host.bind(py).getattr("__call__")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            let o = out.value.bind(py);
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
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_ping",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_ping")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn start_job(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::worker_ipc::start_job_fields_to_py(
                        py, fields_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_start_job")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = (crate::py_objects::native::<dyn ::lk_agents_abi::worker::ipc_host::JobExecution>(o.as_any()).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("JobProcess._core_start_job", ": expected a native JobRun")))).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn start_entrypoint<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_start_entrypoint")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = (crate::py_objects::object_ref_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn wait<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, task_arg)?];
                Ok((
                    host.bind(py).getattr("_core_wait")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn cancel<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, task_arg)?];
                Ok((
                    host.bind(py).getattr("_core_cancel")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn close_session<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_close_session",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_close_session")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn session_end<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_session_end",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_session_end")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn session_report<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::telemetry::report::SessionReport>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_session_report",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_session_report")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::telemetry::session_report_from_py(o.as_any())).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn disconnect<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_disconnect",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_disconnect")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn shutdown_callback<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
        callback: ::lk_agents_abi::HostRef,
        reason: ::std::option::Option<::std::string::String>,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let callback_arg = callback;
        let reason_arg = reason;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "JobProcess._core_shutdown_callback",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::py_objects::object_ref_to_py(py, callback_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(reason_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_shutdown_callback")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
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
                Ok((
                    host.bind(py).getattr("_core_log")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            let o = out.value.bind(py);
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
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("__call__")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
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
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::py_objects::object_ref_to_py(py, job_arg)?,
                    crate::py_objects::object_ref_to_py(py, participant_arg)?,
                ];
                Ok((
                    host.bind(py).getattr("__call__")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
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
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::JobLauncher>,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_create")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("PoolCallbacks._core_create: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (super::worker::JobLauncherProxy::in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn emit(
        &self,
        event: ::std::string::String,
        process: ::std::sync::Arc<dyn ::lk_agents_abi::worker::ipc_host::SupervisedJob>,
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
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(event_arg, py)?, <dyn ::lk_agents_abi::worker::ipc_host::SupervisedJob as super::NativeToHost>::to_host(py, process_arg)?];
                Ok((host.bind(py).getattr("emit")?, ::pyo3::types::PyTuple::new(py, args)?, None))
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
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::worker_server::running_job_info_to_py(
                        py, info_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_launch_job")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn terminate_job(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(job_id_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_terminate_job")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn load(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_load")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = (crate::py_values::number_from_py::<f64>(o.as_any(), "ServerHost._core_load")).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn start(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::worker_server::process_start_options_to_py(
                        py,
                        options_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_start")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn started(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_started",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_started")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn idle_processes(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(target_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_idle_processes")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn worker_load(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(load_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_worker_load")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn join_jobs(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_join_jobs",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_join_jobs")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn close(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ServerHost._core_close",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_close")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn active_jobs(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_health")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
                let out = (crate::py_values::number_from_py::<usize>(o.as_any(), "ServerHost._core_health")).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn metrics(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_metrics")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                let o = out.value.bind(py);
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

impl crate::py_errors::HostException for ::lk_agents_abi::worker::http::ExchangeError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Closed(..) => ::pyo3::exceptions::PyConnectionError::new_err(
                ::std::string::ToString::to_string(self),
            ),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
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
