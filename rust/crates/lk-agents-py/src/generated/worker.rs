//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `worker` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

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

// Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.
type ExitSignalsImpl = <crate::Selected as super::Implementations>::CliSignals;

/// The process launcher's signal and final event-loop cleanup owner.
#[pyclass(frozen, name = "ExitSignals", module = "livekit.agents._core")]
pub struct ExitSignals(::std::sync::Arc<ExitSignalsImpl>);

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
        let context = ::std::clone::Clone::clone(&self.context);
        ::std::boxed::Box::pin(async move {
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_connect")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
            let v = obj.call_method0("_core_framework_info")?;
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
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_chat_history")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_agent_info")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_run_input")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_session_state")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_rtc_stats")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_session_usage")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(update_arg, py)?];
                Ok((host.bind(py).getattr("_core_update_io")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(provisional_arg, py)?];
                Ok((host.bind(py).getattr("_core_finalize_simulation")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        ::pyo3::Python::attach(|py| {
            // One place converts the call's failures to its declared error.
            #[allow(clippy::redundant_closure_call)]
            let out = (|| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                self.host.bind(py).call_method1("_core_running_jobs", ::pyo3::types::PyTuple::new(py, args)?)
            })();
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = &out;
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; (|| -> ::pyo3::PyResult<_> { let tuple = item.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 8 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 8 items", "DevServer._core_running_jobs"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; let item_5 = tuple.get_item(5)?; let item_6 = tuple.get_item(6)?; let item_7 = tuple.get_item(7)?; Ok(((item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "DevServer._core_running_jobs") }).collect::<PyResult<Vec<_>>>()))?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<::std::string::String>())?, (item_3.extract::<::std::string::String>())?, (item_4.extract::<::std::string::String>())?, (item_5.extract::<::std::string::String>())?, (item_6.extract::<::std::string::String>())?, (item_7.extract::<bool>())?,)) })() }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn reload_jobs(&self, jobs: ::std::vec::Vec<(::std::vec::Vec<u8>, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, ::std::string::String, bool)>) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let jobs_arg = jobs;
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(jobs_arg, py)?];
                Ok((host.bind(py).getattr("_core_reload_jobs")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
