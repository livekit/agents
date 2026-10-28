//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `worker` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.
type WorkerCliImpl = <crate::Selected as super::Implementations>::WorkerCli;

#[pyfunction]
#[pyo3(signature = (argv))]
pub fn parse_cli(argv: ::std::vec::Vec<::std::string::String>) -> PyResult<::lk_agents_abi::worker::cli::CliArgs> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <WorkerCliImpl as ::lk_agents_abi::worker::cli::WorkerCli>::parse_cli(argv).map_err(|e: ::lk_agents_abi::worker::cli::CliError| <::lk_agents_abi::worker::cli::CliError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
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
        Ok(SessionJobProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::worker::job::SessionJob>(|obj| SessionJobProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::worker::job::SessionJob for SessionJobProxy {
    async fn connect(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
        Ok(SessionPortProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, framework_info })
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
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 4 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 4 items", "SessionPort._core_agent_info"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::option::Option<::std::string::String>>())?, (item_2.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()))?, (item_3.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ChatItem>().map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>()))?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn run_input(&self, text: &str) -> ::std::result::Result<(::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>, ::std::option::Option<::std::string::String>), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_run_input"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>>().map_err(::pyo3::PyErr::from))?, (item_1.extract::<::std::option::Option<::std::string::String>>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn session_state(&self) -> ::std::result::Result<(::std::string::String, ::std::string::String, ::std::string::String, ::std::collections::BTreeMap<::std::string::String, ::std::string::String>, ::std::option::Option<f64>), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "SessionPort._core_session_state"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((item_0.extract::<::std::string::String>())?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<::std::string::String>())?, (item_3.extract::<::std::collections::BTreeMap<::std::string::String, ::std::string::String>>())?, (item_4.extract::<::std::option::Option<f64>>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn rtc_stats(&self) -> ::std::result::Result<::std::option::Option<(::std::vec::Vec<::serde_json::Value>, ::std::vec::Vec<::serde_json::Value>)>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "SessionPort._core_finalize_simulation"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<::std::option::Option<(bool, ::std::string::String)>>())?, (item_1.extract::<::std::option::Option<::std::string::String>>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
