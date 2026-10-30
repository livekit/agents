//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `cli` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/cli.rs.
type CliImpl = <crate::Selected as super::Implementations>::Cli;

#[pyfunction]
#[pyo3(signature = (argv))]
pub fn parse_cli<'py>(py: Python<'py>, argv: ::std::vec::Vec<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let out = py.detach(|| <CliImpl as ::lk_agents_abi::cli::Cli>::parse_cli(argv));
    Ok(crate::generated::records::cli::cli_parse_result_to_py(py, out)?)
}

/// One structured JSON line, including host-formatted exceptions and stack traces.
#[pyfunction]
#[pyo3(signature = (record))]
pub fn format_log_record(py: Python<'_>, record: Bound<'_, PyAny>) -> PyResult<::std::string::String> {
    let record = crate::generated::records::cli::log_record_from_py(&record)?;
    let out = py.detach(|| <CliImpl as ::lk_agents_abi::cli::Cli>::format_log_record(record));
    Ok(out)
}

/// The development launcher connection.
#[pyfunction]
#[pyo3(signature = (cli_addr, agent_name, url, api_key, api_secret, server))]
pub fn dev_channel<'py>(py: Python<'py>, cli_addr: ::std::string::String, agent_name: ::std::string::String, url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, server: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let server = crate::generated::cli::DevServerProxy::from_host(&server)?;
    crate::host::future_into_py(py, async move {
        let out = <CliImpl as ::lk_agents_abi::cli::Cli>::dev_channel(cli_addr, agent_name, url, api_key, api_secret, server).await;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/cli.rs.
type ShutdownImpl = <crate::Selected as super::Implementations>::CliShutdown;

fn _extract_cli_shutdown_new_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "timeout")
}

fn _extract_cli_shutdown_new_drain_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "drain_timeout")
}

/// Native enforcement of the process's one shutdown deadline.
#[pyclass(frozen, name = "Shutdown", module = "livekit.agents._core")]
pub struct Shutdown(pub(crate) ::std::sync::Arc<ShutdownImpl>);

#[pymethods]
impl Shutdown {
    #[new]
    #[pyo3(signature = (timeout, drain_timeout))]
    fn new(py: Python<'_>, #[pyo3(from_py_with = _extract_cli_shutdown_new_timeout)] timeout: f64, #[pyo3(from_py_with = _extract_cli_shutdown_new_drain_timeout)] drain_timeout: f64) -> PyResult<Shutdown> {
        crate::host::check_pid()?;
        let out = py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::new(timeout, drain_timeout)).map_err(|e: ::lk_agents_abi::cli::CliError| <::lk_agents_abi::cli::CliError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(Shutdown(::std::sync::Arc::new(out)))
    }

    /// Own process termination signals. Fails if another launcher is already listening.
    fn listen(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::listen(&self.0)).map_err(|e: ::lk_agents_abi::cli::CliError| <::lk_agents_abi::cli::CliError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// Wait until shutdown starts, whether requested by the host or a native signal.
    fn wait<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::wait(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Begin shutdown once. Repeated requests never extend the deadline.
    fn begin(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::begin(&self.0));
        Ok(out)
    }

    /// Seconds left until the deadline, or the configured budget before shutdown begins.
    fn remaining(&self, py: Python<'_>) -> PyResult<f64> {
        crate::host::check_pid()?;
        let out = py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::remaining(&self.0));
        Ok(out)
    }

    /// Drain time remaining while reserving the configured cleanup budget.
    fn drain_remaining(&self, py: Python<'_>) -> PyResult<f64> {
        crate::host::check_pid()?;
        let out = py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::drain_remaining(&self.0));
        Ok(out)
    }

    /// Disarm only after host cleanup and event-loop closure finish.
    fn closed(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::closed(&self.0));
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/cli.rs.

/// [`::lk_agents_abi::cli::DevServer`] over a host `livekit.agents.cli._DevServer`.
#[derive(Clone)]
pub struct DevServerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl DevServerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::cli::DevServer>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::cli::DevServer>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::cli::DevServer>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::cli::DevServer>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::cli::DevServer>> {
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

impl ::lk_agents_abi::cli::DevServer for DevServerProxy {
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
