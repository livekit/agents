//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `cli` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/cli.rs.
type CliImpl = <crate::generated::implementations::Selected as super::Implementations>::Cli;

#[pyfunction]
#[pyo3(signature = (argv))]
pub fn parse_cli<'py>(
    py: Python<'py>,
    argv: ::std::vec::Vec<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <CliImpl as ::lk_agents_abi::cli::Cli>::parse_cli(argv));
    crate::generated::records::cli::cli_parse_result_to_py(py, out)
}

/// [`LogLevel::host_level`].
#[pyfunction]
#[pyo3(signature = (level))]
pub fn host_log_level(py: Python<'_>, level: Bound<'_, PyAny>) -> PyResult<u32> {
    crate::py_calls::check_pid()?;
    let level = crate::generated::records::cli::log_level_from_py(level.as_any())?;
    let out = py.detach(|| <CliImpl as ::lk_agents_abi::cli::Cli>::host_log_level(level));
    ::pyo3::PyResult::Ok(out)
}

/// One structured JSON line, including host-formatted exceptions and stack traces.
#[pyfunction]
#[pyo3(signature = (record))]
pub fn format_log_record(
    py: Python<'_>,
    record: Bound<'_, PyAny>,
) -> PyResult<::std::string::String> {
    let record = crate::generated::records::cli::log_record_from_py(record.as_any())?;
    let out = py.detach(|| <CliImpl as ::lk_agents_abi::cli::Cli>::format_log_record(record));
    ::pyo3::PyResult::Ok(out)
}

/// The development launcher connection.
#[pyfunction]
#[pyo3(signature = (cli_addr, agent_name, url, api_key, api_secret, server))]
pub fn dev_channel<'py>(
    py: Python<'py>,
    cli_addr: ::std::string::String,
    agent_name: ::std::string::String,
    url: ::std::string::String,
    api_key: ::std::string::String,
    api_secret: ::std::string::String,
    server: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let server = crate::py_objects::native::<dyn ::lk_agents_abi::worker::server_host::ServerJobs>(
        server.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "server",
            ": expected a native ServerRuntime"
        ))
    })?;
    crate::py_calls::future_into_py(py, async move {
        let () = <CliImpl as ::lk_agents_abi::cli::Cli>::dev_channel(
            cli_addr, agent_name, url, api_key, api_secret, server,
        )
        .await;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/cli.rs.
type ShutdownImpl =
    <crate::generated::implementations::Selected as super::Implementations>::CliShutdown;

fn _extract_cli_shutdown_new_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<CliShutdownNewTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "timeout")
}
type CliShutdownNewTimeoutInput = f64;

fn _extract_cli_shutdown_new_drain_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<CliShutdownNewDrainTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "drain_timeout")
}
type CliShutdownNewDrainTimeoutInput = f64;

/// Native enforcement of the process's one shutdown deadline.
#[pyclass(frozen, name = "Shutdown", module = "livekit.agents._core")]
pub struct Shutdown(pub(crate) ::std::sync::Arc<ShutdownImpl>);

#[pymethods]
impl Shutdown {
    #[new]
    #[pyo3(signature = (timeout, drain_timeout))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_cli_shutdown_new_timeout)]
        timeout: CliShutdownNewTimeoutInput,
        #[pyo3(from_py_with = _extract_cli_shutdown_new_drain_timeout)]
        drain_timeout: CliShutdownNewDrainTimeoutInput,
    ) -> PyResult<Shutdown> {
        crate::py_calls::check_pid()?;
        let out = py
            .detach(|| {
                <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::new(timeout, drain_timeout)
            })
            .map_err(|e: ::lk_agents_abi::cli::CliError| {
                <::lk_agents_abi::cli::CliError as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        ::pyo3::PyResult::Ok(Shutdown(::std::sync::Arc::new(out)))
    }

    /// Own process termination signals. Fails if another launcher is already listening.
    fn listen(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::listen(&self.0))
            .map_err(|e: ::lk_agents_abi::cli::CliError| {
                <::lk_agents_abi::cli::CliError as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        Ok(())
    }

    /// Wait until shutdown starts, whether requested by the host or a native signal.
    fn wait<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::wait(&this).await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Begin shutdown once. Repeated requests never extend the deadline.
    fn begin(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::begin(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// Seconds left until the deadline, or the configured budget before shutdown begins.
    fn remaining(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out =
            py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::remaining(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// Drain time remaining while reserving the configured cleanup budget.
    fn drain_remaining(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::drain_remaining(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// Disarm only after host cleanup and event-loop closure finish.
    fn closed(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ShutdownImpl as ::lk_agents_abi::cli::CliShutdown>::closed(&self.0));
        Ok(())
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::cli::CliError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::py_errors::host_exception(
            "livekit.agents._bindings",
            "CLIError",
            ::std::string::ToString::to_string(self),
        )
    }
}
