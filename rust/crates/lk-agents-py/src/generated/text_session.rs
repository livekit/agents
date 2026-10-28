//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `text_session` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
type TextSessionRunnerImpl = <crate::Selected as super::Implementations>::TextSessionRunner;

#[pyclass(frozen, name = "TextSessionRunner", module = "livekit.agents._core")]
pub struct TextSessionRunner(::std::sync::Arc<TextSessionRunnerImpl>);

#[pymethods]
impl TextSessionRunner {
    #[new]
    #[pyo3(signature = (root_agent, agents, max_depth))]
    fn new(root_agent: ::std::string::String, agents: ::std::vec::Vec<::lk_agents_abi::text_session::TextSessionAgent>, max_depth: usize) -> PyResult<TextSessionRunner> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::new(root_agent, agents, max_depth).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(TextSessionRunner(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (request))]
    fn run(&self, request: ::lk_agents_abi::text_session::TextSessionRequest) -> PyResult<TextSessionUpdates> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::run(&self.0, request).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(TextSessionUpdates(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new(out))))
    }

    /// Acknowledge selected presentation before the child can release its parent.
    #[pyo3(signature = (delivery))]
    fn release(&self, delivery: ::lk_agents_abi::text_session::TextSessionDelivery) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::release(&self.0, delivery).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (invocation_id, operation, message))]
    fn progress(&self, invocation_id: ::std::string::String, operation: u64, message: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::progress(&self.0, &invocation_id, operation, message).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

/// One stream of `::lk_agents_abi::text_session::TextSessionUpdate` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "TextSessionUpdates", module = "livekit.agents._core")]
pub struct TextSessionUpdates(::lk_agents_runtime::source::NativeSource<::lk_agents_abi::text_session::TextSessionUpdate>);

#[pymethods]
impl TextSessionUpdates {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.recv(py)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.aclose(py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.

/// `TextSessionState.to_json()`.
#[pyfunction]
pub fn _core_persist_text_session_state(record: ::lk_agents_abi::text_session::TextSessionState) -> PyResult<String> {
    ::serde_json::to_string(&record)
        .map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("TextSessionState: {e}")))
}

/// `TextSessionState.from_json(data)`.
#[pyfunction]
pub fn _core_restore_text_session_state(data: &str) -> PyResult<::lk_agents_abi::text_session::TextSessionState> {
    ::serde_json::from_str(data)
        .map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("TextSessionState: {e}")))
}
