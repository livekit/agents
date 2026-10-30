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

fn _extract_text_session_runner_new_max_depth(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TextSessionRunnerNewMaxDepthInput> {
    crate::python::number_from_py::<usize>(obj.as_any(), "max_depth")
}
type TextSessionRunnerNewMaxDepthInput = usize;

fn _extract_text_session_runner_progress_operation(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TextSessionRunnerProgressOperationInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "operation")
}
type TextSessionRunnerProgressOperationInput = u64;

#[pyclass(frozen, name = "TextSessionRunner", module = "livekit.agents._core")]
pub struct TextSessionRunner(pub(crate) ::std::sync::Arc<TextSessionRunnerImpl>);

#[pymethods]
impl TextSessionRunner {
    #[new]
    #[pyo3(signature = (root_agent, agents, max_depth))]
    fn new(
        py: Python<'_>,
        root_agent: ::std::string::String,
        agents: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_text_session_runner_new_max_depth)]
        max_depth: TextSessionRunnerNewMaxDepthInput,
    ) -> PyResult<TextSessionRunner> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let agents = agents.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::text_session::text_session_agent_from_py(
                        item.as_any(),
                    )
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = py.detach(|| <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::new(&env, root_agent, agents, max_depth)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as crate::python::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(TextSessionRunner(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (request))]
    fn run(&self, py: Python<'_>, request: Bound<'_, PyAny>) -> PyResult<TextSessionUpdates> {
        crate::host::check_pid()?;
        let request = crate::generated::records::text_session::text_session_request_from_py(
            request.as_any(),
        )?;
        let out = py.detach(|| <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::run(&self.0, request)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as crate::python::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(TextSessionUpdates(
            ::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new(out)),
        ))
    }

    /// The latest usage snapshot of the turn, across its local invocations; never advances it.
    #[getter]
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::usage(
                &self.0,
            )
        });
        crate::generated::records::usage::usage_to_py(py, out)
    }

    /// Acknowledge selected presentation before the child can release its parent.
    #[pyo3(signature = (delivery))]
    fn release(&self, py: Python<'_>, delivery: Bound<'_, PyAny>) -> PyResult<()> {
        crate::host::check_pid()?;
        let delivery = crate::generated::records::text_session::text_session_delivery_from_py(
            delivery.as_any(),
        )?;
        py.detach(|| {
            <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::release(
                &self.0, delivery,
            )
        })
        .map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| {
            <::lk_agents_abi::agent::run::InvalidRun as crate::python::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    #[pyo3(signature = (invocation_id, operation, message))]
    fn progress(
        &self,
        py: Python<'_>,
        invocation_id: ::std::string::String,
        #[pyo3(from_py_with = _extract_text_session_runner_progress_operation)]
        operation: TextSessionRunnerProgressOperationInput,
        message: ::std::string::String,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| {
            <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::progress(
                &self.0,
                &invocation_id,
                operation,
                message,
            )
        })
        .map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| {
            <::lk_agents_abi::agent::run::InvalidRun as crate::python::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }
}

/// One stream of `::lk_agents_abi::text_session::TextSessionUpdate` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "TextSessionUpdates", module = "livekit.agents._core")]
pub struct TextSessionUpdates(
    ::lk_agents_runtime::net::source::NativeSource<
        ::lk_agents_abi::text_session::TextSessionUpdate,
    >,
);

#[pymethods]
impl TextSessionUpdates {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::recv(&self.0, py, |py, item| {
            Ok(crate::generated::records::text_session::text_session_update_to_py(py, item)?
                .unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::aclose(&self.0, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.

/// `TextSessionState.to_json()`.
#[pyfunction]
pub fn _core_persist_text_session_state(record: Bound<'_, PyAny>) -> PyResult<String> {
    let record = crate::generated::records::text_session::text_session_state_from_py(&record)?;
    ::serde_json::to_string(&record)
        .map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("TextSessionState: {e}")))
}

/// `TextSessionState.from_json(data)`.
#[pyfunction]
pub fn _core_restore_text_session_state<'py>(
    py: Python<'py>,
    data: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let record: ::lk_agents_abi::text_session::TextSessionState = ::serde_json::from_str(data)
        .map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("TextSessionState: {e}")))?;
    crate::generated::records::text_session::text_session_state_to_py(py, record)
}
