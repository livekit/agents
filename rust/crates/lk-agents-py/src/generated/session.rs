//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `session` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/session/remote.rs.
type RemoteAgentStateImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RemoteAgentView;

/// Retained communication with one configured remote, scoped to the selected conversation.
#[pyclass(frozen, name = "RemoteAgentState", module = "livekit.agents.storage.session_state")]
pub struct RemoteAgentState(pub(crate) ::std::sync::Arc<RemoteAgentStateImpl>);

#[pymethods]
impl RemoteAgentState {
    /// Pinned destination, without credentials.
    #[getter]
    fn endpoint(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        let out = py.detach(|| <RemoteAgentStateImpl as ::lk_agents_abi::session::remote::RemoteAgentView>::endpoint(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// Observed communication in local journal order.
    #[getter]
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| <RemoteAgentStateImpl as ::lk_agents_abi::session::remote::RemoteAgentView>::events(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| {
                        crate::generated::records::delegation_remote::a2_a_journal_entry_to_py(
                            py, x,
                        )
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// Observed messages in either direction, indexed by message ID; includes contextless replies.
    #[getter]
    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| <RemoteAgentStateImpl as ::lk_agents_abi::session::remote::RemoteAgentView>::messages(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok({
            let d = ::pyo3::types::PyDict::new(py);
            for (k, x) in out {
                d.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                    crate::py_protocol::a2a_message_to_py(py, x)?,
                )?;
            }
            d.into_any()
        })
    }

    /// Observed tasks, indexed by task ID; their protocol history may be absent.
    #[getter]
    fn tasks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| <RemoteAgentStateImpl as ::lk_agents_abi::session::remote::RemoteAgentView>::tasks(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok({
            let dict = ::pyo3::types::PyDict::new(py);
            for (key, value) in out {
                dict.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(key, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(
                        super::session::RemoteTask(::std::sync::Arc::new(value)),
                        py,
                    )?,
                )?;
            }
            dict.into_any()
        })
    }

    /// Grouping of the same observations by context ID, without another message history.
    #[getter]
    fn contexts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| <RemoteAgentStateImpl as ::lk_agents_abi::session::remote::RemoteAgentView>::contexts(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok({
            let dict = ::pyo3::types::PyDict::new(py);
            for (key, value) in out {
                dict.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(key, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(
                        super::session::RemoteContextState(::std::sync::Arc::new(value)),
                        py,
                    )?,
                )?;
            }
            dict.into_any()
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/session/remote.rs.
type RemoteContextStateImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RemoteContextView;

/// Observed messages and tasks grouped by their remote context ID.
#[pyclass(frozen, name = "RemoteContextState", module = "livekit.agents.storage.session_state")]
pub struct RemoteContextState(pub(crate) ::std::sync::Arc<RemoteContextStateImpl>);

#[pymethods]
impl RemoteContextState {
    #[getter]
    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| <RemoteContextStateImpl as ::lk_agents_abi::session::remote::RemoteContextView>::messages(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok({
            let d = ::pyo3::types::PyDict::new(py);
            for (k, x) in out {
                d.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                    crate::py_protocol::a2a_message_to_py(py, x)?,
                )?;
            }
            d.into_any()
        })
    }

    #[getter]
    fn tasks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| <RemoteContextStateImpl as ::lk_agents_abi::session::remote::RemoteContextView>::tasks(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok({
            let dict = ::pyo3::types::PyDict::new(py);
            for (key, value) in out {
                dict.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(key, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(
                        super::session::RemoteTask(::std::sync::Arc::new(value)),
                        py,
                    )?,
                )?;
            }
            dict.into_any()
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/session/remote.rs.
type RemoteTaskImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RemoteTaskView;

/// A task's current protocol observation and its task-owned local routing facts.
#[pyclass(frozen, name = "RemoteTask", module = "livekit.agents.storage.session_state")]
pub struct RemoteTask(pub(crate) ::std::sync::Arc<RemoteTaskImpl>);

#[pymethods]
impl RemoteTask {
    /// The observed SDK Task, including an unspecified status when only deltas were received.
    #[getter]
    fn task<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| <RemoteTaskImpl as ::lk_agents_abi::session::remote::RemoteTaskView>::task(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_protocol::a2a_task_to_py(py, out)
    }

    /// Retained input occurrences, indexed by occurrence ID independently of question wording.
    #[getter]
    fn input_requests<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = py.detach(|| <RemoteTaskImpl as ::lk_agents_abi::session::remote::RemoteTaskView>::input_requests(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok({
            let d = ::pyo3::types::PyDict::new(py);
            for (k, x) in out {
                d.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                    crate::generated::records::delegation_remote::input_request_to_py(py, x)?,
                )?;
            }
            d.into_any()
        })
    }

    /// Explicit local intent; only an observed task state confirms remote cancellation.
    #[getter]
    fn cancel_requested(&self, py: Python<'_>) -> PyResult<bool> {
        let out = py.detach(|| <RemoteTaskImpl as ::lk_agents_abi::session::remote::RemoteTaskView>::cancel_requested(&self.0)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// Request cancellation through the current bound native owner and its configured
    /// remote client. Detached views and tasks without a live driver cannot dispatch.
    /// Success requires persisted observation of the canceled task state.
    fn cancel<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <RemoteTaskImpl as ::lk_agents_abi::session::remote::RemoteTaskView>::cancel(&this, )
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}
