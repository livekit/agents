//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `text_session` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
type TextSessionRunnerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::TextSessionRunner;

fn _extract_text_session_runner_new_max_depth(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TextSessionRunnerNewMaxDepthInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "max_depth")
}
type TextSessionRunnerNewMaxDepthInput = usize;

fn _extract_text_session_runner_progress_operation(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TextSessionRunnerProgressOperationInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "operation")
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
        crate::py_calls::check_pid()?;
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
        let out = py
            .detach(|| {
                <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::new(
                    &env, root_agent, agents, max_depth,
                )
            })
            .map_err(|e: ::lk_agents_abi::errors::Invalid| {
                <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        ::pyo3::PyResult::Ok(TextSessionRunner(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (request))]
    fn run(&self, py: Python<'_>, request: Bound<'_, PyAny>) -> PyResult<TextSessionUpdates> {
        crate::py_calls::check_pid()?;
        let request = crate::generated::records::text_session::text_session_request_from_py(
            request.as_any(),
        )?;
        let out = py
            .detach(|| {
                <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::run(
                    &self.0, request,
                )
            })
            .map_err(|e: ::lk_agents_abi::errors::Invalid| {
                <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        ::pyo3::PyResult::Ok(TextSessionUpdates(
            ::lk_agents_runtime::net::source::SharedStream::new(
                ::std::boxed::Box::new(out),
                ::lk_agents_runtime::process::host_clock(),
            ),
        ))
    }

    #[pyo3(signature = (request, stream, resources))]
    fn run_stream(
        &self,
        py: Python<'_>,
        request: Bound<'_, PyAny>,
        stream: Bound<'_, PyAny>,
        resources: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let request = crate::generated::records::text_session::text_session_request_from_py(
            request.as_any(),
        )?;
        let stream = stream
            .cast::<super::agent::CoreRunStream>()
            .map(|value| ::std::sync::Arc::clone(&value.get().0))
            .map_err(::pyo3::PyErr::from)?;
        let resources = resources
            .cast::<super::agent::CoreRunResources>()
            .map(|value| ::std::sync::Arc::clone(&value.get().0))
            .map_err(::pyo3::PyErr::from)?;
        py.detach(|| {
            <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::run_stream(
                &self.0, request, stream, resources,
            )
        })
        .map_err(|e: ::lk_agents_abi::errors::Invalid| {
            <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    /// The latest usage snapshot of the turn, across its local invocations; never advances it.
    #[getter]
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
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
        crate::py_calls::check_pid()?;
        let delivery = crate::generated::records::text_session::text_session_delivery_from_py(
            delivery.as_any(),
        )?;
        py.detach(|| {
            <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::release(
                &self.0, delivery,
            )
        })
        .map_err(|e: ::lk_agents_abi::errors::Invalid| {
            <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
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
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <TextSessionRunnerImpl as ::lk_agents_abi::text_session::TextSessionRunner>::progress(
                &self.0,
                &invocation_id,
                operation,
                message,
            )
        })
        .map_err(|e: ::lk_agents_abi::errors::Invalid| {
            <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }
}

/// One stream of `::lk_agents_abi::text_session::TextSessionUpdate` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "TextSessionUpdates", module = "livekit.agents._core")]
pub struct TextSessionUpdates(
    ::lk_agents_runtime::net::source::SharedStream<
        ::lk_agents_abi::text_session::TextSessionUpdate,
    >,
);

#[pymethods]
impl TextSessionUpdates {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::recv_native(&self.0, py, |py, item| {
            Ok(crate::generated::records::text_session::text_session_update_to_py(py, item)?
                .unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::close_native(&self.0, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
type CoreLocalInteractionImpl =
    <crate::generated::implementations::Selected as super::Implementations>::LocalInteraction;

fn _extract_local_interaction_new_depth(
    obj: &Bound<'_, PyAny>,
) -> PyResult<LocalInteractionNewDepthInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "depth")
}
type LocalInteractionNewDepthInput = usize;

fn _extract_local_interaction_progress_operation(
    obj: &Bound<'_, PyAny>,
) -> PyResult<LocalInteractionProgressOperationInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "operation")
}
type LocalInteractionProgressOperationInput = u64;

/// One call of an agent tool outside a TextSession: the child agent runs in process as a delegate
/// of the call (tools.md "Agents as tools"). Its decisions are `lk_agents_core::agent_tool`'s.
#[pyclass(frozen, name = "_core_LocalInteraction", module = "livekit.agents._core")]
pub struct CoreLocalInteraction(pub(crate) ::std::sync::Arc<CoreLocalInteractionImpl>);

#[pymethods]
impl CoreLocalInteraction {
    /// `child` runs for `function_call`, the parent's call of `tool`; `depth` counts the agent
    /// tools it runs under, itself included.
    #[new]
    #[pyo3(signature = (child, tool, function_call, depth))]
    fn new(
        py: Python<'_>,
        child: Bound<'_, PyAny>,
        tool: Bound<'_, PyAny>,
        function_call: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_local_interaction_new_depth)]
        depth: LocalInteractionNewDepthInput,
    ) -> PyResult<CoreLocalInteraction> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let child =
            crate::generated::records::text_session::text_session_agent_from_py(child.as_any())?;
        let tool =
            crate::generated::records::text_session::local_agent_tool_from_py(tool.as_any())?;
        let function_call =
            crate::generated::records::chat_items::chat_item_from_py(function_call.as_any())?;
        let out = py
            .detach(|| {
                <CoreLocalInteractionImpl as ::lk_agents_abi::text_session::LocalInteraction>::new(
                    &env,
                    child,
                    tool,
                    function_call,
                    depth,
                )
            })
            .map_err(|e: ::lk_agents_abi::errors::Invalid| {
                <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        ::pyo3::PyResult::Ok(CoreLocalInteraction(::std::sync::Arc::new(out)))
    }

    /// Run the child on `arguments` from the parent's `history`, under the caller's span: its
    /// tools' progress reaches `host.progress`; on a call (`live`) the request is announced
    /// through `host.delegated` and each question it asks is answered through `host.ask`, the
    /// answer resuming it. Returns the call's outcome; off a call, the child's question.
    #[pyo3(signature = (host, arguments, history, live, conn_options))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        host: Bound<'_, PyAny>,
        arguments: Bound<'_, PyAny>,
        history: Bound<'_, PyAny>,
        live: bool,
        conn_options: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::a2a::DelegationHostProxy::from_host(host.as_any())?;
        let arguments = crate::py_values::value_from_py(arguments.as_any(), "arguments")?;
        let history = history.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let conn_options =
            crate::generated::records::connect::api_connect_options_from_py(conn_options.as_any())?;
        let context = crate::py_calls::entered();
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreLocalInteractionImpl as ::lk_agents_abi::text_session::LocalInteraction>::run(&this, &env, host, arguments, history, live, conn_options, context, cx)
                .await.map_err(|e: ::lk_agents_abi::errors::Invalid| <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::tool_report::tool_outcome_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// The tool call was cancelled: the running child is cancelled; `settled()` waits for it.
    fn cancelled(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreLocalInteractionImpl as ::lk_agents_abi::text_session::LocalInteraction>::cancelled(&self.0));
        Ok(())
    }

    /// After `cancelled()`: the child's settlement, bounded by the teardown limit.
    fn settled<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreLocalInteractionImpl as ::lk_agents_abi::text_session::LocalInteraction>::settled(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// The call's retained state: the child's own history while it waits on the person.
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreLocalInteractionImpl as ::lk_agents_abi::text_session::LocalInteraction>::extra(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(crate::py_values::extra_to_py(py, &out)?.into_any())
    }

    /// What the child consumed across its invocations, the running one included.
    #[getter]
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreLocalInteractionImpl as ::lk_agents_abi::text_session::LocalInteraction>::usage(
                &self.0,
            )
        });
        crate::generated::records::usage::usage_to_py(py, out)
    }

    /// `ctx.update(message)` from a tool body of the child's invocation `invocation_id`.
    #[pyo3(signature = (invocation_id, operation, message))]
    fn progress(
        &self,
        py: Python<'_>,
        invocation_id: ::std::string::String,
        #[pyo3(from_py_with = _extract_local_interaction_progress_operation)]
        operation: LocalInteractionProgressOperationInput,
        message: ::std::string::String,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <CoreLocalInteractionImpl as ::lk_agents_abi::text_session::LocalInteraction>::progress(
                &self.0,
                &invocation_id,
                operation,
                message,
            )
        })
        .map_err(|e: ::lk_agents_abi::errors::Invalid| {
            <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
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

impl crate::py_errors::HostException for ::lk_agents_abi::text_session::RoutingError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::py_errors::host_exception(
            "livekit.agents.text_session",
            "TextSessionRoutingError",
            ::std::string::ToString::to_string(self),
        )
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::text_session::TextSessionError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        match self {
            Self::Routing(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Run(inner) => crate::py_errors::HostException::to_py_err(inner),
        }
    }
}
