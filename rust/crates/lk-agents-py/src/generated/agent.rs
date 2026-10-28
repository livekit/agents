//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `agent` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
type AgentRunnerImpl = <crate::Selected as super::Implementations>::AgentRunner;

/// Runs invocations of one agent definition: each `run` streams its updates, then its settled
/// result; closing the stream cancels the invocation, whose work still settles.
#[pyclass(frozen, name = "AgentRunner", module = "livekit.agents._core")]
pub struct AgentRunner(::std::sync::Arc<AgentRunnerImpl>);

#[pymethods]
impl AgentRunner {
    /// The runner of `definition` on `llm`, its callbacks run by `host`.
    #[new]
    #[pyo3(signature = (definition, llm, host))]
    fn new(definition: ::lk_agents_abi::agent::AgentDefinition, llm: Bound<'_, PyAny>, host: Bound<'_, PyAny>) -> PyResult<AgentRunner> {
        ::lk_agents_abi::python::check_pid()?;
        let llm = <dyn ::lk_agents_abi::Llm as ::lk_agents_abi::python::HostProxy>::from_host(&llm, None)?;
        let host = super::agent::AgentHostProxy::from_host(&host)?;
        let out = <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::new(definition, llm, host).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(&e)))?;
        Ok(AgentRunner(::std::sync::Arc::new(out)))
    }

    /// Starts one invocation; nothing runs until the first pull.
    #[pyo3(signature = (request))]
    fn run(&self, request: ::lk_agents_abi::agent::run::RunRequest) -> PyResult<AgentRunUpdates> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::run(&self.0, request).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(&e)))?;
        Ok(AgentRunUpdates(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new(out))))
    }
}

/// One stream of `::lk_agents_abi::agent::run::RunUpdate` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "AgentRunUpdates", module = "livekit.agents._core")]
pub struct AgentRunUpdates(::lk_agents_runtime::source::NativeSource<::lk_agents_abi::agent::run::RunUpdate>);

#[pymethods]
impl AgentRunUpdates {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.recv(py)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.aclose(py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.

/// [`::lk_agents_abi::agent::AgentHost`] over a host `livekit.agents.voice.agent_host.AgentHost`:
/// each call runs in a copy of the host context its `#[host(context)]` parameter captured.
pub struct AgentHostProxy(::std::sync::Arc<Py<PyAny>>);

impl AgentHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::AgentHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::AgentHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(AgentHostProxy(::std::sync::Arc::new(obj.clone().unbind()))))
    }
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::agent::AgentHost for AgentHostProxy {
    async fn render_instructions(&self, invocation: &::lk_agents_abi::agent::Invocation) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.0);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let context = ::lk_agents_abi::agent::HostContext::host_context(invocation);
        let out = match context {
            Some(context) => ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args = [::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_arg, py)?];
                Ok((host.bind(py).getattr("render_instructions")?, ::pyo3::types::PyTuple::new(py, args)?))
            })
            .await,
            None => Err(::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.render_instructions: the call captured no host context")),
        };
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn prepare_tool(&self, invocation: &::lk_agents_abi::agent::Invocation, arguments: ::serde_json::Map<::std::string::String, ::serde_json::Value>) -> ::std::result::Result<::std::result::Result<::lk_agents_abi::agent::PreparedArguments, ::lk_agents_abi::agent::ToolArgumentError>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.0);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let arguments_arg = arguments;
        let context = ::lk_agents_abi::agent::HostContext::host_context(invocation);
        let out = match context {
            Some(context) => ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args = [::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_arg, py)?, ::lk_agents_abi::python::extra_to_py(py, &arguments_arg)?.into_any()];
                Ok((host.bind(py).getattr("prepare_tool")?, ::pyo3::types::PyTuple::new(py, args)?))
            })
            .await,
            None => Err(::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.prepare_tool: the call captured no host context")),
        };
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    if e.is_instance(py, &py.import("livekit.agents.voice.agent_host").and_then(|m| m.getattr("ToolArgumentError")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?) {
                        return Ok(Err(<::lk_agents_abi::agent::ToolArgumentError as ::std::convert::From<String>>::from(e.value(py).to_string())));
                    }
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::agent::PreparedArguments>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(Ok(out))
        })
    }

    async fn read_back(&self, invocation: &::lk_agents_abi::agent::Invocation, prepared: &::lk_agents_abi::HostRef) -> ::std::result::Result<::lk_agents_abi::agent::ReadBackAnswer, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.0);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let context = ::lk_agents_abi::agent::HostContext::host_context(invocation);
        let out = match context {
            Some(context) => ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args = [::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(prepared_arg, py)?];
                Ok((host.bind(py).getattr("read_back")?, ::pyo3::types::PyTuple::new(py, args)?))
            })
            .await,
            None => Err(::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.read_back: the call captured no host context")),
        };
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::agent::ReadBackAnswer>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn run_tool(&self, invocation: &::lk_agents_abi::agent::Invocation, prepared: &::lk_agents_abi::HostRef) -> ::std::result::Result<::lk_agents_abi::llm::tool_executor::ToolReport, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.0);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let context = ::lk_agents_abi::agent::HostContext::host_context(invocation);
        let out = match context {
            Some(context) => ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args = [::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(prepared_arg, py)?];
                Ok((host.bind(py).getattr("run_tool")?, ::pyo3::types::PyTuple::new(py, args)?))
            })
            .await,
            None => Err(::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.run_tool: the call captured no host context")),
        };
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::llm::tool_executor::ToolReport>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn validate_output(&self, invocation: &::lk_agents_abi::agent::Invocation, candidate: ::lk_agents_abi::agent::OutputCandidate) -> ::std::result::Result<::lk_agents_abi::agent::ValidatedOutput, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.0);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let candidate_arg = candidate;
        let context = ::lk_agents_abi::agent::HostContext::host_context(invocation);
        let out = match context {
            Some(context) => ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args = [::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(candidate_arg, py)?];
                Ok((host.bind(py).getattr("validate_output")?, ::pyo3::types::PyTuple::new(py, args)?))
            })
            .await,
            None => Err(::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.validate_output: the call captured no host context")),
        };
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::agent::ValidatedOutput>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn cancel(&self, invocation: &::lk_agents_abi::agent::Invocation) {
        let host = ::std::sync::Arc::clone(&self.0);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let context = ::lk_agents_abi::agent::HostContext::host_context(invocation);
        let Some(context) = context.cloned() else {
            ::lk_agents_abi::python::log::warn!("AgentHost.cancel failed: {}", ::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.cancel: the call captured no host context"));
            return;
        };
        let runtime = match ::lk_agents_runtime::process::handle() {
            Ok(runtime) => runtime,
            Err(e) => {
                ::lk_agents_abi::python::log::warn!("AgentHost.cancel failed: {e}");
                return;
            }
        };
        runtime.spawn(async move {
            let context = &context;
            if let Err(e) = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args = [::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_arg, py)?];
                Ok((host.bind(py).getattr("cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await {
                ::lk_agents_abi::python::log::warn!("AgentHost.cancel failed: {e}");
            }
        });
    }
}
