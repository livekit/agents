//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `agent` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/agent/assertions.rs.
type RunEventsImpl = <crate::Selected as super::Implementations>::RunEvents;

/// The observed events of one invocation and the position `next_event` reads from. Each call
/// returns the index of the event it matched.
#[pyclass(name = "RunEvents", module = "livekit.agents._core")]
pub struct RunEvents(RunEventsImpl);

#[pymethods]
impl RunEvents {
    #[new]
    #[pyo3(signature = (events))]
    fn new(events: Bound<'_, PyAny>) -> PyResult<RunEvents> {
        ::lk_agents_abi::python::check_pid()?;
        let events = events.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::agent_run::observed_event_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::new(events);
        Ok(RunEvents(out))
    }

    /// The event at `index` (from the end when negative), meeting `expected` when given.
    #[pyo3(signature = (index, expected))]
    fn at(&self, index: i64, expected: Option<Bound<'_, PyAny>>) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let expected = expected.map(|v| crate::generated::records::agent_run::event_expectation_from_py(&v)).transpose()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::at(&self.0, index, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Advances to the next event of `expected`'s kind, or the next event when `None`.
    #[pyo3(signature = (expected))]
    fn next_event(&mut self, expected: Option<Bound<'_, PyAny>>) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let expected = expected.map(|v| crate::generated::records::agent_run::event_expectation_from_py(&v)).transpose()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::next_event(&mut self.0, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Advances past the next event only if it meets `expected`.
    #[pyo3(signature = (expected))]
    fn skip_next_event_if(&mut self, expected: Bound<'_, PyAny>) -> PyResult<::std::option::Option<usize>> {
        ::lk_agents_abi::python::check_pid()?;
        let expected = crate::generated::records::agent_run::event_expectation_from_py(&expected)?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::skip_next_event_if(&mut self.0, expected);
        Ok(out)
    }

    #[pyo3(signature = (count))]
    fn skip_next(&mut self, count: usize) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::skip_next(&mut self.0, count).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn no_more_events(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::no_more_events(&self.0).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// The first event of `start..stop` meeting `expected`.
    #[pyo3(signature = (start, stop, expected))]
    fn contains(&self, start: usize, stop: usize, expected: Bound<'_, PyAny>) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let expected = crate::generated::records::agent_run::event_expectation_from_py(&expected)?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::contains(&self.0, start, stop, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// The message of a failure about the event at `index`, or the current one: `message` and
    /// the events with that one marked.
    #[pyo3(signature = (message, index))]
    fn failure(&self, message: ::std::string::String, index: ::std::option::Option<usize>) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::failure(&self.0, message, index);
        Ok(out)
    }
}

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
    fn new(definition: Bound<'_, PyAny>, llm: Bound<'_, PyAny>, host: Bound<'_, PyAny>) -> PyResult<AgentRunner> {
        ::lk_agents_abi::python::check_pid()?;
        let definition = crate::generated::records::agent_run::agent_definition_from_py(&definition)?;
        let llm = super::llm::LlmProxy::from_host(&llm)?;
        let host = super::agent::AgentHostProxy::from_host(&host)?;
        let out = <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::new(definition, llm, host).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AgentRunner(::std::sync::Arc::new(out)))
    }

    /// Starts one invocation; nothing runs until the first pull.
    #[pyo3(signature = (request))]
    fn run(&self, request: Bound<'_, PyAny>) -> PyResult<AgentRunUpdates> {
        ::lk_agents_abi::python::check_pid()?;
        let request = crate::generated::records::agent_run::run_request_from_py(&request)?;
        let out = <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::run(&self.0, request).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AgentRunUpdates(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new(out))))
    }

    /// `ctx.update(message)` from the tool body running as host `operation` of invocation
    /// `invocation_id`: the run streams it as progress. Never waits; an update racing its call's
    /// end is dropped. Refused for an invocation this runner is not running.
    #[pyo3(signature = (invocation_id, operation, message))]
    fn progress(&self, invocation_id: ::std::string::String, operation: u64, message: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::progress(&self.0, &invocation_id, operation, message).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

/// One stream of `::lk_agents_abi::agent::run::RunUpdate` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "AgentRunUpdates", module = "livekit.agents._core")]
pub struct AgentRunUpdates(::lk_agents_runtime::source::NativeSource<::lk_agents_abi::agent::run::RunUpdate>);

#[pymethods]
impl AgentRunUpdates {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.recv(py, |py, item| Ok(crate::generated::records::agent_run::run_update_to_py(py, item)?.unbind()))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.aclose(py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.

/// [`::lk_agents_abi::agent::AgentHost`] over a host `livekit.agents.voice.agent_host.AgentHost`.
#[derive(Clone)]
pub struct AgentHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl AgentHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::AgentHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::AgentHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::agent::AgentHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(AgentHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), notifier: crate::host::Notifier::new(crate::host::capture_context(obj.py())?) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::agent::AgentHost>(|obj| AgentHostProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::agent::AgentHost for AgentHostProxy {
    async fn checkpoint(&self, invocation: &::lk_agents_abi::agent::Invocation) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.checkpoint: the call captured no host context"))));
        };
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("checkpoint")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn render_instructions(&self, invocation: &::lk_agents_abi::agent::Invocation) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.render_instructions: the call captured no host context"))));
        };
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("render_instructions")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let arguments_arg = arguments;
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.prepare_tool: the call captured no host context"))));
        };
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?, ::lk_agents_abi::python::extra_to_py(py, &arguments_arg)?.into_any()];
                Ok((host.bind(py).getattr("prepare_tool")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
            let out = (crate::generated::records::agent_run::prepared_arguments_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(Ok(out))
        })
    }

    async fn read_back(&self, invocation: &::lk_agents_abi::agent::Invocation, prepared: &::lk_agents_abi::HostRef) -> ::std::result::Result<::lk_agents_abi::agent::ReadBackAnswer, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.read_back: the call captured no host context"))));
        };
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(prepared_arg, py)?];
                Ok((host.bind(py).getattr("read_back")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::records::agent_run::read_back_answer_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn run_tool(&self, invocation: &::lk_agents_abi::agent::Invocation, prepared: &::lk_agents_abi::HostRef) -> ::std::result::Result<(::lk_agents_abi::llm::tool_executor::ToolReport, ::serde_json::Map<::std::string::String, ::serde_json::Value>), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.run_tool: the call captured no host context"))));
        };
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(prepared_arg, py)?];
                Ok((host.bind(py).getattr("run_tool")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "AgentHost.run_tool"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((item_0.extract::<::lk_agents_abi::llm::tool_executor::ToolReport>())?, (match ::lk_agents_abi::python::value_from_py(&item_1, "AgentHost.run_tool")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("AgentHost.run_tool", ": expected a dict"))) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn validate_output(&self, invocation: &::lk_agents_abi::agent::Invocation, candidate: ::lk_agents_abi::agent::OutputCandidate) -> ::std::result::Result<::lk_agents_abi::agent::ValidatedOutput, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let candidate_arg = candidate;
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.validate_output: the call captured no host context"))));
        };
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?, crate::generated::records::agent_run::output_candidate_to_py(py, candidate_arg)?];
                Ok((host.bind(py).getattr("validate_output")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::records::agent_run::validated_output_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn cancel(&self, invocation: &::lk_agents_abi::agent::Invocation) {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            ::lk_agents_abi::python::log::warn!("AgentHost.cancel failed: {}", ::pyo3::exceptions::PyRuntimeError::new_err("AgentHost.cancel: the call captured no host context"));
            return;
        };
        self.notifier.notify(context, "AgentHost.cancel", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?];
            Ok((host.bind(py).getattr("cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.

/// [`::lk_agents_abi::agent::HostInstructions`] over a host `livekit.agents.voice.agent_session._DynamicInstructions`.
#[derive(Clone)]
pub struct HostInstructionsProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl HostInstructionsProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::HostInstructions>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::HostInstructions>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::agent::HostInstructions>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(HostInstructionsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::agent::HostInstructions>(|obj| HostInstructionsProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::agent::HostInstructions for HostInstructionsProxy {
    async fn render(&self, speech: &::lk_agents_abi::CoreObject) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let speech_arg = ::std::clone::Clone::clone(speech);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = crate::host::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(speech_arg, py)?];
                Ok((host.bind(py).getattr("_core_render")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
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
}
