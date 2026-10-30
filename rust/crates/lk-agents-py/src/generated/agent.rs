//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `agent` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
type VoiceAgentImpl = <crate::Selected as super::Implementations>::VoiceAgent;

/// `_core.VoiceAgent(definition, host, host_context)`: an [`AgentFrame`], built where the
/// definition converts.
#[pyclass(frozen, name = "VoiceAgent", module = "livekit.agents._core")]
pub struct VoiceAgent(pub(crate) ::std::sync::Arc<VoiceAgentImpl>);

#[pymethods]
impl VoiceAgent {
    #[new]
    #[pyo3(signature = (definition, host, host_context))]
    fn new(
        py: Python<'_>,
        definition: Bound<'_, PyAny>,
        host: Bound<'_, PyAny>,
        host_context: Bound<'_, PyAny>,
    ) -> PyResult<VoiceAgent> {
        crate::host::check_pid()?;
        let definition =
            crate::generated::records::agent_run::agent_definition_from_py(&definition)?;
        let host = crate::generated::agent::AgentCallbacksProxy::from_host(&host)?;
        let host_context = crate::conversions::host_ref_from_py(&host_context)?;
        let out = py.detach(|| {
            <VoiceAgentImpl as ::lk_agents_abi::agent::VoiceAgent>::new(
                definition,
                host,
                host_context,
            )
        });
        Ok(VoiceAgent(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::agent::AgentFrame`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::agent::AgentFrame>(),
        read: |obj| {
            let this = obj.cast::<VoiceAgent>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::agent::AgentFrame> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/assertions.rs.
type RunEventsImpl = <crate::Selected as super::Implementations>::RunEvents;

fn _extract_run_events_at_index(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "index")
}

fn _extract_run_events_skip_next_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "count")
}

fn _extract_run_events_contains_start(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "start")
}

fn _extract_run_events_contains_stop(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "stop")
}

fn _extract_run_events_failure_index(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<usize>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (::lk_agents_abi::python::number_from_py::<usize>(&obj, "index")).map(Some)
    }
}

/// The observed events (items) of one invocation and the position `next_event` reads from. Each call
/// returns the index of the event it matched.
#[pyclass(name = "RunEvents", module = "livekit.agents._core")]
pub struct RunEvents(pub(crate) RunEventsImpl);

#[pymethods]
impl RunEvents {
    #[new]
    #[pyo3(signature = (events))]
    fn new(py: Python<'_>, events: Bound<'_, PyAny>) -> PyResult<RunEvents> {
        crate::host::check_pid()?;
        let events = events.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::chat_items::chat_item_from_py(&item)
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = py.detach(|| {
            <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::new(events)
        });
        Ok(RunEvents(out))
    }

    /// The event at `index` (from the end when negative), meeting `expected` when given.
    #[pyo3(signature = (index, expected))]
    fn at(
        &self,
        #[pyo3(from_py_with = _extract_run_events_at_index)] index: i64,
        expected: Option<Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        crate::host::check_pid()?;
        let expected = expected
            .map(|v| crate::generated::records::agent_run::event_expectation_from_py(&v))
            .transpose()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::at(&self.0, index, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Advances to the next event of `expected`'s kind, or the next event when `None`.
    #[pyo3(signature = (expected))]
    fn next_event(&mut self, expected: Option<Bound<'_, PyAny>>) -> PyResult<usize> {
        crate::host::check_pid()?;
        let expected = expected
            .map(|v| crate::generated::records::agent_run::event_expectation_from_py(&v))
            .transpose()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::next_event(&mut self.0, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Advances past the next event only if it meets `expected`.
    #[pyo3(signature = (expected))]
    fn skip_next_event_if(
        &mut self,
        expected: Bound<'_, PyAny>,
    ) -> PyResult<::std::option::Option<usize>> {
        crate::host::check_pid()?;
        let expected = crate::generated::records::agent_run::event_expectation_from_py(&expected)?;
        let out =
            <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::skip_next_event_if(
                &mut self.0,
                expected,
            );
        Ok(out)
    }

    #[pyo3(signature = (count))]
    fn skip_next(
        &mut self,
        #[pyo3(from_py_with = _extract_run_events_skip_next_count)] count: usize,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::skip_next(&mut self.0, count).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn no_more_events(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::no_more_events(&self.0).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// The first event of `start..stop` meeting `expected`.
    #[pyo3(signature = (start, stop, expected))]
    fn contains(
        &self,
        #[pyo3(from_py_with = _extract_run_events_contains_start)] start: usize,
        #[pyo3(from_py_with = _extract_run_events_contains_stop)] stop: usize,
        expected: Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        crate::host::check_pid()?;
        let expected = crate::generated::records::agent_run::event_expectation_from_py(&expected)?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::contains(&self.0, start, stop, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// The message of a failure about the event at `index`, or the current one: `message` and
    /// the events with that one marked.
    #[pyo3(signature = (message, index))]
    fn failure(
        &self,
        message: ::std::string::String,
        #[pyo3(from_py_with = _extract_run_events_failure_index)] index: ::std::option::Option<
            usize,
        >,
    ) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::failure(
            &self.0, message, index,
        );
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
type AgentRunnerImpl = <crate::Selected as super::Implementations>::AgentRunner;

fn _extract_agent_runner_progress_operation(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "operation")
}

/// Runs invocations of one agent definition: each `run` streams its updates, then its settled
/// result; closing the stream cancels the invocation, whose work still settles.
#[pyclass(frozen, name = "AgentRunner", module = "livekit.agents._core")]
pub struct AgentRunner(pub(crate) ::std::sync::Arc<AgentRunnerImpl>);

#[pymethods]
impl AgentRunner {
    /// The runner of `definition` on `llm`, its callbacks run by `host`.
    #[new]
    #[pyo3(signature = (definition, llm, host))]
    fn new(
        py: Python<'_>,
        definition: Bound<'_, PyAny>,
        llm: Bound<'_, PyAny>,
        host: Bound<'_, PyAny>,
    ) -> PyResult<AgentRunner> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let definition =
            crate::generated::records::agent_run::agent_definition_from_py(&definition)?;
        let llm = crate::generated::llm::LlmProxy::from_host(&llm)?;
        let host = crate::generated::agent::AgentCallbacksProxy::from_host(&host)?;
        let out = py.detach(|| <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::new(&env, definition, llm, host)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AgentRunner(::std::sync::Arc::new(out)))
    }

    /// Starts one invocation; nothing runs until the first pull.
    #[pyo3(signature = (request))]
    fn run(&self, py: Python<'_>, request: Bound<'_, PyAny>) -> PyResult<AgentRunUpdates> {
        crate::host::check_pid()?;
        let request = crate::generated::records::agent_run::run_request_from_py(&request)?;
        let out = py.detach(|| <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::run(&self.0, request)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AgentRunUpdates(::lk_agents_runtime::net::source::NativeSource::new(
            ::std::boxed::Box::new(out),
        )))
    }

    /// The latest usage snapshot of invocation `invocation_id`; never advances it.
    #[pyo3(signature = (invocation_id))]
    fn usage<'py>(
        &self,
        py: Python<'py>,
        invocation_id: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = py.detach(|| <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::usage(&self.0, &invocation_id)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(crate::generated::records::usage::usage_to_py(py, out)?)
    }

    /// `ctx.update(message)` from the tool body running as host `operation` of invocation
    /// `invocation_id`: the run streams it as progress. Never waits; an update racing its call's
    /// end is dropped. Refused for an invocation this runner is not running.
    #[pyo3(signature = (invocation_id, operation, message))]
    fn progress(
        &self,
        py: Python<'_>,
        invocation_id: ::std::string::String,
        #[pyo3(from_py_with = _extract_agent_runner_progress_operation)] operation: u64,
        message: ::std::string::String,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::progress(&self.0, &invocation_id, operation, message)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

/// One stream of `::lk_agents_abi::agent::run::RunUpdate` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "AgentRunUpdates", module = "livekit.agents._core")]
pub struct AgentRunUpdates(
    ::lk_agents_runtime::net::source::NativeSource<::lk_agents_abi::agent::run::RunUpdate>,
);

#[pymethods]
impl AgentRunUpdates {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::recv(&self.0, py, |py, item| {
            Ok(crate::generated::records::agent_run::run_update_to_py(py, item)?.unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::aclose(&self.0, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.

/// [`::lk_agents_abi::agent::AgentCallbacks`] over a host `livekit.agents.voice.agent_host.AgentCallbacks`.
#[derive(Clone)]
pub struct AgentCallbacksProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl AgentCallbacksProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::AgentCallbacks>> {
        if let Some(native) =
            ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::AgentCallbacks>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::AgentCallbacks>> {
        if let Some(native) =
            ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::AgentCallbacks>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::agent::AgentCallbacks>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(AgentCallbacksProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::agent::AgentCallbacks for AgentCallbacksProxy {
    fn checkpoint<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<bool, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AgentCallbacks.checkpoint",
            invocation,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("checkpoint")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
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
                let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn render_instructions<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AgentCallbacks.render_instructions",
            invocation,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?];
                Ok((
                    host.bind(py).getattr("render_instructions")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
                let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn list_tools<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::agent::FunctionToolInfo>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AgentCallbacks.list_tools",
            invocation,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("list_tools")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
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
                let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::agent_run::function_tool_info_from_py(&item) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn prepare_tool<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
        arguments: ::serde_json::Map<::std::string::String, ::serde_json::Value>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::result::Result<
                ::lk_agents_abi::agent::PreparedArguments,
                ::lk_agents_abi::agent::ToolArgumentError,
            >,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let arguments_arg = arguments;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AgentCallbacks.prepare_tool",
            invocation,
            &[::lk_agents_abi::invoke::Scope::ToolCall],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    ::lk_agents_abi::python::extra_to_py(py, &arguments_arg)?.into_any(),
                ];
                Ok((host.bind(py).getattr("prepare_tool")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
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
        })
    }

    fn read_back<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
        prepared: &::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::lk_agents_abi::agent::ReadBackAnswer,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AgentCallbacks.read_back",
            invocation,
            &[::lk_agents_abi::invoke::Scope::ToolCall],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    crate::conversions::host_ref_to_py(py, prepared_arg)?,
                ];
                Ok((host.bind(py).getattr("read_back")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
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
                let out = (crate::generated::records::agent_run::read_back_answer_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn run_tool<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
        prepared: &::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            (
                ::lk_agents_abi::llm::tool_executor::ToolReport,
                ::serde_json::Map<::std::string::String, ::serde_json::Value>,
            ),
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AgentCallbacks.run_tool",
            invocation,
            &[::lk_agents_abi::invoke::Scope::ToolCall],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    crate::conversions::host_ref_to_py(py, prepared_arg)?,
                ];
                Ok((host.bind(py).getattr("run_tool")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
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
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "AgentCallbacks.run_tool"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((crate::generated::records::tool_report::tool_report_from_py(&item_0))?, (match ::lk_agents_abi::python::value_from_py(&item_1, "AgentCallbacks.run_tool")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("AgentCallbacks.run_tool", ": expected a dict"))) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn validate_output<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
        candidate: ::lk_agents_abi::agent::OutputCandidate,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::lk_agents_abi::agent::ValidatedOutput,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let candidate_arg = candidate;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AgentCallbacks.validate_output",
            invocation,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    crate::generated::records::agent_run::output_candidate_to_py(
                        py,
                        candidate_arg,
                    )?,
                ];
                Ok((
                    host.bind(py).getattr("validate_output")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
                let out = (crate::generated::records::agent_run::validated_output_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn render_template<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
        template: ::lk_agents_abi::agent::ToolTemplate,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::option::Option<::std::string::String>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let template_arg = template;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AgentCallbacks.render_template",
            invocation,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    crate::generated::records::agent_run::tool_template_to_py(py, template_arg)?,
                ];
                Ok((
                    host.bind(py).getattr("render_template")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
                let out = (if o.is_none() { Ok(None) } else { (o.extract::<::std::string::String>()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn cancel(&self, invocation: &::lk_agents_abi::agent::Invocation) {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AgentCallbacks.cancel",
            invocation,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
    }

    fn answer_expert(
        &self,
        invocation: &::lk_agents_abi::agent::Invocation,
        answer: ::std::string::String,
    ) {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let answer_arg = answer;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AgentCallbacks.answer_expert",
            invocation,
            &[::lk_agents_abi::invoke::Scope::ToolCall],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(answer_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("answer_expert")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::HostInstructions>> {
        if let Some(native) =
            ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::HostInstructions>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::HostInstructions>> {
        if let Some(native) =
            ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::HostInstructions>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::agent::HostInstructions>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(HostInstructionsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::agent::HostInstructions for HostInstructionsProxy {
    fn render<'a>(
        &'a self,
        speech: ::std::sync::Arc<dyn ::lk_agents_abi::voice::speech_handle::Speech>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let speech_arg = speech;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "HostInstructions._core_render",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [<dyn ::lk_agents_abi::voice::speech_handle::Speech as super::NativeToHost>::to_host(py, speech_arg)?];
                Ok((host.bind(py).getattr("_core_render")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
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
                let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}
