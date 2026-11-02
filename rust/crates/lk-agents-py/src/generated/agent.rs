//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `agent` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/agent/mod.rs.
type VoiceAgentImpl =
    <crate::generated::implementations::Selected as super::Implementations>::VoiceAgent;

fn _extract_voice_agent_new_toolset_scope(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceAgentNewToolsetScopeInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "toolset_scope")
}
type VoiceAgentNewToolsetScopeInput = u64;

/// `_core.VoiceAgent(definition, host, host_context)`: an [`AgentFrame`], built where the
/// definition converts.
#[pyclass(frozen, name = "VoiceAgent", module = "livekit.agents._core")]
pub struct VoiceAgent(pub(crate) ::std::sync::Arc<VoiceAgentImpl>);

#[pymethods]
impl VoiceAgent {
    #[new]
    #[pyo3(signature = (definition, host, host_context, toolset_scope = 0))]
    fn new(
        py: Python<'_>,
        definition: Bound<'_, PyAny>,
        host: Bound<'_, PyAny>,
        host_context: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_voice_agent_new_toolset_scope)]
        toolset_scope: VoiceAgentNewToolsetScopeInput,
    ) -> PyResult<VoiceAgent> {
        crate::py_calls::check_pid()?;
        let definition =
            crate::generated::records::agent_run::agent_definition_from_py(definition.as_any())?;
        let host = crate::generated::agent::AgentCallbacksProxy::from_host(host.as_any())?;
        let host_context = crate::py_objects::object_ref_from_py(host_context.as_any())?;
        let out = py.detach(|| {
            <VoiceAgentImpl as ::lk_agents_abi::agent::VoiceAgent>::new(
                definition,
                host,
                host_context,
                toolset_scope,
            )
        });
        ::pyo3::PyResult::Ok(VoiceAgent(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::agent::AgentFrame`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::agent::AgentFrame>(),
        read: |obj| {
            let this = obj.cast::<VoiceAgent>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::agent::AgentFrame> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/assertions.rs.
type RunEventsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RunEvents;

fn _extract_run_events_at_index(obj: &Bound<'_, PyAny>) -> PyResult<RunEventsAtIndexInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "index")
}
type RunEventsAtIndexInput = i64;
type RunEventsSkipNextEventIfOutput = ::std::option::Option<usize>;

fn _extract_run_events_skip_next_count(
    obj: &Bound<'_, PyAny>,
) -> PyResult<RunEventsSkipNextCountInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "count")
}
type RunEventsSkipNextCountInput = usize;

fn _extract_run_events_contains_start(
    obj: &Bound<'_, PyAny>,
) -> PyResult<RunEventsContainsStartInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "start")
}
type RunEventsContainsStartInput = usize;

fn _extract_run_events_contains_stop(
    obj: &Bound<'_, PyAny>,
) -> PyResult<RunEventsContainsStopInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "stop")
}
type RunEventsContainsStopInput = usize;

fn _extract_run_events_failure_index(
    obj: &Bound<'_, PyAny>,
) -> PyResult<RunEventsFailureIndexInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<usize>(obj.as_any(), "index")).map(Some)
    }
}
type RunEventsFailureIndexInput = ::std::option::Option<usize>;

/// The observed events (items) of one invocation and the position `next_event` reads from. Each call
/// returns the index of the event it matched.
#[pyclass(name = "RunEvents", module = "livekit.agents._core")]
pub struct RunEvents(pub(crate) RunEventsImpl);

#[pymethods]
impl RunEvents {
    #[new]
    #[pyo3(signature = (events))]
    fn new(py: Python<'_>, events: Bound<'_, PyAny>) -> PyResult<RunEvents> {
        crate::py_calls::check_pid()?;
        let events = events.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = py.detach(|| {
            <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::new(events)
        });
        ::pyo3::PyResult::Ok(RunEvents(out))
    }

    /// The event at `index` (from the end when negative), meeting `expected` when given.
    #[pyo3(signature = (index, expected))]
    fn at(
        &self,
        #[pyo3(from_py_with = _extract_run_events_at_index)] index: RunEventsAtIndexInput,
        expected: Option<Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        crate::py_calls::check_pid()?;
        let expected = expected
            .map(|v| crate::generated::records::agent_run::event_expectation_from_py(v.as_any()))
            .transpose()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::at(&self.0, index, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// Advances to the next event of `expected`'s kind, or the next event when `None`.
    #[pyo3(signature = (expected))]
    fn next_event(&mut self, expected: Option<Bound<'_, PyAny>>) -> PyResult<usize> {
        crate::py_calls::check_pid()?;
        let expected = expected
            .map(|v| crate::generated::records::agent_run::event_expectation_from_py(v.as_any()))
            .transpose()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::next_event(&mut self.0, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// Advances past the next event only if it meets `expected`.
    #[pyo3(signature = (expected))]
    fn skip_next_event_if(
        &mut self,
        expected: Bound<'_, PyAny>,
    ) -> PyResult<RunEventsSkipNextEventIfOutput> {
        crate::py_calls::check_pid()?;
        let expected =
            crate::generated::records::agent_run::event_expectation_from_py(expected.as_any())?;
        let out =
            <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::skip_next_event_if(
                &mut self.0,
                expected,
            );
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (count))]
    fn skip_next(
        &mut self,
        #[pyo3(from_py_with = _extract_run_events_skip_next_count)]
        count: RunEventsSkipNextCountInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::skip_next(&mut self.0, count).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn no_more_events(&self) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::no_more_events(&self.0).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// The first event of `start..stop` meeting `expected`.
    #[pyo3(signature = (start, stop, expected))]
    fn contains(
        &self,
        #[pyo3(from_py_with = _extract_run_events_contains_start)]
        start: RunEventsContainsStartInput,
        #[pyo3(from_py_with = _extract_run_events_contains_stop)] stop: RunEventsContainsStopInput,
        expected: Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        crate::py_calls::check_pid()?;
        let expected =
            crate::generated::records::agent_run::event_expectation_from_py(expected.as_any())?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::contains(&self.0, start, stop, expected).map_err(|e: ::lk_agents_abi::agent::assertions::AssertionFailure| <::lk_agents_abi::agent::assertions::AssertionFailure as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// The message of a failure about the event at `index`, or the current one: `message` and
    /// the events with that one marked.
    #[pyo3(signature = (message, index))]
    fn failure(
        &self,
        message: ::std::string::String,
        #[pyo3(from_py_with = _extract_run_events_failure_index)] index: RunEventsFailureIndexInput,
    ) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = <RunEventsImpl as ::lk_agents_abi::agent::assertions::RunEvents>::failure(
            &self.0, message, index,
        );
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/authoring.rs.
type CoreAgentConfigImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AgentConfig;
type AgentConfigRegisterOutput = (u64, ::std::option::Option<u64>);
type AgentConfigCallbacksOutput = ::std::vec::Vec<u64>;
type AgentConfigHookOutput = ::std::option::Option<u64>;

#[pyclass(name = "_core_AgentConfig", module = "livekit.agents._core")]
pub struct CoreAgentConfig(pub(crate) CoreAgentConfigImpl);

#[pymethods]
impl CoreAgentConfig {
    #[new]
    #[pyo3(signature = (instructions, name, retries))]
    fn new(
        py: Python<'_>,
        instructions: ::std::string::String,
        name: ::std::option::Option<::std::string::String>,
        retries: Option<Bound<'_, PyAny>>,
    ) -> PyResult<CoreAgentConfig> {
        crate::py_calls::check_pid()?;
        let retries =
            retries.map(|v| crate::py_values::value_from_py(v.as_any(), "retries")).transpose()?;
        let out = py.detach(|| <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::new(instructions, name, retries)).map_err(|e: ::lk_agents_abi::agent::authoring::AuthoringError| <::lk_agents_abi::agent::authoring::AuthoringError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(CoreAgentConfig(out))
    }

    #[pyo3(signature = (discovered))]
    fn resolve_name(
        &mut self,
        discovered: ::std::option::Option<::std::string::String>,
    ) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out =
            <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::resolve_name(
                &mut self.0,
                discovered,
            );
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn instructions(&self) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out =
            <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::instructions(
                &self.0,
            );
        ::pyo3::PyResult::Ok(out)
    }

    fn freeze(&mut self) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::freeze(
            &mut self.0,
        );
        Ok(())
    }

    #[pyo3(signature = (what))]
    fn check_mutable(&self, what: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::check_mutable(&self.0, what).map_err(|e: ::lk_agents_abi::agent::authoring::AuthoringError| <::lk_agents_abi::agent::authoring::AuthoringError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (kind))]
    fn register(&mut self, kind: Bound<'_, PyAny>) -> PyResult<AgentConfigRegisterOutput> {
        crate::py_calls::check_pid()?;
        let kind =
            crate::generated::records::agent_authoring::callback_kind_from_py(kind.as_any())?;
        let out = <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::register(&mut self.0, kind).map_err(|e: ::lk_agents_abi::agent::authoring::AuthoringError| <::lk_agents_abi::agent::authoring::AuthoringError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (kind))]
    fn callbacks(&self, kind: Bound<'_, PyAny>) -> PyResult<AgentConfigCallbacksOutput> {
        crate::py_calls::check_pid()?;
        let kind =
            crate::generated::records::agent_authoring::callback_kind_from_py(kind.as_any())?;
        let out =
            <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::callbacks(
                &self.0, kind,
            );
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (name))]
    fn hook(&self, name: ::std::string::String) -> PyResult<AgentConfigHookOutput> {
        crate::py_calls::check_pid()?;
        let out = <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::hook(
            &self.0, name,
        );
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (overridden))]
    fn has_enter(&self, overridden: bool) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out =
            <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::has_enter(
                &self.0, overridden,
            );
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn dynamic_instructions(&self) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::dynamic_instructions(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn validated_output(&self) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::validated_output(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (retries))]
    fn configuration<'py>(
        &self,
        py: Python<'py>,
        retries: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let retries =
            retries.map(|v| crate::py_values::value_from_py(v.as_any(), "retries")).transpose()?;
        let out = <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::configuration(&self.0, retries).map_err(|e: ::lk_agents_abi::agent::authoring::AuthoringError| <::lk_agents_abi::agent::authoring::AuthoringError as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::generated::records::agent_authoring::agent_configuration_to_py(py, out)
    }

    #[pyo3(signature = (given))]
    fn check_tool_options(&self, given: ::std::vec::Vec<::std::string::String>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::check_tool_options(&self.0, given).map_err(|e: ::lk_agents_abi::agent::authoring::AuthoringError| <::lk_agents_abi::agent::authoring::AuthoringError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (parts))]
    fn join_instructions(
        &self,
        parts: ::std::vec::Vec<::std::string::String>,
    ) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::join_instructions(&self.0, parts);
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn retries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = <CoreAgentConfigImpl as ::lk_agents_abi::agent::authoring::AgentConfig>::retries(
            &self.0,
        );
        crate::generated::records::agent_run::retries_to_py(py, out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/resources.rs.
type CoreRunResourcesImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RunResources;
type RunResourcesThreadIdOutput = ::std::option::Option<::std::string::String>;

#[pyclass(frozen, name = "_core_RunResources", module = "livekit.agents._core")]
pub struct CoreRunResources(pub(crate) ::std::sync::Arc<CoreRunResourcesImpl>);

#[pymethods]
impl CoreRunResources {
    #[new]
    #[pyo3(signature = (namespace, thread_id, activation, lane, toolsets))]
    fn new(
        py: Python<'_>,
        namespace: Option<Bound<'_, PyAny>>,
        thread_id: ::std::option::Option<::std::string::String>,
        activation: Option<Bound<'_, PyAny>>,
        lane: ::std::string::String,
        toolsets: Bound<'_, PyAny>,
    ) -> PyResult<CoreRunResources> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let namespace = namespace
            .map(|v| {
                v.cast::<super::storage::Storage>()
                    .map(|value| ::std::sync::Arc::clone(&value.get().0))
                    .map_err(::pyo3::PyErr::from)
            })
            .transpose()?;
        let activation = activation
            .map(|v| {
                v.cast::<super::storage::CoreContextActivation>()
                    .map(|value| value.get().0.as_ref().clone())
                    .map_err(::pyo3::PyErr::from)
            })
            .transpose()?;
        let toolsets = toolsets.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::llm::ToolsetHooksProxy::from_host(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = py.detach(|| <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::new(&env, namespace, thread_id, activation, lane, toolsets)).map_err(|e: ::lk_agents_abi::agent::resources::ResourceError| <::lk_agents_abi::agent::resources::ResourceError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(CoreRunResources(::std::sync::Arc::new(out)))
    }

    /// Open storage and toolsets before any run starts. Cancelling the caller requests cleanup;
    /// the native owner retains every operation until it settles.
    fn open<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::open(&self.0, )).map_err(|e: ::lk_agents_abi::agent::resources::ResourceError| <::lk_agents_abi::agent::resources::ResourceError as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let () = call.await.map_err(|e: ::lk_agents_abi::agent::resources::ResourceError| <::lk_agents_abi::agent::resources::ResourceError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[getter]
    fn namespace(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::storage::Storage>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::namespace(&self.0)).map_err(|e: ::lk_agents_abi::agent::resources::ResourceError| <::lk_agents_abi::agent::resources::ResourceError as crate::py_errors::HostException>::to_py_err(&e))?;
        out.map(|value| ::pyo3::PyResult::Ok(super::storage::Storage(value))).transpose()
    }

    #[getter]
    fn thread_id(&self, py: Python<'_>) -> PyResult<RunResourcesThreadIdOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::thread_id(&self.0)).map_err(|e: ::lk_agents_abi::agent::resources::ResourceError| <::lk_agents_abi::agent::resources::ResourceError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn activation(&self, py: Python<'_>) -> PyResult<super::storage::CoreContextActivation> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::activation(&self.0)).map_err(|e: ::lk_agents_abi::agent::resources::ResourceError| <::lk_agents_abi::agent::resources::ResourceError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(super::storage::CoreContextActivation(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::closed(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (hydration))]
    fn retain_hydration(&self, py: Python<'_>, hydration: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let hydration =
            crate::generated::agent::ContextHydrationProxy::from_host(hydration.as_any())?;
        py.detach(|| <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::retain_hydration(&self.0, hydration));
        Ok(())
    }

    /// One owned cleanup; cancelled or repeated waiters can rejoin it.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| {
            <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::aclose(
                &self.0,
            )
        });
        crate::py_calls::future_into_py(py, async move {
            let () = call.await.map_err(|e: ::lk_agents_abi::agent::resources::ResourceError| <::lk_agents_abi::agent::resources::ResourceError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Settle failed host-side initialization without replacing its original exception with a
    /// cleanup failure. Cleanup failures remain logged by the native owner.
    fn rollback<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| {
            <CoreRunResourcesImpl as ::lk_agents_abi::agent::resources::RunResources>::rollback(
                &self.0,
            )
        });
        crate::py_calls::future_into_py(py, async move {
            let () = call.await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
type AgentRunnerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AgentRunner;

fn _extract_agent_runner_progress_operation(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AgentRunnerProgressOperationInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "operation")
}
type AgentRunnerProgressOperationInput = u64;

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
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let definition =
            crate::generated::records::agent_run::agent_definition_from_py(definition.as_any())?;
        let llm = crate::generated::llm::LlmProxy::from_host(llm.as_any())?;
        let host = crate::generated::agent::AgentCallbacksProxy::from_host(host.as_any())?;
        let out = py.detach(|| <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::new(&env, definition, llm, host)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(AgentRunner(::std::sync::Arc::new(out)))
    }

    /// Starts one invocation; nothing runs until the first pull.
    #[pyo3(signature = (request))]
    fn run(&self, py: Python<'_>, request: Bound<'_, PyAny>) -> PyResult<AgentRunUpdates> {
        crate::py_calls::check_pid()?;
        let request = crate::generated::records::agent_run::run_request_from_py(request.as_any())?;
        let out = py.detach(|| <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::run(&self.0, request)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(AgentRunUpdates(::lk_agents_runtime::net::source::SharedStream::new(
            ::std::boxed::Box::new(out),
            ::lk_agents_runtime::process::host_clock(),
        )))
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
        let request = crate::generated::records::agent_run::run_request_from_py(request.as_any())?;
        let stream = stream
            .cast::<super::agent::CoreRunStream>()
            .map(|value| ::std::sync::Arc::clone(&value.get().0))
            .map_err(::pyo3::PyErr::from)?;
        let resources = resources
            .cast::<super::agent::CoreRunResources>()
            .map(|value| ::std::sync::Arc::clone(&value.get().0))
            .map_err(::pyo3::PyErr::from)?;
        py.detach(|| {
            <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::run_stream(
                &self.0, request, stream, resources,
            )
        })
        .map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| {
            <::lk_agents_abi::agent::run::InvalidRun as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
        Ok(())
    }

    /// The latest usage snapshot of invocation `invocation_id`; never advances it.
    #[pyo3(signature = (invocation_id))]
    fn usage<'py>(
        &self,
        py: Python<'py>,
        invocation_id: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::usage(&self.0, &invocation_id)).map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| <::lk_agents_abi::agent::run::InvalidRun as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::generated::records::usage::usage_to_py(py, out)
    }

    /// `ctx.update(message)` from the tool body running as host `operation` of invocation
    /// `invocation_id`: the run streams it as progress. Never waits; an update racing its call's
    /// end is dropped. Refused for an invocation this runner is not running.
    #[pyo3(signature = (invocation_id, operation, message))]
    fn progress(
        &self,
        py: Python<'_>,
        invocation_id: ::std::string::String,
        #[pyo3(from_py_with = _extract_agent_runner_progress_operation)]
        operation: AgentRunnerProgressOperationInput,
        message: ::std::string::String,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AgentRunnerImpl as ::lk_agents_abi::agent::run::AgentRunner>::progress(
                &self.0,
                &invocation_id,
                operation,
                message,
            )
        })
        .map_err(|e: ::lk_agents_abi::agent::run::InvalidRun| {
            <::lk_agents_abi::agent::run::InvalidRun as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
        Ok(())
    }
}

/// One stream of `::lk_agents_abi::agent::run::RunUpdate` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "AgentRunUpdates", module = "livekit.agents._core")]
pub struct AgentRunUpdates(
    ::lk_agents_runtime::net::source::SharedStream<::lk_agents_abi::agent::run::RunUpdate>,
);

#[pymethods]
impl AgentRunUpdates {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::recv_native(&self.0, py, |py, item| {
            Ok(crate::generated::records::agent_run::run_update_to_py(py, item)?.unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::close_native(&self.0, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/stream.rs.
type CoreRunStreamImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RunStream;

fn _extract_run_stream_end_result_consumer(
    obj: &Bound<'_, PyAny>,
) -> PyResult<RunStreamEndResultConsumerInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "consumer")
}
type RunStreamEndResultConsumerInput = u64;

fn _extract_run_stream_result_consumer(
    obj: &Bound<'_, PyAny>,
) -> PyResult<RunStreamResultConsumerInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "consumer")
}
type RunStreamResultConsumerInput = u64;

#[pyclass(frozen, name = "_core_RunStream", module = "livekit.agents._core")]
pub struct CoreRunStream(pub(crate) ::std::sync::Arc<CoreRunStreamImpl>);

#[pymethods]
impl CoreRunStream {
    #[new]
    fn new(py: Python<'_>) -> PyResult<CoreRunStream> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py
            .detach(|| <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::new(&env));
        ::pyo3::PyResult::Ok(CoreRunStream(::std::sync::Arc::new(out)))
    }

    fn enter(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::enter(&self.0)).map_err(|e: ::lk_agents_abi::agent::stream::StreamRefused| <::lk_agents_abi::agent::stream::StreamRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn events(&self, py: Python<'_>) -> PyResult<super::agent::CoreStreamEvents> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::events(&self.0)).map_err(|e: ::lk_agents_abi::agent::stream::StreamRefused| <::lk_agents_abi::agent::stream::StreamRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(super::agent::CoreStreamEvents(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (delta))]
    fn text(&self, py: Python<'_>, delta: bool) -> PyResult<super::agent::CoreStreamText> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::text(&self.0, delta)).map_err(|e: ::lk_agents_abi::agent::stream::StreamRefused| <::lk_agents_abi::agent::stream::StreamRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(super::agent::CoreStreamText(::std::sync::Arc::new(out)))
    }

    fn begin_result(&self, py: Python<'_>) -> PyResult<u64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::begin_result(&self.0)).map_err(|e: ::lk_agents_abi::agent::stream::StreamRefused| <::lk_agents_abi::agent::stream::StreamRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (consumer, hydrated))]
    fn end_result(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_run_stream_end_result_consumer)]
        consumer: RunStreamEndResultConsumerInput,
        hydrated: bool,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::end_result(
                &self.0, consumer, hydrated,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (consumer))]
    fn result<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_run_stream_result_consumer)]
        consumer: RunStreamResultConsumerInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| {
            <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::result(
                &self.0, consumer,
            )
        });
        crate::py_calls::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::agent::stream::StreamError| <::lk_agents_abi::agent::stream::StreamError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::run_stream_events::stream_outcome_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::cancel(&self.0)
        });
        Ok(())
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| {
            <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::aclose(&self.0)
        });
        crate::py_calls::future_into_py(py, async move {
            let () = call.await.map_err(|e: ::lk_agents_abi::agent::stream::StreamError| <::lk_agents_abi::agent::stream::StreamError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn rollback<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| {
            <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::rollback(&self.0)
        });
        crate::py_calls::future_into_py(py, async move {
            let () = call.await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[getter]
    fn done(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::done(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn cancelled(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::cancelled(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreRunStreamImpl as ::lk_agents_abi::agent::stream::RunStream>::usage(&self.0)
        });
        crate::generated::records::usage::usage_to_py(py, out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/stream.rs.
type CoreStreamEventsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::StreamEvents;

#[pyclass(frozen, name = "_core_StreamEvents", module = "livekit.agents._core")]
pub struct CoreStreamEvents(pub(crate) ::std::sync::Arc<CoreStreamEventsImpl>);

#[pymethods]
impl CoreStreamEvents {
    fn begin_read(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreStreamEventsImpl as ::lk_agents_abi::agent::stream::StreamEvents>::begin_read(&self.0)).map_err(|e: ::lk_agents_abi::agent::stream::StreamError| <::lk_agents_abi::agent::stream::StreamError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (accepted))]
    fn end_read(&self, py: Python<'_>, accepted: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <CoreStreamEventsImpl as ::lk_agents_abi::agent::stream::StreamEvents>::end_read(
                &self.0, accepted,
            )
        });
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreStreamEventsImpl as ::lk_agents_abi::agent::stream::StreamEvents>::recv(&this, )
                .await.map_err(|e: ::lk_agents_abi::agent::stream::StreamError| <::lk_agents_abi::agent::stream::StreamError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        crate::generated::records::run_stream_events::presented_event_to_py(
                            py, value,
                        )
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreStreamEventsImpl as ::lk_agents_abi::agent::stream::StreamEvents>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::agent::stream::StreamError| <::lk_agents_abi::agent::stream::StreamError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/stream.rs.
type CoreStreamTextImpl =
    <crate::generated::implementations::Selected as super::Implementations>::StreamText;

#[pyclass(frozen, name = "_core_StreamText", module = "livekit.agents._core")]
pub struct CoreStreamText(pub(crate) ::std::sync::Arc<CoreStreamTextImpl>);

#[pymethods]
impl CoreStreamText {
    fn begin_read(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreStreamTextImpl as ::lk_agents_abi::agent::stream::StreamText>::begin_read(&self.0)).map_err(|e: ::lk_agents_abi::agent::stream::StreamError| <::lk_agents_abi::agent::stream::StreamError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (accepted))]
    fn end_read(&self, py: Python<'_>, accepted: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <CoreStreamTextImpl as ::lk_agents_abi::agent::stream::StreamText>::end_read(
                &self.0, accepted,
            )
        });
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreStreamTextImpl as ::lk_agents_abi::agent::stream::StreamText>::recv(&this, )
                .await.map_err(|e: ::lk_agents_abi::agent::stream::StreamError| <::lk_agents_abi::agent::stream::StreamError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreStreamTextImpl as ::lk_agents_abi::agent::stream::StreamText>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::agent::stream::StreamError| <::lk_agents_abi::agent::stream::StreamError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/mod.rs.

/// [`::lk_agents_abi::agent::AgentCallbacks`] over a host `livekit.agents.voice.agent_host.AgentCallbacks`.
#[derive(Clone)]
pub struct AgentCallbacksProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The `notify` calls, delivered in order.
    notifier: crate::py_calls::Notifier,
}

impl AgentCallbacksProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::AgentCallbacks>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::agent::AgentCallbacks>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::agent::AgentCallbacks>(obj)
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
            &crate::py_calls::Entry,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::agent_run::function_tool_info_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
            "AgentCallbacks.prepare_tool",
            invocation,
            &[::lk_agents_abi::invoke::Scope::ToolCall],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    crate::py_values::extra_to_py(py, &arguments_arg)?.into_any(),
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
                        if e.is_instance(py, &py.import("livekit.agents.voice.agent_host").and_then(|m| m.getattr("ToolArgumentError")).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?) {
                        return Ok(Err(<::lk_agents_abi::agent::ToolArgumentError as ::std::convert::From<String>>::from(e.value(py).to_string())));
                    }
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::generated::records::agent_run::prepared_arguments_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
            "AgentCallbacks.read_back",
            invocation,
            &[::lk_agents_abi::invoke::Scope::ToolCall],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    crate::py_objects::object_ref_to_py(py, prepared_arg)?,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::generated::records::agent_run::read_back_answer_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
            "AgentCallbacks.run_tool",
            invocation,
            &[::lk_agents_abi::invoke::Scope::ToolCall],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    crate::py_objects::object_ref_to_py(py, prepared_arg)?,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "AgentCallbacks.run_tool"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((crate::generated::records::tool_report::tool_report_from_py(item_0.as_any()))?, (match crate::py_values::value_from_py(item_1.as_any(), "AgentCallbacks.run_tool")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("AgentCallbacks.run_tool", ": expected a dict"))) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn validate_tool_completion<'a>(
        &'a self,
        invocation: &::lk_agents_abi::agent::Invocation,
        prepared: &::lk_agents_abi::HostRef,
        report: ::lk_agents_abi::llm::tool_executor::ToolReport,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::lk_agents_abi::llm::tool_executor::ToolReport,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let report_arg = report;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AgentCallbacks.validate_tool_completion",
            invocation,
            &[::lk_agents_abi::invoke::Scope::ToolCall],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    crate::generated::records::agent_run::invocation_to_py(py, invocation_arg)?,
                    crate::py_objects::object_ref_to_py(py, prepared_arg)?,
                    crate::generated::records::tool_report::tool_report_to_py(py, report_arg)?,
                ];
                Ok((
                    host.bind(py).getattr("validate_tool_completion")?,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::generated::records::tool_report::tool_report_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::generated::records::agent_run::validated_output_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (o.extract::<::std::string::String>()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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

// Declared in rust/crates/lk-agents-abi/src/agent/mod.rs.

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
            crate::py_objects::native::<dyn ::lk_agents_abi::agent::HostInstructions>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::agent::HostInstructions>(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
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
            &crate::py_calls::Entry,
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
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/resources.rs.

/// [`::lk_agents_abi::agent::resources::ContextHydration`] over a host `livekit.agents.storage.context_store.ContextStore`.
#[derive(Clone)]
pub struct ContextHydrationProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl ContextHydrationProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::resources::ContextHydration>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::agent::resources::ContextHydration,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::agent::resources::ContextHydration>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::agent::resources::ContextHydration,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::agent::resources::ContextHydration>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(ContextHydrationProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::agent::resources::ContextHydration for ContextHydrationProxy {
    fn release(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_release_hydration")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::agent::assertions::AssertionFailure {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyAssertionError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::agent::authoring::AuthoringError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Retries(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::Kind(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::BuiltToolOptions(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::agent::resources::ResourceError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Storage(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Host(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::agent::run::InvalidRun {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::agent::stream::StreamRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::agent::stream::StreamError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Refused(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Start(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Cancelled => crate::py_errors::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Provider(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Resources(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}
