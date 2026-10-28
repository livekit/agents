//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `agent` records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static INVOCATION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::Invocation {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { operation, invocation_id, agent_name, execution_id, frame_id, context_scope, application_context, history, function_call, approval, host_context: _ } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("operation", ::pyo3::IntoPyObjectExt::into_bound_py_any(operation, py)?)?;
        kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
        kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
        kwargs.set_item("execution_id", match execution_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("frame_id", match frame_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("context_scope", match context_scope { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("application_context", match application_context { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("function_call", match function_call { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("approval", match approval { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        INVOCATION.import(py, "livekit.agents._generated.agent_run", "Invocation")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static ARGUMENT_IDENTITY_JSON: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static ARGUMENT_IDENTITY_FINGERPRINT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::ArgumentIdentity {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(ARGUMENT_IDENTITY_JSON.import(py, "livekit.agents._generated.agent_run", "ArgumentIdentityJson")?)? {
            return Ok(Self::Json(::lk_agents_abi::python::attr_with(obj, "ArgumentIdentityJson", "value", |v| ::lk_agents_abi::python::value_from_py(v, "ArgumentIdentityJson.value"))?));
        }
        if obj.is_instance(ARGUMENT_IDENTITY_FINGERPRINT.import(py, "livekit.agents._generated.agent_run", "ArgumentIdentityFingerprint")?)? {
            return Ok(Self::Fingerprint(::lk_agents_abi::python::attr_with(obj, "ArgumentIdentityFingerprint", "value", |v| v.extract::<::std::string::String>())?));
        }
        Err(super::not_a(obj, "ArgumentIdentityJson | ArgumentIdentityFingerprint"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static PREPARED_ARGUMENTS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::PreparedArguments {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(PREPARED_ARGUMENTS.import(py, "livekit.agents._generated.agent_run", "PreparedArguments")?)? {
            return Err(super::not_a(obj, "PreparedArguments"));
        }
        Ok(Self {
            prepared: ::lk_agents_abi::python::attr_with(obj, "PreparedArguments", "prepared", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))?,
            identity: ::lk_agents_abi::python::attr_with(obj, "PreparedArguments", "identity", |v| v.extract::<::lk_agents_abi::agent::ArgumentIdentity>().map_err(::std::convert::Into::<PyErr>::into))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static READ_BACK_ANSWER_QUESTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static READ_BACK_ANSWER_INPUT_REQUIRED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::ReadBackAnswer {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(READ_BACK_ANSWER_QUESTION.import(py, "livekit.agents._generated.agent_run", "ReadBackAnswerQuestion")?)? {
            return Ok(Self::Question(::lk_agents_abi::python::attr_with(obj, "ReadBackAnswerQuestion", "question", |v| v.extract::<::std::string::String>())?));
        }
        if obj.is_instance(READ_BACK_ANSWER_INPUT_REQUIRED.import(py, "livekit.agents._generated.agent_run", "ReadBackAnswerInputRequired")?)? {
            return Ok(Self::InputRequired { question: ::lk_agents_abi::python::attr_with(obj, "ReadBackAnswerInputRequired", "question", |v| v.extract::<::std::string::String>())?, verbatim: ::lk_agents_abi::python::attr_with(obj, "ReadBackAnswerInputRequired", "verbatim", |v| v.extract::<bool>())? });
        }
        Err(super::not_a(obj, "ReadBackAnswerQuestion | ReadBackAnswerInputRequired"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static OUTPUT_CANDIDATE_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static OUTPUT_CANDIDATE_VALUE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::OutputCandidate {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Text(text) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                OUTPUT_CANDIDATE_TEXT.import(py, "livekit.agents._generated.agent_run", "OutputCandidateText")?.call((), Some(&kwargs))
            },
            Self::Value(value) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("value", ::lk_agents_abi::python::value_to_py(py, &value)?)?;
                OUTPUT_CANDIDATE_VALUE.import(py, "livekit.agents._generated.agent_run", "OutputCandidateValue")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static SERIALIZED_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::SerializedOutput {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(SERIALIZED_OUTPUT.import(py, "livekit.agents._generated.agent_run", "SerializedOutput")?)? {
            return Err(super::not_a(obj, "SerializedOutput"));
        }
        Ok(Self {
            value: ::lk_agents_abi::python::attr_with(obj, "SerializedOutput", "value", |v| ::lk_agents_abi::python::value_from_py(v, "SerializedOutput.value"))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static VALIDATED_OUTPUT_ACCEPTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static VALIDATED_OUTPUT_RETRY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static VALIDATED_OUTPUT_FAILED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::ValidatedOutput {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(VALIDATED_OUTPUT_ACCEPTED.import(py, "livekit.agents._generated.agent_run", "ValidatedOutputAccepted")?)? {
            return Ok(Self::Accepted { text: ::lk_agents_abi::python::attr_with(obj, "ValidatedOutputAccepted", "text", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, value: ::lk_agents_abi::python::attr_with(obj, "ValidatedOutputAccepted", "value", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?, serialized: ::lk_agents_abi::python::attr_with(obj, "ValidatedOutputAccepted", "serialized", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::agent::SerializedOutput>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })? });
        }
        if obj.is_instance(VALIDATED_OUTPUT_RETRY.import(py, "livekit.agents._generated.agent_run", "ValidatedOutputRetry")?)? {
            return Ok(Self::Retry(::lk_agents_abi::python::attr_with(obj, "ValidatedOutputRetry", "reason", |v| v.extract::<::std::string::String>())?));
        }
        if obj.is_instance(VALIDATED_OUTPUT_FAILED.import(py, "livekit.agents._generated.agent_run", "ValidatedOutputFailed")?)? {
            return Ok(Self::Failed(::lk_agents_abi::python::attr_with(obj, "ValidatedOutputFailed", "error", |v| v.extract::<::std::string::String>())?));
        }
        Err(super::not_a(obj, "ValidatedOutputAccepted | ValidatedOutputRetry | ValidatedOutputFailed"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static AGENT_DEFINITION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::AgentDefinition {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(AGENT_DEFINITION.import(py, "livekit.agents._generated.agent_run", "AgentDefinition")?)? {
            return Err(super::not_a(obj, "AgentDefinition"));
        }
        Ok(Self {
            name: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "name", |v| v.extract::<::std::string::String>())?,
            instructions: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "instructions", |v| v.extract::<::std::string::String>())?,
            dynamic_instructions: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "dynamic_instructions", |v| v.extract::<bool>())?,
            tools: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "tools", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::agent::FunctionToolInfo>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            output: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "output", |v| v.extract::<::lk_agents_abi::agent::OutputSpec>().map_err(::std::convert::Into::<PyErr>::into))?,
            retries: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "retries", |v| v.extract::<::lk_agents_abi::agent::Retries>().map_err(::std::convert::Into::<PyErr>::into))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static FUNCTION_TOOL_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::FunctionToolInfo {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(FUNCTION_TOOL_INFO.import(py, "livekit.agents._generated.agent_run", "FunctionToolInfo")?)? {
            return Err(super::not_a(obj, "FunctionToolInfo"));
        }
        Ok(Self {
            spec: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "spec", |v| v.extract::<::lk_agents_abi::ToolSpec>().map_err(::std::convert::Into::<PyErr>::into))?,
            raw_schema: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "raw_schema", |v| v.extract::<bool>())?,
            needs_approval: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "needs_approval", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::agent::ReadBack>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            on_duplicate: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "on_duplicate", |v| v.extract::<::lk_agents_abi::llm::mcp::DuplicateMode>().map_err(::std::convert::Into::<PyErr>::into))?,
            duplicate_scope: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "duplicate_scope", |v| v.extract::<::lk_agents_abi::llm::mcp::DuplicateScope>().map_err(::std::convert::Into::<PyErr>::into))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::ReadBack {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
        match &*value {
            "default" => Ok(Self::Default),
            "function" => Ok(Self::Function),
            other => Err(::pyo3::exceptions::PyValueError::new_err(format!("ReadBack: {other:?} is not one of 'default', 'function'"))),
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static OUTPUT_SPEC_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static OUTPUT_SPEC_STRUCTURED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::OutputSpec {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(OUTPUT_SPEC_TEXT.import(py, "livekit.agents._generated.agent_run", "OutputSpecText")?)? {
            return Ok(Self::Text { validated: ::lk_agents_abi::python::attr_with(obj, "OutputSpecText", "validated", |v| v.extract::<bool>())? });
        }
        if obj.is_instance(OUTPUT_SPEC_STRUCTURED.import(py, "livekit.agents._generated.agent_run", "OutputSpecStructured")?)? {
            return Ok(Self::Structured { schema: ::lk_agents_abi::python::attr_with(obj, "OutputSpecStructured", "schema", |v| ::lk_agents_abi::python::value_from_py(v, "OutputSpecStructured.schema"))? });
        }
        Err(super::not_a(obj, "OutputSpecText | OutputSpecStructured"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
static RETRIES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::Retries {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(RETRIES.import(py, "livekit.agents._generated.agent_run", "Retries")?)? {
            return Err(super::not_a(obj, "Retries"));
        }
        Ok(Self {
            tool: ::lk_agents_abi::python::attr_with(obj, "Retries", "tool", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "Retries.tool"))?,
            output: ::lk_agents_abi::python::attr_with(obj, "Retries", "output", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "Retries.output"))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/assertions.rs.
static OBSERVED_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::assertions::ObservedEvent {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(OBSERVED_EVENT.import(py, "livekit.agents._generated.agent_run", "ObservedEvent")?)? {
            return Err(super::not_a(obj, "ObservedEvent"));
        }
        Ok(Self {
            item: ::lk_agents_abi::python::attr_with(obj, "ObservedEvent", "item", |v| v.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into))?,
            new_agent_types: ::lk_agents_abi::python::attr_with(obj, "ObservedEvent", "new_agent_types", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>())?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/assertions.rs.
static EVENT_EXPECTATION_MESSAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static EVENT_EXPECTATION_FUNCTION_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static EVENT_EXPECTATION_FUNCTION_CALL_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static EVENT_EXPECTATION_AGENT_HANDOFF: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::assertions::EventExpectation {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(EVENT_EXPECTATION_MESSAGE.import(py, "livekit.agents._generated.agent_run", "EventExpectationMessage")?)? {
            return Ok(Self::Message { role: ::lk_agents_abi::python::attr_with(obj, "EventExpectationMessage", "role", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
        }
        if obj.is_instance(EVENT_EXPECTATION_FUNCTION_CALL.import(py, "livekit.agents._generated.agent_run", "EventExpectationFunctionCall")?)? {
            return Ok(Self::FunctionCall { name: ::lk_agents_abi::python::attr_with(obj, "EventExpectationFunctionCall", "name", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, arguments: ::lk_agents_abi::python::attr_with(obj, "EventExpectationFunctionCall", "arguments", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "EventExpectationFunctionCall.arguments")).map(Some) })? });
        }
        if obj.is_instance(EVENT_EXPECTATION_FUNCTION_CALL_OUTPUT.import(py, "livekit.agents._generated.agent_run", "EventExpectationFunctionCallOutput")?)? {
            return Ok(Self::FunctionCallOutput { output: ::lk_agents_abi::python::attr_with(obj, "EventExpectationFunctionCallOutput", "output", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, is_error: ::lk_agents_abi::python::attr_with(obj, "EventExpectationFunctionCallOutput", "is_error", |v| if v.is_none() { Ok(None) } else { (v.extract::<bool>()).map(Some) })? });
        }
        if obj.is_instance(EVENT_EXPECTATION_AGENT_HANDOFF.import(py, "livekit.agents._generated.agent_run", "EventExpectationAgentHandoff")?)? {
            return Ok(Self::AgentHandoff { new_agent_type: ::lk_agents_abi::python::attr_with(obj, "EventExpectationAgentHandoff", "new_agent_type", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
        }
        Err(super::not_a(obj, "EventExpectationMessage | EventExpectationFunctionCall | EventExpectationFunctionCallOutput | EventExpectationAgentHandoff"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static RUN_RESULT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::RunResult {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { outcome, interrupted, new_messages, all_messages, usage, verbatim } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("outcome", ::pyo3::IntoPyObjectExt::into_bound_py_any(outcome, py)?)?;
        kwargs.set_item("interrupted", ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted, py)?)?;
        kwargs.set_item("new_messages", ::pyo3::types::PyList::new(py, new_messages.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("all_messages", ::pyo3::types::PyList::new(py, all_messages.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("usage", ::pyo3::IntoPyObjectExt::into_bound_py_any(usage, py)?)?;
        kwargs.set_item("verbatim", match verbatim { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        RUN_RESULT.import(py, "livekit.agents._generated.agent_run", "RunResult")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static RUN_OUTCOME_COMPLETED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_OUTCOME_INPUT_REQUIRED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::RunOutcome {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Completed { output } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("output", ::pyo3::IntoPyObjectExt::into_bound_py_any(output, py)?)?;
                RUN_OUTCOME_COMPLETED.import(py, "livekit.agents._generated.agent_run", "RunOutcomeCompleted")?.call((), Some(&kwargs))
            },
            Self::InputRequired { question, verbatim, answer_tool } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
                kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
                kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                RUN_OUTCOME_INPUT_REQUIRED.import(py, "livekit.agents._generated.agent_run", "RunOutcomeInputRequired")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static OUTPUT_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static OUTPUT_VALIDATED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static OUTPUT_NOTHING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::Output {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Text(text) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                OUTPUT_TEXT.import(py, "livekit.agents._generated.agent_run", "OutputText")?.call((), Some(&kwargs))
            },
            Self::Validated(_) => OUTPUT_VALIDATED.import(py, "livekit.agents._generated.agent_run", "OutputValidated")?.call0(),
            Self::Nothing => OUTPUT_NOTHING.import(py, "livekit.agents._generated.agent_run", "OutputNothing")?.call0(),
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static RUN_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::RunUsage {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { requests, input_tokens, output_tokens, tool_calls } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("requests", ::pyo3::IntoPyObjectExt::into_bound_py_any(requests, py)?)?;
        kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
        kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
        kwargs.set_item("tool_calls", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_calls, py)?)?;
        RUN_USAGE.import(py, "livekit.agents._generated.agent_run", "RunUsage")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static RUN_ERROR_NO_MODEL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_CANCELLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_MODEL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_HOST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_TOOL_RETRIES_EXCEEDED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_ERROR_OUTPUT_RETRIES_EXCEEDED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::RunError {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::NoModel => RUN_ERROR_NO_MODEL.import(py, "livekit.agents._generated.agent_run", "RunErrorNoModel")?.call0(),
            Self::Cancelled => RUN_ERROR_CANCELLED.import(py, "livekit.agents._generated.agent_run", "RunErrorCancelled")?.call0(),
            Self::Model(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                RUN_ERROR_MODEL.import(py, "livekit.agents._generated.agent_run", "RunErrorModel")?.call((), Some(&kwargs))
            },
            Self::Host(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                RUN_ERROR_HOST.import(py, "livekit.agents._generated.agent_run", "RunErrorHost")?.call((), Some(&kwargs))
            },
            Self::ToolRetriesExceeded { tool, retries } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("tool", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool, py)?)?;
                kwargs.set_item("retries", ::pyo3::IntoPyObjectExt::into_bound_py_any(retries, py)?)?;
                RUN_ERROR_TOOL_RETRIES_EXCEEDED.import(py, "livekit.agents._generated.agent_run", "RunErrorToolRetriesExceeded")?.call((), Some(&kwargs))
            },
            Self::OutputRetriesExceeded { retries } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("retries", ::pyo3::IntoPyObjectExt::into_bound_py_any(retries, py)?)?;
                RUN_ERROR_OUTPUT_RETRIES_EXCEEDED.import(py, "livekit.agents._generated.agent_run", "RunErrorOutputRetriesExceeded")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static STREAM_EVENT_TEXT_DELTA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static STREAM_EVENT_TEXT_END: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static STREAM_EVENT_TOOL_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static STREAM_EVENT_TOOL_RESULT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static STREAM_EVENT_PROGRESS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::StreamEvent {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::TextDelta { message_id, delta } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
                kwargs.set_item("delta", ::pyo3::IntoPyObjectExt::into_bound_py_any(delta, py)?)?;
                STREAM_EVENT_TEXT_DELTA.import(py, "livekit.agents._generated.agent_run", "StreamEventTextDelta")?.call((), Some(&kwargs))
            },
            Self::TextEnd { message_id, interrupted } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
                kwargs.set_item("interrupted", ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted, py)?)?;
                STREAM_EVENT_TEXT_END.import(py, "livekit.agents._generated.agent_run", "StreamEventTextEnd")?.call((), Some(&kwargs))
            },
            Self::ToolCall { call_id, tool_name, arguments } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
                kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
                kwargs.set_item("arguments", ::lk_agents_abi::python::value_to_py(py, &arguments)?)?;
                STREAM_EVENT_TOOL_CALL.import(py, "livekit.agents._generated.agent_run", "StreamEventToolCall")?.call((), Some(&kwargs))
            },
            Self::ToolResult { call_id, tool_name, status, output } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
                kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
                kwargs.set_item("status", ::pyo3::IntoPyObjectExt::into_bound_py_any(status, py)?)?;
                kwargs.set_item("output", ::pyo3::IntoPyObjectExt::into_bound_py_any(output, py)?)?;
                STREAM_EVENT_TOOL_RESULT.import(py, "livekit.agents._generated.agent_run", "StreamEventToolResult")?.call((), Some(&kwargs))
            },
            Self::Progress { call_id, tool_name, message } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
                kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                STREAM_EVENT_PROGRESS.import(py, "livekit.agents._generated.agent_run", "StreamEventProgress")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::ToolResultStatus {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static RUN_REQUEST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::agent::run::RunRequest {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(RUN_REQUEST.import(py, "livekit.agents._generated.agent_run", "RunRequest")?)? {
            return Err(super::not_a(obj, "RunRequest"));
        }
        Ok(Self {
            invocation_id: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "invocation_id", |v| v.extract::<::std::string::String>())?,
            execution_id: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "execution_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            message: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "message", |v| v.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into))?,
            history: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "history", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            answer_tool: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "answer_tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            delegated: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "delegated", |v| v.extract::<bool>())?,
            conn_options: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "conn_options", |v| v.extract::<::lk_agents_abi::ApiConnectOptions>().map_err(::std::convert::Into::<PyErr>::into))?,
            host_context: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "host_context", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static RUN_SETTLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::RunSettled {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { result, output } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("result", ::pyo3::IntoPyObjectExt::into_bound_py_any(result, py)?)?;
        kwargs.set_item("output", match output { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        RUN_SETTLED.import(py, "livekit.agents._generated.agent_run", "RunSettled")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
static RUN_UPDATE_STREAM: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_UPDATE_INCOMPLETE_CLEANUP: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_UPDATE_FINISHED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static RUN_UPDATE_FAILED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::agent::run::RunUpdate {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Stream(event) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("event", ::pyo3::IntoPyObjectExt::into_bound_py_any(event, py)?)?;
                RUN_UPDATE_STREAM.import(py, "livekit.agents._generated.agent_run", "RunUpdateStream")?.call((), Some(&kwargs))
            },
            Self::IncompleteCleanup(pending) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("pending", ::pyo3::types::PyList::new(py, pending.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
                RUN_UPDATE_INCOMPLETE_CLEANUP.import(py, "livekit.agents._generated.agent_run", "RunUpdateIncompleteCleanup")?.call((), Some(&kwargs))
            },
            Self::Finished(settled) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("settled", ::pyo3::IntoPyObjectExt::into_bound_py_any(settled, py)?)?;
                RUN_UPDATE_FINISHED.import(py, "livekit.agents._generated.agent_run", "RunUpdateFinished")?.call((), Some(&kwargs))
            },
            Self::Failed(error) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
                RUN_UPDATE_FAILED.import(py, "livekit.agents._generated.agent_run", "RunUpdateFailed")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::agent::AgentHost {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::agent::AgentHost>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`AgentHost` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/agent.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::agent::HostInstructions {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::agent::HostInstructions>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`HostInstructions` is a declared proxy: it is taken from the host object, not a view"))
    }
}
