//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `llm` records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::mcp::McpCleanupStatus {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Pending => "pending",
            Self::Clean => "clean",
            Self::Incomplete => "incomplete",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
static MCP_CLEANUP: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::mcp::McpCleanup {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { status, service_settled, calls_settled, reader_settled, child_exit_code, child_exit_signal, child_reaped } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("status", ::pyo3::IntoPyObjectExt::into_bound_py_any(status, py)?)?;
        kwargs.set_item("service_settled", ::pyo3::IntoPyObjectExt::into_bound_py_any(service_settled, py)?)?;
        kwargs.set_item("calls_settled", ::pyo3::IntoPyObjectExt::into_bound_py_any(calls_settled, py)?)?;
        kwargs.set_item("reader_settled", ::pyo3::IntoPyObjectExt::into_bound_py_any(reader_settled, py)?)?;
        kwargs.set_item("child_exit_code", match child_exit_code { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("child_exit_signal", match child_exit_signal { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("child_reaped", ::pyo3::IntoPyObjectExt::into_bound_py_any(child_reaped, py)?)?;
        MCP_CLEANUP.import(py, "livekit.agents._generated.mcp_session", "McpCleanup")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
static MCP_INVOCATION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::mcp::McpInvocation {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { host_context } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("host_context", match host_context { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        MCP_INVOCATION.import(py, "livekit.agents._generated.mcp_session", "McpInvocation")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
static REALTIME_JSON_OPTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::llm::realtime_native::RealtimeJsonOption {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(REALTIME_JSON_OPTION.import(py, "livekit.agents._generated.realtime_options", "RealtimeJsonOption")?)? {
            return Err(super::not_a(obj, "RealtimeJsonOption"));
        }
        Ok(Self {
            value: ::lk_agents_abi::python::attr_with(obj, "RealtimeJsonOption", "value", |v| ::lk_agents_abi::python::value_from_py(v, "RealtimeJsonOption.value"))?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::realtime_native::RealtimeJsonOption {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { value } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("value", ::lk_agents_abi::python::value_to_py(py, &value)?)?;
        REALTIME_JSON_OPTION.import(py, "livekit.agents._generated.realtime_options", "RealtimeJsonOption")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
static REALTIME_TOOL_CHOICE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::llm::realtime_native::RealtimeToolChoice {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(REALTIME_TOOL_CHOICE.import(py, "livekit.agents._generated.realtime_options", "RealtimeToolChoice")?)? {
            return Err(super::not_a(obj, "RealtimeToolChoice"));
        }
        Ok(Self {
            value: ::lk_agents_abi::python::attr_with(obj, "RealtimeToolChoice", "value", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::ToolChoice>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::realtime_native::RealtimeToolChoice {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { value } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("value", match value { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        REALTIME_TOOL_CHOICE.import(py, "livekit.agents._generated.realtime_options", "RealtimeToolChoice")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
static REALTIME_DURATION_OPTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::llm::realtime_native::RealtimeDurationOption {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(REALTIME_DURATION_OPTION.import(py, "livekit.agents._generated.realtime_options", "RealtimeDurationOption")?)? {
            return Err(super::not_a(obj, "RealtimeDurationOption"));
        }
        Ok(Self {
            value: ::lk_agents_abi::python::attr_with(obj, "RealtimeDurationOption", "value", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeDurationOption.value")).map(Some) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::realtime_native::RealtimeDurationOption {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { value } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("value", match value { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        REALTIME_DURATION_OPTION.import(py, "livekit.agents._generated.realtime_options", "RealtimeDurationOption")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
static REALTIME_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::llm::realtime_native::RealtimeOptions {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(REALTIME_OPTIONS.import(py, "livekit.agents._generated.realtime_options", "RealtimeOptions")?)? {
            return Err(super::not_a(obj, "RealtimeOptions"));
        }
        Ok(Self {
            input_audio_transcription: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "input_audio_transcription", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeJsonOption>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            input_audio_noise_reduction: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "input_audio_noise_reduction", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeJsonOption>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            turn_detection: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "turn_detection", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeJsonOption>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            tool_choice: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "tool_choice", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeToolChoice>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            speed: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "speed", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeOptions.speed")).map(Some) })?,
            tracing: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "tracing", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeJsonOption>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            truncation: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "truncation", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeJsonOption>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            reasoning: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "reasoning", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeJsonOption>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            max_session_duration: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "max_session_duration", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeDurationOption>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            voice: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "voice", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            max_response_output_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "max_response_output_tokens", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeJsonOption>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::realtime_native::RealtimeOptions {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { input_audio_transcription, input_audio_noise_reduction, turn_detection, tool_choice, speed, tracing, truncation, reasoning, max_session_duration, voice, max_response_output_tokens } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("input_audio_transcription", match input_audio_transcription { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("input_audio_noise_reduction", match input_audio_noise_reduction { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("turn_detection", match turn_detection { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("tool_choice", match tool_choice { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("speed", match speed { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("tracing", match tracing { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("truncation", match truncation { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("reasoning", match reasoning { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("max_session_duration", match max_session_duration { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("voice", match voice { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("max_response_output_tokens", match max_response_output_tokens { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        REALTIME_OPTIONS.import(py, "livekit.agents._generated.realtime_options", "RealtimeOptions")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
static NATIVE_REALTIME_CONFIG: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::llm::realtime_native::NativeRealtimeConfig {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(NATIVE_REALTIME_CONFIG.import(py, "livekit.agents._generated.realtime_options", "NativeRealtimeConfig")?)? {
            return Err(super::not_a(obj, "NativeRealtimeConfig"));
        }
        Ok(Self {
            api_key: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "api_key", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            base_url: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "base_url", |v| v.extract::<::std::string::String>())?,
            conn_options: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "conn_options", |v| v.extract::<::lk_agents_abi::ApiConnectOptions>().map_err(::std::convert::Into::<PyErr>::into))?,
            options: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "options", |v| v.extract::<::lk_agents_abi::llm::realtime_native::RealtimeOptions>().map_err(::std::convert::Into::<PyErr>::into))?,
            dialect: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "dialect", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            model: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "model", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            voice: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "voice", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            modalities: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "modalities", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
            azure_deployment: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "azure_deployment", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            entra_token: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "entra_token", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            azure_defaults: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "azure_defaults", |v| v.extract::<bool>())?,
            api_secret: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "api_secret", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            inference_provider: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "inference_provider", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            inference_class: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "inference_class", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            label: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "label", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::realtime_native::NativeRealtimeConfig {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { api_key, base_url, conn_options, options, dialect, model, voice, modalities, azure_deployment, entra_token, azure_defaults, api_secret, inference_provider, inference_class, label } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("api_key", match api_key { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("base_url", ::pyo3::IntoPyObjectExt::into_bound_py_any(base_url, py)?)?;
        kwargs.set_item("conn_options", ::pyo3::IntoPyObjectExt::into_bound_py_any(conn_options, py)?)?;
        kwargs.set_item("options", ::pyo3::IntoPyObjectExt::into_bound_py_any(options, py)?)?;
        kwargs.set_item("dialect", match dialect { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("model", match model { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("voice", match voice { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("modalities", match modalities { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
        kwargs.set_item("azure_deployment", match azure_deployment { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("entra_token", match entra_token { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("azure_defaults", ::pyo3::IntoPyObjectExt::into_bound_py_any(azure_defaults, py)?)?;
        kwargs.set_item("api_secret", match api_secret { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("inference_provider", match inference_provider { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("inference_class", match inference_class { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("label", match label { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        NATIVE_REALTIME_CONFIG.import(py, "livekit.agents._generated.realtime_options", "NativeRealtimeConfig")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/tool_executor.rs.
static TOOL_OUTCOME_VALUE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_NOTHING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_TOOL_ERROR: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_MODEL_RETRY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_STOP_RESPONSE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_INTERNAL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_INVALID: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_CANCELLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_UPDATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_COMPLETED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_FAILED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_REJECTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_INPUT_REQUIRED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOOL_OUTCOME_WORKING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::llm::tool_executor::ToolOutcome {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(TOOL_OUTCOME_VALUE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeValue")?)? {
            return Ok(Self::Value { text: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeValue", "text", |v| v.extract::<::std::string::String>())?, reply_required: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeValue", "reply_required", |v| v.extract::<bool>())?, verbatim: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeValue", "verbatim", |v| v.extract::<bool>())?, more: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeValue", "more", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
        }
        if obj.is_instance(TOOL_OUTCOME_NOTHING.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeNothing")?)? {
            return Ok(Self::Nothing);
        }
        if obj.is_instance(TOOL_OUTCOME_TOOL_ERROR.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeToolError")?)? {
            return Ok(Self::ToolError(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeToolError", "message", |v| v.extract::<::std::string::String>())?));
        }
        if obj.is_instance(TOOL_OUTCOME_MODEL_RETRY.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeModelRetry")?)? {
            return Ok(Self::ModelRetry(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeModelRetry", "message", |v| v.extract::<::std::string::String>())?));
        }
        if obj.is_instance(TOOL_OUTCOME_STOP_RESPONSE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeStopResponse")?)? {
            return Ok(Self::StopResponse);
        }
        if obj.is_instance(TOOL_OUTCOME_INTERNAL.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInternal")?)? {
            return Ok(Self::Internal);
        }
        if obj.is_instance(TOOL_OUTCOME_INVALID.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInvalid")?)? {
            return Ok(Self::Invalid);
        }
        if obj.is_instance(TOOL_OUTCOME_CANCELLED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeCancelled")?)? {
            return Ok(Self::Cancelled);
        }
        if obj.is_instance(TOOL_OUTCOME_UPDATE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeUpdate")?)? {
            return Ok(Self::Update { message: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeUpdate", "message", |v| v.extract::<::std::string::String>())?, raw: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeUpdate", "raw", |v| v.extract::<::std::string::String>())? });
        }
        if obj.is_instance(TOOL_OUTCOME_COMPLETED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeCompleted")?)? {
            return Ok(Self::Completed { text: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeCompleted", "text", |v| v.extract::<::std::string::String>())?, verbatim: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeCompleted", "verbatim", |v| v.extract::<bool>())? });
        }
        if obj.is_instance(TOOL_OUTCOME_FAILED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeFailed")?)? {
            return Ok(Self::Failed(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeFailed", "error", |v| v.extract::<::std::string::String>())?));
        }
        if obj.is_instance(TOOL_OUTCOME_REJECTED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeRejected")?)? {
            return Ok(Self::Rejected(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeRejected", "reason", |v| v.extract::<::std::string::String>())?));
        }
        if obj.is_instance(TOOL_OUTCOME_INPUT_REQUIRED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInputRequired")?)? {
            return Ok(Self::InputRequired { question: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeInputRequired", "question", |v| v.extract::<::std::string::String>())?, verbatim: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeInputRequired", "verbatim", |v| v.extract::<bool>())?, tool: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeInputRequired", "tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
        }
        if obj.is_instance(TOOL_OUTCOME_WORKING.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeWorking")?)? {
            return Ok(Self::Working(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeWorking", "message", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?));
        }
        Err(super::not_a(obj, "ToolOutcomeValue | ToolOutcomeNothing | ToolOutcomeToolError | ToolOutcomeModelRetry | ToolOutcomeStopResponse | ToolOutcomeInternal | ToolOutcomeInvalid | ToolOutcomeCancelled | ToolOutcomeUpdate | ToolOutcomeCompleted | ToolOutcomeFailed | ToolOutcomeRejected | ToolOutcomeInputRequired | ToolOutcomeWorking"))
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::tool_executor::ToolOutcome {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Value { text, reply_required, verbatim, more } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                kwargs.set_item("reply_required", ::pyo3::IntoPyObjectExt::into_bound_py_any(reply_required, py)?)?;
                kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
                kwargs.set_item("more", match more { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                TOOL_OUTCOME_VALUE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeValue")?.call((), Some(&kwargs))
            },
            Self::Nothing => TOOL_OUTCOME_NOTHING.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeNothing")?.call0(),
            Self::ToolError(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                TOOL_OUTCOME_TOOL_ERROR.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeToolError")?.call((), Some(&kwargs))
            },
            Self::ModelRetry(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                TOOL_OUTCOME_MODEL_RETRY.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeModelRetry")?.call((), Some(&kwargs))
            },
            Self::StopResponse => TOOL_OUTCOME_STOP_RESPONSE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeStopResponse")?.call0(),
            Self::Internal => TOOL_OUTCOME_INTERNAL.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInternal")?.call0(),
            Self::Invalid => TOOL_OUTCOME_INVALID.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInvalid")?.call0(),
            Self::Cancelled => TOOL_OUTCOME_CANCELLED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeCancelled")?.call0(),
            Self::Update { message, raw } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                kwargs.set_item("raw", ::pyo3::IntoPyObjectExt::into_bound_py_any(raw, py)?)?;
                TOOL_OUTCOME_UPDATE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeUpdate")?.call((), Some(&kwargs))
            },
            Self::Completed { text, verbatim } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
                TOOL_OUTCOME_COMPLETED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeCompleted")?.call((), Some(&kwargs))
            },
            Self::Failed(error) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
                TOOL_OUTCOME_FAILED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeFailed")?.call((), Some(&kwargs))
            },
            Self::Rejected(reason) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("reason", ::pyo3::IntoPyObjectExt::into_bound_py_any(reason, py)?)?;
                TOOL_OUTCOME_REJECTED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeRejected")?.call((), Some(&kwargs))
            },
            Self::InputRequired { question, verbatim, tool } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
                kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
                kwargs.set_item("tool", match tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                TOOL_OUTCOME_INPUT_REQUIRED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInputRequired")?.call((), Some(&kwargs))
            },
            Self::Working(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", match message { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                TOOL_OUTCOME_WORKING.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeWorking")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/tool_executor.rs.

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::llm::tool_executor::Approval {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
        match &*value {
            "asked" => Ok(Self::Asked),
            "granted" => Ok(Self::Granted),
            other => Err(::pyo3::exceptions::PyValueError::new_err(format!("Approval: {other:?} is not one of 'asked', 'granted'"))),
        }
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::llm::tool_executor::Approval {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Asked => "asked",
            Self::Granted => "granted",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/tool_executor.rs.
static TOOL_REPORT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::llm::tool_executor::ToolReport {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(TOOL_REPORT.import(py, "livekit.agents._generated.tool_report", "ToolReport")?)? {
            return Err(super::not_a(obj, "ToolReport"));
        }
        Ok(Self {
            outcome: ::lk_agents_abi::python::attr_with(obj, "ToolReport", "outcome", |v| v.extract::<::lk_agents_abi::llm::tool_executor::ToolOutcome>().map_err(::std::convert::Into::<PyErr>::into))?,
            approval: ::lk_agents_abi::python::attr_with(obj, "ToolReport", "approval", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::llm::tool_executor::Approval>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            transition: ::lk_agents_abi::python::attr_with(obj, "ToolReport", "transition", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/llm.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::Llm {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::Llm>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`Llm` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::llm::mcp::McpTransport {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::llm::mcp::McpTransport>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`McpTransport` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::llm::mcp::McpProgress {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::llm::mcp::McpProgress>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`McpProgress` is a declared proxy: it is taken from the host object, not a view"))
    }
}
