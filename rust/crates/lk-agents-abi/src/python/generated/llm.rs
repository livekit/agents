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
