//! GENERATED FILE -- DO NOT EDIT.
//! Interpreter interfaces resolved with the binding declarations.
use pyo3::prelude::*;

/// Python interface: `asyncio`.
pub use crate::py_calls::interfaces::AsyncioModule;
impl AsyncioModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("asyncio")
    }
    pub fn get_running_loop<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("get_running_loop")
    }
}

/// Python interface: `contextvars`.
pub use crate::py_calls::interfaces::ContextVarsModule;
impl ContextVarsModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("contextvars")
    }
    pub fn copy_context<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("copy_context")
    }
}

/// Python interface: `inspect`.
pub use crate::py_calls::interfaces::InspectModule;
impl InspectModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("inspect")
    }
    pub fn isawaitable<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("isawaitable")
    }
    pub fn iscoroutine<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("iscoroutine")
    }
    pub fn getattr_static<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("getattr_static")
    }
}

/// Python interface: `contextvars.Context`.
pub use crate::py_calls::interfaces::Context;
impl Context {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "contextvars", "Context")?.as_any().clone())
    }
    pub fn copy<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("copy")
    }
    pub fn run<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("run")
    }
}

/// Python interface: `asyncio.AbstractEventLoop`.
pub use crate::py_calls::interfaces::EventLoop;
impl EventLoop {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "asyncio", "AbstractEventLoop")?.as_any().clone())
    }
    pub fn call_soon_threadsafe<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("call_soon_threadsafe")
    }
    pub fn create_task<'py>(
        obj: &Bound<'py, PyAny>,
        coro: &Bound<'py, PyAny>,
        context: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = obj.py();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("context", context)?;
        crate::py_calls::call(&obj.getattr("create_task")?, (coro,), Some(&kwargs))
    }
}

/// Python interface: `asyncio.Task`.
pub use crate::py_calls::interfaces::Task;
impl Task {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "asyncio", "Task")?.as_any().clone())
    }
    pub fn cancel<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("cancel")
    }
    pub fn result<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("result")
    }
    pub fn add_done_callback<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("add_done_callback")
    }
}

/// Python interface: `lk_agents_host`.
pub use crate::py_calls::interfaces::AwaitableAdapterModule;
impl AwaitableAdapterModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("lk_agents_host")
    }
    pub fn awaiting<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("awaiting")
    }
}

/// Python interface: `traceback`.
pub use crate::py_calls::interfaces::TracebackModule;
impl TracebackModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("traceback")
    }
    pub fn format_exception<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("format_exception")
    }
}

/// Python interface: `livekit.agents.telemetry`.
pub use crate::py_calls::interfaces::TelemetryModule;
impl TelemetryModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("livekit.agents.telemetry")
    }
    pub fn _activate<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("_activate")
    }
    pub fn _current<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("_current")
    }
}

/// Python interface: `builtins.BaseException`.
pub use crate::py_errors::interfaces::BaseException;
impl BaseException {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "builtins", "BaseException")?.as_any().clone())
    }
    pub fn add_note<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("add_note")
    }
    pub fn traceback<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("__traceback__")
    }
}

/// Python interface: `livekit.agents._bindings.APIError`.
pub use crate::py_errors::interfaces::ApiError;
impl ApiError {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "livekit.agents._bindings", "APIError")?.as_any().clone())
    }
}

/// Python interface: `livekit.agents._bindings.APIStatusError`.
pub use crate::py_errors::interfaces::ApiStatusError;
impl ApiStatusError {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "livekit.agents._bindings", "APIStatusError")?.as_any().clone())
    }
}

/// Python interface: `livekit.agents._bindings.APIConnectionError`.
pub use crate::py_errors::interfaces::ApiConnectionError;
impl ApiConnectionError {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "livekit.agents._bindings", "APIConnectionError")?.as_any().clone())
    }
}

/// Python interface: `livekit.agents._bindings.APITimeoutError`.
pub use crate::py_errors::interfaces::ApiTimeoutError;
impl ApiTimeoutError {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "livekit.agents._bindings", "APITimeoutError")?.as_any().clone())
    }
}

/// Python interface: `logging`.
pub use crate::py_logging::interfaces::LoggingModule;
impl LoggingModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("logging")
    }
    pub fn get_logger<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("getLogger")
    }
}

/// Python interface: `logging.Logger`.
pub use crate::py_logging::interfaces::Logger;
impl Logger {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "logging", "Logger")?.as_any().clone())
    }
    pub fn get_effective_level<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("getEffectiveLevel")
    }
    pub fn is_enabled_for<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("isEnabledFor")
    }
    pub fn make_record<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("makeRecord")
    }
    pub fn handle<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("handle")
    }
}

/// Python interface: `logging.LogRecord`.
pub use crate::py_logging::interfaces::LogRecord;
impl LogRecord {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "logging", "LogRecord")?.as_any().clone())
    }
    pub fn created<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("created")
    }
    pub fn set_created<'py, V>(obj: &Bound<'py, PyAny>, value: V) -> PyResult<()>
    where
        V: IntoPyObject<'py>,
    {
        obj.setattr("created", value)
    }
    pub fn msecs<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("msecs")
    }
    pub fn set_msecs<'py, V>(obj: &Bound<'py, PyAny>, value: V) -> PyResult<()>
    where
        V: IntoPyObject<'py>,
    {
        obj.setattr("msecs", value)
    }
}

/// Python interface: `atexit`.
pub use crate::py_logging::interfaces::AtExitModule;
impl AtExitModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("atexit")
    }
    pub fn register<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("register")
    }
}

/// Python interface: `livekit.agents._core`.
pub use crate::py_logging::interfaces::CoreModule;
impl CoreModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("livekit.agents._core")
    }
    pub fn flush_logs<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("flush_logs")
    }
}

/// Python interface: `collections.abc.AsyncIterable`.
pub use crate::py_streams::interfaces::AsyncIterable;
impl AsyncIterable {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "collections.abc", "AsyncIterable")?.as_any().clone())
    }
    pub fn aiter<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("__aiter__")
    }
}

/// Python interface: `collections.abc.AsyncIterator`.
pub use crate::py_streams::interfaces::AsyncIterator;
impl AsyncIterator {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "collections.abc", "AsyncIterator")?.as_any().clone())
    }
    pub fn anext<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("__anext__")
    }
}

/// Python interface: `supports aclose()`.
pub use crate::py_streams::interfaces::AsyncClose;
impl AsyncClose {
    pub fn aclose<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("aclose")
    }
    pub fn matches(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        obj.hasattr("aclose")
    }
}

/// Python interface: `framework host stream input`.
pub use crate::py_streams::interfaces::StreamInput;
impl StreamInput {
    pub fn push<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("_core_push")
    }
    pub fn flush<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("_core_flush")
    }
    pub fn end_input<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("_core_end_input")
    }
}

/// Python interface: `builtins.memoryview`.
pub use crate::py_values::interfaces::MemoryView;
impl MemoryView {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        Ok(CLASS.import(py, "builtins", "memoryview")?.as_any().clone())
    }
    pub fn cast<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("cast")
    }
}

/// Python interface: `streaming turn detector`.
pub use crate::py_values::interfaces::TurnDetector;
impl TurnDetector {
    pub fn streaming(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(obj.hasattr("model")? && obj.hasattr("provider")? && obj.hasattr("stream")?)
    }
}

/// Python interface: `livekit.agents.types`.
pub use crate::py_values::interfaces::AgentsTypesModule;
impl AgentsTypesModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("livekit.agents.types")
    }
    pub fn not_given<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("NotGiven")
    }
}

/// Python interface: `time`.
pub use crate::py_values::interfaces::TimeModule;
impl TimeModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("time")
    }
    pub fn time<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("time")
    }
    pub fn perf_counter<'py>(
        obj: &Bound<'py, impl pyo3::PyTypeInfo>,
    ) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("perf_counter")
    }
}

/// Python interface: `numpy`.
pub use crate::py_values::interfaces::NumpyModule;
impl NumpyModule {
    pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
        py.import("numpy")
    }
    pub fn asarray<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("asarray")
    }
    pub fn int16<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("int16")
    }
    pub fn float32<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {
        obj.as_any().getattr("float32")
    }
}
