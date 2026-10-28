//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `a2a` proxy bridges and in-place records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
static MESSAGE_METADATA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::client::MessageMetadata {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(MESSAGE_METADATA.import(py, "livekit.agents._generated.a2a_client", "MessageMetadata")?)? {
            return Err(super::not_a(obj, "MessageMetadata"));
        }
        Ok(Self {
            kind: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "kind", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            channel: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "channel", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            verbatim: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "verbatim", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            answer_tool: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "answer_tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            merged_into: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "merged_into", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            reason: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "reason", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            interrupted: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "interrupted", |v| if v.is_none() { Ok(None) } else { (v.extract::<bool>()).map(Some) })?,
            interruption_reason: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "interruption_reason", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            application: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "application", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "MessageMetadata.application")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("MessageMetadata.application: expected a dict")) }).map(Some) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::client::MessageMetadata {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { kind, channel, verbatim, answer_tool, merged_into, reason, interrupted, interruption_reason, application } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("kind", match kind { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("channel", match channel { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("verbatim", match verbatim { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("merged_into", match merged_into { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("reason", match reason { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("interrupted", match interrupted { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("interruption_reason", match interruption_reason { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("application", match application { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
        MESSAGE_METADATA.import(py, "livekit.agents._generated.a2a_client", "MessageMetadata")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
static TEXT_PART: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static DATA_PART: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static FILE_PART: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::client::MessagePart {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(TEXT_PART.import(py, "livekit.agents._generated.a2a_client", "TextPart")?)? {
            return Ok(Self::TextPart { text: ::lk_agents_abi::python::attr_with(obj, "TextPart", "text", |v| v.extract::<::std::string::String>())?, metadata: ::lk_agents_abi::python::attr_with(obj, "TextPart", "metadata", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "TextPart.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("TextPart.metadata: expected a dict")) }).map(Some) })? });
        }
        if obj.is_instance(DATA_PART.import(py, "livekit.agents._generated.a2a_client", "DataPart")?)? {
            return Ok(Self::DataPart { data: ::lk_agents_abi::python::attr_with(obj, "DataPart", "data", |v| ::lk_agents_abi::python::value_from_py(v, "DataPart.data"))?, metadata: ::lk_agents_abi::python::attr_with(obj, "DataPart", "metadata", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "DataPart.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("DataPart.metadata: expected a dict")) }).map(Some) })? });
        }
        if obj.is_instance(FILE_PART.import(py, "livekit.agents._generated.a2a_client", "FilePart")?)? {
            return Ok(Self::FilePart { url: ::lk_agents_abi::python::attr_with(obj, "FilePart", "url", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, data: ::lk_agents_abi::python::attr_with(obj, "FilePart", "data", |v| if v.is_none() { Ok(None) } else { ({ if v.is_instance_of::<::pyo3::types::PyBytes>() || v.is_instance_of::<::pyo3::types::PyByteArray>() || v.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&v, "FilePart.data") } else { v.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "FilePart.data") }).collect::<PyResult<Vec<_>>>()) } }).map(Some) })?, filename: ::lk_agents_abi::python::attr_with(obj, "FilePart", "filename", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, media_type: ::lk_agents_abi::python::attr_with(obj, "FilePart", "media_type", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, metadata: ::lk_agents_abi::python::attr_with(obj, "FilePart", "metadata", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "FilePart.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FilePart.metadata: expected a dict")) }).map(Some) })? });
        }
        Err(super::not_a(obj, "TextPart | DataPart | FilePart"))
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::client::MessagePart {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::TextPart { text, metadata } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
                TEXT_PART.import(py, "livekit.agents._generated.a2a_client", "TextPart")?.call((), Some(&kwargs))
            },
            Self::DataPart { data, metadata } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("data", ::lk_agents_abi::python::value_to_py(py, &data)?)?;
                kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
                DATA_PART.import(py, "livekit.agents._generated.a2a_client", "DataPart")?.call((), Some(&kwargs))
            },
            Self::FilePart { url, data, filename, media_type, metadata } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("url", match url { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("data", match data { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyBytes::new(py, &x).into_any() })?;
                kwargs.set_item("filename", match filename { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("media_type", match media_type { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
                FILE_PART.import(py, "livekit.agents._generated.a2a_client", "FilePart")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
static MESSAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::client::Message {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(MESSAGE.import(py, "livekit.agents._generated.a2a_client", "Message")?)? {
            return Err(super::not_a(obj, "Message"));
        }
        Ok(Self {
            parts: ::lk_agents_abi::python::attr_with(obj, "Message", "parts", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::a2a::client::MessagePart>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            metadata: ::lk_agents_abi::python::attr_with(obj, "Message", "metadata", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::a2a::client::MessageMetadata>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            message_id: ::lk_agents_abi::python::attr_with(obj, "Message", "message_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            task_id: ::lk_agents_abi::python::attr_with(obj, "Message", "task_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            context_id: ::lk_agents_abi::python::attr_with(obj, "Message", "context_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            role: ::lk_agents_abi::python::attr_with(obj, "Message", "role", |v| v.extract::<::std::string::String>())?,
            channel: ::lk_agents_abi::python::attr_with(obj, "Message", "channel", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            received_at: ::lk_agents_abi::python::attr_with(obj, "Message", "received_at", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "Message.received_at")).map(Some) })?,
            raw: ::lk_agents_abi::python::attr_with(obj, "Message", "raw", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "Message.raw")).map(Some) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::client::Message {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { parts, metadata, message_id, task_id, context_id, role, channel, received_at, raw } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("parts", ::pyo3::types::PyList::new(py, parts.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("message_id", match message_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("task_id", match task_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("context_id", match context_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("role", ::pyo3::IntoPyObjectExt::into_bound_py_any(role, py)?)?;
        kwargs.set_item("channel", match channel { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("received_at", match received_at { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("raw", match raw { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
        MESSAGE.import(py, "livekit.agents._generated.a2a_client", "Message")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::a2a::turn::TurnHost {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::a2a::turn::TurnHost>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`TurnHost` is a declared proxy: it is taken from the host object, not a view"))
    }
}
