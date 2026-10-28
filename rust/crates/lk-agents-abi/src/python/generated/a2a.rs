//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `a2a` records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.
static AUTH_GRANT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::auth::AuthGrant {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(AUTH_GRANT.import(py, "livekit.agents._generated.a2a_auth", "AuthGrant")?)? {
            return Err(super::not_a(obj, "AuthGrant"));
        }
        Ok(Self {
            all: ::lk_agents_abi::python::attr_with(obj, "AuthGrant", "all", |v| v.extract::<bool>())?,
            scope: ::lk_agents_abi::python::attr_with(obj, "AuthGrant", "scope", |v| v.extract::<::std::string::String>())?,
            thread_ids: ::lk_agents_abi::python::attr_with(obj, "AuthGrant", "thread_ids", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>())?,
            list: ::lk_agents_abi::python::attr_with(obj, "AuthGrant", "list", |v| v.extract::<bool>())?,
            create: ::lk_agents_abi::python::attr_with(obj, "AuthGrant", "create", |v| v.extract::<bool>())?,
            write: ::lk_agents_abi::python::attr_with(obj, "AuthGrant", "write", |v| v.extract::<bool>())?,
            read: ::lk_agents_abi::python::attr_with(obj, "AuthGrant", "read", |v| v.extract::<bool>())?,
            delete: ::lk_agents_abi::python::attr_with(obj, "AuthGrant", "delete", |v| v.extract::<bool>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::auth::AuthGrant {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { all, scope, thread_ids, list, create, write, read, delete } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("all", ::pyo3::IntoPyObjectExt::into_bound_py_any(all, py)?)?;
        kwargs.set_item("scope", ::pyo3::IntoPyObjectExt::into_bound_py_any(scope, py)?)?;
        kwargs.set_item("thread_ids", ::pyo3::types::PyList::new(py, thread_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("list", ::pyo3::IntoPyObjectExt::into_bound_py_any(list, py)?)?;
        kwargs.set_item("create", ::pyo3::IntoPyObjectExt::into_bound_py_any(create, py)?)?;
        kwargs.set_item("write", ::pyo3::IntoPyObjectExt::into_bound_py_any(write, py)?)?;
        kwargs.set_item("read", ::pyo3::IntoPyObjectExt::into_bound_py_any(read, py)?)?;
        kwargs.set_item("delete", ::pyo3::IntoPyObjectExt::into_bound_py_any(delete, py)?)?;
        AUTH_GRANT.import(py, "livekit.agents._generated.a2a_auth", "AuthGrant")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::auth::AuthOperation {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
        match &*value {
            "create" => Ok(Self::Create),
            "list" => Ok(Self::List),
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "delete" => Ok(Self::Delete),
            other => Err(::pyo3::exceptions::PyValueError::new_err(format!("AuthOperation: {other:?} is not one of 'create', 'list', 'read', 'write', 'delete'"))),
        }
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::auth::AuthOperation {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Create => "create",
            Self::List => "list",
            Self::Read => "read",
            Self::Write => "write",
            Self::Delete => "delete",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::auth::PrincipalKind {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
        match &*value {
            "project" => Ok(Self::Project),
            "grant" => Ok(Self::Grant),
            other => Err(::pyo3::exceptions::PyValueError::new_err(format!("PrincipalKind: {other:?} is not one of 'project', 'grant'"))),
        }
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::auth::PrincipalKind {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Project => "project",
            Self::Grant => "grant",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.
static AUTH_PRINCIPAL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::auth::AuthPrincipal {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(AUTH_PRINCIPAL.import(py, "livekit.agents._generated.a2a_auth", "AuthPrincipal")?)? {
            return Err(super::not_a(obj, "AuthPrincipal"));
        }
        Ok(Self {
            kind: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "kind", |v| v.extract::<::lk_agents_abi::a2a::auth::PrincipalKind>().map_err(::std::convert::Into::<PyErr>::into))?,
            grant: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "grant", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::a2a::auth::AuthGrant>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            identity: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "identity", |v| v.extract::<::std::string::String>())?,
            claims: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "claims", |v| match ::lk_agents_abi::python::value_from_py(v, "AuthPrincipal.claims")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("AuthPrincipal.claims: expected a dict")) })?,
            endpoint_call: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "endpoint_call", |v| v.extract::<bool>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::auth::AuthPrincipal {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { kind, grant, identity, claims, endpoint_call } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("kind", ::pyo3::IntoPyObjectExt::into_bound_py_any(kind, py)?)?;
        kwargs.set_item("grant", match grant { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
        kwargs.set_item("claims", ::lk_agents_abi::python::extra_to_py(py, &claims)?.into_any())?;
        kwargs.set_item("endpoint_call", ::pyo3::IntoPyObjectExt::into_bound_py_any(endpoint_call, py)?)?;
        AUTH_PRINCIPAL.import(py, "livekit.agents._generated.a2a_auth", "AuthPrincipal")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/card.rs.
static ENDPOINT_DESCRIPTOR: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::card::EndpointDescriptor {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(ENDPOINT_DESCRIPTOR.import(py, "livekit.agents._generated.a2a_card", "EndpointDescriptor")?)? {
            return Err(super::not_a(obj, "EndpointDescriptor"));
        }
        Ok(Self {
            name: ::lk_agents_abi::python::attr_with(obj, "EndpointDescriptor", "name", |v| v.extract::<::std::string::String>())?,
            description: ::lk_agents_abi::python::attr_with(obj, "EndpointDescriptor", "description", |v| v.extract::<::std::string::String>())?,
            version: ::lk_agents_abi::python::attr_with(obj, "EndpointDescriptor", "version", |v| v.extract::<::std::string::String>())?,
            url: ::lk_agents_abi::python::attr_with(obj, "EndpointDescriptor", "url", |v| v.extract::<::std::string::String>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::card::EndpointDescriptor {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { name, description, version, url } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
        kwargs.set_item("description", ::pyo3::IntoPyObjectExt::into_bound_py_any(description, py)?)?;
        kwargs.set_item("version", ::pyo3::IntoPyObjectExt::into_bound_py_any(version, py)?)?;
        kwargs.set_item("url", ::pyo3::IntoPyObjectExt::into_bound_py_any(url, py)?)?;
        ENDPOINT_DESCRIPTOR.import(py, "livekit.agents._generated.a2a_card", "EndpointDescriptor")?.call((), Some(&kwargs))
    }
}

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
            return Ok(Self::FilePart { url: ::lk_agents_abi::python::attr_with(obj, "FilePart", "url", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, data: ::lk_agents_abi::python::attr_with(obj, "FilePart", "data", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; ::lk_agents_abi::python::number_from_py::<u8>(item, "FilePart.data") }).collect::<PyResult<Vec<_>>>()).map(Some) })?, filename: ::lk_agents_abi::python::attr_with(obj, "FilePart", "filename", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, media_type: ::lk_agents_abi::python::attr_with(obj, "FilePart", "media_type", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, metadata: ::lk_agents_abi::python::attr_with(obj, "FilePart", "metadata", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "FilePart.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FilePart.metadata: expected a dict")) }).map(Some) })? });
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
                kwargs.set_item("data", match data { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
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

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
static ARTIFACT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::client::Artifact {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(ARTIFACT.import(py, "livekit.agents._generated.a2a_client", "Artifact")?)? {
            return Err(super::not_a(obj, "Artifact"));
        }
        Ok(Self {
            artifact_id: ::lk_agents_abi::python::attr_with(obj, "Artifact", "artifact_id", |v| v.extract::<::std::string::String>())?,
            name: ::lk_agents_abi::python::attr_with(obj, "Artifact", "name", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            description: ::lk_agents_abi::python::attr_with(obj, "Artifact", "description", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            parts: ::lk_agents_abi::python::attr_with(obj, "Artifact", "parts", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::a2a::client::MessagePart>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            metadata: ::lk_agents_abi::python::attr_with(obj, "Artifact", "metadata", |v| match ::lk_agents_abi::python::value_from_py(v, "Artifact.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("Artifact.metadata: expected a dict")) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::client::Artifact {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { artifact_id, name, description, parts, metadata } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("artifact_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(artifact_id, py)?)?;
        kwargs.set_item("name", match name { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("description", match description { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("parts", ::pyo3::types::PyList::new(py, parts.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("metadata", ::lk_agents_abi::python::extra_to_py(py, &metadata)?.into_any())?;
        ARTIFACT.import(py, "livekit.agents._generated.a2a_client", "Artifact")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
static TASK_UPDATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::client::TaskUpdate {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(TASK_UPDATE.import(py, "livekit.agents._generated.a2a_client", "TaskUpdate")?)? {
            return Err(super::not_a(obj, "TaskUpdate"));
        }
        Ok(Self {
            state: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "state", |v| v.extract::<::std::string::String>())?,
            text: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "text", |v| v.extract::<::std::string::String>())?,
            task_id: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "task_id", |v| v.extract::<::std::string::String>())?,
            context_id: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "context_id", |v| v.extract::<::std::string::String>())?,
            metadata: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "metadata", |v| match ::lk_agents_abi::python::value_from_py(v, "TaskUpdate.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("TaskUpdate.metadata: expected a dict")) })?,
            artifacts: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "artifacts", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::a2a::client::Artifact>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::client::TaskUpdate {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { state, text, task_id, context_id, metadata, artifacts } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("state", ::pyo3::IntoPyObjectExt::into_bound_py_any(state, py)?)?;
        kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
        kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
        kwargs.set_item("context_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(context_id, py)?)?;
        kwargs.set_item("metadata", ::lk_agents_abi::python::extra_to_py(py, &metadata)?.into_any())?;
        kwargs.set_item("artifacts", ::pyo3::types::PyList::new(py, artifacts.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        TASK_UPDATE.import(py, "livekit.agents._generated.a2a_client", "TaskUpdate")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
static AGENT_CARD: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::client::AgentCard {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { name, description, extensions, input_schema, structured_input } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
        kwargs.set_item("description", ::pyo3::IntoPyObjectExt::into_bound_py_any(description, py)?)?;
        kwargs.set_item("extensions", ::pyo3::types::PyList::new(py, extensions.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("input_schema", ::lk_agents_abi::python::extra_to_py(py, &input_schema)?.into_any())?;
        kwargs.set_item("structured_input", ::pyo3::IntoPyObjectExt::into_bound_py_any(structured_input, py)?)?;
        AGENT_CARD.import(py, "livekit.agents._generated.a2a_client", "AgentCard")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/door.rs.
static DOOR_CONFIG: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::door::DoorConfig {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(DOOR_CONFIG.import(py, "livekit.agents._generated.a2a_door", "DoorConfig")?)? {
            return Err(super::not_a(obj, "DoorConfig"));
        }
        Ok(Self {
            lane: ::lk_agents_abi::python::attr_with(obj, "DoorConfig", "lane", |v| v.extract::<::std::string::String>())?,
            holder: ::lk_agents_abi::python::attr_with(obj, "DoorConfig", "holder", |v| v.extract::<::std::string::String>())?,
            endpoint: ::lk_agents_abi::python::attr_with(obj, "DoorConfig", "endpoint", |v| v.extract::<::lk_agents_abi::a2a::card::EndpointDescriptor>().map_err(::std::convert::Into::<PyErr>::into))?,
            options: ::lk_agents_abi::python::attr_with(obj, "DoorConfig", "options", |v| ::lk_agents_abi::python::value_from_py(v, "DoorConfig.options"))?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::door::DoorConfig {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { lane, holder, endpoint, options } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
        kwargs.set_item("holder", ::pyo3::IntoPyObjectExt::into_bound_py_any(holder, py)?)?;
        kwargs.set_item("endpoint", ::pyo3::IntoPyObjectExt::into_bound_py_any(endpoint, py)?)?;
        kwargs.set_item("options", ::lk_agents_abi::python::value_to_py(py, &options)?)?;
        DOOR_CONFIG.import(py, "livekit.agents._generated.a2a_door", "DoorConfig")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/door.rs.
static DOOR_FRAME: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::door::DoorFrame {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { json, sse, is_error } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("json", ::pyo3::IntoPyObjectExt::into_bound_py_any(json, py)?)?;
        kwargs.set_item("sse", ::pyo3::IntoPyObjectExt::into_bound_py_any(sse, py)?)?;
        kwargs.set_item("is_error", ::pyo3::IntoPyObjectExt::into_bound_py_any(is_error, py)?)?;
        DOOR_FRAME.import(py, "livekit.agents._generated.a2a_door", "DoorFrame")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/remote.rs.
static REMOTE_EVENT_PENDING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static REMOTE_EVENT_PROGRESS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static REMOTE_EVENT_INPUT_REQUIRED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static REMOTE_EVENT_OUTCOME: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static REMOTE_EVENT_FOLLOW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::remote::RemoteEvent {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Pending => REMOTE_EVENT_PENDING.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventPending")?.call0(),
            Self::Progress { text } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                REMOTE_EVENT_PROGRESS.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventProgress")?.call((), Some(&kwargs))
            },
            Self::InputRequired { question, request_id, verbatim, verbatim_full, outcome } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
                kwargs.set_item("request_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id, py)?)?;
                kwargs.set_item("verbatim", match verbatim { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                kwargs.set_item("verbatim_full", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim_full, py)?)?;
                kwargs.set_item("outcome", ::pyo3::IntoPyObjectExt::into_bound_py_any(outcome, py)?)?;
                REMOTE_EVENT_INPUT_REQUIRED.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventInputRequired")?.call((), Some(&kwargs))
            },
            Self::Outcome { outcome } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("outcome", ::pyo3::IntoPyObjectExt::into_bound_py_any(outcome, py)?)?;
                REMOTE_EVENT_OUTCOME.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventOutcome")?.call((), Some(&kwargs))
            },
            Self::Follow { task_id, thread_id } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
                kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
                REMOTE_EVENT_FOLLOW.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventFollow")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
static TURN_CONTEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::turn::TurnContext {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { thread_id, lane, message, batch, history, state, context_scope_id, context_scopes, read_only, task_id, tasks, caller_history, answer_tool, host_context: _ } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
        kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
        kwargs.set_item("message", ::lk_agents_abi::python::extra_to_py(py, &message)?.into_any())?;
        kwargs.set_item("batch", ::pyo3::types::PyList::new(py, batch.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("state", match state { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
        kwargs.set_item("context_scope_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(context_scope_id, py)?)?;
        kwargs.set_item("context_scopes", ::pyo3::types::PyList::new(py, context_scopes.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("read_only", ::pyo3::IntoPyObjectExt::into_bound_py_any(read_only, py)?)?;
        kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
        kwargs.set_item("tasks", ::pyo3::types::PyList::new(py, tasks.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("caller_history", match caller_history { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
        kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        TURN_CONTEXT.import(py, "livekit.agents._generated.storage_turn", "TurnContext")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
static PENDING_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::turn::PendingCall {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(PENDING_CALL.import(py, "livekit.agents._generated.storage_turn", "PendingCall")?)? {
            return Err(super::not_a(obj, "PendingCall"));
        }
        Ok(Self {
            name: ::lk_agents_abi::python::attr_with(obj, "PendingCall", "name", |v| v.extract::<::std::string::String>())?,
            call_id: ::lk_agents_abi::python::attr_with(obj, "PendingCall", "call_id", |v| v.extract::<::std::string::String>())?,
            answer_tool: ::lk_agents_abi::python::attr_with(obj, "PendingCall", "answer_tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
static HOST_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::turn::HostEvent {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(HOST_EVENT.import(py, "livekit.agents._generated.storage_turn", "HostEvent")?)? {
            return Err(super::not_a(obj, "HostEvent"));
        }
        Ok(Self {
            r#type: ::lk_agents_abi::python::attr_with(obj, "HostEvent", "type", |v| v.extract::<::std::string::String>())?,
            payload: ::lk_agents_abi::python::attr_with(obj, "HostEvent", "payload", |v| ::lk_agents_abi::python::value_from_py(v, "HostEvent.payload"))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
static HANDLER_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::turn::HandlerOutput {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(HANDLER_OUTPUT.import(py, "livekit.agents._generated.storage_turn", "HandlerOutput")?)? {
            return Err(super::not_a(obj, "HandlerOutput"));
        }
        Ok(Self {
            output: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "output", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "HandlerOutput.output")).map(Some) })?,
            new_items: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "new_items", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; ::lk_agents_abi::python::value_from_py(item, "HandlerOutput.new_items") }).collect::<PyResult<Vec<_>>>())?,
            state: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "state", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "HandlerOutput.state")).map(Some) })?,
            scope_writes: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "scope_writes", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::storage::context::ContextScopeWrite>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            usage: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "usage", |v| v.extract::<::lk_agents_abi::storage::RunUsage>().map_err(::std::convert::Into::<PyErr>::into))?,
            input_required: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "input_required", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            pending_call: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "pending_call", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::a2a::turn::PendingCall>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            interrupted: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "interrupted", |v| v.extract::<bool>())?,
            metadata: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "metadata", |v| match ::lk_agents_abi::python::value_from_py(v, "HandlerOutput.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("HandlerOutput.metadata: expected a dict")) })?,
            error: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "error", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            events: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "events", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::a2a::turn::HostEvent>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
static TURN_RESULT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::turn::TurnResult {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { task, output, thread_id, answered_by, run_id } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("task", ::pyo3::IntoPyObjectExt::into_bound_py_any(task, py)?)?;
        kwargs.set_item("output", match output { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
        kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
        kwargs.set_item("answered_by", ::pyo3::IntoPyObjectExt::into_bound_py_any(answered_by, py)?)?;
        kwargs.set_item("run_id", match run_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        TURN_RESULT.import(py, "livekit.agents._generated.storage_turn", "TurnResult")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::turn::Verdict {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
        match &*value {
            "continue" => Ok(Self::Continue),
            "yield" => Ok(Self::Yield),
            other => Err(::pyo3::exceptions::PyValueError::new_err(format!("Verdict: {other:?} is not one of 'continue', 'yield'"))),
        }
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::turn::Verdict {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Continue => "continue",
            Self::Yield => "yield",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
static CHECKPOINT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static PROGRESS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TOKEN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::a2a::turn::TurnEvent {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(CHECKPOINT.import(py, "livekit.agents._generated.storage_turn", "Checkpoint")?)? {
            return Ok(Self::Checkpoint);
        }
        if obj.is_instance(OUTPUT.import(py, "livekit.agents._generated.storage_turn", "Output")?)? {
            return Ok(Self::Output { output: ::lk_agents_abi::python::attr_with(obj, "Output", "output", |v| v.extract::<::lk_agents_abi::a2a::turn::HandlerOutput>().map_err(::std::convert::Into::<PyErr>::into))? });
        }
        if obj.is_instance(PROGRESS.import(py, "livekit.agents._generated.storage_turn", "Progress")?)? {
            return Ok(Self::Progress { task_id: ::lk_agents_abi::python::attr_with(obj, "Progress", "task_id", |v| v.extract::<::std::string::String>())?, text: ::lk_agents_abi::python::attr_with(obj, "Progress", "text", |v| v.extract::<::std::string::String>())? });
        }
        if obj.is_instance(TOKEN.import(py, "livekit.agents._generated.storage_turn", "Token")?)? {
            return Ok(Self::Token { task_id: ::lk_agents_abi::python::attr_with(obj, "Token", "task_id", |v| v.extract::<::std::string::String>())?, delta: ::lk_agents_abi::python::attr_with(obj, "Token", "delta", |v| v.extract::<::std::string::String>())? });
        }
        Err(super::not_a(obj, "Checkpoint | Output | Progress | Token"))
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
