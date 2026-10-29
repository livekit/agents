//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! Each record crosses as an instance of its generated model.

/// `obj` is not an instance of the model(s) `expected` names.
fn not_a(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, expected: &str) -> ::pyo3::PyErr {
    use ::pyo3::types::{PyAnyMethods as _, PyTypeMethods as _};
    let got = obj.get_type().name().map(|n| n.to_string()).unwrap_or_default();
    ::pyo3::exceptions::PyTypeError::new_err(format!("expected {expected}, got {got}"))
}

pub mod _retry {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/types/retry.rs.
    static _CORE__RETRY_DECISION_STOP: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static _CORE__RETRY_DECISION_RAISE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static _CORE__RETRY_DECISION_EXHAUSTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static _CORE__RETRY_DECISION_RETRY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn _core__retry_decision_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::types::retry::RetryDecision) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::types::retry::RetryDecision::Stop => crate::host::construct(_CORE__RETRY_DECISION_STOP.import(py, "livekit.agents._generated._retry", "_core_RetryDecisionStop")?.as_any(), (), None),
                ::lk_agents_abi::types::retry::RetryDecision::Raise => crate::host::construct(_CORE__RETRY_DECISION_RAISE.import(py, "livekit.agents._generated._retry", "_core_RetryDecisionRaise")?.as_any(), (), None),
                ::lk_agents_abi::types::retry::RetryDecision::Exhausted(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(_CORE__RETRY_DECISION_EXHAUSTED.import(py, "livekit.agents._generated._retry", "_core_RetryDecisionExhausted")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::types::retry::RetryDecision::Retry(delay) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(delay, py)?)?;
                    crate::host::construct(_CORE__RETRY_DECISION_RETRY.import(py, "livekit.agents._generated._retry", "_core_RetryDecisionRetry")?.as_any(), (), Some(&kwargs))
                },
            }
    }
}

pub mod a2a_auth {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.
    static AUTH_GRANT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn auth_grant_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::auth::AuthGrant> {
        let py = obj.py();
            if !obj.is_instance(AUTH_GRANT.import(py, "livekit.agents._generated.a2a_auth", "AuthGrant")?)? {
                return Err(super::not_a(obj, "AuthGrant"));
            }
            Ok(::lk_agents_abi::a2a::auth::AuthGrant {
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

    pub fn auth_grant_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::auth::AuthGrant) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::auth::AuthGrant { all, scope, thread_ids, list, create, write, read, delete } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("all", ::pyo3::IntoPyObjectExt::into_bound_py_any(all, py)?)?;
            kwargs.set_item("scope", ::pyo3::IntoPyObjectExt::into_bound_py_any(scope, py)?)?;
            kwargs.set_item("thread_ids", ::pyo3::types::PyList::new(py, thread_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("list", ::pyo3::IntoPyObjectExt::into_bound_py_any(list, py)?)?;
            kwargs.set_item("create", ::pyo3::IntoPyObjectExt::into_bound_py_any(create, py)?)?;
            kwargs.set_item("write", ::pyo3::IntoPyObjectExt::into_bound_py_any(write, py)?)?;
            kwargs.set_item("read", ::pyo3::IntoPyObjectExt::into_bound_py_any(read, py)?)?;
            kwargs.set_item("delete", ::pyo3::IntoPyObjectExt::into_bound_py_any(delete, py)?)?;
            crate::host::construct(AUTH_GRANT.import(py, "livekit.agents._generated.a2a_auth", "AuthGrant")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.

    pub fn auth_operation_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::auth::AuthOperation> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "create" => Ok(::lk_agents_abi::a2a::auth::AuthOperation::Create),
                "list" => Ok(::lk_agents_abi::a2a::auth::AuthOperation::List),
                "read" => Ok(::lk_agents_abi::a2a::auth::AuthOperation::Read),
                "write" => Ok(::lk_agents_abi::a2a::auth::AuthOperation::Write),
                "delete" => Ok(::lk_agents_abi::a2a::auth::AuthOperation::Delete),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("AuthOperation: {other:?} is not one of 'create', 'list', 'read', 'write', 'delete'"))),
            }
    }

    pub fn auth_operation_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::auth::AuthOperation) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::a2a::auth::AuthOperation::Create => "create",
                ::lk_agents_abi::a2a::auth::AuthOperation::List => "list",
                ::lk_agents_abi::a2a::auth::AuthOperation::Read => "read",
                ::lk_agents_abi::a2a::auth::AuthOperation::Write => "write",
                ::lk_agents_abi::a2a::auth::AuthOperation::Delete => "delete",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.

    pub fn principal_kind_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::auth::PrincipalKind> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "project" => Ok(::lk_agents_abi::a2a::auth::PrincipalKind::Project),
                "grant" => Ok(::lk_agents_abi::a2a::auth::PrincipalKind::Grant),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("PrincipalKind: {other:?} is not one of 'project', 'grant'"))),
            }
    }

    pub fn principal_kind_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::auth::PrincipalKind) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::a2a::auth::PrincipalKind::Project => "project",
                ::lk_agents_abi::a2a::auth::PrincipalKind::Grant => "grant",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.
    static AUTH_PRINCIPAL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn auth_principal_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::auth::AuthPrincipal> {
        let py = obj.py();
            if !obj.is_instance(AUTH_PRINCIPAL.import(py, "livekit.agents._generated.a2a_auth", "AuthPrincipal")?)? {
                return Err(super::not_a(obj, "AuthPrincipal"));
            }
            Ok(::lk_agents_abi::a2a::auth::AuthPrincipal {
                kind: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "kind", |v| crate::generated::records::a2a_auth::principal_kind_from_py(v))?,
                grant: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "grant", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::a2a_auth::auth_grant_from_py(v)).map(Some) })?,
                identity: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "identity", |v| v.extract::<::std::string::String>())?,
                claims: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "claims", |v| match ::lk_agents_abi::python::value_from_py(v, "AuthPrincipal.claims")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("AuthPrincipal.claims: expected a dict")) })?,
                endpoint_call: ::lk_agents_abi::python::attr_with(obj, "AuthPrincipal", "endpoint_call", |v| v.extract::<bool>())?,
            })
    }

    pub fn auth_principal_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::auth::AuthPrincipal) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::auth::AuthPrincipal { kind, grant, identity, claims, endpoint_call } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("kind", crate::generated::records::a2a_auth::principal_kind_to_py(py, kind)?)?;
            kwargs.set_item("grant", match grant { None => py.None().into_bound(py), Some(x) => crate::generated::records::a2a_auth::auth_grant_to_py(py, x)? })?;
            kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
            kwargs.set_item("claims", ::lk_agents_abi::python::extra_to_py(py, &claims)?.into_any())?;
            kwargs.set_item("endpoint_call", ::pyo3::IntoPyObjectExt::into_bound_py_any(endpoint_call, py)?)?;
            crate::host::construct(AUTH_PRINCIPAL.import(py, "livekit.agents._generated.a2a_auth", "AuthPrincipal")?.as_any(), (), Some(&kwargs))
    }
}

pub mod a2a_card {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/a2a/card.rs.
    static ENDPOINT_DESCRIPTOR: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn endpoint_descriptor_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::card::EndpointDescriptor> {
        let py = obj.py();
            if !obj.is_instance(ENDPOINT_DESCRIPTOR.import(py, "livekit.agents._generated.a2a_card", "EndpointDescriptor")?)? {
                return Err(super::not_a(obj, "EndpointDescriptor"));
            }
            Ok(::lk_agents_abi::a2a::card::EndpointDescriptor {
                name: ::lk_agents_abi::python::attr_with(obj, "EndpointDescriptor", "name", |v| v.extract::<::std::string::String>())?,
                description: ::lk_agents_abi::python::attr_with(obj, "EndpointDescriptor", "description", |v| v.extract::<::std::string::String>())?,
                version: ::lk_agents_abi::python::attr_with(obj, "EndpointDescriptor", "version", |v| v.extract::<::std::string::String>())?,
                url: ::lk_agents_abi::python::attr_with(obj, "EndpointDescriptor", "url", |v| v.extract::<::std::string::String>())?,
            })
    }

    pub fn endpoint_descriptor_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::card::EndpointDescriptor) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::card::EndpointDescriptor { name, description, version, url } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("description", ::pyo3::IntoPyObjectExt::into_bound_py_any(description, py)?)?;
            kwargs.set_item("version", ::pyo3::IntoPyObjectExt::into_bound_py_any(version, py)?)?;
            kwargs.set_item("url", ::pyo3::IntoPyObjectExt::into_bound_py_any(url, py)?)?;
            crate::host::construct(ENDPOINT_DESCRIPTOR.import(py, "livekit.agents._generated.a2a_card", "EndpointDescriptor")?.as_any(), (), Some(&kwargs))
    }
}

pub mod a2a_client {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
    static MESSAGE_METADATA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn message_metadata_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::client::MessageMetadata> {
        let py = obj.py();
            if !obj.is_instance(MESSAGE_METADATA.import(py, "livekit.agents._generated.a2a_client", "MessageMetadata")?)? {
                return Err(super::not_a(obj, "MessageMetadata"));
            }
            Ok(::lk_agents_abi::a2a::client::MessageMetadata {
                kind: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "kind", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                channel: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "channel", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                verbatim: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "verbatim", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                merged_into: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "merged_into", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                reason: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "reason", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                interrupted: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "interrupted", |v| if v.is_none() { Ok(None) } else { (v.extract::<bool>()).map(Some) })?,
                interruption_reason: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "interruption_reason", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                application: ::lk_agents_abi::python::attr_with(obj, "MessageMetadata", "application", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "MessageMetadata.application")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("MessageMetadata.application: expected a dict")) }).map(Some) })?,
            })
    }

    pub fn message_metadata_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::client::MessageMetadata) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::client::MessageMetadata { kind, channel, verbatim, merged_into, reason, interrupted, interruption_reason, application } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("kind", match kind { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("channel", match channel { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("verbatim", match verbatim { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("merged_into", match merged_into { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("reason", match reason { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("interrupted", match interrupted { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("interruption_reason", match interruption_reason { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("application", match application { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
            crate::host::construct(MESSAGE_METADATA.import(py, "livekit.agents._generated.a2a_client", "MessageMetadata")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
    static TEXT_PART: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static DATA_PART: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static FILE_PART: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn message_part_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::client::MessagePart> {
        let py = obj.py();
            if obj.is_instance(TEXT_PART.import(py, "livekit.agents._generated.a2a_client", "TextPart")?)? {
                return Ok(::lk_agents_abi::a2a::client::MessagePart::TextPart { text: ::lk_agents_abi::python::attr_with(obj, "TextPart", "text", |v| v.extract::<::std::string::String>())?, metadata: ::lk_agents_abi::python::attr_with(obj, "TextPart", "metadata", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "TextPart.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("TextPart.metadata: expected a dict")) }).map(Some) })? });
            }
            if obj.is_instance(DATA_PART.import(py, "livekit.agents._generated.a2a_client", "DataPart")?)? {
                return Ok(::lk_agents_abi::a2a::client::MessagePart::DataPart { data: ::lk_agents_abi::python::attr_with(obj, "DataPart", "data", |v| ::lk_agents_abi::python::value_from_py(v, "DataPart.data"))?, metadata: ::lk_agents_abi::python::attr_with(obj, "DataPart", "metadata", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "DataPart.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("DataPart.metadata: expected a dict")) }).map(Some) })? });
            }
            if obj.is_instance(FILE_PART.import(py, "livekit.agents._generated.a2a_client", "FilePart")?)? {
                return Ok(::lk_agents_abi::a2a::client::MessagePart::FilePart { url: ::lk_agents_abi::python::attr_with(obj, "FilePart", "url", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, data: ::lk_agents_abi::python::attr_with(obj, "FilePart", "data", |v| if v.is_none() { Ok(None) } else { ({ if v.is_instance_of::<::pyo3::types::PyBytes>() || v.is_instance_of::<::pyo3::types::PyByteArray>() || v.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&v, "FilePart.data") } else { v.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "FilePart.data") }).collect::<PyResult<Vec<_>>>()) } }).map(Some) })?, filename: ::lk_agents_abi::python::attr_with(obj, "FilePart", "filename", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, media_type: ::lk_agents_abi::python::attr_with(obj, "FilePart", "media_type", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, metadata: ::lk_agents_abi::python::attr_with(obj, "FilePart", "metadata", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "FilePart.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FilePart.metadata: expected a dict")) }).map(Some) })? });
            }
            Err(super::not_a(obj, "TextPart | DataPart | FilePart"))
    }

    pub fn message_part_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::client::MessagePart) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::a2a::client::MessagePart::TextPart { text, metadata } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                    kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
                    crate::host::construct(TEXT_PART.import(py, "livekit.agents._generated.a2a_client", "TextPart")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::a2a::client::MessagePart::DataPart { data, metadata } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("data", ::lk_agents_abi::python::value_to_py(py, &data)?)?;
                    kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
                    crate::host::construct(DATA_PART.import(py, "livekit.agents._generated.a2a_client", "DataPart")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::a2a::client::MessagePart::FilePart { url, data, filename, media_type, metadata } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("url", match url { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("data", match data { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyBytes::new(py, &x).into_any() })?;
                    kwargs.set_item("filename", match filename { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("media_type", match media_type { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
                    crate::host::construct(FILE_PART.import(py, "livekit.agents._generated.a2a_client", "FilePart")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
    static MESSAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn message_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::client::Message> {
        let py = obj.py();
            if !obj.is_instance(MESSAGE.import(py, "livekit.agents._generated.a2a_client", "Message")?)? {
                return Err(super::not_a(obj, "Message"));
            }
            Ok(::lk_agents_abi::a2a::client::Message {
                parts: ::lk_agents_abi::python::attr_with(obj, "Message", "parts", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::a2a_client::message_part_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "Message", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::a2a_client::message_metadata_from_py(v)).map(Some) })?,
                message_id: ::lk_agents_abi::python::attr_with(obj, "Message", "message_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                task_id: ::lk_agents_abi::python::attr_with(obj, "Message", "task_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                context_id: ::lk_agents_abi::python::attr_with(obj, "Message", "context_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                role: ::lk_agents_abi::python::attr_with(obj, "Message", "role", |v| v.extract::<::std::string::String>())?,
                channel: ::lk_agents_abi::python::attr_with(obj, "Message", "channel", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                received_at: ::lk_agents_abi::python::attr_with(obj, "Message", "received_at", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "Message.received_at")).map(Some) })?,
                raw: ::lk_agents_abi::python::attr_with(obj, "Message", "raw", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "Message.raw")).map(Some) })?,
            })
    }

    pub fn message_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::client::Message) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::client::Message { parts, metadata, message_id, task_id, context_id, role, channel, received_at, raw } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("parts", ::pyo3::types::PyList::new(py, parts.into_iter().map(|x| PyResult::Ok(crate::generated::records::a2a_client::message_part_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::a2a_client::message_metadata_to_py(py, x)? })?;
            kwargs.set_item("message_id", match message_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("task_id", match task_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("context_id", match context_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("role", ::pyo3::IntoPyObjectExt::into_bound_py_any(role, py)?)?;
            kwargs.set_item("channel", match channel { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("received_at", match received_at { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("raw", match raw { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
            crate::host::construct(MESSAGE.import(py, "livekit.agents._generated.a2a_client", "Message")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.

    pub fn message_input_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::client::MessageInput> {
        if obj.is_instance_of::<::pyo3::types::PyString>() {
            return Ok(::lk_agents_abi::a2a::client::MessageInput::Text(obj.extract::<::std::string::String>()?));
        }
        if obj.is_instance(&obj.py().import("livekit.agents._generated.a2a_client")?.getattr("Message")?)? {
            return Ok(::lk_agents_abi::a2a::client::MessageInput::Message(crate::generated::records::a2a_client::message_from_py(obj)?));
        }
        Err(super::not_a(obj, "MessageInput"))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
    static ARTIFACT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn artifact_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::client::Artifact> {
        let py = obj.py();
            if !obj.is_instance(ARTIFACT.import(py, "livekit.agents._generated.a2a_client", "Artifact")?)? {
                return Err(super::not_a(obj, "Artifact"));
            }
            Ok(::lk_agents_abi::a2a::client::Artifact {
                artifact_id: ::lk_agents_abi::python::attr_with(obj, "Artifact", "artifact_id", |v| v.extract::<::std::string::String>())?,
                name: ::lk_agents_abi::python::attr_with(obj, "Artifact", "name", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                description: ::lk_agents_abi::python::attr_with(obj, "Artifact", "description", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                parts: ::lk_agents_abi::python::attr_with(obj, "Artifact", "parts", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::a2a_client::message_part_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "Artifact", "metadata", |v| match ::lk_agents_abi::python::value_from_py(v, "Artifact.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("Artifact.metadata: expected a dict")) })?,
            })
    }

    pub fn artifact_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::client::Artifact) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::client::Artifact { artifact_id, name, description, parts, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("artifact_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(artifact_id, py)?)?;
            kwargs.set_item("name", match name { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("description", match description { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("parts", ::pyo3::types::PyList::new(py, parts.into_iter().map(|x| PyResult::Ok(crate::generated::records::a2a_client::message_part_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("metadata", ::lk_agents_abi::python::extra_to_py(py, &metadata)?.into_any())?;
            crate::host::construct(ARTIFACT.import(py, "livekit.agents._generated.a2a_client", "Artifact")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
    static TASK_UPDATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn task_update_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::client::TaskUpdate> {
        let py = obj.py();
            if !obj.is_instance(TASK_UPDATE.import(py, "livekit.agents._generated.a2a_client", "TaskUpdate")?)? {
                return Err(super::not_a(obj, "TaskUpdate"));
            }
            Ok(::lk_agents_abi::a2a::client::TaskUpdate {
                state: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "state", |v| v.extract::<::std::string::String>())?,
                text: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "text", |v| v.extract::<::std::string::String>())?,
                task_id: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "task_id", |v| v.extract::<::std::string::String>())?,
                context_id: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "context_id", |v| v.extract::<::std::string::String>())?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "metadata", |v| match ::lk_agents_abi::python::value_from_py(v, "TaskUpdate.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("TaskUpdate.metadata: expected a dict")) })?,
                artifacts: ::lk_agents_abi::python::attr_with(obj, "TaskUpdate", "artifacts", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::a2a_client::artifact_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
            })
    }

    pub fn task_update_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::client::TaskUpdate) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::client::TaskUpdate { state, text, task_id, context_id, metadata, artifacts } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("state", ::pyo3::IntoPyObjectExt::into_bound_py_any(state, py)?)?;
            kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
            kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
            kwargs.set_item("context_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(context_id, py)?)?;
            kwargs.set_item("metadata", ::lk_agents_abi::python::extra_to_py(py, &metadata)?.into_any())?;
            kwargs.set_item("artifacts", ::pyo3::types::PyList::new(py, artifacts.into_iter().map(|x| PyResult::Ok(crate::generated::records::a2a_client::artifact_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(TASK_UPDATE.import(py, "livekit.agents._generated.a2a_client", "TaskUpdate")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
    static AGENT_CARD: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn agent_card_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::client::AgentCard) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::client::AgentCard { name, description, extensions, input_schema, structured_input } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("description", ::pyo3::IntoPyObjectExt::into_bound_py_any(description, py)?)?;
            kwargs.set_item("extensions", ::pyo3::types::PyList::new(py, extensions.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("input_schema", ::lk_agents_abi::python::extra_to_py(py, &input_schema)?.into_any())?;
            kwargs.set_item("structured_input", ::pyo3::IntoPyObjectExt::into_bound_py_any(structured_input, py)?)?;
            crate::host::construct(AGENT_CARD.import(py, "livekit.agents._generated.a2a_client", "AgentCard")?.as_any(), (), Some(&kwargs))
    }
}

pub mod a2a_door {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/a2a/door.rs.
    static DOOR_CONFIG: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn door_config_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::door::DoorConfig> {
        let py = obj.py();
            if !obj.is_instance(DOOR_CONFIG.import(py, "livekit.agents._generated.a2a_door", "DoorConfig")?)? {
                return Err(super::not_a(obj, "DoorConfig"));
            }
            Ok(::lk_agents_abi::a2a::door::DoorConfig {
                lane: ::lk_agents_abi::python::attr_with(obj, "DoorConfig", "lane", |v| v.extract::<::std::string::String>())?,
                holder: ::lk_agents_abi::python::attr_with(obj, "DoorConfig", "holder", |v| v.extract::<::std::string::String>())?,
                endpoint: ::lk_agents_abi::python::attr_with(obj, "DoorConfig", "endpoint", |v| crate::generated::records::a2a_card::endpoint_descriptor_from_py(v))?,
                options: ::lk_agents_abi::python::attr_with(obj, "DoorConfig", "options", |v| crate::generated::records::storage_options::turn_options_from_py(v))?,
            })
    }

    pub fn door_config_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::door::DoorConfig) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::door::DoorConfig { lane, holder, endpoint, options } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
            kwargs.set_item("holder", ::pyo3::IntoPyObjectExt::into_bound_py_any(holder, py)?)?;
            kwargs.set_item("endpoint", crate::generated::records::a2a_card::endpoint_descriptor_to_py(py, endpoint)?)?;
            kwargs.set_item("options", crate::generated::records::storage_options::turn_options_to_py(py, options)?)?;
            crate::host::construct(DOOR_CONFIG.import(py, "livekit.agents._generated.a2a_door", "DoorConfig")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/door.rs.
    static DOOR_FRAME: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn door_frame_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::door::DoorFrame) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::door::DoorFrame { json, sse, is_error } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("json", ::pyo3::IntoPyObjectExt::into_bound_py_any(json, py)?)?;
            kwargs.set_item("sse", ::pyo3::IntoPyObjectExt::into_bound_py_any(sse, py)?)?;
            kwargs.set_item("is_error", ::pyo3::IntoPyObjectExt::into_bound_py_any(is_error, py)?)?;
            crate::host::construct(DOOR_FRAME.import(py, "livekit.agents._generated.a2a_door", "DoorFrame")?.as_any(), (), Some(&kwargs))
    }
}

pub mod a2a_remote {
    use pyo3::prelude::*;

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

    pub fn remote_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::remote::RemoteEvent) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::a2a::remote::RemoteEvent::Pending => crate::host::construct(REMOTE_EVENT_PENDING.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventPending")?.as_any(), (), None),
                ::lk_agents_abi::a2a::remote::RemoteEvent::Progress { text } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                    crate::host::construct(REMOTE_EVENT_PROGRESS.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventProgress")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::a2a::remote::RemoteEvent::InputRequired { question, request_id, verbatim, verbatim_full, outcome } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
                    kwargs.set_item("request_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id, py)?)?;
                    kwargs.set_item("verbatim", match verbatim { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("verbatim_full", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim_full, py)?)?;
                    kwargs.set_item("outcome", crate::generated::records::tool_report::tool_outcome_to_py(py, outcome)?)?;
                    crate::host::construct(REMOTE_EVENT_INPUT_REQUIRED.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventInputRequired")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::a2a::remote::RemoteEvent::Outcome { outcome } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("outcome", crate::generated::records::tool_report::tool_outcome_to_py(py, outcome)?)?;
                    crate::host::construct(REMOTE_EVENT_OUTCOME.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventOutcome")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::a2a::remote::RemoteEvent::Follow { task_id, thread_id } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
                    kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
                    crate::host::construct(REMOTE_EVENT_FOLLOW.import(py, "livekit.agents._generated.a2a_remote", "RemoteEventFollow")?.as_any(), (), Some(&kwargs))
                },
            }
    }
}

pub mod agent_run {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static INVOCATION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn invocation_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::Invocation) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::agent::Invocation { operation, invocation_id, agent_name, execution_id, frame_id, context_scope, application_context, history, function_call, approval, speech, host_context: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("operation", ::pyo3::IntoPyObjectExt::into_bound_py_any(operation, py)?)?;
            kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
            kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
            kwargs.set_item("execution_id", match execution_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("frame_id", match frame_id { None => py.None().into_bound(py), Some(x) => crate::generated::records::text_session::frame_id_to_py(py, x)? })?;
            kwargs.set_item("context_scope", match context_scope { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("application_context", match application_context { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("function_call", match function_call { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("approval", match approval { None => py.None().into_bound(py), Some(x) => crate::generated::records::tool_report::approval_to_py(py, x)? })?;
            kwargs.set_item("speech", match speech { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(INVOCATION.import(py, "livekit.agents._generated.agent_run", "Invocation")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static ARGUMENT_IDENTITY_JSON: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ARGUMENT_IDENTITY_FINGERPRINT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn argument_identity_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::ArgumentIdentity> {
        let py = obj.py();
            if obj.is_instance(ARGUMENT_IDENTITY_JSON.import(py, "livekit.agents._generated.agent_run", "ArgumentIdentityJson")?)? {
                return Ok(::lk_agents_abi::agent::ArgumentIdentity::Json(::lk_agents_abi::python::attr_with(obj, "ArgumentIdentityJson", "value", |v| ::lk_agents_abi::python::value_from_py(v, "ArgumentIdentityJson.value"))?));
            }
            if obj.is_instance(ARGUMENT_IDENTITY_FINGERPRINT.import(py, "livekit.agents._generated.agent_run", "ArgumentIdentityFingerprint")?)? {
                return Ok(::lk_agents_abi::agent::ArgumentIdentity::Fingerprint(::lk_agents_abi::python::attr_with(obj, "ArgumentIdentityFingerprint", "value", |v| v.extract::<::std::string::String>())?));
            }
            Err(super::not_a(obj, "ArgumentIdentityJson | ArgumentIdentityFingerprint"))
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static PREPARED_ARGUMENTS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn prepared_arguments_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::PreparedArguments> {
        let py = obj.py();
            if !obj.is_instance(PREPARED_ARGUMENTS.import(py, "livekit.agents._generated.agent_run", "PreparedArguments")?)? {
                return Err(super::not_a(obj, "PreparedArguments"));
            }
            Ok(::lk_agents_abi::agent::PreparedArguments {
                prepared: ::lk_agents_abi::python::attr_with(obj, "PreparedArguments", "prepared", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))?,
                identity: ::lk_agents_abi::python::attr_with(obj, "PreparedArguments", "identity", |v| crate::generated::records::agent_run::argument_identity_from_py(v))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static READ_BACK_ANSWER_QUESTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static READ_BACK_ANSWER_INPUT_REQUIRED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn read_back_answer_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::ReadBackAnswer> {
        let py = obj.py();
            if obj.is_instance(READ_BACK_ANSWER_QUESTION.import(py, "livekit.agents._generated.agent_run", "ReadBackAnswerQuestion")?)? {
                return Ok(::lk_agents_abi::agent::ReadBackAnswer::Question(::lk_agents_abi::python::attr_with(obj, "ReadBackAnswerQuestion", "question", |v| v.extract::<::std::string::String>())?));
            }
            if obj.is_instance(READ_BACK_ANSWER_INPUT_REQUIRED.import(py, "livekit.agents._generated.agent_run", "ReadBackAnswerInputRequired")?)? {
                return Ok(::lk_agents_abi::agent::ReadBackAnswer::InputRequired { question: ::lk_agents_abi::python::attr_with(obj, "ReadBackAnswerInputRequired", "question", |v| v.extract::<::std::string::String>())?, verbatim: ::lk_agents_abi::python::attr_with(obj, "ReadBackAnswerInputRequired", "verbatim", |v| v.extract::<bool>())? });
            }
            Err(super::not_a(obj, "ReadBackAnswerQuestion | ReadBackAnswerInputRequired"))
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static OUTPUT_CANDIDATE_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static OUTPUT_CANDIDATE_VALUE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn output_candidate_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::OutputCandidate) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::agent::OutputCandidate::Text(text) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                    crate::host::construct(OUTPUT_CANDIDATE_TEXT.import(py, "livekit.agents._generated.agent_run", "OutputCandidateText")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::OutputCandidate::Value(value) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("value", ::lk_agents_abi::python::value_to_py(py, &value)?)?;
                    crate::host::construct(OUTPUT_CANDIDATE_VALUE.import(py, "livekit.agents._generated.agent_run", "OutputCandidateValue")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static SERIALIZED_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn serialized_output_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::SerializedOutput> {
        let py = obj.py();
            if !obj.is_instance(SERIALIZED_OUTPUT.import(py, "livekit.agents._generated.agent_run", "SerializedOutput")?)? {
                return Err(super::not_a(obj, "SerializedOutput"));
            }
            Ok(::lk_agents_abi::agent::SerializedOutput {
                value: ::lk_agents_abi::python::attr_with(obj, "SerializedOutput", "value", |v| ::lk_agents_abi::python::value_from_py(v, "SerializedOutput.value"))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static VALIDATED_OUTPUT_ACCEPTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static VALIDATED_OUTPUT_RETRY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static VALIDATED_OUTPUT_FAILED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn validated_output_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::ValidatedOutput> {
        let py = obj.py();
            if obj.is_instance(VALIDATED_OUTPUT_ACCEPTED.import(py, "livekit.agents._generated.agent_run", "ValidatedOutputAccepted")?)? {
                return Ok(::lk_agents_abi::agent::ValidatedOutput::Accepted { text: ::lk_agents_abi::python::attr_with(obj, "ValidatedOutputAccepted", "text", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, value: ::lk_agents_abi::python::attr_with(obj, "ValidatedOutputAccepted", "value", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?, serialized: ::lk_agents_abi::python::attr_with(obj, "ValidatedOutputAccepted", "serialized", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::agent_run::serialized_output_from_py(v)).map(Some) })? });
            }
            if obj.is_instance(VALIDATED_OUTPUT_RETRY.import(py, "livekit.agents._generated.agent_run", "ValidatedOutputRetry")?)? {
                return Ok(::lk_agents_abi::agent::ValidatedOutput::Retry(::lk_agents_abi::python::attr_with(obj, "ValidatedOutputRetry", "reason", |v| v.extract::<::std::string::String>())?));
            }
            if obj.is_instance(VALIDATED_OUTPUT_FAILED.import(py, "livekit.agents._generated.agent_run", "ValidatedOutputFailed")?)? {
                return Ok(::lk_agents_abi::agent::ValidatedOutput::Failed(::lk_agents_abi::python::attr_with(obj, "ValidatedOutputFailed", "error", |v| v.extract::<::std::string::String>())?));
            }
            Err(super::not_a(obj, "ValidatedOutputAccepted | ValidatedOutputRetry | ValidatedOutputFailed"))
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static TOOL_TEMPLATE_DUPLICATE_REJECT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TOOL_TEMPLATE_DUPLICATE_CONFIRM: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TOOL_TEMPLATE_REPLY_AT_TAIL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TOOL_TEMPLATE_REPLY_MAYBE_COVERED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_template_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::ToolTemplate) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::agent::ToolTemplate::DuplicateReject { function_name, fnc_calls_json, fnc_calls_text } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("function_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(function_name, py)?)?;
                    kwargs.set_item("fnc_calls_json", ::pyo3::types::PyList::new(py, fnc_calls_json.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
                    kwargs.set_item("fnc_calls_text", ::pyo3::IntoPyObjectExt::into_bound_py_any(fnc_calls_text, py)?)?;
                    crate::host::construct(TOOL_TEMPLATE_DUPLICATE_REJECT.import(py, "livekit.agents._generated.agent_run", "ToolTemplateDuplicateReject")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::ToolTemplate::DuplicateConfirm { function_name, fnc_calls_json, fnc_calls_text } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("function_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(function_name, py)?)?;
                    kwargs.set_item("fnc_calls_json", ::pyo3::types::PyList::new(py, fnc_calls_json.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
                    kwargs.set_item("fnc_calls_text", ::pyo3::IntoPyObjectExt::into_bound_py_any(fnc_calls_text, py)?)?;
                    crate::host::construct(TOOL_TEMPLATE_DUPLICATE_CONFIRM.import(py, "livekit.agents._generated.agent_run", "ToolTemplateDuplicateConfirm")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::ToolTemplate::ReplyAtTail { call_ids } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("call_ids", ::pyo3::types::PyList::new(py, call_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
                    crate::host::construct(TOOL_TEMPLATE_REPLY_AT_TAIL.import(py, "livekit.agents._generated.agent_run", "ToolTemplateReplyAtTail")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::ToolTemplate::ReplyMaybeCovered { call_ids } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("call_ids", ::pyo3::types::PyList::new(py, call_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
                    crate::host::construct(TOOL_TEMPLATE_REPLY_MAYBE_COVERED.import(py, "livekit.agents._generated.agent_run", "ToolTemplateReplyMaybeCovered")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static AGENT_DEFINITION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn agent_definition_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::AgentDefinition> {
        let py = obj.py();
            if !obj.is_instance(AGENT_DEFINITION.import(py, "livekit.agents._generated.agent_run", "AgentDefinition")?)? {
                return Err(super::not_a(obj, "AgentDefinition"));
            }
            Ok(::lk_agents_abi::agent::AgentDefinition {
                name: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "name", |v| v.extract::<::std::string::String>())?,
                instructions: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "instructions", |v| v.extract::<::std::string::String>())?,
                dynamic_instructions: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "dynamic_instructions", |v| v.extract::<bool>())?,
                tools: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "tools", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::agent_run::function_tool_info_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                output: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "output", |v| crate::generated::records::agent_run::output_spec_from_py(v))?,
                retries: ::lk_agents_abi::python::attr_with(obj, "AgentDefinition", "retries", |v| crate::generated::records::agent_run::retries_from_py(v))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static FUNCTION_TOOL_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn function_tool_info_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::FunctionToolInfo> {
        let py = obj.py();
            if !obj.is_instance(FUNCTION_TOOL_INFO.import(py, "livekit.agents._generated.agent_run", "FunctionToolInfo")?)? {
                return Err(super::not_a(obj, "FunctionToolInfo"));
            }
            Ok(::lk_agents_abi::agent::FunctionToolInfo {
                spec: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "spec", |v| v.extract::<::lk_agents_abi::ToolSpec>().map_err(::std::convert::Into::<PyErr>::into))?,
                raw_schema: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "raw_schema", |v| v.extract::<bool>())?,
                needs_approval: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "needs_approval", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::agent_run::read_back_from_py(v)).map(Some) })?,
                on_duplicate: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "on_duplicate", |v| v.extract::<::lk_agents_abi::llm::mcp::DuplicateMode>().map_err(::std::convert::Into::<PyErr>::into))?,
                duplicate_scope: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "duplicate_scope", |v| v.extract::<::lk_agents_abi::llm::mcp::DuplicateScope>().map_err(::std::convert::Into::<PyErr>::into))?,
                retries: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "retries", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(v, "FunctionToolInfo.retries")).map(Some) })?,
                remote: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "remote", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.

    pub fn read_back_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::ReadBack> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "default" => Ok(::lk_agents_abi::agent::ReadBack::Default),
                "function" => Ok(::lk_agents_abi::agent::ReadBack::Function),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("ReadBack: {other:?} is not one of 'default', 'function'"))),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static OUTPUT_SPEC_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static OUTPUT_SPEC_STRUCTURED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn output_spec_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::OutputSpec> {
        let py = obj.py();
            if obj.is_instance(OUTPUT_SPEC_TEXT.import(py, "livekit.agents._generated.agent_run", "OutputSpecText")?)? {
                return Ok(::lk_agents_abi::agent::OutputSpec::Text { validated: ::lk_agents_abi::python::attr_with(obj, "OutputSpecText", "validated", |v| v.extract::<bool>())? });
            }
            if obj.is_instance(OUTPUT_SPEC_STRUCTURED.import(py, "livekit.agents._generated.agent_run", "OutputSpecStructured")?)? {
                return Ok(::lk_agents_abi::agent::OutputSpec::Structured { schema: ::lk_agents_abi::python::attr_with(obj, "OutputSpecStructured", "schema", |v| ::lk_agents_abi::python::value_from_py(v, "OutputSpecStructured.schema"))? });
            }
            Err(super::not_a(obj, "OutputSpecText | OutputSpecStructured"))
    }

    // Declared in rust/crates/lk-agents-abi/src/agent.rs.
    static RETRIES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn retries_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::Retries> {
        let py = obj.py();
            if !obj.is_instance(RETRIES.import(py, "livekit.agents._generated.agent_run", "Retries")?)? {
                return Err(super::not_a(obj, "Retries"));
            }
            Ok(::lk_agents_abi::agent::Retries {
                tools: ::lk_agents_abi::python::attr_with(obj, "Retries", "tools", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "Retries.tools"))?,
                output: ::lk_agents_abi::python::attr_with(obj, "Retries", "output", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "Retries.output"))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/assertions.rs.
    static EVENT_EXPECTATION_MESSAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static EVENT_EXPECTATION_FUNCTION_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static EVENT_EXPECTATION_FUNCTION_CALL_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn event_expectation_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::assertions::EventExpectation> {
        let py = obj.py();
            if obj.is_instance(EVENT_EXPECTATION_MESSAGE.import(py, "livekit.agents._generated.agent_run", "EventExpectationMessage")?)? {
                return Ok(::lk_agents_abi::agent::assertions::EventExpectation::Message { role: ::lk_agents_abi::python::attr_with(obj, "EventExpectationMessage", "role", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
            }
            if obj.is_instance(EVENT_EXPECTATION_FUNCTION_CALL.import(py, "livekit.agents._generated.agent_run", "EventExpectationFunctionCall")?)? {
                return Ok(::lk_agents_abi::agent::assertions::EventExpectation::FunctionCall { name: ::lk_agents_abi::python::attr_with(obj, "EventExpectationFunctionCall", "name", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, arguments: ::lk_agents_abi::python::attr_with(obj, "EventExpectationFunctionCall", "arguments", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "EventExpectationFunctionCall.arguments")).map(Some) })? });
            }
            if obj.is_instance(EVENT_EXPECTATION_FUNCTION_CALL_OUTPUT.import(py, "livekit.agents._generated.agent_run", "EventExpectationFunctionCallOutput")?)? {
                return Ok(::lk_agents_abi::agent::assertions::EventExpectation::FunctionCallOutput { output: ::lk_agents_abi::python::attr_with(obj, "EventExpectationFunctionCallOutput", "output", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, is_error: ::lk_agents_abi::python::attr_with(obj, "EventExpectationFunctionCallOutput", "is_error", |v| if v.is_none() { Ok(None) } else { (v.extract::<bool>()).map(Some) })? });
            }
            Err(super::not_a(obj, "EventExpectationMessage | EventExpectationFunctionCall | EventExpectationFunctionCallOutput"))
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
    static RUN_RESULT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_result_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::RunResult) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::agent::run::RunResult { outcome, interrupted, new_messages, all_messages, usage, verbatim } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("outcome", crate::generated::records::agent_run::run_outcome_to_py(py, outcome)?)?;
            kwargs.set_item("interrupted", ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted, py)?)?;
            kwargs.set_item("new_messages", ::pyo3::types::PyList::new(py, new_messages.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("all_messages", ::pyo3::types::PyList::new(py, all_messages.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("usage", crate::generated::records::agent_run::run_usage_to_py(py, usage)?)?;
            kwargs.set_item("verbatim", match verbatim { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(RUN_RESULT.import(py, "livekit.agents._generated.agent_run", "RunResult")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
    static RUN_OUTCOME_COMPLETED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static RUN_OUTCOME_INPUT_REQUIRED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_outcome_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::RunOutcome) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::agent::run::RunOutcome::Completed { output } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("output", crate::generated::records::agent_run::output_to_py(py, output)?)?;
                    crate::host::construct(RUN_OUTCOME_COMPLETED.import(py, "livekit.agents._generated.agent_run", "RunOutcomeCompleted")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::RunOutcome::InputRequired { question, verbatim, answer_tool } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
                    kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
                    kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    crate::host::construct(RUN_OUTCOME_INPUT_REQUIRED.import(py, "livekit.agents._generated.agent_run", "RunOutcomeInputRequired")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
    static OUTPUT_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static OUTPUT_VALIDATED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static OUTPUT_NOTHING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn output_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::Output) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::agent::run::Output::Text(text) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                    crate::host::construct(OUTPUT_TEXT.import(py, "livekit.agents._generated.agent_run", "OutputText")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::Output::Validated(_) => crate::host::construct(OUTPUT_VALIDATED.import(py, "livekit.agents._generated.agent_run", "OutputValidated")?.as_any(), (), None),
                ::lk_agents_abi::agent::run::Output::Nothing => crate::host::construct(OUTPUT_NOTHING.import(py, "livekit.agents._generated.agent_run", "OutputNothing")?.as_any(), (), None),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
    static RUN_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::RunUsage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::agent::run::RunUsage { requests, input_tokens, output_tokens, tool_calls } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("requests", ::pyo3::IntoPyObjectExt::into_bound_py_any(requests, py)?)?;
            kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
            kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
            kwargs.set_item("tool_calls", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_calls, py)?)?;
            crate::host::construct(RUN_USAGE.import(py, "livekit.agents._generated.agent_run", "RunUsage")?.as_any(), (), Some(&kwargs))
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

    pub fn run_error_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::RunError) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::agent::run::RunError::NoModel => crate::host::construct(RUN_ERROR_NO_MODEL.import(py, "livekit.agents._generated.agent_run", "RunErrorNoModel")?.as_any(), (), None),
                ::lk_agents_abi::agent::run::RunError::Cancelled => crate::host::construct(RUN_ERROR_CANCELLED.import(py, "livekit.agents._generated.agent_run", "RunErrorCancelled")?.as_any(), (), None),
                ::lk_agents_abi::agent::run::RunError::Model(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(RUN_ERROR_MODEL.import(py, "livekit.agents._generated.agent_run", "RunErrorModel")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::RunError::Host(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(RUN_ERROR_HOST.import(py, "livekit.agents._generated.agent_run", "RunErrorHost")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::RunError::ToolRetriesExceeded { tool, retries } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("tool", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool, py)?)?;
                    kwargs.set_item("retries", ::pyo3::IntoPyObjectExt::into_bound_py_any(retries, py)?)?;
                    crate::host::construct(RUN_ERROR_TOOL_RETRIES_EXCEEDED.import(py, "livekit.agents._generated.agent_run", "RunErrorToolRetriesExceeded")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::RunError::OutputRetriesExceeded { retries } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("retries", ::pyo3::IntoPyObjectExt::into_bound_py_any(retries, py)?)?;
                    crate::host::construct(RUN_ERROR_OUTPUT_RETRIES_EXCEEDED.import(py, "livekit.agents._generated.agent_run", "RunErrorOutputRetriesExceeded")?.as_any(), (), Some(&kwargs))
                },
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

    pub fn stream_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::StreamEvent) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::agent::run::StreamEvent::TextDelta { message_id, delta } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
                    kwargs.set_item("delta", ::pyo3::IntoPyObjectExt::into_bound_py_any(delta, py)?)?;
                    crate::host::construct(STREAM_EVENT_TEXT_DELTA.import(py, "livekit.agents._generated.agent_run", "StreamEventTextDelta")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::StreamEvent::TextEnd { message_id, interrupted } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
                    kwargs.set_item("interrupted", ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted, py)?)?;
                    crate::host::construct(STREAM_EVENT_TEXT_END.import(py, "livekit.agents._generated.agent_run", "StreamEventTextEnd")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::StreamEvent::ToolCall { call_id, tool_name, arguments } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
                    kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
                    kwargs.set_item("arguments", ::lk_agents_abi::python::value_to_py(py, &arguments)?)?;
                    crate::host::construct(STREAM_EVENT_TOOL_CALL.import(py, "livekit.agents._generated.agent_run", "StreamEventToolCall")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::StreamEvent::ToolResult { call_id, tool_name, status, output } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
                    kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
                    kwargs.set_item("status", crate::generated::records::agent_run::tool_result_status_to_py(py, status)?)?;
                    kwargs.set_item("output", ::pyo3::IntoPyObjectExt::into_bound_py_any(output, py)?)?;
                    crate::host::construct(STREAM_EVENT_TOOL_RESULT.import(py, "livekit.agents._generated.agent_run", "StreamEventToolResult")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::StreamEvent::Progress { call_id, tool_name, message } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
                    kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(STREAM_EVENT_PROGRESS.import(py, "livekit.agents._generated.agent_run", "StreamEventProgress")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.

    pub fn tool_result_status_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::ToolResultStatus) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::agent::run::ToolResultStatus::Completed => "completed",
                ::lk_agents_abi::agent::run::ToolResultStatus::Failed => "failed",
                ::lk_agents_abi::agent::run::ToolResultStatus::Cancelled => "cancelled",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
    static RUN_REQUEST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_request_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::agent::run::RunRequest> {
        let py = obj.py();
            if !obj.is_instance(RUN_REQUEST.import(py, "livekit.agents._generated.agent_run", "RunRequest")?)? {
                return Err(super::not_a(obj, "RunRequest"));
            }
            Ok(::lk_agents_abi::agent::run::RunRequest {
                invocation_id: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "invocation_id", |v| v.extract::<::std::string::String>())?,
                execution_id: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "execution_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                message: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "message", |v| v.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into))?,
                history: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "history", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
                answer_tool: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "answer_tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                delegated: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "delegated", |v| v.extract::<bool>())?,
                conn_options: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "conn_options", |v| v.extract::<::lk_agents_abi::ApiConnectOptions>().map_err(::std::convert::Into::<PyErr>::into))?,
                host_context: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "host_context", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))?,
                tools_host: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "tools_host", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
    static RUN_SETTLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_settled_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::RunSettled) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::agent::run::RunSettled { result, output } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("result", crate::generated::records::agent_run::run_result_to_py(py, result)?)?;
            kwargs.set_item("output", match output { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(RUN_SETTLED.import(py, "livekit.agents._generated.agent_run", "RunSettled")?.as_any(), (), Some(&kwargs))
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

    pub fn run_update_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::RunUpdate) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::agent::run::RunUpdate::Stream(event) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("event", crate::generated::records::agent_run::stream_event_to_py(py, event)?)?;
                    crate::host::construct(RUN_UPDATE_STREAM.import(py, "livekit.agents._generated.agent_run", "RunUpdateStream")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::RunUpdate::IncompleteCleanup(pending) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("pending", ::pyo3::types::PyList::new(py, pending.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
                    crate::host::construct(RUN_UPDATE_INCOMPLETE_CLEANUP.import(py, "livekit.agents._generated.agent_run", "RunUpdateIncompleteCleanup")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::RunUpdate::Finished(settled) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("settled", crate::generated::records::agent_run::run_settled_to_py(py, settled)?)?;
                    crate::host::construct(RUN_UPDATE_FINISHED.import(py, "livekit.agents._generated.agent_run", "RunUpdateFinished")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::RunUpdate::Failed(error) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("error", crate::generated::records::agent_run::run_error_to_py(py, error)?)?;
                    crate::host::construct(RUN_UPDATE_FAILED.import(py, "livekit.agents._generated.agent_run", "RunUpdateFailed")?.as_any(), (), Some(&kwargs))
                },
            }
    }
}

pub mod amd {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/amd.rs.
    static A_M_D_PREDICTION_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn a_m_d_prediction_event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::amd::AmdPrediction> {
        let py = obj.py();
            if !obj.is_instance(A_M_D_PREDICTION_EVENT.import(py, "livekit.agents._generated.amd", "AMDPredictionEvent")?)? {
                return Err(super::not_a(obj, "AMDPredictionEvent"));
            }
            Ok(::lk_agents_abi::voice::amd::AmdPrediction {
                speech_duration: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "speech_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "AMDPredictionEvent.speech_duration").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("AMDPredictionEvent.speech_duration: {e}")))))?,
                category: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "category", |v| v.extract::<::lk_agents_abi::voice::amd::AmdCategory>().map_err(::std::convert::Into::<PyErr>::into))?,
                reason: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "reason", |v| v.extract::<::lk_agents_abi::voice::amd::AmdReason>().map_err(::std::convert::Into::<PyErr>::into))?,
                transcript: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "transcript", |v| v.extract::<::std::string::String>())?,
                delay: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "delay", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "AMDPredictionEvent.delay").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("AMDPredictionEvent.delay: {e}")))))?,
            })
    }

    pub fn a_m_d_prediction_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::amd::AmdPrediction) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::amd::AmdPrediction { speech_duration, category, reason, transcript, delay } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("speech_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(speech_duration.as_secs_f64(), py)?)?;
            kwargs.set_item("category", ::pyo3::IntoPyObjectExt::into_bound_py_any(category, py)?)?;
            kwargs.set_item("reason", ::pyo3::IntoPyObjectExt::into_bound_py_any(reason, py)?)?;
            kwargs.set_item("transcript", ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript, py)?)?;
            kwargs.set_item("delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(delay.as_secs_f64(), py)?)?;
            crate::host::construct(A_M_D_PREDICTION_EVENT.import(py, "livekit.agents._generated.amd", "AMDPredictionEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/amd.rs.

    pub fn amd_constants_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::amd::AmdConstants) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::amd::AmdConstants { default_llm_model, default_stt_model, prompt, evaluated_llm_models, evaluated_stt_models, human_speech_threshold, human_silence_threshold, machine_silence_threshold, no_speech_threshold, detection_timeout, max_endpointing_delay, max_extensions, max_extension_secs } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("default_llm_model", ::pyo3::IntoPyObjectExt::into_bound_py_any(default_llm_model, py)?)?;
            dict.set_item("default_stt_model", ::pyo3::IntoPyObjectExt::into_bound_py_any(default_stt_model, py)?)?;
            dict.set_item("prompt", ::pyo3::IntoPyObjectExt::into_bound_py_any(prompt, py)?)?;
            dict.set_item("evaluated_llm_models", ::pyo3::types::PyList::new(py, evaluated_llm_models.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            dict.set_item("evaluated_stt_models", ::pyo3::types::PyList::new(py, evaluated_stt_models.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            dict.set_item("human_speech_threshold", ::pyo3::IntoPyObjectExt::into_bound_py_any(human_speech_threshold, py)?)?;
            dict.set_item("human_silence_threshold", ::pyo3::IntoPyObjectExt::into_bound_py_any(human_silence_threshold, py)?)?;
            dict.set_item("machine_silence_threshold", ::pyo3::IntoPyObjectExt::into_bound_py_any(machine_silence_threshold, py)?)?;
            dict.set_item("no_speech_threshold", ::pyo3::IntoPyObjectExt::into_bound_py_any(no_speech_threshold, py)?)?;
            dict.set_item("detection_timeout", ::pyo3::IntoPyObjectExt::into_bound_py_any(detection_timeout, py)?)?;
            dict.set_item("max_endpointing_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_endpointing_delay, py)?)?;
            dict.set_item("max_extensions", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_extensions, py)?)?;
            dict.set_item("max_extension_secs", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_extension_secs, py)?)?;
            Ok(dict.into_any())
    }
}

pub mod avatar {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/utils/video.rs.

    pub fn audio_receiver_frame_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::utils::video::AudioReceiverFrame> {
        if obj.is_instance(&obj.py().import("livekit.rtc")?.getattr("AudioFrame")?)? {
            return Ok(::lk_agents_abi::utils::video::AudioReceiverFrame::Audio(obj.extract::<::lk_agents_abi::AudioFrame>().map_err(::std::convert::Into::<PyErr>::into)?));
        }
        if obj.is_instance(&obj.py().import("livekit.agents.voice.avatar")?.getattr("AudioSegmentEnd")?)? {
            return Ok(::lk_agents_abi::utils::video::AudioReceiverFrame::SegmentEnd);
        }
        Err(super::not_a(obj, "AudioReceiverFrame"))
    }

    pub fn audio_receiver_frame_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::utils::video::AudioReceiverFrame) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::utils::video::AudioReceiverFrame::Audio(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
            ::lk_agents_abi::utils::video::AudioReceiverFrame::SegmentEnd => crate::host::construct(py.import("livekit.agents.voice.avatar")?.getattr("AudioSegmentEnd")?.as_any(), (), None)?,
        })
    }
}

pub mod background_audio_wire {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
    static SOUND_SOURCE_FILE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static SOUND_SOURCE_STREAM: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn sound_source_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::background_audio::SoundSource> {
        let py = obj.py();
            if obj.is_instance(SOUND_SOURCE_FILE.import(py, "livekit.agents._generated.background_audio_wire", "SoundSourceFile")?)? {
                return Ok(::lk_agents_abi::voice::background_audio::SoundSource::File(::lk_agents_abi::python::attr_with(obj, "SoundSourceFile", "path", |v| v.extract::<::std::string::String>())?));
            }
            if obj.is_instance(SOUND_SOURCE_STREAM.import(py, "livekit.agents._generated.background_audio_wire", "SoundSourceStream")?)? {
                return Ok(::lk_agents_abi::voice::background_audio::SoundSource::Stream(::lk_agents_abi::python::attr_with(obj, "SoundSourceStream", "stream", |v| crate::generated::voice::FrameStreamProxy::from_host(&v))?));
            }
            Err(super::not_a(obj, "SoundSourceFile | SoundSourceStream"))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
    static SOUND: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn sound_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::background_audio::Sound> {
        let py = obj.py();
            if !obj.is_instance(SOUND.import(py, "livekit.agents._generated.background_audio_wire", "Sound")?)? {
                return Err(super::not_a(obj, "Sound"));
            }
            Ok(::lk_agents_abi::voice::background_audio::Sound {
                source: ::lk_agents_abi::python::attr_with(obj, "Sound", "source", |v| crate::generated::records::background_audio_wire::sound_source_from_py(v))?,
                volume: ::lk_agents_abi::python::attr_with(obj, "Sound", "volume", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "Sound.volume"))?,
                probability: ::lk_agents_abi::python::attr_with(obj, "Sound", "probability", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "Sound.probability"))?,
                fade_in: ::lk_agents_abi::python::attr_with(obj, "Sound", "fade_in", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "Sound.fade_in"))?,
                fade_out: ::lk_agents_abi::python::attr_with(obj, "Sound", "fade_out", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "Sound.fade_out"))?,
            })
    }
}

pub mod duplex {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.
    static DUPLEX_CAPABILITIES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn duplex_capabilities_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::duplex::DuplexCapabilities> {
        let py = obj.py();
            if !obj.is_instance(DUPLEX_CAPABILITIES.import(py, "livekit.agents._generated.duplex", "DuplexCapabilities")?)? {
                return Err(super::not_a(obj, "DuplexCapabilities"));
            }
            Ok(::lk_agents_abi::llm::duplex::DuplexCapabilities {
                user_transcription: ::lk_agents_abi::python::attr_with(obj, "DuplexCapabilities", "user_transcription", |v| v.extract::<bool>())?,
                auto_tool_reply_generation: ::lk_agents_abi::python::attr_with(obj, "DuplexCapabilities", "auto_tool_reply_generation", |v| v.extract::<bool>())?,
                mutable_chat_context: ::lk_agents_abi::python::attr_with(obj, "DuplexCapabilities", "mutable_chat_context", |v| v.extract::<bool>())?,
                mutable_instructions: ::lk_agents_abi::python::attr_with(obj, "DuplexCapabilities", "mutable_instructions", |v| v.extract::<bool>())?,
                mutable_tools: ::lk_agents_abi::python::attr_with(obj, "DuplexCapabilities", "mutable_tools", |v| v.extract::<bool>())?,
            })
    }

    pub fn duplex_capabilities_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::duplex::DuplexCapabilities) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::duplex::DuplexCapabilities { user_transcription, auto_tool_reply_generation, mutable_chat_context, mutable_instructions, mutable_tools } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("user_transcription", ::pyo3::IntoPyObjectExt::into_bound_py_any(user_transcription, py)?)?;
            kwargs.set_item("auto_tool_reply_generation", ::pyo3::IntoPyObjectExt::into_bound_py_any(auto_tool_reply_generation, py)?)?;
            kwargs.set_item("mutable_chat_context", ::pyo3::IntoPyObjectExt::into_bound_py_any(mutable_chat_context, py)?)?;
            kwargs.set_item("mutable_instructions", ::pyo3::IntoPyObjectExt::into_bound_py_any(mutable_instructions, py)?)?;
            kwargs.set_item("mutable_tools", ::pyo3::IntoPyObjectExt::into_bound_py_any(mutable_tools, py)?)?;
            crate::host::construct(DUPLEX_CAPABILITIES.import(py, "livekit.agents._generated.duplex", "DuplexCapabilities")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.
    static DUPLEX_AUDIO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn duplex_audio_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::duplex::DuplexAudio> {
        let py = obj.py();
            if !obj.is_instance(DUPLEX_AUDIO.import(py, "livekit.agents._generated.duplex", "DuplexAudio")?)? {
                return Err(super::not_a(obj, "DuplexAudio"));
            }
            Ok(::lk_agents_abi::llm::duplex::DuplexAudio {
                frame: ::lk_agents_abi::python::attr_with(obj, "DuplexAudio", "frame", |v| v.extract::<::lk_agents_abi::AudioFrame>().map_err(::std::convert::Into::<PyErr>::into))?,
                start_ms: ::lk_agents_abi::python::attr_with(obj, "DuplexAudio", "start_ms", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<i64>(v, "DuplexAudio.start_ms")).map(Some) })?,
            })
    }

    pub fn duplex_audio_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::duplex::DuplexAudio) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::duplex::DuplexAudio { frame, start_ms } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("frame", ::pyo3::IntoPyObjectExt::into_bound_py_any(frame, py)?)?;
            kwargs.set_item("start_ms", match start_ms { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(DUPLEX_AUDIO.import(py, "livekit.agents._generated.duplex", "DuplexAudio")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.
    static DUPLEX_TRANSCRIPT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn duplex_transcript_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::duplex::DuplexTranscript> {
        let py = obj.py();
            if !obj.is_instance(DUPLEX_TRANSCRIPT.import(py, "livekit.agents._generated.duplex", "DuplexTranscript")?)? {
                return Err(super::not_a(obj, "DuplexTranscript"));
            }
            Ok(::lk_agents_abi::llm::duplex::DuplexTranscript {
                text: ::lk_agents_abi::python::attr_with(obj, "DuplexTranscript", "text", |v| v.extract::<::std::string::String>())?,
                start_ms: ::lk_agents_abi::python::attr_with(obj, "DuplexTranscript", "start_ms", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<i64>(v, "DuplexTranscript.start_ms")).map(Some) })?,
                end_ms: ::lk_agents_abi::python::attr_with(obj, "DuplexTranscript", "end_ms", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<i64>(v, "DuplexTranscript.end_ms")).map(Some) })?,
            })
    }

    pub fn duplex_transcript_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::duplex::DuplexTranscript) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::duplex::DuplexTranscript { text, start_ms, end_ms } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
            kwargs.set_item("start_ms", match start_ms { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("end_ms", match end_ms { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(DUPLEX_TRANSCRIPT.import(py, "livekit.agents._generated.duplex", "DuplexTranscript")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.

    pub fn duplex_event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::duplex::DuplexEvent> {
        if obj.is_instance(&obj.py().import("livekit.agents._generated.duplex")?.getattr("DuplexTranscript")?)? {
            return Ok(::lk_agents_abi::llm::duplex::DuplexEvent::Transcript(crate::generated::records::duplex::duplex_transcript_from_py(obj)?));
        }
        if obj.is_instance_of::<::pyo3::types::PyDict>() {
            return Ok(::lk_agents_abi::llm::duplex::DuplexEvent::Session(obj.extract::<::lk_agents_abi::llm::realtime::Event>().map_err(::std::convert::Into::<PyErr>::into)?));
        }
        Err(super::not_a(obj, "DuplexEvent"))
    }
}

pub mod evals {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/evals.rs.
    static JUDGMENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn judgment_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::evals::Judgment> {
        let py = obj.py();
            if !obj.is_instance(JUDGMENT.import(py, "livekit.agents._generated.evals", "Judgment")?)? {
                return Err(super::not_a(obj, "Judgment"));
            }
            Ok(::lk_agents_abi::evals::Judgment {
                verdict: ::lk_agents_abi::python::attr_with(obj, "Judgment", "verdict", |v| v.extract::<::std::string::String>())?,
                reasoning: ::lk_agents_abi::python::attr_with(obj, "Judgment", "reasoning", |v| v.extract::<::std::string::String>())?,
                instructions: ::lk_agents_abi::python::attr_with(obj, "Judgment", "instructions", |v| v.extract::<::std::string::String>())?,
            })
    }

    pub fn judgment_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::evals::Judgment) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::evals::Judgment { verdict, reasoning, instructions } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("verdict", ::pyo3::IntoPyObjectExt::into_bound_py_any(verdict, py)?)?;
            kwargs.set_item("reasoning", ::pyo3::IntoPyObjectExt::into_bound_py_any(reasoning, py)?)?;
            kwargs.set_item("instructions", ::pyo3::IntoPyObjectExt::into_bound_py_any(instructions, py)?)?;
            crate::host::construct(JUDGMENT.import(py, "livekit.agents._generated.evals", "Judgment")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/evals.rs.
    static JUDGE_SPEC_BUILT_IN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static JUDGE_SPEC_HOST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn judge_spec_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::evals::JudgeSpec> {
        let py = obj.py();
            if obj.is_instance(JUDGE_SPEC_BUILT_IN.import(py, "livekit.agents._generated.evals", "JudgeSpecBuiltIn")?)? {
                return Ok(::lk_agents_abi::evals::JudgeSpec::BuiltIn { name: ::lk_agents_abi::python::attr_with(obj, "JudgeSpecBuiltIn", "name", |v| v.extract::<::std::string::String>())? });
            }
            if obj.is_instance(JUDGE_SPEC_HOST.import(py, "livekit.agents._generated.evals", "JudgeSpecHost")?)? {
                return Ok(::lk_agents_abi::evals::JudgeSpec::Host { name: ::lk_agents_abi::python::attr_with(obj, "JudgeSpecHost", "name", |v| v.extract::<::std::string::String>())?, evaluator: ::lk_agents_abi::python::attr_with(obj, "JudgeSpecHost", "evaluator", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))? });
            }
            Err(super::not_a(obj, "JudgeSpecBuiltIn | JudgeSpecHost"))
    }

    // Declared in rust/crates/lk-agents-abi/src/evals.rs.
    static JUDGE_OUTCOME_JUDGED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static JUDGE_OUTCOME_FAILED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn judge_outcome_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::evals::JudgeOutcome) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::evals::JudgeOutcome::Judged { name, judgment } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
                    kwargs.set_item("judgment", crate::generated::records::evals::judgment_to_py(py, judgment)?)?;
                    crate::host::construct(JUDGE_OUTCOME_JUDGED.import(py, "livekit.agents._generated.evals", "JudgeOutcomeJudged")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::evals::JudgeOutcome::Failed { name, error } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
                    kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
                    crate::host::construct(JUDGE_OUTCOME_FAILED.import(py, "livekit.agents._generated.evals", "JudgeOutcomeFailed")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/evals.rs.
    static INTENT_JUDGMENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn intent_judgment_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::evals::IntentJudgment) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::evals::IntentJudgment { success, reason, usage } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("success", ::pyo3::IntoPyObjectExt::into_bound_py_any(success, py)?)?;
            kwargs.set_item("reason", ::pyo3::IntoPyObjectExt::into_bound_py_any(reason, py)?)?;
            kwargs.set_item("usage", match usage { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(INTENT_JUDGMENT.import(py, "livekit.agents._generated.evals", "IntentJudgment")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/evals.rs.
    static EVALUATION_SCORE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn evaluation_score_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::evals::EvaluationScore) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::evals::EvaluationScore { score, all_passed, any_passed, majority_passed, none_failed } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("score", ::pyo3::IntoPyObjectExt::into_bound_py_any(score, py)?)?;
            kwargs.set_item("all_passed", ::pyo3::IntoPyObjectExt::into_bound_py_any(all_passed, py)?)?;
            kwargs.set_item("any_passed", ::pyo3::IntoPyObjectExt::into_bound_py_any(any_passed, py)?)?;
            kwargs.set_item("majority_passed", ::pyo3::IntoPyObjectExt::into_bound_py_any(majority_passed, py)?)?;
            kwargs.set_item("none_failed", ::pyo3::IntoPyObjectExt::into_bound_py_any(none_failed, py)?)?;
            crate::host::construct(EVALUATION_SCORE.import(py, "livekit.agents._generated.evals", "EvaluationScore")?.as_any(), (), Some(&kwargs))
    }
}

pub mod events {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static USER_STATE_CHANGED_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn user_state_changed_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::UserStateChanged) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::UserStateChanged { old_state, new_state, created_at } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("old_state", ::pyo3::IntoPyObjectExt::into_bound_py_any(old_state, py)?)?;
            kwargs.set_item("new_state", ::pyo3::IntoPyObjectExt::into_bound_py_any(new_state, py)?)?;
            kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(created_at, py)?)?;
            crate::host::construct(USER_STATE_CHANGED_EVENT.import(py, "livekit.agents._generated.events", "UserStateChangedEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static AGENT_STATE_CHANGED_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn agent_state_changed_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::AgentStateChanged) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::AgentStateChanged { old_state, new_state, created_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("old_state", ::pyo3::IntoPyObjectExt::into_bound_py_any(old_state, py)?)?;
            kwargs.set_item("new_state", ::pyo3::IntoPyObjectExt::into_bound_py_any(new_state, py)?)?;
            crate::host::construct(AGENT_STATE_CHANGED_EVENT.import(py, "livekit.agents._generated.events", "AgentStateChangedEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static USER_INPUT_TRANSCRIBED_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn user_input_transcribed_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::UserInputTranscribed) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::UserInputTranscribed { transcript, is_final, item_id, speaker_id, language, created_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("transcript", ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript, py)?)?;
            kwargs.set_item("is_final", ::pyo3::IntoPyObjectExt::into_bound_py_any(is_final, py)?)?;
            kwargs.set_item("item_id", match item_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("speaker_id", match speaker_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("language", match language { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(USER_INPUT_TRANSCRIBED_EVENT.import(py, "livekit.agents._generated.events", "UserInputTranscribedEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static USER_TRANSCRIPTION_TIMEOUT_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn user_transcription_timeout_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::UserTranscriptionTimeout) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::UserTranscriptionTimeout { speech_duration, vad_speech_started_at, created_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("speech_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(speech_duration, py)?)?;
            kwargs.set_item("vad_speech_started_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(vad_speech_started_at, py)?)?;
            crate::host::construct(USER_TRANSCRIPTION_TIMEOUT_EVENT.import(py, "livekit.agents._generated.events", "UserTranscriptionTimeoutEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static EOT_PREDICTION_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn eot_prediction_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::EotPrediction) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::EotPrediction { probability, threshold, inference_duration, delay, created_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("probability", ::pyo3::IntoPyObjectExt::into_bound_py_any(probability, py)?)?;
            kwargs.set_item("threshold", ::pyo3::IntoPyObjectExt::into_bound_py_any(threshold, py)?)?;
            kwargs.set_item("inference_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(inference_duration, py)?)?;
            kwargs.set_item("delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(delay, py)?)?;
            crate::host::construct(EOT_PREDICTION_EVENT.import(py, "livekit.agents._generated.events", "EotPredictionEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static __AGENT_BACKCHANNEL_OPPORTUNITY_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn __agent_backchannel_opportunity_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::AgentBackchannelOpportunityEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::AgentBackchannelOpportunityEvent { probability, threshold, end_of_turn_probability, end_of_turn_threshold, language, created_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("probability", ::pyo3::IntoPyObjectExt::into_bound_py_any(probability, py)?)?;
            kwargs.set_item("threshold", ::pyo3::IntoPyObjectExt::into_bound_py_any(threshold, py)?)?;
            kwargs.set_item("end_of_turn_probability", ::pyo3::IntoPyObjectExt::into_bound_py_any(end_of_turn_probability, py)?)?;
            kwargs.set_item("end_of_turn_threshold", ::pyo3::IntoPyObjectExt::into_bound_py_any(end_of_turn_threshold, py)?)?;
            kwargs.set_item("language", match language { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(__AGENT_BACKCHANNEL_OPPORTUNITY_EVENT.import(py, "livekit.agents._generated.events", "_AgentBackchannelOpportunityEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static AGENT_FALSE_INTERRUPTION_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn agent_false_interruption_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::AgentFalseInterruption) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::AgentFalseInterruption { resumed, created_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("resumed", ::pyo3::IntoPyObjectExt::into_bound_py_any(resumed, py)?)?;
            crate::host::construct(AGENT_FALSE_INTERRUPTION_EVENT.import(py, "livekit.agents._generated.events", "AgentFalseInterruptionEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static TOOL_CALL_UPDATED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_call_updated_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ToolCallUpdated) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ToolCallUpdated { id, call_id, message } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
            kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
            crate::host::construct(TOOL_CALL_UPDATED.import(py, "livekit.agents._generated.events", "ToolCallUpdated")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static TOOL_CALL_ENDED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_call_ended_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ToolCallEnded) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ToolCallEnded { id, call_id, message, status } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
            kwargs.set_item("message", match message { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("status", ::pyo3::IntoPyObjectExt::into_bound_py_any(status, py)?)?;
            crate::host::construct(TOOL_CALL_ENDED.import(py, "livekit.agents._generated.events", "ToolCallEnded")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static TOOL_REPLY_UPDATED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_reply_updated_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ToolReplyUpdated) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ToolReplyUpdated { update_ids, status, speech_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("update_ids", ::pyo3::types::PyList::new(py, update_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("status", ::pyo3::IntoPyObjectExt::into_bound_py_any(status, py)?)?;
            kwargs.set_item("speech_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(speech_id, py)?)?;
            crate::host::construct(TOOL_REPLY_UPDATED.import(py, "livekit.agents._generated.events", "ToolReplyUpdated")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static USER_TURN_EXCEEDED_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn user_turn_exceeded_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::UserTurnExceededEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::UserTurnExceededEvent { transcript, accumulated_transcript, accumulated_word_count, duration, created_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("transcript", ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript, py)?)?;
            kwargs.set_item("accumulated_transcript", ::pyo3::IntoPyObjectExt::into_bound_py_any(accumulated_transcript, py)?)?;
            kwargs.set_item("accumulated_word_count", ::pyo3::IntoPyObjectExt::into_bound_py_any(accumulated_word_count, py)?)?;
            kwargs.set_item("duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(duration.as_secs_f64(), py)?)?;
            crate::host::construct(USER_TURN_EXCEEDED_EVENT.import(py, "livekit.agents._generated.events", "UserTurnExceededEvent")?.as_any(), (), Some(&kwargs))
    }
}

pub mod inference_stt {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/inference/fallback.rs.

    pub fn fallback_model_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::inference::fallback::SttFallbackModel> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::inference::fallback::SttFallbackModel {
                model: { let v = obj.get_item("model")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("FallbackModel.model"))?; ((&v).extract::<::std::string::String>())? },
                extra_kwargs: match obj.get_item("extra_kwargs")? { Some(v) if !v.is_none() => Some((match ::lk_agents_abi::python::value_from_py((&v), "FallbackModel.extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FallbackModel.extra_kwargs: expected a dict")) })?), _ => None },
            })
    }

    pub fn fallback_model_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::inference::fallback::SttFallbackModel) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::inference::fallback::SttFallbackModel { model, extra_kwargs } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("model", ::pyo3::IntoPyObjectExt::into_bound_py_any(model, py)?)?;
            if let Some(x) = extra_kwargs {
                dict.set_item("extra_kwargs", ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any())?;
            }
            Ok(dict.into_any())
    }
}

pub mod inference_tts {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/inference/fallback.rs.

    pub fn fallback_model_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::inference::fallback::TtsFallbackModel> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::inference::fallback::TtsFallbackModel {
                model: { let v = obj.get_item("model")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("FallbackModel.model"))?; ((&v).extract::<::std::string::String>())? },
                voice: match obj.get_item("voice")? { Some(v) if !v.is_none() => ((&v).extract::<::std::string::String>())?, _ => "".to_owned() },
                extra_kwargs: match obj.get_item("extra_kwargs")? { Some(v) if !v.is_none() => Some((match ::lk_agents_abi::python::value_from_py((&v), "FallbackModel.extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FallbackModel.extra_kwargs: expected a dict")) })?), _ => None },
            })
    }

    pub fn fallback_model_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::inference::fallback::TtsFallbackModel) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::inference::fallback::TtsFallbackModel { model, voice, extra_kwargs } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("model", ::pyo3::IntoPyObjectExt::into_bound_py_any(model, py)?)?;
            dict.set_item("voice", ::pyo3::IntoPyObjectExt::into_bound_py_any(voice, py)?)?;
            if let Some(x) = extra_kwargs {
                dict.set_item("extra_kwargs", ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any())?;
            }
            Ok(dict.into_any())
    }
}

pub mod io_events {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
    static PLAYBACK_SEGMENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn playback_segment_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::io::PlaybackSegment> {
        let py = obj.py();
            if !obj.is_instance(PLAYBACK_SEGMENT.import(py, "livekit.agents._generated.io_events", "PlaybackSegment")?)? {
                return Err(super::not_a(obj, "PlaybackSegment"));
            }
            Ok(::lk_agents_abi::voice::io::PlaybackSegment {
                attachment: ::lk_agents_abi::python::attr_with(obj, "PlaybackSegment", "attachment", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "PlaybackSegment.attachment"))?,
                segment: ::lk_agents_abi::python::attr_with(obj, "PlaybackSegment", "segment", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "PlaybackSegment.segment"))?,
            })
    }

    pub fn playback_segment_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::io::PlaybackSegment) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::io::PlaybackSegment { attachment, segment } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("attachment", ::pyo3::IntoPyObjectExt::into_bound_py_any(attachment, py)?)?;
            kwargs.set_item("segment", ::pyo3::IntoPyObjectExt::into_bound_py_any(segment, py)?)?;
            crate::host::construct(PLAYBACK_SEGMENT.import(py, "livekit.agents._generated.io_events", "PlaybackSegment")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
    static AVATAR_PLAYBACK_SEGMENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn avatar_playback_segment_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::io::AvatarPlaybackSegment> {
        let py = obj.py();
            if !obj.is_instance(AVATAR_PLAYBACK_SEGMENT.import(py, "livekit.agents._generated.io_events", "AvatarPlaybackSegment")?)? {
                return Err(super::not_a(obj, "AvatarPlaybackSegment"));
            }
            Ok(::lk_agents_abi::voice::io::AvatarPlaybackSegment {
                playback: ::lk_agents_abi::python::attr_with(obj, "AvatarPlaybackSegment", "playback", |v| crate::generated::records::io_events::playback_segment_from_py(v))?,
                stream_id: ::lk_agents_abi::python::attr_with(obj, "AvatarPlaybackSegment", "stream_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    pub fn avatar_playback_segment_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::io::AvatarPlaybackSegment) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::io::AvatarPlaybackSegment { playback, stream_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("playback", crate::generated::records::io_events::playback_segment_to_py(py, playback)?)?;
            kwargs.set_item("stream_id", match stream_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(AVATAR_PLAYBACK_SEGMENT.import(py, "livekit.agents._generated.io_events", "AvatarPlaybackSegment")?.as_any(), (), Some(&kwargs))
    }
}

pub mod keyterms {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/turn/keyterms.rs.
    static RESOLVED_KEYTERM_DETECTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn resolved_keyterm_detection_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::turn::keyterms::ResolvedKeytermDetection) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::turn::keyterms::ResolvedKeytermDetection { enabled, turn_interval, max_keyterms, timeout, instructions, model, tool_name } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("enabled", ::pyo3::IntoPyObjectExt::into_bound_py_any(enabled, py)?)?;
            kwargs.set_item("turn_interval", ::pyo3::IntoPyObjectExt::into_bound_py_any(turn_interval, py)?)?;
            kwargs.set_item("max_keyterms", match max_keyterms { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("timeout", ::pyo3::IntoPyObjectExt::into_bound_py_any(timeout, py)?)?;
            kwargs.set_item("instructions", ::pyo3::IntoPyObjectExt::into_bound_py_any(instructions, py)?)?;
            kwargs.set_item("model", match model { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
            crate::host::construct(RESOLVED_KEYTERM_DETECTION.import(py, "livekit.agents._generated.keyterms", "ResolvedKeytermDetection")?.as_any(), (), Some(&kwargs))
    }
}

pub mod llm_fallback {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/fallback.rs.
    static AVAILABILITY_CHANGED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn availability_changed_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::fallback::AvailabilityChanged) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::fallback::AvailabilityChanged { index, label, available } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("index", ::pyo3::IntoPyObjectExt::into_bound_py_any(index, py)?)?;
            kwargs.set_item("label", ::pyo3::IntoPyObjectExt::into_bound_py_any(label, py)?)?;
            kwargs.set_item("available", ::pyo3::IntoPyObjectExt::into_bound_py_any(available, py)?)?;
            crate::host::construct(AVAILABILITY_CHANGED.import(py, "livekit.agents._generated.llm_fallback", "AvailabilityChanged")?.as_any(), (), Some(&kwargs))
    }
}

pub mod mcp_session {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.

    pub fn mcp_cleanup_status_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::mcp::McpCleanupStatus) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::llm::mcp::McpCleanupStatus::Pending => "pending",
                ::lk_agents_abi::llm::mcp::McpCleanupStatus::Clean => "clean",
                ::lk_agents_abi::llm::mcp::McpCleanupStatus::Incomplete => "incomplete",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
    static MCP_CLEANUP: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn mcp_cleanup_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::mcp::McpCleanup) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::mcp::McpCleanup { status, service_settled, calls_settled, reader_settled, child_exit_code, child_exit_signal, child_reaped } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("status", crate::generated::records::mcp_session::mcp_cleanup_status_to_py(py, status)?)?;
            kwargs.set_item("service_settled", ::pyo3::IntoPyObjectExt::into_bound_py_any(service_settled, py)?)?;
            kwargs.set_item("calls_settled", ::pyo3::IntoPyObjectExt::into_bound_py_any(calls_settled, py)?)?;
            kwargs.set_item("reader_settled", ::pyo3::IntoPyObjectExt::into_bound_py_any(reader_settled, py)?)?;
            kwargs.set_item("child_exit_code", match child_exit_code { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("child_exit_signal", match child_exit_signal { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("child_reaped", ::pyo3::IntoPyObjectExt::into_bound_py_any(child_reaped, py)?)?;
            crate::host::construct(MCP_CLEANUP.import(py, "livekit.agents._generated.mcp_session", "McpCleanup")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
    static MCP_INVOCATION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn mcp_invocation_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::mcp::McpInvocation) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::mcp::McpInvocation { host_context } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("host_context", match host_context { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(MCP_INVOCATION.import(py, "livekit.agents._generated.mcp_session", "McpInvocation")?.as_any(), (), Some(&kwargs))
    }
}

pub mod realtime_options {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
    static REALTIME_JSON_OPTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_json_option_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime_native::RealtimeJsonOption> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_JSON_OPTION.import(py, "livekit.agents._generated.realtime_options", "RealtimeJsonOption")?)? {
                return Err(super::not_a(obj, "RealtimeJsonOption"));
            }
            Ok(::lk_agents_abi::llm::realtime_native::RealtimeJsonOption {
                value: ::lk_agents_abi::python::attr_with(obj, "RealtimeJsonOption", "value", |v| ::lk_agents_abi::python::value_from_py(v, "RealtimeJsonOption.value"))?,
            })
    }

    pub fn realtime_json_option_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::realtime_native::RealtimeJsonOption) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::realtime_native::RealtimeJsonOption { value } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("value", ::lk_agents_abi::python::value_to_py(py, &value)?)?;
            crate::host::construct(REALTIME_JSON_OPTION.import(py, "livekit.agents._generated.realtime_options", "RealtimeJsonOption")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
    static REALTIME_TOOL_CHOICE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_tool_choice_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime_native::RealtimeToolChoice> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_TOOL_CHOICE.import(py, "livekit.agents._generated.realtime_options", "RealtimeToolChoice")?)? {
                return Err(super::not_a(obj, "RealtimeToolChoice"));
            }
            Ok(::lk_agents_abi::llm::realtime_native::RealtimeToolChoice {
                value: ::lk_agents_abi::python::attr_with(obj, "RealtimeToolChoice", "value", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::ToolChoice>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            })
    }

    pub fn realtime_tool_choice_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::realtime_native::RealtimeToolChoice) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::realtime_native::RealtimeToolChoice { value } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("value", match value { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(REALTIME_TOOL_CHOICE.import(py, "livekit.agents._generated.realtime_options", "RealtimeToolChoice")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
    static REALTIME_DURATION_OPTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_duration_option_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime_native::RealtimeDurationOption> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_DURATION_OPTION.import(py, "livekit.agents._generated.realtime_options", "RealtimeDurationOption")?)? {
                return Err(super::not_a(obj, "RealtimeDurationOption"));
            }
            Ok(::lk_agents_abi::llm::realtime_native::RealtimeDurationOption {
                value: ::lk_agents_abi::python::attr_with(obj, "RealtimeDurationOption", "value", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeDurationOption.value")).map(Some) })?,
            })
    }

    pub fn realtime_duration_option_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::realtime_native::RealtimeDurationOption) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::realtime_native::RealtimeDurationOption { value } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("value", match value { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(REALTIME_DURATION_OPTION.import(py, "livekit.agents._generated.realtime_options", "RealtimeDurationOption")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
    static REALTIME_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime_native::RealtimeOptions> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_OPTIONS.import(py, "livekit.agents._generated.realtime_options", "RealtimeOptions")?)? {
                return Err(super::not_a(obj, "RealtimeOptions"));
            }
            Ok(::lk_agents_abi::llm::realtime_native::RealtimeOptions {
                input_audio_transcription: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "input_audio_transcription", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_json_option_from_py(v)).map(Some) })?,
                input_audio_noise_reduction: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "input_audio_noise_reduction", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_json_option_from_py(v)).map(Some) })?,
                turn_detection: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "turn_detection", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_json_option_from_py(v)).map(Some) })?,
                tool_choice: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "tool_choice", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_tool_choice_from_py(v)).map(Some) })?,
                speed: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "speed", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeOptions.speed")).map(Some) })?,
                tracing: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "tracing", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_json_option_from_py(v)).map(Some) })?,
                truncation: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "truncation", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_json_option_from_py(v)).map(Some) })?,
                reasoning: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "reasoning", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_json_option_from_py(v)).map(Some) })?,
                max_session_duration: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "max_session_duration", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_duration_option_from_py(v)).map(Some) })?,
                voice: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "voice", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                max_response_output_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "max_response_output_tokens", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::realtime_options::realtime_json_option_from_py(v)).map(Some) })?,
            })
    }

    pub fn realtime_options_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::realtime_native::RealtimeOptions) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::realtime_native::RealtimeOptions { input_audio_transcription, input_audio_noise_reduction, turn_detection, tool_choice, speed, tracing, truncation, reasoning, max_session_duration, voice, max_response_output_tokens } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("input_audio_transcription", match input_audio_transcription { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_json_option_to_py(py, x)? })?;
            kwargs.set_item("input_audio_noise_reduction", match input_audio_noise_reduction { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_json_option_to_py(py, x)? })?;
            kwargs.set_item("turn_detection", match turn_detection { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_json_option_to_py(py, x)? })?;
            kwargs.set_item("tool_choice", match tool_choice { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_tool_choice_to_py(py, x)? })?;
            kwargs.set_item("speed", match speed { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("tracing", match tracing { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_json_option_to_py(py, x)? })?;
            kwargs.set_item("truncation", match truncation { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_json_option_to_py(py, x)? })?;
            kwargs.set_item("reasoning", match reasoning { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_json_option_to_py(py, x)? })?;
            kwargs.set_item("max_session_duration", match max_session_duration { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_duration_option_to_py(py, x)? })?;
            kwargs.set_item("voice", match voice { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("max_response_output_tokens", match max_response_output_tokens { None => py.None().into_bound(py), Some(x) => crate::generated::records::realtime_options::realtime_json_option_to_py(py, x)? })?;
            crate::host::construct(REALTIME_OPTIONS.import(py, "livekit.agents._generated.realtime_options", "RealtimeOptions")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
    static NATIVE_REALTIME_CONFIG: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn native_realtime_config_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime_native::NativeRealtimeConfig> {
        let py = obj.py();
            if !obj.is_instance(NATIVE_REALTIME_CONFIG.import(py, "livekit.agents._generated.realtime_options", "NativeRealtimeConfig")?)? {
                return Err(super::not_a(obj, "NativeRealtimeConfig"));
            }
            Ok(::lk_agents_abi::llm::realtime_native::NativeRealtimeConfig {
                api_key: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "api_key", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                base_url: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "base_url", |v| v.extract::<::std::string::String>())?,
                conn_options: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "conn_options", |v| v.extract::<::lk_agents_abi::ApiConnectOptions>().map_err(::std::convert::Into::<PyErr>::into))?,
                options: ::lk_agents_abi::python::attr_with(obj, "NativeRealtimeConfig", "options", |v| crate::generated::records::realtime_options::realtime_options_from_py(v))?,
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

    pub fn native_realtime_config_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::realtime_native::NativeRealtimeConfig) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::realtime_native::NativeRealtimeConfig { api_key, base_url, conn_options, options, dialect, model, voice, modalities, azure_deployment, entra_token, azure_defaults, api_secret, inference_provider, inference_class, label } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("api_key", match api_key { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("base_url", ::pyo3::IntoPyObjectExt::into_bound_py_any(base_url, py)?)?;
            kwargs.set_item("conn_options", ::pyo3::IntoPyObjectExt::into_bound_py_any(conn_options, py)?)?;
            kwargs.set_item("options", crate::generated::records::realtime_options::realtime_options_to_py(py, options)?)?;
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
            crate::host::construct(NATIVE_REALTIME_CONFIG.import(py, "livekit.agents._generated.realtime_options", "NativeRealtimeConfig")?.as_any(), (), Some(&kwargs))
    }
}

pub mod recorder {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/recorder.rs.
    static RECORDING_SUMMARY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn recording_summary_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::recorder::RecordingSummary) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::recorder::RecordingSummary { bytes_written, started_at, last_flushed, dropped_input_samples, dropped_output_samples } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("bytes_written", ::pyo3::IntoPyObjectExt::into_bound_py_any(bytes_written, py)?)?;
            kwargs.set_item("started_at", match started_at { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("last_flushed", match last_flushed { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("dropped_input_samples", ::pyo3::IntoPyObjectExt::into_bound_py_any(dropped_input_samples, py)?)?;
            kwargs.set_item("dropped_output_samples", ::pyo3::IntoPyObjectExt::into_bound_py_any(dropped_output_samples, py)?)?;
            crate::host::construct(RECORDING_SUMMARY.import(py, "livekit.agents._generated.recorder", "RecordingSummary")?.as_any(), (), Some(&kwargs))
    }
}

pub mod session_wire {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static SPEECH_CREATED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn speech_created_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::SpeechCreated) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::SpeechCreated { user_initiated, source, speech_handle } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("user_initiated", ::pyo3::IntoPyObjectExt::into_bound_py_any(user_initiated, py)?)?;
            kwargs.set_item("source", ::pyo3::IntoPyObjectExt::into_bound_py_any(source, py)?)?;
            kwargs.set_item("speech_handle", ::pyo3::IntoPyObjectExt::into_bound_py_any(speech_handle, py)?)?;
            crate::host::construct(SPEECH_CREATED_WIRE.import(py, "livekit.agents._generated.session_wire", "SpeechCreatedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static TOOL_CALL_STARTED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_call_started_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ToolCallStarted) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ToolCallStarted { function_call } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("function_call", ::pyo3::IntoPyObjectExt::into_bound_py_any(function_call, py)?)?;
            crate::host::construct(TOOL_CALL_STARTED_WIRE.import(py, "livekit.agents._generated.session_wire", "ToolCallStartedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.

    pub fn tool_update_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ToolUpdate) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::voice::events::ToolUpdate::Started(x) => crate::generated::records::session_wire::tool_call_started_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::ToolUpdate::Updated(x) => crate::generated::records::events::tool_call_updated_to_py(py, x)?,
            ::lk_agents_abi::voice::events::ToolUpdate::Ended(x) => crate::generated::records::events::tool_call_ended_to_py(py, x)?,
            ::lk_agents_abi::voice::events::ToolUpdate::Reply(x) => crate::generated::records::events::tool_reply_updated_to_py(py, x)?,
        })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static TOOL_EXECUTION_UPDATED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_execution_updated_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ToolExecutionUpdated) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ToolExecutionUpdated { update } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("update", crate::generated::records::session_wire::tool_update_wire_to_py(py, update)?)?;
            crate::host::construct(TOOL_EXECUTION_UPDATED_WIRE.import(py, "livekit.agents._generated.session_wire", "ToolExecutionUpdatedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static START_OF_SPEECH_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn start_of_speech_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::StartOfSpeechEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::StartOfSpeechEvent { at: _, speech_duration: _, from_vad: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            crate::host::construct(START_OF_SPEECH_EVENT.import(py, "livekit.agents._generated.session_wire", "StartOfSpeechEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static PREEMPTIVE_GENERATION_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn preemptive_generation_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::PreemptiveGenerationEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::PreemptiveGenerationEvent { new_transcript: _, transcript_confidence: _, started_speaking_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            crate::host::construct(PREEMPTIVE_GENERATION_EVENT.import(py, "livekit.agents._generated.session_wire", "PreemptiveGenerationEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static BACKCHANNEL_CONFIRMED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn backchannel_confirmed_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::BackchannelConfirmedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::BackchannelConfirmedWire {  } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            crate::host::construct(BACKCHANNEL_CONFIRMED_WIRE.import(py, "livekit.agents._generated.session_wire", "BackchannelConfirmedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static OVERLAP_INTERRUPTION_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn overlap_interruption_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::OverlapInterruptionEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::OverlapInterruptionEvent { ignore_user_transcript_until: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            crate::host::construct(OVERLAP_INTERRUPTION_EVENT.import(py, "livekit.agents._generated.session_wire", "OverlapInterruptionEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static USER_TURN_COMPLETED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn user_turn_completed_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::UserTurnCompletedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::UserTurnCompletedWire { turn, new_transcript, transcript_confidence } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("turn", ::pyo3::IntoPyObjectExt::into_bound_py_any(turn, py)?)?;
            kwargs.set_item("new_transcript", ::pyo3::IntoPyObjectExt::into_bound_py_any(new_transcript, py)?)?;
            kwargs.set_item("transcript_confidence", ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript_confidence, py)?)?;
            crate::host::construct(USER_TURN_COMPLETED_WIRE.import(py, "livekit.agents._generated.session_wire", "UserTurnCompletedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static END_OF_SPEECH_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn end_of_speech_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::EndOfSpeechEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::EndOfSpeechEvent { at: _, from_vad: _, silence_duration } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("silence_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(silence_duration, py)?)?;
            crate::host::construct(END_OF_SPEECH_EVENT.import(py, "livekit.agents._generated.session_wire", "EndOfSpeechEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static ITEM_ADDED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn item_added_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ItemAddedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ItemAddedWire { item, agent, visibility, run } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("item", ::pyo3::IntoPyObjectExt::into_bound_py_any(item, py)?)?;
            kwargs.set_item("agent", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent, py)?)?;
            kwargs.set_item("visibility", ::pyo3::IntoPyObjectExt::into_bound_py_any(visibility, py)?)?;
            kwargs.set_item("run", match run { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(ITEM_ADDED_WIRE.import(py, "livekit.agents._generated.session_wire", "ItemAddedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static FUNCTION_TOOLS_EXECUTED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn function_tools_executed_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ToolsExecutedEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ToolsExecutedEvent { function_calls: _, function_call_outputs: _, handoff_required: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            crate::host::construct(FUNCTION_TOOLS_EXECUTED_WIRE.import(py, "livekit.agents._generated.session_wire", "FunctionToolsExecutedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static DIAGNOSTIC_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn diagnostic_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::DiagnosticWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::DiagnosticWire { level, message, fields } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("level", ::pyo3::IntoPyObjectExt::into_bound_py_any(level, py)?)?;
            kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
            kwargs.set_item("fields", { let d = ::pyo3::types::PyDict::new(py); for (k, x) in fields { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?; } d.into_any() })?;
            crate::host::construct(DIAGNOSTIC_WIRE.import(py, "livekit.agents._generated.session_wire", "DiagnosticWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static RUN_ITEM_ADDED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_item_added_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::RunItemAddedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::RunItemAddedWire { run, item } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("run", match run { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("item", ::pyo3::IntoPyObjectExt::into_bound_py_any(item, py)?)?;
            crate::host::construct(RUN_ITEM_ADDED_WIRE.import(py, "livekit.agents._generated.session_wire", "RunItemAddedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static RUN_STARTED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_started_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::RunStartedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::RunStartedWire { run } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("run", match run { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(RUN_STARTED_WIRE.import(py, "livekit.agents._generated.session_wire", "RunStartedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static RUN_REFUSED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_refused_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::RunRefusedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::RunRefusedWire { run, cause } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("run", match run { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("cause", ::pyo3::IntoPyObjectExt::into_bound_py_any(cause, py)?)?;
            crate::host::construct(RUN_REFUSED_WIRE.import(py, "livekit.agents._generated.session_wire", "RunRefusedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static RUN_SETTLED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_settled_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::RunSettledWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::RunSettledWire { run, outcome, final_output, question, run_error, speech_handle } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("run", match run { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("outcome", ::pyo3::IntoPyObjectExt::into_bound_py_any(outcome, py)?)?;
            kwargs.set_item("final_output", match final_output { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("question", match question { None => py.None().into_bound(py), Some(x) => crate::generated::records::session_wire::run_question_to_py(py, x)? })?;
            kwargs.set_item("run_error", match run_error { None => py.None().into_bound(py), Some(x) => crate::generated::records::agent_run::run_error_to_py(py, x)? })?;
            kwargs.set_item("speech_handle", match speech_handle { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(RUN_SETTLED_WIRE.import(py, "livekit.agents._generated.session_wire", "RunSettledWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static RUN_QUESTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_question_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::RunQuestion) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::RunQuestion { question, verbatim, answer_tool } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
            kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
            kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(RUN_QUESTION.import(py, "livekit.agents._generated.session_wire", "RunQuestion")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static SPEECH_DONE_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn speech_done_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::SpeechDoneWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::SpeechDoneWire { speech_id, outcome } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("speech_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(speech_id, py)?)?;
            kwargs.set_item("outcome", ::pyo3::IntoPyObjectExt::into_bound_py_any(outcome, py)?)?;
            crate::host::construct(SPEECH_DONE_WIRE.import(py, "livekit.agents._generated.session_wire", "SpeechDoneWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static FLOOR_CHANGED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn floor_changed_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::FloorChangedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::FloorChangedWire { speech_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("speech_id", match speech_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(FLOOR_CHANGED_WIRE.import(py, "livekit.agents._generated.session_wire", "FloorChangedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static SPAN_STARTED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn span_started_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::SpanStartedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::SpanStartedWire { span_id, parent_id, name, start_time, attributes } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("span_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(span_id, py)?)?;
            kwargs.set_item("parent_id", match parent_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("start_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(start_time, py)?)?;
            kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
            crate::host::construct(SPAN_STARTED_WIRE.import(py, "livekit.agents._generated.session_wire", "SpanStartedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static SPAN_ENDED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn span_ended_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::SpanEndedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::SpanEndedWire { span_id, end_time, attributes, error, events, metrics } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("span_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(span_id, py)?)?;
            kwargs.set_item("end_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(end_time, py)?)?;
            kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
            kwargs.set_item("error", match error { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("events", ::pyo3::types::PyList::new(py, events.into_iter().map(|x| PyResult::Ok(crate::generated::records::session_wire::span_event_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("metrics", ::pyo3::types::PyList::new(py, metrics.into_iter().map(|x| PyResult::Ok(crate::generated::records::session_wire::span_metric_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(SPAN_ENDED_WIRE.import(py, "livekit.agents._generated.session_wire", "SpanEndedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static SPAN_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn span_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::SpanEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::SpanEvent { name, time, attributes } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("time", ::pyo3::IntoPyObjectExt::into_bound_py_any(time, py)?)?;
            kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
            crate::host::construct(SPAN_EVENT.import(py, "livekit.agents._generated.session_wire", "SpanEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static SPAN_METRIC: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn span_metric_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::SpanMetric) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::SpanMetric { instrument, value, attributes } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("instrument", ::pyo3::IntoPyObjectExt::into_bound_py_any(instrument, py)?)?;
            kwargs.set_item("value", ::pyo3::IntoPyObjectExt::into_bound_py_any(value, py)?)?;
            kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
            crate::host::construct(SPAN_METRIC.import(py, "livekit.agents._generated.session_wire", "SpanMetric")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static CONTEXT_CARRIED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn context_carried_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ContextCarriedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ContextCarriedWire { from_agent, to_agent, items } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("from_agent", ::pyo3::IntoPyObjectExt::into_bound_py_any(from_agent, py)?)?;
            kwargs.set_item("to_agent", ::pyo3::IntoPyObjectExt::into_bound_py_any(to_agent, py)?)?;
            kwargs.set_item("items", ::pyo3::types::PyList::new(py, items.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(CONTEXT_CARRIED_WIRE.import(py, "livekit.agents._generated.session_wire", "ContextCarriedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static CLOSE_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn close_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::CloseWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::CloseWire { reason, error } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("reason", ::pyo3::IntoPyObjectExt::into_bound_py_any(reason, py)?)?;
            kwargs.set_item("error", match error { None => py.None().into_bound(py), Some(x) => crate::generated::records::session_wire::session_error_wire_to_py(py, x)? })?;
            crate::host::construct(CLOSE_WIRE.import(py, "livekit.agents._generated.session_wire", "CloseWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static METRICS_COLLECTED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn metrics_collected_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::MetricsCollectedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::MetricsCollectedWire { metrics, speech_id, leg } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("metrics", ::pyo3::IntoPyObjectExt::into_bound_py_any(metrics, py)?)?;
            kwargs.set_item("speech_id", match speech_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("leg", match leg { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(METRICS_COLLECTED_WIRE.import(py, "livekit.agents._generated.session_wire", "MetricsCollectedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static SESSION_USAGE_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn session_usage_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::SessionUsageWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::SessionUsageWire { model_usage } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("model_usage", ::pyo3::types::PyList::new(py, model_usage.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(SESSION_USAGE_WIRE.import(py, "livekit.agents._generated.session_wire", "SessionUsageWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static OVERLAPPING_SPEECH_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn overlapping_speech_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::OverlappingSpeechWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::OverlappingSpeechWire { event } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("event", ::pyo3::IntoPyObjectExt::into_bound_py_any(event, py)?)?;
            crate::host::construct(OVERLAPPING_SPEECH_WIRE.import(py, "livekit.agents._generated.session_wire", "OverlappingSpeechWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static SESSION_ERROR_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn session_error_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::SessionErrorWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::SessionErrorWire { source, recoverable, label, error, speech_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("source", ::pyo3::IntoPyObjectExt::into_bound_py_any(source, py)?)?;
            kwargs.set_item("recoverable", ::pyo3::IntoPyObjectExt::into_bound_py_any(recoverable, py)?)?;
            kwargs.set_item("label", ::pyo3::IntoPyObjectExt::into_bound_py_any(label, py)?)?;
            kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
            kwargs.set_item("speech_id", match speech_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(SESSION_ERROR_WIRE.import(py, "livekit.agents._generated.session_wire", "SessionErrorWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static HISTORY_RELOADED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn history_reloaded_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::HistoryReloadedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::HistoryReloadedWire { items } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("items", ::pyo3::types::PyList::new(py, items.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(HISTORY_RELOADED_WIRE.import(py, "livekit.agents._generated.session_wire", "HistoryReloadedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.

    pub fn host_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::HostWire) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::voice::events::HostWire::HistoryReloaded(x) => crate::generated::records::session_wire::history_reloaded_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::UserTranscript(x) => crate::generated::records::events::user_input_transcribed_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::StartOfSpeech(x) => crate::generated::records::session_wire::start_of_speech_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::EndOfSpeech(x) => crate::generated::records::session_wire::end_of_speech_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::SpeechCreated(x) => crate::generated::records::session_wire::speech_created_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::Diagnostic(x) => crate::generated::records::session_wire::diagnostic_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::FunctionToolsExecuted(x) => crate::generated::records::session_wire::function_tools_executed_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::UserTurnCompleted(x) => crate::generated::records::session_wire::user_turn_completed_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::EotPrediction(x) => crate::generated::records::events::eot_prediction_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::AgentBackchannelOpportunity(x) => crate::generated::records::events::__agent_backchannel_opportunity_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::PreemptiveGeneration(x) => crate::generated::records::session_wire::preemptive_generation_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::UserTurnExceeded(x) => crate::generated::records::events::user_turn_exceeded_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::TranscriptionTimeout(x) => crate::generated::records::events::user_transcription_timeout_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::BackchannelConfirmed(x) => crate::generated::records::session_wire::backchannel_confirmed_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::OverlapInterruption(x) => crate::generated::records::session_wire::overlap_interruption_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::SpanStarted(x) => crate::generated::records::session_wire::span_started_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::SpanEnded(x) => crate::generated::records::session_wire::span_ended_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::AgentStateChanged(x) => crate::generated::records::events::agent_state_changed_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::UserStateChanged(x) => crate::generated::records::events::user_state_changed_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::AgentFalseInterruption(x) => crate::generated::records::events::agent_false_interruption_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::AmdResult(x) => crate::generated::records::amd::a_m_d_prediction_event_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::ItemAdded(x) => crate::generated::records::session_wire::item_added_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::ContextCarried(x) => crate::generated::records::session_wire::context_carried_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::RunStarted(x) => crate::generated::records::session_wire::run_started_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::RunItemAdded(x) => crate::generated::records::session_wire::run_item_added_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::RunRefused(x) => crate::generated::records::session_wire::run_refused_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::RunSettled(x) => crate::generated::records::session_wire::run_settled_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::SpeechDone(x) => crate::generated::records::session_wire::speech_done_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::FloorChanged(x) => crate::generated::records::session_wire::floor_changed_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::Close(x) => crate::generated::records::session_wire::close_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::MetricsCollected(x) => crate::generated::records::session_wire::metrics_collected_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::SessionUsageUpdated(x) => crate::generated::records::session_wire::session_usage_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::OverlappingSpeech(x) => crate::generated::records::session_wire::overlapping_speech_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::Error(x) => crate::generated::records::session_wire::session_error_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::ToolExecutionUpdated(x) => crate::generated::records::session_wire::tool_execution_updated_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::ParticipantLinked(x) => crate::generated::records::session_wire::participant_linked_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::ParticipantUnlinked(x) => crate::generated::records::session_wire::participant_unlinked_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::TextInput(x) => crate::generated::records::session_wire::text_input_wire_to_py(py, x)?,
            ::lk_agents_abi::voice::events::HostWire::DeleteRoomRequested(x) => crate::generated::records::session_wire::delete_room_requested_to_py(py, x)?,
        })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/room_io.rs.
    static PARTICIPANT_LINKED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn participant_linked_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::room_io::ParticipantLinked) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::room_io::ParticipantLinked { identity } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
            crate::host::construct(PARTICIPANT_LINKED.import(py, "livekit.agents._generated.session_wire", "ParticipantLinked")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/room_io.rs.
    static PARTICIPANT_UNLINKED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn participant_unlinked_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::room_io::ParticipantUnlinked) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::room_io::ParticipantUnlinked { identity, reason } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
            kwargs.set_item("reason", ::pyo3::IntoPyObjectExt::into_bound_py_any(reason, py)?)?;
            crate::host::construct(PARTICIPANT_UNLINKED.import(py, "livekit.agents._generated.session_wire", "ParticipantUnlinked")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/room_io.rs.
    static TEXT_INPUT_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_input_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::room_io::TextInput) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::room_io::TextInput { text, participant, info } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
            kwargs.set_item("participant", ::pyo3::IntoPyObjectExt::into_bound_py_any(participant, py)?)?;
            kwargs.set_item("info", ::pyo3::IntoPyObjectExt::into_bound_py_any(info, py)?)?;
            crate::host::construct(TEXT_INPUT_WIRE.import(py, "livekit.agents._generated.session_wire", "TextInputWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/room_io.rs.
    static DELETE_ROOM_REQUESTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn delete_room_requested_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::room_io::DeleteRoomRequested) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::room_io::DeleteRoomRequested { room } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("room", ::pyo3::IntoPyObjectExt::into_bound_py_any(room, py)?)?;
            crate::host::construct(DELETE_ROOM_REQUESTED.import(py, "livekit.agents._generated.session_wire", "DeleteRoomRequested")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.
    static START_PLAN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn start_plan_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::session_effects::StartPlan) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::session_effects::StartPlan { recording, is_primary, hosting, records } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("recording", crate::generated::records::telemetry::recording_options_to_py(py, recording)?)?;
            kwargs.set_item("is_primary", ::pyo3::IntoPyObjectExt::into_bound_py_any(is_primary, py)?)?;
            kwargs.set_item("hosting", ::pyo3::IntoPyObjectExt::into_bound_py_any(hosting, py)?)?;
            kwargs.set_item("records", ::pyo3::IntoPyObjectExt::into_bound_py_any(records, py)?)?;
            crate::host::construct(START_PLAN.import(py, "livekit.agents._generated.session_wire", "StartPlan")?.as_any(), (), Some(&kwargs))
    }
}

pub mod storage_class {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
    static THREAD_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn thread_info_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::ThreadInfo) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::ThreadInfo { thread_id, scope, attributes, created_at, last_active, expires_at, idle_ttl_seconds, database_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
            kwargs.set_item("scope", ::pyo3::IntoPyObjectExt::into_bound_py_any(scope, py)?)?;
            kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
            kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(created_at, py)?)?;
            kwargs.set_item("last_active", ::pyo3::IntoPyObjectExt::into_bound_py_any(last_active, py)?)?;
            kwargs.set_item("expires_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(expires_at, py)?)?;
            kwargs.set_item("idle_ttl_seconds", ::pyo3::IntoPyObjectExt::into_bound_py_any(idle_ttl_seconds, py)?)?;
            kwargs.set_item("database_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(database_id, py)?)?;
            crate::host::construct(THREAD_INFO.import(py, "livekit.agents._generated.storage_class", "ThreadInfo")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
    static THREAD_PAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn thread_page_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::ThreadPage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::ThreadPage { threads, before } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("threads", ::pyo3::types::PyList::new(py, threads.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_class::thread_info_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("before", ::pyo3::IntoPyObjectExt::into_bound_py_any(before, py)?)?;
            crate::host::construct(THREAD_PAGE.import(py, "livekit.agents._generated.storage_class", "ThreadPage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
    static THREAD_ITEM: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn thread_item_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::ThreadItem) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::ThreadItem { agent, seq, message_id, item } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("agent", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent, py)?)?;
            kwargs.set_item("seq", ::pyo3::IntoPyObjectExt::into_bound_py_any(seq, py)?)?;
            kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
            kwargs.set_item("item", ::lk_agents_abi::python::value_to_py(py, &item)?)?;
            crate::host::construct(THREAD_ITEM.import(py, "livekit.agents._generated.storage_class", "ThreadItem")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
    static THREAD_ITEM_PAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn thread_item_page_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::ThreadItemPage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::ThreadItemPage { items, after } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("items", ::pyo3::types::PyList::new(py, items.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_class::thread_item_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("after", ::pyo3::IntoPyObjectExt::into_bound_py_any(after, py)?)?;
            crate::host::construct(THREAD_ITEM_PAGE.import(py, "livekit.agents._generated.storage_class", "ThreadItemPage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
    static THREAD_TASK: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn thread_task_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::ThreadTask) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::ThreadTask { task_id, thread_id, name, status, updated_at } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
            kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
            kwargs.set_item("name", match name { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("status", crate::generated::records::storage_class::task_state_to_py(py, status)?)?;
            kwargs.set_item("updated_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(updated_at, py)?)?;
            crate::host::construct(THREAD_TASK.import(py, "livekit.agents._generated.storage_class", "ThreadTask")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.

    pub fn task_state_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::TaskState> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "submitted" => Ok(::lk_agents_abi::storage::TaskState::Submitted),
                "working" => Ok(::lk_agents_abi::storage::TaskState::Working),
                "input-required" => Ok(::lk_agents_abi::storage::TaskState::InputRequired),
                "completed" => Ok(::lk_agents_abi::storage::TaskState::Completed),
                "failed" => Ok(::lk_agents_abi::storage::TaskState::Failed),
                "canceled" => Ok(::lk_agents_abi::storage::TaskState::Canceled),
                "rejected" => Ok(::lk_agents_abi::storage::TaskState::Rejected),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("TaskState: {other:?} is not one of 'submitted', 'working', 'input-required', 'completed', 'failed', 'canceled', 'rejected'"))),
            }
    }

    pub fn task_state_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::TaskState) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::storage::TaskState::Submitted => "submitted",
                ::lk_agents_abi::storage::TaskState::Working => "working",
                ::lk_agents_abi::storage::TaskState::InputRequired => "input-required",
                ::lk_agents_abi::storage::TaskState::Completed => "completed",
                ::lk_agents_abi::storage::TaskState::Failed => "failed",
                ::lk_agents_abi::storage::TaskState::Canceled => "canceled",
                ::lk_agents_abi::storage::TaskState::Rejected => "rejected",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static ASK: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn ask_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::Ask) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::Ask { question, tool, call_id, answer_tool } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
            kwargs.set_item("tool", match tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("call_id", match call_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(ASK.import(py, "livekit.agents._generated.storage_class", "Ask")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static STREAM_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn stream_info_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::StreamInfo) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::StreamInfo { stream_id, expires_at } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("stream_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(stream_id, py)?)?;
            kwargs.set_item("expires_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(expires_at, py)?)?;
            crate::host::construct(STREAM_INFO.import(py, "livekit.agents._generated.storage_class", "StreamInfo")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static TASK_INTERNAL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn task_internal_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::TaskInternal) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::TaskInternal { ask, streams } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("ask", match ask { None => py.None().into_bound(py), Some(x) => crate::generated::records::storage_class::ask_to_py(py, x)? })?;
            kwargs.set_item("streams", ::pyo3::types::PyList::new(py, streams.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_class::stream_info_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(TASK_INTERNAL.import(py, "livekit.agents._generated.storage_class", "TaskInternal")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static ITEM_ROW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn item_row_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::ItemRow> {
        let py = obj.py();
            if !obj.is_instance(ITEM_ROW.import(py, "livekit.agents._generated.storage_class", "ItemRow")?)? {
                return Err(super::not_a(obj, "ItemRow"));
            }
            Ok(::lk_agents_abi::storage::ItemRow {
                id: ::lk_agents_abi::python::attr_with(obj, "ItemRow", "id", |v| v.extract::<::std::string::String>())?,
                message_id: ::lk_agents_abi::python::attr_with(obj, "ItemRow", "message_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                payload: ::lk_agents_abi::python::attr_with(obj, "ItemRow", "payload", |v| ::lk_agents_abi::python::value_from_py(v, "ItemRow.payload"))?,
            })
    }

    pub fn item_row_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::ItemRow) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::ItemRow { id, message_id, payload } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("message_id", match message_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("payload", ::lk_agents_abi::python::value_to_py(py, &payload)?)?;
            crate::host::construct(ITEM_ROW.import(py, "livekit.agents._generated.storage_class", "ItemRow")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static TASK_DATA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn task_data_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::TaskData) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::TaskData { history, artifacts, metadata, status_message } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("artifacts", match artifacts { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
            kwargs.set_item("metadata", ::lk_agents_abi::python::extra_to_py(py, &metadata)?.into_any())?;
            kwargs.set_item("status_message", match status_message { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
            crate::host::construct(TASK_DATA.import(py, "livekit.agents._generated.storage_class", "TaskData")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static TASK_RECORD: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn task_record_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::TaskRecord) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::TaskRecord { task_id, lane, seq, name, message_id, status, updated_at, data, internal } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
            kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
            kwargs.set_item("seq", ::pyo3::IntoPyObjectExt::into_bound_py_any(seq, py)?)?;
            kwargs.set_item("name", match name { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("message_id", match message_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("status", crate::generated::records::storage_class::task_state_to_py(py, status)?)?;
            kwargs.set_item("updated_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(updated_at, py)?)?;
            kwargs.set_item("data", crate::generated::records::storage_class::task_data_to_py(py, data)?)?;
            kwargs.set_item("internal", match internal { None => py.None().into_bound(py), Some(x) => crate::generated::records::storage_class::task_internal_to_py(py, x)? })?;
            crate::host::construct(TASK_RECORD.import(py, "livekit.agents._generated.storage_class", "TaskRecord")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static TASK_PAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn task_page_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::TaskPage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::TaskPage { tasks, next_page_token, page_size, total_size } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("tasks", ::pyo3::types::PyList::new(py, tasks.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_class::task_record_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("next_page_token", ::pyo3::IntoPyObjectExt::into_bound_py_any(next_page_token, py)?)?;
            kwargs.set_item("page_size", ::pyo3::IntoPyObjectExt::into_bound_py_any(page_size, py)?)?;
            kwargs.set_item("total_size", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_size, py)?)?;
            crate::host::construct(TASK_PAGE.import(py, "livekit.agents._generated.storage_class", "TaskPage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static LANE_ROW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn lane_row_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::LaneRow) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::LaneRow { lane, version, state, lease_holder, lease_until, waiter, drained_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
            kwargs.set_item("version", ::pyo3::IntoPyObjectExt::into_bound_py_any(version, py)?)?;
            kwargs.set_item("state", match state { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
            kwargs.set_item("lease_holder", match lease_holder { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("lease_until", ::pyo3::IntoPyObjectExt::into_bound_py_any(lease_until, py)?)?;
            kwargs.set_item("waiter", match waiter { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("drained_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(drained_id, py)?)?;
            crate::host::construct(LANE_ROW.import(py, "livekit.agents._generated.storage_class", "LaneRow")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static EVENT_ROW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn event_row_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::EventRow) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::EventRow { id, ts_ms, lane, run_id, r#type, payload } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("ts_ms", ::pyo3::IntoPyObjectExt::into_bound_py_any(ts_ms, py)?)?;
            kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
            kwargs.set_item("run_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(run_id, py)?)?;
            kwargs.set_item("type", ::pyo3::IntoPyObjectExt::into_bound_py_any(r#type, py)?)?;
            kwargs.set_item("payload", ::lk_agents_abi::python::value_to_py(py, &payload)?)?;
            crate::host::construct(EVENT_ROW.import(py, "livekit.agents._generated.storage_class", "EventRow")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static INBOX_ROW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn inbox_row_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::InboxRow) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::InboxRow { id, lane, task_id, message_id, message, received_at } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
            kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
            kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
            kwargs.set_item("message", ::lk_agents_abi::python::value_to_py(py, &message)?)?;
            kwargs.set_item("received_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(received_at, py)?)?;
            crate::host::construct(INBOX_ROW.import(py, "livekit.agents._generated.storage_class", "InboxRow")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
    static RUN_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn run_usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::RunUsage> {
        let py = obj.py();
            if !obj.is_instance(RUN_USAGE.import(py, "livekit.agents._generated.storage_class", "RunUsage")?)? {
                return Err(super::not_a(obj, "RunUsage"));
            }
            Ok(::lk_agents_abi::storage::RunUsage {
                requests: ::lk_agents_abi::python::attr_with(obj, "RunUsage", "requests", |v| ::lk_agents_abi::python::number_from_py::<i64>(v, "RunUsage.requests"))?,
                input_tokens: ::lk_agents_abi::python::attr_with(obj, "RunUsage", "input_tokens", |v| ::lk_agents_abi::python::number_from_py::<i64>(v, "RunUsage.input_tokens"))?,
                output_tokens: ::lk_agents_abi::python::attr_with(obj, "RunUsage", "output_tokens", |v| ::lk_agents_abi::python::number_from_py::<i64>(v, "RunUsage.output_tokens"))?,
                tool_calls: ::lk_agents_abi::python::attr_with(obj, "RunUsage", "tool_calls", |v| ::lk_agents_abi::python::number_from_py::<i64>(v, "RunUsage.tool_calls"))?,
            })
    }

    pub fn run_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::RunUsage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::RunUsage { requests, input_tokens, output_tokens, tool_calls } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("requests", ::pyo3::IntoPyObjectExt::into_bound_py_any(requests, py)?)?;
            kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
            kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
            kwargs.set_item("tool_calls", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_calls, py)?)?;
            crate::host::construct(RUN_USAGE.import(py, "livekit.agents._generated.storage_class", "RunUsage")?.as_any(), (), Some(&kwargs))
    }
}

pub mod storage_context {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
    static CONTEXT_FIELD: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn context_field_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::context::ContextField> {
        let py = obj.py();
            if !obj.is_instance(CONTEXT_FIELD.import(py, "livekit.agents._generated.storage_context", "ContextField")?)? {
                return Err(super::not_a(obj, "ContextField"));
            }
            Ok(::lk_agents_abi::storage::context::ContextField {
                name: ::lk_agents_abi::python::attr_with(obj, "ContextField", "name", |v| v.extract::<::std::string::String>())?,
                schema: ::lk_agents_abi::python::attr_with(obj, "ContextField", "schema", |v| ::lk_agents_abi::python::value_from_py(v, "ContextField.schema"))?,
                value: ::lk_agents_abi::python::attr_with(obj, "ContextField", "value", |v| ::lk_agents_abi::python::value_from_py(v, "ContextField.value"))?,
            })
    }

    pub fn context_field_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::context::ContextField) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::context::ContextField { name, schema, value } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("schema", ::lk_agents_abi::python::value_to_py(py, &schema)?)?;
            kwargs.set_item("value", ::lk_agents_abi::python::value_to_py(py, &value)?)?;
            crate::host::construct(CONTEXT_FIELD.import(py, "livekit.agents._generated.storage_context", "ContextField")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
    static CONTEXT_MANIFEST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn context_manifest_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::context::ContextManifest> {
        let py = obj.py();
            if !obj.is_instance(CONTEXT_MANIFEST.import(py, "livekit.agents._generated.storage_context", "ContextManifest")?)? {
                return Err(super::not_a(obj, "ContextManifest"));
            }
            Ok(::lk_agents_abi::storage::context::ContextManifest {
                identity: ::lk_agents_abi::python::attr_with(obj, "ContextManifest", "identity", |v| v.extract::<::std::string::String>())?,
                fields: ::lk_agents_abi::python::attr_with(obj, "ContextManifest", "fields", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>())?,
            })
    }

    pub fn context_manifest_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::context::ContextManifest) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::context::ContextManifest { identity, fields } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
            kwargs.set_item("fields", ::pyo3::types::PyList::new(py, fields.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(CONTEXT_MANIFEST.import(py, "livekit.agents._generated.storage_context", "ContextManifest")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
    static CONTEXT_SCOPE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn context_scope_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::context::ContextScope> {
        let py = obj.py();
            if !obj.is_instance(CONTEXT_SCOPE.import(py, "livekit.agents._generated.storage_context", "ContextScope")?)? {
                return Err(super::not_a(obj, "ContextScope"));
            }
            Ok(::lk_agents_abi::storage::context::ContextScope {
                scope_id: ::lk_agents_abi::python::attr_with(obj, "ContextScope", "scope_id", |v| v.extract::<::std::string::String>())?,
                revision: ::lk_agents_abi::python::attr_with(obj, "ContextScope", "revision", |v| ::lk_agents_abi::python::number_from_py::<i64>(v, "ContextScope.revision"))?,
                fields: ::lk_agents_abi::python::attr_with(obj, "ContextScope", "fields", |v| match ::lk_agents_abi::python::value_from_py(v, "ContextScope.fields")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("ContextScope.fields: expected a dict")) })?,
                contracts: ::lk_agents_abi::python::attr_with(obj, "ContextScope", "contracts", |v| match ::lk_agents_abi::python::value_from_py(v, "ContextScope.contracts")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("ContextScope.contracts: expected a dict")) })?,
                manifests: ::lk_agents_abi::python::attr_with(obj, "ContextScope", "manifests", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::storage_context::context_manifest_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
            })
    }

    pub fn context_scope_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::context::ContextScope) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::context::ContextScope { scope_id, revision, fields, contracts, manifests } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("scope_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(scope_id, py)?)?;
            kwargs.set_item("revision", ::pyo3::IntoPyObjectExt::into_bound_py_any(revision, py)?)?;
            kwargs.set_item("fields", ::lk_agents_abi::python::extra_to_py(py, &fields)?.into_any())?;
            kwargs.set_item("contracts", ::lk_agents_abi::python::extra_to_py(py, &contracts)?.into_any())?;
            kwargs.set_item("manifests", ::pyo3::types::PyList::new(py, manifests.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_context::context_manifest_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(CONTEXT_SCOPE.import(py, "livekit.agents._generated.storage_context", "ContextScope")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
    static CONTEXT_SCOPE_WRITE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn context_scope_write_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::context::ContextScopeWrite> {
        let py = obj.py();
            if !obj.is_instance(CONTEXT_SCOPE_WRITE.import(py, "livekit.agents._generated.storage_context", "ContextScopeWrite")?)? {
                return Err(super::not_a(obj, "ContextScopeWrite"));
            }
            Ok(::lk_agents_abi::storage::context::ContextScopeWrite {
                expected_revision: ::lk_agents_abi::python::attr_with(obj, "ContextScopeWrite", "expected_revision", |v| ::lk_agents_abi::python::number_from_py::<i64>(v, "ContextScopeWrite.expected_revision"))?,
                scope: ::lk_agents_abi::python::attr_with(obj, "ContextScopeWrite", "scope", |v| crate::generated::records::storage_context::context_scope_from_py(v))?,
            })
    }

    pub fn context_scope_write_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::context::ContextScopeWrite) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::context::ContextScopeWrite { expected_revision, scope } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("expected_revision", ::pyo3::IntoPyObjectExt::into_bound_py_any(expected_revision, py)?)?;
            kwargs.set_item("scope", crate::generated::records::storage_context::context_scope_to_py(py, scope)?)?;
            crate::host::construct(CONTEXT_SCOPE_WRITE.import(py, "livekit.agents._generated.storage_context", "ContextScopeWrite")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
    static APP_STATEMENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn app_statement_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::context::AppStatement> {
        let py = obj.py();
            if !obj.is_instance(APP_STATEMENT.import(py, "livekit.agents._generated.storage_context", "AppStatement")?)? {
                return Err(super::not_a(obj, "AppStatement"));
            }
            Ok(::lk_agents_abi::storage::context::AppStatement {
                sql: ::lk_agents_abi::python::attr_with(obj, "AppStatement", "sql", |v| v.extract::<::std::string::String>())?,
                params: ::lk_agents_abi::python::attr_with(obj, "AppStatement", "params", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; ::lk_agents_abi::python::value_from_py(item, "AppStatement.params") }).collect::<PyResult<Vec<_>>>())?,
            })
    }

    pub fn app_statement_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::context::AppStatement) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::context::AppStatement { sql, params } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("sql", ::pyo3::IntoPyObjectExt::into_bound_py_any(sql, py)?)?;
            kwargs.set_item("params", ::pyo3::types::PyList::new(py, params.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(APP_STATEMENT.import(py, "livekit.agents._generated.storage_context", "AppStatement")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
    static CONTEXT_VALUES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn context_values_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::context::ContextValues> {
        let py = obj.py();
            if !obj.is_instance(CONTEXT_VALUES.import(py, "livekit.agents._generated.storage_context", "ContextValues")?)? {
                return Err(super::not_a(obj, "ContextValues"));
            }
            Ok(::lk_agents_abi::storage::context::ContextValues {
                revision: ::lk_agents_abi::python::attr_with(obj, "ContextValues", "revision", |v| ::lk_agents_abi::python::number_from_py::<i64>(v, "ContextValues.revision"))?,
                fields: ::lk_agents_abi::python::attr_with(obj, "ContextValues", "fields", |v| match ::lk_agents_abi::python::value_from_py(v, "ContextValues.fields")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("ContextValues.fields: expected a dict")) })?,
            })
    }

    pub fn context_values_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::storage::context::ContextValues) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::storage::context::ContextValues { revision, fields } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("revision", ::pyo3::IntoPyObjectExt::into_bound_py_any(revision, py)?)?;
            kwargs.set_item("fields", ::lk_agents_abi::python::extra_to_py(py, &fields)?.into_any())?;
            crate::host::construct(CONTEXT_VALUES.import(py, "livekit.agents._generated.storage_context", "ContextValues")?.as_any(), (), Some(&kwargs))
    }
}

pub mod storage_options {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/a2a/options.rs.
    static TURN_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn turn_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::options::TurnOptions> {
        let py = obj.py();
            if !obj.is_instance(TURN_OPTIONS.import(py, "livekit.agents._generated.storage_options", "TurnOptions")?)? {
                return Err(super::not_a(obj, "TurnOptions"));
            }
            Ok(::lk_agents_abi::a2a::options::TurnOptions {
                policy: ::lk_agents_abi::python::attr_with(obj, "TurnOptions", "policy", |v| crate::generated::records::storage_options::policy_spec_from_py(v))?,
                debounce: ::lk_agents_abi::python::attr_with(obj, "TurnOptions", "debounce", |v| crate::generated::records::storage_options::debounce_spec_from_py(v))?,
                wait_timeout: ::lk_agents_abi::python::attr_with(obj, "TurnOptions", "wait_timeout", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "TurnOptions.wait_timeout").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("TurnOptions.wait_timeout: {e}"))))).map(Some) })?,
            })
    }

    pub fn turn_options_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::options::TurnOptions) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::options::TurnOptions { policy, debounce, wait_timeout } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("policy", crate::generated::records::storage_options::policy_spec_to_py(py, policy)?)?;
            kwargs.set_item("debounce", crate::generated::records::storage_options::debounce_spec_to_py(py, debounce)?)?;
            kwargs.set_item("wait_timeout", match wait_timeout { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x.as_secs_f64(), py)? })?;
            crate::host::construct(TURN_OPTIONS.import(py, "livekit.agents._generated.storage_options", "TurnOptions")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/options.rs.

    pub fn policy_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::options::Policy> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "batch" => Ok(::lk_agents_abi::a2a::options::Policy::Batch),
                "interrupt" => Ok(::lk_agents_abi::a2a::options::Policy::Interrupt),
                "queue" => Ok(::lk_agents_abi::a2a::options::Policy::Queue),
                "concurrent" => Ok(::lk_agents_abi::a2a::options::Policy::Concurrent),
                "reject" => Ok(::lk_agents_abi::a2a::options::Policy::Reject),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("Policy: {other:?} is not one of 'batch', 'interrupt', 'queue', 'concurrent', 'reject'"))),
            }
    }

    pub fn policy_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::options::Policy) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::a2a::options::Policy::Batch => "batch",
                ::lk_agents_abi::a2a::options::Policy::Interrupt => "interrupt",
                ::lk_agents_abi::a2a::options::Policy::Queue => "queue",
                ::lk_agents_abi::a2a::options::Policy::Concurrent => "concurrent",
                ::lk_agents_abi::a2a::options::Policy::Reject => "reject",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/options.rs.

    pub fn policy_spec_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::options::PolicySpec> {
        if obj.is_none() {
            return Ok(::lk_agents_abi::a2a::options::PolicySpec::Default);
        }
        if obj.is_instance_of::<::pyo3::types::PyString>() {
            return Ok(::lk_agents_abi::a2a::options::PolicySpec::All(crate::generated::records::storage_options::policy_from_py(obj)?));
        }
        if obj.is_instance_of::<::pyo3::types::PyDict>() {
            return Ok(::lk_agents_abi::a2a::options::PolicySpec::ByChannel(obj.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, crate::generated::records::storage_options::policy_from_py(item)?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>()?));
        }
        Err(super::not_a(obj, "PolicySpec"))
    }

    pub fn policy_spec_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::options::PolicySpec) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::a2a::options::PolicySpec::Default => py.None().into_bound(py),
            ::lk_agents_abi::a2a::options::PolicySpec::All(x) => crate::generated::records::storage_options::policy_to_py(py, x)?,
            ::lk_agents_abi::a2a::options::PolicySpec::ByChannel(x) => { let d = ::pyo3::types::PyDict::new(py); for (k, x) in x { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?, crate::generated::records::storage_options::policy_to_py(py, x)?)?; } d.into_any() },
        })
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/options.rs.

    pub fn debounce_spec_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::options::DebounceSpec> {
        if obj.is_none() {
            return Ok(::lk_agents_abi::a2a::options::DebounceSpec::Default);
        }
        if (obj.is_instance_of::<::pyo3::types::PyFloat>() || (obj.is_instance_of::<::pyo3::types::PyInt>() && !obj.is_instance_of::<::pyo3::types::PyBool>())) {
            return Ok(::lk_agents_abi::a2a::options::DebounceSpec::All(::lk_agents_abi::python::number_from_py::<f64>(obj, "DebounceSpec.All").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("DebounceSpec.All: {e}"))))?));
        }
        if obj.is_instance_of::<::pyo3::types::PyDict>() {
            return Ok(::lk_agents_abi::a2a::options::DebounceSpec::ByChannel(obj.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, ::lk_agents_abi::python::number_from_py::<f64>(item, "DebounceSpec.ByChannel").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("DebounceSpec.ByChannel: {e}"))))?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>()?));
        }
        Err(super::not_a(obj, "DebounceSpec"))
    }

    pub fn debounce_spec_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::options::DebounceSpec) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::a2a::options::DebounceSpec::Default => py.None().into_bound(py),
            ::lk_agents_abi::a2a::options::DebounceSpec::All(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x.as_secs_f64(), py)?,
            ::lk_agents_abi::a2a::options::DebounceSpec::ByChannel(x) => { let d = ::pyo3::types::PyDict::new(py); for (k, x) in x { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x.as_secs_f64(), py)?)?; } d.into_any() },
        })
    }
}

pub mod storage_turn {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
    static TURN_CONTEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn turn_context_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::turn::TurnContext) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::turn::TurnContext { thread_id, lane, message, batch, history, state, context_scope_id, context_scopes, read_only, task_id, tasks, caller_history, answer_tool, host_context: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
            kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
            kwargs.set_item("message", ::lk_agents_abi::python::extra_to_py(py, &message)?.into_any())?;
            kwargs.set_item("batch", ::pyo3::types::PyList::new(py, batch.into_iter().map(|x| PyResult::Ok(crate::generated::records::a2a_client::message_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_class::item_row_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("state", match state { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
            kwargs.set_item("context_scope_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(context_scope_id, py)?)?;
            kwargs.set_item("context_scopes", ::pyo3::types::PyList::new(py, context_scopes.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_context::context_scope_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("read_only", ::pyo3::IntoPyObjectExt::into_bound_py_any(read_only, py)?)?;
            kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
            kwargs.set_item("tasks", ::pyo3::types::PyList::new(py, tasks.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_class::task_record_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("caller_history", match caller_history { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
            kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(TURN_CONTEXT.import(py, "livekit.agents._generated.storage_turn", "TurnContext")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
    static PENDING_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn pending_call_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::turn::PendingCall> {
        let py = obj.py();
            if !obj.is_instance(PENDING_CALL.import(py, "livekit.agents._generated.storage_turn", "PendingCall")?)? {
                return Err(super::not_a(obj, "PendingCall"));
            }
            Ok(::lk_agents_abi::a2a::turn::PendingCall {
                name: ::lk_agents_abi::python::attr_with(obj, "PendingCall", "name", |v| v.extract::<::std::string::String>())?,
                call_id: ::lk_agents_abi::python::attr_with(obj, "PendingCall", "call_id", |v| v.extract::<::std::string::String>())?,
                answer_tool: ::lk_agents_abi::python::attr_with(obj, "PendingCall", "answer_tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
    static HOST_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn host_event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::turn::HostEvent> {
        let py = obj.py();
            if !obj.is_instance(HOST_EVENT.import(py, "livekit.agents._generated.storage_turn", "HostEvent")?)? {
                return Err(super::not_a(obj, "HostEvent"));
            }
            Ok(::lk_agents_abi::a2a::turn::HostEvent {
                r#type: ::lk_agents_abi::python::attr_with(obj, "HostEvent", "type", |v| v.extract::<::std::string::String>())?,
                payload: ::lk_agents_abi::python::attr_with(obj, "HostEvent", "payload", |v| ::lk_agents_abi::python::value_from_py(v, "HostEvent.payload"))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
    static HANDLER_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn handler_output_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::turn::HandlerOutput> {
        let py = obj.py();
            if !obj.is_instance(HANDLER_OUTPUT.import(py, "livekit.agents._generated.storage_turn", "HandlerOutput")?)? {
                return Err(super::not_a(obj, "HandlerOutput"));
            }
            Ok(::lk_agents_abi::a2a::turn::HandlerOutput {
                output: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "output", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "HandlerOutput.output")).map(Some) })?,
                new_items: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "new_items", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; ::lk_agents_abi::python::value_from_py(item, "HandlerOutput.new_items") }).collect::<PyResult<Vec<_>>>())?,
                state: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "state", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "HandlerOutput.state")).map(Some) })?,
                scope_writes: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "scope_writes", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::storage_context::context_scope_write_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                app_writes: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "app_writes", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::storage_context::app_statement_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                usage: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "usage", |v| crate::generated::records::storage_class::run_usage_from_py(v))?,
                input_required: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "input_required", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                pending_call: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "pending_call", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::storage_turn::pending_call_from_py(v)).map(Some) })?,
                interrupted: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "interrupted", |v| v.extract::<bool>())?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "metadata", |v| match ::lk_agents_abi::python::value_from_py(v, "HandlerOutput.metadata")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("HandlerOutput.metadata: expected a dict")) })?,
                error: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "error", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                events: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "events", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::storage_turn::host_event_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                status: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "status", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::storage_class::task_state_from_py(v)).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
    static TURN_RESULT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn turn_result_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::turn::TurnResult) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::a2a::turn::TurnResult { task, output, thread_id, answered_by, run_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("task", crate::generated::records::storage_class::task_record_to_py(py, task)?)?;
            kwargs.set_item("output", match output { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
            kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
            kwargs.set_item("answered_by", ::pyo3::IntoPyObjectExt::into_bound_py_any(answered_by, py)?)?;
            kwargs.set_item("run_id", match run_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(TURN_RESULT.import(py, "livekit.agents._generated.storage_turn", "TurnResult")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.

    pub fn verdict_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::turn::Verdict> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "continue" => Ok(::lk_agents_abi::a2a::turn::Verdict::Continue),
                "yield" => Ok(::lk_agents_abi::a2a::turn::Verdict::Yield),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("Verdict: {other:?} is not one of 'continue', 'yield'"))),
            }
    }

    pub fn verdict_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::a2a::turn::Verdict) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::a2a::turn::Verdict::Continue => "continue",
                ::lk_agents_abi::a2a::turn::Verdict::Yield => "yield",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
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

    pub fn turn_event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::turn::TurnEvent> {
        let py = obj.py();
            if obj.is_instance(CHECKPOINT.import(py, "livekit.agents._generated.storage_turn", "Checkpoint")?)? {
                return Ok(::lk_agents_abi::a2a::turn::TurnEvent::Checkpoint);
            }
            if obj.is_instance(OUTPUT.import(py, "livekit.agents._generated.storage_turn", "Output")?)? {
                return Ok(::lk_agents_abi::a2a::turn::TurnEvent::Output { output: ::lk_agents_abi::python::attr_with(obj, "Output", "output", |v| crate::generated::records::storage_turn::handler_output_from_py(v))? });
            }
            if obj.is_instance(PROGRESS.import(py, "livekit.agents._generated.storage_turn", "Progress")?)? {
                return Ok(::lk_agents_abi::a2a::turn::TurnEvent::Progress { task_id: ::lk_agents_abi::python::attr_with(obj, "Progress", "task_id", |v| v.extract::<::std::string::String>())?, text: ::lk_agents_abi::python::attr_with(obj, "Progress", "text", |v| v.extract::<::std::string::String>())? });
            }
            if obj.is_instance(TOKEN.import(py, "livekit.agents._generated.storage_turn", "Token")?)? {
                return Ok(::lk_agents_abi::a2a::turn::TurnEvent::Token { task_id: ::lk_agents_abi::python::attr_with(obj, "Token", "task_id", |v| v.extract::<::std::string::String>())?, delta: ::lk_agents_abi::python::attr_with(obj, "Token", "delta", |v| v.extract::<::std::string::String>())? });
            }
            Err(super::not_a(obj, "Checkpoint | Output | Progress | Token"))
    }
}

pub mod telemetry {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/metrics/mod.rs.

    pub fn otel_value_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::OtelValue) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::metrics::OtelValue::Int(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
            ::lk_agents_abi::metrics::OtelValue::Float(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
        })
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/policy.rs.

    pub fn telemetry_value_kind_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::telemetry::policy::TelemetryValueKind> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "scalar" => Ok(::lk_agents_abi::telemetry::policy::TelemetryValueKind::Scalar),
                "integer" => Ok(::lk_agents_abi::telemetry::policy::TelemetryValueKind::Integer),
                "string" => Ok(::lk_agents_abi::telemetry::policy::TelemetryValueKind::String),
                "bytes" => Ok(::lk_agents_abi::telemetry::policy::TelemetryValueKind::Bytes),
                "mapping" => Ok(::lk_agents_abi::telemetry::policy::TelemetryValueKind::Mapping),
                "sequence" => Ok(::lk_agents_abi::telemetry::policy::TelemetryValueKind::Sequence),
                "unsupported" => Ok(::lk_agents_abi::telemetry::policy::TelemetryValueKind::Unsupported),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("TelemetryValueKind: {other:?} is not one of 'scalar', 'integer', 'string', 'bytes', 'mapping', 'sequence', 'unsupported'"))),
            }
    }

    pub fn telemetry_value_kind_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::telemetry::policy::TelemetryValueKind) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::telemetry::policy::TelemetryValueKind::Scalar => "scalar",
                ::lk_agents_abi::telemetry::policy::TelemetryValueKind::Integer => "integer",
                ::lk_agents_abi::telemetry::policy::TelemetryValueKind::String => "string",
                ::lk_agents_abi::telemetry::policy::TelemetryValueKind::Bytes => "bytes",
                ::lk_agents_abi::telemetry::policy::TelemetryValueKind::Mapping => "mapping",
                ::lk_agents_abi::telemetry::policy::TelemetryValueKind::Sequence => "sequence",
                ::lk_agents_abi::telemetry::policy::TelemetryValueKind::Unsupported => "unsupported",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/policy.rs.

    pub fn telemetry_record_kind_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::telemetry::policy::TelemetryRecordKind> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "span" => Ok(::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Span),
                "log" => Ok(::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Log),
                "event" => Ok(::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Event),
                "link" => Ok(::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Link),
                "scope" => Ok(::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Scope),
                "value" => Ok(::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Value),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("TelemetryRecordKind: {other:?} is not one of 'span', 'log', 'event', 'link', 'scope', 'value'"))),
            }
    }

    pub fn telemetry_record_kind_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::telemetry::policy::TelemetryRecordKind) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Span => "span",
                ::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Log => "log",
                ::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Event => "event",
                ::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Link => "link",
                ::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Scope => "scope",
                ::lk_agents_abi::telemetry::policy::TelemetryRecordKind::Value => "value",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/policy.rs.

    pub fn telemetry_export_action_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::telemetry::policy::TelemetryExportAction> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "export" => Ok(::lk_agents_abi::telemetry::policy::TelemetryExportAction::Export),
                "hold" => Ok(::lk_agents_abi::telemetry::policy::TelemetryExportAction::Hold),
                "drop" => Ok(::lk_agents_abi::telemetry::policy::TelemetryExportAction::Drop),
                "capacity" => Ok(::lk_agents_abi::telemetry::policy::TelemetryExportAction::Capacity),
                "disabled" => Ok(::lk_agents_abi::telemetry::policy::TelemetryExportAction::Disabled),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("TelemetryExportAction: {other:?} is not one of 'export', 'hold', 'drop', 'capacity', 'disabled'"))),
            }
    }

    pub fn telemetry_export_action_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::telemetry::policy::TelemetryExportAction) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::telemetry::policy::TelemetryExportAction::Export => "export",
                ::lk_agents_abi::telemetry::policy::TelemetryExportAction::Hold => "hold",
                ::lk_agents_abi::telemetry::policy::TelemetryExportAction::Drop => "drop",
                ::lk_agents_abi::telemetry::policy::TelemetryExportAction::Capacity => "capacity",
                ::lk_agents_abi::telemetry::policy::TelemetryExportAction::Disabled => "disabled",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
    static RECORDING_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn recording_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::telemetry::report::RecordingOptions> {
        let py = obj.py();
            if !obj.is_instance(RECORDING_OPTIONS.import(py, "livekit.agents._generated.telemetry", "RecordingOptions")?)? {
                return Err(super::not_a(obj, "RecordingOptions"));
            }
            Ok(::lk_agents_abi::telemetry::report::RecordingOptions {
                audio: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "audio", |v| v.extract::<bool>())?,
                traces: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "traces", |v| v.extract::<bool>())?,
                logs: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "logs", |v| v.extract::<bool>())?,
                transcript: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "transcript", |v| v.extract::<bool>())?,
                redaction: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "redaction", |v| v.extract::<bool>())?,
            })
    }

    pub fn recording_options_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::telemetry::report::RecordingOptions) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::telemetry::report::RecordingOptions { audio, traces, logs, transcript, redaction } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("audio", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio, py)?)?;
            kwargs.set_item("traces", ::pyo3::IntoPyObjectExt::into_bound_py_any(traces, py)?)?;
            kwargs.set_item("logs", ::pyo3::IntoPyObjectExt::into_bound_py_any(logs, py)?)?;
            kwargs.set_item("transcript", ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript, py)?)?;
            kwargs.set_item("redaction", ::pyo3::IntoPyObjectExt::into_bound_py_any(redaction, py)?)?;
            crate::host::construct(RECORDING_OPTIONS.import(py, "livekit.agents._generated.telemetry", "RecordingOptions")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
    static EVALUATION_RECORD: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn evaluation_record_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::EvaluationRecord) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::EvaluationRecord { name, tag, verdict, reasoning, instructions } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("tag", ::pyo3::IntoPyObjectExt::into_bound_py_any(tag, py)?)?;
            kwargs.set_item("verdict", ::pyo3::IntoPyObjectExt::into_bound_py_any(verdict, py)?)?;
            kwargs.set_item("reasoning", ::pyo3::IntoPyObjectExt::into_bound_py_any(reasoning, py)?)?;
            kwargs.set_item("instructions", ::pyo3::IntoPyObjectExt::into_bound_py_any(instructions, py)?)?;
            crate::host::construct(EVALUATION_RECORD.import(py, "livekit.agents._generated.telemetry", "EvaluationRecord")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.

    pub fn outcome_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::Outcome) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::Outcome::Success => "success",
                ::lk_agents_abi::Outcome::Fail => "fail",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
    static SESSION_REPORT_INPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn session_report_input_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::telemetry::report::SessionReportInput> {
        let py = obj.py();
            if !obj.is_instance(SESSION_REPORT_INPUT.import(py, "livekit.agents._generated.telemetry", "SessionReportInput")?)? {
                return Err(super::not_a(obj, "SessionReportInput"));
            }
            Ok(::lk_agents_abi::telemetry::report::SessionReportInput {
                job_id: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "job_id", |v| v.extract::<::std::string::String>())?,
                room_id: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "room_id", |v| v.extract::<::std::string::String>())?,
                room: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "room", |v| v.extract::<::std::string::String>())?,
                agent_name: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "agent_name", |v| v.extract::<::std::string::String>())?,
                sdk_version: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "sdk_version", |v| v.extract::<::std::string::String>())?,
                started_at: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "started_at", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "SessionReportInput.started_at")).map(Some) })?,
                timestamp: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "SessionReportInput.timestamp"))?,
                options: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "options", |v| ::lk_agents_abi::python::value_from_py(v, "SessionReportInput.options"))?,
                recording_options: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "recording_options", |v| crate::generated::records::telemetry::recording_options_from_py(v))?,
                model_usage: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "model_usage", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::metrics::ModelUsage>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
                chat_history: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "chat_history", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
                omitted_events: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "omitted_events", |v| ::lk_agents_abi::python::number_from_py::<usize>(v, "SessionReportInput.omitted_events"))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
    static SPAN_UPDATE_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn span_update_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::telemetry::report::SpanUpdateWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::telemetry::report::SpanUpdateWire { span, attrs } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("span", ::pyo3::IntoPyObjectExt::into_bound_py_any(span, py)?)?;
            kwargs.set_item("attrs", crate::generated::records::telemetry::span_update_attrs_wire_to_py(py, attrs)?)?;
            crate::host::construct(SPAN_UPDATE_WIRE.import(py, "livekit.agents._generated.telemetry", "SpanUpdateWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
    static SPAN_UPDATE_ATTRS_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn span_update_attrs_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::telemetry::report::SpanUpdateAttrsWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::telemetry::report::SpanUpdateAttrsWire { is_interruption, probability, total_duration, prediction_duration, detection_delay } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("is_interruption", ::pyo3::IntoPyObjectExt::into_bound_py_any(is_interruption, py)?)?;
            kwargs.set_item("probability", ::pyo3::IntoPyObjectExt::into_bound_py_any(probability, py)?)?;
            kwargs.set_item("total_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_duration, py)?)?;
            kwargs.set_item("prediction_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(prediction_duration, py)?)?;
            kwargs.set_item("detection_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(detection_delay, py)?)?;
            crate::host::construct(SPAN_UPDATE_ATTRS_WIRE.import(py, "livekit.agents._generated.telemetry", "SpanUpdateAttrsWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
    static LOG_RECORD_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn log_record_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::LogRecordWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::LogRecordWire { logger, body, timestamp_nanos, severity_number, severity_text, attributes } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("logger", ::pyo3::IntoPyObjectExt::into_bound_py_any(logger, py)?)?;
            kwargs.set_item("body", ::pyo3::IntoPyObjectExt::into_bound_py_any(body, py)?)?;
            kwargs.set_item("timestamp_nanos", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp_nanos, py)?)?;
            kwargs.set_item("severity_number", ::pyo3::IntoPyObjectExt::into_bound_py_any(severity_number, py)?)?;
            kwargs.set_item("severity_text", ::pyo3::IntoPyObjectExt::into_bound_py_any(severity_text, py)?)?;
            kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
            crate::host::construct(LOG_RECORD_WIRE.import(py, "livekit.agents._generated.telemetry", "LogRecordWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report/projection.rs.
    static SESSION_REPORT_SNAPSHOT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn session_report_snapshot_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::telemetry::report::projection::SessionReportSnapshot> {
        let py = obj.py();
            if !obj.is_instance(SESSION_REPORT_SNAPSHOT.import(py, "livekit.agents._generated.telemetry", "SessionReportSnapshot")?)? {
                return Err(super::not_a(obj, "SessionReportSnapshot"));
            }
            Ok(::lk_agents_abi::telemetry::report::projection::SessionReportSnapshot {
                job_id: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "job_id", |v| v.extract::<::std::string::String>())?,
                room_id: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "room_id", |v| v.extract::<::std::string::String>())?,
                room: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "room", |v| v.extract::<::std::string::String>())?,
                events: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "events", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; ::lk_agents_abi::python::value_from_py(item, "SessionReportSnapshot.events") }).collect::<PyResult<Vec<_>>>())?,
                omitted_events: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "omitted_events", |v| ::lk_agents_abi::python::number_from_py::<usize>(v, "SessionReportSnapshot.omitted_events"))?,
                audio_recording_path: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "audio_recording_path", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                audio_recording_started_at: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "audio_recording_started_at", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "SessionReportSnapshot.audio_recording_started_at")).map(Some) })?,
                options: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "options", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "SessionReportSnapshot.options")).map(Some) })?,
                chat_history: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "chat_history", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "SessionReportSnapshot.chat_history")).map(Some) })?,
                timestamp: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "SessionReportSnapshot.timestamp"))?,
                usage: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "usage", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::metrics::ModelUsage>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                sdk_version: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "sdk_version", |v| v.extract::<::std::string::String>())?,
            })
    }

    pub fn session_report_snapshot_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::telemetry::report::projection::SessionReportSnapshot) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::telemetry::report::projection::SessionReportSnapshot { job_id, room_id, room, events, omitted_events, audio_recording_path, audio_recording_started_at, options, chat_history, timestamp, usage, sdk_version } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("job_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(job_id, py)?)?;
            kwargs.set_item("room_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(room_id, py)?)?;
            kwargs.set_item("room", ::pyo3::IntoPyObjectExt::into_bound_py_any(room, py)?)?;
            kwargs.set_item("events", ::pyo3::types::PyList::new(py, events.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("omitted_events", ::pyo3::IntoPyObjectExt::into_bound_py_any(omitted_events, py)?)?;
            kwargs.set_item("audio_recording_path", match audio_recording_path { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("audio_recording_started_at", match audio_recording_started_at { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("options", match options { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
            kwargs.set_item("chat_history", match chat_history { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("usage", match usage { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
            kwargs.set_item("sdk_version", ::pyo3::IntoPyObjectExt::into_bound_py_any(sdk_version, py)?)?;
            crate::host::construct(SESSION_REPORT_SNAPSHOT.import(py, "livekit.agents._generated.telemetry", "SessionReportSnapshot")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/telemetry/report/projection.rs.
    static PROJECTED_REPORT_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn projected_report_event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::telemetry::report::projection::ProjectedReportEvent> {
        let py = obj.py();
            if !obj.is_instance(PROJECTED_REPORT_EVENT.import(py, "livekit.agents._generated.telemetry", "ProjectedReportEvent")?)? {
                return Err(super::not_a(obj, "ProjectedReportEvent"));
            }
            Ok(::lk_agents_abi::telemetry::report::projection::ProjectedReportEvent {
                value: ::lk_agents_abi::python::attr_with(obj, "ProjectedReportEvent", "value", |v| ::lk_agents_abi::python::value_from_py(v, "ProjectedReportEvent.value"))?,
                content: ::lk_agents_abi::python::attr_with(obj, "ProjectedReportEvent", "content", |v| v.extract::<bool>())?,
                retained_bytes: ::lk_agents_abi::python::attr_with(obj, "ProjectedReportEvent", "retained_bytes", |v| ::lk_agents_abi::python::number_from_py::<usize>(v, "ProjectedReportEvent.retained_bytes"))?,
            })
    }

    pub fn projected_report_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::telemetry::report::projection::ProjectedReportEvent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::telemetry::report::projection::ProjectedReportEvent { value, content, retained_bytes } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("value", ::lk_agents_abi::python::value_to_py(py, &value)?)?;
            kwargs.set_item("content", ::pyo3::IntoPyObjectExt::into_bound_py_any(content, py)?)?;
            kwargs.set_item("retained_bytes", ::pyo3::IntoPyObjectExt::into_bound_py_any(retained_bytes, py)?)?;
            crate::host::construct(PROJECTED_REPORT_EVENT.import(py, "livekit.agents._generated.telemetry", "ProjectedReportEvent")?.as_any(), (), Some(&kwargs))
    }
}

pub mod text_session {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static FRAME_ID: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn frame_id_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::FrameId> {
        let py = obj.py();
            if !obj.is_instance(FRAME_ID.import(py, "livekit.agents._generated.text_session", "FrameId")?)? {
                return Err(super::not_a(obj, "FrameId"));
            }
            Ok(::lk_agents_abi::text_session::FrameId {
                value: ::lk_agents_abi::python::attr_with(obj, "FrameId", "value", |v| v.extract::<::std::string::String>())?,
            })
    }

    pub fn frame_id_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::FrameId) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::text_session::FrameId { value } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("value", ::pyo3::IntoPyObjectExt::into_bound_py_any(value, py)?)?;
            crate::host::construct(FRAME_ID.import(py, "livekit.agents._generated.text_session", "FrameId")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static PARENT_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn parent_call_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::ParentCall> {
        let py = obj.py();
            if !obj.is_instance(PARENT_CALL.import(py, "livekit.agents._generated.text_session", "ParentCall")?)? {
                return Err(super::not_a(obj, "ParentCall"));
            }
            Ok(::lk_agents_abi::text_session::ParentCall {
                frame_id: ::lk_agents_abi::python::attr_with(obj, "ParentCall", "frame_id", |v| crate::generated::records::text_session::frame_id_from_py(v))?,
                call_id: ::lk_agents_abi::python::attr_with(obj, "ParentCall", "call_id", |v| v.extract::<::std::string::String>())?,
            })
    }

    pub fn parent_call_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::ParentCall) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::text_session::ParentCall { frame_id, call_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("frame_id", crate::generated::records::text_session::frame_id_to_py(py, frame_id)?)?;
            kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
            crate::host::construct(PARENT_CALL.import(py, "livekit.agents._generated.text_session", "ParentCall")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static CHILD_CALL_STATUS_APPROVAL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static CHILD_CALL_STATUS_READY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static CHILD_CALL_STATUS_RUNNING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn child_call_status_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::ChildCallStatus> {
        let py = obj.py();
            if obj.is_instance(CHILD_CALL_STATUS_APPROVAL.import(py, "livekit.agents._generated.text_session", "ChildCallStatusApproval")?)? {
                return Ok(::lk_agents_abi::text_session::ChildCallStatus::Approval);
            }
            if obj.is_instance(CHILD_CALL_STATUS_READY.import(py, "livekit.agents._generated.text_session", "ChildCallStatusReady")?)? {
                return Ok(::lk_agents_abi::text_session::ChildCallStatus::Ready);
            }
            if obj.is_instance(CHILD_CALL_STATUS_RUNNING.import(py, "livekit.agents._generated.text_session", "ChildCallStatusRunning")?)? {
                return Ok(::lk_agents_abi::text_session::ChildCallStatus::Running { frame_id: ::lk_agents_abi::python::attr_with(obj, "ChildCallStatusRunning", "frame_id", |v| crate::generated::records::text_session::frame_id_from_py(v))? });
            }
            Err(super::not_a(obj, "ChildCallStatusApproval | ChildCallStatusReady | ChildCallStatusRunning"))
    }

    pub fn child_call_status_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::ChildCallStatus) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::text_session::ChildCallStatus::Approval => crate::host::construct(CHILD_CALL_STATUS_APPROVAL.import(py, "livekit.agents._generated.text_session", "ChildCallStatusApproval")?.as_any(), (), None),
                ::lk_agents_abi::text_session::ChildCallStatus::Ready => crate::host::construct(CHILD_CALL_STATUS_READY.import(py, "livekit.agents._generated.text_session", "ChildCallStatusReady")?.as_any(), (), None),
                ::lk_agents_abi::text_session::ChildCallStatus::Running { frame_id } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("frame_id", crate::generated::records::text_session::frame_id_to_py(py, frame_id)?)?;
                    crate::host::construct(CHILD_CALL_STATUS_RUNNING.import(py, "livekit.agents._generated.text_session", "ChildCallStatusRunning")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static PENDING_CHILD_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn pending_child_call_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::PendingChildCall> {
        let py = obj.py();
            if !obj.is_instance(PENDING_CHILD_CALL.import(py, "livekit.agents._generated.text_session", "PendingChildCall")?)? {
                return Err(super::not_a(obj, "PendingChildCall"));
            }
            Ok(::lk_agents_abi::text_session::PendingChildCall {
                parent: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "parent", |v| crate::generated::records::text_session::parent_call_from_py(v))?,
                invocation_id: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "invocation_id", |v| v.extract::<::std::string::String>())?,
                execution_id: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "execution_id", |v| v.extract::<::std::string::String>())?,
                tool_name: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "tool_name", |v| v.extract::<::std::string::String>())?,
                agent_name: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "agent_name", |v| v.extract::<::std::string::String>())?,
                validated_arguments: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "validated_arguments", |v| ::lk_agents_abi::python::value_from_py(v, "PendingChildCall.validated_arguments"))?,
                identity: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "identity", |v| v.extract::<::std::string::String>())?,
                status: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "status", |v| crate::generated::records::text_session::child_call_status_from_py(v))?,
            })
    }

    pub fn pending_child_call_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::PendingChildCall) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::text_session::PendingChildCall { parent, invocation_id, execution_id, tool_name, agent_name, validated_arguments, identity, status } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("parent", crate::generated::records::text_session::parent_call_to_py(py, parent)?)?;
            kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
            kwargs.set_item("execution_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(execution_id, py)?)?;
            kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
            kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
            kwargs.set_item("validated_arguments", ::lk_agents_abi::python::value_to_py(py, &validated_arguments)?)?;
            kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
            kwargs.set_item("status", crate::generated::records::text_session::child_call_status_to_py(py, status)?)?;
            crate::host::construct(PENDING_CHILD_CALL.import(py, "livekit.agents._generated.text_session", "PendingChildCall")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static SESSION_FRAME: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn session_frame_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::SessionFrame> {
        let py = obj.py();
            if !obj.is_instance(SESSION_FRAME.import(py, "livekit.agents._generated.text_session", "SessionFrame")?)? {
                return Err(super::not_a(obj, "SessionFrame"));
            }
            Ok(::lk_agents_abi::text_session::SessionFrame {
                id: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "id", |v| crate::generated::records::text_session::frame_id_from_py(v))?,
                agent_name: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "agent_name", |v| v.extract::<::std::string::String>())?,
                history: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "history", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
                parent: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "parent", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::text_session::parent_call_from_py(v)).map(Some) })?,
                context_scope: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "context_scope", |v| v.extract::<::std::string::String>())?,
                completed: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "completed", |v| v.extract::<bool>())?,
                pending_calls: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "pending_calls", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::text_session::pending_child_call_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                answer_tool: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "answer_tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                visible_dialogue: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "visible_dialogue", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ChatMessage>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
                starting_history_ids: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "starting_history_ids", |v| v.try_iter()?.map(|item| { let item = &item?; item.extract::<::std::string::String>() }).collect::<PyResult<::std::collections::BTreeSet<_>>>())?,
                transferred_message_ids: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "transferred_message_ids", |v| v.try_iter()?.map(|item| { let item = &item?; item.extract::<::std::string::String>() }).collect::<PyResult<::std::collections::BTreeSet<_>>>())?,
                merge_chat_history: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "merge_chat_history", |v| v.extract::<bool>())?,
            })
    }

    pub fn session_frame_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::SessionFrame) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::text_session::SessionFrame { id, agent_name, history, parent, context_scope, completed, pending_calls, answer_tool, visible_dialogue, starting_history_ids, transferred_message_ids, merge_chat_history } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", crate::generated::records::text_session::frame_id_to_py(py, id)?)?;
            kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
            kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("parent", match parent { None => py.None().into_bound(py), Some(x) => crate::generated::records::text_session::parent_call_to_py(py, x)? })?;
            kwargs.set_item("context_scope", ::pyo3::IntoPyObjectExt::into_bound_py_any(context_scope, py)?)?;
            kwargs.set_item("completed", ::pyo3::IntoPyObjectExt::into_bound_py_any(completed, py)?)?;
            kwargs.set_item("pending_calls", ::pyo3::types::PyList::new(py, pending_calls.into_iter().map(|x| PyResult::Ok(crate::generated::records::text_session::pending_child_call_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("visible_dialogue", ::pyo3::types::PyList::new(py, visible_dialogue.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("starting_history_ids", ::pyo3::types::PyFrozenSet::new(py, starting_history_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("transferred_message_ids", ::pyo3::types::PyFrozenSet::new(py, transferred_message_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("merge_chat_history", ::pyo3::IntoPyObjectExt::into_bound_py_any(merge_chat_history, py)?)?;
            crate::host::construct(SESSION_FRAME.import(py, "livekit.agents._generated.text_session", "SessionFrame")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static TEXT_SESSION_STATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_session_state_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::TextSessionState> {
        let py = obj.py();
            if !obj.is_instance(TEXT_SESSION_STATE.import(py, "livekit.agents._generated.text_session", "TextSessionState")?)? {
                return Err(super::not_a(obj, "TextSessionState"));
            }
            Ok(::lk_agents_abi::text_session::TextSessionState {
                session_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "session_id", |v| v.extract::<::std::string::String>())?,
                execution_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "execution_id", |v| v.extract::<::std::string::String>())?,
                root_frame_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "root_frame_id", |v| crate::generated::records::text_session::frame_id_from_py(v))?,
                active_frame_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "active_frame_id", |v| crate::generated::records::text_session::frame_id_from_py(v))?,
                ancestors: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "ancestors", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::text_session::frame_id_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                frames: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "frames", |v| v.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((crate::generated::records::text_session::frame_id_from_py(key)?, crate::generated::records::text_session::session_frame_from_py(item)?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())?,
                next_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "next_id", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "TextSessionState.next_id"))?,
            })
    }

    pub fn text_session_state_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::TextSessionState) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::text_session::TextSessionState { session_id, execution_id, root_frame_id, active_frame_id, ancestors, frames, next_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("session_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(session_id, py)?)?;
            kwargs.set_item("execution_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(execution_id, py)?)?;
            kwargs.set_item("root_frame_id", crate::generated::records::text_session::frame_id_to_py(py, root_frame_id)?)?;
            kwargs.set_item("active_frame_id", crate::generated::records::text_session::frame_id_to_py(py, active_frame_id)?)?;
            kwargs.set_item("ancestors", ::pyo3::types::PyList::new(py, ancestors.into_iter().map(|x| PyResult::Ok(crate::generated::records::text_session::frame_id_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("frames", { let d = ::pyo3::types::PyDict::new(py); for (k, x) in frames { d.set_item(crate::generated::records::text_session::frame_id_to_py(py, k)?, crate::generated::records::text_session::session_frame_to_py(py, x)?)?; } d.into_any() })?;
            kwargs.set_item("next_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(next_id, py)?)?;
            crate::host::construct(TEXT_SESSION_STATE.import(py, "livekit.agents._generated.text_session", "TextSessionState")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.

    pub fn chat_history_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::ChatHistory> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "fresh" => Ok(::lk_agents_abi::text_session::ChatHistory::Fresh),
                "fork" => Ok(::lk_agents_abi::text_session::ChatHistory::Fork),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("ChatHistory: {other:?} is not one of 'fresh', 'fork'"))),
            }
    }

    pub fn chat_history_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::ChatHistory) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::text_session::ChatHistory::Fresh => "fresh",
                ::lk_agents_abi::text_session::ChatHistory::Fork => "fork",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static LOCAL_AGENT_TOOL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn local_agent_tool_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::LocalAgentTool> {
        let py = obj.py();
            if !obj.is_instance(LOCAL_AGENT_TOOL.import(py, "livekit.agents._generated.text_session", "LocalAgentTool")?)? {
                return Err(super::not_a(obj, "LocalAgentTool"));
            }
            Ok(::lk_agents_abi::text_session::LocalAgentTool {
                agent_name: ::lk_agents_abi::python::attr_with(obj, "LocalAgentTool", "agent_name", |v| v.extract::<::std::string::String>())?,
                chat_history: ::lk_agents_abi::python::attr_with(obj, "LocalAgentTool", "chat_history", |v| crate::generated::records::text_session::chat_history_from_py(v))?,
                merge_chat_history: ::lk_agents_abi::python::attr_with(obj, "LocalAgentTool", "merge_chat_history", |v| v.extract::<bool>())?,
                context_scope: ::lk_agents_abi::python::attr_with(obj, "LocalAgentTool", "context_scope", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    pub fn local_agent_tool_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::LocalAgentTool) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::text_session::LocalAgentTool { agent_name, chat_history, merge_chat_history, context_scope } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
            kwargs.set_item("chat_history", crate::generated::records::text_session::chat_history_to_py(py, chat_history)?)?;
            kwargs.set_item("merge_chat_history", ::pyo3::IntoPyObjectExt::into_bound_py_any(merge_chat_history, py)?)?;
            kwargs.set_item("context_scope", match context_scope { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(LOCAL_AGENT_TOOL.import(py, "livekit.agents._generated.text_session", "LocalAgentTool")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static ROUTING_ERROR_MAX_DEPTH: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ROUTING_ERROR_UNKNOWN_AGENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ROUTING_ERROR_UNKNOWN_SCOPE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ROUTING_ERROR_UNKNOWN_TOOL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ROUTING_ERROR_MISSING_OUTPUT_TYPE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ROUTING_ERROR_INVALID_STATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ROUTING_ERROR_MISSING_ACCEPTED_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn routing_error_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::RoutingError) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::text_session::RoutingError::MaxDepth { max_depth } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("max_depth", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_depth, py)?)?;
                    crate::host::construct(ROUTING_ERROR_MAX_DEPTH.import(py, "livekit.agents._generated.text_session", "RoutingErrorMaxDepth")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::RoutingError::UnknownAgent(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(ROUTING_ERROR_UNKNOWN_AGENT.import(py, "livekit.agents._generated.text_session", "RoutingErrorUnknownAgent")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::RoutingError::UnknownScope(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(ROUTING_ERROR_UNKNOWN_SCOPE.import(py, "livekit.agents._generated.text_session", "RoutingErrorUnknownScope")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::RoutingError::UnknownTool(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(ROUTING_ERROR_UNKNOWN_TOOL.import(py, "livekit.agents._generated.text_session", "RoutingErrorUnknownTool")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::RoutingError::MissingOutputType(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(ROUTING_ERROR_MISSING_OUTPUT_TYPE.import(py, "livekit.agents._generated.text_session", "RoutingErrorMissingOutputType")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::RoutingError::InvalidState(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(ROUTING_ERROR_INVALID_STATE.import(py, "livekit.agents._generated.text_session", "RoutingErrorInvalidState")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::RoutingError::MissingAcceptedOutput => crate::host::construct(ROUTING_ERROR_MISSING_ACCEPTED_OUTPUT.import(py, "livekit.agents._generated.text_session", "RoutingErrorMissingAcceptedOutput")?.as_any(), (), None),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static TEXT_SESSION_TURN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_session_turn_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::TextSessionTurn) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::text_session::TextSessionTurn { result, state } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("result", crate::generated::records::agent_run::run_result_to_py(py, result)?)?;
            kwargs.set_item("state", crate::generated::records::text_session::text_session_state_to_py(py, state)?)?;
            crate::host::construct(TEXT_SESSION_TURN.import(py, "livekit.agents._generated.text_session", "TextSessionTurn")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static TEXT_SESSION_ERROR_ROUTING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TEXT_SESSION_ERROR_RUN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_session_error_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::TextSessionError) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::text_session::TextSessionError::Routing(error) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("error", crate::generated::records::text_session::routing_error_to_py(py, error)?)?;
                    crate::host::construct(TEXT_SESSION_ERROR_ROUTING.import(py, "livekit.agents._generated.text_session", "TextSessionErrorRouting")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::TextSessionError::Run(error) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("error", crate::generated::records::agent_run::run_error_to_py(py, error)?)?;
                    crate::host::construct(TEXT_SESSION_ERROR_RUN.import(py, "livekit.agents._generated.text_session", "TextSessionErrorRun")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static TEXT_SESSION_AGENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_session_agent_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::TextSessionAgent> {
        let py = obj.py();
            if !obj.is_instance(TEXT_SESSION_AGENT.import(py, "livekit.agents._generated.text_session", "TextSessionAgent")?)? {
                return Err(super::not_a(obj, "TextSessionAgent"));
            }
            Ok(::lk_agents_abi::text_session::TextSessionAgent {
                definition: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "definition", |v| crate::generated::records::agent_run::agent_definition_from_py(v))?,
                llm: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "llm", |v| crate::generated::llm::LlmProxy::from_host(&v))?,
                host: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "host", |v| crate::generated::agent::AgentHostProxy::from_host(&v))?,
                output_type_explicit: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "output_type_explicit", |v| v.extract::<bool>())?,
                agent_tools: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "agent_tools", |v| v.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, crate::generated::records::text_session::local_agent_tool_from_py(item)?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())?,
                host_context: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "host_context", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))?,
                tools_host: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "tools_host", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static TEXT_SESSION_REQUEST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_session_request_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::TextSessionRequest> {
        let py = obj.py();
            if !obj.is_instance(TEXT_SESSION_REQUEST.import(py, "livekit.agents._generated.text_session", "TextSessionRequest")?)? {
                return Err(super::not_a(obj, "TextSessionRequest"));
            }
            Ok(::lk_agents_abi::text_session::TextSessionRequest {
                session_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "session_id", |v| v.extract::<::std::string::String>())?,
                execution_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "execution_id", |v| v.extract::<::std::string::String>())?,
                task_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "task_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                root_scope: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "root_scope", |v| v.extract::<::std::string::String>())?,
                contexts: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "contexts", |v| v.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, item.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())?,
                separate_contexts: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "separate_contexts", |v| v.extract::<bool>())?,
                message: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "message", |v| v.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into))?,
                state: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "state", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::text_session::text_session_state_from_py(v)).map(Some) })?,
                delegated: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "delegated", |v| v.extract::<bool>())?,
                conn_options: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "conn_options", |v| v.extract::<::lk_agents_abi::ApiConnectOptions>().map_err(::std::convert::Into::<PyErr>::into))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static TEXT_SESSION_SETTLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_session_settled_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::TextSessionSettled) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::text_session::TextSessionSettled { turn, output } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("turn", crate::generated::records::text_session::text_session_turn_to_py(py, turn)?)?;
            kwargs.set_item("output", match output { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(TEXT_SESSION_SETTLED.import(py, "livekit.agents._generated.text_session", "TextSessionSettled")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static TEXT_SESSION_UPDATE_STREAM: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TEXT_SESSION_UPDATE_INVOCATION_SETTLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TEXT_SESSION_UPDATE_PARKED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TEXT_SESSION_UPDATE_INCOMPLETE_CLEANUP: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TEXT_SESSION_UPDATE_FINISHED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static TEXT_SESSION_UPDATE_FAILED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_session_update_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::text_session::TextSessionUpdate) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::text_session::TextSessionUpdate::Stream { sequence, frame_id, agent_name, invocation_id, event } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("sequence", ::pyo3::IntoPyObjectExt::into_bound_py_any(sequence, py)?)?;
                    kwargs.set_item("frame_id", crate::generated::records::text_session::frame_id_to_py(py, frame_id)?)?;
                    kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
                    kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
                    kwargs.set_item("event", crate::generated::records::agent_run::stream_event_to_py(py, event)?)?;
                    crate::host::construct(TEXT_SESSION_UPDATE_STREAM.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateStream")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::TextSessionUpdate::InvocationSettled { invocation_id, usage } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
                    kwargs.set_item("usage", crate::generated::records::agent_run::run_usage_to_py(py, usage)?)?;
                    crate::host::construct(TEXT_SESSION_UPDATE_INVOCATION_SETTLED.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateInvocationSettled")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::TextSessionUpdate::Parked { invocation_id, usage } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
                    kwargs.set_item("usage", crate::generated::records::agent_run::run_usage_to_py(py, usage)?)?;
                    crate::host::construct(TEXT_SESSION_UPDATE_PARKED.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateParked")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::TextSessionUpdate::IncompleteCleanup(pending) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("pending", ::pyo3::types::PyList::new(py, pending.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
                    crate::host::construct(TEXT_SESSION_UPDATE_INCOMPLETE_CLEANUP.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateIncompleteCleanup")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::TextSessionUpdate::Finished(settled) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("settled", crate::generated::records::text_session::text_session_settled_to_py(py, settled)?)?;
                    crate::host::construct(TEXT_SESSION_UPDATE_FINISHED.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateFinished")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::text_session::TextSessionUpdate::Failed(error) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("error", crate::generated::records::text_session::text_session_error_to_py(py, error)?)?;
                    crate::host::construct(TEXT_SESSION_UPDATE_FAILED.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateFailed")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/text_session.rs.
    static TEXT_SESSION_DELIVERY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_session_delivery_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::text_session::TextSessionDelivery> {
        let py = obj.py();
            if !obj.is_instance(TEXT_SESSION_DELIVERY.import(py, "livekit.agents._generated.text_session", "TextSessionDelivery")?)? {
                return Err(super::not_a(obj, "TextSessionDelivery"));
            }
            Ok(::lk_agents_abi::text_session::TextSessionDelivery {
                invocation_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionDelivery", "invocation_id", |v| v.extract::<::std::string::String>())?,
                messages: ::lk_agents_abi::python::attr_with(obj, "TextSessionDelivery", "messages", |v| v.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, item.extract::<::std::string::String>()?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())?,
            })
    }
}

pub mod tokenizer {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.

    pub fn tokenizer_token_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::tokenize::bindings::TokenizerToken> {
        if obj.is_instance_of::<::pyo3::types::PyString>() {
            return Ok(::lk_agents_abi::tokenize::bindings::TokenizerToken::Text(obj.extract::<::std::string::String>()?));
        }
        if obj.is_instance_of::<::pyo3::types::PyTuple>() {
            return Ok(::lk_agents_abi::tokenize::bindings::TokenizerToken::Span((|| -> ::pyo3::PyResult<_> { let tuple = obj.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 3 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 3 items", "TokenizerToken.Span"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; Ok(((item_0.extract::<::std::string::String>())?, (::lk_agents_abi::python::number_from_py::<usize>(&item_1, "TokenizerToken.Span"))?, (::lk_agents_abi::python::number_from_py::<usize>(&item_2, "TokenizerToken.Span"))?,)) })()?));
        }
        Err(super::not_a(obj, "TokenizerToken"))
    }
}

pub mod tool_report {
    use pyo3::prelude::*;

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

    pub fn tool_outcome_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::tool_executor::ToolOutcome> {
        let py = obj.py();
            if obj.is_instance(TOOL_OUTCOME_VALUE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeValue")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Value { text: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeValue", "text", |v| v.extract::<::std::string::String>())?, reply_required: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeValue", "reply_required", |v| v.extract::<bool>())?, verbatim: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeValue", "verbatim", |v| v.extract::<bool>())?, more: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeValue", "more", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
            }
            if obj.is_instance(TOOL_OUTCOME_NOTHING.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeNothing")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Nothing);
            }
            if obj.is_instance(TOOL_OUTCOME_TOOL_ERROR.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeToolError")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::ToolError(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeToolError", "message", |v| v.extract::<::std::string::String>())?));
            }
            if obj.is_instance(TOOL_OUTCOME_MODEL_RETRY.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeModelRetry")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::ModelRetry(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeModelRetry", "message", |v| v.extract::<::std::string::String>())?));
            }
            if obj.is_instance(TOOL_OUTCOME_STOP_RESPONSE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeStopResponse")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::StopResponse);
            }
            if obj.is_instance(TOOL_OUTCOME_INTERNAL.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInternal")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Internal);
            }
            if obj.is_instance(TOOL_OUTCOME_INVALID.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInvalid")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Invalid);
            }
            if obj.is_instance(TOOL_OUTCOME_CANCELLED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeCancelled")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Cancelled);
            }
            if obj.is_instance(TOOL_OUTCOME_UPDATE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeUpdate")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Update { message: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeUpdate", "message", |v| v.extract::<::std::string::String>())?, raw: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeUpdate", "raw", |v| v.extract::<::std::string::String>())? });
            }
            if obj.is_instance(TOOL_OUTCOME_COMPLETED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeCompleted")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Completed { text: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeCompleted", "text", |v| v.extract::<::std::string::String>())?, verbatim: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeCompleted", "verbatim", |v| v.extract::<bool>())? });
            }
            if obj.is_instance(TOOL_OUTCOME_FAILED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeFailed")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Failed(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeFailed", "error", |v| v.extract::<::std::string::String>())?));
            }
            if obj.is_instance(TOOL_OUTCOME_REJECTED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeRejected")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Rejected(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeRejected", "reason", |v| v.extract::<::std::string::String>())?));
            }
            if obj.is_instance(TOOL_OUTCOME_INPUT_REQUIRED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInputRequired")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::InputRequired { question: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeInputRequired", "question", |v| v.extract::<::std::string::String>())?, verbatim: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeInputRequired", "verbatim", |v| v.extract::<bool>())?, tool: ::lk_agents_abi::python::attr_with(obj, "ToolOutcomeInputRequired", "tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
            }
            if obj.is_instance(TOOL_OUTCOME_WORKING.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeWorking")?)? {
                return Ok(::lk_agents_abi::llm::tool_executor::ToolOutcome::Working(::lk_agents_abi::python::attr_with(obj, "ToolOutcomeWorking", "message", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?));
            }
            Err(super::not_a(obj, "ToolOutcomeValue | ToolOutcomeNothing | ToolOutcomeToolError | ToolOutcomeModelRetry | ToolOutcomeStopResponse | ToolOutcomeInternal | ToolOutcomeInvalid | ToolOutcomeCancelled | ToolOutcomeUpdate | ToolOutcomeCompleted | ToolOutcomeFailed | ToolOutcomeRejected | ToolOutcomeInputRequired | ToolOutcomeWorking"))
    }

    pub fn tool_outcome_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::tool_executor::ToolOutcome) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Value { text, reply_required, verbatim, more } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                    kwargs.set_item("reply_required", ::pyo3::IntoPyObjectExt::into_bound_py_any(reply_required, py)?)?;
                    kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
                    kwargs.set_item("more", match more { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    crate::host::construct(TOOL_OUTCOME_VALUE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeValue")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Nothing => crate::host::construct(TOOL_OUTCOME_NOTHING.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeNothing")?.as_any(), (), None),
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::ToolError(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(TOOL_OUTCOME_TOOL_ERROR.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeToolError")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::ModelRetry(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(TOOL_OUTCOME_MODEL_RETRY.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeModelRetry")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::StopResponse => crate::host::construct(TOOL_OUTCOME_STOP_RESPONSE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeStopResponse")?.as_any(), (), None),
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Internal => crate::host::construct(TOOL_OUTCOME_INTERNAL.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInternal")?.as_any(), (), None),
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Invalid => crate::host::construct(TOOL_OUTCOME_INVALID.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInvalid")?.as_any(), (), None),
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Cancelled => crate::host::construct(TOOL_OUTCOME_CANCELLED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeCancelled")?.as_any(), (), None),
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Update { message, raw } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    kwargs.set_item("raw", ::pyo3::IntoPyObjectExt::into_bound_py_any(raw, py)?)?;
                    crate::host::construct(TOOL_OUTCOME_UPDATE.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeUpdate")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Completed { text, verbatim } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                    kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
                    crate::host::construct(TOOL_OUTCOME_COMPLETED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeCompleted")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Failed(error) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
                    crate::host::construct(TOOL_OUTCOME_FAILED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeFailed")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Rejected(reason) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("reason", ::pyo3::IntoPyObjectExt::into_bound_py_any(reason, py)?)?;
                    crate::host::construct(TOOL_OUTCOME_REJECTED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeRejected")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::InputRequired { question, verbatim, tool } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
                    kwargs.set_item("verbatim", ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim, py)?)?;
                    kwargs.set_item("tool", match tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    crate::host::construct(TOOL_OUTCOME_INPUT_REQUIRED.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeInputRequired")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::llm::tool_executor::ToolOutcome::Working(message) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", match message { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    crate::host::construct(TOOL_OUTCOME_WORKING.import(py, "livekit.agents._generated.tool_report", "ToolOutcomeWorking")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/tool_executor.rs.

    pub fn approval_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::tool_executor::Approval> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "asked" => Ok(::lk_agents_abi::llm::tool_executor::Approval::Asked),
                "granted" => Ok(::lk_agents_abi::llm::tool_executor::Approval::Granted),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("Approval: {other:?} is not one of 'asked', 'granted'"))),
            }
    }

    pub fn approval_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::tool_executor::Approval) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::llm::tool_executor::Approval::Asked => "asked",
                ::lk_agents_abi::llm::tool_executor::Approval::Granted => "granted",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/tool_executor.rs.
    static TOOL_REPORT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_report_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::tool_executor::ToolReport> {
        let py = obj.py();
            if !obj.is_instance(TOOL_REPORT.import(py, "livekit.agents._generated.tool_report", "ToolReport")?)? {
                return Err(super::not_a(obj, "ToolReport"));
            }
            Ok(::lk_agents_abi::llm::tool_executor::ToolReport {
                outcome: ::lk_agents_abi::python::attr_with(obj, "ToolReport", "outcome", |v| crate::generated::records::tool_report::tool_outcome_from_py(v))?,
                transition: ::lk_agents_abi::python::attr_with(obj, "ToolReport", "transition", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            })
    }
}

pub mod turn_events {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/tokenize/language/eot_thresholds.rs.

    pub fn turn_detector_model_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::tokenize::language::TurnDetectorModel> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "turn-detector-v1" => Ok(::lk_agents_abi::tokenize::language::TurnDetectorModel::V1),
                "turn-detector-v1-mini" => Ok(::lk_agents_abi::tokenize::language::TurnDetectorModel::V1Mini),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("TurnDetectorModel: {other:?} is not one of 'turn-detector-v1', 'turn-detector-v1-mini'"))),
            }
    }

    pub fn turn_detector_model_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::tokenize::language::TurnDetectorModel) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::tokenize::language::TurnDetectorModel::V1 => "turn-detector-v1",
                ::lk_agents_abi::tokenize::language::TurnDetectorModel::V1Mini => "turn-detector-v1-mini",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/tokenize/language/eot_thresholds.rs.

    pub fn threshold_override_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::tokenize::language::ThresholdOverride> {
        if (obj.is_instance_of::<::pyo3::types::PyFloat>() || (obj.is_instance_of::<::pyo3::types::PyInt>() && !obj.is_instance_of::<::pyo3::types::PyBool>())) {
            return Ok(::lk_agents_abi::tokenize::language::ThresholdOverride::All(::lk_agents_abi::python::number_from_py::<f64>(obj, "ThresholdOverride.All")?));
        }
        if obj.is_instance_of::<::pyo3::types::PyDict>() {
            return Ok(::lk_agents_abi::tokenize::language::ThresholdOverride::PerLanguage(obj.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, ::lk_agents_abi::python::number_from_py::<f64>(item, "ThresholdOverride.PerLanguage")?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>()?));
        }
        Err(super::not_a(obj, "ThresholdOverride"))
    }

    pub fn threshold_override_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::tokenize::language::ThresholdOverride) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::tokenize::language::ThresholdOverride::All(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
            ::lk_agents_abi::tokenize::language::ThresholdOverride::PerLanguage(x) => { let d = ::pyo3::types::PyDict::new(py); for (k, x) in x { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?; } d.into_any() },
        })
    }

    // Declared in rust/crates/lk-agents-abi/src/turn/events.rs.
    static EOT_PREDICTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn eot_prediction_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::EotPrediction> {
        let py = obj.py();
            if !obj.is_instance(EOT_PREDICTION.import(py, "livekit.agents._generated.turn_events", "EotPrediction")?)? {
                return Err(super::not_a(obj, "EotPrediction"));
            }
            Ok(::lk_agents_abi::EotPrediction {
                end_of_turn_probability: ::lk_agents_abi::python::attr_with(obj, "EotPrediction", "end_of_turn_probability", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EotPrediction.end_of_turn_probability"))?,
                last_speaking_time: ::lk_agents_abi::python::attr_with(obj, "EotPrediction", "last_speaking_time", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EotPrediction.last_speaking_time"))?,
                detection_delay: ::lk_agents_abi::python::attr_with(obj, "EotPrediction", "detection_delay", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "EotPrediction.detection_delay")).map(Some) })?,
                inference_duration: ::lk_agents_abi::python::attr_with(obj, "EotPrediction", "inference_duration", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "EotPrediction.inference_duration")).map(Some) })?,
                backchannel_probability: ::lk_agents_abi::python::attr_with(obj, "EotPrediction", "backchannel_probability", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "EotPrediction.backchannel_probability")).map(Some) })?,
            })
    }

    pub fn eot_prediction_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::EotPrediction) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::EotPrediction { end_of_turn_probability, last_speaking_time, detection_delay, inference_duration, backchannel_probability } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("end_of_turn_probability", ::pyo3::IntoPyObjectExt::into_bound_py_any(end_of_turn_probability, py)?)?;
            kwargs.set_item("last_speaking_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(last_speaking_time, py)?)?;
            kwargs.set_item("detection_delay", match detection_delay { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("inference_duration", match inference_duration { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("backchannel_probability", match backchannel_probability { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(EOT_PREDICTION.import(py, "livekit.agents._generated.turn_events", "EotPrediction")?.as_any(), (), Some(&kwargs))
    }
}

pub mod voice_session {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/session.rs.
    static UPDATE_AGENT_ARGS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn update_agent_args_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session::UpdateAgentArgs> {
        let py = obj.py();
            if !obj.is_instance(UPDATE_AGENT_ARGS.import(py, "livekit.agents._generated.voice_session", "UpdateAgentArgs")?)? {
                return Err(super::not_a(obj, "UpdateAgentArgs"));
            }
            Ok(::lk_agents_abi::voice::session::UpdateAgentArgs {
                instructions: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "instructions", |v| v.extract::<::std::string::String>())?,
                dynamic_instructions: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "dynamic_instructions", |v| if v.is_none() { Ok(None) } else { (crate::generated::agent::HostInstructionsProxy::from_host(&v)).map(Some) })?,
                agent: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "agent", |v| v.extract::<::std::string::String>())?,
                agent_host: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "agent_host", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
                tools: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "tools", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ToolSpec>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                tools_host: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "tools_host", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
                voice_agent: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "voice_agent", |v| ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::AgentFrame>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.voice_agent", ": expected an agent's voice frame (`_core.VoiceAgent`)"))))?,
                chat_ctx: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "chat_ctx", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                llm: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "llm", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (crate::generated::llm::LlmProxy::from_host(&v)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                tts: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "tts", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (crate::generated::tts::TtsProxy::from_host(&v)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                stt: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "stt", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (crate::generated::stt::SttProxy::from_host(&v)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                vad: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "vad", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (crate::generated::vad::VadProxy::from_host(&v)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                turn_handling: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "turn_handling", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::voice::session_options::TurnHandlingKwargs>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
                min_consecutive_speech_delay: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "min_consecutive_speech_delay", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "UpdateAgentArgs.min_consecutive_speech_delay")).map(Some) })?,
                use_tts_aligned_transcript: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "use_tts_aligned_transcript", |v| if v.is_none() { Ok(None) } else { (v.extract::<bool>()).map(Some) })?,
                expressive: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "expressive", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (v.extract::<::lk_agents_abi::voice::session_options::Expressive>().map_err(::std::convert::Into::<PyErr>::into)).map(::lk_agents_abi::Maybe::Given) })?,
                llm_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "llm_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.llm_node", ": expected an agent's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                tts_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "tts_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.tts_node", ": expected an agent's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                stt_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "stt_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.stt_node", ": expected an agent's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                transcription_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "transcription_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.transcription_node", ": expected an agent's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                realtime_audio_output_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "realtime_audio_output_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.realtime_audio_output_node", ": expected an agent's nodes (`_core.AgentNodes`)")))).map(Some) })?,
            })
    }
}

pub mod worker_cli {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.
    static CLI_ARGS_START: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static CLI_ARGS_CONSOLE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static CLI_ARGS_DOWNLOAD_FILES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn cli_args_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::worker::cli::CliArgs> {
        let py = obj.py();
            if obj.is_instance(CLI_ARGS_START.import(py, "livekit.agents._generated.worker_cli", "CliArgsStart")?)? {
                return Ok(::lk_agents_abi::worker::cli::CliArgs::Start { entrypoint: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "entrypoint", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, log_level: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "log_level", |v| v.extract::<::std::string::String>())?, log_format: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "log_format", |v| v.extract::<::std::string::String>())?, log_style: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "log_style", |v| v.extract::<::std::string::String>())?, dev: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "dev", |v| v.extract::<bool>())?, cli_addr: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "cli_addr", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, simulation: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "simulation", |v| v.extract::<bool>())?, ws_url: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "ws_url", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, api_key: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "api_key", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, api_secret: ::lk_agents_abi::python::attr_with(obj, "CliArgsStart", "api_secret", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })? });
            }
            if obj.is_instance(CLI_ARGS_CONSOLE.import(py, "livekit.agents._generated.worker_cli", "CliArgsConsole")?)? {
                return Ok(::lk_agents_abi::worker::cli::CliArgs::Console { entrypoint: ::lk_agents_abi::python::attr_with(obj, "CliArgsConsole", "entrypoint", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, connect_addr: ::lk_agents_abi::python::attr_with(obj, "CliArgsConsole", "connect_addr", |v| v.extract::<::std::string::String>())?, record: ::lk_agents_abi::python::attr_with(obj, "CliArgsConsole", "record", |v| v.extract::<bool>())? });
            }
            if obj.is_instance(CLI_ARGS_DOWNLOAD_FILES.import(py, "livekit.agents._generated.worker_cli", "CliArgsDownloadFiles")?)? {
                return Ok(::lk_agents_abi::worker::cli::CliArgs::DownloadFiles);
            }
            Err(super::not_a(obj, "CliArgsStart | CliArgsConsole | CliArgsDownloadFiles"))
    }

    pub fn cli_args_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::cli::CliArgs) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::worker::cli::CliArgs::Start { entrypoint, log_level, log_format, log_style, dev, cli_addr, simulation, ws_url, api_key, api_secret } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("entrypoint", match entrypoint { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("log_level", ::pyo3::IntoPyObjectExt::into_bound_py_any(log_level, py)?)?;
                    kwargs.set_item("log_format", ::pyo3::IntoPyObjectExt::into_bound_py_any(log_format, py)?)?;
                    kwargs.set_item("log_style", ::pyo3::IntoPyObjectExt::into_bound_py_any(log_style, py)?)?;
                    kwargs.set_item("dev", ::pyo3::IntoPyObjectExt::into_bound_py_any(dev, py)?)?;
                    kwargs.set_item("cli_addr", match cli_addr { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("simulation", ::pyo3::IntoPyObjectExt::into_bound_py_any(simulation, py)?)?;
                    kwargs.set_item("ws_url", match ws_url { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("api_key", match api_key { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("api_secret", match api_secret { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    crate::host::construct(CLI_ARGS_START.import(py, "livekit.agents._generated.worker_cli", "CliArgsStart")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::cli::CliArgs::Console { entrypoint, connect_addr, record } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("entrypoint", match entrypoint { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("connect_addr", ::pyo3::IntoPyObjectExt::into_bound_py_any(connect_addr, py)?)?;
                    kwargs.set_item("record", ::pyo3::IntoPyObjectExt::into_bound_py_any(record, py)?)?;
                    crate::host::construct(CLI_ARGS_CONSOLE.import(py, "livekit.agents._generated.worker_cli", "CliArgsConsole")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::cli::CliArgs::DownloadFiles => crate::host::construct(CLI_ARGS_DOWNLOAD_FILES.import(py, "livekit.agents._generated.worker_cli", "CliArgsDownloadFiles")?.as_any(), (), None),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/cli.rs.
    static LOG_FIELDS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn log_fields_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::worker::cli::LogFields> {
        let py = obj.py();
            if !obj.is_instance(LOG_FIELDS.import(py, "livekit.agents._generated.worker_cli", "LogFields")?)? {
                return Err(super::not_a(obj, "LogFields"));
            }
            Ok(::lk_agents_abi::worker::cli::LogFields {
                level: ::lk_agents_abi::python::attr_with(obj, "LogFields", "level", |v| v.extract::<::std::string::String>())?,
                name: ::lk_agents_abi::python::attr_with(obj, "LogFields", "name", |v| v.extract::<::std::string::String>())?,
                pid: ::lk_agents_abi::python::attr_with(obj, "LogFields", "pid", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<i64>(v, "LogFields.pid")).map(Some) })?,
                created: ::lk_agents_abi::python::attr_with(obj, "LogFields", "created", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "LogFields.created"))?,
                msg: ::lk_agents_abi::python::attr_with(obj, "LogFields", "msg", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "LogFields.msg")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("LogFields.msg: expected a dict")) }).map(Some) })?,
                extra: ::lk_agents_abi::python::attr_with(obj, "LogFields", "extra", |v| match ::lk_agents_abi::python::value_from_py(v, "LogFields.extra")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("LogFields.extra: expected a dict")) })?,
            })
    }
}

pub mod worker_endpoint {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/worker/endpoint.rs.
    static ENDPOINT_REQUEST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn endpoint_request_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::worker::endpoint::EndpointRequest> {
        let py = obj.py();
            if !obj.is_instance(ENDPOINT_REQUEST.import(py, "livekit.agents._generated.worker_endpoint", "EndpointRequest")?)? {
                return Err(super::not_a(obj, "EndpointRequest"));
            }
            Ok(::lk_agents_abi::worker::endpoint::EndpointRequest {
                request_id: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "request_id", |v| v.extract::<::std::string::String>())?,
                authorized: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "authorized", |v| v.extract::<bool>())?,
                route: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "route", |v| v.extract::<::std::string::String>())?,
                kind: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "kind", |v| v.extract::<::std::string::String>())?,
                client_addr: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "client_addr", |v| v.extract::<::std::string::String>())?,
                scheme: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "scheme", |v| v.extract::<::std::string::String>())?,
                timeout_ms: ::lk_agents_abi::python::attr_with(obj, "EndpointRequest", "timeout_ms", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "EndpointRequest.timeout_ms"))?,
            })
    }

    pub fn endpoint_request_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::endpoint::EndpointRequest) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::endpoint::EndpointRequest { request_id, authorized, route, kind, client_addr, scheme, timeout_ms } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("request_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id, py)?)?;
            kwargs.set_item("authorized", ::pyo3::IntoPyObjectExt::into_bound_py_any(authorized, py)?)?;
            kwargs.set_item("route", ::pyo3::IntoPyObjectExt::into_bound_py_any(route, py)?)?;
            kwargs.set_item("kind", ::pyo3::IntoPyObjectExt::into_bound_py_any(kind, py)?)?;
            kwargs.set_item("client_addr", ::pyo3::IntoPyObjectExt::into_bound_py_any(client_addr, py)?)?;
            kwargs.set_item("scheme", ::pyo3::IntoPyObjectExt::into_bound_py_any(scheme, py)?)?;
            kwargs.set_item("timeout_ms", ::pyo3::IntoPyObjectExt::into_bound_py_any(timeout_ms, py)?)?;
            crate::host::construct(ENDPOINT_REQUEST.import(py, "livekit.agents._generated.worker_endpoint", "EndpointRequest")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/endpoint.rs.
    static ENDPOINT_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn endpoint_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::worker::endpoint::EndpointOptions> {
        let py = obj.py();
            if !obj.is_instance(ENDPOINT_OPTIONS.import(py, "livekit.agents._generated.worker_endpoint", "EndpointOptions")?)? {
                return Err(super::not_a(obj, "EndpointOptions"));
            }
            Ok(::lk_agents_abi::worker::endpoint::EndpointOptions {
                public: ::lk_agents_abi::python::attr_with(obj, "EndpointOptions", "public", |v| v.extract::<bool>())?,
            })
    }

    pub fn endpoint_options_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::endpoint::EndpointOptions) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::endpoint::EndpointOptions { public } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("public", ::pyo3::IntoPyObjectExt::into_bound_py_any(public, py)?)?;
            crate::host::construct(ENDPOINT_OPTIONS.import(py, "livekit.agents._generated.worker_endpoint", "EndpointOptions")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
    static ASGI_SCOPE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn asgi_scope_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::endpoint_host::AsgiScope) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::endpoint_host::AsgiScope { kind, method, scheme, path, raw_path, query_string, headers, client, subprotocols } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("kind", ::pyo3::IntoPyObjectExt::into_bound_py_any(kind, py)?)?;
            kwargs.set_item("method", ::pyo3::IntoPyObjectExt::into_bound_py_any(method, py)?)?;
            kwargs.set_item("scheme", ::pyo3::IntoPyObjectExt::into_bound_py_any(scheme, py)?)?;
            kwargs.set_item("path", ::pyo3::IntoPyObjectExt::into_bound_py_any(path, py)?)?;
            kwargs.set_item("raw_path", ::pyo3::types::PyBytes::new(py, &raw_path).into_any())?;
            kwargs.set_item("query_string", ::pyo3::types::PyBytes::new(py, &query_string).into_any())?;
            kwargs.set_item("headers", ::pyo3::types::PyList::new(py, headers.into_iter().map(|x| PyResult::Ok({ let (x0, x1,) = x; ::pyo3::types::PyTuple::new(py, [::pyo3::types::PyBytes::new(py, &x0).into_any(), ::pyo3::types::PyBytes::new(py, &x1).into_any()])?.into_any() })).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("client", match client { None => py.None().into_bound(py), Some(x) => { let (x0, x1,) = x; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?])?.into_any() } })?;
            kwargs.set_item("subprotocols", ::pyo3::types::PyList::new(py, subprotocols.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(ASGI_SCOPE.import(py, "livekit.agents._generated.worker_endpoint", "AsgiScope")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
    static ASGI_RECEIVED_REQUEST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_RECEIVED_DISCONNECT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_RECEIVED_WS_CONNECT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_RECEIVED_WS_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_RECEIVED_WS_BYTES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_RECEIVED_WS_DISCONNECT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn asgi_received_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::endpoint_host::AsgiReceived) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::worker::endpoint_host::AsgiReceived::Request { body, more_body } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("body", ::pyo3::types::PyBytes::new(py, &body).into_any())?;
                    kwargs.set_item("more_body", ::pyo3::IntoPyObjectExt::into_bound_py_any(more_body, py)?)?;
                    crate::host::construct(ASGI_RECEIVED_REQUEST.import(py, "livekit.agents._generated.worker_endpoint", "AsgiReceivedRequest")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::endpoint_host::AsgiReceived::Disconnect => crate::host::construct(ASGI_RECEIVED_DISCONNECT.import(py, "livekit.agents._generated.worker_endpoint", "AsgiReceivedDisconnect")?.as_any(), (), None),
                ::lk_agents_abi::worker::endpoint_host::AsgiReceived::WsConnect => crate::host::construct(ASGI_RECEIVED_WS_CONNECT.import(py, "livekit.agents._generated.worker_endpoint", "AsgiReceivedWsConnect")?.as_any(), (), None),
                ::lk_agents_abi::worker::endpoint_host::AsgiReceived::WsText(text) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                    crate::host::construct(ASGI_RECEIVED_WS_TEXT.import(py, "livekit.agents._generated.worker_endpoint", "AsgiReceivedWsText")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::endpoint_host::AsgiReceived::WsBytes(data) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("data", ::pyo3::types::PyBytes::new(py, &data).into_any())?;
                    crate::host::construct(ASGI_RECEIVED_WS_BYTES.import(py, "livekit.agents._generated.worker_endpoint", "AsgiReceivedWsBytes")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::endpoint_host::AsgiReceived::WsDisconnect { code } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("code", ::pyo3::IntoPyObjectExt::into_bound_py_any(code, py)?)?;
                    crate::host::construct(ASGI_RECEIVED_WS_DISCONNECT.import(py, "livekit.agents._generated.worker_endpoint", "AsgiReceivedWsDisconnect")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
    static ASGI_SENT_RESPONSE_START: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_RESPONSE_BODY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_RESPONSE_TRAILERS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_WS_ACCEPT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_WS_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_WS_BYTES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_WS_CLOSE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_WS_HTTP_RESPONSE_START: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_WS_HTTP_RESPONSE_BODY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static ASGI_SENT_UNKNOWN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn asgi_sent_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::worker::endpoint_host::AsgiSent> {
        let py = obj.py();
            if obj.is_instance(ASGI_SENT_RESPONSE_START.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentResponseStart")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::ResponseStart { status: ::lk_agents_abi::python::attr_with(obj, "AsgiSentResponseStart", "status", |v| ::lk_agents_abi::python::number_from_py::<u16>(v, "AsgiSentResponseStart.status"))?, headers: ::lk_agents_abi::python::attr_with(obj, "AsgiSentResponseStart", "headers", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; (|| -> ::pyo3::PyResult<_> { let tuple = item.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "AsgiSentResponseStart.headers"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok((({ if item_0.is_instance_of::<::pyo3::types::PyBytes>() || item_0.is_instance_of::<::pyo3::types::PyByteArray>() || item_0.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&item_0, "AsgiSentResponseStart.headers") } else { item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentResponseStart.headers") }).collect::<PyResult<Vec<_>>>()) } })?, ({ if item_1.is_instance_of::<::pyo3::types::PyBytes>() || item_1.is_instance_of::<::pyo3::types::PyByteArray>() || item_1.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&item_1, "AsgiSentResponseStart.headers") } else { item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentResponseStart.headers") }).collect::<PyResult<Vec<_>>>()) } })?,)) })() }).collect::<PyResult<Vec<_>>>())? });
            }
            if obj.is_instance(ASGI_SENT_RESPONSE_BODY.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentResponseBody")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::ResponseBody { body: ::lk_agents_abi::python::attr_with(obj, "AsgiSentResponseBody", "body", |v| { if v.is_instance_of::<::pyo3::types::PyBytes>() || v.is_instance_of::<::pyo3::types::PyByteArray>() || v.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&v, "AsgiSentResponseBody.body") } else { v.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentResponseBody.body") }).collect::<PyResult<Vec<_>>>()) } })?, more_body: ::lk_agents_abi::python::attr_with(obj, "AsgiSentResponseBody", "more_body", |v| v.extract::<bool>())? });
            }
            if obj.is_instance(ASGI_SENT_RESPONSE_TRAILERS.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentResponseTrailers")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::ResponseTrailers);
            }
            if obj.is_instance(ASGI_SENT_WS_ACCEPT.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentWsAccept")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::WsAccept { subprotocol: ::lk_agents_abi::python::attr_with(obj, "AsgiSentWsAccept", "subprotocol", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?, headers: ::lk_agents_abi::python::attr_with(obj, "AsgiSentWsAccept", "headers", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; (|| -> ::pyo3::PyResult<_> { let tuple = item.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "AsgiSentWsAccept.headers"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok((({ if item_0.is_instance_of::<::pyo3::types::PyBytes>() || item_0.is_instance_of::<::pyo3::types::PyByteArray>() || item_0.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&item_0, "AsgiSentWsAccept.headers") } else { item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentWsAccept.headers") }).collect::<PyResult<Vec<_>>>()) } })?, ({ if item_1.is_instance_of::<::pyo3::types::PyBytes>() || item_1.is_instance_of::<::pyo3::types::PyByteArray>() || item_1.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&item_1, "AsgiSentWsAccept.headers") } else { item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentWsAccept.headers") }).collect::<PyResult<Vec<_>>>()) } })?,)) })() }).collect::<PyResult<Vec<_>>>())? });
            }
            if obj.is_instance(ASGI_SENT_WS_TEXT.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentWsText")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::WsText(::lk_agents_abi::python::attr_with(obj, "AsgiSentWsText", "text", |v| v.extract::<::std::string::String>())?));
            }
            if obj.is_instance(ASGI_SENT_WS_BYTES.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentWsBytes")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::WsBytes(::lk_agents_abi::python::attr_with(obj, "AsgiSentWsBytes", "data", |v| { if v.is_instance_of::<::pyo3::types::PyBytes>() || v.is_instance_of::<::pyo3::types::PyByteArray>() || v.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&v, "AsgiSentWsBytes.data") } else { v.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentWsBytes.data") }).collect::<PyResult<Vec<_>>>()) } })?));
            }
            if obj.is_instance(ASGI_SENT_WS_CLOSE.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentWsClose")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::WsClose { code: ::lk_agents_abi::python::attr_with(obj, "AsgiSentWsClose", "code", |v| ::lk_agents_abi::python::number_from_py::<u16>(v, "AsgiSentWsClose.code"))?, reason: ::lk_agents_abi::python::attr_with(obj, "AsgiSentWsClose", "reason", |v| v.extract::<::std::string::String>())? });
            }
            if obj.is_instance(ASGI_SENT_WS_HTTP_RESPONSE_START.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentWsHttpResponseStart")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::WsHttpResponseStart { status: ::lk_agents_abi::python::attr_with(obj, "AsgiSentWsHttpResponseStart", "status", |v| ::lk_agents_abi::python::number_from_py::<u16>(v, "AsgiSentWsHttpResponseStart.status"))?, headers: ::lk_agents_abi::python::attr_with(obj, "AsgiSentWsHttpResponseStart", "headers", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; (|| -> ::pyo3::PyResult<_> { let tuple = item.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "AsgiSentWsHttpResponseStart.headers"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok((({ if item_0.is_instance_of::<::pyo3::types::PyBytes>() || item_0.is_instance_of::<::pyo3::types::PyByteArray>() || item_0.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&item_0, "AsgiSentWsHttpResponseStart.headers") } else { item_0.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentWsHttpResponseStart.headers") }).collect::<PyResult<Vec<_>>>()) } })?, ({ if item_1.is_instance_of::<::pyo3::types::PyBytes>() || item_1.is_instance_of::<::pyo3::types::PyByteArray>() || item_1.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&item_1, "AsgiSentWsHttpResponseStart.headers") } else { item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentWsHttpResponseStart.headers") }).collect::<PyResult<Vec<_>>>()) } })?,)) })() }).collect::<PyResult<Vec<_>>>())? });
            }
            if obj.is_instance(ASGI_SENT_WS_HTTP_RESPONSE_BODY.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentWsHttpResponseBody")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::WsHttpResponseBody { body: ::lk_agents_abi::python::attr_with(obj, "AsgiSentWsHttpResponseBody", "body", |v| { if v.is_instance_of::<::pyo3::types::PyBytes>() || v.is_instance_of::<::pyo3::types::PyByteArray>() || v.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&v, "AsgiSentWsHttpResponseBody.body") } else { v.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "AsgiSentWsHttpResponseBody.body") }).collect::<PyResult<Vec<_>>>()) } })?, more_body: ::lk_agents_abi::python::attr_with(obj, "AsgiSentWsHttpResponseBody", "more_body", |v| v.extract::<bool>())? });
            }
            if obj.is_instance(ASGI_SENT_UNKNOWN.import(py, "livekit.agents._generated.worker_endpoint", "AsgiSentUnknown")?)? {
                return Ok(::lk_agents_abi::worker::endpoint_host::AsgiSent::Unknown(::lk_agents_abi::python::attr_with(obj, "AsgiSentUnknown", "kind", |v| v.extract::<::std::string::String>())?));
            }
            Err(super::not_a(obj, "AsgiSentResponseStart | AsgiSentResponseBody | AsgiSentResponseTrailers | AsgiSentWsAccept | AsgiSentWsText | AsgiSentWsBytes | AsgiSentWsClose | AsgiSentWsHttpResponseStart | AsgiSentWsHttpResponseBody | AsgiSentUnknown"))
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/endpoint_host.rs.
    static ASGI_FAILURE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn asgi_failure_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::worker::endpoint_host::AsgiFailure> {
        let py = obj.py();
            if !obj.is_instance(ASGI_FAILURE.import(py, "livekit.agents._generated.worker_endpoint", "AsgiFailure")?)? {
                return Err(super::not_a(obj, "AsgiFailure"));
            }
            Ok(::lk_agents_abi::worker::endpoint_host::AsgiFailure {
                summary: ::lk_agents_abi::python::attr_with(obj, "AsgiFailure", "summary", |v| v.extract::<::std::string::String>())?,
                traceback: ::lk_agents_abi::python::attr_with(obj, "AsgiFailure", "traceback", |v| v.extract::<::std::string::String>())?,
            })
    }
}

pub mod worker_ipc {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/worker/ipc.rs.
    static IPC_FAILURE_IO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static IPC_FAILURE_MALFORMED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static IPC_FAILURE_TOO_LARGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static IPC_FAILURE_UNEXPECTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static IPC_FAILURE_VERSION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn ipc_failure_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::ipc::IpcFailure) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::worker::ipc::IpcFailure::Io { kind, os_error, message } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("kind", ::pyo3::IntoPyObjectExt::into_bound_py_any(kind, py)?)?;
                    kwargs.set_item("os_error", match os_error { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(IPC_FAILURE_IO.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureIo")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::ipc::IpcFailure::Malformed { message } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(IPC_FAILURE_MALFORMED.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureMalformed")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::ipc::IpcFailure::TooLarge { size } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("size", ::pyo3::IntoPyObjectExt::into_bound_py_any(size, py)?)?;
                    crate::host::construct(IPC_FAILURE_TOO_LARGE.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureTooLarge")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::ipc::IpcFailure::Unexpected { message } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                    crate::host::construct(IPC_FAILURE_UNEXPECTED.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureUnexpected")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::worker::ipc::IpcFailure::Version { received, expected } => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("received", ::pyo3::IntoPyObjectExt::into_bound_py_any(received, py)?)?;
                    kwargs.set_item("expected", ::pyo3::IntoPyObjectExt::into_bound_py_any(expected, py)?)?;
                    crate::host::construct(IPC_FAILURE_VERSION.import(py, "livekit.agents._generated.worker_ipc", "IpcFailureVersion")?.as_any(), (), Some(&kwargs))
                },
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/ipc.rs.
    static PROCESS_COMPLETION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn process_completion_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::ipc::ProcessCompletion) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::ipc::ProcessCompletion { exit_code, failure } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("exit_code", match exit_code { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("failure", match failure { None => py.None().into_bound(py), Some(x) => crate::generated::records::worker_ipc::ipc_failure_to_py(py, x)? })?;
            crate::host::construct(PROCESS_COMPLETION.import(py, "livekit.agents._generated.worker_ipc", "ProcessCompletion")?.as_any(), (), Some(&kwargs))
    }
}

pub mod worker_server {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/worker/server_host.rs.
    static ENDPOINT_ROUTE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn endpoint_route_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::worker::server_host::EndpointRoute> {
        let py = obj.py();
            if !obj.is_instance(ENDPOINT_ROUTE.import(py, "livekit.agents._generated.worker_server", "EndpointRoute")?)? {
                return Err(super::not_a(obj, "EndpointRoute"));
            }
            Ok(::lk_agents_abi::worker::server_host::EndpointRoute {
                path: ::lk_agents_abi::python::attr_with(obj, "EndpointRoute", "path", |v| v.extract::<::std::string::String>())?,
                methods: ::lk_agents_abi::python::attr_with(obj, "EndpointRoute", "methods", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>())?,
                kind: ::lk_agents_abi::python::attr_with(obj, "EndpointRoute", "kind", |v| v.extract::<::std::string::String>())?,
                public: ::lk_agents_abi::python::attr_with(obj, "EndpointRoute", "public", |v| v.extract::<bool>())?,
            })
    }
}

pub mod workflows {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/workflows.rs.
    static WORKFLOW_INSTRUCTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn workflow_instructions_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::workflows::WorkflowInstructionsArg> {
        let py = obj.py();
            if !obj.is_instance(WORKFLOW_INSTRUCTIONS.import(py, "livekit.agents._generated.workflows", "WorkflowInstructions")?)? {
                return Err(super::not_a(obj, "WorkflowInstructions"));
            }
            Ok(::lk_agents_abi::voice::workflows::WorkflowInstructionsArg {
                persona: ::lk_agents_abi::python::attr_with(obj, "WorkflowInstructions", "persona", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (v.extract::<::std::string::String>()).map(::lk_agents_abi::Maybe::Given) })?,
                extra: ::lk_agents_abi::python::attr_with(obj, "WorkflowInstructions", "extra", |v| v.extract::<::std::string::String>())?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/workflows.rs.

    pub fn instructions_arg_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::workflows::InstructionsArg> {
        if obj.is_instance(&obj.py().import("livekit.agents._generated.workflows")?.getattr("WorkflowInstructions")?)? {
            return Ok(::lk_agents_abi::voice::workflows::InstructionsArg::Workflow(crate::generated::records::workflows::workflow_instructions_from_py(obj)?));
        }
        if obj.is_instance_of::<::pyo3::types::PyString>() {
            return Ok(::lk_agents_abi::voice::workflows::InstructionsArg::Fixed(obj.extract::<::std::string::String>()?));
        }
        Err(super::not_a(obj, "InstructionsArg"))
    }
}
