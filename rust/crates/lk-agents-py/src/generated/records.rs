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
    static MESSAGE_INPUT__MESSAGE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn message_input_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::a2a::client::MessageInput> {
        if obj.is_instance_of::<::pyo3::types::PyString>() {
            return Ok(::lk_agents_abi::a2a::client::MessageInput::Text(obj.extract::<::std::string::String>()?));
        }
        if obj.is_instance(MESSAGE_INPUT__MESSAGE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.a2a_client")?.getattr("Message")?.unbind()))?.bind(obj.py()))? {
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

pub mod agent_nodes {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
    static NODE_CHUNK__FLUSH_SENTINEL: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static NODE_CHUNK__CHAT_CHUNK: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn node_chunk_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::nodes::NodeChunk> {
        if obj.is_instance_of::<::pyo3::types::PyString>() {
            return Ok(::lk_agents_abi::voice::nodes::NodeChunk::Text(obj.extract::<::std::string::String>()?));
        }
        if obj.is_instance(NODE_CHUNK__FLUSH_SENTINEL.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents.types")?.getattr("FlushSentinel")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::voice::nodes::NodeChunk::Flush);
        }
        if obj.is_instance(NODE_CHUNK__CHAT_CHUNK.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.llm")?.getattr("ChatChunk")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::voice::nodes::NodeChunk::Chunk(crate::generated::records::llm::chat_chunk_from_py(obj)?));
        }
        Err(super::not_a(obj, "NodeChunk"))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
    static NODE_FRAME: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn node_frame_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::nodes::NodeFrame) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::nodes::NodeFrame { frame, timed_transcript, started_time } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("frame", ::pyo3::IntoPyObjectExt::into_bound_py_any(frame, py)?)?;
            kwargs.set_item("timed_transcript", ::pyo3::types::PyList::new(py, timed_transcript.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("started_time", match started_time { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(NODE_FRAME.import(py, "livekit.agents._generated.agent_nodes", "NodeFrame")?.as_any(), (), Some(&kwargs))
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
            kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(crate::generated::records::chat_items::chat_item_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("function_call", match function_call { None => py.None().into_bound(py), Some(x) => crate::generated::records::chat_items::function_call_to_py(py, x)? })?;
            kwargs.set_item("approval", match approval { None => py.None().into_bound(py), Some(x) => crate::generated::records::tool_report::approval_to_py(py, x)? })?;
            kwargs.set_item("speech", match speech { None => py.None().into_bound(py), Some(x) => <dyn ::lk_agents_abi::voice::speech_handle::Speech as crate::generated::NativeToHost>::to_host(py, x)? })?;
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
                on_duplicate: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "on_duplicate", |v| crate::generated::records::tool_context::duplicate_mode_from_py(v))?,
                duplicate_scope: ::lk_agents_abi::python::attr_with(obj, "FunctionToolInfo", "duplicate_scope", |v| crate::generated::records::tool_context::duplicate_scope_from_py(v))?,
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
            kwargs.set_item("new_messages", ::pyo3::types::PyList::new(py, new_messages.into_iter().map(|x| PyResult::Ok(crate::generated::records::chat_items::chat_item_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("all_messages", ::pyo3::types::PyList::new(py, all_messages.into_iter().map(|x| PyResult::Ok(crate::generated::records::chat_items::chat_item_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("usage", crate::generated::records::agent_run::run_usage_to_py(py, usage)?)?;
            kwargs.set_item("verbatim", match verbatim { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(RUN_RESULT.import(py, "livekit.agents._generated.agent_run", "RunResult")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
    static RUN_OUTCOME_COMPLETED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static RUN_OUTCOME_INPUT_REQUIRED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static RUN_OUTCOME_NO_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
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
                ::lk_agents_abi::agent::run::RunOutcome::NoOutput => crate::host::construct(RUN_OUTCOME_NO_OUTPUT.import(py, "livekit.agents._generated.agent_run", "RunOutcomeNoOutput")?.as_any(), (), None),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/agent/run.rs.
    static OUTPUT_TEXT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();
    static OUTPUT_VALIDATED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn output_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::agent::run::Output) -> PyResult<Bound<'py, PyAny>> {
            match value {
                ::lk_agents_abi::agent::run::Output::Text(text) => {
                    let kwargs = ::pyo3::types::PyDict::new(py);
                    kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
                    crate::host::construct(OUTPUT_TEXT.import(py, "livekit.agents._generated.agent_run", "OutputText")?.as_any(), (), Some(&kwargs))
                },
                ::lk_agents_abi::agent::run::Output::Validated(_) => crate::host::construct(OUTPUT_VALIDATED.import(py, "livekit.agents._generated.agent_run", "OutputValidated")?.as_any(), (), None),
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
                message: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "message", |v| crate::generated::records::chat_items::chat_item_from_py(v))?,
                history: ::lk_agents_abi::python::attr_with(obj, "RunRequest", "history", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::chat_items::chat_item_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
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
    static AMD_CATEGORY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn amd_category_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::amd::AmdCategory> {
        let py = obj.py();
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "human" => Ok(::lk_agents_abi::voice::amd::AmdCategory::Human),
                "machine-ivr" => Ok(::lk_agents_abi::voice::amd::AmdCategory::MachineIvr),
                "machine-vm" => Ok(::lk_agents_abi::voice::amd::AmdCategory::MachineVm),
                "machine-unavailable" => Ok(::lk_agents_abi::voice::amd::AmdCategory::MachineUnavailable),
                "uncertain" => Ok(::lk_agents_abi::voice::amd::AmdCategory::Uncertain),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("AMDCategory: {other:?} is not one of 'human', 'machine-ivr', 'machine-vm', 'machine-unavailable', 'uncertain'"))),
            }
    }

    pub fn amd_category_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::amd::AmdCategory) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::voice::amd::AmdCategory::Human => "human",
                ::lk_agents_abi::voice::amd::AmdCategory::MachineIvr => "machine-ivr",
                ::lk_agents_abi::voice::amd::AmdCategory::MachineVm => "machine-vm",
                ::lk_agents_abi::voice::amd::AmdCategory::MachineUnavailable => "machine-unavailable",
                ::lk_agents_abi::voice::amd::AmdCategory::Uncertain => "uncertain",
            };
            Ok(crate::host::construct(AMD_CATEGORY.import(py, "livekit.agents._generated.amd", "AMDCategory")?.as_any(), (value,), None)?)
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/amd.rs.
    static AMD_PREDICTION_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn amd_prediction_event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::amd::AmdPrediction> {
        let py = obj.py();
            if !obj.is_instance(AMD_PREDICTION_EVENT.import(py, "livekit.agents._generated.amd", "AMDPredictionEvent")?)? {
                return Err(super::not_a(obj, "AMDPredictionEvent"));
            }
            Ok(::lk_agents_abi::voice::amd::AmdPrediction {
                speech_duration: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "speech_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "AMDPredictionEvent.speech_duration").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("AMDPredictionEvent.speech_duration: {e}")))))?,
                category: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "category", |v| crate::generated::records::amd::amd_category_from_py(v))?,
                reason: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "reason", |v| v.extract::<::lk_agents_abi::voice::amd::AmdReason>().map_err(::std::convert::Into::<PyErr>::into))?,
                transcript: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "transcript", |v| v.extract::<::std::string::String>())?,
                delay: ::lk_agents_abi::python::attr_with(obj, "AMDPredictionEvent", "delay", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "AMDPredictionEvent.delay").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("AMDPredictionEvent.delay: {e}")))))?,
            })
    }

    pub fn amd_prediction_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::amd::AmdPrediction) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::amd::AmdPrediction { speech_duration, category, reason, transcript, delay } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("speech_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(speech_duration.as_secs_f64(), py)?)?;
            kwargs.set_item("category", crate::generated::records::amd::amd_category_to_py(py, category)?)?;
            kwargs.set_item("reason", ::pyo3::IntoPyObjectExt::into_bound_py_any(reason, py)?)?;
            kwargs.set_item("transcript", ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript, py)?)?;
            kwargs.set_item("delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(delay.as_secs_f64(), py)?)?;
            crate::host::construct(AMD_PREDICTION_EVENT.import(py, "livekit.agents._generated.amd", "AMDPredictionEvent")?.as_any(), (), Some(&kwargs))
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
    static AUDIO_RECEIVER_FRAME__AUDIO_FRAME: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AUDIO_RECEIVER_FRAME__AUDIO_SEGMENT_END: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn audio_receiver_frame_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::utils::video::AudioReceiverFrame> {
        if obj.is_instance(AUDIO_RECEIVER_FRAME__AUDIO_FRAME.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.rtc")?.getattr("AudioFrame")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::utils::video::AudioReceiverFrame::Audio(obj.extract::<::lk_agents_abi::AudioFrame>().map_err(::std::convert::Into::<PyErr>::into)?));
        }
        if obj.is_instance(AUDIO_RECEIVER_FRAME__AUDIO_SEGMENT_END.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents.voice.avatar")?.getattr("AudioSegmentEnd")?.unbind()))?.bind(obj.py()))? {
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

pub mod chat_items {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.

    pub fn chat_role_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Role> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "developer" => Ok(::lk_agents_abi::Role::Developer),
                "system" => Ok(::lk_agents_abi::Role::System),
                "user" => Ok(::lk_agents_abi::Role::User),
                "assistant" => Ok(::lk_agents_abi::Role::Assistant),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("ChatRole: {other:?} is not one of 'developer', 'system', 'user', 'assistant'"))),
            }
    }

    pub fn chat_role_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::Role) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::Role::Developer => "developer",
                ::lk_agents_abi::Role::System => "system",
                ::lk_agents_abi::Role::User => "user",
                ::lk_agents_abi::Role::Assistant => "assistant",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.

    pub fn inference_detail_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::InferenceDetail> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "auto" => Ok(::lk_agents_abi::InferenceDetail::Auto),
                "high" => Ok(::lk_agents_abi::InferenceDetail::High),
                "low" => Ok(::lk_agents_abi::InferenceDetail::Low),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("InferenceDetail: {other:?} is not one of 'auto', 'high', 'low'"))),
            }
    }

    pub fn inference_detail_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::InferenceDetail) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::InferenceDetail::Auto => "auto",
                ::lk_agents_abi::InferenceDetail::High => "high",
                ::lk_agents_abi::InferenceDetail::Low => "low",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static IMAGE_CONTENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn image_content_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::ImageContent> {
        let py = obj.py();
            if !obj.is_instance(IMAGE_CONTENT.import(py, "livekit.agents._generated.chat_items", "ImageContent")?)? {
                return Err(super::not_a(obj, "ImageContent"));
            }
            Ok(::lk_agents_abi::ImageContent {
                id: ::lk_agents_abi::python::attr_with(obj, "ImageContent", "id", |v| v.extract::<::std::string::String>())?,
                image: ::lk_agents_abi::python::attr_with(obj, "ImageContent", "image", |v| crate::conversions::image_source_from_py(v))?,
                inference_width: ::lk_agents_abi::python::attr_with(obj, "ImageContent", "inference_width", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(v, "ImageContent.inference_width")).map(Some) })?,
                inference_height: ::lk_agents_abi::python::attr_with(obj, "ImageContent", "inference_height", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(v, "ImageContent.inference_height")).map(Some) })?,
                inference_detail: ::lk_agents_abi::python::attr_with(obj, "ImageContent", "inference_detail", |v| crate::generated::records::chat_items::inference_detail_from_py(v))?,
                mime_type: ::lk_agents_abi::python::attr_with(obj, "ImageContent", "mime_type", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    pub fn image_content_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::ImageContent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::ImageContent { id, image, inference_width, inference_height, inference_detail, mime_type } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("image", crate::conversions::image_source_to_py(py, image)?)?;
            kwargs.set_item("inference_width", match inference_width { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("inference_height", match inference_height { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("inference_detail", crate::generated::records::chat_items::inference_detail_to_py(py, inference_detail)?)?;
            kwargs.set_item("mime_type", match mime_type { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(IMAGE_CONTENT.import(py, "livekit.agents._generated.chat_items", "ImageContent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static AUDIO_CONTENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn audio_content_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::AudioContent> {
        let py = obj.py();
            if !obj.is_instance(AUDIO_CONTENT.import(py, "livekit.agents._generated.chat_items", "AudioContent")?)? {
                return Err(super::not_a(obj, "AudioContent"));
            }
            Ok(::lk_agents_abi::AudioContent {
                frame: ::lk_agents_abi::python::attr_with(obj, "AudioContent", "frame", |v| v.extract::<::lk_agents_abi::MediaRef>().map_err(::std::convert::Into::<PyErr>::into))?,
                transcript: ::lk_agents_abi::python::attr_with(obj, "AudioContent", "transcript", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    pub fn audio_content_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::AudioContent) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::AudioContent { frame, transcript } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("frame", ::pyo3::IntoPyObjectExt::into_bound_py_any(frame, py)?)?;
            kwargs.set_item("transcript", match transcript { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(AUDIO_CONTENT.import(py, "livekit.agents._generated.chat_items", "AudioContent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static CHAT_CONTENT__IMAGE_CONTENT: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static CHAT_CONTENT__AUDIO_CONTENT: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn chat_content_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::ChatContent> {
        if obj.is_instance(CHAT_CONTENT__IMAGE_CONTENT.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.chat_items")?.getattr("ImageContent")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::ChatContent::Image(crate::generated::records::chat_items::image_content_from_py(obj)?));
        }
        if obj.is_instance(CHAT_CONTENT__AUDIO_CONTENT.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.chat_items")?.getattr("AudioContent")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::ChatContent::Audio(crate::generated::records::chat_items::audio_content_from_py(obj)?));
        }
        if obj.is_instance_of::<::pyo3::types::PyString>() {
            return Ok(::lk_agents_abi::ChatContent::Text(obj.extract::<::std::string::String>()?));
        }
        Err(super::not_a(obj, "ChatContent"))
    }

    pub fn chat_content_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::ChatContent) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::ChatContent::Image(x) => crate::generated::records::chat_items::image_content_to_py(py, x)?,
            ::lk_agents_abi::ChatContent::Audio(x) => crate::generated::records::chat_items::audio_content_to_py(py, x)?,
            ::lk_agents_abi::ChatContent::Text(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
        })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.

    pub fn metrics_metadata_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::MetricsMetadata> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::MetricsMetadata {
                model_name: match obj.get_item("model_name")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<::std::string::String>())?), _ => None },
                model_provider: match obj.get_item("model_provider")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<::std::string::String>())?), _ => None },
            })
    }

    pub fn metrics_metadata_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::MetricsMetadata) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::MetricsMetadata { model_name, model_provider } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            if let Some(x) = model_name {
                dict.set_item("model_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = model_provider {
                dict.set_item("model_provider", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.

    pub fn metrics_report_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::MetricsReport> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::MetricsReport {
                started_speaking_at: match obj.get_item("started_speaking_at")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.started_speaking_at"))?), _ => None },
                stopped_speaking_at: match obj.get_item("stopped_speaking_at")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.stopped_speaking_at"))?), _ => None },
                transcription_delay: match obj.get_item("transcription_delay")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.transcription_delay"))?), _ => None },
                end_of_turn_delay: match obj.get_item("end_of_turn_delay")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.end_of_turn_delay"))?), _ => None },
                on_user_turn_completed_delay: match obj.get_item("on_user_turn_completed_delay")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.on_user_turn_completed_delay"))?), _ => None },
                llm_node_ttft: match obj.get_item("llm_node_ttft")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.llm_node_ttft"))?), _ => None },
                llm_node_tps: match obj.get_item("llm_node_tps")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.llm_node_tps"))?), _ => None },
                llm_node_ttfs: match obj.get_item("llm_node_ttfs")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.llm_node_ttfs"))?), _ => None },
                tts_node_ttfb: match obj.get_item("tts_node_ttfb")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.tts_node_ttfb"))?), _ => None },
                playback_latency: match obj.get_item("playback_latency")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.playback_latency"))?), _ => None },
                e2e_latency: match obj.get_item("e2e_latency")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "MetricsReport.e2e_latency"))?), _ => None },
                provider_request_ids: match obj.get_item("provider_request_ids")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>())?), _ => None },
                llm_metadata: match obj.get_item("llm_metadata")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::chat_items::metrics_metadata_from_py((&v)))?), _ => None },
                tts_metadata: match obj.get_item("tts_metadata")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::chat_items::metrics_metadata_from_py((&v)))?), _ => None },
                stt_metadata: match obj.get_item("stt_metadata")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::chat_items::metrics_metadata_from_py((&v)))?), _ => None },
            })
    }

    pub fn metrics_report_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::MetricsReport) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::MetricsReport { started_speaking_at, stopped_speaking_at, transcription_delay, end_of_turn_delay, on_user_turn_completed_delay, llm_node_ttft, llm_node_tps, llm_node_ttfs, tts_node_ttfb, playback_latency, e2e_latency, provider_request_ids, llm_metadata, tts_metadata, stt_metadata } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            if let Some(x) = started_speaking_at {
                dict.set_item("started_speaking_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = stopped_speaking_at {
                dict.set_item("stopped_speaking_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = transcription_delay {
                dict.set_item("transcription_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = end_of_turn_delay {
                dict.set_item("end_of_turn_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = on_user_turn_completed_delay {
                dict.set_item("on_user_turn_completed_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = llm_node_ttft {
                dict.set_item("llm_node_ttft", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = llm_node_tps {
                dict.set_item("llm_node_tps", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = llm_node_ttfs {
                dict.set_item("llm_node_ttfs", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = tts_node_ttfb {
                dict.set_item("tts_node_ttfb", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = playback_latency {
                dict.set_item("playback_latency", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = e2e_latency {
                dict.set_item("e2e_latency", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = provider_request_ids {
                dict.set_item("provider_request_ids", ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            }
            if let Some(x) = llm_metadata {
                dict.set_item("llm_metadata", crate::generated::records::chat_items::metrics_metadata_to_py(py, x)?)?;
            }
            if let Some(x) = tts_metadata {
                dict.set_item("tts_metadata", crate::generated::records::chat_items::metrics_metadata_to_py(py, x)?)?;
            }
            if let Some(x) = stt_metadata {
                dict.set_item("stt_metadata", crate::generated::records::chat_items::metrics_metadata_to_py(py, x)?)?;
            }
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static CHAT_MESSAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn chat_message_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::ChatMessage> {
        let py = obj.py();
            if !obj.is_instance(CHAT_MESSAGE.import(py, "livekit.agents._generated.chat_items", "ChatMessage")?)? {
                return Err(super::not_a(obj, "ChatMessage"));
            }
            Ok(::lk_agents_abi::ChatMessage {
                id: ::lk_agents_abi::python::attr_with(obj, "ChatMessage", "id", |v| v.extract::<::std::string::String>())?,
                role: ::lk_agents_abi::python::attr_with(obj, "ChatMessage", "role", |v| crate::generated::records::chat_items::chat_role_from_py(v))?,
                content: ::lk_agents_abi::python::attr_with(obj, "ChatMessage", "content", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::chat_items::chat_content_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                interrupted: ::lk_agents_abi::python::attr_with(obj, "ChatMessage", "interrupted", |v| v.extract::<bool>())?,
                transcript_confidence: ::lk_agents_abi::python::attr_with(obj, "ChatMessage", "transcript_confidence", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "ChatMessage.transcript_confidence")).map(Some) })?,
                extra: ::lk_agents_abi::python::attr_with(obj, "ChatMessage", "extra", |v| match ::lk_agents_abi::python::value_from_py(v, "ChatMessage.extra")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("ChatMessage.extra: expected a dict")) })?,
                metrics: ::lk_agents_abi::python::attr_with(obj, "ChatMessage", "metrics", |v| crate::generated::records::chat_items::metrics_report_from_py(v))?,
                created_at: ::lk_agents_abi::python::attr_with(obj, "ChatMessage", "created_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "ChatMessage.created_at").and_then(|s| ::lk_agents_abi::python::wall_from_py(s, "ChatMessage.created_at")))?,
            })
    }

    pub fn chat_message_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::ChatMessage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::ChatMessage { id, role, content, interrupted, transcript_confidence, extra, metrics, created_at } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("role", crate::generated::records::chat_items::chat_role_to_py(py, role)?)?;
            kwargs.set_item("content", ::pyo3::types::PyList::new(py, content.into_iter().map(|x| PyResult::Ok(crate::generated::records::chat_items::chat_content_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("interrupted", ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted, py)?)?;
            kwargs.set_item("transcript_confidence", match transcript_confidence { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("extra", ::lk_agents_abi::python::extra_to_py(py, &extra)?.into_any())?;
            kwargs.set_item("metrics", crate::generated::records::chat_items::metrics_report_to_py(py, metrics)?)?;
            kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::unix_secs(created_at), py)?)?;
            crate::host::construct(CHAT_MESSAGE.import(py, "livekit.agents._generated.chat_items", "ChatMessage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static FUNCTION_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn function_call_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::FunctionCall> {
        let py = obj.py();
            if !obj.is_instance(FUNCTION_CALL.import(py, "livekit.agents._generated.chat_items", "FunctionCall")?)? {
                return Err(super::not_a(obj, "FunctionCall"));
            }
            Ok(::lk_agents_abi::FunctionCall {
                id: ::lk_agents_abi::python::attr_with(obj, "FunctionCall", "id", |v| v.extract::<::std::string::String>())?,
                call_id: ::lk_agents_abi::python::attr_with(obj, "FunctionCall", "call_id", |v| v.extract::<::std::string::String>())?,
                arguments: ::lk_agents_abi::python::attr_with(obj, "FunctionCall", "arguments", |v| v.extract::<::std::string::String>())?,
                name: ::lk_agents_abi::python::attr_with(obj, "FunctionCall", "name", |v| v.extract::<::std::string::String>())?,
                created_at: ::lk_agents_abi::python::attr_with(obj, "FunctionCall", "created_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "FunctionCall.created_at").and_then(|s| ::lk_agents_abi::python::wall_from_py(s, "FunctionCall.created_at")))?,
                extra: ::lk_agents_abi::python::attr_with(obj, "FunctionCall", "extra", |v| match ::lk_agents_abi::python::value_from_py(v, "FunctionCall.extra")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FunctionCall.extra: expected a dict")) })?,
                group_id: ::lk_agents_abi::python::attr_with(obj, "FunctionCall", "group_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    pub fn function_call_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::FunctionCall) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::FunctionCall { id, call_id, arguments, name, created_at, extra, group_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
            kwargs.set_item("arguments", ::pyo3::IntoPyObjectExt::into_bound_py_any(arguments, py)?)?;
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::unix_secs(created_at), py)?)?;
            kwargs.set_item("extra", ::lk_agents_abi::python::extra_to_py(py, &extra)?.into_any())?;
            kwargs.set_item("group_id", match group_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(FUNCTION_CALL.import(py, "livekit.agents._generated.chat_items", "FunctionCall")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static FUNCTION_CALL_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn function_call_output_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::FunctionCallOutput> {
        let py = obj.py();
            if !obj.is_instance(FUNCTION_CALL_OUTPUT.import(py, "livekit.agents._generated.chat_items", "FunctionCallOutput")?)? {
                return Err(super::not_a(obj, "FunctionCallOutput"));
            }
            Ok(::lk_agents_abi::FunctionCallOutput {
                id: ::lk_agents_abi::python::attr_with(obj, "FunctionCallOutput", "id", |v| v.extract::<::std::string::String>())?,
                name: ::lk_agents_abi::python::attr_with(obj, "FunctionCallOutput", "name", |v| v.extract::<::std::string::String>())?,
                call_id: ::lk_agents_abi::python::attr_with(obj, "FunctionCallOutput", "call_id", |v| v.extract::<::std::string::String>())?,
                output: ::lk_agents_abi::python::attr_with(obj, "FunctionCallOutput", "output", |v| v.extract::<::std::string::String>())?,
                is_error: ::lk_agents_abi::python::attr_with(obj, "FunctionCallOutput", "is_error", |v| v.extract::<bool>())?,
                created_at: ::lk_agents_abi::python::attr_with(obj, "FunctionCallOutput", "created_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "FunctionCallOutput.created_at").and_then(|s| ::lk_agents_abi::python::wall_from_py(s, "FunctionCallOutput.created_at")))?,
                reply_required: ::lk_agents_abi::python::attr_with(obj, "FunctionCallOutput", "reply_required", |v| v.extract::<bool>())?,
            })
    }

    pub fn function_call_output_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::FunctionCallOutput) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::FunctionCallOutput { id, name, call_id, output, is_error, created_at, reply_required } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
            kwargs.set_item("output", ::pyo3::IntoPyObjectExt::into_bound_py_any(output, py)?)?;
            kwargs.set_item("is_error", ::pyo3::IntoPyObjectExt::into_bound_py_any(is_error, py)?)?;
            kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::unix_secs(created_at), py)?)?;
            kwargs.set_item("reply_required", ::pyo3::IntoPyObjectExt::into_bound_py_any(reply_required, py)?)?;
            crate::host::construct(FUNCTION_CALL_OUTPUT.import(py, "livekit.agents._generated.chat_items", "FunctionCallOutput")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static AGENT_HANDOFF: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn agent_handoff_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::AgentHandoffItem> {
        let py = obj.py();
            if !obj.is_instance(AGENT_HANDOFF.import(py, "livekit.agents._generated.chat_items", "AgentHandoff")?)? {
                return Err(super::not_a(obj, "AgentHandoff"));
            }
            Ok(::lk_agents_abi::AgentHandoffItem {
                id: ::lk_agents_abi::python::attr_with(obj, "AgentHandoff", "id", |v| v.extract::<::std::string::String>())?,
                old_agent_id: ::lk_agents_abi::python::attr_with(obj, "AgentHandoff", "old_agent_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                new_agent_id: ::lk_agents_abi::python::attr_with(obj, "AgentHandoff", "new_agent_id", |v| v.extract::<::std::string::String>())?,
                created_at: ::lk_agents_abi::python::attr_with(obj, "AgentHandoff", "created_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "AgentHandoff.created_at").and_then(|s| ::lk_agents_abi::python::wall_from_py(s, "AgentHandoff.created_at")))?,
            })
    }

    pub fn agent_handoff_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::AgentHandoffItem) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::AgentHandoffItem { id, old_agent_id, new_agent_id, created_at } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("old_agent_id", match old_agent_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("new_agent_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(new_agent_id, py)?)?;
            kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::unix_secs(created_at), py)?)?;
            crate::host::construct(AGENT_HANDOFF.import(py, "livekit.agents._generated.chat_items", "AgentHandoff")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static AGENT_CONFIG_UPDATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn agent_config_update_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::AgentConfigUpdate> {
        let py = obj.py();
            if !obj.is_instance(AGENT_CONFIG_UPDATE.import(py, "livekit.agents._generated.chat_items", "AgentConfigUpdate")?)? {
                return Err(super::not_a(obj, "AgentConfigUpdate"));
            }
            Ok(::lk_agents_abi::AgentConfigUpdate {
                id: ::lk_agents_abi::python::attr_with(obj, "AgentConfigUpdate", "id", |v| v.extract::<::std::string::String>())?,
                instructions: ::lk_agents_abi::python::attr_with(obj, "AgentConfigUpdate", "instructions", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                tools_added: ::lk_agents_abi::python::attr_with(obj, "AgentConfigUpdate", "tools_added", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                tools_removed: ::lk_agents_abi::python::attr_with(obj, "AgentConfigUpdate", "tools_removed", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                created_at: ::lk_agents_abi::python::attr_with(obj, "AgentConfigUpdate", "created_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "AgentConfigUpdate.created_at").and_then(|s| ::lk_agents_abi::python::wall_from_py(s, "AgentConfigUpdate.created_at")))?,
                tools: ::lk_agents_abi::python::attr_with(obj, "AgentConfigUpdate", "_tools", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            })
    }

    pub fn agent_config_update_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::AgentConfigUpdate) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::AgentConfigUpdate { id, instructions, tools_added, tools_removed, created_at, tools } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("instructions", match instructions { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("tools_added", match tools_added { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
            kwargs.set_item("tools_removed", match tools_removed { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
            kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::unix_secs(created_at), py)?)?;
            let obj = crate::host::construct(AGENT_CONFIG_UPDATE.import(py, "livekit.agents._generated.chat_items", "AgentConfigUpdate")?.as_any(), (), Some(&kwargs))?;
            if let Some(x) = tools {
                obj.setattr("_tools", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            Ok(obj)
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/chat_context.rs.
    static CHAT_ITEM__CHAT_MESSAGE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static CHAT_ITEM__FUNCTION_CALL: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static CHAT_ITEM__FUNCTION_CALL_OUTPUT: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static CHAT_ITEM__AGENT_HANDOFF: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static CHAT_ITEM__AGENT_CONFIG_UPDATE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn chat_item_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::ChatItem> {
        if obj.is_instance(CHAT_ITEM__CHAT_MESSAGE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.chat_items")?.getattr("ChatMessage")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::ChatItem::Message(crate::generated::records::chat_items::chat_message_from_py(obj)?));
        }
        if obj.is_instance(CHAT_ITEM__FUNCTION_CALL.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.chat_items")?.getattr("FunctionCall")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::ChatItem::FunctionCall(crate::generated::records::chat_items::function_call_from_py(obj)?));
        }
        if obj.is_instance(CHAT_ITEM__FUNCTION_CALL_OUTPUT.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.chat_items")?.getattr("FunctionCallOutput")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::ChatItem::FunctionCallOutput(crate::generated::records::chat_items::function_call_output_from_py(obj)?));
        }
        if obj.is_instance(CHAT_ITEM__AGENT_HANDOFF.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.chat_items")?.getattr("AgentHandoff")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::ChatItem::AgentHandoff(crate::generated::records::chat_items::agent_handoff_from_py(obj)?));
        }
        if obj.is_instance(CHAT_ITEM__AGENT_CONFIG_UPDATE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.chat_items")?.getattr("AgentConfigUpdate")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::ChatItem::AgentConfigUpdate(crate::generated::records::chat_items::agent_config_update_from_py(obj)?));
        }
        Err(super::not_a(obj, "ChatItem"))
    }

    pub fn chat_item_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::ChatItem) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::ChatItem::Message(x) => crate::generated::records::chat_items::chat_message_to_py(py, x)?,
            ::lk_agents_abi::ChatItem::FunctionCall(x) => crate::generated::records::chat_items::function_call_to_py(py, x)?,
            ::lk_agents_abi::ChatItem::FunctionCallOutput(x) => crate::generated::records::chat_items::function_call_output_to_py(py, x)?,
            ::lk_agents_abi::ChatItem::AgentHandoff(x) => crate::generated::records::chat_items::agent_handoff_to_py(py, x)?,
            ::lk_agents_abi::ChatItem::AgentConfigUpdate(x) => crate::generated::records::chat_items::agent_config_update_to_py(py, x)?,
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
    static DUPLEX_EVENT__DUPLEX_TRANSCRIPT: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static DUPLEX_EVENT__EVENT: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn duplex_event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::duplex::DuplexEvent> {
        if obj.is_instance(DUPLEX_EVENT__DUPLEX_TRANSCRIPT.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.duplex")?.getattr("DuplexTranscript")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::duplex::DuplexEvent::Transcript(crate::generated::records::duplex::duplex_transcript_from_py(obj)?));
        }
        if obj.is_instance(DUPLEX_EVENT__EVENT.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("Event")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::duplex::DuplexEvent::Session(crate::generated::records::realtime_wire::event_from_py(obj)?));
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
            kwargs.set_item("usage", match usage { None => py.None().into_bound(py), Some(x) => crate::generated::records::llm::completion_usage_to_py(py, x)? })?;
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

    pub fn user_state_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::events::UserState> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "speaking" => Ok(::lk_agents_abi::voice::events::UserState::Speaking),
                "listening" => Ok(::lk_agents_abi::voice::events::UserState::Listening),
                "away" => Ok(::lk_agents_abi::voice::events::UserState::Away),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("UserState: {other:?} is not one of 'speaking', 'listening', 'away'"))),
            }
    }

    pub fn user_state_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::UserState) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::voice::events::UserState::Speaking => "speaking",
                ::lk_agents_abi::voice::events::UserState::Listening => "listening",
                ::lk_agents_abi::voice::events::UserState::Away => "away",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.

    pub fn agent_state_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::events::AgentState> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "initializing" => Ok(::lk_agents_abi::voice::events::AgentState::Initializing),
                "idle" => Ok(::lk_agents_abi::voice::events::AgentState::Idle),
                "listening" => Ok(::lk_agents_abi::voice::events::AgentState::Listening),
                "thinking" => Ok(::lk_agents_abi::voice::events::AgentState::Thinking),
                "speaking" => Ok(::lk_agents_abi::voice::events::AgentState::Speaking),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("AgentState: {other:?} is not one of 'initializing', 'idle', 'listening', 'thinking', 'speaking'"))),
            }
    }

    pub fn agent_state_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::AgentState) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::voice::events::AgentState::Initializing => "initializing",
                ::lk_agents_abi::voice::events::AgentState::Idle => "idle",
                ::lk_agents_abi::voice::events::AgentState::Listening => "listening",
                ::lk_agents_abi::voice::events::AgentState::Thinking => "thinking",
                ::lk_agents_abi::voice::events::AgentState::Speaking => "speaking",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static USER_STATE_CHANGED_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn user_state_changed_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::UserStateChanged) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::UserStateChanged { old_state, new_state, created_at } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("old_state", crate::generated::records::events::user_state_to_py(py, old_state)?)?;
            kwargs.set_item("new_state", crate::generated::records::events::user_state_to_py(py, new_state)?)?;
            kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(created_at, py)?)?;
            crate::host::construct(USER_STATE_CHANGED_EVENT.import(py, "livekit.agents._generated.events", "UserStateChangedEvent")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static AGENT_STATE_CHANGED_EVENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn agent_state_changed_event_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::AgentStateChanged) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::AgentStateChanged { old_state, new_state, created_at: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("old_state", crate::generated::records::events::agent_state_to_py(py, old_state)?)?;
            kwargs.set_item("new_state", crate::generated::records::events::agent_state_to_py(py, new_state)?)?;
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

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static CLOSE_REASON: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn close_reason_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::events::CloseReason> {
        let py = obj.py();
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "error" => Ok(::lk_agents_abi::voice::events::CloseReason::Error),
                "job_shutdown" => Ok(::lk_agents_abi::voice::events::CloseReason::JobShutdown),
                "participant_disconnected" => Ok(::lk_agents_abi::voice::events::CloseReason::ParticipantDisconnected),
                "user_initiated" => Ok(::lk_agents_abi::voice::events::CloseReason::UserInitiated),
                "task_completed" => Ok(::lk_agents_abi::voice::events::CloseReason::TaskCompleted),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("CloseReason: {other:?} is not one of 'error', 'job_shutdown', 'participant_disconnected', 'user_initiated', 'task_completed'"))),
            }
    }

    pub fn close_reason_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::CloseReason) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::voice::events::CloseReason::Error => "error",
                ::lk_agents_abi::voice::events::CloseReason::JobShutdown => "job_shutdown",
                ::lk_agents_abi::voice::events::CloseReason::ParticipantDisconnected => "participant_disconnected",
                ::lk_agents_abi::voice::events::CloseReason::UserInitiated => "user_initiated",
                ::lk_agents_abi::voice::events::CloseReason::TaskCompleted => "task_completed",
            };
            Ok(crate::host::construct(CLOSE_REASON.import(py, "livekit.agents._generated.events", "CloseReason")?.as_any(), (value,), None)?)
    }
}

pub mod inference_stt {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/inference/fallback.rs.

    pub fn fallback_model_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::inference::fallback::SttFallbackModel> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::inference::fallback::SttFallbackModel {
                model: { let v = obj.get_item("model")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("FallbackModel.model"))?; ((&v).extract::<::std::string::String>())? },
                extra_kwargs: match obj.get_item("extra_kwargs")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((match ::lk_agents_abi::python::value_from_py((&v), "FallbackModel.extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FallbackModel.extra_kwargs: expected a dict")) })?), _ => None },
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
                voice: match obj.get_item("voice")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => ((&v).extract::<::std::string::String>())?, _ => "".to_owned() },
                extra_kwargs: match obj.get_item("extra_kwargs")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((match ::lk_agents_abi::python::value_from_py((&v), "FallbackModel.extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FallbackModel.extra_kwargs: expected a dict")) })?), _ => None },
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

pub mod llm {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/events.rs.
    static COMPLETION_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn completion_usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::CompletionUsage> {
        let py = obj.py();
            if !obj.is_instance(COMPLETION_USAGE.import(py, "livekit.agents._generated.llm", "CompletionUsage")?)? {
                return Err(super::not_a(obj, "CompletionUsage"));
            }
            Ok(::lk_agents_abi::CompletionUsage {
                completion_tokens: ::lk_agents_abi::python::attr_with(obj, "CompletionUsage", "completion_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "CompletionUsage.completion_tokens"))?,
                prompt_tokens: ::lk_agents_abi::python::attr_with(obj, "CompletionUsage", "prompt_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "CompletionUsage.prompt_tokens"))?,
                prompt_cached_tokens: ::lk_agents_abi::python::attr_with(obj, "CompletionUsage", "prompt_cached_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "CompletionUsage.prompt_cached_tokens"))?,
                cache_creation_tokens: ::lk_agents_abi::python::attr_with(obj, "CompletionUsage", "cache_creation_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "CompletionUsage.cache_creation_tokens"))?,
                cache_read_tokens: ::lk_agents_abi::python::attr_with(obj, "CompletionUsage", "cache_read_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "CompletionUsage.cache_read_tokens"))?,
                reasoning_tokens: ::lk_agents_abi::python::attr_with(obj, "CompletionUsage", "reasoning_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "CompletionUsage.reasoning_tokens"))?,
                total_tokens: ::lk_agents_abi::python::attr_with(obj, "CompletionUsage", "total_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "CompletionUsage.total_tokens"))?,
                service_tier: ::lk_agents_abi::python::attr_with(obj, "CompletionUsage", "service_tier", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    pub fn completion_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::CompletionUsage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::CompletionUsage { completion_tokens, prompt_tokens, prompt_cached_tokens, cache_creation_tokens, cache_read_tokens, reasoning_tokens, total_tokens, service_tier } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("completion_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(completion_tokens, py)?)?;
            kwargs.set_item("prompt_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(prompt_tokens, py)?)?;
            kwargs.set_item("prompt_cached_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(prompt_cached_tokens, py)?)?;
            kwargs.set_item("cache_creation_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(cache_creation_tokens, py)?)?;
            kwargs.set_item("cache_read_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(cache_read_tokens, py)?)?;
            kwargs.set_item("reasoning_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(reasoning_tokens, py)?)?;
            kwargs.set_item("total_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_tokens, py)?)?;
            kwargs.set_item("service_tier", match service_tier { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(COMPLETION_USAGE.import(py, "livekit.agents._generated.llm", "CompletionUsage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/events.rs.
    static FUNCTION_TOOL_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn function_tool_call_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::FunctionToolCall> {
        let py = obj.py();
            if !obj.is_instance(FUNCTION_TOOL_CALL.import(py, "livekit.agents._generated.llm", "FunctionToolCall")?)? {
                return Err(super::not_a(obj, "FunctionToolCall"));
            }
            Ok(::lk_agents_abi::FunctionToolCall {
                name: ::lk_agents_abi::python::attr_with(obj, "FunctionToolCall", "name", |v| v.extract::<::std::string::String>())?,
                arguments: ::lk_agents_abi::python::attr_with(obj, "FunctionToolCall", "arguments", |v| v.extract::<::std::string::String>())?,
                call_id: ::lk_agents_abi::python::attr_with(obj, "FunctionToolCall", "call_id", |v| v.extract::<::std::string::String>())?,
                extra: ::lk_agents_abi::python::attr_with(obj, "FunctionToolCall", "extra", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "FunctionToolCall.extra")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("FunctionToolCall.extra: expected a dict")) }).map(Some) })?,
            })
    }

    pub fn function_tool_call_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::FunctionToolCall) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::FunctionToolCall { name, arguments, call_id, extra } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("arguments", ::pyo3::IntoPyObjectExt::into_bound_py_any(arguments, py)?)?;
            kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
            kwargs.set_item("extra", match extra { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
            crate::host::construct(FUNCTION_TOOL_CALL.import(py, "livekit.agents._generated.llm", "FunctionToolCall")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/events.rs.
    static CHOICE_DELTA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn choice_delta_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::ChoiceDelta> {
        let py = obj.py();
            if !obj.is_instance(CHOICE_DELTA.import(py, "livekit.agents._generated.llm", "ChoiceDelta")?)? {
                return Err(super::not_a(obj, "ChoiceDelta"));
            }
            Ok(::lk_agents_abi::ChoiceDelta {
                role: ::lk_agents_abi::python::attr_with(obj, "ChoiceDelta", "role", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::chat_items::chat_role_from_py(v)).map(Some) })?,
                content: ::lk_agents_abi::python::attr_with(obj, "ChoiceDelta", "content", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                tool_calls: ::lk_agents_abi::python::attr_with(obj, "ChoiceDelta", "tool_calls", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::llm::function_tool_call_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                extra: ::lk_agents_abi::python::attr_with(obj, "ChoiceDelta", "extra", |v| if v.is_none() { Ok(None) } else { (match ::lk_agents_abi::python::value_from_py(v, "ChoiceDelta.extra")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err("ChoiceDelta.extra: expected a dict")) }).map(Some) })?,
            })
    }

    pub fn choice_delta_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::ChoiceDelta) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::ChoiceDelta { role, content, tool_calls, extra } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("role", match role { None => py.None().into_bound(py), Some(x) => crate::generated::records::chat_items::chat_role_to_py(py, x)? })?;
            kwargs.set_item("content", match content { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("tool_calls", ::pyo3::types::PyList::new(py, tool_calls.into_iter().map(|x| PyResult::Ok(crate::generated::records::llm::function_tool_call_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("extra", match extra { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::extra_to_py(py, &x)?.into_any() })?;
            crate::host::construct(CHOICE_DELTA.import(py, "livekit.agents._generated.llm", "ChoiceDelta")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/events.rs.
    static CHAT_CHUNK: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn chat_chunk_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::LlmChunk> {
        let py = obj.py();
            if !obj.is_instance(CHAT_CHUNK.import(py, "livekit.agents._generated.llm", "ChatChunk")?)? {
                return Err(super::not_a(obj, "ChatChunk"));
            }
            Ok(::lk_agents_abi::LlmChunk {
                id: ::lk_agents_abi::python::attr_with(obj, "ChatChunk", "id", |v| v.extract::<::std::string::String>())?,
                delta: ::lk_agents_abi::python::attr_with(obj, "ChatChunk", "delta", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::llm::choice_delta_from_py(v)).map(Some) })?,
                usage: ::lk_agents_abi::python::attr_with(obj, "ChatChunk", "usage", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::llm::completion_usage_from_py(v)).map(Some) })?,
                flush: ::std::default::Default::default(),
                node_text: ::std::default::Default::default(),
            })
    }

    pub fn chat_chunk_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::LlmChunk) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::LlmChunk { id, delta, usage, flush: _, node_text: _ } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("delta", match delta { None => py.None().into_bound(py), Some(x) => crate::generated::records::llm::choice_delta_to_py(py, x)? })?;
            kwargs.set_item("usage", match usage { None => py.None().into_bound(py), Some(x) => crate::generated::records::llm::completion_usage_to_py(py, x)? })?;
            crate::host::construct(CHAT_CHUNK.import(py, "livekit.agents._generated.llm", "ChatChunk")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/provider_format/image.rs.
    static SERIALIZED_IMAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn serialized_image_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::provider_format::SerializedImage> {
        let py = obj.py();
            if !obj.is_instance(SERIALIZED_IMAGE.import(py, "livekit.agents._generated.llm", "SerializedImage")?)? {
                return Err(super::not_a(obj, "SerializedImage"));
            }
            Ok(::lk_agents_abi::llm::provider_format::SerializedImage {
                inference_detail: ::lk_agents_abi::python::attr_with(obj, "SerializedImage", "inference_detail", |v| crate::generated::records::chat_items::inference_detail_from_py(v))?,
                mime_type: ::lk_agents_abi::python::attr_with(obj, "SerializedImage", "mime_type", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                data_bytes: ::lk_agents_abi::python::attr_with(obj, "SerializedImage", "data_bytes", |v| if v.is_none() { Ok(None) } else { ({ if v.is_instance_of::<::pyo3::types::PyBytes>() || v.is_instance_of::<::pyo3::types::PyByteArray>() || v.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&v, "SerializedImage.data_bytes") } else { v.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "SerializedImage.data_bytes") }).collect::<PyResult<Vec<_>>>()) } }).map(Some) })?,
                external_url: ::lk_agents_abi::python::attr_with(obj, "SerializedImage", "external_url", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
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

pub mod mcp {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.

    pub fn resolved_tool_options_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::mcp::McpToolOptions) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::mcp::McpToolOptions { flags, on_duplicate, duplicate_scope, report_progress, retries } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("flags", ::pyo3::IntoPyObjectExt::into_bound_py_any(flags, py)?)?;
            dict.set_item("on_duplicate", crate::generated::records::tool_context::duplicate_mode_to_py(py, on_duplicate)?)?;
            dict.set_item("duplicate_scope", crate::generated::records::tool_context::duplicate_scope_to_py(py, duplicate_scope)?)?;
            dict.set_item("report_progress", ::pyo3::IntoPyObjectExt::into_bound_py_any(report_progress, py)?)?;
            dict.set_item("retries", match retries { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.

    pub fn mcp_tool_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::mcp::McpToolOptionsPatch> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::llm::mcp::McpToolOptionsPatch {
                flags: match obj.get_item("flags")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<::lk_agents_abi::llm::mcp::ToolFlag>().map_err(::std::convert::Into::<PyErr>::into))?), _ => None },
                on_duplicate: match obj.get_item("on_duplicate")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::tool_context::duplicate_mode_from_py((&v)))?), _ => None },
                duplicate_scope: match obj.get_item("duplicate_scope")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::tool_context::duplicate_scope_from_py((&v)))?), _ => None },
                report_progress: match obj.get_item("report_progress")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<bool>())?), _ => None },
                retries: match obj.get_item("retries")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<u32>((&v), "MCPToolOptions.retries"))?), _ => None },
            })
    }
}

pub mod mcp_session {
    use pyo3::prelude::*;

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

pub mod metrics {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static METADATA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn metadata_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::Metadata> {
        let py = obj.py();
            if !obj.is_instance(METADATA.import(py, "livekit.agents._generated.metrics", "Metadata")?)? {
                return Err(super::not_a(obj, "Metadata"));
            }
            Ok(::lk_agents_abi::metrics::Metadata {
                model_name: ::lk_agents_abi::python::attr_with(obj, "Metadata", "model_name", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                model_provider: ::lk_agents_abi::python::attr_with(obj, "Metadata", "model_provider", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    pub fn metadata_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::Metadata) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::Metadata { model_name, model_provider } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("model_name", match model_name { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("model_provider", match model_provider { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(METADATA.import(py, "livekit.agents._generated.metrics", "Metadata")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static INTERRUPTION_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn interruption_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::InterruptionMetrics> {
        let py = obj.py();
            if !obj.is_instance(INTERRUPTION_METRICS.import(py, "livekit.agents._generated.metrics", "InterruptionMetrics")?)? {
                return Err(super::not_a(obj, "InterruptionMetrics"));
            }
            Ok(::lk_agents_abi::InterruptionMetrics {
                timestamp: ::lk_agents_abi::python::attr_with(obj, "InterruptionMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "InterruptionMetrics.timestamp"))?,
                total_duration: ::lk_agents_abi::python::attr_with(obj, "InterruptionMetrics", "total_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "InterruptionMetrics.total_duration"))?,
                prediction_duration: ::lk_agents_abi::python::attr_with(obj, "InterruptionMetrics", "prediction_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "InterruptionMetrics.prediction_duration"))?,
                detection_delay: ::lk_agents_abi::python::attr_with(obj, "InterruptionMetrics", "detection_delay", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "InterruptionMetrics.detection_delay"))?,
                num_interruptions: ::lk_agents_abi::python::attr_with(obj, "InterruptionMetrics", "num_interruptions", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "InterruptionMetrics.num_interruptions"))?,
                num_backchannels: ::lk_agents_abi::python::attr_with(obj, "InterruptionMetrics", "num_backchannels", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "InterruptionMetrics.num_backchannels"))?,
                num_requests: ::lk_agents_abi::python::attr_with(obj, "InterruptionMetrics", "num_requests", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "InterruptionMetrics.num_requests"))?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "InterruptionMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn interruption_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::InterruptionMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::InterruptionMetrics { timestamp, total_duration, prediction_duration, detection_delay, num_interruptions, num_backchannels, num_requests, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("total_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_duration, py)?)?;
            kwargs.set_item("prediction_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(prediction_duration, py)?)?;
            kwargs.set_item("detection_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(detection_delay, py)?)?;
            kwargs.set_item("num_interruptions", ::pyo3::IntoPyObjectExt::into_bound_py_any(num_interruptions, py)?)?;
            kwargs.set_item("num_backchannels", ::pyo3::IntoPyObjectExt::into_bound_py_any(num_backchannels, py)?)?;
            kwargs.set_item("num_requests", ::pyo3::IntoPyObjectExt::into_bound_py_any(num_requests, py)?)?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(INTERRUPTION_METRICS.import(py, "livekit.agents._generated.metrics", "InterruptionMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static EOT_INFERENCE_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn eot_inference_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::EotInferenceMetrics> {
        let py = obj.py();
            if !obj.is_instance(EOT_INFERENCE_METRICS.import(py, "livekit.agents._generated.metrics", "EOTInferenceMetrics")?)? {
                return Err(super::not_a(obj, "EOTInferenceMetrics"));
            }
            Ok(::lk_agents_abi::EotInferenceMetrics {
                timestamp: ::lk_agents_abi::python::attr_with(obj, "EOTInferenceMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EOTInferenceMetrics.timestamp"))?,
                total_duration: ::lk_agents_abi::python::attr_with(obj, "EOTInferenceMetrics", "total_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EOTInferenceMetrics.total_duration"))?,
                detection_delay: ::lk_agents_abi::python::attr_with(obj, "EOTInferenceMetrics", "detection_delay", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EOTInferenceMetrics.detection_delay"))?,
                prediction_duration: ::lk_agents_abi::python::attr_with(obj, "EOTInferenceMetrics", "prediction_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EOTInferenceMetrics.prediction_duration"))?,
                num_requests: ::lk_agents_abi::python::attr_with(obj, "EOTInferenceMetrics", "num_requests", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "EOTInferenceMetrics.num_requests"))?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "EOTInferenceMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn eot_inference_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::EotInferenceMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::EotInferenceMetrics { timestamp, total_duration, detection_delay, prediction_duration, num_requests, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("total_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_duration, py)?)?;
            kwargs.set_item("detection_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(detection_delay, py)?)?;
            kwargs.set_item("prediction_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(prediction_duration, py)?)?;
            kwargs.set_item("num_requests", ::pyo3::IntoPyObjectExt::into_bound_py_any(num_requests, py)?)?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(EOT_INFERENCE_METRICS.import(py, "livekit.agents._generated.metrics", "EOTInferenceMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static LLM_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn llm_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::LlmMetrics> {
        let py = obj.py();
            if !obj.is_instance(LLM_METRICS.import(py, "livekit.agents._generated.metrics", "LLMMetrics")?)? {
                return Err(super::not_a(obj, "LLMMetrics"));
            }
            Ok(::lk_agents_abi::metrics::LlmMetrics {
                label: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "label", |v| v.extract::<::std::string::String>())?,
                request_id: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "request_id", |v| v.extract::<::std::string::String>())?,
                timestamp: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "LLMMetrics.timestamp"))?,
                duration: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "LLMMetrics.duration"))?,
                ttft: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "ttft", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "LLMMetrics.ttft"))?,
                cancelled: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "cancelled", |v| v.extract::<bool>())?,
                completion_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "completion_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "LLMMetrics.completion_tokens"))?,
                prompt_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "prompt_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "LLMMetrics.prompt_tokens"))?,
                prompt_cached_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "prompt_cached_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "LLMMetrics.prompt_cached_tokens"))?,
                cache_creation_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "cache_creation_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "LLMMetrics.cache_creation_tokens"))?,
                reasoning_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "reasoning_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "LLMMetrics.reasoning_tokens"))?,
                total_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "total_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "LLMMetrics.total_tokens"))?,
                tokens_per_second: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "tokens_per_second", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "LLMMetrics.tokens_per_second"))?,
                speech_id: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "speech_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "LLMMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn llm_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::LlmMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::LlmMetrics { label, request_id, timestamp, duration, ttft, cancelled, completion_tokens, prompt_tokens, prompt_cached_tokens, cache_creation_tokens, reasoning_tokens, total_tokens, tokens_per_second, speech_id, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("label", ::pyo3::IntoPyObjectExt::into_bound_py_any(label, py)?)?;
            kwargs.set_item("request_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id, py)?)?;
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(duration, py)?)?;
            kwargs.set_item("ttft", ::pyo3::IntoPyObjectExt::into_bound_py_any(ttft, py)?)?;
            kwargs.set_item("cancelled", ::pyo3::IntoPyObjectExt::into_bound_py_any(cancelled, py)?)?;
            kwargs.set_item("completion_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(completion_tokens, py)?)?;
            kwargs.set_item("prompt_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(prompt_tokens, py)?)?;
            kwargs.set_item("prompt_cached_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(prompt_cached_tokens, py)?)?;
            kwargs.set_item("cache_creation_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(cache_creation_tokens, py)?)?;
            kwargs.set_item("reasoning_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(reasoning_tokens, py)?)?;
            kwargs.set_item("total_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_tokens, py)?)?;
            kwargs.set_item("tokens_per_second", ::pyo3::IntoPyObjectExt::into_bound_py_any(tokens_per_second, py)?)?;
            kwargs.set_item("speech_id", match speech_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(LLM_METRICS.import(py, "livekit.agents._generated.metrics", "LLMMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static STT_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn stt_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::SttMetrics> {
        let py = obj.py();
            if !obj.is_instance(STT_METRICS.import(py, "livekit.agents._generated.metrics", "STTMetrics")?)? {
                return Err(super::not_a(obj, "STTMetrics"));
            }
            Ok(::lk_agents_abi::metrics::SttMetrics {
                label: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "label", |v| v.extract::<::std::string::String>())?,
                request_id: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "request_id", |v| v.extract::<::std::string::String>())?,
                timestamp: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "STTMetrics.timestamp"))?,
                duration: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "STTMetrics.duration"))?,
                audio_duration: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "audio_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "STTMetrics.audio_duration"))?,
                streamed: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "streamed", |v| v.extract::<bool>())?,
                input_tokens: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "input_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "STTMetrics.input_tokens"))?,
                output_tokens: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "output_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "STTMetrics.output_tokens"))?,
                total_tokens: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "total_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "STTMetrics.total_tokens"))?,
                input_audio_tokens: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "input_audio_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "STTMetrics.input_audio_tokens"))?,
                acquire_time: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "acquire_time", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "STTMetrics.acquire_time"))?,
                connection_reused: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "connection_reused", |v| v.extract::<bool>())?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "STTMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn stt_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::SttMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::SttMetrics { label, request_id, timestamp, duration, audio_duration, streamed, input_tokens, output_tokens, total_tokens, input_audio_tokens, acquire_time, connection_reused, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("label", ::pyo3::IntoPyObjectExt::into_bound_py_any(label, py)?)?;
            kwargs.set_item("request_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id, py)?)?;
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(duration, py)?)?;
            kwargs.set_item("audio_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio_duration, py)?)?;
            kwargs.set_item("streamed", ::pyo3::IntoPyObjectExt::into_bound_py_any(streamed, py)?)?;
            kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
            kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
            kwargs.set_item("total_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_tokens, py)?)?;
            kwargs.set_item("input_audio_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_audio_tokens, py)?)?;
            kwargs.set_item("acquire_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(acquire_time, py)?)?;
            kwargs.set_item("connection_reused", ::pyo3::IntoPyObjectExt::into_bound_py_any(connection_reused, py)?)?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(STT_METRICS.import(py, "livekit.agents._generated.metrics", "STTMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static TTS_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tts_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::TtsMetrics> {
        let py = obj.py();
            if !obj.is_instance(TTS_METRICS.import(py, "livekit.agents._generated.metrics", "TTSMetrics")?)? {
                return Err(super::not_a(obj, "TTSMetrics"));
            }
            Ok(::lk_agents_abi::metrics::TtsMetrics {
                label: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "label", |v| v.extract::<::std::string::String>())?,
                request_id: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "request_id", |v| v.extract::<::std::string::String>())?,
                timestamp: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "TTSMetrics.timestamp"))?,
                ttfb: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "ttfb", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "TTSMetrics.ttfb"))?,
                duration: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "TTSMetrics.duration"))?,
                audio_duration: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "audio_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "TTSMetrics.audio_duration"))?,
                cancelled: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "cancelled", |v| v.extract::<bool>())?,
                characters_count: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "characters_count", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "TTSMetrics.characters_count"))?,
                input_tokens: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "input_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "TTSMetrics.input_tokens"))?,
                output_tokens: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "output_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "TTSMetrics.output_tokens"))?,
                streamed: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "streamed", |v| v.extract::<bool>())?,
                acquire_time: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "acquire_time", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "TTSMetrics.acquire_time"))?,
                connection_reused: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "connection_reused", |v| v.extract::<bool>())?,
                segment_id: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "segment_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                speech_id: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "speech_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "TTSMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn tts_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::TtsMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::TtsMetrics { label, request_id, timestamp, ttfb, duration, audio_duration, cancelled, characters_count, input_tokens, output_tokens, streamed, acquire_time, connection_reused, segment_id, speech_id, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("label", ::pyo3::IntoPyObjectExt::into_bound_py_any(label, py)?)?;
            kwargs.set_item("request_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id, py)?)?;
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("ttfb", ::pyo3::IntoPyObjectExt::into_bound_py_any(ttfb, py)?)?;
            kwargs.set_item("duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(duration, py)?)?;
            kwargs.set_item("audio_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio_duration, py)?)?;
            kwargs.set_item("cancelled", ::pyo3::IntoPyObjectExt::into_bound_py_any(cancelled, py)?)?;
            kwargs.set_item("characters_count", ::pyo3::IntoPyObjectExt::into_bound_py_any(characters_count, py)?)?;
            kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
            kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
            kwargs.set_item("streamed", ::pyo3::IntoPyObjectExt::into_bound_py_any(streamed, py)?)?;
            kwargs.set_item("acquire_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(acquire_time, py)?)?;
            kwargs.set_item("connection_reused", ::pyo3::IntoPyObjectExt::into_bound_py_any(connection_reused, py)?)?;
            kwargs.set_item("segment_id", match segment_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("speech_id", match speech_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(TTS_METRICS.import(py, "livekit.agents._generated.metrics", "TTSMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static VAD_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn vad_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::VadMetrics> {
        let py = obj.py();
            if !obj.is_instance(VAD_METRICS.import(py, "livekit.agents._generated.metrics", "VADMetrics")?)? {
                return Err(super::not_a(obj, "VADMetrics"));
            }
            Ok(::lk_agents_abi::metrics::VadMetrics {
                label: ::lk_agents_abi::python::attr_with(obj, "VADMetrics", "label", |v| v.extract::<::std::string::String>())?,
                timestamp: ::lk_agents_abi::python::attr_with(obj, "VADMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "VADMetrics.timestamp"))?,
                idle_time: ::lk_agents_abi::python::attr_with(obj, "VADMetrics", "idle_time", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "VADMetrics.idle_time"))?,
                inference_duration_total: ::lk_agents_abi::python::attr_with(obj, "VADMetrics", "inference_duration_total", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "VADMetrics.inference_duration_total"))?,
                inference_count: ::lk_agents_abi::python::attr_with(obj, "VADMetrics", "inference_count", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "VADMetrics.inference_count"))?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "VADMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn vad_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::VadMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::VadMetrics { label, timestamp, idle_time, inference_duration_total, inference_count, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("label", ::pyo3::IntoPyObjectExt::into_bound_py_any(label, py)?)?;
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("idle_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(idle_time, py)?)?;
            kwargs.set_item("inference_duration_total", ::pyo3::IntoPyObjectExt::into_bound_py_any(inference_duration_total, py)?)?;
            kwargs.set_item("inference_count", ::pyo3::IntoPyObjectExt::into_bound_py_any(inference_count, py)?)?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(VAD_METRICS.import(py, "livekit.agents._generated.metrics", "VADMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static EOU_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn eou_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::EouMetrics> {
        let py = obj.py();
            if !obj.is_instance(EOU_METRICS.import(py, "livekit.agents._generated.metrics", "EOUMetrics")?)? {
                return Err(super::not_a(obj, "EOUMetrics"));
            }
            Ok(::lk_agents_abi::metrics::EouMetrics {
                timestamp: ::lk_agents_abi::python::attr_with(obj, "EOUMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EOUMetrics.timestamp"))?,
                end_of_utterance_delay: ::lk_agents_abi::python::attr_with(obj, "EOUMetrics", "end_of_utterance_delay", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EOUMetrics.end_of_utterance_delay"))?,
                transcription_delay: ::lk_agents_abi::python::attr_with(obj, "EOUMetrics", "transcription_delay", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EOUMetrics.transcription_delay"))?,
                on_user_turn_completed_delay: ::lk_agents_abi::python::attr_with(obj, "EOUMetrics", "on_user_turn_completed_delay", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "EOUMetrics.on_user_turn_completed_delay"))?,
                speech_id: ::lk_agents_abi::python::attr_with(obj, "EOUMetrics", "speech_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "EOUMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn eou_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::EouMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::EouMetrics { timestamp, end_of_utterance_delay, transcription_delay, on_user_turn_completed_delay, speech_id, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("end_of_utterance_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(end_of_utterance_delay, py)?)?;
            kwargs.set_item("transcription_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(transcription_delay, py)?)?;
            kwargs.set_item("on_user_turn_completed_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(on_user_turn_completed_delay, py)?)?;
            kwargs.set_item("speech_id", match speech_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(EOU_METRICS.import(py, "livekit.agents._generated.metrics", "EOUMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static REALTIME_MODEL_METRICS_CACHED_TOKEN_DETAILS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_model_metrics_cached_token_details_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::CachedTokenDetails> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_MODEL_METRICS_CACHED_TOKEN_DETAILS.get_or_try_init(py, || -> PyResult<_> { Ok(py.import("livekit.agents._generated.metrics")?.getattr("RealtimeModelMetrics")?.getattr("CachedTokenDetails")?.cast_into::<::pyo3::types::PyType>()?.unbind()) })?.bind(py))? {
                return Err(super::not_a(obj, "RealtimeModelMetrics.CachedTokenDetails"));
            }
            Ok(::lk_agents_abi::metrics::CachedTokenDetails {
                audio_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.CachedTokenDetails", "audio_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.CachedTokenDetails.audio_tokens"))?,
                text_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.CachedTokenDetails", "text_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.CachedTokenDetails.text_tokens"))?,
                image_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.CachedTokenDetails", "image_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.CachedTokenDetails.image_tokens"))?,
            })
    }

    pub fn realtime_model_metrics_cached_token_details_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::CachedTokenDetails) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::CachedTokenDetails { audio_tokens, text_tokens, image_tokens } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("audio_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio_tokens, py)?)?;
            kwargs.set_item("text_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(text_tokens, py)?)?;
            kwargs.set_item("image_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(image_tokens, py)?)?;
            crate::host::construct(REALTIME_MODEL_METRICS_CACHED_TOKEN_DETAILS.get_or_try_init(py, || -> PyResult<_> { Ok(py.import("livekit.agents._generated.metrics")?.getattr("RealtimeModelMetrics")?.getattr("CachedTokenDetails")?.cast_into::<::pyo3::types::PyType>()?.unbind()) })?.bind(py).as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static REALTIME_MODEL_METRICS_INPUT_TOKEN_DETAILS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_model_metrics_input_token_details_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::InputTokenDetails> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_MODEL_METRICS_INPUT_TOKEN_DETAILS.get_or_try_init(py, || -> PyResult<_> { Ok(py.import("livekit.agents._generated.metrics")?.getattr("RealtimeModelMetrics")?.getattr("InputTokenDetails")?.cast_into::<::pyo3::types::PyType>()?.unbind()) })?.bind(py))? {
                return Err(super::not_a(obj, "RealtimeModelMetrics.InputTokenDetails"));
            }
            Ok(::lk_agents_abi::metrics::InputTokenDetails {
                audio_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.InputTokenDetails", "audio_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.InputTokenDetails.audio_tokens"))?,
                text_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.InputTokenDetails", "text_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.InputTokenDetails.text_tokens"))?,
                image_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.InputTokenDetails", "image_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.InputTokenDetails.image_tokens"))?,
                cached_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.InputTokenDetails", "cached_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.InputTokenDetails.cached_tokens"))?,
                cached_tokens_details: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.InputTokenDetails", "cached_tokens_details", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::realtime_model_metrics_cached_token_details_from_py(v)).map(Some) })?,
            })
    }

    pub fn realtime_model_metrics_input_token_details_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::InputTokenDetails) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::InputTokenDetails { audio_tokens, text_tokens, image_tokens, cached_tokens, cached_tokens_details } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("audio_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio_tokens, py)?)?;
            kwargs.set_item("text_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(text_tokens, py)?)?;
            kwargs.set_item("image_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(image_tokens, py)?)?;
            kwargs.set_item("cached_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(cached_tokens, py)?)?;
            kwargs.set_item("cached_tokens_details", match cached_tokens_details { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::realtime_model_metrics_cached_token_details_to_py(py, x)? })?;
            crate::host::construct(REALTIME_MODEL_METRICS_INPUT_TOKEN_DETAILS.get_or_try_init(py, || -> PyResult<_> { Ok(py.import("livekit.agents._generated.metrics")?.getattr("RealtimeModelMetrics")?.getattr("InputTokenDetails")?.cast_into::<::pyo3::types::PyType>()?.unbind()) })?.bind(py).as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static REALTIME_MODEL_METRICS_OUTPUT_TOKEN_DETAILS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_model_metrics_output_token_details_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::OutputTokenDetails> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_MODEL_METRICS_OUTPUT_TOKEN_DETAILS.get_or_try_init(py, || -> PyResult<_> { Ok(py.import("livekit.agents._generated.metrics")?.getattr("RealtimeModelMetrics")?.getattr("OutputTokenDetails")?.cast_into::<::pyo3::types::PyType>()?.unbind()) })?.bind(py))? {
                return Err(super::not_a(obj, "RealtimeModelMetrics.OutputTokenDetails"));
            }
            Ok(::lk_agents_abi::metrics::OutputTokenDetails {
                text_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.OutputTokenDetails", "text_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.OutputTokenDetails.text_tokens"))?,
                audio_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.OutputTokenDetails", "audio_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.OutputTokenDetails.audio_tokens"))?,
                image_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics.OutputTokenDetails", "image_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.OutputTokenDetails.image_tokens"))?,
            })
    }

    pub fn realtime_model_metrics_output_token_details_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::OutputTokenDetails) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::OutputTokenDetails { text_tokens, audio_tokens, image_tokens } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("text_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(text_tokens, py)?)?;
            kwargs.set_item("audio_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio_tokens, py)?)?;
            kwargs.set_item("image_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(image_tokens, py)?)?;
            crate::host::construct(REALTIME_MODEL_METRICS_OUTPUT_TOKEN_DETAILS.get_or_try_init(py, || -> PyResult<_> { Ok(py.import("livekit.agents._generated.metrics")?.getattr("RealtimeModelMetrics")?.getattr("OutputTokenDetails")?.cast_into::<::pyo3::types::PyType>()?.unbind()) })?.bind(py).as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static REALTIME_MODEL_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_model_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::RealtimeModelMetrics> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_MODEL_METRICS.import(py, "livekit.agents._generated.metrics", "RealtimeModelMetrics")?)? {
                return Err(super::not_a(obj, "RealtimeModelMetrics"));
            }
            Ok(::lk_agents_abi::metrics::RealtimeModelMetrics {
                label: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "label", |v| v.extract::<::std::string::String>())?,
                request_id: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "request_id", |v| v.extract::<::std::string::String>())?,
                timestamp: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeModelMetrics.timestamp"))?,
                duration: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeModelMetrics.duration"))?,
                session_duration: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "session_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeModelMetrics.session_duration"))?,
                ttft: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "ttft", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeModelMetrics.ttft"))?,
                cancelled: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "cancelled", |v| v.extract::<bool>())?,
                input_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "input_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.input_tokens"))?,
                output_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "output_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.output_tokens"))?,
                total_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "total_tokens", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "RealtimeModelMetrics.total_tokens"))?,
                tokens_per_second: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "tokens_per_second", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeModelMetrics.tokens_per_second"))?,
                input_token_details: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "input_token_details", |v| crate::generated::records::metrics::realtime_model_metrics_input_token_details_from_py(v))?,
                output_token_details: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "output_token_details", |v| crate::generated::records::metrics::realtime_model_metrics_output_token_details_from_py(v))?,
                acquire_time: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "acquire_time", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeModelMetrics.acquire_time"))?,
                connection_reused: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "connection_reused", |v| v.extract::<bool>())?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "RealtimeModelMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn realtime_model_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::RealtimeModelMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::RealtimeModelMetrics { label, request_id, timestamp, duration, session_duration, ttft, cancelled, input_tokens, output_tokens, total_tokens, tokens_per_second, input_token_details, output_token_details, acquire_time, connection_reused, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("label", ::pyo3::IntoPyObjectExt::into_bound_py_any(label, py)?)?;
            kwargs.set_item("request_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id, py)?)?;
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(duration, py)?)?;
            kwargs.set_item("session_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(session_duration, py)?)?;
            kwargs.set_item("ttft", ::pyo3::IntoPyObjectExt::into_bound_py_any(ttft, py)?)?;
            kwargs.set_item("cancelled", ::pyo3::IntoPyObjectExt::into_bound_py_any(cancelled, py)?)?;
            kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
            kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
            kwargs.set_item("total_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_tokens, py)?)?;
            kwargs.set_item("tokens_per_second", ::pyo3::IntoPyObjectExt::into_bound_py_any(tokens_per_second, py)?)?;
            kwargs.set_item("input_token_details", crate::generated::records::metrics::realtime_model_metrics_input_token_details_to_py(py, input_token_details)?)?;
            kwargs.set_item("output_token_details", crate::generated::records::metrics::realtime_model_metrics_output_token_details_to_py(py, output_token_details)?)?;
            kwargs.set_item("acquire_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(acquire_time, py)?)?;
            kwargs.set_item("connection_reused", ::pyo3::IntoPyObjectExt::into_bound_py_any(connection_reused, py)?)?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(REALTIME_MODEL_METRICS.import(py, "livekit.agents._generated.metrics", "RealtimeModelMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static AVATAR_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn avatar_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::AvatarMetrics> {
        let py = obj.py();
            if !obj.is_instance(AVATAR_METRICS.import(py, "livekit.agents._generated.metrics", "AvatarMetrics")?)? {
                return Err(super::not_a(obj, "AvatarMetrics"));
            }
            Ok(::lk_agents_abi::metrics::AvatarMetrics {
                timestamp: ::lk_agents_abi::python::attr_with(obj, "AvatarMetrics", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "AvatarMetrics.timestamp"))?,
                playback_latency: ::lk_agents_abi::python::attr_with(obj, "AvatarMetrics", "playback_latency", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "AvatarMetrics.playback_latency"))?,
                session_started_time: ::lk_agents_abi::python::attr_with(obj, "AvatarMetrics", "session_started_time", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "AvatarMetrics.session_started_time")).map(Some) })?,
                avatar_joined_time: ::lk_agents_abi::python::attr_with(obj, "AvatarMetrics", "avatar_joined_time", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "AvatarMetrics.avatar_joined_time")).map(Some) })?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "AvatarMetrics", "metadata", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::metrics::metadata_from_py(v)).map(Some) })?,
            })
    }

    pub fn avatar_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::AvatarMetrics) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::AvatarMetrics { timestamp, playback_latency, session_started_time, avatar_joined_time, metadata } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("timestamp", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp, py)?)?;
            kwargs.set_item("playback_latency", ::pyo3::IntoPyObjectExt::into_bound_py_any(playback_latency, py)?)?;
            kwargs.set_item("session_started_time", match session_started_time { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("avatar_joined_time", match avatar_joined_time { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("metadata", match metadata { None => py.None().into_bound(py), Some(x) => crate::generated::records::metrics::metadata_to_py(py, x)? })?;
            crate::host::construct(AVATAR_METRICS.import(py, "livekit.agents._generated.metrics", "AvatarMetrics")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/base.rs.
    static AGENT_METRICS__STT_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AGENT_METRICS__LLM_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AGENT_METRICS__TTS_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AGENT_METRICS__VAD_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AGENT_METRICS__EOU_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AGENT_METRICS__EOT_INFERENCE_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AGENT_METRICS__REALTIME_MODEL_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AGENT_METRICS__INTERRUPTION_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static AGENT_METRICS__AVATAR_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn agent_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::AgentMetrics> {
        if obj.is_instance(AGENT_METRICS__STT_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("STTMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::Stt(crate::generated::records::metrics::stt_metrics_from_py(obj)?));
        }
        if obj.is_instance(AGENT_METRICS__LLM_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("LLMMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::Llm(crate::generated::records::metrics::llm_metrics_from_py(obj)?));
        }
        if obj.is_instance(AGENT_METRICS__TTS_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("TTSMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::Tts(crate::generated::records::metrics::tts_metrics_from_py(obj)?));
        }
        if obj.is_instance(AGENT_METRICS__VAD_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("VADMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::Vad(crate::generated::records::metrics::vad_metrics_from_py(obj)?));
        }
        if obj.is_instance(AGENT_METRICS__EOU_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("EOUMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::Eou(crate::generated::records::metrics::eou_metrics_from_py(obj)?));
        }
        if obj.is_instance(AGENT_METRICS__EOT_INFERENCE_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("EOTInferenceMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::EotInference(crate::generated::records::metrics::eot_inference_metrics_from_py(obj)?));
        }
        if obj.is_instance(AGENT_METRICS__REALTIME_MODEL_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("RealtimeModelMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::RealtimeModel(crate::generated::records::metrics::realtime_model_metrics_from_py(obj)?));
        }
        if obj.is_instance(AGENT_METRICS__INTERRUPTION_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("InterruptionMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::Interruption(crate::generated::records::metrics::interruption_metrics_from_py(obj)?));
        }
        if obj.is_instance(AGENT_METRICS__AVATAR_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.metrics")?.getattr("AvatarMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::AgentMetrics::Avatar(crate::generated::records::metrics::avatar_metrics_from_py(obj)?));
        }
        Err(super::not_a(obj, "AgentMetrics"))
    }

    pub fn agent_metrics_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::AgentMetrics) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::metrics::AgentMetrics::Stt(x) => crate::generated::records::metrics::stt_metrics_to_py(py, x)?,
            ::lk_agents_abi::metrics::AgentMetrics::Llm(x) => crate::generated::records::metrics::llm_metrics_to_py(py, x)?,
            ::lk_agents_abi::metrics::AgentMetrics::Tts(x) => crate::generated::records::metrics::tts_metrics_to_py(py, x)?,
            ::lk_agents_abi::metrics::AgentMetrics::Vad(x) => crate::generated::records::metrics::vad_metrics_to_py(py, x)?,
            ::lk_agents_abi::metrics::AgentMetrics::Eou(x) => crate::generated::records::metrics::eou_metrics_to_py(py, x)?,
            ::lk_agents_abi::metrics::AgentMetrics::EotInference(x) => crate::generated::records::metrics::eot_inference_metrics_to_py(py, x)?,
            ::lk_agents_abi::metrics::AgentMetrics::RealtimeModel(x) => crate::generated::records::metrics::realtime_model_metrics_to_py(py, x)?,
            ::lk_agents_abi::metrics::AgentMetrics::Interruption(x) => crate::generated::records::metrics::interruption_metrics_to_py(py, x)?,
            ::lk_agents_abi::metrics::AgentMetrics::Avatar(x) => crate::generated::records::metrics::avatar_metrics_to_py(py, x)?,
        })
    }
}

pub mod realtime_options {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static REALTIME_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::RealtimeOptions> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_OPTIONS.import(py, "livekit.agents._generated.realtime_options", "RealtimeOptions")?)? {
                return Err(super::not_a(obj, "RealtimeOptions"));
            }
            Ok(::lk_agents_abi::llm::realtime::RealtimeOptions {
                input_audio_transcription: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "input_audio_transcription", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "RealtimeOptions.input_audio_transcription")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                input_audio_noise_reduction: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "input_audio_noise_reduction", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "RealtimeOptions.input_audio_noise_reduction")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                turn_detection: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "turn_detection", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "RealtimeOptions.turn_detection")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                tool_choice: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "tool_choice", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::ToolChoice>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                speed: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "speed", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeOptions.speed")).map(::lk_agents_abi::Maybe::Given) })?,
                tracing: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "tracing", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "RealtimeOptions.tracing")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                truncation: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "truncation", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "RealtimeOptions.truncation")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                reasoning: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "reasoning", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "RealtimeOptions.reasoning")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                max_session_duration: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "max_session_duration", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "RealtimeOptions.max_session_duration")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                voice: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "voice", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (v.extract::<::std::string::String>()).map(::lk_agents_abi::Maybe::Given) })?,
                max_response_output_tokens: ::lk_agents_abi::python::attr_with(obj, "RealtimeOptions", "max_response_output_tokens", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::value_from_py(v, "RealtimeOptions.max_response_output_tokens")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static OPEN_AI_REALTIME_CONFIG: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn open_ai_realtime_config_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::OpenAIRealtimeConfig> {
        let py = obj.py();
            if !obj.is_instance(OPEN_AI_REALTIME_CONFIG.import(py, "livekit.agents._generated.realtime_options", "OpenAIRealtimeConfig")?)? {
                return Err(super::not_a(obj, "OpenAIRealtimeConfig"));
            }
            Ok(::lk_agents_abi::llm::realtime::OpenAIRealtimeConfig {
                api_key: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "api_key", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                base_url: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "base_url", |v| v.extract::<::std::string::String>())?,
                conn_options: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "conn_options", |v| v.extract::<::lk_agents_abi::ApiConnectOptions>().map_err(::std::convert::Into::<PyErr>::into))?,
                options: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "options", |v| crate::generated::records::realtime_options::realtime_options_from_py(v))?,
                dialect: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "dialect", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                model: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "model", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                voice: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "voice", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                modalities: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "modalities", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                azure_deployment: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "azure_deployment", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                entra_token: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "entra_token", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                azure_defaults: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "azure_defaults", |v| v.extract::<bool>())?,
                api_secret: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "api_secret", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                inference_provider: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "inference_provider", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                inference_class: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "inference_class", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                label: ::lk_agents_abi::python::attr_with(obj, "OpenAIRealtimeConfig", "label", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }
}

pub mod realtime_wire {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static REMOTE_ITEM_ADDED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn remote_item_added_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::RemoteItemAdded> {
        let py = obj.py();
            if !obj.is_instance(REMOTE_ITEM_ADDED.import(py, "livekit.agents._generated.realtime_wire", "RemoteItemAdded")?)? {
                return Err(super::not_a(obj, "RemoteItemAdded"));
            }
            Ok(::lk_agents_abi::llm::realtime::RemoteItemAdded {
                previous_item_id: ::lk_agents_abi::python::attr_with(obj, "RemoteItemAdded", "previous_item_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                item: ::lk_agents_abi::python::attr_with(obj, "RemoteItemAdded", "item", |v| (crate::generated::records::chat_items::chat_item_from_py(v)).map(::std::boxed::Box::new))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static REALTIME_METRICS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn realtime_metrics_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::RealtimeMetrics> {
        let py = obj.py();
            if !obj.is_instance(REALTIME_METRICS.import(py, "livekit.agents._generated.realtime_wire", "RealtimeMetrics")?)? {
                return Err(super::not_a(obj, "RealtimeMetrics"));
            }
            Ok(::lk_agents_abi::llm::realtime::RealtimeMetrics {
                metrics: ::lk_agents_abi::python::attr_with(obj, "RealtimeMetrics", "metrics", |v| (crate::generated::records::metrics::agent_metrics_from_py(v)).map(::std::boxed::Box::new))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static FUNCTION_CALL_READY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn function_call_ready_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::FunctionCallReady> {
        let py = obj.py();
            if !obj.is_instance(FUNCTION_CALL_READY.import(py, "livekit.agents._generated.realtime_wire", "FunctionCallReady")?)? {
                return Err(super::not_a(obj, "FunctionCallReady"));
            }
            Ok(::lk_agents_abi::FunctionCallReady {
                call: ::lk_agents_abi::python::attr_with(obj, "FunctionCallReady", "call", |v| (crate::generated::records::chat_items::function_call_from_py(v)).map(::std::boxed::Box::new))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static AUDIO_DELTA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn audio_delta_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::AudioDelta> {
        let py = obj.py();
            if !obj.is_instance(AUDIO_DELTA.import(py, "livekit.agents._generated.realtime_wire", "AudioDelta")?)? {
                return Err(super::not_a(obj, "AudioDelta"));
            }
            Ok(::lk_agents_abi::AudioDelta {
                message_id: ::lk_agents_abi::python::attr_with(obj, "AudioDelta", "message_id", |v| v.extract::<::std::string::String>())?,
                response_id: ::lk_agents_abi::python::attr_with(obj, "AudioDelta", "response_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                frame: ::lk_agents_abi::python::attr_with(obj, "AudioDelta", "frame", |v| v.extract::<::lk_agents_abi::AudioFrame>().map_err(::std::convert::Into::<PyErr>::into))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static MESSAGE_MODALITIES: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn message_modalities_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::MessageModalities> {
        let py = obj.py();
            if !obj.is_instance(MESSAGE_MODALITIES.import(py, "livekit.agents._generated.realtime_wire", "MessageModalities")?)? {
                return Err(super::not_a(obj, "MessageModalities"));
            }
            Ok(::lk_agents_abi::MessageModalities {
                message_id: ::lk_agents_abi::python::attr_with(obj, "MessageModalities", "message_id", |v| v.extract::<::std::string::String>())?,
                response_id: ::lk_agents_abi::python::attr_with(obj, "MessageModalities", "response_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                modalities: ::lk_agents_abi::python::attr_with(obj, "MessageModalities", "modalities", |v| crate::generated::records::realtime_wire::modalities_from_py(v))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static GENERATION_DONE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn generation_done_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::GenerationDone> {
        let py = obj.py();
            if !obj.is_instance(GENERATION_DONE.import(py, "livekit.agents._generated.realtime_wire", "GenerationDone")?)? {
                return Err(super::not_a(obj, "GenerationDone"));
            }
            Ok(::lk_agents_abi::GenerationDone {
                response_id: ::lk_agents_abi::python::attr_with(obj, "GenerationDone", "response_id", |v| v.extract::<::std::string::String>())?,
                status: ::std::default::Default::default(),
                usage: ::std::default::Default::default(),
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static SERVER_GENERATION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn server_generation_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::ServerGeneration> {
        let py = obj.py();
            if !obj.is_instance(SERVER_GENERATION.import(py, "livekit.agents._generated.realtime_wire", "ServerGeneration")?)? {
                return Err(super::not_a(obj, "ServerGeneration"));
            }
            Ok(::lk_agents_abi::llm::realtime::ServerGeneration {
                generation: ::lk_agents_abi::python::attr_with(obj, "ServerGeneration", "generation", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static GENERATION_CREATED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn generation_created_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::GenerationCreated> {
        let py = obj.py();
            if !obj.is_instance(GENERATION_CREATED.import(py, "livekit.agents._generated.realtime_wire", "GenerationCreated")?)? {
                return Err(super::not_a(obj, "GenerationCreated"));
            }
            Ok(::lk_agents_abi::GenerationCreated {
                response_id: ::lk_agents_abi::python::attr_with(obj, "GenerationCreated", "response_id", |v| v.extract::<::std::string::String>())?,
                req_id: ::lk_agents_abi::python::attr_with(obj, "GenerationCreated", "req_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static MESSAGE_STARTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn message_started_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::MessageStarted> {
        let py = obj.py();
            if !obj.is_instance(MESSAGE_STARTED.import(py, "livekit.agents._generated.realtime_wire", "MessageStarted")?)? {
                return Err(super::not_a(obj, "MessageStarted"));
            }
            Ok(::lk_agents_abi::MessageStarted {
                message_id: ::lk_agents_abi::python::attr_with(obj, "MessageStarted", "message_id", |v| v.extract::<::std::string::String>())?,
                response_id: ::lk_agents_abi::python::attr_with(obj, "MessageStarted", "response_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static TEXT_DELTA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn text_delta_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::TextDelta> {
        let py = obj.py();
            if !obj.is_instance(TEXT_DELTA.import(py, "livekit.agents._generated.realtime_wire", "TextDelta")?)? {
                return Err(super::not_a(obj, "TextDelta"));
            }
            Ok(::lk_agents_abi::TextDelta {
                message_id: ::lk_agents_abi::python::attr_with(obj, "TextDelta", "message_id", |v| v.extract::<::std::string::String>())?,
                response_id: ::lk_agents_abi::python::attr_with(obj, "TextDelta", "response_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                delta: ::lk_agents_abi::python::attr_with(obj, "TextDelta", "delta", |v| v.extract::<::std::string::String>())?,
                start_time: ::lk_agents_abi::python::attr_with(obj, "TextDelta", "start_time", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "TextDelta.start_time")).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static MESSAGE_DONE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn message_done_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::MessageDone> {
        let py = obj.py();
            if !obj.is_instance(MESSAGE_DONE.import(py, "livekit.agents._generated.realtime_wire", "MessageDone")?)? {
                return Err(super::not_a(obj, "MessageDone"));
            }
            Ok(::lk_agents_abi::MessageDone {
                message_id: ::lk_agents_abi::python::attr_with(obj, "MessageDone", "message_id", |v| v.extract::<::std::string::String>())?,
                response_id: ::lk_agents_abi::python::attr_with(obj, "MessageDone", "response_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static INPUT_SPEECH_STOPPED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn input_speech_stopped_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::InputSpeechStopped> {
        let py = obj.py();
            if !obj.is_instance(INPUT_SPEECH_STOPPED.import(py, "livekit.agents._generated.realtime_wire", "InputSpeechStopped")?)? {
                return Err(super::not_a(obj, "InputSpeechStopped"));
            }
            Ok(::lk_agents_abi::InputSpeechStopped {
                user_transcription_enabled: ::lk_agents_abi::python::attr_with(obj, "InputSpeechStopped", "user_transcription_enabled", |v| v.extract::<bool>())?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static INPUT_TRANSCRIPTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn input_transcription_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::InputTranscription> {
        let py = obj.py();
            if !obj.is_instance(INPUT_TRANSCRIPTION.import(py, "livekit.agents._generated.realtime_wire", "InputTranscription")?)? {
                return Err(super::not_a(obj, "InputTranscription"));
            }
            Ok(::lk_agents_abi::InputTranscription {
                item_id: ::lk_agents_abi::python::attr_with(obj, "InputTranscription", "item_id", |v| v.extract::<::std::string::String>())?,
                transcript: ::lk_agents_abi::python::attr_with(obj, "InputTranscription", "transcript", |v| v.extract::<::std::string::String>())?,
                is_final: ::lk_agents_abi::python::attr_with(obj, "InputTranscription", "is_final", |v| v.extract::<bool>())?,
                confidence: ::lk_agents_abi::python::attr_with(obj, "InputTranscription", "confidence", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "InputTranscription.confidence")).map(Some) })?,
                turn_started_at: ::lk_agents_abi::python::attr_with(obj, "InputTranscription", "turn_started_at", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "InputTranscription.turn_started_at")).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static INPUT_SPEECH_STARTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn input_speech_started_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::InputSpeechStarted> {
        let py = obj.py();
            if !obj.is_instance(INPUT_SPEECH_STARTED.import(py, "livekit.agents._generated.realtime_wire", "InputSpeechStarted")?)? {
                return Err(super::not_a(obj, "InputSpeechStarted"));
            }
            Ok(::lk_agents_abi::llm::realtime::InputSpeechStarted {
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static SESSION_RECONNECTED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn session_reconnected_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::SessionReconnected> {
        let py = obj.py();
            if !obj.is_instance(SESSION_RECONNECTED.import(py, "livekit.agents._generated.realtime_wire", "SessionReconnected")?)? {
                return Err(super::not_a(obj, "SessionReconnected"));
            }
            Ok(::lk_agents_abi::llm::realtime::SessionReconnected {
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static HOST_REALTIME_ERROR: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn host_realtime_error_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::HostRealtimeError> {
        let py = obj.py();
            if !obj.is_instance(HOST_REALTIME_ERROR.import(py, "livekit.agents._generated.realtime_wire", "HostRealtimeError")?)? {
                return Err(super::not_a(obj, "HostRealtimeError"));
            }
            Ok(::lk_agents_abi::llm::realtime::HostRealtimeError {
                label: ::lk_agents_abi::python::attr_with(obj, "HostRealtimeError", "label", |v| v.extract::<::std::string::String>())?,
                message: ::lk_agents_abi::python::attr_with(obj, "HostRealtimeError", "message", |v| v.extract::<::std::string::String>())?,
                recoverable: ::lk_agents_abi::python::attr_with(obj, "HostRealtimeError", "recoverable", |v| v.extract::<bool>())?,
                timestamp: ::lk_agents_abi::python::attr_with(obj, "HostRealtimeError", "timestamp", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "HostRealtimeError.timestamp"))?,
                req_id: ::lk_agents_abi::python::attr_with(obj, "HostRealtimeError", "req_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                code: ::lk_agents_abi::python::attr_with(obj, "HostRealtimeError", "code", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.

    pub fn modalities_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Modalities> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "text" => Ok(::lk_agents_abi::Modalities::TextOnly),
                "audio" => Ok(::lk_agents_abi::Modalities::AudioAndText),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("Modalities: {other:?} is not one of 'text', 'audio'"))),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static EVENT__INPUT_SPEECH_STARTED: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__INPUT_SPEECH_STOPPED: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__INPUT_TRANSCRIPTION: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__GENERATION_CREATED: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__SERVER_GENERATION: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__MESSAGE_STARTED: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__TEXT_DELTA: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__AUDIO_DELTA: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__MESSAGE_MODALITIES: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__MESSAGE_DONE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__FUNCTION_CALL_READY: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__GENERATION_DONE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__REMOTE_ITEM_ADDED: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__SESSION_RECONNECTED: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__REALTIME_METRICS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static EVENT__HOST_REALTIME_ERROR: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::realtime::Event> {
        if obj.is_instance(EVENT__INPUT_SPEECH_STARTED.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("InputSpeechStarted")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::InputSpeechStarted(crate::generated::records::realtime_wire::input_speech_started_from_py(obj)?));
        }
        if obj.is_instance(EVENT__INPUT_SPEECH_STOPPED.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("InputSpeechStopped")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::InputSpeechStopped(crate::generated::records::realtime_wire::input_speech_stopped_from_py(obj)?));
        }
        if obj.is_instance(EVENT__INPUT_TRANSCRIPTION.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("InputTranscription")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::InputTranscription(crate::generated::records::realtime_wire::input_transcription_from_py(obj)?));
        }
        if obj.is_instance(EVENT__GENERATION_CREATED.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("GenerationCreated")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::GenerationCreated(crate::generated::records::realtime_wire::generation_created_from_py(obj)?));
        }
        if obj.is_instance(EVENT__SERVER_GENERATION.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("ServerGeneration")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::ServerGeneration(crate::generated::records::realtime_wire::server_generation_from_py(obj)?));
        }
        if obj.is_instance(EVENT__MESSAGE_STARTED.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("MessageStarted")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::MessageStarted(crate::generated::records::realtime_wire::message_started_from_py(obj)?));
        }
        if obj.is_instance(EVENT__TEXT_DELTA.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("TextDelta")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::TextDelta(crate::generated::records::realtime_wire::text_delta_from_py(obj)?));
        }
        if obj.is_instance(EVENT__AUDIO_DELTA.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("AudioDelta")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::AudioDelta(crate::generated::records::realtime_wire::audio_delta_from_py(obj)?));
        }
        if obj.is_instance(EVENT__MESSAGE_MODALITIES.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("MessageModalities")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::MessageModalities(crate::generated::records::realtime_wire::message_modalities_from_py(obj)?));
        }
        if obj.is_instance(EVENT__MESSAGE_DONE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("MessageDone")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::MessageDone(crate::generated::records::realtime_wire::message_done_from_py(obj)?));
        }
        if obj.is_instance(EVENT__FUNCTION_CALL_READY.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("FunctionCallReady")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::FunctionCall(crate::generated::records::realtime_wire::function_call_ready_from_py(obj)?));
        }
        if obj.is_instance(EVENT__GENERATION_DONE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("GenerationDone")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::GenerationDone(crate::generated::records::realtime_wire::generation_done_from_py(obj)?));
        }
        if obj.is_instance(EVENT__REMOTE_ITEM_ADDED.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("RemoteItemAdded")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::RemoteItemAdded(crate::generated::records::realtime_wire::remote_item_added_from_py(obj)?));
        }
        if obj.is_instance(EVENT__SESSION_RECONNECTED.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("SessionReconnected")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::SessionReconnected(crate::generated::records::realtime_wire::session_reconnected_from_py(obj)?));
        }
        if obj.is_instance(EVENT__REALTIME_METRICS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("RealtimeMetrics")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::Metrics(crate::generated::records::realtime_wire::realtime_metrics_from_py(obj)?));
        }
        if obj.is_instance(EVENT__HOST_REALTIME_ERROR.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.realtime_wire")?.getattr("HostRealtimeError")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::llm::realtime::Event::HostError(crate::generated::records::realtime_wire::host_realtime_error_from_py(obj)?));
        }
        Err(super::not_a(obj, "Event"))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.

    pub fn realtime_command_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::RealtimeCommand) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::RealtimeCommand::UpdateInstructions { .. } => return Err(::pyo3::exceptions::PyTypeError::new_err("UpdateInstructions never crosses to the host")),
            ::lk_agents_abi::RealtimeCommand::UpdateChatCtx { .. } => return Err(::pyo3::exceptions::PyTypeError::new_err("UpdateChatCtx never crosses to the host")),
            ::lk_agents_abi::RealtimeCommand::UpdateTools { .. } => return Err(::pyo3::exceptions::PyTypeError::new_err("UpdateTools never crosses to the host")),
            ::lk_agents_abi::RealtimeCommand::PushAudio(x) => crate::generated::records::realtime_wire::push_audio_to_py(py, x)?,
            ::lk_agents_abi::RealtimeCommand::PushItem { .. } => return Err(::pyo3::exceptions::PyTypeError::new_err("PushItem never crosses to the host")),
            ::lk_agents_abi::RealtimeCommand::GenerateReply(x) => crate::generated::records::realtime_wire::generate_reply_to_py(py, x)?,
            ::lk_agents_abi::RealtimeCommand::Say(x) => crate::generated::records::realtime_wire::say_to_py(py, x)?,
            ::lk_agents_abi::RealtimeCommand::CancelReply { .. } => return Err(::pyo3::exceptions::PyTypeError::new_err("CancelReply never crosses to the host")),
            ::lk_agents_abi::RealtimeCommand::CommitAudio(x) => crate::generated::records::realtime_wire::commit_audio_to_py(py, x)?,
            ::lk_agents_abi::RealtimeCommand::ClearAudio(x) => crate::generated::records::realtime_wire::clear_audio_to_py(py, x)?,
            ::lk_agents_abi::RealtimeCommand::Interrupt(x) => crate::generated::records::realtime_wire::interrupt_to_py(py, x)?,
            ::lk_agents_abi::RealtimeCommand::Truncate(x) => crate::generated::records::realtime_wire::truncate_to_py(py, x)?,
        })
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static PUSH_AUDIO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn push_audio_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::PushAudio) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::PushAudio { frame } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("frame", ::pyo3::IntoPyObjectExt::into_bound_py_any(frame, py)?)?;
            crate::host::construct(PUSH_AUDIO.import(py, "livekit.agents._generated.realtime_wire", "PushAudio")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static GENERATE_REPLY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn generate_reply_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::GenerateReply) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::GenerateReply { req_id, instructions, tool_choice } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("req_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(req_id, py)?)?;
            kwargs.set_item("instructions", match instructions { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("tool_choice", match tool_choice { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(GENERATE_REPLY.import(py, "livekit.agents._generated.realtime_wire", "GenerateReply")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static SAY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn say_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::Say) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::Say { req_id, text, add_to_chat_ctx } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("req_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(req_id, py)?)?;
            kwargs.set_item("text", ::pyo3::IntoPyObjectExt::into_bound_py_any(text, py)?)?;
            kwargs.set_item("add_to_chat_ctx", ::pyo3::IntoPyObjectExt::into_bound_py_any(add_to_chat_ctx, py)?)?;
            crate::host::construct(SAY.import(py, "livekit.agents._generated.realtime_wire", "Say")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static COMMIT_AUDIO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn commit_audio_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::realtime::CommitAudio) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::realtime::CommitAudio {  } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            crate::host::construct(COMMIT_AUDIO.import(py, "livekit.agents._generated.realtime_wire", "CommitAudio")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static CLEAR_AUDIO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn clear_audio_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::realtime::ClearAudio) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::realtime::ClearAudio {  } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            crate::host::construct(CLEAR_AUDIO.import(py, "livekit.agents._generated.realtime_wire", "ClearAudio")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static INTERRUPT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn interrupt_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::realtime::Interrupt) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::llm::realtime::Interrupt { response_id } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("response_id", match response_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(INTERRUPT.import(py, "livekit.agents._generated.realtime_wire", "Interrupt")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
    static TRUNCATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn truncate_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::Truncate) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::Truncate { message_id, has_audio, audio_end_ms, audio_transcript } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
            kwargs.set_item("has_audio", ::pyo3::IntoPyObjectExt::into_bound_py_any(has_audio, py)?)?;
            kwargs.set_item("audio_end_ms", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio_end_ms, py)?)?;
            kwargs.set_item("audio_transcript", match audio_transcript { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(TRUNCATE.import(py, "livekit.agents._generated.realtime_wire", "Truncate")?.as_any(), (), Some(&kwargs))
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

pub mod session_options {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn detector_kind_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::DetectorKind> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "streaming" => Ok(::lk_agents_abi::voice::session_options::DetectorKind::Streaming),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("DetectorKind: {other:?} is not one of 'streaming'"))),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn session_kwargs_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::SessionKwargs> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::voice::session_options::SessionKwargs {
                turn_handling: match obj.get_item("turn_handling")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::turn_handling_options_from_py((&v)))?), _ => None },
                stt_context_options: { let v = obj.get_item("stt_context_options")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.stt_context_options"))?; (crate::conversions::host_value_from_py((&v)))? },
                max_tool_steps: { let v = obj.get_item("max_tool_steps")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.max_tool_steps"))?; (::lk_agents_abi::python::number_from_py::<i64>((&v), "SessionKwargs.max_tool_steps"))? },
                use_tts_aligned_transcript: match obj.get_item("use_tts_aligned_transcript")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { ((&v).extract::<bool>()).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
                tts_text_transforms: match obj.get_item("tts_text_transforms")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if (&v).is_none() { Ok(None) } else { (crate::conversions::text_transform_set_from_py((&v))).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
                min_consecutive_speech_delay: { let v = obj.get_item("min_consecutive_speech_delay")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.min_consecutive_speech_delay"))?; (::lk_agents_abi::python::number_from_py::<f64>((&v), "SessionKwargs.min_consecutive_speech_delay"))? },
                expressive: { let v = obj.get_item("expressive")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.expressive"))?; (crate::conversions::expressive_from_py((&v)))? },
                aec_warmup_duration: match obj.get_item("aec_warmup_duration")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if (&v).is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>((&v), "SessionKwargs.aec_warmup_duration")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
                ivr_detection: { let v = obj.get_item("ivr_detection")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.ivr_detection"))?; ((&v).extract::<bool>())? },
                user_away_timeout: { let v = obj.get_item("user_away_timeout")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.user_away_timeout"))?; (if (&v).is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>((&v), "SessionKwargs.user_away_timeout")).map(Some) })? },
                transcription_timeout: { let v = obj.get_item("transcription_timeout")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.transcription_timeout"))?; (if (&v).is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>((&v), "SessionKwargs.transcription_timeout")).map(Some) })? },
                session_close_transcript_timeout: { let v = obj.get_item("session_close_transcript_timeout")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.session_close_transcript_timeout"))?; (::lk_agents_abi::python::number_from_py::<f64>((&v), "SessionKwargs.session_close_transcript_timeout"))? },
                conn_options: { let v = obj.get_item("conn_options")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionKwargs.conn_options"))?; (crate::conversions::session_conn_options_from_py((&v)))? },
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn update_kwargs_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::UpdateKwargs> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::voice::session_options::UpdateKwargs {
                endpointing_opts: match obj.get_item("endpointing_opts")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::endpointing_options_from_py((&v)))?), _ => None },
                turn_detection: match obj.get_item("turn_detection")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (crate::conversions::turn_detection_kwarg_from_py((&v))).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
                expressive: match obj.get_item("expressive")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (crate::conversions::expressive_from_py((&v))).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
                keyterms: match obj.get_item("keyterms")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::std::string::String>() }).collect::<PyResult<Vec<_>>>())?), _ => None },
                interruption: match obj.get_item("interruption")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::interruption_options_from_py((&v)))?), _ => None },
                min_consecutive_speech_delay: match obj.get_item("min_consecutive_speech_delay")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "UpdateKwargs.min_consecutive_speech_delay"))?), _ => None },
                tts_text_transforms: match obj.get_item("tts_text_transforms")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if (&v).is_none() { Ok(None) } else { (crate::conversions::text_transform_set_from_py((&v))).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn session_legs_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::SessionLegs> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::voice::session_options::SessionLegs {
                stt: { let v = obj.get_item("stt")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.stt"))?; ((&v).extract::<bool>())? },
                vad: { let v = obj.get_item("vad")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.vad"))?; ((&v).extract::<bool>())? },
                using_default_vad: { let v = obj.get_item("using_default_vad")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.using_default_vad"))?; ((&v).extract::<bool>())? },
                llm: { let v = obj.get_item("llm")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.llm"))?; ((&v).extract::<bool>())? },
                tts: { let v = obj.get_item("tts")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.tts"))?; ((&v).extract::<bool>())? },
                realtime: { let v = obj.get_item("realtime")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.realtime"))?; ((&v).extract::<bool>())? },
                detector: { let v = obj.get_item("detector")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.detector"))?; (if (&v).is_none() { Ok(None) } else { (crate::generated::records::session_options::detector_kind_from_py((&v))).map(Some) })? },
                adaptive_detector: { let v = obj.get_item("adaptive_detector")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.adaptive_detector"))?; ((&v).extract::<bool>())? },
                audio_enabled: { let v = obj.get_item("audio_enabled")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.audio_enabled"))?; ((&v).extract::<bool>())? },
                audio_output: { let v = obj.get_item("audio_output")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.audio_output"))?; ((&v).extract::<bool>())? },
                sink_can_pause: { let v = obj.get_item("sink_can_pause")?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err("SessionLegs.sink_can_pause"))?; ((&v).extract::<bool>())? },
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn resolved_endpointing_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::session_options::ResolvedEndpointing) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::session_options::ResolvedEndpointing { mode, min_delay, max_delay, alpha } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("mode", ::pyo3::IntoPyObjectExt::into_bound_py_any(mode, py)?)?;
            dict.set_item("min_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(min_delay, py)?)?;
            dict.set_item("max_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_delay, py)?)?;
            dict.set_item("alpha", ::pyo3::IntoPyObjectExt::into_bound_py_any(alpha, py)?)?;
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn resolved_interruption_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::session_options::ResolvedInterruption) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::session_options::ResolvedInterruption { enabled, mode, discard_audio_if_uninterruptible, min_duration, min_words, resume_false_interruption, false_interruption_timeout, backchannel_boundary } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("enabled", ::pyo3::IntoPyObjectExt::into_bound_py_any(enabled, py)?)?;
            if let Some(x) = mode {
                dict.set_item("mode", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            dict.set_item("discard_audio_if_uninterruptible", ::pyo3::IntoPyObjectExt::into_bound_py_any(discard_audio_if_uninterruptible, py)?)?;
            dict.set_item("min_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(min_duration, py)?)?;
            dict.set_item("min_words", ::pyo3::IntoPyObjectExt::into_bound_py_any(min_words, py)?)?;
            dict.set_item("resume_false_interruption", ::pyo3::IntoPyObjectExt::into_bound_py_any(resume_false_interruption, py)?)?;
            dict.set_item("false_interruption_timeout", match false_interruption_timeout { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("backchannel_boundary", match backchannel_boundary { None => py.None().into_bound(py), Some(x) => { let (x0, x1,) = x; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?])?.into_any() } })?;
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn resolved_preemptive_generation_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::session_options::ResolvedPreemptiveGeneration) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::session_options::ResolvedPreemptiveGeneration { enabled, preemptive_tts, max_speech_duration, max_retries } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("enabled", ::pyo3::IntoPyObjectExt::into_bound_py_any(enabled, py)?)?;
            dict.set_item("preemptive_tts", ::pyo3::IntoPyObjectExt::into_bound_py_any(preemptive_tts, py)?)?;
            dict.set_item("max_speech_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_speech_duration, py)?)?;
            dict.set_item("max_retries", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_retries, py)?)?;
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn resolved_user_turn_limit_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::session_options::ResolvedUserTurnLimit) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::session_options::ResolvedUserTurnLimit { max_words, max_duration } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("max_words", match max_words { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("max_duration", match max_duration { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn endpointing_overrides_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::session_options::EndpointingOverrides) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::session_options::EndpointingOverrides { mode, min_delay, max_delay, alpha } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            if let Some(x) = mode {
                dict.set_item("mode", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = min_delay {
                dict.set_item("min_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = max_delay {
                dict.set_item("max_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            if let Some(x) = alpha {
                dict.set_item("alpha", ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?;
            }
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn resolved_session_options_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::session_options::ResolvedSessionOptions) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::session_options::ResolvedSessionOptions { endpointing, interruption, preemptive_generation, user_turn_limit, turn_detection_mode, endpointing_overrides, max_tool_steps, min_consecutive_speech_delay, use_tts_aligned_transcript, tts_text_transforms, ivr_detection, user_away_timeout, transcription_timeout, aec_warmup_duration, session_close_transcript_timeout } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("endpointing", crate::generated::records::session_options::resolved_endpointing_to_py(py, endpointing)?)?;
            dict.set_item("interruption", crate::generated::records::session_options::resolved_interruption_to_py(py, interruption)?)?;
            dict.set_item("preemptive_generation", crate::generated::records::session_options::resolved_preemptive_generation_to_py(py, preemptive_generation)?)?;
            dict.set_item("user_turn_limit", crate::generated::records::session_options::resolved_user_turn_limit_to_py(py, user_turn_limit)?)?;
            dict.set_item("turn_detection_mode", match turn_detection_mode { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("endpointing_overrides", crate::generated::records::session_options::endpointing_overrides_to_py(py, endpointing_overrides)?)?;
            dict.set_item("max_tool_steps", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_tool_steps, py)?)?;
            dict.set_item("min_consecutive_speech_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(min_consecutive_speech_delay, py)?)?;
            dict.set_item("use_tts_aligned_transcript", match use_tts_aligned_transcript { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("tts_text_transforms", match tts_text_transforms { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
            dict.set_item("ivr_detection", ::pyo3::IntoPyObjectExt::into_bound_py_any(ivr_detection, py)?)?;
            dict.set_item("user_away_timeout", match user_away_timeout { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("transcription_timeout", match transcription_timeout { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("aec_warmup_duration", match aec_warmup_duration { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("session_close_transcript_timeout", ::pyo3::IntoPyObjectExt::into_bound_py_any(session_close_transcript_timeout, py)?)?;
            Ok(dict.into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn interruption_gate_verdict_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::session_options::InterruptionGateVerdict) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::session_options::InterruptionGateVerdict { build, reason, level, message } = value;
            let dict = ::pyo3::types::PyDict::new(py);
            dict.set_item("build", ::pyo3::IntoPyObjectExt::into_bound_py_any(build, py)?)?;
            dict.set_item("reason", match reason { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("level", match level { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            dict.set_item("message", match message { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            Ok(dict.into_any())
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
            kwargs.set_item("speech_handle", <dyn ::lk_agents_abi::voice::speech_handle::Speech as crate::generated::NativeToHost>::to_host(py, speech_handle)?)?;
            crate::host::construct(SPEECH_CREATED_WIRE.import(py, "livekit.agents._generated.session_wire", "SpeechCreatedWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static TOOL_CALL_STARTED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_call_started_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::ToolCallStarted) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::ToolCallStarted { function_call } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("function_call", crate::generated::records::chat_items::function_call_to_py(py, function_call)?)?;
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
            kwargs.set_item("item", crate::generated::records::chat_items::chat_item_to_py(py, item)?)?;
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
            kwargs.set_item("item", crate::generated::records::chat_items::chat_item_to_py(py, item)?)?;
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
            kwargs.set_item("speech_handle", match speech_handle { None => py.None().into_bound(py), Some(x) => <dyn ::lk_agents_abi::voice::speech_handle::Speech as crate::generated::NativeToHost>::to_host(py, x)? })?;
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
            kwargs.set_item("reason", crate::generated::records::events::close_reason_to_py(py, reason)?)?;
            kwargs.set_item("error", match error { None => py.None().into_bound(py), Some(x) => crate::generated::records::session_wire::session_error_wire_to_py(py, x)? })?;
            crate::host::construct(CLOSE_WIRE.import(py, "livekit.agents._generated.session_wire", "CloseWire")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/events.rs.
    static METRICS_COLLECTED_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn metrics_collected_wire_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::voice::events::MetricsCollectedWire) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::voice::events::MetricsCollectedWire { metrics, speech_id, leg } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("metrics", crate::generated::records::metrics::agent_metrics_to_py(py, metrics)?)?;
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
            kwargs.set_item("model_usage", ::pyo3::types::PyList::new(py, model_usage.into_iter().map(|x| PyResult::Ok(crate::generated::records::usage::model_usage_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
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
            kwargs.set_item("items", ::pyo3::types::PyList::new(py, items.into_iter().map(|x| PyResult::Ok(crate::generated::records::chat_items::chat_item_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
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
            ::lk_agents_abi::voice::events::HostWire::AmdResult(x) => crate::generated::records::amd::amd_prediction_event_to_py(py, x)?,
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
            let ::lk_agents_abi::storage::TaskInternal { ask, streams, answered } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("ask", match ask { None => py.None().into_bound(py), Some(x) => crate::generated::records::storage_class::ask_to_py(py, x)? })?;
            kwargs.set_item("streams", ::pyo3::types::PyList::new(py, streams.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_class::stream_info_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("answered", ::pyo3::IntoPyObjectExt::into_bound_py_any(answered, py)?)?;
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
                tool_executions: ::lk_agents_abi::python::attr_with(obj, "HandlerOutput", "tool_executions", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::storage_turn::tool_execution_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
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

    // Declared in rust/crates/lk-agents-abi/src/storage/journal.rs.
    static TOOL_EXECUTION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tool_execution_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::journal::ToolExecution> {
        let py = obj.py();
            if !obj.is_instance(TOOL_EXECUTION.import(py, "livekit.agents._generated.storage_turn", "ToolExecution")?)? {
                return Err(super::not_a(obj, "ToolExecution"));
            }
            Ok(::lk_agents_abi::storage::journal::ToolExecution {
                phase: ::lk_agents_abi::python::attr_with(obj, "ToolExecution", "phase", |v| crate::generated::records::storage_turn::tool_phase_from_py(v))?,
                call_id: ::lk_agents_abi::python::attr_with(obj, "ToolExecution", "call_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                name: ::lk_agents_abi::python::attr_with(obj, "ToolExecution", "name", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                is_error: ::lk_agents_abi::python::attr_with(obj, "ToolExecution", "is_error", |v| if v.is_none() { Ok(None) } else { (v.extract::<bool>()).map(Some) })?,
                text: ::lk_agents_abi::python::attr_with(obj, "ToolExecution", "text", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/storage/journal.rs.

    pub fn tool_phase_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::storage::journal::ToolPhase> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "started" => Ok(::lk_agents_abi::storage::journal::ToolPhase::Started),
                "update" => Ok(::lk_agents_abi::storage::journal::ToolPhase::Update),
                "ended" => Ok(::lk_agents_abi::storage::journal::ToolPhase::Ended),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("ToolPhase: {other:?} is not one of 'started', 'update', 'ended'"))),
            }
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
                model_usage: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "model_usage", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::usage::model_usage_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                chat_history: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "chat_history", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::chat_items::chat_item_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                omitted_events: ::lk_agents_abi::python::attr_with(obj, "SessionReportInput", "omitted_events", |v| ::lk_agents_abi::python::number_from_py::<usize>(v, "SessionReportInput.omitted_events"))?,
            })
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
                usage: ::lk_agents_abi::python::attr_with(obj, "SessionReportSnapshot", "usage", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::usage::model_usage_from_py(item) }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
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
            kwargs.set_item("usage", match usage { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(crate::generated::records::usage::model_usage_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
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
                history: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "history", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::chat_items::chat_item_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                parent: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "parent", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::text_session::parent_call_from_py(v)).map(Some) })?,
                context_scope: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "context_scope", |v| v.extract::<::std::string::String>())?,
                completed: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "completed", |v| v.extract::<bool>())?,
                pending_calls: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "pending_calls", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::text_session::pending_child_call_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
                answer_tool: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "answer_tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                visible_dialogue: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "visible_dialogue", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::chat_items::chat_message_from_py(item) }).collect::<PyResult<Vec<_>>>())?,
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
            kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(crate::generated::records::chat_items::chat_item_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("parent", match parent { None => py.None().into_bound(py), Some(x) => crate::generated::records::text_session::parent_call_to_py(py, x)? })?;
            kwargs.set_item("context_scope", ::pyo3::IntoPyObjectExt::into_bound_py_any(context_scope, py)?)?;
            kwargs.set_item("completed", ::pyo3::IntoPyObjectExt::into_bound_py_any(completed, py)?)?;
            kwargs.set_item("pending_calls", ::pyo3::types::PyList::new(py, pending_calls.into_iter().map(|x| PyResult::Ok(crate::generated::records::text_session::pending_child_call_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("visible_dialogue", ::pyo3::types::PyList::new(py, visible_dialogue.into_iter().map(|x| PyResult::Ok(crate::generated::records::chat_items::chat_message_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
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
                host: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "host", |v| crate::generated::agent::AgentCallbacksProxy::from_host(&v))?,
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
                message: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "message", |v| crate::generated::records::chat_items::chat_item_from_py(v))?,
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

pub mod tool_context {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs.

    pub fn duplicate_mode_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::mcp::DuplicateMode> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "allow" => Ok(::lk_agents_abi::llm::mcp::DuplicateMode::Allow),
                "reject" => Ok(::lk_agents_abi::llm::mcp::DuplicateMode::Reject),
                "replace" => Ok(::lk_agents_abi::llm::mcp::DuplicateMode::Replace),
                "confirm" => Ok(::lk_agents_abi::llm::mcp::DuplicateMode::Confirm),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("DuplicateMode: {other:?} is not one of 'allow', 'reject', 'replace', 'confirm'"))),
            }
    }

    pub fn duplicate_mode_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::mcp::DuplicateMode) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::llm::mcp::DuplicateMode::Allow => "allow",
                ::lk_agents_abi::llm::mcp::DuplicateMode::Reject => "reject",
                ::lk_agents_abi::llm::mcp::DuplicateMode::Replace => "replace",
                ::lk_agents_abi::llm::mcp::DuplicateMode::Confirm => "confirm",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
    }

    // Declared in rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs.

    pub fn duplicate_scope_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::mcp::DuplicateScope> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "name" => Ok(::lk_agents_abi::llm::mcp::DuplicateScope::Name),
                "name_and_args" => Ok(::lk_agents_abi::llm::mcp::DuplicateScope::NameAndArgs),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("DuplicateScope: {other:?} is not one of 'name', 'name_and_args'"))),
            }
    }

    pub fn duplicate_scope_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::llm::mcp::DuplicateScope) -> PyResult<Bound<'py, PyAny>> {
            let value = match value {
                ::lk_agents_abi::llm::mcp::DuplicateScope::Name => "name",
                ::lk_agents_abi::llm::mcp::DuplicateScope::NameAndArgs => "name_and_args",
            };
            Ok(::pyo3::types::PyString::new(py, value).into_any())
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

pub mod turn {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn endpointing_mode_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::EndpointingMode> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "fixed" => Ok(::lk_agents_abi::voice::session_options::EndpointingMode::Fixed),
                "dynamic" => Ok(::lk_agents_abi::voice::session_options::EndpointingMode::Dynamic),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("EndpointingMode: {other:?} is not one of 'fixed', 'dynamic'"))),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn interruption_mode_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::InterruptionMode> {
            let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
            match &*value {
                "adaptive" => Ok(::lk_agents_abi::voice::session_options::InterruptionMode::Adaptive),
                "vad" => Ok(::lk_agents_abi::voice::session_options::InterruptionMode::Vad),
                other => Err(::pyo3::exceptions::PyValueError::new_err(format!("InterruptionMode: {other:?} is not one of 'adaptive', 'vad'"))),
            }
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn endpointing_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::EndpointingKwargs> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::voice::session_options::EndpointingKwargs {
                mode: match obj.get_item("mode")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::endpointing_mode_from_py((&v)))?), _ => None },
                min_delay: match obj.get_item("min_delay")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "EndpointingOptions.min_delay"))?), _ => None },
                max_delay: match obj.get_item("max_delay")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "EndpointingOptions.max_delay"))?), _ => None },
                alpha: match obj.get_item("alpha")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "EndpointingOptions.alpha"))?), _ => None },
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn interruption_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::InterruptionKwargs> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::voice::session_options::InterruptionKwargs {
                enabled: match obj.get_item("enabled")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<bool>())?), _ => None },
                mode: match obj.get_item("mode")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::interruption_mode_from_py((&v)))?), _ => None },
                discard_audio_if_uninterruptible: match obj.get_item("discard_audio_if_uninterruptible")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<bool>())?), _ => None },
                min_duration: match obj.get_item("min_duration")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "InterruptionOptions.min_duration"))?), _ => None },
                min_words: match obj.get_item("min_words")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<i64>((&v), "InterruptionOptions.min_words"))?), _ => None },
                resume_false_interruption: match obj.get_item("resume_false_interruption")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<bool>())?), _ => None },
                false_interruption_timeout: match obj.get_item("false_interruption_timeout")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if (&v).is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>((&v), "InterruptionOptions.false_interruption_timeout")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
                backchannel_boundary: match obj.get_item("backchannel_boundary")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if (&v).is_none() { Ok(None) } else { (crate::conversions::backchannel_boundary_from_py((&v))).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn preemptive_generation_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::PreemptiveKwargs> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::voice::session_options::PreemptiveKwargs {
                enabled: match obj.get_item("enabled")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<bool>())?), _ => None },
                preemptive_tts: match obj.get_item("preemptive_tts")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some(((&v).extract::<bool>())?), _ => None },
                max_speech_duration: match obj.get_item("max_speech_duration")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<f64>((&v), "PreemptiveGenerationOptions.max_speech_duration"))?), _ => None },
                max_retries: match obj.get_item("max_retries")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((::lk_agents_abi::python::number_from_py::<i64>((&v), "PreemptiveGenerationOptions.max_retries"))?), _ => None },
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn user_turn_limit_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::UserTurnLimitKwargs> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::voice::session_options::UserTurnLimitKwargs {
                max_words: match obj.get_item("max_words")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if (&v).is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<i64>((&v), "UserTurnLimitOptions.max_words")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
                max_duration: match obj.get_item("max_duration")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if (&v).is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>((&v), "UserTurnLimitOptions.max_duration")).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.

    pub fn turn_handling_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session_options::TurnHandlingKwargs> {
            let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, "dict"))?;
            Ok(::lk_agents_abi::voice::session_options::TurnHandlingKwargs {
                turn_detection: match obj.get_item("turn_detection")? { Some(v) => (if ::lk_agents_abi::python::is_not_given((&v))? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (crate::conversions::turn_detection_kwarg_from_py((&v))).map(::lk_agents_abi::Maybe::Given) })?, None => ::lk_agents_abi::Maybe::NotGiven },
                endpointing: match obj.get_item("endpointing")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::endpointing_options_from_py((&v)))?), _ => None },
                interruption: match obj.get_item("interruption")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::interruption_options_from_py((&v)))?), _ => None },
                preemptive_generation: match obj.get_item("preemptive_generation")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::preemptive_generation_options_from_py((&v)))?), _ => None },
                user_turn_limit: match obj.get_item("user_turn_limit")? { Some(v) if !v.is_none() && !::lk_agents_abi::python::is_not_given(&v)? => Some((crate::generated::records::turn::user_turn_limit_options_from_py((&v)))?), _ => None },
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

pub mod usage {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/metrics/usage/records.rs.
    static LLM_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn llm_model_usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::LlmModelUsage> {
        let py = obj.py();
            if !obj.is_instance(LLM_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "LLMModelUsage")?)? {
                return Err(super::not_a(obj, "LLMModelUsage"));
            }
            Ok(::lk_agents_abi::metrics::LlmModelUsage {
                provider: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "provider", |v| v.extract::<::std::string::String>())?,
                model: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "model", |v| v.extract::<::std::string::String>())?,
                input_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_tokens"))?,
                input_cached_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_cached_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_cached_tokens"))?,
                input_cache_creation_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_cache_creation_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_cache_creation_tokens"))?,
                input_audio_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_audio_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_audio_tokens"))?,
                input_cached_audio_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_cached_audio_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_cached_audio_tokens"))?,
                input_text_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_text_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_text_tokens"))?,
                input_cached_text_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_cached_text_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_cached_text_tokens"))?,
                input_image_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_image_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_image_tokens"))?,
                input_cached_image_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "input_cached_image_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.input_cached_image_tokens"))?,
                output_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "output_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.output_tokens"))?,
                output_audio_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "output_audio_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.output_audio_tokens"))?,
                output_text_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "output_text_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.output_text_tokens"))?,
                output_reasoning_tokens: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "output_reasoning_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "LLMModelUsage.output_reasoning_tokens"))?,
                session_duration: ::lk_agents_abi::python::attr_with(obj, "LLMModelUsage", "session_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "LLMModelUsage.session_duration"))?,
            })
    }

    pub fn llm_model_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::LlmModelUsage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::LlmModelUsage { provider, model, input_tokens, input_cached_tokens, input_cache_creation_tokens, input_audio_tokens, input_cached_audio_tokens, input_text_tokens, input_cached_text_tokens, input_image_tokens, input_cached_image_tokens, output_tokens, output_audio_tokens, output_text_tokens, output_reasoning_tokens, session_duration } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("provider", ::pyo3::IntoPyObjectExt::into_bound_py_any(provider, py)?)?;
            kwargs.set_item("model", ::pyo3::IntoPyObjectExt::into_bound_py_any(model, py)?)?;
            kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
            kwargs.set_item("input_cached_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_cached_tokens, py)?)?;
            kwargs.set_item("input_cache_creation_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_cache_creation_tokens, py)?)?;
            kwargs.set_item("input_audio_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_audio_tokens, py)?)?;
            kwargs.set_item("input_cached_audio_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_cached_audio_tokens, py)?)?;
            kwargs.set_item("input_text_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_text_tokens, py)?)?;
            kwargs.set_item("input_cached_text_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_cached_text_tokens, py)?)?;
            kwargs.set_item("input_image_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_image_tokens, py)?)?;
            kwargs.set_item("input_cached_image_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_cached_image_tokens, py)?)?;
            kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
            kwargs.set_item("output_audio_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_audio_tokens, py)?)?;
            kwargs.set_item("output_text_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_text_tokens, py)?)?;
            kwargs.set_item("output_reasoning_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_reasoning_tokens, py)?)?;
            kwargs.set_item("session_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(session_duration, py)?)?;
            crate::host::construct(LLM_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "LLMModelUsage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/usage/records.rs.
    static TTS_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn tts_model_usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::TtsModelUsage> {
        let py = obj.py();
            if !obj.is_instance(TTS_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "TTSModelUsage")?)? {
                return Err(super::not_a(obj, "TTSModelUsage"));
            }
            Ok(::lk_agents_abi::metrics::TtsModelUsage {
                provider: ::lk_agents_abi::python::attr_with(obj, "TTSModelUsage", "provider", |v| v.extract::<::std::string::String>())?,
                model: ::lk_agents_abi::python::attr_with(obj, "TTSModelUsage", "model", |v| v.extract::<::std::string::String>())?,
                input_tokens: ::lk_agents_abi::python::attr_with(obj, "TTSModelUsage", "input_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "TTSModelUsage.input_tokens"))?,
                output_tokens: ::lk_agents_abi::python::attr_with(obj, "TTSModelUsage", "output_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "TTSModelUsage.output_tokens"))?,
                characters_count: ::lk_agents_abi::python::attr_with(obj, "TTSModelUsage", "characters_count", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "TTSModelUsage.characters_count"))?,
                audio_duration: ::lk_agents_abi::python::attr_with(obj, "TTSModelUsage", "audio_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "TTSModelUsage.audio_duration"))?,
            })
    }

    pub fn tts_model_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::TtsModelUsage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::TtsModelUsage { provider, model, input_tokens, output_tokens, characters_count, audio_duration } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("provider", ::pyo3::IntoPyObjectExt::into_bound_py_any(provider, py)?)?;
            kwargs.set_item("model", ::pyo3::IntoPyObjectExt::into_bound_py_any(model, py)?)?;
            kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
            kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
            kwargs.set_item("characters_count", ::pyo3::IntoPyObjectExt::into_bound_py_any(characters_count, py)?)?;
            kwargs.set_item("audio_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio_duration, py)?)?;
            crate::host::construct(TTS_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "TTSModelUsage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/usage/records.rs.
    static STT_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn stt_model_usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::SttModelUsage> {
        let py = obj.py();
            if !obj.is_instance(STT_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "STTModelUsage")?)? {
                return Err(super::not_a(obj, "STTModelUsage"));
            }
            Ok(::lk_agents_abi::metrics::SttModelUsage {
                provider: ::lk_agents_abi::python::attr_with(obj, "STTModelUsage", "provider", |v| v.extract::<::std::string::String>())?,
                model: ::lk_agents_abi::python::attr_with(obj, "STTModelUsage", "model", |v| v.extract::<::std::string::String>())?,
                input_tokens: ::lk_agents_abi::python::attr_with(obj, "STTModelUsage", "input_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "STTModelUsage.input_tokens"))?,
                input_audio_tokens: ::lk_agents_abi::python::attr_with(obj, "STTModelUsage", "input_audio_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "STTModelUsage.input_audio_tokens"))?,
                output_tokens: ::lk_agents_abi::python::attr_with(obj, "STTModelUsage", "output_tokens", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "STTModelUsage.output_tokens"))?,
                audio_duration: ::lk_agents_abi::python::attr_with(obj, "STTModelUsage", "audio_duration", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "STTModelUsage.audio_duration"))?,
            })
    }

    pub fn stt_model_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::SttModelUsage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::SttModelUsage { provider, model, input_tokens, input_audio_tokens, output_tokens, audio_duration } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("provider", ::pyo3::IntoPyObjectExt::into_bound_py_any(provider, py)?)?;
            kwargs.set_item("model", ::pyo3::IntoPyObjectExt::into_bound_py_any(model, py)?)?;
            kwargs.set_item("input_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_tokens, py)?)?;
            kwargs.set_item("input_audio_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(input_audio_tokens, py)?)?;
            kwargs.set_item("output_tokens", ::pyo3::IntoPyObjectExt::into_bound_py_any(output_tokens, py)?)?;
            kwargs.set_item("audio_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio_duration, py)?)?;
            crate::host::construct(STT_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "STTModelUsage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/usage/records.rs.
    static INTERRUPTION_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn interruption_model_usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::InterruptionModelUsage> {
        let py = obj.py();
            if !obj.is_instance(INTERRUPTION_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "InterruptionModelUsage")?)? {
                return Err(super::not_a(obj, "InterruptionModelUsage"));
            }
            Ok(::lk_agents_abi::metrics::InterruptionModelUsage {
                provider: ::lk_agents_abi::python::attr_with(obj, "InterruptionModelUsage", "provider", |v| v.extract::<::std::string::String>())?,
                model: ::lk_agents_abi::python::attr_with(obj, "InterruptionModelUsage", "model", |v| v.extract::<::std::string::String>())?,
                total_requests: ::lk_agents_abi::python::attr_with(obj, "InterruptionModelUsage", "total_requests", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "InterruptionModelUsage.total_requests"))?,
            })
    }

    pub fn interruption_model_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::InterruptionModelUsage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::InterruptionModelUsage { provider, model, total_requests } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("provider", ::pyo3::IntoPyObjectExt::into_bound_py_any(provider, py)?)?;
            kwargs.set_item("model", ::pyo3::IntoPyObjectExt::into_bound_py_any(model, py)?)?;
            kwargs.set_item("total_requests", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_requests, py)?)?;
            crate::host::construct(INTERRUPTION_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "InterruptionModelUsage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/usage/records.rs.
    static EOT_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn eot_model_usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::EotModelUsage> {
        let py = obj.py();
            if !obj.is_instance(EOT_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "EOTModelUsage")?)? {
                return Err(super::not_a(obj, "EOTModelUsage"));
            }
            Ok(::lk_agents_abi::metrics::EotModelUsage {
                provider: ::lk_agents_abi::python::attr_with(obj, "EOTModelUsage", "provider", |v| v.extract::<::std::string::String>())?,
                model: ::lk_agents_abi::python::attr_with(obj, "EOTModelUsage", "model", |v| v.extract::<::std::string::String>())?,
                total_requests: ::lk_agents_abi::python::attr_with(obj, "EOTModelUsage", "total_requests", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "EOTModelUsage.total_requests"))?,
            })
    }

    pub fn eot_model_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::EotModelUsage) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::metrics::EotModelUsage { provider, model, total_requests } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("provider", ::pyo3::IntoPyObjectExt::into_bound_py_any(provider, py)?)?;
            kwargs.set_item("model", ::pyo3::IntoPyObjectExt::into_bound_py_any(model, py)?)?;
            kwargs.set_item("total_requests", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_requests, py)?)?;
            crate::host::construct(EOT_MODEL_USAGE.import(py, "livekit.agents._generated.usage", "EOTModelUsage")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/metrics/usage/records.rs.
    static MODEL_USAGE__LLM_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static MODEL_USAGE__TTS_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static MODEL_USAGE__STT_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static MODEL_USAGE__INTERRUPTION_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();
    static MODEL_USAGE__EOT_MODEL_USAGE: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn model_usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::metrics::ModelUsage> {
        if obj.is_instance(MODEL_USAGE__LLM_MODEL_USAGE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.usage")?.getattr("LLMModelUsage")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::ModelUsage::Llm(crate::generated::records::usage::llm_model_usage_from_py(obj)?));
        }
        if obj.is_instance(MODEL_USAGE__TTS_MODEL_USAGE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.usage")?.getattr("TTSModelUsage")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::ModelUsage::Tts(crate::generated::records::usage::tts_model_usage_from_py(obj)?));
        }
        if obj.is_instance(MODEL_USAGE__STT_MODEL_USAGE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.usage")?.getattr("STTModelUsage")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::ModelUsage::Stt(crate::generated::records::usage::stt_model_usage_from_py(obj)?));
        }
        if obj.is_instance(MODEL_USAGE__INTERRUPTION_MODEL_USAGE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.usage")?.getattr("InterruptionModelUsage")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::ModelUsage::Interruption(crate::generated::records::usage::interruption_model_usage_from_py(obj)?));
        }
        if obj.is_instance(MODEL_USAGE__EOT_MODEL_USAGE.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.usage")?.getattr("EOTModelUsage")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::metrics::ModelUsage::Eot(crate::generated::records::usage::eot_model_usage_from_py(obj)?));
        }
        Err(super::not_a(obj, "ModelUsage"))
    }

    pub fn model_usage_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::metrics::ModelUsage) -> PyResult<Bound<'py, PyAny>> {
        Ok(match value {
            ::lk_agents_abi::metrics::ModelUsage::Llm(x) => crate::generated::records::usage::llm_model_usage_to_py(py, x)?,
            ::lk_agents_abi::metrics::ModelUsage::Tts(x) => crate::generated::records::usage::tts_model_usage_to_py(py, x)?,
            ::lk_agents_abi::metrics::ModelUsage::Stt(x) => crate::generated::records::usage::stt_model_usage_to_py(py, x)?,
            ::lk_agents_abi::metrics::ModelUsage::Interruption(x) => crate::generated::records::usage::interruption_model_usage_to_py(py, x)?,
            ::lk_agents_abi::metrics::ModelUsage::Eot(x) => crate::generated::records::usage::eot_model_usage_to_py(py, x)?,
        })
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
                chat_ctx: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "chat_ctx", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::chat_items::chat_item_from_py(item) }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                llm: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "llm", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (crate::generated::llm::LlmProxy::from_host(&v)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                tts: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "tts", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (crate::generated::tts::TtsProxy::from_host(&v)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                stt: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "stt", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (crate::generated::stt::SttProxy::from_host(&v)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                vad: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "vad", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (if v.is_none() { Ok(None) } else { (crate::generated::vad::VadProxy::from_host(&v)).map(Some) }).map(::lk_agents_abi::Maybe::Given) })?,
                turn_handling: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "turn_handling", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::turn::turn_handling_options_from_py(v)).map(Some) })?,
                min_consecutive_speech_delay: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "min_consecutive_speech_delay", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "UpdateAgentArgs.min_consecutive_speech_delay")).map(Some) })?,
                use_tts_aligned_transcript: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "use_tts_aligned_transcript", |v| if v.is_none() { Ok(None) } else { (v.extract::<bool>()).map(Some) })?,
                expressive: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "expressive", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (crate::conversions::expressive_from_py(v)).map(::lk_agents_abi::Maybe::Given) })?,
                llm_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "llm_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.llm_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                tts_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "tts_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.tts_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                stt_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "stt_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.stt_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                transcription_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "transcription_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.transcription_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                realtime_audio_output_node: ::lk_agents_abi::python::attr_with(obj, "UpdateAgentArgs", "realtime_audio_output_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("UpdateAgentArgs.realtime_audio_output_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
            })
    }

    // Declared in rust/crates/lk-agents-abi/src/voice/session.rs.
    static VOICE_SESSION_ARGS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn voice_session_args_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::session::VoiceSessionArgs> {
        let py = obj.py();
            if !obj.is_instance(VOICE_SESSION_ARGS.import(py, "livekit.agents._generated.voice_session", "VoiceSessionArgs")?)? {
                return Err(super::not_a(obj, "VoiceSessionArgs"));
            }
            Ok(::lk_agents_abi::voice::session::VoiceSessionArgs {
                instructions: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "instructions", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                dynamic_instructions: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "dynamic_instructions", |v| if v.is_none() { Ok(None) } else { (crate::generated::agent::HostInstructionsProxy::from_host(&v)).map(Some) })?,
                agent: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "agent", |v| v.extract::<::std::string::String>())?,
                agent_host: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "agent_host", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
                chat_ctx: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "chat_ctx", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::chat_items::chat_item_from_py(item) }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                storage: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "storage", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::storage::voice::VoiceStorage>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.storage", ": expected a native voice storage manager")))).map(Some) })?,
                stt: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "stt", |v| if v.is_none() { Ok(None) } else { (crate::generated::stt::SttProxy::from_host(&v)).map(Some) })?,
                vad: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "vad", |v| if v.is_none() { Ok(None) } else { (crate::generated::vad::VadProxy::from_host(&v)).map(Some) })?,
                llm: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "llm", |v| if v.is_none() { Ok(None) } else { (crate::generated::llm::LlmProxy::from_host(&v)).map(Some) })?,
                realtime: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "realtime", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Realtime>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.realtime", ": expected OpenAIRealtimeModel")))).map(Some) })?,
                tts: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "tts", |v| if v.is_none() { Ok(None) } else { (crate::generated::tts::TtsProxy::from_host(&v)).map(Some) })?,
                on_event: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "on_event", |v| if v.is_none() { Ok(None) } else { (crate::generated::voice::SessionEventsProxy::from_host(&v)).map(Some) })?,
                hooks: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "hooks", |v| if v.is_none() { Ok(None) } else { (crate::generated::voice::SessionHooksProxy::from_host(&v)).map(Some) })?,
                participant_identity: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "participant_identity", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                auto_gain_control: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "auto_gain_control", |v| v.extract::<bool>())?,
                input_sample_rate: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "input_sample_rate", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "VoiceSessionArgs.input_sample_rate"))?,
                output_sample_rate: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "output_sample_rate", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "VoiceSessionArgs.output_sample_rate"))?,
                agent_turn_handling: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "agent_turn_handling", |v| if v.is_none() { Ok(None) } else { (crate::generated::records::turn::turn_handling_options_from_py(v)).map(Some) })?,
                agent_min_consecutive_speech_delay: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "agent_min_consecutive_speech_delay", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "VoiceSessionArgs.agent_min_consecutive_speech_delay")).map(Some) })?,
                agent_use_tts_aligned_transcript: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "agent_use_tts_aligned_transcript", |v| if v.is_none() { Ok(None) } else { (v.extract::<bool>()).map(Some) })?,
                turn_detector: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "turn_detector", |v| if v.is_none() { Ok(None) } else { (crate::generated::turn::EotDetectorProxy::from_host(&v)).map(Some) })?,
                interruption_detector: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "interruption_detector", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::turn::interruption::InterruptionSource>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.interruption_detector", ": interruption_detector must be an `inference.AdaptiveInterruptionDetector`")))).map(Some) })?,
                text_output: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "text_output", |v| v.extract::<bool>())?,
                room_io_config: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "room_io_config", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::voice::room_io::RoomIoConfig>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
                tools: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "tools", |v| if v.is_none() { Ok(None) } else { (v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; crate::generated::records::agent_run::function_tool_info_from_py(item) }).collect::<PyResult<Vec<_>>>()).map(Some) })?,
                tools_host: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "tools_host", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
                voice_agent: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "voice_agent", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::agent::AgentFrame>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.voice_agent", ": expected an agent's voice frame (`_core.VoiceAgent`)")))).map(Some) })?,
                transcription_node: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "transcription_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.transcription_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                realtime_audio_output_node: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "realtime_audio_output_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.realtime_audio_output_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                llm_node: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "llm_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.llm_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                tts_node: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "tts_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.tts_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                stt_node: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "stt_node", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.stt_node", ": expected the session's nodes (`_core.AgentNodes`)")))).map(Some) })?,
                host_audio_input: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "host_audio_input", |v| if v.is_none() { Ok(None) } else { (crate::generated::voice::AudioInputProxy::from_host(&v)).map(Some) })?,
                host_audio_output: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "host_audio_output", |v| if v.is_none() { Ok(None) } else { (crate::generated::voice::AudioOutputProxy::from_host(&v)).map(Some) })?,
                host_video_input: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "host_video_input", |v| if v.is_none() { Ok(None) } else { (crate::generated::voice::VideoInputProxy::from_host(&v)).map(Some) })?,
                video_sampler: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "video_sampler", |v| if v.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = v.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "VoiceSessionArgs.video_sampler"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((::lk_agents_abi::python::number_from_py::<f64>(&item_0, "VoiceSessionArgs.video_sampler"))?, (::lk_agents_abi::python::number_from_py::<f64>(&item_1, "VoiceSessionArgs.video_sampler"))?,)) })()).map(Some) })?,
                host_video_sampler: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "host_video_sampler", |v| v.extract::<bool>())?,
                host_transcription_output: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "host_transcription_output", |v| if v.is_none() { Ok(None) } else { (crate::generated::voice::TextOutputProxy::from_host(&v)).map(Some) })?,
                using_default_vad: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "using_default_vad", |v| v.extract::<bool>())?,
                hook_bound: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "hook_bound", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(v, "VoiceSessionArgs.hook_bound")).map(Some) })?,
                max_unrecoverable_errors: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "max_unrecoverable_errors", |v| ::lk_agents_abi::python::number_from_py::<u32>(v, "VoiceSessionArgs.max_unrecoverable_errors"))?,
                stt_context: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "stt_context", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::turn::keyterms::SttContextConfig>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
                keyterm_llm: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "keyterm_llm", |v| if v.is_none() { Ok(None) } else { (crate::generated::llm::LlmProxy::from_host(&v)).map(Some) })?,
                keyterm_tools: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "keyterm_tools", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
                agent_expressive: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "agent_expressive", |v| if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (crate::conversions::expressive_from_py(v)).map(::lk_agents_abi::Maybe::Given) })?,
                session_id: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "session_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
                remote_session: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "remote_session", |v| if v.is_none() { Ok(None) } else { (crate::generated::worker::SessionPortProxy::from_host(&v)).map(Some) })?,
                console: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "console", |v| if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::console::ConsolePort>(&v).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("VoiceSessionArgs.console", ": expected a native ConsoleIo")))).map(Some) })?,
                job: ::lk_agents_abi::python::attr_with(obj, "VoiceSessionArgs", "job", |v| if v.is_none() { Ok(None) } else { (crate::generated::worker::SessionJobProxy::from_host(&v)).map(Some) })?,
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

    // Declared in rust/crates/lk-agents-abi/src/worker/job.rs.
    static START_JOB_FIELDS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn start_job_fields_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::worker::job::StartJobFields> {
        let py = obj.py();
            if !obj.is_instance(START_JOB_FIELDS.import(py, "livekit.agents._generated.worker_ipc", "StartJobFields")?)? {
                return Err(super::not_a(obj, "StartJobFields"));
            }
            Ok(::lk_agents_abi::worker::job::StartJobFields {
                job: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "job", |v| { if v.is_instance_of::<::pyo3::types::PyBytes>() || v.is_instance_of::<::pyo3::types::PyByteArray>() || v.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&v, "StartJobFields.job") } else { v.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "StartJobFields.job") }).collect::<PyResult<Vec<_>>>()) } })?,
                name: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "name", |v| v.extract::<::std::string::String>())?,
                identity: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "identity", |v| v.extract::<::std::string::String>())?,
                metadata: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "metadata", |v| v.extract::<::std::string::String>())?,
                url: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "url", |v| v.extract::<::std::string::String>())?,
                token: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "token", |v| v.extract::<::std::string::String>())?,
                worker_id: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "worker_id", |v| v.extract::<::std::string::String>())?,
                fake_job: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "fake_job", |v| v.extract::<bool>())?,
                received_at: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "received_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "StartJobFields.received_at"))?,
                accepted_at: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "accepted_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "StartJobFields.accepted_at"))?,
                assigned_at: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "assigned_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "StartJobFields.assigned_at"))?,
                launched_at: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "launched_at", |v| ::lk_agents_abi::python::number_from_py::<f64>(v, "StartJobFields.launched_at"))?,
                traceparent: ::lk_agents_abi::python::attr_with(obj, "StartJobFields", "traceparent", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            })
    }

    pub fn start_job_fields_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::job::StartJobFields) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::job::StartJobFields { job, name, identity, metadata, url, token, worker_id, fake_job, received_at, accepted_at, assigned_at, launched_at, traceparent } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("job", ::pyo3::types::PyBytes::new(py, &job).into_any())?;
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
            kwargs.set_item("metadata", ::pyo3::IntoPyObjectExt::into_bound_py_any(metadata, py)?)?;
            kwargs.set_item("url", ::pyo3::IntoPyObjectExt::into_bound_py_any(url, py)?)?;
            kwargs.set_item("token", ::pyo3::IntoPyObjectExt::into_bound_py_any(token, py)?)?;
            kwargs.set_item("worker_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(worker_id, py)?)?;
            kwargs.set_item("fake_job", ::pyo3::IntoPyObjectExt::into_bound_py_any(fake_job, py)?)?;
            kwargs.set_item("received_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(received_at, py)?)?;
            kwargs.set_item("accepted_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(accepted_at, py)?)?;
            kwargs.set_item("assigned_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(assigned_at, py)?)?;
            kwargs.set_item("launched_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(launched_at, py)?)?;
            kwargs.set_item("traceparent", match traceparent { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(START_JOB_FIELDS.import(py, "livekit.agents._generated.worker_ipc", "StartJobFields")?.as_any(), (), Some(&kwargs))
    }
}

pub mod worker_server {
    use pyo3::prelude::*;

    // Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
    static JOB_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn job_info_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::worker::JobInfo) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::worker::JobInfo { id, dispatch_id, r#type, room, room_sid, agent_name, metadata, deployment, enable_recording, enable_redaction, proto } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
            kwargs.set_item("dispatch_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(dispatch_id, py)?)?;
            kwargs.set_item("type", ::pyo3::IntoPyObjectExt::into_bound_py_any(r#type, py)?)?;
            kwargs.set_item("room", match room { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("room_sid", match room_sid { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
            kwargs.set_item("metadata", ::pyo3::IntoPyObjectExt::into_bound_py_any(metadata, py)?)?;
            kwargs.set_item("deployment", ::pyo3::IntoPyObjectExt::into_bound_py_any(deployment, py)?)?;
            kwargs.set_item("enable_recording", ::pyo3::IntoPyObjectExt::into_bound_py_any(enable_recording, py)?)?;
            kwargs.set_item("enable_redaction", ::pyo3::IntoPyObjectExt::into_bound_py_any(enable_redaction, py)?)?;
            kwargs.set_item("proto", ::pyo3::types::PyBytes::new(py, &proto).into_any())?;
            crate::host::construct(JOB_INFO.import(py, "livekit.agents._generated.worker_server", "JobInfo")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
    static RUNNING_JOB_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn running_job_info_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::worker::RunningJobInfo) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::worker::RunningJobInfo { accept_arguments, job, url, token, worker_id, fake_job, received_at, accepted_at, assigned_at } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("accept_arguments", crate::generated::records::worker_server::accept_arguments_to_py(py, accept_arguments)?)?;
            kwargs.set_item("job", crate::generated::records::worker_server::job_info_to_py(py, job)?)?;
            kwargs.set_item("url", ::pyo3::IntoPyObjectExt::into_bound_py_any(url, py)?)?;
            kwargs.set_item("token", ::pyo3::IntoPyObjectExt::into_bound_py_any(token, py)?)?;
            kwargs.set_item("worker_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(worker_id, py)?)?;
            kwargs.set_item("fake_job", ::pyo3::IntoPyObjectExt::into_bound_py_any(fake_job, py)?)?;
            kwargs.set_item("received_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(received_at, py)?)?;
            kwargs.set_item("accepted_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(accepted_at, py)?)?;
            kwargs.set_item("assigned_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(assigned_at, py)?)?;
            crate::host::construct(RUNNING_JOB_INFO.import(py, "livekit.agents._generated.worker_server", "RunningJobInfo")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
    static ACCEPT_ARGUMENTS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn accept_arguments_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::worker::AcceptArguments) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::worker::AcceptArguments { name, identity, metadata, attributes } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
            kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
            kwargs.set_item("metadata", ::pyo3::IntoPyObjectExt::into_bound_py_any(metadata, py)?)?;
            kwargs.set_item("attributes", ::pyo3::types::PyList::new(py, attributes.into_iter().map(|x| PyResult::Ok({ let (x0, x1,) = x; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?])?.into_any() })).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            crate::host::construct(ACCEPT_ARGUMENTS.import(py, "livekit.agents._generated.worker_server", "AcceptArguments")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
    static PROCESS_POOL_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn process_pool_options_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::worker::ProcessPoolOptions) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::worker::ProcessPoolOptions { num_idle_processes, initialize_timeout, close_timeout, session_end_timeout, memory_warn_mb, memory_limit_mb, http_proxy } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("num_idle_processes", ::pyo3::IntoPyObjectExt::into_bound_py_any(num_idle_processes, py)?)?;
            kwargs.set_item("initialize_timeout", ::pyo3::IntoPyObjectExt::into_bound_py_any(initialize_timeout, py)?)?;
            kwargs.set_item("close_timeout", ::pyo3::IntoPyObjectExt::into_bound_py_any(close_timeout, py)?)?;
            kwargs.set_item("session_end_timeout", ::pyo3::IntoPyObjectExt::into_bound_py_any(session_end_timeout, py)?)?;
            kwargs.set_item("memory_warn_mb", ::pyo3::IntoPyObjectExt::into_bound_py_any(memory_warn_mb, py)?)?;
            kwargs.set_item("memory_limit_mb", ::pyo3::IntoPyObjectExt::into_bound_py_any(memory_limit_mb, py)?)?;
            kwargs.set_item("http_proxy", match http_proxy { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
            crate::host::construct(PROCESS_POOL_OPTIONS.import(py, "livekit.agents._generated.worker_server", "ProcessPoolOptions")?.as_any(), (), Some(&kwargs))
    }

    // Declared in rust/crates/lk-agents-abi/src/worker/worker.rs.
    static PROCESS_START_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
        ::pyo3::sync::PyOnceLock::new();

    pub fn process_start_options_to_py<'py>(py: Python<'py>, value: ::lk_agents_abi::worker::worker::ProcessStartOptions) -> PyResult<Bound<'py, PyAny>> {
            let ::lk_agents_abi::worker::worker::ProcessStartOptions { devmode, job_processes, env, job_executor_type, multiprocessing_context, pool } = value;
            let kwargs = ::pyo3::types::PyDict::new(py);
            kwargs.set_item("devmode", ::pyo3::IntoPyObjectExt::into_bound_py_any(devmode, py)?)?;
            kwargs.set_item("job_processes", ::pyo3::IntoPyObjectExt::into_bound_py_any(job_processes, py)?)?;
            kwargs.set_item("env", ::pyo3::types::PyList::new(py, env.into_iter().map(|x| PyResult::Ok({ let (x0, x1,) = x; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?])?.into_any() })).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
            kwargs.set_item("job_executor_type", ::pyo3::IntoPyObjectExt::into_bound_py_any(job_executor_type, py)?)?;
            kwargs.set_item("multiprocessing_context", ::pyo3::IntoPyObjectExt::into_bound_py_any(multiprocessing_context, py)?)?;
            kwargs.set_item("pool", crate::generated::records::worker_server::process_pool_options_to_py(py, pool)?)?;
            crate::host::construct(PROCESS_START_OPTIONS.import(py, "livekit.agents._generated.worker_server", "ProcessStartOptions")?.as_any(), (), Some(&kwargs))
    }

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
    static INSTRUCTIONS_ARG__WORKFLOW_INSTRUCTIONS: ::pyo3::sync::PyOnceLock<Py<PyAny>> = ::pyo3::sync::PyOnceLock::new();

    pub fn instructions_arg_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::workflows::InstructionsArg> {
        if obj.is_instance(INSTRUCTIONS_ARG__WORKFLOW_INSTRUCTIONS.get_or_try_init(obj.py(), || PyResult::Ok(obj.py().import("livekit.agents._generated.workflows")?.getattr("WorkflowInstructions")?.unbind()))?.bind(obj.py()))? {
            return Ok(::lk_agents_abi::voice::workflows::InstructionsArg::Workflow(crate::generated::records::workflows::workflow_instructions_from_py(obj)?));
        }
        if obj.is_instance_of::<::pyo3::types::PyString>() {
            return Ok(::lk_agents_abi::voice::workflows::InstructionsArg::Fixed(obj.extract::<::std::string::String>()?));
        }
        Err(super::not_a(obj, "InstructionsArg"))
    }
}
