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

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
static MESSAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::a2a::turn::Message {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { message_id, parts, metadata, channel, received_at, task_id, context_id, raw } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
        kwargs.set_item("parts", ::pyo3::types::PyList::new(py, parts.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("metadata", ::lk_agents_abi::python::extra_to_py(py, &metadata)?.into_any())?;
        kwargs.set_item("channel", ::pyo3::IntoPyObjectExt::into_bound_py_any(channel, py)?)?;
        kwargs.set_item("received_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(received_at, py)?)?;
        kwargs.set_item("task_id", match task_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("context_id", match context_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("raw", ::lk_agents_abi::python::value_to_py(py, &raw)?)?;
        MESSAGE.import(py, "livekit.agents._generated.storage_turn", "Message")?.call((), Some(&kwargs))
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
        let Self { thread_id, lane, message, batch, history, state, context_scope_id, context_scopes, read_only, task_id, tasks, caller_history, answer_tool } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
        kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
        kwargs.set_item("message", ::lk_agents_abi::python::value_to_py(py, &message)?)?;
        kwargs.set_item("batch", ::pyo3::types::PyList::new(py, batch.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("state", match state { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
        kwargs.set_item("context_scope_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(context_scope_id, py)?)?;
        kwargs.set_item("context_scopes", ::pyo3::types::PyList::new(py, context_scopes.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("read_only", ::pyo3::IntoPyObjectExt::into_bound_py_any(read_only, py)?)?;
        kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
        kwargs.set_item("tasks", ::pyo3::types::PyList::new(py, tasks.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("caller_history", match caller_history { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
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
        Err(super::not_a(obj, "Checkpoint | Output"))
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
