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
