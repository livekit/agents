//! Values as the host spells them: JSON, the `NotGiven` bridge, and wall-clock seconds.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};
use serde_json::{Map, Number, Value};

use crate::Extra;


// JSON, not a `HostRef`: Rust branches on `extra` and provider formats inline it. One mapping rule
// serves both directions so they stay inverses; non-JSON raises naming the path, never coerced.

#[allow(dead_code)]
pub fn parse_str<T: std::str::FromStr>(s: &str, path: &str) -> PyResult<T>
where
    T::Err: std::fmt::Display,
{
    s.parse().map_err(|e| PyValueError::new_err(format!("{path}: {e}")))
}

pub fn or_default<'a, 'py, T>(v: &'a Bound<'py, PyAny>) -> PyResult<T>
where
    T: FromPyObject<'a, 'py> + Default,
    T::Error: Into<PyErr>,
{
    if v.is_none() { Ok(T::default()) } else { v.extract().map_err(Into::into) }
}

/// The top level must be a dict, because the field is typed as one.
pub fn extra_from_py(obj: &Bound<'_, PyDict>) -> PyResult<Extra> {
    map_from_py(obj, "extra")
}

pub fn map_from_py(obj: &Bound<'_, PyDict>, path: &str) -> PyResult<Map<String, Value>> {
    let mut out = Map::with_capacity(obj.len());
    for (key, value) in obj.iter() {
        // Not `key.str()`: `json.dumps` coerces non-string keys; this rule is deliberately stricter.
        let key: String = key
            .cast::<PyString>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "{path}: JSON object keys must be str, got {}",
                    type_name(&key)
                ))
            })?
            .extract()?;
        let child = format!("{path}.{key}");
        out.insert(key, value_from_py(&value, &child)?);
    }
    Ok(out)
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "?".to_owned())
}

/// `pub(crate)` for `to_dict`, which converts media under the same rule.
pub fn value_from_py(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    // `bool` before `int`: `True` is an `int`, and an int-first order turns `is_summary` into `1`,
    // which `extra.get("is_summary") is True` reads as false.
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::Number(i.into()));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(Value::Number(u.into()));
        }
        return Err(PyValueError::new_err(format!(
            "{path}: integer is out of range for JSON (must fit in i64 or u64)"
        )));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f = obj.extract::<f64>()?;
        return Number::from_f64(f).map(Value::Number).ok_or_else(|| {
            PyValueError::new_err(format!("{path}: {f} has no JSON representation"))
        });
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.extract()?));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        return Ok(Value::Object(map_from_py(d, path)?));
    }
    if let Ok(items) = obj.cast::<PyList>() {
        let mut out = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            out.push(value_from_py(&item, &format!("{path}[{i}]"))?);
        }
        return Ok(Value::Array(out));
    }
    // `json.dumps` accepts tuples, so name the fix.
    let hint = if obj.is_instance_of::<PyTuple>() { " (use a list)" } else { "" };
    Err(PyTypeError::new_err(format!(
        "{path}: {} is not JSON{hint}. `extra` holds provider payloads that get inlined into the \
         request body, so every value in it has to be JSON-serialisable.",
        type_name(obj)
    )))
}

pub fn extra_to_py<'py>(py: Python<'py>, extra: &Extra) -> PyResult<Bound<'py, PyDict>> {
    map_to_py(py, extra)
}

fn map_to_py<'py>(py: Python<'py>, map: &Map<String, Value>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (key, value) in map {
        out.set_item(key, value_to_py(py, value)?)?;
    }
    Ok(out)
}

pub fn value_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Number(n) => {
            // Mirrors `value_from_py`: ints stay ints and floats stay floats, even `5.0`.
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                n.as_f64()
                    .ok_or_else(|| PyValueError::new_err("unrepresentable JSON number"))?
                    .into_pyobject(py)?
                    .into_any()
            }
        }
        Value::String(s) => s.into_pyobject(py)?.into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_any()
        }
        // The core's one bytes shape inside JSON (`format::BYTES_TAG`, a provider payload's image
        // or signature): a one-key object only, so a user dict is never mistaken for one.
        Value::Object(map) if map.len() == 1 && map.contains_key(crate::llm::provider_format::BYTES_TAG) => {
            use base64::Engine as _;
            let Some(Value::String(b64)) = map.get(crate::llm::provider_format::BYTES_TAG) else {
                return map_to_py(py, map).map(Bound::into_any);
            };
            let raw = base64::engine::general_purpose::STANDARD.decode(b64).map_err(|_| {
                PyValueError::new_err(format!("malformed base64 in a {} leaf", crate::llm::provider_format::BYTES_TAG))
            })?;
            pyo3::types::PyBytes::new(py, &raw).into_any()
        }
        Value::Object(map) => map_to_py(py, map)?.into_any(),
    })
}

// `NotGiven` stays a Python class (pydantic schema, singleton identity, `__bool__`); Rust only
// recognises it, and every `NotGivenOr` parameter extracts through this.

static NOT_GIVEN_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Cached: an `isinstance` check per argument would otherwise be an import per call.
pub(crate) fn not_given_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    NOT_GIVEN_TYPE
        .get_or_try_init(py, || {
            py.import("livekit.agents.types")?
                .getattr("NotGiven")?
                .cast_into::<PyType>()
                .map(|t| t.unbind())
                .map_err(PyErr::from)
        })
        .map(|t| t.bind(py))
}

/// A stamp the core cannot represent is a `ValueError` naming the argument.
pub fn wall_from_py(secs: f64, path: &str) -> PyResult<std::time::SystemTime> {
    crate::wall_from_secs(secs)
        .ok_or_else(|| PyValueError::new_err(format!("{path}={secs} is not a usable wall clock")))
}

/// Pre-epoch is negative.
pub fn wall_to_py(at: std::time::SystemTime) -> f64 {
    crate::epoch_secs(at)
}

/// For arguments where `None` is a value: `Agent(llm=None)` means "no LLM", not "unchanged". A
/// separate type because `impl FromPyObject for Maybe<Option<T>>` would overlap the blanket impl.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MaybeNull<T> {
    #[default]
    NotGiven,
    Null,
    Given(T),
}

impl<T> MaybeNull<T> {
    pub fn into_core(self) -> crate::Maybe<Option<T>> {
        match self {
            MaybeNull::NotGiven => crate::Maybe::NotGiven,
            MaybeNull::Null => crate::Maybe::Given(None),
            MaybeNull::Given(v) => crate::Maybe::Given(Some(v)),
        }
    }

    pub fn try_map<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<MaybeNull<U>, E> {
        Ok(match self {
            MaybeNull::NotGiven => MaybeNull::NotGiven,
            MaybeNull::Null => MaybeNull::Null,
            MaybeNull::Given(v) => MaybeNull::Given(f(v)?),
        })
    }
}

impl<'a, 'py, T> FromPyObject<'a, 'py> for MaybeNull<T>
where
    T: FromPyObject<'a, 'py>,
{
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, PyErr> {
        // `NotGiven` first: it is the absent sentinel, and `None` is a value from here on.
        if ob.is_instance(not_given_type(ob.py())?)? {
            return Ok(MaybeNull::NotGiven);
        }
        if ob.is_none() {
            return Ok(MaybeNull::Null);
        }
        T::extract(ob).map(MaybeNull::Given).map_err(Into::into)
    }
}

/// pyo3's derive replaces a `from_py_with` failure with "failed to extract field .." and keeps the
/// real message only as `__cause__`; these errors end up in log lines, so the cause is lifted.
pub fn extract_generated<'a, 'py, T: FromPyObject<'a, 'py, Error = PyErr>>(obj: &'a Bound<'py, PyAny>) -> PyResult<T> {
    obj.extract::<T>().map_err(|err| {
        let py = obj.py();
        let Some(cause) = err.cause(py) else { return err };
        let flat = PyErr::from_type(cause.get_type(py), format!("{err}: {cause}"));
        flat.set_cause(py, Some(err));
        flat
    })
}

// --------------------------------------------------------------------- arguments by type

/// A type the host passes as a string and the core reads with `FromStr`. An argument that does not
/// parse is a `ValueError` naming the argument, where any other type's failure is pyo3's own.
pub trait HostStr: std::str::FromStr {}

/// Autoref dispatch over [`ArgStr`] and [`ArgPlain`]: `(&Arg::new(v, "name")).get()`.
#[doc(hidden)]
pub struct Arg<'a, 'py, T>(&'a Bound<'py, PyAny>, &'static str, std::marker::PhantomData<T>);

impl<'a, 'py, T> Arg<'a, 'py, T> {
    pub fn new(value: &'a Bound<'py, PyAny>, name: &'static str) -> Self {
        Arg(value, name, std::marker::PhantomData)
    }
}

#[doc(hidden)]
pub trait ArgStr<T> {
    fn get(&self) -> PyResult<T>;
}

impl<T: HostStr> ArgStr<T> for Arg<'_, '_, T>
where
    T::Err: std::fmt::Display,
{
    fn get(&self) -> PyResult<T> {
        let s = self.0.extract::<pyo3::pybacked::PyBackedStr>().map_err(|e| {
            pyo3::impl_::extract_argument::argument_extraction_error(self.0.py(), self.1, e.into())
        })?;
        parse_str(&s, self.1)
    }
}

#[doc(hidden)]
pub trait ArgPlain<T> {
    fn get(&self) -> PyResult<T>;
}

impl<'a, 'py, T> ArgPlain<T> for &Arg<'a, 'py, T>
where
    T: FromPyObject<'a, 'py>,
    T::Error: Into<PyErr>,
{
    fn get(&self) -> PyResult<T> {
        self.0
            .extract::<T>()
            .map_err(|e| pyo3::impl_::extract_argument::argument_extraction_error(self.0.py(), self.1, e.into()))
    }
}

// --------------------------------------------------------------------- records (`#[derive(Host)]`)

/// The key a record's discriminant sits under, its value, and whether it reads a dict.
#[doc(hidden)]
pub trait HostReadTag {
    const KEY: &'static str;
    const TAG: &'static str;
    const ITEMS: bool;
}

#[doc(hidden)]
pub trait HostEmitTag {
    const TAG: Option<&'static str>;
}

/// A `#[binding(fields_only)]` record's fields, read at the level of the record that flattens it.
#[doc(hidden)]
pub trait FlatRecord: Sized {
    /// A flattening reader must still require a key.
    const ALL_DEFAULTED: bool;
    fn read_flat(obj: &Bound<'_, PyAny>, owner: &'static str, items: bool) -> PyResult<Self>;
}

/// Negative or non-finite seconds clamp to zero: a user option must not panic `from_secs_f64`.
pub fn secs_from_py(v: &Bound<'_, PyAny>) -> PyResult<std::time::Duration> {
    let s = v.extract::<f64>()?;
    Ok(if s.is_finite() && s > 0.0 { std::time::Duration::from_secs_f64(s) } else { std::time::Duration::ZERO })
}

/// As pyo3's derive raises it: `TypeError("failed to extract field <owner>.<key>")` caused by the
/// reader's own error.
pub fn field_error(py: Python<'_>, inner: PyErr, owner: &str, key: &str) -> PyErr {
    let err = PyTypeError::new_err(format!("failed to extract field {owner}.{key}"));
    err.set_cause(py, Some(inner));
    err
}

/// A missing attribute raises as itself, a failed read through [`field_error`].
pub fn attr_with<'py, T>(
    obj: &Bound<'py, PyAny>,
    owner: &str,
    key: &str,
    read: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<T> {
    let value = obj.getattr(key)?;
    read(&value).map_err(|e| field_error(obj.py(), e, owner, key))
}

pub fn attr<'py, T>(obj: &Bound<'py, PyAny>, owner: &str, key: &str) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    attr_with(obj, owner, key, |v| v.extract::<T>().map_err(Into::into))
}

pub fn attr_or<'py, T>(
    obj: &Bound<'py, PyAny>,
    owner: &str,
    key: &str,
    read: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
    default: impl FnOnce() -> T,
) -> PyResult<T> {
    match obj.getattr(key) {
        Ok(value) => read(&value).map_err(|e| field_error(obj.py(), e, owner, key)),
        Err(_) => Ok(default()),
    }
}

/// A failure is a `ValueError` naming the message.
pub fn proto_in<M>(v: &Bound<'_, PyAny>, decode: impl FnOnce(&[u8]) -> Result<M, String>) -> PyResult<M> {
    let data = v.cast::<pyo3::types::PyBytes>()?;
    decode(pyo3::types::PyBytesMethods::as_bytes(data)).map_err(|e| {
        let name = std::any::type_name::<M>().rsplit("::").next().unwrap_or("message");
        PyValueError::new_err(format!("not a {name}: {e}"))
    })
}

/// Implemented by a `cell = "arc"` object's expansion.
pub trait ArcClass: Sized {
    fn shared(obj: &Bound<'_, PyAny>) -> PyResult<std::sync::Arc<Self>>;
    fn wrap<'py>(py: Python<'py>, value: std::sync::Arc<Self>) -> PyResult<Bound<'py, PyAny>>;
}

// --------------------------------------------------------------------- core objects as host classes

pub use crate::utils::host_ref::CoreObjectClass;

/// Submitted beside the class: `CoreObjectClass { to_host: core_object_to_host::<T> }`.
pub fn core_object_to_host<'py, T>(
    py: Python<'py>,
    obj: &std::sync::Arc<dyn std::any::Any + Send + Sync>,
) -> Option<PyResult<Bound<'py, PyAny>>>
where
    T: Clone + 'static,
    T: for<'x> IntoPyObject<'x>,
{
    let v = obj.downcast_ref::<T>()?.clone();
    Some(pyo3::IntoPyObjectExt::into_bound_py_any(v, py))
}

/// `None` for a member whose dict carries no tag (autoref dispatch:
/// `(&EmitTagOf::<T>::new()).emit_tag()`).
#[doc(hidden)]
pub struct EmitTagOf<T: ?Sized>(std::marker::PhantomData<T>);

impl<T: ?Sized> EmitTagOf<T> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        EmitTagOf(std::marker::PhantomData)
    }
}

#[doc(hidden)]
pub trait EmitTagDeclared {
    fn emit_tag(&self) -> Option<&'static str>;
}

impl<T: HostEmitTag + ?Sized> EmitTagDeclared for EmitTagOf<T> {
    fn emit_tag(&self) -> Option<&'static str> {
        T::TAG
    }
}

#[doc(hidden)]
pub trait EmitTagNone {
    fn emit_tag(&self) -> Option<&'static str>;
}

impl<T: ?Sized> EmitTagNone for &EmitTagOf<T> {
    fn emit_tag(&self) -> Option<&'static str> {
        None
    }
}

/// A missing key raises as itself, a failed read through [`field_error`].
pub fn item_with<'py, T>(
    obj: &Bound<'py, PyAny>,
    owner: &str,
    key: &str,
    read: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<T> {
    let value = obj.get_item(key)?;
    read(&value).map_err(|e| field_error(obj.py(), e, owner, key))
}

pub fn item<'py, T>(obj: &Bound<'py, PyAny>, owner: &str, key: &str) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    item_with(obj, owner, key, |v| v.extract::<T>().map_err(Into::into))
}

pub fn item_or<'py, T>(
    obj: &Bound<'py, PyAny>,
    owner: &str,
    key: &str,
    read: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
    default: impl FnOnce() -> T,
) -> PyResult<T> {
    match obj.get_item(key) {
        Ok(value) => read(&value).map_err(|e| field_error(obj.py(), e, owner, key)),
        Err(_) => Ok(default()),
    }
}

/// Autoref dispatch: `(&&&ArgRef::<T>::new(v, "name")).hold()` borrows a class off its host
/// object, an `arc` cell's shared value, else extracts the type's own `FromPyObject`.
#[doc(hidden)]
pub struct ArgRef<'a, 'py, T>(&'a Bound<'py, PyAny>, &'static str, std::marker::PhantomData<T>);

impl<'a, 'py, T> ArgRef<'a, 'py, T> {
    pub fn new(value: &'a Bound<'py, PyAny>, name: &'static str) -> Self {
        ArgRef(value, name, std::marker::PhantomData)
    }

    fn note(&self, e: PyErr) -> PyErr {
        pyo3::impl_::extract_argument::argument_extraction_error(self.0.py(), self.1, e)
    }
}

#[doc(hidden)]
pub trait ArgRefClass<'py, T: pyo3::PyClass> {
    fn hold(&self) -> PyResult<pyo3::PyRef<'py, T>>;
}

impl<'a, 'py, T: pyo3::PyClass> ArgRefClass<'py, T> for &&ArgRef<'a, 'py, T> {
    fn hold(&self) -> PyResult<pyo3::PyRef<'py, T>> {
        self.0.extract::<pyo3::PyRef<'py, T>>().map_err(|e| self.note(e.into()))
    }
}

#[doc(hidden)]
pub trait ArgRefShared<T> {
    fn hold(&self) -> PyResult<std::sync::Arc<T>>;
}

impl<T: ArcClass> ArgRefShared<T> for &ArgRef<'_, '_, T> {
    fn hold(&self) -> PyResult<std::sync::Arc<T>> {
        T::shared(self.0).map_err(|e| self.note(e))
    }
}

#[doc(hidden)]
pub struct Owned<T>(pub T);

impl<T> std::ops::Deref for Owned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[doc(hidden)]
pub trait ArgRefOwned<T> {
    fn hold(&self) -> PyResult<Owned<T>>;
}

impl<'a, 'py, T> ArgRefOwned<T> for ArgRef<'a, 'py, T>
where
    T: FromPyObject<'a, 'py>,
    T::Error: Into<PyErr>,
{
    fn hold(&self) -> PyResult<Owned<T>> {
        self.0.extract::<T>().map(Owned).map_err(|e| self.note(e.into()))
    }
}

/// By `&T: IntoPyObject` when there is one, over a clone otherwise: `(&&Lend(v)).lend(py)?`.
#[doc(hidden)]
pub struct Lend<'a, T: ?Sized>(pub &'a T);

#[doc(hidden)]
pub trait LendRef<'py> {
    fn lend(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

impl<'a, 'py, T: ?Sized> LendRef<'py> for &Lend<'a, T>
where
    &'a T: IntoPyObject<'py>,
{
    fn lend(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3::IntoPyObjectExt::into_bound_py_any(self.0, py)
    }
}

#[doc(hidden)]
pub trait LendClone<'py> {
    fn lend(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

impl<'a, 'py, T> LendClone<'py> for Lend<'a, T>
where
    T: Clone + IntoPyObject<'py>,
{
    fn lend(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3::IntoPyObjectExt::into_bound_py_any(self.0.clone(), py)
    }
}
