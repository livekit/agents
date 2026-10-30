//! Values as the host spells them: JSON, the `NotGiven` bridge, and wall-clock seconds.

use pyo3::exceptions::{PyAttributeError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};
use serde_json::{Map, Number, Value};

use lk_agents_abi::Extra;

/// A `#[host(string)]` or `#[host(wire_enum)]` value from the host's `str`; one that does not parse
/// is a `ValueError`.
pub fn str_from_py<T: std::str::FromStr>(obj: &Bound<'_, PyAny>) -> PyResult<T>
where
    T::Err: std::fmt::Display,
{
    let s = obj.extract::<pyo3::pybacked::PyBackedStr>()?;
    s.parse().map_err(|e: T::Err| PyValueError::new_err(e.to_string()))
}

/// A `#[host(string)]` or `#[host(wire_enum)]` value as the host's `str`.
pub fn str_to_py<'py>(py: Python<'py>, value: &impl std::fmt::Display) -> Bound<'py, PyAny> {
    PyString::new(py, &value.to_string()).into_any()
}

// JSON, not a `HostRef`: Rust branches on `extra` and provider formats inline it. One mapping rule
// serves both directions so they stay inverses; non-JSON raises naming the path, never coerced.

/// Attach a validation path without replacing an exception, its subclass, or its cause. Python
/// 3.11 exception notes appear in tracebacks and pytest's match text.
pub fn at_path(py: Python<'_>, error: PyErr, path: &str) -> PyErr {
    let _ = error
        .value(py)
        .getattr("add_note")
        .and_then(|add_note| crate::host::call(&add_note, (format!("at {path}"),), None));
    error
}

pub fn extract_at<'a, 'py, T>(obj: &'a Bound<'py, PyAny>, path: &str) -> PyResult<T>
where
    T: FromPyObject<'a, 'py>,
    T::Error: Into<PyErr>,
{
    obj.extract::<T>().map_err(|e| at_path(obj.py(), e.into(), path))
}

/// Numeric declarations do not accept Python's bool-as-int inheritance. Integer widths and
/// signedness are then checked by the destination's own FromPyObject implementation.
pub fn number_from_py<'a, 'py, T>(obj: &'a Bound<'py, PyAny>, path: &str) -> PyResult<T>
where
    T: FromPyObject<'a, 'py>,
    T::Error: Into<PyErr>,
{
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!("{path}: expected a number, got bool")));
    }
    extract_at(obj, path)
}

/// IEEE binary32 permits explicit NaN/infinity; a finite binary64 input must fit before rounding.
pub fn f32_from_py(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<f32> {
    let value = number_from_py::<f64>(obj, path)?;
    let narrowed = value as f32;
    if value.is_finite() && !narrowed.is_finite() {
        return Err(PyValueError::new_err(format!("{path}: {value} is out of range for f32")));
    }
    Ok(narrowed)
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
        Value::Object(map)
            if map.len() == 1
                && map.contains_key(lk_agents_abi::llm::provider_format::BYTES_TAG) =>
        {
            use base64::Engine as _;
            let Some(Value::String(b64)) = map.get(lk_agents_abi::llm::provider_format::BYTES_TAG)
            else {
                return map_to_py(py, map).map(Bound::into_any);
            };
            let raw = base64::engine::general_purpose::STANDARD.decode(b64).map_err(|_| {
                PyValueError::new_err(format!(
                    "malformed base64 in a {} leaf",
                    lk_agents_abi::llm::provider_format::BYTES_TAG
                ))
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

pub fn is_not_given(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.is_instance(not_given_type(obj.py())?)
}

/// A stamp the core cannot represent is a `ValueError` naming the argument.
pub fn wall_from_py(secs: f64, path: &str) -> PyResult<std::time::SystemTime> {
    lk_agents_abi::wall_from_secs(secs)
        .ok_or_else(|| PyValueError::new_err(format!("{path}={secs} is not a usable wall clock")))
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
    pub fn into_core(self) -> lk_agents_abi::Maybe<Option<T>> {
        match self {
            MaybeNull::NotGiven => lk_agents_abi::Maybe::NotGiven,
            MaybeNull::Null => lk_agents_abi::Maybe::Given(None),
            MaybeNull::Given(v) => lk_agents_abi::Maybe::Given(Some(v)),
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
        Ok(if is_not_given(&ob)? {
            MaybeNull::NotGiven
        } else if ob.is_none() {
            MaybeNull::Null
        } else {
            MaybeNull::Given(T::extract(ob).map_err(Into::into)?)
        })
    }
}

/// Duration seconds, with no clamping and no panic for unsupported values.
pub fn secs_from_py(v: &Bound<'_, PyAny>) -> PyResult<std::time::Duration> {
    secs_from_py_at(v, "duration")
}

pub fn secs_from_py_at(v: &Bound<'_, PyAny>, path: &str) -> PyResult<std::time::Duration> {
    duration_from_secs(number_from_py(v, path)?, path)
}

pub fn duration_from_secs(seconds: f64, path: &str) -> PyResult<std::time::Duration> {
    std::time::Duration::try_from_secs_f64(seconds)
        .map_err(|e| PyValueError::new_err(format!("{path}={seconds}: {e}")))
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
        Err(e) if attribute_missing(obj, key, &e)? => Ok(default()),
        Err(e) => Err(e),
    }
}

fn attribute_missing(obj: &Bound<'_, PyAny>, key: &str, error: &PyErr) -> PyResult<bool> {
    let py = obj.py();
    if !error.is_instance_of::<PyAttributeError>(py) {
        return Ok(false);
    }
    // AttributeError from a descriptor is still a failed lookup, not omission. Static lookup
    // checks whether the attribute is declared without invoking the descriptor a second time.
    static GETATTR_STATIC: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let lookup = GETATTR_STATIC.get_or_try_init(py, || {
        py.import("inspect")?.getattr("getattr_static").map(Bound::unbind)
    })?;
    Ok(match crate::host::call(lookup.bind(py), (obj, key), None) {
        Ok(_) => false,
        Err(e) => e.is_instance_of::<PyAttributeError>(py),
    })
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
        Err(e) if item_missing(obj, key, &e) => Ok(default()),
        Err(e) => Err(e),
    }
}

fn item_missing(obj: &Bound<'_, PyAny>, key: &str, error: &PyErr) -> bool {
    if !error.is_instance_of::<PyKeyError>(obj.py()) {
        return false;
    }
    // A dictionary override can raise KeyError for an entry that exists. Inspect its snapshot
    // without invoking that getter again. Other mappings must declare their own missing policy;
    // an arbitrary getter failure alone does not establish absence.
    obj.cast::<PyDict>().is_ok_and(|dict| matches!(dict.get_item(key), Ok(None)))
}
