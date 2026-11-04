//! Python values, buffers, union normalization, JSON, metadata codecs and missing values.

/// Interpreter interface namespaces. Generated inherent methods perform the declared
/// imports and member access; they accept Python objects without asserting their runtime type.
pub mod interfaces {
    #[lk_agents_abi::host(interface(host = "builtins.memoryview", attrs(cast)))]
    pub struct MemoryView;

    #[lk_agents_abi::host(interface(
        protocol = "streaming turn detector",
        protocols(streaming(required(model, provider, stream)))
    ))]
    pub struct TurnDetector;

    #[lk_agents_abi::host(interface(
        module = "livekit.agents.types",
        attrs(not_given(name = "NotGiven"))
    ))]
    pub struct AgentsTypesModule;

    #[lk_agents_abi::host(interface(module = "time", attrs(time, perf_counter)))]
    pub struct TimeModule;

    #[lk_agents_abi::host(interface(module = "numpy", attrs(asarray, int16, float32)))]
    pub struct NumpyModule;
}

use crate::generated::accessors as access;
use lk_agents_abi::Extra;
use lk_agents_abi::Maybe;
use lk_agents_abi::voice::session_options::Expressive;
use lk_agents_abi::voice::session_options::ExpressiveConfig;
use lk_agents_abi::voice::session_options::TurnDetectionKwarg;
use lk_agents_abi::voice::session_options::UnknownLiteral;
use pyo3::buffer::{PyBuffer, PyUntypedBuffer};
use pyo3::exceptions::PyAttributeError;
use pyo3::exceptions::PyKeyError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyBool;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyFloat;
use pyo3::types::PyInt;
use pyo3::types::PyList;
use pyo3::types::PyMemoryView;
use pyo3::types::PyString;
use pyo3::types::PyTuple;
use pyo3::types::PyType;
use serde_json::Map;
use serde_json::Number;
use serde_json::Value;
use std::sync::Arc;

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
    let _ = access::BaseException::add_note(error.value(py))
        .and_then(|add_note| crate::py_calls::call(&add_note, (format!("at {path}"),), None));
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

/// A stamp the core cannot represent is a `ValueError` naming the argument.
pub fn wall_from_py(secs: f64, path: &str) -> PyResult<std::time::SystemTime> {
    lk_agents_abi::wall_from_secs(secs)
        .ok_or_else(|| PyValueError::new_err(format!("{path}={secs} is not a usable wall clock")))
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
        access::InspectModule::getattr_static(&(access::InspectModule::import(py)?))
            .map(Bound::unbind)
    })?;
    Ok(match crate::py_calls::call(lookup.bind(py), (obj, key), None) {
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

/// A parameter whose omission (`NOT_GIVEN`, or left out) differs from every value, `None`
/// included: the value is converted by the declared type's own rule.
pub struct Omittable<'py>(Maybe<Bound<'py, PyAny>>);

impl<'py> Omittable<'py> {
    pub const NOT_GIVEN: Self = Omittable(Maybe::NotGiven);

    pub fn map<T>(
        self,
        convert: impl FnOnce(Bound<'py, PyAny>) -> PyResult<T>,
    ) -> PyResult<Maybe<T>> {
        Ok(match self.0 {
            Maybe::NotGiven => Maybe::NotGiven,
            Maybe::Given(value) => Maybe::Given(convert(value)?),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Omittable<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Omittable(match crate::py_values::is_not_given(&ob)? {
            true => Maybe::NotGiven,
            false => Maybe::Given(ob.to_owned()),
        }))
    }
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

static NOT_GIVEN_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

fn not_given_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    NOT_GIVEN_TYPE
        .get_or_try_init(py, || {
            access::AgentsTypesModule::not_given(&(access::AgentsTypesModule::import(py)?))?
                .cast_into::<PyType>()
                .map(|t| t.unbind())
                .map_err(PyErr::from)
        })
        .map(|t| t.bind(py))
}

pub fn is_not_given(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.is_instance(not_given_type(obj.py())?)
}

/// Translate a provider's performance-counter stamp at the host clock boundary.
pub fn perf_to_wall(value: &Bound<'_, PyAny>, _path: &str) -> PyResult<Option<f64>> {
    let py = value.py();
    let time = access::TimeModule::import(py)?;
    let wall: f64 =
        crate::py_calls::call(&access::TimeModule::time(&time)?, (), None)?.extract()?;
    let perf: f64 =
        crate::py_calls::call(&access::TimeModule::perf_counter(&time)?, (), None)?.extract()?;
    Ok(Some(wall - (perf - value.extract::<f64>()?)))
}

pub fn wall_to_perf(py: Python<'_>, value: f64) -> PyResult<Bound<'_, PyAny>> {
    let time = access::TimeModule::import(py)?;
    let wall: f64 =
        crate::py_calls::call(&access::TimeModule::time(&time)?, (), None)?.extract()?;
    let perf: f64 =
        crate::py_calls::call(&access::TimeModule::perf_counter(&time)?, (), None)?.extract()?;
    pyo3::IntoPyObjectExt::into_bound_py_any(perf - (wall - value), py)
}

pub fn exception_message(value: &Bound<'_, PyAny>, _path: &str) -> PyResult<String> {
    Ok(value.to_string())
}

/// Of a detector object only whether it streams crosses; the check mirrors `runtime_checkable`:
/// `model`, `provider` and `stream`.
pub fn turn_detection_kwarg_from_py(obj: &Bound<'_, PyAny>) -> PyResult<TurnDetectionKwarg> {
    if obj.is_none() {
        return Ok(TurnDetectionKwarg::Null);
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(TurnDetectionKwarg::Mode(
            obj.extract::<String>()?
                .parse()
                .map_err(|e: UnknownLiteral| PyValueError::new_err(e.to_string()))?,
        ));
    }
    let streaming = access::TurnDetector::streaming(obj)?;
    Ok(TurnDetectionKwarg::Detector { streaming })
}

/// `bool | ExpressiveOptions`: `False`/`None` is off, `True` the default options, a dict (even an
/// empty one) on with those options.
pub fn expressive_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Expressive> {
    if obj.cast::<PyDict>().is_err() {
        return Ok(Expressive(obj.is_truthy()?.then(ExpressiveConfig::default)));
    }
    Ok(Expressive(Some(crate::generated::records::expressive::expressive_options_from_py(obj)?)))
}

/// Byte buffers have one-dimensional C layout. PCM additionally accepts signed 16-bit little
/// endian samples (the fixed AudioFrame format); it never reinterprets floats or big-endian PCM.
#[derive(Clone, Copy)]
pub enum BufferFormat {
    Bytes,
    I16Le,
    /// AudioByteStream accepts chunks that split a sample across calls.
    PcmChunk,
}

fn layout_error(path: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{path}: expected a one-dimensional contiguous byte or i16le buffer"
    ))
}

/// Exactly one copy from host storage into the shared allocation. Fan-out clones the Arc.
pub fn snapshot_buffer(
    obj: &Bound<'_, PyAny>,
    format: BufferFormat,
    path: &str,
    max_bytes: Option<usize>,
) -> PyResult<Arc<[u8]>> {
    Ok(copy_bytes(obj.py(), &validated(obj, format, path, max_bytes)?))
}

/// Exactly one copy into an owned vector: exact bytes with the GIL released, any other exporter
/// while attached.
pub fn owned_bytes(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Vec<u8>> {
    if obj.is_exact_instance_of::<PyBytes>() {
        let bytes = obj.extract::<pyo3::pybacked::PyBackedBytes>()?;
        return Ok(obj.py().detach(move || bytes.to_vec()));
    }
    validated(obj, BufferFormat::Bytes, path, None)?.to_vec(obj.py())
}

/// A `#[host(pcm)]` input: bytes, or a 16-bit PCM view (`rtc.AudioFrame.data`) as its bytes, a
/// partial sample included (a chunker reassembles samples across calls).
pub fn pcm_bytes(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Vec<u8>> {
    if obj.is_exact_instance_of::<PyBytes>() {
        return owned_bytes(obj, path);
    }
    validated(obj, BufferFormat::PcmChunk, path, None)?.to_vec(obj.py())
}

/// The exporter's bytes, acquired once and checked against `format`.
fn validated(
    obj: &Bound<'_, PyAny>,
    format: BufferFormat,
    path: &str,
    max_bytes: Option<usize>,
) -> PyResult<PyBuffer<u8>> {
    // The builtin view acquires the exporter once. All subsequent metadata access and casts use
    // that same view, so a custom exporter cannot hand out different buffers between checks.
    let view = PyMemoryView::from(obj).map_err(|e| crate::py_values::at_path(obj.py(), e, path))?;
    let buffer = PyUntypedBuffer::get(view.as_any())?;
    if let Some(limit) = max_bytes
        && buffer.len_bytes() > limit
    {
        return Err(PyValueError::new_err(format!(
            "{path}: buffer has {} bytes; limit is {limit}",
            buffer.len_bytes()
        )));
    }
    if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
        return Err(layout_error(path));
    }
    let pcm = matches!(format, BufferFormat::I16Le | BufferFormat::PcmChunk);
    let valid_format = match buffer.format().to_bytes() {
        b"B" | b"b" | b"c" => true,
        b"<h" => pcm,
        b"h" | b"@h" | b"=h" => pcm && cfg!(target_endian = "little"),
        _ => false,
    };
    if !valid_format {
        return Err(layout_error(path));
    }
    if matches!(format, BufferFormat::I16Le) && buffer.len_bytes() % 2 != 0 {
        return Err(PyValueError::new_err(format!("{path}: i16le data has a partial sample")));
    }
    // Byte casting is a zero-copy buffer operation. Copy the validated representation directly;
    // no aligned i16 Rust references or mutable host references escape the GIL-held snapshot.
    let byte_view = crate::py_calls::call(&access::MemoryView::cast(&view)?, ("B",), None)?;
    PyBuffer::<u8>::get(&byte_view)
}

fn copy_bytes(py: Python<'_>, bytes: &PyBuffer<u8>) -> Arc<[u8]> {
    let len = bytes.len_bytes();
    let mut output = if len >= 1024 * 1024 {
        // Only the destination length crosses this detached section. The acquired exporter
        // remains pinned; its mutable bytes are first read after Python is attached again.
        py.detach(move || {
            let mut output = Arc::<[u8]>::new_uninit_slice(len);
            let destination = Arc::get_mut(&mut output).expect("new allocation is unique");
            // Commit destination pages outside the GIL. The final byte covers an unaligned
            // allocation's trailing page; 4 KiB strides cover supported larger page sizes too.
            // SAFETY: each address is an allocated byte of this unique destination.
            // Volatile writes preserve the page touches when a later memcpy overwrites them.
            unsafe {
                for page in destination.chunks_mut(4096) {
                    page[0].as_mut_ptr().write_volatile(0);
                }
                if let Some(last) = destination.last_mut() {
                    last.as_mut_ptr().write_volatile(0);
                }
            }
            output
        })
    } else {
        Arc::<[u8]>::new_uninit_slice(len)
    };
    let destination = Arc::get_mut(&mut output).expect("new allocation is unique");
    // SAFETY: the retained buffer has been validated as contiguous bytes. Python is attached
    // again before the source is read; no mutable host reference escapes. The unique
    // destination is disjoint from the exporter and has exactly `len` writable bytes.
    // A bulk copy keeps large SDK buffers from holding the GIL in a per-byte Rust loop.
    unsafe {
        if len != 0 {
            std::ptr::copy_nonoverlapping(
                bytes.buf_ptr().cast::<u8>(),
                destination.as_mut_ptr().cast::<u8>(),
                len,
            );
        }
        output.assume_init()
    }
}

/// `Bytes` owns the snapshot or retains an exact bytes object for its entire native lifetime.
/// A read-only memoryview can still have writable aliases and therefore always takes a copy.
pub fn retained_bytes(
    obj: &Bound<'_, PyAny>,
    path: &str,
    max_bytes: Option<usize>,
) -> PyResult<bytes::Bytes> {
    if obj.is_exact_instance_of::<PyBytes>() {
        let size = obj.cast::<PyBytes>()?.as_bytes().len();
        if let Some(limit) = max_bytes
            && size > limit
        {
            return Err(PyValueError::new_err(format!(
                "{path}: buffer has {size} bytes; limit is {limit}"
            )));
        }
        return Ok(bytes::Bytes::from_owner(obj.extract::<pyo3::pybacked::PyBackedBytes>()?));
    }
    Ok(bytes::Bytes::from_owner(snapshot_buffer(obj, BufferFormat::Bytes, path, max_bytes)?))
}

/// PCM ownership is a single snapshot; the converter supplies the field and diagnostic path.
pub fn pcm_snapshot(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<bytes::Bytes> {
    Ok(bytes::Bytes::from_owner(snapshot_buffer(obj, BufferFormat::I16Le, path, None)?))
}

pub fn video_bytes(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<bytes::Bytes> {
    retained_bytes(obj, path, None)
}

/// NumPy layout is a host representation primitive; callers supply the value, never a field key.
pub fn numpy_i16(py: Python<'_>, value: Vec<i16>) -> PyResult<Bound<'_, PyAny>> {
    let numpy = access::NumpyModule::import(py)?;
    crate::py_calls::call(
        &access::NumpyModule::asarray(&numpy)?,
        (value, access::NumpyModule::int16(&numpy)?),
        None,
    )
}

pub fn numpy_f32(py: Python<'_>, value: Vec<f32>) -> PyResult<Bound<'_, PyAny>> {
    let numpy = access::NumpyModule::import(py)?;
    crate::py_calls::call(
        &access::NumpyModule::asarray(&numpy)?,
        (value, access::NumpyModule::float32(&numpy)?),
        None,
    )
}
