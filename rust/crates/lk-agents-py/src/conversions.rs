//! The host conversions no declaration expresses: a type declared `handwritten(.., py)` in ABI
//! crosses by its `<type>_from_py` here, which the generated code calls.

use lk_agents_abi::python::MaybeNull;
use lk_agents_abi::telemetry::policy::{MAX_COLLECTION, MAX_DEPTH, TelemetryValue};
use lk_agents_abi::voice::session_options::{
    BackchannelBoundary, Expressive, ExpressiveConfig, HostValue, Instructions, SessionConnOptions,
    TextTransformSet, TurnDetectionKwarg, UnknownLiteral,
};
use lk_agents_abi::{AlignedTranscript, ApiConnectOptions, ImageSource, MediaRef};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyMapping, PySequence, PyString, PyType,
};

/// A scalar applies to both edges.
pub fn backchannel_boundary_from_py(obj: &Bound<'_, PyAny>) -> PyResult<BackchannelBoundary> {
    Ok(match obj.extract::<(f64, f64)>() {
        Ok((start, end)) => BackchannelBoundary::Split(start, end),
        Err(_) => BackchannelBoundary::Both(obj.extract()?),
    })
}

/// `NOT_GIVEN` and `None` are the defaults; a `SessionConnectOptions` is read by attribute.
pub fn session_conn_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<SessionConnOptions> {
    let MaybeNull::Given(options) = obj.extract::<MaybeNull<Bound<'_, PyAny>>>()? else {
        return Ok(SessionConnOptions::default());
    };
    let leg = |name: &str| -> PyResult<Option<ApiConnectOptions>> {
        let conn: Option<ApiConnectOptions> = options.getattr(name)?.extract()?;
        if let Some(conn) = &conn {
            conn.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(conn)
    };
    Ok(SessionConnOptions {
        stt: leg("stt_conn_options")?,
        llm: leg("llm_conn_options")?,
        tts: leg("tts_conn_options")?,
    })
}

/// `NOT_GIVEN` and `None` are "not given", anything else is "given".
pub fn host_value_from_py(obj: &Bound<'_, PyAny>) -> PyResult<HostValue> {
    Ok(match obj.extract::<MaybeNull<Bound<'_, PyAny>>>()? {
        MaybeNull::NotGiven | MaybeNull::Null => HostValue::NotGiven,
        MaybeNull::Given(_) => HostValue::Given,
    })
}

/// `Sequence[TextTransforms]`: an unknown name raises rather than silently not running.
pub fn text_transform_set_from_py(obj: &Bound<'_, PyAny>) -> PyResult<TextTransformSet> {
    TextTransformSet::from_names(&obj.extract::<Vec<String>>()?)
        .map_err(|e| PyValueError::new_err(e.to_string()))
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
    let streaming = obj.hasattr("model")? && obj.hasattr("provider")? && obj.hasattr("stream")?;
    Ok(TurnDetectionKwarg::Detector { streaming })
}

/// `bool | ExpressiveOptions`: `False`/`None` is off, `True` the default options, a dict (even an
/// empty one) on with those options.
pub fn expressive_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Expressive> {
    let Ok(d) = obj.cast::<PyDict>() else {
        return Ok(Expressive(obj.is_truthy()?.then(ExpressiveConfig::default)));
    };
    let get = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        Ok(d.get_item(key)?.filter(|v| !v.is_none()))
    };
    Ok(Expressive(Some(ExpressiveConfig {
        speech_steering: get("speech_steering")?.map(|v| v.extract()).transpose()?,
        tts_instructions_template: get("tts_instructions_template")?
            .map(|v| v.extract::<String>().map(Instructions::plain))
            .transpose()?,
        tts_instructions_append: get("tts_instructions_append")?
            .map(|v| v.extract())
            .transpose()?,
    })))
}

/// A URL as itself; any other image (an `rtc.VideoFrame`) by its side-table key.
pub fn image_source_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ImageSource> {
    Ok(match obj.cast::<PyString>() {
        Ok(url) => ImageSource::Url(url.to_str()?.to_owned()),
        Err(_) => ImageSource::Frame(obj.extract::<MediaRef>()?),
    })
}

/// The host's own frame back out of the side table, not a copy: callers compare by identity.
pub fn image_source_to_py<'py>(py: Python<'py>, value: ImageSource) -> PyResult<Bound<'py, PyAny>> {
    match value {
        ImageSource::Url(url) => Ok(PyString::new(py, &url).into_any()),
        ImageSource::Frame(frame) => frame.into_pyobject(py),
    }
}

/// A host SDK value read once, without walking past the telemetry depth and collection limits.
pub fn telemetry_value_from_py(obj: &Bound<'_, PyAny>) -> PyResult<TelemetryValue> {
    read(obj, 0)
}

fn read(obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<TelemetryValue> {
    if depth > MAX_DEPTH {
        return Ok(TelemetryValue::Unsupported);
    }
    if obj.is_none() {
        return Ok(TelemetryValue::None);
    }
    // `bool` before `int`: `True` is an `int`.
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(TelemetryValue::Bool(b.is_true()));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(obj.extract::<i64>().map_or(TelemetryValue::Unsupported, TelemetryValue::Int));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(TelemetryValue::Float(obj.extract()?));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(TelemetryValue::String(s.to_cow()?.into_owned()));
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(TelemetryValue::Bytes(b.as_bytes().to_vec()));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        if d.len() > MAX_COLLECTION {
            return Ok(TelemetryValue::Unsupported);
        }
        return map(d.items().into_iter().collect(), depth);
    }
    if let Some(entries) = record_fields(obj)? {
        return match entries {
            Some(entries) => map(entries, depth),
            None => Ok(TelemetryValue::Unsupported),
        };
    }
    if let Ok(m) = obj.cast::<PyMapping>() {
        if m.len()? > MAX_COLLECTION {
            return Ok(TelemetryValue::Unsupported);
        }
        return map(m.items()?.into_iter().collect(), depth);
    }
    if let Ok(seq) = obj.cast::<PySequence>() {
        let len = seq.len()?;
        if len > MAX_COLLECTION {
            return Ok(TelemetryValue::Unsupported);
        }
        let items = seq.try_iter()?.map(|item| read(&item?, depth + 1));
        return Ok(TelemetryValue::Array(items.collect::<PyResult<_>>()?));
    }
    Ok(TelemetryValue::Unsupported)
}

/// A mapping with a key that is not a string has no attribute form: it is refused as a whole,
/// charged like any other unsupported value, rather than losing the entry silently.
fn map(entries: Vec<Bound<'_, PyAny>>, depth: usize) -> PyResult<TelemetryValue> {
    if entries.len() > MAX_COLLECTION {
        return Ok(TelemetryValue::Unsupported);
    }
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = entry.extract()?;
        let Ok(key) = key.cast::<PyString>() else {
            return Ok(TelemetryValue::Unsupported);
        };
        out.push((key.to_cow()?.into_owned(), read(&value, depth + 1)?));
    }
    Ok(TelemetryValue::Map(out))
}

/// A pydantic model's fields and declared extra, or a dataclass's public fields, as
/// `(name, value)` tuples (`Some(None)` past the collection limit, counted before any value is
/// read); `None` for any other object. Private attributes (`_cache`, provider state) are never
/// read: they are private and may be large.
#[allow(clippy::type_complexity, reason = "not a record, past the limit, or its fields")]
fn record_fields<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Option<Vec<Bound<'py, PyAny>>>>> {
    if obj.is_instance_of::<PyType>() {
        return Ok(None);
    }
    let py = obj.py();
    if obj.hasattr("__pydantic_fields__")? {
        let mut parts = Vec::new();
        for name in ["__dict__", "__pydantic_extra__"] {
            if let Ok(d) = obj.getattr(name)?.cast_into::<PyDict>() {
                parts.push(d);
            }
        }
        if parts.iter().map(|d| d.len()).sum::<usize>() > MAX_COLLECTION {
            return Ok(Some(None));
        }
        return Ok(Some(Some(parts.iter().flat_map(|d| d.items()).collect())));
    }
    if let Ok(fields) = obj.getattr("__dataclass_fields__") {
        let mut names = Vec::new();
        for name in fields.cast_into::<PyDict>()?.keys() {
            if !name.cast::<PyString>()?.to_cow()?.starts_with('_') {
                names.push(name);
            }
        }
        if names.len() > MAX_COLLECTION {
            return Ok(Some(None));
        }
        let mut entries = Vec::with_capacity(names.len());
        for name in names {
            let value = obj.getattr(name.cast::<PyString>()?)?;
            entries.push((name, value).into_pyobject(py)?.into_any());
        }
        return Ok(Some(Some(entries)));
    }
    Ok(None)
}

pub fn telemetry_value_to_py<'py>(
    py: Python<'py>,
    value: TelemetryValue,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        TelemetryValue::None | TelemetryValue::Unsupported => py.None().into_bound(py),
        TelemetryValue::Bool(b) => PyBool::new(py, b).to_owned().into_any(),
        TelemetryValue::Int(i) => i.into_pyobject(py)?.into_any(),
        TelemetryValue::Float(f) => f.into_pyobject(py)?.into_any(),
        TelemetryValue::String(s) => s.into_pyobject(py)?.into_any(),
        TelemetryValue::Bytes(b) => PyBytes::new(py, &b).into_any(),
        TelemetryValue::Array(items) => pyo3::types::PyList::new(
            py,
            items
                .into_iter()
                .map(|item| telemetry_value_to_py(py, item))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any(),
        TelemetryValue::Map(entries) => {
            let d = PyDict::new(py);
            for (key, value) in entries {
                d.set_item(key, telemetry_value_to_py(py, value)?)?;
            }
            d.into_any()
        }
    })
}

/// A known word, else none: `False` (and anything that is not one of the words) reads as none.
pub fn aligned_transcript_from_py(obj: &Bound<'_, PyAny>) -> PyResult<AlignedTranscript> {
    Ok(obj.extract::<String>().ok().and_then(|s| s.parse().ok()).unwrap_or_default())
}

/// None is the host's `False`, never a string.
pub fn aligned_transcript_to_py<'py>(
    py: Python<'py>,
    value: AlignedTranscript,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        AlignedTranscript::None => pyo3::types::PyBool::new(py, false).to_owned().into_any(),
        other => PyString::new(py, other.as_str()).into_any(),
    })
}
