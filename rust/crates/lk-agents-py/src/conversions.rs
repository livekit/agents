//! The host conversions no declaration expresses: a type declared `handwritten(.., py)` in ABI
//! crosses by its `<type>_from_py` here, which the generated code calls.

use lk_agents_abi::{ApiConnectOptions, ImageSource, MediaRef};
use lk_agents_abi::python::MaybeNull;
use lk_agents_abi::voice::session_options::{
    BackchannelBoundary, Expressive, ExpressiveConfig, HostValue, Instructions,
    SessionConnOptions, TextTransformSet, TurnDetectionKwarg, UnknownLiteral,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

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
