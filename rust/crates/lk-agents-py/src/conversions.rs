//! The host conversions no declaration expresses: a type declared `handwritten(.., py)` in ABI
//! crosses by its `<type>_from_py` here, which the generated code calls.

use lk_agents_abi::python::{
    BufferFormat, MaybeNull, attr_with, number_from_py, retained_bytes, snapshot_buffer,
};
use lk_agents_abi::telemetry::policy::{MAX_COLLECTION, MAX_DEPTH, TelemetryValue};
use lk_agents_abi::voice::session_options::{
    BackchannelBoundary, Expressive, ExpressiveConfig, HostValue, Instructions, SessionConnOptions,
    TextTransformSet, TurnDetectionKwarg, UnknownLiteral,
};
use lk_agents_abi::tts::tts::AudioEncoding;
use lk_agents_abi::turn::keyterms::DetectionLlm;
use lk_agents_abi::Maybe;
use lk_agents_abi::errors::CapturedError;
use lk_agents_abi::utils::dates::Date;
use lk_agents_abi::utils::host_ref::StaleHostRef;
use lk_agents_abi::utils::video::VideoFrame;
use lk_agents_abi::voice::events::OverlappingSpeechEvent;
use lk_agents_abi::voice::session_effects::FillerReply;
use lk_agents_abi::voice::room_io::{
    AudioInputConfig, AudioOutputConfig, ParticipantIdentity, ParticipantKind, RoomIoConfig,
    StreamInfo, TextOutputConfig, TrackPublishOptions, TrackSource,
};
use std::time::Duration;
use lk_agents_abi::llm::llm::ProviderToolDeclaration;
use lk_agents_abi::llm::provider_format::tools::ToolFlag;
use lk_agents_abi::{
    AlignedTranscript, ApiConnectOptions, AudioFrame, HostRef, ImageSource, MediaRef, SynthesizedAudio, TimedString,
    ToolChoice, ToolFlags, ToolSpec, VadEvent, VadEventType,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDate, PyDict, PyFloat, PyInt, PyMapping, PySequence, PyString, PyType,
};

/// A scalar applies to both edges.
pub fn backchannel_boundary_from_py(obj: &Bound<'_, PyAny>) -> PyResult<BackchannelBoundary> {
    Ok(match obj.extract::<(f64, f64)>() {
        Ok((start, end)) => BackchannelBoundary::Split(start, end),
        Err(_) => BackchannelBoundary::Both(obj.extract()?),
    })
}

/// An `APIConnectOptions` read by attribute.
pub fn api_connect_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ApiConnectOptions> {
    Ok(ApiConnectOptions {
        max_retry: obj.getattr("max_retry")?.extract()?,
        retry_interval: obj.getattr("retry_interval")?.extract()?,
        pool_timeout: obj.getattr("pool_timeout")?.extract()?,
        connect_timeout: obj.getattr("connect_timeout")?.extract()?,
        write_timeout: obj.getattr("write_timeout")?.extract()?,
        read_timeout: obj.getattr("read_timeout")?.extract()?,
        total_timeout: obj.getattr("total_timeout")?.extract()?,
    })
}

/// The host's frozen `APIConnectOptions`.
pub fn api_connect_options_to_py<'py>(
    py: Python<'py>,
    value: ApiConnectOptions,
) -> PyResult<Bound<'py, PyAny>> {
    static CLASS: pyo3::sync::PyOnceLock<Py<PyType>> = pyo3::sync::PyOnceLock::new();
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_retry", value.max_retry)?;
    kwargs.set_item("retry_interval", value.retry_interval)?;
    kwargs.set_item("pool_timeout", value.pool_timeout)?;
    kwargs.set_item("connect_timeout", value.connect_timeout)?;
    kwargs.set_item("write_timeout", value.write_timeout)?;
    kwargs.set_item("read_timeout", value.read_timeout)?;
    kwargs.set_item("total_timeout", value.total_timeout)?;
    let class = CLASS.import(py, "livekit.agents.types", "APIConnectOptions")?;
    crate::host::construct(class, (), Some(&kwargs))
}

/// `NOT_GIVEN` and `None` are the defaults; a `SessionConnectOptions` is read by attribute.
pub fn session_conn_options_from_py(obj: &Bound<'_, PyAny>) -> PyResult<SessionConnOptions> {
    let MaybeNull::Given(options) = obj.extract::<MaybeNull<Bound<'_, PyAny>>>()? else {
        return Ok(SessionConnOptions::default());
    };
    let leg = |name: &str| -> PyResult<Option<ApiConnectOptions>> {
        let conn = options.getattr(name)?;
        if conn.is_none() {
            return Ok(None);
        }
        let conn = api_connect_options_from_py(&conn)?;
        conn.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Some(conn))
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
    if obj.cast::<PyDict>().is_err() {
        return Ok(Expressive(obj.is_truthy()?.then(ExpressiveConfig::default)));
    }
    Ok(Expressive(Some(crate::generated::records::expressive::expressive_options_from_py(obj)?)))
}

/// `None` declined, a `str` is the text to say, anything else a speech the callable scheduled.
pub fn filler_reply_from_py(obj: &Bound<'_, PyAny>) -> PyResult<FillerReply> {
    if obj.is_none() {
        return Ok(FillerReply::Declined);
    }
    Ok(obj.extract::<String>().map_or(FillerReply::Spoke, FillerReply::Text))
}

/// A model string, resolved through the gateway; anything else is the caller's own `LLM`.
pub fn detection_llm_from_py(obj: &Bound<'_, PyAny>) -> PyResult<DetectionLlm> {
    Ok(match obj.extract::<String>() {
        Ok(model) => DetectionLlm::Model(model),
        Err(_) => DetectionLlm::Instance,
    })
}

/// An instructions string, as its common text.
pub fn instructions_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Instructions> {
    Ok(Instructions::plain(obj.extract::<String>()?))
}

/// A URL as itself; any other image (an `rtc.VideoFrame`) by its side-table key.
pub fn image_source_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ImageSource> {
    Ok(match obj.cast::<PyString>() {
        Ok(url) => ImageSource::Url(url.to_str()?.to_owned()),
        Err(_) => ImageSource::Frame(media_ref_from_py(obj)?),
    })
}

/// The host's own frame back out of the side table, not a copy: callers compare by identity.
pub fn image_source_to_py<'py>(py: Python<'py>, value: ImageSource) -> PyResult<Bound<'py, PyAny>> {
    match value {
        ImageSource::Url(url) => Ok(PyString::new(py, &url).into_any()),
        ImageSource::Frame(frame) => media_ref_to_py(py, frame),
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

/// The detector's verdict as the host's `OverlappingSpeechEvent`, the audio it ran on and its
/// probabilities as numpy arrays.
pub fn overlapping_speech_event_to_py<'py>(
    py: Python<'py>,
    value: OverlappingSpeechEvent,
) -> PyResult<Bound<'py, PyAny>> {
    static CLASS: pyo3::sync::PyOnceLock<Py<PyType>> = pyo3::sync::PyOnceLock::new();
    let numpy = py.import("numpy")?;
    let array = |values: Bound<'py, PyAny>, dtype: &str| -> PyResult<Bound<'py, PyAny>> {
        crate::host::construct(&numpy.getattr("asarray")?, (values, numpy.getattr(dtype)?), None)
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("created_at", value.created_at)?;
    kwargs.set_item("detected_at", value.detected_at)?;
    kwargs.set_item("is_interruption", value.is_interruption)?;
    kwargs.set_item("agent_ended", value.agent_ended)?;
    kwargs.set_item("total_duration", value.total_duration)?;
    kwargs.set_item("prediction_duration", value.prediction_duration)?;
    kwargs.set_item("detection_delay", value.detection_delay)?;
    kwargs.set_item("overlap_started_at", value.overlap_started_at)?;
    if let Some(speech) = value.speech_input {
        kwargs.set_item("speech_input", array(speech.into_pyobject(py)?.into_any(), "int16")?)?;
    }
    if let Some(p) = value.probabilities {
        kwargs.set_item("probabilities", array(p.into_pyobject(py)?.into_any(), "float32")?)?;
    }
    kwargs.set_item("probability", value.probability)?;
    kwargs.set_item("num_requests", value.num_requests)?;
    let class = CLASS.import(py, "livekit.agents.inference.interruption", "OverlappingSpeechEvent")?;
    crate::host::construct(class, (), Some(&kwargs))
}

/// Built as the host's own `TimedString` (a `str` subclass), an absent timing an absent keyword.
pub fn timed_string_to_py<'py>(py: Python<'py>, value: TimedString) -> PyResult<Bound<'py, PyAny>> {
    static CLASS: pyo3::sync::PyOnceLock<Py<PyType>> = pyo3::sync::PyOnceLock::new();
    let kwargs = PyDict::new(py);
    kwargs.set_item("text", value.text)?;
    for (key, v) in [
        ("start_time", value.start_time),
        ("end_time", value.end_time),
        ("confidence", value.confidence),
        ("start_time_offset", value.start_time_offset),
    ] {
        if let Some(v) = v {
            kwargs.set_item(key, v)?;
        }
    }
    if let Some(v) = value.speaker_id {
        kwargs.set_item("speaker_id", v)?;
    }
    crate::host::construct(CLASS.import(py, "livekit.agents.types", "TimedString")?, (), Some(&kwargs))
}

/// The `str` itself and its timings; `NOT_GIVEN`, `None`, anything unreadable and a missing
/// attribute (an older subclass's `speaker_id`) read as absent.
pub fn timed_string_from_py(obj: &Bound<'_, PyAny>) -> PyResult<TimedString> {
    let lenient = |key: &str| obj.getattr(key).ok().and_then(|v| v.extract::<f64>().ok());
    Ok(TimedString {
        text: obj.str()?.extract()?,
        start_time: lenient("start_time"),
        end_time: lenient("end_time"),
        confidence: lenient("confidence"),
        start_time_offset: lenient("start_time_offset"),
        speaker_id: obj.getattr("speaker_id").ok().and_then(|v| v.extract().ok()),
    })
}

/// `started_time` and `timed_transcript` come off the frame's `userdata`; a plain `str` among the
/// words is skipped.
pub fn synthesized_audio_from_py(obj: &Bound<'_, PyAny>) -> PyResult<SynthesizedAudio> {
    let py = obj.py();
    let frame = obj.getattr("frame")?;
    let types = py.import("livekit.agents.types")?;
    let userdata = frame.getattr("userdata")?.cast_into::<PyDict>()?;
    let get = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        Ok(userdata.get_item(types.getattr(key)?)?.filter(|v| !v.is_none()))
    };
    // a `perf_counter()` stamp, as wall-clock seconds
    let started_time = match get("USERDATA_TTS_STARTED_TIME")? {
        Some(v) => {
            let started: f64 = v.extract()?;
            let time = py.import("time")?;
            let wall: f64 = crate::host::read(&time, "time")?.extract()?;
            let perf: f64 = crate::host::read(&time, "perf_counter")?.extract()?;
            Some(wall - (perf - started))
        }
        None => None,
    };
    let timed_transcript = match get("USERDATA_TIMED_TRANSCRIPT")? {
        Some(words) => {
            let class = types.getattr("TimedString")?;
            let mut out = Vec::new();
            for word in words.try_iter()? {
                let word = word?;
                if word.is_instance(&class)? {
                    out.push(timed_string_from_py(&word)?);
                }
            }
            out
        }
        None => Vec::new(),
    };
    Ok(SynthesizedAudio {
        frame: audio_frame_from_py(&frame)?,
        request_id: obj.getattr("request_id")?.extract()?,
        is_final: obj.getattr("is_final")?.extract()?,
        segment_id: obj.getattr("segment_id")?.extract()?,
        delta_text: obj.getattr("delta_text")?.extract()?,
        usage: match obj.getattr("usage")? {
            usage if usage.is_none() => None,
            usage => Some(crate::generated::records::usage::token_usage_from_py(&usage)?),
        },
        characters_count: obj.getattr("characters_count")?.extract()?,
        started_time,
        timed_transcript,
        encoding: AudioEncoding::default(),
    })
}

/// The two userdata fields go back onto the frame, the started time as a `perf_counter()` stamp.
pub fn synthesized_audio_to_py<'py>(
    py: Python<'py>,
    value: SynthesizedAudio,
) -> PyResult<Bound<'py, PyAny>> {
    static CLASS: pyo3::sync::PyOnceLock<Py<PyType>> = pyo3::sync::PyOnceLock::new();
    let types = py.import("livekit.agents.types")?;
    let frame = audio_frame_to_py(py, value.frame)?;
    let userdata = frame.getattr("userdata")?;
    if !value.timed_transcript.is_empty() {
        let words = value
            .timed_transcript
            .into_iter()
            .map(|w| timed_string_to_py(py, w))
            .collect::<PyResult<Vec<_>>>()?;
        userdata.set_item(types.getattr("USERDATA_TIMED_TRANSCRIPT")?, words)?;
    }
    if let Some(wall) = value.started_time {
        let time = py.import("time")?;
        let now: f64 = crate::host::read(&time, "time")?.extract()?;
        let perf: f64 = crate::host::read(&time, "perf_counter")?.extract()?;
        userdata.set_item(types.getattr("USERDATA_TTS_STARTED_TIME")?, perf - (now - wall))?;
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("frame", frame)?;
    kwargs.set_item("request_id", value.request_id)?;
    kwargs.set_item("is_final", value.is_final)?;
    kwargs.set_item("segment_id", value.segment_id)?;
    kwargs.set_item("delta_text", value.delta_text)?;
    if let Some(usage) = value.usage {
        kwargs.set_item("usage", crate::generated::records::usage::token_usage_to_py(py, usage)?)?;
    }
    kwargs.set_item("characters_count", value.characters_count)?;
    crate::host::construct(CLASS.import(py, "livekit.agents.tts", "SynthesizedAudio")?, (), Some(&kwargs))
}

/// `frames` is read only for an end of speech, the one event whose audio is recognized
/// (`inference_done` fires ~30/s).
pub fn vad_event_from_py(obj: &Bound<'_, PyAny>) -> PyResult<VadEvent> {
    let kind = crate::generated::records::vad::vad_event_type_from_py(&obj.getattr("type")?)?;
    let number = |key: &str| obj.getattr(key)?.extract::<f64>();
    let frames = obj.getattr("frames")?;
    Ok(VadEvent {
        kind,
        samples_index: obj.getattr("samples_index")?.extract()?,
        timestamp: number("timestamp")?,
        speech_duration: number("speech_duration")?,
        silence_duration: number("silence_duration")?,
        frames: match kind == VadEventType::EndOfSpeech && !frames.is_none() {
            true => frames.extract::<Vec<Bound<'_, PyAny>>>()?.iter().map(audio_frame_from_py).collect::<PyResult<_>>()?,
            false => Vec::new(),
        },
        probability: number("probability")?,
        inference_duration: number("inference_duration")?,
        speaking: obj.getattr("speaking")?.extract()?,
        raw_accumulated_silence: number("raw_accumulated_silence")?,
        raw_accumulated_speech: number("raw_accumulated_speech")?,
    })
}

/// `RoomOptions._to_core()`'s dict. A malformed `participant_kinds` raises: an empty list would read
/// as "use the defaults" and widen the auto-link accept list.
pub fn room_io_config_from_py(obj: &Bound<'_, PyAny>) -> PyResult<RoomIoConfig> {
    let get = |key: &str| obj.get_item(key);
    let leg = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        let v = get(key)?;
        Ok((!v.is_none()).then_some(v))
    };
    let audio_input = match leg("audio_input")? {
        None => None,
        Some(a) => Some(AudioInputConfig {
            sample_rate: a.get_item("sample_rate")?.extract()?,
            num_channels: a.get_item("num_channels")?.extract()?,
            frame_size_ms: a.get_item("frame_size_ms")?.extract()?,
            noise_cancellation: optional_host_ref(&a.get_item("noise_cancellation")?)?,
            auto_gain_control: a.get_item("auto_gain_control")?.extract()?,
            pre_connect_audio: a.get_item("pre_connect_audio")?.extract()?,
            pre_connect_audio_timeout: Duration::from_secs_f64(
                a.get_item("pre_connect_audio_timeout")?.extract::<f64>()?.max(0.0),
            ),
        }),
    };
    let audio_output = match leg("audio_output")? {
        None => None,
        Some(a) => {
            let source = TrackSource::try_from(a.get_item("track_source")?.extract::<i32>()?)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let default = TrackPublishOptions::default();
            Some(AudioOutputConfig {
                publish: TrackPublishOptions {
                    name: match a.get_item("track_name") {
                        Ok(name) => name.extract()?,
                        Err(_) => default.name.clone(),
                    },
                    source,
                    sample_rate: a.get_item("sample_rate")?.extract()?,
                    num_channels: a.get_item("num_channels")?.extract()?,
                    ..default
                },
            })
        }
    };
    let text_output = match leg("text_output")? {
        None => None,
        Some(t) => Some(TextOutputConfig {
            sync_transcription: t.get_item("sync_transcription")?.extract()?,
            transcription_speed_factor: t.get_item("transcription_speed_factor")?.extract()?,
            json_format: t.get_item("json_format")?.extract()?,
        }),
    };
    Ok(RoomIoConfig {
        participant_identity: get("participant_identity")?
            .extract::<Option<String>>()?
            .map(ParticipantIdentity),
        participant_kinds: get("participant_kinds")?
            .extract::<Vec<i32>>()?
            .into_iter()
            .map(|k| {
                ParticipantKind::try_from(k).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "{e}: it would widen the auto-link accept list to the default rather than \
                         narrow it, so it is refused instead"
                    ))
                })
            })
            .collect::<PyResult<_>>()?,
        close_on_disconnect: get("close_on_disconnect")?.extract()?,
        delete_room_on_close: get("delete_room_on_close")?.extract()?,
        text_input: get("text_input")?.extract()?,
        video_input: get("video_input")?.extract()?,
        audio_input,
        audio_output,
        text_output,
    })
}

/// An SDK `rtc.TextStreamInfo` or `rtc.ByteStreamInfo`: only a byte stream has a `name`, only a
/// text stream `attachments`; `None` attributes read as none.
pub fn stream_info_from_py(obj: &Bound<'_, PyAny>) -> PyResult<StreamInfo> {
    let optional = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        Ok(obj.getattr(key).ok().filter(|v| !v.is_none()))
    };
    Ok(StreamInfo {
        name: optional("name")?.map(|v| v.extract()).transpose()?,
        id: obj.getattr("stream_id")?.extract()?,
        topic: obj.getattr("topic")?.extract()?,
        mime_type: obj.getattr("mime_type")?.extract()?,
        total_length: optional("size")?.map(|v| v.extract()).transpose()?,
        attributes: optional("attributes")?.map(|v| v.extract()).transpose()?.unwrap_or_default(),
        timestamp: obj.getattr("timestamp")?.extract()?,
        attachments: optional("attachments")?.map(|v| v.extract()).transpose()?.unwrap_or_default(),
    })
}

/// The SDK's header again: a `ByteStreamInfo` when it names a file, else a `TextStreamInfo`.
pub fn stream_info_to_py<'py>(py: Python<'py>, value: StreamInfo) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("stream_id", value.id)?;
    kwargs.set_item("mime_type", value.mime_type)?;
    kwargs.set_item("topic", value.topic)?;
    kwargs.set_item("timestamp", value.timestamp)?;
    kwargs.set_item("size", value.total_length)?;
    kwargs.set_item("attributes", value.attributes)?;
    let class = match value.name {
        Some(name) => {
            kwargs.set_item("name", name)?;
            "ByteStreamInfo"
        }
        None => {
            kwargs.set_item("attachments", value.attachments)?;
            "TextStreamInfo"
        }
    };
    crate::host::construct(&py.import("livekit.rtc")?.getattr(class)?, (), Some(&kwargs))
}

/// One offered tool as the host's dict: identity and schema are required (an explicit JSON null
/// stays a value); `description` and `ignore_on_enter` default only when omitted.
pub fn tool_spec_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ToolSpec> {
    use lk_agents_abi::python::{item, item_or, item_with, value_from_py};
    const OWNER: &str = "ToolSpec";
    let provider_tool = item_or(
        obj,
        OWNER,
        "provider_tool",
        |value| {
            if value.is_none() {
                return Ok(None);
            }
            Ok(Some(ProviderToolDeclaration {
                provider: item(value, OWNER, "provider")?,
                payload: item_with(value, OWNER, "payload", |payload| {
                    value_from_py(payload, "ToolSpec.provider_tool.payload")
                })?,
            }))
        },
        || None,
    )?;
    Ok(ToolSpec {
        provider_tool,
        name: item(obj, OWNER, "name")?,
        description: item_or(obj, OWNER, "description", |v| v.extract(), String::new)?,
        json_schema: item_with(obj, OWNER, "json_schema", |v| value_from_py(v, "ToolSpec.json_schema"))?,
        flags: ToolFlags {
            ignore_on_enter: item_or(obj, OWNER, "ignore_on_enter", |v| v.extract(), || false)?,
        },
    })
}

/// A literal (`"auto"`, `"required"`, `"none"`) or the named-tool dict. `None` is not a choice: an
/// optional `tool_choice` reads it as "the provider decides" before it gets here.
pub fn tool_choice_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ToolChoice> {
    if obj.is_none() {
        return Err(PyValueError::new_err("tool_choice"));
    }
    if let Ok(name) = obj.extract::<String>() {
        return ToolChoice::from_literal(&name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown tool_choice {name:?}")));
    }
    let named = obj.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err("tool_choice must be a string or a named-tool dict")
    })?;
    let function = named
        .get_item("function")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("NamedToolChoice.function"))?;
    let name = function
        .get_item("name")
        .map_err(|_| pyo3::exceptions::PyKeyError::new_err("NamedToolChoice.function.name"))?;
    Ok(ToolChoice::Named(name.extract()?))
}

/// A literal string, or the two-level named-tool dict.
pub fn tool_choice_to_py<'py>(py: Python<'py>, value: ToolChoice) -> PyResult<Bound<'py, PyAny>> {
    lk_agents_abi::python::value_to_py(py, &value.to_host_json())
}

/// Unknown bits raise: dropping one could make a cancellable tool uncancellable.
pub fn tool_flag_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ToolFlag> {
    ToolFlag::try_from(obj.extract::<u8>()?).map_err(PyValueError::new_err)
}

/// The flag's bits: the host's `ToolFlag` is an `IntFlag`.
pub fn tool_flag_to_py<'py>(py: Python<'py>, value: ToolFlag) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.bits().into_pyobject(py)?.into_any())
}

/// Validated rather than saturated.
pub fn date_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Date> {
    let part = |name: &str| -> PyResult<i64> { obj.getattr(name)?.extract() };
    Date::new(part("year")?, part("month")?, part("day")?)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn date_to_py<'py>(py: Python<'py>, value: Date) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyDate::new(py, value.year(), value.month() as u8, value.day() as u8)?.into_any())
}

/// `data` is snapshotted once through the buffer protocol. The scalars are read first, so an
/// object that is not an audio frame (a video frame tried as one) fails before its buffer is
/// copied.
pub fn audio_frame_from_py(obj: &Bound<'_, PyAny>) -> PyResult<AudioFrame> {
    const OWNER: &str = "AudioFrameFromPy";
    let sample_rate =
        attr_with(obj, OWNER, "sample_rate", |v| number_from_py(v, "AudioFrame.sample_rate"))?;
    let num_channels =
        attr_with(obj, OWNER, "num_channels", |v| number_from_py(v, "AudioFrame.num_channels"))?;
    let samples_per_channel = attr_with(obj, OWNER, "samples_per_channel", |v| {
        number_from_py(v, "AudioFrame.samples_per_channel")
    })?;
    let data = attr_with(obj, OWNER, "data", |v| {
        snapshot_buffer(v, BufferFormat::I16Le, "AudioFrame.data", None)
    })?;
    Ok(AudioFrame { data, sample_rate, num_channels, samples_per_channel })
}

pub fn audio_frame_to_py<'py>(py: Python<'py>, value: AudioFrame) -> PyResult<Bound<'py, PyAny>> {
    static CLASS: pyo3::sync::PyOnceLock<Py<PyType>> = pyo3::sync::PyOnceLock::new();
    let kwargs = PyDict::new(py);
    kwargs.set_item("data", PyBytes::new(py, &value.data))?;
    kwargs.set_item("sample_rate", value.sample_rate)?;
    kwargs.set_item("num_channels", value.num_channels)?;
    kwargs.set_item("samples_per_channel", value.samples_per_channel)?;
    crate::host::construct(CLASS.import(py, "livekit.rtc", "AudioFrame")?, (), Some(&kwargs))
}

/// `data` held as the host's own `bytes` when the frame's buffer is one, else copied once
/// through the buffer protocol. Read-only views of mutable exporters still copy.
pub fn video_frame_from_py(obj: &Bound<'_, PyAny>) -> PyResult<VideoFrame> {
    const OWNER: &str = "VideoFrameFromPy";
    let width = attr_with(obj, OWNER, "width", |v| number_from_py(v, "VideoFrame.width"))?;
    let height = attr_with(obj, OWNER, "height", |v| number_from_py(v, "VideoFrame.height"))?;
    let buffer_type = attr_with(obj, OWNER, "type", |v| number_from_py(v, "VideoFrame.type"))?;
    let data = attr_with(obj, OWNER, "data", |v| retained_bytes(v, "VideoFrame.data", None))?;
    Ok(VideoFrame { width, height, buffer_type, data })
}

/// `rtc.VideoFrame(width, height, type, data)`.
pub fn video_frame_to_py<'py>(py: Python<'py>, value: VideoFrame) -> PyResult<Bound<'py, PyAny>> {
    static CLASS: pyo3::sync::PyOnceLock<Py<PyType>> = pyo3::sync::PyOnceLock::new();
    let data = PyBytes::new(py, &value.data);
    let args = (value.width, value.height, value.buffer_type, data);
    crate::host::construct(CLASS.import(py, "livekit.rtc", "VideoFrame")?, args, None)
}

/// Interned; the key keeps it alive.
pub fn host_ref_from_py(obj: &Bound<'_, PyAny>) -> PyResult<HostRef> {
    Ok(lk_agents_abi::python::intern_host(obj.clone().unbind()))
}

/// The interned object itself: callers compare by identity. A stale or foreign key raises.
pub fn host_ref_to_py<'py>(py: Python<'py>, value: HostRef) -> PyResult<Bound<'py, PyAny>> {
    resolve_owned(py, value.id(), value.belongs_to(lk_agents_abi::python::media().registry()))
}

pub fn media_ref_from_py(obj: &Bound<'_, PyAny>) -> PyResult<MediaRef> {
    Ok(lk_agents_abi::python::intern_media(obj.clone().unbind()))
}

pub fn media_ref_to_py<'py>(py: Python<'py>, value: MediaRef) -> PyResult<Bound<'py, PyAny>> {
    resolve_owned(py, value.id(), value.belongs_to(lk_agents_abi::python::media().registry()))
}

fn optional_host_ref(obj: &Bound<'_, PyAny>) -> PyResult<Option<HostRef>> {
    if obj.is_none() { Ok(None) } else { host_ref_from_py(obj).map(Some) }
}

/// The host's own exception, released: `None` when another clone already took it.
pub fn captured_error_to_py<'py>(py: Python<'py>, value: CapturedError) -> PyResult<Bound<'py, PyAny>> {
    Ok(lk_agents_abi::python::release(value).map_or_else(|| py.None().into_bound(py), |obj| obj.into_bound(py)))
}

/// The host object `id` names, provided its key was issued by this runtime's table.
fn resolve_owned(py: Python<'_>, id: u64, owned: bool) -> PyResult<Bound<'_, PyAny>> {
    if !owned {
        return Err(lk_agents_abi::python::HostException::to_py_err(&StaleHostRef { id }));
    }
    Ok(lk_agents_abi::python::resolve_key(py, id)?.into_bound(py))
}

/// A parameter whose omission (`NOT_GIVEN`, or left out) differs from every value, `None`
/// included: the value is converted by the declared type's own rule.
pub struct Omittable<'py>(Maybe<Bound<'py, PyAny>>);

impl<'py> Omittable<'py> {
    pub const NOT_GIVEN: Self = Omittable(Maybe::NotGiven);

    pub fn map<T>(self, convert: impl FnOnce(Bound<'py, PyAny>) -> PyResult<T>) -> PyResult<Maybe<T>> {
        Ok(match self.0 {
            Maybe::NotGiven => Maybe::NotGiven,
            Maybe::Given(value) => Maybe::Given(convert(value)?),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Omittable<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Omittable(match lk_agents_abi::python::is_not_given(&ob)? {
            true => Maybe::NotGiven,
            false => Maybe::Given(ob.to_owned()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    // The package is not importable here: a stub frame and event stand in for it.
    const STUBS: &str = r#"
class AudioFrame:
    def __init__(self, n):
        self._d = bytes(n * 2); self.sample_rate = 24000
        self.num_channels = 1; self.samples_per_channel = n
    @property
    def data(self):
        return memoryview(self._d)

class VADEvent:
    def __init__(self, type, frames):
        self.type = type; self.frames = frames; self.samples_index = 0; self.timestamp = 0.0
        self.speech_duration = 0.0; self.silence_duration = 0.0; self.probability = 0.9
        self.inference_duration = 0.0; self.speaking = False
        self.raw_accumulated_silence = 0.0; self.raw_accumulated_speech = 0.0
"#;

    /// One tool spec as `agent_activity.py` builds it, with `patch` applied after.
    fn parse_spec(py: Python<'_>, patch: &str) -> PyResult<lk_agents_abi::ToolSpec> {
        let globals = PyDict::new(py);
        let program = format!(
            "out = {{'name': 'lookup', 'description': 'look something up', \
             'json_schema': {{'type': 'object', 'properties': {{}}}}, 'ignore_on_enter': True}}\n{patch}\n"
        );
        py.run(&std::ffi::CString::new(program).unwrap(), Some(&globals), None).unwrap();
        super::tool_spec_from_py(&globals.get_item("out").unwrap().unwrap())
    }

    // Catches: a key that reaches this reader and stops there. `ignore_on_enter` is asserted `true`
    // because `false` is both the default and what the N184 bug produced.
    #[test]
    fn every_key_of_a_tool_spec_lands_including_the_flag_that_used_to_be_dropped() {
        Python::initialize();
        Python::attach(|py| {
            let spec = parse_spec(py, "").expect("the producers' dict reads");
            assert_eq!((spec.name.as_str(), spec.description.as_str()), ("lookup", "look something up"));
            assert_eq!(spec.json_schema["type"], "object");
            assert!(spec.provider_tool.is_none());
            assert!(spec.flags.ignore_on_enter, "the flag was read as the default rather than the value");
        });
    }

    // Defaults apply only to omission. Explicit JSON null remains a JSON value; fields whose
    // types are not nullable must refuse it.
    #[test]
    fn tool_defaults_apply_to_omission_and_nonnullable_fields_refuse_null() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(parse_spec(py, "out['json_schema'] = None").unwrap().json_schema, serde_json::Value::Null);
            assert!(parse_spec(py, "out['description'] = None").is_err());
            assert!(parse_spec(py, "out['ignore_on_enter'] = None").is_err());
            assert!(parse_spec(py, "del out['json_schema']").is_err());
            assert!(!parse_spec(py, "del out['ignore_on_enter']").unwrap().flags.ignore_on_enter);
            assert_eq!(parse_spec(py, "del out['description']").unwrap().description, "");
        });
    }

    // Catches: copying the audio of every VAD event off the host (`inference_done` fires ~30 a
    // second per stream) when only an end of speech's is recognized.
    #[test]
    fn only_an_end_of_speech_carries_its_audio_off_the_host() {
        Python::initialize();
        Python::attach(|py| {
            let read = |code: &str| {
                let globals = PyDict::new(py);
                let program = std::ffi::CString::new(format!("{STUBS}\nout = {code}\n")).unwrap();
                py.run(&program, Some(&globals), None).unwrap();
                super::vad_event_from_py(&globals.get_item("out").unwrap().unwrap()).unwrap()
            };
            assert!(read("VADEvent('inference_done', [AudioFrame(4)])").frames.is_empty());
            let ev = read("VADEvent('end_of_speech', [AudioFrame(4), AudioFrame(4)])");
            assert_eq!(ev.frames.len(), 2);
        });
    }

    // Catches: a date saturated or wrapped into range instead of refused.
    #[test]
    fn a_date_crosses_both_ways_and_an_impossible_one_is_refused() {
        Python::initialize();
        Python::attach(|py| {
            let leap = lk_agents_abi::utils::dates::Date::new(2024, 2, 29).unwrap();
            assert_eq!(super::date_from_py(&super::date_to_py(py, leap).unwrap()).unwrap(), leap);
            let globals = PyDict::new(py);
            let program = c"from types import SimpleNamespace\nout = SimpleNamespace(year=2024, month=2, day=30)\n";
            py.run(program, Some(&globals), None).unwrap();
            assert!(super::date_from_py(&globals.get_item("out").unwrap().unwrap()).is_err());
        });
    }

    // Catches: reading `data` through `tobytes()` or keeping a view the host can still mutate, a
    // bool taken as frame metadata, or the frame crossing back with its samples aliased.
    #[test]
    fn an_audio_frame_snapshots_its_buffer_both_ways() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            let program = c"
from types import SimpleNamespace
class Samples(bytearray):
    def tobytes(self):
        raise AssertionError('must use buffer protocol')
samples = Samples(b'\\x01\\x00\\x02\\x00')
frame = SimpleNamespace(data=samples, sample_rate=16000, num_channels=1, samples_per_channel=2)
";
            py.run(program, Some(&globals), None).unwrap();
            let get = |name: &str| globals.get_item(name).unwrap().unwrap();
            let frame = super::audio_frame_from_py(&get("frame")).unwrap();
            get("samples").set_item(0, 9).unwrap();
            assert_eq!(&*frame.data, &[1, 0, 2, 0]);
            let host = super::audio_frame_to_py(py, frame.clone()).unwrap();
            assert_eq!(host.getattr("samples_per_channel").unwrap().extract::<u32>().unwrap(), 2);
            let data = host.getattr("data").unwrap();
            if !data.getattr("readonly").unwrap().extract::<bool>().unwrap() {
                data.set_item(0, 9).unwrap();
            }
            assert_eq!(&*frame.data, &[1, 0, 2, 0], "host output must not alias native storage");
            get("frame").setattr("sample_rate", true).unwrap();
            assert!(super::audio_frame_from_py(&get("frame")).is_err(), "metadata rejects bool");
        });
    }

    // Catches: a frame keeping a view of a host buffer the host goes on mutating, or a bool taken
    // as frame metadata.
    #[test]
    fn a_video_frame_snapshots_a_sliced_mutable_buffer() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            let program = c"
from types import SimpleNamespace
owner = bytearray(b'01234567')
frame = SimpleNamespace(data=memoryview(owner)[2:6], width=1, height=1, type=0)
";
            py.run(program, Some(&globals), None).unwrap();
            let get = |name: &str| globals.get_item(name).unwrap().unwrap();
            let frame = super::video_frame_from_py(&get("frame")).unwrap();
            get("owner").set_item(2, 99).unwrap();
            assert_eq!(&*frame.data, b"2345");
            get("frame").setattr("width", true).unwrap();
            assert!(super::video_frame_from_py(&get("frame")).is_err(), "metadata rejects bool");
        });
    }

    // Catches: a stale or foreign key resolving to null, or a foreign table's key resolving the
    // local entry that happens to share its id.
    #[test]
    fn a_host_ref_resolves_only_from_the_table_that_issued_it() {
        use lk_agents_abi::python::{Table, intern_host, media};
        Python::initialize();
        Python::attach(|py| {
            let stale_name = |r: lk_agents_abi::HostRef| {
                let error = super::host_ref_to_py(py, r).unwrap_err();
                error.get_type(py).name().unwrap().to_string()
            };
            assert_eq!(stale_name(lk_agents_abi::HostRef::new(u64::MAX, media().registry())), "StaleHandle");
            let original = PyDict::new(py);
            let local = intern_host(original.clone().into_any().unbind());
            assert!(super::host_ref_to_py(py, local.clone()).unwrap().is(&original));
            let foreign_table = Table::new();
            let foreign_id = foreign_table.insert(PyDict::new(py).into_any().unbind());
            let foreign = lk_agents_abi::HostRef::new(foreign_id, foreign_table.registry());
            assert_ne!(foreign, local);
            assert_eq!(stale_name(foreign), "StaleHandle");
        });
    }
}
