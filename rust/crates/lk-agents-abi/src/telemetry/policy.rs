//! Destination privacy and admission limits and records; the decisions are
//! `lk_agents_core::telemetry::policy`'s. The limits reach the host as generated constants.

/// Nested SDK values deeper than this are refused.
#[crate::host(export(name = "TELEMETRY_MAX_DEPTH"))]
pub const MAX_DEPTH: usize = 4;

/// SDK mappings and sequences longer than this are refused.
pub const MAX_COLLECTION: usize = 128;

/// Spans with more events or links than this are refused.
#[crate::host(export(name = "TELEMETRY_MAX_EVENTS"))]
pub const MAX_EVENTS: usize = 32;

/// Retained bytes one SDK record may charge.
#[crate::host(export(name = "SDK_RECORD_BYTES"))]
pub const SDK_RECORD_BYTES: usize = 64 * 1024;

/// Records one framework export queue holds.
#[crate::host(export(name = "SDK_PENDING_RECORDS"))]
pub const SDK_PENDING_RECORDS: usize = 64;

#[crate::host(export(name = "SDK_BATCH_ITEMS"))]
pub const SDK_BATCH_ITEMS: usize = 16;

#[crate::host(export(name = "SDK_SCHEDULE_DELAY_MILLIS"))]
pub const SDK_SCHEDULE_DELAY_MILLIS: usize = 1000;

#[crate::host(export(name = "SDK_EXPORT_TIMEOUT_MILLIS"))]
pub const SDK_EXPORT_TIMEOUT_MILLIS: usize = 5000;

/// Framework Cloud span pipelines one process keeps.
#[crate::host(export(name = "SDK_MAX_PROVIDERS"))]
pub const SDK_MAX_PROVIDERS: usize = 4;

#[crate::host(export(name = "SDK_SHUTDOWN_TIMEOUT_SECONDS"))]
pub const SDK_SHUTDOWN_TIMEOUT_SECONDS: f64 = 10.0;

#[crate::host(export(name = "UPLOAD_TIMEOUT_SECONDS"))]
pub const UPLOAD_TIMEOUT_SECONDS: u64 = 900;

/// What a redacted destination sees in place of an exception message or error status.
#[crate::host(export(name = "REDACTED_EXCEPTION"))]
pub const REDACTED_EXCEPTION: &str = "exception details redacted";

/// What a redacted destination sees in place of a log body that may carry content.
#[crate::host(export(name = "REDACTED_LOG_BODY"))]
pub const REDACTED_LOG_BODY: &str = "log body redacted";

pub const REPORT_EVENT_ITEMS: usize = 1024;
pub const REPORT_EVENT_BYTES: usize = 1024 * 1024;
pub const REPORT_EVENT_RECORD_BYTES: usize = 64 * 1024;
pub const REPORT_RETAINED_BYTES: usize = 4 * 1024 * 1024;

/// An OpenTelemetry `AnyValue` as the host SDK holds it, read once from the host object: a
/// mapping's non-string keys are skipped, a host record (a pydantic model, a dataclass) reads as
/// the mapping of its fields.
#[crate::host(handwritten(extract, emit, host = "object"))]
#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    Array(Vec<TelemetryValue>),
    Map(Vec<(String, TelemetryValue)>),
    /// What OTel cannot carry: another host object, an integer beyond 64 bits, or a collection
    /// nested deeper than [`MAX_DEPTH`] or longer than [`MAX_COLLECTION`], which is not walked.
    /// Admission refuses it; a snapshot drops it.
    Unsupported,
}

#[cfg(feature = "python")]
mod host_value {
    use pyo3::prelude::*;
    use pyo3::types::{
        PyBool, PyBytes, PyDict, PyFloat, PyInt, PyMapping, PySequence, PyString, PyType,
    };

    use super::{MAX_COLLECTION, MAX_DEPTH, TelemetryValue};

    impl<'a, 'py> FromPyObject<'a, 'py> for TelemetryValue {
        type Error = PyErr;

        fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            read(&obj, 0)
        }
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
            return Ok(obj
                .extract::<i64>()
                .map_or(TelemetryValue::Unsupported, TelemetryValue::Int));
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
            return map(d.items().into_iter().collect(), depth);
        }
        if let Some(entries) = record_fields(obj)? {
            return map(entries, depth);
        }
        if let Ok(m) = obj.cast::<PyMapping>() {
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

    fn map(entries: Vec<Bound<'_, PyAny>>, depth: usize) -> PyResult<TelemetryValue> {
        if entries.len() > MAX_COLLECTION {
            return Ok(TelemetryValue::Unsupported);
        }
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = entry.extract()?;
            if let Ok(key) = key.cast::<PyString>() {
                out.push((key.to_cow()?.into_owned(), read(&value, depth + 1)?));
            }
        }
        Ok(TelemetryValue::Map(out))
    }

    /// A pydantic model's fields, extra and private attributes, or a dataclass's fields, as
    /// `(name, value)` tuples; `None` for any other object.
    fn record_fields<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        if obj.is_instance_of::<PyType>() {
            return Ok(None);
        }
        let py = obj.py();
        if obj.hasattr("__pydantic_fields__")? {
            let mut entries = Vec::new();
            for name in ["__dict__", "__pydantic_extra__", "__pydantic_private__"] {
                if let Ok(d) = obj.getattr(name)?.cast_into::<PyDict>() {
                    entries.extend(d.items().into_iter());
                }
            }
            return Ok(Some(entries));
        }
        if let Ok(fields) = obj.getattr("__dataclass_fields__") {
            // what the instance retains: its `__dict__`, and slot fields outside it
            let storage = obj.getattr("__dict__").ok().and_then(|d| d.cast_into::<PyDict>().ok());
            let mut entries: Vec<_> =
                storage.as_ref().map(|d| d.items().into_iter().collect()).unwrap_or_default();
            for name in fields.cast_into::<PyDict>()?.keys() {
                if storage.as_ref().is_some_and(|d| d.contains(&name).unwrap_or(false)) {
                    continue;
                }
                let value = obj.getattr(name.cast::<PyString>()?)?;
                entries.push((name, value).into_pyobject(py)?.into_any());
            }
            return Ok(Some(entries));
        }
        Ok(None)
    }

    impl<'py> IntoPyObject<'py> for TelemetryValue {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            Ok(match self {
                TelemetryValue::None | TelemetryValue::Unsupported => py.None().into_bound(py),
                TelemetryValue::Bool(b) => PyBool::new(py, b).to_owned().into_any(),
                TelemetryValue::Int(i) => i.into_pyobject(py)?.into_any(),
                TelemetryValue::Float(f) => f.into_pyobject(py)?.into_any(),
                TelemetryValue::String(s) => s.into_pyobject(py)?.into_any(),
                TelemetryValue::Bytes(b) => PyBytes::new(py, &b).into_any(),
                TelemetryValue::Array(items) => items.into_pyobject(py)?.into_any(),
                TelemetryValue::Map(entries) => {
                    let d = PyDict::new(py);
                    for (key, value) in entries {
                        d.set_item(key, value)?;
                    }
                    d.into_any()
                }
            })
        }
    }
}

#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryRecordKind {
    Span,
    Log,
    Event,
    Link,
    Scope,
    Value,
}

#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryExportAction {
    Export,
    Hold,
    Drop,
    Capacity,
    Disabled,
}

/// One destination's content policy: what its snapshot keeps, drops or replaces.
#[crate::host(class(name = "_core_TelemetryPolicy"))]
pub trait TelemetryPolicy: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(allow_pii: bool, capture_content: bool, project_redaction: bool) -> Self;

    #[host(getter)]
    fn redacted(&self) -> bool;

    /// This destination's copy of `value`: fields it may not see dropped or replaced, at every
    /// depth. `destination=False` copies a value into the framework's own queue, keeping the
    /// destination-private markers.
    #[host(call)]
    fn snapshot(&self, value: TelemetryValue, destination: bool) -> TelemetryValue;

    #[host(call)]
    fn keep_event(&self, name: String) -> bool;

    /// A replacement body, or None to snapshot the original SDK value.
    #[host(call)]
    fn log_body(&self, safe_body: bool, body: Option<String>) -> Option<String>;

    #[host(call)]
    fn status_description(&self, is_error: bool, description: Option<String>) -> Option<String>;
}

/// SDK retention charges and Cloud export admission, which no destination's consent changes.
#[crate::host(functions)]
pub trait TelemetryAdmission {
    /// The destination may export, hold, drop, refuse capacity or stop after recording is disabled.
    #[host(function)]
    fn _core_export_action(
        registered: bool,
        signal_enabled: bool,
        job_open: bool,
        held_count: usize,
        recording_disabled: bool,
    ) -> TelemetryExportAction;

    /// Conservative retained bytes of an SDK value; more than `SDK_RECORD_BYTES` refuses it.
    #[host(function)]
    fn _core_value_bytes(value: TelemetryValue) -> usize;

    /// SDK record overhead plus field charges. Counts are supplied before traversing events/links.
    #[host(function)]
    fn _core_record_bytes(
        kind: TelemetryRecordKind,
        fields: Vec<usize>,
        event_count: usize,
        link_count: usize,
    ) -> usize;
}

/// The framework's metric series. Subject folds decide measurements; this refuses unbounded labels
/// and series before the host SDK creates a new one.
#[crate::host(class(name = "_core_MetricSeries"))]
pub trait MetricSeries: Send + Sized + 'static {
    #[host(new)]
    fn new() -> Self;

    #[host(call)]
    fn admit(&mut self, name: String, attributes: Vec<(String, String)>) -> bool;
}
