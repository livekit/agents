//! What every telemetry producer hands the exporter: finished spans, log records and metric
//! measurements, with the job they belong to. Declarations (`telemetry::spans`) produce the
//! attribute lists; `lk-agents-net` encodes these records as OTLP per destination.

use std::borrow::Cow;

/// An attribute key: a declared or semconv constant, or an application's own key.
pub type Key = Cow<'static, str>;

/// Attributes in insertion order. A key appears at most once.
pub type KeyValues = Vec<(Key, Value)>;

/// An OTLP `AnyValue`.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Double(f64),
    String(String),
    Bytes(Vec<u8>),
    Array(Vec<Value>),
    KvList(Vec<(String, Value)>),
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Double(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_owned())
    }
}

/// A W3C trace id. Never zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TraceId(pub u128);

/// A W3C span id. Never zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpanId(pub u64);

/// A span's context: what a child, a link, an effect or a `traceparent` carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanRef {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub sampled: bool,
}

/// The job a record belongs to: the key the Cloud destination gates, batches and stamps by.
/// `None` for records produced outside any job (worker lifecycle, application spans between jobs).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobKey(pub String);

/// A span context crossing an API that cannot carry one (a provider request): the span a
/// provider that records spans of its own (a fallback adapter) opens them under, and the job and
/// conversation they record for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraceContext {
    pub span: Option<SpanRef>,
    pub job: Option<JobKey>,
    pub conversation: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpanKind {
    #[default]
    Internal,
    Client,
    Server,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Status {
    #[default]
    Unset,
    Ok,
    /// The description is content: redacting destinations replace it.
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventRecord {
    pub name: Key,
    pub time_unix_nanos: u64,
    pub attributes: KeyValues,
}

/// One ended span.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanRecord {
    pub context: SpanRef,
    pub parent: Option<SpanId>,
    pub name: Key,
    pub kind: SpanKind,
    pub start_unix_nanos: u64,
    pub end_unix_nanos: u64,
    pub attributes: KeyValues,
    pub events: Vec<EventRecord>,
    pub links: Vec<SpanRef>,
    pub status: Status,
    /// Spans of instrumentation scopes other than the framework's (application spans) carry
    /// their own scope name.
    pub scope: Key,
    pub job: Option<JobKey>,
}

/// An OTLP log record. The framework emits only the session report's records; process logs
/// go to stdout.
#[derive(Debug, Clone, PartialEq)]
pub struct LogRecord {
    pub scope: Key,
    pub body: Value,
    pub time_unix_nanos: u64,
    pub severity_number: i32,
    pub severity_text: Key,
    pub attributes: KeyValues,
    pub span: Option<SpanRef>,
    pub job: Option<JobKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentKind {
    Counter,
    Histogram,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Int(i64),
    Double(f64),
}

impl From<f64> for Number {
    fn from(v: f64) -> Self {
        Number::Double(v)
    }
}

impl From<u64> for Number {
    fn from(v: u64) -> Self {
        Number::Int(i64::try_from(v).unwrap_or(i64::MAX))
    }
}

/// One measurement of a declared instrument. Labels are declared and bounded; the destination
/// adds job identity where it aggregates per job.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub instrument: &'static str,
    pub kind: InstrumentKind,
    pub unit: &'static str,
    /// A histogram's explicit bucket upper bounds; empty for a counter.
    pub bounds: &'static [f64],
    pub value: Number,
    pub labels: KeyValues,
    pub job: Option<JobKey>,
}

/// Everything a producer can hand the exporter.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Span(SpanRecord),
    Log(LogRecord),
    Measurement(Measurement),
}

/// Where producers send records. Recording never blocks or fails: a full queue drops the
/// record and counts it.
pub trait RecordSink: Send + Sync {
    fn record(&self, record: Record);
}
