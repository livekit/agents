//! What every telemetry producer hands the exporter: finished spans, log records and metric
//! measurements, with the job they belong to. Declarations (`telemetry::spans`) produce the
//! attribute lists; `lk-agents-net` encodes these records as OTLP per destination.

use std::borrow::Cow;

/// The framework's instrumentation scope, and its `service.name` unless the application names one.
pub const FRAMEWORK: &str = "livekit-agents";

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

/// A span's W3C identity: what a child, a link, an effect, a host call or a `traceparent`
/// carries. Ids are never zero; [`SpanContext::valid`] refuses a host's zero ids.
#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanContext {
    pub trace_id: u128,
    pub span_id: u64,
    pub sampled: bool,
}

impl SpanContext {
    /// `None` for an invalid (zero) id.
    pub fn valid(self) -> Option<SpanContext> {
        (self.trace_id != 0 && self.span_id != 0).then_some(self)
    }

    /// A W3C version-00 `traceparent` header; `None` for a malformed one or zero ids, which never
    /// refuses a request.
    pub fn parse_traceparent(value: &str) -> Option<SpanContext> {
        let [version, trace_id, span_id, flags] = value.split('-').collect::<Vec<_>>()[..] else {
            return None;
        };
        let hex = |text: &str, len| {
            (text.len() == len && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
                .then(|| u128::from_str_radix(text, 16).ok())
                .flatten()
        };
        if version != "00" {
            return None;
        }
        SpanContext {
            trace_id: hex(trace_id, 32)?,
            span_id: u64::try_from(hex(span_id, 16)?).ok()?,
            sampled: hex(flags, 2)? & 1 != 0,
        }
        .valid()
    }

    /// The W3C `traceparent` header.
    pub fn traceparent(self) -> String {
        format!("00-{:032x}-{:016x}-{:02x}", self.trace_id, self.span_id, u8::from(self.sampled))
    }
}

/// The job a record belongs to: the key the Cloud destination gates, batches and stamps by.
/// `None` for records produced outside any job (worker lifecycle, application spans between jobs).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobKey(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpanKind {
    #[default]
    Internal,
    Client,
    Server,
    /// Only a host span is a producer or a consumer: no declaration is one.
    Producer,
    Consumer,
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
    pub context: SpanContext,
    pub parent: Option<u64>,
    pub name: Key,
    pub kind: SpanKind,
    pub start_unix_nanos: u64,
    pub end_unix_nanos: u64,
    pub attributes: KeyValues,
    pub events: Vec<EventRecord>,
    pub links: Vec<SpanContext>,
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
    pub span: Option<SpanContext>,
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

#[cfg(test)]
mod tests {
    use super::SpanContext;

    // Catches: a malformed, uppercase, oversized or zero-id `traceparent` accepted, or the header
    // not round-tripping.
    #[test]
    fn traceparent_round_trips_and_refuses_invalid_headers() {
        let valid = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
        let span = SpanContext::parse_traceparent(valid).unwrap();
        assert!(span.sampled);
        assert_eq!(span.traceparent(), valid);
        for invalid in [
            "",
            "00-00000000000000000000000000000000-0123456789abcdef-01",
            "00-0123456789abcdef0123456789abcdef-0000000000000000-01",
            "00-0123456789ABCDEF0123456789abcdef-0123456789abcdef-01",
            "01-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
            "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01-extra",
        ] {
            assert!(SpanContext::parse_traceparent(invalid).is_none(), "{invalid}");
        }
        assert!(SpanContext::parse_traceparent(&format!("{valid}0")).is_none());
    }
}
