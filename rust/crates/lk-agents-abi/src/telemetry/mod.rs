//! Telemetry records: Rust emits completed records, the host pulls and materialises them in its own
//! OTel SDK. Timestamps cross as `HrTime`: epoch nanos do not fit a JS `number`. Span attributes
//! are flat and log attributes are not; don't unify them, the type rejects a nested span attribute.

pub mod policy;
pub mod provider;
use std::time::{SystemTime, UNIX_EPOCH};

/// OTel JS's `HrTime`.
pub type HrTime = (i64, u32);

pub fn hr_time(epoch_nanos: u128) -> HrTime {
    ((epoch_nanos / 1_000_000_000) as i64, (epoch_nanos % 1_000_000_000) as u32)
}

pub fn epoch_nanos(t: HrTime) -> u128 {
    t.0.max(0) as u128 * 1_000_000_000 + t.1 as u128
}

/// A wall-clock stamp; pre-epoch reads as the epoch.
pub fn hr_wall(t: SystemTime) -> HrTime {
    hr_time(t.duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0))
}

/// Identical in both SDKs (1..24).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Severity {
    Trace = 1,
    Debug = 5,
    Info = 9,
    Warn = 13,
    Error = 17,
    Fatal = 21,
}

impl Severity {
    pub fn text(self) -> &'static str {
        match self {
            Severity::Trace => "TRACE",
            Severity::Debug => "DEBUG",
            Severity::Info => "INFO",
            Severity::Warn => "WARN",
            Severity::Error => "ERROR",
            Severity::Fatal => "FATAL",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpanAttr {
    Str(String),
    Num(f64),
    Bool(bool),
    StrArray(Vec<String>),
}

pub fn attr_str(s: impl Into<String>) -> SpanAttr {
    SpanAttr::Str(s.into())
}
pub fn attr_num(n: f64) -> SpanAttr {
    SpanAttr::Num(n)
}
pub fn attr_bool(b: bool) -> SpanAttr {
    SpanAttr::Bool(b)
}
pub fn attr_strs(v: impl IntoIterator<Item = impl Into<String>>) -> SpanAttr {
    SpanAttr::StrArray(v.into_iter().map(Into::into).collect())
}

/// Every string is a dashboard wire contract. None carries user content.
pub mod attrs {
    pub const SPEECH_ID: &str = "lk.speech_id";
    /// Provider correlation ids, for joining against provider logs.
    pub const PROVIDER_REQUEST_IDS: &str = "lk.provider_request_ids";
    pub const TTS_LABEL: &str = "lk.tts.label";
    pub const TTS_STREAMING: &str = "lk.tts.streaming";

    pub const IS_INTERRUPTION: &str = "lk.is_interruption";
    pub const INTERRUPTION_PROBABILITY: &str = "lk.interruption.probability";
    pub const INTERRUPTION_TOTAL_DURATION: &str = "lk.interruption.total_duration";
    pub const INTERRUPTION_PREDICTION_DURATION: &str = "lk.interruption.prediction_duration";
    pub const INTERRUPTION_DETECTION_DELAY: &str = "lk.interruption.detection_delay";

    /// Not declared in `trace_types.py`: adding them would fail its byte-for-byte comparison with
    /// the reference.
    pub mod node {
        pub const METRIC: &str = "lk.metric";
        /// `root`, `live`, `stale` or `unknown`.
        pub const FRAME_RESOLUTION: &str = "lk.frame.resolution";
        /// The TTS reported no duplex path, so the adapter was selected. Not a failure.
        pub const TTS_NO_DUPLEX_PATH: &str = "lk.tts.no_duplex_path";
        pub const TTS_FRAMES: &str = "lk.tts.frames";
        pub const TTS_BYTES: &str = "lk.tts.bytes";
        pub const ERROR_RETRYABLE: &str = "lk.error.retryable";
    }
}

/// `(key, value)` pairs, not a map: attribute order is part of what the dashboard diffs.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanAttribute {
    pub key: String,
    pub value: SpanAttr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogRecord {
    pub logger: String,
    pub body: String,
    pub timestamp: HrTime,
    pub severity_number: u8,
    pub severity_text: String,
    /// Recursive, unlike a span's.
    pub attributes: serde_json::Value,
}

impl LogRecord {
    pub fn at(
        timestamp: HrTime,
        logger: impl Into<String>,
        body: impl Into<String>,
        severity: Severity,
        attributes: serde_json::Value,
    ) -> Self {
        LogRecord {
            logger: logger.into(),
            body: body.into(),
            timestamp,
            severity_number: severity as u8,
            severity_text: severity.text().to_owned(),
            attributes,
        }
    }
}

pub mod attributes;
pub mod report;

pub mod loop_monitor;
pub mod observation;
pub mod transport;
pub mod upload;
