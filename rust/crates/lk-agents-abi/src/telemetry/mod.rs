//! Telemetry records: Rust emits completed records, the host pulls and materialises them in its own
//! OTel SDK. Timestamps cross as `HrTime`: epoch nanos do not fit a JS `number`. Span attributes
//! are flat and log attributes are not; don't unify them, the type rejects a nested span attribute.

use std::sync::Mutex;
pub mod provider;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::FrameId;

/// OTel JS's `HrTime`.
pub type HrTime = (i64, u32);

pub fn hr_time(epoch_nanos: u128) -> HrTime {
    ((epoch_nanos / 1_000_000_000) as i64, (epoch_nanos % 1_000_000_000) as u32)
}

pub fn epoch_nanos(t: HrTime) -> u128 {
    t.0.max(0) as u128 * 1_000_000_000 + t.1 as u128
}

pub fn now_epoch_nanos() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
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

/// Does `key` carry the dot-delimited `pii` segment the Cloud collector strips? Not a substring:
/// `lk.piiuser_input` is not marked.
pub fn marks_pii(key: &str) -> bool {
    key.split('.').any(|segment| segment == "pii")
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

/// With **Rust's** timestamps, so the duration is the work's, not the drain's.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanRecord {
    pub name: String,
    /// W3C `traceparent`, so Rust and host spans nest under each other.
    pub parent: Option<String>,
    pub start_time: HrTime,
    pub end_time: HrTime,
    pub attributes: Vec<SpanAttribute>,
    /// Already redacted on the Rust side.
    pub error: Option<String>,
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
    pub fn now(
        logger: impl Into<String>,
        body: impl Into<String>,
        severity: Severity,
        attributes: serde_json::Value,
    ) -> Self {
        LogRecord {
            logger: logger.into(),
            body: body.into(),
            timestamp: hr_time(now_epoch_nanos()),
            severity_number: severity as u8,
            severity_text: severity.text().to_owned(),
            attributes,
        }
    }
}

/// Shared, so neither host can invent another kind.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Span { frame: FrameId, record: SpanRecord },
    Log { frame: FrameId, record: LogRecord },
}

impl Record {
    pub fn kind(&self) -> &'static str {
        match self {
            Record::Span { .. } => "span",
            Record::Log { .. } => "log",
        }
    }

    pub fn frame(&self) -> FrameId {
        match self {
            Record::Span { frame, .. } | Record::Log { frame, .. } => *frame,
        }
    }
}

/// Bounded, drop-oldest, `dropped` counting the loss: a host that stops draining must not grow it.
pub struct TelemetryBuffer {
    inner: Mutex<Buffer>,
    capacity: usize,
}

struct Buffer {
    records: Vec<Record>,
    dropped: u64,
}

impl Default for TelemetryBuffer {
    fn default() -> Self {
        TelemetryBuffer::new(8192)
    }
}

impl TelemetryBuffer {
    pub fn new(capacity: usize) -> Self {
        TelemetryBuffer { inner: Mutex::new(Buffer { records: Vec::new(), dropped: 0 }), capacity }
    }

    fn push(&self, record: Record) {
        let mut b = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if b.records.len() >= self.capacity {
            b.records.remove(0);
            b.dropped += 1;
        }
        b.records.push(record);
    }

    pub fn push_span(&self, frame: FrameId, record: SpanRecord) {
        self.push(Record::Span { frame, record });
    }

    pub fn push_log(&self, frame: FrameId, record: LogRecord) {
        self.push(Record::Log { frame, record });
    }

    /// Allocation-free when empty (polled on a timer).
    pub fn drain(&self) -> Vec<Record> {
        let mut b = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if b.records.is_empty() {
            return Vec::new();
        }
        std::mem::take(&mut b.records)
    }

    pub fn dropped(&self) -> u64 {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).dropped
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Start is wall-clock, the duration monotonic, so an NTP step cannot make it negative. Reads the
/// OS clock, not a [`Clock`](crate::Clock): a virtual date would reach the real dashboard.
pub struct SpanBuilder {
    name: String,
    parent: Option<String>,
    frame: FrameId,
    start_epoch: u128,
    start_instant: Instant,
    attributes: Vec<SpanAttribute>,
    error: Option<String>,
}

impl SpanBuilder {
    pub fn start(name: impl Into<String>, frame: FrameId, parent: Option<String>) -> Self {
        SpanBuilder {
            name: name.into(),
            parent,
            frame,
            start_epoch: now_epoch_nanos(),
            start_instant: Instant::now(),
            attributes: Vec::new(),
            error: None,
        }
    }

    /// Use [`pii_attr`](Self::pii_attr) for user content.
    pub fn attr(&mut self, key: impl Into<String>, value: SpanAttr) -> &mut Self {
        self.attributes.push(SpanAttribute { key: key.into(), value });
        self
    }

    /// The key must pass [`marks_pii`]. Debug-panics on an unmarked key and drops it in release.
    pub fn pii_attr(&mut self, key: impl Into<String>, value: SpanAttr) -> &mut Self {
        let key = key.into();
        debug_assert!(
            marks_pii(&key),
            "attribute `{key}` carries user content but has no dot-delimited `pii` segment; the \
             Cloud collector strips `.pii.` and nothing else"
        );
        if marks_pii(&key) {
            self.attributes.push(SpanAttribute { key, value });
        }
        self
    }

    pub fn error(&mut self, message: impl Into<String>) -> &mut Self {
        self.error = Some(message.into());
        self
    }

    pub fn end(self, buffer: &TelemetryBuffer) {
        let end_epoch = self.start_epoch + self.start_instant.elapsed().as_nanos();
        buffer.push_span(
            self.frame,
            SpanRecord {
                name: self.name,
                parent: self.parent,
                start_time: hr_time(self.start_epoch),
                end_time: hr_time(end_epoch),
                attributes: self.attributes,
                error: self.error,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hr_time_splits_exactly_at_the_boundary_that_a_js_number_cannot_hold() {
        // No exact JS `number` for this value, hence the pair.
        let ns: u128 = 1_756_300_000_123_456_789;
        assert_eq!(hr_time(ns), (1_756_300_000, 123_456_789));
        assert!(ns > 9_007_199_254_740_991, "the premise of this test");
        assert_eq!(epoch_nanos(hr_time(ns)), ns, "Python reassembles the pair losslessly");
        assert_eq!(hr_time(999_999_999), (0, 999_999_999));
        assert_eq!(hr_time(1_000_000_000), (1, 0));
    }





    // Catches: `now_epoch_nanos` answering a constant, or `SpanBuilder::end` combining start and
    // elapsed by anything but addition.
    #[test]
    fn a_span_lands_on_the_real_timeline_and_lasts_as_long_as_the_work_did() {
        let buf = TelemetryBuffer::default();
        let frame = FrameId::from_raw(42);
        let mut s = SpanBuilder::start("llm_node", frame, Some("00-a-b-01".to_owned()));
        s.attr("lk.tts.label", attr_str("fake"));
        std::thread::sleep(std::time::Duration::from_millis(2));
        s.end(&buf);

        let drained = buf.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].frame(), frame, "§8e: the frame rides every record");
        let Record::Span { record, .. } = &drained[0] else { panic!("expected a span") };
        assert_eq!(record.parent.as_deref(), Some("00-a-b-01"));
        assert_eq!(record.attributes.len(), 1);
        // 2024-01-01 .. 2100-01-01.
        for (label, (secs, nanos)) in [("start", record.start_time), ("end", record.end_time)] {
            assert!(
                (1_704_067_200..4_102_444_800).contains(&secs),
                "{label} is {secs}s, not a plausible wall clock"
            );
            assert!(nanos < 1_000_000_000, "{label} nanos must stay inside the second");
        }
        let span_ns = epoch_nanos(record.end_time) - epoch_nanos(record.start_time);
        assert!(
            (2_000_000..5_000_000_000).contains(&span_ns),
            "a 2 ms sleep must read as a few ms, not {span_ns} ns"
        );
    }

    #[test]
    fn the_buffer_drops_oldest_rather_than_growing_without_bound() {
        let buf = TelemetryBuffer::new(2);
        for i in 0..5 {
            SpanBuilder::start(format!("s{i}"), FrameId::from_raw(i), None).end(&buf);
        }
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.dropped(), 3, "loss must be counted, never silent");
        let drained = buf.drain();
        match &drained[0] {
            Record::Span { record, .. } => assert_eq!(record.name, "s3", "the newest window wins"),
            _ => panic!(),
        }
    }

    #[test]
    fn a_log_record_keeps_its_attributes_nested_and_in_declaration_order() {
        let buf = TelemetryBuffer::default();
        buf.push_log(
            FrameId::ROOT,
            LogRecord::now(
                "livekit.agents.tts",
                "boom",
                Severity::Error,
                serde_json::json!({ "error": { "retryable": true, "statusCode": 503 } }),
            ),
        );
        let drained = buf.drain();
        assert_eq!(drained[0].kind(), "log");
        assert_eq!(drained[0].frame(), FrameId::ROOT);
        match &drained[0] {
            Record::Log { record, .. } => {
                assert_eq!(record.severity_number, 17);
                assert_eq!(record.severity_text, "ERROR");
                // Nested on purpose, key order preserved.
                assert_eq!(record.attributes["error"]["statusCode"], 503);
                let keys: Vec<_> =
                    record.attributes["error"].as_object().unwrap().keys().cloned().collect();
                assert_eq!(keys, ["retryable", "statusCode"], "preserve_order, not BTreeMap");
            }
            _ => panic!("expected a log"),
        }
    }

    #[test]
    fn the_pii_marker_is_a_dot_delimited_segment_not_a_substring() {
        assert!(marks_pii("lk.pii.user_input"));
        assert!(marks_pii("pii.transcript"));
        assert!(marks_pii("lk.chat.pii"));
        // The three near-misses the Cloud collector does NOT strip.
        assert!(!marks_pii("lk.piiuser_input"));
        assert!(!marks_pii("lk.pii_user_input"));
        assert!(!marks_pii("lk.user_input"));
    }


    #[test]
    #[should_panic(expected = "no dot-delimited `pii` segment")]
    fn an_unmarked_sensitive_attribute_fails_a_test_rather_than_a_code_review() {
        let mut s = SpanBuilder::start("llm_node", FrameId::ROOT, None);
        s.pii_attr("lk.user_input", attr_str("my card number is"));
    }
}

pub mod attributes;
pub mod gen_ai;
pub mod report;
