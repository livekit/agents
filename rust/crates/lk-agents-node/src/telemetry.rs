//! The napi projection of §1.10's drained records, and nothing else.
//!
//! **The record types moved to [`lk_agents_abi::telemetry`]** -- `SpanRecord`, `LogRecord`,
//! `Severity`, the `HrTime` split, the bounded drop-oldest [`lk_agents_abi::telemetry::TelemetryBuffer`] and the
//! [`lk_agents_abi::telemetry::SpanBuilder`]. They were written here first, deliberately napi-free, so that they could be
//! lifted the moment a second binding needed them; §1.10 requires the shape be "defined once ...
//! and generated into both languages", and two hand-written copies of one dashboard contract is
//! exactly the drift it names. Read that module for the design.
//!
//! What stays here is the FFI encoding, which is genuinely Node's and is not shared with Python:
//!
//! | Core type | This file | Why it cannot be shared |
//! |---|---|---|
//! | `SpanAttr` (a 4-arm enum) | [`SpanAttribute::value`], an `Either4` | `Either4` is `napi::bindgen_prelude`; pyo3 projects the same four arms as a `PyObject` |
//! | `Record` (an enum with payload) | [`TelemetryRecord`], a tagged struct | napi has no enum-with-payload, so `kind` discriminates and exactly one of `span`/`log` is set |
//! | `FrameId` | a bare `i64` field | `u64` would cross as a `BigInt` and infect every signature in the generated `.d.ts` |
//!
//! The struct **names** here are load-bearing and must not be renamed: they are what the napi CLI
//! writes into `native/index.d.ts`, and `node/@livekit-agents/src/native.ts` re-exports them by
//! name. That is why they shadow the core names rather than being called `JsSpanRecord`; the core
//! types are imported under a `Core*` alias instead.

use lk_agents_abi::telemetry::{
    HrTime, LogRecord as CoreLogRecord, Record, SpanAttr, SpanRecord as CoreSpanRecord,
};
use napi::bindgen_prelude::Either4;
use napi_derive::napi;

/// A `(key, value)` pair rather than a map: napi renders `HashMap` as a `Record<string, T>` whose
/// iteration order is JS's, and the attribute order is part of what the dashboard diffs.
#[napi(object)]
pub struct SpanAttribute {
    pub key: String,
    /// OTel JS's `AttributeValue`, spelled out: a nested value here is not a Rust type error but
    /// a *silently dropped attribute* at the exporter, so the TS says exactly what is legal.
    #[napi(ts_type = "string | number | boolean | Array<string>")]
    pub value: Either4<String, f64, bool, Vec<String>>,
}

impl From<lk_agents_abi::telemetry::SpanAttribute> for SpanAttribute {
    fn from(a: lk_agents_abi::telemetry::SpanAttribute) -> Self {
        SpanAttribute {
            key: a.key,
            value: match a.value {
                SpanAttr::Str(s) => Either4::A(s),
                SpanAttr::Num(n) => Either4::B(n),
                SpanAttr::Bool(b) => Either4::C(b),
                SpanAttr::StrArray(v) => Either4::D(v),
            },
        }
    }
}

/// A completed span, with **Rust's** timestamps -- so the duration is the work's, not the drain's.
#[napi(object)]
pub struct SpanRecord {
    pub name: String,
    /// W3C `traceparent` of the parent, so a Rust span nests under a host span and a host span
    /// nests under a Rust one. §8e hands this down on the frame.
    pub parent: Option<String>,
    #[napi(ts_type = "[seconds: number, nanos: number]")]
    pub start_time: HrTime,
    #[napi(ts_type = "[seconds: number, nanos: number]")]
    pub end_time: HrTime,
    pub attributes: Vec<SpanAttribute>,
    /// `record_exception` (`telemetry/utils.py`), already redacted on the Rust side.
    pub error: Option<String>,
}

impl From<CoreSpanRecord> for SpanRecord {
    fn from(r: CoreSpanRecord) -> Self {
        SpanRecord {
            name: r.name,
            parent: r.parent,
            start_time: r.start_time,
            end_time: r.end_time,
            attributes: r.attributes.into_iter().map(Into::into).collect(),
            error: r.error,
        }
    }
}

/// A completed log record. `attributes` is `serde_json::Value` because `LogAttributes` really is
/// recursive on both sides (`AnyValueMap` / `_ExtendedAttributes`) and the session summary is a
/// nested document.
#[napi(object)]
pub struct LogRecord {
    pub logger: String,
    pub body: String,
    #[napi(ts_type = "[seconds: number, nanos: number]")]
    pub timestamp: HrTime,
    /// 1..24. Named `severity_number` and not `severity` to match the OTel field exactly -- the
    /// host writes it straight through.
    pub severity_number: u8,
    pub severity_text: String,
    /// `AnyValueMap`: recursive, unlike a span's. `@opentelemetry/api-logs` 0.221.0's
    /// `LogAttributes` and Python's `_ExtendedAttributes` both admit nested maps and arrays, so
    /// nothing has to be flattened on this half.
    #[napi(ts_type = "Record<string, unknown>")]
    pub attributes: serde_json::Value,
}

impl From<CoreLogRecord> for LogRecord {
    fn from(r: CoreLogRecord) -> Self {
        LogRecord {
            logger: r.logger,
            body: r.body,
            timestamp: r.timestamp,
            severity_number: r.severity_number,
            severity_text: r.severity_text,
            attributes: r.attributes,
        }
    }
}

/// One drained item. A tagged union across the FFI because napi has no enum-with-payload: `kind`
/// discriminates and exactly one of `span`/`log` is present, which is what the generated TS
/// declares as a discriminated union.
#[napi(object)]
pub struct TelemetryRecord {
    /// `"span"` or `"log"`.
    pub kind: String,
    /// §8e's free correlation field. `0` is `FrameId::ROOT`.
    pub frame: i64,
    pub span: Option<SpanRecord>,
    pub log: Option<LogRecord>,
}

impl From<Record> for TelemetryRecord {
    fn from(r: Record) -> Self {
        // `kind()` and `frame()` come from the core so the two hosts cannot spell the
        // discriminant differently; only the *encoding* of the payload is Node's.
        let (kind, frame) = (r.kind().to_owned(), r.frame().raw());
        match r {
            Record::Span { record, .. } => {
                TelemetryRecord { kind, frame, span: Some(record.into()), log: None }
            }
            Record::Log { record, .. } => {
                TelemetryRecord { kind, frame, span: None, log: Some(record.into()) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::FrameId;
    use lk_agents_abi::telemetry::{
        LogRecord as CoreLog, semconv, Severity, SpanBuilder, TelemetryBuffer, attr_bool, attr_num,
        attr_str, attr_strs,
    };

    #[test]
    fn the_four_span_attribute_shapes_each_land_on_their_own_either4_arm() {
        // A silent mis-mapping here would send every number to the exporter as a string, which
        // renders as a working dashboard with unusable panels.
        let buf = TelemetryBuffer::default();
        let mut s = SpanBuilder::start(semconv::span::TTS_NODE, FrameId::from_raw(7), None);
        s.attr("a", attr_str("x"));
        s.attr("b", attr_num(1.5));
        s.attr("c", attr_bool(true));
        s.attr("d", attr_strs(["p", "q"]));
        s.end(&buf);

        let out: Vec<TelemetryRecord> = buf.drain().into_iter().map(Into::into).collect();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, "span");
        assert_eq!(out[0].frame, 7, "§8e: the frame rides every record");
        let attrs = &out[0].span.as_ref().unwrap().attributes;
        assert!(matches!(&attrs[0].value, Either4::A(s) if s == "x"));
        assert!(matches!(attrs[1].value, Either4::B(n) if n == 1.5));
        assert!(matches!(attrs[2].value, Either4::C(true)));
        assert!(matches!(&attrs[3].value, Either4::D(v) if v == &["p".to_owned(), "q".to_owned()]));
    }

    #[test]
    fn a_log_crosses_as_the_other_arm_of_the_tagged_union() {
        let buf = TelemetryBuffer::default();
        buf.push_log(
            FrameId::ROOT,
            CoreLog::now(
                "livekit.agents.tts",
                "boom",
                Severity::Error,
                serde_json::json!({ "error": { "retryable": true } }),
            ),
        );
        let out: Vec<TelemetryRecord> = buf.drain().into_iter().map(Into::into).collect();
        assert_eq!(out[0].kind, "log");
        assert_eq!(out[0].frame, 0, "`0` is `FrameId::ROOT` on the wire");
        assert!(out[0].span.is_none(), "exactly one of span/log is ever set");
        let log = out[0].log.as_ref().unwrap();
        assert_eq!(log.severity_number, 17);
        assert_eq!(log.attributes["error"]["retryable"], true);
    }
}
