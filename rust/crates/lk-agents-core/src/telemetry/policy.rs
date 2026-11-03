//! Destination content policy, Cloud admission and the session report's retention charges. No
//! host objects cross here.

use lk_agents_abi::telemetry::policy::{
    REDACTED_EXCEPTION, REPORT_EVENT_RECORD_BYTES, TelemetryValue,
};
use lk_agents_abi::telemetry::record::{KeyValues, LogRecord, SpanRecord, Status, Value};
use lk_agents_abi::telemetry::semconv::otel;

/// Does `key` carry the dot-delimited `pii` segment the Cloud collector strips? Not a substring:
/// `lk.piiuser_input` is not marked.
pub fn marks_pii(key: &str) -> bool {
    key.split('.').any(|segment| segment == "pii")
}

pub fn content_allowed(capture_content: bool, redaction: bool) -> bool {
    capture_content && !redaction
}

const REFUSED_BYTES: usize = REPORT_EVENT_RECORD_BYTES + 1;

fn total(base: usize, charges: impl IntoIterator<Item = usize>) -> usize {
    charges.into_iter().fold(base, usize::saturating_add).min(REFUSED_BYTES)
}

fn string_bytes(s: &str) -> usize {
    64usize.saturating_add(s.chars().count().saturating_mul(4))
}

/// What retaining `value` costs the host: Python keeps up to four bytes per character and its
/// integers in 30-bit digits of four bytes. More than [`REPORT_EVENT_RECORD_BYTES`] (or a value
/// the host could not represent) refuses it.
pub fn value_bytes(value: &TelemetryValue) -> usize {
    match value {
        TelemetryValue::None | TelemetryValue::Bool(_) | TelemetryValue::Float(_) => 32,
        TelemetryValue::Int(i) => {
            let bits = (u64::BITS - i.unsigned_abs().leading_zeros()) as usize;
            32 + bits.div_ceil(30) * 4
        }
        TelemetryValue::String(s) => string_bytes(s).min(REFUSED_BYTES),
        TelemetryValue::Bytes(b) => 64usize.saturating_add(b.len()).min(REFUSED_BYTES),
        TelemetryValue::Array(items) => total(64, items.iter().map(value_bytes)),
        TelemetryValue::Map(entries) => total(
            128,
            entries.iter().flat_map(|(key, value)| [string_bytes(key), value_bytes(value)]),
        ),
        TelemetryValue::Unsupported => REFUSED_BYTES,
    }
}

/// What the Cloud destination does with one of a job's records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportAction {
    Export,
    /// Kept until the job's recording decision.
    Hold,
    Drop,
    /// Dropped: the job's hold is full.
    Capacity,
    /// Dropped: the project disabled the job's recording.
    Disabled,
}

/// Spans one undecided job holds in the exporter before its recording decision (as 1.8.10).
pub const HELD_SPANS: usize = 1024;

/// The Cloud destination's admission of a job's record: export, hold until the job decides, or
/// drop. Only spans are held, up to [`HELD_SPANS`].
pub fn cloud_export_action(
    registered: bool,
    signal_enabled: bool,
    job_open: bool,
    held_spans: usize,
) -> ExportAction {
    if registered {
        if signal_enabled { ExportAction::Export } else { ExportAction::Drop }
    } else if !job_open {
        ExportAction::Drop
    } else if held_spans < HELD_SPANS {
        ExportAction::Hold
    } else {
        ExportAction::Capacity
    }
}

/// GenAI content keys. Cloud's ingest does not recognize them, so the Cloud destination removes
/// them itself for a redacted session.
pub const GENAI_CONTENT: [&str; 8] = [
    otel::GEN_AI_INPUT_MESSAGES,
    otel::GEN_AI_OUTPUT_MESSAGES,
    otel::GEN_AI_SYSTEM_INSTRUCTIONS,
    otel::GEN_AI_TOOL_DEFINITIONS,
    otel::GEN_AI_TOOL_CALL_ARGUMENTS,
    otel::GEN_AI_TOOL_CALL_RESULT,
    otel::GEN_AI_TOOL_DESCRIPTION,
    otel::GEN_AI_EVALUATION_EXPLANATION,
];

/// What one exporter destination removes or replaces in its copy of a record
/// (docs/internals/telemetry.md "Destinations").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentPolicy {
    /// `lk.pii.*` (any `pii` dot segment) at any depth.
    pub strip_pii: bool,
    /// GenAI content keys at any depth, `exception.stacktrace`; exception messages and error
    /// status descriptions become [`REDACTED_EXCEPTION`].
    pub redact: bool,
}

impl ContentPolicy {
    /// LiveKit Cloud: `lk.pii.*` is sent as recorded (Cloud's ingest strips it by key); GenAI
    /// content and exception text only for a redacted session.
    pub fn cloud(redaction: bool) -> Self {
        ContentPolicy { strip_pii: false, redact: redaction }
    }

    /// The application's endpoint: everything removable unless the application allowed PII.
    pub fn application(allow_pii: bool) -> Self {
        ContentPolicy { strip_pii: !allow_pii, redact: !allow_pii }
    }

    fn removes(&self, key: &str) -> bool {
        (self.strip_pii && marks_pii(key))
            || (self.redact && (key == otel::EXCEPTION_STACKTRACE || GENAI_CONTENT.contains(&key)))
    }

    fn value(&self, value: &mut Value) {
        match value {
            Value::Array(items) => items.iter_mut().for_each(|item| self.value(item)),
            Value::KvList(entries) => entries.retain_mut(|(key, value)| {
                let keep = !self.removes(key);
                if keep {
                    self.value(value);
                }
                keep
            }),
            _ => {}
        }
    }

    pub fn attributes(&self, attributes: &mut KeyValues) {
        if !self.strip_pii && !self.redact {
            return;
        }
        attributes.retain_mut(|(key, value)| {
            if self.redact && key == otel::EXCEPTION_MESSAGE {
                *value = Value::String(REDACTED_EXCEPTION.into());
                return true;
            }
            let keep = !self.removes(key);
            if keep {
                self.value(value);
            }
            keep
        });
    }

    pub fn span(&self, span: &mut SpanRecord) {
        self.attributes(&mut span.attributes);
        for event in &mut span.events {
            self.attributes(&mut event.attributes);
        }
        if let Status::Error(description) = &mut span.status
            && self.redact
            && !description.is_empty()
        {
            *description = REDACTED_EXCEPTION.into();
        }
    }

    pub fn log(&self, log: &mut LogRecord) {
        self.attributes(&mut log.attributes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pii_marker_is_a_whole_segment() {
        assert!(marks_pii("lk.pii.input"));
        assert!(!marks_pii("lk.pii_input"));
    }

    #[test]
    fn retention_refuses_large_or_unsupported_values() {
        use TelemetryValue as V;
        assert_eq!(value_bytes(&V::String("abc".into())), 76);
        assert_eq!(value_bytes(&V::Map(vec![("abc".into(), V::None)])), 236);
        assert_eq!(value_bytes(&V::Int(i64::MAX)), 44);
        assert_eq!(value_bytes(&V::String("x".repeat(REPORT_EVENT_RECORD_BYTES))), REFUSED_BYTES);
        assert_eq!(value_bytes(&V::Array(vec![V::Unsupported])), REFUSED_BYTES);
    }

    // Catches: Cloud stripping `lk.pii.*` (its ingest does), Cloud redacting an unredacted
    // session, or the application endpoint keeping PII at depth without `allow_pii`.
    #[test]
    fn each_destination_removes_what_its_policy_says() {
        let attributes = || -> KeyValues {
            vec![
                ("lk.pii.user_transcript".into(), "secret".into()),
                ("gen_ai.tool.call.arguments".into(), "secret".into()),
                ("exception.message".into(), "secret".into()),
                ("exception.stacktrace".into(), "secret".into()),
                (
                    "options".into(),
                    Value::Array(vec![Value::KvList(vec![
                        ("lk.pii.keyterms".into(), "secret".into()),
                        ("gen_ai.input.messages".into(), "secret".into()),
                        ("model".into(), "gpt".into()),
                    ])]),
                ),
            ]
        };
        let keys = |policy: ContentPolicy| {
            let mut a = attributes();
            policy.attributes(&mut a);
            let Value::Array(items) = &a[a.len() - 1].1 else { panic!() };
            let Value::KvList(nested) = &items[0] else { panic!() };
            let mut keys: Vec<String> = a.iter().map(|(k, _)| k.to_string()).collect();
            keys.extend(nested.iter().map(|(k, _)| format!("options.{k}")));
            let message = a.iter().find(|(k, _)| k == "exception.message").map(|(_, v)| v.clone());
            (keys.len(), message)
        };
        let secret = Some(Value::String("secret".into()));
        let redacted = Some(Value::String(REDACTED_EXCEPTION.into()));
        assert_eq!(keys(ContentPolicy::cloud(false)), (8, secret.clone()));
        // Keeps both lk.pii keys, drops two GenAI keys and the stacktrace.
        assert_eq!(keys(ContentPolicy::cloud(true)), (5, redacted.clone()));
        assert_eq!(keys(ContentPolicy::application(true)), (8, secret));
        // Only the exception message, options and its model remain.
        assert_eq!(keys(ContentPolicy::application(false)), (3, redacted));
    }

    #[test]
    fn the_cloud_hold_is_bounded_per_job_and_never_exports_an_undecided_job() {
        assert_eq!(cloud_export_action(true, true, false, 0), ExportAction::Export);
        assert_eq!(cloud_export_action(true, false, true, 0), ExportAction::Drop);
        assert_eq!(cloud_export_action(false, true, false, 0), ExportAction::Drop);
        assert_eq!(cloud_export_action(false, false, true, HELD_SPANS - 1), ExportAction::Hold);
        assert_eq!(cloud_export_action(false, false, true, HELD_SPANS), ExportAction::Capacity);
    }
}
