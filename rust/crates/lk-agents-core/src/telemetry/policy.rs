//! Shared content filtering and conservative SDK retention accounting. No host objects cross here.

use lk_agents_abi::telemetry::policy::{
    MAX_COLLECTION, MAX_DEPTH, MAX_EVENTS, REDACTED_EXCEPTION, REDACTED_LOG_BODY,
    SDK_PENDING_RECORDS as MAX_PENDING, SDK_RECORD_BYTES as MAX_RECORD_BYTES, TelemetryAdmission,
    TelemetryExportAction, TelemetryPolicy, TelemetryRecordKind, TelemetryValue,
};

const REFUSED_BYTES: usize = MAX_RECORD_BYTES + 1;

/// Does `key` carry the dot-delimited `pii` segment the Cloud collector strips? Not a substring:
/// `lk.piiuser_input` is not marked.
pub fn marks_pii(key: &str) -> bool {
    key.split('.').any(|segment| segment == "pii")
}

/// The session report's own log bodies, which carry no user content.
pub fn report_body_is_safe(body: &str) -> bool {
    matches!(
        body,
        "session report" | "session event" | "chat item" | "evaluation" | "tag" | "outcome"
    )
}

pub fn content_allowed(capture_content: bool, redaction: bool) -> bool {
    capture_content && !redaction
}

pub struct DestinationPolicy {
    redacted: bool,
}

fn is_pii_attribute(key: &str) -> bool {
    marks_pii(key)
        || key.starts_with("gen_ai.prompt.variable")
        || matches!(
            key,
            "chat.item"
                | "session.options"
                | "session.tags"
                | "tag"
                | "gen_ai.input.messages"
                | "gen_ai.output.messages"
                | "gen_ai.system_instructions"
                | "gen_ai.tool.call.arguments"
                | "gen_ai.tool.call.result"
                | "gen_ai.tool.description"
                | "gen_ai.tool.definitions"
                | "gen_ai.retrieval.query.text"
                | "gen_ai.retrieval.documents"
                | "gen_ai.memory.query.text"
                | "gen_ai.memory.records"
                | "gen_ai.evaluation.explanation"
        )
}

fn total(base: usize, charges: impl IntoIterator<Item = usize>) -> usize {
    charges.into_iter().fold(base, usize::saturating_add).min(REFUSED_BYTES)
}

fn string_bytes(s: &str) -> usize {
    64usize.saturating_add(s.chars().count().saturating_mul(4))
}

/// Charges the host's own representation: Python retains up to four bytes per character and
/// its integers in 30-bit digits of four bytes, which also bounds fixed-width SDK integers.
fn value_bytes(value: &TelemetryValue) -> usize {
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

impl DestinationPolicy {
    /// Whether to keep the field `key` of a mapping under `parent` and, if present, the
    /// replacement for its value.
    fn attribute_action(
        &self,
        parent: Option<&str>,
        key: &str,
        destination: bool,
    ) -> (bool, Option<String>) {
        if (destination && matches!(key, "lk.telemetry.export_key" | "lk.telemetry.safe_body"))
            || (self.redacted
                && (is_pii_attribute(key)
                    || key == "exception.stacktrace"
                    || matches!(
                        (parent, key),
                        (Some("evaluation"), "reasoning" | "instructions")
                            | (Some("outcome"), "reason")
                    )))
        {
            return (false, None);
        }
        let replacement =
            (self.redacted && key == "exception.message").then(|| REDACTED_EXCEPTION.to_owned());
        (true, replacement)
    }

    fn filtered(
        &self,
        value: TelemetryValue,
        parent: Option<&str>,
        destination: bool,
    ) -> TelemetryValue {
        match value {
            TelemetryValue::Map(entries) => TelemetryValue::Map(
                entries
                    .into_iter()
                    .filter_map(|(key, value)| {
                        let (keep, replacement) = self.attribute_action(parent, &key, destination);
                        keep.then(|| {
                            let value = match replacement {
                                Some(text) => TelemetryValue::String(text),
                                None => self.filtered(value, Some(&key), destination),
                            };
                            (key, value)
                        })
                    })
                    .collect(),
            ),
            TelemetryValue::Array(items) => TelemetryValue::Array(
                items.into_iter().map(|item| self.filtered(item, parent, destination)).collect(),
            ),
            TelemetryValue::Unsupported => TelemetryValue::None,
            other => other,
        }
    }
}

impl TelemetryPolicy for DestinationPolicy {
    fn new(allow_pii: bool, capture_content: bool, project_redaction: bool) -> Self {
        Self { redacted: !allow_pii || !content_allowed(capture_content, project_redaction) }
    }

    fn redacted(&self) -> bool {
        self.redacted
    }

    fn snapshot(&self, value: TelemetryValue, destination: bool) -> TelemetryValue {
        self.filtered(value, None, destination)
    }

    fn keep_event(&self, name: String) -> bool {
        !self.redacted
            || !matches!(
                name.as_str(),
                "gen_ai.system.message"
                    | "gen_ai.user.message"
                    | "gen_ai.assistant.message"
                    | "gen_ai.tool.message"
                    | "gen_ai.choice"
                    | "gen_ai.client.inference.operation.details"
            )
    }

    fn log_body(&self, safe_body: bool, body: Option<String>) -> Option<String> {
        let safe_body = safe_body && body.as_deref().is_some_and(report_body_is_safe);
        (self.redacted && !safe_body).then(|| REDACTED_LOG_BODY.to_owned())
    }

    fn status_description(&self, is_error: bool, description: Option<String>) -> Option<String> {
        if self.redacted && is_error && description.as_ref().is_some_and(|s| !s.is_empty()) {
            Some(REDACTED_EXCEPTION.to_owned())
        } else {
            description
        }
    }
}

pub struct Admission;

impl TelemetryAdmission for Admission {
    fn _core_export_action(
        registered: bool,
        signal_enabled: bool,
        job_open: bool,
        held_count: usize,
        recording_disabled: bool,
    ) -> TelemetryExportAction {
        if recording_disabled {
            TelemetryExportAction::Disabled
        } else if registered {
            if signal_enabled { TelemetryExportAction::Export } else { TelemetryExportAction::Drop }
        } else if !job_open {
            TelemetryExportAction::Drop
        } else if held_count < MAX_PENDING {
            TelemetryExportAction::Hold
        } else {
            TelemetryExportAction::Capacity
        }
    }

    fn _core_value_bytes(value: TelemetryValue) -> usize {
        value_bytes(&value)
    }

    fn _core_record_bytes(
        kind: TelemetryRecordKind,
        fields: Vec<usize>,
        event_count: usize,
        link_count: usize,
    ) -> usize {
        if event_count > MAX_EVENTS || link_count > MAX_EVENTS {
            return REFUSED_BYTES;
        }
        let base = match kind {
            TelemetryRecordKind::Span | TelemetryRecordKind::Log => 512,
            TelemetryRecordKind::Event | TelemetryRecordKind::Link | TelemetryRecordKind::Scope => {
                128
            }
            TelemetryRecordKind::Value => 0,
        };
        total(base, fields)
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
    fn destination_consent_cannot_widen_capture_or_override_project_redaction() {
        for allow in [false, true] {
            for capture in [false, true] {
                for project in [false, true] {
                    let policy = DestinationPolicy::new(allow, capture, project);
                    assert_eq!(policy.redacted(), !(allow && capture && !project));
                    assert_eq!(
                        policy.attribute_action(None, "lk.pii.input", true).0,
                        !policy.redacted()
                    );
                    assert_eq!(policy.keep_event("gen_ai.user.message".into()), !policy.redacted());
                    assert!(policy.attribute_action(None, "lk.piiuser_input", true).0);
                    assert!(policy.keep_event("llm.complete".into()));
                }
            }
        }
    }

    #[test]
    fn redaction_covers_nested_fields_exception_details_and_generic_log_bodies() {
        let policy = DestinationPolicy::new(false, true, false);
        for key in
            ["gen_ai.tool.call.arguments", "gen_ai.prompt.variable.user", "exception.stacktrace"]
        {
            assert_eq!(policy.attribute_action(None, key, true), (false, None));
        }
        assert_eq!(
            policy.attribute_action(None, "exception.message", true),
            (true, Some(REDACTED_EXCEPTION.into()))
        );
        assert_eq!(
            policy.status_description(true, Some("secret".into())),
            Some(REDACTED_EXCEPTION.into())
        );
        assert_eq!(policy.log_body(false, None), Some(REDACTED_LOG_BODY.into()));
        assert_eq!(policy.log_body(true, Some("evaluation".into())), None);
        assert_eq!(policy.attribute_action(None, "lk.telemetry.safe_body", true), (false, None));
        let opted_in = DestinationPolicy::new(true, true, false);
        assert_eq!(opted_in.attribute_action(None, "exception.message", true), (true, None));
        assert_eq!(opted_in.log_body(false, None), None);
    }

    #[test]
    fn report_content_and_forged_safe_body_markers_cannot_bypass_destination_redaction() {
        let policy = DestinationPolicy::new(false, true, false);
        assert_eq!(
            policy.log_body(true, Some("customer secret".into())),
            Some(REDACTED_LOG_BODY.into())
        );
        for key in ["chat.item", "session.options", "session.tags", "tag"] {
            assert_eq!(policy.attribute_action(None, key, true), (false, None));
        }
        for (parent, key) in
            [("evaluation", "reasoning"), ("evaluation", "instructions"), ("outcome", "reason")]
        {
            assert_eq!(policy.attribute_action(Some(parent), key, true), (false, None));
            assert_eq!(policy.attribute_action(Some("operation"), key, true), (true, None));
        }
        for (parent, key) in [("evaluation", "verdict"), ("outcome", "outcome")] {
            assert_eq!(policy.attribute_action(Some(parent), key, true), (true, None));
        }
        assert_eq!(policy.log_body(true, None), Some(REDACTED_LOG_BODY.into()));
        assert_eq!(policy.log_body(true, Some("evaluation".into())), None);
        let opted_in = DestinationPolicy::new(true, true, false);
        assert_eq!(opted_in.attribute_action(Some("evaluation"), "reasoning", true), (true, None));
        assert_eq!(opted_in.log_body(true, Some("customer secret".into())), None);
    }

    // Catches: a destination snapshot that filters only the top level, or keeps a value the
    // host could not represent.
    #[test]
    fn a_snapshot_filters_every_depth() {
        use TelemetryValue as V;
        let map = |entries: &[(&str, V)]| {
            V::Map(entries.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect())
        };
        let value = map(&[
            ("evaluation", V::Array(vec![map(&[("reasoning", V::String("secret".into()))])])),
            ("exception.message", V::String("secret".into())),
            ("lk.telemetry.safe_body", V::Bool(true)),
            ("big", V::Unsupported),
        ]);
        let redacted = DestinationPolicy::new(false, true, false);
        assert_eq!(
            redacted.snapshot(value.clone(), true),
            map(&[
                ("evaluation", V::Array(vec![map(&[])])),
                ("exception.message", V::String(REDACTED_EXCEPTION.into())),
                ("big", V::None),
            ])
        );
        let source = DestinationPolicy::new(true, true, false).snapshot(value, false);
        assert!(matches!(source, V::Map(entries) if entries.len() == 4));
    }

    #[test]
    fn admission_refuses_large_or_unsupported_sdk_records() {
        use TelemetryValue as V;
        let charge = Admission::_core_value_bytes;
        assert_eq!(charge(V::String("abc".into())), 76);
        assert_eq!(charge(V::Map(vec![("abc".into(), V::None)])), 236);
        assert_eq!(charge(V::Int(i64::MAX)), 44);
        assert_eq!(charge(V::String("x".repeat(MAX_RECORD_BYTES))), REFUSED_BYTES);
        assert_eq!(charge(V::Array(vec![V::Unsupported])), REFUSED_BYTES);
        assert_eq!(
            Admission::_core_record_bytes(TelemetryRecordKind::Span, vec![32, 76], 0, 0),
            620
        );
        assert_eq!(
            Admission::_core_record_bytes(TelemetryRecordKind::Span, vec![], MAX_EVENTS + 1, 0),
            REFUSED_BYTES
        );
        assert_eq!(
            Admission::_core_record_bytes(TelemetryRecordKind::Log, vec![MAX_RECORD_BYTES], 0, 0),
            REFUSED_BYTES
        );
    }

    #[test]
    fn cloud_admission_never_exports_an_undecided_or_disabled_job() {
        assert_eq!(
            Admission::_core_export_action(true, true, false, 0, false),
            TelemetryExportAction::Export
        );
        assert_eq!(
            Admission::_core_export_action(true, false, true, 0, false),
            TelemetryExportAction::Drop
        );
        assert_eq!(
            Admission::_core_export_action(false, true, false, 0, false),
            TelemetryExportAction::Drop
        );
        assert_eq!(
            Admission::_core_export_action(false, true, true, MAX_PENDING - 1, false),
            TelemetryExportAction::Hold
        );
        assert_eq!(
            Admission::_core_export_action(false, true, true, MAX_PENDING, false),
            TelemetryExportAction::Capacity
        );
        assert_eq!(
            Admission::_core_export_action(true, true, true, 0, true),
            TelemetryExportAction::Disabled
        );
    }
}
