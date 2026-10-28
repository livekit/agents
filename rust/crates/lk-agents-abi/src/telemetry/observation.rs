//! Immutable observation data. Subjects supply identity, trace relationships and both clocks at
//! the decision that creates the record; dispatch never consults a live job or speech handle.

use super::{HrTime, LogRecord, SpanAttribute};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationIdentity {
    pub thread_id: String,
    pub lane: String,
    pub job_id: String,
    pub operation_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationTime {
    pub wall: HrTime,
    pub monotonic_ns: u64,
}

/// A validated W3C version-00 parent, independent of delivery of span-start observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceParent {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub sampled: bool,
}

impl TraceParent {
    pub fn parse(value: &str) -> Option<Self> {
        let bytes = value.as_bytes();
        if bytes.len() != 55 || &bytes[..3] != b"00-" || bytes[35] != b'-' || bytes[52] != b'-' {
            return None;
        }
        fn decode<const N: usize>(text: &[u8]) -> Option<[u8; N]> {
            let mut out = [0; N];
            fn hex(b: u8) -> Option<u8> {
                match b {
                    b'0'..=b'9' => Some(b - b'0'),
                    b'a'..=b'f' => Some(b - b'a' + 10),
                    _ => None,
                }
            }
            for (value, pair) in out.iter_mut().zip(text.chunks_exact(2)) {
                *value = hex(pair[0])? * 16 + hex(pair[1])?;
            }
            Some(out)
        }
        let trace_id = decode(&bytes[3..35])?;
        let span_id = decode(&bytes[36..52])?;
        let flags: [u8; 1] = decode(&bytes[53..])?;
        if trace_id == [0; 16] || span_id == [0; 8] {
            return None;
        }
        Some(Self { trace_id, span_id, sampled: flags[0] & 1 != 0 })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObservationKind {
    SpanStarted {
        name: String,
        parent_operation: Option<String>,
        trace_parent: Option<TraceParent>,
        attributes: Vec<SpanAttribute>,
    },
    SpanEnded {
        attributes: Vec<SpanAttribute>,
        failed: bool,
    },
    Log(LogRecord),
    /// Failure diagnostics contain categories and counts, never exception text or content.
    ObserverDropped {
        records: u64,
        bytes: u64,
    },
    IncompleteCleanup {
        pending: u64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub identity: ObservationIdentity,
    pub at: ObservationTime,
    /// Content capture is fixed at creation; changing configuration while queued cannot widen it.
    pub capture_content: bool,
    pub kind: ObservationKind,
}

impl Observation {
    /// Payload admission accounting excludes allocator bookkeeping but includes every retained
    /// string, vector and nested log value. Runtime also charges the inline record size.
    pub fn retained_bytes(&self) -> usize {
        use super::SpanAttr;
        fn attrs(attributes: &[SpanAttribute]) -> usize {
            attributes.iter().fold(0usize, |size, attr| {
                let bytes = match &attr.value {
                    SpanAttr::Str(s) => s.len(),
                    SpanAttr::StrArray(v) => v
                        .iter()
                        .map(String::len)
                        .sum::<usize>()
                        .saturating_add(v.len() * std::mem::size_of::<String>()),
                    SpanAttr::Num(_) | SpanAttr::Bool(_) => 0,
                };
                size.saturating_add(attr.key.len())
                    .saturating_add(bytes)
                    .saturating_add(std::mem::size_of::<SpanAttribute>())
            })
        }
        fn json_size(value: &serde_json::Value) -> usize {
            use serde_json::Value;
            std::mem::size_of::<Value>().saturating_add(match value {
                Value::String(s) => s.len(),
                Value::Array(a) => a.iter().map(json_size).fold(0, usize::saturating_add),
                Value::Object(o) => o
                    .iter()
                    .map(|(k, v)| k.len().saturating_add(json_size(v)))
                    .fold(0, usize::saturating_add),
                _ => 0,
            })
        }
        let id = &self.identity;
        let payload = match &self.kind {
            ObservationKind::SpanStarted { name, parent_operation, attributes, .. } => name
                .len()
                .saturating_add(parent_operation.as_ref().map_or(0, String::len))
                .saturating_add(attrs(attributes)),
            ObservationKind::SpanEnded { attributes, .. } => attrs(attributes),
            ObservationKind::Log(log) => log
                .logger
                .len()
                .saturating_add(log.body.len())
                .saturating_add(log.severity_text.len())
                .saturating_add(json_size(&log.attributes)),
            ObservationKind::ObserverDropped { .. } | ObservationKind::IncompleteCleanup { .. } => {
                0
            }
        };
        [
            std::mem::size_of::<Self>(),
            id.thread_id.len(),
            id.lane.len(),
            id.job_id.len(),
            id.operation_id.len(),
            payload,
        ]
        .into_iter()
        .fold(0, usize::saturating_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_parent_refuses_malformed_oversized_and_zero_identifiers() {
        let valid = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
        assert!(TraceParent::parse(valid).unwrap().sampled);
        for invalid in [
            "",
            "00-00000000000000000000000000000000-0123456789abcdef-01",
            "00-0123456789abcdef0123456789abcdef-0000000000000000-01",
            "00-0123456789ABCDEF0123456789abcdef-0123456789abcdef-01",
        ] {
            assert!(TraceParent::parse(invalid).is_none());
        }
        assert!(TraceParent::parse(&format!("{valid}extra")).is_none());
    }
}
