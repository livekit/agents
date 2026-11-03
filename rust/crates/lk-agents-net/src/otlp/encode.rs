//! Records to OTLP messages: each destination's content policy and identity stamp, then the
//! request body (protobuf or JSON, optionally gzip).

use std::borrow::Cow;
use std::io::Write as _;

use super::proto::collector::logs::v1::ExportLogsServiceRequest;
use super::proto::collector::metrics::v1::ExportMetricsServiceRequest;
use super::proto::collector::trace::v1::ExportTraceServiceRequest;
use super::proto::common::v1::{
    AnyValue, ArrayValue, InstrumentationScope, KeyValue, KeyValueList, any_value,
};
use super::proto::logs::v1 as logs;
use super::proto::metrics::v1 as metrics;
use super::proto::resource::v1::Resource;
use super::proto::trace::v1 as trace;
use lk_agents_abi::telemetry::attributes::{
    ATTRIBUTE_REDACTION_ENABLED, ATTRIBUTE_SIMULATION_ENABLED, ATTRIBUTE_SIMULATION_JOB_ID,
    ATTRIBUTE_SIMULATION_RUN_ID,
};
use lk_agents_abi::telemetry::record::{
    Key, KeyValues, LogRecord, Record, SpanKind, SpanRecord, SpanRef, Status, Value,
};

use super::config::{CloudRecording, JobIdentity, ProcessIdentity, Protocol};

// Protocol shared keys and the spellings Cloud's consumers read today.
const ROOM_ID: &str = "lk.room_id";
const LEGACY_ROOM_ID: &str = "room_id";
const JOB_ID: &str = "lk.job_id";
const LEGACY_JOB_ID: &str = "job_id";
const ROOM_NAME: &str = "lk.pii.room_name";
const AGENT_NAME: &str = "lk.agent_name";
const CLOUD_AGENT_ID: &str = "lk.cloud_agent_id";
const DEPLOYMENT_ID: &str = "lk.deployment_id";
const SDK_NAME: &str = "lk.sdk.name";
const SDK_VERSION: &str = "lk.sdk.version";
const SERVICE_NAME: &str = "service.name";

/// Sets each key, replacing a producer's value: the stamp is the one place identity is written.
pub(crate) fn stamp(attributes: &mut KeyValues, stamp: &KeyValues) {
    for (key, value) in stamp {
        match attributes.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value.clone(),
            None => attributes.push((key.clone(), value.clone())),
        }
    }
}

fn push_str(out: &mut KeyValues, key: &'static str, value: &str) {
    if !value.is_empty() {
        out.push((Cow::Borrowed(key), Value::String(value.to_owned())));
    }
}

/// A job's identity attributes. `room_name` is content and stays off resources.
pub(crate) fn job_stamp(
    process: &ProcessIdentity,
    job: &JobIdentity,
    room_name: bool,
) -> KeyValues {
    let mut out = KeyValues::new();
    push_str(&mut out, ROOM_ID, &job.room_id);
    push_str(&mut out, LEGACY_ROOM_ID, &job.room_id);
    push_str(&mut out, JOB_ID, &job.job_id);
    push_str(&mut out, LEGACY_JOB_ID, &job.job_id);
    if room_name {
        push_str(&mut out, ROOM_NAME, &job.room_name);
    }
    push_str(&mut out, AGENT_NAME, job.agent_name.as_deref().unwrap_or_default());
    push_str(&mut out, CLOUD_AGENT_ID, process.cloud_agent_id.as_deref().unwrap_or_default());
    push_str(&mut out, DEPLOYMENT_ID, process.deployment_id.as_deref().unwrap_or_default());
    out
}

/// The session flags Cloud's ingest reads on every span and log record.
pub(crate) fn session_flags(recording: &CloudRecording) -> KeyValues {
    let mut out: KeyValues = vec![
        (Cow::Borrowed(ATTRIBUTE_REDACTION_ENABLED), Value::Bool(recording.redaction)),
        (Cow::Borrowed(ATTRIBUTE_SIMULATION_ENABLED), Value::Bool(recording.simulation.is_some())),
    ];
    if let Some(simulation) = &recording.simulation {
        push_str(&mut out, ATTRIBUTE_SIMULATION_RUN_ID, simulation.run_id.as_deref().unwrap_or(""));
        push_str(&mut out, ATTRIBUTE_SIMULATION_JOB_ID, simulation.job_id.as_deref().unwrap_or(""));
    }
    out
}

pub(crate) fn cloud_resource(process: &ProcessIdentity, job: &JobIdentity) -> Resource {
    let mut attributes: KeyValues = vec![(Cow::Borrowed(SERVICE_NAME), "livekit-agents".into())];
    push_str(&mut attributes, SDK_NAME, process.sdk_name.as_str());
    push_str(&mut attributes, SDK_VERSION, &process.sdk_version);
    attributes.extend(job_stamp(process, job, false));
    resource(&attributes)
}

pub(crate) fn app_resource(process: &ProcessIdentity, pairs: &[(String, String)]) -> Resource {
    let mut attributes: KeyValues =
        pairs.iter().map(|(k, v)| (Cow::Owned(k.clone()), Value::String(v.clone()))).collect();
    push_str(&mut attributes, SDK_NAME, process.sdk_name.as_str());
    push_str(&mut attributes, SDK_VERSION, &process.sdk_version);
    resource(&attributes)
}

fn resource(attributes: &KeyValues) -> Resource {
    Resource { attributes: key_values(attributes), ..Default::default() }
}

fn any_value(value: &Value) -> AnyValue {
    use any_value::Value as V;
    let value = match value {
        Value::Bool(v) => V::BoolValue(*v),
        Value::Int(v) => V::IntValue(*v),
        Value::Double(v) => V::DoubleValue(*v),
        Value::String(v) => V::StringValue(v.clone()),
        Value::Bytes(v) => V::BytesValue(v.clone()),
        Value::Array(items) => {
            V::ArrayValue(ArrayValue { values: items.iter().map(any_value).collect() })
        }
        Value::KvList(entries) => V::KvlistValue(KeyValueList {
            values: entries.iter().map(|(k, v)| key_value(k, v)).collect(),
        }),
    };
    AnyValue { value: Some(value) }
}

fn key_value(key: &str, value: &Value) -> KeyValue {
    KeyValue { key: key.to_owned(), value: Some(any_value(value)) }
}

pub(crate) fn key_values(attributes: &KeyValues) -> Vec<KeyValue> {
    attributes.iter().map(|(k, v)| key_value(k, v)).collect()
}

fn flags(context: &SpanRef) -> u32 {
    u32::from(context.sampled)
}

fn span(record: &SpanRecord) -> trace::Span {
    let kind = match record.kind {
        SpanKind::Internal => trace::span::SpanKind::Internal,
        SpanKind::Client => trace::span::SpanKind::Client,
        SpanKind::Server => trace::span::SpanKind::Server,
    };
    let status = match &record.status {
        Status::Unset => None,
        Status::Ok => {
            Some(trace::Status { code: trace::status::StatusCode::Ok as i32, ..Default::default() })
        }
        Status::Error(message) => Some(trace::Status {
            code: trace::status::StatusCode::Error as i32,
            message: message.clone(),
        }),
    };
    trace::Span {
        trace_id: record.context.trace_id.0.to_be_bytes().to_vec(),
        span_id: record.context.span_id.0.to_be_bytes().to_vec(),
        parent_span_id: record.parent.map(|p| p.0.to_be_bytes().to_vec()).unwrap_or_default(),
        flags: flags(&record.context),
        name: record.name.to_string(),
        kind: kind as i32,
        start_time_unix_nano: record.start_unix_nanos,
        end_time_unix_nano: record.end_unix_nanos,
        attributes: key_values(&record.attributes),
        events: record
            .events
            .iter()
            .map(|event| trace::span::Event {
                time_unix_nano: event.time_unix_nanos,
                name: event.name.to_string(),
                attributes: key_values(&event.attributes),
                ..Default::default()
            })
            .collect(),
        links: record
            .links
            .iter()
            .map(|link| trace::span::Link {
                trace_id: link.trace_id.0.to_be_bytes().to_vec(),
                span_id: link.span_id.0.to_be_bytes().to_vec(),
                flags: flags(link),
                ..Default::default()
            })
            .collect(),
        status,
        ..Default::default()
    }
}

fn log(record: &LogRecord) -> logs::LogRecord {
    logs::LogRecord {
        time_unix_nano: record.time_unix_nanos,
        observed_time_unix_nano: record.time_unix_nanos,
        severity_number: record.severity_number,
        severity_text: record.severity_text.to_string(),
        body: Some(any_value(&record.body)),
        attributes: key_values(&record.attributes),
        flags: record.span.as_ref().map_or(0, flags),
        trace_id: record.span.map(|s| s.trace_id.0.to_be_bytes().to_vec()).unwrap_or_default(),
        span_id: record.span.map(|s| s.span_id.0.to_be_bytes().to_vec()).unwrap_or_default(),
        ..Default::default()
    }
}

fn scope(name: &Key) -> Option<InstrumentationScope> {
    Some(InstrumentationScope { name: name.to_string(), ..Default::default() })
}

/// Groups by instrumentation scope in first-seen order.
fn by_scope<'a, T>(items: impl Iterator<Item = (&'a Key, T)>) -> Vec<(&'a Key, Vec<T>)> {
    let mut groups: Vec<(&Key, Vec<T>)> = Vec::new();
    for (scope, item) in items {
        match groups.iter_mut().find(|(s, _)| *s == scope) {
            Some(group) => group.1.push(item),
            None => groups.push((scope, vec![item])),
        }
    }
    groups
}

pub(crate) fn traces_request(
    resource: Resource,
    spans: &[SpanRecord],
) -> ExportTraceServiceRequest {
    let scope_spans = by_scope(spans.iter().map(|s| (&s.scope, span(s))))
        .into_iter()
        .map(|(name, spans)| trace::ScopeSpans { scope: scope(name), spans, ..Default::default() })
        .collect();
    ExportTraceServiceRequest {
        resource_spans: vec![trace::ResourceSpans {
            resource: Some(resource),
            scope_spans,
            ..Default::default()
        }],
    }
}

pub(crate) fn logs_request(resource: Resource, records: &[LogRecord]) -> ExportLogsServiceRequest {
    let scope_logs = by_scope(records.iter().map(|r| (&r.scope, log(r))))
        .into_iter()
        .map(|(name, log_records)| logs::ScopeLogs {
            scope: scope(name),
            log_records,
            ..Default::default()
        })
        .collect();
    ExportLogsServiceRequest {
        resource_logs: vec![logs::ResourceLogs {
            resource: Some(resource),
            scope_logs,
            ..Default::default()
        }],
    }
}

/// The framework's instrumentation scope for measurements.
pub(crate) const METRICS_SCOPE: &str = "livekit-agents";

pub(crate) fn metrics_request(
    resource: Resource,
    metrics: Vec<metrics::Metric>,
) -> ExportMetricsServiceRequest {
    ExportMetricsServiceRequest {
        resource_metrics: vec![metrics::ResourceMetrics {
            resource: Some(resource),
            scope_metrics: vec![metrics::ScopeMetrics {
                scope: scope(&Cow::Borrowed(METRICS_SCOPE)),
                metrics,
                ..Default::default()
            }],
            ..Default::default()
        }],
    }
}

/// An encoded request body and its headers.
pub(crate) struct Body {
    pub bytes: Vec<u8>,
    pub content_type: &'static str,
    pub gzip: bool,
}

/// One export request.
pub(crate) enum Message {
    Traces(ExportTraceServiceRequest),
    Logs(ExportLogsServiceRequest),
    Metrics(ExportMetricsServiceRequest),
}

pub(crate) fn body(message: &Message, protocol: Protocol, gzip: bool) -> Result<Body, String> {
    use prost::Message as _;
    let (bytes, content_type) = match protocol {
        Protocol::Protobuf => {
            let bytes = match message {
                Message::Traces(m) => m.encode_to_vec(),
                Message::Logs(m) => m.encode_to_vec(),
                Message::Metrics(m) => m.encode_to_vec(),
            };
            (bytes, "application/x-protobuf")
        }
        Protocol::Json => {
            let json = match message {
                Message::Traces(m) => super::json::traces(m),
                Message::Logs(m) => super::json::logs(m),
                Message::Metrics(m) => super::json::metrics(m),
            };
            (serde_json::to_vec(&json).map_err(|e| e.to_string())?, "application/json")
        }
    };
    let bytes = if gzip {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(&bytes).map_err(|e| e.to_string())?;
        encoder.finish().map_err(|e| e.to_string())?
    } else {
        bytes
    };
    Ok(Body { bytes, content_type, gzip })
}

/// The retained bytes a queued record is charged: its strings and values, plus overhead.
pub(crate) fn charge(record: &Record) -> usize {
    fn value(v: &Value) -> usize {
        16 + match v {
            Value::String(s) => s.len(),
            Value::Bytes(b) => b.len(),
            Value::Array(items) => items.iter().map(value).sum(),
            Value::KvList(entries) => entries.iter().map(|(k, v)| k.len() + value(v)).sum(),
            _ => 0,
        }
    }
    fn attributes(kvs: &KeyValues) -> usize {
        kvs.iter().map(|(k, v)| 16 + k.len() + value(v)).sum()
    }
    match record {
        Record::Span(s) => {
            256 + s.name.len()
                + attributes(&s.attributes)
                + s.events
                    .iter()
                    .map(|e| 64 + e.name.len() + attributes(&e.attributes))
                    .sum::<usize>()
                + 48 * s.links.len()
                + match &s.status {
                    Status::Error(d) => d.len(),
                    _ => 0,
                }
        }
        Record::Log(l) => 256 + value(&l.body) + attributes(&l.attributes),
        Record::Measurement(m) => 64 + attributes(&m.labels),
    }
}
