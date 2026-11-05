//! Records to OTLP messages: each destination's content policy and identity stamp, then the
//! request body (protobuf, optionally gzip).

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
use lk_agents_abi::telemetry::decl::{Attrs, Content};
use lk_agents_abi::telemetry::record::{
    FRAMEWORK, Key, KeyValues, LogRecord, Record, SpanContext, SpanKind, SpanRecord, Status, Value,
};
use lk_agents_abi::telemetry::semconv::otel::SERVICE_NAME;
use lk_agents_abi::telemetry::spans::common::{JobIdentity as Stamp, SdkResource, SessionFlags};

use super::config::{CloudRecording, JobIdentity, ProcessIdentity};

/// Sets each key, replacing a producer's value: the stamp is the one place identity is written.
pub(crate) fn stamp(attributes: &mut KeyValues, stamp: &KeyValues) {
    for (key, value) in stamp {
        match attributes.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value.clone(),
            None => attributes.push((key.clone(), value.clone())),
        }
    }
}

/// A job's identity: the room name is content, which the caller keeps off resources.
fn identity(process: &ProcessIdentity, job: &JobIdentity) -> Stamp {
    Stamp {
        lk_room_id: job.room_id.clone(),
        room_id: job.room_id.clone(),
        lk_job_id: job.job_id.clone(),
        job_id: job.job_id.clone(),
        room_name: job.room_name.clone().map(Content),
        agent_name: job.agent_name.clone(),
        cloud_agent_id: process.cloud_agent_id.clone(),
        deployment_id: process.deployment_id.clone(),
    }
}

/// What the application destination stamps on a job's spans, before its content policy.
pub(crate) fn app_stamp(process: &ProcessIdentity, job: &JobIdentity) -> KeyValues {
    let mut out = KeyValues::new();
    identity(process, job).write(&mut out);
    out
}

/// What the Cloud destination stamps on a job's every span, log record and measurement: its
/// identity and the session flags Cloud's ingest reads.
pub(crate) fn cloud_stamp(
    process: &ProcessIdentity,
    job: &JobIdentity,
    recording: &CloudRecording,
) -> KeyValues {
    let mut out = app_stamp(process, job);
    let simulation = recording.simulation.as_ref();
    SessionFlags {
        redaction: recording.redaction,
        simulation: simulation.is_some(),
        simulation_run_id: simulation.and_then(|s| s.run_id.clone()),
        simulation_job_id: simulation.and_then(|s| s.job_id.clone()),
    }
    .write(&mut out);
    out
}

fn sdk(process: &ProcessIdentity, out: &mut KeyValues) {
    SdkResource {
        name: process.sdk_name.as_str().to_owned(),
        version: process.sdk_version.clone(),
    }
    .write(out);
}

pub(crate) fn cloud_resource(process: &ProcessIdentity, job: &JobIdentity) -> Resource {
    let mut attributes: KeyValues = vec![(Cow::Borrowed(SERVICE_NAME), FRAMEWORK.into())];
    sdk(process, &mut attributes);
    Stamp { room_name: None, ..identity(process, job) }.write(&mut attributes);
    resource(&attributes)
}

pub(crate) fn app_resource(process: &ProcessIdentity, pairs: &[(String, String)]) -> Resource {
    let mut attributes: KeyValues =
        pairs.iter().map(|(k, v)| (Cow::Owned(k.clone()), Value::String(v.clone()))).collect();
    sdk(process, &mut attributes);
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

fn flags(context: &SpanContext) -> u32 {
    u32::from(context.sampled)
}

fn span(record: &SpanRecord) -> trace::Span {
    let kind = match record.kind {
        SpanKind::Internal => trace::span::SpanKind::Internal,
        SpanKind::Client => trace::span::SpanKind::Client,
        SpanKind::Server => trace::span::SpanKind::Server,
        SpanKind::Producer => trace::span::SpanKind::Producer,
        SpanKind::Consumer => trace::span::SpanKind::Consumer,
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
        trace_id: record.context.trace_id.to_be_bytes().to_vec(),
        span_id: record.context.span_id.to_be_bytes().to_vec(),
        parent_span_id: record.parent.map(|p| p.to_be_bytes().to_vec()).unwrap_or_default(),
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
                trace_id: link.trace_id.to_be_bytes().to_vec(),
                span_id: link.span_id.to_be_bytes().to_vec(),
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
        trace_id: record.span.map(|s| s.trace_id.to_be_bytes().to_vec()).unwrap_or_default(),
        span_id: record.span.map(|s| s.span_id.to_be_bytes().to_vec()).unwrap_or_default(),
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

pub(crate) fn metrics_request(
    resource: Resource,
    metrics: Vec<metrics::Metric>,
) -> ExportMetricsServiceRequest {
    ExportMetricsServiceRequest {
        resource_metrics: vec![metrics::ResourceMetrics {
            resource: Some(resource),
            scope_metrics: vec![metrics::ScopeMetrics {
                scope: scope(&Cow::Borrowed(FRAMEWORK)),
                metrics,
                ..Default::default()
            }],
            ..Default::default()
        }],
    }
}

/// An encoded request body.
pub(crate) struct Body {
    bytes: Vec<u8>,
    gzip: bool,
}

impl Body {
    /// `request` with this body and its content headers.
    pub fn attach(self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let request = request.header(reqwest::header::CONTENT_TYPE, "application/x-protobuf");
        let request = if self.gzip {
            request.header(reqwest::header::CONTENT_ENCODING, "gzip")
        } else {
            request
        };
        request.body(self.bytes)
    }
}

pub(crate) fn body(message: &impl prost::Message, gzip: bool) -> Result<Body, String> {
    let bytes = message.encode_to_vec();
    let bytes = if gzip {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(&bytes).map_err(|e| e.to_string())?;
        encoder.finish().map_err(|e| e.to_string())?
    } else {
        bytes
    };
    Ok(Body { bytes, gzip })
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
