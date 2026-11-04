//! OTLP/JSON (the OTLP specification's protobuf JSON mapping) for the messages the exporter
//! sends: lowerCamelCase fields, trace and span ids as hex, enums as integers, 64-bit integers as
//! decimal strings, other bytes as base64. Default values are omitted.

use base64::Engine as _;
use serde_json::{Map, Value as Json, json};

use super::proto::collector::logs::v1::ExportLogsServiceRequest;
use super::proto::collector::metrics::v1::ExportMetricsServiceRequest;
use super::proto::collector::trace::v1::ExportTraceServiceRequest;
use super::proto::common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value};
use super::proto::logs::v1::LogRecord;
use super::proto::metrics::v1::{
    HistogramDataPoint, Metric, NumberDataPoint, metric, number_data_point,
};
use super::proto::resource::v1::Resource;
use super::proto::trace::v1::Span;

/// An object without its absent, empty-string and empty-list fields. Scalar defaults are
/// written, as the mapping allows, so a set optional or oneof zero is never lost.
fn object<const N: usize>(fields: [(&str, Json); N]) -> Json {
    let fields = fields.into_iter().filter(|(_, v)| match v {
        Json::Null => false,
        Json::String(s) => !s.is_empty(),
        Json::Array(a) => !a.is_empty(),
        _ => true,
    });
    Json::Object(fields.map(|(k, v)| (k.to_owned(), v)).collect::<Map<_, _>>())
}

fn hex(bytes: &[u8]) -> Json {
    Json::String(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn int64(v: impl ToString) -> Json {
    Json::String(v.to_string())
}

fn list<T>(items: &[T], f: impl Fn(&T) -> Json) -> Json {
    Json::Array(items.iter().map(f).collect())
}

fn any_value(value: &AnyValue) -> Json {
    use any_value::Value as V;
    match &value.value {
        None => json!({}),
        Some(V::StringValue(s)) => json!({ "stringValue": s }),
        Some(V::BoolValue(b)) => json!({ "boolValue": b }),
        Some(V::IntValue(i)) => json!({ "intValue": i.to_string() }),
        Some(V::DoubleValue(d)) => json!({ "doubleValue": d }),
        Some(V::BytesValue(b)) => {
            json!({ "bytesValue": base64::engine::general_purpose::STANDARD.encode(b) })
        }
        Some(V::ArrayValue(a)) => json!({ "arrayValue": { "values": list(&a.values, any_value) } }),
        Some(V::KvlistValue(kv)) => {
            json!({ "kvlistValue": { "values": list(&kv.values, key_value) } })
        }
    }
}

fn key_value(kv: &KeyValue) -> Json {
    json!({ "key": kv.key, "value": kv.value.as_ref().map_or(json!({}), any_value) })
}

fn resource(resource: &Option<Resource>) -> Json {
    resource
        .as_ref()
        .map_or(Json::Null, |r| object([("attributes", list(&r.attributes, key_value))]))
}

fn scope(scope: &Option<InstrumentationScope>) -> Json {
    scope.as_ref().map_or(Json::Null, |s| {
        object([
            ("name", json!(s.name)),
            ("version", json!(s.version)),
            ("attributes", list(&s.attributes, key_value)),
        ])
    })
}

fn span(s: &Span) -> Json {
    object([
        ("traceId", hex(&s.trace_id)),
        ("spanId", hex(&s.span_id)),
        ("parentSpanId", hex(&s.parent_span_id)),
        ("flags", json!(s.flags)),
        ("name", json!(s.name)),
        ("kind", json!(s.kind)),
        ("startTimeUnixNano", int64(s.start_time_unix_nano)),
        ("endTimeUnixNano", int64(s.end_time_unix_nano)),
        ("attributes", list(&s.attributes, key_value)),
        (
            "events",
            list(&s.events, |e| {
                object([
                    ("timeUnixNano", int64(e.time_unix_nano)),
                    ("name", json!(e.name)),
                    ("attributes", list(&e.attributes, key_value)),
                ])
            }),
        ),
        (
            "links",
            list(&s.links, |l| {
                object([
                    ("traceId", hex(&l.trace_id)),
                    ("spanId", hex(&l.span_id)),
                    ("flags", json!(l.flags)),
                ])
            }),
        ),
        (
            "status",
            s.status.as_ref().map_or(Json::Null, |st| {
                object([("message", json!(st.message)), ("code", json!(st.code))])
            }),
        ),
    ])
}

fn log(l: &LogRecord) -> Json {
    object([
        ("timeUnixNano", int64(l.time_unix_nano)),
        ("observedTimeUnixNano", int64(l.observed_time_unix_nano)),
        ("severityNumber", json!(l.severity_number)),
        ("severityText", json!(l.severity_text)),
        ("body", l.body.as_ref().map_or(Json::Null, any_value)),
        ("attributes", list(&l.attributes, key_value)),
        ("flags", json!(l.flags)),
        ("traceId", hex(&l.trace_id)),
        ("spanId", hex(&l.span_id)),
    ])
}

fn number(p: &NumberDataPoint) -> Json {
    let value = match p.value {
        Some(number_data_point::Value::AsInt(i)) => ("asInt", json!(i.to_string())),
        Some(number_data_point::Value::AsDouble(d)) => ("asDouble", json!(d)),
        None => ("asInt", Json::Null),
    };
    object([
        ("attributes", list(&p.attributes, key_value)),
        ("startTimeUnixNano", int64(p.start_time_unix_nano)),
        ("timeUnixNano", int64(p.time_unix_nano)),
        value,
    ])
}

fn histogram(p: &HistogramDataPoint) -> Json {
    object([
        ("attributes", list(&p.attributes, key_value)),
        ("startTimeUnixNano", int64(p.start_time_unix_nano)),
        ("timeUnixNano", int64(p.time_unix_nano)),
        ("count", int64(p.count)),
        ("sum", p.sum.map_or(Json::Null, |v| json!(v))),
        ("bucketCounts", list(&p.bucket_counts, |c| json!(c.to_string()))),
        ("explicitBounds", list(&p.explicit_bounds, |b| json!(b))),
        ("min", p.min.map_or(Json::Null, |v| json!(v))),
        ("max", p.max.map_or(Json::Null, |v| json!(v))),
    ])
}

fn metric(m: &Metric) -> Json {
    let data = match &m.data {
        Some(metric::Data::Sum(s)) => (
            "sum",
            object([
                ("dataPoints", list(&s.data_points, number)),
                ("aggregationTemporality", json!(s.aggregation_temporality)),
                ("isMonotonic", json!(s.is_monotonic)),
            ]),
        ),
        Some(metric::Data::Histogram(h)) => (
            "histogram",
            object([
                ("dataPoints", list(&h.data_points, histogram)),
                ("aggregationTemporality", json!(h.aggregation_temporality)),
            ]),
        ),
        // The exporter aggregates only sums and histograms.
        _ => ("sum", Json::Null),
    };
    object([
        ("name", json!(m.name)),
        ("description", json!(m.description)),
        ("unit", json!(m.unit)),
        data,
    ])
}

pub(crate) fn traces(request: &ExportTraceServiceRequest) -> Json {
    json!({ "resourceSpans": list(&request.resource_spans, |r| object([
        ("resource", resource(&r.resource)),
        ("scopeSpans", list(&r.scope_spans, |s| object([
            ("scope", scope(&s.scope)),
            ("spans", list(&s.spans, span)),
        ]))),
    ])) })
}

pub(crate) fn logs(request: &ExportLogsServiceRequest) -> Json {
    json!({ "resourceLogs": list(&request.resource_logs, |r| object([
        ("resource", resource(&r.resource)),
        ("scopeLogs", list(&r.scope_logs, |s| object([
            ("scope", scope(&s.scope)),
            ("logRecords", list(&s.log_records, log)),
        ]))),
    ])) })
}

pub(crate) fn metrics(request: &ExportMetricsServiceRequest) -> Json {
    json!({ "resourceMetrics": list(&request.resource_metrics, |r| object([
        ("resource", resource(&r.resource)),
        ("scopeMetrics", list(&r.scope_metrics, |s| object([
            ("scope", scope(&s.scope)),
            ("metrics", list(&s.metrics, metric)),
        ]))),
    ])) })
}
