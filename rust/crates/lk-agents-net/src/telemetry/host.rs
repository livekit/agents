//! `_core.TelemetrySpan`: a span the host's OpenTelemetry API starts, recorded like a framework
//! span. Names and keys are the application's own.

use std::collections::BTreeMap;

use lk_agents_abi::telemetry::decl::json;
use lk_agents_abi::telemetry::record::{
    EventRecord, JobKey, Key, KeyValues, Record, SpanId, SpanKind, SpanRecord, SpanRef, Status,
    TraceId,
};
use parking_lot::Mutex;
use serde_json::Value;

use super::Tracer;
use crate::env::ProcessEnv;

pub struct HostSpan {
    tracer: Tracer,
    context: SpanRef,
    job: Option<JobKey>,
    open: Mutex<Option<Open>>,
}

struct Open {
    name: String,
    scope: String,
    kind: SpanKind,
    parent: Option<SpanId>,
    start: u64,
    attributes: KeyValues,
    events: Vec<EventRecord>,
    status: Status,
}

fn attributes(values: BTreeMap<String, Value>) -> KeyValues {
    values.into_iter().filter_map(|(k, v)| Some((Key::Owned(k), json(&v)?))).collect()
}

impl lk_agents_abi::telemetry::host::HostSpan for HostSpan {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        name: String,
        scope: String,
        parent: Option<(u128, u64, bool)>,
        kind: i64,
        start_time: Option<u64>,
        ids: Option<(u128, u64)>,
        job: Option<String>,
    ) -> Self {
        let tracer = env.tracer.clone();
        let parent = parent.filter(|(trace, span, _)| *trace != 0 && *span != 0);
        let context = match (ids, parent) {
            (Some((trace, span)), _) => SpanRef {
                trace_id: TraceId(trace),
                span_id: SpanId(span),
                sampled: parent.is_none_or(|p| p.2),
            },
            (None, parent) => {
                let parent = parent.map(|(trace, span, sampled)| SpanRef {
                    trace_id: TraceId(trace),
                    span_id: SpanId(span),
                    sampled,
                });
                tracer.root().continuing(parent).new_context()
            }
        };
        // OpenTelemetry's SpanKind: INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER.
        let kind = match kind {
            1 => SpanKind::Server,
            2 => SpanKind::Client,
            _ => SpanKind::Internal,
        };
        let open = Open {
            name,
            scope,
            kind,
            parent: parent.map(|(_, span, _)| SpanId(span)),
            start: start_time.unwrap_or_else(|| tracer.now_nanos()),
            attributes: KeyValues::new(),
            events: Vec::new(),
            status: Status::Unset,
        };
        HostSpan { tracer, context, job: job.map(JobKey), open: Mutex::new(Some(open)) }
    }

    fn context(&self) -> (u128, u64, bool) {
        (self.context.trace_id.0, self.context.span_id.0, self.context.sampled)
    }

    fn is_recording(&self) -> bool {
        self.open.lock().is_some()
    }

    fn set_attribute(&self, key: String, value: Value) {
        let Some(value) = json(&value) else { return };
        if let Some(open) = self.open.lock().as_mut() {
            match open.attributes.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = value,
                None => open.attributes.push((Key::Owned(key), value)),
            }
        }
    }

    fn add_event(
        &self,
        name: String,
        attributes: Option<BTreeMap<String, Value>>,
        timestamp: Option<u64>,
    ) {
        let time = timestamp.unwrap_or_else(|| self.tracer.now_nanos());
        let attributes = self::attributes(attributes.unwrap_or_default());
        if let Some(open) = self.open.lock().as_mut() {
            open.events.push(EventRecord {
                name: Key::Owned(name),
                time_unix_nanos: time,
                attributes,
            });
        }
    }

    fn set_status(&self, code: i64, description: Option<String>) {
        // OpenTelemetry's StatusCode: UNSET, OK, ERROR. OK is final.
        if let Some(open) = self.open.lock().as_mut() {
            if open.status == Status::Ok {
                return;
            }
            open.status = match code {
                1 => Status::Ok,
                2 => Status::Error(description.unwrap_or_default()),
                _ => Status::Unset,
            };
        }
    }

    fn update_name(&self, name: String) {
        if let Some(open) = self.open.lock().as_mut() {
            open.name = name;
        }
    }

    fn end(&self, end_time: Option<u64>) {
        let Some(open) = self.open.lock().take() else { return };
        let end = end_time.unwrap_or_else(|| self.tracer.now_nanos()).max(open.start);
        self.tracer.record(Record::Span(SpanRecord {
            context: self.context,
            parent: open.parent,
            name: Key::Owned(open.name),
            kind: open.kind,
            start_unix_nanos: open.start,
            end_unix_nanos: end,
            attributes: open.attributes,
            events: open.events,
            links: Vec::new(),
            status: open.status,
            scope: Key::Owned(open.scope),
            job: self.job.clone(),
        }));
    }
}
