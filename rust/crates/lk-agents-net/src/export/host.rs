use std::collections::BTreeMap;

use lk_agents_abi::telemetry::decl::json;
use lk_agents_abi::telemetry::host::{EndedSpan, HostSpan as _, StatusCode};
use lk_agents_abi::telemetry::record::{
    EventRecord, JobKey, Key, KeyValues, Record, SpanContext, SpanKind, SpanRecord, Status,
};
use parking_lot::Mutex;
use serde_json::Value;

use crate::env::ProcessEnv;
use lk_agents_telemetry::Tracer;

pub struct HostSpan {
    tracer: Tracer,
    context: SpanContext,
    job: Option<JobKey>,
    open: Mutex<Option<Open>>,
}

struct Open {
    name: String,
    scope: String,
    kind: SpanKind,
    parent: Option<u64>,
    start: u64,
    attributes: KeyValues,
    events: Vec<EventRecord>,
    status: Status,
}

fn attributes(values: BTreeMap<String, Value>) -> KeyValues {
    values.into_iter().filter_map(|(k, v)| Some((Key::Owned(k), json(&v)?))).collect()
}

impl HostSpan {
    #[allow(clippy::too_many_arguments, reason = "The host span's constructor arguments")]
    fn open(
        tracer: Tracer,
        name: String,
        scope: String,
        parent: Option<SpanContext>,
        kind: Option<SpanKind>,
        start_time: Option<u64>,
        ids: Option<SpanContext>,
        job: Option<String>,
    ) -> Self {
        let parent = parent.and_then(SpanContext::valid);
        let context = match ids.and_then(SpanContext::valid) {
            Some(ids) => ids,
            None => tracer.root().continuing(parent).child_context(),
        };
        let open = Open {
            name,
            scope,
            kind: kind.unwrap_or_default(),
            parent: parent.map(|p| p.span_id),
            start: start_time.unwrap_or_else(|| tracer.now_nanos()),
            attributes: KeyValues::new(),
            events: Vec::new(),
            status: Status::Unset,
        };
        HostSpan { tracer, context, job: job.map(JobKey), open: Mutex::new(Some(open)) }
    }
}

/// Records a span the application's OpenTelemetry SDK ended, as its host span would have.
pub fn record_ended(tracer: Tracer, span: EndedSpan, job: Option<String>) {
    let Some(ids) = span.context else { return };
    let host = HostSpan::open(
        tracer,
        span.name,
        span.scope,
        span.parent,
        Some(span.kind),
        span.start_time,
        Some(ids),
        job,
    );
    for (key, value) in span.attributes {
        host.set_attribute(key, value);
    }
    for event in span.events {
        host.add_event(event.name, Some(event.attributes), Some(event.timestamp));
    }
    host.set_status(span.status, span.description);
    host.end(span.end_time);
}

impl lk_agents_abi::telemetry::host::HostSpan for HostSpan {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        name: String,
        scope: String,
        parent: Option<SpanContext>,
        kind: Option<SpanKind>,
        start_time: Option<u64>,
        ids: Option<SpanContext>,
        job: Option<String>,
    ) -> Self {
        HostSpan::open(env.tracer.clone(), name, scope, parent, kind, start_time, ids, job)
    }

    fn context(&self) -> SpanContext {
        self.context
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

    fn set_attributes(&self, attributes: BTreeMap<String, Value>) {
        for (key, value) in attributes {
            self.set_attribute(key, value);
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

    fn set_status(&self, code: StatusCode, description: Option<String>) {
        let mut open = self.open.lock();
        let Some(open) = open.as_mut().filter(|o| o.status != Status::Ok) else { return };
        open.status = match code {
            StatusCode::Ok => Status::Ok,
            StatusCode::Error => Status::Error(description.unwrap_or_default()),
            StatusCode::Unset => Status::Unset,
        };
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
