//! Framework spans of the async I/O crates (docs/internals/telemetry.md "Span API"): the process
//! [`Tracer`], the context [`Cx`] work runs in, and [`ScopedSpan`], ended when its scope exits.
//! Ended spans and measurements reach the process's `RecordSink` (the OTLP exporter).

mod errors;
pub mod host;

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use lk_agents_abi::Clock;
use lk_agents_abi::telemetry::decl::{
    Attrs, EventDecl, HasEvent, InstrumentKind, KeyValues, MetricDecl, Number, SpanDecl,
};
use lk_agents_abi::telemetry::record::{
    EventRecord, JobKey, Key, Measurement, Record, RecordSink, SpanId, SpanRecord, SpanRef, Status,
    TraceId,
};
use lk_agents_abi::telemetry::spans::common::{EverySpan, Exception};
use parking_lot::{Mutex, RwLock};

use crate::otlp::Exporter;

pub use errors::{Failure, SpanError};

/// The instrumentation scope of framework spans.
pub const SCOPE: &str = "livekit-agents";

/// What an `Ok` value of a `#[span]` function ends its span with.
pub trait SpanEnd<D: SpanDecl> {
    fn end(&self) -> D::End;
}

/// The process's span recording: where records go, the clock that times them and the id source.
/// One per process, lent through `ProcessEnv`; the DST builds its own with a fixed seed.
#[derive(Clone)]
pub struct Tracer(Arc<Inner>);

struct Inner {
    sink: Arc<dyn RecordSink>,
    exporter: Option<Arc<ExporterSlot>>,
    clock: Arc<dyn Clock>,
    ids: Mutex<u64>,
    capture_content: AtomicBool,
}

struct Discard;

impl RecordSink for Discard {
    fn record(&self, _: Record) {}
}

/// The process's exporter once the host installed telemetry: what the process tracer records
/// to. Records before the installation are discarded.
#[derive(Default)]
pub struct ExporterSlot(RwLock<Option<Arc<Exporter>>>);

impl ExporterSlot {
    pub fn get(&self) -> Option<Arc<Exporter>> {
        self.0.read().clone()
    }

    /// Installs the exporter `start` builds, unless one is installed.
    pub fn install(&self, start: impl FnOnce() -> Exporter) -> Arc<Exporter> {
        self.0.write().get_or_insert_with(|| Arc::new(start())).clone()
    }

    /// Abandons a fork-inherited exporter without dropping it: its task belongs to the parent's
    /// runtime.
    pub fn forget(&self) {
        std::mem::forget(self.0.write().take());
    }
}

impl RecordSink for ExporterSlot {
    fn record(&self, record: Record) {
        if let Some(exporter) = self.0.read().as_ref() {
            exporter.record(record);
        }
    }
}

impl Tracer {
    pub fn new(sink: Arc<dyn RecordSink>, clock: Arc<dyn Clock>, seed: u64) -> Self {
        Tracer::build(sink, None, clock, seed)
    }

    /// The process's tracer, recording to the exporter `slot` installs.
    pub fn process(slot: Arc<ExporterSlot>, clock: Arc<dyn Clock>, seed: u64) -> Self {
        Tracer::build(slot.clone(), Some(slot), clock, seed)
    }

    fn build(
        sink: Arc<dyn RecordSink>,
        exporter: Option<Arc<ExporterSlot>>,
        clock: Arc<dyn Clock>,
        seed: u64,
    ) -> Self {
        let capture_content = AtomicBool::new(true);
        Tracer(Arc::new(Inner { sink, exporter, clock, ids: Mutex::new(seed), capture_content }))
    }

    /// The process exporter, for its job registry and configuration; `None` before installation
    /// or for a tracer without one.
    pub fn exporter(&self) -> Option<Arc<Exporter>> {
        self.0.exporter.as_ref()?.get()
    }

    /// A forked child's ids diverge from its parent's.
    pub fn reseed(&self, seed: u64) {
        *self.0.ids.lock() = seed;
    }

    /// GenAI message capture: whether request and response messages and judge explanations are
    /// recorded on spans.
    pub fn capture_content(&self) -> bool {
        self.0.capture_content.load(Ordering::Relaxed)
    }

    pub fn set_capture_content(&self, enabled: bool) {
        self.0.capture_content.store(enabled, Ordering::Relaxed);
    }

    /// Records nothing: a Rust caller with no exporter.
    pub fn disabled(clock: Arc<dyn Clock>) -> Self {
        Tracer::new(Arc::new(Discard), clock, 1)
    }

    /// A context with no current span, outside any job.
    pub fn root(&self) -> Cx {
        Cx { tracer: self.clone(), current: None, job: None, conversation: None }
    }

    /// Hands a record produced elsewhere (a voice core span) to the sink.
    pub fn record(&self, record: Record) {
        self.0.sink.record(record);
    }

    /// Wall-clock nanoseconds from the injected clock.
    pub fn now_nanos(&self) -> u64 {
        nanos(self.0.clock.wall())
    }

    /// splitmix64: never zero.
    fn next_id(&self) -> u64 {
        let mut state = self.0.ids.lock();
        loop {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if z != 0 {
                return z;
            }
        }
    }
}

/// A W3C `traceparent` header's span; `None` for an invalid one, which never refuses a request.
pub fn parse_traceparent(value: &str) -> Option<SpanRef> {
    let parent = lk_agents_core::telemetry::transport::TraceParent::parse(value)?;
    Some(SpanRef {
        trace_id: TraceId(u128::from_be_bytes(parent.trace_id)),
        span_id: SpanId(u64::from_be_bytes(parent.span_id)),
        sampled: parent.sampled,
    })
}

/// `span` as a W3C `traceparent` header.
pub fn traceparent(span: SpanRef) -> Option<String> {
    lk_agents_core::telemetry::transport::TraceParent::format(
        span.trace_id.0,
        span.span_id.0,
        span.sampled,
    )
}

/// A process tracer's seed.
pub fn random_seed() -> u64 {
    rand::random()
}

fn nanos(at: SystemTime) -> u64 {
    let since = at.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    since.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// The span work runs under, explicitly passed: nothing is entered into a thread-local, so the
/// context stays correct across `.await` and concurrent tasks.
#[derive(Clone)]
pub struct Cx {
    tracer: Tracer,
    current: Option<Current>,
    job: Option<JobKey>,
    conversation: Option<Arc<str>>,
}

#[derive(Clone, Copy)]
struct Current {
    span: SpanRef,
    /// The declared span's name; `None` for a span from outside the framework (an application
    /// span, a `traceparent`).
    name: Option<&'static str>,
}

impl Cx {
    pub fn tracer(&self) -> &Tracer {
        &self.tracer
    }

    /// The current span, what a host call activates and a `traceparent` carries.
    pub fn span(&self) -> Option<SpanRef> {
        self.current.map(|c| c.span)
    }

    pub fn job(&self) -> Option<&JobKey> {
        self.job.as_ref()
    }

    /// The same context, recording for `job`.
    pub fn with_job(mut self, job: Option<JobKey>) -> Cx {
        self.job = job;
        self
    }

    /// The same context, its spans carrying `gen_ai.conversation.id`.
    pub fn with_conversation(mut self, conversation: Option<&str>) -> Cx {
        self.conversation = conversation.map(Arc::from);
        self
    }

    /// The same context under `span`, a framework span of type `P` recorded elsewhere (a voice
    /// effect's `SpanRef`).
    pub fn under<P: SpanDecl>(mut self, span: SpanRef) -> Cx {
        self.current = Some(Current { span, name: Some(P::NAME) });
        self
    }

    /// The same context under a span from outside the framework: an application span captured
    /// at a host entry, or a `traceparent`. `None` starts new traces.
    pub fn continuing(mut self, span: Option<SpanRef>) -> Cx {
        self.current = span.map(|span| Current { span, name: None });
        self
    }

    /// Opens a child of the current span.
    pub fn start_span<D: SpanDecl>(&self, decl: D) -> ScopedSpan<D> {
        self.start_span_at(decl, self.tracer.now_nanos())
    }

    /// Opens a child of the current span that started at `start` (unix nanoseconds): work that
    /// began before the process knew of it (a job, from its receipt).
    pub fn start_span_at<D: SpanDecl>(&self, decl: D, start: u64) -> ScopedSpan<D> {
        debug_assert!(
            parent_declared(D::PARENTS, self.current.map(|c| c.name)),
            "{} opened under {:?}; declared parents {:?}",
            D::NAME,
            self.current.map(|c| c.name),
            D::PARENTS,
        );
        let span = self.new_context();
        let mut attributes = KeyValues::new();
        decl.write(&mut attributes);
        let mut labels = KeyValues::new();
        decl.labels(&mut labels);
        let mut measured = Vec::new();
        decl.measure(&mut measured);
        let open = Open {
            parent: self.current.map(|c| c.span.span_id),
            start,
            attributes,
            labels,
            measured,
            events: Vec::new(),
            failure: None,
        };
        ScopedSpan {
            cx: Cx { current: Some(Current { span, name: Some(D::NAME) }), ..self.clone() },
            open: Mutex::new(Some(open)),
            _decl: PhantomData,
        }
    }

    /// A new span's context under the current span: its trace, or a new one.
    fn new_context(&self) -> SpanRef {
        let tracer = &self.tracer;
        let trace_id = match self.current {
            Some(c) => c.span.trace_id,
            None => TraceId((u128::from(tracer.next_id()) << 64) | u128::from(tracer.next_id())),
        };
        SpanRef {
            trace_id,
            span_id: SpanId(tracer.next_id()),
            sampled: self.current.is_none_or(|c| c.span.sampled),
        }
    }

    /// Records one measurement of a declared instrument, for this context's job.
    pub fn measure<M: MetricDecl>(&self, labels: &M, value: impl Into<Number>) {
        let mut out = KeyValues::new();
        labels.write(&mut out);
        self.tracer.record(Record::Measurement(Measurement {
            instrument: M::NAME,
            kind: M::KIND,
            unit: M::UNIT,
            value: value.into(),
            labels: out,
            job: self.job.clone(),
        }));
    }
}

/// `Current` accepts any parent; `External` a missing or foreign one; otherwise the parent must
/// be one of the declared span names.
fn parent_declared(parents: &[&str], current: Option<Option<&'static str>>) -> bool {
    use lk_agents_abi::telemetry::decl::{Current as AnyParent, External, SpanParent};
    parents.contains(&<AnyParent as SpanParent>::PARENT_NAME)
        || match current {
            None | Some(None) => parents.contains(&<External as SpanParent>::PARENT_NAME),
            Some(Some(name)) => parents.contains(&name),
        }
}

struct Open {
    parent: Option<SpanId>,
    start: u64,
    attributes: KeyValues,
    labels: KeyValues,
    measured: Vec<(&'static str, Number)>,
    events: Vec<EventRecord>,
    failure: Option<Recorded>,
}

struct Recorded {
    error_type: String,
    message: String,
}

/// A span ended when dropped, as cancelled, unless ended before (docs "ScopedSpan").
#[must_use = "a span is ended when dropped"]
pub struct ScopedSpan<D: SpanDecl> {
    cx: Cx,
    open: Mutex<Option<Open>>,
    _decl: PhantomData<fn(D)>,
}

impl<D: SpanDecl> ScopedSpan<D> {
    /// The child context. It borrows the span, so the span outlives every use of it.
    pub fn cx(&self) -> &Cx {
        &self.cx
    }

    /// On `Err`, records the error, then returns the result unchanged.
    pub fn check<T, E: SpanError>(&self, result: Result<T, E>) -> Result<T, E> {
        if let Err(error) = &result {
            self.fail(error);
        }
        result
    }

    /// Records a failure as `record_exception` does: status, `error.type` and an `exception`
    /// event. The last failure recorded is the span's status.
    pub fn fail(&self, error: &(impl SpanError + ?Sized)) {
        let time = self.cx.tracer.now_nanos();
        let error_type = error.error_type();
        let message = error.message();
        let mut attributes = KeyValues::new();
        Exception {
            kind: error_type.clone(),
            message: message.clone(),
            stacktrace: error.stacktrace(),
        }
        .write(&mut attributes);
        if let Some(open) = self.open.lock().as_mut() {
            open.events.push(EventRecord {
                name: Key::Borrowed(<Exception as EventDecl>::NAME),
                time_unix_nanos: time,
                attributes,
            });
            open.failure = Some(Recorded { error_type, message });
        }
    }

    /// Adds a declared event at the current time.
    pub fn event<E: EventDecl>(&self, event: E)
    where
        D: HasEvent<E>,
    {
        self.event_at(event, self.cx.tracer.now_nanos());
    }

    /// Adds a declared event that happened at `time` (unix nanoseconds).
    pub fn event_at<E: EventDecl>(&self, event: E, time: u64)
    where
        D: HasEvent<E>,
    {
        let mut attributes = KeyValues::new();
        event.write(&mut attributes);
        if let Some(open) = self.open.lock().as_mut() {
            open.events.push(EventRecord {
                name: Key::Borrowed(E::NAME),
                time_unix_nanos: time,
                attributes,
            });
        }
    }

    /// Records the end attributes and ends the span.
    pub fn end(self, end: D::End) {
        self.finish(Some(&end));
    }

    fn finish(&self, end: Option<&D::End>) {
        let Some(mut open) = self.open.lock().take() else { return };
        let tracer = &self.cx.tracer;
        let end_time = tracer.now_nanos();
        if let Some(end) = end {
            end.write(&mut open.attributes);
            end.labels(&mut open.labels);
            end.measure(&mut open.measured);
        }
        // Dropped without `end`: cancelled, unless it failed (an error is never a cancellation).
        let cancelled = (end.is_none() && open.failure.is_none()).then_some(true);
        EverySpan {
            conversation_id: self.cx.conversation.as_deref().map(str::to_owned),
            error_type: open.failure.as_ref().map(|f| f.error_type.clone()),
            cancelled,
        }
        .write(&mut open.attributes);
        let job = self.cx.job.clone();
        if end.is_some() {
            // A span end field is one sample of a duration distribution, in seconds.
            for (instrument, value) in open.measured {
                tracer.record(Record::Measurement(Measurement {
                    instrument,
                    kind: InstrumentKind::Histogram,
                    unit: "s",
                    value,
                    labels: open.labels.clone(),
                    job: job.clone(),
                }));
            }
        }
        let context = self.cx.span().expect("a span's context is current");
        tracer.record(Record::Span(SpanRecord {
            context,
            parent: open.parent,
            name: Key::Borrowed(D::NAME),
            kind: D::KIND,
            start_unix_nanos: open.start,
            end_unix_nanos: end_time.max(open.start),
            attributes: open.attributes,
            events: open.events,
            links: Vec::new(),
            status: open.failure.map_or(Status::Unset, |f| Status::Error(f.message)),
            scope: Key::Borrowed(SCOPE),
            job,
        }));
    }
}

impl<D: SpanDecl> Drop for ScopedSpan<D> {
    fn drop(&mut self) {
        self.finish(None);
    }
}

#[cfg(test)]
#[path = "span_tests.rs"]
mod tests;
