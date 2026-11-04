//! Framework spans of the async I/O crates (docs/internals/telemetry.md "Span API"): the process
//! [`Tracer`], the context [`Cx`] work runs in, and [`ScopedSpan`], ended when its scope exits.
//! Callers spell every touchpoint `telemetry::`: [`start`], [`end`], [`end_with`], [`check`],
//! [`fail`], [`event`], [`measure`], [`reserve`] and `#[telemetry::span]`.
//! Ended spans and measurements reach the process's `RecordSink` (the OTLP exporter).

pub mod host;
pub mod otlp;
pub mod session_report;

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use lk_agents_abi::Clock;
use lk_agents_abi::telemetry::decl::{
    Attrs, EventDecl, HasEvent, KeyValues, MetricDecl, Number, Parent, SpanDecl, finish_span,
};
use lk_agents_abi::telemetry::record::{
    EventRecord, Key, Measurement, Record, RecordSink, SpanId, SpanRecord, SpanRef, Status,
    TraceContext, TraceId,
};
use lk_agents_abi::telemetry::spans::common::{EverySpan, Exception};
use parking_lot::{Mutex, RwLock};

use otlp::Exporter;

/// `#[telemetry::span(Decl { .. })]` on a function taking `cx: &telemetry::Cx`: the function is
/// that one span.
pub use lk_agents_abi::telemetry::decl::span;
pub use lk_agents_abi::telemetry::errors::{Failure, SpanError};
pub use lk_agents_abi::telemetry::{Cx, Recorder};
pub use lk_agents_core::telemetry::span::Reserved;
use lk_agents_core::telemetry::span::{FRAMEWORK_SCOPE, IdSource, seeded_ids};

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
    ids: Mutex<IdSource>,
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

    /// Removes the installed exporter, so a later install starts a new one.
    pub fn take(&self) -> Option<Arc<Exporter>> {
        self.0.write().take()
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
        Tracer(Arc::new(Inner {
            sink,
            exporter,
            clock,
            ids: Mutex::new(seeded_ids(seed)),
            capture_content,
        }))
    }

    /// The process exporter, for its job registry and configuration; `None` before installation
    /// or for a tracer without one.
    pub fn exporter(&self) -> Option<Arc<Exporter>> {
        self.0.exporter.as_ref()?.get()
    }

    /// A forked child's ids diverge from its parent's.
    pub fn reseed(&self, seed: u64) {
        *self.0.ids.lock() = seeded_ids(seed);
    }

    /// GenAI message capture: whether request and response messages and judge explanations are
    /// recorded on spans.
    pub fn capture_content(&self) -> bool {
        self.0.capture_content()
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
        Cx::root(self.0.clone())
    }

    /// The caller's span at a host-to-framework entry.
    pub fn entry(&self, caller: Option<SpanRef>) -> Cx {
        Cx::entry(self.0.clone(), caller)
    }

    /// The context a [`TraceContext`] carries across a provider request.
    pub fn resume(&self, trace: &TraceContext) -> Cx {
        Cx::resume(self.0.clone(), trace)
    }

    /// Hands a record produced elsewhere (a voice core span) to the sink.
    pub fn record(&self, record: Record) {
        self.0.sink.record(record);
    }

    /// Wall-clock nanoseconds from the injected clock.
    pub fn now_nanos(&self) -> u64 {
        self.0.now_nanos()
    }
}

impl Recorder for Inner {
    fn record(&self, record: Record) {
        self.sink.record(record);
    }

    fn now_nanos(&self) -> u64 {
        nanos(self.clock.wall())
    }

    /// Never zero.
    fn next_id(&self) -> u64 {
        let mut ids = self.ids.lock();
        loop {
            let id = ids();
            if id != 0 {
                return id;
            }
        }
    }

    fn capture_content(&self) -> bool {
        self.capture_content.load(Ordering::Relaxed)
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

fn debug_check_parent<D: SpanDecl>(cx: &Cx) {
    debug_assert!(
        Parent::allows(D::PARENTS, cx.current_name()),
        "{} opened under {:?}; declared parents {:?}",
        D::NAME,
        cx.current_name(),
        D::PARENTS,
    );
}

/// Opens a child of `cx`'s current span.
pub fn start<D: SpanDecl>(cx: &Cx, decl: D) -> ScopedSpan<D> {
    start_at(cx, decl, cx.now_nanos())
}

/// Opens a child of `cx`'s current span that started at `start` (unix nanoseconds): work that
/// began before the process knew of it (a job, from its receipt).
pub fn start_at<D: SpanDecl>(cx: &Cx, decl: D, start: u64) -> ScopedSpan<D> {
    const {
        assert!(!D::MANUAL, "a `handle = Manual` declaration is opened by its owning entity");
    }
    debug_check_parent::<D>(cx);
    let span = cx.child_context();
    let mut attributes = KeyValues::new();
    decl.write(&mut attributes);
    let mut labels = KeyValues::new();
    decl.labels(&mut labels);
    let mut measured = Vec::new();
    decl.measure(&mut measured);
    let open = Open {
        parent: cx.span().map(|s| s.span_id),
        start,
        attributes,
        labels,
        measured,
        events: Vec::new(),
        failure: None,
    };
    ScopedSpan { cx: cx.clone().under::<D>(span), open: Mutex::new(Some(open)), _decl: PhantomData }
}

/// Ends `span`: an `Err` is recorded, an `Ok` ends it with its [`SpanEnd`] attributes. Returns
/// the result unchanged.
pub fn end<D: SpanDecl, T: SpanEnd<D>, E: SpanError>(
    span: ScopedSpan<D>,
    result: Result<T, E>,
) -> Result<T, E> {
    match &result {
        Ok(value) => span.finish(Some(&value.end())),
        Err(error) => span.record_failure(error),
    }
    result
}

/// Ends `span` with explicit end attributes.
pub fn end_with<D: SpanDecl>(span: ScopedSpan<D>, end: D::End) {
    span.finish(Some(&end));
}

/// On `Err`, records the error on `span`, then returns the result unchanged; the span stays open.
pub fn check<D: SpanDecl, T, E: SpanError>(
    span: &ScopedSpan<D>,
    result: Result<T, E>,
) -> Result<T, E> {
    if let Err(error) = &result {
        span.record_failure(error);
    }
    result
}

/// Records a failure on `span` as `record_exception` does: status, `error.type` and an
/// `exception` event. The last failure recorded is the span's status.
pub fn fail<D: SpanDecl>(span: &ScopedSpan<D>, error: &(impl SpanError + ?Sized)) {
    span.record_failure(error);
}

/// Adds a declared event to `span` at the current time.
pub fn event<D: SpanDecl + HasEvent<E>, E: EventDecl>(span: &ScopedSpan<D>, event: E) {
    event_at(span, event, span.cx.now_nanos());
}

/// Adds a declared event to `span` that happened at `time` (unix nanoseconds).
pub fn event_at<D: SpanDecl + HasEvent<E>, E: EventDecl>(
    span: &ScopedSpan<D>,
    event: E,
    time: u64,
) {
    let mut attributes = KeyValues::new();
    event.write(&mut attributes);
    if let Some(open) = span.open.lock().as_mut() {
        open.events.push(EventRecord {
            name: Key::Borrowed(E::NAME),
            time_unix_nanos: time,
            attributes,
        });
    }
}

/// Records one measurement of a declared instrument, for `cx`'s job.
pub fn measure<M: MetricDecl>(cx: &Cx, labels: &M, value: impl Into<Number>) {
    let mut out = KeyValues::new();
    labels.write(&mut out);
    let Some(recorder) = cx.recorder() else { return };
    recorder.record(Record::Measurement(Measurement {
        instrument: M::NAME,
        kind: M::KIND,
        unit: M::UNIT,
        value: value.into(),
        labels: out,
        job: cx.job().cloned(),
    }));
}

/// A child of `cx`'s current span that a voice core opens later, starting now: startup work runs
/// under it before the core exists.
pub fn reserve<D: SpanDecl>(cx: &Cx) -> Reserved<D> {
    debug_check_parent::<D>(cx);
    Reserved::new(cx.child_context(), cx.now_nanos())
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

    fn record_failure(&self, error: &(impl SpanError + ?Sized)) {
        let time = self.cx.now_nanos();
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

    fn finish(&self, end: Option<&D::End>) {
        let Some(open) = self.open.lock().take() else { return };
        let Some(tracer) = self.cx.recorder() else { return };
        let end_time = tracer.now_nanos();
        // Dropped without `end`: cancelled, unless it failed (an error is never a cancellation).
        let every = EverySpan {
            conversation_id: self.cx.conversation().map(str::to_owned),
            error_type: open.failure.as_ref().map(|f| f.error_type.clone()),
            cancelled: (end.is_none() && open.failure.is_none()).then_some(true),
        };
        let span = SpanRecord {
            context: self.cx.span().expect("a span's context is current"),
            parent: open.parent,
            name: Key::Borrowed(D::NAME),
            kind: D::KIND,
            start_unix_nanos: open.start,
            end_unix_nanos: end_time.max(open.start),
            attributes: open.attributes,
            events: open.events,
            links: Vec::new(),
            status: open.failure.map_or(Status::Unset, |f| Status::Error(f.message)),
            scope: Key::Borrowed(FRAMEWORK_SCOPE),
            job: self.cx.job().cloned(),
        };
        finish_span(span, open.labels, open.measured, end, every, &mut |r| tracer.record(r));
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
