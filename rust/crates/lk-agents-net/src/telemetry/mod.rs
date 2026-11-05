//! Framework spans of the async I/O crates (docs/internals/telemetry.md "Span API"): the process
//! [`Tracer`], the context [`Cx`] work runs in, and [`ScopedSpan`], ended when its scope exits.
//! Callers spell every touchpoint `telemetry::`: [`start`], [`end`], [`event`], [`measure`],
//! [`reserve`] and `#[telemetry::span]`.
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
    Attrs, EventDecl, HasEvent, Instrument, KeyValues, MetricDecl, Number, Parent, SpanDecl,
    finish_span,
};
use lk_agents_abi::telemetry::record::{
    EventRecord, FRAMEWORK, Key, Measurement, Record, RecordSink, SpanRecord, Status,
};
use lk_agents_abi::telemetry::spans::common::{EverySpan, Exception};
use parking_lot::{Mutex, RwLock};

use otlp::Exporter;

/// `#[telemetry::span(Decl { .. })]` on a function taking `cx: &telemetry::Cx`: the function is
/// that one span.
pub use lk_agents_abi::telemetry::decl::{Failed, SpanEnd, span};
pub use lk_agents_abi::telemetry::errors::{ErrorType, Failure, SpanError};
pub use lk_agents_abi::telemetry::instruments::{
    AudioCatchUp, AudioConsumer, AudioDropped, AudioPolicy,
};
pub use lk_agents_abi::telemetry::record::SpanContext;
pub use lk_agents_abi::telemetry::{Cx, Recorder};
pub use lk_agents_core::telemetry::span::Reserved;
use lk_agents_core::telemetry::span::{IdSource, seeded_ids};

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
    };
    ScopedSpan { cx: cx.clone().under::<D>(span), open: Mutex::new(Some(open)), _decl: PhantomData }
}

/// Ends `span` with what it ends with ([`SpanEnd`]): its end attributes, an error, a result
/// (`result.as_ref()`: an `Err` fails it) or a [`Failed`] (both).
pub fn end<D: SpanDecl>(span: ScopedSpan<D>, outcome: impl SpanEnd<D>) {
    span.finish(Some(&outcome));
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
        bounds: M::BOUNDS,
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
    parent: Option<u64>,
    start: u64,
    attributes: KeyValues,
    labels: KeyValues,
    measured: Vec<(Instrument, Number)>,
    events: Vec<EventRecord>,
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

    /// Ends the span; `None`: dropped without `end`, cancelled.
    fn finish(&self, outcome: Option<&dyn SpanEnd<D>>) {
        let Some(mut open) = self.open.lock().take() else { return };
        let Some(tracer) = self.cx.recorder() else { return };
        let Some(context) = self.cx.span() else { return };
        let end_time = tracer.now_nanos();
        let error = outcome.and_then(|o| o.error());
        if let Some(error) = error {
            let mut attributes = KeyValues::new();
            Exception {
                kind: error.exception_type().into_owned(),
                message: error.message(),
                stacktrace: error.stacktrace(),
            }
            .write(&mut attributes);
            open.events.push(EventRecord {
                name: Key::Borrowed(<Exception as EventDecl>::NAME),
                time_unix_nanos: end_time,
                attributes,
            });
        }
        let every = EverySpan {
            conversation_id: self.cx.conversation().map(str::to_owned),
            error_type: error.map(|e| e.error_type()),
            cancelled: outcome.is_none().then_some(true),
        };
        let span = SpanRecord {
            context,
            parent: open.parent,
            name: Key::Borrowed(D::NAME),
            kind: D::KIND,
            start_unix_nanos: open.start,
            end_unix_nanos: end_time.max(open.start),
            attributes: open.attributes,
            events: open.events,
            links: Vec::new(),
            status: error.map_or(Status::Unset, |e| Status::Error(e.message())),
            scope: Key::Borrowed(FRAMEWORK),
            job: self.cx.job().cloned(),
        };
        let end = outcome.and_then(|o| o.attributes());
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
