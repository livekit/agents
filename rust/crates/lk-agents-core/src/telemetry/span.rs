//! Spans owned by sans-I/O state machines (`telemetry.md`, "Sans-I/O: the entity owns its span").
//! The entity a span describes holds it in a field named `span` ([`Span`]) and the machine's
//! [`Trace`] (`self.telemetry`) opens and ends it: [`Trace::step`] sets the step's instant once,
//! [`Trace::open`] / [`Trace::open_in`] open, [`Trace::end`] ends. A declaration's `#[at]` field
//! dates a start, an end or an event before the step that records it. Ended spans leave as
//! [`Record`]s the caller drains each step.

use lk_agents_abi::Ts;
use lk_agents_abi::errors::{ApiErrorKind, HostError, ProviderError};
use lk_agents_abi::telemetry::decl::{
    Attrs, Current, EventDecl, HasEvent, MetricDecl, ParentOf, SpanDecl,
};
use lk_agents_abi::telemetry::record::{
    EventRecord, InstrumentKind, JobKey, Key, KeyValues, Measurement, Number, Record, SpanId,
    SpanKind, SpanRecord, SpanRef, Status, TraceId,
};
use lk_agents_abi::telemetry::spans::common::{EverySpan, Exception};
use lk_agents_abi::telemetry::spans::session::AgentSession;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::time::{Duration, SystemTime};

/// The instrumentation scope of every framework span.
pub const FRAMEWORK_SCOPE: &str = "livekit-agents";

/// Where span and trace ids come from: random in production, seeded in the DST.
pub type IdSource = Box<dyn FnMut() -> u64 + Send>;

/// A deterministic id source (splitmix64).
pub fn seeded_ids(seed: u64) -> IdSource {
    let mut state = seed;
    Box::new(move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

/// A failure a span records, as `record_exception` writes it (`telemetry.md`, "Errors").
pub trait SpanFailure {
    /// `error.type` and `exception.type`.
    fn error_type(&self) -> String;
    fn message(&self) -> String;
    fn stacktrace(&self) -> Option<String> {
        None
    }
}

impl SpanFailure for HostError {
    fn error_type(&self) -> String {
        if self.kind.is_empty() { "HostError".to_owned() } else { self.kind.clone() }
    }
    fn message(&self) -> String {
        self.message.clone()
    }
    fn stacktrace(&self) -> Option<String> {
        self.traceback.clone()
    }
}

impl SpanFailure for ProviderError {
    fn error_type(&self) -> String {
        match self {
            ProviderError::Host { error, .. } => error.error_type(),
            ProviderError::Local(error) => match error.kind {
                ApiErrorKind::Api => "api_error",
                ApiErrorKind::Status => "status_error",
                ApiErrorKind::Connection => "connection_error",
                ApiErrorKind::Timeout => "timeout",
                ApiErrorKind::Cancelled => "cancelled",
            }
            .to_owned(),
        }
    }
    fn message(&self) -> String {
        match self {
            ProviderError::Host { error, .. } => error.message.clone(),
            ProviderError::Local(error) => error.message.clone(),
        }
    }
    fn stacktrace(&self) -> Option<String> {
        match self {
            ProviderError::Host { error, .. } => error.stacktrace(),
            ProviderError::Local(_) => None,
        }
    }
}

/// A span context allocated before its owner opens it: startup work in the runtime runs under
/// it before the voice core learns the session started, so the core opens the span with this
/// context and start instead of drawing its own.
pub struct Reserved<D: SpanDecl> {
    pub context: SpanRef,
    /// Unix nanoseconds: when the work it describes began.
    pub start: u64,
    _decl: PhantomData<fn() -> D>,
}

impl<D: SpanDecl> Reserved<D> {
    pub fn new(context: SpanRef, start: u64) -> Self {
        Reserved { context, start, _decl: PhantomData }
    }
}

impl<D: SpanDecl> Clone for Reserved<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: SpanDecl> Copy for Reserved<D> {}

impl<D: SpanDecl> std::fmt::Debug for Reserved<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reserved").field("name", &D::NAME).field("span", &self.context).finish()
    }
}

impl<D: SpanDecl> PartialEq for Reserved<D> {
    fn eq(&self, other: &Self) -> bool {
        self.context == other.context && self.start == other.start
    }
}

/// The handle the entity a span describes owns, in a field named `span`.
pub type Span<D> = ManualSpan<D>;

/// What an open span recorded so far.
struct Open {
    context: SpanRef,
    parent: Option<SpanId>,
    name: &'static str,
    kind: SpanKind,
    start: u64,
    attributes: KeyValues,
    labels: KeyValues,
    events: Vec<EventRecord>,
    status: Status,
    error_type: Option<String>,
    /// The latest end of its children: a backdated end never ends it before them.
    children_end: u64,
    /// Opening order, for the close sweep.
    order: u64,
}

/// The session's span state: the id source, the wall-clock anchor, the open spans and the ended
/// ones not yet drained.
pub struct Trace {
    ids: IdSource,
    /// The step's instant, set once at the machine's step entry.
    now: Ts,
    /// The machine's root span (`agent_session`): the parent [`Trace::open`] implies.
    root: Option<SpanRef>,
    /// The root ended: spans opened after it record nothing.
    closed: bool,
    anchor: (Ts, SystemTime),
    job: Option<JobKey>,
    conversation_id: Option<String>,
    open: BTreeMap<SpanId, Open>,
    opened: u64,
    swept: u32,
    records: Vec<Record>,
}

impl Trace {
    /// `at` and `wall` are one instant on the monotonic and wall clocks (the `ProcessEnv` clock).
    pub fn new(
        ids: IdSource,
        at: Ts,
        wall: SystemTime,
        job: Option<JobKey>,
        conversation_id: Option<String>,
    ) -> Self {
        Trace {
            ids,
            now: at,
            root: None,
            closed: false,
            anchor: (at, wall),
            job,
            conversation_id,
            open: BTreeMap::new(),
            opened: 0,
            swept: 0,
            records: Vec::new(),
        }
    }

    /// The step's instant, once at the machine's step entry. No other call takes a time.
    pub fn step(&mut self, now: Ts) {
        self.now = self.now.max(now);
    }

    /// The step's instant.
    pub fn now(&self) -> Ts {
        self.now
    }

    /// Opens the machine's root span, under `parent` (a span outside this trace) or as a trace
    /// root, with the reservation's context and start when one was made: what
    /// [`Trace::open`] implies as parent.
    pub fn open_root<D: SpanDecl>(
        &mut self,
        parent: Option<SpanRef>,
        reserved: Option<Reserved<D>>,
        decl: D,
    ) -> Span<D>
    where
        Current: ParentOf<D>,
    {
        let span = match reserved {
            Some(r) => self.start(r.start, parent, Some(r.context), decl),
            None => {
                let at = self.start_of(None, &decl);
                self.start(at, parent, None, decl)
            }
        };
        self.root = Some(span.context);
        span
    }

    /// Opens a span under the root, at the step's instant or its declaration's `#[at]`.
    pub fn open<D: SpanDecl>(&mut self, decl: D) -> Span<D>
    where
        AgentSession: ParentOf<D>,
    {
        let Some(root) = self.root.filter(|_| !self.closed) else { return Span::dormant() };
        let at = self.start_of(Some(root), &decl);
        self.start(at, Some(root), None, decl)
    }

    /// Opens a span under the root reserved before this trace knew of it, with the reservation's
    /// context and start.
    pub fn open_reserved<D: SpanDecl>(&mut self, reserved: Reserved<D>, decl: D) -> Span<D>
    where
        AgentSession: ParentOf<D>,
    {
        let Some(root) = self.root.filter(|_| !self.closed) else { return Span::dormant() };
        let at = reserved.start.max(self.floor(root));
        self.start(at, Some(root), Some(reserved.context), decl)
    }

    /// Opens a child of `parent`, at the step's instant or its declaration's `#[at]`, clamped to
    /// the parent's start.
    pub fn open_in<P: SpanDecl, D: SpanDecl>(&mut self, parent: &Span<P>, decl: D) -> Span<D>
    where
        P: ParentOf<D>,
    {
        if !parent.recording {
            return Span::dormant();
        }
        let at = self.start_of(Some(parent.context), &decl);
        self.start(at, Some(parent.context), None, decl)
    }

    /// Opens a child of `parent` reserved before this trace knew of it, with the reservation's
    /// context and start, clamped to the parent's start.
    pub fn open_in_reserved<P: SpanDecl, D: SpanDecl>(
        &mut self,
        parent: &Span<P>,
        reserved: Reserved<D>,
        decl: D,
    ) -> Span<D>
    where
        P: ParentOf<D>,
    {
        if !parent.recording {
            return Span::dormant();
        }
        let at = reserved.start.max(self.floor(parent.context));
        self.start(at, Some(parent.context), Some(reserved.context), decl)
    }

    /// Ends `span` with its end attributes, at the step's instant or the end's `#[at]`.
    pub fn end<D: SpanDecl>(&mut self, mut span: Span<D>, end: D::End) {
        span.ended = true;
        let Some(mut open) = self.open.remove(&span.context.span_id) else { return };
        let at = self.wall(end.at().unwrap_or(self.now));
        let mut attributes = KeyValues::new();
        end.write(&mut attributes);
        end.labels(&mut open.labels);
        let mut measured = Vec::new();
        end.measure(&mut measured);
        self.finish(open, at, attributes, measured, false);
    }

    /// Adds a declared event, at the step's instant or the event's `#[at]`.
    pub fn event<D: SpanDecl, E: EventDecl>(&mut self, span: &Span<D>, event: E)
    where
        D: HasEvent<E>,
    {
        let at = self.wall(event.at().unwrap_or(self.now));
        push_event(self, span.context.span_id, at, E::NAME, &event);
    }

    /// Records a failure on `span`: status `ERROR`, `error.type` and an `exception` event.
    pub fn fail<D: SpanDecl>(&mut self, span: &Span<D>, error: &dyn SpanFailure) {
        let at = self.wall(self.now);
        let exception = Exception {
            kind: error.error_type(),
            message: error.message(),
            stacktrace: error.stacktrace(),
        };
        push_event(self, span.context.span_id, at, Exception::NAME, &exception);
        if let Some(open) = self.open.get_mut(&span.context.span_id) {
            open.status = Status::Error(exception.message);
            open.error_type = Some(exception.kind);
        }
    }

    /// Replaces `span`'s start attributes, as OpenTelemetry's `set_attributes` does on a live
    /// span: for what the entity learns after the span opened.
    pub fn set<D: SpanDecl>(&mut self, span: &Span<D>, decl: D) {
        let Some(open) = self.open.get_mut(&span.context.span_id) else { return };
        open.attributes.clear();
        decl.write(&mut open.attributes);
        open.labels.clear();
        decl.labels(&mut open.labels);
    }

    /// What a child, a link or an effect of `span` carries; `None` for a span that records
    /// nothing (opened outside the root's life).
    pub fn context<D: SpanDecl>(&self, span: &Span<D>) -> Option<SpanRef> {
        span.recording.then_some(span.context)
    }

    /// The root ended: later spans record nothing, and the sweep ends what is still open.
    pub fn close_root(&mut self) -> u32 {
        self.closed = true;
        let now = self.now;
        self.close(now)
    }

    /// A span's start: its declaration's `#[at]`, else the step's instant, never before its
    /// parent's start.
    fn start_of<D: SpanDecl>(&self, parent: Option<SpanRef>, decl: &D) -> u64 {
        let at = self.wall(decl.at().unwrap_or(self.now));
        parent.map_or(at, |p| at.max(self.floor(p)))
    }

    /// The start of open span `span`, or 0.
    fn floor(&self, span: SpanRef) -> u64 {
        self.open.get(&span.span_id).map_or(0, |o| o.start)
    }

    /// Ended spans and measurements since the last drain.
    pub fn drain(&mut self) -> Vec<Record> {
        std::mem::take(&mut self.records)
    }

    /// Ends every span still open, child first, as cancelled. The owner ends its spans before
    /// closing; a span reaching this sweep is a bug the DST reports. Returns how many it ended.
    pub fn close(&mut self, now: Ts) -> u32 {
        let at = self.wall(now);
        let depth = |id: SpanId, open: &BTreeMap<SpanId, Open>| {
            let mut depth = 0u32;
            let mut parent = open.get(&id).and_then(|o| o.parent);
            while let Some(p) = parent.and_then(|p| open.get(&p)) {
                depth += 1;
                parent = p.parent;
            }
            depth
        };
        let mut ids: Vec<(u32, u64, SpanId)> =
            self.open.iter().map(|(id, o)| (depth(*id, &self.open), o.order, *id)).collect();
        ids.sort_by(|a, b| b.cmp(a));
        let swept = ids.len() as u32;
        for (_, _, id) in ids {
            if let Some(open) = self.open.remove(&id) {
                self.finish(open, at, KeyValues::new(), Vec::new(), true);
            }
        }
        self.swept += swept;
        swept
    }

    /// Spans the close sweep has ended so far.
    pub fn swept(&self) -> u32 {
        self.swept
    }

    /// Records one measurement of a declared instrument, labeled by `metric`.
    pub fn measure<M: MetricDecl>(&mut self, metric: &M, value: Number) {
        let mut labels = KeyValues::new();
        metric.write(&mut labels);
        self.record::<M>(labels, value);
    }

    /// Records one measurement of `M` without its labels, when their values are unknown.
    pub fn measure_unlabeled<M: MetricDecl>(&mut self, value: Number) {
        self.record::<M>(KeyValues::new(), value);
    }

    fn record<M: MetricDecl>(&mut self, labels: KeyValues, value: Number) {
        self.records.push(Record::Measurement(Measurement {
            instrument: M::NAME,
            kind: M::KIND,
            unit: M::UNIT,
            value,
            labels,
            job: self.job.clone(),
        }));
    }

    /// Unix nanoseconds of a session instant.
    fn wall(&self, ts: Ts) -> u64 {
        let (at, wall) = self.anchor;
        let wall = if ts >= at {
            wall + ts.saturating_duration_since(at)
        } else {
            wall.checked_sub(at.saturating_duration_since(ts)).unwrap_or(wall)
        };
        let since = wall.duration_since(SystemTime::UNIX_EPOCH).unwrap_or(Duration::ZERO);
        u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
    }

    fn next_id(&mut self) -> u64 {
        loop {
            let id = (self.ids)();
            if id != 0 {
                return id;
            }
        }
    }

    fn start<D: SpanDecl>(
        &mut self,
        at: u64,
        parent: Option<SpanRef>,
        reserved: Option<SpanRef>,
        decl: D,
    ) -> ManualSpan<D> {
        const {
            assert!(D::MANUAL, "a ManualSpan opens only a `handle = Manual` declaration");
        }
        debug_assert!(
            reserved.zip(parent).is_none_or(|(r, p)| r.trace_id == p.trace_id),
            "{} reserved in another trace than its parent's",
            D::NAME
        );
        let context = reserved.unwrap_or_else(|| {
            let trace_id = match parent {
                Some(p) => p.trace_id,
                None => TraceId(u128::from(self.next_id()) << 64 | u128::from(self.next_id())),
            };
            SpanRef {
                trace_id,
                span_id: SpanId(self.next_id()),
                sampled: parent.is_none_or(|p| p.sampled),
            }
        });
        let mut attributes = KeyValues::new();
        decl.write(&mut attributes);
        let mut labels = KeyValues::new();
        decl.labels(&mut labels);
        self.opened += 1;
        let open = Open {
            context,
            parent: parent.map(|p| p.span_id),
            name: D::NAME,
            kind: D::KIND,
            start: at,
            attributes,
            labels,
            events: Vec::new(),
            status: Status::Unset,
            error_type: None,
            children_end: at,
            order: self.opened,
        };
        self.open.insert(context.span_id, open);
        ManualSpan { context, ended: false, recording: true, _decl: PhantomData }
    }

    fn finish(&mut self, open: Open, at: u64, end: KeyValues, measured: Measured, swept: bool) {
        let Open {
            context,
            parent,
            name,
            kind,
            start,
            mut attributes,
            labels,
            events,
            status,
            error_type,
            children_end,
            order: _,
        } = open;
        let at = at.max(children_end);
        if let Some(parent) = parent.and_then(|p| self.open.get_mut(&p)) {
            parent.children_end = parent.children_end.max(at);
        }
        attributes.extend(end);
        EverySpan {
            conversation_id: self.conversation_id.clone(),
            error_type,
            cancelled: swept.then_some(true),
        }
        .write(&mut attributes);
        for (instrument, value) in measured {
            self.records.push(Record::Measurement(Measurement {
                instrument,
                kind: InstrumentKind::Histogram,
                unit: "s",
                value,
                labels: labels.clone(),
                job: self.job.clone(),
            }));
        }
        self.records.push(Record::Span(SpanRecord {
            context,
            parent,
            name: Key::Borrowed(name),
            kind,
            start_unix_nanos: start,
            end_unix_nanos: at,
            attributes,
            events,
            links: Vec::new(),
            status,
            scope: Key::Borrowed(FRAMEWORK_SCOPE),
            job: self.job.clone(),
        }));
    }
}

type Measured = Vec<(&'static str, Number)>;

/// A span ended only by the entity that owns it (`telemetry.md`, "Two span handles").
#[must_use = "a ManualSpan is ended by its owner"]
pub struct ManualSpan<D: SpanDecl> {
    context: SpanRef,
    ended: bool,
    /// `false`: opened outside its trace's root life; it records nothing.
    recording: bool,
    _decl: PhantomData<fn() -> D>,
}

impl<D: SpanDecl> std::fmt::Debug for ManualSpan<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManualSpan").field("name", &D::NAME).field("span", &self.context).finish()
    }
}

impl<D: SpanDecl> ManualSpan<D> {
    /// A span opened outside its trace's root life: every call on it records nothing.
    fn dormant() -> Self {
        let none = SpanRef { trace_id: TraceId(0), span_id: SpanId(0), sampled: false };
        ManualSpan { context: none, ended: false, recording: false, _decl: PhantomData }
    }

    /// What a child, a link or an effect carries.
    pub fn context(&self) -> SpanRef {
        self.context
    }

    /// Opens a child at `now`.
    pub fn child<C: SpanDecl>(&self, trace: &mut Trace, now: Ts, decl: C) -> ManualSpan<C>
    where
        D: ParentOf<C>,
    {
        self.child_at(trace, now, decl)
    }

    /// Opens a child at `started_at`, the time of its first evidence, clamped to this span's
    /// start.
    pub fn child_at<C: SpanDecl>(&self, trace: &mut Trace, started_at: Ts, decl: C) -> ManualSpan<C>
    where
        D: ParentOf<C>,
    {
        let floor = trace.open.get(&self.context.span_id).map_or(0, |o| o.start);
        let at = trace.wall(started_at).max(floor);
        trace.start(at, Some(self.context), None, decl)
    }

    /// Opens a child reserved before this trace knew of it, with the reservation's context and
    /// start, clamped to this span's start.
    pub fn child_reserved<C: SpanDecl>(
        &self,
        trace: &mut Trace,
        reserved: Reserved<C>,
        decl: C,
    ) -> ManualSpan<C>
    where
        D: ParentOf<C>,
    {
        let floor = trace.open.get(&self.context.span_id).map_or(0, |o| o.start);
        trace.start(reserved.start.max(floor), Some(self.context), Some(reserved.context), decl)
    }

    /// The span's start, unix nanoseconds; `None` once ended.
    pub fn started(&self, trace: &Trace) -> Option<u64> {
        trace.open.get(&self.context.span_id).map(|o| o.start)
    }

    /// Replaces the start attributes, as OpenTelemetry's `set_attributes` does on a live span:
    /// for what the entity learns after the span opened (a re-armed wait's source, the speech
    /// that adopted a turn).
    pub fn set(&self, trace: &mut Trace, decl: D) {
        let Some(open) = trace.open.get_mut(&self.context.span_id) else { return };
        open.attributes.clear();
        decl.write(&mut open.attributes);
        open.labels.clear();
        decl.labels(&mut open.labels);
    }

    /// Adds a declared event at `now`.
    pub fn event<E: EventDecl>(&self, trace: &mut Trace, now: Ts, event: E)
    where
        D: HasEvent<E>,
    {
        let at = trace.wall(now);
        push_event(trace, self.context.span_id, at, E::NAME, &event);
    }

    /// Records a failure: status `ERROR`, `error.type` and an `exception` event.
    pub fn fail(&self, trace: &mut Trace, now: Ts, error: &dyn SpanFailure) {
        let at = trace.wall(now);
        let exception = Exception {
            kind: error.error_type(),
            message: error.message(),
            stacktrace: error.stacktrace(),
        };
        push_event(trace, self.context.span_id, at, Exception::NAME, &exception);
        if let Some(open) = trace.open.get_mut(&self.context.span_id) {
            open.status = Status::Error(exception.message);
            open.error_type = Some(exception.kind);
        }
    }

    /// Ends the span with its end attributes. A span the close sweep already ended stays ended.
    pub fn end(mut self, trace: &mut Trace, now: Ts, end: D::End) {
        self.ended = true;
        let Some(mut open) = trace.open.remove(&self.context.span_id) else { return };
        let mut attributes = KeyValues::new();
        end.write(&mut attributes);
        end.labels(&mut open.labels);
        let mut measured = Vec::new();
        end.measure(&mut measured);
        let at = trace.wall(now);
        trace.finish(open, at, attributes, measured, false);
    }
}

fn push_event(trace: &mut Trace, span: SpanId, at: u64, name: &'static str, event: &dyn Attrs) {
    let Some(open) = trace.open.get_mut(&span) else { return };
    let mut attributes = KeyValues::new();
    event.write(&mut attributes);
    let time_unix_nanos = at.max(open.start);
    open.events.push(EventRecord { name: Key::Borrowed(name), time_unix_nanos, attributes });
}

impl<D: SpanDecl> Drop for ManualSpan<D> {
    fn drop(&mut self) {
        debug_assert!(
            self.ended || std::thread::panicking(),
            "the {} span was dropped without end",
            D::NAME
        );
    }
}

#[cfg(test)]
#[path = "span_tests.rs"]
mod tests;
