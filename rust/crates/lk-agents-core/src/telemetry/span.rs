//! Spans owned by sans-I/O state machines (`telemetry.md`, "`ManualSpan`"). A [`ManualSpan`] lives
//! as long as the entity it describes and ends only by an explicit [`ManualSpan::end`]; the
//! session's [`Trace`] holds what every open span recorded so far, dates it from the step's
//! monotonic `now` and hands ended spans out as [`Record`]s the caller drains each step.

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
            anchor: (at, wall),
            job,
            conversation_id,
            open: BTreeMap::new(),
            opened: 0,
            swept: 0,
            records: Vec::new(),
        }
    }

    /// Opens a span that follows its caller: under `parent` (a span outside this trace), or as
    /// a trace root.
    pub fn open<D: SpanDecl>(&mut self, now: Ts, parent: Option<SpanRef>, decl: D) -> ManualSpan<D>
    where
        Current: ParentOf<D>,
    {
        let at = self.wall(now);
        self.start(at, parent, decl)
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

    fn start<D: SpanDecl>(&mut self, at: u64, parent: Option<SpanRef>, decl: D) -> ManualSpan<D> {
        let trace_id = match parent {
            Some(p) => p.trace_id,
            None => TraceId(u128::from(self.next_id()) << 64 | u128::from(self.next_id())),
        };
        let context = SpanRef {
            trace_id,
            span_id: SpanId(self.next_id()),
            sampled: parent.is_none_or(|p| p.sampled),
        };
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
        ManualSpan { context, ended: false, _decl: PhantomData }
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
    _decl: PhantomData<fn() -> D>,
}

impl<D: SpanDecl> std::fmt::Debug for ManualSpan<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManualSpan").field("name", &D::NAME).field("span", &self.context).finish()
    }
}

impl<D: SpanDecl> ManualSpan<D> {
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
        trace.start(at, Some(self.context), decl)
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
