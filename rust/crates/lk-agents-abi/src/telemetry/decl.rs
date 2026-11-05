//! The telemetry declaration model: what a span, an event, a metric instrument or a report record
//! is, and how each attribute is encoded on the wire. Declarations are `Attrs` structs with a role
//! attribute (`spans`, `instruments`, `report::records`); `lk-agents-codegen` reads them into the
//! registry, the consumer schema and the Python constants.

pub use super::record::{InstrumentKind, KeyValues, Number, SpanKind, SpanContext, Value};
pub use lk_agents_host_derive::{Attrs, SpanValue, event, metric, record, span};

use super::record::Key;

/// Attributes a value writes: a span's start or end, an event, a metric's labels, a record.
pub trait Attrs {
    /// The `#[at]` field: when the span started or ended, or the event happened, when that was
    /// before the step that records it. `None`: the step's instant.
    fn at(&self) -> Option<crate::Ts> {
        None
    }
    fn write(&self, out: &mut KeyValues);
    /// The fields marked `label`: the labels of the measurements recorded from this span.
    fn labels(&self, out: &mut KeyValues);
    /// The fields marked `metric = Instrument`, as measurements of those instruments.
    fn measure(&self, out: &mut Vec<(Instrument, Number)>);
}

impl Attrs for () {
    fn write(&self, _: &mut KeyValues) {}
    fn labels(&self, _: &mut KeyValues) {}
    fn measure(&self, _: &mut Vec<(Instrument, Number)>) {}
}

/// An optional group: written when present.
impl<T: Attrs> Attrs for Option<T> {
    fn at(&self) -> Option<crate::Ts> {
        self.as_ref().and_then(Attrs::at)
    }
    fn write(&self, out: &mut KeyValues) {
        if let Some(t) = self {
            t.write(out);
        }
    }
    fn labels(&self, out: &mut KeyValues) {
        if let Some(t) = self {
            t.labels(out);
        }
    }
    fn measure(&self, out: &mut Vec<(Instrument, Number)>) {
        if let Some(t) = self {
            t.measure(out);
        }
    }
}

/// A span: its start attributes are the struct, its end attributes `End`.
pub trait SpanDecl: Attrs {
    const NAME: &'static str;
    const KIND: SpanKind;
    type End: Attrs;
    /// Why the span may end after its parent. `None`: it ends within its parent.
    const OUTLIVES_PARENT: Option<&'static str>;
    /// `handle = Manual`: opened only as an entity-owned `ManualSpan` (sans-I/O crates), never a
    /// `ScopedSpan`; each handle refuses the other's declarations at compile time.
    const MANUAL: bool;
    /// The declared parents, for checks against a recorded span.
    const PARENTS: &'static [Parent];
}

/// What a span ends with (`telemetry::end`, `Trace::end`): its declared end attributes, an error
/// or an `Err` (the span failed), or [`Failed`] (both). Every declaration's `End` is one.
pub trait SpanEnd<D: SpanDecl> {
    /// `None`: ended without end attributes.
    fn attributes(&self) -> Option<&D::End>;
    /// The failure the span ended with: status `ERROR`, `error.type` and an `exception` event.
    fn error(&self) -> Option<&dyn super::errors::SpanError>;
}

/// An `Ok` ends with its value's attributes; an `Err` ends the span as failed.
impl<D: SpanDecl, T: SpanEnd<D>, E: super::errors::SpanError> SpanEnd<D> for Result<T, E> {
    fn attributes(&self) -> Option<&D::End> {
        self.as_ref().ok().and_then(SpanEnd::attributes)
    }
    fn error(&self) -> Option<&dyn super::errors::SpanError> {
        match self {
            Ok(value) => value.error(),
            Err(error) => Some(error),
        }
    }
}

/// An error ends the span as failed, without end attributes.
impl<D: SpanDecl, E: super::errors::SpanError> SpanEnd<D> for E {
    fn attributes(&self) -> Option<&D::End> {
        None
    }
    fn error(&self) -> Option<&dyn super::errors::SpanError> {
        Some(self)
    }
}

/// A failed span that still records its end attributes (a failed attempt's request ids).
pub struct Failed<'a, End> {
    pub end: End,
    pub error: &'a dyn super::errors::SpanError,
}

impl<D: SpanDecl> SpanEnd<D> for Failed<'_, D::End> {
    fn attributes(&self) -> Option<&D::End> {
        Some(&self.end)
    }
    fn error(&self) -> Option<&dyn super::errors::SpanError> {
        Some(self.error)
    }
}

/// A declared parent, as checks against recorded spans read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parent {
    /// The declared span of this name.
    Span(&'static str),
    /// [`Current`]: any parent.
    Current,
    /// [`External`]: a span from outside the framework, or none.
    External,
}

impl Parent {
    /// Whether `parents` allow opening under `current`: the current span's declared name, `None`
    /// for a span from outside the framework or no span.
    pub fn allows(parents: &[Parent], current: Option<&str>) -> bool {
        parents.iter().any(|p| match (p, current) {
            (Parent::Current, _) | (Parent::External, None) => true,
            (Parent::Span(name), Some(current)) => *name == current,
            _ => false,
        })
    }
}

/// What a span may be opened under: a declared span, [`Current`] or [`External`].
pub trait SpanParent {
    const PARENT: Parent;
}

impl<T: SpanDecl> SpanParent for T {
    const PARENT: Parent = Parent::Span(<T as SpanDecl>::NAME);
}

/// `Self` is a declared parent of `D`.
pub trait ParentOf<D: SpanDecl>: SpanParent {}

/// The span follows its caller: it nests under whatever span is current.
pub struct Current;

impl SpanParent for Current {
    const PARENT: Parent = Parent::Current;
}

/// The span continues a context from outside the process (a job dispatch, an A2A request), or
/// starts a trace.
pub struct External;

impl SpanParent for External {
    const PARENT: Parent = Parent::External;
}

/// What a span's declaration says of its place in the tree.
#[derive(Debug, Clone, Copy)]
pub struct Declared {
    pub name: &'static str,
    pub parents: &'static [Parent],
    pub outlives_parent: bool,
}

impl Declared {
    pub const fn of<D: SpanDecl>() -> Declared {
        Declared {
            name: D::NAME,
            parents: D::PARENTS,
            outlives_parent: D::OUTLIVES_PARENT.is_some(),
        }
    }
}

/// A span event.
pub trait EventDecl: Attrs {
    const NAME: &'static str;
}

/// `E` is declared in the span's `events = [..]`.
pub trait HasEvent<E: EventDecl>: SpanDecl {}

/// A metric instrument; its attributes are its labels.
pub trait MetricDecl: Attrs {
    const NAME: &'static str;
    const KIND: InstrumentKind;
    const UNIT: &'static str;
    /// A histogram's explicit bucket upper bounds; empty for a counter.
    const BOUNDS: &'static [f64];
}

/// What a measurement records of its instrument's declaration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instrument {
    pub name: &'static str,
    pub kind: InstrumentKind,
    pub unit: &'static str,
    pub bounds: &'static [f64],
}

impl Instrument {
    pub const fn of<M: MetricDecl>() -> Instrument {
        Instrument { name: M::NAME, kind: M::KIND, unit: M::UNIT, bounds: M::BOUNDS }
    }
}

/// A report record, exported to Cloud as an OTLP log of `SCOPE` with the string body `BODY`.
pub trait RecordDecl: Attrs {
    const SCOPE: &'static str;
    const BODY: &'static str;
    /// The span the record belongs to (`#[span_id]`).
    fn span(&self) -> Option<SpanContext>;
}

/// A field's wire value. `None` omits the attribute.
pub trait SpanValue {
    fn value(&self) -> Option<Value>;
}

/// Writes `value` under `key` when present. Called by `#[derive(Attrs)]`.
pub fn put(out: &mut KeyValues, key: &'static str, value: Option<Value>) {
    if let Some(value) = value {
        out.push((Key::Borrowed(key), value));
    }
}

/// Writes `values` into `out`, each replacing the value already under its key.
pub fn overwrite(out: &mut KeyValues, values: KeyValues) {
    for (key, value) in values {
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => out.push((key, value)),
        }
    }
}

/// Ends a span, for both span handles. `end`'s attributes and labels replace the start's under
/// the same key (a fallback's served model), `every` replaces both, and with an `end` each measured
/// field is emitted as one measurement of its instrument under the span's labels, before the span.
/// A key appears once.
pub fn finish_span<E: Attrs>(
    mut span: super::record::SpanRecord,
    mut labels: KeyValues,
    mut measured: Vec<(Instrument, Number)>,
    end: Option<&E>,
    every: super::spans::common::EverySpan,
    emit: &mut dyn FnMut(super::record::Record),
) {
    use super::record::{Measurement, Record};
    if let Some(end) = end {
        let mut values = KeyValues::new();
        end.write(&mut values);
        overwrite(&mut span.attributes, values);
        let mut values = KeyValues::new();
        end.labels(&mut values);
        overwrite(&mut labels, values);
        end.measure(&mut measured);
        for (instrument, value) in measured {
            emit(Record::Measurement(Measurement {
                instrument: instrument.name,
                kind: instrument.kind,
                unit: instrument.unit,
                bounds: instrument.bounds,
                value,
                labels: labels.clone(),
                job: span.job.clone(),
            }));
        }
    }
    let mut values = KeyValues::new();
    every.write(&mut values);
    overwrite(&mut span.attributes, values);
    emit(Record::Span(span));
}

/// A field's value as a measurement. Called by `#[derive(Attrs)]`.
pub fn measurement<T: SpanValue>(field: &T) -> Option<Number> {
    match field.value()? {
        Value::Int(i) => Some(Number::Int(i)),
        Value::Double(d) => Some(Number::Double(d)),
        _ => None,
    }
}

impl SpanValue for bool {
    fn value(&self) -> Option<Value> {
        Some(Value::Bool(*self))
    }
}

impl SpanValue for i64 {
    fn value(&self) -> Option<Value> {
        Some(Value::Int(*self))
    }
}

impl SpanValue for u32 {
    fn value(&self) -> Option<Value> {
        Some(Value::Int(i64::from(*self)))
    }
}

impl SpanValue for u64 {
    fn value(&self) -> Option<Value> {
        Some(Value::Int(i64::try_from(*self).unwrap_or(i64::MAX)))
    }
}

impl SpanValue for f64 {
    fn value(&self) -> Option<Value> {
        Some(Value::Double(*self))
    }
}

impl SpanValue for String {
    fn value(&self) -> Option<Value> {
        Some(Value::String(self.clone()))
    }
}

impl<T: SpanValue> SpanValue for Option<T> {
    fn value(&self) -> Option<Value> {
        self.as_ref()?.value()
    }
}

impl<T: SpanValue> SpanValue for Vec<T> {
    fn value(&self) -> Option<Value> {
        Some(Value::Array(self.iter().filter_map(SpanValue::value).collect()))
    }
}

/// A duration or delay, in seconds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Seconds(pub f64);

impl SpanValue for Seconds {
    fn value(&self) -> Option<Value> {
        Some(Value::Double(self.0))
    }
}

/// A value that may be impossible to compute. Unmeasured, it is omitted, or written as the
/// attribute's declared `unmeasured =` value.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Measured<T>(pub Option<T>);

/// The declared encoding of an unmeasured value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unmeasured {
    Zero,
    MinusOne,
}

/// A measurable type's `0` and `-1`.
pub trait Sentinel: SpanValue + Sized {
    fn sentinel(unmeasured: Unmeasured) -> Self;
}

impl Sentinel for Seconds {
    fn sentinel(unmeasured: Unmeasured) -> Self {
        Seconds(f64::sentinel(unmeasured))
    }
}

impl Sentinel for f64 {
    fn sentinel(unmeasured: Unmeasured) -> Self {
        match unmeasured {
            Unmeasured::Zero => 0.0,
            Unmeasured::MinusOne => -1.0,
        }
    }
}

impl Sentinel for i64 {
    fn sentinel(unmeasured: Unmeasured) -> Self {
        match unmeasured {
            Unmeasured::Zero => 0,
            Unmeasured::MinusOne => -1,
        }
    }
}

impl<T: SpanValue> SpanValue for Measured<T> {
    fn value(&self) -> Option<Value> {
        self.0.as_ref()?.value()
    }
}

impl<T: Sentinel> Measured<T> {
    /// The value, or the declared encoding when unmeasured. Called by `#[derive(Attrs)]`.
    pub fn value_or(&self, unmeasured: Unmeasured) -> Option<Value> {
        match &self.0 {
            Some(v) => v.value(),
            None => T::sentinel(unmeasured).value(),
        }
    }
}

/// A boolean consumers read as the string `"true"` or `"false"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolString(pub bool);

impl SpanValue for BoolString {
    fn value(&self) -> Option<Value> {
        Some(Value::String(if self.0 { "true" } else { "false" }.into()))
    }
}

/// User content under an `lk.pii.*` key, removed by Cloud's ingest for a redacted session.
#[derive(Debug, Clone, PartialEq)]
pub struct Content<T>(pub T);

impl<T: SpanValue> SpanValue for Content<T> {
    fn value(&self) -> Option<Value> {
        self.0.value()
    }
}

/// User content under a GenAI message key, removed by the Cloud destination itself for a redacted
/// session.
#[derive(Debug, Clone, PartialEq)]
pub struct GenAiContent<T>(pub T);

impl<T: SpanValue> SpanValue for GenAiContent<T> {
    fn value(&self) -> Option<Value> {
        self.0.value()
    }
}

/// A JSON string of `T`, for structured blobs consumers parse.
#[derive(Debug, Clone, PartialEq)]
pub struct Json<T>(pub T);

impl<T: serde::Serialize> SpanValue for Json<T> {
    fn value(&self) -> Option<Value> {
        serde_json::to_string(&self.0).ok().map(Value::String)
    }
}

/// A JSON object, the payload of a [`KvList`] until its shape is declared.
pub type JsonObject = serde_json::Map<String, serde_json::Value>;

/// An OTLP key-value list of `T`.
#[derive(Debug, Clone, PartialEq)]
pub struct KvList<T>(pub T);

/// `T`'s serde form, its absent (`null`) fields omitted.
impl<T: serde::Serialize> SpanValue for KvList<T> {
    fn value(&self) -> Option<Value> {
        json(&serde_json::to_value(&self.0).ok()?)
    }
}

/// JSON as an OTLP value; `null` is omitted.
pub fn json(v: &serde_json::Value) -> Option<Value> {
    use serde_json::Value as J;
    Some(match v {
        J::Null => return None,
        J::Bool(b) => Value::Bool(*b),
        J::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Double(n.as_f64()?),
        },
        J::String(s) => Value::String(s.clone()),
        J::Array(a) => Value::Array(a.iter().filter_map(json).collect()),
        J::Object(o) => {
            Value::KvList(o.iter().filter_map(|(k, v)| Some((k.clone(), json(v)?))).collect())
        }
    })
}

/// A metric label: an enum or a bounded string. Ids, transcripts and error messages are never
/// labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label(pub String);

impl SpanValue for Label {
    fn value(&self) -> Option<Value> {
        Some(Value::String(self.0.clone()))
    }
}

#[cfg(test)]
#[path = "decl_tests.rs"]
mod tests;
