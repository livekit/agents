use super::*;
use crate::telemetry::semconv::otel;

#[derive(SpanValue, Debug, Clone, Copy)]
enum Origin {
    ServerSpeech,
}

#[derive(Attrs)]
struct Group {
    #[attr("lk.test.group", label)]
    group: Label,
}

#[span(name = "test_parent", parent = External)]
#[derive(Attrs)]
struct TestParent;

#[event(name = "test_event")]
#[derive(Attrs)]
struct TestEvent;

#[span(name = "test_span", parent = TestParent | Current, end = TestEnd, events = [TestEvent], kind = Client)]
#[derive(Attrs)]
struct TestSpan {
    #[attr("lk.test.origin")]
    origin: Origin,
    #[attr(otel::GEN_AI_REQUEST_MODEL, label)]
    model: Option<String>,
    #[attrs]
    group: Option<Group>,
    #[attr("lk.pii.test")]
    content: Content<Json<serde_json::Value>>,
    #[attr("lk.test.flag")]
    flag: BoolString,
    #[attr("lk.test.ids")]
    ids: Vec<String>,
    #[attr("lk.test.sip")]
    sip: Template<String>,
}

#[derive(Attrs)]
struct TestEnd {
    #[attr("lk.test.omitted", metric = "lk.test.metric")]
    omitted: Measured<Seconds>,
    #[attr("lk.test.zero", unmeasured = 0)]
    zero: Measured<Seconds>,
    #[attr("lk.test.minus_one", unmeasured = -1)]
    minus_one: Measured<i64>,
    #[attr("lk.test.measured", metric = "lk.test.metric")]
    measured: Measured<Seconds>,
}

#[record(scope = "chat_history", body = "chat item")]
#[derive(Attrs)]
struct TestRecord {
    #[attr("chat.item")]
    item: KvList<JsonObject>,
    #[span_id]
    turn: Option<SpanRef>,
}

fn parent_of<P: ParentOf<D>, D: SpanDecl>() {}
fn has_event<D: HasEvent<E>, E: EventDecl>() {}

fn written(attrs: &impl Attrs) -> KeyValues {
    let mut out = KeyValues::new();
    attrs.write(&mut out);
    out
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

// Catches: a field written under the wrong key or encoding, an absent value written, a
// flattened group or a template dropped, or attributes reordered.
#[test]
fn a_declaration_writes_each_field_in_order() {
    let span = TestSpan {
        origin: Origin::ServerSpeech,
        model: None,
        group: Some(Group { group: Label("g".into()) }),
        content: Content(Json(serde_json::json!({"a": 1}))),
        flag: BoolString(false),
        ids: vec!["r1".into()],
        sip: Template(vec![("callID".into(), "c".into())]),
    };
    assert_eq!(
        written(&span),
        vec![
            ("lk.test.origin".into(), s("server_speech")),
            ("lk.test.group".into(), s("g")),
            ("lk.pii.test".into(), s("{\"a\":1}")),
            ("lk.test.flag".into(), s("false")),
            ("lk.test.ids".into(), Value::Array(vec![s("r1")])),
            ("lk.test.sip.callID".into(), s("c")),
        ]
    );
    let mut labels = KeyValues::new();
    span.labels(&mut labels);
    assert_eq!(labels, vec![("lk.test.group".into(), s("g"))]);
}

// Catches: an unmeasured value written as anything but its declared encoding, or a
// measurement recorded for an unmeasured value.
#[test]
fn unmeasured_values_follow_their_declared_encoding() {
    let end = TestEnd {
        omitted: Measured(None),
        zero: Measured(None),
        minus_one: Measured(None),
        measured: Measured(Some(Seconds(0.5))),
    };
    assert_eq!(
        written(&end),
        vec![
            ("lk.test.zero".into(), Value::Double(0.0)),
            ("lk.test.minus_one".into(), Value::Int(-1)),
            ("lk.test.measured".into(), Value::Double(0.5)),
        ]
    );
    let mut measured = Vec::new();
    end.measure(&mut measured);
    assert_eq!(measured, vec![("lk.test.metric", Number::Double(0.5))]);
}

// Catches: the role constants or the compile-time parent and event relations not generated.
#[test]
fn roles_carry_their_declarations() {
    assert_eq!(
        (TestSpan::NAME, TestSpan::KIND, TestSpan::OUTLIVES_PARENT),
        ("test_span", SpanKind::Client, None)
    );
    parent_of::<TestParent, TestSpan>();
    parent_of::<Current, TestSpan>();
    parent_of::<External, TestParent>();
    has_event::<TestSpan, TestEvent>();
    assert_eq!(<TestEvent as EventDecl>::NAME, "test_event");
    let turn = SpanRef {
        trace_id: crate::telemetry::record::TraceId(1),
        span_id: crate::telemetry::record::SpanId(2),
        sampled: true,
    };
    let mut item = JsonObject::new();
    item.insert("message".into(), serde_json::json!({"role": "user", "n": 2, "x": null}));
    let record = TestRecord { item: KvList(item), turn: Some(turn) };
    assert_eq!(
        (TestRecord::SCOPE, TestRecord::BODY, record.span()),
        ("chat_history", "chat item", Some(turn))
    );
    assert_eq!(
        written(&record),
        vec![(
            "chat.item".into(),
            Value::KvList(vec![(
                "message".into(),
                Value::KvList(vec![("role".into(), s("user")), ("n".into(), Value::Int(2))])
            )])
        )]
    );
}
