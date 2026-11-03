//! The telemetry registry: what the reader accepts and refuses, what each output says, and that
//! the declarations cover what livekit-agents 1.8.10 sends.

use crate::decl::tree::Crate;
use crate::telemetry::{self, Content, Encoding, Parent, Registry, Wire};
use lk_agents_bindgen::telemetry::{SpanKind, Unmeasured};

const DECL: &str = "
    pub struct Current;
    pub struct External;
    pub struct Seconds(pub f64);
    pub struct Measured<T>(pub Option<T>);
    pub struct Content<T>(pub T);
    pub struct GenAiContent<T>(pub T);
    pub struct Json<T>(pub T);
    pub struct KvList<T>(pub T);
    pub struct Label(pub String);
    pub struct BoolString(pub bool);
    pub struct Template<T>(pub Vec<(String, T)>);
";

const SEMCONV: &str = r#"
    pub mod otel {
        pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
        pub const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";
        pub const GEN_AI_INVOKE_AGENT_DURATION: &str = "gen_ai.invoke_agent.duration";
    }
    pub mod protocol {
        pub const LK_ROOM_ID: &str = "lk.room_id";
    }
"#;

const PRELUDE: &str = "
    use crate::telemetry::decl::{
        BoolString, Content, Current, External, GenAiContent, Json, Label, Measured, Seconds,
        Template,
    };
    use crate::telemetry::semconv::{otel, protocol};
";

/// The registry of a crate whose `telemetry::spans` holds `decls`.
fn registry(decls: &str) -> Result<Registry, String> {
    let spans = format!("{PRELUDE}\n{decls}");
    let krate = Crate::from_files(
        "abi",
        &[
            ("lib.rs", "pub mod telemetry;"),
            ("telemetry/mod.rs", "pub mod decl; pub mod semconv; pub mod spans;"),
            ("telemetry/decl.rs", DECL),
            ("telemetry/semconv.rs", SEMCONV),
            ("telemetry/spans.rs", &spans),
        ],
    )?;
    telemetry::read(&krate)
}

const TURN: &str = r#"
    /// The session.
    #[span(name = "session", parent = External | Current)]
    #[derive(Attrs)]
    pub struct Session {
        /// The room.
        #[attr(protocol::LK_ROOM_ID)]
        pub room: String,
    }

    /// A turn.
    #[span(name = "turn", parent = Session, end = Option<TurnEnd>, events = [Resumed], kind = Client, outlives_parent = "late")]
    #[derive(Attrs)]
    pub struct Turn {
        /// What opened it.
        #[attr("lk.origin")]
        pub origin: Origin,
        #[attr(otel::GEN_AI_REQUEST_MODEL, label = "model_name")]
        pub model: Option<String>,
        #[attrs]
        pub group: Option<Group>,
    }

    #[derive(Attrs)]
    pub struct TurnEnd {
        /// Seconds to the decision.
        #[attr("lk.delay", unmeasured = 0, metric = "lk.turn.delay")]
        pub delay: Measured<Seconds>,
        /// The transcript.
        #[attr("lk.pii.transcript")]
        pub transcript: Content<String>,
        #[attr(otel::GEN_AI_INPUT_MESSAGES)]
        pub messages: Option<GenAiContent<Json<Value>>>,
        /// Request ids.
        #[attr("lk.ids")]
        pub ids: Vec<String>,
        /// One key per SIP attribute.
        #[attr("lk.sip")]
        pub sip: Template<String>,
    }

    #[derive(Attrs)]
    pub struct Group {
        /// Grouped.
        #[attr("lk.grouped")]
        pub grouped: BoolString,
    }

    #[derive(SpanValue)]
    pub enum Origin { Speech, ServerSpeech }

    /// The user resumed.
    #[event(name = "resumed")]
    #[derive(Attrs)]
    pub struct Resumed;

    /// Delay to the decision.
    #[metric(name = "lk.turn.delay", kind = Histogram, unit = "s")]
    #[derive(Attrs)]
    pub struct TurnDelay {
        /// The model.
        #[attr("model_name")]
        pub model: Label,
    }

    /// Agent invocations.
    #[metric(name = otel::GEN_AI_INVOKE_AGENT_DURATION, kind = Histogram, unit = "s")]
    #[derive(Attrs)]
    pub struct Invoke {
        #[attr("model_name")]
        pub model: Label,
    }

    /// Stamped by the exporter.
    #[derive(Attrs)]
    pub struct Identity {
        /// The job.
        #[attr("job_id")]
        pub job: String,
    }
"#;

// Catches: a span's name, kind, parents, events or outlive reason, an attribute's key, registry,
// wire type, encoding, unit, content kind or unmeasured encoding read wrong, an optional end not
// making its fields optional, or an unflattened group not reported as a standalone group.
#[test]
fn declarations_are_read_into_the_registry() {
    let r = registry(TURN).expect("reads");
    let turn = r.spans.iter().find(|s| s.name == "turn").expect("turn");
    assert_eq!(
        (turn.kind, turn.parents.clone(), turn.events.clone(), turn.outlives_parent.as_deref()),
        (
            SpanKind::Client,
            vec![Parent::Span("session".into())],
            vec!["resumed".into()],
            Some("late")
        )
    );
    let session = r.spans.iter().find(|s| s.name == "session").expect("session");
    assert_eq!(session.parents, vec![Parent::External, Parent::Current]);
    assert_eq!(session.start[0].registry, "protocol");
    let start: Vec<_> =
        turn.start.iter().map(|a| (a.key.as_str(), a.required, a.label.as_deref())).collect();
    assert_eq!(
        start,
        [
            ("lk.origin", true, None),
            ("gen_ai.request.model", false, Some("model_name")),
            ("lk.grouped", false, None)
        ]
    );
    assert_eq!(
        turn.start[0].encoding,
        Some(Encoding::Enum(vec!["speech".into(), "server_speech".into()]))
    );
    assert_eq!(turn.start[2].encoding, Some(Encoding::BoolString));
    let end = &turn.end;
    assert!(end.iter().all(|a| !a.required), "an optional end makes every field optional");
    assert_eq!(
        (end[0].wire.clone(), end[0].seconds, end[0].unmeasured, end[0].metric.as_deref()),
        (Wire::Double, true, Some(Unmeasured::Zero), Some("lk.turn.delay"))
    );
    assert_eq!(end[1].content, Some(Content::Pii));
    assert_eq!(
        (end[2].content, end[2].encoding.clone()),
        (Some(Content::GenAi), Some(Encoding::Json))
    );
    assert_eq!(end[3].wire, Wire::Array(Box::new(Wire::String)));
    assert!(end[4].template);
    let metrics: Vec<_> = r
        .metrics
        .iter()
        .map(|m| (m.name.as_str(), m.upstream, m.labels[0].label.as_deref()))
        .collect();
    assert_eq!(
        metrics,
        [
            ("lk.turn.delay", false, Some("model_name")),
            ("gen_ai.invoke_agent.duration", true, Some("model_name"))
        ]
    );
    let groups: Vec<_> = r.groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(
        groups,
        ["identity"],
        "Group is flattened and TurnEnd is an end: neither stands alone"
    );
    assert!(r.owned.iter().any(|a| a.key == "lk.origin" && a.brief == "What opened it."));
    assert!(r.referenced.iter().any(|a| a.key == "gen_ai.request.model"));
}

// Catches: a mistake the type system cannot see reaching the generated registry.
#[test]
fn registry_rules_refuse_mistakes() {
    let span = |fields: &str| {
        format!(
            "/// S.\n#[span(name = \"s\", parent = Current)]\n#[derive(Attrs)]\npub struct S {{ {fields} }}"
        )
    };
    for (decls, message) in [
        (span("/// R.\n#[attr(\"lk.room_id\")] pub r: String,"), "name its `semconv::protocol` constant"),
        (span("/// T.\n#[attr(\"lk.text\")] pub t: Content<String>,"), "keys under `lk.pii.*`"),
        (span("/// T.\n#[attr(\"lk.pii.text\")] pub t: String,"), "is `Content<T>`"),
        (span("/// T.\n#[attr(\"lk.text\")] pub t: GenAiContent<String>,"), "GenAI registry's message keys"),
        (span("/// T.\n#[attr(\"lk.t\", unmeasured = 0)] pub t: Seconds,"), "for a `Measured` value"),
        (span("/// T.\n#[attr(\"lk.t\", metric = \"m\")] pub t: String,"), "is a number"),
        (span("/// T.\n#[attr(\"lk.t\", metric = \"m\")] pub t: Measured<Seconds>,"), "not a declared instrument"),
        (span("/// T.\n#[attr(\"lk.t\")] pub t: u8,"), "not an attribute type"),
        (span("#[attr(\"lk.t\")] pub t: String,"), "needs a `///` brief"),
        (span("/// A.\n#[attr(\"lk.t\")] pub a: String, #[attr(\"lk.t\")] pub b: String,"), "written twice"),
        (
            format!("pub const OWN: &str = \"lk.own\";\n{}", span("/// T.\n#[attr(OWN)] pub t: String,")),
            "is not a `semconv` constant",
        ),
        (
            format!("{}\n/// E.\n#[event(name = \"e\")]\n#[derive(Attrs)]\npub struct E {{ /// T.\n#[attr(\"lk.t\")] pub t: i64 }}", span("/// T.\n#[attr(\"lk.t\")] pub t: String,")),
            "one key, one type",
        ),
        ("/// S.\n#[span(name = \"s\", parent = Nowhere)]\n#[derive(Attrs)]\npub struct S;".into(), "Nowhere"),
        ("#[span(name = \"s\", parent = Current)]\n#[derive(Attrs)]\npub struct S;".into(), "a `///` brief"),
        (
            "/// M.\n#[metric(name = \"m\", kind = Counter, unit = \"\")]\n#[derive(Attrs)]\npub struct M { /// Id.\n#[attr(\"lk.id\")] pub id: String }".into(),
            "is a `Label`",
        ),
        (
            "/// S.\n#[span(name = \"s\", parent = Current)]\n#[derive(Attrs)]\npub struct S;\n/// T.\n#[span(name = \"s\", parent = Current)]\n#[derive(Attrs)]\npub struct T;".into(),
            "declared twice",
        ),
        (
            format!("{}\n/// M.\n#[metric(name = \"m\", kind = Histogram, unit = \"s\")]\n#[derive(Attrs)]\npub struct M {{ /// Model.\n#[attr(\"model_name\")] pub m: Label }}", span("/// D.\n#[attr(\"lk.d\", metric = \"m\")] pub d: Measured<Seconds>, #[attr(otel::GEN_AI_REQUEST_MODEL, label)] pub g: String,")),
            "is not a label of `m`",
        ),
        (
            format!("{}\n/// M.\n#[metric(name = \"m\", kind = Histogram, unit = \"s\")]\n#[derive(Attrs)]\npub struct M {{ /// Model.\n#[attr(\"model_name\")] pub m: Label }}", span("/// D.\n#[attr(\"lk.d\", metric = \"m\")] pub d: Measured<Seconds>,")),
            "needs the label `model_name`",
        ),
    ] {
        let error = registry(&decls).err().unwrap_or_else(|| panic!("accepted:\n{decls}"));
        assert!(error.contains(message), "{message}: {error}");
    }
}

// Catches: an output missing a declaration or misreporting it, a dependency key redefined in
// the Weaver registry, the pins lost, or the trace rules diverging from the declared parents.
#[test]
fn outputs_describe_the_declarations() {
    let r = registry(TURN).expect("reads");
    let files: std::collections::BTreeMap<String, String> =
        telemetry::otel(&r).into_iter().collect();
    let schema: serde_json::Value =
        serde_json::from_str(&files["otel/schema.json"]).expect("schema is JSON");
    let turn = &schema["spans"][1];
    assert_eq!((turn["name"].as_str(), turn["kind"].as_str()), (Some("turn"), Some("client")));
    assert_eq!(
        turn["end"][0],
        serde_json::json!({"key": "lk.delay", "type": "double", "required": false, "unit": "s", "unmeasured": 0, "metric": "lk.turn.delay"})
    );
    assert_eq!(turn["end"][1]["content"], "pii");
    assert_eq!(turn["start"][0]["members"], serde_json::json!(["speech", "server_speech"]));
    assert_eq!(schema["spans"][0]["parents"], serde_json::json!(["<external>", "<current>"]));
    assert_eq!(schema["groups"][0]["name"], "identity");

    let attributes = &files["otel/registry/attributes.yaml"];
    assert!(
        attributes.contains("id: \"lk.origin\"") && attributes.contains("value: \"server_speech\""),
        "{attributes}"
    );
    assert!(attributes.contains("type: \"template[string]\""), "{attributes}");
    assert!(
        !attributes.contains("id: \"gen_ai.request.model\"")
            && !attributes.contains("id: \"lk.room_id\""),
        "dependency keys are referenced, never redefined: {attributes}"
    );
    let spans = &files["otel/registry/spans.yaml"];
    assert!(
        spans.contains("  - id: \"span.turn\"\n    type: \"span\"\n    span_kind: \"client\""),
        "{spans}"
    );
    assert!(
        spans.contains(
            "      - ref: \"gen_ai.request.model\"\n        requirement_level: \"recommended\""
        ),
        "{spans}"
    );
    let metrics = &files["otel/registry/metrics.yaml"];
    assert!(
        metrics.contains("metric_name: \"lk.turn.delay\"") && !metrics.contains("invoke_agent"),
        "{metrics}"
    );
    assert!(files["otel/registry/imports.yaml"].contains("- \"gen_ai.invoke_agent.duration\""));
    let manifest = &files["otel/manifest.yaml"];
    assert!(
        manifest.contains("livekit/protocol.git@") && manifest.contains("[otel/model/shared]"),
        "{manifest}"
    );
    assert!(
        manifest.contains("semantic-conventions.git@v1.44.0[model]")
            && manifest.contains("547934afa90a34298441caf80c43b27255c3a292[model]"),
        "{manifest}"
    );

    let python = telemetry::semconv_py(&r).expect("constants");
    for line in [
        "LK_PII_TRANSCRIPT: Final = \"lk.pii.transcript\"",
        "SPAN_TURN: Final = \"turn\"",
        "EVENT_RESUMED: Final = \"resumed\"",
        "METRIC_LK_TURN_DELAY: Final = \"lk.turn.delay\"",
        "GEN_AI_REQUEST_MODEL: Final = \"gen_ai.request.model\"",
    ] {
        assert!(python.contains(line), "{line}: {python}");
    }
    let rules = telemetry::trace_schema(&r);
    assert!(
        rules.contains("        \"session\": {ANY, ROOT},\n        \"turn\": {\"session\"},\n"),
        "{rules}"
    );
    assert!(rules.contains("    (\"turn\", \"session\"): \"late\",\n"), "{rules}");
    assert!(
        rules.starts_with("# GENERATED FILE -- DO NOT EDIT.") && rules.contains("def check_trace("),
        "{rules}"
    );
}

/// The 1.8.10 keys the declarations write under another name, on purpose.
const RENAMED: [(&str, &str); 2] = [("lk.room_sid", "lk.room_id"), ("lk.track_sid", "lk.track_id")];
/// Spans whose kind changed from 1.8.10's `INTERNAL`, on purpose.
const NEW_KINDS: [&str; 5] =
    ["llm_request", "tts_request", "realtime_inference", "rpc_handler", "rpc_call"];
/// Log attributes Python's logging handler added in 1.8.10; report records do not carry them.
const LOGGING_HANDLER: [&str; 1] = ["logger.name"];

fn compatible(baseline: &str, wire: &Wire) -> bool {
    match (baseline, wire) {
        ("empty", _) => true,
        ("string" | "string(json)", Wire::String)
        | ("int", Wire::Int)
        | ("double", Wire::Double)
        | ("bool", Wire::Bool) => true,
        (b, Wire::KvList) => b.starts_with("kvlist"),
        (b, Wire::Array(inner)) => {
            b.strip_prefix("array<").and_then(|b| b.strip_suffix('>')).is_some_and(|element| {
                element == "empty" || element.split('|').all(|e| compatible(e, inner))
            })
        }
        _ => false,
    }
}

// Catches: a span, parent, attribute, event, metric, label or record key that livekit-agents
// 1.8.10 sends and Cloud reads disappearing from (or changing type in) the declarations, other
// than the documented deliberate differences and the exporter-owned identity stamp.
#[test]
fn declarations_cover_the_1_8_10_wire() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let abi = Crate::read(&root, "rust/crates/lk-agents-abi", "lk_agents_abi").expect("abi");
    let r = telemetry::read(&abi).expect("the declarations read");
    let text = std::fs::read_to_string(root.join("tests/telemetry/baseline/schema_1_8_10.json"))
        .expect("baseline");
    let baseline: serde_json::Value = serde_json::from_str(&text).expect("baseline JSON");
    let stamped: Vec<&str> = r
        .groups
        .iter()
        .filter(|g| ["job_identity", "session_flags", "every_span"].contains(&g.name.as_str()))
        .flat_map(|g| g.attrs.iter().map(|a| a.key.as_str()))
        .collect();
    let rename = |key: &str| {
        RENAMED.iter().find(|(old, _)| *old == key).map_or(key, |(_, new)| *new).to_owned()
    };
    let check = |at: &str, attrs: &serde_json::Value, declared: &[&telemetry::Attr]| {
        let mut missing = Vec::new();
        for (key, types) in attrs.as_object().expect("attributes") {
            let key = rename(key);
            if stamped.contains(&key.as_str()) || LOGGING_HANDLER.contains(&key.as_str()) {
                continue;
            }
            let Some(a) = declared.iter().find(|a| a.key == key) else {
                missing.push(format!("{at}: `{key}` is not declared"));
                continue;
            };
            for t in types.as_array().expect("types").iter().map(|t| t.as_str().expect("a type")) {
                if !compatible(t, &a.wire) {
                    missing.push(format!("{at}: `{key}` was {t}, is declared {:?}", a.wire));
                }
            }
        }
        missing
    };
    let mut missing = Vec::new();
    for (name, span) in baseline["spans"].as_object().expect("spans") {
        let Some(d) = r.spans.iter().find(|s| &s.name == name) else {
            missing.push(format!("span {name} is not declared"));
            continue;
        };
        for parent in span["parents"].as_array().expect("parents") {
            let ok = match parent.as_str().expect("a parent") {
                "<root>" => {
                    d.parents.iter().any(|p| matches!(p, Parent::External | Parent::Current))
                }
                p => {
                    d.parents.iter().any(|q| q == &Parent::Current || q == &Parent::Span(p.into()))
                }
            };
            if !ok {
                missing.push(format!("{name}: parent {parent} is not declared"));
            }
        }
        if !NEW_KINDS.contains(&name.as_str()) && d.kind != SpanKind::Internal {
            missing.push(format!("{name}: kind changed to {:?}", d.kind));
        }
        let declared: Vec<_> = d.start.iter().chain(&d.end).collect();
        missing.extend(check(name, &span["attributes"], &declared));
        for (event, attrs) in span["events"].as_object().into_iter().flatten() {
            let Some(e) =
                r.events.iter().find(|e| &e.name == event).filter(|_| d.events.contains(event))
            else {
                missing.push(format!("{name}: event {event} is not declared"));
                continue;
            };
            let declared: Vec<_> = e.attrs.iter().collect();
            missing.extend(check(&format!("{name}/{event}"), attrs, &declared));
        }
    }
    for (name, metric) in baseline["metrics"].as_object().expect("metrics") {
        let Some(d) = r.metrics.iter().find(|m| &m.name == name) else {
            missing.push(format!("metric {name} is not declared"));
            continue;
        };
        let kind = match d.kind {
            lk_agents_bindgen::telemetry::InstrumentKind::Counter => "sum",
            lk_agents_bindgen::telemetry::InstrumentKind::Histogram => "histogram",
        };
        if (metric["kind"].as_str(), metric["unit"].as_str()) != (Some(kind), Some(d.unit.as_str()))
        {
            missing.push(format!(
                "metric {name}: was {} {}, is {kind} {:?}",
                metric["kind"], metric["unit"], d.unit
            ));
        }
        let declared: Vec<_> = d.labels.iter().collect();
        missing.extend(check(name, &metric["attributes"], &declared));
    }
    for (name, log) in baseline["logs"].as_object().expect("logs") {
        // Process logs go to stdout, never OTLP.
        let Some((scope, body)) =
            name.split_once(" / ").filter(|(scope, _)| !scope.starts_with('<'))
        else {
            continue;
        };
        let Some(d) = r.records.iter().find(|rec| rec.scope == scope && rec.body == body) else {
            missing.push(format!("record {name} is not declared"));
            continue;
        };
        let declared: Vec<_> = d.attrs.iter().collect();
        missing.extend(check(name, &log["attributes"], &declared));
    }
    assert!(missing.is_empty(), "the declarations drop 1.8.10's wire:\n  {}", missing.join("\n  "));
    for (old, new) in RENAMED {
        assert!(
            r.referenced.iter().any(|a| a.key == new && a.registry == "protocol"),
            "{old} becomes the shared {new}"
        );
    }
}
