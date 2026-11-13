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
    pub const GENAI_CONTENT: &[&str] = &[otel::GEN_AI_INPUT_MESSAGES];

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
        #[attr("lk.delay", unmeasured = 0, metric = TurnDelay)]
        pub delay: Measured<Seconds>,
        /// The transcript.
        #[attr("lk.pii.transcript")]
        pub transcript: Content<String>,
        #[attr(otel::GEN_AI_INPUT_MESSAGES)]
        pub messages: Option<GenAiContent<Json<Value>>>,
        /// Request ids.
        #[attr("lk.ids")]
        pub ids: Vec<String>,
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

    pub const SECONDS: &[f64] = &[0.5, 1.0];

    /// Delay to the decision.
    #[metric(name = "lk.turn.delay", kind = Histogram, unit = "s", bounds = SECONDS)]
    #[derive(Attrs)]
    pub struct TurnDelay {
        /// The model.
        #[attr("model_name")]
        pub model: Label,
    }

    /// Agent invocations.
    #[metric(name = otel::GEN_AI_INVOKE_AGENT_DURATION, kind = Histogram, unit = "s", bounds = SECONDS)]
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
        (span("/// T.\n#[attr(\"lk.t\", metric = S)] pub t: String,"), "is a number"),
        (span("/// T.\n#[attr(\"lk.t\", metric = S)] pub t: Measured<Seconds>,"), "is not a `#[metric]`"),
        (span("/// T.\n#[attr(\"lk.t\")] pub t: u8,"), "not an attribute type"),
        (span("#[attr(\"lk.t\")] pub t: String,"), "needs a `///` brief"),
        (span("/// A.\n#[attr(\"lk.t\")] pub a: String, #[attr(\"lk.t\")] pub b: String,"), "written twice"),
        (
            format!("pub const OWN: &str = \"lk.room_id\";\n{}", span("/// T.\n#[attr(OWN)] pub t: String,")),
            "name its `semconv::protocol` constant",
        ),
        (span("/// T.\n#[attr(\"gen_ai.mine\")] pub t: String,"), "GenAI registry's namespace"),
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
            format!("{}\n/// M.\n#[metric(name = \"m\", kind = Histogram, unit = \"s\", bounds = B)]\n#[derive(Attrs)]\npub struct M {{ /// Model.\n#[attr(\"model_name\")] pub m: Label }}\npub const B: &[f64] = &[1.0];", span("/// D.\n#[attr(\"lk.d\", metric = M)] pub d: Measured<Seconds>, #[attr(otel::GEN_AI_REQUEST_MODEL, label)] pub g: String,")),
            "is not a label of `m`",
        ),
        (
            format!("{}\n/// M.\n#[metric(name = \"m\", kind = Histogram, unit = \"s\", bounds = B)]\n#[derive(Attrs)]\npub struct M {{ /// Model.\n#[attr(\"model_name\")] pub m: Label }}\npub const B: &[f64] = &[1.0];", span("/// D.\n#[attr(\"lk.d\", metric = M)] pub d: Measured<Seconds>,")),
            "needs the label `model_name`",
        ),
        (
            "/// M.\n#[metric(name = \"m\", kind = Histogram, unit = \"s\", bounds = B)]\n#[derive(Attrs)]\npub struct M;\npub const B: &[f64] = &[1.0, 0.5];".into(),
            "increase strictly",
        ),
        (
            format!("/// E.\n#[derive(Attrs)]\npub struct EverySpan {{ /// C.\n#[attr(\"lk.cancelled\")] pub c: Option<bool> }}\n{}", span("/// C.\n#[attr(\"lk.cancelled\")] pub c: bool,")),
            "written on every span",
        ),
        (
            "/// S.\n#[span(name = \"s\", parent = Current)]\n#[derive(Attrs)]\npub struct S;\npub const SPANS: &[Declared] = &[];".into(),
            "the `#[span]` declarations are",
        ),
    ] {
        let error = registry(&decls).err().unwrap_or_else(|| panic!("accepted:\n{decls}"));
        assert!(error.contains(message), "{message}: {error}");
    }
}

// Catches: an output missing a declaration or misreporting it, a dependency key redefined in
// the Weaver registry, or the pins lost.
#[test]
fn outputs_describe_the_declarations() {
    let r = registry(TURN).expect("reads");
    let files: std::collections::BTreeMap<String, String> =
        telemetry::otel(&r).into_iter().collect();
    let schema: serde_json::Value =
        serde_json::from_str(&files["rust/otel/schema.json"]).expect("schema is JSON");
    let turn = &schema["spans"][1];
    assert_eq!((turn["name"].as_str(), turn["kind"].as_str()), (Some("turn"), Some("client")));
    assert_eq!(
        turn["end"][0],
        serde_json::json!({"key": "lk.delay", "type": "double", "required": false, "unit": "s", "unmeasured": 0, "metric": "lk.turn.delay"})
    );
    let delay = schema["metrics"].as_array().unwrap().iter().find(|m| m["name"] == "lk.turn.delay");
    assert_eq!(delay.unwrap()["bounds"], serde_json::json!([0.5, 1.0]));
    assert_eq!(turn["end"][1]["content"], "pii");
    assert_eq!(turn["start"][0]["members"], serde_json::json!(["speech", "server_speech"]));
    assert_eq!(schema["spans"][0]["parents"], serde_json::json!(["<external>", "<current>"]));
    assert_eq!(schema["groups"][0]["name"], "identity");

    // The Weaver files are YAML in its JSON subset, after their comment header.
    let yaml = |path: &str| -> serde_json::Value {
        let text = &files[path];
        assert!(text.starts_with("# GENERATED FILE -- DO NOT EDIT."), "{text}");
        let body: String = text.lines().filter(|l| !l.starts_with('#')).collect();
        serde_json::from_str(&body).expect("JSON")
    };
    let attributes = yaml("rust/otel/registry/attributes.yaml");
    let owned: Vec<&serde_json::Value> = attributes["groups"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|g| g["attributes"].as_array().unwrap())
        .collect();
    let origin = owned.iter().find(|a| a["id"] == "lk.origin").expect("lk.origin");
    assert_eq!(origin["type"]["members"][1]["value"], "server_speech");
    assert!(
        !owned.iter().any(|a| a["id"] == "gen_ai.request.model" || a["id"] == "lk.room_id"),
        "dependency keys are referenced, never redefined: {attributes}"
    );
    let spans = yaml("rust/otel/registry/spans.yaml");
    let turn = spans["groups"].as_array().unwrap().iter().find(|g| g["id"] == "span.turn");
    let turn = turn.expect("span.turn");
    assert_eq!((turn["type"].as_str(), turn["span_kind"].as_str()), (Some("span"), Some("client")));
    assert!(turn["attributes"].as_array().unwrap().contains(
        &serde_json::json!({"ref": "gen_ai.request.model", "requirement_level": "recommended"})
    ));
    let metrics = yaml("rust/otel/registry/metrics.yaml");
    assert_eq!(metrics["groups"][0]["metric_name"], "lk.turn.delay");
    assert_eq!(metrics["groups"].as_array().unwrap().len(), 1, "imports are not redeclared");
    let imports = yaml("rust/otel/registry/imports.yaml");
    assert_eq!(imports["imports"]["metrics"], serde_json::json!(["gen_ai.invoke_agent.duration"]));
    let manifest = &files["rust/otel/manifest.yaml"];
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
}
