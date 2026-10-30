//! Declared records' Python models: legacy modules, unions, constants and nested classes.

use super::*;

// Catches: declared records overwriting the legacy models of the module they share (a family
// migrated record by record), or a declared record redefining a legacy model's name.
#[test]
fn declared_records_join_a_legacy_module_without_replacing_it() {
    let legacy = "# GENERATED FILE -- DO NOT EDIT.\n\nfrom __future__ import annotations\n\nfrom pydantic import BaseModel\n\n\nclass Legacy(BaseModel):\n    x: int\n";
    let models = "# GENERATED FILE -- DO NOT EDIT.\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Declared:\n    y: int\n";
    let merged = crate::records::merge(legacy, models).expect("merges");
    assert!(
        merged.contains("class Legacy(BaseModel)") && merged.contains("class Declared:"),
        "{merged}"
    );
    assert_eq!(merged.matches("from __future__ import annotations").count(), 1, "{merged}");
    let clash = models.replace("class Declared", "class Legacy");
    assert!(crate::records::merge(legacy, &clash).unwrap_err().contains("`Legacy`"));
}

// Catches: a value-or-record union (`str | Message`) crossing through wrapper classes, a variant
// read by the wrong class (a `bool` taken as an `int`), or two variants no class tells apart.
#[test]
fn a_transparent_record_crosses_as_its_variants_values() {
    let decls = read(
        r#"#[crate::host(record(module = "session", extract, emit))]
        pub struct Message { pub text: String }
        #[crate::host(record(module = "session", extract, emit, transparent))]
        pub enum Input { Text(String), Count(i64), Flag(bool), Message(Message) }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let session = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(session.contains("Input: TypeAlias = str | int | bool | Message"), "{session}");
    assert!(!session.contains("class InputText"), "{session}");
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("obj.is_instance_of::<::pyo3::types::PyString>()")
            && shell.contains("!obj.is_instance_of::<::pyo3::types::PyBool>()")
            && shell.contains("crate::generated::records::session::message_from_py(obj)")
            && shell.contains("::lk_agents_abi::llm::class::Input::Text(x) =>"),
        "{shell}"
    );
    let ambiguous = read(
        r#"#[crate::host(record(module = "session", extract, transparent))]
        pub enum Two { A(String), B(String) }"#,
    )
    .expect("resolves");
    assert!(
        crate::records::conversions(&ambiguous).err().is_some_and(|e| e.contains("share a class"))
    );
}

// Catches: a model's fixed discriminator (`type`) forced into Rust state, missing from the model,
// or demanded of the host when it builds one.
#[test]
fn a_record_constant_is_a_model_field_at_its_fixed_value() {
    let decls = read(
        r#"#[crate::host(record(module = "session", extract, emit, constant(type = "stt_metrics")))]
        pub struct SttMetrics { pub duration: f64 }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let session = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(
        session.contains("    duration: float\n")
            && session.contains("    type: Literal[\"stt_metrics\"] = \"stt_metrics\"\n"),
        "{session}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(!shell.contains("\"type\""), "{shell}");
}

// Catches: a legacy union naming a record migrated to a declaration failing generation, the
// union referencing its model before the module defines it, or the record not marked for the
// legacy conversions (by-reference emit, union tag) its members owe.
#[test]
fn a_legacy_union_names_a_declared_record_member() {
    let mut decls = read(
        r#"#[crate::host(record(module = "events", extract, emit, constant(type = "updated")))]
        #[derive(Clone)]
        pub struct Updated { pub id: String }"#,
    )
    .expect("resolves");
    let mut legacy = crate::parse::file(
        "events.rs",
        r#"
        #[derive(Host)]
        #[python(pydantic, module = "events", name = "Started", tag = "type", tag_value = "started")]
        #[binding(emit)]
        pub struct Started { pub id: String }
        #[derive(Host)]
        #[python(union, module = "events", name = "Update")]
        #[binding(emit)]
        pub enum Update { Started(Started), Updated(Updated) }
        "#,
    )
    .expect("parses");
    crate::records::link_legacy(&mut legacy, &mut decls).expect("links");
    let union = legacy.iter().find(|d| d.rust_name == "Update").expect("Update");
    let crate::ir::Body::Union(members) = &union.body else { panic!("a union") };
    assert!(
        matches!(&members[1].ty, crate::ir::Ty::Foreign(f) if f.python == "Updated" && f.python_from.is_empty())
    );
    let legacy_py = "# GENERATED FILE -- DO NOT EDIT.\n\nfrom __future__ import annotations\n\nfrom pydantic import BaseModel\n\n\nclass Started(BaseModel):\n    id: str\n\n\nUpdate = Started | Updated\n";
    let models = crate::records::python(&decls).expect("models");
    let merged = crate::records::merge(legacy_py, &models[0].text).expect("merges");
    let (defined, used) =
        (merged.find("class Updated:").unwrap(), merged.find("= Started | Updated").unwrap());
    assert!(defined < used && merged.find("from pydantic").unwrap() < defined, "{merged}");
}

// Catches: a scalar-or-map option (`Policy | dict[str, Policy] | None`) needing wrapper classes,
// `None` not crossing as the unit variant, a policy string read as a model instance, an `int`
// refused where seconds are a `float`, or `int` and `float` variants left ambiguous.
#[test]
fn a_transparent_record_crosses_scalars_maps_and_none() {
    let decls = read(
        r#"#[crate::host(record(module = "options", extract, emit))]
        pub enum Policy { Batch, Queue }
        #[crate::host(record(module = "options", extract, emit, transparent))]
        pub enum PolicySpec { Default, All(Policy), ByChannel(std::collections::BTreeMap<String, Policy>) }
        #[crate::host(record(module = "options", extract, emit, transparent))]
        pub enum DebounceSpec { Default, All(std::time::Duration) }
        #[crate::host(record(module = "options", extract, emit))]
        pub struct TurnOptions {
            #[host(default = None)]
            pub policy: PolicySpec,
            #[host(default = None)]
            pub debounce: DebounceSpec,
        }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let options = &models.iter().find(|o| o.path == "options.py").expect("options.py").text;
    assert!(
        options.contains("PolicySpec: TypeAlias = Policy | dict[str, Policy] | None")
            && options.contains("    policy: PolicySpec = None\n"),
        "{options}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("return Ok(::lk_agents_abi::llm::class::PolicySpec::Default);")
            && shell.contains("::lk_agents_abi::llm::class::PolicySpec::Default => py.None().into_bound(py),")
            && shell.contains("PolicySpec::All(crate::generated::records::options::policy_from_py(obj)?)")
            && shell.contains("obj.is_instance_of::<::pyo3::types::PyFloat>() || (obj.is_instance_of::<::pyo3::types::PyInt>()"),
        "{shell}"
    );
    let not_none = read(
        r#"#[crate::host(record(module = "options", extract, transparent))]
        pub enum Spec { All(String) }
        #[crate::host(record(module = "options", extract))]
        pub struct Holder { #[host(default = None)] pub spec: Spec }"#,
    );
    assert!(not_none.err().is_some_and(|e| e.contains("default None")));
    let ambiguous = read(
        r#"#[crate::host(record(module = "options", extract, transparent))]
        pub enum Number { Count(i64), Seconds(f64) }"#,
    )
    .expect("resolves");
    assert!(
        crate::records::conversions(&ambiguous).err().is_some_and(|e| e.contains("share a class"))
    );
}

// Catches: a nested model (`RealtimeModelMetrics.InputTokenDetails`) emitted at module level,
// named without its owner where it is used, read off the module rather than its owner, or nested
// in what is no record struct.
#[test]
fn a_nested_record_is_a_class_of_its_owners_model() {
    let decls = read(
        r#"#[crate::host(record(module = "metrics", extract, emit, nested_in = "RealtimeModelMetrics"))]
        pub struct CachedTokenDetails { pub audio_tokens: u32 }
        #[crate::host(record(module = "metrics", extract, emit, nested_in = "RealtimeModelMetrics"))]
        pub struct InputTokenDetails { pub cached_tokens_details: Option<CachedTokenDetails> }
        /// One response.
        #[crate::host(record(module = "metrics", extract, emit))]
        pub struct RealtimeModelMetrics { pub input_token_details: InputTokenDetails }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let metrics = &models.iter().find(|o| o.path == "metrics.py").expect("metrics.py").text;
    assert!(
        metrics.contains("class RealtimeModelMetrics:\n    \"\"\"One response.\"\"\"\n\n    @dataclass(frozen=True, kw_only=True, slots=True)\n    class CachedTokenDetails:\n        audio_tokens: int\n")
            && metrics.contains("        cached_tokens_details: RealtimeModelMetrics.CachedTokenDetails | None\n")
            && metrics.contains("\n    input_token_details: RealtimeModelMetrics.InputTokenDetails\n")
            && !metrics.contains("\nclass InputTokenDetails"),
        "{metrics}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains(".getattr(\"RealtimeModelMetrics\")?.getattr(\"InputTokenDetails\")?")
            && shell.contains("pub fn realtime_model_metrics_input_token_details_from_py("),
        "{shell}"
    );
    let orphan = read(
        r#"#[crate::host(record(module = "metrics", extract, nested_in = "Missing"))]
        pub struct Details { pub n: u32 }"#,
    )
    .expect("resolves");
    assert!(
        crate::records::python(&orphan).err().is_some_and(|e| e.contains("nested in `Missing`"))
    );
}

// Catches: a computed default (`STTMetrics.total_tokens`, input plus output when left out) lost,
// overriding an explicit value (even 0), computed in Python rather than by the Rust function, or
// accepted where the host never reads it.
#[test]
fn an_omitted_field_is_computed_from_the_record_read() {
    let decls = read(
        r#"#[crate::host(record(module = "metrics", extract, emit))]
        pub struct SttMetrics {
            pub input_tokens: u32,
            #[host(omitted = Self::sum)]
            pub total_tokens: u32,
        }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let metrics = &models.iter().find(|o| o.path == "metrics.py").expect("metrics.py").text;
    assert!(
        metrics.contains("    total_tokens: int = None  # type: ignore[assignment]\n")
            && metrics.contains("        if self.total_tokens is None:\n            object.__setattr__(self, \"total_tokens\", _core._core_omitted_stt_metrics_total_tokens(self))\n"),
        "{metrics}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("total_tokens: ::std::default::Default::default(),")
            && shell.contains("record.total_tokens = match total_tokens { Some(v) => v, None => ::lk_agents_abi::llm::class::SttMetrics::sum(&record) };"),
        "{shell}"
    );
    let adapters: String =
        crate::adapter::emit(&decls).expect("emits").into_iter().map(|o| o.text).collect();
    assert!(
        adapters.contains("pub fn _core_omitted_stt_metrics_total_tokens<'py>(")
            && adapters
                .contains("wrap_pyfunction!(llm::_core_omitted_stt_metrics_total_tokens, m)"),
        "{adapters}"
    );
    let emitted_only = read(
        r#"#[crate::host(record(module = "metrics", emit))]
        pub struct Out { #[host(omitted = Self::sum)] pub total: u32 }"#,
    );
    assert!(emitted_only.err().is_some_and(|e| e.contains("`omitted`")));
}

// Catches: a dispatched member routed by anything but its model class, an event emitted under a
// name its model does not carry, a mirror or an ignored member lost, or a route outside a
// dispatched record accepted.
#[test]
fn a_dispatched_record_routes_its_members_by_model_class() {
    let source = r#"
        #[crate::host(record(module = "events", emit, constant(type = "agent_state_changed")))]
        pub struct AgentStateChanged { pub new_state: String }
        #[crate::host(record(module = "events", emit, constant(type = "user_turn_completed")))]
        pub struct TurnCompleted { pub text: String }
        #[crate::host(record(module = "events", emit, constant(type = "speech_done")))]
        pub struct SpeechDone { pub id: String }
        #[crate::host(record(module = "events", emit, transparent, dispatch = "session_dispatch"))]
        pub enum HostWire {
            #[host(event(mirror = "_agent_state"))]
            State(AgentStateChanged),
            #[host(ignored)]
            Turn(TurnCompleted),
            Done(SpeechDone),
        }"#;
    let decls = read(source).expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let dispatch =
        &models.iter().find(|o| o.path == "session_dispatch.py").expect("dispatch module").text;
    for want in [
        "from .events import AgentStateChanged, TurnCompleted\n",
        "PUBLIC_EVENTS: dict[type, str] = {\n    AgentStateChanged: \"agent_state_changed\",\n}\n",
        "MIRRORED_STATES: dict[type, str] = {\n    AgentStateChanged: \"_agent_state\",\n}\n",
        "IGNORED_EVENTS: frozenset[type] = frozenset(\n    {\n        TurnCompleted,\n    }\n)\n",
    ] {
        assert!(dispatch.contains(want), "{want}\n{dispatch}");
    }
    assert!(!dispatch.contains("SpeechDone"), "{dispatch}");
    let untyped = source.replace(", constant(type = \"agent_state_changed\")", "");
    let error = crate::records::python(&read(&untyped).expect("resolves")).err().expect("refused");
    assert!(error.contains("HostWire::State: an event's model names it"), "{error}");
    let undispatched = source.replace(", dispatch = \"session_dispatch\"", "");
    assert!(read(&undispatched).unwrap_err().contains("routes a member of a dispatched record"));
}

#[test]
fn transparent_records_extract_and_emit_existing_host_classes() {
    let decls = read(
        r#"
        #[crate::host(handwritten(extract, emit, host = "livekit.rtc.AudioFrame"))]
        pub struct AudioFrame;
        #[crate::host(record(module = "avatar", extract, emit, transparent))]
        pub enum AudioReceiverFrame {
            Audio(AudioFrame),
            #[host(host_type = "livekit.agents.voice.avatar.AudioSegmentEnd")]
            SegmentEnd,
        }
    "#,
    )
    .unwrap();
    let output = crate::records::conversions(&decls).unwrap();
    let rust = &output.text;
    assert!(rust.contains("obj.py().import(\"livekit.rtc\")?.getattr(\"AudioFrame\")?"));
    assert!(rust.contains(
        "obj.py().import(\"livekit.agents.voice.avatar\")?.getattr(\"AudioSegmentEnd\")?"
    ));
    assert!(!rust.contains("if obj.is_none()"));
    let models = crate::records::python(&decls).unwrap();
    let model = &models.iter().find(|o| o.path == "avatar.py").unwrap().text;
    assert!(model.contains("from livekit.rtc import AudioFrame"));
    assert!(model.contains("from livekit.agents.voice.avatar import AudioSegmentEnd"));
    assert!(model.contains("AudioReceiverFrame: TypeAlias = AudioFrame | AudioSegmentEnd"));
    assert!(!model.contains("class AudioSegmentEnd"));
    let invalid = read(
        r#"
        #[crate::host(handwritten(extract, emit, host = "livekit.rtc.AudioFrame | object"))]
        pub struct Frame;
        #[crate::host(record(module = "avatar", extract, transparent))]
        pub enum Bad { Audio(Frame), End }
    "#,
    )
    .unwrap();
    assert!(crate::records::conversions(&invalid).err().unwrap().contains("names one host class"));
}

// Catches: a plugin-facing dataclass emitted frozen or keyword-only (breaking positional
// construction and mutation), its `__post_init__` hook dropped or calling `_core`, a renamed field
// crossing under its Rust name, a host-stamped `Option` overwritten with `None` on emit, or a
// required field accepted after a defaulted one (a `TypeError` at import).
#[test]
fn a_dataclass_record_is_mutable_positional_and_calls_its_post_init_hook() {
    let decls = read(
        r#"#[crate::host(record(module = "stt", dataclass, post_init = "event_post_init", extract, emit))]
        pub struct Event {
            #[host(rename = "type")]
            pub kind: String,
            #[host(default = 0.0)]
            pub start_time: f64,
            #[host(annotation = "float", default_factory = "time.time")]
            pub created_at: Option<f64>,
        }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let stt = &models.iter().find(|o| o.path == "stt.py").expect("stt.py").text;
    assert!(
        stt.contains("@dataclass\nclass Event:\n    type: str\n    start_time: float = 0.0\n")
            && stt.contains("    created_at: float = field(default_factory=time.time)\n")
            && stt.contains(
                "    def __post_init__(self) -> None:\n        _bindings.event_post_init(self)\n"
            ),
        "{stt}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("if let Some(x) = created_at {") && shell.contains("kwargs.set_item(\"type\""),
        "{shell}"
    );
    let unordered = read(
        r#"#[crate::host(record(module = "stt", dataclass, extract))]
        pub struct Event { #[host(default = 0.0)] pub start_time: f64, pub text: String }"#,
    );
    assert!(unordered.err().is_some_and(|e| e.contains("positional")));
}

// Catches: a list field read from any iterable, so a `str` given for `list[str]` becomes its
// characters instead of a type error.
#[test]
fn a_list_field_reads_a_list_or_tuple_only() {
    let decls = read(
        r#"#[crate::host(record(module = "session", extract))]
        pub struct Keyterms { pub terms: Vec<String> }"#,
    )
    .expect("resolves");
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("v.is_instance_of::<::pyo3::types::PyList>() || v.is_instance_of::<::pyo3::types::PyTuple>()")
            && shell.contains("Keyterms.terms: expected a list or tuple"),
        "{shell}"
    );
}
