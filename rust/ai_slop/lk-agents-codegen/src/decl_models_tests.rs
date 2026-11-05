//! Declared records' Python models: legacy modules, unions, constants and nested classes.

use super::*;

#[test]
fn transparent_provider_members_distinguish_native_capabilities_from_host_classes() {
    let decls = read(
        r#"
        #[crate::host(native(refused = "expected a native model"))]
        pub trait Native: Send + Sync {}
        #[crate::host(class(name = "NativeModel", native = Native))]
        pub trait Model: Send + Sync + Sized + 'static {
            #[host(new)]
            fn new() -> Self;
        }
        #[crate::host(proxy(host = "plugins.Model"))]
        pub trait Plugin: Send + Sync {
            #[host(read_once)]
            fn label(&self) -> String;
        }
        #[crate::host(record(module = "members", extract, transparent))]
        #[derive(Clone)]
        pub enum Member {
            Native(std::sync::Arc<dyn Native>),
            Host(std::sync::Arc<dyn Plugin>),
        }
        "#,
    )
    .expect("typed native-or-host members resolve");
    let output = crate::records::conversions(&decls).expect("conversions").text;
    let native = output.find("native::<dyn ::lk_agents_abi::llm::class::Native>(obj)").unwrap();
    let host = output.find("getattr(\"Model\")").unwrap();
    assert!(native < host, "native capabilities are selected before a host proxy");
    assert!(output.contains("PluginProxy::from_host(obj.as_any())"), "{output}");
    let models = crate::records::python(&decls).expect("models");
    let model = &models.iter().find(|out| out.path == "members.py").unwrap().text;
    assert!(model.contains("Member: TypeAlias = NativeModel | Model"), "{model}");
}

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

// Catches: a type-hint-only `Literal` (a model catalog no native entry takes) refused for not
// crossing, a union of them lost, or conversions written for either.
#[test]
fn literal_aliases_and_their_union_are_models_without_conversions() {
    let decls = read(
        r#"#[crate::host(record(module = "models", name = "AModels"))]
        pub enum A { #[host(rename = "a/1")] One }
        #[crate::host(record(module = "models", name = "BModels"))]
        pub enum B { #[host(rename = "b/1")] One }
        #[crate::host(record(module = "models", name = "AllModels", transparent))]
        pub enum All { A(A), B(B) }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let text = &models.iter().find(|o| o.path == "models.py").expect("models.py").text;
    assert!(
        text.contains("AModels: TypeAlias = Literal[\"a/1\"]")
            && text.contains("AllModels: TypeAlias = AModels | BModels"),
        "{text}"
    );
    let shell = crate::records::conversions(&decls).expect("conversions").text;
    assert!(!shell.contains("models"), "{shell}");
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

// Catches: a total TypedDict's defaulted key written with a hand annotation but no
// `NotRequired`, so the type demands a key the reader accepts missing.
#[test]
fn a_defaulted_typed_dict_key_is_not_required_whatever_its_annotation() {
    let decls = read(
        r#"#[crate::host(record(module = "session", typed_dict, extract))]
        pub struct Dial {
            pub number: String,
            #[host(default = None, annotation = "str | None")]
            pub token: Option<String>,
            #[host(default = None)]
            pub caller: Option<String>,
        }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let session = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(
        session.contains("class Dial(TypedDict):\n")
            && session.contains("    number: str\n")
            && session.contains("    token: NotRequired[str | None]\n")
            && session.contains("    caller: NotRequired[str]\n"),
        "{session}"
    );
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
        metrics.contains("class RealtimeModelMetrics:\n    \"\"\"One response.\"\"\"\n\n    @dataclass(kw_only=True, slots=True)\n    class CachedTokenDetails:\n        audio_tokens: int\n")
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
// name its model does not carry, an ignored member lost, or a route outside a
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
            #[host(event)]
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
        #[crate::host(codec(extract, emit, host = "livekit.rtc.AudioFrame"))]
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
        #[crate::host(codec(extract, emit, host = "livekit.rtc.AudioFrame | object"))]
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
                "    def __post_init__(self) -> None:\n        from .. import _bindings\n\n        _bindings.event_post_init(self)\n"
            ),
        "{stt}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("if let Some(x) = created_at {")
            && shell.contains("kwargs.set_item(\"type\""),
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

#[test]
fn an_extract_only_string_enum_needs_no_python_type_cache() {
    let declarations = read(
        r#"
        #[crate::host(record(module = "config", enum, extract))]
        pub enum Kind { Enabled, Disabled }
    "#,
    )
    .expect("enum resolves");
    let output = crate::records::conversions(&declarations).expect("conversions emit");
    assert!(output.text.contains("fn kind_from_py"));
    assert!(!output.text.contains("static KIND:"));
}

#[test]
fn external_records_generate_structural_conversions_and_reuse_the_host_class() {
    let declarations = super::read(r#"
        #[crate::host(record(module = "options", name = "Options", host_type = "package.Options", extract, emit))]
        pub struct Options { pub retry: u32, pub timeout: Option<f64> }
    "#).unwrap();
    let rust = crate::records::conversions(&declarations).unwrap().text;
    assert!(rust.contains("options_from_py"));
    assert!(rust.contains("attr_with(obj, \"Options\", \"retry\""));
    assert!(!rust.contains("obj.is_instance(OPTIONS"));
    let python = crate::records::python(&declarations).unwrap();
    let python = &python.iter().find(|out| out.path == "options.py").unwrap().text;
    assert!(python.contains("from package import Options as _HostOptions"));
    assert!(python.contains("Options = _HostOptions"));
    assert!(!python.contains("class Options"));
}

#[test]
fn capture_variants_generate_the_reverse_error_mapping_and_refuse_ambiguity() {
    let source = r#"
        #[crate::host(error(kind = "RuntimeError"))]
        pub enum Failure {
            #[host(delegate, capture)]
            Host(crate::errors::HostError),
        }
    "#;
    let declarations = read(source).unwrap();
    let binding = &declarations.errors.iter().find(|e| e.family == "llm").unwrap().binding;
    assert!(binding.contains("impl crate::py_errors::FromHostErr"));
    assert!(binding.contains("Self::Host(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, err))"));
    let ambiguous = source.replace(
        "Host(crate::errors::HostError),",
        "Host(crate::errors::HostError), #[host(delegate, capture)] Other(crate::errors::HostError),",
    );
    assert!(read(&ambiguous).unwrap_err().contains("only one variant captures"));
}

#[test]
fn boundary_records_generate_complete_converters_and_flattened_host_shapes() {
    let declarations = read(
        r#"
        #[host(record(module = "wire", mapping, extract))]
        pub struct Flags { #[host(default = false)] pub enabled: bool }
        #[host(record(module = "wire", mapping, extract))]
        pub struct Options {
            pub sample_rate: u32,
            #[host(flatten)] pub flags: Flags,
        }
    "#,
    )
    .unwrap();
    let rust = crate::records::conversions(&declarations).unwrap().text;
    assert!(rust.contains("pub fn options_from_py"));
    assert!(rust.contains("item_with(obj, \"Options\", \"sample_rate\""));
    assert!(rust.contains("flags_from_py(obj)"));
    assert!(rust.contains("let value = ::lk_agents_abi::llm::class::Options"));
    assert!(!rust.contains("access::"));
    let python = crate::records::python(&declarations).unwrap();
    let python = &python.iter().find(|out| out.path == "wire.py").unwrap().text;
    assert!(python.contains("enabled:"));
    assert!(!python.contains("flags:"));
}

#[test]
fn generated_record_codecs_run_after_scalar_validation_and_preserve_source_keys() {
    let declarations = read(
        r#"
        #[host(record(module = "wire", host_type = "sdk.Frame", boundary, extract, emit))]
        pub struct Frame {
            #[host(late, extract_with = "py_values::pcm_snapshot")] pub data: Vec<u8>,
            pub sample_rate: u32,
        }
    "#,
    )
    .unwrap();
    let rust = crate::records::conversions(&declarations).unwrap().text;
    assert!(rust.find("let sample_rate").unwrap() < rust.find("let data").unwrap());
    assert!(rust.contains("crate::py_values::pcm_snapshot(v, \"Frame.data\")"));
    assert!(rust.contains("kwargs.set_item(\"sample_rate\""));
    assert!(!rust.contains("with_sample_rate"));
}

#[test]
fn structural_boundary_extraction_normalizes_attributes_and_flattened_values() {
    let source = r#"
        #[host(record(module = "wire", host_type = "sdk.PublishOptions", boundary, extract))]
        pub struct PublishWire { pub source: u32 }
        #[host(record(module = "wire", host_type = "sdk.Options", boundary, extract))]
        pub struct Options {
            #[host(read_as = f64, normalize = crate::llm::class::duration_from_seconds)]
            pub timeout: std::time::Duration,
            #[host(flatten, read_as = PublishWire, try_normalize = crate::llm::class::publish_from_wire)]
            pub publish: u32,
        }
    "#;
    let decls = read(source).unwrap();
    let rust = crate::records::conversions(&decls).unwrap().text;
    let converter = rust.split("pub fn options_from_py").nth(1).unwrap();
    assert!(converter.contains("::lk_agents_abi::llm::class::duration_from_seconds(crate::py_values::attr_with(obj, \"Options\", \"timeout\""), "{converter}");
    assert!(converter.contains("number_from_py::<f64>"), "{converter}");
    assert!(converter.contains("::lk_agents_abi::llm::class::publish_from_wire((crate::generated::records::wire::publish_wire_from_py(obj))?).map_err"), "{converter}");
    assert!(!converter.contains("\"publish\", |v|"), "flattened values read the whole object");
    let normalized = converter.find("let timeout =").unwrap();
    let constructed = converter.find("let value = ::lk_agents_abi::llm::class::Options").unwrap();
    assert!(normalized < constructed);

    let missing_boundary =
        source.replace("host_type = \"sdk.Options\", boundary", "host_type = \"sdk.Options\"");
    let decls = read(&missing_boundary).unwrap();
    assert!(
        crate::records::conversions(&decls).err().unwrap().contains("requires a boundary record")
    );
}

#[test]
fn boundary_declarations_reject_inferred_or_ambiguous_structure() {
    for source in [
        r#"#[host(record(module = "wire", host_type = "sdk.X", boundary, extract))] pub struct X { #[host(lenient)] pub value: u32 }"#,
        r#"#[host(record(module = "wire", host_type = "sdk.X", boundary, extract))] pub struct X { #[host(source(attrs(undeclared), constant = "sdk.KEY"))] pub value: Option<u32> }"#,
        r#"#[host(record(module = "wire", host_type = "sdk.X", boundary, extract, emit))] pub struct X { #[host(emit_if_none = "missing")] pub value: Vec<u32> }"#,
    ] {
        let declarations = read(source).unwrap();
        assert!(crate::records::conversions(&declarations).is_err(), "{source}");
    }
}

// Catches: every record frozen again (each frozen class generates its setattr, delattr, hash and
// pickling hooks at import), or an identity used as a dict key left unhashable.
#[test]
fn only_a_record_whose_rust_type_derives_hash_is_frozen() {
    let decls = crate::decl::tests::read(
        r#"
        #[crate::host(record(module = "ids", extract, emit))]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct FrameId { pub value: String }
        #[crate::host(record(module = "ids", extract, emit))]
        #[derive(Debug, Clone, PartialEq)]
        pub struct Snapshot { pub value: String }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let ids = &models.iter().find(|o| o.path == "ids.py").expect("ids.py").text;
    assert!(
        ids.contains("@dataclass(frozen=True, kw_only=True, slots=True)\nclass FrameId:")
            && ids.contains("@dataclass(kw_only=True, slots=True)\nclass Snapshot:"),
        "{ids}"
    );
}
