use crate::decl::tests::read;

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

// Catches: declared records overwriting the authored models of their module, or redefining one.
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

// Catches: a scalar-or-map option needing wrapper classes, `None` not crossing as the unit
// variant, or `int` and `float` variants left ambiguous.
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
            && shell.contains(
                "::lk_agents_abi::llm::class::PolicySpec::Default => py.None().into_bound(py),"
            )
            && shell.contains(
                "PolicySpec::All(crate::generated::records::options::policy_from_py(obj)?)"
            )
            && shell.contains("obj.is_instance_of::<::pyo3::types::PyDelta>()")
            && shell.contains("extract_at::<::std::time::Duration>(obj, "),
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

// Catches: an options record's `Kwargs` missing, required (not `total=False`), annotated with
// `NotGivenOr`, a duration not a `timedelta`, its field docs dropped, or not exported beside it.
#[test]
fn an_options_record_emits_its_keyword_arguments() {
    let decls = read(
        r#"use crate::utils::maybe::Maybe;
        #[crate::host(record(module = "options", path = "livekit.agents", dataclass, extract, kwargs))]
        pub struct ServerOptions {
            /// Seconds of grace.
            pub drain_timeout: Maybe<std::time::Duration>,
            pub http_proxy: Maybe<Option<String>>,
        }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let options = &models.iter().find(|o| o.path == "options.py").expect("options.py").text;
    assert!(
        options.contains(
            "class ServerOptionsKwargs(TypedDict, total=False):\n    \"\"\"`ServerOptions`'s fields as keyword arguments.\"\"\"\n\n    drain_timeout: timedelta\n    \"\"\"Seconds of grace.\"\"\"\n    http_proxy: str | None\n"
        ) && options.contains("ServerOptionsKwargs.__module__ = \"livekit.agents\"\n"),
        "{options}"
    );
    let placed = crate::exports::placed(&decls, &[]);
    assert!(placed["livekit.agents"].iter().any(|(_, n)| n == "ServerOptionsKwargs"));
    let typed = read(
        r#"use crate::utils::maybe::Maybe;
        #[crate::host(record(module = "options", typed_dict, kwargs, extract))]
        pub struct Keys { pub a: Maybe<String> }"#,
    );
    assert!(typed.err().is_some_and(|e| e.contains("kwargs follows an options struct")));
}

// Catches: a `one_or_many` list refusing one item (`content="hello"`), its kwargs annotation
// refusing it, a field without a default optional in the kwargs, or an enum class's method hook
// dropped.
#[test]
fn a_model_takes_one_item_for_its_list_and_an_enum_class_has_hook_methods() {
    let decls = read(
        r#"#[crate::host(record(module = "chat", pydantic, extract, kwargs))]
        pub struct Message {
            #[host(one_or_many)]
            pub content: Vec<String>,
            #[host(default = false)]
            pub interrupted: bool,
        }
        #[crate::host(record(module = "chat", enum))]
        #[python(method(name = "path", ret = "str", hook = "clip_path"))]
        pub enum Clip { #[host(rename = "a.ogg")] A }"#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let chat = &models.iter().find(|o| o.path == "chat.py").expect("chat.py").text;
    assert!(
        chat.contains(
            "content: Annotated[list[str], BeforeValidator(lambda v: v if isinstance(v, list) else [v])]\n"
        ) && chat.contains("    content: Required[str | list[str]]\n    interrupted: bool\n")
            && chat.contains("    def path(self) -> str:\n")
            && chat.contains("return _bindings.clip_path(self)"),
        "{chat}"
    );
    let dataclass = read(
        r#"#[crate::host(record(module = "chat", dataclass, extract))]
        pub struct Message { #[host(one_or_many)] pub content: Vec<String> }"#,
    );
    assert!(dataclass.err().is_some_and(|e| e.contains("pydantic model's list field")));
}

// Catches: a nested model emitted at module level or named without its owner.
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

// Catches: a computed default lost, overriding an explicit value (even 0), or computed in Python
// rather than by the Rust function.
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

// Catches: a dispatched member routed by anything but its model class, an ignored member lost,
// or an emitted event missing from `EventTypes` and `AgentEvent`.
#[test]
fn a_dispatched_record_routes_its_members_by_model_class() {
    let source = r#"
        #[crate::host(record(module = "events", emit, constant(type = "agent_state_changed")))]
        pub struct AgentStateChanged { pub new_state: String }
        #[crate::host(record(module = "events", emit, constant(type = "user_turn_completed")))]
        pub struct TurnCompleted { pub text: String }
        #[crate::host(record(module = "events", emit, constant(type = "speech_done")))]
        pub struct SpeechDone { pub id: String }
        #[crate::host(record(module = "events", emit, constant(type = "close")))]
        pub struct Closed { pub reason: String }
        #[crate::host(record(module = "events", emit, transparent, dispatch = "session_dispatch"))]
        pub enum HostWire {
            #[host(event)]
            State(AgentStateChanged),
            #[host(ignored)]
            Turn(TurnCompleted),
            #[host(emits = "Closed")]
            Done(SpeechDone),
        }"#;
    let decls = read(source).expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let dispatch =
        &models.iter().find(|o| o.path == "session_dispatch.py").expect("dispatch module").text;
    for want in [
        "from .events import AgentStateChanged, Closed, TurnCompleted\n",
        "EventTypes = Literal[\n    \"agent_state_changed\",\n    \"close\",\n]\n",
        "AgentEvent = Annotated[\n    AgentStateChanged\n    | Closed,\n    Field(discriminator=\"type\"),\n]\n",
        "PUBLIC_EVENTS: dict[type, str] = {\n    AgentStateChanged: \"agent_state_changed\",\n}\n",
        "IGNORED_EVENTS: frozenset[type] = frozenset(\n    {\n        TurnCompleted,\n    }\n)\n",
    ] {
        assert!(dispatch.contains(want), "{want}\n{dispatch}");
    }
    assert!(!dispatch.contains("SpeechDone"), "{dispatch}");
    let unknown = source.replace("emits = \"Closed\"", "emits = \"Missing\"");
    let error = crate::records::python(&read(&unknown).expect("resolves")).err().expect("refused");
    assert!(error.contains("emits names no declared record `Missing`"), "{error}");
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

// Catches: a plugin-facing dataclass emitted frozen or keyword-only, its `__post_init__` hook
// dropped, a renamed field crossing under its Rust name, or a required field after a defaulted one.
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
    let declarations = read(r#"
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
fn a_read_only_external_record_imports_its_class_for_typing_only() {
    let declarations = read(r#"
        #[crate::host(record(module = "spans", name = "Span", host_type = "optional.sdk.Span", extract))]
        pub struct Span { pub name: String }
    "#).unwrap();
    let python = crate::records::python(&declarations).unwrap();
    let python = &python.iter().find(|out| out.path == "spans.py").unwrap().text;
    assert!(python.contains("if TYPE_CHECKING:\n    from optional.sdk import Span as _HostSpan"));
    assert!(python.contains("Span: TypeAlias = \"_HostSpan\""));
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

// Catches: a foreign class's nested attribute copied by hand (`ctx.trace_flags.sampled`), or a
// constructor constant retyped at each call site.
#[test]
fn boundary_paths_read_nested_attributes_and_fixed_keywords_build_the_host_class() {
    let source = r#"
        #[host(record(module = "wire", host_type = "otel.SpanContext", boundary, extract, emit, fixed(is_remote = false)))]
        pub struct Ids {
            pub trace_id: u64,
            #[host(path = "trace_flags.sampled", emit_class = "otel.TraceFlags")]
            pub sampled: bool,
        }
    "#;
    let rust = crate::records::conversions(&read(source).unwrap()).unwrap().text;
    assert!(rust.contains("obj.clone().getattr(\"trace_flags\")?.getattr(\"sampled\")?"), "{rust}");
    assert!(rust.contains("kwargs.set_item(\"is_remote\", false)?"), "{rust}");
    assert!(rust.contains("kwargs.set_item(\"trace_flags\", crate::py_calls::call("), "{rust}");
    assert!(rust.contains("\"otel\", \"TraceFlags\""), "{rust}");
    let optional = r#"#[host(record(module = "wire", host_type = "app.Asked", boundary, extract))] pub struct Asked { #[host(path = "tool.info.name", missing_default)] pub tool: Option<String> }"#;
    let rust_optional = crate::records::conversions(&read(optional).unwrap()).unwrap().text;
    assert!(
        rust_optional.contains(".filter(|m| !m.is_none()).and_then(|m| m.getattr(\"info\").ok())"),
        "{rust_optional}"
    );
    let unbuilt = source.replace(", emit_class = \"otel.TraceFlags\"", "");
    assert!(crate::records::conversions(&read(&unbuilt).unwrap()).is_err());
    let unemitted = r#"#[host(record(module = "wire", host_type = "otel.X", boundary, extract, fixed(a = 1)))] pub struct X { pub a: u64 }"#;
    assert!(read(unemitted).is_err());
}

// Catches: an ASGI message converted by a handwritten match (renamed keys, defaults and the
// bytes/text choice retyped in Python), or a shared tag accepting both or neither key.
#[test]
fn tagged_dict_unions_read_and_emit_plain_dicts_by_their_tag() {
    let source = r#"
        #[host(record(module = "asgi", typed_dict, tag = "type", extract, emit))]
        pub enum Message {
            #[host(tag = "http.response.body | websocket.http.response.body")]
            Body { #[host(rename = "body", missing_default)] data: Vec<u8>, #[host(rename = "more_body", missing_default)] more: bool },
            #[host(tag = "websocket.send")]
            Text { text: String },
            #[host(tag = "websocket.send")]
            Binary { #[host(rename = "bytes")] data: Vec<u8> },
            #[host(tag = "websocket.close")]
            Close { #[host(default = 1000)] code: u16, reason: Option<String> },
            #[host(tag = "http.response.trailers")]
            Trailers,
        }
    "#;
    let decls = read(source).unwrap();
    let rust = crate::records::conversions(&decls).unwrap().text;
    assert!(
        rust.contains("\"http.response.body\" => ::lk_agents_abi::llm::class::Message::Body"),
        "{rust}"
    );
    assert!(rust.contains("\"websocket.http.response.body\" =>"), "{rust}");
    assert!(rust.contains("item(\"more_body\")?.filter(|v| !v.is_none())"), "{rust}");
    assert!(rust.contains("websocket.send carries exactly one of text, bytes"), "{rust}");
    assert!(rust.contains("(true, false) => ::lk_agents_abi::llm::class::Message::Text"), "{rust}");
    assert!(rust.contains("None => 1000"), "{rust}");
    assert!(rust.contains("dict.set_item(\"type\", \"http.response.body\")?"), "{rust}");
    assert!(
        rust.contains(
            "Message::Trailers => { dict.set_item(\"type\", \"http.response.trailers\")?;"
        ),
        "{rust}"
    );
    let python = crate::records::python(&decls).unwrap();
    let python = &python.iter().find(|out| out.path == "asgi.py").unwrap().text;
    assert!(python.contains("Message: TypeAlias = MutableMapping[str, Any]"), "{python}");
    assert!(!python.contains("class MessageBody"), "{python}");
    assert!(read(&source.replace("#[host(tag = \"http.response.trailers\")]", "")).is_err());
    assert!(read(&source.replace("typed_dict, tag", "tag")).is_err());
}

// Catches: an isinstance ladder over host outcome classes handwritten in Python, or a fallback
// member that shadows the classes declared before it.
#[test]
fn transparent_members_read_attributes_and_take_any_other_value_last() {
    let source = r#"
        #[host(record(module = "outcomes", transparent, extract))]
        pub enum Outcome {
            #[host(host_type = "app.Failed", path = "error")]
            Failed(String),
            #[host(host_type = "app.Done", path = "result.value", extract_with = "py_values::jsonable")]
            Done(serde_json::Value),
            #[host(otherwise, extract_with = "py_values::jsonable")]
            Value(serde_json::Value),
        }
    "#;
    let rust = crate::records::conversions(&read(source).unwrap()).unwrap().text;
    let reader = rust.split("pub fn outcome_from_py").nth(1).unwrap();
    assert!(reader.contains("let member = obj.clone().getattr(\"error\")?;"), "{reader}");
    assert!(reader.contains("obj.clone().getattr(\"result\")?.getattr(\"value\")?"), "{reader}");
    assert!(reader.contains("crate::py_values::jsonable(v, \"Outcome.Value\")"), "{reader}");
    assert!(!reader.contains("not_a(obj"), "{reader}");
    let early = source.replace("#[host(otherwise, ", "#[host(").replace(
        "#[host(host_type = \"app.Failed\", path = \"error\")]",
        "#[host(otherwise, path = \"error\")]",
    );
    assert!(crate::records::conversions(&read(&early).unwrap()).is_err());
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

// Catches: a generic union losing its type parameter, a branch not inheriting its base, or a
// cached view on a slotted model.
#[test]
fn a_generic_union_of_dataclass_subclasses_shares_its_base_and_caches_a_view() {
    let source = |base: &str| {
        format!(
            r#"
        #[crate::host(record(module = "results", name = "_Base", {base}))]
        #[python(
            method(name = "items", ret = "list[int]", hook = "base_items"),
            property(name = "cursor", ret = "int", hook = "base_cursor", cached)
        )]
        pub struct Base {{ #[host(rename = "_items")] pub items: Vec<i64> }}
        #[crate::host(record(module = "results", path = "pkg.results", dataclass, kw_only,
            base = "_Base", generic = "T", constant(status = "done")))]
        pub struct Done {{ #[host(annotation = "T")] pub value: String }}
        #[crate::host(record(module = "results", path = "pkg.results", dataclass, kw_only,
            base = "_Base", constant(status = "empty")))]
        pub struct Empty {{}}
        #[crate::host(record(module = "results", name = "Outcome", path = "pkg.results",
            transparent, generic = "T", generic_default = "str"))]
        pub enum Outcome {{ #[host(annotation = "Done[T]")] Done(Done), Empty(Empty) }}
        "#
        )
    };
    let decls = read(&source("dataclass, kw_only")).expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let text = &models.iter().find(|o| o.path == "results.py").expect("results.py").text;
    assert!(
        text.contains("T = TypeVar(\"T\", default=str)\n")
            && text.contains("class Done(_Base, Generic[T]):\n")
            && text.contains("class Empty(_Base):\n")
            && text.contains("    @functools.cached_property\n    def cursor(self) -> int:\n")
            && text.contains("        return _bindings.base_items(self)\n")
            && text.contains(
                "Outcome = TypeAliasType(\n    \"Outcome\",\n    Done[T] | Empty,\n    type_params=(T,),\n)"
            ),
        "{text}"
    );
    let slotted = read(&source("dataclass, kw_only, slots"));
    assert!(slotted.err().is_some_and(|e| e.contains("unslotted")));
    let crossing = read(
        r#"#[crate::host(record(module = "results", name = "_Base", dataclass))]
        pub struct Base { pub items: Vec<i64> }
        #[crate::host(record(module = "results", dataclass, base = "_Base", extract))]
        pub struct Done { pub value: String }"#,
    );
    assert!(crossing.err().is_some_and(|e| e.contains("it is never read back")));
}

// Catches: an emitted dataclass subclass redeclaring its base's fields in its model, its emission
// missing the flattened base's keywords, or an emitted subclass that holds no base record.
#[test]
fn an_emitted_dataclass_subclass_flattens_its_base_into_its_keywords() {
    let source = r#"
        #[crate::host(record(module = "runs", name = "_Base", emit, dataclass, kw_only))]
        #[derive(Clone)]
        pub struct Base { pub usage: u32, #[host(rename = "_items")] pub items: Vec<String> }
        #[crate::host(record(module = "runs", emit, dataclass, kw_only, base = "_Base"))]
        #[derive(Clone)]
        pub struct Done { #[host(flatten)] pub base: Base, pub question: String }
    "#;
    let decls = read(source).expect("resolves");
    let rust = crate::records::conversions(&decls).unwrap().text;
    assert!(
        rust.contains("let ::lk_agents_abi::llm::class::Base { usage, items } = base;")
            && rust.contains("kwargs.set_item(\"_items\"")
            && rust.contains("kwargs.set_item(\"question\""),
        "{rust}"
    );
    let models = crate::records::python(&decls).unwrap();
    let model = &models.iter().find(|out| out.path == "runs.py").unwrap().text;
    let done = &model[model.find("class Done(_Base)").expect("subclasses its base")..];
    assert!(done.contains("question: str") && !done.contains("usage:"), "{model}");
    let unflattened = source.replace("#[host(flatten)] pub base: Base, ", "");
    let err = crate::records::python(&read(&unflattened).unwrap()).err().expect("refused");
    assert!(err.contains("an emitted dataclass subclass flattens its base"), "{err}");
}

// Catches: an emit-only codec crossing through a record not converted into it first, a field hook
// not called on the emitted value, or `resolve` accepted on a field the host passes in.
#[test]
fn emitted_values_cross_as_a_declared_record_or_through_a_field_hook() {
    let decls = read(
        r#"
        #[crate::host(record(module = "runs", emit))]
        pub struct Finished { pub done: bool }
        #[crate::host(codec(emit, host = "Finished", via = crate::llm::class::Finished))]
        pub struct Settled { pub done: bool }
        impl From<Settled> for Finished { fn from(s: Settled) -> Self { Finished { done: s.done } } }
        #[crate::host(record(module = "runs", emit))]
        pub struct Update {
            pub settled: Settled,
            #[host(resolve(module = "pkg.io", name = "Handle._of"))]
            pub handle: crate::utils::host_ref::HostRef,
        }
    "#,
    )
    .expect("resolves");
    let rust = crate::records::conversions(&decls).unwrap().text;
    assert!(
        rust.contains("let scalar: ::lk_agents_abi::llm::class::Finished = settled.into();")
            && rust.contains("finished_to_py(py, scalar)")
            && rust.contains("py.import(\"pkg.io\")?.getattr(\"Handle\")?.getattr(\"_of\")?"),
        "{rust}"
    );
    let err = read(
        r#"
        #[crate::host(record(module = "runs", extract))]
        pub struct Taken {
            #[host(resolve(module = "pkg.io", name = "Handle._of"))]
            pub handle: crate::utils::host_ref::HostRef,
        }
    "#,
    )
    .unwrap_err();
    assert!(err.contains("resolve shapes an emit-only record's field"), "{err}");
}
