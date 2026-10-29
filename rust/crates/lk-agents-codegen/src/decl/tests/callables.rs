//! Generated callable arguments, async functions and native class contracts.

use super::*;

#[test]
fn async_module_functions_own_arguments_and_await_the_implementation() {
    let declarations = read(r#"
        #[crate::host(functions)]
        pub trait Formatting {
            #[host(function, name = "prepare")]
            async fn run(text: String, #[host(kw, default = false)] strict: bool) -> Result<Vec<String>, crate::errors::Bad>;
        }
    "#).expect("async module function resolves");
    assert_eq!(declarations.classes[0].methods[0].kind, MethodKind::AsyncFunction);
    let out = crate::adapter::emit(&declarations).expect("async function adapter emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").unwrap().text;
    assert!(
        llm.contains(
            "pub fn prepare<'py>(py: Python<'py>, text: ::std::string::String, strict: bool)"
        ),
        "{llm}"
    );
    assert!(llm.contains("future_into_py(py, async move"), "{llm}");
    assert!(llm.contains("::run(text, strict).await.map_err("), "{llm}");
    assert!(!llm.contains("&self") && !llm.contains("let this ="), "{llm}");
    let stubs = crate::class_stub::emit(&declarations).expect("async function stub emits");
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(
        text.contains("async def prepare(text: str, *, strict: bool = False) -> list[str]"),
        "{text}"
    );
}

#[test]
fn raw_rust_parameter_names_keep_their_python_keyword_spelling() {
    let declarations = read(
        r#"
        #[crate::host(functions)]
        pub trait Schema {
            #[host(function)] fn resolve(r#ref: &str) -> String;
        }
    "#,
    )
    .unwrap();
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        adapter.contains("r#ref: ::std::string::String") && adapter.contains("::resolve(&r#ref)"),
        "{adapter}"
    );
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("def resolve(ref: str) -> str") && !text.contains("r#ref"), "{text}");
}

#[test]
fn wire_enum_parameters_keep_their_existing_native_conversion() {
    let declarations = read(
        r#"
        #[crate::host(wire_enum)]
        pub enum Role { User, Assistant }
        #[crate::host(functions)]
        pub trait Roles {
            #[host(function)] fn echo(role: Role) -> Role;
        }
    "#,
    )
    .expect("legacy wire enum is a declared string boundary");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(adapter.contains("role: ::lk_agents_abi::llm::class::Role"), "{adapter}");
    assert!(adapter.contains("::echo(role)") && !adapter.contains("from_py"), "{adapter}");
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("def echo(role: str) -> str"), "{text}");
}

#[test]
fn maybe_parameters_preserve_omission_null_and_given_values() {
    let declarations = read(r#"
        use crate::utils::maybe::Maybe;
        use serde_json::Value;
        #[crate::host(functions)]
        pub trait Settings {
            #[host(function)] fn update(#[host(kw)] scalar: Maybe<Option<f64>>, data: Maybe<Option<Value>>);
        }
    "#).expect("nullable omission is declared with the shared Maybe type");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(adapter.contains("scalar = ::lk_agents_abi::Maybe::NotGiven"), "{adapter}");
    assert!(
        adapter.contains("scalar: ::lk_agents_abi::Maybe<::std::option::Option<f64>>"),
        "{adapter}"
    );
    assert!(adapter.contains("data: ::lk_agents_abi::Maybe<Bound<'_, PyAny>>"), "{adapter}");
    assert!(adapter.contains("Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven"), "{adapter}");
    assert!(adapter.contains("if value.is_none() { Ok(None) }"), "{adapter}");
    assert!(adapter.contains("value_from_py(&value"), "{adapter}");
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("from ..types import NotGivenOr"), "{text}");
    assert!(text.contains("scalar: NotGivenOr[float | None] = ..."), "{text}");
    assert!(text.contains("data: NotGivenOr[object | None] = ..."), "{text}");
}

#[test]
fn declared_mutable_classes_preserve_python_identity_and_subclassing() {
    let declarations = read(
        r#"
        #[crate::host(class(name = "Filter", module = "package.filters", subclass))]
        pub trait Filter: Sized {
            #[host(new)] fn new() -> Self;
            #[host(call)] fn update(&mut self, value: f64);
        }
    "#,
    )
    .expect("mutable subclass contract resolves");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        adapter.contains("#[pyclass(subclass, name = \"Filter\", module = \"package.filters\")]"),
        "{adapter}"
    );
    assert!(adapter.contains("fn update(&mut self"), "{adapter}");
}

// Pure exports retain their pre-declaration fork behavior; resource entries keep the guard.
#[test]
fn fork_guards_follow_group_defaults_and_entry_overrides() {
    let declarations = read(
        r#"
        #[crate::host(class(pid = false))]
        pub trait Filter: Sized {
            #[host(new)] fn new() -> Self;
            #[host(call)] fn apply(&self, value: f64) -> f64;
        }
        #[crate::host(functions(pid = false))]
        pub trait Algorithms {
            #[host(function)] fn pure() -> u32;
            #[host(function, pid = true)] async fn admitted() -> u32;
        }
        #[crate::host(functions)]
        pub trait Runtime {
            #[host(function)] fn guarded() -> u32;
            #[host(function, pid = false)] fn pure_override() -> u32;
        }
    "#,
    )
    .expect("typed fork policy is declared for classes, function groups and entries");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let llm = &adapters.iter().find(|output| output.path == "llm.rs").unwrap().text;
    assert_eq!(llm.matches("::lk_agents_abi::python::check_pid()?").count(), 2, "{llm}");
    for invalid in ["false", "0"] {
        let value = if invalid == "false" { "\"false\"" } else { invalid };
        let declaration = format!(
            "#[crate::host(functions(pid = {value}))] pub trait Wrong {{ #[host(function)] fn call(); }}"
        );
        assert!(
            read(&declaration).err().is_some_and(|error| error.contains("`pid` takes a boolean"))
        );
    }
}

#[test]
fn input_maps_accept_covariant_mapping_values_while_results_remain_dicts() {
    let declarations = read(r#"
        use std::collections::BTreeMap;
        #[crate::host(functions)]
        pub trait Bytes {
            #[host(function)] fn copy(values: BTreeMap<String, Vec<u8>>) -> BTreeMap<String, Vec<u8>>;
            #[host(function)] fn objects(values: Vec<serde_json::Value>) -> Vec<serde_json::Value>;
        }
    "#).unwrap();
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("from collections.abc import Mapping"), "{text}");
    assert!(
        text.contains("values: Mapping[str, bytes | bytearray | memoryview | list[int]]"),
        "{text}"
    );
    assert!(text.contains("-> dict[str, bytes]"), "{text}");
    assert!(text.contains("values: Sequence[object]"), "{text}");
    assert!(text.contains("-> list[object]"), "{text}");
}

#[test]
fn authoring_wrappers_preserve_native_constructor_defaults_and_keyword_arguments() {
    let declarations = read(r#"
        #[crate::host(class(name = "BasicTokenizer"))]
        #[python(wrapper(module = "tokenize_basic", path = "livekit.agents.tokenize.basic", name = "Tokenizer", base = "..tokenize:TokenizerBase"))]
        pub trait BasicTokenizer: Sized {
            #[host(new)]
            fn new(#[host(kw, default = "english")] language: String, #[host(default = false)] retain_format: bool) -> Self;
        }
        #[crate::host(functions)]
        pub trait Tokenization {
            #[host(function)]
            #[python(forward(module = "tokenize_basic"))]
            fn split_words(text: String) -> Vec<String>;
        }
    "#).unwrap();
    let modules = crate::forward::emit(&[], &crate::ir::Exposed::default(), &declarations).unwrap();
    let module = &modules.iter().find(|out| out.path == "tokenize_basic.py").unwrap().text;
    assert!(module.contains("class Tokenizer(TokenizerBase):"), "{module}");
    assert!(module.contains("language: str = \"english\""), "{module}");
    assert!(module.contains("retain_format: bool = False"), "{module}");
    assert!(
        module.contains("_core.BasicTokenizer(language=language, retain_format=retain_format)"),
        "{module}"
    );
    assert!(module.contains("from .._core import split_words"), "{module}");
    assert!(
        module.contains("Tokenizer.__module__ = \"livekit.agents.tokenize.basic\""),
        "{module}"
    );
}

#[test]
fn declared_class_instances_are_accepted_by_batch_operations() {
    let declarations = read(
        r#"
        #[crate::host(class(name = "Document"))]
        pub trait Document: Clone + Sized {
            #[host(new)] fn new(text: String) -> Self;
        }
        #[crate::host(functions)]
        pub trait Search {
            #[host(class(name = "Document"))] type Document: Document;
            #[host(function)] fn search(documents: Vec<Self::Document>) -> Vec<usize>;
        }
    "#,
    )
    .expect("a batch accepts native document handles");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(adapter.contains("cast::<super::llm::Document>()"), "{adapter}");
    assert!(adapter.contains("::search(documents)"), "{adapter}");
}

#[test]
fn transparent_tuple_values_use_shared_shape_checks_and_accept_iterable_batches() {
    let declarations = read(r#"
        #[crate::host(record(module = "tokenize", extract, emit, transparent))]
        pub enum Token { Text(String), Span((String, usize, usize)) }
        #[crate::host(proxy(host = "livekit.agents.tokenize.Tokenizer"))]
        pub trait Tokenizer: Send + Sync {
            #[host(call)] fn tokenize(&self, text: &str) -> Result<Vec<Token>, crate::errors::HostError>;
        }
    "#).expect("a typed token can carry an offset tuple");
    let shell = crate::records::conversions(&declarations).unwrap().text;
    assert!(shell.contains("obj.is_instance_of::<::pyo3::types::PyTuple>()"), "{shell}");
    assert!(
        shell.contains("tuple.len() != 3") && shell.contains("expected a tuple of 3 items"),
        "{shell}"
    );
    let tuple = Ty::Tuple(vec![Ty::Str, Ty::Int(Int::Usize), Ty::Int(Int::Usize)]);
    let shared =
        crate::adapter::from_object("obj", &tuple, "\"Token.Span\"", "Token.Span").unwrap();
    assert!(shell.contains("tuple.get_item(2)?") && shell.contains(&shared), "{shell}");
    let models = crate::records::python(&declarations).unwrap();
    let model = &models.iter().find(|model| model.path == "tokenize.py").unwrap().text;
    assert!(model.contains("Token: TypeAlias = str | tuple[str, int, int]"), "{model}");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|adapter| adapter.path == "llm.rs").unwrap().text;
    assert!(adapter.contains("try_iter()") && adapter.contains("token_from_py"), "{adapter}");
    let ambiguous = read(
        r#"
        #[crate::host(record(module = "tokenize", extract, transparent))]
        pub enum Two { First((String, usize)), Second((usize, usize)) }
    "#,
    )
    .unwrap();
    assert!(
        crate::records::conversions(&ambiguous)
            .err()
            .is_some_and(|error| error.contains("share a class"))
    );
}

#[test]
fn native_only_inputs_use_the_registered_class_without_constructing_a_host_proxy() {
    let declaration = r#"
        use std::sync::Arc;
        #[crate::host(export(kw_only = "format", defaults(format = "JPEG".to_owned())))]
        pub fn unrelated(format: String) {}
        pub struct Unrelated;
        #[crate::host(export(defaults(format = "JPEG".to_owned())))]
        impl Unrelated {}
        #[crate::host(native)]
        pub trait AudioInput: Send + Sync { fn label(&self) -> String; }
        pub struct RoomAudioInput;
        #[crate::host(object(name = "RoomAudioInput", as_dyn(AudioInput = as_audio_input)))]
        impl RoomAudioInput { pub fn as_audio_input(&self) -> Arc<dyn AudioInput> { unimplemented!() } }
        #[crate::host(class(name = "Client"))]
        pub trait Client: Sized {
            #[host(new)] fn new() -> Self;
            #[host(call)] fn set_audio_input(&self, input: Arc<dyn AudioInput>);
        }
    "#;
    let declarations =
        read(declaration).expect("legacy native registration supplies the host type");
    assert!(declarations.proxies.is_empty(), "native-only traits never generate a host proxy");
    let argument = &declarations.classes[0].methods[1].params[0].ty;
    assert!(matches!(argument, Ty::Native { classes, .. }
        if classes == &["livekit.agents._core.RoomAudioInput"]));
    let converted =
        crate::adapter::from_object("obj", argument, "\"Client.input\"", "test").unwrap();
    assert!(
        converted.contains("python::native::<dyn ::lk_agents_abi::llm::class::AudioInput>(&obj)"),
        "{converted}"
    );
    assert!(
        converted.contains(".ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err"),
        "unregistered objects must be refused: {converted}"
    );
    assert!(converted.contains("expected native livekit.agents._core.RoomAudioInput"), "{converted}");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let code = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(code.contains("python::native::<dyn ::lk_agents_abi::llm::class::AudioInput>"), "{code}");
    assert!(!code.contains("HostProxy") && !code.contains("AudioInputProxy"), "{code}");
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let stub = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(stub.contains("def set_audio_input(self, input: RoomAudioInput) -> None"), "{stub}");
    assert!(stub.contains("import RoomAudioInput"), "{stub}");
    let boxed = declaration.replace("input: Arc<dyn AudioInput>", "input: Box<dyn AudioInput>");
    assert!(read(&boxed).unwrap_err().contains("a native-only trait crosses as `Arc<dyn AudioInput>`"));
    let unregistered = declaration.replace("as_dyn(AudioInput = as_audio_input)", "pid = false");
    assert!(
        read(&unregistered)
            .unwrap_err()
            .contains("native-only trait needs a registered host class")
    );
}

// Catches: a `native(refused = ..)` trait taken through its in-place `HostProxy` (pyo3 in ABI) or
// constructing a proxy, its declared refusal replaced, or one of its classes missing from the stub.
#[test]
fn a_refused_native_trait_is_any_of_its_classes_and_refuses_other_objects() {
    let declarations = read(
        r#"
        use std::sync::Arc;
        #[crate::host(native(refused = "expected a native tracker"))]
        pub trait Tracker: Send + Sync {}
        #[crate::host(class(name = "PlaybackTracker", native = Tracker))]
        pub trait Playback: Sized { #[host(new)] fn new() -> Self; }
        #[crate::host(class(name = "MuteTracker", native = Tracker))]
        pub trait Mute: Sized { #[host(new)] fn new() -> Self; }
        #[crate::host(functions)]
        pub trait Functions { #[host(function)] fn track(tracker: Arc<dyn Tracker>); }
    "#,
    )
    .unwrap();
    assert!(declarations.proxies.is_empty());
    let functions = declarations.classes.iter().find(|c| c.form == Form::Functions).unwrap();
    let argument = &functions.methods[0].params[0].ty;
    let converted = crate::adapter::from_object("obj", argument, "\"track\"", "test").unwrap();
    assert!(
        converted.contains("python::native::<dyn ::lk_agents_abi::llm::class::Tracker>(&obj)")
            && converted.contains("\": expected a native tracker\"")
            && !converted.contains("HostProxy"),
        "{converted}"
    );
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(text.contains("def track(tracker: MuteTracker | PlaybackTracker) -> None"), "{text}");
}

#[test]
fn declared_native_classes_supply_the_same_native_only_input_annotation() {
    let declarations = read(
        r#"
        use std::sync::Arc;
        #[crate::host(native)]
        pub trait Tap: Send + Sync {}
        #[crate::host(class(name = "Recorder", native = Tap))]
        pub trait Recorder: Sized { #[host(new)] fn new() -> Self; }
        #[crate::host(functions)]
        pub trait Functions { #[host(function)] fn use_tap(tap: Arc<dyn Tap>); }
    "#,
    )
    .unwrap();
    assert!(declarations.proxies.is_empty());
    assert_eq!(declarations.classes[0].native, ["::lk_agents_abi::llm::class::Tap"]);
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(text.contains("def use_tap(tap: Recorder) -> None"), "{text}");
}

#[test]
fn dictionary_record_inputs_keep_their_declared_model_while_results_remain_dicts() {
    let declarations = read(
        r#"
        #[derive(Host)]
        #[python(typed_dict, module = "forms", name = "SettingsOptions")]
        #[binding(extract, emit, from_items)]
        pub struct Settings { pub name: String }
        #[crate::host(functions)]
        pub trait Functions {
            #[host(function)] fn echo(settings: Settings) -> Settings;
        }
    "#,
    )
    .unwrap();
    let ty = &declarations.classes[0].methods[0].params[0].ty;
    assert!(matches!(ty, Ty::Record(Record { host: HostForm::Model { module, name }, .. })
        if module == "forms" && name == "SettingsOptions"));
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let stub = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(stub.contains("from .forms import SettingsOptions"), "{stub}");
    assert!(stub.contains("def echo(settings: SettingsOptions) -> dict[str, object]"), "{stub}");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let code = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        code.contains("settings: ::lk_agents_abi::llm::class::Settings"),
        "native dictionary conversion must remain the type's own: {code}"
    );
    assert!(code.contains("::echo(settings)"), "{code}");
}

#[test]
fn callable_annotations_follow_the_declared_call_and_its_iterable_result_reader() {
    let declarations = read(r#"
        use std::sync::Arc;
        #[crate::host(record(module = "tokens", extract, emit, transparent))]
        pub enum Token { Text(String), Span((String, usize, usize)) }
        #[crate::host(proxy(host = "collections.abc.Callable"))]
        pub trait Callback: Send + Sync {
            #[host(call, name = "__call__")]
            fn tokenize(&self, text: &str) -> Result<Vec<Token>, crate::errors::HostError>;
        }
        #[crate::host(proxy(host = "collections.abc.Callable"))]
        pub trait OtherCallback: Send + Sync {
            #[host(call, name = "__call__")]
            fn apply(&self, text: &str, enabled: bool, #[host(context)] cx: &crate::invoke::CallContext) -> Result<Option<Vec<Token>>, crate::errors::HostError>;
        }
        #[crate::host(record(module = "callbacks", extract))]
        pub struct Settings { pub callback: Arc<dyn Callback> }
        #[crate::host(functions)]
        pub trait Functions {
            #[host(function)] fn consume(callback: Arc<dyn Callback>, other: Arc<dyn OtherCallback>);
        }
    "#).unwrap();
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(text.contains("from collections.abc import Callable, Iterable"), "{text}");
    assert!(text.contains("from .tokens import Token"), "{text}");
    assert!(text.contains("callback: Callable[[str], Iterable[Token]]"), "{text}");
    assert!(text.contains("other: Callable[[str, bool], Iterable[Token] | None]"), "{text}");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let code = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        code.contains(".try_iter()"),
        "the annotation must match the existing iterator reader: {code}"
    );
    assert!(
        code.contains("lk_agents_runtime::host::call_now(&crate::host::Entry")
            && code.contains("host.bind(py).getattr(\"__call__\")?"),
        "the declared callback must dispatch through HostInvoker's Python entry: {code}"
    );
    let conversions = crate::records::conversions(&declarations).unwrap().text;
    assert!(
        conversions.contains("crate::generated::llm::CallbackProxy::from_host(&v)"),
        "modern record fields use the same generated Python extractor: {conversions}"
    );
    assert!(
        !conversions.contains("HostProxy>::from_host"),
        "a modern callback record needs no ABI proxy shim: {conversions}"
    );
    let abi = crate::records::rust(&declarations).unwrap();
    assert!(
        abi.iter().all(|out| !out.text.contains("Callback")),
        "new callbacks must remain Python-only"
    );
}

#[test]
fn pure_wire_string_enum_map_keys_emit_in_the_rust_map_order_and_reject_host_input() {
    let source = r#"
        use std::collections::BTreeMap;
        #[crate::host(wire_strs)]
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        pub enum Field { Zed, Alpha }
        #[crate::host(functions)]
        pub trait Matrix {
            #[host(function)] fn values() -> BTreeMap<Field, Vec<String>>;
        }
    "#;
    let declarations = read(source).unwrap();
    let output = &declarations.classes[0].methods[0].ret.ok;
    assert_eq!(
        output.rust(),
        "::std::collections::BTreeMap<::lk_agents_abi::llm::class::Field, ::std::vec::Vec<::std::string::String>>"
    );
    assert!(
        matches!(output, Ty::Map(key, _) if matches!(&**key, Ty::Record(Record { host: HostForm::WireStrs, extract: false, emit: true, .. })))
    );
    let expression = crate::records::emit(output, "matrix").unwrap();
    assert!(
        expression.contains("for (k, x) in matrix {"),
        "iterate the enum-keyed map directly: {expression}"
    );
    assert!(
        expression.contains("into_bound_py_any(k.as_str(), py)?"),
        "each key uses the existing pure string protocol: {expression}"
    );
    assert!(
        !expression.contains("sort")
            && !expression.contains("collect::<::std::collections::BTreeMap"),
        "string keys must not be re-sorted: {expression}"
    );
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(text.contains("def values() -> dict[str, list[str]]"), "{text}");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let code = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        code.contains("let out = <MatrixImpl as ::lk_agents_abi::llm::class::Matrix>::values();")
            && code.contains("for (k, x) in out {"),
        "the adapter must retain the native enum-keyed map: {code}"
    );
    assert!(code.contains("k.as_str()"), "no ABI PyO3 enum conversion is needed: {code}");
    let input = source.replace(
        "fn values() -> BTreeMap<Field, Vec<String>>;",
        "fn values(values: BTreeMap<Field, Vec<String>>);",
    );
    assert!(read(&input).unwrap_err().contains("Field` is not declared for input"));
    // Existing explicit string conversion (RequestKind's declaration) must retain input.
    let legacy = input
        .replace("#[crate::host(wire_strs)]", "#[crate::host(string)] #[crate::host(wire_strs)]");
    let declarations = read(&legacy).expect("explicit host input wins over pure wire metadata");
    assert!(matches!(&declarations.classes[0].methods[0].params[0].ty,
        Ty::Map(key, _) if matches!(&**key, Ty::Record(Record { host: HostForm::Annotation(name), extract: true, .. }) if name == "str")));
}

#[test]
fn a_grandfathered_proxy_keeps_its_exact_legacy_abi_conversion() {
    let tree = crate::decl::tree::Crate::from_files(
        "lk_agents_abi",
        &[
            (
                "lib.rs",
                "pub mod stt; pub use stt::Stt; #[host(error(original))] pub struct HostError;",
            ),
            (
                "stt.rs",
                r#"
            #[host(proxy(host = "livekit.agents.stt.STT"))]
            pub trait Stt: Send + Sync {
                #[host(call)] fn count(&self) -> Result<u32, crate::HostError>;
            }
        "#,
            ),
        ],
    )
    .unwrap();
    let declarations = crate::decl::resolve::declarations(&tree).unwrap();
    let bridges = crate::records::rust(&declarations).unwrap();
    let bridge = &bridges.iter().find(|out| out.path == "stt.rs").unwrap().text;
    assert!(
        bridge.contains("impl crate::python::HostProxy for dyn ::lk_agents_abi::Stt {"),
        "{bridge}"
    );
    assert!(
        bridge.contains("crate::python::declared_proxy::<dyn ::lk_agents_abi::Stt>(obj)"),
        "{bridge}"
    );
}

#[test]
fn ordered_typed_maps_keep_string_extraction_and_host_insertion_order() {
    let declarations = read(r#"
        #[crate::host(functions)]
        pub trait Replacements {
            #[host(function)]
            fn replace(values: indexmap::IndexMap<String, String>) -> indexmap::IndexMap<String, String>;
        }
    "#).unwrap();
    let method = &declarations.classes[0].methods[0];
    assert!(
        matches!(&method.params[0].ty, Ty::OrderedMap(key, value) if **key == Ty::Str && **value == Ty::Str)
    );
    assert_eq!(
        method.params[0].ty.rust(),
        "::indexmap::IndexMap<::std::string::String, ::std::string::String>"
    );
    let expression =
        crate::adapter::from_object("obj", &method.params[0].ty, "\"values\"", "test").unwrap();
    assert!(
        expression.contains("dict.iter()")
            && expression.contains("collect::<PyResult<::indexmap::IndexMap<_, _>>>()"),
        "input order must reach the ordered map unchanged: {expression}"
    );
    assert!(
        expression.contains("key.extract::<::std::string::String>()")
            && expression.contains("item.extract::<::std::string::String>()"),
        "both keys and values use strict string extraction: {expression}"
    );
    assert!(
        !expression.contains("value_from_py") && !expression.contains("serde_json"),
        "there is no JSON detour: {expression}"
    );
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let stub = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(stub.contains("def replace(values: dict[str, str]) -> dict[str, str]"), "{stub}");
    let emitted = crate::records::emit(&method.ret.ok, "values").unwrap();
    assert!(
        emitted.contains("for (k, x) in values {"),
        "output order must also be retained: {emitted}"
    );
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let code = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        code.contains("values: Bound<'_, PyAny>"),
        "the generated shared reader, not a new host wrapper, decodes the map: {code}"
    );
    assert!(code.contains("::indexmap::IndexMap<_, _>"), "{code}");
}

#[test]
fn list_item_shape_policy_keeps_concrete_list_and_typed_record_annotations() {
    let declarations = read(r#"
        #[crate::host(record(module = "forms", extract))]
        pub struct Item { pub name: String }
        #[crate::host(functions)]
        pub trait Inputs {
            #[host(function)] fn read(
                #[host(default = None, shape_error(items, kind = "ValueError", message = "expected a dict"))]
                strict: Option<Vec<Item>>,
                #[host(default = None)] ordinary: Option<Vec<Item>>,
            );
        }
    "#).unwrap();
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("from .forms import Item"), "{text}");
    assert!(
        text.contains("strict: list[Item] | None = None"),
        "the policy accepts only a list and retains the declared item model: {text}"
    );
    assert!(
        text.contains("ordinary: Sequence[Item] | None = None"),
        "ordinary sequence annotations remain unchanged: {text}"
    );
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let code = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        code.contains("cast::<::pyo3::types::PyList>()"),
        "the strict annotation must reflect the existing reader: {code}"
    );
}

#[test]
fn writable_property_keeps_its_name_validates_the_value_and_propagates_failures() {
    let declarations = read(
        r#"
        #[crate::host(class(name = "Pool", pid = false))]
        pub trait Pool: Sized {
            #[host(getter)] fn target(&self) -> usize;
            #[host(setter, name = "target", pid = true)]
            fn set_target(&self, count: usize) -> Result<(), crate::errors::Bad>;
        }
    "#,
    )
    .unwrap();
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    syn::parse_file(adapter).expect("a fallible writable property emits valid Rust");
    assert!(adapter.contains("#[setter(target)]"), "{adapter}");
    assert!(adapter.contains("number_from_py::<usize>(&obj, \"count\")"), "{adapter}");
    assert!(
        adapter.contains("#[pyo3(from_py_with = _extract_pool_set_target_count)]"),
        "{adapter}"
    );
    let call = "::set_target(&self.0, count)";
    let guard = "::lk_agents_abi::python::check_pid()?";
    assert_eq!(adapter.matches(guard).count(), 1, "only this setter overrides the PID policy");
    assert!(adapter.find(guard).unwrap() < adapter.find(call).unwrap(), "{adapter}");
    assert!(adapter.contains(&format!("{call}.map_err(")), "{adapter}");
    assert!(adapter.contains("HostException>::to_py_err(&e)"), "{adapter}");
    assert!(!adapter.contains("#[pyo3(name = \"target\")]"), "the setter annotation owns its name");
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let stub = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(stub.contains("@property\n    def target(self) -> int"), "{stub}");
    assert!(stub.contains("@target.setter\n    def target(self, count: int) -> None"), "{stub}");
}

#[test]
fn property_setters_refuse_signatures_python_cannot_assign() {
    for declaration in [
        "#[host(setter)] fn set_value(&self);",
        "#[host(setter)] fn set_value(&self, first: usize, second: usize);",
        "#[host(setter)] fn set_value(&self, value: usize) -> usize;",
    ] {
        let error = refuse(declaration);
        assert!(error.contains("a setter takes one value and returns unit"), "{error}");
    }
    for declaration in [
        "#[host(setter)] async fn set_value(&self, value: usize);",
        "#[host(setter)] fn set_value(value: usize);",
    ] {
        let error = refuse(declaration);
        assert!(error.contains("the signature does not fit `setter`"), "{error}");
    }
}

#[test]
fn generated_methods_accept_legacy_native_classes_only_with_clone_extraction() {
    let source = r#"
        #[derive(Host)]
        #[class(from_py_object)]
        #[binding(class, new)]
        pub struct StartJobFields { pub job: Vec<u8> }

        #[crate::host(class(name = "Supervisor"))]
        pub trait Supervisor: Sized {
            #[host(call)]
            fn start_job(&self, fields: StartJobFields) -> StartJobFields;
        }
    "#;
    let declarations = read(source).expect("a native class uses its existing class conversion");
    let method = &declarations.classes[0].methods[0];
    assert!(matches!(&method.params[0].ty,
        Ty::Record(record) if record.extract && record.emit
            && record.host == HostForm::CoreClass("StartJobFields".into())));
    assert_eq!(method.params[0].ty, method.ret.ok);
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    syn::parse_file(adapter).expect("the generated method uses the native Rust argument type");
    assert!(adapter.contains("fields: ::lk_agents_abi::llm::class::StartJobFields"), "{adapter}");
    assert!(adapter.contains("::start_job(&self.0, fields)"), "{adapter}");
    assert!(
        !adapter.contains("start_job_fields_from_py"),
        "native classes keep their own conversion"
    );

    let without_extraction = source.replace("#[class(from_py_object)]", "#[class]");
    let error =
        read(&without_extraction).expect_err("output-only native classes cannot be arguments");
    assert!(
        error.contains("StartJobFields") && error.contains("not declared for input"),
        "{error}"
    );

    let output_only = without_extraction
        .replace("fn start_job(&self, fields: StartJobFields)", "fn start_job(&self)");
    let declarations =
        read(&output_only).expect("a native class can still be returned by a method");
    assert!(matches!(&declarations.classes[0].methods[0].ret.ok,
        Ty::Record(record) if !record.extract && record.emit
            && record.host == HostForm::CoreClass("StartJobFields".into())));
}

#[test]
fn host_owned_future_preserves_typed_completion_without_another_runtime_future() {
    let declarations = read(
        r#"
        #[crate::host(class(name = "Reader"))]
        pub trait Reader: Sized {
            #[host(call, future = Vec<u8>)]
            fn read(&self) -> Result<crate::HostRef, crate::errors::HostError>;
            #[host(call, future = ())]
            fn close(&self) -> crate::HostRef;
            #[host(getter)] fn data(&self) -> Vec<u8>;
        }
    "#,
    )
    .unwrap();
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let stub = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(stub.contains("from asyncio import Future"), "{stub}");
    assert!(stub.contains("def read(self) -> Future[bytes]"), "{stub}");
    assert!(stub.contains("def close(self) -> Future[None]"), "{stub}");
    let class = &declarations.classes[0];
    let mut imports = crate::class_stub::Imports::new(&declarations);
    assert_eq!(
        crate::class_stub::signature_parts(class, &class.methods[0], &mut imports).1,
        "Future[bytes]",
        "forwarded methods must keep the native future's completion type"
    );
    assert!(
        stub.contains("def data(self) -> bytes"),
        "PyO3 converts native byte vectors to bytes: {stub}"
    );
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        !adapter.contains("future_into_py"),
        "the adapter returns the original host future: {adapter}"
    );
}

#[test]
fn host_owned_future_refuses_incompatible_methods_and_completion_types() {
    for body in [
        "#[host(call, future = ())] fn read(&self) -> bool;",
        "#[host(getter, future = ())] fn read(&self) -> crate::HostRef;",
        "#[host(await, future = ())] async fn read(&self) -> crate::HostRef;",
    ] {
        let error = refuse(body);
        assert!(error.contains("future requires a synchronous call returning HostRef"), "{error}");
    }
    let error = refuse("#[host(call, future = \"bytes\")] fn read(&self) -> crate::HostRef;");
    assert!(error.contains("future is its Rust completion type"), "{error}");
    let error = read(
        r#"
        #[crate::host(handwritten(extract, host = "InputOnly"))]
        pub struct InputOnly;
        #[crate::host(class(name = "Reader"))]
        pub trait Reader: Sized {
            #[host(call, future = InputOnly)] fn read(&self) -> crate::HostRef;
        }
    "#,
    )
    .unwrap_err();
    assert!(error.contains("not declared for output"), "{error}");
}

#[test]
fn property_assignment_captures_context_without_an_extra_python_argument() {
    let declarations = read(
        r#"
        #[crate::host(class(name = "Pool"))]
        pub trait Pool: Sized {
            #[host(getter)] fn target(&self) -> usize;
            #[host(setter, name = "target")]
            fn set_target(&self, count: usize, #[host(context)] cx: crate::invoke::CallContext);
        }
    "#,
    )
    .unwrap();
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    syn::parse_file(adapter).expect("a contextual property setter emits valid Rust");
    assert!(adapter.contains("#[setter(target)]"), "{adapter}");
    assert!(adapter.contains("let cx = crate::host::entered();"), "{adapter}");
    assert!(adapter.contains("::set_target(&self.0, count, cx)"), "{adapter}");
    assert!(!adapter.contains("#[pyo3(signature"), "property assignment has no call signature");
    assert!(!adapter.contains("cx: "), "the binding captures context: {adapter}");
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let stub = stubs.iter().map(|out| out.out.text.as_str()).collect::<String>();
    assert!(stub.contains("@target.setter\n    def target(self, count: int) -> None"), "{stub}");
    assert!(!stub.contains("cx"), "context is hidden from Python: {stub}");
}

#[test]
fn native_class_stubs_keep_host_types_distinct_and_import_union_members() {
    let declarations = read(r#"
        use std::sync::Arc;
        #[crate::host(proxy(host = "service.Receiver"))]
        pub trait Callback {}
        #[crate::host(proxy(host = "other_service.Receiver"))]
        pub trait OtherCallback {}
        #[crate::host(handwritten(extract, emit, host = "media.AudioFrame | rtc.VideoFrame | app.avatar.End | typing.Literal['not.a.type']"))]
        pub struct Frame;
        #[crate::host(class(name = "Receiver"))]
        pub trait Native: Sized {
            #[host(new)] fn new(callback: Arc<dyn Callback>, other: Arc<dyn OtherCallback>) -> Self;
            #[host(call)] fn receive(&self) -> Frame;
        }
    "#).unwrap();
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let stub = &stubs.iter().find(|stub| stub.out.path == "_core_llm.pyi").unwrap().out.text;
    assert!(stub.contains("from service import Receiver as HostReceiver\n"), "{stub}");
    assert!(stub.contains("from other_service import Receiver as HostHostReceiver\n"), "{stub}");
    assert!(stub.contains("callback: HostReceiver"), "{stub}");
    assert!(stub.contains("other: HostHostReceiver"), "{stub}");
    assert!(stub.contains("from media import AudioFrame\n"), "{stub}");
    assert!(stub.contains("from livekit import rtc\n"), "{stub}");
    assert!(stub.contains("from app.avatar import End\n"), "{stub}");
    assert!(stub.contains("from typing import Literal\n"), "{stub}");
    assert!(stub.contains("AudioFrame | rtc.VideoFrame | End | Literal['not.a.type']"), "{stub}");
    assert!(!stub.contains("from not.a import"), "literal values are not type names: {stub}");
}

#[test]
fn static_class_calls_construct_values_and_compare_borrowed_instances() {
    let declarations = read(
        r#"
        #[crate::host(class(name = "Value", pid = false))]
        pub trait Value: Sized {
            #[host(new)] fn new(text: String) -> Self;
            #[host(call)] fn from_parts(parts: Vec<String>) -> Self;
            #[host(call, name = "__eq__")] fn equal(&self, other: &Self) -> bool;
        }
    "#,
    )
    .expect("static class construction and a borrowed comparison resolve");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let text = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        text.contains(
            "#[staticmethod]\n    #[pyo3(signature = (parts))]\n    fn from_parts(parts:"
        ),
        "{text}"
    );
    assert!(text.contains("::from_parts(parts)") && !text.contains("::from_parts(&self"), "{text}");
    assert!(
        text.contains("other: PyRef<'_, Value>") && text.contains("::equal(&self.0, &*other.0)"),
        "{text}"
    );
    let equality = text.split("#[pyo3(name = \"__eq__\")]").nth(1).unwrap();
    let attributes = equality.split("fn equal").next().unwrap();
    assert!(!attributes.contains("signature"), "fixed Python protocols have no custom signature");
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(
        text.contains("@staticmethod\n    def from_parts(parts: Sequence[str]) -> Value"),
        "{text}"
    );
    assert!(text.contains("def __eq__(self, other: Value) -> bool"), "{text}");
    let unsupported = read(
        r#"
        #[crate::host(class)]
        pub trait Value: Sized {
            #[host(await)] async fn equal(&self, other: &Self) -> bool;
        }
    "#,
    )
    .unwrap_err();
    assert!(
        unsupported.contains("own-class input is borrowed as &Self by a synchronous class call"),
        "{unsupported}"
    );
}
