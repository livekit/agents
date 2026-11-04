//! Contracts of the resolved declaration model and its generated adapters.

#[test]
fn emit_only_enum_variants_do_not_require_exception_readers() {
    let decls = crate::decl::tests::read(
        r#"
        #[crate::host(error(kind = "RuntimeError"))]
        pub struct Failure(pub String);
        #[crate::host(record(module = "session", emit))]
        pub enum Outcome {
            Failed {
                #[host(default = None)]
                cause: Option<Failure>,
            },
            #[host(field = "cause")]
            Error(Failure),
        }
        "#,
    )
    .expect("emitted exception fields resolve");
    let models = crate::records::python(&decls).expect("models");
    let session = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(session.contains("cause: BaseException | None = None"), "{session}");
    assert!(session.contains("cause: BaseException"), "{session}");
    let conversions = crate::records::conversions(&decls).expect("exception values emit").text;
    assert!(conversions.contains("fn outcome_to_py"), "{conversions}");
    assert!(conversions.contains("ExceptionValue::to_exception"), "{conversions}");
    assert!(!conversions.contains("fn outcome_from_py"), "{conversions}");

    let error = crate::decl::tests::read(
        r#"
        #[crate::host(error(kind = "RuntimeError"))]
        pub struct Failure(pub String);
        #[crate::host(record(module = "session", extract))]
        pub enum Outcome { #[host(field = "cause")] Error(Failure) }
        "#,
    )
    .expect_err("exceptions cannot be read back as native failures");
    assert!(error.contains("emitted record field only"), "{error}");
}

// Catches: a `record(.., enum)` crossing as a bare string (consumers compare `CloseReason.ERROR`
// and read `.value`), or a member read back by anything but its string.
#[test]
fn an_enum_record_crosses_as_its_str_enum_members() {
    let decls = crate::decl::tests::read(
        r#"
        /// Why a session closed.
        #[crate::host(record(module = "session", path = "livekit.agents.session", extract, emit, enum))]
        pub enum CloseReason {
            Error,
            /// The job ended.
            #[host(rename = "job-shutdown")]
            JobShutdown,
        }
        "#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let session = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(
        session.contains("from enum import StrEnum, unique\n")
            && session.contains("@unique\nclass CloseReason(StrEnum):\n    \"\"\"Why a session closed.\"\"\"\n\n    ERROR = \"error\"\n    JOB_SHUTDOWN = \"job-shutdown\"\n    \"\"\"The job ended.\"\"\"\n")
            && session.contains("\nCloseReason.__module__ = \"livekit.agents.session\"\n"),
        "{session}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("::lk_agents_abi::llm::class::CloseReason::JobShutdown => \"job-shutdown\"")
            && shell.contains("crate::py_calls::call(CLOSE_REASON.import(py, \"livekit.agents._generated.session\", \"CloseReason\")?.as_any(), (value,), None)")
            && shell.contains("\"job-shutdown\" => Ok(::lk_agents_abi::llm::class::CloseReason::JobShutdown)"),
        "{shell}"
    );
    let err = crate::decl::tests::read(
        r#"#[crate::host(record(module = "session", emit, enum))]
        pub enum Mixed { A, B(String) }"#,
    )
    .expect_err("refused");
    assert!(err.contains("record(.., enum) is a unit-only enum"), "{err}");
}

// Catches: a host timestamp the model cannot default (`time.time` at construction), one Rust
// sends although the host stamps it (`skip`), or a stamped field not read back.
#[test]
fn a_default_factory_field_is_stamped_by_the_host() {
    let decls = crate::decl::tests::read(
        r#"
        #[crate::host(record(module = "session", extract, emit))]
        pub struct Stamped {
            pub state: String,
            #[host(default_factory = "time.time")] pub at: f64,
            #[host(skip, default_factory = "time.time")] pub created_at: f64,
        }
        "#,
    )
    .expect("resolves");
    let models = crate::records::python(&decls).expect("models");
    let session = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(
        session.contains("import time\nfrom dataclasses import dataclass, field\n")
            && session.contains("    at: float = field(default_factory=time.time)\n")
            && session.contains("    created_at: float = field(default_factory=time.time)\n"),
        "{session}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("kwargs.set_item(\"at\"")
            && !shell.contains("kwargs.set_item(\"created_at\"")
            && shell.contains("\"Stamped\", \"created_at\""),
        "{shell}"
    );
}

// Catches: a `_core` call whose context is captured later than its entry (or not at all), a
// captured context surfacing as a host parameter, or a proxy passing what it runs in to the host.
#[test]
fn a_call_context_is_captured_at_entry_and_never_crosses() {
    use crate::decl::tests::{proxy, read};
    let decls = read(&format!(
        "{}\nuse crate::invoke::CallContext;\n#[crate::host(class(name = \"Session\"))]\npub trait Session: Sized {{\n    #[host(new)] fn new() -> Self;\n    #[host(await)] async fn say(&self, text: String, #[host(context)] cx: CallContext) -> Result<(), HostError>;\n}}",
        proxy("#[host(await)] async fn listen(&self, span: &Cx, #[host(context)] cx: &crate::invoke::CallContext, text: String) -> Result<(), HostError>;")
    ))
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains("#[pyo3(signature = (text))]")
            && llm.contains("let cx = crate::py_calls::entered();\n        let this =")
            && llm.contains("::say(&this, text, cx)"),
        "{llm}"
    );
    assert!(
        llm.contains("\"Callbacks.listen\", cx, &[], ")
            && llm.contains("let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [")
            && !llm.contains("cx_arg"),
        "{llm}"
    );
    let stubs = crate::class_stub::emit(&decls).expect("stubs");
    let stub = stubs.iter().map(|s| s.out.text.as_str()).collect::<String>();
    assert!(stub.contains("async def say(self, text: str) -> None"), "{stub}");
}

#[test]
fn shared_associated_classes_preserve_the_existing_arc_in_both_directions() {
    let decls = crate::decl::tests::read(
        r#"
        use std::sync::Arc;
        #[crate::host(class(name = "Store"))]
        pub trait Store: Sized { #[host(new)] fn new() -> Self; }
        #[crate::host(class(name = "Session"))]
        pub trait Session: Sized {
            #[host(class(name = "Store"))] type Store;
            #[host(new)] fn new(store: Option<Arc<Self::Store>>) -> Self;
            #[host(getter)] fn store(&self) -> Arc<Self::Store>;
        }
    "#,
    )
    .expect("shared associated classes resolve");
    let output = crate::adapter::emit(&decls).expect("bindings emit");
    let text = output.iter().map(|output| output.text.as_str()).collect::<String>();
    assert!(text.contains("::std::sync::Arc::clone(&value.get().0)"), "{text}");
    assert!(text.contains("super::llm::Store(out)"), "{text}");
    assert!(!text.contains("super::llm::Store(::std::sync::Arc::new(out))"), "{text}");
}

#[test]
fn nullable_defaults_distinguish_an_explicit_null_from_an_omitted_key() {
    let decls = crate::decl::tests::read(
        r#"
        #[crate::host(record(module = "options", typed_dict, extract))]
        pub struct Options {
            #[host(default = 15.0)] pub away_timeout: Option<f64>,
        }
    "#,
    )
    .expect("nullable defaults resolve");
    let generated = crate::records::conversions(&decls).expect("conversions emit").text;
    assert!(generated.contains("Some(v) if !crate::py_values::is_not_given(&v)?"), "{generated}");
    assert!(generated.contains("_ => Some(15.0)"), "{generated}");
}

#[test]
fn record_options_keep_omitted_null_and_value_distinct() {
    let decls = crate::decl::tests::read(
        r#"
        use crate::utils::maybe::Maybe;
        #[crate::host(record(module = "session", extract))]
        pub struct Details { pub text: String }
        #[crate::host(record(module = "session", extract))]
        pub struct Options {
            pub persona: Maybe<String>,
            pub reset: Maybe<Option<String>>,
            pub details: Maybe<Details>,
        }
    "#,
    )
    .expect("omittable record options resolve");
    let models = crate::records::python(&decls).expect("models");
    let model = &models.iter().find(|o| o.path == "session.py").unwrap().text;
    assert!(model.contains("from ..types import NOT_GIVEN\n"), "{model}");
    assert!(model.contains("from ..types import NotGivenOr\n"), "{model}");
    assert!(model.contains("persona: NotGivenOr[str] = NOT_GIVEN"), "{model}");
    assert!(model.contains("reset: NotGivenOr[str | None] = NOT_GIVEN"), "{model}");
    assert!(model.contains("details: NotGivenOr[Details] = NOT_GIVEN"), "{model}");
    let adapter = crate::records::conversions(&decls).expect("conversions").text;
    syn::parse_file(&adapter).expect("generated conversion syntax");
    assert!(adapter.contains(
        "if crate::py_values::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (v.extract::<::std::string::String>()).map(::lk_agents_abi::Maybe::Given) }"
    ), "{adapter}");
    assert!(adapter.contains(
        "else { (if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) }).map(::lk_agents_abi::Maybe::Given) }"
    ), "{adapter}");
    assert!(adapter.contains(
        "(crate::generated::records::session::details_from_py(v)).map(::lk_agents_abi::Maybe::Given)"
    ), "{adapter}");
}

#[test]
fn omittable_record_options_refuse_competing_defaults_and_output() {
    for attr in ["default = None", "default = \"persona\"", "default_factory = \"types.default\""] {
        let result = crate::decl::tests::read(&format!(
            r#"
            use crate::utils::maybe::Maybe;
            #[crate::host(record(module = "session", extract))]
            pub struct Options {{ #[host({attr})] pub persona: Maybe<String> }}
        "#
        ));
        let error = result.expect_err("the omission sentinel is the field's only default");
        assert!(error.contains("Options.persona"), "{error}");
        assert!(error.contains("default"), "{error}");
    }
    let result = crate::decl::tests::read(
        r#"
        use crate::utils::maybe::Maybe;
        #[crate::host(record(module = "session", emit))]
        pub struct Options { pub persona: Maybe<String> }
    "#,
    );
    assert!(result.expect_err("omission is input-only").contains("omission is an input state"));
}

#[test]
fn merged_record_annotations_keep_both_modules_typing_imports() {
    let decls = crate::decl::tests::read(
        r#"
        use crate::utils::maybe::Maybe;
        #[crate::host(record(module = "session", extract))]
        pub struct Options { pub persona: Maybe<String> }
    "#,
    )
    .unwrap();
    let models = crate::records::python(&decls).unwrap();
    let model = &models.iter().find(|o| o.path == "session.py").unwrap().text;
    let legacy = "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from ..existing import Existing\n\nclass Legacy:\n    pass\n";
    let merged = crate::records::merge(legacy, model).unwrap();
    assert!(merged.contains("    from ..types import NotGivenOr\n"), "{merged}");
    assert!(merged.contains("    from ..existing import Existing\n"), "{merged}");
    assert_eq!(merged.matches("if TYPE_CHECKING:").count(), 1, "{merged}");
}

// Catches: the generated bindings warning (non-CamelCase `_core_X` types, non-snake helpers)
// or the rename reaching the Python name a string carries.
#[test]
fn private_core_classes_take_rust_names_and_keep_their_python_names() {
    let text = "#[pyclass(name = \"_core_Policy\")]\npub struct _core_Policy(::std::sync::Arc<_core_PolicyImpl>);\nfn _extract_supervisor__core_pid() {}\npub fn _core__retry_to_py() {}\n";
    assert_eq!(
        crate::adapter::rust_names(text),
        "#[pyclass(name = \"_core_Policy\")]\npub struct CorePolicy(::std::sync::Arc<CorePolicyImpl>);\nfn _extract_supervisor_core_pid() {}\npub fn _core_retry_to_py() {}\n"
    );
}

#[test]
fn constants_and_export_signatures_share_resolved_record_and_alias_types() {
    let declarations = crate::decl::tests::read(
        r#"
        #[host(file(module = "defaults"))] const _: () = ();
        #[host(record(module = "defaults", emit))]
        pub struct Settings { #[host(rename = "type")] pub kind: &'static str }
        #[host(constant)] pub const DEFAULT: Settings = Settings { kind: "fixed" };
        pub type Names = Vec<&'static str>;
        #[host(export)] pub fn names() -> Names { unreachable!() }
    "#,
    )
    .unwrap();
    let models = crate::python::emit(&declarations, &[]).unwrap();
    assert!(models[0].text.contains("DEFAULT: Final = Settings(type=\"fixed\")"));
    assert!(
        matches!(&declarations.exports[0].ty, crate::decl::Ty::List(item) if matches!(**item, crate::decl::Ty::Str))
    );
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    assert!(stubs.iter().any(|stub| stub.out.text.contains("def names() -> list[str]: ...")));
}

#[test]
fn marker_classes_keep_identity_semantics_and_unions_import_generated_definitions() {
    let declarations = crate::decl::tests::read(
        r#"
        #[crate::host(record(module = "marker", path = "livekit.agents.marker", marker, emit))]
        pub struct End {}
        #[crate::host(record(module = "marker", extract, transparent))]
        pub enum Item { Text(String), #[host(host_type = "livekit.agents.marker.End")] End }
    "#,
    )
    .unwrap();
    let outputs = crate::records::python(&declarations).unwrap();
    let model = &outputs.iter().find(|output| output.path == "marker.py").unwrap().text;
    assert!(model.contains("class End:\n    pass"), "{model}");
    assert!(!model.contains("@dataclass"), "{model}");
    assert!(!model.contains("from livekit.agents.marker import"), "{model}");
    assert!(!model.contains("from .marker import"), "{model}");
    for invalid in [
        "#[crate::host(record(module = \"marker\", marker, emit))] pub struct Invalid { pub field: String }",
        "#[crate::host(record(module = \"marker\", marker, dataclass, emit))] pub struct Invalid {}",
    ] {
        assert!(
            crate::decl::tests::read(invalid).unwrap_err().contains("empty identity-only record")
        );
    }
}

#[test]
fn native_record_defaults_and_exported_constants_share_resolved_values() {
    let declarations = crate::decl::tests::read(r#"
        #[crate::host(file(module = "options"))] const _: () = ();
        const PHASE: Option<f64> = Some(7.5);
        #[crate::host(record(module = "options", dataclass, frozen, defaults_from = DEFAULT, extract, emit))]
        pub struct Options { pub retries: u32, pub timeout: Option<f64>, pub total: Option<f64> }
        #[crate::host(constant)]
        pub const DEFAULT: Options = Options { retries: 2, timeout: PHASE, total: None };
    "#).unwrap();
    let models = crate::records::python(&declarations).unwrap();
    let model = &models.iter().find(|model| model.path == "options.py").unwrap().text;
    assert!(model.contains("@dataclass(frozen=True)"));
    assert!(model.contains("retries: int = 2"));
    assert!(model.contains("timeout: float | None = 7.5"));
    assert!(model.contains("total: float | None = None"));
    let constants = crate::python::emit(&declarations, &[]).unwrap();
    assert!(
        constants
            .iter()
            .any(|module| module.text.contains("Options(retries=2, timeout=7.5, total=None)"))
    );
    for source in [
        "#[crate::host(record(module = \"options\", dataclass, defaults_from = DEFAULT, emit))] pub struct Options { #[host(default = 3)] pub retries: u32 } const DEFAULT: Options = Options { retries: 2 };",
        "#[crate::host(record(module = \"options\", defaults_from = DEFAULT, emit))] pub struct Options { pub retries: u32 } const DEFAULT: Options = make_options();",
        "#[crate::host(record(module = \"options\", defaults_from = DEFAULT, emit))] pub struct Options { pub retries: u32 } const DEFAULT: Options = OTHER; const OTHER: Options = DEFAULT;",
    ] {
        assert!(crate::decl::tests::read(source).is_err());
    }
}

#[test]
fn checked_representations_and_model_methods_are_resolved_before_emission() {
    let declarations = crate::decl::tests::read(r#"
        #[crate::host(codec(extract, emit, host = "int", via = u8))] pub struct Flag(u8);
        #[crate::host(record(module = "values", dataclass, frozen, extract, emit))]
        #[python(method(name = "adjust", args = "amount: int = 1, *, names: tuple[str, str] = ('a,b', 'c'), thread: .storage_class.ThreadInfo | None = None", ret = "int", hook = "adjust"))]
        pub struct Options { pub flag: Flag }
    "#).unwrap();
    let shell = crate::records::conversions(&declarations).unwrap().text;
    assert!(
        shell.contains("from_representation::<::lk_agents_abi::llm::class::Flag, u8>"),
        "{shell}"
    );
    assert!(shell.contains("scalar: u8"), "{shell}");
    let models = crate::records::python(&declarations).unwrap();
    let model = &models.iter().find(|model| model.path == "values.py").unwrap().text;
    assert!(model.contains("from .storage_class import ThreadInfo"), "{model}");
    assert!(model.contains("thread: ThreadInfo | None = None"), "{model}");
    assert!(
        model.contains("return _bindings.adjust(self, amount, names=names, thread=thread)"),
        "{model}"
    );
    let invalid = crate::decl::tests::read(
        r#"
        #[crate::host(codec(extract, host = "object", via = Self))] pub struct Recursive;
        #[crate::host(record(module = "values", extract))] pub struct Options { pub value: Recursive }
    "#,
    );
    assert!(invalid.is_err());
}

#[test]
fn explicit_model_schemas_preserve_layout_and_postponed_callback_types() {
    let declarations = crate::decl::tests::read(r#"
        #[crate::host(record(module = "values", path = "package.values", dataclass, frozen, kw_only, slots, deferred))]
        pub struct Event {
            #[host(default = false, kw_only)] pub active: bool,
            #[host(hide_repr, annotation = "package.callbacks.Callback")] pub callback: crate::HostRef,
        }
        #[crate::host(record(module = "values", typed_dict, constant(type = "named")))]
        pub struct Choice { pub name: String, #[host(default = None)] pub note: Option<String> }
        #[crate::host(record(module = "values", enum, enum_base = "Enum"))]
        pub enum Status { Ready, Done }
    "#).unwrap();
    let models = crate::records::python(&declarations).unwrap();
    let text = &models.iter().find(|model| model.path == "values.py").unwrap().text;
    assert!(text.contains("@dataclass(frozen=True, kw_only=True, slots=True)"), "{text}");
    assert!(text.contains("field(default=False, kw_only=True)"), "{text}");
    assert!(text.contains("field(repr=False)"), "{text}");
    assert!(
        text.contains("if TYPE_CHECKING:\n    from package.callbacks import Callback"),
        "{text}"
    );
    assert!(!text.contains("get_annotations(Event"), "{text}");
    assert!(text.contains("class Status(Enum)"), "{text}");
    assert!(text.contains("type: Literal[\"named\"]"), "{text}");
    assert!(text.contains("note: NotRequired[str]"), "{text}");
    let converters = crate::records::conversions(&declarations).unwrap();
    assert!(!converters.text.contains("event_from_py"));
    for invalid in [
        "#[crate::host(record(module = \"values\", dataclass))] pub struct Bad { #[host(default = false)] pub first: bool, pub second: String }",
        "#[crate::host(record(module = \"values\", pydantic, deferred))] pub struct Bad {}",
        "#[crate::host(record(module = \"values\", enum, enum_base = \"object\"))] pub enum Bad { Value }",
    ] {
        assert!(crate::decl::tests::read(invalid).is_err());
    }
}

#[test]
fn flag_members_are_discovered_from_the_native_public_constants() {
    let declarations = crate::decl::tests::read(r#"
        #[crate::host(codec(extract, emit, host = "package.flags.Options", via = u8, attribute = "value", flag(module = "flags")))]
        pub struct Options(u8);
        impl Options {
            pub const NONE: Self = Self(0);
            pub const FIRST: Self = Self(1);
            const PRIVATE: Self = Self(2);
            pub const OTHER_TYPE: u8 = 8;
        }
    "#).unwrap();
    let models = crate::native_enums::flags(&declarations);
    assert_eq!(models.len(), 1);
    assert!(models[0].text.contains("class Options(Flag)"));
    assert!(models[0].text.contains("    NONE = 0\n    FIRST = 1\n"));
    assert!(!models[0].text.contains("PRIVATE"));
    assert!(!models[0].text.contains("OTHER_TYPE"));
    let invalid = crate::decl::tests::read(
        r#"
        #[crate::host(codec(extract, host = "package.flags.Options", via = u8, attribute = "value", flag(module = "flags")))]
        pub struct Options(u8);
        impl Options { pub const FIRST: Self = Self(1 + 2); }
    "#,
    );
    assert!(invalid.is_err());
}
