//! The generator's own tests: one declaration in, one generated string out.
//!
//! Every test here names the bug it catches, because a codegen test that only asserts "the output
//! is what the code produces" is a change detector -- it fails on every edit and tells the reader
//! nothing about which edits were wrong. What is asserted is the property that would have shipped
//! broken: a key that is written when it should be absent, a method name spelled twice, a Rust
//! keyword emitted bare.
//!
//! `parse::file` + one emitter rather than the whole `run()`: neither touches the filesystem.

use crate::{ir, parse, python, stub};

/// Every generated **Python** module's text, concatenated.
///
/// [`generate`]'s twin for the host side. Added with the `total=` work, and worth saying why it
/// did not exist before: the Python emitter was covered only by `generated-check`, which
/// regenerates the committed files and diffs them. That catches a change to a declaration that
/// already exists and cannot catch anything about a shape nothing declares yet -- which is exactly
/// the state `Kind::TypedDict` was in.
fn generate_python(src: &str) -> String {
    let decls = parse::file("test.rs", src).expect("the declaration parses");
    python::emit_with(&decls, &[], &[])
        .expect("the declaration generates")
        .into_iter()
        .map(|o| o.text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The error a bad Python declaration produces.
fn refuse_python(src: &str) -> String {
    let decls = match parse::file("test.rs", src) {
        Ok(d) => d,
        Err(e) => return e,
    };
    python::emit_with(&decls, &[], &[])
        .err()
        .unwrap_or_else(|| panic!("expected a refusal, got a generated file"))
}

/// The `_core` classes' stub, for the `class` kind.
fn generate_stub(src: &str) -> String {
    let decls = parse::file("test.rs", src).expect("the declaration parses");
    stub::emit(&decls, &ir::Exposed::default())
        .expect("the stub generates")
        .expect("there is a class")
        .out
        .text
}

// --- typed dicts --------------------------------------------------------------------------------

// Catches: `total` dropped or defaulted the wrong way. `total=True` (Python's default, written as
// nothing) makes every key required; `total=False` makes every key optional -- either mistake is
// a user-visible API change that fails only at the call that omits (or relies on) a key.
#[test]
fn a_typed_dict_says_total_false_only_when_declared() {
    for (total, want) in [
        ("total = \"False\", ", "class Opts(TypedDict, total=False):"),
        ("", "class Opts(TypedDict):"),
    ] {
        let out = generate_python(&format!(
            "#[derive(Host)]\n#[python(typed_dict, {total}module = \"m\", name = \"Opts\")]\npub struct Opts {{ pub enabled: bool }}"
        ));
        assert!(out.contains(want), "{out}");
        assert_eq!(out.contains("total="), !total.is_empty(), "{out}");
    }
}

// Catches: a declaration key that is quietly ignored or misread instead of refused by name. Each
// would generate a file that looks deliberate and matches nothing: `total` on a kind with no
// `total=`; a Rust `false` for a Python bool (`walk` throws the parse error away, dropping every
// later key too); a misspelt `typed_dict_from` (the other `TypedDict` class); `tag_after` naming
// no emitted field (the tag silently leads); an `order` that is not a permutation of the fields.
#[test]
fn a_declaration_key_that_cannot_be_honoured_is_refused_by_name() {
    let opts = |attr: &str| {
        format!(
            "#[derive(Host)]\n#[python({attr}, module = \"m\", name = \"Opts\")]\npub struct Opts {{ pub a: bool, pub b: bool }}"
        )
    };
    let cases = [
        (opts("dataclass, total = \"False\""), &["dataclass", "total"][..]),
        (opts("typed_dict, total = \"false\""), &["not a Python bool", "\"false\""]),
        (opts("typed_dict, typed_dict_from = \"typing_extension\""), &["typing_extension"]),
        (
            opts("pydantic, tag = \"type\", tag_value = \"h\", tag_after = \"nope\""),
            &["tag_after", "nope"],
        ),
        (opts("typed_dict, order = \"a\""), &["leaves out `b`"]),
        (opts("typed_dict, order = \"a, b, a\""), &["twice"]),
        (opts("typed_dict, order = \"a, b, c\""), &["`c`"]),
    ];
    for (src, words) in cases {
        let err = refuse_python(&src);
        for w in words {
            assert!(err.contains(w), "{src}\n-> {err}");
        }
    }
}

// Catches: an `import = ".."` landing in the wrong block. `typing_extensions:Required` in the
// relative block put a second `from typing_extensions import ..` under the generator's own;
// `typing:Mapping` must join the stdlib `typing` line; any other stdlib module
// (`collections.abc:Callable`) goes above third-party. Each import appears exactly once.
#[test]
fn an_import_joins_its_block_once() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(typed_dict, total = "False", module = "m", name = "Opts")]
        pub struct Opts {
            #[python(ty = "Required[str]", import = "typing_extensions:Required")]
            pub name: Option<String>,
            #[python(ty = "Mapping[str, Any]", import = "typing:Mapping")]
            pub items: Option<String>,
            #[python(ty = "Callable[[str], str]", import = "collections.abc:Callable")]
            pub render: Option<String>,
        }
        "#,
    );
    assert!(out.contains("from typing_extensions import Required, TypedDict"), "{out}");
    assert_eq!(out.matches("from typing_extensions import").count(), 1, "{out}");
    assert_eq!(out.matches("from typing import").count(), 1, "{out}");
    assert!(out.contains("Mapping"), "{out}");
    let abc = out.find("from collections.abc import Callable").expect("emitted");
    let te = out.find("from typing_extensions import").expect("emitted");
    assert!(abc < te, "stdlib must sort above third-party:\n{out}");
}

// Catches: `#[python(skip)]` being confused with `#[python(exclude)]`, which is what happened when
// this was first written. `exclude` is pydantic's `Field(..., exclude=True)` -- the field is still
// a field and users still see it. `skip` means the host class has no such member at all.
// `LlmChunk.flush` is the live case: an in-band TTS sentinel the reference's `ChatChunk` has no
// name for, so emitting it would publish a field no reference user has ever seen.
#[test]
fn a_python_skipped_field_is_absent_from_the_class_rather_than_excluded() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(pydantic, module = "m", name = "Chunk")]
        pub struct Chunk {
            pub id: String,
            #[python(skip)]
            pub flush: bool,
        }
        "#,
    );
    assert!(out.contains("id: str"), "{out}");
    assert!(!out.contains("flush"), "{out}");
    assert!(!out.contains("exclude=True"), "{out}");
}

// Catches: a multi-line `doc = ".."` on a member emitted with every continuation line at column 0.
// Inside a method body that reads as if the docstring had ended, and Python keeps it as the
// docstring anyway -- so it renders wrong and nothing fails. `ChatChunk.has_response` is the first
// member whose docstring has a second paragraph.
#[test]
fn a_multi_line_member_docstring_is_indented_to_the_method_body() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(
            pydantic,
            module = "m",
            name = "Chunk",
            method(name = "ok", ret = "bool", doc = "First line.\n\nSecond paragraph.")
        )]
        pub struct Chunk {
            pub id: String,
        }
        "#,
    );
    assert!(out.contains("        \"\"\"First line."), "{out}");
    assert!(out.contains("\n        Second paragraph.\n"), "{out}");
    // The closing quotes get their own line for a multi-line docstring, like every other one.
    assert!(out.contains("\n        \"\"\"\n"), "{out}");
}

// Catches: a `default_factory` that is an *expression* being split on its first `.` and refused as
// naming a module. `lambda: utils.shortuuid("item_")` is the reference's own spelling for four
// `id` fields in `chat_context.py`; the old code read `lambda: utils` as a module name and
// produced an error about an import it could not add, which points at the wrong problem entirely
// -- nothing needs importing, the expression already names what it uses.
#[test]
fn a_default_factory_that_is_an_expression_is_not_read_as_a_module_path() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(pydantic, module = "m", name = "Msg")]
        pub struct Msg {
            #[python(default_factory = "lambda: utils.shortuuid(\"item_\")")]
            pub id: String,
        }
        "#,
    );
    assert!(
        out.contains(r#"id: str = Field(default_factory=lambda: utils.shortuuid("item_"))"#),
        "{out}"
    );
    // And the dotted-name case still resolves its import, which is what the split existed for.
    let dotted = generate_python(
        r#"
        #[derive(Host)]
        #[python(pydantic, module = "m", name = "Msg")]
        pub struct Msg {
            #[python(default_factory = "time.time")]
            pub created_at: f64,
        }
        "#,
    );
    assert!(dotted.contains("import time"), "{dotted}");
}

// Catches: the discriminant always leading the class. That is right for every `voice/events.py`
// model and wrong for `chat_context.py`'s, which write `id` then `type` -- and pydantic field
// order is `model_dump()` key order and `repr` order, so it is user-visible rather than cosmetic.
//
// The other way to place it is a `()` marker field, which `ir.rs`'s `tag` doc records as the
// rejected first design: it makes every Rust construction site write `ty: ()` for a marker only
// Python reads. Trying it broke `lk-agents-remote`'s codec on the first build.
#[test]
fn tag_after_puts_the_discriminant_behind_a_named_field() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(
            pydantic,
            module = "m",
            name = "Item",
            tag = "type",
            tag_value = "handoff",
            tag_after = "id"
        )]
        pub struct Item {
            pub id: String,
            pub other: String,
        }
        "#,
    );
    let id_at = out.find("id: str").expect("emitted");
    let tag_at = out.find(r#"type: Literal["handoff"]"#).expect("emitted");
    let other_at = out.find("other: str").expect("emitted");
    assert!(id_at < tag_at && tag_at < other_at, "tag must sit between them:\n{out}");
}

// --- the `_core` class ---------------------------------------------------------------------------

/// A three-field record as a class, with and without the constructor. The `Option` field carries
/// `#[binding(default)]`, the `u32` keeps its width, the `String` clones.
fn class_decl(flags: &str) -> String {
    format!(
        r#"
        #[derive(Host)]
        #[binding(class, {flags}from = core::Thing)]
        pub struct Thing {{
            /// The name.
            pub name: String,
            pub count: u32,
            #[binding(default)]
            pub note: Option<String>,
        }}
        "#
    )
}

// Catches: a generated class arriving without its stub, which would *remove* the type information
// the hand-written pyclass had in `_core.pyi`. One `@property` per getter with the Python type
// the pydantic modules would give the field, and an `__init__` only under `new`, keyword-only,
// with a defaulted field spelled `= None`.
#[test]
fn the_stub_declares_a_property_per_getter_and_an_init_under_new() {
    let out = generate_stub(&class_decl("new, "));
    assert!(out.contains("class Thing:\n"), "{out}");
    assert!(out.contains("    def __init__(self, *, name: str, count: int, note: str | None = None) -> None: ...\n"), "{out}");
    assert!(out.contains("    @property\n    def name(self) -> str: ...\n    @property\n    def count(self) -> int: ...\n    @property\n    def note(self) -> str | None: ...\n"), "{out}");
    assert!(out.starts_with("# GENERATED FILE -- DO NOT EDIT."), "{out}");
    let out = generate_stub(&class_decl(""));
    assert!(!out.contains("__init__"), "{out}");
}

// Catches: a converted field's stub type. The dict table has no answer for a `with` field, so the
// stub takes the declaration's `#[python(ty = ..)]` and imports what it names, or says `Any`.
#[test]
fn the_stub_types_a_converted_field_from_its_python_annotation() {
    let out = generate_stub(
        r#"
        #[derive(Host)]
        #[binding(class, from = core::Dob)]
        pub struct Dob {
            #[python(ty = "datetime.date")]
            #[binding(with = "crate::dates::date_to_py")]
            pub date_of_birth: Date,
            #[binding(with = "crate::json::value_to_py")]
            pub extra: Value,
        }
        "#,
    );
    assert!(out.contains("import datetime\nfrom typing import Any\n"), "{out}");
    assert!(out.contains("def date_of_birth(self) -> datetime.date: ..."), "{out}");
    assert!(out.contains("def extra(self) -> Any: ..."), "{out}");
}

// --- exposed core items: #[host(export)] / #[host(object)] ----------------------------------------
//
// `parse::exposed` + the stub over one source that may also hold the `#[derive(Host)]`
// declarations a signature names.

fn exports_stub(src: &str) -> String {
    let decls = parse::file("test.rs", src).expect("the declarations parse");
    let x = parse::exposed("test.rs", src).expect("the items parse");
    stub::emit(&decls, &x).expect("the stub generates").expect("there are items").out.text
}

// Catches: a raw-identifier parameter reaching the stub as `r#ref: str`, which is not Python:
// `ast.parse` of `_core_classes.pyi` failed on it (2026-09-24). pyo3 already calls it `ref`.
#[test]
fn a_keyword_parameter_is_its_python_name_in_the_stub() {
    let stub = exports_stub(
        "#[host(export(kw_only))]\npub fn resolve_ref(root: &str, r#ref: &str) -> bool { true }",
    );
    assert!(stub.contains("def resolve_ref(*, root: str, ref: str) -> bool: ..."), "{stub}");
    assert!(!stub.contains("r#"), "{stub}");
}

// Catches: a class stub that drops the hand-written methods (then `_core.pyi` has to keep a second
// copy of the class), or an extra stub line whose name is never imported.
#[test]
fn an_object_stub_is_one_class_with_its_extra_lines_and_their_imports() {
    let stub = exports_stub(
        "/// The gate.\n#[host(object(name = \"Gate\", stub = \"def extra(self) -> Result: ...\", stub_import = \"from .x import Result\"))]\n\
         impl G {\n#[host(new(kw_only))] pub fn new(a: u8) -> Self { todo!() }\n#[host(method)] pub fn s(b: &[u8]) -> bool { true }\n}",
    );
    assert!(stub.contains("from .x import Result\n"), "{stub}");
    assert!(stub.contains("class Gate:\n    \"\"\"The gate.\"\"\"\n\n    def __init__(self, *, a: int) -> None: ...\n    @staticmethod\n    def s(b: bytes) -> bool: ...\n    def extra(self) -> Result: ...\n"), "{stub}");
}

// --- the Python shapes the option records need (2026-09-23) --------------------------------------

// Catches: a factory spelled for the wrong kind. On a dataclass pydantic's `Field` is not a
// factory at all -- the `FieldInfo` becomes the shared default, so `SpeechEvent(type=..)
// .alternatives` would be a `FieldInfo`, not a fresh `[]`; a pydantic model keeps `Field`.
#[test]
fn a_default_factory_is_spelled_for_its_kind() {
    for (kind, want, absent) in [
        ("dataclass", "alternatives: list[str] = field(default_factory=list)", "pydantic"),
        ("pydantic", "alternatives: list[str] = Field(default_factory=list)", "from dataclasses"),
    ] {
        let out = generate_python(&format!(
            "#[derive(Host)]\n#[python({kind}, module = \"m\", name = \"Ev\")]\npub struct Ev {{\n#[python(default_factory = \"list\")]\npub alternatives: Vec<String>,\n}}"
        ));
        assert!(out.contains(want), "{out}");
        assert!(!out.contains(absent), "{out}");
    }
}

// Catches: a `typing.TypedDict` in the reference generated as a `typing_extensions` one. On
// Python < 3.13 they are different classes with different metaclasses, and `typing.is_typeddict`
// answers `False` for the second -- `voice/amd/detector.py` and `inference/stt.py` use `typing`.
#[test]
fn typed_dict_from_typing_imports_the_stdlib_class() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(typed_dict, total = "False", typed_dict_from = "typing", module = "m", name = "Opts")]
        pub struct Opts {
            #[python(ty = "float")]
            pub timeout: Option<f64>,
        }
        "#,
    );
    assert!(out.contains("from typing import TypedDict\n"), "{out}");
    assert!(!out.contains("typing_extensions"), "{out}");
    assert!(out.contains("class Opts(TypedDict, total=False):\n    timeout: float\n"), "{out}");
}

// Catches: the host's field order following the Rust one when the two must differ. The reference
// declares `resume_false_interruption` before `false_interruption_timeout`; the binding's emit
// side writes them the other way round, in Rust order, and a `TypedDict`'s `__annotations__` order
// is public.
#[test]
fn order_sets_the_host_field_order_without_touching_the_rust_one() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(typed_dict, module = "m", name = "Opts", order = "b, a")]
        pub struct Opts {
            pub a: bool,
            /// B's doc.
            pub b: f64,
        }
        "#,
    );
    assert!(out.contains("    b: float\n    \"\"\"B's doc.\"\"\"\n    a: bool\n"), "{out}");
}

// Catches: a host field that can only trail. `TextOutputOptions.next_in_chain` is a host object
// between two declared fields, and a dataclass's positional `__init__` makes the slot public.
#[test]
fn order_places_a_host_field_between_declared_ones() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(
            dataclass,
            module = "m",
            name = "Opts",
            host_field(name = "next", ty = "Out | None", default = "None"),
            order = "a, next, b"
        )]
        pub struct Opts {
            #[python(default = "1.0")]
            pub a: f64,
            #[python(default = "False")]
            pub b: bool,
        }
        "#,
    );
    assert!(
        out.contains("    a: float = 1.0\n    next: Out | None = None\n    b: bool = False\n"),
        "{out}"
    );
}

// Catches: a `__post_init__` hook calling `_bindings` in a module that never imported it -- a
// `NameError` on the first construction, not at import. `SpeechData` was the first dataclass
// whose only hook was `post_init`.
#[test]
fn a_post_init_hook_imports_the_bindings_module() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(dataclass, module = "m", name = "Data", post_init = "data_post_init")]
        pub struct Data { pub text: String }
        "#,
    );
    assert!(out.contains("from .. import _bindings\n"), "{out}");
    assert!(
        out.contains(
            "    def __post_init__(self) -> None:\n        _bindings.data_post_init(self)\n"
        ),
        "{out}"
    );
}

// Catches: a `type_import` of an absolute module emitted as a runtime import. `openai` is not a
// dependency of the package, so `from openai.types.chat import ..` at module scope would break
// `import livekit.agents` for everyone without it.
#[test]
fn an_absolute_type_import_stays_under_type_checking() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(typed_dict, total = "False", module = "m", name = "Opts")]
        pub struct Opts {
            #[python(ty = "Metadata | None", type_import = "openai.types.shared_params:Metadata")]
            pub metadata: Option<String>,
        }
        "#,
    );
    assert!(
        out.contains("if TYPE_CHECKING:\n    from openai.types.shared_params import Metadata\n"),
        "{out}"
    );
    assert_eq!(out.matches("from openai").count(), 1, "{out}");
}

// Catches: a plain union alias pulling in `Annotated` and pydantic's `Field`. Only a
// discriminated union uses them; `LLMModels = OpenAIModels | GoogleModels | ..` lives in a module
// with no model at all.
#[test]
fn a_plain_union_imports_neither_annotated_nor_pydantic() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(literal, module = "m", name = "A")]
        pub enum A { #[host(rename = "a")] A }

        #[derive(Host)]
        #[python(union, module = "m", name = "U")]
        pub enum U {
            A(A),
            Auto(#[python(ty = "Literal[\"auto\"]")] String),
        }
        "#,
    );
    assert!(out.contains("U = A | Literal[\"auto\"]\n"), "{out}");
    assert!(!out.contains("Annotated") && !out.contains("pydantic"), "{out}");
}

// Catches: a one-line docstring that ends in a quote. `"""e.g. "finance""""` is a syntax error,
// and it is what a provider option noted with a quoted example generates.
#[test]
fn a_docstring_ending_in_a_quote_still_closes() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(typed_dict, total = "False", module = "m", name = "Opts")]
        pub struct Opts {
            /// e.g. "finance"
            #[python(ty = "str")]
            pub domain: Option<String>,
        }
        "#,
    );
    assert!(out.contains("    \"\"\"e.g. \"finance\\\"\"\"\"\n"), "{out}");
}

// --- host-stream dispatch -------------------------------------------------------------------------

const ROUTED: &str = "#[derive(Host)]\n#[python(pydantic, module = \"events\", name = \"TimeoutEvent\", tag = \"type\", tag_value = \"user_timeout\")]\n\
    #[binding(emit, tag = \"type\", tag_value = \"timeout\", from = core::Timeout)]\n\
    pub struct Timeout { pub at: f64 }\n\
    #[derive(Host)]\n#[binding(emit, tag = \"type\", tag_value = \"done\", from = core::Done)]\n\
    pub struct Done { pub id: String }\n\
    #[derive(Host)]\n#[binding(emit, from = core::Stream)]\n#[python(dispatch = \"dispatch\")]\n\
    pub enum Stream {\n    #[python(event, mirror = \"_timeout\")]\n    Timeout(Timeout),\n    #[python(ignored)]\n    Done(Done),\n}";

// Catches: a routing table keyed by the Python discriminant instead of the wire kind the binding
// sends (they differ -- `transcription_timeout` is emitted as `user_transcription_timeout`), an
// event member routed to a class the host cannot import, or a routed member with no module.
#[test]
fn a_routed_union_generates_its_dispatch_table_keyed_by_the_wire_kind() {
    let out = generate_python(ROUTED);
    assert!(out.contains("from .events import TimeoutEvent\n"), "{out}");
    assert!(out.contains("PUBLIC_EVENTS: dict[str, tuple[type[Any], str]] = {\n    \"timeout\": (TimeoutEvent, \"user_timeout\"),\n}"), "{out}");
    assert!(
        out.contains(
            "IGNORED_EVENTS: frozenset[str] = frozenset(\n    {\n        \"done\",\n    }\n)"
        ),
        "{out}"
    );
    assert!(
        out.contains("MIRRORED_STATES: dict[str, str] = {\n    \"timeout\": \"_timeout\",\n}"),
        "{out}"
    );
    let e = refuse_python(&ROUTED.replace("#[python(dispatch = \"dispatch\")]\n", ""));
    assert!(e.contains("names no #[python(dispatch"), "{e}");
    let e = refuse_python(&ROUTED.replace("#[python(ignored)]", "#[python(event)]"));
    assert!(e.contains("needs a pydantic Python target"), "{e}");
    let e =
        refuse_python(&ROUTED.replace("#[python(ignored)]", "#[python(ignored, mirror = \"_x\")]"));
    assert!(e.contains("mirrors a state but is not an event"), "{e}");
}

// Catches: the stub reading a method's kind from anything but its signature (the expansion's rule),
// so a receiver-less method missing `@staticmethod`, an `async fn` or a `Reply<T>` method not
// awaitable, or the awaitable typed as the `Reply` rather than what it resolves to.
#[test]
fn a_methods_stub_kind_is_its_signatures() {
    let stub = exports_stub(
        "#[host(object(cell = \"arc\"))]\nimpl C {\n\
         #[host(method)] pub fn make(n: u8) -> u8 { n }\n\
         #[host(method)] pub async fn wait(&self) -> u8 { 0 }\n\
         #[host(method)] pub fn ask(&self) -> Result<Reply<bool>, E> { todo!() }\n}",
    );
    assert!(stub.contains("    @staticmethod\n    def make(n: int) -> int: ...\n"), "{stub}");
    assert!(stub.contains("    async def wait(self) -> int: ...\n"), "{stub}");
    assert!(stub.contains("    async def ask(self) -> bool: ...\n"), "{stub}");
}

// Catches: the stub and the expansion disagreeing on which `Option` parameters default -- the
// stub must show `= None` exactly where pyo3's signature has it (the shared rule), and never on a
// `#[host(required)]` one.
#[test]
fn the_stub_defaults_an_option_where_the_expansion_does() {
    let stub = exports_stub(
        "#[host(export)]\npub fn f(a: Option<u8>, b: u8, c: Option<u8>, #[host(required)] d: Option<u8>, e: Option<u8>) {}",
    );
    assert!(stub.contains("def f(a: int | None, b: int, c: int | None, d: int | None, e: int | None = None) -> None: ..."), "{stub}");
}

// --- record-level defaults -----------------------------------------------------------------------

// Catches: a record-level default reaching a field that says its own, missing one that does not,
// or `zero_defaults` defaulting a string (the reference's `provider: str` is required). The
// record's `#[binding(default)]` is also the stub's "optional to the caller".
#[test]
fn record_level_defaults_reach_each_field_that_says_nothing() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(pydantic, zero_defaults, module = "m", name = "Usage")]
        pub struct Usage {
            pub provider: String,
            pub tokens: u64,
            pub duration: f64,
            pub note: Option<String>,
            #[python(default = "1")]
            pub requests: u32,
            #[python(required)]
            pub timestamp: f64,
        }
        "#,
    );
    assert!(out.contains("    provider: str\n"), "{out}");
    assert!(
        out.contains("    tokens: int = 0\n") && out.contains("    duration: float = 0.0\n"),
        "{out}"
    );
    assert!(
        out.contains("    note: str | None = None\n") && out.contains("    requests: int = 1\n"),
        "{out}"
    );
    assert!(out.contains("    timestamp: float\n"), "{out}");
    let out = generate_stub(
        r#"
        #[derive(Host)]
        #[binding(class, new, default, from = core::Thing)]
        pub struct Thing { pub name: String, pub count: u32 }
        "#,
    );
    assert!(
        out.contains("def __init__(self, *, name: str = \"\", count: int = 0) -> None"),
        "{out}"
    );
}

// Catches: the file's `module`/`path` overriding a record's own, reaching a record with no Python
// target, or a second default (or an unknown key) accepted in silence.
#[test]
fn a_file_default_fills_module_and_path_where_the_record_says_nothing() {
    let src = r#"
        #[crate::host(file(module = "events", path = "livekit.agents.voice.events"))]
        const _: () = ();
        #[derive(Host)]
        #[python(pydantic, name = "A")]
        pub struct A { pub x: u32 }
        #[derive(Host)]
        #[python(pydantic, module = "other", path = "livekit.agents.other", name = "B")]
        pub struct B { pub x: u32 }
        #[derive(Host)]
        #[binding(emit)]
        pub struct C { pub x: u32 }
    "#;
    let decls = parse::file("test.rs", src).expect("the declaration parses");
    let py = |name: &str| {
        let d = decls.iter().find(|d| d.rust_name == name).expect("declared");
        d.targets
            .iter()
            .find(|t| t.host == ir::Host::Python)
            .map(|t| (t.module.clone(), t.path.clone()))
    };
    assert_eq!(py("A"), Some(("events".into(), Some("livekit.agents.voice.events".into()))));
    assert_eq!(py("B"), Some(("other".into(), Some("livekit.agents.other".into()))));
    assert_eq!(py("C"), None);
    let twice = format!("{src}\n#[crate::host(file(module = \"x\"))]\nconst _: () = ();");
    assert!(parse::file("test.rs", &twice).unwrap_err().contains("two #[host(file(..))]"));
    let bad = "#[crate::host(file(module = \"x\", kind = \"typed_dict\"))]\nconst _: () = ();";
    assert!(parse::file("test.rs", bad).unwrap_err().contains("not `file.kind`"));
}

// Catches: a host-only field's `default_factory` dropped, or written as pydantic's shared
// `Field` in a dataclass, or its module left unimported.
#[test]
fn a_host_field_default_factory_is_a_dataclass_field_with_its_import() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(
            dataclass,
            module = "m",
            name = "Ev",
            host_field(name = "created_at", ty = "float", default_factory = "time.time")
        )]
        pub struct Ev {
            pub kind: String,
        }
        "#,
    );
    assert!(out.contains("created_at: float = field(default_factory=time.time)"), "{out}");
    assert!(out.contains("import time\n"), "{out}");
    assert!(out.contains("from dataclasses import dataclass, field"), "{out}");
}

/// A provider trait beside one record, as the generated module holds both.
const PROVIDER: &str = r#"
    #[derive(Host)]
    #[python(dataclass, module = "m", path = "livekit.agents.m", name = "Caps")]
    pub struct Caps { pub streaming: bool }

    /// A provider.
    #[crate::host(proxy(host = "livekit.agents.m.P"))]
    #[python(provider(module = "m", path = "livekit.agents.m", events = "error", error(name = "PError", kind = "p_error")))]
    pub trait P: Send + Sync {
        /// Its name.
        #[python(label)]
        fn label(&self) -> String;
        /// What it can do.
        #[python(init)]
        fn capabilities(&self) -> Caps;
        #[python(conn)]
        fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn P>> { None }
        /// Ask it.
        #[python(hook = "ask", impl = "_ask_impl")]
        async fn ask(&self, text: &str, language: Option<&str>) -> Result<Caps, ProviderError>;
        MARKER
    }
"#;

fn generate_provider(src: &str) -> Result<String, String> {
    let decls = parse::file("test.rs", src)?;
    let exposed = parse::exposed("test.rs", src)?;
    let bases = crate::bases::emit(&decls, &exposed, &crate::decl::Declarations::default())?;
    Ok(python::emit_with(&decls, &bases, &[])?.into_iter().map(|o| o.text).collect())
}

// Catches: a trait method the base class does not answer reaching a runtime `AttributeError` on
// `_core_<method>` instead of failing the codegen, which is what keeps the two from drifting.
#[test]
fn a_provider_method_without_a_python_role_is_refused() {
    let err = generate_provider(&PROVIDER.replace("MARKER", "fn flush(&self);"))
        .expect_err("an unanswered method is a codegen error");
    assert!(err.contains("P::flush") && err.contains("what is this method"), "{err}");
}

// Catches: a `_core_` method missing for a trait method, `conn_options` not reaching the plugin
// through the conn view, or an optional core argument passed as `None` where the plugin API
// takes `NOT_GIVEN`.
#[test]
fn every_provider_method_gets_its_core_surface() {
    let out = generate_provider(&PROVIDER.replace("MARKER", "")).expect("generates");
    // Wrapping aside.
    let flat = out.split_whitespace().collect::<Vec<_>>().join(" ").replace("( ", "(");
    for core in ["_core_label", "_core_capabilities", "_core_ask"] {
        assert!(flat.contains(&format!("def {core}(self")), "{core}: {out}");
    }
    // The declared proxy binds `conn_options` itself: no view of the provider.
    assert!(!out.contains("_core_with_conn_options") && !out.contains("class _PConn"), "{out}");
    for want in [
        "async def _ask_impl(self, text: str, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: APIConnectOptions ) -> Caps: ...",
        "return await self.ask(text, language=NOT_GIVEN if language is None else language, conn_options=conn )",
        "P.__module__ = \"livekit.agents.m\"",
    ] {
        assert!(flat.contains(want), "{want}: {out}");
    }
}

// Catches: a str class whose members are not the declared exports (a property named after the fn,
// a method emitted as a property), or one built without normalising through `new`.
#[test]
fn a_str_class_normalises_and_applies_its_exports() {
    let src = r#"
/// A code.
#[crate::host(string)]
#[python(str_class(module = "code", path = "livekit.agents", new = "normalize", property(base = "base_of"), method(to_name = "name_of")))]
pub struct Code(String);

/// The base.
#[crate::host(export(pid = false))]
pub fn base_of(code: &str) -> &str { code }

#[crate::host(export(pid = false))]
pub fn name_of(code: &str) -> Option<&str> { None }

#[crate::host(export(pid = false))]
pub fn normalize(code: &str) -> String { code.to_owned() }
"#;
    let x = parse::exposed("a.rs", src).expect("parses");
    let out =
        crate::forward::emit(&[], &x, &crate::decl::Declarations::default()).expect("generates");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].path, "code.py");
    let text = &out[0].text;
    for want in [
        "class Code(str):\n    \"\"\"A code.\"\"\"\n",
        "    def __new__(cls, code: str) -> Code:\n        return super().__new__(cls, _core.normalize(code))\n",
        "    @property\n    def base(self) -> str:\n        \"\"\"The base.\"\"\"\n        return _core.base_of(self)\n",
        "\n    def to_name(self) -> str | None:\n        return _core.name_of(self)\n",
        "Code.__module__ = \"livekit.agents\"\n",
    ] {
        assert!(text.contains(want), "{want}: {text}");
    }
}

// Catches: a wrapper whose constructor drops the keyword-only star or a parameter's default, or a
// forwarded export missing from the module's `__all__`.
#[test]
fn a_wrapper_builds_its_core_object_and_forwards_re_export() {
    let src = r#"
pub struct Tok { n: usize }

/// A tokenizer.
#[crate::host(object(name = "CoreTok", pid = false))]
#[python(wrapper(module = "tok", path = "livekit.agents.tok", name = "Tok", base = "..base:_Base"))]
impl Tok {
    #[host(new(kw_only, defaults(n = 20)))]
    pub fn new(n: usize) -> Self { Tok { n } }
}

#[crate::host(export(pid = false))]
#[python(forward(module = "tok"))]
pub fn split(text: &str) -> Vec<String> { Vec::new() }
"#;
    let x = parse::exposed("a.rs", src).expect("parses");
    let out =
        crate::forward::emit(&[], &x, &crate::decl::Declarations::default()).expect("generates");
    assert_eq!(out.len(), 1);
    let text = &out[0].text;
    for want in [
        "from .. import _core\nfrom .._core import split\nfrom ..base import _Base\n",
        "__all__ = [\n    \"Tok\",\n    \"split\",\n]\n",
        "class Tok(_Base):\n    \"\"\"A tokenizer.\"\"\"\n",
        "    def __init__(self, *, n: int = 20) -> None:\n        self._core = _core.CoreTok(n=n)\n",
        "Tok.__module__ = \"livekit.agents.tok\"\n",
    ] {
        assert!(text.contains(want), "{want}: {text}");
    }
}

// Catches: a constant's string written unescaped (a quote closing the Python literal early), its
// docstring dropped, a module of constants alone missing its `Final` import, or a slice written
// as anything but a tuple (a one-item tuple without its comma is its item).
#[test]
fn a_constant_is_a_final_with_its_doc() {
    let src = r#"
#[crate::host(file(module = "types", path = "livekit.agents.types"))]
const _: () = ();

/// The key.
#[crate::host(constant)]
pub const KEY: &str = "lk.\"key\"";

#[crate::host(constant)]
pub const LIMIT: i32 = 3;

#[crate::host(constant)]
pub const KEYS: &[&str] = &["a", "b"];

#[crate::host(constant)]
pub const ONE: &[i32] = &[1];
"#;
    let consts = parse::constants("a.rs", src).expect("parses");
    let out = python::emit_with(&[], &[], &consts).expect("generates");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].path, "types.py");
    let text = &out[0].text;
    assert!(text.contains("from typing import Final\n"), "{text}");
    assert!(text.contains("KEY: Final = \"lk.\\\"key\\\"\"\n\"\"\"The key.\"\"\"\n"), "{text}");
    assert!(text.contains("LIMIT: Final = 3\n"), "{text}");
    assert!(text.contains("KEYS: Final = (\"a\", \"b\")\n"), "{text}");
    assert!(text.contains("ONE: Final = (1,)\n"), "{text}");
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
        session.contains("from enum import Enum, unique\n")
            && session.contains("@unique\nclass CloseReason(str, Enum):\n    \"\"\"Why a session closed.\"\"\"\n\n    ERROR = \"error\"\n    JOB_SHUTDOWN = \"job-shutdown\"\n    \"\"\"The job ended.\"\"\"\n")
            && session.contains("\nCloseReason.__module__ = \"livekit.agents.session\"\n"),
        "{session}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("::lk_agents_abi::llm::class::CloseReason::JobShutdown => \"job-shutdown\"")
            && shell.contains("crate::host::construct(CLOSE_REASON.import(py, \"livekit.agents._generated.session\", \"CloseReason\")?.as_any(), (value,), None)?")
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
        proxy("#[host(await)] async fn listen(&self, #[host(context)] cx: &crate::invoke::CallContext, text: String) -> Result<(), HostError>;")
    ))
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains("#[pyo3(signature = (text))]")
            && llm.contains("let cx = crate::host::entered();\n        let this =")
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
fn provider_markup_forwards_legacy_and_declared_classes_through_the_same_contract() {
    let records = r#"
        #[derive(Host)]
        #[python(dataclass, module = "m", path = "livekit.agents.m", name = "MarkupInfo")]
        pub struct MarkupInfo { pub nonverbals: String }
        #[derive(Host)]
        #[python(typed_dict, module = "expressive", path = "livekit.agents.m", name = "SpeechSteeringOptions")]
        #[binding(extract, from_items)]
        pub struct SpeechSteering { pub rate: f64 }
    "#;
    let provider =
        PROVIDER.replace("MARKER", "#[python(markup)] fn provider_key(&self) -> String;");
    let common = format!("{records}\n{provider}");
    let legacy = r#"
        /// Markup helper.
        #[crate::host(object(name = "Markup"))]
        impl Markup {
            #[host(new)] pub fn new(provider: &str) -> Self { todo!() }
            /// The dialect key.
            #[host(method(name = "_provider_key"))]
            pub fn provider_key(&self) -> String { todo!() }
            /// Instructions for this voice.
            #[host(method(kw_only))]
            pub fn llm_instructions(&self, speech_steering: Option<&SpeechSteering>) -> Option<String> { todo!() }
            #[host(method)] pub fn normalize(&self, text: &str) -> String { todo!() }
            #[host(method)] pub fn convert(&self, text: &str) -> String { todo!() }
        }
    "#;
    let declared = r#"
        /// Markup helper.
        #[crate::host(class(name = "Markup"))]
        pub trait Markup: Sized {
            #[host(new)] fn new(provider: String) -> Self;
            /// The dialect key.
            #[host(call, name = "_provider_key")] fn provider_key(&self) -> String;
            /// Instructions for this voice.
            #[host(call)] fn llm_instructions(&self, #[host(kw, default = None)] speech_steering: Option<SpeechSteering>) -> Option<String>;
            #[host(call)] fn normalize(&self, text: String) -> String;
            #[host(call)] fn convert(&self, text: String) -> String;
        }
    "#;
    let records = parse::file("test.rs", &common).unwrap();
    let legacy_exposed = parse::exposed("test.rs", &format!("{common}\n{legacy}")).unwrap();
    let legacy_bases =
        crate::bases::emit(&records, &legacy_exposed, &crate::decl::Declarations::default())
            .unwrap();
    let text = format!("{common}\n{declared}");
    let tree = crate::decl::tree::Crate::from_files(
        "lk_agents_abi",
        &[("lib.rs", "pub mod tts;"), ("tts.rs", &text)],
    )
    .unwrap();
    let declarations = crate::decl::resolve::declarations(&tree).unwrap();
    let exposed = parse::exposed("test.rs", &common).unwrap();
    let declared_bases = crate::bases::emit(&records, &exposed, &declarations).unwrap();
    // Legacy dictionary extractors intentionally expose Any; the declaration keeps the
    // record's TypedDict name. All forwarding, docs, defaults and calls must otherwise match.
    let old_annotation = "speech_steering: Any | None = None";
    assert_eq!(legacy_bases[0].body.matches(old_annotation).count(), 1);
    assert_eq!(
        legacy_bases[0].body.replacen(
            old_annotation,
            "speech_steering: SpeechSteeringOptions | None = None",
            1
        ),
        declared_bases[0].body,
        "switching declaration form must preserve the full forwarding body except the typed annotation"
    );
    let body = &declared_bases[0].body;
    assert!(body.contains("def llm_instructions(self, *, speech_steering: SpeechSteeringOptions | None = None) -> str | None:"), "{body}");
    assert!(
        body.contains("return self._core().llm_instructions(speech_steering=speech_steering)"),
        "{body}"
    );
    assert!(
        body.contains("def normalize(self, text: str) -> str:")
            && body.contains("return self._core().normalize(text)"),
        "{body}"
    );
    assert_eq!(body.matches("def convert(").count(), 1, "each declared method forwards once");
    assert_eq!(
        declared_bases[0]
            .imports
            .iter()
            .filter(|(module, name)| module == ".expressive" && name == "SpeechSteeringOptions")
            .count(),
        1,
        "the declared record must import its canonical TypedDict exactly once"
    );
    assert!(
        !declared_bases[0]
            .imports
            .iter()
            .any(|(module, name)| module == ".m" && name == "SpeechSteeringOptions"),
        "records already beside the base must not import their own module"
    );
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
        "if ::lk_agents_abi::python::is_not_given(v)? { Ok(::lk_agents_abi::Maybe::NotGiven) } else { (v.extract::<::std::string::String>()).map(::lk_agents_abi::Maybe::Given) }"
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
