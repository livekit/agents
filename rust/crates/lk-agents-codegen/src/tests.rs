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
    let decls = parse::file("test.rs", "lk_agents_abi", src).expect("the declaration parses");
    python::emit(&decls)
        .expect("the declaration generates")
        .into_iter()
        .map(|o| o.text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The error a bad Python declaration produces.
fn refuse_python(src: &str) -> String {
    let decls = match parse::file("test.rs", "lk_agents_abi", src) {
        Ok(d) => d,
        Err(e) => return e,
    };
    python::emit(&decls).err().unwrap_or_else(|| panic!("expected a refusal, got a generated file"))
}

/// The `_core` classes' stub, for the `class` kind.
fn generate_stub(src: &str) -> String {
    let decls = parse::file("test.rs", "lk_agents_abi", src).expect("the declaration parses");
    stub::emit(&decls, &ir::Exposed::default()).expect("the stub generates").expect("there is a class").text
}

// --- typed dicts --------------------------------------------------------------------------------

// Catches: `total` dropped or defaulted the wrong way. `total=True` (Python's default, written as
// nothing) makes every key required; `total=False` makes every key optional -- either mistake is
// a user-visible API change that fails only at the call that omits (or relies on) a key.
#[test]
fn a_typed_dict_says_total_false_only_when_declared() {
    for (total, want) in [("total = \"False\", ", "class Opts(TypedDict, total=False):"), ("", "class Opts(TypedDict):")] {
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
// no emitted field (the tag silently leads); an `order` that is not a permutation of the fields;
// a declaration in a path-less group (module "").
#[test]
fn a_declaration_key_that_cannot_be_honoured_is_refused_by_name() {
    let opts = |attr: &str| format!("#[derive(Host)]\n#[python({attr}, module = \"m\", name = \"Opts\")]\npub struct Opts {{ pub a: bool, pub b: bool }}");
    let cases = [
        (opts("dataclass, total = \"False\""), &["dataclass", "total"][..]),
        (opts("typed_dict, total = \"false\""), &["not a Python bool", "\"false\""]),
        (opts("typed_dict, typed_dict_from = \"typing_extension\""), &["typing_extension"]),
        (opts("pydantic, tag = \"type\", tag_value = \"h\", tag_after = \"nope\""), &["tag_after", "nope"]),
        (opts("typed_dict, order = \"a\""), &["leaves out `b`"]),
        (opts("typed_dict, order = \"a, b, a\""), &["twice"]),
        (opts("typed_dict, order = \"a, b, c\""), &["`c`"]),
        ("#[crate::host(group(kind = \"typed_dict\"))]\npub mod m { pub struct Opts { pub a: Option<bool> } }".into(), &["module"]),
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

// Catches: pydantic's own `Field(deprecated="..")` being confused with `#[python(deprecated)]`,
// which is a different mechanism -- a module-level `__getattr__` with a fixed sentence. This is
// the argument pydantic itself takes, and it is what `ChatMessage.hash` carries, so a caller
// reading the field gets a `DeprecationWarning` rather than nothing.
#[test]
fn a_deprecated_msg_composes_with_the_fields_own_default() {
    let out = generate_python(
        r#"
        #[derive(Host)]
        #[python(pydantic, module = "m", name = "Msg")]
        pub struct Msg {
            #[python(default = "None", deprecated_msg = "hash is deprecated")]
            pub hash: Option<String>,
        }
        "#,
    );
    assert!(
        out.contains(r#"hash: str | None = Field(default=None, deprecated="hash is deprecated")"#),
        "{out}"
    );
    assert!(out.contains("from pydantic import"), "{out}");
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
    let decls = parse::file("test.rs", "lk_agents_abi", src).expect("the declarations parse");
    let x = parse::exposed("test.rs", "core", src).expect("the items parse");
    stub::emit(&decls, &x).expect("the stub generates").expect("there are items").text
}

// Catches: a raw-identifier parameter reaching the stub as `r#ref: str`, which is not Python:
// `ast.parse` of `_core_classes.pyi` failed on it (2026-09-24). pyo3 already calls it `ref`.
#[test]
fn a_keyword_parameter_is_its_python_name_in_the_stub() {
    let stub = exports_stub("#[host(export(kw_only))]\npub fn resolve_ref(root: &str, r#ref: &str) -> bool { true }");
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

/// Every generated **Node** module's text, concatenated.
fn generate_node(src: &str) -> String {
    let decls = parse::file("test.rs", "lk_agents_abi", src).expect("the declaration parses");
    crate::node::emit(&decls)
        .expect("the declaration generates")
        .into_iter()
        .map(|o| o.text)
        .collect::<Vec<_>>()
        .join("\n")
}

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
    assert!(out.contains("    a: float = 1.0\n    next: Out | None = None\n    b: bool = False\n"), "{out}");
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
    assert!(out.contains("    def __post_init__(self) -> None:\n        _bindings.data_post_init(self)\n"), "{out}");
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

// --- host groups and literal consts ---------------------------------------------------------------

const GROUP: &str = r#"
    #[crate::host(group(kind = "typed_dict", typed_dict_from = "typing", total = "False", node))]
    pub mod gateway {
        #[crate::host(group(path = "livekit.agents.inference.stt"))]
        pub mod stt {
            /// Doc of the literal.
            pub const DEEPGRAM_MODELS: &[&str] = &["deepgram/nova-3", "deepgram/nova-2"];
            #[host(name = "STTEncoding")]
            pub const STT_ENCODING: &[&str] = &["pcm_s16le"];
            #[host(name = "STTModels", union = "DeepgramModels, STTEncoding")]
            pub const STT_MODELS: &[&str] = &["auto"];
            #[host(union = "Opts, str")]
            pub const OPTS_OR_NAME: &[&str] = &[];
            pub const NOT_A_LITERAL: u32 = 3;

            pub struct Opts {
                pub interim: Option<bool>, // a comment, not a docstring
                pub timeout: Option<Option<f64>>,
                #[python(ty = "str | list[str]")]
                pub keyterm: Option<Value>,
            }
        }
    }
"#;

// Catches: a model vocabulary that costs more to declare than the Python it replaces. A string
// slice is one line per name; its Python name comes from the const's words unless written, and a
// union names its members and appends its own strings.
#[test]
fn a_string_slice_const_in_a_group_is_a_literal_and_a_union() {
    let out = generate_python(GROUP);
    assert!(out.contains("DeepgramModels = Literal[\"deepgram/nova-3\", \"deepgram/nova-2\"]\n\"\"\"Doc of the literal.\"\"\"\n"), "{out}");
    assert!(out.contains("STTEncoding = Literal[\"pcm_s16le\"]\n"), "{out}");
    assert!(out.contains("STTModels = DeepgramModels | STTEncoding | Literal[\"auto\"]\n"), "{out}");
    assert!(out.contains("OptsOrName = Opts | str\n"), "{out}");
    assert!(!out.contains("NotALiteral") && !out.contains("NOT_A_LITERAL"), "{out}");
}

// Catches: the group's defaults not reaching a declaration -- a struct with no attributes of its
// own must still be a `typing` `total=False` `TypedDict` under the group's module and path, with
// an outer `Option` read as absence (`Option<Option<f64>>` is `float | None`).
#[test]
fn a_group_gives_its_structs_kind_module_path_and_total() {
    let decls = parse::file("test.rs", "lk_agents_abi", GROUP).expect("parses");
    let py = python::emit(&decls).expect("generates");
    assert_eq!(py.len(), 1);
    assert_eq!(py[0].path, "inference_stt.py");
    let out = &py[0].text;
    assert!(out.contains("from typing import Literal, TypedDict\n"), "{out}");
    assert!(
        out.contains("class Opts(TypedDict, total=False):\n    interim: bool\n    timeout: float | None\n    keyterm: str | list[str]\n"),
        "{out}"
    );
    assert!(!out.contains("a comment"), "{out}");
    assert!(out.contains("Opts.__module__ = \"livekit.agents.inference.stt\""), "{out}");
}

// Catches: `node` in a group not giving each declaration its TypeScript twin, and a builtin-only
// Python annotation still demanding a hand-written `#[node(ty = ..)]`.
#[test]
fn a_group_with_node_emits_the_typescript_twins() {
    let out = generate_node(GROUP);
    assert!(out.contains("export type DeepgramModels =\n  | 'deepgram/nova-3'\n  | 'deepgram/nova-2';\n"), "{out}");
    assert!(out.contains("export type STTModels =\n  | DeepgramModels\n  | STTEncoding\n  | 'auto';\n"), "{out}");
    assert!(out.contains("export type OptsOrName =\n  | Opts\n  | string;\n"), "{out}");
    assert!(out.contains("  interim?: boolean;\n  timeout?: number | null;\n  keyterm?: string | string[];\n"), "{out}");
}

// Catches: the Python-to-TypeScript rule guessing at a name it cannot see.
#[test]
fn ts_of_spells_builtins_and_refuses_other_names() {
    use crate::node::ts_of;
    assert_eq!(ts_of("Literal[\"a\", 1, False] | None").as_deref(), Some("'a' | 1 | false | null"));
    assert_eq!(ts_of("list[tuple[str, float]]").as_deref(), Some("[string, number][]"));
    assert_eq!(ts_of("dict[str, Any]").as_deref(), Some("Record<string, unknown>"));
    assert_eq!(ts_of("str | list[str]").as_deref(), Some("string | string[]"));
    assert_eq!(ts_of("ReasoningEffort | None"), None);
}

// --- host-stream dispatch -------------------------------------------------------------------------

const ROUTED: &str = "#[derive(Host)]\n#[python(pydantic, module = \"events\", name = \"TimeoutEvent\", tag = \"type\", tag_value = \"user_timeout\")]\n\
    #[binding(emit, tag = \"type\", tag_value = \"timeout\", from = core::Timeout)]\n\
    pub struct Timeout { pub at: f64 }\n\
    #[derive(Host)]\n#[binding(emit, tag = \"type\", tag_value = \"done\", from = core::Done)]\n\
    pub struct Done { pub id: String }\n\
    #[derive(Host)]\n#[binding(emit, from = core::Stream)]\n#[python(dispatch = \"dispatch\")]\n\
    pub enum Stream {\n    #[python(event)]\n    Timeout(Timeout),\n    #[python(ignored)]\n    Done(Done),\n}";

// Catches: a routing table keyed by the Python discriminant instead of the wire kind the binding
// sends (they differ -- `transcription_timeout` is emitted as `user_transcription_timeout`), an
// event member routed to a class the host cannot import, or a routed member with no module.
#[test]
fn a_routed_union_generates_its_dispatch_table_keyed_by_the_wire_kind() {
    let out = generate_python(ROUTED);
    assert!(out.contains("from .events import TimeoutEvent\n"), "{out}");
    assert!(out.contains("PUBLIC_EVENTS: dict[str, tuple[type[Any], str]] = {\n    \"timeout\": (TimeoutEvent, \"user_timeout\"),\n}"), "{out}");
    assert!(out.contains("IGNORED_EVENTS: frozenset[str] = frozenset(\n    {\n        \"done\",\n    }\n)"), "{out}");
    let e = refuse_python(&ROUTED.replace("#[python(dispatch = \"dispatch\")]\n", ""));
    assert!(e.contains("names no #[python(dispatch"), "{e}");
    let e = refuse_python(&ROUTED.replace("#[python(ignored)]", "#[python(event)]"));
    assert!(e.contains("needs a pydantic Python target"), "{e}");
}

// --- node targets --------------------------------------------------------------------------------

// Catches: a Node target that writes neither `name` nor `module` not taking the Python target's
// (the 85 records that repeated both), or one that writes its own being overridden.
#[test]
fn a_node_target_is_named_and_placed_as_the_python_one() {
    let decls = parse::file(
        "test.rs",
        "lk_agents_abi",
        "#[derive(Host)]\n#[python(pydantic, module = \"events\", name = \"ThingEvent\")]\n#[node(interface)]\npub struct Thing { pub a: u8 }\n\
         #[derive(Host)]\n#[python(pydantic, module = \"events\", name = \"Other\")]\n#[node(interface, module = \"voice\", name = \"OtherJs\")]\npub struct Other { pub a: u8 }",
    )
    .expect("the declarations parse");
    let node = decls[0].target(crate::ir::Host::Node).expect("a node target");
    assert_eq!((node.name.as_str(), node.module.as_str()), ("ThingEvent", "events"));
    let node = decls[1].target(crate::ir::Host::Node).expect("a node target");
    assert_eq!((node.name.as_str(), node.module.as_str()), ("OtherJs", "voice"));
}

// Catches: a stream's stub missing a method its class has, or the push argument typed `Any` where
// the declaration named the host type.
#[test]
fn a_stream_class_has_its_stub() {
    let stub = exports_stub(
        "#[host(error(with = \"f\"))]\npub enum ProviderError { A }\n\
         #[host(stream(name = \"S\", doc = \"One stream.\", sink = X, push = \"push_frame\", arg = \"frame\", input = AudioFrame, \
         input_py = \"rtc.AudioFrame\", stub_import = \"from livekit import rtc\", end_input, output = u8, stub = \"def extra(self) -> None: ...\"))]\npub struct Src;",
    );
    assert!(stub.contains("from livekit import rtc\n"), "{stub}");
    assert!(stub.contains("class S:\n    \"\"\"One stream.\"\"\"\n\n    def push_frame(self, frame: rtc.AudioFrame) -> None: ...\n    def flush(self) -> None: ...\n    def end_input(self) -> None: ...\n    def extra(self) -> None: ...\n    async def recv(self) -> Any: ...\n    async def aclose(self) -> None: ...\n"), "{stub}");
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
    let stub = exports_stub("#[host(export)]\npub fn f(a: Option<u8>, b: u8, c: Option<u8>, #[host(required)] d: Option<u8>, e: Option<u8>) {}");
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
    assert!(out.contains("    tokens: int = 0\n") && out.contains("    duration: float = 0.0\n"), "{out}");
    assert!(out.contains("    note: str | None = None\n") && out.contains("    requests: int = 1\n"), "{out}");
    assert!(out.contains("    timestamp: float\n"), "{out}");
    let out = generate_stub(
        r#"
        #[derive(Host)]
        #[binding(class, new, default, from = core::Thing)]
        pub struct Thing { pub name: String, pub count: u32 }
        "#,
    );
    assert!(out.contains("def __init__(self, *, name: str = \"\", count: int = 0) -> None"), "{out}");
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
    let decls = parse::file("test.rs", "lk_agents_abi", src).expect("the declaration parses");
    let py = |name: &str| {
        let d = decls.iter().find(|d| d.rust_name == name).expect("declared");
        d.targets.iter().find(|t| t.host == ir::Host::Python).map(|t| (t.module.clone(), t.path.clone()))
    };
    assert_eq!(py("A"), Some(("events".into(), Some("livekit.agents.voice.events".into()))));
    assert_eq!(py("B"), Some(("other".into(), Some("livekit.agents.other".into()))));
    assert_eq!(py("C"), None);
    let twice = format!("{src}\n#[crate::host(file(module = \"x\"))]\nconst _: () = ();");
    assert!(parse::file("test.rs", "m", &twice).unwrap_err().contains("two #[host(file(..))]"));
    let bad = "#[crate::host(file(module = \"x\", kind = \"typed_dict\"))]\nconst _: () = ();";
    assert!(parse::file("test.rs", "m", bad).unwrap_err().contains("not `file.kind`"));
}

// --- the session schema ------------------------------------------------------------------------

// Catches: a `#[table]` record of `lk-agents-storage` that the generator cannot read (the derive's
// own `#[table]` beside `#[python(..)]`), a `type` column that
// loses its name, a JSONB column rendered as something narrower than `Any`, or one of the five row
// models of plan 13.5 missing from the `storage` module.
#[test]
fn the_session_schema_renders_its_five_pydantic_row_models() {
    let src = include_str!("../../lk-agents-storage/src/schema.rs");
    let decls = parse::file("schema.rs", "lk_agents_storage::schema", src).expect("the schema parses");
    let outs = python::emit(&decls).expect("the schema generates");
    let storage = outs.iter().find(|o| o.path.contains("storage")).expect("a storage module");
    for model in ["ChatItemRow", "EventRow", "LaneRow", "InboxRow", "TaskRow"] {
        assert!(storage.text.contains(&format!("class {model}(BaseModel):")), "{model} missing:\n{}", storage.text);
    }
    assert!(storage.text.contains("    type: str"), "{}", storage.text);
    assert!(storage.text.contains("    payload: Any"), "{}", storage.text);
    assert!(storage.text.contains("    state: Any | None"), "{}", storage.text);
    assert!(storage.text.contains("    lease_until: int"), "{}", storage.text);
}
