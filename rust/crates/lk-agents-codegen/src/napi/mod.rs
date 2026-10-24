//! The Node binding: napi-rs wrappers over the same IR the pyo3 emitters read, into
//! `lk-agents-node/src/generated/`. Records become `#[napi(object)]` structs, exports `#[napi]`
//! fns, objects `#[napi]` classes over the same cell, error types `e_to_node`.
//!
//! Names are camelCase via [`crate::node::camel`], with an explicit `js_name` where napi could
//! differ. Every field's `ts_type` is [`crate::node::field_ts`]'s, so `index.d.ts` agrees with
//! `node.rs`. Semantics are Python's: the declaration's defaults, `NotGiven`, and errors carrying
//! the Python exception kind. Unsupported items are skipped and listed in `generated/mod.rs`'s
//! header, never a failed run.

mod functions;
mod objects;
mod proxies;
mod records;
mod types;

use std::cell::RefCell;
use std::rc::Rc;

use crate::ir::*;
use crate::python::Out;
use functions::{error_fn, function};
use objects::{object, stream};
use proxies::proxy;
use records::{emit_rec, rec, Rec};

/// An item from any other crate is a gap, not a compile error.
pub const LINKED: &[&str] =
    &["lk_agents_abi", "lk_agents_voice", "lk_agents_inference", "lk_agents_worker", "lk_agents_media"];

/// `#[host(object)]` classes whose methods land on a hand-written napi class, by path.
const ON_HAND_WRITTEN: &[(&str, &str)] = &[("LiveSession", "crate::session::LiveSession")];

fn on_hand_written(o: &Object) -> Option<&'static str> {
    ON_HAND_WRITTEN.iter().find(|(n, _)| *n == o.name).map(|(_, c)| *c)
}

/// Names `lk-agents-node` writes by hand; napi keys by Rust ident and `index.d.ts` is one namespace.
const HAND_WRITTEN: &[&str] = &[
    "AgentCore",
    "LiveSession",
    "TtsInfo",
    "DriveOutcome",
    "FrameView",
    "JsProviderError",
    "JsErrorOut",
    "SpanAttribute",
    "SpanRecord",
    "LogRecord",
    "TelemetryRecord",
    "JsAudioFrame",
    "JsSynthesizedAudio",
    "JsOpenOutcome",
    "JsNextOutcome",
];

/// Non-declaration core types with a Node shape: `(core ident, napi type, TypeScript, core -> napi
/// with `{x}` a `&Core`, napi -> core with `{v}` the napi value)`.
const HAND_TYPES: &[(&str, &str, &str, &str, &str)] = &[
    (
        "AudioFrame",
        "crate::providers::host::JsAudioFrame",
        "JsAudioFrame",
        "crate::providers::host::audio_frame_to_js({x})",
        "crate::providers::host::audio_frame_from_js({v})",
    ),
    ("LanguageCode", "String", "string", "({x}).as_str().to_owned()", "lk_agents_abi::LanguageCode::new(&{v})"),
    (
        "Role",
        "String",
        "'developer' | 'system' | 'user' | 'assistant'",
        "({x}).as_str().to_owned()",
        "lk_agents_abi::Role::from_literal(&{v}).ok_or_else(|| crate::host_error::JsError::new(\"ValueError\", format!(\"not a chat role: {}\", {v})))?",
    ),
    (
        "AudioEncoding",
        "String",
        "string",
        "crate::providers::host::encoding_to_js({x})",
        "crate::providers::host::encoding_from_js({v})",
    ),
];

const JSON_MAP_ALIASES: &[&str] = &["Extra", "Object"];

/// Records hand-written `lk-agents-node` code names by their generated twin.
const USED_BY_HAND: &[&str] = &["TimedString"];

/// `#[host(error(with = ..))]` types; their Node converters are in `lk-agents-node/src/host_error.rs`.
const ERROR_WITH: &[(&str, &str)] = &[
    ("SchemaError", "crate::host_error::schema_error_to_node"),
    ("ProviderError", "crate::host_error::provider_error_to_node"),
    ("IdleRefused", "crate::host_error::idle_refused_to_node"),
    ("CommitSuperseded", "crate::host_error::commit_superseded_to_node"),
];

const HEADER: &str = "//! GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`); see `napi.rs` in `lk-agents-codegen`.\n";

/// Anything wider crosses as an `i64`.
const NATIVE_INTS: &[&str] = &["u8", "i8", "u16", "i16", "u32", "i32", "i64"];

/// `None` while it is being built.
type RecSlot = (String, Option<Result<Rc<Rec>, String>>);

struct Cx<'a> {
    decls: &'a [Decl],
    x: &'a Exposed,
    /// `None` while a plan is being built (a cycle is a refusal).
    recs: RefCell<Vec<RecSlot>>,
    errors: RefCell<Vec<String>>,
    /// Converting a parameter (where a declared enum is parsed, as `exports.rs` parses it).
    in_params: std::cell::Cell<bool>,
}

#[derive(Clone, Copy, PartialEq)]
enum Dir {
    ToJs,
    FromJs,
}

pub fn emit(decls: &[Decl], x: &Exposed) -> Result<Vec<Out>, String> {
    let cx = Cx { decls, x, recs: RefCell::new(Vec::new()), errors: RefCell::new(Vec::new()), in_params: std::cell::Cell::new(false) };
    let mut gaps: Vec<(String, String)> = Vec::new();
    let mut wanted: Vec<String> = Vec::new();

    let mut fns = Vec::new();
    for f in x.fns.iter().filter(|f| !f.python_only) {
        match function(&cx, f, &mut wanted) {
            Ok(b) => fns.push(b),
            Err(e) => gaps.push((f.rust_path.clone(), e)),
        }
    }
    let fn_count = fns.len();
    let mut objects = Vec::new();
    let mut class_names: Vec<String> = Vec::new();
    for o in &x.objects {
        match object(&cx, o, &mut wanted, &mut gaps) {
            Ok(b) => {
                class_names.push(o.name.clone());
                objects.push(b)
            }
            Err(e) => gaps.push((o.core_path.clone(), e)),
        }
    }
    let mut streams = Vec::new();
    for s in &x.streams {
        let mut mine = Vec::new();
        match stream(&cx, s, &mut mine) {
            Ok(b) => {
                class_names.push(s.name.clone());
                streams.push(b);
                wanted.extend(mine);
            }
            Err(e) => gaps.push((s.name.clone(), e)),
        }
    }
    for k in x.kwargs.iter().filter(|k| k.client) {
        gaps.push((k.path.clone(), "not yet: a #[host(client)] constructor lands on a hand-written pyclass".to_owned()));
    }
    let mut proxies = Vec::new();
    for p in &x.proxies {
        match proxy(&cx, p, &mut wanted) {
            Ok(b) => proxies.push(b),
            Err(e) => gaps.push((p.trait_path.clone(), e)),
        }
    }
    let proxy_count = proxies.len();

    let mut order: Vec<String> = Vec::new();
    for d in decls {
        let Some(t) = d.target(Host::Node) else { continue };
        if t.inline || matches!(d.body, Body::Unit(_)) || d.rust_name.starts_with("py:") {
            continue;
        }
        match rec(&cx, &key(d)) {
            Ok(_) => push_unique(&mut order, &key(d)),
            Err(e) => gaps.push((format!("{} (record {})", d.rust_name, t.name), e)),
        }
    }
    for w in wanted {
        push_unique(&mut order, &w);
    }
    // Only where declared: a test's source is not the tree.
    for h in USED_BY_HAND.iter().filter(|h| decls.iter().any(|d| &d.rust_name == *h)) {
        match rec(&cx, h) {
            Ok(_) => push_unique(&mut order, h),
            Err(e) => return Err(format!("`{h}` is used by hand-written lk-agents-node code but cannot be generated: {e}")),
        }
    }
    let mut i = 0;
    while i < order.len() {
        let r = rec(&cx, &order[i]).expect("only planned records are queued");
        for dep in &r.deps {
            push_unique(&mut order, dep);
        }
        i += 1;
    }
    let mut records = Vec::new();
    for name in &order {
        let r = rec(&cx, name).expect("planned");
        if class_names.contains(&r.ident) {
            return Err(format!("{}: record `{}` and a generated class share one napi name", r.decl, r.ident));
        }
        records.push(emit_rec(&cx, &r));
    }

    let errors: Vec<String> = cx
        .errors
        .borrow()
        .iter()
        .map(|n| error_fn(x.errors.iter().find(|e| &e.name == n).expect("recorded from the list")))
        .collect();

    let mut out = Vec::new();
    let mut modules = Vec::new();
    let mut file = |name: &str, items: Vec<String>, prelude: &str| {
        if items.is_empty() {
            return;
        }
        let mut b = format!("{HEADER}\n#![allow(clippy::all, dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]\n\n{prelude}");
        for i in items {
            b.push('\n');
            b.push_str(&i);
        }
        out.push(Out { path: format!("{name}.rs"), text: b });
        modules.push(name.to_owned());
    };
    file("records", records, "use napi::bindgen_prelude::JsObjectValue as _;\nuse napi_derive::napi;\n");
    file("functions", fns, "use napi::Env;\nuse napi_derive::napi;\n");
    file("objects", objects, "use napi::Env;\nuse napi_derive::napi;\n");
    file("errors", errors, "use crate::host_error::JsError;\n");
    file("proxies", proxies, "");
    file("streams", streams, "use napi::Env;\nuse napi_derive::napi;\n");

    let (n_fns, n_classes, n_records) = (fn_count, class_names.len(), order.len());
    let mut m = format!(
        "{HEADER}//!\n//! The Node twin of `lk-agents-py/src/generated`: {} core items bound ({n_fns} functions and \
         consts, {n_classes} classes, {n_records} records, {proxy_count} proxies).\n//!\n//! Not bound, and why -- each is a gap in the \
         generator or in a declaration, not in this crate:\n//!\n",
        n_fns + n_classes + n_records + proxy_count,
    );
    for (item, why) in &gaps {
        m.push_str(&format!("//! - `{item}`: {why}\n"));
    }
    m.push('\n');
    for name in &modules {
        m.push_str(&format!("pub mod {name};\n"));
    }
    out.push(Out { path: "mod.rs".to_owned(), text: m });
    Ok(out)
}

fn push_unique(v: &mut Vec<String>, s: &str) {
    if !v.iter().any(|x| x == s) {
        v.push(s.to_owned());
    }
}

/// Node name, else binding class name, else the Rust identifier; disambiguated when two records
/// share a host name, since `index.d.ts` is one namespace.
pub(crate) fn js_name(all: &[Decl], d: &Decl) -> String {
    let base = host_name(d);
    let clash = all.iter().any(|o| key(o) != key(d) && !matches!(o.body, Body::Unit(_)) && host_name(o) == base);
    if !(clash || HAND_WRITTEN.contains(&base.as_str())) {
        return base;
    }
    match d.rust_module.last() {
        Some(m) if d.rust_name == base => {
            let mut cs = m.chars();
            cs.next().map(|c| c.to_uppercase().chain(cs).collect::<String>()).unwrap_or_default() + &base
        }
        _ => d.rust_name.clone(),
    }
}

/// Rust name under its inline modules: two `group` modules may each declare a `CartesiaOptions`.
pub(crate) fn key(d: &Decl) -> String {
    if d.rust_module.is_empty() { d.rust_name.clone() } else { format!("{}::{}", d.rust_module.join("::"), d.rust_name) }
}

fn by_key<'a>(decls: &'a [Decl], k: &str) -> Option<&'a Decl> {
    decls.iter().find(|d| key(d) == k)
}

fn host_name(d: &Decl) -> String {
    d.target(Host::Node)
        .map(|t| t.name.clone())
        .or_else(|| d.targets_for(Host::Binding).find(|t| t.kind == Kind::Class).map(|t| t.name.clone()))
        .unwrap_or_else(|| d.rust_name.clone())
}

fn summary(doc: &[String]) -> Vec<String> {
    let end = doc.iter().position(|l| l.is_empty()).unwrap_or(doc.len());
    doc[..end].iter().map(|l| l.replace("[`", "`").replace("`]", "`")).collect()
}

fn linked(path: &str) -> Result<(), String> {
    let krate = path.split("::").next().unwrap_or_default();
    if LINKED.contains(&krate) { Ok(()) } else { Err(format!("`{krate}` is not linked by lk-agents-node")) }
}

/// A blank doc line is `///`, not `/// `: rustfmt would flag the trailing space.
fn doc_line(line: &str, indent: &str) -> String {
    if line.is_empty() { format!("{indent}///\n") } else { format!("{indent}/// {line}\n") }
}

/// Every raw-able Rust keyword, strict and reserved: a partial list once generated an unparsable
/// `pub for: bool`. `self`, `Self`, `super`, `crate` cannot be raw.
const KEYWORDS: [&str; 47] = [
    "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "do", "dyn",
    "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl", "in", "let",
    "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub", "ref", "return",
    "static", "struct", "trait", "true", "try", "type", "typeof", "unsafe", "unsized", "use",
    "virtual", "where", "while",
];

/// Raw when a Rust keyword: pyo3's `IntoPyObject` unraws `r#type` back to `"type"`.
fn raw(name: &str) -> String {
    if KEYWORDS.contains(&name) { format!("r#{name}") } else { name.to_owned() }
}

/// A function parameter's Rust identifier. Not [`raw`]: napi-derive writes a parameter's TS name
/// as the camelCase of the pattern's *text*, so `r#ref` reached `index.d.ts` verbatim (and broke
/// `napi build`). A trailing `_` is a word boundary to that camelCase, so `ref_` is `ref` in TS.
fn arg(name: &str) -> String {
    if KEYWORDS.contains(&name) { format!("{name}_") } else { name.to_owned() }
}

fn snake(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase() {
            let prev_lower = i > 0 && (chars[i - 1].is_ascii_lowercase() || chars[i - 1].is_ascii_digit());
            let next_lower = i + 1 < chars.len() && chars[i + 1].is_ascii_lowercase();
            let prev_upper = i > 0 && chars[i - 1].is_ascii_uppercase();
            if i > 0 && (prev_lower || (prev_upper && next_lower)) {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(*c);
        }
    }
    out
}
