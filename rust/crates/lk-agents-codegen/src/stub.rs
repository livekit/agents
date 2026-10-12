//! The `_core` classes' stub, `_generated/_core_classes.pyi`, star-imported by `_core.pyi`: a
//! generated class must bring its stub or griffe and the equivalence gate lose its signatures.
//! Getters are `@property` (read-only on a frozen pyclass). Field types are
//! [`python::render`]'s, except: a converted field is its `ty` or `Any`, a unit enum is `str`, a
//! nested payload is `dict[str, Any]`.

use crate::ir::*;
use crate::python::{self, Out};

pub const PATH: &str = "_core_classes.pyi";

/// `None` when no declaration is a class; then no file is written.
pub fn emit(decls: &[Decl], exposed: &Exposed) -> Result<Option<Out>, String> {
    let classes: Vec<(&Decl, &Target)> = decls
        .iter()
        .flat_map(|d| d.targets_for(Host::Binding).map(move |t| (d, t)))
        .filter(|(_, t)| t.kind == Kind::Class)
        .collect();
    if classes.is_empty()
        && exposed.fns.is_empty()
        && exposed.objects.is_empty()
        && exposed.streams.is_empty()
    {
        return Ok(None);
    }

    let mut sources: Vec<&str> = Vec::new();
    for (d, _) in &classes {
        if !sources.contains(&d.source.as_str()) {
            sources.push(&d.source);
        }
    }

    // Two blank lines between classes; exported functions are one block, as in `_core.pyi`.
    let mut body: Vec<String> = Vec::new();
    for (d, t) in &classes {
        body.push(class(decls, d, t)?);
    }
    let items = items(decls, exposed)?;
    let (fns, objects) = items.split_at(exposed.fns.len());
    if !fns.is_empty() {
        body.push(fns.concat());
    }
    body.extend(objects.iter().cloned());
    let body = body.join("\n\n");

    let mut b = String::new();
    b.push_str("# GENERATED FILE -- DO NOT EDIT.\n#\n");
    b.push_str("# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:\n");
    for s in &sources {
        b.push_str(&format!("#     {s}\n"));
    }
    b.push_str(
        "# and the `#[host(export)]` / `#[host(object)]` items in `EXPORT_SOURCES`.\n\
         #\n# The stub of every `_core` class and function the binding generates rather than\n\
         # writes. `_core.pyi` star-imports this file, so the hand-written stub carries only what\n\
         # is still hand-written.\n\
         #\n# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a\n\
         # non-empty diff, so a hand edit here is reverted rather than merged.\n\n",
    );
    // Imports spotted in the rendered text; anything else comes from an object's `stub_import`.
    let mut wrote = false;
    if body.contains("datetime.") {
        b.push_str("import datetime\n");
        wrote = true;
    }
    let typing: Vec<&str> = ["Any", "Literal"].into_iter().filter(|n| body.contains(n)).collect();
    if !typing.is_empty() {
        b.push_str(&format!("from typing import {}\n", typing.join(", ")));
        wrote = true;
    }
    for l in imports(exposed) {
        b.push_str(&format!("{l}\n"));
        wrote = true;
    }
    if wrote {
        b.push('\n');
    }
    b.push_str(&body);
    Ok(Some(Out { path: PATH.to_owned(), text: b }))
}

fn class(all: &[Decl], d: &Decl, t: &Target) -> Result<String, String> {
    let Body::Struct(fields) = &d.body else {
        return Err(format!("{}: only a struct can be a `class` target", d.rust_name));
    };
    let readable: Vec<&Field> = fields.iter().filter(|f| !f.private && !f.emit_skip).collect();
    let mut b = String::new();
    b.push_str(&format!("class {}:\n", t.name));
    let had_doc = python::docstring(&mut b, &d.doc, 4);
    if t.constructor {
        if had_doc {
            b.push('\n');
        }
        let args: Vec<String> = readable
            .iter()
            .map(|f| match default(f) {
                Some(v) => format!("{}: {} = {v}", f.name, ty(all, f)),
                None => format!("{}: {}", f.name, ty(all, f)),
            })
            .collect();
        let flat = format!("    def __init__(self, *, {}) -> None: ...\n", args.join(", "));
        if flat.len() <= 101 || args.is_empty() {
            b.push_str(&flat);
        } else {
            b.push_str("    def __init__(\n        self,\n        *,\n");
            for a in &args {
                b.push_str(&format!("        {a},\n"));
            }
            b.push_str("    ) -> None: ...\n");
        }
    }
    for f in &readable {
        b.push_str("    @property\n");
        b.push_str(&format!("    def {}(self) -> {}: ...\n", f.name, ty(all, f)));
    }
    if !t.constructor && readable.is_empty() && !had_doc {
        b.push_str("    ...\n");
    }
    Ok(b)
}

fn ty(all: &[Decl], f: &Field) -> String {
    if let Ty::Declared(n) = &f.ty
        && let Some(d) = all.iter().find(|d| &d.rust_name == n)
    {
        return match &d.body {
            Body::Unit(_) => "str".to_owned(),
            _ => "dict[str, Any]".to_owned(),
        };
    }
    if let Ty::Option(inner) = &f.ty
        && let Ty::Declared(n) = &**inner
        && let Some(d) = all.iter().find(|d| &d.rust_name == n)
    {
        return match &d.body {
            Body::Unit(_) => "str | None".to_owned(),
            _ => "dict[str, Any] | None".to_owned(),
        };
    }
    python::render(all, &f.ty)
}

/// `#[python(default)]` verbatim; a bare `#[binding(default)]` is the type's `Default` spelled
/// per type; a `#[binding(default = "<expr>")]` is `...` (still optional to the caller).
fn default(f: &Field) -> Option<String> {
    if let Some(d) = &f.default {
        return Some(d.clone());
    }
    if f.extract_default_value.is_some() {
        return Some("...".to_owned());
    }
    if !f.extract_default {
        return None;
    }
    Some(
        match &f.ty {
            Ty::Option(_) => "None",
            Ty::Bool => "False",
            Ty::Int => "0",
            Ty::Float => "0.0",
            Ty::Str => "\"\"",
            Ty::List(_) => "[]",
            Ty::Map(..) => "{}",
            _ => "...",
        }
        .to_owned(),
    )
}


#[derive(Clone, Copy)]
struct Cx<'a> {
    decls: &'a [Decl],
    x: &'a Exposed,
}

/// Functions, then objects, then stream classes.
fn items(decls: &[Decl], x: &Exposed) -> Result<Vec<String>, String> {
    let cx = Cx { decls, x };
    let mut out = Vec::new();
    for f in &x.fns {
        if f.is_attr {
            out.push(format!("{}: {}\n", f.name, py_ty(cx, &f.sig.ret, false)?));
            continue;
        }
        out.push(def(cx, "", if f.is_async { "async " } else { "" }, &f.name, &f.sig, false)?);
    }
    for o in &x.objects {
        let mut b = format!("class {}:\n", o.name);
        let summary = summary(&o.doc);
        if python::docstring(&mut b, &summary, 4) {
            b.push('\n');
        }
        for c in &o.consts {
            b.push_str(&format!("    {}: {}\n", c.name, py_ty(cx, &c.ty, false)?));
        }
        for m in &o.methods {
            let d = match m.kind {
                MethodKind::New => def(cx, "    ", "", "__init__", &Sig { ret: RTy::Unit, ..m.sig.clone() }, true)?,
                MethodKind::Getter => format!("    @property\n{}", def(cx, "    ", "", &m.name, &m.sig, true)?),
                MethodKind::Setter => format!("    @{}.setter\n{}", m.name, def(cx, "    ", "", &m.name, &m.sig, true)?),
                MethodKind::Static => {
                    let own = RTy::Declared { name: o.core_path.rsplit("::").next().unwrap_or_default().to_owned(), by_ref: false };
                    let sig = if m.sig.ret == RTy::SelfTy { Sig { ret: own, ..m.sig.clone() } } else { m.sig.clone() };
                    format!("    @staticmethod\n{}", def(cx, "    ", "", &m.name, &sig, false)?)
                }
                MethodKind::Async => def(cx, "    ", "async ", &m.name, &m.sig, true)?,
                MethodKind::Method => match &m.reply {
                    Some(r) => def(cx, "    ", "async ", &m.name, &Sig { ret: r.ret.clone(), ..m.sig.clone() }, true)?,
                    None => def(cx, "    ", "", &m.name, &m.sig, true)?,
                },
            };
            b.push_str(&d);
        }
        for l in &o.stub_extra {
            b.push_str(&format!("    {l}\n"));
        }
        out.push(b);
    }
    for s in &x.streams {
        let mut b = format!("class {}:\n", s.name);
        if python::docstring(&mut b, &summary(&s.doc), 4) {
            b.push('\n');
        }
        if s.sink_ty.is_some() {
            b.push_str(&format!("    def {}(self, {}: {}) -> None: ...\n", s.push, s.arg, s.input_py));
            b.push_str("    def flush(self) -> None: ...\n");
            if s.end_input {
                b.push_str("    def end_input(self) -> None: ...\n");
            }
        }
        for l in s.stub_extra.iter().flat_map(|l| l.lines()) {
            b.push_str(&format!("    {l}\n"));
        }
        b.push_str("    async def recv(self) -> Any: ...\n    async def aclose(self) -> None: ...\n");
        out.push(b);
    }
    Ok(out)
}

/// In order, once each.
fn imports(x: &Exposed) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for l in x.objects.iter().flat_map(|o| &o.stub_imports).chain(x.streams.iter().flat_map(|s| &s.stub_imports)) {
        if !out.contains(l) {
            out.push(l.clone());
        }
    }
    out
}

fn def(cx: Cx, pad: &str, prefix: &str, name: &str, sig: &Sig, method: bool) -> Result<String, String> {
    let mut parts: Vec<String> = if method { vec!["self".to_owned()] } else { Vec::new() };
    for (stub, star) in args(cx, sig)? {
        if star {
            parts.push("*".to_owned());
        }
        parts.push(stub);
    }
    let ret = match &sig.stream {
        Some(class) => class.clone(),
        None => py_ty(cx, &sig.ret, false)?,
    };
    let flat = format!("{pad}{prefix}def {name}({}) -> {ret}: ...\n", parts.join(", "));
    if flat.len() <= 100 {
        return Ok(flat);
    }
    let mut b = format!("{pad}{prefix}def {name}(\n");
    for p in &parts {
        b.push_str(&format!("{pad}    {p},\n"));
    }
    b.push_str(&format!("{pad}) -> {ret}: ...\n"));
    Ok(b)
}

/// `(name: type [= default], star before it)`: an ambient parameter has none, a keyword record
/// one per field.
fn args(cx: Cx, sig: &Sig) -> Result<Vec<(String, bool)>, String> {
    let mut out = Vec::new();
    for (i, p) in sig.params.iter().enumerate() {
        if p.ambient.is_some() {
            continue;
        }
        let star = sig.kw_only == KwOnly::All && i == 0 || sig.kw_only == KwOnly::From(p.name.clone());
        let explicit = sig.defaults.iter().find(|(k, _)| k == &p.name).map(|(_, v)| v.clone());
        if let RTy::Declared { name, .. } = &p.ty
            && let Some(k) = cx.x.kwargs.iter().find(|k| &k.name == name)
        {
            for (j, f) in k.fields.iter().enumerate() {
                let fp = Param {
                    name: f.name.clone(),
                    ty: f.ty.clone(),
                    lenient: f.lenient,
                    label: None,
                    ambient: None,
                };
                out.push((one(cx, &fp, f.default.clone())?, star && j == 0));
            }
            continue;
        }
        out.push((one(cx, p, explicit)?, star));
    }
    Ok(out)
}

fn one(cx: Cx, p: &Param, explicit: Option<String>) -> Result<String, String> {
    let (py, implied) = match &p.ty {
        RTy::Maybe(_) => (py_ty(cx, &p.ty, true)?, Some("...".to_owned())),
        t => (py_ty(cx, t, true)?, None),
    };
    Ok(match explicit.map(|v| py_default(&v)).or(implied) {
        Some(v) => format!("{}: {py} = {v}", p.name),
        None => format!("{}: {py}", p.name),
    })
}

fn py_default(v: &str) -> String {
    match v {
        "true" => "True".to_owned(),
        "false" => "False".to_owned(),
        "None" => "None".to_owned(),
        _ if v.replace('_', "").parse::<f64>().is_ok() => v.replace('_', ""),
        _ if v.starts_with('"') => v.strip_suffix(".to_owned()").or_else(|| v.strip_suffix(".into()")).unwrap_or(v).to_owned(),
        _ => "...".to_owned(),
    }
}

fn exposed_object<'a>(cx: Cx<'a>, name: &str) -> Option<&'a Object> {
    cx.x.objects.iter().find(|o| o.core_path.rsplit("::").next() == Some(name))
}

enum DeclKind<'a> {
    Unit,
    Record(&'a Decl),
    /// Undeclared: crosses by its own `IntoPyObject`/`FromPyObject`; the stub says `Any`.
    Opaque,
}

fn declared<'a>(cx: Cx<'a>, name: &str) -> DeclKind<'a> {
    if cx.x.wire_enums.iter().any(|w| w.name == name) {
        return DeclKind::Unit;
    }
    // Two declarations may share a Rust name; the one with a binding twin crosses.
    let named = || cx.decls.iter().filter(|d| d.rust_name == name);
    match named().find(|d| d.targets_for(Host::Binding).next().is_some()).or_else(|| named().next()) {
        None => DeclKind::Opaque,
        Some(d) => match d.body {
            Body::Unit(_) => DeclKind::Unit,
            _ => DeclKind::Record(d),
        },
    }
}

/// A `#[binding(class)]` record with no dict twin of `kind` crosses as that class.
fn class_twin(d: &Decl, kind: Kind) -> bool {
    !d.targets_for(Host::Binding).any(|t| t.kind == kind) && d.targets_for(Host::Binding).any(|t| t.kind == Kind::Class)
}

/// A side-table key (`MediaRef`, `HostRef`): the host object itself.
fn host_key(t: &RTy) -> bool {
    matches!(t, RTy::Declared { name, .. } if name == "MediaRef") || matches!(t, RTy::HostRef)
}

fn py_ty(cx: Cx, t: &RTy, arg: bool) -> Result<String, String> {
    Ok(match t {
        RTy::Unit | RTy::SelfTy => "None".to_owned(),
        RTy::Prim(p) if p == "bool" => "bool".to_owned(),
        RTy::Prim(p) if p.starts_with('f') => "float".to_owned(),
        RTy::Prim(p) if p == "char" => "str".to_owned(),
        RTy::Prim(_) => "int".to_owned(),
        RTy::Str | RTy::StrRef | RTy::IntoString => "str".to_owned(),
        RTy::Bytes => "bytes".to_owned(),
        RTy::Vec(i) if !arg && matches!(&**i, RTy::Prim(p) if p == "u8") => "bytes".to_owned(),
        RTy::Option(i) => format!("{} | None", py_ty(cx, i, arg)?),
        RTy::Vec(i) | RTy::Slice(i) | RTy::Iter(i) | RTy::IntoIter(i) => format!("list[{}]", py_ty(cx, i, arg)?),
        RTy::Tuple(items) => {
            format!("tuple[{}]", items.iter().map(|i| py_ty(cx, i, arg)).collect::<Result<Vec<_>, _>>()?.join(", "))
        }
        RTy::Json { .. } => "Any".to_owned(),
        RTy::JsonMap { .. } => "dict[str, Any]".to_owned(),
        RTy::Map(v) => format!("dict[str, {}]", py_ty(cx, v, arg)?),
        RTy::Duration | RTy::SystemTime => "float".to_owned(),
        RTy::Dyn(_) | RTy::HostRef | RTy::StreamHalf { .. } => "Any".to_owned(),
        RTy::Maybe(i) => format!("NotGivenOr[{}]", py_ty(cx, i, arg)?),
        RTy::Shared(name) => exposed_object(cx, name).map(|o| o.name.clone()).unwrap_or_else(|| "Any".to_owned()),
        RTy::Declared { .. } if host_key(t) => "Any".to_owned(),
        RTy::Declared { name, .. } => {
            if let Some(k) = cx.x.kwargs.iter().find(|k| &k.name == name) {
                return Err(format!("`{}` is a keyword record; it is spread, never typed", k.name));
            }
            if let Some(o) = exposed_object(cx, name) {
                return Ok(o.name.clone());
            }
            match declared(cx, name) {
                DeclKind::Unit => "str".to_owned(),
                DeclKind::Record(d) if class_twin(d, if arg { Kind::Extract } else { Kind::Emit }) => d
                    .targets_for(Host::Binding)
                    .find(|t| t.kind == Kind::Class)
                    .map(|t| t.name.clone())
                    .expect("a class twin"),
                DeclKind::Record(_) if arg => "Any".to_owned(),
                DeclKind::Record(_) => "dict[str, Any]".to_owned(),
                DeclKind::Opaque => "Any".to_owned(),
            }
        }
    })
}

fn summary(doc: &[String]) -> Vec<String> {
    let end = doc.iter().position(|l| l.is_empty()).unwrap_or(doc.len());
    doc[..end].iter().map(|l| unlink(l)).collect()
}

/// `` [`x`] `` -> `` `x` ``; a `[text](url)` link is kept.
fn unlink(line: &str) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(i) = rest.find("[`") {
        let Some(j) = rest[i..].find("`]") else { break };
        let after = &rest[i + j + 2..];
        out.push_str(&rest[..i]);
        if after.starts_with('(') || after.starts_with('[') {
            out.push_str(&rest[i..i + j + 2]);
        } else {
            out.push_str(&rest[i + 1..i + j + 1]);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}
