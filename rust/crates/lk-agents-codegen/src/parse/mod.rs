//! Source in, [`Decl`]s out: `syn::parse_file` on the source text, so output keeps source order
//! and `///` docs. `// ---` banners are not tokens and are read from the raw text ([`sections`]).
//! Every failure is a hard error naming the declaration: skipping would silently drop a class.

mod exposed;
mod meta;
mod signature;

use crate::ir::*;
use lk_agents_bindgen::doc;
use syn::spanned::Spanned as _;

#[cfg(test)]
pub use exposed::exposed;
pub use exposed::{aliases, exposed_with, global_aliases};
use meta::{flag, has_host_derive, head_of, host_attr_strict, meta_str, metas, rust_arg, str_arg, str_args, unprefixed};

pub type Res<T> = Result<T, String>;

/// `label` (the repo-relative path) goes into the generated header and every error.
pub fn file(label: &str, module: &str, text: &str) -> Res<Vec<Decl>> {
    let parsed = syn::parse_file(text).map_err(|e| format!("{label}: {e}"))?;
    let marks = sections(text);
    let consts = consts(&parsed);
    let mut out = Vec::new();
    let file = file_defaults(label, &parsed.items)?;
    items(label, &marks, &consts, &parsed.items, None, file.as_ref(), &mut out)?;
    for d in &mut out {
        let inner: String = d.rust_module.iter().map(|m| format!("{m}::")).collect();
        d.core_path = format!("{module}::{inner}{}", d.rust_name);
    }
    link(&mut out)?;
    Ok(out)
}

/// An inline `mod` with `#[crate::host(group(..))]` contributes its own, with the group's defaults;
/// `file` ([`file_defaults`]) fills a record's `module`/`path` where no group does.
fn items(
    label: &str,
    marks: &[(usize, String)],
    consts: &[(String, String)],
    items: &[syn::Item],
    group: Option<&Group>,
    file: Option<&Group>,
    out: &mut Vec<Decl>,
) -> Res<()> {
    for item in items {
        let at = |span: proc_macro2::Span| section_at(marks, span.start().line);
        match item {
            // Inside a group the group's attribute is the marker for every struct and enum.
            syn::Item::Struct(s) if group.is_some() || has_host_derive(&s.attrs) => {
                out.push(strukt(label, at(s.span()), s, consts, group.or(file))?)
            }
            syn::Item::Enum(e) if group.is_some() || has_host_derive(&e.attrs) => {
                out.push(enom(label, at(e.span()), e, group.or(file))?)
            }
            syn::Item::Const(c) if group.is_some_and(|g| !g.module.is_empty()) => {
                if let Some(d) = literal(label, at(c.span()), c, group.expect("checked"))? {
                    out.push(d);
                }
            }
            syn::Item::Mod(m) => {
                let Some(mut g) = group_of(label, &m.attrs, group)? else { continue };
                g.rust_module = group.map(|p| p.rust_module.clone()).unwrap_or_default();
                g.rust_module.push(m.ident.to_string());
                let Some((_, inner)) = &m.content else {
                    return Err(format!("{label}: a host group must be an inline `mod {{ .. }}`"));
                };
                self::items(label, marks, consts, inner, Some(&g), file, out)?;
            }
            _ => {}
        }
    }
    Ok(())
}


/// `#[crate::host(group(path, kind, typed_dict_from, total, node))]` on an inline `mod`: shared
/// defaults. `path` is the public module; the generated one is it under `livekit.agents.` with
/// `.` -> `_`. `node` gives every Python target a same-named Node twin.
#[derive(Debug, Clone, Default)]
pub struct Group {
    pub module: String,
    pub path: Option<String>,
    pub kind: Option<Kind>,
    pub typed_dict_from: Option<String>,
    pub total: Option<bool>,
    pub node: bool,
    /// Outermost first.
    pub rust_module: Vec<String>,
}

/// A nested group inherits every default it does not set; a `path` of its own also names its module.
fn group_of(label: &str, attrs: &[syn::Attribute], parent: Option<&Group>) -> Res<Option<Group>> {
    for a in attrs {
        // By path: a bare `host` is ambiguous with the `#[derive(Host)]` helper.
        if !a.path().segments.last().is_some_and(|s| s.ident == "host") {
            continue;
        }
        let out = metas(std::slice::from_ref(a));
        if !out.iter().any(|m| m.key.starts_with("group.")) {
            continue;
        }
        let get = |k: &str| out.iter().find(|m| m.key == format!("group.{k}")).and_then(|m| m.value.clone());
        let own_path = get("path");
        let module = get("module")
            .or_else(|| {
                own_path
                    .as_deref()
                    .and_then(|p| p.strip_prefix("livekit.agents."))
                    .map(|p| p.replace('.', "_"))
            })
            .or_else(|| parent.map(|p| p.module.clone()).filter(|m| !m.is_empty()))
            .unwrap_or_default();
        let path = own_path.or_else(|| parent.and_then(|p| p.path.clone()));
        let kind = match get("kind") {
            None => parent.and_then(|p| p.kind),
            Some(k) => Some(
                KIND_WORDS
                    .iter()
                    .find(|(w, _)| *w == k)
                    .map(|(_, kind)| *kind)
                    .ok_or_else(|| format!("{label}: group kind = \"{k}\" is not a kind word"))?,
            ),
        };
        let total = match get("total").as_deref() {
            None => parent.and_then(|p| p.total),
            Some("True") => Some(true),
            Some("False") => Some(false),
            Some(other) => return Err(format!("{label}: group total = \"{other}\"; write \"True\" or \"False\"")),
        };
        return Ok(Some(Group {
            module,
            path,
            kind,
            typed_dict_from: get("typed_dict_from").or_else(|| parent.and_then(|p| p.typed_dict_from.clone())),
            total,
            node: out.iter().any(|m| m.key == "group.node" && m.value.is_none()) || parent.is_some_and(|p| p.node),
            rust_module: Vec::new(),
        }));
    }
    Ok(None)
}

/// `#[crate::host(file(module = "..", path = ".."))] const _: () = ();`: the Python `module` and
/// `path` of every record in the file that does not name its own. Unlike a group it declares
/// nothing: a record still needs its own `#[derive(Host)]` and `#[python(<kind>)]`.
fn file_defaults(label: &str, items: &[syn::Item]) -> Res<Option<Group>> {
    let mut found = None;
    for item in items {
        let syn::Item::Const(c) = item else { continue };
        let Some(m) = host_attr_strict(label, &c.attrs)? else { continue };
        if head_of(&m) != "file" {
            continue;
        }
        if found.is_some() {
            return Err(format!("{label}: two #[host(file(..))] defaults"));
        }
        if let Some(k) = m.iter().find(|x| !["file.module", "file.path"].contains(&x.key.as_str())) {
            return Err(format!("{label}: #[host(file(..))] takes module and path, not `{}`", k.key));
        }
        let module = meta_str(&m, "file.module").ok_or_else(|| format!("{label}: #[host(file(..))] needs module = \"..\""))?;
        found = Some(Group { module, path: meta_str(&m, "file.path"), ..Group::default() });
    }
    Ok(found)
}

fn bare_target(host: Host, kind: Kind, module: &str, name: &str, path: Option<String>) -> Target {
    Target {
        host,
        kind,
        module: module.to_owned(),
        name: name.to_owned(),
        path,
        total: true,
        config: None,
        validator: None,
        tag: None,
        host_fields: Vec::new(),
        tag_after: None,
        base: None,
        nested_in: None,
        post_init: None,
        repr: None,
        discriminator: None,
        members: Vec::new(),
        inline: false,
        into_type: None,
        constructor: false,
        order: Vec::new(),
        typed_dict_from: None,
    }
}

/// `pub const DEEPGRAM_MODELS: &[&str] = &[..];` in a group: `DeepgramModels = Literal[..]`
/// (or `#[host(name = "..")]`). With `#[host(union = "A, B")]`: `A | B | Literal[..]`.
fn literal(label: &str, section: Option<String>, c: &syn::ItemConst, g: &Group) -> Res<Option<Decl>> {
    let meta = metas(&c.attrs);
    let owner = c.ident.to_string();
    let bad = || format!("{label}: {owner} must be a `&[&str]` of string literals");
    // Any other const in a group is the module's own.
    let is_str_slice = matches!(&*c.ty, syn::Type::Reference(r) if matches!(&*r.elem,
        syn::Type::Slice(sl) if quote(&sl.elem).replace(' ', "") == "&str"));
    if !is_str_slice {
        return Ok(None);
    }
    let name = meta.iter().find(|m| m.key == "name").and_then(|m| m.value.clone()).unwrap_or_else(|| {
        owner
            .split('_')
            .filter(|w| !w.is_empty())
            .map(|w| {
                let lower = w.to_lowercase();
                let mut cs = lower.chars();
                cs.next().map(|c| c.to_uppercase().chain(cs).collect::<String>()).unwrap_or_default()
            })
            .collect()
    });
    let syn::Expr::Reference(r) = &*c.expr else { return Err(bad()) };
    let syn::Expr::Array(arr) = &*r.expr else { return Err(bad()) };
    let mut values = Vec::new();
    for e in &arr.elems {
        let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(v), .. }) = e else { return Err(bad()) };
        values.push(v.value());
    }
    let key = |n: &str| format!("py:{}:{n}", g.module);
    let (body, kind, node_kind) =
        match meta.iter().find(|m| m.key == "union").and_then(|m| m.value.clone()) {
            None => (
                Body::Unit(values.into_iter().map(|value| UnitVariant { value, doc: Vec::new() }).collect()),
                Kind::Literal,
                Kind::Enum,
            ),
            Some(members) => {
                let mut out = Vec::new();
                for m in members.split(',').map(str::trim).filter(|m| !m.is_empty()) {
                    let ty = match m {
                        "str" | "int" | "float" | "bool" => foreign(m, ts_builtin(m)),
                        _ => Ty::Declared(key(m)),
                    };
                    out.push(Variant { rust_name: m.to_owned(), ty, twin: None, dispatch: None, kind: None });
                }
                for v in values {
                    let ty = foreign(&format!("Literal[\"{v}\"]"), &format!("'{v}'"));
                    out.push(Variant { rust_name: v, ty, twin: None, dispatch: None, kind: None });
                }
                (Body::Union(out), Kind::Union, Kind::Union)
            }
        };
    let mut targets = vec![bare_target(Host::Python, kind, &g.module, &name, None)];
    if g.node {
        targets.push(bare_target(Host::Node, node_kind, &g.module, &name, None));
    }
    Ok(Some(Decl {
        rust_name: key(&name),
        source: label.to_owned(),
        doc: doc(&c.attrs),
        section,
        rust_module: g.rust_module.clone(),
        body,
        targets,
        dispatch: None,
        dispatch_also: Vec::new(),
        core_path: String::new(),
    }))
}

fn ts_builtin(py: &str) -> &'static str {
    match py {
        "str" => "string",
        "bool" => "boolean",
        _ => "number",
    }
}

fn foreign(python: &str, node: &str) -> Ty {
    Ty::Foreign(Foreign {
        python: python.to_owned(),
        rust: None,
        python_from: Vec::new(),
        python_type_from: Vec::new(),
        node: Some(node.to_owned()),
        node_from: None,
    })
}

/// Resolve `py:<module>:<Name>` union members to declarations. Run per file, then over all files.
pub fn link(decls: &mut [Decl]) -> Res<()> {
    let names: Vec<(String, String)> = decls
        .iter()
        .filter_map(|d| {
            let t = d.target(Host::Python)?;
            Some((format!("py:{}:{}", t.module, t.name), d.rust_name.clone()))
        })
        .collect();
    for d in decls.iter_mut() {
        let Body::Union(members) = &mut d.body else { continue };
        for m in members {
            let Ty::Declared(n) = &mut m.ty else { continue };
            if !n.starts_with("py:") {
                continue;
            }
            if let Some((_, rust)) = names.iter().find(|(k, _)| k == n) {
                *n = rust.clone();
            }
        }
    }
    Ok(())
}

/// Every `impl T { const NAME: &str = "value" }` as `("T::NAME", "value")`, for
/// `literal_const`. Strings only: anything else should be an error, not an odd literal.
fn consts(file: &syn::File) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for item in &file.items {
        let syn::Item::Impl(imp) = item else { continue };
        let syn::Type::Path(p) = &*imp.self_ty else { continue };
        let Some(owner) = p.path.segments.last().map(|s| s.ident.to_string()) else { continue };
        for inner in &imp.items {
            let syn::ImplItem::Const(c) = inner else { continue };
            let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(v), .. }) = &c.expr else {
                continue;
            };
            out.push((format!("{owner}::{}", c.ident), v.value()));
        }
    }
    out
}


/// `// --- <title> ---` banners as `(line, title)`, from the raw text: the lexer drops comments.
fn sections(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("// ---") else { continue };
        let title = rest.trim().trim_end_matches('-').trim();
        if !title.is_empty() {
            out.push((i + 1, title.to_owned()));
        }
    }
    out
}

/// The last banner at or above `line`.
fn section_at(marks: &[(usize, String)], line: usize) -> Option<String> {
    marks.iter().rev().find(|(at, _)| *at <= line).map(|(_, t)| t.clone())
}


fn strukt(
    label: &str,
    section: Option<String>,
    s: &syn::ItemStruct,
    consts: &[(String, String)],
    group: Option<&Group>,
) -> Res<Decl> {
    let rust_name = s.ident.to_string();
    let named = match &s.fields {
        syn::Fields::Named(n) => n,
        _ => {
            return Err(format!(
                "{label}: {rust_name} is a tuple struct; #[derive(Host)] needs named fields"
            ));
        }
    };
    let inherit = lk_agents_bindgen::RecordDefaults::of(&s.attrs).map_err(|e| format!("{label}: {rust_name}: {e}"))?;
    let mut fields = Vec::new();
    for f in &named.named {
        fields.push(field(label, &rust_name, f, inherit)?);
    }
    Ok(Decl {
        targets: targets(label, &rust_name, &s.attrs, consts, group)?,
        rust_name,
        source: label.to_owned(),
        doc: doc(&s.attrs),
        section,
        rust_module: group.map(|g| g.rust_module.clone()).unwrap_or_default(),
        body: Body::Struct(fields),
        dispatch: None,
        dispatch_also: Vec::new(),
        core_path: String::new(),
    })
}

fn enom(label: &str, section: Option<String>, e: &syn::ItemEnum, group: Option<&Group>) -> Res<Decl> {
    let rust_name = e.ident.to_string();
    let all_unit = e.variants.iter().all(|v| matches!(v.fields, syn::Fields::Unit));
    let body = if all_unit {
        let mut vs = Vec::new();
        for v in &e.variants {
            vs.push(unit_variant(label, &rust_name, v)?);
        }
        Body::Unit(vs)
    } else {
        // Each variant carries exactly one type, the member. Multi-field variants are rejected:
        // an invented name would drift.
        let mut members = Vec::new();
        for v in e.variants.iter().filter(|v| !flag(&v.attrs, "binding", "skip")) {
            let syn::Fields::Unnamed(u) = &v.fields else {
                return Err(format!(
                    "{label}: {rust_name}::{} must be a one-type variant to be a host union",
                    v.ident
                ));
            };
            if u.unnamed.len() != 1 {
                return Err(format!(
                    "{label}: {rust_name}::{} carries {} types; a host union variant carries one",
                    v.ident,
                    u.unnamed.len()
                ));
            }
            let t = ty(label, &rust_name, &u.unnamed[0].ty, &u.unnamed[0].attrs)?;
            if !matches!(t, Ty::Declared(_) | Ty::Foreign(_)) {
                return Err(format!(
                    "{label}: {rust_name}::{} is {t:?}; a union member must be a declared or \
                     foreign type",
                    v.ident
                ));
            }
            let dispatch = match (flag(&v.attrs, "python", "event"), flag(&v.attrs, "python", "ignored")) {
                (true, true) => {
                    return Err(format!("{label}: {rust_name}::{} is both an event and ignored", v.ident));
                }
                (true, false) => Some(Dispatch::Event),
                (false, true) => Some(Dispatch::Ignored),
                (false, false) => None,
            };
            members.push(Variant {
                rust_name: v.ident.to_string(),
                ty: t,
                twin: rust_arg(&v.attrs, "binding", "twin"),
                kind: str_arg(&v.attrs, "python", "kind")?,
                dispatch,
            });
        }
        Body::Union(members)
    };
    let dispatch = str_arg(&e.attrs, "python", "dispatch")?;
    let dispatch_also = str_args(&e.attrs, "python", "dispatch_also")?;
    if !dispatch_also.is_empty() && dispatch.is_none() {
        return Err(format!("{label}: {rust_name} names #[python(dispatch_also)] but no #[python(dispatch = \"..\")] module"));
    }
    let routed = matches!(&body, Body::Union(ms) if ms.iter().any(|m| m.dispatch.is_some()));
    if routed && dispatch.is_none() {
        return Err(format!("{label}: {rust_name} routes members (`#[python(event|ignored)]`) but names no #[python(dispatch = \"..\")] module"));
    }
    Ok(Decl {
        targets: targets(label, &rust_name, &e.attrs, &[], group)?,
        rust_name,
        source: label.to_owned(),
        doc: doc(&e.attrs),
        section,
        rust_module: group.map(|g| g.rust_module.clone()).unwrap_or_default(),
        body,
        dispatch,
        dispatch_also,
        core_path: String::new(),
    })
}

fn unit_variant(label: &str, owner: &str, v: &syn::Variant) -> Res<UnitVariant> {
    let rust_name = v.ident.to_string();
    // No default: the reference's enums follow no casing rule, and a guessed value breaks `==`.
    let value = unprefixed(&v.attrs, "rename").ok_or_else(|| {
        format!(
            "{label}: {owner}::{rust_name} needs #[host(rename = \"..\")] -- the wire value is \
                 user-visible and must not be guessed"
        )
    })?;
    Ok(UnitVariant { value, doc: doc(&v.attrs) })
}

fn field(label: &str, owner: &str, f: &syn::Field, inherit: lk_agents_bindgen::RecordDefaults) -> Res<Field> {
    let rust_name = f.ident.as_ref().expect("named fields only").to_string();
    let name = unprefixed(&f.attrs, "rename").unwrap_or_else(|| rust_name.clone());
    let literal = str_arg(&f.attrs, "python", "literal")?;
    let ty = ty(label, owner, &f.ty, &f.attrs)?;
    let py_obj = flag(&f.attrs, "binding", "py");
    // A `()` field must either discriminate (`literal`) or be host-valued (`#[binding(py)]`).
    if matches!(ty, Ty::Unit) && literal.is_none() && !py_obj {
        return Err(format!(
            "{label}: {owner}.{rust_name} is `()` with no #[python(literal = \"..\")] and no \
             #[binding(py)] -- a field that carries nothing must at least say what it is for"
        ));
    }
    let rust_ty = quote(&f.ty).replace(' ', "");
    let default = str_arg(&f.attrs, "python", "default")?.or_else(|| zero_default(inherit, &ty, &f.attrs, literal.is_some()));
    Ok(Field {
        rust_ty,
        rust_name,
        name,
        doc: doc(&f.attrs),
        ty,
        literal,
        default,
        default_factory: str_arg(&f.attrs, "python", "default_factory")?,
        serializer: str_arg(&f.attrs, "python", "serializer")?,
        exclude: flag(&f.attrs, "python", "exclude"),
        py_skip: flag(&f.attrs, "python", "skip"),
        private: flag(&f.attrs, "python", "private"),
        deprecated_msg: str_arg(&f.attrs, "python", "deprecated_msg")?,
        imports: str_args(&f.attrs, "python", "import")?,
        type_imports: str_args(&f.attrs, "python", "type_import")?,
        deprecated: flag(&f.attrs, "python", "deprecated"),
        emit_skip: flag(&f.attrs, "binding", "skip"),
        extract_default: flag(&f.attrs, "binding", "default") || inherit.default,
        // `flag` matches only the bare word, `str_arg`/`rust_arg` only the valued form.
        extract_default_value: rust_arg(&f.attrs, "binding", "default"),
    })
}


/// `#[python(zero_defaults)]` ([`lk_agents_bindgen::RecordDefaults`]): a number or `Option` field
/// with no default of its own defaults to its zero, unless `#[python(required)]`.
fn zero_default(inherit: lk_agents_bindgen::RecordDefaults, ty: &Ty, attrs: &[syn::Attribute], literal: bool) -> Option<String> {
    let factory = str_arg(attrs, "python", "default_factory").ok().flatten();
    if !inherit.zero || literal || factory.is_some() || flag(attrs, "python", "required") {
        return None;
    }
    Some(match ty {
        Ty::Int => "0",
        Ty::Float => "0.0",
        Ty::Option(_) => "None",
        _ => return None,
    }.to_owned())
}

fn ty(label: &str, owner: &str, t: &syn::Type, attrs: &[syn::Attribute]) -> Res<Ty> {
    // The escape hatch wins for the Python annotation.
    if let Some(python) = str_arg(attrs, "python", "ty")? {
        // The Rust type is kept too: the binding needs it (`Foreign::rust`).
        let rust = rust_ty(label, owner, t).ok().map(Box::new);
        return Ok(Ty::Foreign(Foreign {
            python,
            rust,
            python_from: str_args(attrs, "python", "import")?,
            python_type_from: str_args(attrs, "python", "type_import")?,
            node: str_arg(attrs, "node", "ty")?,
            node_from: str_arg(attrs, "node", "from")?,
        }));
    }
    rust_ty(label, owner, t)
}

fn rust_ty(label: &str, owner: &str, t: &syn::Type) -> Res<Ty> {
    match t {
        syn::Type::Tuple(tu) if tu.elems.is_empty() => Ok(Ty::Unit),
        syn::Type::Tuple(tu) => {
            let mut v = Vec::new();
            for e in &tu.elems {
                v.push(ty(label, owner, e, &[])?);
            }
            Ok(Ty::Tuple(v))
        }
        syn::Type::Path(p) => path_ty(label, owner, p),
        other => Err(format!("{label}: {owner} has an unsupported field type {:?}", quote(other))),
    }
}

fn path_ty(label: &str, owner: &str, p: &syn::TypePath) -> Res<Ty> {
    let seg = p.path.segments.last().expect("a path has a last segment");
    let head = seg.ident.to_string();
    let args: Vec<&syn::Type> = match &seg.arguments {
        syn::PathArguments::AngleBracketed(a) => a
            .args
            .iter()
            .filter_map(|a| match a {
                syn::GenericArgument::Type(t) => Some(t),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let arg = |i: usize| -> Res<Ty> {
        let t = args
            .get(i)
            .ok_or_else(|| format!("{label}: {owner} uses {head} with too few type arguments"))?;
        ty(label, owner, t, &[])
    };
    Ok(match head.as_str() {
        "String" | "str" => Ty::Str,
        "bool" => Ty::Bool,
        "f32" | "f64" => Ty::Float,
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => Ty::Int,
        "Value" => Ty::Any,
        "Option" => Ty::Option(Box::new(arg(0)?)),
        "Vec" | "VecDeque" => Ty::List(Box::new(arg(0)?)),
        "HashMap" | "BTreeMap" => Ty::Map(Box::new(arg(0)?), Box::new(arg(1)?)),
        // Another declaration; unknown names are an error in `crate::resolve`.
        _ if head.chars().next().is_some_and(char::is_uppercase) => Ty::Declared(head),
        _ => return Err(format!("{label}: {owner} has an unmappable field type `{head}`")),
    })
}

fn quote(t: &syn::Type) -> String {
    use quote::ToTokens as _;
    t.to_token_stream().to_string()
}

/// `#[python(total = "False")]`. A string, not `false`: [`walk`] parses only string literals and
/// `metas` drops the error, so `false` would silently drop every later key.
fn total(
    label: &str,
    rust_name: &str,
    attrs: &[syn::Attribute],
    outer: &str,
    kind: Kind,
) -> Res<bool> {
    let Some(raw) = str_arg(attrs, outer, "total")? else { return Ok(true) };
    if kind != Kind::TypedDict {
        return Err(format!(
            "{label}: {rust_name}'s {outer} target is `{}`, which has no `total=`; \
             `total` belongs on `typed_dict`",
            kind_word(kind)
        ));
    }
    match raw.as_str() {
        "True" => Ok(true),
        "False" => Ok(false),
        other => Err(format!(
            "{label}: {rust_name}'s {outer} total = \"{other}\" is not a Python bool; \
             write \"True\" or \"False\""
        )),
    }
}


fn targets(
    label: &str,
    rust_name: &str,
    attrs: &[syn::Attribute],
    consts: &[(String, String)],
    group: Option<&Group>,
) -> Res<Vec<Target>> {
    let mut out = Vec::new();
    for (host, outer) in
        [(Host::Python, "python"), (Host::Node, "node"), (Host::Binding, "binding")]
    {
        let mut kinds = kinds_of(attrs, outer);
        if let (Host::Python, true, Some(k)) = (host, kinds.is_empty(), group.and_then(|g| g.kind)) {
            kinds.push(k);
        }
        if kinds.is_empty() {
            continue;
        }
        // Group defaults apply to the Python target; Node inherits Python's module and name.
        let grouped = group.filter(|_| host == Host::Python);
        let python = out.iter().find(|t: &&Target| t.host == Host::Python).cloned();
        let module = match (str_arg(attrs, outer, "module")?, grouped) {
            (Some(m), _) => m,
            (None, Some(g)) if !g.module.is_empty() => g.module.clone(),
            (None, _) if host == Host::Binding => String::new(),
            (None, _) if host == Host::Node && python.is_some() => python.as_ref().expect("checked").module.clone(),
            (None, _) => {
                return Err(format!("{label}: {rust_name}'s {outer} target needs module = \"..\""))
            }
        };
        // Binding targets use the Rust identifier; Node uses the Python name unless it says its own.
        let name = match (str_arg(attrs, outer, "name")?, &python) {
            (Some(n), _) => n,
            (None, Some(py)) if host == Host::Node => py.name.clone(),
            (None, _) => rust_name.to_owned(),
        };
        for kind in kinds {
            out.push(Target {
                host,
                kind,
                module: module.clone(),
                name: name.clone(),
                constructor: flag(attrs, outer, "new"),
                tag: tag(label, rust_name, attrs, outer, consts)?,
                tag_after: str_arg(attrs, outer, "tag_after")?,
                host_fields: host_fields(attrs, outer),
                path: str_arg(attrs, outer, "path")?.or_else(|| grouped.and_then(|g| g.path.clone())),
                base: str_arg(attrs, outer, "base")?,
                nested_in: str_arg(attrs, outer, "nested_in")?,
                post_init: str_arg(attrs, outer, "post_init")?,
                repr: str_arg(attrs, outer, "repr")?,
                discriminator: str_arg(attrs, outer, "discriminator")?,
                total: match (str_arg(attrs, outer, "total")?, grouped.and_then(|g| g.total)) {
                    (None, Some(t)) if kind == Kind::TypedDict => t,
                    _ => total(label, rust_name, attrs, outer, kind)?,
                },
                config: str_arg(attrs, outer, "config")?,
                validator: str_arg(attrs, outer, "validator")?,
                inline: flag(attrs, outer, "inline"),
                // `into` (extract) or `from` (emit): the same field, the core type on the other
                // side; `from` because `#[binding(emit, into = "..")]` reads backwards.
                into_type: rust_arg(attrs, outer, "from"),
                members: members(attrs, outer, rust_name),
                // Read for every kind; `binding.rs` rejects it where it means nothing.
                order: str_arg(attrs, outer, "order")?
                    .map(|o| {
                        o.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect()
                    })
                    .unwrap_or_default(),
                typed_dict_from: str_arg(attrs, outer, "typed_dict_from")?
                    .or_else(|| grouped.and_then(|g| g.typed_dict_from.clone())),
            });
        }
    }
    if group.is_some_and(|g| g.node) && kinds_of(attrs, "node").is_empty() {
        let twins: Vec<Target> = out
            .iter()
            .filter(|t| t.host == Host::Python)
            .map(|t| {
                let kind = match t.kind {
                    Kind::Literal | Kind::Enum => Kind::Enum,
                    Kind::Union => Kind::Union,
                    _ => Kind::Interface,
                };
                bare_target(Host::Node, kind, &t.module, &t.name, None)
            })
            .collect();
        out.extend(twins);
    }
    // No target: read for its fields alone (flattened into another, or crossed by hand).
    Ok(out)
}

/// Every `#[<outer>(host_field(..))]`, in order; a second `name` starts the next one.
fn host_fields(attrs: &[syn::Attribute], outer: &str) -> Vec<HostField> {
    let mut out: Vec<HostField> = Vec::new();
    let mut current: Option<Vec<(String, String)>> = None;
    let finish = |cur: Option<Vec<(String, String)>>, out: &mut Vec<HostField>| {
        let Some(kvs) = cur else { return };
        let get = |k: &str| kvs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        let (Some(name), Some(ty)) = (get("name"), get("ty")) else { return };
        out.push(HostField {
            name,
            ty,
            default: get("default"),
            deprecated_msg: get("deprecated_msg"),
            default_factory: get("default_factory"),
            private: get("private").is_some(),
            doc: get("doc").unwrap_or_default(),
        });
    };
    for m in metas(attrs) {
        let Some(rest) = m.key.strip_prefix(&format!("{outer}.host_field.")) else { continue };
        if rest == "name" {
            finish(current.take(), &mut out);
            current = Some(Vec::new());
        }
        if let (Some(kvs), Some(value)) = (current.as_mut(), m.value) {
            kvs.push((rest.to_owned(), value));
        }
    }
    finish(current, &mut out);
    out
}

/// Every `#[<outer>(property(..))]` / `#[<outer>(method(..))]`, in order; a second `name` starts
/// the next member.
fn members(attrs: &[syn::Attribute], outer: &str, owner: &str) -> Vec<Member> {
    let flat = metas(attrs);
    let mut out: Vec<Member> = Vec::new();
    let mut current: Option<(bool, Vec<(String, String)>)> = None;
    let finish = |cur: Option<(bool, Vec<(String, String)>)>, out: &mut Vec<Member>| {
        let Some((is_property, kvs)) = cur else { return };
        let get = |k: &str| kvs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        let Some(name) = get("name") else { return };
        out.push(Member {
            hook: get("hook").unwrap_or_else(|| format!("{}_{name}", snake(owner))),
            is_property,
            args: get("args").unwrap_or_default(),
            ret: get("ret").unwrap_or_else(|| "None".to_owned()),
            doc: get("doc").unwrap_or_default(),
            // Explicit: an implicit setter would give every property a writable alias.
            setter: get("setter"),
            name,
        });
    };
    for m in flat {
        let Some(rest) = m.key.strip_prefix(&format!("{outer}.")) else { continue };
        let (kind, key) = match rest.split_once('.') {
            Some(("property", k)) => (Some(true), k),
            Some(("method", k)) => (Some(false), k),
            _ => continue,
        };
        let is_property = kind.expect("matched above");
        if key == "name" {
            finish(current.take(), &mut out);
            current = Some((is_property, Vec::new()));
        }
        if let (Some((_, kvs)), Some(value)) = (current.as_mut(), m.value) {
            kvs.push((key.to_owned(), value));
        }
    }
    finish(current.take(), &mut out);
    out
}

/// `FunctionToolsExecuted` -> `function_tools_executed`.
fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.char_indices() {
        if c.is_uppercase() {
            if i != 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn tag(
    label: &str,
    rust_name: &str,
    attrs: &[syn::Attribute],
    outer: &str,
    consts: &[(String, String)],
) -> Res<Option<(String, String)>> {
    let Some(name) = str_arg(attrs, outer, "tag")? else { return Ok(None) };
    if let Some(value) = str_arg(attrs, outer, "tag_value")? {
        return Ok(Some((name, value)));
    }
    let c = rust_arg(attrs, outer, "tag_const").ok_or_else(|| {
        format!("{label}: {rust_name} has tag = \"{name}\" but no tag_value or tag_const")
    })?;
    let key = format!("{rust_name}::{c}");
    let value =
        consts.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()).ok_or_else(|| {
            format!(
                "{label}: {rust_name} names tag_const `{c}`, but there is no \
             `impl {rust_name} {{ const {c}: &str = \"..\" }}` in this file"
            )
        })?;
    Ok(Some((name, value)))
}

/// Every kind word on `#[<outer>(..)]`. The binding takes two (`emit, extract`): two attributes
/// would be flattened together by `metas`, and two declarations would be two structs.
fn kinds_of(attrs: &[syn::Attribute], outer: &str) -> Vec<Kind> {
    KIND_WORDS.iter().filter(|(w, _)| flag(attrs, outer, w)).map(|(_, k)| *k).collect()
}

/// Scan order: `extract` before `emit`, and `class` last so its getters can name the dict twin.
const KIND_WORDS: [(&str, Kind); 11] = [
    ("pydantic", Kind::Pydantic),
    ("dataclass", Kind::Dataclass),
    ("typed_dict", Kind::TypedDict),
    ("enum", Kind::Enum),
    ("plain", Kind::Plain),
    ("interface", Kind::Interface),
    ("literal", Kind::Literal),
    ("union", Kind::Union),
    ("extract", Kind::Extract),
    ("emit", Kind::Emit),
    ("class", Kind::Class),
];
