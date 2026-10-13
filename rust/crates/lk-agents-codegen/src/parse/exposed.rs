//! `EXPORT_SOURCES` in, [`Exposed`] out: every `#[host(export|object|error|stream|kwargs|proxy)]`
//! item, with the file's `type` aliases read through.

use super::meta::{has_host_derive, head_of, host_attr_strict, host_attrs_strict, meta_rust, meta_str, want_head, Meta};
use super::signature::{python_gated, reply_split, rty, signature};
use super::{quote, Res};
use crate::ir::*;
use lk_agents_bindgen::doc;
use syn::spanned::Spanned as _;

/// `type X = ..;` aliases, read through by [`rty`]. A file's own win; a name two files alias
/// differently is no alias.
#[derive(Debug, Clone, Default)]
pub struct Aliases(Vec<(String, syn::Type)>);

impl Aliases {
    pub(super) fn get(&self, name: &str) -> Option<&syn::Type> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
}

pub fn aliases(text: &str) -> Vec<(String, String)> {
    let Ok(parsed) = syn::parse_file(text) else { return Vec::new() };
    parsed
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Type(t) if t.generics.params.is_empty() => {
                Some((t.ident.to_string(), quote(&t.ty)))
            }
            _ => None,
        })
        .collect()
}

pub fn global_aliases(all: &[(String, String)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut ambiguous: Vec<String> = Vec::new();
    for (n, t) in all {
        match out.iter().find(|(m, _)| m == n) {
            Some((_, u)) if u.replace(' ', "") != t.replace(' ', "") => ambiguous.push(n.clone()),
            Some(_) => {}
            None => out.push((n.clone(), t.clone())),
        }
    }
    out.retain(|(n, _)| !ambiguous.contains(n));
    out
}

/// Every `#[host(..)]` item in one `EXPORT_SOURCES` file, with only the file's own aliases.
#[cfg(test)]
pub fn exposed(label: &str, module: &str, text: &str) -> Res<Exposed> {
    exposed_with(label, module, text, &[])
}

/// Every exposed `#[host(..)]` item in one `EXPORT_SOURCES` file. `module` is written in the
/// list: a private module re-exported at the crate root has no path of its own.
pub fn exposed_with(label: &str, module: &str, text: &str, global: &[(String, String)]) -> Res<Exposed> {
    let parsed = syn::parse_file(text).map_err(|e| format!("{label}: {e}"))?;
    let mut al = Aliases::default();
    for (n, t) in aliases(text).into_iter().chain(global.iter().cloned()) {
        if al.get(&n).is_none()
            && let Ok(ty) = syn::parse_str::<syn::Type>(&t)
        {
            al.0.push((n, ty));
        }
    }
    let mut out = Exposed::default();
    for item in &parsed.items {
        match item {
            syn::Item::Fn(f) => {
                let Some(m) = host_attr_strict(label, &f.attrs)? else { continue };
                let rust = f.sig.ident.to_string();
                let at = format!("{label}: {rust}");
                want_head(&at, &m, "export")?;
                out.fns.push(export_fn(&at, &m, &f.sig, &f.attrs, format!("{module}::{rust}"), rust, &al)?);
            }
            syn::Item::Const(c) => {
                let Some(m) = host_attr_strict(label, &c.attrs)? else { continue };
                if head_of(&m) == "file" {
                    continue;
                }
                let rust = c.ident.to_string();
                let at = format!("{label}: {rust}");
                want_head(&at, &m, "export")?;
                out.fns.push(export_const(&at, &m, &c.ty, &c.attrs, format!("{module}::{rust}"), &rust, &al)?);
            }
            syn::Item::Impl(imp) => {
                let Some(m) = host_attr_strict(label, &imp.attrs)? else { continue };
                if head_of(&m) == "export" {
                    exported_impl(label, module, imp, &mut out, &al)?;
                } else {
                    out.objects.push(object(label, module, imp, &m, &al)?);
                }
            }
            // A `#[derive(Host)]` record's `#[host(..)]` is the record vocabulary, not this one.
            syn::Item::Enum(e) if has_host_derive(&e.attrs) => {}
            syn::Item::Struct(st) if has_host_derive(&st.attrs) => {}
            syn::Item::Enum(e) => {
                for m in host_attrs_strict(label, &e.attrs)? {
                    match head_of(&m).as_str() {
                        "wire_enum" => out.wire_enums.push(WireEnum {
                            name: e.ident.to_string(),
                            values: e
                                .variants
                                .iter()
                                .map(|v| {
                                    Ok(host_attr_strict(label, &v.attrs)?
                                        .and_then(|m| meta_str(&m, "rename"))
                                        .unwrap_or_else(|| lk_agents_bindgen::to_snake(&v.ident.to_string())))
                                })
                                .collect::<Res<_>>()?,
                            open: false,
                        }),
                        "string" => out.wire_enums.push(WireEnum {
                            name: e.ident.to_string(),
                            values: Vec::new(),
                            open: true,
                        }),
                        "error" => out.errors.push(error_type(label, module, &e.ident, &m, Some(&e.variants))?),
                        "stream" => out.streams.push(stream(label, module, &e.ident, &e.attrs, &m, &al, None)?),
                        // The wire strings, Rust side only: the enum crosses by its own impls, if at all.
                        "wire_strs" => {}
                        other => return Err(format!("{label}: enum {}: #[host({other})] -- an enum takes wire_enum, wire_strs, string, error or stream", e.ident)),
                    }
                }
            }
            syn::Item::Struct(st) => {
                for m in host_attrs_strict(label, &st.attrs)? {
                    match head_of(&m).as_str() {
                        "client" | "kwargs" => out.kwargs.push(kw_record(label, module, st, &m, &al)?),
                        "error" => out.errors.push(error_type(label, module, &st.ident, &m, None)?),
                        "stream" => out.streams.push(stream(label, module, &st.ident, &st.attrs, &m, &al, None)?),
                        // `Display` out, `FromStr` in; the macro writes nothing.
                        "string" => out.wire_enums.push(WireEnum {
                            name: st.ident.to_string(),
                            values: Vec::new(),
                            open: true,
                        }),
                        other => return Err(format!("{label}: struct {}: #[host({other})] -- a struct takes client, kwargs, error, stream or string", st.ident)),
                    }
                }
            }
            syn::Item::Type(t) => {
                for m in host_attrs_strict(label, &t.attrs)? {
                    want_head(&format!("{label}: type {}", t.ident), &m, "stream")?;
                    out.streams.push(stream(label, module, &t.ident, &t.attrs, &m, &al, Some(&t.ty))?);
                }
            }
            syn::Item::Trait(tr) => {
                let Some(m) = host_attr_strict(label, &tr.attrs)? else { continue };
                want_head(&format!("{label}: trait {}", tr.ident), &m, "proxy")?;
                out.proxies.push(proxy(label, module, text, &parsed, tr, &m, &al)?);
            }
            _ => {}
        }
    }
    Ok(out)
}

fn export_fn(
    at: &str,
    m: &[Meta],
    sig: &syn::Signature,
    attrs: &[syn::Attribute],
    rust_path: String,
    rust: String,
    al: &Aliases,
) -> Res<ExportFn> {
    let is_async = sig.asyncness.is_some();
    let (receiver, sig) = signature(at, m, "export", sig, al)?;
    if receiver != Receiver::None {
        return Err(format!("{at}: a free fn has no receiver"));
    }
    Ok(ExportFn {
        python_only: python_gated(attrs),
        rust_path,
        name: meta_str(m, "export.name").unwrap_or(rust),
        doc: doc(attrs),
        sig,
        is_const: false,
        is_attr: false,
        is_async,
    })
}

fn export_const(
    at: &str,
    m: &[Meta],
    ty: &syn::Type,
    attrs: &[syn::Attribute],
    rust_path: String,
    rust: &str,
    al: &Aliases,
) -> Res<ExportFn> {
    for key in ["defaults", "kw_only", "err"] {
        if m.iter().any(|x| x.key.starts_with(&format!("export.{key}"))) {
            return Err(format!("{at}: `{key}` on a const -- a const takes no arguments and cannot fail"));
        }
    }
    let ret = rty(at, ty, al)?;
    let name = meta_str(m, "export.name").unwrap_or_else(|| rust.to_lowercase());
    Ok(ExportFn {
        python_only: python_gated(attrs),
        rust_path,
        // An UPPER_CASE name is a module attribute, as the in-place expansion reads it.
        is_attr: name.chars().any(|c| c.is_ascii_uppercase()) && !name.chars().any(|c| c.is_ascii_lowercase()),
        name,
        doc: doc(attrs),
        sig: Sig {
            params: Vec::new(),
            ret,
            err: None,
            err_ty: None,
            defaults: Vec::new(),
            kw_only: KwOnly::None,
            stream: None,
        },
        is_const: true,
        is_async: false,
    })
}

/// `#[host(export)] impl T { .. }`: associated items as module functions, called through `T::`.
fn exported_impl(label: &str, module: &str, imp: &syn::ItemImpl, out: &mut Exposed, al: &Aliases) -> Res<()> {
    let syn::Type::Path(p) = &*imp.self_ty else {
        return Err(format!("{label}: #[host(export)] on an impl is for `impl Type`"));
    };
    let core = p.path.segments.last().expect("a path").ident.to_string();
    if imp.trait_.is_some() || !imp.generics.params.is_empty() {
        return Err(format!("{label}: impl {core}: #[host(export)] is for an inherent, non-generic impl"));
    }
    let mut any = false;
    for inner in &imp.items {
        match inner {
            syn::ImplItem::Const(c) => {
                let Some(m) = host_attr_strict(label, &c.attrs)? else { continue };
                let rust = c.ident.to_string();
                let at = format!("{label}: {core}::{rust}");
                want_head(&at, &m, "export")?;
                out.fns.push(export_const(&at, &m, &c.ty, &c.attrs, format!("{module}::{core}::{rust}"), &rust, al)?);
                any = true;
            }
            syn::ImplItem::Fn(f) => {
                let Some(m) = host_attr_strict(label, &f.attrs)? else { continue };
                let rust = f.sig.ident.to_string();
                let at = format!("{label}: {core}::{rust}");
                want_head(&at, &m, "export")?;
                out.fns.push(export_fn(&at, &m, &f.sig, &f.attrs, format!("{module}::{core}::{rust}"), rust, al)?);
                any = true;
            }
            _ => {}
        }
    }
    if !any {
        return Err(format!("{label}: impl {core}: #[host(export)] on an impl with no #[host(export)] item"));
    }
    Ok(())
}

fn object(label: &str, module: &str, imp: &syn::ItemImpl, m: &[Meta], al: &Aliases) -> Res<Object> {
    // A body-less `send` method is an ordinary one once its body is written, as the expansion sees it.
    let mut imp = imp.clone();
    lk_agents_bindgen::fill_sends(&mut imp).map_err(|e| format!("{label}: {e}"))?;
    let imp = &imp;
    let syn::Type::Path(p) = &*imp.self_ty else {
        return Err(format!("{label}: #[host(object)] is for `impl Type`"));
    };
    let core = p.path.segments.last().expect("a path").ident.to_string();
    let at = format!("{label}: impl {core}");
    want_head(&at, m, "object")?;
    if imp.trait_.is_some() || !imp.generics.params.is_empty() {
        return Err(format!("{at}: #[host(object)] is for an inherent, non-generic impl"));
    }
    let name = meta_str(m, "object.name").unwrap_or_else(|| core.clone());
    let cell = match meta_str(m, "object.cell").as_deref() {
        None | Some("plain") => Cell::Plain,
        Some("mutex") => Cell::Mutex,
        Some("arc") => Cell::Arc,
        // A wrapper over a value the binding cannot make a class itself: a plain class to the stub.
        Some("value" | "input") => Cell::Plain,
        Some(other) => return Err(format!("{at}: cell = \"{other}\" -- write plain, mutex, arc, value or input")),
    };
    let mut methods = Vec::new();
    let mut consts = Vec::new();
    for inner in &imp.items {
        if let syn::ImplItem::Const(c) = inner {
            let Some(fm) = host_attr_strict(label, &c.attrs)? else { continue };
            let rust = c.ident.to_string();
            let cat = format!("{at}::{rust}");
            want_head(&cat, &fm, "const")?;
            consts.push(ObjConst {
                name: meta_str(&fm, "const.name").unwrap_or_else(|| rust.clone()),
                ty: rty(&cat, &c.ty, al)?,
            });
            continue;
        }
        let syn::ImplItem::Fn(f) = inner else { continue };
        let Some(fm) = host_attr_strict(label, &f.attrs)? else { continue };
        let rust = f.sig.ident.to_string();
        let at = format!("{at}::{rust}");
        let head = head_of(&fm);
        // A method's kind comes from its signature: `async fn`, no receiver, `-> Reply<T>`.
        if !["new", "getter", "setter", "method"].contains(&head.as_str()) {
            return Err(format!("{at}: #[host({head})] -- a method marker is new, getter, setter or method"));
        }
        let receiverless = !f.sig.inputs.iter().any(|i| matches!(i, syn::FnArg::Receiver(_)));
        let (sig_in, reply) = match reply_split(&at, &f.sig, al)? {
            Some((s, r)) => (s, Some(r)),
            None => (f.sig.clone(), None),
        };
        let kind = match head.as_str() {
            "new" => MethodKind::New,
            "getter" => MethodKind::Getter,
            "setter" => MethodKind::Setter,
            _ if f.sig.asyncness.is_some() => MethodKind::Async,
            _ if receiverless => MethodKind::Static,
            _ => MethodKind::Method,
        };
        let (receiver, sig) = signature(&at, &fm, &head, &sig_in, al)?;
        if matches!(kind, MethodKind::Getter | MethodKind::Setter) && receiver == Receiver::None {
            return Err(format!("{at}: #[host({head})] needs a receiver"));
        }
        if kind == MethodKind::New && receiver != Receiver::None {
            return Err(format!("{at}: #[host(new)] takes no receiver"));
        }
        if kind == MethodKind::New && sig.ret != RTy::SelfTy {
            return Err(format!("{at}: #[host(new)] returns Self or Result<Self, E>"));
        }
        let default_name = match kind {
            MethodKind::Setter => rust.strip_prefix("set_").unwrap_or(&rust).to_owned(),
            _ => rust.clone(),
        };
        methods.push(Method {
            name: meta_str(&fm, &format!("{head}.name")).unwrap_or(default_name),
            rust_name: rust,
            kind,
            receiver,
            sig,
            reply,
        });
    }
    if methods.iter().filter(|x| x.kind == MethodKind::New).count() > 1 {
        return Err(format!("{at}: more than one #[host(new)]"));
    }
    Ok(Object {
        core_path: format!("{module}::{core}"),
        name,
        cell,
        doc: doc(&imp.attrs),
        methods,
        stub_extra: m.iter().filter(|x| x.key == "object.stub").filter_map(|x| x.value.clone()).collect(),
        stub_imports: m.iter().filter(|x| x.key == "object.stub_import").filter_map(|x| x.value.clone()).collect(),
        consts,
    })
}

fn error_type(
    label: &str,
    module: &str,
    ident: &syn::Ident,
    m: &[Meta],
    variants: Option<&syn::punctuated::Punctuated<syn::Variant, syn::Token![,]>>,
) -> Res<ErrorType> {
    let at = format!("{label}: {ident}");
    let with = meta_rust(m, "error.with")
        .or_else(|| m.iter().any(|x| x.key == "error.original").then(|| "crate::runtime::raise_host_error".to_owned()));
    let kind = meta_str(m, "error.kind");
    if with.is_some() && kind.is_some() {
        return Err(format!("{at}: `kind` and `with` are two answers to one question"));
    }
    let mut out = Vec::new();
    for v in variants.into_iter().flatten() {
        let Some(vm) = host_attr_strict(label, &v.attrs)? else { continue };
        // `#[host(delegate)]` on a one-field tuple variant: the wrapped error's own map answers.
        if vm.iter().any(|x| x.key == "delegate") {
            let syn::Fields::Unnamed(u) = &v.fields else {
                return Err(format!("{at}::{}: `delegate` wraps one error, in a one-field tuple variant", v.ident));
            };
            let inner = match u.unnamed.iter().map(|f| &f.ty).collect::<Vec<_>>().as_slice() {
                [syn::Type::Path(p)] => p.path.segments.last().map(|s| s.ident.to_string()),
                _ => None,
            }
            .ok_or_else(|| format!("{at}::{}: `delegate` wraps one error, in a one-field tuple variant", v.ident))?;
            out.push((v.ident.to_string(), "(inner)", format!("={inner}")));
            continue;
        }
        let Some(k) = meta_str(&vm, "kind") else {
            return Err(format!("{at}::{}: a variant of an error type takes #[host(kind = \"..\")] or #[host(delegate)]", v.ident));
        };
        if with.is_some() {
            return Err(format!("{at}::{}: a variant `kind` beside `with` -- the converter is the whole answer", v.ident));
        }
        let pat = match v.fields {
            syn::Fields::Unit => "",
            syn::Fields::Unnamed(_) => "(..)",
            syn::Fields::Named(_) => " { .. }",
        };
        out.push((v.ident.to_string(), pat, k));
    }
    Ok(ErrorType {
        name: ident.to_string(),
        path: format!("{module}::{ident}"),
        source: label.to_owned(),
        kind: kind.unwrap_or_else(|| "ValueError".to_owned()),
        with,
        variants: out,
    })
}

fn kw_record(label: &str, module: &str, st: &syn::ItemStruct, m: &[Meta], al: &Aliases) -> Res<KwRecord> {
    let at = format!("{label}: {}", st.ident);
    let syn::Fields::Named(named) = &st.fields else {
        return Err(format!("{at}: a keyword record has named fields -- they are the keywords"));
    };
    let mut fields = Vec::new();
    for f in &named.named {
        let name = f.ident.as_ref().expect("named").to_string();
        let fm = host_attr_strict(label, &f.attrs)?.unwrap_or_default();
        if let Some(bad) = fm.iter().find(|x| !["kwarg", "kwarg.default", "kwarg.lenient"].contains(&x.key.as_str())) {
            return Err(format!("{at}.{name}: #[host({})] -- a keyword field takes kwarg(default, lenient)", bad.key));
        }
        fields.push(KwField {
            ty: rty(&format!("{at}.{name}"), &f.ty, al)?,
            default: meta_rust(&fm, "kwarg.default"),
            lenient: fm.iter().any(|x| x.key == "kwarg.lenient"),
            name,
        });
    }
    // `client(..)`: a class's keyword constructor; no Node support.
    let client = head_of(m) == "client";
    Ok(KwRecord {
        name: st.ident.to_string(),
        path: format!("{module}::{}", st.ident),
        fields,
        client,
    })
}

fn stream(
    label: &str,
    module: &str,
    ident: &syn::Ident,
    _attrs: &[syn::Attribute],
    m: &[Meta],
    al: &Aliases,
    alias_of: Option<&syn::Type>,
) -> Res<Stream> {
    let at = format!("{label}: {ident}");
    let known = [
        "name", "sink", "push", "input", "output", "end_input",
        "close_ends_input", "optional_sink", "error", "deadline", "stub", "stub_import", "source",
        "arg", "input_py", "doc",
    ];
    for x in m.iter().skip(1) {
        let key = x.key.strip_prefix("stream.").unwrap_or(&x.key);
        if !known.contains(&key) {
            return Err(format!("{at}: stream({key}) -- a stream takes {}", known.join(", ")));
        }
    }
    let name = meta_str(m, "stream.name").ok_or_else(|| format!("{at}: stream(name = \"..\") is the class name"))?;
    let ty = |key: &str| -> Res<Option<RTy>> {
        match meta_rust(m, &format!("stream.{key}")) {
            None => Ok(None),
            Some(t) => {
                let parsed = syn::parse_str::<syn::Type>(&t).map_err(|e| format!("{at}: {key} = \"{t}\": {e}"))?;
                Ok(Some(rty(&format!("{at}.{key}"), &parsed, al)?))
            }
        }
    };
    let sink_ty = meta_rust(m, "stream.sink");
    let input = ty("input")?;
    if sink_ty.is_some() != input.is_some() {
        return Err(format!("{at}: `sink` and `input` come together -- a stream pushes something or nothing"));
    }
    let output = ty("output")?.ok_or_else(|| format!("{at}: stream(output = \"T\") is what recv() yields"))?;
    let flag = |k: &str| m.iter().any(|x| x.key == format!("stream.{k}"));
    if sink_ty.is_none() && (flag("end_input") || flag("close_ends_input") || flag("optional_sink")) {
        return Err(format!("{at}: end_input/close_ends_input/optional_sink on a stream with no sink"));
    }
    let is_box = |t: &syn::Type| matches!(t, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Box"));
    let boxed = match meta_rust(m, "stream.source") {
        Some(src) => al.get(src.rsplit("::").next().unwrap_or(&src)).is_some_and(is_box),
        None => alias_of.is_some_and(is_box),
    };
    let input = input.unwrap_or(RTy::Unit);
    Ok(Stream {
        name,
        source_ty: meta_rust(m, "stream.source").unwrap_or_else(|| format!("{module}::{ident}")),
        sink_ty,
        push: meta_str(m, "stream.push").unwrap_or_else(|| "push".to_owned()),
        arg: meta_str(m, "stream.arg").unwrap_or_else(|| "text".to_owned()),
        input_py: meta_str(m, "stream.input_py").unwrap_or_else(|| "str".to_owned()),
        boxed,
        input,
        output,
        end_input: flag("end_input"),
        close_ends_input: flag("close_ends_input"),
        optional_sink: meta_str(m, "stream.optional_sink"),
        err_with: meta_rust(m, "stream.error").unwrap_or_else(|| "ProviderError".to_owned()),
        deadline: meta_rust(m, "stream.deadline").unwrap_or_else(|| "crate::inference::NATIVE_CLOSE_DEADLINE".to_owned()),
        // The class's own `__doc__`: one source type backs several classes.
        doc: meta_str(m, "stream.doc").map(|d| vec![d]).unwrap_or_default(),
        stub_extra: m.iter().filter(|x| x.key == "stream.stub").filter_map(|x| x.value.clone()).collect(),
        stub_imports: m.iter().filter(|x| x.key == "stream.stub_import").filter_map(|x| x.value.clone()).collect(),
    })
}


fn span_text(text: &str, span: proc_macro2::Span) -> String {
    let (start, end) = (span.start(), span.end());
    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::new();
    for l in start.line..=end.line {
        let line: Vec<char> = lines.get(l - 1).copied().unwrap_or("").chars().collect();
        let a = if l == start.line { start.column } else { 0 };
        let b = if l == end.line { end.column.min(line.len()) } else { line.len() };
        out.push_str(&line[a.min(b)..b].iter().collect::<String>());
        out.push(' ');
    }
    let mut collapsed = out.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [("( ", "("), (" )", ")"), (", )", ")"), (",)", ")")] {
        collapsed = collapsed.replace(from, to);
    }
    collapsed
}

/// A file's `use` paths, flattened and absolute, as `(path, bound name)`; a glob binds `*`.
fn flat_uses(module: &str, file: &syn::File) -> Vec<(String, String)> {
    fn walk(prefix: &str, t: &syn::UseTree, out: &mut Vec<(String, String)>) {
        let join = |a: &str, b: &str| if a.is_empty() { b.to_owned() } else { format!("{a}::{b}") };
        match t {
            syn::UseTree::Path(p) => walk(&join(prefix, &p.ident.to_string()), &p.tree, out),
            syn::UseTree::Name(n) => {
                let name = n.ident.to_string();
                if name == "self" {
                    let bound = prefix.rsplit("::").next().unwrap_or(prefix).to_owned();
                    out.push((prefix.to_owned(), bound));
                } else {
                    out.push((join(prefix, &name), name));
                }
            }
            syn::UseTree::Rename(r) => out.push((
                format!("{} as {}", join(prefix, &r.ident.to_string()), r.rename),
                r.rename.to_string(),
            )),
            syn::UseTree::Glob(_) => out.push((join(prefix, "*"), "*".to_owned())),
            syn::UseTree::Group(g) => g.items.iter().for_each(|i| walk(prefix, i, out)),
        }
    }
    let krate = module.split("::").next().unwrap_or(module);
    let parent = module.rsplit_once("::").map(|(p, _)| p).unwrap_or(krate);
    let mut out = Vec::new();
    for item in &file.items {
        let syn::Item::Use(u) = item else { continue };
        if u.attrs.iter().any(|a| a.path().is_ident("cfg")) {
            continue;
        }
        let mut here = Vec::new();
        walk("", &u.tree, &mut here);
        for (path, bound) in here {
            let path = if let Some(r) = path.strip_prefix("crate::") {
                format!("{krate}::{r}")
            } else if let Some(r) = path.strip_prefix("self::") {
                format!("{module}::{r}")
            } else if let Some(r) = path.strip_prefix("super::") {
                format!("{parent}::{r}")
            } else {
                path
            };
            out.push((path, bound));
        }
    }
    out
}

/// `crate::`/`self::`/`super::` in a copied signature mean the core crate, not the binding.
fn absolute(module: &str, s: &str) -> String {
    let krate = module.split("::").next().unwrap_or(module);
    let parent = module.rsplit_once("::").map(|(p, _)| p).unwrap_or(krate);
    s.replace("crate::", &format!("{krate}::"))
        .replace("super::", &format!("{parent}::"))
        .replace("self::", &format!("{module}::"))
}

fn proxy(
    label: &str,
    module: &str,
    text: &str,
    file: &syn::File,
    tr: &syn::ItemTrait,
    m: &[Meta],
    _al: &Aliases,
) -> Res<Proxy> {
    let at = format!("{label}: trait {}", tr.ident);
    if !tr.generics.params.is_empty() {
        return Err(format!("{at}: a generic trait has no single proxy"));
    }
    // A bare proxy: every method, defaulted or not, calls `_core_<method>` on the host object.
    let host = meta_str(m, "proxy.host");
    let name = match (meta_rust(m, "proxy.name"), &host) {
        (Some(n), _) => n,
        (None, Some(_)) => format!("Py{}", tr.ident),
        (None, None) => return Err(format!("{at}: proxy(name = \"..\") names the generated struct")),
    };
    let state: Vec<(String, String)> = m
        .iter()
        .any(|x| x.key == "proxy.clock")
        .then(|| ("clock".to_owned(), "std::sync::Arc<dyn lk_agents_abi::Clock>".to_owned()))
        .into_iter()
        .collect();
    let async_trait = tr.attrs.iter().any(|a| a.path().segments.last().is_some_and(|s| s.ident == "async_trait"));
    let mut methods = Vec::new();
    for inner in &tr.items {
        let syn::TraitItem::Fn(f) = inner else { continue };
        let rust = f.sig.ident.to_string();
        let mat = format!("{at}::{rust}");
        let marker = host_attr_strict(label, &f.attrs)?;
        if host.is_some() && marker.as_ref().is_some_and(|m| m.iter().any(|x| x.key != "soon")) {
            return Err(format!("{mat}: a bare proxy's methods carry no marker but `soon` -- each calls `_core_{rust}` and crosses by its signature"));
        }
        let fm = match marker {
            Some(fm) => fm,
            None if host.is_some() => Vec::new(),
            None => {
                if f.default.is_none() {
                    return Err(format!("{mat}: a required method with no host marker -- the proxy cannot implement it (soon)"));
                }
                continue;
            }
        };
        let (ret, ret_result, ret_ref) = match &f.sig.output {
            syn::ReturnType::Default => (String::new(), false, false),
            syn::ReturnType::Type(_, t) => (
                span_text(text, t.span()),
                matches!(&**t, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Result")),
                matches!(&**t, syn::Type::Reference(_)),
            ),
        };
        let has_params = f.sig.inputs.iter().any(|i| matches!(i, syn::FnArg::Typed(_)));
        let action = match meta_str(&fm, "soon") {
            Some(method) => ProxyAction::Soon { method },
            None if fm.is_empty() => ProxyAction::Core,
            None => return Err(format!("{mat}: #[host({})] -- a proxy method takes soon = \"..\" or nothing", head_of(&fm))),
        };
        match &action {
            // A bare getter lending a borrow is read once and stored; any other host call cannot lend.
            ProxyAction::Core if ret_ref && f.sig.asyncness.is_none() && !has_params => {}
            _ if ret_ref => return Err(format!("{mat}: a host call cannot lend a reference")),
            ProxyAction::Soon { .. } if !ret.is_empty() && !ret_result => {
                return Err(format!("{mat}: `soon` is fire-and-forget: it returns () or Result<(), E>"));
            }
            _ => {}
        }
        let sig = absolute(module, &span_text(text, f.sig.span()));
        methods.push(ProxyMethod { name: rust, sig, action });
    }
    // Only names the copied signatures use: no crate-private paths, no unused-import warnings.
    let words: Vec<String> = methods
        .iter()
        .map(|x| x.sig.as_str())
        .chain(state.iter().map(|(_, t)| t.as_str()))
        .flat_map(|s| s.split(|c: char| !(c.is_alphanumeric() || c == '_')))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect();
    let mut uses: Vec<String> = flat_uses(module, file)
        .into_iter()
        .filter(|(_, bound)| bound != "*" && words.contains(bound))
        .map(|(path, _)| path)
        .collect();
    uses.push(format!("{module}::*"));
    Ok(Proxy {
        trait_path: format!("{module}::{}", tr.ident),
        name,
        host,
        source: label.to_owned(),
        async_trait,
        uses,
        state,
        methods,
    })
}
