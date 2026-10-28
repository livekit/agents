//! `EXPORT_SOURCES` in, [`Exposed`] out: every `#[host(export|object|error|stream|kwargs|proxy)]`
//! item, with the file's `type` aliases read through.

use super::meta::{
    Meta, has_host_derive, head_of, host_attr_strict, host_attrs_strict, meta_rust, meta_str,
    want_head,
};
use super::signature::{reply_split, rty, signature};
use super::{Res, quote};
use crate::ir::*;
use lk_agents_bindgen::doc;

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
pub fn exposed(label: &str, text: &str) -> Res<Exposed> {
    exposed_with(label, text, &[])
}

/// Every exposed `#[host(..)]` item in one `EXPORT_SOURCES` file.
pub fn exposed_with(label: &str, text: &str, global: &[(String, String)]) -> Res<Exposed> {
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
                let f = export_fn(&at, &m, &f.sig, &f.attrs, rust, &al)?;
                if let Some(target) = python_key(&at, &item_attrs(item), "forward", &["module"])?
                    .and_then(|m| meta_str(&m, "python.forward.module"))
                {
                    out.forwards.push((target, f.name.clone(), label.to_owned()));
                }
                out.fns.push(f);
            }
            syn::Item::Const(c) => {
                let Some(m) = host_attr_strict(label, &c.attrs)? else { continue };
                if head_of(&m) == "file" {
                    continue;
                }
                let rust = c.ident.to_string();
                let at = format!("{label}: {rust}");
                want_head(&at, &m, "export")?;
                out.fns.push(export_const(&at, &m, &c.ty, &c.attrs, &rust, &al)?);
            }
            syn::Item::Impl(imp) => {
                let Some(m) = host_attr_strict(label, &imp.attrs)? else { continue };
                if head_of(&m) == "export" {
                    exported_impl(label, imp, &mut out, &al)?;
                } else {
                    let o = object(label, imp, &m, &al)?;
                    let at = format!("{label}: {}", o.name);
                    if let Some(w) =
                        python_key(&at, &imp.attrs, "wrapper", &["module", "path", "name", "base"])?
                    {
                        let get = |k: &str| meta_str(&w, &format!("python.wrapper.{k}"));
                        let need = |k: &str| {
                            get(k).ok_or_else(|| format!("{at}: wrapper needs {k} = \"..\""))
                        };
                        out.wrappers.push(Wrapper {
                            name: need("name")?,
                            module: need("module")?,
                            path: get("path"),
                            source: label.to_owned(),
                            doc: doc(&imp.attrs),
                            base: need("base")?,
                            core: o.name.clone(),
                        });
                    }
                    out.objects.push(o);
                }
            }
            // A `#[derive(Host)]` record's `#[host(..)]` is the record vocabulary, not this one.
            syn::Item::Enum(e) if has_host_derive(&e.attrs) => {}
            syn::Item::Struct(st) if has_host_derive(&st.attrs) => {}
            syn::Item::Enum(e) => {
                for m in host_attrs_strict(label, &e.attrs)? {
                    match head_of(&m).as_str() {
                        "wire_enum" | "string" => {
                            out.wire_enums.push(WireEnum { name: e.ident.to_string() })
                        }
                        "stream" => out.streams.push(stream(label, &e.ident, &m)?),
                        // The binding maps the error itself; the wire strings are Rust side
                        // only: the enum crosses by its own impls, if at all.
                        "error" | "wire_strs" | "handwritten" => {}
                        other => {
                            return Err(format!(
                                "{label}: enum {}: #[host({other})] -- an enum takes wire_enum, wire_strs, string, error or stream",
                                e.ident
                            ));
                        }
                    }
                }
            }
            syn::Item::Struct(st) => {
                for m in host_attrs_strict(label, &st.attrs)? {
                    match head_of(&m).as_str() {
                        "client" | "kwargs" => out.kwargs.push(kw_record(label, st, &al)?),
                        // The binding maps the error itself.
                        "error" => {}
                        "stream" => out.streams.push(stream(label, &st.ident, &m)?),
                        // `Display` out, `FromStr` in; the macro writes nothing.
                        "string" => {
                            out.wire_enums.push(WireEnum { name: st.ident.to_string() });
                            if let Some(c) = str_class(label, st)? {
                                out.str_classes.push(c);
                            }
                        }
                        other => {
                            return Err(format!(
                                "{label}: struct {}: #[host({other})] -- a struct takes client, kwargs, error, stream or string",
                                st.ident
                            ));
                        }
                    }
                }
            }
            syn::Item::Type(t) => {
                for m in host_attrs_strict(label, &t.attrs)? {
                    want_head(&format!("{label}: type {}", t.ident), &m, "stream")?;
                    out.streams.push(stream(label, &t.ident, &m)?);
                }
            }
            syn::Item::Trait(tr) => {
                let Some(m) = host_attr_strict(label, &tr.attrs)? else { continue };
                // A native-only trait has no host surface to describe; a class is `decl`'s.
                if m.first().is_some_and(|x| matches!(x.key.split('.').next(), Some("native" | "class"))) {
                    continue;
                }
                want_head(&format!("{label}: trait {}", tr.ident), &m, "proxy")?;
                let host = meta_str(&m, "proxy.host");
                out.providers.extend(super::provider::provider(label, tr, host.as_deref(), text)?);
                out.proxies.push(proxy(label, tr)?);
            }
            _ => {}
        }
    }
    Ok(out)
}

fn item_attrs(item: &syn::Item) -> Vec<syn::Attribute> {
    match item {
        syn::Item::Fn(f) => f.attrs.clone(),
        _ => Vec::new(),
    }
}

/// The item's `#[python(<head>(..))]` entries, if any; any other key is refused.
fn python_key(
    at: &str,
    attrs: &[syn::Attribute],
    head: &str,
    keys: &[&str],
) -> Res<Option<Vec<Meta>>> {
    let m: Vec<Meta> =
        super::meta::metas(attrs).into_iter().filter(|m| m.key.starts_with("python.")).collect();
    if m.is_empty() {
        return Ok(None);
    }
    for x in &m {
        let key = x.key.strip_prefix(&format!("python.{head}")).unwrap_or("!");
        let key = key.strip_prefix('.').unwrap_or(key);
        if !(key.is_empty() || keys.contains(&key)) {
            return Err(format!(
                "{at}: #[python({head}(..))] takes {}, not `{}`",
                keys.join(", "),
                x.key
            ));
        }
    }
    Ok(Some(m))
}

/// A `#[host(string)]` struct's `#[python(str_class(..))]`, if it has one.
fn str_class(label: &str, st: &syn::ItemStruct) -> Res<Option<StrClass>> {
    let m: Vec<Meta> = super::meta::metas(&st.attrs)
        .into_iter()
        .filter(|m| m.key.starts_with("python."))
        .collect();
    if m.is_empty() {
        return Ok(None);
    }
    let at = format!("{label}: {}", st.ident);
    let get = |k: &str| meta_str(&m, &format!("python.str_class.{k}"));
    let mut properties = Vec::new();
    let mut methods = Vec::new();
    for x in &m {
        let key = x.key.as_str();
        if let Some(name) = key.strip_prefix("python.str_class.property.") {
            properties.push((name.to_owned(), x.value.clone().unwrap_or_default()));
        } else if let Some(name) = key.strip_prefix("python.str_class.method.") {
            methods.push((name.to_owned(), x.value.clone().unwrap_or_default()));
        } else if ![
            "python.str_class",
            "python.str_class.module",
            "python.str_class.path",
            "python.str_class.new",
            "python.str_class.property",
            "python.str_class.method",
        ]
        .contains(&key)
        {
            return Err(format!(
                "{at}: #[python(str_class(..))] takes module, path, new, property(..) and method(..), not `{key}`"
            ));
        }
    }
    Ok(Some(StrClass {
        name: st.ident.to_string(),
        module: get("module").ok_or_else(|| format!("{at}: str_class needs module = \"..\""))?,
        path: get("path"),
        source: label.to_owned(),
        doc: doc(&st.attrs),
        new: get("new").ok_or_else(|| format!("{at}: str_class needs new = \"<exported fn>\""))?,
        properties,
        methods,
    }))
}

fn export_fn(
    at: &str,
    m: &[Meta],
    sig: &syn::Signature,
    attrs: &[syn::Attribute],
    rust: String,
    al: &Aliases,
) -> Res<ExportFn> {
    let is_async = sig.asyncness.is_some();
    let (receiver, sig) = signature(at, m, "export", sig, al)?;
    if receiver != Receiver::None {
        return Err(format!("{at}: a free fn has no receiver"));
    }
    Ok(ExportFn {
        name: meta_str(m, "export.name").unwrap_or(rust),
        doc: doc(attrs),
        sig,
        is_attr: false,
        is_async,
    })
}

fn export_const(
    at: &str,
    m: &[Meta],
    ty: &syn::Type,
    attrs: &[syn::Attribute],
    rust: &str,
    al: &Aliases,
) -> Res<ExportFn> {
    for key in ["defaults", "kw_only", "err"] {
        if m.iter().any(|x| x.key.starts_with(&format!("export.{key}"))) {
            return Err(format!(
                "{at}: `{key}` on a const -- a const takes no arguments and cannot fail"
            ));
        }
    }
    let ret = rty(at, ty, al)?;
    let name = meta_str(m, "export.name").unwrap_or_else(|| rust.to_lowercase());
    Ok(ExportFn {
        // An UPPER_CASE name is a module attribute, as the in-place expansion reads it.
        is_attr: name.chars().any(|c| c.is_ascii_uppercase())
            && !name.chars().any(|c| c.is_ascii_lowercase()),
        name,
        doc: doc(attrs),
        sig: Sig {
            params: Vec::new(),
            ret,
            defaults: Vec::new(),
            kw_only: KwOnly::None,
            stream: None,
        },
        is_async: false,
    })
}

/// `#[host(export)] impl T { .. }`: associated items as module functions, called through `T::`.
fn exported_impl(label: &str, imp: &syn::ItemImpl, out: &mut Exposed, al: &Aliases) -> Res<()> {
    let syn::Type::Path(p) = &*imp.self_ty else {
        return Err(format!("{label}: #[host(export)] on an impl is for `impl Type`"));
    };
    let core = p.path.segments.last().expect("a path").ident.to_string();
    if imp.trait_.is_some() || !imp.generics.params.is_empty() {
        return Err(format!(
            "{label}: impl {core}: #[host(export)] is for an inherent, non-generic impl"
        ));
    }
    let mut any = false;
    for inner in &imp.items {
        match inner {
            syn::ImplItem::Const(c) => {
                let Some(m) = host_attr_strict(label, &c.attrs)? else { continue };
                let rust = c.ident.to_string();
                let at = format!("{label}: {core}::{rust}");
                want_head(&at, &m, "export")?;
                out.fns.push(export_const(&at, &m, &c.ty, &c.attrs, &rust, al)?);
                any = true;
            }
            syn::ImplItem::Fn(f) => {
                let Some(m) = host_attr_strict(label, &f.attrs)? else { continue };
                let rust = f.sig.ident.to_string();
                let at = format!("{label}: {core}::{rust}");
                want_head(&at, &m, "export")?;
                out.fns.push(export_fn(&at, &m, &f.sig, &f.attrs, rust, al)?);
                any = true;
            }
            _ => {}
        }
    }
    if !any {
        return Err(format!(
            "{label}: impl {core}: #[host(export)] on an impl with no #[host(export)] item"
        ));
    }
    Ok(())
}

fn object(label: &str, imp: &syn::ItemImpl, m: &[Meta], al: &Aliases) -> Res<Object> {
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
            return Err(format!(
                "{at}: #[host({head})] -- a method marker is new, getter, setter or method"
            ));
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
            kind,
            sig,
            reply,
            doc: doc(&f.attrs),
        });
    }
    if methods.iter().filter(|x| x.kind == MethodKind::New).count() > 1 {
        return Err(format!("{at}: more than one #[host(new)]"));
    }
    Ok(Object {
        core,
        name,
        doc: doc(&imp.attrs),
        methods,
        stub_extra: m
            .iter()
            .filter(|x| x.key == "object.stub")
            .filter_map(|x| x.value.clone())
            .collect(),
        stub_imports: m
            .iter()
            .filter(|x| x.key == "object.stub_import")
            .filter_map(|x| x.value.clone())
            .collect(),
        consts,
    })
}

fn kw_record(label: &str, st: &syn::ItemStruct, al: &Aliases) -> Res<KwRecord> {
    let at = format!("{label}: {}", st.ident);
    let syn::Fields::Named(named) = &st.fields else {
        return Err(format!("{at}: a keyword record has named fields -- they are the keywords"));
    };
    let mut fields = Vec::new();
    for f in &named.named {
        let name = f.ident.as_ref().expect("named").to_string();
        let fm = host_attr_strict(label, &f.attrs)?.unwrap_or_default();
        if let Some(bad) = fm
            .iter()
            .find(|x| !["kwarg", "kwarg.default", "kwarg.lenient"].contains(&x.key.as_str()))
        {
            return Err(format!(
                "{at}.{name}: #[host({})] -- a keyword field takes kwarg(default, lenient)",
                bad.key
            ));
        }
        fields.push(KwField {
            ty: rty(&format!("{at}.{name}"), &f.ty, al)?,
            default: meta_rust(&fm, "kwarg.default"),
            name,
        });
    }
    Ok(KwRecord { name: st.ident.to_string(), fields })
}

fn stream(label: &str, ident: &syn::Ident, m: &[Meta]) -> Res<Stream> {
    let at = format!("{label}: {ident}");
    let known = [
        "name",
        "sink",
        "push",
        "input",
        "output",
        "end_input",
        "close_ends_input",
        "optional_sink",
        "error",
        "deadline",
        "stub",
        "stub_import",
        "source",
        "arg",
        "input_py",
        "doc",
    ];
    for x in m.iter().skip(1) {
        let key = x.key.strip_prefix("stream.").unwrap_or(&x.key);
        if !known.contains(&key) {
            return Err(format!("{at}: stream({key}) -- a stream takes {}", known.join(", ")));
        }
    }
    let values = |key: &str| -> Vec<String> {
        let key = format!("stream.{key}");
        m.iter().filter(|x| x.key == key).filter_map(|x| x.value.clone()).collect()
    };
    Ok(Stream {
        name: meta_str(m, "stream.name")
            .ok_or_else(|| format!("{at}: stream(name = \"..\") is the class name"))?,
        sink_ty: meta_rust(m, "stream.sink"),
        push: meta_str(m, "stream.push").unwrap_or_else(|| "push".to_owned()),
        arg: meta_str(m, "stream.arg").unwrap_or_else(|| "text".to_owned()),
        input_py: meta_str(m, "stream.input_py").unwrap_or_else(|| "str".to_owned()),
        end_input: m.iter().any(|x| x.key == "stream.end_input"),
        // The class's own `__doc__`: one source type backs several classes.
        doc: meta_str(m, "stream.doc").map(|d| vec![d]).unwrap_or_default(),
        stub_extra: values("stub"),
        stub_imports: values("stub_import"),
    })
}

fn proxy(label: &str, tr: &syn::ItemTrait) -> Res<Proxy> {
    if !tr.generics.params.is_empty() {
        return Err(format!("{label}: trait {}: a generic trait has no single proxy", tr.ident));
    }
    let methods = tr
        .items
        .iter()
        .filter_map(|inner| match inner {
            syn::TraitItem::Fn(f) => Some(f.sig.ident.to_string()),
            _ => None,
        })
        .collect();
    Ok(Proxy { trait_name: tr.ident.to_string(), methods })
}
