//! `lk-agents-bindgen`: a `#[host(..)]` item's pyo3 binding, expanded in the crate that owns it
//! (behind its `python` feature), registered through `lk_agents_abi::python::Registration`.
//!
//! A proc macro sees one item, so cross-item lookups are traits: values cross by
//! `IntoPyObject`/`FromPyObject`, errors by `HostException` (else `ValueError`, via autoref), and
//! aliases resolve because the wrapper is emitted in the item's module.
//!
//! `cell = "mutex"`: a non-`Sync` core type with `&mut self` methods becomes a frozen class over
//! `Mutex<T>`, beside `T`. The attribute language is flattened once, by [`meta`], shared with
//! `lk-agents-codegen`: Rust values unquoted, host text quoted.

mod cross;
mod kwargs;
pub mod meta;
mod proxy;
mod python;
mod record;
mod shape;
mod stream;
#[cfg(test)]
mod tests;

pub use python::{class, error, export_const, export_fn, export_impl, object, string, wire_enum};
pub use kwargs::{client, kwargs};
pub use proxy::proxy;
pub use record::{RecordDefaults, record};
pub use stream::stream;

pub use meta::{Meta, Metas, Value, doc, is_host, to_snake};
use proc_macro2::{Span, TokenStream};

/// Core crates whose `python` feature enables `pyo3` and `lk-agents-abi/python`; markers in any
/// other file stay inert.
pub const IN_PLACE_CRATES: &[&str] = &[
    "rust/crates/lk-agents-abi/src/",
    "rust/crates/lk-agents-rtc/src/",
    "rust/crates/lk-agents-media/src/",
    "rust/crates/lk-agents-worker/src/",
    "rust/crates/lk-agents-voice/src/",
    "rust/crates/lk-agents-inference/src/",
];

/// `file` as rustc names it: workspace-relative or absolute.
pub fn in_place(file: Option<&std::path::Path>) -> bool {
    let Some(file) = file else { return false };
    let file = file.to_string_lossy().replace('\\', "/");
    IN_PLACE_CRATES.iter().any(|dir| {
        let tail = dir.strip_prefix("rust/").unwrap_or(dir);
        file.starts_with(tail) || file.contains(&format!("/{tail}"))
    })
}

/// Raw when a Rust keyword (`type`); pyo3 unraws it.
pub fn host_ident(name: &str) -> syn::Ident {
    match syn::parse_str::<syn::Ident>(name) {
        Ok(ident) => ident,
        Err(_) => syn::Ident::new_raw(name, Span::call_site()),
    }
}


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Receiver {
    None,
    Ref,
    Mut,
}

#[derive(Clone)]
pub enum Ret {
    Unit,
    Plain(syn::Type),
    /// `Result<T, E>` (or a one-argument alias of it): the `T`.
    Result(syn::Type),
}

#[derive(Clone)]
pub enum KwOnly {
    None,
    All,
    From(String),
}

#[derive(Clone)]
pub struct Param {
    pub name: syn::Ident,
    pub ty: syn::Type,
    /// The name a conversion error gives the argument (default: its own).
    pub label: Option<String>,
    /// The runtime's own value (`lk_agents_abi::python::Ambient`), not a host argument.
    pub ambient: bool,
    /// Anything that does not extract reads as the type's default.
    pub lenient: bool,
    /// A `#[host(kwargs)]` struct spread into one keyword per field.
    pub kwargs: bool,
    /// An `Option<T>` the host must pass.
    pub required: bool,
}

#[derive(Clone)]
pub struct Sig {
    pub rust: syn::Ident,
    pub name: String,
    pub receiver: Receiver,
    pub params: Vec<Param>,
    pub ret: Ret,
    /// In the order written.
    pub defaults: Vec<(String, TokenStream)>,
    pub kw_only: KwOnly,
    /// `err = "RuntimeError"`: every failure raises it. `None`: `HostException`, else `ValueError`.
    pub err: Option<String>,
    /// Call `check_pid()` first (the default).
    pub pid: bool,
    /// Release the GIL around the call.
    pub detach: bool,
    /// `///` lines, one leading space removed, trailing blanks dropped.
    pub doc: Vec<String>,
    pub is_async: bool,
    /// The returned halves, as that `#[host(stream)]` class.
    pub stream: Option<syn::Path>,
    /// `T` of a `Reply<T>` return: sent now, awaited by the host.
    pub reply: Option<syn::Type>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MethodKind {
    New,
    Method,
    Getter,
    Setter,
    Static,
}

#[derive(Clone)]
pub struct Method {
    pub kind: MethodKind,
    pub sig: Sig,
}

#[derive(Clone)]
pub enum Cell {
    /// `#[lk_agents_abi::class]` on the struct.
    Plain,
    Mutex { rust_name: syn::Ident },
    /// The cell an `async` method needs: the future outlives the borrow of `self`.
    Arc { rust_name: syn::Ident },
    /// `T` by value, for types pyo3 cannot make a class of (an enum with data). `input`: the host
    /// constructs and passes it in, so no `IntoPyObject` for `T`.
    Value { rust_name: syn::Ident, pyclass: Vec<String>, derive: Option<String>, input: bool },
}

impl Cell {
    pub fn class(&self) -> Option<&syn::Ident> {
        match self {
            Cell::Plain => None,
            Cell::Mutex { rust_name } | Cell::Arc { rust_name } | Cell::Value { rust_name, .. } => Some(rust_name),
        }
    }
}

#[derive(Clone)]
pub struct Object {
    pub self_ty: syn::Type,
    pub name: String,
    pub module: String,
    pub cell: Cell,
    pub doc: Vec<String>,
    pub methods: Vec<Method>,
    /// `#[host(const)]` associated consts: class attributes.
    pub consts: Vec<ConstExport>,
    /// `as_dyn(Trait = "method")`: a native `Arc<dyn Trait>`, taken as itself wherever a host
    /// object is proxied as that trait.
    pub as_dyn: Vec<(syn::Path, syn::Ident)>,
}

#[derive(Clone)]
pub struct Class {
    pub name: String,
    pub module: String,
    pub mutable: bool,
    /// pyo3's opt-in: an argument is taken by value (a clone); otherwise only borrowed.
    pub from_py_object: bool,
    pub eq: bool,
    /// pyo3's `subclass`: Python may derive from the class. The reference's pure-Python classes
    /// (`ExpFilter`, `MovingAverage`, ...) are subclassed by users; a pyclass without it raises
    /// `TypeError: type '..' is not an acceptable base type`.
    pub subclass: bool,
}

pub const DEFAULT_MODULE: &str = "livekit.agents._core";

/// The host's `__doc__`: the first paragraph, intra-doc links unlinked.
pub fn summary(doc: &[String]) -> Vec<String> {
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

/// One callable under the `head` group (`export`, `new`, `method`, `getter`, `static`).
pub fn sig(head: &str, m: &Metas, s: &syn::Signature, attrs: &[syn::Attribute], object_pid: Option<bool>) -> syn::Result<Sig> {
    let at = &s.ident;
    if !s.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(at, "a generic fn does not expand in place"));
    }
    let mut receiver = Receiver::None;
    let mut params = Vec::new();
    for input in &s.inputs {
        match input {
            syn::FnArg::Receiver(r) => {
                // `self: &Arc<Self>`: an `arc` cell's method that spawns over the shared value.
                let arc_self = r.colon_token.is_some()
                    && matches!(&*r.ty, syn::Type::Reference(rf) if quote::ToTokens::to_token_stream(&rf.elem).to_string().replace(' ', "") == "Arc<Self>");
                if r.reference.is_none() && !arc_self {
                    return Err(syn::Error::new_spanned(r, "a `self`-consuming method cannot be called on a shared host object"));
                }
                receiver = if r.mutability.is_some() { Receiver::Mut } else { Receiver::Ref };
            }
            syn::FnArg::Typed(t) => {
                let pm = Metas::of(&t.attrs)?;
                if let Some(bad) = pm.0.iter().find(|m| !PARAM_MARKERS.contains(&m.key.as_str())) {
                    return Err(syn::Error::new_spanned(&t.pat, format!("the parameter marker `{}` has no in-place meaning: give the type a host shape of its own", bad.key)));
                }
                let syn::Pat::Ident(id) = &*t.pat else {
                    return Err(syn::Error::new_spanned(&t.pat, "a destructuring parameter has no host name"));
                };
                params.push(Param {
                    name: id.ident.clone(),
                    ty: (*t.ty).clone(),
                    label: pm.text("label")?.map(str::to_owned),
                    ambient: pm.flag("ambient"),
                    lenient: pm.flag("lenient"),
                    kwargs: pm.flag("kwargs"),
                    required: pm.flag("required"),
                });
            }
        }
    }
    let ret = match &s.output {
        syn::ReturnType::Default => Ret::Unit,
        syn::ReturnType::Type(_, t) => match result_ok(t) {
            Some(ok) => Ret::Result(ok),
            None if matches!(&**t, syn::Type::Tuple(tu) if tu.elems.is_empty()) => Ret::Unit,
            None => Ret::Plain((**t).clone()),
        },
    };
    let err = m.text(&format!("{head}.err"))?.map(str::to_owned);
    if err.is_some() && !matches!(ret, Ret::Result(_)) {
        return Err(syn::Error::new_spanned(at, "`err` on a fn that does not return a Result"));
    }
    let prefix = format!("{head}.defaults.");
    let mut defaults = Vec::new();
    for (p, v) in m.under(&prefix) {
        if !params.iter().any(|q| q.name == p) {
            return Err(syn::Error::new_spanned(at, format!("defaults({p} = ..) names no parameter")));
        }
        let v = meta::rust_of(v).unwrap_or_default();
        let expr: TokenStream = v.parse().map_err(|_| syn::Error::new_spanned(at, format!("defaults({p}): not a Rust expression")))?;
        defaults.push((p.to_owned(), expr));
    }
    if let Some(p) = params.iter().find(|p| p.required && defaults.iter().any(|(d, _)| p.name == d)) {
        return Err(syn::Error::new_spanned(at, format!("`{}` is `required` and has a default", p.name)));
    }
    let key = format!("{head}.kw_only");
    let kw_only = match m.0.iter().find(|x| x.key == key).map(|x| &x.value) {
        None => KwOnly::None,
        Some(Value::Word) => KwOnly::All,
        Some(_) => {
            let p = m.text(&key)?.unwrap_or_default();
            if !params.iter().any(|q| q.name == p) {
                return Err(syn::Error::new_spanned(at, format!("kw_only = \"{p}\" names no parameter")));
            }
            KwOnly::From(p.to_owned())
        }
    };
    let pid = m.bool(&format!("{head}.pid"))?.or(object_pid).unwrap_or(true);
    let rust = s.ident.clone();
    let stream = m.path(&format!("{head}.stream"))?;
    let reply = match &ret {
        Ret::Plain(t) | Ret::Result(t) => reply_of(t),
        Ret::Unit => None,
    };
    let mut sig = Sig {
        reply,
        // `set_x` sets `x`.
        name: m.text(&format!("{head}.name"))?.map(str::to_owned).unwrap_or_else(|| match head {
            "setter" => rust.to_string().strip_prefix("set_").map(str::to_owned).unwrap_or_else(|| rust.to_string()),
            _ => rust.to_string(),
        }),
        rust,
        receiver,
        params,
        ret,
        defaults,
        kw_only,
        err,
        pid,
        detach: m.flag(&format!("{head}.detach")),
        doc: doc(attrs),
        is_async: s.asyncness.is_some(),
        stream,
    };
    sig.defaults = implied_defaults(&sig);
    Ok(sig)
}

pub const PARAM_MARKERS: [&str; 5] = ["label", "ambient", "lenient", "kwargs", "required"];

/// By the last path segment.
pub fn is_named(t: &syn::Type, name: &str) -> bool {
    matches!(t, syn::Type::Path(tp) if tp.qself.is_none() && tp.path.segments.last().is_some_and(|s| s.ident == name))
}

pub struct Shape<'a> {
    pub name: &'a str,
    pub option: bool,
    /// Defaults to `NotGiven` unwritten.
    pub maybe: bool,
    pub required: bool,
    /// Has a `defaults(..)` entry.
    pub written: bool,
}

/// `Option<T>` parameters not `required` whose followers all default: `None` unwritten (Python's
/// rule). Shared by the expansion and the codegen's stub.
pub fn implied_none<'a>(host: &[Shape<'a>]) -> Vec<&'a str> {
    let defaulted = |p: &Shape| p.written || p.maybe || (p.option && !p.required);
    let mut out = Vec::new();
    for (i, p) in host.iter().enumerate() {
        if p.option && !p.required && !p.written && host[i + 1..].iter().all(defaulted) {
            out.push(p.name);
        }
    }
    out
}

fn implied_defaults(s: &Sig) -> Vec<(String, TokenStream)> {
    let names: Vec<String> = s.params.iter().filter(|p| !p.ambient).map(|p| p.name.to_string()).collect();
    let host: Vec<Shape> = s
        .params
        .iter()
        .filter(|p| !p.ambient)
        .zip(&names)
        .map(|(p, name)| Shape {
            name,
            option: is_named(&p.ty, "Option"),
            maybe: is_named(&p.ty, "Maybe"),
            required: p.required,
            written: s.defaults.iter().any(|(d, _)| d == name),
        })
        .collect();
    let mut out = s.defaults.clone();
    out.extend(implied_none(&host).into_iter().map(|n| (n.to_owned(), quote::quote!(None))));
    out.sort_by_key(|(d, _)| names.iter().position(|n| n == d));
    out
}

fn reply_of(t: &syn::Type) -> Option<syn::Type> {
    let syn::Type::Path(tp) = t else { return None };
    let seg = tp.path.segments.last()?;
    if seg.ident != "Reply" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    a.args.iter().find_map(|g| if let syn::GenericArgument::Type(t) = g { Some(t.clone()) } else { None })
}

fn result_ok(t: &syn::Type) -> Option<syn::Type> {
    let syn::Type::Path(p) = t else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    a.args.iter().find_map(|g| match g {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    })
}

const SIG_KEYS: [&str; 8] = ["name", "defaults.", "kw_only", "err", "pid", "detach", "stream", ""];

fn sig_keys(head: &str) -> Vec<String> {
    SIG_KEYS.iter().map(|k| if k.is_empty() { head.to_owned() } else { format!("{head}.{k}") }).collect()
}

pub fn read_export(attr: TokenStream, f: &syn::ItemFn) -> syn::Result<Sig> {
    let m = Metas::parse(attr)?;
    let keys = sig_keys("export");
    m.only(&format!("fn {}", f.sig.ident), &keys.iter().map(String::as_str).collect::<Vec<_>>())?;
    sig("export", &m, &f.sig, &f.attrs, None)
}

/// `#[host(object ..)]` on an inherent impl, and its `#[host(new|method|getter|static ..)]` fns.
pub fn read_object(attr: TokenStream, imp: &syn::ItemImpl) -> syn::Result<Object> {
    let m = Metas::parse(attr)?;
    let syn::Type::Path(p) = &*imp.self_ty else {
        return Err(syn::Error::new_spanned(&imp.self_ty, "#[host(object)] is for `impl Type`"));
    };
    if imp.trait_.is_some() || !imp.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&imp.self_ty, "#[host(object)] is for an inherent, non-generic impl"));
    }
    let core = p.path.segments.last().expect("a path").ident.to_string();
    // `stub`/`stub_import` are the codegen's (the `.pyi`).
    m.only(
        &format!("impl {core}"),
        &[
            "object", "object.name", "object.module", "object.cell", "object.rust_name", "object.pid", "object.stub",
            "object.stub_import", "object.as_dyn.", "object.pyclass.", "object.derive.",
        ],
    )?;
    let name = m.text("object.name")?.map(str::to_owned).unwrap_or_else(|| core.clone());
    let rust_name = m.ident("object.rust_name")?;
    let wrapper = || rust_name.clone().unwrap_or_else(|| syn::Ident::new(&format!("Py{name}"), Span::call_site()));
    let words = |prefix: &str| m.under(prefix).map(|(w, _)| w.to_owned()).collect::<Vec<_>>();
    let (pyclass, derive) = (words("object.pyclass."), words("object.derive."));
    let cell = match m.text("object.cell")? {
        None | Some("plain") => match &rust_name {
            // The class is the type: a `rust_name` can only repeat it.
            Some(r) if *r != core => {
                return Err(syn::Error::new_spanned(&imp.self_ty, "`rust_name` names the class a `mutex`/`arc` cell wraps; a plain object is its own class"));
            }
            _ => Cell::Plain,
        },
        Some("mutex") => Cell::Mutex { rust_name: wrapper() },
        Some("arc") => Cell::Arc { rust_name: wrapper() },
        Some(kind @ ("value" | "input")) => Cell::Value {
            input: kind == "input",
            rust_name: wrapper(),
            pyclass: pyclass.clone(),
            derive: (!derive.is_empty()).then(|| derive.join(", ")),
        },
        Some(other) => {
            return Err(syn::Error::new_spanned(&imp.self_ty, format!("cell = \"{other}\": plain, mutex, arc, value or input")));
        }
    };
    if !matches!(cell, Cell::Value { .. }) && !(pyclass.is_empty() && derive.is_empty()) {
        return Err(syn::Error::new_spanned(&imp.self_ty, "`pyclass(..)`/`derive(..)` are a `value` cell's class options; a plain object's are its #[lk_agents_abi::class]"));
    }
    let mut as_dyn = Vec::new();
    for (tr, method) in m.under("object.as_dyn.") {
        let tr: syn::Path = syn::parse_str(tr).map_err(|_| syn::Error::new_spanned(&imp.self_ty, "as_dyn(Trait = method)"))?;
        let method: syn::Ident = meta::rust_of(method)
            .and_then(|v| syn::parse_str(&v).ok())
            .ok_or_else(|| syn::Error::new_spanned(&imp.self_ty, "as_dyn(Trait = method): the method is an identifier"))?;
        as_dyn.push((tr, method));
    }
    let object_pid = m.bool("object.pid")?;
    let mut methods = Vec::new();
    let mut consts = Vec::new();
    for item in &imp.items {
        let (attrs, sig_in) = match item {
            syn::ImplItem::Fn(f) => (&f.attrs, &f.sig),
            syn::ImplItem::Const(c) if c.attrs.iter().any(is_host) => {
                let cm = Metas::of(&c.attrs)?;
                cm.only(&format!("const {}", c.ident), &["const", "const.name"])?;
                consts.push(ConstExport {
                    path: quote::quote!(<#p>::#c.ident),
                    rust: c.ident.clone(),
                    name: cm.text("const.name")?.map(str::to_owned).unwrap_or_else(|| c.ident.to_string()),
                    ty: c.ty.clone(),
                    pid: false,
                    doc: doc(&c.attrs),
                    attr: false,
                });
                continue;
            }
            _ => continue,
        };
        if !attrs.iter().any(is_host) {
            continue;
        }
        let fm = Metas::of(attrs)?;
        let head = fm.head().to_owned();
        // `method`'s kind is the signature's: async, receiver-less (static), or `Reply<T>`.
        if !["new", "method", "getter", "setter"].contains(&head.as_str()) {
            return Err(syn::Error::new_spanned(&sig_in.ident, format!("host({head}) on a method: new, method, getter or setter")));
        }
        let mut keys = sig_keys(&head);
        if head == "method" {
            // Consumed by `fill_sends`, which wrote the body.
            keys.push("method.send".to_owned());
        }
        fm.only(&format!("{core}::{}", sig_in.ident), &keys.iter().map(String::as_str).collect::<Vec<_>>())?;
        let s = sig(&head, &fm, sig_in, attrs, object_pid)?;
        let kind = match head.as_str() {
            "new" => MethodKind::New,
            "getter" => MethodKind::Getter,
            "setter" => MethodKind::Setter,
            _ if s.receiver == Receiver::None => MethodKind::Static,
            _ => MethodKind::Method,
        };
        if s.is_async && !matches!(cell, Cell::Arc { .. }) {
            return Err(syn::Error::new_spanned(&sig_in.ident, "an `async fn` needs cell = \"arc\": the future outlives the borrow of self"));
        }
        let wants_self = matches!(kind, MethodKind::Getter | MethodKind::Setter);
        if wants_self && s.receiver == Receiver::None || kind == MethodKind::New && s.receiver != Receiver::None {
            return Err(syn::Error::new_spanned(&sig_in.ident, format!("host({head}) {}", if wants_self { "needs a receiver" } else { "takes no receiver" })));
        }
        if kind == MethodKind::New && !returns_self(&s.ret) {
            return Err(syn::Error::new_spanned(&sig_in.ident, "#[host(new)] returns Self or Result<Self, E>"));
        }
        if kind == MethodKind::Getter && (!s.params.is_empty() || s.detach) {
            return Err(syn::Error::new_spanned(&sig_in.ident, "#[host(getter)] takes only the receiver"));
        }
        let unit = |t: &syn::Type| matches!(t, syn::Type::Tuple(tu) if tu.elems.is_empty());
        let returns_nothing = match &s.ret {
            Ret::Unit => true,
            Ret::Result(t) => unit(t),
            Ret::Plain(_) => false,
        };
        if kind == MethodKind::Setter && (s.params.len() != 1 || !returns_nothing) {
            return Err(syn::Error::new_spanned(&sig_in.ident, "#[host(setter)] takes one value and returns nothing, or refuses (`Result<(), E>`)"));
        }
        if matches!(cell, Cell::Arc { .. }) && s.receiver == Receiver::Mut {
            return Err(syn::Error::new_spanned(&sig_in.ident, "a `&mut self` method in an `arc` cell: use cell = \"mutex\", or take &self"));
        }
        methods.push(Method { kind, sig: s });
    }
    if methods.iter().filter(|x| x.kind == MethodKind::New).count() > 1 {
        return Err(syn::Error::new_spanned(&imp.self_ty, "more than one #[host(new)]"));
    }
    Ok(Object {
        self_ty: (*imp.self_ty).clone(),
        name,
        module: m.text("object.module")?.unwrap_or(DEFAULT_MODULE).to_owned(),
        cell,
        doc: doc(&imp.attrs),
        methods,
        consts,
        as_dyn,
    })
}

/// `#[host(method(send = <expr>))]` on a body-less method (`pub fn f(&self, ..) -> R;`): the body
/// is `self.send(<expr>)` written for it, the expression built from the parameters. Returning
/// `Result<_, E>` the refusal is returned (`?`, converted by `From`); returning nothing it is
/// dropped. Run on every `#[host(object)]` impl before anything else reads it -- the expansion,
/// the strip and the codegen -- so each sees an ordinary method. A body-less fn without `send`, and
/// `send` beside a written body, are refused.
pub fn fill_sends(imp: &mut syn::ItemImpl) -> syn::Result<()> {
    struct Bodiless {
        attrs: Vec<syn::Attribute>,
        vis: syn::Visibility,
        sig: syn::Signature,
    }
    impl syn::parse::Parse for Bodiless {
        fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
            let attrs = input.call(syn::Attribute::parse_outer)?;
            let vis = input.parse()?;
            let sig = input.parse()?;
            input.parse::<syn::Token![;]>()?;
            Ok(Bodiless { attrs, vis, sig })
        }
    }
    let send_of = |attrs: &[syn::Attribute]| -> syn::Result<Option<TokenStream>> {
        if !attrs.iter().any(is_host) {
            return Ok(None);
        }
        Metas::of(attrs)?.expr("method.send")
    };
    for item in &mut imp.items {
        match item {
            syn::ImplItem::Fn(f) => {
                if send_of(&f.attrs)?.is_some() {
                    return Err(syn::Error::new_spanned(&f.sig.ident, "`send` writes the body: declare the method without one (`fn f(..) -> R;`)"));
                }
            }
            syn::ImplItem::Verbatim(tokens) => {
                let Ok(Bodiless { mut attrs, vis, sig }) = syn::parse2::<Bodiless>(tokens.clone()) else { continue };
                let Some(send) = send_of(&attrs)? else {
                    return Err(syn::Error::new_spanned(&sig.ident, "a method without a body needs #[host(method(send = <expr>))], which writes it"));
                };
                if !matches!(sig.inputs.first(), Some(syn::FnArg::Receiver(r)) if r.reference.is_some()) {
                    return Err(syn::Error::new_spanned(&sig.ident, "`send` calls `self.send(..)`: the method takes `&self`"));
                }
                let block: syn::Block = match &sig.output {
                    syn::ReturnType::Default => syn::parse_quote!({ let _ = self.send(#send); }),
                    syn::ReturnType::Type(_, t) if result_ok(t).is_some() => {
                        // `?` converts the refusal; it is the identity when the two types agree.
                        attrs.push(syn::parse_quote!(#[allow(clippy::needless_question_mark)]));
                        syn::parse_quote!({ ::core::result::Result::Ok(self.send(#send)?) })
                    }
                    syn::ReturnType::Type(..) => {
                        return Err(syn::Error::new_spanned(&sig.output, "a `send` method returns nothing (the refusal dropped) or a `Result` (the refusal returned)"));
                    }
                };
                *item = syn::ImplItem::Fn(syn::ImplItemFn { attrs, vis, defaultness: None, sig, block });
            }
            _ => {}
        }
    }
    Ok(())
}

fn returns_self(ret: &Ret) -> bool {
    let (Ret::Plain(t) | Ret::Result(t)) = ret else { return false };
    matches!(t, syn::Type::Path(p) if p.path.is_ident("Self"))
}

/// A zero-argument host function (or attribute) returning a const's value.
#[derive(Clone)]
pub struct ConstExport {
    /// `X`, or `<Type>::X`.
    pub path: TokenStream,
    pub rust: syn::Ident,
    /// `name = ".."`, else the identifier lower-cased.
    pub name: String,
    pub ty: syn::Type,
    pub pid: bool,
    pub doc: Vec<String>,
    /// An UPPER_CASE name: a module attribute, not a function.
    pub attr: bool,
}

/// `owner` is the impl's type for an associated const.
pub fn read_const(
    m: &Metas,
    ident: &syn::Ident,
    ty: &syn::Type,
    attrs: &[syn::Attribute],
    owner: Option<&syn::Type>,
) -> syn::Result<ConstExport> {
    m.only(&format!("const {ident}"), &["export", "export.name", "export.pid"])?;
    let path = match owner {
        Some(t) => quote::quote!(<#t>::#ident),
        None => quote::quote!(#ident),
    };
    let name = m.text("export.name")?.map(str::to_owned).unwrap_or_else(|| ident.to_string().to_lowercase());
    Ok(ConstExport {
        path,
        rust: ident.clone(),
        attr: name.chars().any(|c| c.is_ascii_uppercase()) && !name.chars().any(|c| c.is_ascii_lowercase()),
        name,
        ty: ty.clone(),
        pid: m.bool("export.pid")?.unwrap_or(true),
        doc: doc(attrs),
    })
}

pub fn read_export_impl(attr: TokenStream, imp: &syn::ItemImpl) -> syn::Result<Vec<ConstExport>> {
    let m = Metas::parse(attr)?;
    m.only("impl", &["export"])?;
    if imp.trait_.is_some() || !imp.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&imp.self_ty, "#[host(export)] is for an inherent, non-generic impl"));
    }
    let mut out = Vec::new();
    for item in &imp.items {
        match item {
            syn::ImplItem::Const(c) if c.attrs.iter().any(is_host) => {
                out.push(read_const(&Metas::of(&c.attrs)?, &c.ident, &c.ty, &c.attrs, Some(&imp.self_ty))?);
            }
            syn::ImplItem::Fn(f) if f.attrs.iter().any(is_host) => {
                return Err(syn::Error::new_spanned(&f.sig.ident, "an `#[host(export)] impl` exports its consts; a method belongs to a #[host(object)]"));
            }
            _ => {}
        }
    }
    Ok(out)
}

pub fn read_class(attr: TokenStream, st: &syn::ItemStruct) -> syn::Result<Class> {
    let m = Metas::parse(attr)?;
    m.only(&format!("struct {}", st.ident), &["name", "module", "mutable", "from_py_object", "eq", "subclass"])?;
    if !st.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&st.generics, "a host class is a concrete type"));
    }
    Ok(Class {
        name: m.text("name")?.map(str::to_owned).unwrap_or_else(|| st.ident.to_string()),
        module: m.text("module")?.unwrap_or(DEFAULT_MODULE).to_owned(),
        mutable: m.flag("mutable"),
        from_py_object: m.flag("from_py_object"),
        eq: m.flag("eq"),
        subclass: m.flag("subclass"),
    })
}
