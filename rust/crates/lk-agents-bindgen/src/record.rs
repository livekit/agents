//! `#[derive(Host)]` records: their host shapes, written beside them under `python`. `emit`:
//! `IntoPyObject`, a dict in declaration order. `extract`: `FromPyObject` from attributes (items
//! with `from_items`). `class`: getters, keyword constructor and `__repr__`. Error texts match
//! pyo3's derive (`failed to extract field <Record>.<field>`).
//!
//! A field crosses by its own type's traits, except structural types ([`emit_value`] /
//! [`extract_value`]). Cross-item facts are traits: `HostReadTag`/`HostEmitTag` for union
//! discriminants, `FlatRecord` for `#[binding(fields_only)]`.
//!
//! `model_dump`, whatever the feature: `lk_agents_abi::ModelDump`, pydantic's
//! `model_dump(mode="json", exclude_none=True)` in the host class's field order.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::shape::{args, is, is_json, is_json_map, one};
use crate::{Metas, doc};


#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Extract,
    Emit,
    Class,
}

#[derive(Clone)]
enum TagValue {
    Lit(String),
    /// `tag_const = "TYPE"`: `Self::TYPE`.
    Const(syn::Ident),
}

#[derive(Clone)]
struct Tag {
    key: String,
    value: TagValue,
}

impl Tag {
    /// `owner` is the type whose const `tag_const` names.
    fn value(&self, owner: &syn::Ident) -> TokenStream {
        match &self.value {
            TagValue::Lit(s) => quote!(#s),
            TagValue::Const(c) => quote!(<#owner>::#c),
        }
    }
}

struct Target {
    kind: Kind,
    /// The core type an emit target projects, when it is not the record itself.
    from: Option<syn::Path>,
    from_items: bool,
    new: bool,
}

#[derive(Clone)]
enum Dflt {
    None,
    Bare,
    Expr(TokenStream),
}

struct Field {
    ident: syn::Ident,
    key: String,
    ty: syn::Type,
    private: bool,
    emit_skip: bool,
    extract_skip: bool,
    skip_if_none: bool,
    default: Dflt,
    flatten: bool,
    py: bool,
    /// `#[binding(dump_with = "..")]`: `fn(&Record, &DumpCx) -> Dumped` renders the field.
    dump_with: Option<syn::Path>,
    doc: Vec<String>,
}

impl Field {
    fn emitted(&self) -> bool {
        !self.private && !self.emit_skip
    }
    fn read(&self) -> bool {
        !self.private && !self.extract_skip
    }
    fn defaulted(&self) -> bool {
        !matches!(self.default, Dflt::None)
    }
}

struct Variant {
    ident: syn::Ident,
    ty: syn::Type,
    /// `#[binding(twin = "..")]`: emits through that declaration's `host_dict(py, &member)`.
    twin: Option<syn::Path>,
}

enum Body {
    Struct(Vec<Field>),
    Union(Vec<Variant>),
    Unit,
}

struct Record {
    ident: syn::Ident,
    body: Body,
    targets: Vec<Target>,
    binding_tag: Option<Tag>,
    python_tag: Option<Tag>,
    node_tag: Option<Tag>,
    /// `#[binding(read_tag = "kind")]`: the key an extract target reads its tag under, when the
    /// dict it reads keys the kind differently from the one it emits.
    read_key: Option<String>,
    fields_only: bool,
    /// `#[binding(model_dump)]`, and the host class's `tag_after`.
    model_dump: bool,
    tag_after: Option<String>,
}

impl Record {
    fn target(&self, kind: Kind) -> Option<&Target> {
        self.targets.iter().find(|t| t.kind == kind)
    }

    /// The binding's own, else python's, then node's (the codegen's `decl_tag`).
    fn dict_tag(&self) -> Option<&Tag> {
        self.binding_tag.as_ref().or(self.python_tag.as_ref()).or(self.node_tag.as_ref())
    }

    /// The host class's literal first.
    fn object_tag(&self) -> Option<&Tag> {
        self.python_tag.as_ref().or(self.dict_tag())
    }
}


/// What a field without its own word inherits from its record, read the same way by the binding
/// and by `lk-agents-codegen`:
///
/// * `#[binding(default)]` on the record: every field reads as `default`;
/// * a `#[python(typed_dict, total = "False")]` record with a read target: the same (any key may be
///   absent);
/// * `#[binding(skip_if_none)]` on the record: every `Option` field is `skip_if_none`;
/// * `#[python(zero_defaults)]`: every number and `Option` field defaults to its `Default` on the
///   host (`0`, `0.0`, `None`), unless it is `#[python(required)]` or says its own.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecordDefaults {
    pub default: bool,
    pub skip_if_none: bool,
    pub zero: bool,
}

impl RecordDefaults {
    pub fn of(attrs: &[syn::Attribute]) -> syn::Result<RecordDefaults> {
        let b = Metas::named(attrs, "binding")?;
        let py = Metas::named(attrs, "python")?;
        let open_dict = py.flag("typed_dict") && py.text("total")? == Some("False");
        Ok(RecordDefaults {
            default: b.flag("default") || (open_dict && b.flag("extract")),
            skip_if_none: b.flag("skip_if_none"),
            zero: py.flag("zero_defaults"),
        })
    }
}

fn tag_of(ms: &Metas, at: &syn::Ident) -> syn::Result<Option<Tag>> {
    let Some(key) = ms.text("tag")? else { return Ok(None) };
    let value = match (ms.text("tag_value")?, ms.ident("tag_const")?) {
        (Some(v), _) => TagValue::Lit(v.to_owned()),
        (None, Some(c)) => TagValue::Const(c),
        (None, None) => return Err(syn::Error::new_spanned(at, format!("tag = \"{key}\" needs tag_value or tag_const"))),
    };
    Ok(Some(Tag { key: key.to_owned(), value }))
}

fn read(input: &syn::DeriveInput) -> syn::Result<Record> {
    let ident = input.ident.clone();
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&input.generics, "#[derive(Host)] is for a concrete type"));
    }
    let b = Metas::named(&input.attrs, "binding")?;
    let py = Metas::named(&input.attrs, "python")?;
    let node = Metas::named(&input.attrs, "node")?;
    b.only(
        &format!("#[binding] on {ident}"),
        &["emit", "extract", "class", "new", "from_items", "fields_only", "from", "tag", "tag_value", "tag_const", "default", "skip_if_none", "read_tag", "model_dump"],
    )?;
    let inherit = RecordDefaults::of(&input.attrs)?;
    let from = b.path("from")?;
    if from.as_ref().is_some_and(|p| p.segments.last().is_some_and(|s| s.ident == ident)) {
        return Err(syn::Error::new_spanned(&ident, "`from` names the record itself: leave it out"));
    }
    let mut targets = Vec::new();
    for (word, kind) in [("extract", Kind::Extract), ("emit", Kind::Emit), ("class", Kind::Class)] {
        if !b.flag(word) {
            continue;
        }
        targets.push(Target {
            kind,
            from: if kind == Kind::Emit { from.clone() } else { None },
            from_items: b.flag("from_items"),
            new: b.flag("new"),
        });
    }
    if from.is_some() && targets.iter().any(|t| t.kind != Kind::Emit) {
        return Err(syn::Error::new_spanned(&ident, "`from` is an emit target's projection; an extract or class target is the record itself"));
    }
    let body = match &input.data {
        syn::Data::Struct(s) => {
            let mut fields = Vec::new();
            for f in &s.fields {
                fields.push(field(f, inherit)?);
            }
            Body::Struct(fields)
        }
        syn::Data::Enum(e) if e.variants.iter().all(|v| matches!(v.fields, syn::Fields::Unit)) => Body::Unit,
        syn::Data::Enum(e) => {
            let mut vs = Vec::new();
            let mut skipped = false;
            for v in &e.variants {
                // `#[binding(skip)]`: a variant not on the wire (an extract union reads no such kind).
                if Metas::named(&v.attrs, "binding")?.flag("skip") {
                    skipped = true;
                    continue;
                }
                let syn::Fields::Unnamed(u) = &v.fields else {
                    return Err(syn::Error::new_spanned(&v.ident, "a host union variant carries one record"));
                };
                if u.unnamed.len() != 1 {
                    return Err(syn::Error::new_spanned(&v.ident, "a host union variant carries one record"));
                }
                let twin = Metas::named(&v.attrs, "binding")?.path("twin")?;
                vs.push(Variant { ident: v.ident.clone(), ty: u.unnamed[0].ty.clone(), twin });
            }
            if skipped && b.flag("emit") {
                return Err(syn::Error::new_spanned(&ident, "an emit union writes every variant: #[binding(skip)] is for an extract union"));
            }
            Body::Union(vs)
        }
        syn::Data::Union(_) => return Err(syn::Error::new_spanned(&ident, "#[derive(Host)] is for a struct or an enum")),
    };
    Ok(Record {
        binding_tag: tag_of(&b, &ident)?,
        python_tag: tag_of(&py, &ident)?,
        node_tag: tag_of(&node, &ident)?,
        read_key: b.text("read_tag")?.map(str::to_owned),
        fields_only: b.flag("fields_only"),
        model_dump: b.flag("model_dump"),
        tag_after: py.text("tag_after")?.map(str::to_owned),
        ident,
        body,
        targets,
    })
}

fn field(f: &syn::Field, inherit: RecordDefaults) -> syn::Result<Field> {
    let ident = f.ident.clone().ok_or_else(|| syn::Error::new_spanned(f, "a host record has named fields"))?;
    let b = Metas::named(&f.attrs, "binding")?;
    let h = Metas::of(&f.attrs)?;
    let p = Metas::named(&f.attrs, "python")?;
    for m in &b.0 {
        if !["skip", "extract_skip", "skip_if_none", "default", "flatten", "py", "dump_with"].contains(&m.key.as_str()) {
            return Err(syn::Error::new_spanned(
                &ident,
                format!("#[binding({})] is not a field word: a value with its own host shape implements IntoPyObject/FromPyObject in its crate", m.key),
            ));
        }
    }
    let default = match (b.flag("default") || inherit.default, b.expr("default")?) {
        (_, Some(e)) => Dflt::Expr(e),
        (true, None) => Dflt::Bare,
        (false, None) => Dflt::None,
    };
    let key = unraw(h.text("rename")?.unwrap_or(&ident.to_string()));
    Ok(Field {
        key,
        ty: f.ty.clone(),
        private: p.flag("private"),
        emit_skip: b.flag("skip"),
        extract_skip: b.flag("extract_skip"),
        skip_if_none: b.flag("skip_if_none") || (inherit.skip_if_none && one(&f.ty, "Option").is_some()),
        default,
        flatten: b.flag("flatten"),
        py: b.flag("py"),
        dump_with: b.path("dump_with")?,
        doc: doc(&f.attrs),
        ident,
    })
}

fn unraw(s: &str) -> String {
    s.strip_prefix("r#").unwrap_or(s).to_owned()
}


/// Types no crate here can implement a pyo3 trait for: [`emit_value`]/[`extract_value`] handle them.
fn structural(t: &syn::Type) -> bool {
    if is(t, "Duration") || is(t, "SystemTime") || is_json(t) || is_json_map(t) || one(t, "Box").is_some() {
        return true;
    }
    if let Some(i) = one(t, "Option").or_else(|| one(t, "Vec")) {
        return structural(i);
    }
    if (is(t, "HashMap") || is(t, "BTreeMap")) && let [_, v] = args(t).as_slice() {
        return structural(v);
    }
    if let syn::Type::Tuple(tu) = t {
        return tu.elems.iter().any(structural);
    }
    false
}

/// `e: &T` as a value pyo3 converts; non-structural types are cloned into their own `IntoPyObject`.
fn emit_value(t: &syn::Type, e: TokenStream) -> TokenStream {
    let rt = quote!(::lk_agents_abi::python);
    if let Some(inner) = one(t, "Box") {
        return emit_value(inner, quote!(&**#e));
    }
    if is(t, "Duration") {
        return quote!((#e).as_secs_f64());
    }
    if is(t, "SystemTime") {
        return quote!(#rt::wall_to_py(*#e));
    }
    if is_json(t) {
        return quote!(#rt::value_to_py(py, #e)?);
    }
    if is_json_map(t) {
        return quote!(#rt::extra_to_py(py, #e)?);
    }
    if structural(t) {
        if let Some(inner) = one(t, "Option") {
            let x = emit_value(inner, quote!(v));
            return quote!(match #e { ::std::option::Option::Some(v) => ::std::option::Option::Some(#x), ::std::option::Option::None => ::std::option::Option::None });
        }
        if let Some(inner) = one(t, "Vec") {
            let x = emit_value(inner, quote!(v));
            let ok = crate::shape::ok(&x);
            return quote!(#e.iter().map(|v| #ok).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>()?);
        }
        if let [_, v] = args(t).as_slice() {
            let x = emit_value(v, quote!(v));
            let map = if is(t, "BTreeMap") { quote!(::std::collections::BTreeMap) } else { quote!(::std::collections::HashMap) };
            return quote!(#e.iter().map(|(k, v)| ::std::result::Result::Ok::<_, ::pyo3::PyErr>((::std::clone::Clone::clone(k), #x))).collect::<::pyo3::PyResult<#map<_, _>>>()?);
        }
        if let syn::Type::Tuple(tu) = t {
            let parts = tu.elems.iter().enumerate().map(|(i, t)| {
                let i = syn::Index::from(i);
                emit_value(t, quote!(&(#e).#i))
            });
            return quote!((#(#parts,)*));
        }
    }
    quote!(::std::clone::Clone::clone(#e))
}

/// `None` when the type's own `FromPyObject` reads it. `label` names the value in a JSON error.
fn extract_value(t: &syn::Type, v: TokenStream, label: &str) -> Option<TokenStream> {
    let rt = quote!(::lk_agents_abi::python);
    if !structural(t) {
        return None;
    }
    if let Some(inner) = one(t, "Box") {
        let x = extract_value(inner, v.clone(), label).unwrap_or_else(|| quote!(#v.extract::<#inner>().map_err(::std::convert::Into::<::pyo3::PyErr>::into)));
        return Some(quote!((#x).map(::std::boxed::Box::new)));
    }
    if is(t, "Duration") {
        return Some(quote!(#rt::secs_from_py(#v)));
    }
    if is(t, "SystemTime") {
        return Some(quote!(#v.extract::<f64>().and_then(|s| #rt::wall_from_py(s, #label))));
    }
    if is_json(t) {
        return Some(quote!(#rt::value_from_py(#v, #label)));
    }
    if is_json_map(t) {
        // Checked under the path `extra` whatever the field, as the codegen's `json` reader did.
        return Some(quote!(#v.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|d| #rt::extra_from_py(&d))));
    }
    if let Some(inner) = one(t, "Option") {
        let x = extract_value(inner, quote!(v), label)?;
        return Some(quote!({ let v = #v; if v.is_none() { ::std::result::Result::Ok(::std::option::Option::None) } else { (#x).map(::std::option::Option::Some) } }));
    }
    if let Some(inner) = one(t, "Vec") {
        let x = extract_value(inner, quote!(v), label)?;
        return Some(quote!(#v.extract::<::std::vec::Vec<::pyo3::Bound<'_, ::pyo3::PyAny>>>().and_then(|items| items.iter().map(|v| #x).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>())));
    }
    None
}


/// Gated on the calling crate's `python` feature.
pub fn record(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    let r = read(input)?;
    let mut out = TokenStream::new();
    match &r.body {
        Body::Unit => out.extend(unit(&r)),
        Body::Struct(fields) => {
            if r.fields_only {
                out.extend(flat(&r, fields)?);
            }
            for t in &r.targets {
                out.extend(match t.kind {
                    Kind::Emit => emit_struct(&r, t, fields)?,
                    Kind::Extract => extract_struct(&r, t, fields)?,
                    Kind::Class => class(&r, t, fields)?,
                });
            }
            out.extend(tags(&r));
        }
        Body::Union(vs) => {
            for t in &r.targets {
                out.extend(match t.kind {
                    Kind::Emit => emit_union(&r, vs)?,
                    Kind::Extract => extract_union(&r, t, vs)?,
                    Kind::Class => return Err(syn::Error::new_spanned(&r.ident, "a union is not a class")),
                });
            }
        }
    }
    let dump = match (&r.body, r.model_dump) {
        (Body::Struct(fields), true) => model_dump(&r, fields),
        (_, true) => return Err(syn::Error::new_spanned(&r.ident, "model_dump is for a struct")),
        _ => TokenStream::new(),
    };
    if out.is_empty() {
        return Ok(dump);
    }
    Ok(quote! {
        #dump
        #[cfg(feature = "python")]
        const _: () = {
            #[allow(unused_imports)]
            use ::pyo3::types::{PyAnyMethods as _, PyDictMethods as _};
            #out
        };
    })
}

/// `ModelDump`: every emitted field in declaration order, the host class's discriminant first or
/// after its `tag_after` field.
fn model_dump(r: &Record, fields: &[Field]) -> TokenStream {
    let ident = &r.ident;
    let tag = r.object_tag().map(|t| {
        let (key, value) = (&t.key, t.value(ident));
        quote!(cx.insert(&mut d, #key, ::std::option::Option::Some(::std::convert::From::from(#value)));)
    });
    let mut sets = Vec::new();
    if r.tag_after.is_none() {
        sets.extend(tag.clone());
    }
    for f in fields.iter().filter(|f| f.emitted()) {
        let (key, fi) = (&f.key, &f.ident);
        let at = format!("{ident}.{key}");
        sets.push(match &f.dump_with {
            Some(path) => quote!(cx.insert(&mut d, #key, #path(self, cx)?);),
            None => quote!(cx.insert(&mut d, #key, ::lk_agents_abi::ModelDump::model_dump(&self.#fi, cx, #at)?);),
        });
        if r.tag_after.as_deref() == Some(key.as_str()) {
            sets.extend(tag.clone());
        }
    }
    quote! {
        impl ::lk_agents_abi::ModelDump for #ident {
            fn model_dump(&self, cx: &::lk_agents_abi::DumpCx<'_>, _at: &'static str) -> ::lk_agents_abi::Dumped {
                let mut d = ::lk_agents_abi::DumpMap::new();
                #(#sets)*
                ::std::result::Result::Ok(::std::option::Option::Some(::lk_agents_abi::DumpValue::Object(d)))
            }
        }
    }
}

/// `as_str()` both ways (`FromStr`'s error is the `ValueError`).
fn unit(r: &Record) -> TokenStream {
    let ident = &r.ident;
    quote! {
        impl<'py> ::pyo3::IntoPyObject<'py> for #ident {
            type Target = ::pyo3::types::PyString;
            type Output = ::pyo3::Bound<'py, ::pyo3::types::PyString>;
            type Error = ::std::convert::Infallible;
            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::std::result::Result<::pyo3::Bound<'py, ::pyo3::types::PyString>, ::std::convert::Infallible> {
                ::std::result::Result::Ok(::pyo3::types::PyString::new(py, self.as_str()))
            }
        }
        impl<'py> ::pyo3::IntoPyObject<'py> for &#ident {
            type Target = ::pyo3::types::PyString;
            type Output = ::pyo3::Bound<'py, ::pyo3::types::PyString>;
            type Error = ::std::convert::Infallible;
            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::std::result::Result<::pyo3::Bound<'py, ::pyo3::types::PyString>, ::std::convert::Infallible> {
                ::std::result::Result::Ok(::pyo3::types::PyString::new(py, self.as_str()))
            }
        }
        impl ::lk_agents_abi::python::HostStr for #ident {}
        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #ident {
            type Error = ::pyo3::PyErr;
            fn extract(ob: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::std::result::Result<Self, ::pyo3::PyErr> {
                let s = ob.extract::<::pyo3::pybacked::PyBackedStr>().map_err(::std::convert::Into::<::pyo3::PyErr>::into)?;
                <Self as ::core::str::FromStr>::from_str(&s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(&e)))
            }
        }
    }
}

/// The discriminants a union's reader and a kind list ask the record for.
fn tags(r: &Record) -> TokenStream {
    let ident = &r.ident;
    let mut out = TokenStream::new();
    if let Some(t) = r.target(Kind::Extract) {
        let tag = if t.from_items { r.dict_tag() } else { r.object_tag() };
        if let Some(tag) = tag {
            let (key, value, items) = (r.read_key.as_ref().unwrap_or(&tag.key), tag.value(ident), t.from_items);
            out.extend(quote! {
                impl ::lk_agents_abi::python::HostReadTag for #ident {
                    const KEY: &'static str = #key;
                    const TAG: &'static str = #value;
                    const ITEMS: bool = #items;
                }
            });
        }
    }
    if r.target(Kind::Emit).is_some() {
        let value = match r.dict_tag() {
            Some(tag) => {
                let v = tag.value(ident);
                quote!(::std::option::Option::Some(#v))
            }
            None => quote!(::std::option::Option::None),
        };
        out.extend(quote! {
            impl ::lk_agents_abi::python::HostEmitTag for #ident {
                const TAG: ::std::option::Option<&'static str> = #value;
            }
        });
    }
    out
}

fn dict_body(r: &Record, fields: &[Field], py_args: bool) -> TokenStream {
    let mut sets = Vec::new();
    if let Some(tag) = r.dict_tag() {
        let (key, value) = (&tag.key, tag.value(&r.ident));
        sets.push(quote!(d.set_item(#key, #value)?;));
    }
    for f in fields.iter().filter(|f| f.emitted()) {
        let (key, ident) = (&f.key, &f.ident);
        if f.py && py_args {
            if f.skip_if_none {
                sets.push(quote!(if let ::std::option::Option::Some(o) = #ident { d.set_item(#key, o)?; }));
            } else {
                sets.push(quote!(d.set_item(#key, #ident)?;));
            }
            continue;
        }
        if f.skip_if_none {
            // `Maybe<T>`: `NotGiven` writes no key; `Option<T>`: `None` writes none.
            let inner = one(&f.ty, "Option").or_else(|| one(&f.ty, "Maybe"));
            let Some(inner) = inner else {
                sets.push(quote!(compile_error!("skip_if_none is for an Option or a Maybe field");));
                continue;
            };
            let x = emit_value(inner, quote!(x));
            let pat = if is(&f.ty, "Maybe") { quote!(::lk_agents_abi::Maybe::Given(x)) } else { quote!(::std::option::Option::Some(x)) };
            sets.push(quote!(if let #pat = &v.#ident { d.set_item(#key, #x)?; }));
        } else {
            let x = emit_value(&f.ty, quote!(&v.#ident));
            sets.push(quote!(d.set_item(#key, #x)?;));
        }
    }
    quote! {
        let d = ::pyo3::types::PyDict::new(py);
        #(#sets)*
        ::std::result::Result::Ok(d)
    }
}

fn emit_struct(r: &Record, t: &Target, fields: &[Field]) -> syn::Result<TokenStream> {
    let ident = &r.ident;
    if let Some(f) = fields.iter().find(|f| f.emitted() && f.flatten) {
        return Err(syn::Error::new_spanned(&f.ident, "#[binding(flatten)] is the read direction's; an emit target would nest the keys"));
    }
    let py_fields: Vec<&Field> = fields.iter().filter(|f| f.emitted() && f.py).collect();
    let use_dict = quote!(use ::pyo3::types::PyDictMethods as _;);
    if !py_fields.is_empty() {
        // Host objects the call site passes, in declaration order: no `IntoPyObject` can hold them.
        if t.from.is_some() {
            return Err(syn::Error::new_spanned(ident, "a #[binding(py)] record is built by the call site, not from another record"));
        }
        let params = py_fields.iter().map(|f| {
            let i = &f.ident;
            if f.skip_if_none {
                quote!(#i: ::std::option::Option<::pyo3::Bound<'py, ::pyo3::PyAny>>)
            } else {
                quote!(#i: ::pyo3::Bound<'py, ::pyo3::PyAny>)
            }
        });
        let body = dict_body(r, fields, true);
        return Ok(quote! {
            impl #ident {
                /// This record as the host's dict, with the host objects it keys passed in.
                #[allow(clippy::too_many_arguments)]
                pub fn host_dict<'py>(&self, py: ::pyo3::Python<'py>, #(#params),*) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::types::PyDict>> {
                    #use_dict
                    let v = self;
                    #body
                }
            }
        });
    }
    let body = dict_body(r, fields, false);
    if let Some(from) = &t.from {
        // A second projection of another record: its dict, by this field list.
        return Ok(quote! {
            impl #ident {
                /// `v` as the host's dict, keyed by this declaration's fields.
                pub fn host_dict<'py>(py: ::pyo3::Python<'py>, v: &#from) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::types::PyDict>> {
                    #use_dict
                    #body
                }
            }
        });
    }
    Ok(quote! {
        impl<'py> ::pyo3::IntoPyObject<'py> for &#ident {
            type Target = ::pyo3::types::PyDict;
            type Output = ::pyo3::Bound<'py, ::pyo3::types::PyDict>;
            type Error = ::pyo3::PyErr;
            #[allow(unused_variables)]
            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::types::PyDict>> {
                #use_dict
                let v = self;
                #body
            }
        }
        impl<'py> ::pyo3::IntoPyObject<'py> for #ident {
            type Target = ::pyo3::types::PyDict;
            type Output = ::pyo3::Bound<'py, ::pyo3::types::PyDict>;
            type Error = ::pyo3::PyErr;
            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::types::PyDict>> {
                ::pyo3::IntoPyObject::into_pyobject(&self, py)
            }
        }
    })
}

/// `wrap`: wrap the failure as pyo3's derive does, else raise it as is.
fn read_value(f: &Field, v: TokenStream, owner: &TokenStream, wrap: bool) -> TokenStream {
    let key = &f.key;
    let ty = &f.ty;
    match extract_value(ty, quote!(v), key) {
        Some(x) if wrap => quote! {
            ::pyo3::impl_::frompyobject::extract_struct_field_with(|v: &::pyo3::Bound<'_, ::pyo3::PyAny>| -> ::pyo3::PyResult<#ty> { #x }, &#v, #owner, #key)?
        },
        Some(x) => quote!({ let v = &#v; #x? }),
        None if wrap => quote!(::pyo3::impl_::frompyobject::extract_struct_field::<#ty>(&#v, #owner, #key)?),
        None => quote!(#v.extract::<#ty>()?),
    }
}

fn default_of(f: &Field) -> TokenStream {
    match &f.default {
        Dflt::Expr(e) => e.clone(),
        _ => quote!(::std::default::Default::default()),
    }
}

/// `owner` is the struct name the error names; `items`: read by key.
fn attr_field(f: &Field, owner: &TokenStream, items: &TokenStream) -> TokenStream {
    let key = &f.key;
    let getter = quote! {
        if #items {
            ::pyo3::types::PyAnyMethods::get_item(obj, ::pyo3::intern!(obj.py(), #key))
        } else {
            ::pyo3::types::PyAnyMethods::getattr(obj, ::pyo3::intern!(obj.py(), #key))
        }
    };
    let x = read_value(f, quote!(value), owner, true);
    match f.default {
        Dflt::None => quote!({ let value = #getter?; #x }),
        _ => {
            let d = default_of(f);
            quote!(if let ::std::result::Result::Ok(value) = #getter { #x } else { #d })
        }
    }
}

/// In the all-optional dict reader: present keys raise their own error.
fn manual_field(f: &Field) -> TokenStream {
    let key = &f.key;
    let x = read_value(f, quote!(v), &quote!(""), false);
    let d = default_of(f);
    quote!(match ::pyo3::types::PyDictMethods::get_item(d, #key)? { ::std::option::Option::Some(v) => #x, ::std::option::Option::None => #d })
}

/// `#[binding(fields_only)]`, for `flatten`.
fn flat(r: &Record, fields: &[Field]) -> syn::Result<TokenStream> {
    let ident = &r.ident;
    let owner = quote!(owner);
    let items = quote!(items);
    let mut inits = Vec::new();
    for f in fields.iter().filter(|f| f.read()) {
        if f.flatten {
            return Err(syn::Error::new_spanned(&f.ident, "a fields_only record does not flatten another"));
        }
        let i = &f.ident;
        let x = attr_field(f, &owner, &items);
        inits.push(quote!(#i: #x));
    }
    let partial = fields.iter().any(|f| !f.read());
    let rest = partial.then(|| quote!(..::std::default::Default::default()));
    let all = fields.iter().filter(|f| f.read()).all(Field::defaulted);
    Ok(quote! {
        impl ::lk_agents_abi::python::FlatRecord for #ident {
            const ALL_DEFAULTED: bool = #all;
            fn read_flat(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, owner: &'static str, items: bool) -> ::pyo3::PyResult<Self> {
                ::std::result::Result::Ok(#ident { #(#inits,)* #rest })
            }
        }
    })
}

fn extract_struct(r: &Record, t: &Target, fields: &[Field]) -> syn::Result<TokenStream> {
    let ident = &r.ident;
    // What pyo3's field error names: `failed to extract field SpeechEvent.alternatives`.
    let name = ident.to_string();
    let owner = quote!(#name);
    let read: Vec<&Field> = fields.iter().filter(|f| f.read()).collect();
    // The dict carries its kind and nothing else: checked, nothing read.
    if read.is_empty() {
        if !(t.from_items && fields.is_empty() && r.dict_tag().is_some()) {
            return Err(syn::Error::new_spanned(ident, "this extract target reads nothing"));
        }
        return Ok(quote! {
            impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #ident {
                type Error = ::pyo3::PyErr;
                fn extract(obj: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                    (*obj).cast::<::pyo3::types::PyDict>()?;
                    ::std::result::Result::Ok(#ident {})
                }
            }
        });
    }
    // pyo3 refuses to derive an all-optional reader. A flattened record's own defaults decide it
    // too, which only it knows: asserted, not guessed.
    let flats: Vec<&Field> = read.iter().copied().filter(|f| f.flatten).collect();
    let manual = read.iter().filter(|f| !f.flatten).all(|f| f.defaulted()) && flats.is_empty();
    if manual && !t.from_items {
        return Err(syn::Error::new_spanned(ident, "every field of this object reader has a default, so it matches anything: add from_items, or require a key"));
    }
    let asserts = flats.iter().map(|f| {
        let ty = &f.ty;
        let only = read.iter().all(|g| g.flatten || g.defaulted());
        if only {
            quote!(const _: () = assert!(!<#ty as ::lk_agents_abi::python::FlatRecord>::ALL_DEFAULTED, "an all-optional flattening reader is not expanded in place");)
        } else {
            quote!()
        }
    });
    let items = t.from_items;
    let mut inits = Vec::new();
    for f in fields {
        let i = &f.ident;
        if f.private {
            continue;
        }
        if f.extract_skip {
            inits.push(quote!(#i: ::std::default::Default::default()));
            continue;
        }
        if f.flatten {
            let ty = &f.ty;
            inits.push(quote!(#i: <#ty as ::lk_agents_abi::python::FlatRecord>::read_flat(obj, #owner, #items)?));
            continue;
        }
        let x = if manual { manual_field(f) } else { attr_field(f, &owner, &quote!(#items)) };
        inits.push(quote!(#i: #x));
    }
    let rest = fields.iter().any(|f| f.private).then(|| quote!(..::std::default::Default::default()));
    let prelude = if manual {
        quote!(let d = (*obj).cast::<::pyo3::types::PyDict>()?;)
    } else {
        quote!(let obj: &::pyo3::Bound<'py, ::pyo3::PyAny> = &obj;)
    };
    Ok(quote! {
        #(#asserts)*
        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #ident {
            type Error = ::pyo3::PyErr;
            #[allow(unused_variables)]
            fn extract(obj: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                #prelude
                ::std::result::Result::Ok(#ident { #(#inits,)* #rest })
            }
        }
    })
}

/// A getter per emitted field, the keyword constructor under `new`, and `__repr__` (`Debug`).
fn class(r: &Record, t: &Target, fields: &[Field]) -> syn::Result<TokenStream> {
    let ident = &r.ident;
    let name = ident.to_string();
    let mut methods = Vec::new();
    if t.new {
        let mut sig = Vec::new();
        let mut params = Vec::new();
        let mut init = Vec::new();
        for f in fields {
            let i = &f.ident;
            if f.private || f.emit_skip {
                init.push(quote!(#i: ::std::default::Default::default()));
                continue;
            }
            let ty = &f.ty;
            let arg = format_ident!("{}", f.key);
            sig.push(match &f.default {
                Dflt::Expr(e) => quote!(#arg = #e),
                Dflt::Bare if one(ty, "Option").is_some() => quote!(#arg = None),
                Dflt::Bare => quote!(#arg = <#ty>::default()),
                Dflt::None => quote!(#arg),
            });
            params.push(quote!(#arg: #ty));
            init.push(if *i == arg { quote!(#i) } else { quote!(#i: #arg) });
        }
        let d1 = format!(" `{name}(**fields)`: every field as a keyword argument. The field list appears here");
        let d2 = " once, as the host's constructor signature, and nowhere else by hand.";
        let signature = (!sig.is_empty()).then(|| quote!(#[pyo3(signature = (*, #(#sig),*))]));
        methods.push(quote! {
            #[doc = #d1]
            #[doc = #d2]
            #[new]
            #signature
            #[allow(clippy::too_many_arguments)]
            fn __host_py_new(#(#params),*) -> Self {
                #ident { #(#init),* }
            }
        });
    }
    for f in fields.iter().filter(|f| f.emitted()) {
        if f.skip_if_none || f.py || f.flatten {
            return Err(syn::Error::new_spanned(&f.ident, "a getter is always there and reads one field: skip_if_none, py and flatten have no class meaning"));
        }
        let getter = crate::host_ident(&f.key);
        let wrapper = format_ident!("__host_py_get_{}", f.key);
        let docs = f.doc.iter().map(|l| format!(" {l}"));
        let field = &f.ident;
        let x = emit_value(&f.ty, quote!(&self.#field));
        methods.push(quote! {
            #(#[doc = #docs])*
            #[getter(#getter)]
            fn #wrapper<'py>(&self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                ::pyo3::IntoPyObjectExt::into_bound_py_any(#x, py)
            }
        });
    }
    let repr = format!("{name}({{:?}})");
    let message = format!("the record's class is `{name}`: its #[lk_agents_abi::class] names it so");
    Ok(quote! {
        const _: () = assert!(::lk_agents_abi::python::same_name(<#ident as ::pyo3::PyClass>::NAME, #name), #message);
        #[::pyo3::pymethods]
        impl #ident {
            #(#methods)*
            fn __repr__(&self) -> ::std::string::String {
                ::std::format!(#repr, self)
            }
        }
    })
}

fn emit_union(r: &Record, vs: &[Variant]) -> syn::Result<TokenStream> {
    let ident = &r.ident;
    let arms = vs.iter().map(|v| {
        let vi = &v.ident;
        match &v.twin {
            Some(p) => quote!(#ident::#vi(x) => #p::host_dict(py, x)?.into_any()),
            None => quote!(#ident::#vi(x) => ::pyo3::IntoPyObject::into_pyobject(x, py)?.into_any()),
        }
    });
    // The arm's emitting type: the member, or the projection it emits through.
    let emitters: Vec<TokenStream> = vs
        .iter()
        .map(|v| match &v.twin {
            Some(p) => quote!(#p),
            None => {
                let t = &v.ty;
                quote!(#t)
            }
        })
        .collect();
    Ok(quote! {
        impl<'py> ::pyo3::IntoPyObject<'py> for &#ident {
            type Target = ::pyo3::PyAny;
            type Output = ::pyo3::Bound<'py, ::pyo3::PyAny>;
            type Error = ::pyo3::PyErr;
            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                ::std::result::Result::Ok(match self { #(#arms,)* })
            }
        }
        impl<'py> ::pyo3::IntoPyObject<'py> for #ident {
            type Target = ::pyo3::PyAny;
            type Output = ::pyo3::Bound<'py, ::pyo3::PyAny>;
            type Error = ::pyo3::PyErr;
            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                ::pyo3::IntoPyObject::into_pyobject(&self, py)
            }
        }
        impl #ident {
            /// Every discriminant this union's members write, in declaration order, once each.
            pub fn host_kinds() -> ::std::vec::Vec<&'static str> {
                let mut out: ::std::vec::Vec<&'static str> = ::std::vec::Vec::new();
                #[allow(unused_imports)]
                use ::lk_agents_abi::python::{EmitTagDeclared as _, EmitTagNone as _};
                #(
                    if let ::std::option::Option::Some(k) = (&::lk_agents_abi::python::EmitTagOf::<#emitters>::new()).emit_tag()
                        && !out.contains(&k)
                    {
                        out.push(k);
                    }
                )*
                out
            }
        }
    })
}

fn extract_union(r: &Record, t: &Target, vs: &[Variant]) -> syn::Result<TokenStream> {
    let ident = &r.ident;
    let first = &vs.first().ok_or_else(|| syn::Error::new_spanned(ident, "a union with no arms"))?.ty;
    let rust = ident.to_string();
    let _ = t;
    let arms = vs.iter().map(|v| {
        let (vi, ty) = (&v.ident, &v.ty);
        quote! {
            if tag == <#ty as ::lk_agents_abi::python::HostReadTag>::TAG {
                return ::std::result::Result::Ok(::std::option::Option::Some(#ident::#vi(::lk_agents_abi::python::extract_generated::<#ty>(obj)?)));
            }
        }
    });
    let checks = vs.iter().map(|v| {
        let ty = &v.ty;
        quote! {
            assert!(::lk_agents_abi::python::same_name(<#ty as ::lk_agents_abi::python::HostReadTag>::KEY, <#first as ::lk_agents_abi::python::HostReadTag>::KEY), "every member of a union reads its discriminant under one key");
            assert!(<#ty as ::lk_agents_abi::python::HostReadTag>::ITEMS == <#first as ::lk_agents_abi::python::HostReadTag>::ITEMS, "every member of a union is read the same way");
        }
    });
    Ok(quote! {
        const _: () = { #(#checks)* };
        impl #ident {
            /// The member this object's discriminant names, or `None` for one this build does not know.
            pub fn extract_known(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::option::Option<Self>> {
                let tag: ::std::string::String = Self::__host_tag(obj)?;
                Self::__host_of_tag(obj, &tag)
            }

            fn __host_tag(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::string::String> {
                use ::pyo3::types::PyAnyMethods as _;
                let key = <#first as ::lk_agents_abi::python::HostReadTag>::KEY;
                if <#first as ::lk_agents_abi::python::HostReadTag>::ITEMS { obj.get_item(key)?.extract() } else { obj.getattr(key)?.extract() }
            }

            fn __host_of_tag(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, tag: &str) -> ::pyo3::PyResult<::std::option::Option<Self>> {
                #(#arms)*
                ::std::result::Result::Ok(::std::option::Option::None)
            }
        }
        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #ident {
            type Error = ::pyo3::PyErr;
            fn extract(obj: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                let tag = Self::__host_tag(&obj)?;
                Self::__host_of_tag(&obj, &tag)?.ok_or_else(|| {
                    let key = <#first as ::lk_agents_abi::python::HostReadTag>::KEY;
                    ::pyo3::exceptions::PyValueError::new_err(::std::format!("{}: unknown {} {:?}", #rust, key, tag))
                })
            }
        }
    })
}
