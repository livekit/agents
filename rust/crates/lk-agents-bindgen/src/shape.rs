//! Structural types whose host shape is not pyo3's own and which no workspace crate can implement
//! a trait for: `Duration` (float seconds, not `timedelta`), `SystemTime` (unix seconds), JSON
//! values and `Map<String, Value>` (checked), borrowed `&[T]`/`&str`/... (taken owned, lent), and a
//! returned `impl Iterator` (a list). Recognised by last path segment, since an expansion cannot
//! resolve aliases (`Extra`, `Object`).

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub(crate) fn last(t: &syn::Type) -> Option<&syn::PathSegment> {
    match t {
        syn::Type::Path(p) if p.qself.is_none() => p.path.segments.last(),
        syn::Type::Group(g) => last(&g.elem),
        syn::Type::Paren(p) => last(&p.elem),
        _ => None,
    }
}

pub(crate) fn is(t: &syn::Type, name: &str) -> bool {
    last(t).is_some_and(|s| s.ident == name)
}

/// `[T]` of `Option<T>`.
pub(crate) fn args(t: &syn::Type) -> Vec<&syn::Type> {
    let Some(seg) = last(t) else { return Vec::new() };
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return Vec::new() };
    a.args
        .iter()
        .filter_map(|g| match g {
            syn::GenericArgument::Type(t) => Some(t),
            _ => None,
        })
        .collect()
}

pub(crate) fn one<'a>(t: &'a syn::Type, name: &str) -> Option<&'a syn::Type> {
    if !is(t, name) {
        return None;
    }
    match args(t).as_slice() {
        [inner] => Some(inner),
        _ => None,
    }
}

pub(crate) fn is_json(t: &syn::Type) -> bool {
    is(t, "Value") && args(t).is_empty()
}

/// ABI's aliases of `Map<String, Value>`.
pub(crate) fn is_json_map(t: &syn::Type) -> bool {
    ((is(t, "Extra") || is(t, "Object")) && args(t).is_empty())
        || (is(t, "Map") && matches!(args(t).as_slice(), [k, v] if is(k, "String") && is_json(v)))
}

fn iter_item(t: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::ImplTrait(it) = t else { return None };
    it.bounds.iter().find_map(|b| {
        let syn::TypeParamBound::Trait(tb) = b else { return None };
        let seg = tb.path.segments.last()?;
        if seg.ident != "Iterator" {
            return None;
        }
        let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
        a.args.iter().find_map(|g| match g {
            syn::GenericArgument::AssocType(at) if at.ident == "Item" => Some(&at.ty),
            _ => None,
        })
    })
}


/// `None`: pyo3 crosses it.
pub struct Out {
    pub ty: TokenStream,
    pub expr: TokenStream,
    pub py: bool,
    /// Contains a `?` (runs where a `PyResult` is returned).
    pub fallible: bool,
}

pub fn output(t: &syn::Type, e: TokenStream) -> Option<Out> {
    let rt = quote!(::lk_agents_abi::python);
    // A borrow of the receiver or an argument cannot outlive the call: the owned value crosses.
    // A `&'static` borrow is kept (pyo3 crosses `&'static str` itself).
    if let syn::Type::Reference(r) = t
        && r.lifetime.as_ref().is_none_or(|l| l.ident != "static")
    {
        let inner = &*r.elem;
        if is_str(inner) {
            return Some(Out { ty: quote!(::std::string::String), expr: quote!(::std::borrow::ToOwned::to_owned(#e)), py: false, fallible: false });
        }
        let (owned, expr): (syn::Type, TokenStream) = match inner {
            syn::Type::Slice(s) => {
                let elem = &s.elem;
                (syn::parse_quote!(::std::vec::Vec<#elem>), quote!(#e.to_vec()))
            }
            other => (other.clone(), quote!(::std::clone::Clone::clone(#e))),
        };
        return Some(match output(&owned, quote!(v)) {
            None => Out { ty: quote!(#owned), expr, py: false, fallible: false },
            Some(o) => {
                let x = o.expr;
                Out { ty: o.ty, expr: quote!({ let v = #expr; #x }), py: o.py, fallible: o.fallible }
            }
        });
    }
    if is(t, "Duration") {
        return Some(Out { ty: quote!(f64), expr: quote!(#e.as_secs_f64()), py: false, fallible: false });
    }
    // A side-table key is the host object it names.
    if is(t, "HostRef") || is(t, "MediaRef") {
        return Some(Out { ty: quote!(::pyo3::Py<::pyo3::PyAny>), expr: quote!(#rt::resolve_key(py, #e.id())?), py: true, fallible: true });
    }
    // An `Arc<T>` of an `arc`-cell class: that class over the same `Arc`, a shared handle.
    if let Some(inner) = one(t, "Arc")
        && !matches!(inner, syn::Type::TraitObject(_))
    {
        return Some(Out {
            ty: quote!(::pyo3::Bound<'py, ::pyo3::PyAny>),
            expr: quote!(<#inner as #rt::ArcClass>::wrap(py, #e)?),
            py: true,
            fallible: true,
        });
    }
    if is(t, "SystemTime") {
        return Some(Out { ty: quote!(f64), expr: quote!(#rt::wall_to_py(#e)), py: false, fallible: false });
    }
    if is_json(t) {
        return Some(Out {
            ty: quote!(::pyo3::Bound<'py, ::pyo3::PyAny>),
            expr: quote!(#rt::value_to_py(py, &#e)?),
            py: true,
            fallible: true,
        });
    }
    if is_json_map(t) {
        return Some(Out {
            ty: quote!(::pyo3::Bound<'py, ::pyo3::types::PyDict>),
            expr: quote!(#rt::extra_to_py(py, &#e)?),
            py: true,
            fallible: true,
        });
    }
    if let Some(item) = iter_item(t) {
        let inner = output(item, quote!(v));
        return Some(match inner {
            None => Out { ty: quote!(::std::vec::Vec<#item>), expr: quote!(#e.collect::<::std::vec::Vec<_>>()), py: false, fallible: false },
            Some(i) => collect(e, i),
        });
    }
    if let Some(inner) = one(t, "Option") {
        let i = output(inner, quote!(v))?;
        let (ty, x) = (&i.ty, &i.expr);
        let expr = if i.fallible {
            let ok = ok(x);
            quote!(#e.map(|v| #ok).transpose()?)
        } else {
            quote!(#e.map(|v| #x))
        };
        return Some(Out { ty: quote!(::std::option::Option<#ty>), expr, py: i.py, fallible: i.fallible });
    }
    if let Some(inner) = one(t, "Vec") {
        let i = output(inner, quote!(v))?;
        return Some(collect(e, i));
    }
    if is(t, "HashMap")
        && let [k, v] = args(t).as_slice()
    {
        let i = output(v, quote!(v))?;
        let (ty, x) = (&i.ty, &i.expr);
        let expr = if i.fallible {
            quote!(#e.into_iter().map(|(k, v)| ::std::result::Result::Ok::<_, ::pyo3::PyErr>((k, #x))).collect::<::pyo3::PyResult<::std::collections::HashMap<_, _>>>()?)
        } else {
            quote!(#e.into_iter().map(|(k, v)| (k, #x)).collect::<::std::collections::HashMap<_, _>>())
        };
        return Some(Out { ty: quote!(::std::collections::HashMap<#k, #ty>), expr, py: i.py, fallible: i.fallible });
    }
    if let syn::Type::Tuple(tu) = t
        && !tu.elems.is_empty()
    {
        let names: Vec<_> = (0..tu.elems.len()).map(|i| format_ident!("v{i}")).collect();
        let parts: Vec<_> = tu.elems.iter().zip(&names).map(|(t, n)| (t, output(t, quote!(#n)))).collect();
        if parts.iter().all(|(_, o)| o.is_none()) {
            return None;
        }
        let tys = parts.iter().map(|(t, o)| o.as_ref().map_or_else(|| quote!(#t), |o| o.ty.clone()));
        let xs = parts.iter().zip(&names).map(|((_, o), n)| o.as_ref().map_or_else(|| quote!(#n), |o| o.expr.clone()));
        let py = parts.iter().any(|(_, o)| o.as_ref().is_some_and(|o| o.py));
        let fallible = parts.iter().any(|(_, o)| o.as_ref().is_some_and(|o| o.fallible));
        return Some(Out {
            ty: quote!((#(#tys,)*)),
            expr: quote!({ let (#(#names,)*) = #e; (#(#xs,)*) }),
            py,
            fallible,
        });
    }
    None
}

/// `Ok(x)` for a fallible `x`, bound first (`clippy::needless_question_mark` in the caller crate).
pub fn ok(x: &TokenStream) -> TokenStream {
    quote!({ let x = #x; ::std::result::Result::Ok::<_, ::pyo3::PyErr>(x) })
}

fn collect(e: TokenStream, i: Out) -> Out {
    let (ty, x) = (&i.ty, &i.expr);
    let expr = if i.fallible {
        let ok = ok(x);
        quote!(#e.into_iter().map(|v| #ok).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>()?)
    } else {
        quote!(#e.into_iter().map(|v| #x).collect::<::std::vec::Vec<_>>())
    };
    Out { ty: quote!(::std::vec::Vec<#ty>), expr, py: i.py, fallible: i.fallible }
}


/// Host type, the statement rebinding it to the core value (`None`: pyo3 extracts it), and how
/// the core call is handed it.
pub struct In {
    pub ty: TokenStream,
    pub prelude: Option<TokenStream>,
    pub arg: TokenStream,
    pub py: bool,
}

pub fn input(name: &syn::Ident, label: &str, t: &syn::Type) -> In {
    // A borrow the core wants of a value pyo3 extracts owned: take it owned, lend it.
    if let syn::Type::Reference(r) = t
        && r.mutability.is_none()
    {
        let inner = &*r.elem;
        if let syn::Type::Slice(s) = inner {
            let elem = &*s.elem;
            // `&[&str]`: a list of `str`, borrowed element-wise.
            if let syn::Type::Reference(er) = elem
                && is_str(&er.elem)
            {
                return In {
                    ty: quote!(::std::vec::Vec<::std::string::String>),
                    prelude: Some(quote!(let #name: ::std::vec::Vec<&str> = #name.iter().map(::std::string::String::as_str).collect();)),
                    arg: quote!(&#name),
                    py: false,
                };
            }
            if is(elem, "u8") {
                return native(name, t);
            }
            let v: syn::Type = syn::parse_quote!(::std::vec::Vec<#elem>);
            let mut i = input(name, label, &v);
            i.arg = quote!(&#name);
            return i;
        }
        if is_str(inner) {
            return native(name, t);
        }
        if is_json(inner) || is_json_map(inner) || ["String", "Vec", "HashMap", "Option", "Duration", "SystemTime"].iter().any(|n| is(inner, n)) {
            let mut i = input(name, label, inner);
            i.arg = quote!(&#name);
            return i;
        }
        // A class borrowed off its host object, an `arc` cell's shared value, or any other type's
        // own reading, lent (`lk_agents_abi::python::ArgRef`).
        if is_opaque(inner) {
            return In {
                ty: quote!(&::pyo3::Bound<'py, ::pyo3::PyAny>),
                prelude: Some(quote!(let #name = {
                    #[allow(unused_imports)]
                    use ::lk_agents_abi::python::{ArgRefClass as _, ArgRefOwned as _, ArgRefShared as _};
                    (&&&::lk_agents_abi::python::ArgRef::<#inner>::new(#name, #label)).hold()?
                };)),
                arg: quote!(&*#name),
                py: true,
            };
        }
        return native(name, t);
    }
    // `Option<&T>`: convert the owned value, then lend it.
    if let Some(inner) = one(t, "Option")
        && let syn::Type::Reference(r) = inner
        && !is_str(&r.elem)
    {
        let elem = &*r.elem;
        let owned: syn::Type = syn::parse_quote!(::std::option::Option<#elem>);
        // A class (or any type's own reading) borrowed off its host object, or `None`.
        if is_opaque(elem) && !is_json(elem) && !is_json_map(elem) && !is(elem, "Duration") && !is(elem, "SystemTime") {
            return In {
                ty: quote!(::std::option::Option<&::pyo3::Bound<'py, ::pyo3::PyAny>>),
                prelude: Some(quote!(let #name = #name.map(|v| {
                    #[allow(unused_imports)]
                    use ::lk_agents_abi::python::{ArgRefClass as _, ArgRefOwned as _, ArgRefShared as _};
                    (&&&::lk_agents_abi::python::ArgRef::<#elem>::new(v, #label)).hold()
                }).transpose()?;)),
                arg: quote!(#name.as_ref().map(|v| &**v)),
                py: true,
            };
        }
        let mut i = input(name, label, &owned);
        if i.prelude.is_none() {
            return native(name, t);
        }
        i.arg = quote!(#name.as_ref());
        return i;
    }
    // A host object as the core's `Arc<dyn Trait>`: a native class registered for the trait, else
    // its `#[host(proxy)]` over the object and the loop running now.
    if let Some(tr) = arc_dyn(t) {
        return In {
            ty: quote!(&::pyo3::Bound<'py, ::pyo3::PyAny>),
            prelude: Some(quote!(let #name = <dyn #tr as ::lk_agents_abi::python::HostProxy>::from_host(#name, ::std::option::Option::None)?;)),
            arg: quote!(#name),
            py: true,
        };
    }
    if let Some(inner) = one(t, "Option")
        && let Some(tr) = arc_dyn(inner)
    {
        return In {
            ty: quote!(::std::option::Option<::pyo3::Bound<'py, ::pyo3::PyAny>>),
            prelude: Some(quote!(let #name = #name.map(|v| <dyn #tr as ::lk_agents_abi::python::HostProxy>::from_host(&v, ::std::option::Option::None)).transpose()?;)),
            arg: quote!(#name),
            py: true,
        };
    }
    // `Maybe<Arc<dyn Trait>>` / `Maybe<Option<Arc<dyn Trait>>>`: `NOT_GIVEN` keeps the leg, `None`
    // (the latter) removes it, an object replaces it.
    if let Some(inner) = one(t, "Maybe") {
        let from_host = |tr: &syn::Path| quote!(<dyn #tr as ::lk_agents_abi::python::HostProxy>::from_host(&v, ::std::option::Option::None));
        let (ty, given) = if let Some(tr) = arc_dyn(inner) {
            let f = from_host(tr);
            (quote!(::pyo3::Bound<'py, ::pyo3::PyAny>), quote!(#f?))
        } else if let Some(tr) = one(inner, "Option").and_then(arc_dyn) {
            let f = from_host(tr);
            (quote!(::std::option::Option<::pyo3::Bound<'py, ::pyo3::PyAny>>), quote!(v.map(|v| #f).transpose()?))
        } else {
            (TokenStream::new(), TokenStream::new())
        };
        if !ty.is_empty() {
            return In {
                ty: quote!(::lk_agents_abi::Maybe<#ty>),
                prelude: Some(quote!(let #name = match #name {
                    ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven,
                    ::lk_agents_abi::Maybe::Given(v) => ::lk_agents_abi::Maybe::Given(#given),
                };)),
                arg: quote!(#name),
                py: true,
            };
        }
    }
    if let Some(inner) = one(t, "Vec")
        && let Some(tr) = arc_dyn(inner)
    {
        return In {
            ty: quote!(::std::vec::Vec<::pyo3::Bound<'py, ::pyo3::PyAny>>),
            prelude: Some(quote!(let #name = #name.iter().map(|v| <dyn #tr as ::lk_agents_abi::python::HostProxy>::from_host(v, ::std::option::Option::None)).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>()?;)),
            arg: quote!(#name),
            py: true,
        };
    }
    // A side-table key: the host object itself, interned.
    for (key, intern) in [("HostRef", quote!(intern_host)), ("MediaRef", quote!(intern_media))] {
        if is(t, key) {
            return In {
                ty: quote!(::pyo3::Py<::pyo3::PyAny>),
                prelude: Some(quote!(let #name = ::lk_agents_abi::python::#intern(#name);)),
                arg: quote!(#name),
                py: false,
            };
        }
        if one(t, "Option").is_some_and(|i| is(i, key)) {
            return In {
                ty: quote!(::std::option::Option<::pyo3::Py<::pyo3::PyAny>>),
                prelude: Some(quote!(let #name = #name.map(::lk_agents_abi::python::#intern);)),
                arg: quote!(#name),
                py: false,
            };
        }
        if one(t, "Vec").is_some_and(|i| is(i, key)) {
            return In {
                ty: quote!(::std::vec::Vec<::pyo3::Py<::pyo3::PyAny>>),
                prelude: Some(quote!(let #name: ::std::vec::Vec<_> = #name.into_iter().map(::lk_agents_abi::python::#intern).collect();)),
                arg: quote!(#name),
                py: false,
            };
        }
    }
    // `impl IntoIterator<Item = X>`: a list, handed over as the `Vec` (`impl Into<String>` items
    // are `str`).
    if let Some(item) = impl_item(t, "IntoIterator") {
        let item: syn::Type = match impl_item_is_into_string(item) {
            true => syn::parse_quote!(::std::string::String),
            false => item.clone(),
        };
        let v: syn::Type = syn::parse_quote!(::std::vec::Vec<#item>);
        return input(name, label, &v);
    }
    if impl_item_is_into_string(t) {
        return In { ty: quote!(::std::string::String), prelude: None, arg: quote!(#name), py: false };
    }
    if one(t, "Option").is_some_and(impl_item_is_into_string) {
        return In { ty: quote!(::std::option::Option<::std::string::String>), prelude: None, arg: quote!(#name), py: false };
    }
    match leaf_in(t, label) {
        Some((ty, conv, py)) => In { ty, prelude: Some(quote!(let #name = { let v = #name; #conv };)), arg: quote!(#name), py },
        None => native(name, t),
    }
}

fn arc_dyn(t: &syn::Type) -> Option<&syn::Path> {
    let inner = one(t, "Arc")?;
    let syn::Type::TraitObject(o) = inner else { return None };
    o.bounds.iter().find_map(|b| match b {
        syn::TypeParamBound::Trait(tb) => Some(&tb.path),
        _ => None,
    })
}

fn impl_item<'a>(t: &'a syn::Type, trait_name: &str) -> Option<&'a syn::Type> {
    let syn::Type::ImplTrait(it) = t else { return None };
    it.bounds.iter().find_map(|b| {
        let syn::TypeParamBound::Trait(tb) = b else { return None };
        let seg = tb.path.segments.last()?;
        if seg.ident != trait_name {
            return None;
        }
        let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
        a.args.iter().find_map(|g| match g {
            syn::GenericArgument::AssocType(at) if at.ident == "Item" => Some(&at.ty),
            _ => None,
        })
    })
}

fn impl_item_is_into_string(t: &syn::Type) -> bool {
    let syn::Type::ImplTrait(it) = t else { return false };
    it.bounds.iter().any(|b| {
        matches!(b, syn::TypeParamBound::Trait(tb) if tb.path.segments.last().is_some_and(|s| {
            s.ident == "Into" && matches!(&s.arguments, syn::PathArguments::AngleBracketed(a) if a.args.len() == 1 && matches!(a.args.first(), Some(syn::GenericArgument::Type(t)) if is(t, "String")))
        }))
    })
}

/// Types pyo3 extracts and reports itself. Any other named type is read through
/// `lk_agents_abi::python::Arg` (its `HostStr` reading if it has one).
const NATIVE: &[&str] = &[
    "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32", "f64",
    "String", "str", "Vec", "HashMap", "HashSet", "BTreeMap", "Option", "Bound", "Borrowed", "Py", "PyRef", "PyRefMut",
    "PyObject", "Self", "Cow", "Box", "Arc",
];

pub fn is_opaque(t: &syn::Type) -> bool {
    match last(t) {
        Some(seg) => !NATIVE.iter().any(|n| seg.ident == n) && matches!(seg.arguments, syn::PathArguments::None),
        None => false,
    }
}

/// Over `v: &Bound<'py, PyAny>`.
fn arg(t: &syn::Type, label: &str) -> TokenStream {
    quote! {{
        #[allow(unused_imports)]
        use ::lk_agents_abi::python::{ArgPlain as _, ArgStr as _};
        (&::lk_agents_abi::python::Arg::<#t>::new(v, #label)).get()?
    }}
}

fn is_str(t: &syn::Type) -> bool {
    matches!(t, syn::Type::Path(p) if p.path.is_ident("str"))
}

fn native(name: &syn::Ident, t: &syn::Type) -> In {
    In { ty: quote!(#t), prelude: None, arg: quote!(#name), py: false }
}

/// `(host type, expression over `v`, needs 'py)`, or `None` when pyo3 extracts the core type. The
/// expression may use `?`.
fn leaf_in(t: &syn::Type, label: &str) -> Option<(TokenStream, TokenStream, bool)> {
    let rt = quote!(::lk_agents_abi::python);
    if is(t, "Duration") {
        return Some((
            quote!(f64),
            quote!(::std::time::Duration::try_from_secs_f64(v).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(::std::format!("{}: {}", #label, e)))?),
            false,
        ));
    }
    if is(t, "SystemTime") {
        return Some((quote!(f64), quote!(#rt::wall_from_py(v, #label)?), false));
    }
    if is_json(t) {
        return Some((quote!(&::pyo3::Bound<'py, ::pyo3::PyAny>), quote!(#rt::value_from_py(v, #label)?), true));
    }
    if is_json_map(t) {
        return Some((quote!(&::pyo3::Bound<'py, ::pyo3::types::PyDict>), quote!(#rt::map_from_py(v, #label)?), true));
    }
    if is_opaque(t) {
        return Some((quote!(&::pyo3::Bound<'py, ::pyo3::PyAny>), arg(t, label), true));
    }
    if let Some(inner) = one(t, "Option") {
        let (ty, conv, py) = leaf_in(inner, label)?;
        let ok = ok(&conv);
        return Some((quote!(::std::option::Option<#ty>), quote!(v.map(|v| #ok).transpose()?), py));
    }
    if let Some(inner) = one(t, "Vec") {
        let (ty, conv, py) = leaf_in(inner, label)?;
        // `&Bound` elements come in owned: `Vec<Bound<'py, PyAny>>`, each lent to the reader.
        let (ty, conv) = match strip_ref(&ty) {
            Some(owned) => (owned, quote!({ let v = &v; #conv })),
            None => (ty, conv),
        };
        let ok = ok(&conv);
        return Some((quote!(::std::vec::Vec<#ty>), quote!(v.into_iter().map(|v| #ok).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>()?), py));
    }
    None
}

/// Over tokens (host types are written as `&::pyo3::Bound<..>`).
fn strip_ref(ty: &TokenStream) -> Option<TokenStream> {
    let t: syn::Type = syn::parse2(ty.clone()).ok()?;
    match t {
        syn::Type::Reference(r) => {
            let e = r.elem;
            Some(quote!(#e))
        }
        _ => None,
    }
}
