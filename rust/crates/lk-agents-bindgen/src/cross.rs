//! How a value crosses a proxy call: owned types by their own `IntoPyObject`/`FromPyObject`;
//! structural ones (primitives, `Duration` seconds, JSON, bytes, prost messages as bytes,
//! side-table keys, `Arc` of an `arc`-cell class, and containers of those) by the rules here.

use proc_macro2::TokenStream;
use quote::quote;

pub fn head(t: &syn::Type) -> Option<(String, Vec<syn::Type>)> {
    let syn::Type::Path(p) = t else { return None };
    let seg = p.path.segments.last()?;
    let args = match &seg.arguments {
        syn::PathArguments::AngleBracketed(a) => {
            a.args.iter().filter_map(|a| if let syn::GenericArgument::Type(t) = a { Some(t.clone()) } else { None }).collect()
        }
        _ => Vec::new(),
    };
    Some((seg.ident.to_string(), args))
}

pub fn is_unit(t: &syn::Type) -> bool {
    matches!(t, syn::Type::Tuple(tu) if tu.elems.is_empty())
}

const PRIMS: [&str; 15] = ["bool", "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize", "f32", "f64", "String", "char"];

fn is_proto(t: &syn::Type) -> bool {
    matches!(t, syn::Type::Path(p) if p.path.segments.iter().any(|s| s.ident == "proto"))
}

fn is_str(t: &syn::Type) -> bool {
    matches!(t, syn::Type::Path(p) if p.path.is_ident("str"))
}

/// `v` (`t`, or `&t` when `by_ref`) as a value `IntoPyObject` takes; may use `py` and `?`.
pub fn to_py(t: &syn::Type, v: &TokenStream, by_ref: bool) -> TokenStream {
    let rt = quote!(::lk_agents_abi::python);
    let r = if by_ref { quote!(#v) } else { quote!(&#v) };
    if is_proto(t) {
        return quote!(::pyo3::types::PyBytes::new(py, &::prost::Message::encode_to_vec(#r)));
    }
    if let syn::Type::Tuple(tu) = t
        && !tu.elems.is_empty()
    {
        let parts = tu.elems.iter().enumerate().map(|(i, e)| {
            let i = syn::Index::from(i);
            to_py(e, &quote!((#v).#i), by_ref)
        });
        return quote!((#(#parts,)*));
    }
    if let syn::Type::Reference(rf) = t {
        if let syn::Type::Slice(s) = &*rf.elem {
            let elem = &*s.elem;
            if matches!(head(elem), Some((h, _)) if h == "u8") {
                return quote!(::pyo3::types::PyBytes::new(py, #v));
            }
            if matches!(head(elem), Some((h, _)) if h == "String") {
                return quote!(#v.to_vec());
            }
            let e = to_py(elem, &quote!(e), true);
            return quote!(#v.iter().map(|e| -> ::pyo3::PyResult<_> { ::std::result::Result::Ok(#e) }).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>()?);
        }
        if is_str(&rf.elem) {
            return quote!(#v);
        }
        return to_py(&rf.elem, v, true);
    }
    let Some((h, args)) = head(t) else {
        return if by_ref { quote!(::std::clone::Clone::clone(#v)) } else { quote!(#v) };
    };
    match h.as_str() {
        "String" if by_ref => quote!(#v.as_str()),
        _ if PRIMS.contains(&h.as_str()) => {
            if by_ref { quote!(*#v) } else { quote!(#v) }
        }
        "str" => quote!(#v),
        "Duration" => quote!(#v.as_secs_f64()),
        "Value" if args.is_empty() => quote!(#rt::value_to_py(py, #r)?),
        "Vec" if args.len() == 1 && matches!(head(&args[0]), Some((h, _)) if h == "u8") => quote!(::pyo3::types::PyBytes::new(py, #r)),
        "Option" if args.len() == 1 => {
            let src = if by_ref { quote!(#v.as_ref()) } else { quote!(#v) };
            let e = to_py(&args[0], &quote!(e), by_ref);
            quote!(match #src { ::std::option::Option::Some(e) => ::std::option::Option::Some(#e), ::std::option::Option::None => ::std::option::Option::None })
        }
        "Vec" if args.len() == 1 => {
            let e = to_py(&args[0], &quote!(e), true);
            quote!(#v.iter().map(|e| -> ::pyo3::PyResult<_> { ::std::result::Result::Ok(#e) }).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>()?)
        }
        "HashMap" if args.len() == 2 => {
            let e = to_py(&args[1], &quote!(e), true);
            quote!(#v.iter().map(|(k, e)| -> ::pyo3::PyResult<_> { ::std::result::Result::Ok((::std::clone::Clone::clone(k), #e)) }).collect::<::pyo3::PyResult<::std::collections::HashMap<_, _>>>()?)
        }
        // A side-table key is the host object it names.
        "HostRef" | "MediaRef" => quote!(::pyo3::IntoPyObjectExt::into_bound_py_any(#r, py)?),
        // Everything else crosses by its own `IntoPyObject`: moved when owned; when lent, by
        // `&T`'s when it has one, over a clone otherwise.
        _ if by_ref => quote!({
            #[allow(unused_imports)]
            use #rt::{LendClone as _, LendRef as _};
            (&&#rt::Lend(#v)).lend(py)?
        }),
        _ => quote!(#v),
    }
}

/// `o: &Bound<PyAny>` as a `PyResult<t>` expression.
pub fn from_py(t: &syn::Type, o: &TokenStream) -> TokenStream {
    from_py_at(t, o, &quote!("value"))
}

/// Shared validation for generated arguments, record fields, and host callback results.
pub fn checked_input(t: &syn::Type) -> bool {
    let Some((h, args)) = head(t) else { return false };
    matches!(h.as_str(), "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "f32" | "f64")
        || matches!(h.as_str(), "Vec" | "HashMap" | "BTreeMap")
        || (matches!(h.as_str(), "Option" | "Maybe") && args.first().is_some_and(checked_input))
}

pub fn from_py_at(t: &syn::Type, o: &TokenStream, path: &TokenStream) -> TokenStream {
    let rt = quote!(::lk_agents_abi::python);
    if is_proto(t) {
        return quote!(#rt::proto_in::<#t>(#o, |b| <#t as ::prost::Message>::decode(b).map_err(|e| e.to_string())));
    }
    let Some((h, args)) = head(t) else {
        return quote!(#rt::extract_at::<#t>(#o, #path));
    };
    match h.as_str() {
        "Maybe" if args.len() == 1 => {
            let inner = from_py_at(&args[0], o, path);
            quote!({
                if #rt::is_not_given(#o)? {
                    ::std::result::Result::Ok(::lk_agents_abi::Maybe::NotGiven)
                } else {
                    (#inner).map(::lk_agents_abi::Maybe::Given)
                }
            })
        }
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "f64" => quote!(#rt::number_from_py::<#t>(#o, #path)),
        "f32" => quote!(#rt::f32_from_py(#o, #path)),
        "Value" if args.is_empty() => quote!(#rt::value_from_py(#o, #path)),
        "Vec" if args.len() == 1 && matches!(head(&args[0]), Some((h, _)) if h == "u8") => {
            quote!(
                #rt::snapshot_buffer(#o, #rt::BufferFormat::Bytes, #path, None).map(|b| b.to_vec())
            )
        }
        "Option" if args.len() == 1 => {
            let inner = from_py_at(&args[0], o, path);
            quote!((if (#o).is_none() { ::std::result::Result::Ok(::std::option::Option::None) } else { (#inner).map(::std::option::Option::Some) }))
        }
        "Vec" if args.len() == 1 => {
            let child_path = quote!(&::std::format!("{}[{}]", path, index));
            let inner = from_py_at(&args[0], &quote!(item), &child_path);
            quote!({
                let path = #path;
                (#o).try_iter().and_then(|it| it.enumerate().map(|(index, item)| item.and_then(|item| { let item = &item; #inner })).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>())
            })
        }
        "Duration" => quote!(#rt::secs_from_py_at(#o, #path)),
        "HashMap" | "BTreeMap" if args.len() == 2 => {
            let key_path = quote!(&::std::format!("{}.<key>", path));
            let key = from_py_at(&args[0], &quote!(&key), &key_path);
            let child_path = quote!(&::std::format!("{}[{}]", path, index));
            let inner = from_py_at(&args[1], &quote!(&value), &child_path);
            quote!({
                let path = #path;
                (#o).cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|d| ::pyo3::types::PyDictMethods::iter(&*d).enumerate().map(|(index, (key, value))| {
                    ::std::result::Result::Ok(((#key)?, (#inner)?))
                }).collect::<::pyo3::PyResult<#t>>())
            })
        }
        "MediaRef" => quote!(::std::result::Result::Ok::<_, ::pyo3::PyErr>(#rt::intern_media((#o).clone().unbind()))),
        "HostRef" => quote!(::std::result::Result::Ok::<_, ::pyo3::PyErr>(#rt::intern_host((#o).clone().unbind()))),
        "Arc" if args.len() == 1 && !matches!(args[0], syn::Type::TraitObject(_)) => {
            let inner = &args[0];
            quote!(<#inner as #rt::ArcClass>::shared(#o))
        }
        _ => quote!(#rt::extract_at::<#t>(#o, #path)),
    }
}
