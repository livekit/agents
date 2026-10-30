//! How an in-place argument is read: owned types by their own `FromPyObject`; structural ones
//! (primitives, `Duration` seconds, JSON, bytes, prost messages as bytes, side-table keys, `Arc` of
//! an `arc`-cell class, and containers of those) by the rules here.

use proc_macro2::TokenStream;
use quote::quote;

pub fn head(t: &syn::Type) -> Option<(String, Vec<syn::Type>)> {
    let syn::Type::Path(p) = t else { return None };
    let seg = p.path.segments.last()?;
    let args = match &seg.arguments {
        syn::PathArguments::AngleBracketed(a) => a
            .args
            .iter()
            .filter_map(
                |a| if let syn::GenericArgument::Type(t) = a { Some(t.clone()) } else { None },
            )
            .collect(),
        _ => Vec::new(),
    };
    Some((seg.ident.to_string(), args))
}

fn is_proto(t: &syn::Type) -> bool {
    matches!(t, syn::Type::Path(p) if p.path.segments.iter().any(|s| s.ident == "proto"))
}

/// Shared validation for generated arguments, record fields, and host callback results.
pub fn checked_input(t: &syn::Type) -> bool {
    let Some((h, args)) = head(t) else { return false };
    matches!(
        h.as_str(),
        "u8" | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "f32"
            | "f64"
    ) || matches!(h.as_str(), "Vec" | "HashMap" | "BTreeMap")
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
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128"
        | "isize" | "f64" => quote!(#rt::number_from_py::<#t>(#o, #path)),
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
        "MediaRef" => {
            quote!(::std::result::Result::Ok::<_, ::pyo3::PyErr>(#rt::intern_media((#o).clone().unbind())))
        }
        "HostRef" => {
            quote!(::std::result::Result::Ok::<_, ::pyo3::PyErr>(#rt::intern_host((#o).clone().unbind())))
        }
        "Arc" if args.len() == 1 && !matches!(args[0], syn::Type::TraitObject(_)) => {
            let inner = &args[0];
            quote!(<#inner as #rt::ArcClass>::shared(#o))
        }
        _ => quote!(#rt::extract_at::<#t>(#o, #path)),
    }
}
