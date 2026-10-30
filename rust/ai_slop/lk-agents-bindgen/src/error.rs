//! `#[host(error ..)]`: an error's `HostException`, the host exception kind each variant raises.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::python::raise;

/// `#[host(error ..)]`: `HostException`, picked by `lk_agents_abi::python`'s autoref dispatch.
pub fn error(attr: TokenStream, item: &syn::Item) -> syn::Result<TokenStream> {
    let m = crate::Metas::parse(attr)?;
    m.only(
        "error",
        &[
            "error",
            "error.kind",
            "error.with",
            "error.original",
            "error.declare",
            "error.by",
            "error.kinds.",
        ],
    )?;
    let (ident, generics, variants, attrs) = match item {
        syn::Item::Enum(e) => (&e.ident, &e.generics, Some(&e.variants), &e.attrs),
        syn::Item::Struct(s) => (&s.ident, &s.generics, None, &s.attrs),
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "#[host(error)] is for an enum or a struct",
            ));
        }
    };
    if !generics.params.is_empty() {
        return Err(syn::Error::new_spanned(generics, "#[host(error)] is for a concrete type"));
    }
    let kind = m.text("error.kind")?;
    let with = m.path("error.with")?;
    // `declare = "RuntimeError"`: `kind` is a class of `_core`'s own, derived from that builtin.
    let declared = match (m.text("error.declare")?, kind) {
        (Some(base), Some(class)) if !class.contains('.') => {
            let doc = crate::summary(&crate::doc(attrs)).join(" ");
            Some((format_ident!("{class}"), format_ident!("Py{base}"), doc))
        }
        (Some(_), _) => {
            return Err(syn::Error::new_spanned(
                ident,
                "`declare` names the base of the class `kind` names, undotted",
            ));
        }
        (None, _) => None,
    };
    if let Some((class, base, doc)) = &declared {
        let class_name = class.to_string();
        return Ok(quote! {
            #[cfg(feature = "python")]
            const _: () = {
                ::pyo3::create_exception!(_core, #class, ::pyo3::exceptions::#base, #doc);
                impl ::lk_agents_abi::python::HostException for #ident {
                    fn to_py_err(&self) -> ::pyo3::PyErr { #class::new_err(::std::string::ToString::to_string(self)) }
                }
                ::lk_agents_abi::python::inventory::submit! {
                    ::lk_agents_abi::python::Registration(|m| {
                        use ::pyo3::types::PyModuleMethods as _;
                        m.add(#class_name, m.py().get_type::<#class>())
                    })
                }
            };
            #[allow(dead_code)]
            impl #ident {
                /// The host exception class `#[host(kind)]` names for this value.
                pub fn host_kind(&self) -> &str { #class_name }
            }
        });
    }
    // The kind table's own reading, for a binding that names the class instead of raising it.
    let mut host_kind = None;
    let body = if let Some(by) = m.ident("error.by")? {
        if kind.is_some() || with.is_some() || m.flag("error.original") || variants.is_some() {
            return Err(syn::Error::new_spanned(
                ident,
                "`by` selects a struct's class by one of its fields, alone",
            ));
        }
        let field = match item {
            syn::Item::Struct(st) => st.fields.iter().find(|f| f.ident.as_ref() == Some(&by)),
            _ => None,
        };
        let Some(field) = field else {
            return Err(syn::Error::new_spanned(
                ident,
                format!("`by = {by}` names a field of this struct"),
            ));
        };
        let (arms, kind_arms) = kind_selected(ident, &field.ty, &m)?;
        host_kind = Some(quote!(match self.#by { #(#kind_arms)* }));
        quote!(match self.#by { #(#arms)* })
    } else if m.flag("error.original") {
        if kind.is_some() || with.is_some() {
            return Err(syn::Error::new_spanned(
                ident,
                "`original`, `kind` and `with` are three answers to one question",
            ));
        }
        quote!(::lk_agents_abi::python::raise_host_error(::std::clone::Clone::clone(self)))
    } else if let Some(w) = with {
        // A converter of the type's own crate, `fn(E) -> PyErr`.
        if kind.is_some() {
            return Err(syn::Error::new_spanned(
                ident,
                "`with` and `kind` are two answers to one question",
            ));
        }
        quote!(#w(::std::clone::Clone::clone(self)))
    } else {
        let default_kind = kind.unwrap_or("ValueError");
        let default = raise(default_kind, quote!(::std::string::ToString::to_string(self)));
        let mut arms = Vec::new();
        let mut kind_arms = Vec::new();
        for v in variants.into_iter().flatten() {
            let vm = crate::Metas::of(&v.attrs)?;
            if vm.0.is_empty() {
                continue;
            }
            vm.only(&format!("{ident}::{}", v.ident), &["kind", "delegate", "kwargs"])?;
            let vi = &v.ident;
            // `delegate = field`: a named variant raises as that field's error.
            if let Some(field) = vm.ident("delegate")? {
                if !matches!(&v.fields, syn::Fields::Named(n) if n.named.iter().any(|f| f.ident.as_ref() == Some(&field)))
                {
                    return Err(syn::Error::new_spanned(
                        vi,
                        "`delegate = field` names a field of this named variant",
                    ));
                }
                arms.push(quote!(Self::#vi { #field, .. } => ::lk_agents_abi::python::HostException::to_py_err(#field),));
                kind_arms.push(quote!(Self::#vi { #field, .. } => #field.host_kind(),));
                continue;
            }
            if vm.flag("delegate") {
                if !matches!(&v.fields, syn::Fields::Unnamed(u) if u.unnamed.len() == 1) {
                    return Err(syn::Error::new_spanned(
                        vi,
                        "`delegate` wraps one error, in a one-field tuple variant",
                    ));
                }
                arms.push(quote!(Self::#vi(inner) => ::lk_agents_abi::python::HostException::to_py_err(inner),));
                kind_arms.push(quote!(Self::#vi(inner) => inner.host_kind(),));
                continue;
            }
            let Some(k) = vm.text("kind")? else {
                return Err(syn::Error::new_spanned(
                    vi,
                    "a variant of an error type takes #[host(kind = \"..\")] or #[host(delegate)]",
                ));
            };
            let pat = match &v.fields {
                syn::Fields::Unit => quote!(Self::#vi),
                syn::Fields::Unnamed(_) => quote!(Self::#vi(..)),
                syn::Fields::Named(_) => quote!(Self::#vi { .. }),
            };
            let message = quote!(::std::string::ToString::to_string(self));
            // `kwargs`: the variant's fields are the host class's keyword arguments too.
            let r = if vm.flag("kwargs") {
                let (syn::Fields::Named(named), Some((module, class))) =
                    (&v.fields, k.rsplit_once('.'))
                else {
                    return Err(syn::Error::new_spanned(
                        vi,
                        "`kwargs` passes a named-field variant's fields to a dotted host class",
                    ));
                };
                let fields: Vec<_> = named.named.iter().filter_map(|f| f.ident.clone()).collect();
                let names = fields.iter().map(|f| f.to_string());
                arms.push(quote!(Self::#vi { #(#fields),* } => ::lk_agents_abi::python::host_exception_with(#module, #class, #message, |kwargs| {
                    #( ::pyo3::types::PyDictMethods::set_item(kwargs, #names, #fields)?; )*
                    ::std::result::Result::Ok(())
                }),));
                kind_arms.push(quote!(#pat => #k,));
                continue;
            } else {
                raise(k, message)
            };
            arms.push(quote!(#pat => #r,));
            kind_arms.push(quote!(#pat => #k,));
        }
        host_kind = Some(quote! {
            #[allow(unreachable_patterns)]
            match self { #(#kind_arms)* _ => #default_kind, }
        });
        if arms.is_empty() {
            default
        } else {
            quote! {
                #[allow(unreachable_patterns)]
                match self { #(#arms)* _ => #default, }
            }
        }
    };
    let host_kind = host_kind.map(|body| {
        quote! {
            #[allow(dead_code)]
            impl #ident {
                /// The host exception class `#[host(kind)]` names for this value.
                pub fn host_kind(&self) -> &str { #body }
            }
        }
    });
    Ok(quote! {
        #[cfg(feature = "python")]
        impl ::lk_agents_abi::python::HostException for #ident {
            fn to_py_err(&self) -> ::pyo3::PyErr { #body }
        }
        #host_kind
    })
}

/// `error(by = field, kinds(Variant(class = "module.Class", kwargs(a, b)), ..))` on a struct: the
/// class `field`'s variant names, called with the message and the listed fields as keyword
/// arguments. Every variant of the field's enum is listed (the match is exhaustive).
fn kind_selected(
    ident: &syn::Ident,
    ty: &syn::Type,
    m: &crate::Metas,
) -> syn::Result<(Vec<TokenStream>, Vec<TokenStream>)> {
    let mut variants: Vec<&str> = Vec::new();
    for (key, _) in m.under("error.kinds.") {
        let variant = key.split('.').next().unwrap_or(key);
        if !variants.contains(&variant) {
            variants.push(variant);
        }
    }
    if variants.is_empty() {
        return Err(syn::Error::new_spanned(
            ident,
            "`by` takes `kinds(Variant(class = \"module.Class\"), ..)`",
        ));
    }
    let (mut arms, mut kind_arms) = (Vec::new(), Vec::new());
    for variant in variants {
        let prefix = format!("error.kinds.{variant}.");
        let class = m.text(&format!("{prefix}class"))?.ok_or_else(|| {
            syn::Error::new_spanned(
                ident,
                format!("`kinds({variant}(..))` names its `class = \"module.Class\"`"),
            )
        })?;
        let Some((module, name)) = class.rsplit_once('.') else {
            return Err(syn::Error::new_spanned(
                ident,
                format!("`{class}`: a kind-selected class is dotted"),
            ));
        };
        let mut fields = Vec::new();
        for (key, value) in m.under(&prefix) {
            match (key, value) {
                ("class", _) => {}
                (kwarg, crate::meta::Value::Word) if kwarg.starts_with("kwargs.") => {
                    fields.push(format_ident!("{}", &kwarg["kwargs.".len()..]));
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        format!("`kinds({variant}(..))`: `{key}` is not a key here"),
                    ));
                }
            }
        }
        let names = fields.iter().map(|f| f.to_string());
        let v = format_ident!("{variant}");
        arms.push(quote!(<#ty>::#v => ::lk_agents_abi::python::host_exception_with(#module, #name, ::std::string::ToString::to_string(self), |kwargs| {
            #( ::pyo3::types::PyDictMethods::set_item(kwargs, #names, ::std::clone::Clone::clone(&self.#fields))?; )*
            ::std::result::Result::Ok(())
        }),));
        kind_arms.push(quote!(<#ty>::#v => #class,));
    }
    Ok((arms, kind_arms))
}
