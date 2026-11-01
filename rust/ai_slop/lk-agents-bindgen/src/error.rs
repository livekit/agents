//! `#[host(error ..)]`: the host exception a declared error raises, which `lk-agents-codegen` writes
//! into the binding, and its `host_kind`, which `lk-agents-host-derive` writes beside the error.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::Metas;

/// The binding half of one declared error.
pub struct HostException {
    /// `impl crate::py_errors::HostException for <error>`, and the exception class it declares.
    pub tokens: TokenStream,
    /// `declare = ".."`: the class `tokens` creates, which the module registers.
    pub declared: Option<String>,
}

/// One error declaration, read once for both halves.
struct Read {
    /// The `to_py_err` body.
    raise: TokenStream,
    /// The `host_kind` body; `original` has its own, hand-written beside the error.
    kind: Option<TokenStream>,
    /// `(class, base builtin, doc)` of a class of `_core`'s own.
    declared: Option<(syn::Ident, syn::Ident, String)>,
}

/// `error(raised = "..")`: a refusal the host raises, which a generated proxy reads. It has no
/// exception of its own to raise.
pub fn raised(m: &Metas) -> bool {
    m.0.iter().any(|x| x.key == "error.raised")
}

/// `impl <error> { pub fn host_kind(&self) -> &str }`: the host class each value raises as.
pub fn host_kind(attr: TokenStream, item: &syn::Item) -> syn::Result<TokenStream> {
    let ident = match item {
        syn::Item::Enum(e) => &e.ident,
        syn::Item::Struct(s) => &s.ident,
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "#[host(error)] is for an enum or a struct",
            ));
        }
    };
    let Some(body) = read(attr, item, None)?.kind else { return Ok(TokenStream::new()) };
    Ok(quote! {
        #[allow(dead_code)]
        impl #ident {
            /// The host exception class `#[host(kind)]` names for this value.
            pub fn host_kind(&self) -> &str { #body }
        }
    })
}

/// How the binding hands a field's owned `value` (of type `ty`) to the host: an expression of a
/// `Bound<'py, PyAny>`, with `py` in scope.
pub type ToHost<'a> = &'a dyn Fn(&syn::Type, TokenStream) -> syn::Result<TokenStream>;

/// `impl crate::py_errors::HostException for <ty>`, `ty` being the error's path from the binding.
pub fn host_exception(
    attr: TokenStream,
    item: &syn::Item,
    ty: &syn::Path,
    to_host: ToHost<'_>,
) -> syn::Result<HostException> {
    let r = read(attr, item, Some(to_host))?;
    let raise = r.raise;
    let tokens = quote! {
        impl crate::py_errors::HostException for #ty {
            fn to_py_err(&self) -> ::pyo3::PyErr { #raise }
        }
    };
    let Some((class, base, doc)) = r.declared else {
        return Ok(HostException { tokens, declared: None });
    };
    Ok(HostException {
        tokens: quote! {
            ::pyo3::create_exception!(_core, #class, ::pyo3::exceptions::#base, #doc);
            #tokens
        },
        declared: Some(class.to_string()),
    })
}

/// `to_host` is `None` for `host_kind`, which reads no field.
fn read(attr: TokenStream, item: &syn::Item, to_host: Option<ToHost<'_>>) -> syn::Result<Read> {
    let to_host = |ty: &syn::Type, value: TokenStream| match to_host {
        Some(f) => f(ty, value),
        None => Ok(TokenStream::new()),
    };
    let m = Metas::parse(attr)?;
    m.only(
        "error",
        &[
            "error",
            "error.kind",
            "error.original",
            "error.declare",
            "error.by",
            "error.cause",
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
    // `declare = "RuntimeError"`: `kind` is a class of `_core`'s own, derived from that builtin.
    match (m.text("error.declare")?, kind) {
        (Some(base), Some(class)) if !class.contains('.') => {
            let doc = crate::summary(&crate::doc(attrs)).join(" ");
            let (class, base) = (format_ident!("{class}"), format_ident!("Py{base}"));
            let name = class.to_string();
            return Ok(Read {
                raise: quote!(#class::new_err(::std::string::ToString::to_string(self))),
                kind: Some(quote!(#name)),
                declared: Some((class, base, doc)),
            });
        }
        (Some(_), _) => {
            return Err(syn::Error::new_spanned(
                ident,
                "`declare` names the base of the class `kind` names, undotted",
            ));
        }
        (None, _) => {}
    }
    let (raise, kind) = if let Some(by) = m.ident("error.by")? {
        if kind.is_some() || m.flag("error.original") || variants.is_some() {
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
        let (arms, kind_arms) = kind_selected(item, ident, &field.ty, &m, &to_host)?;
        (quote!(match self.#by { #(#arms)* }), Some(quote!(match self.#by { #(#kind_arms)* })))
    } else if m.flag("error.original") {
        if kind.is_some() {
            return Err(syn::Error::new_spanned(
                ident,
                "`original` and `kind` are two answers to one question",
            ));
        }
        (quote!(crate::py_errors::raise_host_error(::std::clone::Clone::clone(self))), None)
    } else {
        let default_kind = kind.unwrap_or("ValueError");
        let default = raise(default_kind, quote!(::std::string::ToString::to_string(self)));
        let mut arms = Vec::new();
        let mut kind_arms = Vec::new();
        for v in variants.into_iter().flatten() {
            let vm = Metas::of(&v.attrs)?;
            if vm.0.is_empty() {
                continue;
            }
            vm.only(&format!("{ident}::{}", v.ident), &["kind", "delegate", "kwargs", "capture"])?;
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
                arms.push(quote!(Self::#vi { #field, .. } => crate::py_errors::HostException::to_py_err(#field),));
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
                arms.push(
                    quote!(Self::#vi(inner) => crate::py_errors::HostException::to_py_err(inner),),
                );
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
            kind_arms.push(quote!(#pat => #k,));
            // `kwargs`: the variant's fields are the host class's keyword arguments too.
            if vm.flag("kwargs") {
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
                let values = named
                    .named
                    .iter()
                    .zip(&fields)
                    .map(|(f, name)| to_host(&f.ty, quote!(::std::clone::Clone::clone(#name))))
                    .collect::<syn::Result<Vec<_>>>()?;
                arms.push(quote!(Self::#vi { #(#fields),* } => crate::py_errors::host_exception_with(#module, #class, #message, |kwargs| {
                    let py = kwargs.py();
                    #( ::pyo3::types::PyDictMethods::set_item(kwargs, #names, #values)?; )*
                    ::std::result::Result::Ok(())
                }),));
            } else {
                let r = raise(k, message);
                arms.push(quote!(#pat => #r,));
            }
        }
        let kind = Some(quote! {
            #[allow(unreachable_patterns)]
            match self { #(#kind_arms)* _ => #default_kind, }
        });
        if arms.is_empty() {
            (default, kind)
        } else {
            let raise = quote! {
                #[allow(unreachable_patterns)]
                match self { #(#arms)* _ => #default, }
            };
            (raise, kind)
        }
    };
    // `cause = field`: the `Option<Box<E>>` the exception was raised from, its `__cause__`.
    let raise = match m.ident("error.cause")? {
        Some(field) => quote! {
            let err = #raise;
            if let ::std::option::Option::Some(cause) = &self.#field {
                ::pyo3::Python::attach(|py| err.set_cause(py, ::std::option::Option::Some(crate::py_errors::HostException::to_py_err(&**cause))));
            }
            err
        },
        None => raise,
    };
    Ok(Read { raise, kind, declared: None })
}

/// `error(by = field, kinds(Variant(class = "module.Class", kwargs(a, b)), ..))` on a struct: the
/// class `field`'s variant names, called with the message and the listed fields as keyword
/// arguments. Every variant of the field's enum is listed (the match is exhaustive).
fn kind_selected(
    item: &syn::Item,
    ident: &syn::Ident,
    ty: &syn::Type,
    m: &Metas,
    to_host: &dyn Fn(&syn::Type, TokenStream) -> syn::Result<TokenStream>,
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
        let values = fields
            .iter()
            .map(|f| {
                let syn::Item::Struct(st) = item else {
                    unreachable!("`by` is checked to be on a struct")
                };
                let Some(field) = st.fields.iter().find(|x| x.ident.as_ref() == Some(f)) else {
                    return Err(syn::Error::new_spanned(
                        ident,
                        format!("`kwargs({f})` names a field of this struct"),
                    ));
                };
                to_host(&field.ty, quote!(::std::clone::Clone::clone(&self.#f)))
            })
            .collect::<syn::Result<Vec<_>>>()?;
        let v = format_ident!("{variant}");
        arms.push(quote!(<#ty>::#v => crate::py_errors::host_exception_with(#module, #name, ::std::string::ToString::to_string(self), |kwargs| {
            let py = kwargs.py();
            #( ::pyo3::types::PyDictMethods::set_item(kwargs, #names, #values)?; )*
            ::std::result::Result::Ok(())
        }),));
        kind_arms.push(quote!(<#ty>::#v => #class,));
    }
    Ok((arms, kind_arms))
}

/// A builtin (`ValueError`) or a dotted host class the binding imports and instantiates.
fn raise(kind: &str, message: TokenStream) -> TokenStream {
    match kind.rsplit_once('.') {
        Some((module, class)) => {
            quote!(crate::py_errors::host_exception(#module, #class, #message))
        }
        None => {
            let class = format_ident!("Py{kind}");
            quote!(::pyo3::exceptions::#class::new_err(#message))
        }
    }
}
