//! Unions: tagged enums by their discriminant, untagged ones by the first arm that fits.

use proc_macro2::TokenStream;
use quote::quote;

use super::*;

pub(super) fn emit_union(r: &Record, vs: &[Variant]) -> syn::Result<TokenStream> {
    let ident = &r.ident;
    let arms = vs.iter().map(|v| {
        let vi = &v.ident;
        match &v.twin {
            Some(p) => quote!(#ident::#vi(x) => #p::host_dict(py, x)?.into_any()),
            None => {
                quote!(#ident::#vi(x) => ::pyo3::IntoPyObject::into_pyobject(x, py)?.into_any())
            }
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

/// Each arm as its member, `None` for a unit arm.
pub(super) fn emit_untagged(r: &Record, arms: &[Arm]) -> TokenStream {
    let ident = &r.ident;
    let arms = arms.iter().map(|a| {
        let ai = &a.ident;
        match &a.fits {
            Fits::Member(_) => {
                quote!(#ident::#ai(x) => ::pyo3::IntoPyObject::into_pyobject(x, py)?.into_any())
            }
            Fits::None | Fits::Rest => quote!(#ident::#ai => py.None().into_bound(py)),
            Fits::Class { module, name } => {
                quote!(#ident::#ai => py.import(#module)?.getattr(#name)?.call0()?)
            }
        }
    });
    quote! {
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
    }
}

/// The first arm the object fits; when none does, the last member's error.
pub(super) fn extract_untagged(r: &Record, arms: &[Arm]) -> TokenStream {
    let ident = &r.ident;
    let unfit = format!("{ident}: the value fits no arm");
    let tries = arms.iter().map(|a| {
        let ai = &a.ident;
        match &a.fits {
            Fits::Member(ty) => quote! {
                match obj.extract::<#ty>() {
                    ::std::result::Result::Ok(v) => return ::std::result::Result::Ok(#ident::#ai(v)),
                    ::std::result::Result::Err(e) => last = ::std::option::Option::Some(::std::convert::Into::<::pyo3::PyErr>::into(e)),
                }
            },
            Fits::None => quote! {
                if obj.is_none() {
                    return ::std::result::Result::Ok(#ident::#ai);
                }
            },
            Fits::Class { module, name } => quote! {
                if obj.is_instance(&obj.py().import(#module)?.getattr(#name)?)? {
                    return ::std::result::Result::Ok(#ident::#ai);
                }
            },
            Fits::Rest => quote!(return ::std::result::Result::Ok(#ident::#ai);),
        }
    });
    quote! {
        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #ident {
            type Error = ::pyo3::PyErr;
            #[allow(unreachable_code, unused_assignments, unused_mut)]
            fn extract(obj: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                let mut last: ::std::option::Option<::pyo3::PyErr> = ::std::option::Option::None;
                #(#tries)*
                ::std::result::Result::Err(last.unwrap_or_else(|| ::pyo3::exceptions::PyValueError::new_err(#unfit)))
            }
        }
    }
}

pub(super) fn extract_union(r: &Record, t: &Target, vs: &[Variant]) -> syn::Result<TokenStream> {
    let ident = &r.ident;
    let first =
        &vs.first().ok_or_else(|| syn::Error::new_spanned(ident, "a union with no arms"))?.ty;
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
