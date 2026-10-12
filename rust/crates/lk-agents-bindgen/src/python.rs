//! The pyo3 each item expands to, gated by an emitted `cfg` on the calling crate's `python`
//! feature (a proc macro cannot see features). `::pyo3` is the caller's own dependency, since
//! pyo3's macros spell `::pyo3`. Names, `__text_signature__`, `__doc__`, raised exceptions and the
//! fork guard match what the codegen used to emit.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::visit_mut::VisitMut as _;

use crate::{Cell, Class, ConstExport, KwOnly, Method, MethodKind, Object, Receiver, Ret, Sig};

/// `#[host(export ..)]` on `f`: a registered `#[pyfunction]` beside it.
pub fn export_fn(attr: TokenStream, f: &syn::ItemFn) -> syn::Result<TokenStream> {
    let s = crate::read_export(attr, f)?;
    let wrapper = format_ident!("__host_py_{}", s.rust);
    let docs = doc_attrs(&crate::summary(&s.doc));
    let func = if s.params.iter().any(|p| p.kwargs) {
        let name = &s.name;
        spread(&s, Spread {
            head: None,
            attrs: quote!(#docs #[::pyo3::pyfunction] #[pyo3(name = #name)]),
            wrapper: wrapper.clone(),
            recv: TokenStream::new(),
            ret: ret_ty(&s, None),
            call: &Call::Free,
            wrap: None,
        })
    } else {
        let pyo3 = pyo3_attr(Some(&s.name), &s);
        let body = body(&s, &Call::Free);
        let (lt, py, params, ret) = (generics(&s), py_param(&s), params(&s), ret_ty(&s, None));
        quote! {
            #docs
            #[allow(clippy::too_many_arguments)]
            #[::pyo3::pyfunction]
            #pyo3
            fn #wrapper #lt (#py #params) -> ::pyo3::PyResult<#ret> { #body }
        }
    };
    Ok(quote! {
        #[cfg(feature = "python")]
        const _: () = {
            #func
            ::lk_agents_abi::python::inventory::submit! {
                ::lk_agents_abi::python::Registration(|m| {
                    use ::pyo3::types::PyModuleMethods as _;
                    m.add_function(::pyo3::wrap_pyfunction!(#wrapper, m)?)
                })
            }
        };
    })
}

/// `#[host(export ..)]` on a const: a zero-argument `#[pyfunction]` returning it.
pub fn export_const(attr: TokenStream, c: &syn::ItemConst) -> syn::Result<TokenStream> {
    let m = crate::Metas::parse(attr)?;
    Ok(const_fn(&crate::read_const(&m, &c.ident, &c.ty, &c.attrs, None)?))
}

/// `#[host(export)]` on a plain impl: one `const_fn` per exported associated const.
pub fn export_impl(attr: TokenStream, imp: &syn::ItemImpl) -> syn::Result<TokenStream> {
    Ok(crate::read_export_impl(attr, imp)?.iter().map(const_fn).collect())
}

fn const_fn(c: &ConstExport) -> TokenStream {
    if c.attr {
        // A module attribute: the value itself, registered by name, as the codegen's `attr` did.
        let (name, path) = (&c.name, &c.path);
        return quote! {
            #[cfg(feature = "python")]
            ::lk_agents_abi::python::inventory::submit! {
                ::lk_agents_abi::python::Registration(|m| {
                    use ::pyo3::types::PyModuleMethods as _;
                    m.add(#name, #path)
                })
            }
        };
    }
    let wrapper = format_ident!("__host_py_{}", c.rust.to_string().to_lowercase());
    let docs = doc_attrs(&crate::summary(&c.doc));
    let name = &c.name;
    let pid = c.pid.then(|| quote!(::lk_agents_abi::python::check_pid()?;));
    let path = &c.path;
    // A const's `&str` is `&'static str`; a function's return type must say so.
    let mut ty = c.ty.clone();
    StaticRefs.visit_type_mut(&mut ty);
    let ty = &ty;
    let (ret, out) = match crate::shape::output(ty, quote!(out)) {
        Some(o) => {
            let (t, x) = (o.ty, o.expr);
            (t, quote!(let out = #x;))
        }
        None => (quote!(#ty), TokenStream::new()),
    };
    quote! {
        #[cfg(feature = "python")]
        const _: () = {
            #docs
            #[::pyo3::pyfunction]
            #[pyo3(name = #name)]
            fn #wrapper() -> ::pyo3::PyResult<#ret> {
                #pid
                let out = #path;
                #out
                ::std::result::Result::Ok(out)
            }
            ::lk_agents_abi::python::inventory::submit! {
                ::lk_agents_abi::python::Registration(|m| {
                    use ::pyo3::types::PyModuleMethods as _;
                    m.add_function(::pyo3::wrap_pyfunction!(#wrapper, m)?)
                })
            }
        };
    }
}

/// `#[host(string)]`: any string, `Display` out and `FromStr` in (a `ValueError` on failure).
pub fn string(name: &syn::Ident) -> TokenStream {
    quote! {
        #[cfg(feature = "python")]
        const _: () = {
            impl<'py> ::pyo3::IntoPyObject<'py> for #name {
                type Target = ::pyo3::types::PyString;
                type Output = ::pyo3::Bound<'py, ::pyo3::types::PyString>;
                type Error = ::std::convert::Infallible;
                fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::std::result::Result<::pyo3::Bound<'py, ::pyo3::types::PyString>, ::std::convert::Infallible> {
                    ::std::result::Result::Ok(::pyo3::types::PyString::new(py, &::std::string::ToString::to_string(&self)))
                }
            }
            impl ::lk_agents_abi::python::HostStr for #name {}
            impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #name {
                type Error = ::pyo3::PyErr;
                fn extract(ob: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::std::result::Result<Self, ::pyo3::PyErr> {
                    let s = ob.extract::<::pyo3::pybacked::PyBackedStr>().map_err(::std::convert::Into::<::pyo3::PyErr>::into)?;
                    <Self as ::core::str::FromStr>::from_str(&s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(&e)))
                }
            }
        };
    }
}

struct SelfTo(syn::Type);

impl syn::visit_mut::VisitMut for SelfTo {
    fn visit_type_mut(&mut self, t: &mut syn::Type) {
        if matches!(t, syn::Type::Path(p) if p.qself.is_none() && p.path.is_ident("Self")) {
            *t = self.0.clone();
            return;
        }
        syn::visit_mut::visit_type_mut(self, t);
    }
}

/// Every elided reference lifetime made `'static`.
struct StaticRefs;

impl syn::visit_mut::VisitMut for StaticRefs {
    fn visit_type_reference_mut(&mut self, r: &mut syn::TypeReference) {
        if r.lifetime.is_none() {
            r.lifetime = Some(syn::Lifetime::new("'static", Span::call_site()));
        }
        syn::visit_mut::visit_type_reference_mut(self, r);
    }
}

/// `#[host(wire_enum)]`: a closed set of wire strings, `as_str` / `FromStr` (a `ValueError`).
pub fn wire_enum(name: &syn::Ident) -> TokenStream {
    quote! {
        #[cfg(feature = "python")]
        const _: () = {
            impl<'py> ::pyo3::IntoPyObject<'py> for #name {
                type Target = ::pyo3::types::PyString;
                type Output = ::pyo3::Bound<'py, ::pyo3::types::PyString>;
                type Error = ::std::convert::Infallible;
                fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::std::result::Result<::pyo3::Bound<'py, ::pyo3::types::PyString>, ::std::convert::Infallible> {
                    ::std::result::Result::Ok(::pyo3::types::PyString::new(py, self.as_str()))
                }
            }
            impl ::lk_agents_abi::python::HostStr for #name {}
            impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #name {
                type Error = ::pyo3::PyErr;
                fn extract(ob: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::std::result::Result<Self, ::pyo3::PyErr> {
                    let s = ob.extract::<::pyo3::pybacked::PyBackedStr>().map_err(::std::convert::Into::<::pyo3::PyErr>::into)?;
                    <Self as ::core::str::FromStr>::from_str(&s).map_err(::pyo3::exceptions::PyValueError::new_err)
                }
            }
        };
    }
}

/// `#[host(error ..)]`: `HostException`, picked by `lk_agents_abi::python`'s autoref dispatch.
pub fn error(attr: TokenStream, item: &syn::Item) -> syn::Result<TokenStream> {
    let m = crate::Metas::parse(attr)?;
    m.only("error", &["error", "error.kind", "error.with", "error.original"])?;
    let (ident, generics, variants) = match item {
        syn::Item::Enum(e) => (&e.ident, &e.generics, Some(&e.variants)),
        syn::Item::Struct(s) => (&s.ident, &s.generics, None),
        other => return Err(syn::Error::new_spanned(other, "#[host(error)] is for an enum or a struct")),
    };
    if !generics.params.is_empty() {
        return Err(syn::Error::new_spanned(generics, "#[host(error)] is for a concrete type"));
    }
    let kind = m.text("error.kind")?;
    let with = m.path("error.with")?;
    // The kind table's own reading, for a binding that names the class instead of raising it.
    let mut host_kind = None;
    let body = if m.flag("error.original") {
        if kind.is_some() || with.is_some() {
            return Err(syn::Error::new_spanned(ident, "`original`, `kind` and `with` are three answers to one question"));
        }
        quote!(::lk_agents_abi::python::raise_host_error(::std::clone::Clone::clone(self)))
    } else if let Some(w) = with {
        // A converter of the type's own crate, `fn(E) -> PyErr`.
        if kind.is_some() {
            return Err(syn::Error::new_spanned(ident, "`with` and `kind` are two answers to one question"));
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
            vm.only(&format!("{ident}::{}", v.ident), &["kind", "delegate"])?;
            let vi = &v.ident;
            if vm.flag("delegate") {
                if !matches!(&v.fields, syn::Fields::Unnamed(u) if u.unnamed.len() == 1) {
                    return Err(syn::Error::new_spanned(vi, "`delegate` wraps one error, in a one-field tuple variant"));
                }
                arms.push(quote!(Self::#vi(inner) => ::lk_agents_abi::python::HostException::to_py_err(inner),));
                kind_arms.push(quote!(Self::#vi(inner) => inner.host_kind(),));
                continue;
            }
            let Some(k) = vm.text("kind")? else {
                return Err(syn::Error::new_spanned(vi, "a variant of an error type takes #[host(kind = \"..\")] or #[host(delegate)]"));
            };
            let pat = match &v.fields {
                syn::Fields::Unit => quote!(Self::#vi),
                syn::Fields::Unnamed(_) => quote!(Self::#vi(..)),
                syn::Fields::Named(_) => quote!(Self::#vi { .. }),
            };
            let r = raise(k, quote!(::std::string::ToString::to_string(self)));
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

/// A builtin (`ValueError`) or a dotted host class the runtime imports and instantiates.
fn raise(kind: &str, message: TokenStream) -> TokenStream {
    match kind.rsplit_once('.') {
        Some((module, class)) => quote!(::lk_agents_abi::python::host_exception(#module, #class, #message)),
        None => {
            let class = format_ident!("Py{kind}");
            quote!(::pyo3::exceptions::#class::new_err(#message))
        }
    }
}

/// `#[host(object ..)]`: the `#[pymethods]` of its class. A plain object's class is `imp`'s type
/// (name checked against `#[lk_agents_abi::class]`); a `mutex`/`arc` cell's class is emitted here,
/// frozen, with `T`'s `IntoPyObject` and, for `arc`, `ArcClass`.
pub fn object(attr: TokenStream, imp: &syn::ItemImpl) -> syn::Result<TokenStream> {
    let o = crate::read_object(attr, imp)?;
    // A method spreading a keyword record is its own `#[pymethods]` block, beside the class's.
    let (spread, plain): (Vec<_>, Vec<_>) = o.methods.iter().partition(|m| m.sig.params.iter().any(|p| p.kwargs));
    let methods = plain.iter().map(|m| method(&o, m)).collect::<Vec<_>>();
    let spread = spread.iter().map(|m| method(&o, m)).collect::<Vec<_>>();
    let consts = o.consts.iter().map(class_const).collect::<Vec<_>>();
    let core = &o.self_ty;
    let natives = o.as_dyn.iter().map(|(tr, m)| {
        let class = o.cell.class().map(|c| quote!(#c)).unwrap_or_else(|| quote!(#core));
        let get = match &o.cell {
            Cell::Plain => quote!(n.#m()),
            _ => quote!(n.inner.#m()),
        };
        quote! {
            ::lk_agents_abi::python::inventory::submit! {
                ::lk_agents_abi::python::HostNative {
                    trait_id: || ::std::any::TypeId::of::<dyn #tr>(),
                    read: |obj| {
                        use ::pyo3::types::PyAnyMethods as _;
                        let n = obj.extract::<::pyo3::PyRef<'_, #class>>().ok()?;
                        let v: ::std::sync::Arc<dyn #tr> = #get;
                        ::std::option::Option::Some(::std::boxed::Box::new(v) as ::std::boxed::Box<dyn ::std::any::Any>)
                    },
                }
            }
        }
    });
    Ok(match &o.cell {
        Cell::Plain => {
            let name = &o.name;
            let message = format!(
                "#[host(object(name = \"{name}\"))] and the type's #[lk_agents_abi::class] disagree on the class name"
            );
            quote! {
                #[cfg(feature = "python")]
                const _: () = {
                    // The stub reads the impl's name and pyo3 the struct's; they are one name.
                    assert!(::lk_agents_abi::python::same_name(<#core as ::pyo3::PyClass>::NAME, #name), #message);
                    #[::pyo3::pymethods]
                    impl #core { #(#consts)* #(#methods)* }
                    #(#spread)*
                    #(#natives)*
                };
            }
        }
        Cell::Mutex { rust_name } | Cell::Arc { rust_name } | Cell::Value { rust_name, .. } => {
            let docs = doc_attrs(&crate::summary(&o.doc));
            let (name, module) = (&o.name, &o.module);
            let (field, wrap) = match &o.cell {
                Cell::Mutex { .. } => (quote!(::std::sync::Mutex<#core>), quote!(::std::sync::Mutex::new(self))),
                Cell::Arc { .. } => (quote!(::std::sync::Arc<#core>), quote!(::std::sync::Arc::new(self))),
                _ => (quote!(#core), quote!(self)),
            };
            // A `value` cell's class options; it is frozen unless a method mutates.
            let mutates = o.methods.iter().any(|m| m.sig.receiver == Receiver::Mut);
            let input = matches!(o.cell, Cell::Value { input: true, .. });
            let (frozen, options, derive, by_value) = match &o.cell {
                Cell::Value { pyclass, derive, .. } => {
                    let options: Vec<TokenStream> = pyclass.iter().map(|p| p.parse().unwrap_or_default()).collect();
                    let derive: Option<TokenStream> = derive.as_ref().map(|d| d.parse().unwrap_or_default());
                    let extract = pyclass.iter().any(|p| p == "from_py_object");
                    ((!mutates).then(|| quote!(frozen,)), quote!(#(#options,)*), derive.map(|d| quote!(#[derive(#d)])), Some(extract))
                }
                _ => (Some(quote!(frozen,)), TokenStream::new(), None, None),
            };
            let into = (!input).then(|| {
                quote! {
                    impl<'py> ::pyo3::IntoPyObject<'py> for #core {
                        type Target = #rust_name;
                        type Output = ::pyo3::Bound<'py, #rust_name>;
                        type Error = ::pyo3::PyErr;
                        fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, #rust_name>> {
                            ::pyo3::Bound::new(py, #rust_name { inner: #wrap })
                        }
                    }
                }
            });
            let value_impls = by_value.map(|extract| {
                let from = extract.then(|| {
                    quote! {
                        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for #core {
                            type Error = ::pyo3::PyErr;
                            fn extract(obj: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                                ::std::result::Result::Ok(::std::clone::Clone::clone(&obj.extract::<::pyo3::PyRef<'py, #rust_name>>()?.inner))
                            }
                        }
                    }
                });
                let by_ref = (!input).then(|| {
                    quote! {
                        impl<'py> ::pyo3::IntoPyObject<'py> for &#core {
                            type Target = #rust_name;
                            type Output = ::pyo3::Bound<'py, #rust_name>;
                            type Error = ::pyo3::PyErr;
                            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, #rust_name>> {
                                ::pyo3::Bound::new(py, #rust_name { inner: ::std::clone::Clone::clone(self) })
                            }
                        }
                    }
                });
                quote! {
                    #by_ref
                    #from
                }
            });
            let shared = matches!(o.cell, Cell::Arc { .. }).then(|| {
                quote! {
                    // A `CoreObject` holding this type crosses as this class over the same `Arc`.
                    ::lk_agents_abi::python::inventory::submit! {
                        ::lk_agents_abi::python::CoreObjectClass {
                            to_host: |py, obj| {
                                let v = ::std::sync::Arc::clone(obj).downcast::<#core>().ok()?;
                                ::std::option::Option::Some(::pyo3::Bound::new(py, #rust_name { inner: v }).map(::pyo3::Bound::into_any))
                            },
                        }
                    }
                    impl ::lk_agents_abi::python::ArcClass for #core {
                        fn shared(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
                            use ::pyo3::types::PyAnyMethods as _;
                            ::std::result::Result::Ok(::std::sync::Arc::clone(&obj.extract::<::pyo3::PyRef<'_, #rust_name>>()?.inner))
                        }
                        fn wrap<'py>(py: ::pyo3::Python<'py>, value: ::std::sync::Arc<Self>) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                            ::std::result::Result::Ok(::pyo3::Bound::new(py, #rust_name { inner: value })?.into_any())
                        }
                    }
                }
            });
            quote! {
                #[cfg(feature = "python")]
                #docs
                #[::pyo3::pyclass(#frozen #options name = #name, module = #module)]
                #derive
                pub struct #rust_name {
                    pub inner: #field,
                }
                #[cfg(feature = "python")]
                const _: () = {
                    #[::pyo3::pymethods]
                    impl #rust_name { #(#consts)* #(#methods)* }
                    #(#spread)*
                    #into
                    #shared
                    #value_impls
                    #(#natives)*
                    ::lk_agents_abi::python::inventory::submit! {
                        ::lk_agents_abi::python::Registration(|m| {
                            use ::pyo3::types::PyModuleMethods as _;
                            m.add_class::<#rust_name>()
                        })
                    }
                };
            }
        }
    })
}

fn class_const(c: &ConstExport) -> TokenStream {
    let wrapper = format_ident!("__host_py_const_{}", c.rust);
    let name = syn::Ident::new(&c.name, Span::call_site());
    let path = &c.path;
    let mut ty = c.ty.clone();
    StaticRefs.visit_type_mut(&mut ty);
    let (ret, conv, py) = match crate::shape::output(&ty, quote!(out)) {
        Some(o) => {
            let (t, x, py) = (o.ty, o.expr, o.py);
            (t, quote!(let out = #x;), py)
        }
        None => (quote!(#ty), TokenStream::new(), false),
    };
    let (lt, pyp) = if py { (quote!(<'py>), quote!(py: ::pyo3::Python<'py>)) } else { (TokenStream::new(), TokenStream::new()) };
    quote! {
        #[classattr]
        #[pyo3(name = #name)]
        #[allow(non_snake_case)]
        fn #wrapper #lt (#pyp) -> ::pyo3::PyResult<#ret> {
            let out = #path;
            #conv
            ::std::result::Result::Ok(out)
        }
    }
}

/// `#[lk_agents_abi::class ..]`: the struct with `#[pyclass]` under the feature, registered. The
/// doc attributes are swapped for the summary via `cfg_attr`s placed before the `pyclass` (rustc
/// expands in order), so `__doc__` is the first paragraph.
pub fn class(attr: TokenStream, st: &syn::ItemStruct) -> syn::Result<TokenStream> {
    let c = crate::read_class(attr, st)?;
    let mut st = st.clone();
    let lines = crate::doc(&st.attrs);
    let docs: Vec<syn::Attribute> = st.attrs.iter().filter(|a| a.path().is_ident("doc")).cloned().collect();
    st.attrs.retain(|a| !a.path().is_ident("doc"));
    let summary = crate::summary(&lines).into_iter().map(|l| format!(" {l}"));
    let rustdoc = docs.iter().map(|a| {
        let meta = &a.meta;
        let syn::Meta::NameValue(nv) = meta else { return quote!(#a) };
        let value = &nv.value;
        quote!(#[cfg_attr(not(feature = "python"), doc = #value)])
    });
    let Class { name, module, mutable, from_py_object, eq, subclass } = &c;
    let frozen = (!mutable).then(|| quote!(frozen,));
    let eq = eq.then(|| quote!(eq,));
    let subclass = subclass.then(|| quote!(subclass,));
    let extract = if *from_py_object { quote!(from_py_object,) } else { quote!(skip_from_py_object,) };
    let ident = &st.ident;
    Ok(quote! {
        #(#rustdoc)*
        #(#[cfg_attr(feature = "python", doc = #summary)])*
        #[cfg_attr(feature = "python", ::pyo3::pyclass(#frozen #eq #subclass #extract name = #name, module = #module))]
        #st
        #[cfg(feature = "python")]
        ::lk_agents_abi::python::inventory::submit! {
            ::lk_agents_abi::python::Registration(|m| {
                use ::pyo3::types::PyModuleMethods as _;
                m.add_class::<#ident>()
            })
        }
    })
}

enum Call<'a> {
    Free,
    /// `Self::`.
    Own,
    /// The `T` inside `self.inner: Mutex<T>`.
    Locked(&'a syn::Type),
    /// The `T` inside `self.inner: Arc<T>`.
    Shared(&'a syn::Type),
}

fn method(o: &Object, m: &Method) -> TokenStream {
    // In a cell's class, `Self` is the class; the core's `Self` is the value it holds.
    let sig = match &o.cell {
        Cell::Plain => m.sig.clone(),
        _ => {
            let mut sig = m.sig.clone();
            let mut to_core = SelfTo(o.self_ty.clone());
            for p in &mut sig.params {
                to_core.visit_type_mut(&mut p.ty);
            }
            if let Ret::Plain(t) | Ret::Result(t) = &mut sig.ret
                && m.kind != MethodKind::New
            {
                to_core.visit_type_mut(t);
            }
            sig
        }
    };
    let s = &sig;
    let wrapper = match m.kind {
        MethodKind::Setter => format_ident!("__host_py_set_{}", s.rust),
        _ => format_ident!("__host_py_{}", s.rust),
    };
    let call = match &o.cell {
        Cell::Plain => Call::Own,
        Cell::Mutex { .. } => Call::Locked(&o.self_ty),
        Cell::Arc { .. } | Cell::Value { .. } => Call::Shared(&o.self_ty),
    };
    let (marker, name) = match m.kind {
        MethodKind::New => (quote!(#[new]), None),
        MethodKind::Static => (quote!(#[staticmethod]), Some(s.name.as_str())),
        MethodKind::Method => (TokenStream::new(), Some(s.name.as_str())),
        MethodKind::Getter => {
            let getter = crate::host_ident(&s.name);
            (quote!(#[getter(#getter)]), None)
        }
        MethodKind::Setter => {
            let setter = crate::host_ident(&s.name);
            (quote!(#[setter(#setter)]), None)
        }
    };
    let pyo3 = pyo3_attr(name, s);
    let recv = match s.receiver {
        Receiver::None => TokenStream::new(),
        // A mutex or arc cell's class is frozen: the lock, not the borrow flag, gives `&mut`.
        Receiver::Mut if matches!(call, Call::Own) || matches!(o.cell, Cell::Value { .. }) => quote!(&mut self,),
        Receiver::Ref | Receiver::Mut => quote!(&self,),
    };
    let wrap = match (&o.cell, m.kind) {
        (Cell::Mutex { rust_name }, MethodKind::New) => Some((rust_name, Wrap::Mutex)),
        (Cell::Arc { rust_name }, MethodKind::New) => Some((rust_name, Wrap::Arc)),
        (Cell::Value { rust_name, .. }, MethodKind::New) => Some((rust_name, Wrap::Value)),
        _ => None,
    };
    // The first paragraph; a `__dunder__` keeps Python's, and constructors/setters get none.
    let docs = if s.name.starts_with("__") || matches!(m.kind, MethodKind::Setter | MethodKind::New) {
        TokenStream::new()
    } else {
        doc_attrs(&crate::summary(&s.doc))
    };
    if s.params.iter().any(|p| p.kwargs) {
        let name = name.map(|n| quote!(#[pyo3(name = #n)]));
        let class = match &o.cell {
            Cell::Plain => { let c = &o.self_ty; quote!(#c) }
            c => { let c = c.class().expect("a cell class"); quote!(#c) }
        };
        return spread(s, Spread {
            head: Some(quote!(impl #class)),
            attrs: quote!(#docs #marker #name),
            wrapper,
            recv,
            ret: ret_ty(s, wrap.map(|(c, _)| c)),
            call: &call,
            wrap,
        });
    }
    let body = body_wrapped(s, &call, wrap);
    let (lt, py, params, ret) = (generics(s), py_param(s), params(s), ret_ty(s, wrap.map(|(c, _)| c)));
    quote! {
        #docs
        #[allow(clippy::too_many_arguments)]
        #marker
        #pyo3
        fn #wrapper #lt (#recv #py #params) -> ::pyo3::PyResult<#ret> { #body }
    }
}

/// `#[pyo3(name, signature)]`, each only when it differs from what pyo3 infers.
fn pyo3_attr(name: Option<&str>, s: &Sig) -> TokenStream {
    let mut parts = Vec::new();
    if let Some(name) = name {
        parts.push(quote!(name = #name));
    }
    // A core `Maybe<T>` parameter defaults to `NotGiven` without saying so, as the codegen's did.
    let implied = |p: &crate::Param| {
        matches!(&p.ty, syn::Type::Path(tp) if tp.path.segments.last().is_some_and(|s| s.ident == "Maybe"))
            .then(|| quote!(::lk_agents_abi::Maybe::NotGiven))
    };
    let host: Vec<&crate::Param> = s.params.iter().filter(|p| !p.ambient).collect();
    if !s.defaults.is_empty() || !matches!(s.kw_only, KwOnly::None) || host.iter().any(|p| implied(p).is_some()) {
        let mut items = Vec::new();
        for (i, p) in host.iter().enumerate() {
            let star = match &s.kw_only {
                KwOnly::All => i == 0,
                KwOnly::From(first) => p.name == first,
                KwOnly::None => false,
            };
            if star {
                items.push(quote!(*));
            }
            let name = &p.name;
            match s.defaults.iter().find(|(d, _)| p.name == d).map(|(_, v)| v.clone()).or_else(|| implied(p)) {
                Some(value) => items.push(quote!(#name = #value)),
                None => items.push(quote!(#name)),
            }
        }
        parts.push(quote!(signature = (#(#items),*)));
    }
    if parts.is_empty() { TokenStream::new() } else { quote!(#[pyo3(#(#parts),*)]) }
}

/// `shape::input` per host parameter; an `ambient` one has none.
fn inputs(s: &Sig) -> Vec<Option<crate::shape::In>> {
    s.params
        .iter()
        .map(|p| {
            if p.ambient || p.kwargs {
                return None;
            }
            let label = p.label.clone().unwrap_or_else(|| p.name.to_string());
            // A string type defaulted to a literal: taken as the `str` it is spelled as, parsed
            // (`FromStr`'s refusal is a `ValueError` naming the argument).
            let literal = s.defaults.iter().any(|(d, v)| p.name == d && v.to_string().starts_with('"'));
            if literal && crate::shape::is_opaque(&p.ty) {
                let (name, ty) = (&p.name, &p.ty);
                return Some(crate::shape::In {
                    ty: quote!(&str),
                    prelude: Some(quote!(let #name: #ty = ::lk_agents_abi::python::parse_str(#name, #label)?;)),
                    arg: quote!(#name),
                    py: false,
                });
            }
            if p.lenient {
                let (name, ty) = (&p.name, &p.ty);
                return Some(crate::shape::In {
                    ty: quote!(&::pyo3::Bound<'py, ::pyo3::PyAny>),
                    prelude: Some(quote!(let #name: #ty = ::pyo3::types::PyAnyMethods::extract(#name).unwrap_or_default();)),
                    arg: quote!(#name),
                    py: true,
                });
            }
            Some(crate::shape::input(&p.name, &label, &p.ty))
        })
        .collect()
}

/// `None` when pyo3 converts it itself (`shape::output`).
fn output(s: &Sig) -> Option<crate::shape::Out> {
    if s.stream.is_some() {
        return None;
    }
    match &s.ret {
        Ret::Unit => None,
        Ret::Plain(t) | Ret::Result(t) => crate::shape::output(t, quote!(out)),
    }
}

/// To release the GIL, to return an awaitable, or because a conversion needs it.
fn needs_py(s: &Sig) -> bool {
    s.detach || s.is_async || s.reply.is_some() || inputs(s).iter().flatten().any(|i| i.py) || (!s.is_async && output(s).is_some_and(|o| o.py))
}

fn generics(s: &Sig) -> TokenStream {
    if needs_py(s) { quote!(<'py>) } else { TokenStream::new() }
}

fn py_param(s: &Sig) -> TokenStream {
    if needs_py(s) { quote!(py: ::pyo3::Python<'py>,) } else { TokenStream::new() }
}

fn params(s: &Sig) -> TokenStream {
    let items = s.params.iter().zip(inputs(s)).filter_map(|(p, i)| {
        let i = i?;
        let (name, ty) = (&p.name, &i.ty);
        Some(quote!(#name: #ty))
    });
    quote!(#(#items),*)
}

/// The core's, a cell constructor's class, a stream class, or an `async` callable's awaitable.
fn ret_ty(s: &Sig, wrap: Option<&syn::Ident>) -> TokenStream {
    if let Some(class) = wrap {
        return quote!(#class);
    }
    if s.is_async || s.reply.is_some() {
        return quote!(::pyo3::Bound<'py, ::pyo3::PyAny>);
    }
    if let Some(class) = &s.stream {
        return quote!(#class);
    }
    match (&s.ret, output(s)) {
        (Ret::Unit, _) => quote!(()),
        (_, Some(o)) => o.ty,
        (Ret::Plain(t) | Ret::Result(t), None) => quote!(#t),
    }
}

fn body(s: &Sig, call: &Call) -> TokenStream {
    body_wrapped(s, call, None)
}

/// A `(sink, source)` pair, or a source alone.
fn stream_wrap(s: &Sig) -> Option<TokenStream> {
    let class = s.stream.as_ref()?;
    let (Ret::Plain(t) | Ret::Result(t)) = &s.ret else { return None };
    Some(match t {
        syn::Type::Tuple(tu) if tu.elems.len() == 2 => quote!(#class::from_pair(out.0, out.1)),
        _ => quote!(#class::from_source(out)),
    })
}

/// How a cell's constructor holds the new value.
#[derive(Clone, Copy)]
enum Wrap {
    Mutex,
    Arc,
    Value,
}

fn body_wrapped(s: &Sig, call: &Call, wrap: Option<(&syn::Ident, Wrap)>) -> TokenStream {
    let (pre, rest) = body_parts(s, call, wrap);
    quote!(#pre #rest)
}

/// Over `out`, a `Result` of the core's error.
fn fail_rule(s: &Sig) -> TokenStream {
    match &s.err {
        Some(kind) => {
            let r = raise(kind, quote!(e.to_string()));
            quote!(let out = out.map_err(|e| #r)?;)
        }
        None => quote! {
            let out = out.map_err(|e| {
                #[allow(unused_imports)]
                use ::lk_agents_abi::python::{Declared as _, Fallback as _};
                (&::lk_agents_abi::python::Fail(&e)).py_err()
            })?;
        },
    }
}

/// Fork guard and argument conversions, then call, error rule, conversion and `Ok`. A
/// `#[host(kwargs)]` parameter's fields are converted in between (`crate::kwargs`).
fn body_parts(s: &Sig, call: &Call, wrap: Option<(&syn::Ident, Wrap)>) -> (TokenStream, TokenStream) {
    let pid = s.pid.then(|| quote!(::lk_agents_abi::python::check_pid()?;));
    let ins = inputs(s);
    // A receiver or an `Arc<dyn ..>` parameter that is a `HostClock` scopes its clock over the
    // conversions after it.
    let probe = |v: TokenStream| quote!(let _clock = {
        #[allow(unused_imports)]
        use ::lk_agents_abi::python::{ClockDeclared as _, ClockNone as _};
        (&&::lk_agents_abi::python::ClockOf(#v)).scope()
    };);
    // Only a conversion that builds a proxy reads the clock: an `Arc<dyn ..>` argument, or a
    // keyword record's (whose fields the expansion cannot see).
    let proxies = s.params.iter().any(|p| p.kwargs || quote::ToTokens::to_token_stream(&p.ty).to_string().contains("dyn "));
    let recv_clock = match (call, s.receiver) {
        _ if !proxies => None,
        (_, Receiver::None) | (Call::Free | Call::Locked(_), _) => None,
        (Call::Own, _) => Some(probe(quote!(self))),
        (Call::Shared(_), _) => Some(probe(quote!(&self.inner))),
    };
    let preludes: Vec<TokenStream> = s
        .params
        .iter()
        .zip(&ins)
        .filter_map(|(p, i)| {
            let prelude = i.as_ref()?.prelude.clone()?;
            let is_dyn_arc = matches!(crate::cross::head(&p.ty), Some((h, a)) if h == "Arc" && matches!(a.first(), Some(syn::Type::TraitObject(_))));
            let name = &p.name;
            let scope = is_dyn_arc.then(|| probe(quote!(&#name)));
            Some(quote!(#prelude #scope))
        })
        .collect();
    let preludes = recv_clock.into_iter().chain(preludes);
    let rust = &s.rust;
    let args = s.params.iter().zip(&ins).map(|(p, i)| match i {
        Some(i) => i.arg.clone(),
        None if p.kwargs => {
            let name = &p.name;
            if matches!(p.ty, syn::Type::Reference(_)) { quote!(&#name) } else { quote!(#name) }
        }
        None => {
            let ty = &p.ty;
            quote!(<#ty as ::lk_agents_abi::python::Ambient>::ambient())
        }
    });
    let target = match (call, s.receiver) {
        (Call::Free, _) => quote!(#rust(#(#args),*)),
        (Call::Own, Receiver::None) => quote!(Self::#rust(#(#args),*)),
        (Call::Own, _) => quote!(Self::#rust(self, #(#args),*)),
        (Call::Locked(core) | Call::Shared(core), Receiver::None) => quote!(<#core>::#rust(#(#args),*)),
        (Call::Locked(_), _) => {
            quote!(self.inner.lock().unwrap_or_else(::std::sync::PoisonError::into_inner).#rust(#(#args),*))
        }
        (Call::Shared(_), _) if s.is_async => quote!(inner.#rust(#(#args),*)),
        (Call::Shared(_), _) => quote!(self.inner.#rust(#(#args),*)),
    };
    let target = if s.is_async { quote!(#target.await) } else { target };
    let target = if s.detach { quote!(py.detach(|| #target)) } else { target };
    let fail = match &s.ret {
        Ret::Result(_) => fail_rule(s),
        _ => TokenStream::new(),
    };
    let pre = quote! {
        #pid
        #(#preludes)*
    };
    // `command(reply)`: sent now (a refusal raises at the call), the reply awaited by the host.
    if let Some(value) = &s.reply {
        let (inner, awaited_fail) = match result_ok(value) {
            Some(ok) => (ok, fail_rule(&Sig { err: None, ..s.clone() })),
            None => (value.clone(), TokenStream::new()),
        };
        let convert = match crate::shape::output(&inner, quote!(out)) {
            Some(o) if o.py => {
                let x = o.expr;
                quote!(let out = ::pyo3::Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(#x, py))?;)
            }
            Some(o) => {
                let x = o.expr;
                quote!(let out = #x;)
            }
            None => TokenStream::new(),
        };
        let rest = quote! {
            let out = #target;
            #fail
            let reply = out;
            ::lk_agents_abi::python::future_into_py(py, async move {
                let out = reply.await;
                #awaited_fail
                #convert
                ::std::result::Result::Ok(out)
            })
        };
        return (pre, rest);
    }
    let ok = match wrap {
        Some((class, Wrap::Mutex)) => quote!(#class { inner: ::std::sync::Mutex::new(out) }),
        Some((class, Wrap::Arc)) => quote!(#class { inner: ::std::sync::Arc::new(out) }),
        Some((class, Wrap::Value)) => quote!(#class { inner: out }),
        None => quote!(out),
    };
    let convert = match (stream_wrap(s), output(s), wrap) {
        (Some(w), _, _) => quote!(let out = #w;),
        (None, Some(o), None) if s.is_async && o.py => {
            let x = o.expr;
            quote!(let out = ::pyo3::Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(#x, py))?;)
        }
        (None, Some(o), None) => {
            let x = o.expr;
            quote!(let out = #x;)
        }
        _ => TokenStream::new(),
    };
    if s.is_async {
        let shared = matches!(call, Call::Shared(_)).then(|| quote!(let inner = ::std::sync::Arc::clone(&self.inner);));
        let rest = quote! {
            #shared
            ::lk_agents_abi::python::future_into_py(py, async move {
                let out = #target;
                #fail
                #convert
                ::std::result::Result::Ok(#ok)
            })
        };
        return (pre, rest);
    }
    let rest = quote! {
        let out = #target;
        #fail
        #convert
        ::std::result::Result::Ok(#ok)
    };
    (pre, rest)
}

fn result_ok(t: &syn::Type) -> Option<syn::Type> {
    let syn::Type::Path(p) = t else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    a.args.iter().find_map(|g| if let syn::GenericArgument::Type(t) = g { Some(t.clone()) } else { None })
}

/// A wrapper with a `#[host(kwargs)]` parameter, assembled by `__host_kwargs_<Struct>!`, which
/// puts its fields in the gap.
struct Spread<'a> {
    /// `impl Class` for a method (its own `#[pymethods]` block), `None` for a function.
    head: Option<TokenStream>,
    attrs: TokenStream,
    wrapper: syn::Ident,
    recv: TokenStream,
    ret: TokenStream,
    call: &'a Call<'a>,
    wrap: Option<(&'a syn::Ident, Wrap)>,
}

fn spread(s: &Sig, k: Spread) -> TokenStream {
    let kw = s.params.iter().position(|p| p.kwargs).expect("a kwargs parameter");
    let kp = &s.params[kw];
    let st = match &kp.ty {
        syn::Type::Reference(r) => (*r.elem).clone(),
        t => t.clone(),
    };
    let st_ident = match &st {
        syn::Type::Path(p) => p.path.segments.last().expect("a path").ident.clone(),
        _ => return quote!(compile_error!("#[host(kwargs)] is on a parameter of a struct type");),
    };
    let mac = format_ident!("__host_kwargs_{}", st_ident);
    let ins = inputs(s);
    let implied = |p: &crate::Param| {
        matches!(&p.ty, syn::Type::Path(tp) if tp.path.segments.last().is_some_and(|s| s.ident == "Maybe"))
            .then(|| quote!(::lk_agents_abi::Maybe::NotGiven))
    };
    let sig_item = |i: usize, p: &crate::Param| -> TokenStream {
        let star = match &s.kw_only {
            KwOnly::All => i == 0,
            KwOnly::From(first) => p.name == first,
            KwOnly::None => false,
        };
        let star = star.then(|| quote!(*,));
        let name = &p.name;
        match s.defaults.iter().find(|(d, _)| p.name == d).map(|(_, v)| v.clone()).or_else(|| implied(p)) {
            Some(value) => quote!(#star #name = #value,),
            None => quote!(#star #name,),
        }
    };
    let host: Vec<(usize, &crate::Param)> = s.params.iter().enumerate().filter(|(_, p)| !p.ambient).collect();
    let pos = host.iter().position(|(i, _)| *i == kw).expect("kwargs is a host parameter");
    let sig_pre: Vec<TokenStream> = host[..pos].iter().enumerate().map(|(j, (_, p))| sig_item(j, p)).collect();
    // The fields take the kwargs parameter's place, `*` before them when it starts the keywords.
    let star = match &s.kw_only {
        KwOnly::All => pos == 0,
        KwOnly::From(first) => kp.name == first,
        KwOnly::None => false,
    };
    let sig_pre = if star { quote!(#(#sig_pre)* *,) } else { quote!(#(#sig_pre)*) };
    let sig_post: Vec<TokenStream> = host[pos + 1..].iter().map(|(_, p)| sig_item(usize::MAX, p)).collect();
    let param = |i: usize| -> Option<TokenStream> {
        let p = &s.params[i];
        let input = ins[i].as_ref()?;
        let (name, ty) = (&p.name, &input.ty);
        Some(quote!(#name: #ty,))
    };
    let params_pre: Vec<TokenStream> = (0..kw).filter_map(param).collect();
    let params_post: Vec<TokenStream> = (kw + 1..s.params.len()).filter_map(param).collect();
    let (pre, rest) = body_parts(s, k.call, k.wrap);
    let (attrs, wrapper, recv, ret, var) = (k.attrs, k.wrapper, k.recv, k.ret, &kp.name);
    let head = match k.head {
        Some(h) => quote!(impl { #h }),
        None => quote!(fn),
    };
    quote! {
        #mac! {
            #head
            { #attrs }
            { #sig_pre } { #(#sig_post)* }
            { #wrapper }
            { #recv py: ::pyo3::Python<'py>, #(#params_pre)* } { #(#params_post)* }
            { #ret }
            { #var }
            { #pre } { #rest }
        }
    }
}

/// `#[doc = " line"]` per line, as `///` would: pyo3 strips the one leading space.
fn doc_attrs(lines: &[String]) -> TokenStream {
    let lines = lines.iter().map(|l| format!(" {l}"));
    quote!(#(#[doc = #lines])*)
}
