//! `#[host(stream(..))]` on a source type: a frozen `_core` class over a `sink` (sync `push` /
//! `flush` / `end_input`; optional) and a `source` (`recv` / `aclose`). A method returning the
//! halves names it (`export(stream = "Class")`). `doc` is per class: one source type may back
//! several. `error` defaults to `ProviderError`; `stub`/`stub_import`/`input_py` are the codegen's.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::Metas;

pub fn stream(attr: TokenStream, item: &syn::Item) -> syn::Result<TokenStream> {
    let m = Metas::parse(attr)?;
    let (ident, boxed_alias) = match item {
        syn::Item::Type(t) => (&t.ident, matches!(&*t.ty, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Box"))),
        syn::Item::Struct(s) => (&s.ident, false),
        other => return Err(syn::Error::new_spanned(other, "#[host(stream)] is for the source type (an alias or a struct)")),
    };
    let known = [
        "stream", "stream.name", "stream.doc", "stream.output", "stream.sink", "stream.input", "stream.push", "stream.arg",
        "stream.end_input", "stream.close_ends_input", "stream.optional_sink", "stream.source", "stream.error", "stream.deadline",
        "stream.stub", "stream.stub_import", "stream.input_py",
    ];
    m.only(&format!("stream {ident}"), &known)?;
    let parse = |key: &str| m.ty(key);
    let name = m.text("stream.name")?.ok_or_else(|| syn::Error::new_spanned(ident, "stream(name = \"..\") is the class name"))?;
    let class = format_ident!("{name}");
    let output = parse("stream.output")?.ok_or_else(|| syn::Error::new_spanned(ident, "stream(output = T) is what recv() yields"))?;
    let _ = output;
    let sink_ty = parse("stream.sink")?;
    let input = parse("stream.input")?;
    if sink_ty.is_some() != input.is_some() {
        return Err(syn::Error::new_spanned(ident, "`sink` and `input` come together"));
    }
    let source_ty: syn::Type = match parse("stream.source")? {
        Some(t) => t,
        None => syn::parse_quote!(#ident),
    };
    // A boxed source closes by value through its box; a struct source is boxed to close.
    let boxed = m.has("stream.source") || boxed_alias;
    let error: syn::Type = parse("stream.error")?.unwrap_or_else(|| syn::parse_quote!(::lk_agents_abi::errors::ProviderError));
    let deadline = m.expr("stream.deadline")?.unwrap_or_else(|| quote!(::lk_agents_abi::python::NATIVE_CLOSE_DEADLINE));
    let doc = m.text("stream.doc")?.map(|d| format!(" {d}"));
    let doc = doc.iter();
    let optional = m.text("stream.optional_sink")?;
    let end_input = m.flag("stream.end_input");
    let close_ends_input = m.flag("stream.close_ends_input");
    let rt = quote!(::lk_agents_abi::python);
    let err = quote!(|e: #error| #rt::HostException::to_py_err(&e));
    let slot = quote!(::std::sync::Arc<#rt::AsyncMutex<::std::option::Option<#source_ty>>>);
    let new_slot = quote!(::std::sync::Arc::new(#rt::AsyncMutex::new(::std::option::Option::Some(source))));
    let (sink_field, ctor, sink_fn) = match (&sink_ty, optional) {
        (Some(t), None) => (
            quote!(pub sink: ::std::sync::Arc<#t>,),
            quote! {
                pub fn new(sink: impl ::std::convert::Into<::std::sync::Arc<#t>>, source: #source_ty) -> Self {
                    #class { sink: sink.into(), source: #new_slot }
                }
                /// A method's returned `(sink, source)`.
                pub fn from_pair(sink: impl ::std::convert::Into<::std::sync::Arc<#t>>, source: #source_ty) -> Self {
                    Self::new(sink, source)
                }
            },
            TokenStream::new(),
        ),
        (Some(t), Some(msg)) => (
            quote!(pub sink: ::std::option::Option<::std::sync::Arc<#t>>,),
            quote! {
                pub fn new(sink: ::std::option::Option<impl ::std::convert::Into<::std::sync::Arc<#t>>>, source: #source_ty) -> Self {
                    #class { sink: sink.map(::std::convert::Into::into), source: #new_slot }
                }
                /// A method's returned `(sink, source)`.
                pub fn from_pair(sink: impl ::std::convert::Into<::std::sync::Arc<#t>>, source: #source_ty) -> Self {
                    Self::new(::std::option::Option::Some(sink), source)
                }
                /// A method's returned source, its input given at the call.
                pub fn from_source(source: #source_ty) -> Self {
                    Self::new(::std::option::Option::None::<::std::sync::Arc<#t>>, source)
                }
            },
            quote! {
                fn sink(&self) -> ::pyo3::PyResult<&::std::sync::Arc<#t>> {
                    self.sink.as_ref().ok_or_else(|| ::pyo3::exceptions::PyRuntimeError::new_err(#msg))
                }
            },
        ),
        (None, _) => (
            TokenStream::new(),
            quote! {
                pub fn new(source: #source_ty) -> Self {
                    #class { source: #new_slot }
                }
                /// A method's returned source.
                pub fn from_source(source: #source_ty) -> Self {
                    Self::new(source)
                }
            },
            TokenStream::new(),
        ),
    };
    let sink = if optional.is_some() { quote!(self.sink()?) } else { quote!(self.sink) };
    let mut methods = Vec::new();
    if let Some(input) = &input {
        let push = format_ident!("{}", m.text("stream.push")?.unwrap_or("push"));
        let arg = format_ident!("{}", m.text("stream.arg")?.unwrap_or("text"));
        methods.push(quote! {
            pub fn #push(&self, #arg: #input) -> ::pyo3::PyResult<()> {
                #rt::check_pid()?;
                #sink.push(#arg).map_err(#err)
            }
            pub fn flush(&self) -> ::pyo3::PyResult<()> {
                #rt::check_pid()?;
                #sink.flush().map_err(#err)
            }
        });
        if end_input {
            methods.push(quote! {
                pub fn end_input(&self) -> ::pyo3::PyResult<()> {
                    #rt::check_pid()?;
                    #sink.end_input();
                    ::std::result::Result::Ok(())
                }
            });
        }
    }
    let end = match (&sink_ty, close_ends_input, optional) {
        (Some(_), true, None) => quote!(self.sink.end_input();),
        (Some(_), true, Some(_)) => quote!(if let ::std::option::Option::Some(sink) = &self.sink { sink.end_input(); }),
        _ => TokenStream::new(),
    };
    let close = if boxed { quote!(src.close) } else { quote!(::std::boxed::Box::new(src).close) };
    Ok(quote! {
        #[cfg(feature = "python")]
        #(#[doc = #doc])*
        #[::pyo3::pyclass(frozen, name = #name, module = "livekit.agents._core")]
        pub struct #class {
            #sink_field
            pub source: #slot,
        }
        #[cfg(feature = "python")]
        const _: () = {
            #[allow(unused_imports)]
            use ::lk_agents_abi::{StreamSink as _, StreamSource as _};
            impl #class {
                #ctor
                #sink_fn
            }
            #[::pyo3::pymethods]
            impl #class {
                #(#methods)*
                pub fn recv<'py>(&self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                    #rt::check_pid()?;
                    let source = ::std::sync::Arc::clone(&self.source);
                    #rt::future_into_py(py, async move {
                        let mut guard = source.lock().await;
                        let ::std::option::Option::Some(src) = guard.as_mut() else { return ::std::result::Result::Ok(::std::option::Option::None) };
                        match src.next().await {
                            ::std::result::Result::Ok(::std::option::Option::Some(v)) => ::pyo3::Python::attach(|py| {
                                ::std::result::Result::Ok(::std::option::Option::Some(::pyo3::IntoPyObjectExt::into_py_any(v, py)?))
                            }),
                            ::std::result::Result::Ok(::std::option::Option::None) => ::std::result::Result::Ok(::std::option::Option::None),
                            ::std::result::Result::Err(e) => ::std::result::Result::Err((#err)(e)),
                        }
                    })
                }
                pub fn aclose<'py>(&self, py: ::pyo3::Python<'py>) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                    #rt::check_pid()?;
                    #end
                    let source = ::std::sync::Arc::clone(&self.source);
                    #rt::future_into_py(py, async move {
                        if let ::std::option::Option::Some(src) = source.lock().await.take() {
                            #close(#deadline).await;
                        }
                        ::std::result::Result::Ok(())
                    })
                }
            }
            #rt::inventory::submit! {
                #rt::Registration(|m| {
                    use ::pyo3::types::PyModuleMethods as _;
                    m.add_class::<#class>()
                })
            }
        };
    })
}
