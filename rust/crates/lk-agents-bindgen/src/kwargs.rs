//! `#[host(kwargs)]` / `#[host(client(..))]` on a config struct: one keyword per field. `client`
//! is a class's keyword constructor, handing the struct to `build(py, config)`. `kwargs` spreads a
//! marked parameter into keywords via `__host_kwargs_<Struct>!` (an expansion sees one item).
//! Fields take `#[host(kwarg(default = <expr>))]` and `#[host(kwarg(lenient))]`.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::Metas;

pub struct KwField {
    pub ident: syn::Ident,
    /// For the pyo3 signature.
    pub sig: TokenStream,
    pub param: TokenStream,
    pub prelude: Option<TokenStream>,
    /// `field` or `field: <conversion>`.
    pub init: TokenStream,
}

fn fields(st: &syn::ItemStruct) -> syn::Result<Vec<KwField>> {
    let syn::Fields::Named(named) = &st.fields else {
        return Err(syn::Error::new_spanned(&st.ident, "a keyword record has named fields: they are the keywords"));
    };
    let mut out = Vec::new();
    for f in &named.named {
        let ident = f.ident.clone().expect("named");
        let fm = Metas::of(&f.attrs)?;
        fm.only(&format!("{}.{ident}", st.ident), &["kwarg", "kwarg.default", "kwarg.lenient"])?;
        let default = fm.expr("kwarg.default")?;
        // A `Maybe<T>` field defaults to `NotGiven` without saying so, as a `Maybe` parameter does.
        let default = default.or_else(|| {
            matches!(&f.ty, syn::Type::Path(tp) if tp.path.segments.last().is_some_and(|s| s.ident == "Maybe"))
                .then(|| quote!(::lk_agents_abi::Maybe::NotGiven))
        });
        let ty = &f.ty;
        let label = ident.to_string();
        let input = if fm.flag("kwarg.lenient") {
            crate::shape::In {
                ty: quote!(&::pyo3::Bound<'py, ::pyo3::PyAny>),
                prelude: Some(quote!(let #ident: #ty = ::pyo3::types::PyAnyMethods::extract(#ident).unwrap_or_default();)),
                arg: quote!(#ident),
                py: true,
            }
        } else {
            crate::shape::input(&ident, &label, ty)
        };
        let host_ty = &input.ty;
        out.push(KwField {
            sig: match &default {
                Some(d) => quote!(#ident = #d),
                None => quote!(#ident),
            },
            param: quote!(#ident: #host_ty),
            prelude: input.prelude,
            init: {
                let arg = input.arg;
                if ident == arg.to_string() { quote!(#ident) } else { quote!(#ident: #arg) }
            },
            ident,
        });
    }
    Ok(out)
}

/// `#[host(kwargs)]`: the struct back, and the macro a wrapper spreads its fields through.
pub fn kwargs(attr: TokenStream, st: &syn::ItemStruct) -> syn::Result<TokenStream> {
    let m = Metas::parse(attr)?;
    m.only(&format!("struct {}", st.ident), &["kwargs"])?;
    let fs = fields(st)?;
    let ident = &st.ident;
    let mac = format_ident!("__host_kwargs_{}", ident);
    let sig = fs.iter().map(|f| &f.sig);
    let params = fs.iter().map(|f| &f.param);
    let preludes = fs.iter().filter_map(|f| f.prelude.as_ref());
    let inits = fs.iter().map(|f| &f.init);
    let (sig2, params2, inits2) = (sig.clone(), params.clone(), inits.clone());
    let preludes2 = preludes.clone();
    Ok(quote! {
        #[cfg(feature = "python")]
        #[doc(hidden)]
        #[allow(unused_macros)]
        macro_rules! #mac {
            // A method: its own `#[pymethods]` block (a macro call is no `#[pymethods]` item).
            (
                impl { $($head:tt)* }
                { $($attrs:tt)* }
                { $($sig_pre:tt)* } { $($sig_post:tt)* }
                { $name:ident }
                { $($params_pre:tt)* } { $($params_post:tt)* }
                { $ret:ty }
                { $var:ident }
                { $($pre:tt)* } { $($body:tt)* }
            ) => {
                #[::pyo3::pymethods]
                $($head)* {
                    $($attrs)*
                    #[pyo3(signature = ($($sig_pre)* #(#sig,)* $($sig_post)*))]
                    #[allow(clippy::too_many_arguments)]
                    fn $name<'py>($($params_pre)* #(#params,)* $($params_post)*) -> ::pyo3::PyResult<$ret> {
                        $($pre)*
                        #(#preludes)*
                        let $var = #ident { #(#inits),* };
                        $($body)*
                    }
                }
            };
            (
                fn
                { $($attrs:tt)* }
                { $($sig_pre:tt)* } { $($sig_post:tt)* }
                { $name:ident }
                { $($params_pre:tt)* } { $($params_post:tt)* }
                { $ret:ty }
                { $var:ident }
                { $($pre:tt)* } { $($body:tt)* }
            ) => {
                $($attrs)*
                #[pyo3(signature = ($($sig_pre)* #(#sig2,)* $($sig_post)*))]
                #[allow(clippy::too_many_arguments)]
                fn $name<'py>($($params_pre)* #(#params2,)* $($params_post)*) -> ::pyo3::PyResult<$ret> {
                    $($pre)*
                    #(#preludes2)*
                    let $var = #ident { #(#inits2),* };
                    $($body)*
                }
            };
        }
    })
}

/// `#[host(client(..))]`: the class's keyword constructor, from the struct's fields.
pub fn client(attr: TokenStream, st: &syn::ItemStruct) -> syn::Result<TokenStream> {
    let m = Metas::parse(attr)?;
    m.only(&format!("struct {}", st.ident), &["client", "client.class", "client.build", "client.positional"])?;
    let class = m.path("client.class")?.ok_or_else(|| syn::Error::new_spanned(&st.ident, "client(class = ..) names the class"))?;
    let build = match m.path("client.build")? {
        Some(b) => quote!(#b),
        None => quote!(#class::build),
    };
    let fs = fields(st)?;
    let positional = match m.ident("client.positional")? {
        None => 0,
        Some(p) => 1 + fs.iter().position(|f| f.ident == p).ok_or_else(|| syn::Error::new_spanned(&st.ident, format!("positional = {p} names no field")))?,
    };
    let sig = fs.iter().enumerate().map(|(i, f)| {
        let s = &f.sig;
        if i == positional { quote!(*, #s) } else { quote!(#s) }
    });
    let params = fs.iter().map(|f| &f.param);
    let preludes = fs.iter().filter_map(|f| f.prelude.as_ref());
    let inits = fs.iter().map(|f| &f.init);
    let ident = &st.ident;
    let doc = format!(" `{}`'s keyword constructor, from `{ident}`.", class.segments.last().map(|s| s.ident.to_string()).unwrap_or_default());
    Ok(quote! {
        #[cfg(feature = "python")]
        const _: () = {
            #[doc = #doc]
            #[::pyo3::pymethods]
            impl #class {
                #[new]
                #[pyo3(signature = (#(#sig),*))]
                #[allow(clippy::too_many_arguments)]
                fn __host_py_new<'py>(py: ::pyo3::Python<'py>, #(#params),*) -> ::pyo3::PyResult<Self> {
                    ::lk_agents_abi::python::check_pid()?;
                    #(#preludes)*
                    #build(py, #ident { #(#inits),* })
                }
            }
        };
    })
}
