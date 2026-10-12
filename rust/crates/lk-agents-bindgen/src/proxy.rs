//! `#[host(proxy(host = ".."))]` on a core trait: the core calling a host object. Expands to
//! `Py<Trait><B: HostBridge = HostHandle>`, `impl Trait` for it, and `HostProxy for dyn Trait`
//! (a registered native class first, else the proxy).
//!
//! Every method calls `_core_<method>`, crossing by signature: `fn m(&self) -> T` is read once at
//! construction (every call on a `live` trait, unless it lends a borrow); `fn m(&self, ..)` is
//! `call_soon_threadsafe`; `-> T` is called now under the GIL; `async` awaits on the host loop.
//! Returned `dyn` streams/views wrap the returned object on this proxy's loop; errors are the
//! captured host exception. `clock` keeps the caller's (else ambient) clock and hands it to views;
//! `cancel_on_drop` cancels an awaited host call when its Rust waiter goes away.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::cross::{from_py, head, is_unit, to_py};
use crate::{Metas, is_host};

struct Method {
    ident: syn::Ident,
    sig: syn::Signature,
    is_async: bool,
    params: Vec<(syn::Ident, syn::Type)>,
    /// `None`: returns nothing.
    ret: Option<syn::Type>,
    ret_ref: bool,
    action: Action,
}

enum Action {
    Core,
    Soon(String),
}

struct Proxy {
    trait_ident: syn::Ident,
    name: syn::Ident,
    host: Option<String>,
    live: bool,
    cancel_on_drop: bool,
    /// Keep the clock it was built with (the caller's, else ambient) and hand it to returned views.
    clock: bool,
    async_trait: bool,
    methods: Vec<Method>,
}

fn read(attr: TokenStream, tr: &syn::ItemTrait) -> syn::Result<Proxy> {
    let m = Metas::parse(attr)?;
    let at = &tr.ident;
    for x in &m.0 {
        let ok = ["proxy", "proxy.host", "proxy.name", "proxy.live", "proxy.cancel_on_drop", "proxy.clock"].contains(&x.key.as_str());
        if !ok {
            return Err(syn::Error::new_spanned(at, format!("proxy({}) has no in-place meaning", x.key.trim_start_matches("proxy."))));
        }
    }
    if !tr.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&tr.generics, "a generic trait has no single proxy"));
    }
    let host = m.text("proxy.host")?.map(str::to_owned);
    let name = match (m.ident("proxy.name")?, &host) {
        (Some(n), _) => n,
        (None, Some(_)) => format_ident!("Py{}", tr.ident),
        (None, None) => return Err(syn::Error::new_spanned(at, "proxy(name = \"..\") names the struct of a trait without `host`")),
    };
    let async_trait = tr.attrs.iter().any(|a| a.path().segments.last().is_some_and(|s| s.ident == "async_trait"));
    let mut methods = Vec::new();
    for item in &tr.items {
        let syn::TraitItem::Fn(f) = item else { continue };
        let fm = Metas::of(&f.attrs)?;
        let action = match (fm.head(), &host) {
            ("", Some(_)) => Action::Core,
            ("soon", None) => Action::Soon(fm.text("soon")?.unwrap_or_default().to_owned()),
            ("", None) if f.default.is_some() => continue,
            ("", None) => return Err(syn::Error::new_spanned(&f.sig.ident, "a required method with no marker: the proxy cannot implement it")),
            (other, Some(_)) => {
                return Err(syn::Error::new_spanned(&f.sig.ident, format!("host({other}): a bare proxy's methods carry no marker")));
            }
            (other, None) => return Err(syn::Error::new_spanned(&f.sig.ident, format!("host({other}) on a proxy method has no in-place meaning (soon)"))),
        };
        let mut sig = f.sig.clone();
        let mut params = Vec::new();
        for input in &mut sig.inputs {
            if let syn::FnArg::Typed(t) = input {
                t.attrs.retain(|a| !is_host(a));
                let syn::Pat::Ident(id) = &*t.pat else {
                    return Err(syn::Error::new_spanned(&t.pat, "a destructuring parameter has no name to convert"));
                };
                params.push((id.ident.clone(), (*t.ty).clone()));
            }
        }
        let (ret, ret_ref) = match &sig.output {
            syn::ReturnType::Default => (None, false),
            syn::ReturnType::Type(_, t) if is_unit(t) => (None, false),
            syn::ReturnType::Type(_, t) => (Some((**t).clone()), matches!(**t, syn::Type::Reference(_))),
        };
        if let Action::Soon(_) = &action
            && ret.as_ref().is_some_and(|t| head(t).is_none_or(|(h, _)| h != "Result"))
        {
            return Err(syn::Error::new_spanned(&f.sig.ident, "`soon` is fire-and-forget: it returns () or Result<(), E>"));
        }
        methods.push(Method { ident: f.sig.ident.clone(), is_async: sig.asyncness.is_some(), sig, params, ret, ret_ref, action });
    }
    Ok(Proxy {
        trait_ident: tr.ident.clone(),
        name,
        host,
        live: m.flag("proxy.live"),
        clock: m.flag("proxy.clock"),
        cancel_on_drop: m.flag("proxy.cancel_on_drop"),
        async_trait,
        methods,
    })
}

fn is_getter(m: &Method) -> bool {
    !m.is_async && m.params.is_empty() && m.ret.is_some() && matches!(m.action, Action::Core)
}

/// Every getter, unless the trait is `live`: then only one that lends a borrow.
fn once(p: &Proxy, m: &Method) -> bool {
    is_getter(m) && (!p.live || m.ret_ref)
}

/// The return without its borrow.
fn stored(t: &syn::Type) -> syn::Type {
    match t {
        syn::Type::Reference(r) => match &*r.elem {
            syn::Type::Path(p) if p.path.is_ident("str") => syn::parse_quote!(::std::string::String),
            other => other.clone(),
        },
        other => other.clone(),
    }
}

/// The proxy, and whether the code runs in `from_bridge` (locals) or a method (`self.`).
struct Cx<'a> {
    p: &'a Proxy,
    ctor: bool,
}

impl Cx<'_> {
    fn bridge(&self) -> TokenStream {
        if self.ctor { quote!(bridge) } else { quote!(self.bridge) }
    }
    fn clock(&self) -> TokenStream {
        match (self.p.clock, self.ctor) {
            (true, true) => quote!(::std::option::Option::Some(::std::clone::Clone::clone(&clock))),
            (true, false) => quote!(::std::option::Option::Some(::std::clone::Clone::clone(&self.clock))),
            (false, _) => quote!(::std::option::Option::None),
        }
    }
    fn handle(&self, o: &TokenStream) -> TokenStream {
        let bridge = self.bridge();
        quote!({
            let o: &::pyo3::Bound<'_, ::pyo3::PyAny> = #o;
            ::lk_agents_abi::python::HostHandle::from_parts(o.clone().unbind(), ::std::clone::Clone::clone(::lk_agents_abi::python::HostBridge::task_locals(&#bridge)))
        })
    }
}

/// `Arc<dyn X>` / `Box<dyn X>` (not a stream half): `("Arc" | "Box", X)`.
fn dyn_of(t: &syn::Type) -> Option<(String, syn::Path)> {
    let (h, args) = head(t)?;
    if (h != "Arc" && h != "Box") || args.len() != 1 {
        return None;
    }
    let syn::Type::TraitObject(o) = &args[0] else { return None };
    o.bounds.iter().find_map(|b| match b {
        syn::TypeParamBound::Trait(tb) => Some((h.clone(), tb.path.clone())),
        _ => None,
    })
}

fn is_stream_half(p: &syn::Path) -> Option<(bool, syn::Type)> {
    let seg = p.segments.last()?;
    let sink = match seg.ident.to_string().as_str() {
        "StreamSink" => true,
        "StreamSource" => false,
        _ => return None,
    };
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    a.args.iter().find_map(|g| if let syn::GenericArgument::Type(t) = g { Some((sink, t.clone())) } else { None })
}

/// Whether `t` crosses by [`crate::cross::from_py`]'s structural rules.
fn structural(t: &syn::Type) -> bool {
    const LEAVES: [&str; 20] = [
        "bool", "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize", "f32", "f64", "String", "char",
        "Duration", "Value", "HostRef", "MediaRef", "str",
    ];
    if matches!(t, syn::Type::Path(p) if p.path.segments.iter().any(|s| s.ident == "proto")) {
        return true;
    }
    let Some((h, args)) = head(t) else { return false };
    match h.as_str() {
        _ if LEAVES.contains(&h.as_str()) => true,
        "Option" | "Vec" if args.len() == 1 => structural(&args[0]),
        "HashMap" if args.len() == 2 => structural(&args[1]),
        "Arc" if args.len() == 1 => !matches!(args[0], syn::Type::TraitObject(_)),
        _ => false,
    }
}

/// `o: &Bound<PyAny>` as a `PyResult<T>` expression.
fn value(cx: &Cx, t: &syn::Type, o: &TokenStream) -> syn::Result<TokenStream> {
    if structural(t) {
        return Ok(from_py(t, o));
    }
    if let Some((h, args)) = head(t)
        && args.len() == 1
    {
        // An absent view, or a list of them.
        if h == "Option" {
            let v = value(cx, &args[0], o)?;
            return Ok(quote!(if #o.is_none() { ::std::result::Result::Ok(::std::option::Option::None) } else { (#v).map(::std::option::Option::Some) }));
        }
        if h == "Vec" {
            let v = value(cx, &args[0], &quote!(&i))?;
            return Ok(quote!(#o.try_iter().and_then(|it| it.map(|i| i.and_then(|i| #v)).collect::<::pyo3::PyResult<::std::vec::Vec<_>>>())));
        }
    }
    // A tuple the signature spells as views or data: the host's tuple, item by item. (A stream
    // pair spelled through aliases asks its type, below.)
    if let syn::Type::Tuple(tu) = t
        && !tu.elems.is_empty()
        && tu.elems.iter().all(|e| structural(e) || dyn_of(e).is_some_and(|(_, p)| is_stream_half(&p).is_none()))
    {
        let mut lets = Vec::new();
        let mut vals = Vec::new();
        for (i, e) in tu.elems.iter().enumerate() {
            let ti = format_ident!("t{i}");
            let v = value(cx, e, &quote!((&#ti)))?;
            lets.push(quote!(let #ti = t.get_item(#i)?;));
            vals.push(quote!((#v)?));
        }
        return Ok(quote!((|| -> ::pyo3::PyResult<_> {
            use ::pyo3::types::PyTupleMethods as _;
            let t = #o.cast::<::pyo3::types::PyTuple>()?;
            #(#lets)*
            ::std::result::Result::Ok((#(#vals,)*))
        })()));
    }
    // A boxed or shared view of another proxied trait.
    if let Some((wrapper, path)) = dyn_of(t)
        && wrapper == "Arc"
        && is_stream_half(&path).is_none()
    {
        let handle = cx.handle(o);
        let clock = cx.clock();
        return Ok(quote!(<dyn #path as ::lk_agents_abi::python::HostProxy>::view(#handle, #clock).map(::std::sync::Arc::from)));
    }
    if let Some((wrapper, path)) = dyn_of(t)
        && wrapper == "Box"
        && is_stream_half(&path).is_none()
    {
        let handle = cx.handle(o);
        let clock = cx.clock();
        return Ok(quote!(<dyn #path as ::lk_agents_abi::python::HostProxy>::view(#handle, #clock)));
    }
    // Anything else asks its type: a host stream or a view (`ProxyReturn`), else its own
    // `FromPyObject` -- a record, a string type, a class, a tuple of those.
    let handle = cx.handle(o);
    let clock = cx.clock();
    Ok(quote!({
        #[allow(unused_imports)]
        use ::lk_agents_abi::python::{ProxyValueClass as _, ProxyValuePlain as _, ProxyValueReturn as _};
        (&&&::lk_agents_abi::python::ProxyValue::<#t>::new()).read(#o, #handle, #clock)
    }))
}

/// `|py| (a, b,).into_pyobject(py)`: the host call's arguments.
fn args_tuple(m: &Method) -> TokenStream {
    if m.params.is_empty() {
        return quote!(|py| ::std::result::Result::Ok(::pyo3::types::PyTuple::empty(py)));
    }
    let items = m.params.iter().map(|(n, t)| to_py(t, &quote!(#n), false));
    quote!(|py| ::pyo3::IntoPyObject::into_pyobject((#(#items,)*), py))
}

pub fn proxy(attr: TokenStream, tr: &syn::ItemTrait) -> syn::Result<TokenStream> {
    let p = read(attr, tr)?;
    let name = &p.name;
    let tr_ident = &p.trait_ident;
    let rt = quote!(::lk_agents_abi::python);
    let bridge_ty = if p.cancel_on_drop { quote!(#rt::CancellingHandle) } else { quote!(#rt::HostHandle) };
    let cx = Cx { p: &p, ctor: true };
    let mut fields = Vec::new();
    let mut lets = Vec::new();
    for m in &p.methods {
        if !once(&p, m) {
            continue;
        }
        let ident = &m.ident;
        let ret = m.ret.as_ref().expect("a getter returns");
        let ty = stored(ret);
        let core = format!("_core_{ident}");
        let conv = value(&cx, &ty, &quote!(o))?;
        lets.push(quote! {
            let #ident: #ty = #rt::HostBridge::read(&bridge, "", |o| { let o = &o.call_method0(#core)?; #conv })?;
        });
        fields.push((ident.clone(), ty));
    }
    let clock_ty = quote!(::std::sync::Arc<dyn ::lk_agents_abi::Clock>);
    let clock_field = p.clock.then(|| quote!(pub clock: #clock_ty,));
    let clock_param = p.clock.then(|| quote!(, clock: #clock_ty));
    let clock_name = p.clock.then(|| quote!(clock));
    let clock_arg = p.clock.then(|| quote!(, clock.unwrap_or_else(<#clock_ty as #rt::Ambient>::ambient)));
    let field_decls = fields.iter().map(|(n, t)| quote!(pub #n: #t));
    let field_names: Vec<_> = fields.iter().map(|(n, _)| n.clone()).collect();
    let surface = match (&p.host, p.methods.first()) {
        (Some(host), Some(first)) => {
            let probe = format!("_core_{}", first.ident);
            quote!(let bridge = #rt::HostBridge::core_surface(bridge, #host, #probe)?;)
        }
        _ => TokenStream::new(),
    };
    let unread = p.methods.iter().all(|m| once(&p, m));
    let dead = unread.then(|| quote!(#[allow(dead_code)]));
    let mut bodies = Vec::new();
    let mcx = Cx { p: &p, ctor: false };
    for m in &p.methods {
        let sig = &m.sig;
        let body = method(&mcx, m)?;
        bodies.push(quote!(#sig { #body }));
    }
    let async_trait = p.async_trait.then(|| quote!(#[::async_trait::async_trait]));
    let doc = format!(" `{tr_ident}` over a host object's `_core_*` surface (`#[host(proxy)]`).");
    Ok(quote! {
        #[cfg(feature = "python")]
        #[doc = #doc]
        pub struct #name<B: #rt::HostBridge = #bridge_ty> {
            #dead
            pub bridge: B,
            #(#field_decls,)*
            #clock_field
        }
        #[cfg(feature = "python")]
        const _: () = {
            #[allow(unused_imports)]
            use ::pyo3::types::{PyAnyMethods as _, PyDictMethods as _};
            impl<B: #rt::HostBridge> #name<B> {
                /// The proxy over `bridge`'s object: its getters read once, here.
                pub fn from_bridge(bridge: B #clock_param) -> ::pyo3::PyResult<Self> {
                    #surface
                    #(#lets)*
                    ::std::result::Result::Ok(#name { bridge, #(#field_names,)* #clock_name })
                }
            }
            #async_trait
            impl<B: #rt::HostBridge> #tr_ident for #name<B> {
                #(#bodies)*
            }
            impl #rt::HostProxy for dyn #tr_ident {
                fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, clock: ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
                    let _ = &clock;
                    if let ::std::option::Option::Some(native) = #rt::native::<dyn #tr_ident>(obj) {
                        return ::std::result::Result::Ok(native);
                    }
                    let bridge = <#bridge_ty>::new(obj.py(), obj.clone().unbind())?;
                    ::std::result::Result::Ok(::std::sync::Arc::new(#name::from_bridge(bridge #clock_arg)?))
                }
                fn view(handle: #rt::HostHandle, clock: ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
                    let _ = &clock;
                    ::std::result::Result::Ok(::std::boxed::Box::new(#name::<#rt::HostHandle>::from_bridge(handle #clock_arg)?))
                }
            }
        };
    })
}

fn method(cx: &Cx, m: &Method) -> syn::Result<TokenStream> {
    let rt = quote!(::lk_agents_abi::python);
    let err = quote!(#rt::host_err);
    let ident = &m.ident;
    let args = args_tuple(m);
    if let Action::Soon(host) = &m.action {
        let r = quote!(#rt::HostBridge::call_soon(&self.bridge, #host, #args));
        return Ok(match &m.ret {
            None => quote!(let _ = #r;),
            Some(_) => quote!(#r.map_err(#err)),
        });
    }
    let host = format!("_core_{ident}");
    if once(cx.p, m) {
        return Ok(if m.ret_ref { quote!(&self.#ident) } else { quote!(::std::clone::Clone::clone(&self.#ident)) });
    }
    let Some(ret) = &m.ret else {
        return Ok(if m.is_async {
            quote!(let _ = #rt::HostBridge::await_method(&self.bridge, #host, #args).await;)
        } else {
            quote!(let _ = #rt::HostBridge::call_soon(&self.bridge, #host, #args);)
        });
    };
    let name = cx.p.name.to_string();
    if !m.is_async {
        let conv = match head(ret) {
            Some((h, a)) if h == "Result" => match a.first() {
                Some(t) if is_unit(t) => quote!({ let _ = o; ::std::result::Result::Ok(()) }),
                Some(t) => value(cx, t, &quote!(o))?,
                None => return Err(syn::Error::new_spanned(ident, "Result<T, E>")),
            },
            _ => value(cx, ret, &quote!(o))?,
        };
        let invoke = if m.params.is_empty() {
            quote!(let _ = py; let o = &o.call_method0(#host)?;)
        } else {
            quote!(let o = &o.call_method1(#host, (#args)(py)?)?;)
        };
        let call = quote!(#rt::HostBridge::read(&self.bridge, "", |o| { let py = o.py(); #invoke #conv }));
        let warn = format!("{name}.{host} failed: {{}}");
        return Ok(match head(ret) {
            Some((h, _)) if h == "Result" => quote!(#call.map_err(#err)),
            Some((h, _)) if h == "Option" => quote! {
                match #call {
                    ::std::result::Result::Ok(v) => v,
                    ::std::result::Result::Err(e) => { ::lk_agents_abi::python::log::warn!(#warn, e); ::std::option::Option::None }
                }
            },
            _ => quote! {
                match #call {
                    ::std::result::Result::Ok(v) => v,
                    ::std::result::Result::Err(e) => { ::lk_agents_abi::python::log::warn!(#warn, e); ::std::default::Default::default() }
                }
            },
        });
    }
    let call = quote!(#rt::HostBridge::await_method(&self.bridge, #host, #args).await);
    let (h, targs) = head(ret).ok_or_else(|| syn::Error::new_spanned(ident, "an awaited method returns (), a Result or an Option"))?;
    match h.as_str() {
        "Result" => {
            let t = targs.first().ok_or_else(|| syn::Error::new_spanned(ident, "Result<T, E>"))?;
            if is_unit(t) {
                return Ok(quote!(#call.map(|_| ()).map_err(#err)));
            }
            let conv = value(cx, t, &quote!(o))?;
            Ok(quote! {
                let out = match #call {
                    ::std::result::Result::Ok(out) => out,
                    ::std::result::Result::Err(e) => return ::std::result::Result::Err(#err(e)),
                };
                ::pyo3::Python::attach(|py| { let o = out.bind(py); #conv }).map_err(#err)
            })
        }
        "Option" => {
            let inner = targs.first().ok_or_else(|| syn::Error::new_spanned(ident, "Option<T>"))?;
            match head(inner) {
                Some((ih, iargs)) if ih == "Result" => {
                    let t = iargs.first().ok_or_else(|| syn::Error::new_spanned(ident, "Result<T, E>"))?;
                    let conv = value(cx, t, &quote!(o))?;
                    Ok(quote! {
                        match #call {
                            ::std::result::Result::Err(e) => ::std::option::Option::Some(::std::result::Result::Err(#err(e))),
                            ::std::result::Result::Ok(out) => ::pyo3::Python::attach(|py| {
                                let o = out.bind(py);
                                if o.is_none() {
                                    return ::std::option::Option::None;
                                }
                                ::std::option::Option::Some((#conv).map_err(#err))
                            }),
                        }
                    })
                }
                _ => awaited_plain(cx, m, &call, inner, true),
            }
        }
        _ => awaited_plain(cx, m, &call, ret, false),
    }
}

/// No `Result`: a failure is logged and reads as the default; with `opt`, as `None`.
fn awaited_plain(cx: &Cx, m: &Method, call: &TokenStream, t: &syn::Type, opt: bool) -> syn::Result<TokenStream> {
    let host = format!("_core_{}", m.ident);
    let name = cx.p.name.to_string();
    let failed = format!("{name}.{host} failed: {{}}");
    let unreadable = format!("{name}.{host} returned what it cannot be read as: {{}}");
    let conv = value(cx, t, &quote!(o))?;
    let fallback = if opt { quote!(::std::option::Option::None) } else { quote!(::std::default::Default::default()) };
    let none = opt.then(|| quote!(if o.is_none() { return ::std::option::Option::None; }));
    let ok = if opt { quote!(::std::option::Option::Some(v)) } else { quote!(v) };
    Ok(quote! {
        let out = match #call {
            ::std::result::Result::Ok(out) => out,
            ::std::result::Result::Err(e) => { ::lk_agents_abi::python::log::warn!(#failed, e); return #fallback; }
        };
        ::pyo3::Python::attach(|py| {
            let o = out.bind(py);
            #none
            match #conv {
                ::std::result::Result::Ok(v) => #ok,
                ::std::result::Result::Err(e) => { ::lk_agents_abi::python::log::warn!(#unreadable, e); #fallback }
            }
        })
    })
}
