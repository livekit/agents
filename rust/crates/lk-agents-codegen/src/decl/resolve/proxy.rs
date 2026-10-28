//! Proxy declarations: a `#[host(proxy)]` trait whose methods declare how the core calls the host.

use super::*;

impl Resolver<'_> {
    /// A declaration-model proxy: every method declares its call and its context parameter.
    pub(super) fn proxy_trait(&self, tr: &syn::ItemTrait, proxy: &Entry) -> Result<Proxy, String> {
        let at = self.at;
        voc::only(at, &proxy.nested, &["host"])?;
        let host = voc::text(at, &proxy.nested, "host")?.ok_or_else(|| {
            format!("{at}: proxy(host = \"module.Class\") names the class it calls")
        })?;
        if !tr.generics.params.is_empty() {
            return Err(format!("{at}: a generic trait has no single proxy"));
        }
        let k = self.krate;
        let module = &k.modules[self.module];
        let item = module
            .items
            .iter()
            .position(|i| matches!(i, syn::Item::Trait(t) if t.ident == tr.ident))
            .expect("read from here");
        let trait_path = k.extern_path(self.module, item).map_err(|e| format!("{at}: {e}"))?;
        let mut methods = Vec::new();
        for inner in &tr.items {
            match inner {
                // A provided method without a key is native only: the proxy keeps its default.
                syn::TraitItem::Fn(f)
                    if f.default.is_some()
                        && voc::read(&f.attrs).map_err(|e| format!("{at}: {e}"))?.is_empty() => {}
                syn::TraitItem::Fn(f) => methods.push(self.proxy_method(f, &trait_path)?),
                other => {
                    return Err(format!(
                        "{at}: a proxy declares methods only, not `{}`",
                        quote::quote!(#other)
                    ));
                }
            }
        }
        if methods.iter().filter(|m| m.kind == ProxyCall::Bind).count() > 1 {
            return Err(format!("{at}: a proxy binds one value"));
        }
        Ok(Proxy {
            rust_name: tr.ident.to_string(),
            trait_path,
            family: module.path.first().cloned().ok_or_else(|| {
                format!("{at}: a proxy is declared in a subject module, not the crate root")
            })?,
            source: module.file.clone(),
            host,
            async_trait: tr
                .attrs
                .iter()
                .any(|a| a.path().segments.last().is_some_and(|s| s.ident == "async_trait")),
            methods,
        })
    }

    fn proxy_method(&self, f: &syn::TraitItemFn, trait_path: &str) -> Result<ProxyMethod, String> {
        let at = format!("{}::{}", self.at, f.sig.ident);
        let entries = voc::read(&f.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(&at, &entries, &["await", "call", "notify", "read_once", "bind", "name"])?;
        let kinds = [
            ("await", ProxyCall::Await),
            ("call", ProxyCall::Call),
            ("notify", ProxyCall::Notify),
            ("read_once", ProxyCall::ReadOnce),
            ("bind", ProxyCall::Bind),
        ];
        let mut declared = kinds.iter().filter(|(k, _)| voc::find(&entries, k).is_some());
        let (key, kind) = match (declared.next(), declared.next()) {
            (Some((key, kind)), None) => (*key, *kind),
            _ => {
                return Err(format!(
                    "{at}: a proxy method declares exactly one of await, call, notify, read_once, bind"
                ));
            }
        };
        let receiver = f.sig.receiver();
        // `self: Box<Self>`: the call consumes the proxy (a writer's close), awaited.
        let consumes = receiver.is_some_and(boxed_self);
        let shared_receiver =
            receiver.is_some_and(shared) || (consumes && kind == ProxyCall::Await);
        let sync = f.sig.asyncness.is_none();
        let inputs = f.sig.inputs.iter().filter(|i| matches!(i, syn::FnArg::Typed(_))).count();
        let fits = shared_receiver
            && match kind {
                ProxyCall::Await => !sync,
                ProxyCall::Call => sync,
                ProxyCall::Notify => sync && matches!(f.sig.output, syn::ReturnType::Default),
                ProxyCall::ReadOnce => sync && inputs == 0,
                ProxyCall::Bind => sync && inputs == 1,
            };
        if !fits {
            return Err(format!(
                "{at}: the signature does not fit `{key}` (await: `async fn(&self | self: Box<Self>, ..) -> Result<T, E>`; call: `fn(&self, ..) -> Result<T, E>`; notify: `fn(&self, ..)` returning nothing; read_once: `fn(&self) -> T` or `-> &T`; bind: `fn(&self, x: &X) -> Option<Arc<dyn Self>>`)"
            ));
        }
        let mut params = Vec::new();
        let mut context = None;
        for input in f.sig.inputs.iter() {
            let syn::FnArg::Typed(pt) = input else { continue };
            let syn::Pat::Ident(ident) = &*pt.pat else {
                return Err(format!("{at}: a parameter is a plain name"));
            };
            let name = ident.ident.to_string();
            let name = name.trim_start_matches('_').to_owned();
            let pat = format!("{at}({name})");
            let entries = voc::read(&pt.attrs).map_err(|e| format!("{pat}: {e}"))?;
            voc::only(&pat, &entries, &["context"])?;
            if voc::find(&entries, "context").is_some() && context.replace(params.len()).is_some() {
                return Err(format!("{at}: one parameter is the call's #[host(context)]"));
            }
            let (by_ref, ty) = match &*pt.ty {
                syn::Type::Reference(r) if r.mutability.is_none() => {
                    (true, self.borrowed(&r.elem, &pat)?)
                }
                syn::Type::Reference(_) => {
                    return Err(format!("{pat}: a parameter is never `&mut`"));
                }
                other => match optional_ref(other) {
                    // `Option<&T>`: lent when present.
                    Some(elem) => (true, Ty::Optional(Box::new(self.borrowed(elem, &pat)?))),
                    None => (false, self.ty(other, &pat)?),
                },
            };
            // It crosses to the host.
            check_output(&pat, &ty)?;
            params.push(Param { name, ty, by_ref, kw_only: false, default: None });
        }
        let mut lends = false;
        let ret = match kind {
            ProxyCall::Notify => ProxyRet { ok: Ty::Unit, refusal: None, err: None },
            // An awaited call returning nothing (a close): its failure is logged, as a notify's is.
            ProxyCall::Await if matches!(f.sig.output, syn::ReturnType::Default) => {
                ProxyRet { ok: Ty::Unit, refusal: None, err: None }
            }
            ProxyCall::Await | ProxyCall::Call => self.proxy_ret(&f.sig.output, &at)?,
            ProxyCall::ReadOnce => {
                let syn::ReturnType::Type(_, t) = &f.sig.output else {
                    return Err(format!("{at}: read_once returns the value it reads"));
                };
                // `-> &T` lends the value the proxy holds.
                let ok = match &**t {
                    syn::Type::Reference(r) if r.mutability.is_none() => {
                        lends = true;
                        self.borrowed(&r.elem, &at)?
                    }
                    t => self.ty(t, &at)?,
                };
                check_input(&at, &ok)?;
                ProxyRet { ok, refusal: None, err: None }
            }
            ProxyCall::Bind => {
                let syn::ReturnType::Type(_, t) = &f.sig.output else {
                    return Err(format!("{at}: bind returns `Option<Arc<dyn Self>>`"));
                };
                let ok = self.ty(t, &at)?;
                let own = matches!(&ok, Ty::Optional(p) if matches!(&**p, Ty::Proxy { rust, .. } if rust == trait_path));
                if !own || context.is_some() {
                    return Err(format!(
                        "{at}: bind takes the value it binds and returns `Option<Arc<dyn Self>>` of this trait"
                    ));
                }
                ProxyRet { ok, refusal: None, err: None }
            }
        };
        let wrap = match voc::find(&entries, "bind") {
            Some(bind) => {
                voc::only(&at, &bind.nested, &["wrap"])?;
                match voc::rust(&at, &bind.nested, "wrap")? {
                    None => None,
                    Some(t) => match self.shape(self.module, &t, &[], &at)? {
                        Shape::Named { found: Found::Item { module, item }, args }
                            if args.is_empty() =>
                        {
                            Some(
                                self.krate
                                    .extern_path(module, item)
                                    .map_err(|e| format!("{at}: {e}"))?,
                            )
                        }
                        _ => return Err(format!("{at}: bind(wrap = T) names a crate type")),
                    },
                }
            }
            None => None,
        };
        Ok(ProxyMethod {
            rust_name: f.sig.ident.to_string(),
            name: voc::text(&at, &entries, "name")?.unwrap_or_else(|| f.sig.ident.to_string()),
            kind,
            params,
            context,
            ret,
            wrap,
            consumes,
            lends,
        })
    }

    /// `Result<T, E>` or `Result<Result<T, R>, E>`: a host failure is always reported.
    fn proxy_ret(&self, out: &syn::ReturnType, at: &str) -> Result<ProxyRet, String> {
        let result = |shape: &Shape| match shape {
            Shape::Named { found: Found::External(path), args }
                if is(path, &["std", "result", "Result"]) && args.len() == 2 =>
            {
                Some((clone_shape(&args[0]), clone_shape(&args[1])))
            }
            _ => None,
        };
        let refused = || {
            format!(
                "{at}: an awaited proxy call returns `Result<T, E>` (E a declared #[host(error)]): a host failure is reported, never swallowed"
            )
        };
        let syn::ReturnType::Type(_, t) = out else { return Err(refused()) };
        let (ok, err) = result(&self.shape(self.module, t, &[], at)?).ok_or_else(refused)?;
        let (_, _, err, _) = self.error_item(&err, at)?;
        let (ok, refusal) = match result(&ok) {
            Some((ok, refusal)) => (ok, Some(self.refusal(&refusal, at)?)),
            None => (ok, None),
        };
        let ok = self.classify(&ok, at)?;
        if ok != Ty::Unit {
            // It crosses from the host.
            check_input(at, &ok)?;
        }
        Ok(ProxyRet { ok, refusal, err: Some(err) })
    }

    /// `#[host(error(raised = "module.Class"))]`.
    fn refusal(&self, shape: &Shape, at: &str) -> Result<Refusal, String> {
        let (_, _, rust, entries) = self.error_item(shape, at)?;
        let declared = voc::find(&entries, "error").expect("error_item checked it");
        let raised = voc::text(at, &declared.nested, "raised")?.ok_or_else(|| {
            format!("{at}: `{rust}` is a refusal the host raises: #[host(error(raised = \"module.Class\"))]")
        })?;
        let (module, class) = raised
            .rsplit_once('.')
            .ok_or_else(|| format!("{at}: raised = \"{raised}\" is a dotted class path"))?;
        Ok(Refusal { rust, module: module.to_owned(), class: class.to_owned() })
    }
}

/// `Option<&T>`'s `T`.
fn optional_ref(t: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(p) = t else { return None };
    let [seg] = p.path.segments.iter().collect::<Vec<_>>()[..] else { return None };
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    let mut args = a.args.iter();
    match (args.next(), args.next()) {
        (Some(syn::GenericArgument::Type(syn::Type::Reference(r))), None)
            if seg.ident == "Option" && r.mutability.is_none() =>
        {
            Some(&r.elem)
        }
        _ => None,
    }
}

/// `self: Box<Self>`.
fn boxed_self(r: &syn::Receiver) -> bool {
    r.reference.is_none()
        && matches!(&*r.ty, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Box"))
}
