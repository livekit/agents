//! ABI declarations into the IR. Every type a declaration names is found through the crate's
//! modules and `use`s ([`Crate::resolve`]), aliases are expanded with their generic arguments,
//! and the result is classified by what its declaration says. Anything else fails, naming the
//! declaration.

use super::tree::{Crate, Found};
use super::vocabulary::{self as voc, Entry, Value};
use super::*;

/// Every `#[host(class)]` of the crate, in module order.
pub fn declarations(krate: &Crate) -> Result<Declarations, String> {
    let mut out = Declarations::default();
    for (m, module) in krate.modules.iter().enumerate() {
        for item in &module.items {
            let syn::Item::Trait(tr) = item else { continue };
            let entries = voc::read(&tr.attrs)
                .map_err(|e| format!("{}: trait {}: {e}", module.file, tr.ident))?;
            let at = format!("{}: {}", module.file, tr.ident);
            let resolver = || Resolver { krate, module: m, at: &at, assoc: Vec::new() };
            if let Some(class) = voc::find(&entries, "class") {
                voc::only(&at, &entries, &["class"])?;
                out.classes.push(resolver().class(tr, Some(class))?);
            } else if voc::find(&entries, "functions").is_some() {
                voc::only(&at, &entries, &["functions"])?;
                out.classes.push(resolver().class(tr, None)?);
            } else if let Some(proxy) = voc::find(&entries, "proxy")
                && declares_calls(tr)
            {
                out.proxies.push(resolver().proxy_trait(tr, proxy)?);
            }
        }
    }
    Ok(out)
}

struct Resolver<'a> {
    krate: &'a Crate,
    /// The module the declaration is written in.
    module: usize,
    at: &'a str,
    /// The class's associated types, once resolved: `Self::Chunks` reads them.
    assoc: Vec<Assoc>,
}

/// A path's meaning with its generic arguments, before classification.
enum Shape {
    Named {
        found: Found,
        args: Vec<Shape>,
    },
    Tuple(Vec<Shape>),
    /// `dyn Trait<..> + Send + ..`: the first bound.
    Dyn {
        found: Found,
        args: Vec<Shape>,
    },
    SelfType,
    SelfAssoc(String),
}

impl Resolver<'_> {
    /// A class, or functions when `class` is `None`.
    fn class(&mut self, tr: &syn::ItemTrait, class: Option<&Entry>) -> Result<Class, String> {
        let at = self.at;
        let nested = class.map(|c| c.nested.as_slice()).unwrap_or_default();
        voc::only(at, nested, &["name", "native"])?;
        let name = voc::text(at, nested, "name")?.unwrap_or_else(|| tr.ident.to_string());
        let native = match voc::rust(at, nested, "native")? {
            None => Vec::new(),
            Some(t) => match self.shape(self.module, &t, &[], at)? {
                Shape::Named { found: Found::Item { module, item }, .. } => {
                    match self.proxy(module, item, at)? {
                        Ty::Proxy { rust, .. } => vec![rust],
                        _ => unreachable!("proxy() returns a proxy"),
                    }
                }
                _ => return Err(format!("{at}: `native` names a #[host(proxy)] trait")),
            },
        };
        let krate = self.krate;
        let module = &krate.modules[self.module];
        let item = module
            .items
            .iter()
            .position(|i| matches!(i, syn::Item::Trait(t) if t.ident == tr.ident))
            .expect("read from here");
        let mut methods = Vec::new();
        for inner in &tr.items {
            match inner {
                syn::TraitItem::Type(t) => {
                    let assoc = self.assoc_type(t)?;
                    self.assoc.push(assoc);
                }
                syn::TraitItem::Fn(f) => methods.push(self.method(f)?),
                other => {
                    return Err(format!(
                        "{at}: only methods and associated types are declarable, not `{}`",
                        quote::quote!(#other)
                    ));
                }
            }
        }
        let form = match class {
            None if methods.iter().all(|m| m.kind == MethodKind::Function) => Form::Functions,
            None => {
                return Err(format!("{at}: every method of #[host(functions)] is a `function`"));
            }
            Some(_) if methods.iter().any(|m| m.kind == MethodKind::Function) => {
                return Err(format!(
                    "{at}: a class has no `function`; declare it in a #[host(functions)] trait"
                ));
            }
            Some(_) if methods.iter().any(|m| m.mutable) => {
                if methods.iter().any(|m| m.kind == MethodKind::Await) || !native.is_empty() {
                    return Err(format!(
                        "{at}: a class with `&mut self` methods is owned by one host object: no `await` method, no `native`"
                    ));
                }
                Form::MutableClass
            }
            Some(_) => Form::Class,
        };
        Ok(Class {
            form,
            name,
            rust_name: tr.ident.to_string(),
            trait_path: krate.extern_path(self.module, item).map_err(|e| format!("{at}: {e}"))?,
            family: module.path.first().cloned().ok_or_else(|| {
                format!("{at}: a class is declared in a subject module, not the crate root")
            })?,
            source: module.file.clone(),
            doc: doc(&tr.attrs),
            native,
            methods,
        })
    }

    fn assoc_type(&self, t: &syn::TraitItemType) -> Result<Assoc, String> {
        let at = format!("{}::{}", self.at, t.ident);
        let entries = voc::read(&t.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(&at, &entries, &["source"])?;
        let source = voc::find(&entries, "source").ok_or_else(|| format!("{at}: an associated type declares its kind: #[host(source(item = T, class = \"X\"))]"))?;
        voc::only(&at, &source.nested, &["item", "class"])?;
        let item = voc::rust(&at, &source.nested, "item")?
            .ok_or_else(|| format!("{at}: source(item = T) is what it yields"))?;
        let class = voc::text(&at, &source.nested, "class")?
            .ok_or_else(|| format!("{at}: source(class = \"X\") is its stream class"))?;
        let item = self.ty(&item, &at)?;
        Ok(Assoc { rust_name: t.ident.to_string(), ty: Ty::Source { item: Box::new(item), class } })
    }

    fn method(&self, f: &syn::TraitItemFn) -> Result<Method, String> {
        let at = format!("{}::{}", self.at, f.sig.ident);
        let entries = voc::read(&f.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(&at, &entries, &["new", "call", "await", "getter", "function", "name"])?;
        let kinds: Vec<MethodKind> = [
            ("new", MethodKind::New),
            ("call", MethodKind::Call),
            ("await", MethodKind::Await),
            ("getter", MethodKind::Getter),
            ("function", MethodKind::Function),
        ]
        .into_iter()
        .filter(|(k, _)| voc::find(&entries, k).is_some())
        .map(|(_, kind)| kind)
        .collect();
        let [kind] = kinds[..] else {
            return Err(format!("{at}: declare exactly one of new, call, await, getter, function"));
        };
        let receiver = f.sig.receiver();
        let shape_ok = match kind {
            MethodKind::New | MethodKind::Function => {
                receiver.is_none() && f.sig.asyncness.is_none()
            }
            MethodKind::Await => receiver.is_some_and(shared) && f.sig.asyncness.is_some(),
            MethodKind::Getter => receiver.is_some_and(shared) && f.sig.asyncness.is_none(),
            MethodKind::Call => {
                receiver.is_some_and(|r| r.reference.is_some()) && f.sig.asyncness.is_none()
            }
        };
        if !shape_ok {
            return Err(format!(
                "{at}: the signature does not fit `{}` (new/function: no receiver; getter: `&self`; call: `&self` or `&mut self`; await: `async fn(&self)`)",
                format!("{kind:?}").to_lowercase()
            ));
        }
        let mut params = Vec::new();
        let mut kw_only = false;
        for input in f.sig.inputs.iter() {
            let syn::FnArg::Typed(pt) = input else { continue };
            let syn::Pat::Ident(ident) = &*pt.pat else {
                return Err(format!("{at}: a parameter is a plain name"));
            };
            let pat = format!("{at}({})", ident.ident);
            let entries = voc::read(&pt.attrs).map_err(|e| format!("{pat}: {e}"))?;
            voc::only(&pat, &entries, &["kw", "default"])?;
            kw_only |= voc::find(&entries, "kw").is_some();
            let (by_ref, ty) = match &*pt.ty {
                syn::Type::Reference(r) if r.mutability.is_none() => {
                    (true, self.borrowed(&r.elem, &pat)?)
                }
                syn::Type::Reference(_) => {
                    return Err(format!("{pat}: a parameter is never `&mut`"));
                }
                other => (false, self.ty(other, &pat)?),
            };
            let default = match voc::find(&entries, "default").map(|e| &e.value) {
                None => None,
                Some(Some(Value::Literal(l))) => Some(l.clone()),
                Some(Some(Value::Text(t))) => Some(Literal::Str(t.clone())),
                Some(_) => return Err(format!("{pat}: `default` is a literal")),
            };
            check_default(&pat, &ty, default.as_ref())?;
            if params.last().is_some_and(|p: &Param| p.default.is_some() && !p.kw_only)
                && default.is_none()
                && !kw_only
            {
                return Err(format!(
                    "{pat}: a required positional parameter follows one with a default"
                ));
            }
            check_input(&pat, &ty)?;
            params.push(Param { name: ident.ident.to_string(), ty, by_ref, kw_only, default });
        }
        if kind == MethodKind::Getter && !params.is_empty() {
            return Err(format!("{at}: a getter takes no parameters"));
        }
        let ret = self.ret(&f.sig.output, &at)?;
        if kind == MethodKind::New && ret.ok != Ty::This {
            return Err(format!("{at}: a constructor returns `Self` or `Result<Self, E>`"));
        }
        check_output(&at, &ret.ok)?;
        Ok(Method {
            rust_name: f.sig.ident.to_string(),
            mutable: receiver.is_some_and(|r| r.mutability.is_some()),
            name: voc::text(&at, &entries, "name")?.unwrap_or_else(|| f.sig.ident.to_string()),
            kind,
            params,
            ret,
            doc: doc(&f.attrs),
        })
    }

    fn ret(&self, out: &syn::ReturnType, at: &str) -> Result<Ret, String> {
        let syn::ReturnType::Type(_, t) = out else {
            return Ok(Ret { ok: Ty::Unit, err: None });
        };
        let shape = self.shape(self.module, t, &[], at)?;
        if let Shape::Named { found: Found::External(path), args } = &shape
            && is(path, &["std", "result", "Result"])
        {
            let [ok, err] = &args[..] else {
                return Err(format!("{at}: `Result` takes two arguments"));
            };
            return Ok(Ret { ok: self.classify(ok, at)?, err: Some(self.error(err, at)?) });
        }
        Ok(Ret { ok: self.classify(&shape, at)?, err: None })
    }

    /// A `#[host(error(..))]` type: its module, item, outside path and `#[host]` entries.
    fn error_item(
        &self,
        shape: &Shape,
        at: &str,
    ) -> Result<(usize, usize, String, Vec<Entry>), String> {
        let Shape::Named { found: Found::Item { module, item }, args } = shape else {
            return Err(format!("{at}: the error type is not an ABI declaration"));
        };
        let k = self.krate;
        let attrs = item_attrs(&k.modules[*module].items[*item]);
        let entries = voc::read(attrs).map_err(|e| format!("{at}: {e}"))?;
        let rust = k.extern_path(*module, *item).map_err(|e| format!("{at}: {e}"))?;
        if !args.is_empty() {
            return Err(format!("{at}: a declared error is a concrete type"));
        }
        if voc::find(&entries, "error").is_none() {
            return Err(format!(
                "{at}: `{rust}` is not a declared error: #[host(error(..))]"
            ));
        }
        Ok((*module, *item, rust, entries))
    }

    fn error(&self, shape: &Shape, at: &str) -> Result<ErrorDecl, String> {
        let (module, _, rust, entries) = self.error_item(shape, at)?;
        let module = &module;
        let k = self.krate;
        let declared = voc::find(&entries, "error").expect("error_item checked it");
        let with = match (
            voc::rust(at, &declared.nested, "with")?,
            voc::text(at, &declared.nested, "kind")?,
        ) {
            // `with` is written in the error's module: resolve it there.
            (Some(syn::Type::Path(p)), None) => match k.resolve(*module, &p.path)? {
                Found::Item { module, item } => {
                    k.extern_path(module, item).map_err(|e| format!("{at}: {e}"))?
                }
                other => {
                    return Err(format!(
                        "{at}: `with` names {other:?}, not a function of the crate"
                    ));
                }
            },
            // A builtin exception carrying the error's message.
            (None, Some(kind)) if BUILTIN_ERRORS.contains(&kind.as_str()) => {
                format!(
                    "|e: {rust}| ::pyo3::exceptions::Py{kind}::new_err(::std::string::ToString::to_string(&e))"
                )
            }
            _ => {
                return Err(format!(
                    "{at}: `{rust}` declares its conversion as one of `with = path` or `kind = \"<{}>\"`",
                    BUILTIN_ERRORS.join("|")
                ));
            }
        };
        Ok(ErrorDecl { rust, with })
    }

    /// What `&T` lends: `&str` is a string, `&[T]` a list, anything else `T`.
    fn borrowed(&self, t: &syn::Type, at: &str) -> Result<Ty, String> {
        match t {
            syn::Type::Path(p) if p.qself.is_none() && p.path.is_ident("str") => Ok(Ty::Str),
            syn::Type::Slice(s) => Ok(Ty::List(Box::new(self.ty(&s.elem, at)?))),
            other => self.ty(other, at),
        }
    }

    fn ty(&self, t: &syn::Type, at: &str) -> Result<Ty, String> {
        let shape = self.shape(self.module, t, &[], at)?;
        self.classify(&shape, at)
    }

    /// `t` as written in `module`, with `env` the generic arguments of the alias being expanded.
    fn shape(
        &self,
        module: usize,
        t: &syn::Type,
        env: &[(String, &Shape)],
        at: &str,
    ) -> Result<Shape, String> {
        match t {
            syn::Type::Tuple(tuple) => Ok(Shape::Tuple(
                tuple
                    .elems
                    .iter()
                    .map(|e| self.shape(module, e, env, at))
                    .collect::<Result<_, _>>()?,
            )),
            syn::Type::Paren(p) => self.shape(module, &p.elem, env, at),
            syn::Type::Group(g) => self.shape(module, &g.elem, env, at),
            syn::Type::TraitObject(obj) => {
                let mut traits = obj.bounds.iter().filter_map(|b| match b {
                    syn::TypeParamBound::Trait(t) => Some(t),
                    _ => None,
                });
                let first =
                    traits.next().ok_or_else(|| format!("{at}: a trait object names a trait"))?;
                if traits.any(|t| !t.path.get_ident().is_some_and(|i| i == "Send" || i == "Sync")) {
                    return Err(format!("{at}: a trait object is one trait (plus Send/Sync)"));
                }
                let (found, args) = self.path(module, &first.path, env, at)?;
                Ok(Shape::Dyn { found, args })
            }
            syn::Type::Path(p) if p.qself.is_none() => {
                let segments: Vec<String> =
                    p.path.segments.iter().map(|s| s.ident.to_string()).collect();
                match segments.iter().map(String::as_str).collect::<Vec<_>>()[..] {
                    ["Self"] => return Ok(Shape::SelfType),
                    ["Self", assoc] => return Ok(Shape::SelfAssoc(assoc.to_owned())),
                    [one] => {
                        if let Some((_, shape)) = env.iter().find(|(n, _)| n == one) {
                            return Ok(clone_shape(shape));
                        }
                    }
                    _ => {}
                }
                let (found, args) = self.path(module, &p.path, env, at)?;
                // An alias is what it stands for, its parameters bound to these arguments.
                if let Found::Item { module: m, item } = &found
                    && let syn::Item::Type(alias) = &self.krate.modules[*m].items[*item]
                {
                    let params: Vec<String> =
                        alias.generics.type_params().map(|p| p.ident.to_string()).collect();
                    if params.len() != args.len() {
                        return Err(format!(
                            "{at}: `{}` takes {} type argument(s)",
                            alias.ident,
                            params.len()
                        ));
                    }
                    let bound: Vec<(String, &Shape)> =
                        params.into_iter().zip(args.iter()).collect();
                    return self.shape(*m, &alias.ty, &bound, at);
                }
                Ok(Shape::Named { found, args })
            }
            other => Err(format!("{at}: `{}` is not a declarable type", quote::quote!(#other))),
        }
    }

    fn path(
        &self,
        module: usize,
        path: &syn::Path,
        env: &[(String, &Shape)],
        at: &str,
    ) -> Result<(Found, Vec<Shape>), String> {
        let found = self.krate.resolve(module, path).map_err(|e| format!("{at}: {e}"))?;
        let mut args = Vec::new();
        if let Some(last) = path.segments.last()
            && let syn::PathArguments::AngleBracketed(a) = &last.arguments
        {
            for arg in &a.args {
                match arg {
                    syn::GenericArgument::Type(t) => args.push(self.shape(module, t, env, at)?),
                    syn::GenericArgument::Lifetime(_) => {}
                    other => {
                        return Err(format!(
                            "{at}: `{}` is not a declarable generic argument",
                            quote::quote!(#other)
                        ));
                    }
                }
            }
        }
        Ok((found, args))
    }

    fn classify(&self, shape: &Shape, at: &str) -> Result<Ty, String> {
        let one = |args: &[Shape]| -> Result<Ty, String> {
            match args {
                [a] => self.classify(a, at),
                _ => Err(format!("{at}: expected one type argument")),
            }
        };
        match shape {
            Shape::Tuple(items) if items.is_empty() => Ok(Ty::Unit),
            Shape::Tuple(items) => Ok(Ty::Tuple(items.iter().map(|i| self.classify(i, at)).collect::<Result<_, _>>()?)),
            Shape::SelfType => Ok(Ty::This),
            Shape::SelfAssoc(name) => self
                .assoc
                .iter()
                .find(|a| a.rust_name == *name)
                .map(|a| a.ty.clone())
                .ok_or_else(|| format!("{at}: `Self::{name}` is not a declared associated type (declare it before use)")),
            Shape::Dyn { .. } => Err(format!("{at}: a trait object crosses as `Arc<dyn Trait>` of a #[host(proxy)] trait")),
            Shape::Named { found: Found::External(path), args } => {
                let p: Vec<&str> = path.iter().map(String::as_str).collect();
                Ok(match (&p[..], &args[..]) {
                    (["bool"], []) => Ty::Bool,
                    (["f64"], []) => Ty::Float,
                    (["u8"], []) => Ty::Int(Int::U8),
                    (["u16"], []) => Ty::Int(Int::U16),
                    (["u32"], []) => Ty::Int(Int::U32),
                    (["u64"], []) => Ty::Int(Int::U64),
                    (["i32"], []) => Ty::Int(Int::I32),
                    (["i64"], []) => Ty::Int(Int::I64),
                    (["std", "string", "String"], []) => Ty::Str,
                    (["std", "option", "Option"], a) => Ty::Optional(Box::new(one(a)?)),
                    (["std", "vec", "Vec"], a) => Ty::List(Box::new(one(a)?)),
                    (["serde_json", "Value"], []) => Ty::Json,
                    (["serde_json", "Map"], [k, v]) if self.classify(k, at)? == Ty::Str && self.classify(v, at)? == Ty::Json => Ty::JsonObject,
                    (["std", "sync", "Arc"], [Shape::Dyn { found: Found::Item { module, item }, args }]) if args.is_empty() => {
                        self.proxy(*module, *item, at)?
                    }
                    _ => return Err(format!("{at}: `{}` has no declared conversion", p.join("::"))),
                })
            }
            Shape::Named { found: Found::Module(m), .. } => Err(format!("{at}: `{}` is a module, not a type", self.krate.display(*m))),
            Shape::Named { found: Found::Item { module, item }, args } => {
                if !args.is_empty() {
                    return Err(format!("{at}: generic declarations cross only through an alias that binds them"));
                }
                self.declared(*module, *item, at)
            }
        }
    }

    fn proxy(&self, module: usize, item: usize, at: &str) -> Result<Ty, String> {
        let k = self.krate;
        let syn::Item::Trait(tr) = &k.modules[module].items[item] else {
            return Err(format!("{at}: `Arc<dyn ..>` names a trait"));
        };
        let entries = voc::read(&tr.attrs).map_err(|e| format!("{at}: {e}"))?;
        let proxy = voc::find(&entries, "proxy")
            .ok_or_else(|| format!("{at}: `{}` is not a #[host(proxy)] trait", tr.ident))?;
        let host = voc::text(at, &proxy.nested, "host")?
            .ok_or_else(|| format!("{at}: proxy `{}` names no host class", tr.ident))?;
        let generic = tr.attrs.iter().any(|a| {
            a.path().is_ident("python")
                && a.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                )
                .is_ok_and(|metas| metas.iter().any(|m| m.path().is_ident("provider")))
        });
        let declared = declares_calls(tr).then(|| {
            (k.modules[module].path.first().cloned().unwrap_or_default(), tr.ident.to_string())
        });
        Ok(Ty::Proxy {
            rust: k.extern_path(module, item).map_err(|e| format!("{at}: {e}"))?,
            host,
            generic,
            declared,
        })
    }

    /// A declaration-model proxy: every method declares its call and its context parameter.
    fn proxy_trait(&self, tr: &syn::ItemTrait, proxy: &Entry) -> Result<Proxy, String> {
        let at = self.at;
        voc::only(at, &proxy.nested, &["host"])?;
        let host = voc::text(at, &proxy.nested, "host")?
            .ok_or_else(|| format!("{at}: proxy(host = \"module.Class\") names the class it calls"))?;
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
        let mut methods = Vec::new();
        for inner in &tr.items {
            match inner {
                syn::TraitItem::Fn(f) => methods.push(self.proxy_method(f)?),
                other => {
                    return Err(format!(
                        "{at}: a proxy declares methods only, not `{}`",
                        quote::quote!(#other)
                    ));
                }
            }
        }
        Ok(Proxy {
            rust_name: tr.ident.to_string(),
            trait_path: k.extern_path(self.module, item).map_err(|e| format!("{at}: {e}"))?,
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

    fn proxy_method(&self, f: &syn::TraitItemFn) -> Result<ProxyMethod, String> {
        let at = format!("{}::{}", self.at, f.sig.ident);
        let entries = voc::read(&f.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(&at, &entries, &["await", "notify", "name"])?;
        let kind = match (voc::find(&entries, "await"), voc::find(&entries, "notify")) {
            (Some(_), None) => ProxyCall::Await,
            (None, Some(_)) => ProxyCall::Notify,
            _ => return Err(format!("{at}: a proxy method declares exactly one of await, notify")),
        };
        let shared_receiver = f.sig.receiver().is_some_and(shared);
        let fits = match kind {
            ProxyCall::Await => shared_receiver && f.sig.asyncness.is_some(),
            ProxyCall::Notify => {
                shared_receiver
                    && f.sig.asyncness.is_none()
                    && matches!(f.sig.output, syn::ReturnType::Default)
            }
        };
        if !fits {
            return Err(format!(
                "{at}: the signature does not fit `{}` (await: `async fn(&self, ..) -> Result<T, E>`; notify: `fn(&self, ..)` returning nothing)",
                format!("{kind:?}").to_lowercase()
            ));
        }
        let mut params = Vec::new();
        let mut context = None;
        for input in f.sig.inputs.iter() {
            let syn::FnArg::Typed(pt) = input else { continue };
            let syn::Pat::Ident(ident) = &*pt.pat else {
                return Err(format!("{at}: a parameter is a plain name"));
            };
            let pat = format!("{at}({})", ident.ident);
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
                other => (false, self.ty(other, &pat)?),
            };
            // It crosses to the host.
            check_output(&pat, &ty)?;
            params.push(Param {
                name: ident.ident.to_string(),
                ty,
                by_ref,
                kw_only: false,
                default: None,
            });
        }
        let context = context.ok_or_else(|| {
            format!("{at}: a proxy call runs in the host context one #[host(context)] parameter carries")
        })?;
        let ret = match kind {
            ProxyCall::Notify => ProxyRet { ok: Ty::Unit, refusal: None, err: None },
            ProxyCall::Await => self.proxy_ret(&f.sig.output, &at)?,
        };
        Ok(ProxyMethod {
            rust_name: f.sig.ident.to_string(),
            name: voc::text(&at, &entries, "name")?.unwrap_or_else(|| f.sig.ident.to_string()),
            kind,
            params,
            context,
            ret,
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
        let refused = || format!("{at}: an awaited proxy call returns `Result<T, E>` (E a declared #[host(error)]): a host failure is reported, never swallowed");
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

    /// A crate type, by its declaration: a `#[derive(Host)]` record with its `#[binding]`
    /// directions, or a value class (`#[host(object(cell = "value"))]` on an impl of it).
    fn declared(&self, module: usize, item: usize, at: &str) -> Result<Ty, String> {
        let k = self.krate;
        let it = &k.modules[module].items[item];
        let rust = k.extern_path(module, item).map_err(|e| format!("{at}: {e}"))?;
        let ident = k.item_name(module, item);
        let attrs = item_attrs(it);
        let entries = voc::read(attrs).map_err(|e| format!("{at}: {e}"))?;
        if let Some(own) = voc::find(&entries, "handwritten") {
            voc::only(at, &own.nested, &["extract", "emit", "host"])?;
            let host = voc::text(at, &own.nested, "host")?.ok_or_else(|| {
                format!("{at}: handwritten(host = \"..\") is its host annotation")
            })?;
            let has = |w: &str| voc::find(&own.nested, w).is_some();
            return Ok(Ty::Record(Record {
                rust,
                host: HostForm::Annotation(host),
                extract: has("extract"),
                emit: has("emit"),
            }));
        }
        if derives_host(attrs) {
            let words = binding_words(attrs);
            let (name, module_name) = python_target(attrs);
            voc::only(at, &entries, &["host_type", "rename", "wire"])?;
            let external = voc::text(at, &entries, "host_type")?;
            let from_items = words.iter().any(|w| w == "from_items");
            let host = match (
                external,
                module_name.or_else(|| file_module(&k.modules[module].items)),
            ) {
                _ if from_items => HostForm::Dict,
                (Some(path), _) => {
                    let (module, name) = path
                        .rsplit_once('.')
                        .ok_or_else(|| format!("{at}: host_type is a dotted path"))?;
                    HostForm::External { module: module.to_owned(), name: name.to_owned() }
                }
                (None, Some(module)) => {
                    HostForm::Model { module, name: name.unwrap_or_else(|| ident.clone()) }
                }
                (None, None) => {
                    return Err(format!(
                        "{at}: record `{rust}` has no host form (a #[python] model, #[host(host_type = \"..\")] or #[binding(from_items)])"
                    ));
                }
            };
            return Ok(Ty::Record(Record {
                rust,
                host,
                extract: words.iter().any(|w| w == "extract"),
                emit: words.iter().any(|w| w == "emit"),
            }));
        }
        for i in &k.modules[module].items {
            let syn::Item::Impl(imp) = i else { continue };
            if imp.trait_.is_some()
                || !matches!(&*imp.self_ty, syn::Type::Path(p) if p.path.is_ident(&ident))
            {
                continue;
            }
            let entries = voc::read(&imp.attrs).map_err(|e| format!("{at}: {e}"))?;
            if let Some(object) = voc::find(&entries, "object")
                && voc::text(at, &object.nested, "cell")?.as_deref() == Some("value")
            {
                let pyclass = voc::find(&object.nested, "pyclass")
                    .map(|p| p.nested.iter().map(|e| e.key.clone()).collect::<Vec<_>>())
                    .unwrap_or_default();
                let input = voc::find(&object.nested, "input").is_some();
                let host = HostForm::CoreClass(
                    voc::text(at, &object.nested, "name")?.unwrap_or_else(|| ident.clone()),
                );
                return Ok(Ty::Record(Record {
                    rust,
                    host,
                    extract: pyclass.iter().any(|w| w == "from_py_object"),
                    emit: !input,
                }));
            }
        }
        Err(format!(
            "{at}: `{rust}` is not a declared host type (#[derive(Host)] record or value class)"
        ))
    }
}

/// The Python builtins an error may declare as its `kind`.
const BUILTIN_ERRORS: &[&str] = &["ValueError", "TypeError", "RuntimeError", "KeyError"];

/// Some method declares its call (`await`, `notify`): a declaration-model proxy.
fn declares_calls(tr: &syn::ItemTrait) -> bool {
    tr.items.iter().any(|i| {
        let syn::TraitItem::Fn(f) = i else { return false };
        voc::read(&f.attrs).is_ok_and(|entries| {
            voc::find(&entries, "await").is_some() || voc::find(&entries, "notify").is_some()
        })
    })
}

fn shared(r: &syn::Receiver) -> bool {
    r.reference.is_some() && r.mutability.is_none()
}

fn is(path: &[String], want: &[&str]) -> bool {
    path.iter().map(String::as_str).eq(want.iter().copied())
}

fn clone_shape(s: &Shape) -> Shape {
    match s {
        Shape::Named { found, args } => {
            Shape::Named { found: found.clone(), args: args.iter().map(clone_shape).collect() }
        }
        Shape::Tuple(items) => Shape::Tuple(items.iter().map(clone_shape).collect()),
        Shape::Dyn { found, args } => {
            Shape::Dyn { found: found.clone(), args: args.iter().map(clone_shape).collect() }
        }
        Shape::SelfType => Shape::SelfType,
        Shape::SelfAssoc(n) => Shape::SelfAssoc(n.clone()),
    }
}

/// A host may pass it: every record in it declares `extract`; classes and sources are outputs.
fn check_input(at: &str, ty: &Ty) -> Result<(), String> {
    match ty {
        Ty::Record(r) if !r.extract => Err(format!(
            "{at}: record `{}` is not declared for input (#[binding(extract)])",
            r.rust
        )),
        Ty::Optional(t) | Ty::List(t) => check_input(at, t),
        Ty::Tuple(ts) => ts.iter().try_for_each(|t| check_input(at, t)),
        Ty::This | Ty::Source { .. } | Ty::Unit => {
            Err(format!("{at}: `{}` is not an input", ty.rust()))
        }
        _ => Ok(()),
    }
}

/// The host may receive it: every record in it declares `emit`; proxies are inputs.
fn check_output(at: &str, ty: &Ty) -> Result<(), String> {
    match ty {
        Ty::Record(r) if !r.emit => {
            Err(format!("{at}: record `{}` is not declared for output (#[binding(emit)])", r.rust))
        }
        Ty::Optional(t) | Ty::List(t) => check_output(at, t),
        Ty::Tuple(ts) => ts.iter().try_for_each(|t| check_output(at, t)),
        Ty::Proxy { .. } => Err(format!("{at}: a host proxy is not returned to the host")),
        Ty::Source { item, .. } => check_output(at, item),
        _ => Ok(()),
    }
}

fn check_default(at: &str, ty: &Ty, default: Option<&Literal>) -> Result<(), String> {
    let Some(d) = default else { return Ok(()) };
    let fits = match (ty, d) {
        (Ty::Optional(_), Literal::None) => true,
        (Ty::Optional(inner), d) => return check_default(at, inner, Some(d)),
        (Ty::Bool, Literal::Bool(_))
        | (Ty::Float, Literal::Float(_) | Literal::Int(_))
        | (Ty::Str, Literal::Str(_)) => true,
        (Ty::Int(i), Literal::Int(v)) => !i.rust().starts_with('u') || *v >= 0,
        _ => false,
    };
    if fits {
        Ok(())
    } else {
        Err(format!("{at}: default {d:?} is not a value of `{}`", ty.rust()))
    }
}

fn item_attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Struct(i) => &i.attrs,
        syn::Item::Enum(i) => &i.attrs,
        syn::Item::Trait(i) => &i.attrs,
        syn::Item::Type(i) => &i.attrs,
        syn::Item::Fn(i) => &i.attrs,
        _ => &[],
    }
}

fn derives_host(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().filter(|a| a.path().is_ident("derive")).any(|a| {
        a.parse_args_with(
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        )
        .is_ok_and(|paths| {
            paths.iter().any(|p| p.segments.last().is_some_and(|s| s.ident == "Host"))
        })
    })
}

fn binding_words(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut words = Vec::new();
    for a in attrs.iter().filter(|a| a.path().is_ident("binding")) {
        let _ = a.parse_nested_meta(|m| {
            if let Some(i) = m.path.get_ident() {
                words.push(i.to_string());
            }
            if m.input.peek(syn::Token![=]) {
                m.value()?.parse::<proc_macro2::TokenTree>()?;
            } else if m.input.peek(syn::token::Paren) {
                m.input.parse::<proc_macro2::Group>()?;
            }
            Ok(())
        });
    }
    words
}

/// A record's `#[python(name = "..", module = "..")]`: its host class and generated module.
fn python_target(attrs: &[syn::Attribute]) -> (Option<String>, Option<String>) {
    let (mut name, mut module) = (None, None);
    for a in attrs.iter().filter(|a| a.path().is_ident("python")) {
        let _ = a.parse_nested_meta(|m| {
            if m.path.is_ident("name") {
                name = Some(m.value()?.parse::<syn::LitStr>()?.value());
            } else if m.path.is_ident("module") {
                module = Some(m.value()?.parse::<syn::LitStr>()?.value());
            } else if m.input.peek(syn::Token![=]) {
                m.value()?.parse::<proc_macro2::TokenTree>()?;
            } else if m.input.peek(syn::token::Paren) {
                m.input.parse::<proc_macro2::Group>()?;
            }
            Ok(())
        });
    }
    (name, module)
}

/// The `#[host(file(module = ".."))]` default of the records in a file.
fn file_module(items: &[syn::Item]) -> Option<String> {
    items.iter().find_map(|i| {
        let syn::Item::Const(c) = i else { return None };
        let entries = voc::read(&c.attrs).ok()?;
        let file = voc::find(&entries, "file")?;
        voc::text("", &file.nested, "module").ok().flatten()
    })
}

/// `///` lines, one leading space removed, trailing blanks dropped.
pub fn doc(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut lines: Vec<String> = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            syn::Meta::NameValue(syn::MetaNameValue {
                value: syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }),
                ..
            }) => {
                let v = s.value();
                Some(v.strip_prefix(' ').map(str::to_owned).unwrap_or(v))
            }
            _ => None,
        })
        .collect();
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    lines
}
