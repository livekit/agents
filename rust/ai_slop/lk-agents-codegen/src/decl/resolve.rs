//! ABI declarations into the IR. Every type a declaration names is found through the crate's
//! modules and `use`s ([`Crate::resolve`]), aliases are expanded with their generic arguments,
//! and the result is classified by what its declaration says. Anything else fails, naming the
//! declaration.

use super::tree::{Crate, Found};
use super::vocabulary::{self as voc, Entry, Value};
use super::*;

mod class;
mod native;
mod proxy;
mod record;
pub(crate) use proxy::boxed_future;
use proxy::eager_future;
use record::{check_default, record_entry, record_target, unit_enum};

/// Every `#[host(class)]` of the crate, in module order.
pub fn declarations(krate: &Crate) -> Result<Declarations, String> {
    let mut out = Declarations::default();
    for (m, module) in krate.modules.iter().enumerate() {
        for item in &module.items {
            if let syn::Item::Struct(_) | syn::Item::Enum(_) = item
                && let Some(record) = record_entry(item_attrs(item))
            {
                let at = format!("{}: {}", module.file, item_ident(item));
                let resolver = Resolver { krate, module: m, at: &at, assoc: Vec::new() };
                out.records.push(resolver.record(item, &record)?);
                continue;
            }
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
    crate::records::check_none_defaults(&out)?;
    proxy::check_handed_back(&out)?;
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
#[derive(Clone)]
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
    fn assoc_type(&self, t: &syn::TraitItemType) -> Result<(Assoc, Option<Sink>), String> {
        let at = format!("{}::{}", self.at, t.ident);
        let entries = voc::read(&t.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(&at, &entries, &["source", "handle", "class", "sink", "env"])?;
        if let Some(env) = voc::find(&entries, "env") {
            if entries.len() > 1 || !env.nested.is_empty() || env.value.is_some() {
                return Err(format!("{at}: #[host(env)] alone declares the process environment"));
            }
            return Ok((Assoc { rust_name: t.ident.to_string(), ty: Ty::Env }, None));
        }
        if let Some(sink) = voc::find(&entries, "sink") {
            if entries.len() > 1 {
                return Err(format!("{at}: an associated type declares one kind"));
            }
            let sink = self.sink(t, sink, &at)?;
            let ty = Ty::Sink { item: Box::new(sink.item.clone()), class: sink.class.clone() };
            return Ok((Assoc { rust_name: t.ident.to_string(), ty }, Some(sink)));
        }
        let kinds = (
            voc::find(&entries, "source"),
            voc::find(&entries, "handle"),
            voc::find(&entries, "class"),
        );
        let ty = match kinds {
            (None, None, Some(class)) => {
                voc::only(&at, &class.nested, &["name"])?;
                let class = voc::text(&at, &class.nested, "name")?.ok_or_else(|| {
                    format!(
                        "{at}: class(name = \"X\") names the declared class it is an instance of"
                    )
                })?;
                let family = self
                    .krate
                    .modules
                    .iter()
                    .find_map(|module| {
                        module.items.iter().find_map(|item| {
                            let syn::Item::Trait(tr) = item else { return None };
                            let entries = voc::read(&tr.attrs).ok()?;
                            let entry = voc::find(&entries, "class")?;
                            let name = voc::text(&at, &entry.nested, "name")
                                .ok()?
                                .unwrap_or_else(|| tr.ident.to_string());
                            (name == class).then(|| module.path.first().cloned()).flatten()
                        })
                    })
                    .ok_or_else(|| format!("{at}: `{class}` is not a declared class"))?;
                Ty::Instance { assoc: t.ident.to_string(), class, family }
            }
            (Some(source), None, None) => {
                voc::only(&at, &source.nested, &["item", "class"])?;
                let item = voc::rust(&at, &source.nested, "item")?
                    .ok_or_else(|| format!("{at}: source(item = T) is what it yields"))?;
                let class = voc::text(&at, &source.nested, "class")?
                    .ok_or_else(|| format!("{at}: source(class = \"X\") is its stream class"))?;
                Ty::Source { item: Box::new(self.ty(&item, &at)?), class }
            }
            (None, Some(handle), None) => {
                voc::only(&at, &handle.nested, &["class"])?;
                let class = voc::text(&at, &handle.nested, "class")?.ok_or_else(|| {
                    format!("{at}: handle(class = \"X\") is the shell's class `crate::handles::X`")
                })?;
                Ty::Handle { assoc: t.ident.to_string(), class }
            }
            _ => {
                return Err(format!(
                    "{at}: an associated type declares one kind: #[host(source(item = T, class = \"X\"))], #[host(sink(item = T, class = \"X\"))], #[host(handle(class = \"X\"))] or #[host(class(name = \"X\"))]"
                ));
            }
        };
        Ok((Assoc { rust_name: t.ident.to_string(), ty }, None))
    }

    /// `sink(item = I, class = "X")` on `type S: Trait`: `Trait` is `StreamSink<I>`, or an ABI
    /// trait over it whose methods are `call`s of the stream class.
    fn sink(&self, t: &syn::TraitItemType, entry: &Entry, at: &str) -> Result<Sink, String> {
        voc::only(at, &entry.nested, &["item", "class"])?;
        let item = voc::rust(at, &entry.nested, "item")?
            .ok_or_else(|| format!("{at}: sink(item = T) is what it takes"))?;
        let item = self.ty(&item, at)?;
        check_input(at, &item)?;
        let class = voc::text(at, &entry.nested, "class")?
            .ok_or_else(|| format!("{at}: sink(class = \"X\") is its stream class"))?;
        let stream_sink =
            self.krate.resolve(0, &syn::parse_quote!(crate::utils::stream::StreamSink)).ok();
        let bound = t.bounds.iter().find_map(|b| match b {
            syn::TypeParamBound::Trait(tr) => Some(&tr.path),
            _ => None,
        });
        let bound = bound.ok_or_else(|| format!("{at}: a sink is bound by its input trait"))?;
        let found = self.krate.resolve(self.module, bound).map_err(|e| format!("{at}: {e}"))?;
        let err_shape =
            self.shape(self.module, &syn::parse_quote!(crate::errors::ProviderError), &[], at)?;
        let err = self.error(&err_shape, at)?;
        if stream_sink.as_ref() == Some(&found) {
            let trait_path = format!("::lk_agents_abi::StreamSink<{}>", item.rust());
            return Ok(Sink { class, item, trait_path, err, methods: Vec::new() });
        }
        let Found::Item { module, item: index } = found else {
            return Err(format!(
                "{at}: a sink is bound by `StreamSink<T>` or an ABI trait over it"
            ));
        };
        let syn::Item::Trait(tr) = &self.krate.modules[module].items[index] else {
            return Err(format!(
                "{at}: a sink is bound by `StreamSink<T>` or an ABI trait over it"
            ));
        };
        let declared = voc::read(&tr.attrs).map_err(|e| format!("{at}: {e}"))?;
        if voc::find(&declared, "sink").is_none() {
            return Err(format!("{at}: `{}` is declared #[host(sink)]", tr.ident));
        }
        let over_sink = tr.supertraits.iter().any(|b| {
            matches!(b, syn::TypeParamBound::Trait(s)
                if self.krate.resolve(module, &s.path).ok() == stream_sink)
        });
        if !over_sink {
            return Err(format!("{at}: `{}` extends `StreamSink<T>`", tr.ident));
        }
        let inner = format!("{}: {}", self.krate.modules[module].file, tr.ident);
        let resolver = Resolver { krate: self.krate, module, at: &inner, assoc: Vec::new() };
        let mut methods = Vec::new();
        for f in &tr.items {
            let syn::TraitItem::Fn(f) = f else {
                return Err(format!("{inner}: a sink trait declares only methods"));
            };
            let m = resolver.method(f)?;
            if m.kind != MethodKind::Call || m.mutable {
                return Err(format!(
                    "{inner}::{}: a sink's methods are `call`s on `&self`",
                    m.rust_name
                ));
            }
            methods.push(m);
        }
        let trait_path = self.krate.extern_path(module, index).map_err(|e| format!("{at}: {e}"))?;
        Ok(Sink { class, item, trait_path, err, methods })
    }

    fn method(&self, f: &syn::TraitItemFn) -> Result<Method, String> {
        let at = format!("{}::{}", self.at, f.sig.ident);
        let entries = voc::read(&f.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(
            &at,
            &entries,
            &[
                "new", "call", "await", "getter", "setter", "function", "next", "name", "pid",
                "detach", "future",
            ],
        )?;
        let kinds: Vec<MethodKind> = [
            ("new", MethodKind::New),
            ("call", MethodKind::Call),
            ("await", MethodKind::Await),
            ("getter", MethodKind::Getter),
            ("setter", MethodKind::Setter),
            ("function", MethodKind::Function),
            ("next", MethodKind::Next),
        ]
        .into_iter()
        .filter(|(k, _)| voc::find(&entries, k).is_some())
        .map(|(_, kind)| kind)
        .collect();
        let [mut kind] = kinds[..] else {
            return Err(format!(
                "{at}: declare exactly one of new, call, await, getter, setter, function, next"
            ));
        };
        if kind == MethodKind::Function && f.sig.asyncness.is_some() {
            kind = MethodKind::AsyncFunction;
        }
        if kind == MethodKind::Call && f.sig.receiver().is_none() {
            kind = MethodKind::StaticCall;
        }
        let detach = match voc::find(&entries, "detach") {
            Some(flag) if flag.value.is_some() || !flag.nested.is_empty() => {
                return Err(format!("{at}: detach is a bare flag"));
            }
            value => value.is_some(),
        };
        if detach
            && !matches!(
                kind,
                MethodKind::New | MethodKind::Call | MethodKind::StaticCall | MethodKind::Function
            )
        {
            return Err(format!(
                "{at}: detach requires a synchronous constructor, call or function"
            ));
        }
        let receiver = f.sig.receiver();
        // `fn ..(&self, ..) -> BoxFuture<'static, T>`, maybe in a `Result<_, E>`: the call runs (or
        // refuses) at once, in host call order, and the host awaits the future it returns.
        let eager_future = eager_future(&f.sig.output)
            .filter(|_| kind == MethodKind::Await && f.sig.asyncness.is_none());
        let eager = eager_future.is_some();
        let shape_ok = match kind {
            MethodKind::AsyncFunction => receiver.is_none() && f.sig.asyncness.is_some(),
            MethodKind::New | MethodKind::Function | MethodKind::StaticCall => {
                receiver.is_none() && f.sig.asyncness.is_none()
            }
            MethodKind::Await => {
                receiver.is_some_and(shared) && (f.sig.asyncness.is_some() || eager)
            }
            MethodKind::Next => receiver.is_some_and(shared) && f.sig.asyncness.is_some(),
            MethodKind::Getter | MethodKind::Setter => {
                receiver.is_some_and(shared) && f.sig.asyncness.is_none()
            }
            MethodKind::Call => {
                receiver.is_some_and(|r| r.reference.is_some()) && f.sig.asyncness.is_none()
            }
        };
        if !shape_ok {
            return Err(format!(
                "{at}: the signature does not fit `{}` (new/function: no receiver; getter: `&self`; call: no receiver, `&self` or `&mut self`; await: `async fn(&self)` or `fn(&self) -> BoxFuture<'static, T>`)",
                format!("{kind:?}").to_lowercase()
            ));
        }
        let mut params = Vec::new();
        let mut kw_only = false;
        for input in f.sig.inputs.iter() {
            let syn::FnArg::Typed(pt) = input else {
                continue;
            };
            let syn::Pat::Ident(ident) = &*pt.pat else {
                return Err(format!("{at}: a parameter is a plain name"));
            };
            let pat = format!("{at}({})", ident.ident);
            let entries = voc::read(&pt.attrs).map_err(|e| format!("{pat}: {e}"))?;
            voc::only(
                &pat,
                &entries,
                &["kw", "default", "context", "shape_error", "one_or_many", "pcm"],
            )?;
            kw_only |= voc::find(&entries, "kw").is_some();
            if voc::find(&entries, "context").is_some() {
                let by_ref = matches!(&*pt.ty, syn::Type::Reference(_));
                if !call_context(&pt.ty) || params.iter().any(|p: &Param| p.ty == Ty::CallContext) {
                    return Err(format!(
                        "{pat}: a method's one #[host(context)] parameter is `CallContext`, captured at the call's entry"
                    ));
                }
                params.push(Param {
                    name: ident.ident.to_string(),
                    ty: Ty::CallContext,
                    shape_error: None,
                    one_or_many: false,
                    pcm: false,
                    by_ref,
                    kw_only: false,
                    default: None,
                });
                continue;
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
            let default = match voc::find(&entries, "default").map(|e| &e.value) {
                None => None,
                Some(Some(Value::Literal(l))) => Some(l.clone()),
                Some(Some(Value::Text(t))) => Some(Literal::Str(t.clone())),
                Some(_) => return Err(format!("{pat}: `default` is a literal")),
            };
            check_default(&pat, &ty, default.as_ref())?;
            let shape_error = crate::shape_error::read(&pat, &entries, &ty)?;
            let one_or_many = voc::find(&entries, "one_or_many").is_some();
            if one_or_many
                && (shape_error.is_some()
                    || !matches!(&ty, Ty::List(item) if **item != Ty::Int(crate::decl::Int::U8)))
            {
                return Err(format!("{pat}: one_or_many applies to a list of items"));
            }
            let pcm = voc::find(&entries, "pcm").is_some();
            if pcm && ty != Ty::List(Box::new(Ty::Int(crate::decl::Int::U8))) {
                return Err(format!("{pat}: pcm applies to bytes"));
            }
            if params.iter().rfind(|p: &&Param| !p.ty.hidden()).is_some_and(|p| {
                (p.default.is_some() || matches!(p.ty, Ty::Maybe(_))) && !p.kw_only
            }) && default.is_none()
                && !matches!(ty, Ty::Maybe(_))
                && !kw_only
            {
                return Err(format!(
                    "{pat}: a required positional parameter follows one with a default"
                ));
            }
            if ty == Ty::Env {
                if !by_ref || default.is_some() || kw_only {
                    return Err(format!(
                        "{pat}: the process environment is borrowed: `&Self::Env`"
                    ));
                }
            } else if ty == Ty::This {
                if !by_ref || !matches!(kind, MethodKind::Call | MethodKind::StaticCall) || detach {
                    return Err(format!(
                        "{pat}: an own-class input is borrowed as &Self by a synchronous class call"
                    ));
                }
            } else {
                check_input(&pat, &ty)?;
            }
            params.push(Param {
                name: ident.ident.to_string(),
                ty,
                by_ref,
                kw_only,
                default,
                shape_error,
                one_or_many,
                pcm,
            });
        }
        if kind == MethodKind::Getter && !params.is_empty() {
            return Err(format!("{at}: a getter takes no parameters"));
        }
        let (ret, refused) = match eager_future {
            Some((out, refused)) => {
                (self.ret(&syn::parse_quote!(-> #out), &at)?, self.ret(&refused, &at)?.err)
            }
            None => (self.ret(&f.sig.output, &at)?, None),
        };
        if kind == MethodKind::Setter
            && (params.iter().filter(|parameter| !parameter.ty.hidden()).count() != 1
                || ret.ok != Ty::Unit)
        {
            return Err(format!("{at}: a setter takes one value and returns unit"));
        }
        if kind == MethodKind::Next
            && (!f.sig.inputs.iter().all(|i| matches!(i, syn::FnArg::Receiver(_)))
                || ret.err.is_none()
                || !matches!(ret.ok, Ty::Optional(_)))
        {
            return Err(format!(
                "{at}: next is `async fn next(&self) -> Result<Option<T>, E>`: `None` ends the iteration"
            ));
        }
        if kind == MethodKind::New && ret.ok != Ty::This {
            return Err(format!("{at}: a constructor returns `Self` or `Result<Self, E>`"));
        }
        check_output(&at, &ret.ok)?;
        let future = match voc::find(&entries, "future") {
            None => None,
            Some(entry) => {
                if kind != MethodKind::Call
                    || ret.ok != self.ty(&syn::parse_quote!(crate::HostRef), &at)?
                {
                    return Err(format!(
                        "{at}: future requires a synchronous call returning HostRef"
                    ));
                }
                let Some(Value::Rust(ty)) = &entry.value else {
                    return Err(format!("{at}: future is its Rust completion type"));
                };
                let ty = self.ty(ty, &at)?;
                check_output(&at, &ty)?;
                Some(ty)
            }
        };
        Ok(Method {
            rust_name: f.sig.ident.to_string(),
            mutable: receiver.is_some_and(|r| r.mutability.is_some()),
            name: voc::text(&at, &entries, "name")?.unwrap_or_else(|| f.sig.ident.to_string()),
            kind,
            eager,
            refused,
            detach,
            pid: voc::boolean(&at, &entries, "pid")?,
            params,
            ret,
            future,
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
            return Err(format!("{at}: `{rust}` is not a declared error: #[host(error(..))]"));
        }
        Ok((*module, *item, rust, entries))
    }

    /// A declared error raises through the one conversion its declaration expands to
    /// (`HostException`: `with`, `kind`, per-variant kinds, delegation, `original`).
    fn error(&self, shape: &Shape, at: &str) -> Result<ErrorDecl, String> {
        let (_, _, rust, entries) = self.error_item(shape, at)?;
        let declared = voc::find(&entries, "error").expect("error_item checked it");
        if !declared.nested.is_empty() && declared.nested.iter().all(|e| e.key == "raised") {
            return Err(format!(
                "{at}: `{rust}` is a refusal the host raises (error(raised)), not an error raised to it"
            ));
        }
        let with = format!(
            "|e: {rust}| <{rust} as ::lk_agents_abi::python::HostException>::to_py_err(&e)"
        );
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
                            return Ok((*shape).clone());
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
            Shape::Tuple(items) if let Some((input, output)) = self.duplex(items) => Ok(Ty::HostDuplex {
                input: Box::new(self.classify(input, at)?),
                output: Box::new(self.classify(output, at)?),
            }),
            Shape::Tuple(items) => {
                let items: Vec<Ty> = items.iter().map(|i| self.classify(i, at)).collect::<Result<_, _>>()?;
                match &items[..] {
                    [Ty::Sink { item: input, class }, Ty::Source { item: output, class: other }] if class == other => {
                        Ok(Ty::Duplex { input: input.clone(), output: output.clone(), class: class.clone() })
                    }
                    [Ty::Sink { class, .. }, ..] | [_, Ty::Sink { class, .. }] => Err(format!(
                        "{at}: the sink of `{class}` is returned first beside its source: `(Self::Sink, Self::Source)`"
                    )),
                    _ => Ok(Ty::Tuple(items)),
                }
            }
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
                    (["std" | "core", "time", "Duration"], []) => Ty::Duration,
                    (["std", "time", "SystemTime"], []) => Ty::Timestamp,
                    (["isize"], []) => Ty::Int(Int::Isize),
                    (["u8"], []) => Ty::Int(Int::U8),
                    (["u16"], []) => Ty::Int(Int::U16),
                    (["u32"], []) => Ty::Int(Int::U32),
                    (["u64"], []) => Ty::Int(Int::U64),
                    (["u128"], []) => Ty::Int(Int::U128),
                    (["usize"], []) => Ty::Int(Int::Usize),
                    (["i16"], []) => Ty::Int(Int::I16),
                    (["i32"], []) => Ty::Int(Int::I32),
                    (["i64"], []) => Ty::Int(Int::I64),
                    (["std", "string", "String"], []) => Ty::Str,
                    (["std", "option", "Option"], a) => Ty::Optional(Box::new(one(a)?)),
                    (["std", "vec", "Vec"], a) => Ty::List(Box::new(one(a)?)),
                    (["std", "collections", "BTreeSet"], a) => Ty::Set(Box::new(one(a)?)),
                    (["std", "collections", "BTreeMap"], [k, v]) => {
                        Ty::Map(Box::new(self.classify(k, at)?), Box::new(self.classify(v, at)?))
                    }
                    (["indexmap", "IndexMap"], [k, v]) => {
                        Ty::OrderedMap(Box::new(self.classify(k, at)?), Box::new(self.classify(v, at)?))
                    }
                    (["serde_json", "Value"], []) => Ty::Json,
                    (["serde_json", "Map"], [k, v]) if self.classify(k, at)? == Ty::Str && self.classify(v, at)? == Ty::Json => Ty::JsonObject,
                    (["std", "sync", "Arc"], [Shape::Dyn { found: Found::Item { module, item }, args }]) if args.is_empty() => {
                        self.proxy(*module, *item, false, at)?
                    }
                    (["std", "sync", "Arc"], [Shape::Named { found: Found::Item { module, item }, args }])
                        if args.is_empty() && self.arc_object(*module, *item, at)?.is_some() =>
                    {
                        let (class, family) = self.arc_object(*module, *item, at)?.expect("checked");
                        Ty::Shared { rust: self.krate.extern_path(*module, *item).map_err(|e| format!("{at}: {e}"))?, class, family }
                    }
                    (["std", "boxed", "Box"], [inner @ Shape::Named { .. }]) => {
                        Ty::Boxed(Box::new(self.classify(inner, at)?))
                    }
                    (["std", "boxed", "Box"], [Shape::Dyn { found: Found::Item { module, item }, args }])
                        if args.is_empty() && !self.is_stream_source(*module, *item) =>
                    {
                        self.proxy(*module, *item, true, at)?
                    }
                    (["std", "boxed", "Box"], [Shape::Dyn { found: Found::Item { module, item }, args }])
                        if self.is_stream_source(*module, *item) =>
                    {
                        Ty::HostSource(Box::new(one(args)?))
                    }
                    _ => return Err(format!("{at}: `{}` has no declared conversion", p.join("::"))),
                })
            }
            Shape::Named { found: Found::Module(m), .. } => Err(format!("{at}: `{}` is a module, not a type", self.krate.display(*m))),
            Shape::Named { found: Found::Item { module, item }, args } => {
                if self.krate.resolve(0, &syn::parse_quote!(crate::utils::maybe::Maybe))
                    .is_ok_and(|found| found == Found::Item { module: *module, item: *item }) {
                    return Ok(Ty::Maybe(Box::new(one(args)?)));
                }
                if !args.is_empty() {
                    return Err(format!("{at}: generic declarations cross only through an alias that binds them"));
                }
                self.declared(*module, *item, at)
            }
        }
    }

    /// `(class, family)` of the declared class an ABI type implements in its own module, if any.
    fn arc_object(
        &self,
        module: usize,
        item: usize,
        at: &str,
    ) -> Result<Option<(String, String)>, String> {
        let k = self.krate;
        let ident = k.item_name(module, item);
        for i in &k.modules[module].items {
            let syn::Item::Impl(imp) = i else { continue };
            let Some((_, path, _)) = &imp.trait_ else { continue };
            if !matches!(&*imp.self_ty, syn::Type::Path(p) if p.path.is_ident(&ident)) {
                continue;
            }
            let Ok(Found::Item { module: m, item: t }) = k.resolve(module, path) else { continue };
            let syn::Item::Trait(tr) = &k.modules[m].items[t] else { continue };
            let entries = voc::read(&tr.attrs).map_err(|e| format!("{at}: {e}"))?;
            if let Some(class) = voc::find(&entries, "class") {
                let name = voc::text(at, &class.nested, "name")?.unwrap_or(tr.ident.to_string());
                let family = k.modules[m].path.first().cloned().unwrap_or_default();
                return Ok(Some((name, family)));
            }
        }
        Ok(None)
    }

    /// The item is the ABI's `StreamSource` trait.
    fn is_stream_source(&self, module: usize, item: usize) -> bool {
        let path: syn::Path = syn::parse_quote!(crate::utils::stream::StreamSource);
        self.krate.resolve(0, &path).is_ok_and(|f| f == Found::Item { module, item })
    }

    /// `(Box<dyn StreamSink<I>>, Box<dyn StreamSource<T>>)`: `I` and `T`.
    fn duplex<'s>(&self, items: &'s [Shape]) -> Option<(&'s Shape, &'s Shape)> {
        let boxed = |s: &'s Shape, path: syn::Path| -> Option<&'s Shape> {
            let Shape::Named { found: Found::External(b), args } = s else {
                return None;
            };
            let [Shape::Dyn { found, args }] = &args[..] else {
                return None;
            };
            let wanted = self.krate.resolve(0, &path).ok()?;
            (is(b, &["std", "boxed", "Box"]) && *found == wanted && args.len() == 1)
                .then(|| &args[0])
        };
        match items {
            [sink, source] => Some((
                boxed(sink, syn::parse_quote!(crate::utils::stream::StreamSink))?,
                boxed(source, syn::parse_quote!(crate::utils::stream::StreamSource))?,
            )),
            _ => None,
        }
    }

    pub(super) fn proxy(
        &self,
        module: usize,
        item: usize,
        boxed: bool,
        at: &str,
    ) -> Result<Ty, String> {
        let k = self.krate;
        let syn::Item::Trait(tr) = &k.modules[module].items[item] else {
            return Err(format!("{at}: `Arc<dyn ..>` names a trait"));
        };
        let entries = voc::read(&tr.attrs).map_err(|e| format!("{at}: {e}"))?;
        if voc::find(&entries, "proxy").is_none()
            && let Some(native) = voc::find(&entries, "native")
        {
            return self.native(module, item, native, boxed, at);
        }
        let proxy = voc::find(&entries, "proxy")
            .ok_or_else(|| format!("{at}: `{}` is not a #[host(proxy)] trait", tr.ident))?;
        let host = voc::text(at, &proxy.nested, "host")?
            .ok_or_else(|| format!("{at}: proxy `{}` names no host class", tr.ident))?;
        // A provider base with `events` is an `EventEmitter` generic over the plugin's own events.
        type Metas = syn::punctuated::Punctuated<syn::Meta, syn::Token![,]>;
        let generic = tr.attrs.iter().any(|a| {
            a.path().is_ident("python")
                && a.parse_args_with(Metas::parse_terminated).is_ok_and(|metas| {
                    metas.iter().any(|m| {
                        let syn::Meta::List(provider) = m else { return false };
                        provider.path.is_ident("provider")
                            && provider
                                .parse_args_with(Metas::parse_terminated)
                                .is_ok_and(|args| args.iter().any(|a| a.path().is_ident("events")))
                    })
                })
        });
        let declared = declares_calls(tr).then(|| {
            (k.modules[module].path.first().cloned().unwrap_or_default(), tr.ident.to_string())
        });
        if boxed && declared.is_none() {
            return Err(format!(
                "{at}: `Box<dyn {}>` is owned behind a declared proxy: declare its methods' calls",
                tr.ident
            ));
        }
        Ok(Ty::Proxy {
            rust: k.extern_path(module, item).map_err(|e| format!("{at}: {e}"))?,
            host,
            generic,
            declared,
            boxed,
        })
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
        if let Some(record) = voc::find(&entries, "record") {
            let (module_name, name, extract, emit) = record_target(at, record, &ident)?;
            return Ok(Ty::Record(Record {
                rust,
                host: HostForm::Record { module: module_name, name },
                extract,
                emit,
                literal: unit_enum(it),
            }));
        }
        if voc::find(&entries, "error").is_some() {
            let shape = Shape::Named { found: Found::Item { module, item }, args: Vec::new() };
            return Ok(Ty::Error(self.error(&shape, at)?));
        }
        if voc::find(&entries, "string").is_some() || voc::find(&entries, "wire_enum").is_some() {
            return Ok(Ty::Record(Record {
                rust,
                host: HostForm::Annotation("str".into()),
                extract: true,
                emit: true,
                literal: false,
            }));
        }
        // Beside a derive (for another binding's record), `host` says Python crosses it by its own
        // conversions all the same; `py`: they are `lk-agents-py`'s `conversions` free functions.
        if let Some(own) = voc::find(&entries, "handwritten")
            && (!derives_host(attrs) || voc::find(&own.nested, "host").is_some())
        {
            voc::only(at, &own.nested, &["extract", "emit", "host", "py"])?;
            let host = voc::text(at, &own.nested, "host")?.ok_or_else(|| {
                format!("{at}: handwritten(host = \"..\") is its host annotation")
            })?;
            let has = |w: &str| voc::find(&own.nested, w).is_some();
            return Ok(Ty::Record(Record {
                rust,
                host: if has("py") {
                    HostForm::Converted(host)
                } else {
                    HostForm::Annotation(host)
                },
                extract: has("extract"),
                emit: has("emit"),
                literal: false,
            }));
        }
        if derives_host(attrs) && class_struct(it) {
            return Ok(Ty::Record(Record {
                rust,
                host: HostForm::CoreClass(ident.clone()),
                extract: class_from_py_object(it),
                emit: true,
                literal: false,
            }));
        }
        if derives_host(attrs) {
            let words = binding_words(attrs);
            let (name, module_name, typed_dict) = python_target(attrs);
            voc::only(at, &entries, &["host_type", "rename", "wire", "handwritten"])?;
            // `handwritten(extract)` beside the derive: the read is the type's own conversion.
            let own_extract = match voc::find(&entries, "handwritten") {
                Some(own) => {
                    voc::only(at, &own.nested, &["extract"])?;
                    voc::find(&own.nested, "extract").is_some()
                }
                None => false,
            };
            let external = voc::text(at, &entries, "host_type")?;
            let from_items = words.iter().any(|w| w == "from_items");
            let host = match (
                external,
                module_name.or_else(|| file_default(&k.modules[module].items, "module")),
            ) {
                (Some(path), _) => {
                    let (module, name) = path
                        .rsplit_once('.')
                        .ok_or_else(|| format!("{at}: host_type is a dotted path"))?;
                    HostForm::External { module: module.to_owned(), name: name.to_owned() }
                }
                // `from_items` reads a dict: only a `TypedDict` model is that dict's type.
                (None, Some(_)) if from_items && !typed_dict => HostForm::Dict,
                (None, Some(module)) => {
                    HostForm::Model { module, name: name.unwrap_or_else(|| ident.clone()) }
                }
                (None, None) if from_items => HostForm::Dict,
                (None, None) => {
                    return Err(format!(
                        "{at}: record `{rust}` has no host form (a #[python] model, #[host(host_type = \"..\")] or #[binding(from_items)])"
                    ));
                }
            };
            // A unit-only enum's derive writes its string both ways.
            let unit = unit_enum(it);
            return Ok(Ty::Record(Record {
                rust,
                host,
                extract: unit || own_extract || words.iter().any(|w| w == "extract"),
                emit: unit || words.iter().any(|w| w == "emit"),
                literal: unit,
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
            // A `#[class]` struct's object: the host holds a `_core` handle over a clone, and hands
            // one back when the class is `#[class(from_py_object)]`.
            if let Some(object) = voc::find(&entries, "object")
                && object.nested.is_empty()
                && class_struct(it)
            {
                return Ok(Ty::Record(Record {
                    rust,
                    host: HostForm::CoreClass(ident.clone()),
                    extract: class_from_py_object(it),
                    emit: true,
                    literal: false,
                }));
            }
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
                    literal: false,
                }));
            }
        }
        // Explicit host conversions above retain their input/output contract. `wire_strs`
        // alone supplies only the pure output protocol; it declares no Python input reader.
        if voc::find(&entries, "wire_strs").is_some() {
            return Ok(Ty::Record(Record {
                rust,
                host: HostForm::WireStrs,
                extract: false,
                emit: true,
                literal: false,
            }));
        }
        Err(format!(
            "{at}: `{rust}` is not a declared host type (#[derive(Host)] record or value class)"
        ))
    }
}

fn item_ident(item: &syn::Item) -> String {
    match item {
        syn::Item::Struct(s) => s.ident.to_string(),
        syn::Item::Enum(e) => e.ident.to_string(),
        _ => String::new(),
    }
}

/// `InputRequired` -> `input_required`.
pub(crate) fn snake(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Some method declares its call (`await`, `notify`): a declaration-model proxy.
fn declares_calls(tr: &syn::ItemTrait) -> bool {
    tr.items.iter().any(|i| {
        let syn::TraitItem::Fn(f) = i else {
            return false;
        };
        voc::read(&f.attrs)
            .is_ok_and(|entries| PROXY_CALLS.iter().any(|k| voc::find(&entries, k).is_some()))
    })
}

/// The method keys of a declared proxy (`lk-agents-host-derive` reads the same list).
const PROXY_CALLS: &[&str] = &["await", "call", "notify", "read_once", "bind"];

/// A struct carrying `#[class]` (`#[crate::class]`, `#[lk_agents_abi::class]`).
fn class_struct(item: &syn::Item) -> bool {
    matches!(item, syn::Item::Struct(st) if st.attrs.iter().any(|a| a.path().segments.last().is_some_and(|s| s.ident == "class")))
}

/// `#[class(from_py_object)]`: pyo3 takes the class by value (a clone).
fn class_from_py_object(item: &syn::Item) -> bool {
    let syn::Item::Struct(st) = item else {
        return false;
    };
    st.attrs.iter().any(|a| {
        a.path().segments.last().is_some_and(|s| s.ident == "class")
            && a.parse_args_with(
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )
            .is_ok_and(|ps| ps.iter().any(|p| p.is_ident("from_py_object")))
    })
}

fn shared(r: &syn::Receiver) -> bool {
    r.reference.is_some() && r.mutability.is_none()
}

fn is(path: &[String], want: &[&str]) -> bool {
    path.iter().map(String::as_str).eq(want.iter().copied())
}

/// A host may pass it: every record in it declares `extract`; classes and sources are outputs.
fn check_input(at: &str, ty: &Ty) -> Result<(), String> {
    match ty {
        Ty::Record(r) if !r.extract => Err(format!(
            "{at}: record `{}` is not declared for input (#[binding(extract)])",
            r.rust
        )),
        Ty::Optional(t) | Ty::Maybe(t) | Ty::List(t) | Ty::Set(t) | Ty::Boxed(t) => {
            check_input(at, t)
        }
        Ty::Map(k, v) | Ty::OrderedMap(k, v) => {
            check_input(at, k)?;
            check_input(at, v)
        }
        Ty::Tuple(ts) => ts.iter().try_for_each(|t| check_input(at, t)),
        Ty::This
        | Ty::Source { .. }
        | Ty::Sink { .. }
        | Ty::Duplex { .. }
        | Ty::Handle { .. }
        | Ty::Unit => Err(format!("{at}: `{}` is not an input", ty.rust())),
        Ty::HostSource(item) => check_input(at, item),
        Ty::HostDuplex { input, output } => {
            check_output(at, input)?;
            check_input(at, output)
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
        Ty::Optional(t) | Ty::List(t) | Ty::Set(t) | Ty::Boxed(t) => check_output(at, t),
        Ty::Map(k, v) | Ty::OrderedMap(k, v) => {
            check_output(at, k)?;
            check_output(at, v)
        }
        Ty::Tuple(ts) => ts.iter().try_for_each(|t| check_output(at, t)),
        Ty::Maybe(_) => Err(format!("{at}: omission is an input state, not an output value")),
        // Handed back as the host object it was taken from (`check_handed_back`); a native
        // implementation as the declared class it is.
        Ty::Proxy { declared: Some(_), boxed: false, .. } | Ty::Native { .. } => Ok(()),
        Ty::Proxy { .. } => Err(format!("{at}: a host proxy is not returned to the host")),
        Ty::HostSource(_) | Ty::HostDuplex { .. } => {
            Err(format!("{at}: a host stream is not handed back to the host"))
        }
        Ty::Source { item, .. } => check_output(at, item),
        Ty::Sink { class, .. } => {
            Err(format!("{at}: the sink of `{class}` is returned beside its source"))
        }
        Ty::Duplex { input, output, .. } => {
            check_input(at, input)?;
            check_output(at, output)
        }
        _ => Ok(()),
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

pub(crate) fn derives_host(attrs: &[syn::Attribute]) -> bool {
    derives(attrs, "Host")
}

fn derives(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().filter(|a| a.path().is_ident("derive")).any(|a| {
        a.parse_args_with(
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        )
        .is_ok_and(|paths| paths.iter().any(|p| p.segments.last().is_some_and(|s| s.ident == name)))
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

/// A record's `#[python(name = "..", module = "..", typed_dict)]`: its host class, generated
/// module, and whether that model is a `TypedDict` (a plain dict at runtime).
fn python_target(attrs: &[syn::Attribute]) -> (Option<String>, Option<String>, bool) {
    let (mut name, mut module, mut typed_dict) = (None, None, false);
    for a in attrs.iter().filter(|a| a.path().is_ident("python")) {
        let _ = a.parse_nested_meta(|m| {
            if m.path.is_ident("name") {
                name = Some(m.value()?.parse::<syn::LitStr>()?.value());
            } else if m.path.is_ident("module") {
                module = Some(m.value()?.parse::<syn::LitStr>()?.value());
            } else if m.path.is_ident("typed_dict") {
                typed_dict = true;
            } else if m.input.peek(syn::Token![=]) {
                m.value()?.parse::<proc_macro2::TokenTree>()?;
            } else if m.input.peek(syn::token::Paren) {
                m.input.parse::<proc_macro2::Group>()?;
            }
            Ok(())
        });
    }
    (name, module, typed_dict)
}

/// `key` of a file's `#[host(file(module = "..", path = ".."))]`: its records' defaults.
fn file_default(items: &[syn::Item], key: &str) -> Option<String> {
    items.iter().find_map(|i| {
        let syn::Item::Const(c) = i else { return None };
        let entries = voc::read(&c.attrs).ok()?;
        let file = voc::find(&entries, "file")?;
        voc::text("", &file.nested, key).ok().flatten()
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

/// `CallContext` or `&CallContext` (`lk_agents_abi::invoke::CallContext`): what a call runs in.
pub(super) fn call_context(ty: &syn::Type) -> bool {
    let ty = match ty {
        syn::Type::Reference(r) => &*r.elem,
        t => t,
    };
    matches!(ty, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "CallContext"))
}
