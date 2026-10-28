//! ABI declarations into the IR. Every type a declaration names is found through the crate's
//! modules and `use`s ([`Crate::resolve`]), aliases are expanded with their generic arguments,
//! and the result is classified by what its declaration says. Anything else fails, naming the
//! declaration.

use super::tree::{Crate, Found};
use super::vocabulary::{self as voc, Entry, Value};
use super::*;

mod class;
mod proxy;

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
    fn assoc_type(&self, t: &syn::TraitItemType) -> Result<Assoc, String> {
        let at = format!("{}::{}", self.at, t.ident);
        let entries = voc::read(&t.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(&at, &entries, &["source", "handle", "class"])?;
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
                Ty::Instance { assoc: t.ident.to_string(), class }
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
                    "{at}: an associated type declares one kind: #[host(source(item = T, class = \"X\"))], #[host(handle(class = \"X\"))] or #[host(class(name = \"X\"))]"
                ));
            }
        };
        Ok(Assoc { rust_name: t.ident.to_string(), ty })
    }

    fn method(&self, f: &syn::TraitItemFn) -> Result<Method, String> {
        let at = format!("{}::{}", self.at, f.sig.ident);
        let entries = voc::read(&f.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(&at, &entries, &["new", "call", "await", "getter", "function", "next", "name"])?;
        let kinds: Vec<MethodKind> = [
            ("new", MethodKind::New),
            ("call", MethodKind::Call),
            ("await", MethodKind::Await),
            ("getter", MethodKind::Getter),
            ("function", MethodKind::Function),
            ("next", MethodKind::Next),
        ]
        .into_iter()
        .filter(|(k, _)| voc::find(&entries, k).is_some())
        .map(|(_, kind)| kind)
        .collect();
        let [kind] = kinds[..] else {
            return Err(format!(
                "{at}: declare exactly one of new, call, await, getter, function, next"
            ));
        };
        let receiver = f.sig.receiver();
        let shape_ok = match kind {
            MethodKind::New | MethodKind::Function => {
                receiver.is_none() && f.sig.asyncness.is_none()
            }
            MethodKind::Await | MethodKind::Next => {
                receiver.is_some_and(shared) && f.sig.asyncness.is_some()
            }
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
            return Err(format!("{at}: `{rust}` is not a declared error: #[host(error(..))]"));
        }
        Ok((*module, *item, rust, entries))
    }

    /// A declared error raises through the one conversion its declaration expands to
    /// (`HostException`: `with`, `kind`, per-variant kinds, delegation, `original`).
    fn error(&self, shape: &Shape, at: &str) -> Result<ErrorDecl, String> {
        let (_, _, rust, entries) = self.error_item(shape, at)?;
        let declared = voc::find(&entries, "error").expect("error_item checked it");
        if declared.nested.iter().all(|e| e.key == "raised") {
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
            Shape::Tuple(items) if let Some((input, output)) = self.duplex(items) => Ok(Ty::HostDuplex {
                input: Box::new(self.classify(input, at)?),
                output: Box::new(self.classify(output, at)?),
            }),
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
                    (["usize"], []) => Ty::Int(Int::Usize),
                    (["i32"], []) => Ty::Int(Int::I32),
                    (["i64"], []) => Ty::Int(Int::I64),
                    (["std", "string", "String"], []) => Ty::Str,
                    (["std", "option", "Option"], a) => Ty::Optional(Box::new(one(a)?)),
                    (["std", "vec", "Vec"], a) => Ty::List(Box::new(one(a)?)),
                    (["std", "collections", "BTreeSet"], a) => Ty::Set(Box::new(one(a)?)),
                    (["std", "collections", "BTreeMap"], [k, v]) => {
                        Ty::Map(Box::new(self.classify(k, at)?), Box::new(self.classify(v, at)?))
                    }
                    (["serde_json", "Value"], []) => Ty::Json,
                    (["serde_json", "Map"], [k, v]) if self.classify(k, at)? == Ty::Str && self.classify(v, at)? == Ty::Json => Ty::JsonObject,
                    (["std", "sync", "Arc"], [Shape::Dyn { found: Found::Item { module, item }, args }]) if args.is_empty() => {
                        self.proxy(*module, *item, at)?
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
                if !args.is_empty() {
                    return Err(format!("{at}: generic declarations cross only through an alias that binds them"));
                }
                self.declared(*module, *item, at)
            }
        }
    }

    /// The item is the ABI's `StreamSource` trait.
    fn is_stream_source(&self, module: usize, item: usize) -> bool {
        let path: syn::Path = syn::parse_quote!(crate::utils::stream::StreamSource);
        self.krate.resolve(0, &path).is_ok_and(|f| f == Found::Item { module, item })
    }

    /// `(Box<dyn StreamSink<I>>, Box<dyn StreamSource<T>>)`: `I` and `T`.
    fn duplex<'s>(&self, items: &'s [Shape]) -> Option<(&'s Shape, &'s Shape)> {
        let boxed = |s: &'s Shape, path: syn::Path| -> Option<&'s Shape> {
            let Shape::Named { found: Found::External(b), args } = s else { return None };
            let [Shape::Dyn { found, args }] = &args[..] else { return None };
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
            }));
        }
        if voc::find(&entries, "error").is_some() {
            let shape = Shape::Named { found: Found::Item { module, item }, args: Vec::new() };
            return Ok(Ty::Error(self.error(&shape, at)?));
        }
        if voc::find(&entries, "string").is_some() {
            return Ok(Ty::Record(Record {
                rust,
                host: HostForm::Annotation("str".into()),
                extract: true,
                emit: true,
            }));
        }
        if let Some(own) = voc::find(&entries, "handwritten")
            && !derives_host(attrs)
        {
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
            // A unit-only enum's derive writes its string both ways.
            let unit = matches!(it, syn::Item::Enum(e)
                if e.variants.iter().all(|v| matches!(v.fields, syn::Fields::Unit)));
            return Ok(Ty::Record(Record {
                rust,
                host,
                extract: unit || own_extract || words.iter().any(|w| w == "extract"),
                emit: unit || words.iter().any(|w| w == "emit"),
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
            // A `#[class]` struct's object: the host holds a `_core` handle over a clone.
            if let Some(object) = voc::find(&entries, "object")
                && object.nested.is_empty()
                && class_struct(it)
            {
                return Ok(Ty::Record(Record {
                    rust,
                    host: HostForm::CoreClass(ident.clone()),
                    extract: false,
                    emit: true,
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
                }));
            }
        }
        Err(format!(
            "{at}: `{rust}` is not a declared host type (#[derive(Host)] record or value class)"
        ))
    }
}

impl Resolver<'_> {
    /// A `#[host(record)]` struct or enum, its fields resolved in its own module.
    fn record(&self, item: &syn::Item, record: &Entry) -> Result<RecordDecl, String> {
        let at = self.at;
        let k = self.krate;
        let module = &k.modules[self.module];
        let index =
            module.items.iter().position(|i| std::ptr::eq(i, item)).expect("read from here");
        let ident = item_ident(item);
        let (py_module, name, extract, emit) = record_target(at, record, &ident)?;
        let persisted = voc::find(&record.nested, "persisted").is_some();
        if persisted && !(extract && emit && matches!(item, syn::Item::Struct(_))) {
            return Err(format!(
                "{at}: a persisted record is a struct crossing both ways (extract, emit): the application saves what it was given and restores it"
            ));
        }
        let entries = voc::read(item_attrs(item)).map_err(|e| format!("{at}: {e}"))?;
        voc::only(at, &entries, &["record", "wire_strs"])?;
        if derives_host(item_attrs(item)) {
            return Err(format!("{at}: a #[host(record)] is not also #[derive(Host)]"));
        }
        let body = match item {
            syn::Item::Struct(st) => {
                if !st.generics.params.is_empty() {
                    return Err(format!("{at}: a record is a concrete type"));
                }
                let syn::Fields::Named(named) = &st.fields else {
                    return Err(format!("{at}: a record struct has named fields"));
                };
                RecordBody::Struct(self.fields(named.named.iter(), &name, (extract, emit))?)
            }
            syn::Item::Enum(en) => {
                if en.variants.iter().all(|v| matches!(v.fields, syn::Fields::Unit)) {
                    let mut values = Vec::new();
                    for v in &en.variants {
                        let vat = format!("{at}::{}", v.ident);
                        let entries = voc::read(&v.attrs).map_err(|e| format!("{vat}: {e}"))?;
                        voc::only(&vat, &entries, &["rename"])?;
                        let value = voc::text(&vat, &entries, "rename")?
                            .unwrap_or_else(|| snake(&v.ident.to_string()));
                        values.push((v.ident.to_string(), value));
                    }
                    RecordBody::Literals(values)
                } else {
                    let mut variants = Vec::new();
                    for v in &en.variants {
                        variants.push(self.variant(v, &name, (extract, emit))?);
                    }
                    RecordBody::Variants(variants)
                }
            }
            _ => unreachable!("records are structs and enums"),
        };
        Ok(RecordDecl {
            rust: k.extern_path(self.module, index).map_err(|e| format!("{at}: {e}"))?,
            name,
            module: py_module,
            family: module.path.first().cloned().ok_or_else(|| {
                format!("{at}: a record is declared in a subject module, not the crate root")
            })?,
            source: module.file.clone(),
            doc: doc(item_attrs(item)),
            extract,
            emit,
            persisted,
            body,
        })
    }

    fn variant(
        &self,
        v: &syn::Variant,
        owner: &str,
        dirs: (bool, bool),
    ) -> Result<Variant, String> {
        let at = format!("{}::{}", self.at, v.ident);
        let entries = voc::read(&v.attrs).map_err(|e| format!("{at}: {e}"))?;
        let shape = match &v.fields {
            syn::Fields::Unit => {
                voc::only(&at, &entries, &["name"])?;
                VariantShape::Unit
            }
            syn::Fields::Named(named) => {
                voc::only(&at, &entries, &["name"])?;
                VariantShape::Named(self.fields(named.named.iter(), owner, dirs)?)
            }
            syn::Fields::Unnamed(unnamed) => {
                voc::only(&at, &entries, &["name", "field", "skip", "default"])?;
                let [only] = &unnamed.unnamed.iter().collect::<Vec<_>>()[..] else {
                    return Err(format!("{at}: a tuple variant carries one value"));
                };
                if voc::find(&entries, "skip").is_some() {
                    if dirs.0 {
                        return Err(format!(
                            "{at}: an extracted record reads every value: no `skip`"
                        ));
                    }
                    VariantShape::Tuple(None)
                } else {
                    let field = voc::text(&at, &entries, "field")?.ok_or_else(|| {
                        format!("{at}: a tuple variant names its value: #[host(field = \"..\")]")
                    })?;
                    let ty = self.ty(&only.ty, &at)?;
                    let default = literal_default(&at, &entries)?;
                    check_default(&at, &ty, default.as_ref())?;
                    check_record_field(&at, &ty, dirs)?;
                    VariantShape::Tuple(Some(Field {
                        rust_name: field,
                        ty,
                        default,
                        skip: false,
                        doc: Vec::new(),
                    }))
                }
            }
        };
        let rust_name = v.ident.to_string();
        Ok(Variant {
            name: voc::text(&at, &entries, "name")?
                .unwrap_or_else(|| format!("{owner}{rust_name}")),
            rust_name,
            doc: doc(&v.attrs),
            shape,
        })
    }

    fn fields<'f>(
        &self,
        fields: impl Iterator<Item = &'f syn::Field>,
        owner: &str,
        dirs: (bool, bool),
    ) -> Result<Vec<Field>, String> {
        let mut out = Vec::new();
        for f in fields {
            let ident = f.ident.as_ref().expect("named").to_string();
            let at = format!("{}({owner}.{ident})", self.at);
            let entries = voc::read(&f.attrs).map_err(|e| format!("{at}: {e}"))?;
            voc::only(&at, &entries, &["default", "skip"])?;
            let skip = voc::find(&entries, "skip").is_some();
            if skip && dirs.0 {
                return Err(format!("{at}: an extracted record reads every field: no `skip`"));
            }
            let ty = if skip { Ty::Unit } else { self.ty(&f.ty, &at)? };
            let default = literal_default(&at, &entries)?;
            check_default(&at, &ty, default.as_ref())?;
            if !skip {
                check_record_field(&at, &ty, dirs)?;
            }
            out.push(Field { rust_name: ident, ty, default, skip, doc: doc(&f.attrs) });
        }
        Ok(out)
    }
}

/// `record(module = "..", name = "..", extract, emit)`: its module, host name and directions.
fn record_target(
    at: &str,
    record: &Entry,
    ident: &str,
) -> Result<(String, String, bool, bool), String> {
    voc::only(at, &record.nested, &["module", "name", "extract", "emit", "persisted"])?;
    let module = voc::text(at, &record.nested, "module")?
        .ok_or_else(|| format!("{at}: record(module = \"..\") is its generated Python module"))?;
    let name = voc::text(at, &record.nested, "name")?.unwrap_or_else(|| ident.to_owned());
    let extract = voc::find(&record.nested, "extract").is_some();
    let emit = voc::find(&record.nested, "emit").is_some();
    if !extract && !emit {
        return Err(format!("{at}: a record crosses some way: extract, emit or both"));
    }
    Ok((module, name, extract, emit))
}

fn record_entry(attrs: &[syn::Attribute]) -> Option<Entry> {
    voc::read(attrs).ok().and_then(|entries| voc::find(&entries, "record").cloned())
}

fn literal_default(at: &str, entries: &[Entry]) -> Result<Option<Literal>, String> {
    match voc::find(entries, "default").map(|e| &e.value) {
        None => Ok(None),
        Some(Some(Value::Literal(l))) => Ok(Some(l.clone())),
        Some(Some(Value::Text(t))) => Ok(Some(Literal::Str(t.clone()))),
        Some(_) => Err(format!("{at}: `default` is a literal")),
    }
}

/// A record field crosses each declared direction: its records declare it, and it is a value
/// (no class, proxy or source).
fn check_record_field(at: &str, ty: &Ty, (extract, emit): (bool, bool)) -> Result<(), String> {
    match ty {
        // A proxy field is taken from the host (`extract` only: `check_output` refuses it).
        Ty::This
        | Ty::Source { .. }
        | Ty::Handle { .. }
        | Ty::Instance { .. }
        | Ty::HostSource(_)
        | Ty::HostDuplex { .. }
        | Ty::Error(_)
        | Ty::Unit
        | Ty::Tuple(_) => {
            return Err(format!("{at}: `{}` is not a record field", ty.rust()));
        }
        _ => {}
    }
    if extract {
        check_input(at, ty)?;
    }
    if emit {
        check_output(at, ty)?;
    }
    Ok(())
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
        let syn::TraitItem::Fn(f) = i else { return false };
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
        Ty::Optional(t) | Ty::List(t) | Ty::Set(t) => check_input(at, t),
        Ty::Map(k, v) => {
            check_input(at, k)?;
            check_input(at, v)
        }
        Ty::Tuple(ts) => ts.iter().try_for_each(|t| check_input(at, t)),
        Ty::This
        | Ty::Source { .. }
        | Ty::Handle { .. }
        | Ty::Instance { .. }
        | Ty::Unit
        | Ty::Error(_) => Err(format!("{at}: `{}` is not an input", ty.rust())),
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
        Ty::Optional(t) | Ty::List(t) | Ty::Set(t) => check_output(at, t),
        Ty::Map(k, v) => {
            check_output(at, k)?;
            check_output(at, v)
        }
        Ty::Tuple(ts) => ts.iter().try_for_each(|t| check_output(at, t)),
        Ty::Proxy { .. } => Err(format!("{at}: a host proxy is not returned to the host")),
        Ty::HostSource(_) | Ty::HostDuplex { .. } => {
            Err(format!("{at}: a host stream is not handed back to the host"))
        }
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
