//! `#[host(facade(..))]` traits: the public Python class over a native class.

use super::*;

/// The keys a facade method reads itself; the rest are the class-method keys.
const METHOD_KEYS: &[&str] = &[
    "new",
    "getter",
    "via",
    "native",
    "none_ends_iteration",
    "wrap",
    "exit",
    "future",
    "discard",
    "not_given",
    "origin",
    "returns",
    "annotation",
    "resolve",
];

impl Resolver<'_> {
    pub(super) fn facade_trait(
        &mut self,
        tr: &syn::ItemTrait,
        entry: &Entry,
    ) -> Result<FacadeClass, String> {
        let at = self.at.to_owned();
        let at = at.as_str();
        let nested = &entry.nested;
        voc::only(
            at,
            nested,
            &[
                "path",
                "extended",
                "name",
                "base",
                "native",
                "legs",
                "leg",
                "availability",
                "base_args",
                "keep",
                "stream",
            ],
        )?;
        let native = voc::rust(at, nested, "native")?
            .ok_or_else(|| format!("{at}: a facade names its native class: `native = Trait`"))?;
        let native = match self.shape(self.module, &native, &[], at)? {
            Shape::Named { found: Found::Item { module, item }, .. } => {
                let syn::Item::Trait(native) = &self.krate.modules[module].items[item] else {
                    return Err(format!("{at}: `native` names a #[host(class)] trait"));
                };
                let entries = voc::read(&native.attrs).map_err(|e| format!("{at}: {e}"))?;
                let class = voc::find(&entries, "class")
                    .ok_or_else(|| format!("{at}: `native` names a #[host(class)] trait"))?;
                voc::text(at, &class.nested, "name")?.unwrap_or_else(|| native.ident.to_string())
            }
            _ => return Err(format!("{at}: `native` names a #[host(class)] trait")),
        };
        let flag = |key| voc::find(nested, key).is_some();
        let stream = voc::find(nested, "stream")
            .map(|stream| {
                let keys = &stream.nested;
                voc::only(
                    at,
                    keys,
                    &[
                        "op",
                        "owner",
                        "native_retries",
                        "resolve",
                        "method",
                        "base_args",
                        "state",
                        "default",
                        "annotation",
                        "not_given",
                    ],
                )?;
                let names = |key| {
                    voc::find(keys, key)
                        .map_or(Vec::new(), |l| l.nested.iter().map(|n| n.key.clone()).collect())
                };
                let op = voc::find(keys, "op")
                    .ok_or_else(|| format!("{at}: a stream names the native `op` it opens"))?;
                Ok::<_, String>(StreamForm {
                    op: rust_ident(at, op, "op")?,
                    owner: voc::text(at, keys, "owner")?
                        .ok_or_else(|| format!("{at}: a stream names its `owner` attribute"))?,
                    native_retries: voc::find(keys, "native_retries").is_some(),
                    resolve: per_parameter(at, keys, "resolve")?,
                    method: voc::text(at, keys, "method")?,
                    base_args: names("base_args"),
                    state: names("state"),
                    defaults: per_parameter(at, keys, "default")?,
                    annotations: per_parameter(at, keys, "annotation")?,
                    not_given: voc::find(keys, "not_given").is_some(),
                })
            })
            .transpose()?;
        let mut delegates = Vec::new();
        for item in &tr.items {
            let syn::TraitItem::Type(t) = item else { continue };
            let entries = voc::read(&t.attrs).map_err(|e| format!("{at}: {e}"))?;
            voc::only(at, &entries, &["delegate"])?;
            if voc::find(&entries, "delegate").is_none() {
                return Err(format!("{at}::{}: a facade's associated type is a delegate", t.ident));
            }
            let Some(syn::TypeParamBound::Trait(bound)) = t.bounds.first() else {
                return Err(format!("{at}::{}: a delegate is bounded by its trait", t.ident));
            };
            let bound = &bound.path;
            let ty = match self.krate.resolve(self.module, bound) {
                Ok(Found::Item { module, item }) if self.is_proxy(module, item) => {
                    self.ty(&syn::parse_quote!(std::sync::Arc<dyn #bound>), at)?
                }
                Ok(Found::Item { module, item }) => {
                    let class = self.class_name(module, item, at)?.ok_or_else(|| {
                        format!("{at}::{}: a delegate is a proxy or a class", t.ident)
                    })?;
                    let family =
                        self.krate.modules[module].path.first().cloned().unwrap_or_default();
                    Ty::Instance { assoc: t.ident.to_string(), class, family, shared: false }
                }
                _ => return Err(format!("{at}::{}: a delegate is a proxy or a class", t.ident)),
            };
            let attribute = format!("_{}", snake(&t.ident.to_string()));
            self.assoc.push(Assoc { rust_name: t.ident.to_string(), ty: ty.clone() });
            delegates.push(Delegate { assoc: t.ident.to_string(), attribute, ty });
        }
        let mut methods = Vec::new();
        for item in &tr.items {
            let syn::TraitItem::Fn(f) = item else { continue };
            methods.push(self.facade_method(f, &delegates)?);
        }
        let new = methods.iter().any(|m| m.method.kind == MethodKind::New);
        if stream.is_some() && new {
            return Err(format!("{at}: a stream is built by its base, not a #[host(new)]"));
        }
        let names = |key| {
            voc::find(nested, key)
                .map_or(Vec::new(), |l| l.nested.iter().map(|n| n.key.clone()).collect::<Vec<_>>())
        };
        let (base_args, keep) = (names("base_args"), names("keep"));
        let constructor = methods.iter().find(|m| m.method.kind == MethodKind::New);
        for n in base_args.iter().chain(&keep) {
            if !base_args.is_empty() && constructor.is_none() {
                return Err(format!(
                    "{at}: base_args read the native object a #[host(new)] builds"
                ));
            }
            if keep.contains(n)
                && !constructor.is_some_and(|c| c.method.params.iter().any(|p| &p.name == n))
            {
                return Err(format!("{at}: keep({n}) names no constructor parameter"));
            }
        }
        let name = voc::text(at, nested, "name")?.unwrap_or_else(|| tr.ident.to_string());
        let path = voc::text(at, nested, "path")?
            .ok_or_else(|| format!("{at}: a facade names its public module: `path = \"..\"`"))?;
        let module = crate::exports::module(&path)
            .ok_or_else(|| format!("{at}: `{path}` lies under livekit.agents"))?
            .replacen("public", "facades", 1);
        Ok(FacadeClass {
            name,
            path,
            extended: flag("extended"),
            module,
            base: voc::text(at, nested, "base")?,
            native,
            legs: voc::find(nested, "legs")
                .map_or(Vec::new(), |legs| legs.nested.iter().map(|l| l.key.clone()).collect()),
            leg: voc::text(at, nested, "leg")?,
            availability: voc::find(nested, "availability")
                .map(|a| {
                    let keys = ["observer", "members", "kind", "event", "record"];
                    voc::only(at, &a.nested, &keys)?;
                    let field = |name| {
                        voc::text(at, &a.nested, name)?
                            .ok_or_else(|| format!("{at}: facade availability requires {name}"))
                    };
                    let members = field("members")?;
                    if !constructor
                        .is_some_and(|c| c.method.params.iter().any(|p| p.name == members))
                    {
                        return Err(format!(
                            "{at}: availability members names no constructor parameter"
                        ));
                    }
                    Ok::<_, String>(Availability {
                        observer: field("observer")?,
                        members,
                        kind: field("kind")?,
                        event: field("event")?,
                        record: field("record")?,
                    })
                })
                .transpose()?,
            base_args,
            keep,
            stream,
            delegates,
            source: self.krate.modules[self.module].file.clone(),
            doc: doc(&tr.attrs),
            methods,
        })
    }

    fn facade_method(
        &self,
        f: &syn::TraitItemFn,
        delegates: &[Delegate],
    ) -> Result<FacadeMethod, String> {
        let at = format!("{}::{}", self.at, f.sig.ident);
        let entries = voc::read(&f.attrs).map_err(|e| format!("{at}: {e}"))?;
        voc::only(&at, &entries, METHOD_KEYS)?;
        let flag = |key| voc::find(&entries, key).is_some();
        let kind = if flag("new") {
            MethodKind::New
        } else if flag("getter") {
            MethodKind::Getter
        } else if f.sig.asyncness.is_some() {
            MethodKind::Await
        } else {
            MethodKind::Call
        };
        let mut method = self.method_of(f, kind, &[], &at)?;
        let new_keys = ["via", "native", "none_ends_iteration", "wrap", "exit"];
        if kind == MethodKind::New && new_keys.iter().any(|k| flag(k)) {
            return Err(format!("{at}: a constructor builds the native object"));
        }
        let via = voc::rust(&at, &entries, "via")?
            .map(|via| {
                let name = quote::quote!(#via).to_string();
                delegates
                    .iter()
                    .find(|d| d.assoc == name)
                    .cloned()
                    .ok_or_else(|| format!("{at}: `via = {name}` names no delegate"))
            })
            .transpose()?;
        let none_ends_iteration = flag("none_ends_iteration");
        if none_ends_iteration && kind != MethodKind::Await {
            return Err(format!("{at}: none_ends_iteration is an awaited `__anext__`"));
        }
        let native = match voc::find(&entries, "native") {
            Some(entry) => rust_ident(&at, entry, "native")?,
            None if none_ends_iteration => "next".to_owned(),
            None => method.name.clone(),
        };
        let wrap = voc::rust(&at, &entries, "wrap")?
            .map(|wrap| {
                let syn::Type::Path(path) = &wrap else {
                    return Err(format!("{at}: `wrap = Record::field`"));
                };
                let mut segments = path.path.segments.clone();
                let field = segments.pop().map(|s| s.into_value().ident.to_string());
                let field = field.ok_or_else(|| format!("{at}: `wrap = Record::field`"))?;
                let record: syn::Type = syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: syn::Path { leading_colon: None, segments },
                });
                match self.ty(&record, &at)? {
                    Ty::Record(Record { host: HostForm::Record { module, name }, .. }) => {
                        Ok((module, name, field))
                    }
                    _ => Err(format!("{at}: `wrap` names a #[host(record)] struct")),
                }
            })
            .transpose()?;
        let exit = voc::find(&entries, "exit").map(|e| rust_ident(&at, e, "exit")).transpose()?;
        if exit.is_some() && via.is_some() {
            return Err(format!("{at}: an `async with` enters and exits the native object"));
        }
        let pairs = |key| per_parameter(&at, &entries, key);
        let adapt = Adapt {
            future: flag("future"),
            discard: flag("discard"),
            not_given: flag("not_given"),
            origin: voc::text(&at, &entries, "origin")?,
            returns: voc::text(&at, &entries, "returns")?,
            annotations: pairs("annotation")?,
            resolve: pairs("resolve")?,
        };
        for (name, _) in adapt.annotations.iter().chain(&adapt.resolve) {
            if !method.params.iter().any(|p| &p.name == name) {
                return Err(format!("{at}: names no parameter {name}"));
            }
        }
        if adapt.future && kind != MethodKind::Await {
            return Err(format!("{at}: future needs an awaited method"));
        }
        method.name = f.sig.ident.to_string();
        let mut delegated = Vec::new();
        for input in &f.sig.inputs {
            let syn::FnArg::Typed(pt) = input else { continue };
            let (syn::Pat::Ident(ident), syn::Type::Path(ty)) = (&*pt.pat, &*pt.ty) else {
                continue;
            };
            let segments: Vec<String> =
                ty.path.segments.iter().map(|s| s.ident.to_string()).collect();
            if let [this, assoc] = &segments[..]
                && this == "Self"
                && let Some(d) = delegates.iter().find(|d| &d.assoc == assoc)
            {
                delegated.push((ident.ident.to_string(), d.attribute.clone()));
            }
        }
        Ok(FacadeMethod { method, native, via, delegated, none_ends_iteration, wrap, exit, adapt })
    }

    /// Whether the trait at `module.item` is a `#[host(proxy)]`.
    fn is_proxy(&self, module: usize, item: usize) -> bool {
        let syn::Item::Trait(tr) = &self.krate.modules[module].items[item] else { return false };
        voc::read(&tr.attrs).is_ok_and(|entries| voc::find(&entries, "proxy").is_some())
    }

    /// The host name of the `#[host(class)]` trait at `module.item`, if it is one.
    fn class_name(&self, module: usize, item: usize, at: &str) -> Result<Option<String>, String> {
        let syn::Item::Trait(tr) = &self.krate.modules[module].items[item] else { return Ok(None) };
        let entries = voc::read(&tr.attrs).map_err(|e| format!("{at}: {e}"))?;
        let Some(class) = voc::find(&entries, "class") else { return Ok(None) };
        Ok(Some(voc::text(at, &class.nested, "name")?.unwrap_or_else(|| tr.ident.to_string())))
    }
}

/// `key = ident`: a native method's Rust name.
fn rust_ident(at: &str, entry: &Entry, key: &str) -> Result<String, String> {
    match &entry.value {
        Some(Value::Rust(syn::Type::Path(path))) if path.path.get_ident().is_some() => {
            Ok(path.path.get_ident().expect("checked").to_string())
        }
        _ => Err(format!("{at}: `{key}` names a native method, unquoted")),
    }
}

/// `key(param = "host text", ..)`.
fn per_parameter(at: &str, entries: &[Entry], key: &str) -> Result<Vec<(String, String)>, String> {
    voc::find(entries, key).map_or(Ok(Vec::new()), |list| {
        list.nested
            .iter()
            .map(|p| match &p.value {
                Some(Value::Text(text)) => Ok((p.key.clone(), text.clone())),
                _ => Err(format!("{at}: {key}({} = ..) takes host text", p.key)),
            })
            .collect()
    })
}
