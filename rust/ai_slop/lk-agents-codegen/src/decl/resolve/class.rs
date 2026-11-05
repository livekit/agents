//! Class declarations and their native registration, over the shared type/method resolver.

use super::*;

impl Resolver<'_> {
    /// A class, or functions when `class` is `None`.
    pub(super) fn class(
        &mut self,
        tr: &syn::ItemTrait,
        class: Option<&Entry>,
    ) -> Result<Class, String> {
        let at = self.at;
        let entries = voc::read(&tr.attrs).map_err(|error| format!("{at}: {error}"))?;
        let nested = class
            .or_else(|| voc::find(&entries, "functions"))
            .map(|entry| entry.nested.as_slice())
            .unwrap_or_default();
        let allowed = if class.is_some() {
            &["name", "native", "module", "subclass", "pid"][..]
        } else {
            &["pid"][..]
        };
        voc::only(at, nested, allowed)?;
        let pid = voc::boolean(at, nested, "pid")?.unwrap_or(true);
        let host_module =
            voc::text(at, nested, "module")?.unwrap_or_else(|| "livekit.agents._core".into());
        let subclass = match voc::find(nested, "subclass") {
            Some(flag) if flag.value.is_some() || !flag.nested.is_empty() => {
                return Err(format!("{at}: `subclass` is a bare flag"));
            }
            value => value.is_some(),
        };
        let name = voc::text(at, nested, "name")?.unwrap_or_else(|| tr.ident.to_string());
        let native = match voc::rust(at, nested, "native")? {
            None => Vec::new(),
            Some(t) => match self.shape(self.module, &t, &[], at)? {
                Shape::Named { found: Found::Item { module, item }, .. } => {
                    let syn::Item::Trait(tr) = &self.krate.modules[module].items[item] else {
                        return Err(format!("{at}: `native` names a declared trait"));
                    };
                    let entries = voc::read(&tr.attrs).map_err(|e| format!("{at}: {e}"))?;
                    if voc::find(&entries, "native").is_some() {
                        vec![
                            self.krate
                                .extern_path(module, item)
                                .map_err(|e| format!("{at}: {e}"))?,
                        ]
                    } else {
                        match self.proxy(module, item, false, at)? {
                            Ty::Proxy { rust, .. } => vec![rust],
                            _ => unreachable!("proxy() returns a proxy"),
                        }
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
        let mut sinks = Vec::new();
        for inner in &tr.items {
            match inner {
                syn::TraitItem::Type(t) => {
                    let (assoc, sink) = self.assoc_type(t)?;
                    self.assoc.push(assoc);
                    sinks.extend(sink);
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
        for sink in &sinks {
            let paired = |ty: &Ty| matches!(ty, Ty::Duplex { class, .. } if *class == sink.class);
            if !methods.iter().any(|m| paired(&m.ret.ok)) {
                return Err(format!(
                    "{at}: the sink of `{}` is returned beside its source by a method: `(Self::Sink, Self::Source)`",
                    sink.class
                ));
            }
        }
        let form = match class {
            None if methods
                .iter()
                .all(|m| matches!(m.kind, MethodKind::Function | MethodKind::AsyncFunction)) =>
            {
                Form::Functions
            }
            None => {
                return Err(format!("{at}: every method of #[host(functions)] is a `function`"));
            }
            Some(_)
                if methods.iter().any(|m| {
                    matches!(m.kind, MethodKind::Function | MethodKind::AsyncFunction)
                }) =>
            {
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
            wrapper: crate::parse::wrapper(&module.file, &tr.attrs, &name)?,
            forwards: tr
                .items
                .iter()
                .filter_map(|item| {
                    let syn::TraitItem::Fn(method) = item else { return None };
                    Some(crate::parse::forward_module(&module.file, &method.attrs).and_then(
                        |target| {
                            let method_name = voc::text(at, &voc::read(&method.attrs)?, "name")?
                                .unwrap_or_else(|| method.sig.ident.to_string());
                            Ok(target.map(|target| (target, method_name, module.file.clone())))
                        },
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_iter()
                .flatten()
                .collect(),
            pid,
            host_module,
            subclass,
            name,
            rust_name: tr.ident.to_string(),
            trait_path: krate.extern_path(self.module, item).map_err(|e| format!("{at}: {e}"))?,
            family: module.path.first().cloned().ok_or_else(|| {
                format!("{at}: a class is declared in a subject module, not the crate root")
            })?,
            source: module.file.clone(),
            doc: doc(&tr.attrs),
            native,
            instances: self
                .assoc
                .iter()
                .filter_map(|a| match &a.ty {
                    Ty::Instance { assoc, class, .. } => Some((assoc.clone(), class.clone())),
                    _ => None,
                })
                .collect(),
            sinks,
            methods,
        })
    }
}

/// Authoring delegates are class metadata, separate from resolving native parameter types.
pub(super) fn facade(entries: &[Entry], at: &str) -> Result<Option<Facade>, String> {
    voc::find(entries, "facade")
        .map(|entry| {
            voc::only(
                at,
                &entry.nested,
                &[
                    "module",
                    "class",
                    "target",
                    "future",
                    "discard",
                    "omit",
                    "not_given",
                    "origin",
                    "returns",
                    "annotation",
                    "resolve",
                ],
            )?;
            let per_parameter = |key| {
                voc::find(&entry.nested, key).map_or(Ok(Vec::new()), |list| {
                    list.nested
                        .iter()
                        .map(|p| match &p.value {
                            Some(voc::Value::Text(text)) => Ok((p.key.clone(), text.clone())),
                            _ => Err(format!("{at}: facade {key}({} = ..) takes host text", p.key)),
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
            };
            let required = |name| {
                voc::text(at, &entry.nested, name)?
                    .ok_or_else(|| format!("{at}: facade requires {name}"))
            };
            Ok(Facade {
                module: required("module")?,
                class: required("class")?,
                target: required("target")?,
                future: voc::find(&entry.nested, "future").is_some(),
                discard: voc::find(&entry.nested, "discard").is_some(),
                omit: voc::text(at, &entry.nested, "omit")?,
                not_given: voc::find(&entry.nested, "not_given").is_some(),
                origin: voc::text(at, &entry.nested, "origin")?,
                returns: voc::text(at, &entry.nested, "returns")?,
                annotations: per_parameter("annotation")?,
                resolve: per_parameter("resolve")?,
            })
        })
        .transpose()
}
