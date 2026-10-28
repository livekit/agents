//! `Arc<dyn Trait>` of a `#[host(native)]` trait: the host classes registered for it cross as
//! themselves; there is no host proxy.

use super::*;

impl Resolver<'_> {
    pub(super) fn native(
        &self,
        module: usize,
        item: usize,
        native: &Entry,
        boxed: bool,
        at: &str,
    ) -> Result<Ty, String> {
        let ident = self.krate.item_name(module, item);
        if boxed {
            return Err(format!("{at}: a native-only trait crosses as `Arc<dyn {ident}>`"));
        }
        voc::only(at, &native.nested, &["refused"])?;
        let classes = self.native_classes(module, item, at)?;
        let refused = match voc::text(at, &native.nested, "refused")? {
            Some(refused) => refused,
            None => format!("expected native {}", classes.join(" or ")),
        };
        Ok(Ty::Native {
            rust: self.krate.extern_path(module, item).map_err(|e| format!("{at}: {e}"))?,
            classes,
            refused,
        })
    }

    /// The dotted host classes registered for the native trait: a declared class's
    /// `native = Trait`, or a legacy object's `as_dyn(Trait = ..)`.
    fn native_classes(&self, module: usize, item: usize, at: &str) -> Result<Vec<String>, String> {
        let wanted = Found::Item { module, item };
        let mut hosts = Vec::new();
        for (index, origin) in self.krate.modules.iter().enumerate() {
            for candidate in &origin.items {
                let attrs = match candidate {
                    syn::Item::Impl(imp) => &imp.attrs,
                    syn::Item::Trait(tr) => &tr.attrs,
                    _ => continue,
                };
                let mut registrations = Vec::new();
                for attr in attrs {
                    let meta = lk_agents_bindgen::Metas::of(std::slice::from_ref(attr))
                        .map_err(|e| format!("{at}: {e}"))?;
                    if matches!(meta.head(), "object" | "class") {
                        registrations.push(attr.clone());
                    }
                }
                let entries = voc::read(&registrations).map_err(|e| format!("{at}: {e}"))?;
                // Legacy object exports register `as_dyn(Trait = method)` in the same registry
                // as a declared class's `native = Trait`; reuse that identity for the annotation.
                if let syn::Item::Impl(imp) = candidate
                    && let Some(object) = voc::find(&entries, "object")
                    && let Some(as_dyn) = voc::find(&object.nested, "as_dyn")
                {
                    for registration in &as_dyn.nested {
                        let path: syn::Path = syn::parse_str(&registration.key)
                            .map_err(|e| format!("{at}: as_dyn trait: {e}"))?;
                        if self.krate.resolve(index, &path).is_ok_and(|found| found == wanted) {
                            let syn::Type::Path(ty) = &*imp.self_ty else { continue };
                            let name =
                                voc::text(at, &object.nested, "name")?.unwrap_or_else(|| {
                                    ty.path.segments.last().unwrap().ident.to_string()
                                });
                            hosts.push(format!("livekit.agents._core.{name}"));
                        }
                    }
                }
                if let Some(class) = voc::find(&entries, "class")
                    && let Some(syn::Type::Path(native)) = voc::rust(at, &class.nested, "native")?
                    && self.krate.resolve(index, &native.path).is_ok_and(|found| found == wanted)
                {
                    let name =
                        voc::text(at, &class.nested, "name")?.unwrap_or_else(|| match candidate {
                            syn::Item::Trait(tr) => tr.ident.to_string(),
                            _ => unreachable!("class declaration is a trait"),
                        });
                    let module = voc::text(at, &class.nested, "module")?
                        .unwrap_or_else(|| "livekit.agents._core".into());
                    hosts.push(format!("{module}.{name}"));
                }
            }
        }
        hosts.sort();
        hosts.dedup();
        if hosts.is_empty() {
            return Err(format!("{at}: a native-only trait needs a registered host class"));
        }
        Ok(hosts)
    }
}
