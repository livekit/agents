//! The host class already registered for a native-only trait; no executable ABI proxy.

use super::*;

impl Resolver<'_> {
    pub(super) fn native_host(
        &self,
        module: usize,
        item: usize,
        at: &str,
    ) -> Result<String, String> {
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
        match hosts.as_slice() {
            [host] => Ok(host.clone()),
            _ => Err(format!(
                "{at}: a native-only trait needs one registered host class, found {}",
                hosts.len()
            )),
        }
    }
}
