//! Record schema resolution and field validation.

use super::*;

impl Resolver<'_> {
    /// A `#[host(record)]` struct or enum, its fields resolved in its own module.
    pub(super) fn record(&self, item: &syn::Item, record: &Entry) -> Result<RecordDecl, String> {
        let at = self.at;
        let k = self.krate;
        let module = &k.modules[self.module];
        let index =
            module.items.iter().position(|i| std::ptr::eq(i, item)).expect("read from here");
        let ident = item_ident(item);
        let (py_module, name, extract, emit) = record_target(at, record, &ident)?;
        let persisted = voc::find(&record.nested, "persisted").is_some();
        let transparent = voc::find(&record.nested, "transparent").is_some();
        let enumeration = voc::find(&record.nested, "enum").is_some();
        let dispatch = voc::text(at, &record.nested, "dispatch")?;
        if dispatch.is_some() && !transparent {
            return Err(format!(
                "{at}: a dispatched record is transparent: its members cross as themselves"
            ));
        }
        let constants = match voc::find(&record.nested, "constant") {
            None => Vec::new(),
            Some(c) => c
                .nested
                .iter()
                .map(|e| match &e.value {
                    Some(Value::Text(v)) if e.nested.is_empty() => Ok((e.key.clone(), v.clone())),
                    _ => Err(format!("{at}: constant(name = \"value\", ..) holds host text")),
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        if !constants.is_empty() && !matches!(item, syn::Item::Struct(_)) {
            return Err(format!("{at}: constants belong to a record struct's model"));
        }
        // One-value tuple variants, and at most one unit variant (the host's `None`).
        let values = |en: &syn::ItemEnum| {
            en.variants.iter().all(|v| match &v.fields {
                syn::Fields::Unnamed(f) => f.unnamed.len() == 1,
                syn::Fields::Unit => true,
                syn::Fields::Named(_) => false,
            }) && en.variants.iter().filter(|v| matches!(v.fields, syn::Fields::Unit)).count() <= 1
                && en.variants.iter().any(|v| !matches!(v.fields, syn::Fields::Unit))
        };
        if transparent && !matches!(item, syn::Item::Enum(en) if values(en)) {
            return Err(format!(
                "{at}: a transparent record is an enum of one-value tuple variants, each crossing as its value, and at most one unit variant crossing as None"
            ));
        }
        if name.contains('.') && (persisted || !matches!(item, syn::Item::Struct(_))) {
            return Err(format!("{at}: a nested model is a record struct, not persisted"));
        }
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
                } else if enumeration {
                    return Err(format!("{at}: record(.., enum) is a unit-only enum"));
                } else {
                    let mut variants = Vec::new();
                    for v in &en.variants {
                        variants.push(self.variant(
                            v,
                            &name,
                            (extract, emit),
                            transparent,
                            dispatch.is_some(),
                        )?);
                    }
                    RecordBody::Variants(variants)
                }
            }
            _ => unreachable!("records are structs and enums"),
        };
        let members = crate::parse::members(item_attrs(item), "python", &ident);
        for (i, member) in members.iter().enumerate() {
            let RecordBody::Struct(fields) = &body else {
                return Err(format!("{at}: computed properties belong to record structs"));
            };
            if !member.is_property || !member.args.is_empty() {
                return Err(format!(
                    "{at}: a record declares computed properties without arguments"
                ));
            }
            if fields.iter().any(|field| field.host_name() == member.name)
                || members[..i].iter().any(|other| other.name == member.name)
            {
                return Err(format!("{at}: duplicate record member `{}`", member.name));
            }
        }
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
            in_place: false,
            clone: derives(item_attrs(item), "Clone"),
            legacy: false,
            transparent,
            enumeration,
            constants,
            persisted,
            members,
            dispatch,
            body,
        })
    }

    fn variant(
        &self,
        v: &syn::Variant,
        owner: &str,
        dirs: (bool, bool),
        transparent: bool,
        dispatched: bool,
    ) -> Result<Variant, String> {
        let at = format!("{}::{}", self.at, v.ident);
        let mut entries = voc::read(&v.attrs).map_err(|e| format!("{at}: {e}"))?;
        let route = route(&at, &mut entries, dispatched)?;
        let shape = match &v.fields {
            syn::Fields::Unit => {
                voc::only(&at, &entries, &["name"])?;
                VariantShape::Unit
            }
            syn::Fields::Named(named) => {
                voc::only(&at, &entries, &["name"])?;
                let fields = self.fields(named.named.iter(), owner, dirs)?;
                if fields.iter().any(|f| f.omitted.is_some()) {
                    return Err(format!("{at}: `omitted` belongs to a record struct's field"));
                }
                VariantShape::Named(fields)
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
                    let field = match voc::text(&at, &entries, "field")? {
                        Some(field) if !transparent => field,
                        None if transparent => "value".to_owned(),
                        Some(_) => {
                            return Err(format!("{at}: a transparent variant's value has no name"));
                        }
                        None => {
                            return Err(format!(
                                "{at}: a tuple variant names its value: #[host(field = \"..\")]"
                            ));
                        }
                    };
                    let ty = self.ty(&only.ty, &at)?;
                    let default = literal_default(&at, &entries)?;
                    check_default(&at, &ty, default.as_ref())?;
                    check_record_field(&at, &ty, dirs)?;
                    VariantShape::Tuple(Some(Field {
                        rust_name: field,
                        ty,
                        default,
                        skip: false,
                        omitted: None,
                        factory: None,
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
            route,
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
            voc::only(&at, &entries, &["default", "skip", "omitted", "default_factory"])?;
            let skip = voc::find(&entries, "skip").is_some();
            let factory = voc::text(&at, &entries, "default_factory")?;
            if factory.as_ref().is_some_and(|f| !f.contains('.')) {
                return Err(format!(
                    "{at}: default_factory names a host callable by `module.name`"
                ));
            }
            let omitted = voc::rust(&at, &entries, "omitted")?.map(|path| {
                let path = quote::quote!(#path).to_string().replace(' ', "");
                match path.strip_prefix("crate::") {
                    Some(rest) => format!("::lk_agents_abi::{rest}"),
                    None => path,
                }
            });
            if skip && dirs.0 && factory.is_none() {
                return Err(format!("{at}: an extracted record reads every field: no `skip`"));
            }
            let ty = if skip && factory.is_none() { Ty::Unit } else { self.ty(&f.ty, &at)? };
            let default = literal_default(&at, &entries)?;
            if factory.is_some() && (default.is_some() || omitted.is_some()) {
                return Err(format!(
                    "{at}: default_factory is the field's one default (no `default`, `omitted`)"
                ));
            }
            check_default(&at, &ty, default.as_ref())?;
            if matches!(ty, Ty::Maybe(_)) && factory.is_some() {
                return Err(format!(
                    "{at}: an omittable field defaults to NOT_GIVEN, not default_factory"
                ));
            }
            if !skip || factory.is_some() {
                check_record_field(&at, &ty, dirs)?;
            }
            if omitted.is_some() {
                let scalar =
                    matches!(ty, Ty::Bool | Ty::Int(_) | Ty::Float | Ty::Duration | Ty::Str);
                if !dirs.0 || default.is_some() || !scalar {
                    return Err(format!(
                        "{at}: `omitted` computes a read scalar field the host leaves out: an extracted record's bool, number or string, with no `default`"
                    ));
                }
            }
            out.push(Field {
                rust_name: ident,
                ty,
                default,
                skip,
                omitted,
                factory,
                doc: doc(&f.attrs),
            });
        }
        Ok(out)
    }
}

/// A dispatched member's `event` / `event(mirror = "..")` / `ignored`, taken out of `entries`.
fn route(at: &str, entries: &mut Vec<Entry>, dispatched: bool) -> Result<Option<Route>, String> {
    let mut routes = Vec::new();
    entries.retain(|e| {
        let route = matches!(e.key.as_str(), "event" | "ignored");
        if route {
            routes.push(e.clone());
        }
        !route
    });
    let route = match &routes[..] {
        [] => return Ok(None),
        [one] => one,
        _ => return Err(format!("{at}: a member has one route: event or ignored")),
    };
    if !dispatched {
        return Err(format!("{at}: `{}` routes a member of a dispatched record", route.key));
    }
    if route.value.is_some() {
        return Err(format!("{at}: `{}` takes no value", route.key));
    }
    Ok(Some(match route.key.as_str() {
        "event" => {
            voc::only(at, &route.nested, &["mirror"])?;
            Route::Event { mirror: voc::text(at, &route.nested, "mirror")? }
        }
        _ if route.nested.is_empty() => Route::Ignored,
        _ => return Err(format!("{at}: `ignored` is a bare flag")),
    }))
}

/// `record(module = "..", name = "..", extract, emit)`: its module, host name and directions.
pub(super) fn record_target(
    at: &str,
    record: &Entry,
    ident: &str,
) -> Result<(String, String, bool, bool), String> {
    voc::only(
        at,
        &record.nested,
        &[
            "module",
            "name",
            "extract",
            "emit",
            "persisted",
            "transparent",
            "enum",
            "constant",
            "nested_in",
            "dispatch",
        ],
    )?;
    let module = voc::text(at, &record.nested, "module")?
        .ok_or_else(|| format!("{at}: record(module = \"..\") is its generated Python module"))?;
    let name = voc::text(at, &record.nested, "name")?.unwrap_or_else(|| ident.to_owned());
    // A nested model is named through its owner, `Owner.Name`, wherever it is named.
    let name = match voc::text(at, &record.nested, "nested_in")? {
        Some(owner) => format!("{owner}.{name}"),
        None => name,
    };
    let extract = voc::find(&record.nested, "extract").is_some();
    let emit = voc::find(&record.nested, "emit").is_some();
    if !extract && !emit {
        return Err(format!("{at}: a record crosses some way: extract, emit or both"));
    }
    Ok((module, name, extract, emit))
}

pub(super) fn record_entry(attrs: &[syn::Attribute]) -> Option<Entry> {
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
        | Ty::Sink { .. }
        | Ty::Duplex { .. }
        | Ty::Handle { .. }
        | Ty::Instance { .. }
        | Ty::HostSource(_)
        | Ty::HostDuplex { .. }
        | Ty::Error(_)
        | Ty::Unit => {
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
