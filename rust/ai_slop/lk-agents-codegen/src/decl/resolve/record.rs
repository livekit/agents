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
        let (py_module, name, extract, emit) = record_target(at, record, &ident, unit_enum(item))?;
        let persisted = voc::find(&record.nested, "persisted").is_some();
        let transparent = voc::find(&record.nested, "transparent").is_some();
        let enumeration = voc::find(&record.nested, "enum").is_some();
        let enum_base =
            voc::text(at, &record.nested, "enum_base")?.unwrap_or_else(|| "StrEnum".into());
        if !matches!(enum_base.as_str(), "Enum" | "StrEnum")
            || (!enumeration && voc::find(&record.nested, "enum_base").is_some())
        {
            return Err(format!("{at}: enum_base selects Enum or StrEnum for an enum record"));
        }
        if voc::find(&record.nested, "finish").is_some()
            && voc::find(&record.nested, "try_finish").is_some()
        {
            return Err(format!("{at}: finish and try_finish are alternatives"));
        }
        let mapping = voc::find(&record.nested, "mapping").is_some();
        let typed_dict = voc::find(&record.nested, "typed_dict").is_some() || mapping;
        let pydantic = match voc::find(&record.nested, "pydantic") {
            None if ["base", "repr", "validator", "serializer"]
                .iter()
                .any(|w| voc::find(&record.nested, w).is_some()) =>
            {
                return Err(format!(
                    "{at}: `base`, `repr` and `validator` shape a pydantic record's model"
                ));
            }
            None => None,
            Some(_) if typed_dict || enumeration || transparent || persisted => {
                return Err(format!("{at}: a pydantic record is a model struct"));
            }
            Some(_) if !matches!(item, syn::Item::Struct(_)) => {
                return Err(format!("{at}: a pydantic record is a model struct"));
            }
            Some(_) => Some(Pydantic {
                base: voc::text(at, &record.nested, "base")?,
                repr: voc::text(at, &record.nested, "repr")?,
                validator: voc::text(at, &record.nested, "validator")?,
                serializer: voc::text(at, &record.nested, "serializer")?,
                arbitrary_types: voc::find(&record.nested, "arbitrary_types").is_some(),
                constants_after: voc::text(at, &record.nested, "constants_after")?,
                frozen: voc::find(&record.nested, "frozen").is_some(),
                private: match voc::find(&record.nested, "private") {
                    None => Vec::new(),
                    Some(p) => p
                        .nested
                        .iter()
                        .map(|e| match &e.value {
                            Some(Value::Text(ann)) if e.key.starts_with('_') => {
                                Ok((e.key.clone(), ann.clone()))
                            }
                            _ => Err(format!(
                                "{at}: private(_name = \"annotation\", ..) names host-only attributes"
                            )),
                        })
                        .collect::<Result<_, _>>()?,
                },
            }),
        };
        let dataclass = match voc::find(&record.nested, "dataclass") {
            None if voc::find(&record.nested, "post_init").is_some() => {
                return Err(format!("{at}: `post_init` shapes a mutable dataclass model"));
            }
            None => None,
            Some(_)
                if pydantic.is_some()
                    || typed_dict
                    || enumeration
                    || transparent
                    || persisted
                    || !matches!(item, syn::Item::Struct(_)) =>
            {
                return Err(format!("{at}: a mutable dataclass record is a plain model struct"));
            }
            Some(_) => Some(Dataclass {
                frozen: voc::find(&record.nested, "frozen").is_some(),
                deferred: voc::find(&record.nested, "deferred").is_some(),
                kw_only: voc::find(&record.nested, "kw_only").is_some(),
                slots: voc::find(&record.nested, "slots").is_some(),
                arbitrary_types: voc::find(&record.nested, "arbitrary_types").is_some(),
                post_init: voc::text(at, &record.nested, "post_init")?,
                host_fields: match voc::find(&record.nested, "host_fields") {
                    None => Vec::new(),
                    Some(h) => h
                        .nested
                        .iter()
                        .map(|e| {
                            voc::only(at, &e.nested, &["ty", "default", "doc"])?;
                            let need = |key: &str| {
                                voc::text(at, &e.nested, key)?.ok_or_else(|| {
                                    format!("{at}: host field `{}` needs `{key}`", e.key)
                                })
                            };
                            Ok(HostField {
                                name: e.key.clone(),
                                annotation: need("ty")?,
                                default: need("default")?,
                                doc: voc::text(at, &e.nested, "doc")?,
                            })
                        })
                        .collect::<Result<_, String>>()?,
                },
            }),
        };
        if dataclass.is_none() && voc::find(&record.nested, "host_fields").is_some() {
            return Err(format!("{at}: `host_fields` shape a mutable dataclass model"));
        }
        if dataclass.is_none()
            && ["kw_only", "slots", "deferred"]
                .iter()
                .any(|key| voc::find(&record.nested, key).is_some())
        {
            return Err(format!("{at}: kw_only and slots shape a dataclass model"));
        }
        if dataclass.is_none()
            && pydantic.is_none()
            && voc::find(&record.nested, "arbitrary_types").is_some()
        {
            return Err(format!("{at}: arbitrary_types shapes a dataclass or Pydantic model"));
        }
        if pydantic.is_none()
            && ["constants_after", "private"].iter().any(|w| voc::find(&record.nested, w).is_some())
        {
            return Err(format!("{at}: `constants_after` and `private` shape a pydantic model"));
        }
        let discriminator = voc::text(at, &record.nested, "discriminator")?;
        if discriminator.is_some() && !transparent {
            return Err(format!("{at}: a discriminated union is a transparent record"));
        }
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
            let en = syn::ItemEnum {
                variants: en.variants.iter().filter(|v| !native(v)).cloned().collect(),
                ..en.clone()
            };
            en.variants.iter().all(|v| match &v.fields {
                syn::Fields::Unnamed(f) => f.unnamed.len() == 1,
                syn::Fields::Unit => true,
                syn::Fields::Named(_) => false,
            }) && en
                .variants
                .iter()
                .filter(|v| {
                    matches!(v.fields, syn::Fields::Unit)
                        && voc::read(&v.attrs)
                            .is_ok_and(|entries| voc::find(&entries, "host_type").is_none())
                })
                .count()
                <= 1
                && en.variants.iter().any(|v| !matches!(v.fields, syn::Fields::Unit))
        };
        if transparent && !matches!(item, syn::Item::Enum(en) if values(en)) {
            return Err(format!(
                "{at}: a transparent record is an enum of one-value tuple variants, each crossing as its value, and at most one unit variant crossing as None"
            ));
        }
        if typed_dict && (persisted || !matches!(item, syn::Item::Struct(_))) {
            return Err(format!("{at}: a typed_dict record is a plain struct of keys"));
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
        let mut body = match item {
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
                        values.push((v.ident.to_string(), value, doc(&v.attrs)));
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
        if let Some(default) = voc::rust(at, &record.nested, "defaults_from")? {
            let syn::Type::Path(default) = default else {
                return Err(format!("{at}: default names a data constant"));
            };
            let expression: syn::Expr = syn::parse_quote!(#default);
            let syn::Expr::Struct(value) = k.constant(self.module, &expression)? else {
                return Err(format!("{at}: default is a record constant"));
            };
            if k.resolve(self.module, &value.path)?
                != (Found::Item { module: self.module, item: index })
            {
                return Err(format!("{at}: default must construct this record"));
            }
            let RecordBody::Struct(fields) = &mut body else {
                return Err(format!("{at}: only structs have record defaults"));
            };
            for field in fields {
                if field.default.is_some() {
                    return Err(format!(
                        "{at}: record and field defaults cannot duplicate each other"
                    ));
                }
                let source = value.fields.iter().find(|value| matches!(&value.member, syn::Member::Named(name) if *name == field.rust_name)).ok_or_else(|| format!("{at}: default is missing field {}", field.rust_name))?;
                let default = voc::literal(&source.expr).map_err(|e| format!("{at}: {e}"))?;
                check_default(at, &field.ty, Some(&default))?;
                field.default = Some(default);
            }
        }
        if pydantic.is_some()
            && let RecordBody::Struct(fields) = &body
            && fields.iter().any(|f| f.omitted.is_some())
        {
            return Err(format!(
                "{at}: a pydantic model computes a left-out field in Python: record(.., validator = \"..\")"
            ));
        }
        // A positional dataclass: a field without a default never follows one with a default.
        if let Some(d) = &dataclass
            && let RecordBody::Struct(fields) = &body
        {
            if d.post_init.is_some() && fields.iter().any(|f| f.omitted.is_some()) {
                return Err(format!("{at}: `post_init` computes in Python: no `omitted` field"));
            }
            let defaulted = |f: &&Field| {
                f.default.is_some() || f.factory.is_some() || matches!(f.ty, Ty::Maybe(_))
            };
            let modeled: Vec<&Field> =
                fields.iter().filter(|f| f.modeled() && !d.kw_only && !f.kw_only).collect();
            if let Some(i) = modeled.iter().position(defaulted)
                && let Some(f) = modeled[i..].iter().find(|f| !defaulted(f))
            {
                return Err(format!(
                    "{at}: `{}` has no default after a defaulted field: a dataclass is positional",
                    f.host_name()
                ));
            }
        }
        let members = crate::parse::members(item_attrs(item), "python", &ident)?;
        if dataclass.is_none()
            && crate::records::fields(&body).iter().any(|f| f.kw_only || f.hide_repr)
        {
            return Err(format!("{at}: kw_only and hide_repr fields belong to a dataclass"));
        }
        for (i, member) in members.iter().enumerate() {
            let RecordBody::Struct(fields) = &body else {
                return Err(format!("{at}: computed properties belong to record structs"));
            };
            if (member.is_property && !member.args.is_empty())
                || (!member.is_property && pydantic.is_none() && dataclass.is_none())
            {
                return Err(format!(
                    "{at}: properties take no arguments; model methods require pydantic or dataclass"
                ));
            }
            if fields.iter().any(|field| field.host_name() == member.name)
                || members[..i].iter().any(|other| other.name == member.name)
            {
                return Err(format!("{at}: duplicate record member `{}`", member.name));
            }
        }
        // The file's public path covers the records of the file's own module only.
        let path = match voc::text(at, &record.nested, "path")? {
            Some(path) => Some(path),
            None if super::file_default(&module.items, "module").as_ref() == Some(&py_module) => {
                super::file_default(&module.items, "path")
            }
            None => None,
        };
        let host_type = voc::text(at, &record.nested, "host_type")?;
        let missing = voc::text(at, &record.nested, "missing")?;
        if let Some(name) = &missing {
            syn::parse_str::<syn::Ident>(name)
                .map_err(|_| format!("{at}: missing names a singleton identifier"))?;
        }
        let marker = voc::find(&record.nested, "marker").is_some() || missing.is_some();
        if marker
            && (!matches!(&body, RecordBody::Struct(fields) if fields.is_empty())
                || host_type.is_some()
                || pydantic.is_some()
                || dataclass.is_some()
                || transparent
                || typed_dict
                || enumeration
                || persisted
                || !members.is_empty())
        {
            return Err(format!("{at}: a marker is an empty identity-only record"));
        }
        if host_type.as_ref().is_some_and(|path| {
            !path
                .rsplit_once('.')
                .is_some_and(|(module, name)| !module.is_empty() && !name.is_empty())
        }) {
            return Err(format!("{at}: host_type is a dotted class path"));
        }
        if host_type.is_some()
            && (!matches!(item, syn::Item::Struct(_))
                || transparent
                || typed_dict
                || pydantic.is_some()
                || dataclass.is_some()
                || persisted)
        {
            return Err(format!("{at}: an external host record is a plain struct"));
        }
        let emit_classes = if let Some(classes) = voc::find(&record.nested, "classes") {
            voc::only(at, &classes.nested, &["by", "some", "none"])?;
            let need = |key| {
                voc::text(at, &classes.nested, key)?
                    .ok_or_else(|| format!("{at}: classes needs {key}"))
            };
            Some(EmitClasses { field: need("by")?, some: need("some")?, none: need("none")? })
        } else {
            None
        };
        Ok(RecordDecl {
            marker,
            missing,
            output: voc::rust(at, &record.nested, "output")?
                .map(|ty| self.ty(&ty, at).map(|ty| ty.rust()))
                .transpose()?,
            finish: native_path(at, &record.nested, "finish")?.or(native_path(
                at,
                &record.nested,
                "try_finish",
            )?),
            finish_fallible: voc::find(&record.nested, "try_finish").is_some(),
            dict_keys: voc::find(&record.nested, "dict_keys").is_some(),
            emit_classes,
            mapping,
            boundary: mapping || voc::find(&record.nested, "boundary").is_some(),
            default_on_null: voc::find(&record.nested, "default_on_null").is_some(),
            validate: native_path(at, &record.nested, "validate")?,
            error_owner: voc::text(at, &record.nested, "error_owner")?,
            host_type,
            rust: k.extern_path(self.module, index).map_err(|e| format!("{at}: {e}"))?,
            name,
            module: py_module,
            path,
            family: module.path.first().cloned().ok_or_else(|| {
                format!("{at}: a record is declared in a subject module, not the crate root")
            })?,
            source: module.file.clone(),
            doc: doc(item_attrs(item)),
            extract,
            extract_by_hand: by_hand(at, &record.nested, "extract")?,
            emit,
            emit_by_hand: by_hand(at, &record.nested, "emit")?,
            transparent,
            typed_dict,
            enumeration,
            enum_base,
            pydantic,
            dataclass,
            hashable: derives(item_attrs(item), "Hash"),
            constants,
            persisted,
            members,
            dispatch,
            discriminator,
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
            _ if native(v) => {
                voc::only(&at, &entries, &["native"])?;
                VariantShape::Native
            }
            syn::Fields::Unit => {
                voc::only(
                    &at,
                    &entries,
                    if transparent { &["name", "host_type"] } else { &["name"] },
                )?;
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
                let allowed = if transparent {
                    &["name", "field", "skip", "default", "host_type"][..]
                } else {
                    &["name", "field", "skip", "default"][..]
                };
                voc::only(&at, &entries, allowed)?;
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
                    VariantShape::Tuple(Some(Box::new(Field {
                        kw_only: false,
                        hide_repr: false,
                        conversion: FieldConversion::default(),
                        rust_name: field,
                        ty,
                        default,
                        skip: false,
                        omitted: None,
                        factory: None,
                        host_type: None,
                        private: false,
                        annotation: None,
                        rename: None,
                        doc: Vec::new(),
                    })))
                }
            }
        };
        let host_type = voc::text(&at, &entries, "host_type")?
            .map(|path| {
                path.rsplit_once('.')
                    .map(|(module, name)| (module.to_owned(), name.to_owned()))
                    .ok_or_else(|| format!("{at}: host_type is a dotted class path"))
            })
            .transpose()?;
        let rust_name = v.ident.to_string();
        Ok(Variant {
            name: voc::text(&at, &entries, "name")?
                .unwrap_or_else(|| format!("{owner}{rust_name}")),
            rust_name,
            doc: doc(&v.attrs),
            shape,
            host_type,
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
            voc::only(
                &at,
                &entries,
                &[
                    "default",
                    "skip",
                    "omitted",
                    "default_factory",
                    "host_type",
                    "private",
                    "annotation",
                    "rename",
                    // Rust's own: `record(.., model_dump)` dumps the field through this function.
                    "dump_with",
                    // Rust's own: no host default; `record(.., default)` starts from the type's.
                    "required",
                    "read_as",
                    "normalize",
                    "try_normalize",
                    "extract_with",
                    "emit_with",
                    "lenient",
                    "missing_default",
                    "flatten",
                    "own_value",
                    "omit_none",
                    "late",
                    "when",
                    "none_default",
                    "emit_if_none",
                    "source",
                    "hidden",
                    "omit_empty",
                    "key_error",
                    "getter",
                    "kw_only",
                    "hide_repr",
                ],
            )?;
            let host_type = voc::text(&at, &entries, "host_type")?
                .map(|path| {
                    path.rsplit_once('.')
                        .map(|(module, name)| (module.to_owned(), name.to_owned()))
                        .ok_or_else(|| format!("{at}: host_type is a dotted class path"))
                })
                .transpose()?;
            let skip = voc::find(&entries, "skip").is_some();
            let factory = voc::text(&at, &entries, "default_factory")?;
            if factory.as_ref().is_some_and(|f| {
                !f.contains(['.', '(']) && !["list", "dict", "tuple"].contains(&f.as_str())
            }) {
                return Err(format!(
                    "{at}: default_factory names a host callable by `module.name` (a call with its arguments, `..utils.shortuuid(\"item_\")`), or `list`/`dict`/`tuple`"
                ));
            }
            let private = voc::find(&entries, "private").is_some();
            if private && factory.is_none() {
                return Err(format!("{at}: a private attribute defaults by its default_factory"));
            }
            let annotation = voc::text(&at, &entries, "annotation")?;
            let rename = voc::text(&at, &entries, "rename")?;
            if private && rename.is_some() {
                return Err(format!("{at}: a private attribute is named `_<field>`"));
            }
            let omitted = voc::rust(&at, &entries, "omitted")?.map(|path| {
                let path = quote::quote!(#path).to_string().replace(' ', "");
                match path.strip_prefix("crate::") {
                    Some(rest) => format!("::lk_agents_abi::{rest}"),
                    None => path,
                }
            });
            let read_as = voc::rust(&at, &entries, "read_as")?;
            let ty = if skip && factory.is_none() {
                Ty::Unit
            } else {
                self.ty(read_as.as_ref().unwrap_or(&f.ty), &at)?
            };
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
            let when = if let Some(when) = voc::find(&entries, "when") {
                voc::only(&at, &when.nested, &["field", "variant"])?;
                let field = voc::text(&at, &when.nested, "field")?.ok_or("when needs field")?;
                let variant =
                    native_path(&at, &when.nested, "variant")?.ok_or("when needs variant")?;
                Some((field, variant))
            } else {
                None
            };
            let source = if let Some(source) = voc::find(&entries, "source") {
                voc::only(&at, &source.nested, &["attrs", "constant"])?;
                let attrs = voc::find(&source.nested, "attrs")
                    .ok_or("source needs attrs")?
                    .nested
                    .iter()
                    .map(|field| field.key.clone())
                    .collect();
                Some(FieldSource {
                    attrs,
                    constant: voc::text(&at, &source.nested, "constant")?
                        .ok_or("source needs constant")?,
                })
            } else {
                None
            };
            if voc::find(&entries, "normalize").is_some()
                && voc::find(&entries, "try_normalize").is_some()
            {
                return Err(format!("{at}: normalize and try_normalize are alternatives"));
            }
            let conversion = FieldConversion {
                getter: voc::find(&entries, "getter").is_some(),
                key_error: voc::find(&entries, "key_error").is_some(),
                source,
                hidden: voc::find(&entries, "hidden").is_some(),
                omit_empty: voc::find(&entries, "omit_empty").is_some(),
                when,
                none_default: voc::find(&entries, "none_default").is_some(),
                emit_if_none: voc::text(&at, &entries, "emit_if_none")?,
                late: voc::find(&entries, "late").is_some(),
                normalize: native_path(&at, &entries, "normalize")?.or(native_path(
                    &at,
                    &entries,
                    "try_normalize",
                )?),
                fallible: voc::find(&entries, "try_normalize").is_some(),
                extract_with: voc::text(&at, &entries, "extract_with")?,
                emit_with: voc::text(&at, &entries, "emit_with")?,
                lenient: voc::find(&entries, "lenient").is_some(),
                missing_default: voc::find(&entries, "missing_default").is_some(),
                flatten: voc::find(&entries, "flatten").is_some(),
                own_value: voc::find(&entries, "own_value").is_some(),
                omit_none: voc::find(&entries, "omit_none").is_some(),
            };
            out.push(Field {
                kw_only: voc::find(&entries, "kw_only").is_some(),
                hide_repr: voc::find(&entries, "hide_repr").is_some(),
                conversion,
                rust_name: ident,
                ty,
                default,
                skip,
                omitted,
                factory,
                host_type,
                private,
                annotation,
                rename,
                doc: doc(&f.attrs),
            });
        }
        Ok(out)
    }
}

fn native(v: &syn::Variant) -> bool {
    voc::read(&v.attrs).is_ok_and(|entries| voc::find(&entries, "native").is_some())
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
    literal: bool,
) -> Result<(String, String, bool, bool), String> {
    voc::only(
        at,
        &record.nested,
        &[
            "module",
            "marker",
            "missing",
            "host_type",
            "path",
            "name",
            "extract",
            "emit",
            "persisted",
            "transparent",
            "enum",
            "enum_base",
            "constant",
            "nested_in",
            "dispatch",
            "typed_dict",
            "default",
            "defaults_from",
            "pydantic",
            "dataclass",
            "deferred",
            "kw_only",
            "slots",
            "post_init",
            "host_fields",
            "base",
            "repr",
            "validator",
            "serializer",
            "arbitrary_types",
            "constants_after",
            "frozen",
            "private",
            "discriminator",
            // Rust's own: the record marker expands `ModelDump` (`model_dump(mode="json")`).
            "model_dump",
            "mapping",
            "boundary",
            "default_on_null",
            "validate",
            "error_owner",
            "classes",
            "dict_keys",
            "output",
            "finish",
            "try_finish",
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
    // Explicit model declarations need no converter until a native entry actually uses them.
    // This includes subclassed Pydantic bases, host-facing configuration/event records and
    // `Literal` aliases (unit-only enums).
    if !extract
        && !emit
        && !literal
        && !["pydantic", "dataclass", "typed_dict", "enum", "transparent"]
            .iter()
            .any(|kind| voc::find(&record.nested, kind).is_some())
    {
        return Err(format!("{at}: a record crosses some way: extract, emit or both"));
    }
    Ok((module, name, extract, emit))
}

/// `extract(py)` / `emit(py)`: that direction is `lk-agents-py`'s hand-written conversion.
fn by_hand(at: &str, entries: &[Entry], key: &str) -> Result<bool, String> {
    let Some(e) = voc::find(entries, key) else { return Ok(false) };
    voc::only(at, &e.nested, &["py"])?;
    Ok(voc::find(&e.nested, "py").is_some())
}

/// A unit-only enum: it crosses as its string.
pub(super) fn unit_enum(item: &syn::Item) -> bool {
    matches!(item, syn::Item::Enum(e)
        if e.variants.iter().all(|v| matches!(v.fields, syn::Fields::Unit)))
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
        | Ty::HostSource { .. }
        | Ty::HostDuplex { .. }
        | Ty::Unit => {
            return Err(format!("{at}: `{}` is not a record field", ty.rust()));
        }
        // An emitted failure crosses as the exception it raises as; the host never hands one back.
        Ty::Error(_) if extract => {
            return Err(format!("{at}: `{}` is an emitted record field only", ty.rust()));
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

pub(super) fn check_default(at: &str, ty: &Ty, default: Option<&Literal>) -> Result<(), String> {
    let Some(d) = default else { return Ok(()) };
    let fits = match (ty, d) {
        (Ty::Optional(_), Literal::None) => true,
        // A transparent record crossing `None` (`records::check_none_defaults`).
        (Ty::Record(Record { host: HostForm::Record { .. }, .. }), Literal::None) => true,
        // A unit enum's string; the generated model's type check holds it to a member.
        (Ty::Record(r), Literal::Str(_))
            if r.literal && matches!(r.host, HostForm::Record { .. }) =>
        {
            true
        }
        // A hand-written conversion's host value (`False` for `AlignedTranscript::None`).
        (Ty::Record(Record { host: HostForm::Converted(..), .. }), _) => true,
        (Ty::Record(Record { host: HostForm::Via { via, .. }, .. }), d) => {
            return check_default(at, via, Some(d));
        }
        (Ty::Json, _) => true,
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

fn native_path(at: &str, entries: &[Entry], key: &str) -> Result<Option<String>, String> {
    voc::rust(at, entries, key).map(|value| {
        value.map(|path| {
            quote::quote!(#path)
                .to_string()
                .replace(' ', "")
                .replace("crate::", "::lk_agents_abi::")
        })
    })
}

pub(super) fn primitive_path(at: &str, declaration: &Entry, ident: &str) -> Result<String, String> {
    match voc::find(&declaration.nested, "py").and_then(|entry| entry.value.as_ref()) {
        Some(Value::Text(path)) => Ok(path.clone()),
        None => Ok(format!("py_values::{}", super::snake(ident))),
        _ => Err(format!("{at}: py names a binding primitive path")),
    }
}

/// Restrict checked representations to ordinary data types; a codec cannot recursively select
/// another native class, proxy or itself as its representation.
pub(super) fn representation(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Tuple(tuple) => tuple.elems.iter().all(representation),
        syn::Type::Path(path) if path.qself.is_none() => {
            let Some(last) = path.path.segments.last() else { return false };
            let name = last.ident.to_string();
            match &last.arguments {
                syn::PathArguments::None => {
                    [
                        "bool", "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64",
                        "isize", "f32", "f64", "String",
                    ]
                    .contains(&name.as_str())
                        || path.path == syn::parse_quote!(serde_json::Value)
                }
                syn::PathArguments::AngleBracketed(arguments) => [
                    "Vec", "Option", "BTreeMap", "BTreeSet",
                ]
                .contains(&name.as_str())
                    && arguments.args.iter().all(
                        |arg| matches!(arg, syn::GenericArgument::Type(ty) if representation(ty)),
                    ),
                _ => false,
            }
        }
        _ => false,
    }
}
