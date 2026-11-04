//! The telemetry registry: the span, event, metric and record declarations of
//! `lk_agents_abi::telemetry`, read from the ABI source and checked, then written as the Weaver
//! registry, the consumer schema and the Python constants (`emit`).

mod emit;

pub use emit::{otel, semconv_py};

use crate::decl::tree::{Crate, Found};
use lk_agents_bindgen::telemetry::{self as t, InstrumentKind, Role, SpanKind, Unmeasured};
use std::collections::BTreeMap;

/// The ABI module the wire types and markers live in.
const DECL: [&str; 2] = ["telemetry", "decl"];
/// The ABI module of the dependency registries' constants, one submodule per registry.
const SEMCONV: [&str; 2] = ["telemetry", "semconv"];

/// An OTLP value type.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Wire {
    #[default]
    String,
    Int,
    Double,
    Bool,
    KvList,
    Array(Box<Wire>),
}

/// How a value is written inside its wire type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Encoding {
    Json,
    BoolString,
    Enum(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Content {
    Pii,
    GenAi,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attr {
    pub key: String,
    /// `agents` (owned here) or the dependency registry the key's constant comes from.
    pub registry: String,
    pub wire: Wire,
    pub required: bool,
    pub seconds: bool,
    pub encoding: Option<Encoding>,
    pub content: Option<Content>,
    /// The `unmeasured` encoding of a `Measured` value.
    pub unmeasured: Option<Unmeasured>,
    /// A family of keys under this prefix.
    pub template: bool,
    /// The key the field takes as a metric label (`label`, or `label = "key"`).
    pub label: Option<String>,
    /// The type is bounded (`Label`, an enum, `BoolString`): it may be a metric label.
    pub bounded: bool,
    pub metric: Option<String>,
    pub brief: String,
    /// The module of the struct that declares the field (`telemetry::spans::job`).
    pub declared_in: String,
    /// A typed `KvList`'s fields, read from its value type's serde form.
    pub shape: Option<Shape>,
}

/// The keys of a typed key-value list.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Fields(Vec<ShapeField>),
    /// An internally tagged enum: the tag key, and each variant's tag value and fields.
    Tagged {
        tag: String,
        variants: Vec<(String, Vec<ShapeField>)>,
    },
}

/// One key of a [`Shape`]; an optional value is omitted when absent.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeField {
    pub key: String,
    pub wire: Wire,
    pub required: bool,
    pub shape: Option<Shape>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Parent {
    Span(String),
    Current,
    External,
}

pub struct Span {
    pub name: String,
    pub kind: SpanKind,
    pub parents: Vec<Parent>,
    pub outlives_parent: Option<String>,
    /// `handle = Manual`: an entity-owned span of the sans-I/O crates.
    pub manual: bool,
    pub start: Vec<Attr>,
    pub end: Vec<Attr>,
    pub events: Vec<String>,
    pub brief: String,
}

pub struct Event {
    pub name: String,
    pub attrs: Vec<Attr>,
    pub brief: String,
}

pub struct Metric {
    pub name: String,
    pub kind: InstrumentKind,
    pub unit: String,
    pub labels: Vec<Attr>,
    pub brief: String,
    /// Imported from a dependency registry rather than declared here.
    pub upstream: bool,
}

pub struct Record {
    pub scope: String,
    pub body: String,
    pub attrs: Vec<Attr>,
    pub span_id: bool,
    pub brief: String,
}

/// An attribute group no declaration flattens: written by the exporter or the span handles.
pub struct Group {
    pub name: String,
    pub attrs: Vec<Attr>,
    pub brief: String,
}

pub struct Registry {
    pub spans: Vec<Span>,
    pub events: Vec<Event>,
    pub metrics: Vec<Metric>,
    pub records: Vec<Record>,
    pub groups: Vec<Group>,
    /// Every key owned here, as first declared, with a brief from one of its declarations.
    pub owned: Vec<Attr>,
    /// Every dependency key a declaration names.
    pub referenced: Vec<Attr>,
}

/// An `Attrs` struct: where it is and its role.
struct Decl<'a> {
    module: usize,
    item: &'a syn::ItemStruct,
    role: Option<Role>,
}

struct Reader<'a> {
    krate: &'a Crate,
    decls: BTreeMap<(usize, usize), Decl<'a>>,
    /// The values of the dependency constants, with their registry.
    semconv: BTreeMap<String, String>,
}

/// Reads and checks every telemetry declaration of `krate`.
pub fn read(krate: &Crate) -> Result<Registry, String> {
    let mut reader = Reader { krate, decls: BTreeMap::new(), semconv: BTreeMap::new() };
    for (m, module) in krate.modules.iter().enumerate() {
        if is_semconv(&module.path) {
            for item in &module.items {
                if let syn::Item::Const(c) = item {
                    let value = string(&krate.constant(m, &c.expr)?).ok_or_else(|| {
                        format!("{}::{} is a string constant", krate.display(m), c.ident)
                    })?;
                    reader.semconv.insert(value, module.path[2].clone());
                }
            }
        }
        for (i, item) in module.items.iter().enumerate() {
            let syn::Item::Struct(st) = item else { continue };
            if !derives(&st.attrs, "Attrs") {
                continue;
            }
            let here = format!("{}::{}", krate.display(m), st.ident);
            let mut role = None;
            for a in &st.attrs {
                let Some(name) = a.path().segments.last().map(|s| s.ident.to_string()) else {
                    continue;
                };
                if t::ROLES.contains(&name.as_str()) {
                    let args = a.meta.require_list().map_err(|e| format!("{here}: {e}"))?;
                    let parsed =
                        t::role(&name, args.tokens.clone()).map_err(|e| format!("{here}: {e}"))?;
                    if role.replace(parsed).is_some() {
                        return Err(format!("{here} has two telemetry roles"));
                    }
                }
            }
            reader.decls.insert((m, i), Decl { module: m, item: st, role });
        }
    }
    reader.registry()
}

impl Reader<'_> {
    fn registry(&self) -> Result<Registry, String> {
        let mut r = Registry {
            spans: Vec::new(),
            events: Vec::new(),
            metrics: Vec::new(),
            records: Vec::new(),
            groups: Vec::new(),
            owned: Vec::new(),
            referenced: Vec::new(),
        };
        let mut seen: BTreeMap<String, Attr> = BTreeMap::new();
        let mut note = |attrs: &[Attr], r: &mut Registry| -> Result<(), String> {
            for a in attrs {
                let Some(first) = seen.get_mut(&a.key) else {
                    seen.insert(a.key.clone(), a.clone());
                    match a.registry.as_str() {
                        "agents" => r.owned.push(a.clone()),
                        _ => r.referenced.push(a.clone()),
                    }
                    continue;
                };
                if (&first.wire, &first.encoding, first.template)
                    != (&a.wire, &a.encoding, a.template)
                {
                    return Err(format!(
                        "`{}` is declared as {:?} {:?} and as {:?} {:?}: one key, one type",
                        a.key, first.wire, first.encoding, a.wire, a.encoding
                    ));
                }
                if first.brief.is_empty() && !a.brief.is_empty() {
                    first.brief = a.brief.clone();
                    let all = r.owned.iter_mut().chain(&mut r.referenced);
                    all.filter(|o| o.key == a.key).for_each(|o| o.brief = a.brief.clone());
                }
            }
            Ok(())
        };
        let groups = self.flattened_or_ends();
        for (key, decl) in &self.decls {
            let (module, st) = (decl.module, decl.item);
            let here = format!("{}::{}", self.krate.display(module), st.ident);
            let brief = brief(&st.attrs);
            let attrs = self.attrs(*key)?;
            unique(&attrs, &here)?;
            note(&attrs, &mut r)?;
            if brief.is_empty() && (decl.role.is_some() || !groups.contains(key)) {
                return Err(format!("{here}: a declaration has a `///` brief"));
            }
            match &decl.role {
                None if groups.contains(key) => {}
                None => r.groups.push(Group {
                    name: lk_agents_bindgen::to_snake(&st.ident.to_string()),
                    attrs,
                    brief,
                }),
                Some(Role::Span(span)) => {
                    let end = match &span.end {
                        None => Vec::new(),
                        Some(ty) => {
                            let (target, optional) = self.attrs_type(module, ty, &here)?;
                            let mut end = self.attrs(target)?;
                            end.iter_mut().for_each(|a| a.required &= !optional);
                            end
                        }
                    };
                    unique(&end, &format!("{here}'s end"))?;
                    note(&end, &mut r)?;
                    let mut parents = Vec::new();
                    for p in &span.parents {
                        parents.push(self.parent(module, p, &here)?);
                    }
                    let mut events = Vec::new();
                    for e in &span.events {
                        let Some(Role::Event { name }) = self.struct_at(module, e, &here)?.1 else {
                            return Err(format!("{here}: `{}` is not an `#[event]`", path_str(e)));
                        };
                        events.push(name.value());
                    }
                    r.spans.push(Span {
                        name: span.name.value(),
                        kind: span.kind,
                        parents,
                        outlives_parent: span.outlives_parent.as_ref().map(|l| l.value()),
                        manual: span.manual,
                        start: attrs,
                        end,
                        events,
                        brief,
                    });
                }
                Some(Role::Event { name }) => {
                    r.events.push(Event { name: name.value(), attrs, brief });
                }
                Some(Role::Metric(metric)) => {
                    let mut labels = attrs;
                    for label in &mut labels {
                        if !label.bounded {
                            return Err(format!(
                                "{here}: the label `{}` is a `Label`, a `SpanValue` enum or a `BoolString`",
                                label.key
                            ));
                        }
                        label.label = Some(label.key.clone());
                    }
                    let (name, upstream) = match &metric.name {
                        t::Name::Lit(lit) => (lit.value(), false),
                        t::Name::Path(path) => (self.key(module, path, &here)?.0, true),
                    };
                    let unit = metric.unit.value();
                    r.metrics.push(Metric {
                        name,
                        kind: metric.kind,
                        unit,
                        labels,
                        brief,
                        upstream,
                    });
                }
                Some(Role::Record { scope, body }) => {
                    let span_id =
                        st.fields.iter().any(|f| matches!(t::field(f), Ok(t::Field::SpanId)));
                    let (scope, body) = (scope.value(), body.value());
                    r.records.push(Record { scope, body, attrs, span_id, brief });
                }
            }
        }
        check(&r)?;
        let mut listed = self.manual_list()?;
        let mut manual: Vec<String> =
            r.spans.iter().filter(|s| s.manual).map(|s| s.name.clone()).collect();
        listed.sort();
        manual.sort();
        if listed != manual {
            return Err(format!(
                "telemetry::spans::MANUAL lists {listed:?}; the `handle = Manual` spans are {manual:?}"
            ));
        }
        Ok(r)
    }

    /// The span names `telemetry::spans::MANUAL` lists, one `Declared::of::<D>()` each.
    fn manual_list(&self) -> Result<Vec<String>, String> {
        let here = "telemetry::spans::MANUAL";
        let found = self
            .krate
            .modules
            .iter()
            .enumerate()
            .filter(|(_, m)| m.path == ["telemetry", "spans"])
            .find_map(|(i, m)| {
                m.items.iter().find_map(|item| match item {
                    syn::Item::Const(c) if c.ident == "MANUAL" => Some((i, &*c.expr)),
                    _ => None,
                })
            });
        let Some((module, expr)) = found else { return Ok(Vec::new()) };
        let syn::Expr::Reference(r) = expr else { return Err(format!("{here} is `&[..]`")) };
        let syn::Expr::Array(items) = &*r.expr else { return Err(format!("{here} is `&[..]`")) };
        let mut names = Vec::new();
        for item in &items.elems {
            let ty = match item {
                syn::Expr::Call(call) => match &*call.func {
                    syn::Expr::Path(p) => p.path.segments.last().and_then(|s| match &s.arguments {
                        syn::PathArguments::AngleBracketed(a) => match a.args.first() {
                            Some(syn::GenericArgument::Type(syn::Type::Path(t))) => Some(&t.path),
                            _ => None,
                        },
                        _ => None,
                    }),
                    _ => None,
                },
                _ => None,
            };
            let ty = ty.ok_or_else(|| format!("{here} holds `Declared::of::<D>()` items"))?;
            match self.struct_at(module, ty, here)? {
                (_, Some(Role::Span(span))) => names.push(span.name.value()),
                _ => return Err(format!("{here}: `{}` is not a `#[span]`", path_str(ty))),
            }
        }
        Ok(names)
    }

    /// The structs a declaration flattens or names as a span's end: not groups of their own.
    fn flattened_or_ends(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for decl in self.decls.values() {
            let flattened = decl
                .item
                .fields
                .iter()
                .filter(|f| matches!(t::field(f), Ok(t::Field::Flatten)))
                .map(|f| &f.ty);
            let end = match &decl.role {
                Some(Role::Span(span)) => span.end.as_ref(),
                _ => None,
            };
            for ty in flattened.chain(end) {
                if let Ok((target, _)) = self.attrs_type(decl.module, ty, "") {
                    out.push(target);
                }
            }
        }
        out
    }

    /// The attributes of the `Attrs` struct at `key`, groups flattened.
    fn attrs(&self, key: (usize, usize)) -> Result<Vec<Attr>, String> {
        let decl = &self.decls[&key];
        let (module, st) = (decl.module, decl.item);
        let here = format!("{}::{}", self.krate.display(module), st.ident);
        let mut out = Vec::new();
        for field in &st.fields {
            let at = format!(
                "{here}.{}",
                field.ident.as_ref().map(|i| i.to_string()).unwrap_or_default()
            );
            match t::field(field).map_err(|e| format!("{at}: {e}"))? {
                t::Field::SpanId | t::Field::At => {}
                t::Field::Flatten => {
                    let (target, optional) = self.attrs_type(module, &field.ty, &at)?;
                    for mut a in self.attrs(target)? {
                        a.required &= !optional;
                        out.push(a);
                    }
                }
                t::Field::Attr(attr) => out.push(self.attr(module, field, attr, &at)?),
            }
        }
        Ok(out)
    }

    fn attr(
        &self,
        module: usize,
        field: &syn::Field,
        attr: t::Attr,
        at: &str,
    ) -> Result<Attr, String> {
        let (key, registry) = match &attr.key {
            t::Name::Lit(lit) => {
                let key = lit.value();
                if let Some(registry) = self.semconv.get(&key) {
                    return Err(format!(
                        "{at}: `{key}` belongs to the {registry} registry; name its `semconv::{registry}` constant"
                    ));
                }
                (key, "agents".to_owned())
            }
            t::Name::Path(path) => self.key(module, path, at)?,
        };
        let ty = self.ty(module, &field.ty, at)?;
        let fail = |message: &str| Err(format!("{at} (`{key}`): {message}"));
        let pii = key.split('.').take(2).eq(["lk", "pii"]);
        match (ty.content, pii) {
            (Some(Content::Pii), false) => return fail("`Content` is for keys under `lk.pii.*`"),
            (Some(Content::GenAi), _) if !key.starts_with("gen_ai.") || registry == "agents" => {
                return fail("`GenAiContent` is for the GenAI registry's message keys");
            }
            (Some(Content::GenAi) | None, true) => {
                return fail("a key under `lk.pii.*` is `Content<T>`");
            }
            _ => {}
        }
        if attr.unmeasured != Unmeasured::Omit && !ty.measured {
            return fail("`unmeasured` is for a `Measured` value");
        }
        if attr.metric.is_some() && !matches!(ty.wire, Wire::Int | Wire::Double) {
            return fail("a `metric` field is a number");
        }
        if ty.template && (registry != "agents" || attr.label || attr.metric.is_some()) {
            return fail("a `Template` is a plain prefix declared here");
        }
        let label = attr.label.then(|| attr.label_key.map_or_else(|| key.clone(), |k| k.value()));
        Ok(Attr {
            key,
            registry,
            wire: ty.wire,
            required: !ty.optional,
            seconds: ty.seconds,
            encoding: ty.encoding,
            content: ty.content,
            unmeasured: ty.measured.then_some(attr.unmeasured),
            template: ty.template,
            label,
            bounded: ty.label,
            metric: attr.metric.map(|m| m.value()),
            brief: brief(&field.attrs),
            declared_in: self.krate.display(module).replacen("crate::", "", 1),
            shape: ty.shape,
        })
    }

    /// A dependency constant: its value and its registry.
    fn key(&self, module: usize, path: &syn::Path, at: &str) -> Result<(String, String), String> {
        let found = self.krate.resolve(module, path).map_err(|e| format!("{at}: {e}"))?;
        let Found::Item { module: owner, .. } = found else {
            return Err(format!("{at}: `{}` is not a constant", path_str(path)));
        };
        let owner_path = &self.krate.modules[owner].path;
        if !is_semconv(owner_path) {
            return Err(format!(
                "{at}: `{}` is not a `semconv` constant; a key owned here is a literal",
                path_str(path)
            ));
        }
        let expr =
            syn::Expr::Path(syn::ExprPath { attrs: Vec::new(), qself: None, path: path.clone() });
        let value = string(&self.krate.constant(module, &expr).map_err(|e| format!("{at}: {e}"))?)
            .ok_or_else(|| format!("{at}: `{}` is a string constant", path_str(path)))?;
        Ok((value, owner_path[2].clone()))
    }

    /// The struct `path` names from `module`, with its role.
    fn struct_at(
        &self,
        module: usize,
        path: &syn::Path,
        at: &str,
    ) -> Result<((usize, usize), Option<&Role>), String> {
        match self.krate.resolve(module, path).map_err(|e| format!("{at}: {e}"))? {
            Found::Item { module, item } if self.decls.contains_key(&(module, item)) => {
                Ok(((module, item), self.decls[&(module, item)].role.as_ref()))
            }
            _ => Err(format!("{at}: `{}` is not an `Attrs` struct", path_str(path))),
        }
    }

    fn parent(&self, module: usize, path: &syn::Path, at: &str) -> Result<Parent, String> {
        if let Ok(Found::Item { module: m, item }) = self.krate.resolve(module, path)
            && self.krate.modules[m].path == DECL.map(String::from)
        {
            match self.krate.item_name(m, item).as_str() {
                "Current" => return Ok(Parent::Current),
                "External" => return Ok(Parent::External),
                _ => {}
            }
        }
        match self.struct_at(module, path, at)? {
            (_, Some(Role::Span(span))) => Ok(Parent::Span(span.name.value())),
            _ => Err(format!(
                "{at}: parent `{}` is not a `#[span]`, `Current` or `External`",
                path_str(path)
            )),
        }
    }

    /// The `Attrs` struct a group or end type names, and whether it is optional.
    fn attrs_type(
        &self,
        module: usize,
        ty: &syn::Type,
        at: &str,
    ) -> Result<((usize, usize), bool), String> {
        if let Some(inner) = generic(ty, "Option") {
            return Ok((self.attrs_type(module, inner, at)?.0, true));
        }
        match ty {
            syn::Type::Path(p) if p.qself.is_none() => {
                Ok((self.struct_at(module, &p.path, at)?.0, false))
            }
            _ => Err(format!("{at}: an attribute group is an `Attrs` struct")),
        }
    }

    /// A field type's wire encoding.
    fn ty(&self, module: usize, ty: &syn::Type, at: &str) -> Result<Ty, String> {
        let syn::Type::Path(p) = ty else {
            return Err(format!("{at}: unsupported attribute type"));
        };
        let last = p.path.segments.last().expect("a path has a segment");
        let arg =
            |name: &str| generic(ty, name).ok_or_else(|| format!("{at}: `{name}` takes one type"));
        let leaf = |wire: Wire| Ty { wire, ..Ty::default() };
        if p.path.segments.len() == 1 {
            match last.ident.to_string().as_str() {
                "bool" => return Ok(leaf(Wire::Bool)),
                "i64" | "u32" | "u64" => return Ok(leaf(Wire::Int)),
                "f64" => return Ok(leaf(Wire::Double)),
                _ => {}
            }
        }
        let found = self.krate.resolve(module, &p.path).map_err(|e| format!("{at}: {e}"))?;
        let name = last.ident.to_string();
        match &found {
            Found::External(path) if path == &["std", "vec", "Vec"] => {
                let inner = self.ty(module, arg("Vec")?, at)?;
                if inner.optional || inner.content.is_some() || inner.template || inner.measured {
                    return Err(format!("{at}: a `Vec` holds plain values"));
                }
                return Ok(Ty {
                    wire: Wire::Array(Box::new(inner.wire)),
                    shape: inner.shape,
                    ..Ty::default()
                });
            }
            Found::External(path) if path == &["std", "string", "String"] => {
                return Ok(leaf(Wire::String));
            }
            Found::External(path) if path == &["std", "option", "Option"] => {
                return Ok(Ty { optional: true, ..self.ty(module, arg("Option")?, at)? });
            }
            Found::Item { module: m, .. }
                if self.krate.modules[*m].path == DECL.map(String::from) =>
            {
                let wrap = |this: &Self| this.ty(module, arg(&name)?, at);
                return Ok(match name.as_str() {
                    "Seconds" => Ty { seconds: true, ..leaf(Wire::Double) },
                    "Label" => Ty { label: true, ..leaf(Wire::String) },
                    "BoolString" => Ty {
                        label: true,
                        encoding: Some(Encoding::BoolString),
                        ..leaf(Wire::String)
                    },
                    "Json" => Ty { encoding: Some(Encoding::Json), ..leaf(Wire::String) },
                    "KvList" => {
                        let value = arg("KvList")?;
                        let untyped =
                            matches!(value, syn::Type::Path(p) if p.path.is_ident("JsonObject"));
                        let shape =
                            if untyped { None } else { Some(self.shape(module, value, at)?) };
                        Ty { shape, ..leaf(Wire::KvList) }
                    }
                    "Measured" => Ty { measured: true, ..wrap(self)? },
                    "Template" => Ty { template: true, ..wrap(self)? },
                    "Content" | "GenAiContent" => {
                        let inner = wrap(self)?;
                        if inner.content.is_some() {
                            return Err(format!("{at}: content is wrapped once"));
                        }
                        let content = if name == "Content" { Content::Pii } else { Content::GenAi };
                        Ty { content: Some(content), ..inner }
                    }
                    _ => return Err(format!("{at}: `{name}` is not an attribute type")),
                });
            }
            Found::Item { module: m, item } => {
                if let syn::Item::Enum(e) = &self.krate.modules[*m].items[*item]
                    && derives(&e.attrs, "SpanValue")
                {
                    let members = e
                        .variants
                        .iter()
                        .map(|v| lk_agents_bindgen::to_snake(&v.ident.to_string()))
                        .collect();
                    return Ok(Ty {
                        label: true,
                        encoding: Some(Encoding::Enum(members)),
                        ..leaf(Wire::String)
                    });
                }
            }
            _ => {}
        }
        Err(format!("{at}: `{}` is not an attribute type", path_str(&p.path)))
    }
}

impl Reader<'_> {
    /// The keys a typed `KvList` value writes: a struct's fields, or an internally tagged enum's
    /// variants, as serde serializes them.
    fn shape(&self, module: usize, ty: &syn::Type, at: &str) -> Result<Shape, String> {
        let syn::Type::Path(p) = ty else {
            return Err(format!("{at}: a `KvList` holds a named type"));
        };
        let Found::Item { module: m, item } =
            self.krate.resolve(module, &p.path).map_err(|e| format!("{at}: {e}"))?
        else {
            return Err(format!("{at}: `{}` is not a declared type", path_str(&p.path)));
        };
        match &self.krate.modules[m].items[item] {
            syn::Item::Struct(st) => Ok(Shape::Fields(self.shape_fields(m, &st.fields, at)?)),
            syn::Item::Enum(e) => {
                let tag = serde_arg(&e.attrs, "tag")
                    .ok_or_else(|| format!("{at}: `{}` is an internally tagged enum", e.ident))?;
                let mut variants = Vec::new();
                for v in &e.variants {
                    let syn::Fields::Unnamed(f) = &v.fields else {
                        return Err(format!("{at}: `{}::{}` wraps one struct", e.ident, v.ident));
                    };
                    let Some(Shape::Fields(fields)) =
                        f.unnamed.first().map(|f| self.shape(m, &f.ty, at)).transpose()?
                    else {
                        return Err(format!("{at}: `{}::{}` wraps one struct", e.ident, v.ident));
                    };
                    variants.push((lk_agents_bindgen::to_snake(&v.ident.to_string()), fields));
                }
                Ok(Shape::Tagged { tag, variants })
            }
            _ => Err(format!("{at}: `{}` is not a struct or enum", path_str(&p.path))),
        }
    }

    fn shape_fields(
        &self,
        module: usize,
        fields: &syn::Fields,
        at: &str,
    ) -> Result<Vec<ShapeField>, String> {
        let mut out = Vec::new();
        for f in fields {
            let key = f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default();
            let (ty, required) = match generic(&f.ty, "Option") {
                Some(inner) => (inner, false),
                None => (&f.ty, true),
            };
            let (wire, shape) = self.shape_value(module, ty, &format!("{at}.{key}"))?;
            out.push(ShapeField { key, wire, required, shape });
        }
        Ok(out)
    }

    fn shape_value(
        &self,
        module: usize,
        ty: &syn::Type,
        at: &str,
    ) -> Result<(Wire, Option<Shape>), String> {
        if let Some(inner) = generic(ty, "Vec") {
            let (wire, shape) = self.shape_value(module, inner, at)?;
            return Ok((Wire::Array(Box::new(wire)), shape));
        }
        let syn::Type::Path(p) = ty else { return Err(format!("{at}: unsupported value type")) };
        let name = p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
        Ok(match name.as_str() {
            "bool" => (Wire::Bool, None),
            "i64" | "u32" | "u64" => (Wire::Int, None),
            "f64" => (Wire::Double, None),
            "String" => (Wire::String, None),
            _ => (Wire::KvList, Some(self.shape(module, ty, at)?)),
        })
    }
}

/// A `#[serde(name = "value")]` argument.
fn serde_arg(attrs: &[syn::Attribute], name: &str) -> Option<String> {
    let mut found = None;
    for a in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        let _ = a.parse_nested_meta(|m| {
            if m.input.peek(syn::Token![=]) {
                let value: syn::LitStr = m.value()?.parse()?;
                if m.path.is_ident(name) {
                    found = Some(value.value());
                }
            }
            Ok(())
        });
    }
    found
}

#[derive(Default)]
struct Ty {
    wire: Wire,
    optional: bool,
    seconds: bool,
    encoding: Option<Encoding>,
    content: Option<Content>,
    measured: bool,
    template: bool,
    /// Bounded: may be a metric label.
    label: bool,
    shape: Option<Shape>,
}

/// The registry-wide rules no single declaration can check.
fn check(r: &Registry) -> Result<(), String> {
    let mut names = BTreeMap::new();
    for (kind, name) in r
        .spans
        .iter()
        .map(|s| ("span", s.name.clone()))
        .chain(r.events.iter().map(|e| ("event", e.name.clone())))
        .chain(r.metrics.iter().map(|m| ("metric", m.name.clone())))
        .chain(r.records.iter().map(|rec| ("record", format!("{} / {}", rec.scope, rec.body))))
    {
        if names.insert((kind, name.clone()), ()).is_some() {
            return Err(format!("the {kind} `{name}` is declared twice"));
        }
    }
    for span in &r.spans {
        for p in &span.parents {
            if let Parent::Span(name) = p
                && !r.spans.iter().any(|s| &s.name == name)
            {
                return Err(format!("{}: unknown parent {name}", span.name));
            }
        }
        let labels: Vec<&String> =
            span.start.iter().chain(&span.end).filter_map(|a| a.label.as_ref()).collect();
        for a in span.start.iter().chain(&span.end) {
            let Some(metric) = &a.metric else { continue };
            let Some(m) = r.metrics.iter().find(|m| &m.name == metric) else {
                return Err(format!(
                    "{} (`{}`): `metric = \"{metric}\"` is not a declared instrument",
                    span.name, a.key
                ));
            };
            // The measurement carries the span's labels: exactly the instrument's label keys.
            if let Some(extra) = labels.iter().find(|l| !m.labels.iter().any(|ml| &&ml.key == *l)) {
                return Err(format!(
                    "{} (`{}`): the label `{extra}` is not a label of `{metric}`",
                    span.name, a.key
                ));
            }
            if let Some(missing) =
                m.labels.iter().find(|ml| ml.required && !labels.contains(&&ml.key))
            {
                return Err(format!(
                    "{} (`{}`): `{metric}` needs the label `{}`",
                    span.name, a.key, missing.key
                ));
            }
        }
    }
    for a in &r.owned {
        if a.brief.is_empty() {
            return Err(format!(
                "`{}` is owned here and needs a `///` brief on one of its fields",
                a.key
            ));
        }
    }
    Ok(())
}

/// Two fields of one attribute set under the same key.
fn unique(attrs: &[Attr], here: &str) -> Result<(), String> {
    for (i, a) in attrs.iter().enumerate() {
        if attrs[..i].iter().any(|b| b.key == a.key) {
            return Err(format!("{here}: `{}` is written twice", a.key));
        }
    }
    Ok(())
}

fn is_semconv(path: &[String]) -> bool {
    path.len() == 3 && path[..2] == SEMCONV
}

fn derives(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().filter(|a| a.path().is_ident("derive")).any(|a| {
        a.parse_args_with(
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        )
        .is_ok_and(|paths| paths.iter().any(|p| p.segments.last().is_some_and(|s| s.ident == name)))
    })
}

fn generic<'t>(ty: &'t syn::Type, name: &str) -> Option<&'t syn::Type> {
    let syn::Type::Path(p) = ty else { return None };
    let last = p.path.segments.last()?;
    if last.ident != name {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else { return None };
    match args.args.first()? {
        syn::GenericArgument::Type(t) if args.args.len() == 1 => Some(t),
        _ => None,
    }
}

fn string(expr: &syn::Expr) -> Option<String> {
    match expr {
        syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) => Some(s.value()),
        _ => None,
    }
}

fn path_str(path: &syn::Path) -> String {
    path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::")
}

/// The first paragraph of the doc comment, on one line.
fn brief(attrs: &[syn::Attribute]) -> String {
    lk_agents_bindgen::summary(&lk_agents_bindgen::doc(attrs)).join(" ")
}
