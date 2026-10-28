//! The records' Python models: `_generated/<module>.py` per declared module, merged with the
//! legacy generated module of the same name.

use super::*;
use crate::class_stub::{Imports, docstring, py_literal};
use std::collections::BTreeMap;

/// A legacy generated module with the declared records of the same module: their imports merged
/// into the legacy ones and their models right after them, so a legacy model or union naming a declared record
/// finds it defined; a record reusing a legacy name is refused.
pub fn merge(legacy: &str, models: &str) -> Result<String, String> {
    const FUTURE: &str = "from __future__ import annotations\n\n";
    let body = models.split_once(FUTURE).map_or(models, |(_, body)| body);
    let defined = |text: &str| -> BTreeSet<String> {
        text.lines()
            .filter_map(|l| {
                let name =
                    l.strip_prefix("class ").or_else(|| l.split_once(" = ").map(|(n, _)| n))?;
                let name: String =
                    name.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                (!name.is_empty() && !l.starts_with(' ')).then_some(name)
            })
            .collect()
    };
    let legacy_names = defined(legacy);
    if let Some(name) = defined(body).intersection(&legacy_names).next() {
        return Err(format!("the declared record `{name}` reuses a legacy model's name"));
    }
    let (imports, classes) = split_header(legacy);
    let (declared_imports, declared) = split_header(body);
    let imports = crate::imports::merge(imports, declared_imports)?;
    Ok(format!(
        "{}\n\n\n# The declared records.\n\n\n{}\n\n\n{}",
        imports.trim_end(),
        declared.trim(),
        classes.trim_start()
    ))
}

/// A generated module's header (comments, imports with their continuations and the
/// `TYPE_CHECKING` block, blank lines) and the definitions after it, from the first definition
/// or `# ---` section comment.
fn split_header(text: &str) -> (&str, &str) {
    let mut head = 0;
    for line in text.split_inclusive('\n') {
        let t = line.trim_end();
        let header = t.is_empty()
            || t.starts_with(' ')
            || t == ")"
            || t.starts_with("from ")
            || t.starts_with("import ")
            || t == "if TYPE_CHECKING:"
            || (t.starts_with('#') && !t.starts_with("# ---"));
        if !header {
            break;
        }
        head += line.len();
    }
    text.split_at(head)
}

/// Legacy models naming a declared record (a union member, a field) name its model: in the same
/// module as written, else imported from its module; a Rust name two records share is refused.
/// A record a legacy union holds is marked [`RecordDecl::legacy`].
pub fn link_legacy(legacy: &mut [crate::ir::Decl], decls: &mut Declarations) -> Result<(), String> {
    use crate::ir::{Body, Foreign, Host, Ty as L};
    let owned: BTreeSet<String> = legacy.iter().map(|d| d.rust_name.clone()).collect();
    // Rust ident -> (module, model); `None` when two records share the ident.
    let mut records: BTreeMap<String, Option<(String, String)>> = BTreeMap::new();
    for r in &decls.records {
        let ident = r.rust.rsplit("::").next().unwrap_or(&r.rust).to_owned();
        let model = Some((r.module.clone(), r.name.clone()));
        records.entry(ident).and_modify(|m| *m = None).or_insert(model);
    }
    type Records = BTreeMap<String, Option<(String, String)>>;
    fn walk(
        t: &mut L,
        module: &str,
        owned: &BTreeSet<String>,
        records: &Records,
        held: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        match t {
            L::Declared(n) if !owned.contains(n.as_str()) => match records.get(n.as_str()) {
                Some(None) => {
                    return Err(format!("`{n}` names two declared records: name the one meant"));
                }
                Some(Some((m, name))) => {
                    held.insert(n.clone());
                    let python_from = if m == module {
                        Vec::new()
                    } else {
                        vec![format!(".{m}:{}", name.split('.').next().unwrap_or(name))]
                    };
                    *t = L::Foreign(Foreign {
                        python: name.clone(),
                        python_from,
                        python_type_from: Vec::new(),
                    });
                }
                None => {}
            },
            L::Option(i) | L::List(i) => walk(i, module, owned, records, held)?,
            L::Map(k, v) => {
                walk(k, module, owned, records, held)?;
                walk(v, module, owned, records, held)?;
            }
            L::Tuple(items) => {
                for i in items {
                    walk(i, module, owned, records, held)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    let mut held = BTreeSet::new();
    for d in legacy.iter_mut() {
        let module = d.target(Host::Python).map(|t| t.module.clone()).unwrap_or_default();
        let at = format!("{}: {}", d.source, d.rust_name);
        // A union member is emitted by reference and picked by its tag; a field only by value.
        let (tys, members): (Vec<&mut L>, bool) = match &mut d.body {
            Body::Struct(fields) => (fields.iter_mut().map(|f| &mut f.ty).collect(), false),
            Body::Union(members) => (members.iter_mut().map(|m| &mut m.ty).collect(), true),
            Body::Unit(_) => (Vec::new(), false),
        };
        let mut found = BTreeSet::new();
        for t in tys {
            walk(t, &module, &owned, &records, &mut found).map_err(|e| format!("{at}: {e}"))?;
        }
        if members {
            held.extend(found);
        }
    }
    for r in &mut decls.records {
        r.legacy = held.contains(r.rust.rsplit("::").next().unwrap_or(&r.rust));
    }
    Ok(())
}

/// The models: `_generated/<module>.py` per declared module.
pub fn python(decls: &Declarations) -> Result<Vec<Out>, String> {
    let mut modules: Vec<&str> = decls.records.iter().map(|r| r.module.as_str()).collect();
    modules.sort();
    modules.dedup();
    let mut out = Vec::new();
    for module in modules {
        let records: Vec<&RecordDecl> =
            decls.records.iter().filter(|r| r.module == module).collect();
        let mut sources: Vec<&str> = records.iter().map(|r| r.source.as_str()).collect();
        sources.dedup();
        let mut runtime = Imports::default();
        let mut typing = Imports::default();
        let mut body = String::new();
        let own: Vec<String> = records
            .iter()
            .flat_map(|r| match &r.body {
                RecordBody::Variants(vs) if !r.transparent => std::iter::once(r.name.clone())
                    .chain(vs.iter().map(|v| v.name.clone()))
                    .collect::<Vec<_>>(),
                _ => vec![r.name.clone()],
            })
            .collect();
        for r in records.iter().filter(|r| !r.name.contains('.')) {
            body.push_str(&model_tree(r, &records, module, &own, &mut runtime, &mut typing));
        }
        for r in records.iter().filter(|r| r.name.contains('.')) {
            let (owner, _) = r.name.rsplit_once('.').expect("nested");
            let held = records.iter().any(|o| {
                o.name == owner && matches!(o.body, RecordBody::Struct(_)) && !o.transparent
            });
            if !held {
                return Err(format!(
                    "{}: {}: nested in `{owner}`, which is no record struct of module `{module}`",
                    r.source, r.rust
                ));
            }
        }
        let typing = typing.render(&[]);
        if !typing.is_empty() {
            runtime.add("typing", "TYPE_CHECKING");
        }
        let mut text = String::from(
            "# GENERATED FILE -- DO NOT EDIT.\n#\n# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`\n# declarations in:\n",
        );
        for s in sources {
            text.push_str(&format!("#     {s}\n"));
        }
        text.push_str("\nfrom __future__ import annotations\n\n");
        text.push_str(&runtime.render(&[]));
        if !typing.is_empty() {
            text.push_str("\nif TYPE_CHECKING:\n");
            for line in typing.lines() {
                if line.is_empty() {
                    text.push('\n');
                } else {
                    text.push_str(&format!("    {line}\n"));
                }
            }
        }
        // isort: one blank line after the imports unless a class follows.
        text.push_str(if body.starts_with("\n\n@") { &body } else { &body[1..] });
        out.push(Out { path: format!("{module}.py"), text });
    }
    Ok(out)
}

/// A record's model, with the models nested in it (`record(.., nested_in = "..")`) inside.
fn model_tree(
    r: &RecordDecl,
    records: &[&RecordDecl],
    module: &str,
    own: &[String],
    runtime: &mut Imports,
    typing: &mut Imports,
) -> String {
    let mut nested = String::new();
    let inner = records.iter().filter(|n| {
        n.name.strip_prefix(&format!("{}.", r.name)).is_some_and(|rest| !rest.contains('.'))
    });
    for n in inner {
        let text = model_tree(n, records, module, own, runtime, typing);
        if !nested.is_empty() {
            nested.push('\n');
        }
        for line in text.trim_start_matches('\n').lines() {
            nested.push_str(&if line.is_empty() {
                "\n".to_owned()
            } else {
                format!("    {line}\n")
            });
        }
    }
    model(r, module, own, &nested, runtime, typing)
}

fn model(
    r: &RecordDecl,
    module: &str,
    own: &[String],
    nested: &str,
    runtime: &mut Imports,
    typing: &mut Imports,
) -> String {
    // An emit-only record's derived models (`#[derive(Host)]`) reach the host as dicts.
    let out = r.emit && !r.extract;
    let all: Vec<&Field> = match &r.body {
        RecordBody::Struct(fields) => fields.iter().collect(),
        RecordBody::Variants(variants) => variants
            .iter()
            .flat_map(|v| match &v.shape {
                VariantShape::Named(fs) => fs.iter().collect(),
                VariantShape::Tuple(Some(f)) => vec![f],
                _ => Vec::new(),
            })
            .collect(),
        RecordBody::Literals(_) => Vec::new(),
    };
    for factory in all.iter().filter_map(|f| f.factory.as_deref()) {
        runtime.add("dataclasses", "field");
        runtime.add_module(factory.rsplit_once('.').map_or(factory, |(m, _)| m));
    }
    let mut class = |name: &str, doc: &[String], fields: &[&Field], nested: &str| {
        let name = name.rsplit('.').next().unwrap_or(name);
        let mut s =
            format!("\n\n@dataclass(frozen=True, kw_only=True, slots=True)\nclass {name}:\n");
        s.push_str(&docstring(doc, "    "));
        if !nested.is_empty() {
            if !doc.is_empty() {
                s.push('\n');
            }
            s.push_str(nested);
        }
        let fields: Vec<&&Field> = fields.iter().filter(|f| f.modeled()).collect();
        if fields.is_empty() {
            if doc.is_empty() && nested.is_empty() {
                s.push_str("    pass\n");
            }
            return s;
        }
        if !doc.is_empty() || !nested.is_empty() {
            s.push('\n');
        }
        for f in fields {
            let default = match (&f.default, &f.omitted) {
                _ if let Some(factory) = &f.factory => {
                    format!(" = field(default_factory={factory})")
                }
                (Some(d), _) => format!(" = {}", py_literal(d)),
                // Computed in `__post_init__` when left out: the field is never `None` after.
                (None, Some(_)) => " = None  # type: ignore[assignment]".to_owned(),
                (None, None) => String::new(),
            };
            let ann = annotation(&f.ty, module, own, out, typing);
            s.push_str(&format!("    {}: {ann}{default}\n", f.host_name()));
            s.push_str(&docstring(&f.doc, "    "));
        }
        s
    };
    match &r.body {
        RecordBody::Struct(fields) => {
            runtime.add("dataclasses", "dataclass");
            let mut s = class(&r.name, &r.doc, &fields.iter().collect::<Vec<_>>(), nested);
            if !r.constants.is_empty() {
                runtime.add("typing", "Literal");
                s = s.replace("    pass\n", "");
            }
            for (name, value) in &r.constants {
                s.push_str(&format!("    {name}: Literal[{value:?}] = {value:?}\n"));
            }
            let omitted = r.omitted();
            if !omitted.is_empty() {
                s.push_str(
                    "\n    def __post_init__(self) -> None:\n        from .. import _core\n\n",
                );
                for (f, function) in omitted {
                    let field = f.host_name();
                    s.push_str(&format!(
                        "        if self.{field} is None:\n            object.__setattr__(self, {field:?}, _core.{function}(self))\n"
                    ));
                }
            }
            for member in &r.members {
                s.push_str("\n    @property\n");
                s.push_str(&format!("    def {}(self) -> {}:\n", member.name, member.ret));
                s.push_str(&docstring(
                    &member.doc.lines().map(str::to_owned).collect::<Vec<_>>(),
                    "        ",
                ));
                s.push_str(&format!(
                    "        from .. import _core\n\n        return _core.{}(self)\n",
                    member.hook
                ));
            }
            if r.persisted {
                let (persist, restore) = r.persistence();
                s.push_str(&format!(
                    "\n    def to_json(self) -> str:\n        \"\"\"The saved form: restore it with ``from_json``.\"\"\"\n        from .. import _core\n\n        return _core.{persist}(self)\n\n    @classmethod\n    def from_json(cls, data: str) -> {name}:\n        \"\"\"The record ``to_json`` saved.\"\"\"\n        from .. import _core\n\n        return _core.{restore}(data)\n",
                    name = r.name
                ));
            }
            s
        }
        RecordBody::Variants(variants) if r.transparent => {
            runtime.add("typing", "TypeAlias");
            let members: Vec<String> = variants
                .iter()
                .filter_map(|v| match &v.shape {
                    VariantShape::Tuple(Some(f)) => {
                        Some(annotation(&f.ty, module, own, out, typing))
                    }
                    _ => None,
                })
                // `None` last: `None | float` is no type expression at runtime.
                .chain(
                    variants
                        .iter()
                        .any(|v| matches!(v.shape, VariantShape::Unit))
                        .then(|| "None".to_owned()),
                )
                .collect();
            let members: Vec<&str> = members.iter().map(String::as_str).collect();
            let mut s = format!("\n\n{}\n", union_alias(&r.name, &members));
            s.push_str(&docstring(&r.doc, ""));
            s
        }
        RecordBody::Variants(variants) => {
            runtime.add("dataclasses", "dataclass");
            runtime.add("typing", "TypeAlias");
            let mut s = String::new();
            for v in variants {
                let fields: Vec<&Field> = match &v.shape {
                    VariantShape::Unit | VariantShape::Tuple(None) => Vec::new(),
                    VariantShape::Tuple(Some(f)) => vec![f],
                    VariantShape::Named(fs) => fs.iter().collect(),
                };
                s.push_str(&class(&v.name, &v.doc, &fields, ""));
            }
            let names: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
            s.push_str(&format!("\n\n{}\n", union_alias(&r.name, &names)));
            s.push_str(&docstring(&r.doc, ""));
            s
        }
        RecordBody::Literals(values) if r.enumeration => {
            runtime.add("enum", "Enum");
            let mut s = format!("\n\nclass {}(str, Enum):\n", r.name);
            if !r.doc.is_empty() {
                s.push_str(&docstring(&r.doc, "    "));
                s.push('\n');
            }
            for (_, host) in values {
                s.push_str(&format!("    {} = {host:?}\n", host.to_uppercase()));
            }
            s
        }
        RecordBody::Literals(values) => {
            runtime.add("typing", "Literal");
            runtime.add("typing", "TypeAlias");
            let values: Vec<String> = values.iter().map(|(_, h)| format!("{h:?}")).collect();
            let mut s = format!("\n\n{}: TypeAlias = Literal[{}]\n", r.name, values.join(", "));
            s.push_str(&docstring(&r.doc, ""));
            s
        }
    }
}

/// `Name: TypeAlias = A | B`, wrapped as ruff formats it at 100 columns.
fn union_alias(name: &str, members: &[&str]) -> String {
    let head = format!("{name}: TypeAlias = ");
    let one = members.join(" | ");
    if head.len() + one.len() <= 100 {
        return format!("{head}{one}");
    }
    if one.len() + 4 <= 100 {
        return format!("{head}(\n    {one}\n)");
    }
    format!("{head}(\n    {}\n)", members.join("\n    | "))
}

fn annotation(ty: &Ty, module: &str, own: &[String], out: bool, typing: &mut Imports) -> String {
    match ty {
        Ty::Bool => "bool".into(),
        Ty::Int(_) => "int".into(),
        Ty::Float | Ty::Duration => "float".into(),
        Ty::Str => "str".into(),
        Ty::Json => "object".into(),
        Ty::JsonObject => "dict[str, object]".into(),
        Ty::Optional(t) => format!("{} | None", annotation(t, module, own, out, typing)),
        Ty::Tuple(items) => format!(
            "tuple[{}]",
            items
                .iter()
                .map(|item| annotation(item, module, own, out, typing))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::List(t) => format!("list[{}]", annotation(t, module, own, out, typing)),
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => format!(
            "dict[{}, {}]",
            annotation(k, module, own, out, typing),
            annotation(t, module, own, out, typing)
        ),
        Ty::Set(t) => format!("frozenset[{}]", annotation(t, module, own, out, typing)),
        Ty::Proxy { host, .. } => {
            let (m, name) = host.rsplit_once('.').expect("a proxy's host class is dotted");
            typing.add(m, name);
            name.to_owned()
        }
        Ty::Native { classes, .. } => classes
            .iter()
            .map(|class| {
                let (m, name) = class.rsplit_once('.').expect("a host class is dotted");
                typing.add(m, name);
                name
            })
            .collect::<Vec<_>>()
            .join(" | "),
        Ty::Record(r) if out && matches!(r.host, HostForm::Model { .. }) => {
            "dict[str, object]".into()
        }
        Ty::Record(r) => match &r.host {
            HostForm::Record { module: m, name } if m == module && own.contains(name) => {
                name.clone()
            }
            HostForm::Record { module: m, name } | HostForm::Model { module: m, name } => {
                typing.add(&format!(".{m}"), name);
                name.clone()
            }
            HostForm::CoreClass(name) => {
                typing.add(".._core", name);
                name.clone()
            }
            HostForm::External { module, name } => {
                typing.add(module, name);
                name.clone()
            }
            HostForm::Dict => "dict[str, object]".into(),
            HostForm::WireStrs => "str".into(),
            HostForm::Annotation(text) => text.clone(),
        },
        // Refused by the resolver (`check_record_field`).
        other => unreachable!("`{}` is not a record field", other.rust()),
    }
}
