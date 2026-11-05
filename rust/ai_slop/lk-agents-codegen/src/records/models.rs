//! The records' Python models: `_generated/<module>.py` per declared module, merged with the
//! authored presentation of the same name.

use super::*;
use crate::class_stub::{Imports, docstring, py_literal};
use crate::decl::Route;

/// A authored presentation with the declared records of the same module: their imports merged
/// into the presentation header and their models right after it, so authored base classes naming a record
/// find it defined; a record reusing an authored name is refused.
pub fn merge(presentation: &str, models: &str) -> Result<String, String> {
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
    let authored_names = defined(presentation);
    if let Some(name) = defined(body).intersection(&authored_names).next() {
        return Err(format!("the declared record `{name}` reuses an authored definition"));
    }
    let (imports, classes) = split_header(presentation);
    let (declared_imports, declared) = split_header(body);
    let imports = crate::imports::merge(imports, declared_imports)?;
    // The provenance header names both declaration forms and every source file.
    let declared_sources: String = models
        .split_once(FUTURE)
        .map_or("", |(head, _)| head)
        .lines()
        .filter(|l| l.starts_with("#     ") && !imports.contains(l))
        .map(|l| format!("{l}\n"))
        .collect();
    let imports = imports.trim_end();
    // isort: one blank line between an import statement and the comment, two after a block.
    let gap = match imports.lines().last() {
        Some(last) if !last.starts_with(' ') => "\n\n",
        _ => "\n\n\n",
    };
    Ok(format!(
        "{imports}{gap}# Record declarations:\n{declared_sources}\n\n{}\n\n\n{}",
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

/// The models: `_generated/<module>.py` per declared module.
pub fn python(decls: &Declarations) -> Result<Vec<Out>, String> {
    // Flattening describes the actual host dictionary, not a nested native implementation field.
    let mut host = decls.clone();
    let markers: Vec<_> = decls
        .records
        .iter()
        .filter(|record| record.marker)
        .map(|record| (record.path.clone(), record.name.clone(), record.module.clone()))
        .collect();
    for record in &mut host.records {
        if let RecordBody::Struct(fields) = &record.body {
            record.body = RecordBody::Struct(flatten_fields(fields, decls)?);
        }
        if let RecordBody::Variants(variants) = &mut record.body {
            for variant in variants {
                if let Some((module, name)) = &mut variant.host_type
                    && let Some((_, _, generated)) = markers
                        .iter()
                        .find(|(path, marker, _)| path.as_ref() == Some(module) && marker == name)
                {
                    *module = format!(".{generated}");
                }
            }
        }
    }
    let decls = &host;
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
        let mut typing = Imports::new(decls);
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
        // A union alias is evaluated where it stands: after every class (a dataclass's annotations
        // are lazy). A pydantic model resolves its annotations as it is built, so a module holding
        // one keeps the declaration order: an alias before the models naming it.
        let (aliases, classes): (Vec<&&RecordDecl>, Vec<&&RecordDecl>) =
            records.iter().filter(|r| !r.name.contains('.')).partition(|r| r.transparent);
        let order: Vec<&&RecordDecl> = match records.iter().any(|r| r.pydantic.is_some()) {
            true => records.iter().filter(|r| !r.name.contains('.')).collect(),
            false => classes.into_iter().chain(aliases).collect(),
        };
        let mut dataclasses = Vec::new();
        for r in order {
            let text = model_tree(r, &records, module, &own, &mut runtime, &mut typing);
            if r.dataclass.as_ref().is_some_and(|d| !d.deferred) && r.path.is_some() {
                dataclasses.push((r, text.clone()));
            }
            body.push_str(&text);
        }
        // A public dataclass's annotations are resolved at import (`public_paths`): a name only
        // the type checker imports would fail the import.
        for (r, text) in dataclasses {
            let annotations = text.lines().filter_map(|l| {
                let l = l.strip_prefix("    ")?;
                l.split_once(": ").filter(|(name, _)| !name.contains(' ')).map(|(_, ann)| ann)
            });
            for ann in annotations {
                for (module, name) in typing.entries() {
                    let runtime_too = runtime.entries().iter().any(|(_, n)| n == name);
                    let used = crate::class_stub::annotation_names(ann)
                        .iter()
                        .any(|(_, word)| word == name);
                    if used && !runtime_too {
                        return Err(format!(
                            "{}: {}: `{name}` (from {module}) is imported for the type checker only, but a public dataclass resolves its annotations at import",
                            r.source, r.name
                        ));
                    }
                }
            }
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
        body.push_str(&public_paths(&records, &mut runtime));
        for imports in [&mut runtime, &mut typing] {
            imports.remove_module(&format!(".{module}"));
            imports.remove_module(&format!("{PACKAGE}.{module}"));
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
        text.push_str(
            "#\n# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a\n# non-empty diff, so a hand edit here is reverted rather than merged.\n",
        );
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
        let class = body.starts_with("\n\n@") || body.starts_with("\n\nclass ");
        text.push_str(if class { &body } else { body.strip_prefix('\n').unwrap_or(&body) });
        out.push(Out { path: format!("{module}.py"), text });
    }
    for r in decls.records.iter().filter(|r| r.dispatch.is_some()) {
        out.push(dispatch(r, decls)?);
    }
    Ok(out)
}

/// `X.__module__ = "<path>"` for each class of a record with a public module, after every class:
/// `repr`, `pickle` and the docs report the import path users write, not `_generated.*`. An alias
/// (`Literal`, union) has no settable `__module__`.
///
/// Evaluate the annotation expressions before changing module identity, without recursively
/// resolving forward references inside callback signatures to modules still being initialized.
fn public_paths(records: &[&RecordDecl], runtime: &mut Imports) -> String {
    let mut lines = Vec::new();
    for r in records {
        let Some(path) = &r.path else { continue };
        if r.dataclass.as_ref().is_some_and(|d| !d.deferred) {
            runtime.add("inspect", "get_annotations");
            lines.push(format!(
                "{0}.__annotations__ = get_annotations({0}, eval_str=True)\n",
                r.name
            ));
        }
        let classes: Vec<&str> = match &r.body {
            RecordBody::Struct(_) => vec![&r.name],
            RecordBody::Variants(vs) if !r.transparent => {
                vs.iter().map(|v| v.name.as_str()).collect()
            }
            RecordBody::Literals(_) if r.enumeration => vec![&r.name],
            _ => Vec::new(),
        };
        lines.extend(classes.into_iter().map(|c| format!("{c}.__module__ = {path:?}\n")));
    }
    match lines.is_empty() {
        true => String::new(),
        false => format!("\n\n{}", lines.concat()),
    }
}

/// `_generated/<dispatch>.py`: a dispatched record's routed members, keyed by model class.
fn dispatch(r: &RecordDecl, decls: &Declarations) -> Result<Out, String> {
    let RecordBody::Variants(variants) = &r.body else {
        unreachable!("a dispatched record is a transparent enum")
    };
    let mut imports = Imports::default();
    let (mut events, mut mirrors, mut ignored) = (String::new(), String::new(), String::new());
    for v in variants {
        let Some(route) = &v.route else { continue };
        let at = format!("{}: {}::{}", r.source, r.rust, v.rust_name);
        let VariantShape::Tuple(Some(field)) = &v.shape else {
            return Err(format!("{at}: a routed member is a declared record"));
        };
        let Field {
            ty: Ty::Record(Record { host: HostForm::Record { module, name }, .. }), ..
        } = field.as_ref()
        else {
            return Err(format!("{at}: a routed member is a declared record"));
        };
        imports.add(&format!(".{module}"), name);
        match route {
            Route::Ignored => ignored.push_str(&format!("        {name},\n")),
            Route::Event { mirror } => {
                let event = decls
                    .records
                    .iter()
                    .find(|x| &x.module == module && &x.name == name)
                    .and_then(|x| x.constants.iter().find(|(key, _)| key == "type"))
                    .map(|(_, value)| value)
                    .ok_or_else(|| {
                        format!("{at}: an event's model names it with constant(type = \"..\")")
                    })?;
                events.push_str(&format!("    {name}: {event:?},\n"));
                if let Some(attr) = mirror {
                    mirrors.push_str(&format!("    {name}: {attr:?},\n"));
                }
            }
        }
    }
    let map = |entries: &str| match entries {
        "" => "{}".to_owned(),
        _ => format!("{{\n{entries}}}"),
    };
    let set = match ignored.as_str() {
        "" => "frozenset()".to_owned(),
        _ => format!("frozenset(\n    {{\n{ignored}    }}\n)"),
    };
    let union = &r.name;
    Ok(Out {
        path: format!("{}.py", r.dispatch.as_deref().expect("dispatched")),
        text: format!(
            "# GENERATED FILE -- DO NOT EDIT.\n#\n# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`\n# declarations in:\n#     {source}\n\n{imports}\n# `{union}` members the host emits as public session events: {{model class: event name}}.\nPUBLIC_EVENTS: dict[type, str] = {events}\n\n# Public events whose `new_state` the host keeps on the session: {{model class: attribute}}.\nMIRRORED_STATES: dict[type, str] = {mirrors}\n\n# `{union}` members the host receives and does not act on.\nIGNORED_EVENTS: frozenset[type] = {set}\n",
            source = r.source,
            imports = imports.render(&[]),
            events = map(&events),
            mirrors = map(&mirrors),
        ),
    })
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
    if r.marker {
        let mut doc = docstring(&r.doc, "    ");
        if !doc.is_empty() {
            doc.push('\n');
        }
        if let Some(singleton) = &r.missing {
            runtime.add("typing", "Literal");
            runtime.add("pydantic", "GetCoreSchemaHandler");
            runtime.add("pydantic_core", "CoreSchema");
            runtime.add("pydantic_core", "core_schema");
            return format!(
                "\n\nclass {name}:\n{doc}    __slots__ = ()\n\n    def __bool__(self) -> Literal[False]:\n        return False\n\n    def __repr__(self) -> str:\n        return {singleton:?}\n\n    @classmethod\n    def __get_pydantic_core_schema__(\n        cls, source_type: object, handler: GetCoreSchemaHandler\n    ) -> CoreSchema:\n        return core_schema.is_instance_schema(cls)\n\n\n{singleton} = {name}()\n",
                name = r.name,
            );
        }
        return format!("\n\nclass {}:\n{doc}    pass\n", r.name);
    }
    if let Some(classes) = &r.emit_classes {
        let (some_module, some_class) = classes.some.rsplit_once('.').expect("dotted class");
        let (none_module, none_class) = classes.none.rsplit_once('.').expect("dotted class");
        runtime.add(some_module, &format!("{some_class} as _Some{}", r.name));
        runtime.add(none_module, &format!("{none_class} as _None{}", r.name));
        return format!("\n\n{} = _Some{} | _None{}\n", r.name, r.name, r.name);
    }
    if let Some(host_type) = &r.host_type {
        let (module, class) = host_type.rsplit_once('.').expect("dotted host type");
        runtime.add(module, &format!("{class} as _Host{}", r.name));
        return format!("\n\n{} = _Host{}\n", r.name, r.name);
    }
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
    let all: Vec<&Field> = match &r.body {
        RecordBody::Struct(fields) => fields.iter().collect(),
        RecordBody::Variants(variants) => variants
            .iter()
            .flat_map(|v| match &v.shape {
                VariantShape::Named(fs) => fs.iter().collect(),
                VariantShape::Tuple(Some(f)) => vec![f.as_ref()],
                _ => Vec::new(),
            })
            .collect(),
        RecordBody::Literals(_) => Vec::new(),
    };
    if !r.typed_dict && all.iter().any(|f| matches!(f.ty, Ty::Maybe(_))) {
        runtime.add("..types", "NOT_GIVEN");
    }
    let pydantic = r.pydantic.as_ref();
    // A mutable model's JSON reads `JsonValue`, as the values users and plugins put in it.
    let json_value = pydantic.is_some() || r.dataclass.is_some();
    if all.iter().any(|f| f.kw_only || f.hide_repr) {
        runtime.add("dataclasses", "field");
    }
    if r.dataclass.as_ref().is_some_and(|d| d.arbitrary_types) {
        runtime.add("pydantic", "ConfigDict");
    }
    for factory in all.iter().filter_map(|f| f.factory.as_deref()) {
        match pydantic {
            Some(_) => runtime.add("pydantic", "Field"),
            None => runtime.add("dataclasses", "field"),
        }
        let callable = factory.split('(').next().unwrap_or(factory);
        match callable.rsplit_once('.') {
            // A module of the package (`..utils.shortuuid`): `from .. import utils`.
            Some((m, _)) if m.starts_with("..") => runtime.add("..", &m[2..]),
            // A record of another generated module (`.usage.TokenUsage`): its class, imported.
            Some((m, name)) if m.starts_with('.') => runtime.add(m, name),
            Some((m, _)) => runtime.add_module(m),
            None => {}
        }
    }
    // `..utils.shortuuid("item_")`: a call is a lambda; the package prefix is imported above.
    let factory = |f: &str| {
        let f = match f.strip_prefix("..") {
            Some(f) => f,
            None if f.starts_with('.') => f.rsplit('.').next().unwrap_or(f),
            None => f,
        };
        if f.contains('(') { format!("lambda: {f}") } else { f.to_owned() }
    };
    // A pydantic model's discriminator constants lead its fields, as pydantic declares them.
    let constants: String = r
        .constants
        .iter()
        .map(|(name, value)| format!("    {name}: Literal[{value:?}] = {value:?}\n"))
        .collect();
    // A pydantic model resolves its annotations as it is built: they are runtime imports.
    let class = |name: &str,
                 doc: &[String],
                 fields: &[&Field],
                 nested: &str,
                 imports: &mut Imports| {
        let name = name.rsplit('.').next().unwrap_or(name);
        let mut s = match pydantic {
            Some(p) => {
                format!("\n\nclass {name}({}):\n", p.base.as_deref().unwrap_or("BaseModel"))
            }
            // Mutable and positional: plugins build it by position and read `dataclasses.asdict`.
            None if let Some(dataclass) = &r.dataclass => {
                let options = [
                    (dataclass.frozen, "frozen=True"),
                    (dataclass.kw_only, "kw_only=True"),
                    (dataclass.slots, "slots=True"),
                ]
                .into_iter()
                .filter_map(|(enabled, option)| enabled.then_some(option))
                .collect::<Vec<_>>();
                let options = if options.is_empty() {
                    String::new()
                } else {
                    format!("({})", options.join(", "))
                };
                format!("\n\n@dataclass{options}\nclass {name}:\n")
            }
            None if r.hashable => {
                format!("\n\n@dataclass(frozen=True, kw_only=True, slots=True)\nclass {name}:\n")
            }
            None => format!("\n\n@dataclass(kw_only=True, slots=True)\nclass {name}:\n"),
        };
        s.push_str(&docstring(doc, "    "));
        if r.dataclass.as_ref().is_some_and(|d| d.arbitrary_types) {
            if !doc.is_empty() {
                s.push('\n');
            }
            s.push_str("    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)\n\n");
        }
        if let Some(p) = pydantic
            && (p.frozen || p.arbitrary_types)
        {
            if !doc.is_empty() {
                s.push('\n');
            }
            let mut options = Vec::new();
            if p.frozen {
                options.push("frozen=True");
            }
            if p.arbitrary_types {
                options.push("arbitrary_types_allowed=True");
            }
            s.push_str(&format!("    model_config = ConfigDict({})\n", options.join(", ")));
        }
        if !nested.is_empty() {
            if !doc.is_empty() {
                s.push('\n');
            }
            s.push_str(nested);
        }
        let fields: Vec<&&Field> = fields.iter().filter(|f| f.modeled() && !f.private).collect();
        let after = pydantic.and_then(|p| p.constants_after.as_deref());
        let lead = if pydantic.is_some() && after.is_none() { constants.as_str() } else { "" };
        if fields.is_empty() && lead.is_empty() {
            if doc.is_empty() && nested.is_empty() && pydantic.is_none() {
                s.push_str("    pass\n");
            }
            return s;
        }
        if !doc.is_empty() || !nested.is_empty() {
            s.push('\n');
        }
        s.push_str(lead);
        for f in fields {
            let default = match (&f.default, &f.omitted) {
                _ if let Some(f) = &f.factory => match pydantic {
                    Some(_) => format!(" = Field(default_factory={})", factory(f)),
                    None => format!(" = field(default_factory={})", factory(f)),
                },
                (Some(d), _) => format!(" = {}", py_literal(d)),
                // Computed in `__post_init__` when left out: the field is never `None` after.
                (None, Some(_)) => " = None  # type: ignore[assignment]".to_owned(),
                (None, None) if matches!(f.ty, Ty::Maybe(_)) => " = NOT_GIVEN".to_owned(),
                (None, None) => String::new(),
            };
            let default = if f.kw_only || f.hide_repr {
                let mut options = Vec::new();
                if let Some(factory_name) = &f.factory {
                    options.push(format!("default_factory={}", factory(factory_name)));
                } else if let Some(value) = default.strip_prefix(" = ") {
                    options.push(format!("default={value}"));
                }
                if f.kw_only {
                    options.push("kw_only=True".into());
                }
                if f.hide_repr {
                    options.push("repr=False".into());
                }
                format!(" = field({})", options.join(", "))
            } else {
                default
            };
            let ann = match &f.host_type {
                _ if let Some(text) = &f.annotation => imports.annotation(text, &r.rust),
                Some((m, name)) => {
                    imports.add(m, name);
                    match f.ty {
                        Ty::Optional(_) => format!("{name} | None"),
                        _ => name.clone(),
                    }
                }
                None => annotation(&f.ty, module, own, json_value, imports),
            };
            s.push_str(&format!("    {}: {ann}{default}\n", f.host_name()));
            s.push_str(&docstring(&f.doc, "    "));
            if after == Some(&*f.host_name()) {
                s.push_str(&constants);
            }
        }
        s
    };
    match &r.body {
        RecordBody::Struct(fields) if r.typed_dict => {
            runtime.add("typing_extensions", "TypedDict");
            // The annotations are strings (`from __future__ import annotations`), which hide a
            // `NotRequired` from the class's runtime `__required_keys__`: a dict whose every key is
            // optional says so with `total=False` instead.
            let omittable =
                |f: &crate::decl::Field| matches!(f.ty, Ty::Maybe(_)) || f.default.is_some();
            let total = !r.constants.is_empty() || !fields.iter().all(omittable);
            let mut s = if total {
                format!("\n\nclass {}(TypedDict):\n", r.name)
            } else {
                format!("\n\nclass {}(TypedDict, total=False):\n", r.name)
            };
            s.push_str(&docstring(&r.doc, "    "));
            if !r.doc.is_empty() {
                s.push('\n');
            }
            if !r.constants.is_empty() {
                runtime.add("typing", "Literal");
                for (name, value) in &r.constants {
                    s.push_str(&format!("    {name}: Literal[{value:?}]\n"));
                }
            }
            for f in fields {
                let inner = match &f.ty {
                    _ if f.annotation.is_some() => None,
                    Ty::Maybe(inner) => Some(&**inner),
                    Ty::Optional(inner) if f.default == Some(crate::decl::Literal::None) => {
                        Some(&**inner)
                    }
                    ty if f.default.is_some() => Some(ty),
                    _ => None,
                };
                let ann = match (&f.annotation, inner) {
                    (Some(text), _) => typing.annotation(text, &r.rust),
                    (None, Some(inner)) => annotation(inner, module, own, false, typing),
                    (None, None) => annotation(&f.ty, module, own, false, typing),
                };
                let ann = if total && omittable(f) {
                    runtime.add("typing", "NotRequired");
                    format!("NotRequired[{ann}]")
                } else {
                    ann
                };
                s.push_str(&format!("    {}: {ann}\n", f.host_name()));
                s.push_str(&docstring(&f.doc, "    "));
            }
            s
        }
        RecordBody::Struct(fields) if let Some(p) = pydantic => {
            if p.base.is_none() {
                runtime.add("pydantic", "BaseModel");
            }
            if p.frozen || p.arbitrary_types {
                runtime.add("pydantic", "ConfigDict");
            }
            if !r.constants.is_empty() {
                runtime.add("typing", "Literal");
            }
            let fields = fields.iter().collect::<Vec<_>>();
            let mut s = class(&r.name, &r.doc, &fields, nested, runtime);
            if let Some(serializer) = &p.serializer {
                runtime.add("pydantic", "model_serializer");
                runtime.add("pydantic", "SerializerFunctionWrapHandler");
                s.push_str(&format!("\n    @model_serializer(mode=\"wrap\")\n    def _serialize(self, handler: SerializerFunctionWrapHandler) -> object:\n        from .. import _bindings\n\n        return _bindings.{serializer}(self, handler)\n"));
            }
            // Private attributes trail the fields; their annotations are never evaluated.
            let mut private = String::new();
            for f in fields.iter().filter(|f| f.private) {
                let ann = match &f.annotation {
                    Some(text) => typing.annotation(text, &r.rust),
                    None => annotation(&f.ty, module, own, true, typing),
                };
                private.push_str(&format!(
                    "    {}: {ann} = PrivateAttr(default_factory={})\n{}",
                    f.host_name(),
                    factory(f.factory.as_deref().expect("a private field has a factory")),
                    docstring(&f.doc, "    ")
                ));
            }
            for (name, ann) in &p.private {
                let container = ann.split('[').next().unwrap_or(ann);
                let ann = typing.annotation(ann, &r.rust);
                private.push_str(&format!(
                    "    {name}: {ann} = PrivateAttr(default_factory={container})\n"
                ));
            }
            if !private.is_empty() {
                runtime.add("pydantic", "PrivateAttr");
                s.push_str(&private);
            }
            if let Some(validator) = &p.validator {
                runtime.add("pydantic", "model_validator");
                runtime.add("typing", "Self");
                runtime.add("..", "_bindings");
                s.push_str(&format!(
                    "\n    @model_validator(mode=\"after\")\n    def {validator}(self) -> Self:\n        return _bindings.{validator}(self)\n"
                ));
            }
            if let Some(repr) = &p.repr {
                runtime.add("..", "_bindings");
                if s.ends_with(":\n") {
                    s.pop();
                }
                s.push_str(&format!(
                    "\n    def __repr__(self) -> str:\n        return _bindings.{repr}(self)\n"
                ));
            }
            s.push_str(&properties(r, runtime));
            s
        }
        RecordBody::Struct(fields) => {
            runtime.add("dataclasses", "dataclass");
            // A mutable dataclass's annotations resolve at runtime (`typing.get_type_hints`), as a
            // pydantic model's do; a frozen record's are for the type checker only.
            let imports = if r.dataclass.as_ref().is_some_and(|d| !d.deferred) {
                &mut *runtime
            } else {
                &mut *typing
            };
            let mut s = class(&r.name, &r.doc, &fields.iter().collect::<Vec<_>>(), nested, imports);
            for h in r.dataclass.iter().flat_map(|d| &d.host_fields) {
                s = s.replace("    pass\n", "");
                let imports = if r.dataclass.as_ref().is_some_and(|d| d.deferred) {
                    &mut *typing
                } else {
                    &mut *runtime
                };
                let ann = imports.annotation(&h.annotation, &r.rust);
                let default = match h.default.rsplit_once('.') {
                    Some((module, name)) if module.starts_with('.') => {
                        runtime.add(module, name);
                        name
                    }
                    _ => &h.default,
                };
                s.push_str(&format!("    {}: {ann} = {default}\n", h.name));
                s.push_str(&docstring(&h.doc.iter().cloned().collect::<Vec<_>>(), "    "));
            }
            if !r.constants.is_empty() {
                runtime.add("typing", "Literal");
                s = s.replace("    pass\n", "");
            }
            s.push_str(&constants);
            if let Some(hook) = r.dataclass.as_ref().and_then(|d| d.post_init.as_deref()) {
                s.push_str(&format!(
                    "\n    def __post_init__(self) -> None:\n        from .. import _bindings\n\n        _bindings.{hook}(self)\n"
                ));
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
            s.push_str(&properties(r, runtime));
            if r.persisted {
                let (persist, restore) = r.persistence();
                s.push_str(&format!(
                    "\n    def _to_json(self) -> str:\n        \"\"\"The saved form: restore it with ``_from_json``.\"\"\"\n        from .. import _core\n\n        return _core.{persist}(self)\n\n    @classmethod\n    def _from_json(cls, data: str) -> {name}:\n        \"\"\"The record ``_to_json`` saved.\"\"\"\n        from .. import _core\n\n        return _core.{restore}(data)\n",
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
                    VariantShape::Tuple(_) if v.host_type.is_some() => {
                        let (host_module, name) = v.host_type.as_ref().expect("declared class");
                        if host_module != "builtins" {
                            runtime.add(host_module, name);
                        }
                        Some(name.clone())
                    }
                    VariantShape::Tuple(Some(f)) => {
                        Some(annotation(&f.ty, module, own, false, runtime))
                    }
                    VariantShape::Unit => v.host_type.as_ref().map(|(host_module, name)| {
                        if host_module != "builtins" && host_module != &format!(".{module}") {
                            runtime.add(host_module, name);
                        }
                        name.clone()
                    }),
                    _ => None,
                })
                // `None` last: `None | float` is no type expression at runtime.
                .chain(
                    variants
                        .iter()
                        .any(|v| matches!(v.shape, VariantShape::Unit) && v.host_type.is_none())
                        .then(|| "None".to_owned()),
                )
                .collect();
            let members: Vec<&str> = members.iter().map(String::as_str).collect();
            let alias = match &r.discriminator {
                Some(key) => {
                    runtime.add("typing", "Annotated");
                    runtime.add("pydantic", "Field");
                    format!(
                        "{}: TypeAlias = Annotated[\n    {},\n    Field(discriminator={key:?}),\n]",
                        r.name,
                        members.join(" | ")
                    )
                }
                None => union_alias(&r.name, &members),
            };
            let mut s = format!("\n\n{alias}\n");
            s.push_str(&docstring(&r.doc, ""));
            s
        }
        RecordBody::Variants(variants) => {
            runtime.add("dataclasses", "dataclass");
            runtime.add("typing", "TypeAlias");
            let mut s = String::new();
            for v in variants {
                let fields: Vec<&Field> = match &v.shape {
                    VariantShape::Native => continue,
                    VariantShape::Unit | VariantShape::Tuple(None) => Vec::new(),
                    VariantShape::Tuple(Some(f)) => vec![f.as_ref()],
                    VariantShape::Named(fs) => fs.iter().collect(),
                };
                s.push_str(&class(&v.name, &v.doc, &fields, "", typing));
            }
            let names: Vec<&str> = variants
                .iter()
                .filter(|v| !matches!(v.shape, VariantShape::Native))
                .map(|v| v.name.as_str())
                .collect();
            s.push_str(&format!("\n\n{}\n", union_alias(&r.name, &names)));
            s.push_str(&docstring(&r.doc, ""));
            s
        }
        RecordBody::Literals(values) if r.enumeration => {
            runtime.add("enum", &r.enum_base);
            runtime.add("enum", "unique");
            let mut s = format!("\n\n@unique\nclass {}({}):\n", r.name, r.enum_base);
            if !r.doc.is_empty() {
                s.push_str(&docstring(&r.doc, "    "));
                s.push('\n');
            }
            for (rust, host, doc) in values {
                s.push_str(&format!(
                    "    {} = {host:?}\n",
                    crate::decl::resolve::snake(rust).to_uppercase()
                ));
                s.push_str(&docstring(doc, "    "));
            }
            s
        }
        RecordBody::Literals(values) => {
            runtime.add("typing", "Literal");
            runtime.add("typing", "TypeAlias");
            let values: Vec<String> = values.iter().map(|(_, h, _)| format!("{h:?}")).collect();
            let mut s = format!("\n\n{}: TypeAlias = Literal[{}]\n", r.name, values.join(", "));
            s.push_str(&docstring(&r.doc, ""));
            s
        }
    }
}

/// A record's computed views, answered by `_core`; a pydantic model's are host hooks
/// (`_bindings`), properties or methods, run on the host's own object without crossing.
fn properties(r: &RecordDecl, runtime: &mut Imports) -> String {
    let mut s = String::new();
    for member in &r.members {
        if member.is_property {
            s.push_str("\n    @property\n");
        } else {
            s.push('\n');
        }
        let params = if member.args.is_empty() {
            String::new()
        } else {
            format!(", {}", runtime.annotation(&member.args, &r.rust))
        };
        let ret = runtime.annotation(&member.ret, &r.rust);
        s.push_str(&format!("    def {}(self{params}) -> {ret}:\n", member.name));
        s.push_str(&docstring(
            &member.doc.lines().map(str::to_owned).collect::<Vec<_>>(),
            "        ",
        ));
        let arguments = if member.arguments.is_empty() {
            String::new()
        } else {
            format!(", {}", member.arguments.join(", "))
        };
        match (r.pydantic.as_ref(), r.dataclass.as_ref(), member.is_property) {
            (Some(_), _, _) => {
                runtime.add("..", "_bindings");
                s.push_str(&format!("        return _bindings.{}(self{arguments})\n", member.hook));
            }
            (_, Some(_), _) => s.push_str(&format!("        from .. import _bindings\n\n        return _bindings.{}(self{arguments})\n", member.hook)),
            _ => s.push_str(&format!(
                "        from .. import _core\n\n        return _core.{}(self)\n",
                member.hook
            )),
        }
    }
    s
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

/// `any`: JSON reads `JsonValue` (a pydantic model's public annotation), else `object`.
fn annotation(ty: &Ty, module: &str, own: &[String], any: bool, typing: &mut Imports) -> String {
    match ty {
        Ty::Bool => "bool".into(),
        Ty::Int(_) => "int".into(),
        Ty::Float | Ty::Duration | Ty::Timestamp => "float".into(),
        Ty::Str => "str".into(),
        Ty::Json | Ty::JsonObject if any => {
            typing.add("pydantic", "JsonValue");
            if matches!(ty, Ty::Json) { "JsonValue".into() } else { "dict[str, JsonValue]".into() }
        }
        Ty::Json => "object".into(),
        Ty::JsonObject => "dict[str, object]".into(),
        Ty::Optional(t) => format!("{} | None", annotation(t, module, own, any, typing)),
        Ty::Boxed(t) => annotation(t, module, own, any, typing),
        Ty::Tuple(items) => format!(
            "tuple[{}]",
            items
                .iter()
                .map(|item| annotation(item, module, own, any, typing))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Maybe(t) => {
            typing.add("..types", "NotGivenOr");
            format!("NotGivenOr[{}]", annotation(t, module, own, any, typing))
        }
        // Bytes cross as `bytes`, as class arguments and results do.
        Ty::List(t) if **t == Ty::Int(crate::decl::Int::U8) => "bytes".into(),
        Ty::List(t) => format!("list[{}]", annotation(t, module, own, any, typing)),
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => format!(
            "dict[{}, {}]",
            annotation(k, module, own, any, typing),
            annotation(t, module, own, any, typing)
        ),
        Ty::Set(t) => format!("frozenset[{}]", annotation(t, module, own, any, typing)),
        Ty::Error(_) => "BaseException".into(),
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
        Ty::Record(r) => match &r.host {
            HostForm::Record { module: m, name } if m == module && own.contains(name) => {
                name.clone()
            }
            HostForm::Record { module: m, name } => {
                typing.add(&format!(".{m}"), name);
                name.clone()
            }
            HostForm::WireStrs | HostForm::Str | HostForm::Wire => "str".into(),
            HostForm::Annotation(text)
            | HostForm::Converted(text, ..)
            | HostForm::Via { host: text, .. } => typing.annotation(text, &r.rust),
        },
        // Refused by the resolver (`check_record_field`).
        other => unreachable!("`{}` is not a record field", other.rust()),
    }
}

fn flatten_fields(fields: &[Field], decls: &Declarations) -> Result<Vec<Field>, String> {
    let mut out = Vec::new();
    for field in fields {
        if !field.conversion.flatten {
            out.push(field.clone());
            continue;
        }
        let Ty::Record(held) = &field.ty else {
            return Err("a flattened field is a record".into());
        };
        let nested = decls
            .records
            .iter()
            .find(|r| r.rust == held.rust)
            .ok_or("flattened record is missing")?;
        let RecordBody::Struct(fields) = &nested.body else {
            return Err("a flattened record is a struct".into());
        };
        out.extend(flatten_fields(fields, decls)?);
    }
    let mut names = std::collections::BTreeSet::new();
    for field in &out {
        if !names.insert(field.host_name().to_string()) {
            return Err(format!("duplicate flattened host field {}", field.host_name()));
        }
    }
    Ok(out)
}
