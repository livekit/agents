//! The Python view, one file per module under `livekit/agents/_generated/`. Output must match the
//! reference as source text (the gate compares annotations textually): no reordering, no
//! `Optional[X]`. Field docstrings follow the field so pydantic puts them in the JSON schema.

use crate::ir::*;

pub struct Out {
    pub path: String,
    pub text: String,
}

/// In the order the modules were first named.
pub fn emit(decls: &[Decl]) -> Result<Vec<Out>, String> {
    let mut modules: Vec<String> = Vec::new();
    for d in decls {
        if let Some(t) = d.target(Host::Python)
            && !modules.contains(&t.module) {
                modules.push(t.module.clone());
            }
    }
    let mut out: Vec<Out> = modules.iter().map(|m| module(decls, m)).collect::<Result<_, _>>()?;
    for d in decls {
        if let Some(m) = &d.dispatch {
            if modules.contains(m) || out.iter().any(|o| o.path == format!("{m}.py")) {
                return Err(format!("{}: dispatch module `{m}` is already a generated module", d.rust_name));
            }
            out.push(dispatch(decls, d, m)?);
        }
    }
    Ok(out)
}

/// `#[python(dispatch = "..")]`: `PUBLIC_EVENTS` (wire tag -> class and public name, which may
/// differ: `transcription_timeout`) and `IGNORED_EVENTS`; other members are handled by hand.
/// `EVENT_KINDS` is every kind the stream carries, so the host can tell an unknown one.
fn dispatch(decls: &[Decl], d: &Decl, module: &str) -> Result<Out, String> {
    let Body::Union(members) = &d.body else {
        return Err(format!("{}: #[python(dispatch)] is for a union", d.rust_name));
    };
    // The binding's rule (`decl_tag`): the emit twin's tag, else the declaration's other one.
    let wire_of = |at: &str, name: &str, twin: Option<&String>| -> Result<(String, &Decl), String> {
        let md = decls.iter().find(|x| x.rust_name == name).ok_or_else(|| format!("{at}: `{name}` is not declared"))?;
        let t = md
            .targets_for(Host::Binding)
            .filter(|t| t.kind == Kind::Emit)
            .find(|t| twin.is_none_or(|n| &t.name == n))
            .ok_or_else(|| format!("{at}: `{name}` has no binding emit twin to read its wire kind from"))?;
        let wire = t
            .tag
            .as_ref()
            .or_else(|| md.targets.iter().find_map(|x| x.tag.as_ref()))
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("{at}: `{name}` carries no tag"))?;
        Ok((wire, md))
    };
    let mut events: Vec<(String, String, String, String)> = Vec::new();
    let mut ignored: Vec<String> = Vec::new();
    let mut kinds: Vec<String> = Vec::new();
    for m in members {
        let at = format!("{}::{}", d.rust_name, m.rust_name);
        let Ty::Declared(name) = &m.ty else {
            return Err(format!("{at}: a host-stream member is a declared record"));
        };
        let declared = decls.iter().any(|x| &x.rust_name == name);
        let (wire, md) = match (&m.kind, declared) {
            (None, _) => {
                let (wire, md) = wire_of(&at, name, m.twin.as_ref())?;
                (wire, Some(md))
            }
            (Some(kind), false) => (kind.clone(), None),
            (Some(_), true) => {
                return Err(format!("{at}: `{name}` is declared, so its tag is its kind; drop #[python(kind)]"));
            }
        };
        if !kinds.contains(&wire) {
            kinds.push(wire.clone());
        }
        let Some(route) = m.dispatch else { continue };
        let Some(md) = md else {
            return Err(format!("{at}: a routed member is a declared record"));
        };
        match route {
            Dispatch::Ignored => ignored.push(wire),
            Dispatch::Event => {
                let py = md
                    .targets_for(Host::Python)
                    .find(|t| t.kind == Kind::Pydantic)
                    .ok_or_else(|| format!("{at}: an event member needs a pydantic Python target"))?;
                let public = py.tag.as_ref().map(|(_, v)| v.clone()).ok_or_else(|| format!("{at}: `{}` has no discriminant to emit it under", py.name))?;
                events.push((wire, py.name.clone(), py.module.clone(), public));
            }
        }
    }
    for name in &d.dispatch_also {
        let (wire, _) = wire_of(&format!("{}: dispatch_also", d.rust_name), name, None)?;
        if !kinds.contains(&wire) {
            kinds.push(wire);
        }
    }
    let mut b = String::new();
    header(&mut b, &[d]);
    b.push_str("from typing import Any\n\n");
    let mut by_module: Vec<(String, Vec<String>)> = Vec::new();
    for (_, class, m, _) in &events {
        match by_module.iter_mut().find(|(x, _)| x == m) {
            Some((_, names)) => names.push(class.clone()),
            None => by_module.push((m.clone(), vec![class.clone()])),
        }
    }
    by_module.sort();
    for (m, mut names) in by_module {
        names.sort();
        names.dedup();
        let flat = format!("from .{m} import {}\n", names.join(", "));
        if flat.len() <= 100 {
            b.push_str(&flat);
        } else {
            b.push_str(&format!("from .{m} import (\n"));
            for n in names {
                b.push_str(&format!("    {n},\n"));
            }
            b.push_str(")\n");
        }
    }
    b.push_str(&format!(
        "\n\n# `{}` members the host emits as a public session event and nothing more:\n\
         # {{wire kind: (event class, event name)}}. The payload's keys are the class's fields.\n\
         PUBLIC_EVENTS: dict[str, tuple[type[Any], str]] = {{\n",
        d.rust_name
    ));
    for (wire, class, _, public) in &events {
        b.push_str(&format!("    \"{wire}\": ({class}, \"{public}\"),\n"));
    }
    b.push_str(&format!("}}\n\n# `{}` members the host receives and does not act on.\nIGNORED_EVENTS: frozenset[str] = frozenset(\n    {{\n", d.rust_name));
    for wire in &ignored {
        b.push_str(&format!("        \"{wire}\",\n"));
    }
    b.push_str("    }\n)\n");
    b.push_str(&format!(
        "\n# Every kind the stream carries: each `{}` member's, and the records it carries beside them.\n\
         EVENT_KINDS: frozenset[str] = frozenset(\n    {{\n",
        d.rust_name
    ));
    for wire in &kinds {
        b.push_str(&format!("        \"{wire}\",\n"));
    }
    b.push_str("    }\n)\n");
    Ok(Out { path: format!("{module}.py"), text: b })
}

fn module(decls: &[Decl], name: &str) -> Result<Out, String> {
    let mine: Vec<&Decl> = decls
        .iter()
        .filter(|d| {
            d.target(Host::Python)
                .is_some_and(|t| t.module == name && !t.inline && t.nested_in.is_none())
        })
        .collect();
    // Inline unions are expanded by `render`, never emitted: they have no host name.
    let visible: Vec<&Decl> =
        decls.iter().filter(|d| d.target(Host::Python).is_some_and(|t| t.module == name)).collect();

    let mut b = String::new();
    header(&mut b, &mine);
    imports(&mut b, decls, &visible)?;

    // Two blank lines between top-level statements (PEP 8), exactly two under the imports when
    // a class follows; one when an alias (a `Literal` or a union) does, as isort wants it.
    let alias_first = mine.first().is_some_and(|d| {
        matches!(d.body, Body::Union(_))
            || d.target(Host::Python).is_some_and(|t| t.kind == Kind::Literal)
    });
    let under_imports = if alias_first { "\n\n" } else { "\n\n\n" };
    while !b.ends_with(under_imports) {
        b.push('\n');
    }

    let mut blocks: Vec<String> = Vec::new();
    let mut section: Option<String> = None;
    for d in &mine {
        let mut block = String::new();
        if d.section != section {
            section = d.section.clone();
            if let Some(s) = &section {
                block.push_str(&format!(
                    "# --- {s} {}\n",
                    "-".repeat(96usize.saturating_sub(s.len() + 7))
                ));
                block.push('\n');
            }
        }
        block.push_str(&class(decls, d)?);
        blocks.push(block);
    }
    b.push_str(&blocks.join("\n\n"));
    b.push_str(&modules(&visible));
    Ok(Out { path: format!("{name}.py"), text: b })
}

/// The `__module__` fixups, after the definitions: in the class body pydantic would resolve
/// annotations against the re-export module ("not fully defined" on first use).
fn modules(visible: &[&Decl]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for d in visible {
        let Some(t) = d.target(Host::Python) else { continue };
        let Some(path) = &t.path else { continue };
        // Only a class has a settable `__module__`; assigning one on a union or `Literal` raises.
        if t.inline || matches!(t.kind, Kind::Union | Kind::Literal) {
            continue;
        }
        let name = match &t.nested_in {
            Some(owner) => {
                let outer = visible
                    .iter()
                    .find(|o| &o.rust_name == owner)
                    .and_then(|o| o.target(Host::Python))
                    .map_or_else(|| owner.clone(), |ot| ot.name.clone());
                format!("{outer}.{}", t.name)
            }
            None => t.name.clone(),
        };
        lines.push(format!("{name}.__module__ = \"{path}\""));
    }
    if lines.is_empty() {
        return String::new();
    }
    format!(
        "\n\n# --- the frozen public paths {}\n#\n\
         # Assigned here rather than in each class body: pydantic resolves annotations against\n\
         # `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it\n\
         # looking for `Literal` and every foreign type in the re-export module, which imports\n\
         # none of them. What this fixes is `repr`, `pickle` and the equivalence gate all\n\
         # reporting `_generated.*` for a name whose import path never moved.\n{}\n",
        "-".repeat(65),
        lines.join("\n")
    )
}


/// Names the distinct Rust sources, first-seen order, so the file says where to edit.
fn header(b: &mut String, mine: &[&Decl]) {
    let mut sources: Vec<&str> = Vec::new();
    for d in mine {
        if !sources.contains(&d.source.as_str()) {
            sources.push(&d.source);
        }
    }
    b.push_str("# GENERATED FILE -- DO NOT EDIT.\n#\n");
    b.push_str("# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:\n");
    for s in &sources {
        b.push_str(&format!("#     {s}\n"));
    }
    b.push_str(
        "#\n# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a\n\
         # non-empty diff, so a hand edit here is reverted rather than merged.\n\n",
    );
    b.push_str("from __future__ import annotations\n\n");
}


/// Deterministic, grouped imports. Relative ones come from `#[python(from = "..")]`; a foreign
/// type without one is assumed builtin.
fn imports(b: &mut String, all: &[Decl], mine: &[&Decl]) -> Result<(), String> {
    let mut stdlib: Vec<(&'static str, Vec<String>)> = Vec::new();
    let mut third: Vec<(String, Vec<String>)> = Vec::new();
    let mut rel: Vec<(String, Vec<String>)> = Vec::new();
    let mut deferred: Vec<(String, Vec<String>)> = Vec::new();
    let mut plain: Vec<&str> = Vec::new();

    /// [`needs`] for a bucket whose module names are not `'static`.
    fn needs_s(bucket: &mut Vec<(String, Vec<String>)>, module: &str, name: &str) {
        match bucket.iter_mut().find(|(m, _)| m == module) {
            Some((_, names)) if names.iter().any(|n| n == name) => {}
            Some((_, names)) => names.push(name.to_owned()),
            None => bucket.push((module.to_owned(), vec![name.to_owned()])),
        }
    }

    fn needs(bucket: &mut Vec<(&'static str, Vec<String>)>, module: &'static str, name: &str) {
        match bucket.iter_mut().find(|(m, _)| *m == module) {
            Some((_, names)) if names.iter().any(|n| n == name) => {}
            Some((_, names)) => names.push(name.to_owned()),
            None => bucket.push((module, vec![name.to_owned()])),
        }
    }

    for d in mine {
        let t = d.target(Host::Python).expect("filtered on this");
        match t.kind {
            Kind::Pydantic => {
                needs_s(&mut third, "pydantic", "BaseModel");
                if t.config.is_some() {
                    needs_s(&mut third, "pydantic", "ConfigDict");
                }
                if t.validator.is_some() {
                    needs_s(&mut third, "pydantic", "model_validator");
                    needs_s(&mut third, "typing_extensions", "Self");
                }
            }
            Kind::Dataclass => needs(&mut stdlib, "dataclasses", "dataclass"),
            Kind::TypedDict => match t.typed_dict_from.as_deref() {
                None | Some("typing_extensions") => {
                    needs_s(&mut third, "typing_extensions", "TypedDict")
                }
                Some("typing") => needs(&mut stdlib, "typing", "TypedDict"),
                Some(other) => {
                    return Err(format!(
                        "{}: typed_dict_from = \"{other}\" -- write \"typing\" or \"typing_extensions\"",
                        d.rust_name
                    ));
                }
            },
            Kind::Enum => {
                needs(&mut stdlib, "enum", "Enum");
                needs(&mut stdlib, "enum", "unique");
            }
            Kind::Plain => {}
            Kind::Literal => needs(&mut stdlib, "typing", "Literal"),
            Kind::Union => {}
            Kind::Extract | Kind::Emit | Kind::Class => {
                return Err(format!("{}: `{}` is a binding kind", d.rust_name, kind_word(t.kind)));
            }
            Kind::Interface => {
                return Err(format!(
                    "{}: `interface` is a Node kind, not a Python one",
                    d.rust_name
                ));
            }
        }
        // Only a discriminated union needs pydantic; a plain one is a bare PEP-604 expression.
        if let Body::Union(_) = &d.body
            && t.discriminator.is_some()
        {
            needs(&mut stdlib, "typing", "Annotated");
            needs_s(&mut third, "pydantic", "Field");
        }
        // An inline-union field expands to `Annotated[.., Field(..)]` here.
        for f in d.fields().iter().filter(|f| !f.py_skip) {
            if let Ty::Declared(n) = &f.ty {
                let inline_union = all.iter().any(|o| {
                    &o.rust_name == n
                        && matches!(o.body, Body::Union(_))
                        && o.target(Host::Python).is_some_and(|t| t.inline && t.discriminator.is_some())
                });
                if inline_union {
                    needs(&mut stdlib, "typing", "Annotated");
                    needs_s(&mut third, "pydantic", "Field");
                }
            }
        }
        if t.tag.is_some() {
            needs(&mut stdlib, "typing", "Literal");
        }
        if t.host_fields.iter().any(|h| h.deprecated_msg.is_some()) {
            needs_s(&mut third, "pydantic", "Field");
        }
        if t.host_fields.iter().any(|h| h.private) {
            needs_s(&mut third, "pydantic", "PrivateAttr");
        }
        for f in d.fields().iter().filter(|f| !f.py_skip) {
            if f.literal.is_some() {
                needs(&mut stdlib, "typing", "Literal");
            }
            // A dataclass needs `dataclasses.field(default_factory=..)`: pydantic's `Field` would
            // be a shared `FieldInfo` default.
            if f.default_factory.is_some() && t.kind == Kind::Dataclass {
                needs(&mut stdlib, "dataclasses", "field");
            } else if f.default_factory.is_some() || f.exclude {
                needs_s(&mut third, "pydantic", "Field");
            }
            if f.private {
                needs_s(&mut third, "pydantic", "PrivateAttr");
            }
            if f.serializer.is_some() {
                needs_s(&mut third, "pydantic", "field_serializer");
            }
            if f.deprecated_msg.is_some() {
                needs_s(&mut third, "pydantic", "Field");
            }
            if let Some(factory) = &f.default_factory {
                // A dotted name (`time.time`) imports its module; an expression such as
                // `lambda: utils.shortuuid(..)` imports nothing.
                let dotted_name =
                    factory.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
                if let Some((head, _)) = factory.split_once('.').filter(|_| dotted_name)
                    && !plain.contains(&head) {
                        plain.push(match head {
                            "time" => "time",
                            "datetime" => "datetime",
                            other => {
                                return Err(format!(
                                    "{}: default_factory `{factory}` names module `{other}`, which \
                                     the emitter has no import for -- add it, or use a callable \
                                     imported with #[python(from = ..)]",
                                    d.rust_name
                                ));
                            }
                        });
                    }
            }
            foreign_imports(&f.ty, &mut rel, &mut deferred, &mut stdlib, &mut third);
            // `import = ".."` is collected for every field, not only foreign annotations: a
            // default factory can name a module too.
            for (groups, bucket, type_only) in
                [(&f.imports, &mut rel, false), (&f.type_imports, &mut deferred, true)]
            {
                for group in groups {
                    let Some((module, names)) = group.split_once(':') else { continue };
                    // A `type_import` is always under `TYPE_CHECKING`: an absolute one may name a
                    // package the host does not depend on.
                    let target: &mut Vec<(String, Vec<String>)> = if type_only || module.starts_with('.') {
                        bucket
                    } else if STDLIB.contains(&module) {
                        continue; // already handled by the stdlib routing in `foreign_imports`
                    } else {
                        &mut third
                    };
                    for name in names.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        match target.iter_mut().find(|(m, _)| m == module) {
                            Some((_, have)) if have.iter().any(|n| n == name) => {}
                            Some((_, have)) => have.push(name.to_owned()),
                            None => target.push((module.to_owned(), vec![name.to_owned()])),
                        }
                    }
                }
            }
            if f.deprecated {
                match rel.iter_mut().find(|(m, _)| m == "..log") {
                    Some((_, names)) if names.iter().any(|n| n == "logger") => {}
                    Some((_, names)) => names.push("logger".to_owned()),
                    None => rel.push(("..log".to_owned(), vec!["logger".to_owned()])),
                }
                needs(&mut stdlib, "typing", "Any");
            }
        }
        // A union member may be foreign.
        if let Body::Union(members) = &d.body {
            for m in members {
                foreign_imports(&m.ty, &mut rel, &mut deferred, &mut stdlib, &mut third);
            }
        }
    }
    for d in mine {
        for f in d.fields().iter().filter(|f| !f.py_skip) {
            for dep in declared_deps(&f.ty) {
                let Some(owner) = all.iter().find(|o| o.rust_name == dep) else { continue };
                let Some(ot) = owner.target(Host::Python) else { continue };
                let mt = d.target(Host::Python).expect("filtered on this");
                if ot.module != mt.module {
                    let m = format!(".{}", ot.module);
                    match rel.iter_mut().find(|(x, _)| *x == m) {
                        Some((_, names)) if names.contains(&ot.name) => {}
                        Some((_, names)) => names.push(ot.name.clone()),
                        None => rel.push((m, vec![ot.name.clone()])),
                    }
                }
            }
        }
    }

    if !deferred.is_empty() {
        needs(&mut stdlib, "typing", "TYPE_CHECKING");
    }
    // `_bindings` holds the residue that cannot be generated (serializers, a validator).
    let hooks = mine.iter().any(|d| {
        d.fields().iter().any(|f| f.serializer.is_some())
            || d.target(Host::Python).is_some_and(|t| {
                t.validator.is_some()
                    || t.repr.is_some()
                    || t.post_init.is_some()
                    || !t.members.is_empty()
            })
    });
    if hooks {
        match rel.iter_mut().find(|(m, _)| m == "..") {
            Some((_, names)) if names.iter().any(|n| n == "_bindings") => {}
            Some((_, names)) => names.push("_bindings".to_owned()),
            None => rel.push(("..".to_owned(), vec!["_bindings".to_owned()])),
        }
    }

    let mut wrote = false;
    for m in &plain {
        b.push_str(&format!("import {m}\n"));
        wrote = true;
    }
    for group in [&stdlib] {
        let mut g: Vec<&(&str, Vec<String>)> = group.iter().collect();
        g.sort_by_key(|(m, _)| *m);
        for (m, names) in g {
            from_import(b, m, names);
            wrote = true;
        }
    }
    if wrote {
        b.push('\n');
        wrote = false;
    }
    {
        let mut g: Vec<&(String, Vec<String>)> = third.iter().collect();
        g.sort_by_key(|(m, _)| m.clone());
        for (m, names) in g {
            from_import(b, m, names);
            wrote = true;
        }
        if wrote {
            b.push('\n');
            wrote = false;
        }
    }
    // Relative imports deepest-first, as elsewhere in the package.
    let mut r: Vec<&(String, Vec<String>)> = rel.iter().collect();
    r.sort_by(|a, b| {
        let depth = |s: &str| s.chars().take_while(|c| *c == '.').count();
        depth(&b.0).cmp(&depth(&a.0)).then(a.0.cmp(&b.0))
    });
    for (m, names) in r {
        from_import(b, m, names);
        wrote = true;
    }
    if wrote {
        b.push('\n');
    }
    if !deferred.is_empty() {
        // Runtime imports here would be a cycle; `from __future__ import annotations` defers them.
        b.push_str("if TYPE_CHECKING:\n");
        deferred.sort_by(|a, b| a.0.cmp(&b.0));
        for (m, names) in &deferred {
            let mut line = String::new();
            from_import(&mut line, m, names);
            for l in line.lines() {
                b.push_str(&format!("    {l}\n"));
            }
        }
        b.push('\n');
    }
    Ok(())
}

/// One `from X import a, b`, wrapped at 100 columns, constants first then alphabetical (the
/// package's hand-written style).
fn from_import(b: &mut String, module: &str, names: &[String]) {
    let mut n = names.to_vec();
    n.sort_by(|a, z| {
        let constant = |s: &String| !s.chars().any(|c| c.is_lowercase());
        constant(z).cmp(&constant(a)).then(a.cmp(z))
    });
    let flat = format!("from {module} import {}\n", n.join(", "));
    if flat.len() <= 101 {
        b.push_str(&flat);
        return;
    }
    b.push_str(&format!("from {module} import (\n"));
    for name in &n {
        b.push_str(&format!("    {name},\n"));
    }
    b.push_str(")\n");
}

/// Absolute modules that belong in the stdlib import block; unlisted ones land in third-party.
const STDLIB: &[&str] =
    &["abc", "collections.abc", "dataclasses", "datetime", "enum", "pathlib", "typing", "uuid"];

fn foreign_imports(
    t: &Ty,
    rel: &mut Vec<(String, Vec<String>)>,
    deferred: &mut Vec<(String, Vec<String>)>,
    stdlib: &mut Vec<(&'static str, Vec<String>)>,
    third: &mut Vec<(String, Vec<String>)>,
) {
    let any = |stdlib: &mut Vec<(&'static str, Vec<String>)>| match stdlib
        .iter_mut()
        .find(|(m, _)| *m == "typing")
    {
        Some((_, n)) if n.iter().any(|x| x == "Any") => {}
        Some((_, n)) => n.push("Any".to_owned()),
        None => stdlib.push(("typing", vec!["Any".to_owned()])),
    };
    match t {
        Ty::Any => any(stdlib),
        Ty::Option(i) | Ty::List(i) => foreign_imports(i, rel, deferred, stdlib, third),
        Ty::Map(k, v) => {
            foreign_imports(k, rel, deferred, stdlib, third);
            foreign_imports(v, rel, deferred, stdlib, third);
        }
        Ty::Tuple(items) => {
            items.iter().for_each(|i| foreign_imports(i, rel, deferred, stdlib, third))
        }
        Ty::Foreign(f) => {
            if f.python.split('|').any(|p| p.trim() == "Any") {
                any(stdlib);
            }
            // `typing.Literal` inside a foreign annotation: `import = ..` can only write relative
            // imports, so it is sniffed from the rendered text like `Any`.
            if f.python.contains("Literal[") {
                match stdlib.iter_mut().find(|(m, _)| *m == "typing") {
                    Some((_, n)) if n.iter().any(|x| x == "Literal") => {}
                    Some((_, n)) => n.push("Literal".to_owned()),
                    None => stdlib.push(("typing", vec!["Literal".to_owned()])),
                }
            }
            for (groups, bucket, type_only) in
                [(&f.python_from, &mut *rel, false), (&f.python_type_from, &mut *deferred, true)]
            {
                for group in groups {
                    let Some((module, names)) = group.split_once(':') else { continue };
                    // A non-dotted `import = ".."` module is absolute: route it by shape, not
                    // into the relative bucket (which would duplicate `typing_extensions`).
                    let bucket: &mut Vec<(String, Vec<String>)> = if type_only || module.starts_with('.') {
                        bucket
                    } else if let Some(std) = STDLIB.iter().find(|m| **m == module) {
                        // Stdlib joins its block (sharing `typing`'s bucket); unlisted modules are
                        // third-party. There is no build-time way to ask Python.
                        for name in names.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                            match stdlib.iter_mut().find(|(m, _)| m == std) {
                                Some((_, n)) if n.iter().any(|x| x == name) => {}
                                Some((_, n)) => n.push(name.to_owned()),
                                None => stdlib.push((std, vec![name.to_owned()])),
                            }
                        }
                        continue;
                    } else {
                        third
                    };
                    for name in names.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        match bucket.iter_mut().find(|(m, _)| m == module) {
                            Some((_, have)) if have.iter().any(|n| n == name) => {}
                            Some((_, have)) => have.push(name.to_owned()),
                            None => bucket.push((module.to_owned(), vec![name.to_owned()])),
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

fn declared_deps(t: &Ty) -> Vec<String> {
    match t {
        Ty::Declared(n) => vec![n.clone()],
        Ty::Option(i) | Ty::List(i) => declared_deps(i),
        Ty::Map(k, v) => [declared_deps(k), declared_deps(v)].concat(),
        Ty::Tuple(items) => items.iter().flat_map(declared_deps).collect(),
        _ => Vec::new(),
    }
}


fn class(all: &[Decl], d: &Decl) -> Result<String, String> {
    let t = d.target(Host::Python).expect("filtered on this");
    let mut b = String::new();

    if let Body::Union(members) = &d.body {
        // A type alias, not a class: pydantic reads each member's own `Literal` discriminant.
        b.push_str(&format!(
            "{} = {}\n",
            t.name,
            union_expr(all, members, "    ", t.discriminator.as_deref(), t.name.len() + 3)
        ));
        return Ok(b);
    }

    if let (Kind::Literal, Body::Unit(variants)) = (t.kind, &d.body) {
        // An alias, not a class: `UserState` has no `.SPEAKING` member, unlike `CloseReason`.
        let values: Vec<String> = variants.iter().map(|v| format!("\"{}\"", v.value)).collect();
        let flat = format!("{} = Literal[{}]\n", t.name, values.join(", "));
        // Wrapped as ruff formats it: the values on one indented line when they fit, else one
        // per line.
        if flat.len() <= 101 {
            b.push_str(&flat);
        } else if values.join(", ").len() + 4 <= 100 {
            b.push_str(&format!("{} = Literal[\n    {}\n]\n", t.name, values.join(", ")));
        } else {
            b.push_str(&format!("{} = Literal[\n", t.name));
            for v in &values {
                b.push_str(&format!("    {v},\n"));
            }
            b.push_str("]\n");
        }
        if !d.doc.is_empty() {
            docstring(&mut b, &d.doc, 0);
        }
        return Ok(b);
    }

    if let Body::Unit(variants) = &d.body {
        // `(str, Enum)`, never a bare `Enum` or `#[pyclass]` enum, so `== "job_shutdown"` holds.
        b.push_str("@unique\n");
        b.push_str(&format!("class {}(str, Enum):\n", t.name));
        if docstring(&mut b, &d.doc, 4) {
            b.push('\n');
        }
        for v in variants {
            // The member name is the wire value upper-cased, matching every reference enum.
            let member = v.value.to_uppercase().replace(['-', '.', ' '], "_");
            b.push_str(&format!("    {member} = \"{}\"\n", v.value));
            docstring(&mut b, &v.doc, 4);
        }
        return Ok(b);
    }

    let base = match t.kind {
        // `_BaseMetrics` is already a `BaseModel`; naming both would change the compared `bases`.
        Kind::Pydantic if t.base.is_some() => "",
        Kind::Pydantic => "(BaseModel)",
        Kind::TypedDict if !t.total => "(TypedDict, total=False)",
        Kind::TypedDict => "(TypedDict)",
        Kind::Dataclass | Kind::Plain => "",
        Kind::Enum => "(str, Enum)",
        Kind::Union => {
            return Err(format!("{}: `union` needs a Rust enum with typed variants", d.rust_name));
        }
        Kind::Extract | Kind::Emit | Kind::Class => {
            return Err(format!(
                "{}: `{}` is a binding kind",
                d.rust_name,
                kind_word(d.targets[0].kind)
            ));
        }
        Kind::Literal => {
            return Err(format!(
                "{}: `literal` needs a fieldless Rust enum, not a struct",
                d.rust_name
            ));
        }
        Kind::Interface => return Err(format!("{}: `interface` is a Node kind", d.rust_name)),
    };
    if t.kind == Kind::Dataclass {
        b.push_str("@dataclass\n");
    }
    let base = match &t.base {
        Some(name) => format!("({name})"),
        None => base.to_owned(),
    };
    b.push_str(&format!("class {}{base}:\n", t.name));
    let had_doc = docstring(&mut b, &d.doc, 4);
    let mut body = false;
    // black's blank line under a class docstring.
    if had_doc && (t.config.is_some() || !d.fields().is_empty()) {
        b.push('\n');
    }

    if let Some(cfg) = &t.config {
        b.push_str(&format!("    model_config = ConfigDict({cfg})\n"));
        body = true;
    }

    // Nested classes first: fields below reference them, and the body runs top to bottom.
    for n in all {
        let Some(nt) = n.target(Host::Python) else { continue };
        if nt.nested_in.as_deref() != Some(d.rust_name.as_str()) {
            continue;
        }
        if body || had_doc {
            b.push('\n');
        }
        for line in class(all, n)?.lines() {
            if line.is_empty() {
                b.push('\n');
            } else {
                b.push_str(&format!("    {line}\n"));
            }
        }
        body = true;
    }
    if body
        && !d.fields().is_empty()
        && all.iter().any(|n| {
            n.target(Host::Python).and_then(|t| t.nested_in.as_deref())
                == Some(d.rust_name.as_str())
        })
    {
        b.push('\n');
    }

    // The discriminant leads the fields unless `tag_after` names one to follow.
    let tag_line =
        |name: &str, value: &str| format!("    {name}: Literal[\"{value}\"] = \"{value}\"\n");
    if let Some((name, value)) = &t.tag
        && t.tag_after.is_none()
    {
        b.push_str(&tag_line(name, value));
        body = true;
    }

    if let Some(after) = &t.tag_after
        && !d.fields().iter().any(|f| !f.py_skip && &f.name == after)
    {
        return Err(format!(
            "{}: tag_after = \"{after}\" names no emitted field. A tag placed after a field that \
             is not there would silently lead the class instead, which is the ordering this key \
             exists to control.",
            d.rust_name
        ));
    }

    // Declared fields, then host fields, unless `order` interleaves them.
    for slot in ordered(d, t)? {
        match slot {
            Slot::Field(f) => {
                b.push_str(&format!("    {}\n", field(all, d, f)?));
                docstring(&mut b, &f.doc, 4);
                if let (Some((name, value)), Some(after)) = (&t.tag, &t.tag_after)
                    && &f.name == after
                {
                    b.push_str(&tag_line(name, value));
                }
            }
            Slot::Host(h) => host_field(&mut b, h),
        }
        body = true;
    }

    for f in d.fields().iter().filter(|f| !f.py_skip) {
        let Some(hook) = &f.serializer else { continue };
        b.push('\n');
        b.push_str(&format!("    @field_serializer(\"{}\")\n", f.name));
        b.push_str(&format!("    def _serialize_{}(self, value: Any) -> Any:\n", f.name));
        b.push_str(&format!("        return _bindings.{hook}(value)\n"));
    }

    for m in &t.members {
        b.push('\n');
        if m.is_property {
            b.push_str("    @property\n");
        }
        let args = if m.args.is_empty() { String::new() } else { format!(", {}", m.args) };
        b.push_str(&format!("    def {}(self{args}) -> {}:\n", m.name, m.ret));
        if !m.doc.is_empty() {
            // Via `docstring`, which indents continuation lines of a multi-line doc.
            let lines: Vec<String> = m.doc.split('\n').map(str::to_owned).collect();
            docstring(&mut b, &lines, 8);
        }
        b.push_str(&format!("        return _bindings.{}(self)\n", m.hook));
        if let Some(setter) = &m.setter {
            b.push('\n');
            b.push_str(&format!("    @{}.setter\n", m.name));
            b.push_str(&format!("    def {}(self, value: {}) -> None:\n", m.name, m.ret));
            b.push_str(&format!("        _bindings.{setter}(self, value)\n"));
        }
        body = true;
    }

    if let Some(hook) = &t.post_init {
        if body || had_doc {
            b.push('\n');
        }
        b.push_str("    def __post_init__(self) -> None:\n");
        b.push_str(&format!("        _bindings.{hook}(self)\n"));
        body = true;
    }

    if let Some(hook) = &t.repr {
        if body || had_doc {
            b.push('\n');
        }
        b.push_str("    def __repr__(self) -> str:\n");
        b.push_str(&format!("        return _bindings.{hook}(self)\n"));
        body = true;
    }

    let deprecated: Vec<&str> =
        d.fields().iter().filter(|f| f.deprecated).map(|f| f.name.as_str()).collect();
    if !deprecated.is_empty() {
        let names = deprecated.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", ");
        b.push('\n');
        b.push_str("    def __getattribute__(self, name: str) -> Any:\n");
        b.push_str(&format!("        if name in [{names}]:\n"));
        b.push_str("            logger.warning(\n");
        b.push_str(&format!(
            "                f\"{}.{{name}} is deprecated, automatic resume is now supported\"\n",
            t.name
        ));
        b.push_str("            )\n");
        b.push_str("        return super().__getattribute__(name)\n");
        body = true;
    }

    if let Some(v) = &t.validator {
        b.push('\n');
        b.push_str("    @model_validator(mode=\"after\")\n");
        b.push_str(&format!("    def {v}(self) -> Self:\n"));
        b.push_str(&format!("        return _bindings.{v}(self)\n"));
        body = true;
    }

    if !body && !had_doc {
        b.push_str("    pass\n");
    }
    Ok(b)
}

enum Slot<'a> {
    Field(&'a Field),
    Host(&'a HostField),
}

impl Slot<'_> {
    fn name(&self) -> &str {
        match self {
            Slot::Field(f) => &f.name,
            Slot::Host(h) => &h.name,
        }
    }
}

/// `#[python(order = ..)]` when written (it must name every emitted field exactly once), else
/// declared fields in Rust order followed by `host_field`s.
fn ordered<'a>(d: &'a Decl, t: &'a Target) -> Result<Vec<Slot<'a>>, String> {
    let emitted: Vec<Slot<'a>> = d
        .fields()
        .iter()
        .filter(|f| !f.py_skip)
        .map(Slot::Field)
        .chain(t.host_fields.iter().map(Slot::Host))
        .collect();
    if t.order.is_empty() {
        return Ok(emitted);
    }
    let mut out: Vec<Slot<'a>> = Vec::new();
    for name in &t.order {
        let Some(slot) = emitted.iter().find(|s| s.name() == name) else {
            return Err(format!("{}: order names `{name}`, which is not an emitted field", d.rust_name));
        };
        if out.iter().any(|o| o.name() == name) {
            return Err(format!("{}: order names `{name}` twice", d.rust_name));
        }
        out.push(match slot {
            Slot::Field(f) => Slot::Field(f),
            Slot::Host(h) => Slot::Host(h),
        });
    }
    if let Some(missing) = emitted.iter().find(|s| !t.order.iter().any(|n| n == s.name())) {
        return Err(format!(
            "{}: order leaves out `{}` -- it must name every emitted field once",
            d.rust_name,
            missing.name()
        ));
    }
    Ok(out)
}

fn host_field(b: &mut String, h: &HostField) {
    if h.private {
        let arg = match (&h.default_factory, &h.default) {
            (Some(factory), _) => format!("default_factory={factory}"),
            (None, Some(d)) => format!("default={d}"),
            (None, None) => "default=None".to_owned(),
        };
        b.push_str(&format!("    {}: {} = PrivateAttr({arg})\n", h.name, h.ty));
    } else {
        let value = match (&h.default, &h.deprecated_msg) {
            (d, Some(msg)) => {
                let head = d.clone().unwrap_or_else(|| "...".to_owned());
                format!(" = Field(default={head}, deprecated={msg:?})")
            }
            (Some(d), None) => format!(" = {d}"),
            (None, None) => String::new(),
        };
        b.push_str(&format!("    {}: {}{value}\n", h.name, h.ty));
    }
    if !h.doc.is_empty() {
        let lines: Vec<String> = h.doc.split('\n').map(str::to_owned).collect();
        docstring(b, &lines, 4);
    }
}

fn field(all: &[Decl], d: &Decl, f: &Field) -> Result<String, String> {
    if let Some(lit) = &f.literal {
        return Ok(format!("{}: Literal[\"{lit}\"] = \"{lit}\"", f.name));
    }
    // In a `total=False` `TypedDict` an outer `Option` is the key's absence; a key that may be
    // `None` says so with `#[python(ty = "X | None")]`.
    let ty = match (&f.ty, d.target(Host::Python)) {
        (Ty::Option(inner), Some(t)) if t.kind == Kind::TypedDict && !t.total => render(all, inner),
        _ => render(all, &f.ty),
    };
    if f.private {
        // `default_factory` before `default`, for `PrivateAttr` too: otherwise a
        // `default_factory=list` field would become a shared `default=None`.
        let arg = match (&f.default_factory, &f.default) {
            (Some(factory), _) => format!("default_factory={factory}"),
            (None, Some(default)) => format!("default={default}"),
            (None, None) => "default=None".to_owned(),
        };
        return Ok(format!("{}: {ty} = PrivateAttr({arg})", f.name));
    }
    // pydantic's `deprecated=` turns every default form below into a `Field(..)` call.
    if let Some(msg) = &f.deprecated_msg {
        let head = match (&f.default, &f.default_factory) {
            (_, Some(factory)) => format!("default_factory={factory}"),
            (Some(dflt), None) => format!("default={dflt}"),
            (None, None) => "...".to_owned(),
        };
        let excl = if f.exclude { ", exclude=True" } else { "" };
        return Ok(format!("{}: {ty} = Field({head}, deprecated={msg:?}{excl})", f.name));
    }
    let dataclass = d.target(Host::Python).is_some_and(|t| t.kind == Kind::Dataclass);
    if dataclass && f.exclude {
        return Err(format!(
            "{}.{}: `exclude` is pydantic's; a dataclass field has no serializer to leave it out of",
            d.rust_name, f.name
        ));
    }
    let value = match (&f.default, &f.default_factory, f.exclude) {
        (_, Some(factory), false) if dataclass => format!(" = field(default_factory={factory})"),
        (_, Some(factory), true) => {
            format!(" = Field(default_factory={factory}, exclude=True)")
        }
        (_, Some(factory), false) => format!(" = Field(default_factory={factory})"),
        (Some(dflt), None, true) => format!(" = Field({dflt}, exclude=True)"),
        (None, None, true) => " = Field(..., exclude=True)".to_owned(),
        (Some(dflt), None, false) => format!(" = {dflt}"),
        (None, None, false) => String::new(),
    };
    Ok(format!("{}: {ty}{value}", f.name))
}

/// `Annotated[A | B, Field(discriminator="type")]`, the members on one line when they fit the
/// width, else one per line as `voice/events.py` (what ruff makes of either). `head` is what
/// already sits on the first line (`Name = ` for an alias, nothing inline).
fn union_expr(
    all: &[Decl],
    members: &[Variant],
    pad: &str,
    discriminator: Option<&str>,
    head: usize,
) -> String {
    let names: Vec<String> = members.iter().map(|m| render(all, &m.ty)).collect();
    let flat = names.join(" | ");
    // A plain PEP-604 union unless a discriminator was asked for; pydantic validates them apart.
    let Some(discriminator) = discriminator else {
        // On one line without the parentheses when it fits: ruff drops them then.
        if head + flat.len() <= 100 {
            return flat;
        }
        return format!("(\n{pad}{}\n)", names.join(&format!("\n{pad}| ")));
    };
    let joined =
        if pad.len() + flat.len() + 1 <= 100 { flat } else { names.join(&format!("\n{pad}| ")) };
    // The closing bracket is one level shallower than the members.
    let close = pad.strip_prefix("    ").unwrap_or("");
    format!("Annotated[\n{pad}{joined},\n{pad}Field(discriminator=\"{discriminator}\"),\n{close}]")
}

/// `X | None` and `list[X]`, as the reference writes them (compared as text).
pub fn render(all: &[Decl], t: &Ty) -> String {
    match t {
        Ty::Str => "str".to_owned(),
        Ty::Int => "int".to_owned(),
        Ty::Float => "float".to_owned(),
        Ty::Bool => "bool".to_owned(),
        Ty::Any => "Any".to_owned(),
        Ty::Unit => "None".to_owned(),
        Ty::Option(i) => format!("{} | None", render(all, i)),
        Ty::List(i) => format!("list[{}]", render(all, i)),
        Ty::Map(k, v) => format!("dict[{}, {}]", render(all, k), render(all, v)),
        Ty::Tuple(items) => {
            format!(
                "tuple[{}]",
                items.iter().map(|i| render(all, i)).collect::<Vec<_>>().join(", ")
            )
        }
        Ty::Declared(n) => {
            let Some(d) = all.iter().find(|d| &d.rust_name == n) else { return n.clone() };
            let Some(t) = d.target(Host::Python) else { return n.clone() };
            match (&d.body, t.inline) {
                (Body::Union(members), true) => {
                    union_expr(all, members, "        ", t.discriminator.as_deref(), 0)
                }
                _ => t.name.clone(),
            }
        }
        Ty::Foreign(f) => f.python.clone(),
    }
}

/// Returns whether one was written.
pub fn docstring(b: &mut String, doc: &[String], indent: usize) -> bool {
    if doc.is_empty() {
        return false;
    }
    let pad = " ".repeat(indent);
    // Escape `"""`, and a trailing quote, which would close the literal early.
    let esc = |l: &str| l.replace("\"\"\"", "\\\"\"\"");
    if doc.len() == 1 {
        let mut line = esc(&doc[0]);
        if line.ends_with('"') {
            line.pop();
            line.push_str("\\\"");
        }
        b.push_str(&format!("{pad}\"\"\"{line}\"\"\"\n"));
        return true;
    }
    b.push_str(&format!("{pad}\"\"\"{}\n", esc(&doc[0])));
    for line in &doc[1..] {
        if line.is_empty() {
            b.push('\n');
        } else {
            b.push_str(&format!("{pad}{}\n", esc(line)));
        }
    }
    b.push_str(&format!("{pad}\"\"\"\n"));
    true
}
