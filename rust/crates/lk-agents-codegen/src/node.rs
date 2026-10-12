//! The TypeScript view. Interfaces, not classes: no constructor or `readonly` to get wrong, so
//! safe to regenerate.

use crate::ir::*;
use crate::python::Out;

pub fn emit(decls: &[Decl]) -> Result<Vec<Out>, String> {
    let mut modules: Vec<String> = Vec::new();
    for d in decls {
        if let Some(t) = d.target(Host::Node)
            && !modules.contains(&t.module) {
                modules.push(t.module.clone());
            }
    }
    modules.iter().map(|m| module(decls, m)).collect()
}

fn module(decls: &[Decl], name: &str) -> Result<Out, String> {
    let mine: Vec<&Decl> = decls
        .iter()
        .filter(|d| d.target(Host::Node).is_some_and(|t| t.module == name && !t.inline))
        .collect();

    let mut sources: Vec<&str> = Vec::new();
    for d in &mine {
        if !sources.contains(&d.source.as_str()) {
            sources.push(&d.source);
        }
    }
    let mut b = String::new();
    b.push_str("// GENERATED FILE -- DO NOT EDIT.\n//\n");
    b.push_str("// Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:\n");
    for s in &sources {
        b.push_str(&format!("//     {s}\n"));
    }
    b.push_str(
        "//\n// Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on\n\
         // a non-empty diff, so a hand edit here is reverted rather than merged.\n",
    );

    let mut rel: Vec<(String, Vec<String>)> = Vec::new();
    for d in &mine {
        for f in d.fields() {
            collect(&f.ty, &mut rel);
        }
    }
    if !rel.is_empty() {
        b.push('\n');
        rel.sort_by(|a, b| a.0.cmp(&b.0));
        for (m, names) in &rel {
            let mut n = names.clone();
            n.sort();
            b.push_str(&format!("import type {{ {} }} from '{m}';\n", n.join(", ")));
        }
    }

    let mut section: Option<String> = None;
    for d in &mine {
        if d.section != section {
            section = d.section.clone();
            if let Some(s) = &section {
                b.push_str(&format!(
                    "\n// --- {s} {}\n",
                    "-".repeat(88usize.saturating_sub(s.len() + 8))
                ));
            }
        }
        b.push('\n');
        b.push_str(&item(decls, d)?);
    }
    Ok(Out { path: format!("{name}.ts"), text: b })
}

fn item(all: &[Decl], d: &Decl) -> Result<String, String> {
    let t = d.target(Host::Node).expect("filtered on this");
    let mut b = String::new();
    tsdoc(&mut b, &d.doc, 0);

    match &d.body {
        Body::Union(members) => {
            let names = members
                .iter()
                .map(|m| render(all, &m.ty, &d.rust_name))
                .collect::<Result<Vec<_>, _>>()?;
            b.push_str(&format!("export type {} =\n  | {};\n", t.name, names.join("\n  | ")));
        }
        Body::Unit(variants) => {
            // A string-literal union, not a TS `enum`: `ev.reason === 'job_shutdown'` must type-check.
            b.push_str(&format!("export type {} =\n", t.name));
            for (i, v) in variants.iter().enumerate() {
                let end = if i + 1 == variants.len() { ";" } else { "" };
                b.push_str(&format!("  | '{}'{end}\n", v.value));
            }
        }
        Body::Struct(fields) => {
            b.push_str(&format!("export interface {} {{\n", t.name));
            if let Some((name, value)) = &t.tag {
                b.push_str(&format!("  {}: '{value}';\n", camel(name)));
            }
            for f in fields {
                let Some(field) = field_ts(all, d, f, false)? else { continue };
                tsdoc(&mut b, &f.doc, 2);
                b.push_str(&format!(
                    "  {}{}: {};\n",
                    field.name,
                    if field.optional { "?" } else { "" },
                    field.ty
                ));
            }
            b.push_str("}\n");
        }
    }
    Ok(b)
}

/// One interface field: camelCase name, whether the key may be absent, type; `None` for a
/// host-private attribute. Shared with `napi.rs` (`ts_type`), so `index.d.ts` agrees; `inline`
/// spells out enums and unions, which `index.d.ts` has no alias for.
pub(crate) struct FieldTs {
    pub name: String,
    pub optional: bool,
    pub ty: String,
}

pub(crate) fn field_ts(all: &[Decl], d: &Decl, f: &Field, inline: bool) -> Result<Option<FieldTs>, String> {
    if f.private {
        return Ok(None); // a host-private attribute is not part of the wire shape
    }
    if let Some(lit) = &f.literal {
        return Ok(Some(FieldTs { name: camel(&f.name), optional: false, ty: format!("'{lit}'") }));
    }
    // A foreign annotation over a Rust `Option` is optional too: `total=False` dicts write
    // `timeout: float` without `| None`, so the Rust type says the key may be missing.
    let rust_option =
        matches!(&f.ty, Ty::Foreign(Foreign { rust: Some(r), .. }) if matches!(**r, Ty::Option(_)));
    let optional =
        matches!(f.ty, Ty::Option(_)) || rust_option || f.default.is_some() || f.default_factory.is_some();
    // As in Python: under `total=False` an outer `Option` is absence, which `?` already says.
    let absent = d.target(Host::Python).is_some_and(|t| t.kind == Kind::TypedDict && !t.total);
    let ty = match &f.ty {
        Ty::Option(inner) if absent => render_with(all, inner, &d.rust_name, inline)?,
        other => render_with(all, other, &d.rust_name, inline)?,
    };
    Ok(Some(FieldTs { name: camel(&f.name), optional, ty }))
}

/// napi's camelCase (also agents-js's); leading underscores are kept.
pub(crate) fn camel(name: &str) -> String {
    // SCREAMING_CASE is a constant's name, kept as is.
    if name.chars().any(|c| c.is_ascii_uppercase()) && !name.chars().any(|c| c.is_ascii_lowercase()) {
        return name.to_owned();
    }
    let lead = name.len() - name.trim_start_matches('_').len();
    let mut out = name[..lead].to_owned();
    for (i, word) in name[lead..].split('_').filter(|w| !w.is_empty()).enumerate() {
        if i == 0 {
            out.push_str(word);
        } else {
            let mut cs = word.chars();
            if let Some(c) = cs.next() {
                out.extend(c.to_uppercase());
                out.push_str(cs.as_str());
            }
        }
    }
    out
}

fn render(all: &[Decl], t: &Ty, owner: &str) -> Result<String, String> {
    render_with(all, t, owner, false)
}

/// [`render`], optionally with every string enum and union spelled out (see [`field_ts`]).
pub(crate) fn render_with(all: &[Decl], t: &Ty, owner: &str, inline: bool) -> Result<String, String> {
    Ok(match t {
        Ty::Str => "string".to_owned(),
        Ty::Int | Ty::Float => "number".to_owned(),
        Ty::Bool => "boolean".to_owned(),
        Ty::Any => "unknown".to_owned(),
        Ty::Unit => "null".to_owned(),
        // `| null`, not `| undefined`: JSON can only spell absent-vs-empty this way.
        Ty::Option(i) => format!("{} | null", render_with(all, i, owner, inline)?),
        Ty::List(i) => format!("{}[]", render_with(all, i, owner, inline)?),
        Ty::Map(k, v) => format!("Record<{}, {}>", render_with(all, k, owner, inline)?, render_with(all, v, owner, inline)?),
        Ty::Tuple(items) => format!(
            "[{}]",
            items.iter().map(|i| render_with(all, i, owner, inline)).collect::<Result<Vec<_>, _>>()?.join(", ")
        ),
        Ty::Declared(n) => {
            let decl = all.iter().find(|d| &d.rust_name == n);
            match decl.and_then(|d| d.target(Host::Node).map(|t| (d, t))) {
                // No host name, so spelled out at every use, as in Python.
                Some((d, t)) if t.inline => match &d.body {
                    Body::Union(members) => members
                        .iter()
                        .map(|m| render_with(all, &m.ty, owner, inline))
                        .collect::<Result<Vec<_>, _>>()?
                        .join(" | "),
                    _ => t.name.clone(),
                },
                _ if inline && decl.is_some() => {
                    let d = decl.expect("checked");
                    match &d.body {
                        Body::Unit(vs) => vs.iter().map(|v| format!("'{}'", v.value)).collect::<Vec<_>>().join(" | "),
                        Body::Union(members) => members
                            .iter()
                            .map(|m| render_with(all, &m.ty, owner, inline))
                            .collect::<Result<Vec<_>, _>>()?
                            .join(" | "),
                        Body::Struct(_) => crate::napi::js_name(all, d),
                    }
                }
                Some((_, t)) => t.name.clone(),
                None => n.clone(),
            }
        }
        // Never `any`. A Python annotation of builtins, `Literal`s, containers and unions is
        // spelled by rule; anything naming another type needs its `#[node(ty = ..)]`.
        Ty::Foreign(f) => match (&f.node, ts_of(&f.python)) {
            (Some(n), _) => n.clone(),
            (None, Some(n)) => n,
            (None, None) => {
                return Err(format!(
                    "{owner} targets node but field type `{}` has no #[node(ty = \"..\")] -- give \
                     it one, or drop the node target",
                    f.python
                ));
            }
        },
    })
}

fn collect(t: &Ty, rel: &mut Vec<(String, Vec<String>)>) {
    match t {
        Ty::Option(i) | Ty::List(i) => collect(i, rel),
        Ty::Map(k, v) => {
            collect(k, rel);
            collect(v, rel);
        }
        Ty::Tuple(items) => items.iter().for_each(|i| collect(i, rel)),
        Ty::Foreign(f) => {
            let (Some(name), Some(from)) = (&f.node, &f.node_from) else { return };
            match rel.iter_mut().find(|(m, _)| m == from) {
                Some((_, names)) if names.contains(name) => {}
                Some((_, names)) => names.push(name.clone()),
                None => rel.push((from.clone(), vec![name.clone()])),
            }
        }
        _ => {}
    }
}

fn tsdoc(b: &mut String, doc: &[String], indent: usize) {
    if doc.is_empty() {
        return;
    }
    let pad = " ".repeat(indent);
    b.push_str(&format!("{pad}/**\n"));
    for line in doc {
        if line.is_empty() {
            b.push_str(&format!("{pad} *\n"));
        } else {
            b.push_str(&format!("{pad} * {line}\n"));
        }
    }
    b.push_str(&format!("{pad} */\n"));
}

/// A Python annotation of builtins, `None`, `Literal`, `list`/`tuple`/`dict`,
/// `Required`/`NotRequired` and `|` as TypeScript; `None` for anything else.
pub fn ts_of(py: &str) -> Option<String> {
    let py = py.trim();
    let parts = split_top(py, '|');
    if parts.len() > 1 {
        let mut out: Vec<String> = Vec::new();
        for p in parts {
            let t = ts_of(p)?;
            if !out.contains(&t) {
                out.push(t);
            }
        }
        return Some(out.join(" | "));
    }
    let inner = |head: &str| py.strip_prefix(head).and_then(|r| r.strip_suffix(']'));
    if let Some(args) = inner("Literal[") {
        let items: Vec<String> = split_top(args, ',')
            .into_iter()
            .map(|a| match a.chars().next() {
                Some('"' | '\'') => format!("'{}'", &a[1..a.len() - 1]),
                _ => match a {
                    "True" => "true".to_owned(),
                    "False" => "false".to_owned(),
                    other => other.to_owned(),
                },
            })
            .collect();
        return Some(items.join(" | "));
    }
    if let Some(args) = inner("Required[").or_else(|| inner("NotRequired[")) {
        return ts_of(args);
    }
    if let Some(args) = inner("list[") {
        let t = ts_of(args)?;
        return Some(if t.contains('|') { format!("({t})[]") } else { format!("{t}[]") });
    }
    if let Some(args) = inner("tuple[") {
        let items: Option<Vec<String>> = split_top(args, ',').into_iter().map(ts_of).collect();
        return Some(format!("[{}]", items?.join(", ")));
    }
    if let Some(args) = inner("dict[") {
        let kv = split_top(args, ',');
        let [k, v] = kv.as_slice() else { return None };
        return Some(format!("Record<{}, {}>", ts_of(k)?, ts_of(v)?));
    }
    Some(
        match py {
            "str" => "string",
            "int" | "float" => "number",
            "bool" => "boolean",
            "None" => "null",
            "Any" => "unknown",
            _ => return None,
        }
        .to_owned(),
    )
}

fn split_top(s: &str, sep: char) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0i32, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            c if c == sep && depth == 0 => {
                out.push(s[start..i].trim());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out.into_iter().filter(|p| !p.is_empty()).collect()
}
