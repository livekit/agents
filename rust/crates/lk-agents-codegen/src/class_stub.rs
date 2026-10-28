//! The `.pyi` of the declared classes, one per ABI subject (`_generated/_core_<subject>.pyi`),
//! star-imported by `_core.pyi`. Written from the same IR as the adapters.

use crate::decl::{Class, Declarations, Form, HostForm, Literal, Method, MethodKind, Ty};
use crate::python::Out;

pub fn emit(decls: &Declarations) -> Result<Vec<Out>, String> {
    let mut families: Vec<&str> = decls.classes.iter().map(|c| c.family.as_str()).collect();
    families.sort();
    families.dedup();
    let mut out = Vec::new();
    for family in families {
        let classes: Vec<&Class> = decls.classes.iter().filter(|c| c.family == family).collect();
        let mut imports = Imports::default();
        let mut body = String::new();
        let mut own: Vec<String> = Vec::new();
        for c in &classes {
            body.push_str(&class(c, &mut imports));
            if c.form != Form::Functions {
                own.push(c.name.clone());
            }
            for (name, item) in crate::adapter::sources(c) {
                let item = annotation(c, &item, Dir::Out, &mut imports);
                body.push_str(&format!(
                    "\nclass {name}:\n    async def recv(self) -> {item} | None: ...\n    async def aclose(self) -> None: ...\n"
                ));
                own.push(name);
            }
        }
        let own: Vec<&str> = own.iter().map(String::as_str).collect();
        let mut text = String::from(
            "# GENERATED FILE -- DO NOT EDIT.\n#\n# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the\n# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` star-imports it.\n\nfrom __future__ import annotations\n",
        );
        let imports = imports.render(&own);
        if !imports.is_empty() {
            text.push('\n');
            text.push_str(&imports);
        }
        text.push_str(&body);
        out.push(Out { path: format!("_core_{family}.pyi"), text });
    }
    Ok(out)
}

/// `from <module> import <name>` lines, sorted.
#[derive(Default)]
struct Imports(Vec<(String, String)>);

impl Imports {
    fn add(&mut self, module: &str, name: &str) {
        let entry = (module.to_owned(), name.to_owned());
        if !self.0.contains(&entry) {
            self.0.push(entry);
        }
    }

    /// isort's order: the standard library, other absolute imports, relative ones; each block
    /// sorted and separated by a blank line.
    fn render(&self, own: &[&str]) -> String {
        let block = |m: &str| match m {
            _ if m.starts_with('.') => 2,
            "typing" | "collections.abc" | "enum" | "asyncio" => 0,
            _ => 1,
        };
        let mut modules: Vec<&str> = self.0.iter().map(|(m, _)| m.as_str()).collect();
        modules.sort_by_key(|m| (block(m), *m));
        modules.dedup();
        let mut s = String::new();
        let mut last = None;
        for m in modules {
            let mut names: Vec<&str> = self
                .0
                .iter()
                .filter(|(mm, n)| mm == m && !(m == ".._core" && own.contains(&n.as_str())))
                .map(|(_, n)| n.as_str())
                .collect();
            names.sort();
            if names.is_empty() {
                continue;
            }
            if last.is_some_and(|b| b != block(m)) {
                s.push('\n');
            }
            last = Some(block(m));
            s.push_str(&format!("from {m} import {}\n", names.join(", ")));
        }
        s
    }
}

fn class(c: &Class, imports: &mut Imports) -> String {
    if c.form == Form::Functions {
        return c.methods.iter().map(|m| format!("\n{}", method(c, m, imports))).collect();
    }
    let mut s = format!("\nclass {}:\n", c.name);
    s.push_str(&docstring(&c.doc, "    "));
    for m in &c.methods {
        s.push_str(&method(c, m, imports));
    }
    s
}

fn method(c: &Class, m: &Method, imports: &mut Imports) -> String {
    let mut params =
        if m.kind == MethodKind::Function { Vec::new() } else { vec!["self".to_owned()] };
    let mut star = false;
    for p in &m.params {
        if p.kw_only && !star {
            params.push("*".into());
            star = true;
        }
        let ty = annotation(c, &p.ty, Dir::In, imports);
        params.push(match &p.default {
            Some(d) => format!("{}: {ty} = {}", p.name, py_literal(d)),
            None => format!("{}: {ty}", p.name),
        });
    }
    let ret = annotation(c, &m.ret.ok, Dir::Out, imports);
    let mut s = String::new();
    let ind = if m.kind == MethodKind::Function { "" } else { "    " };
    let doc = docstring(&m.doc, &format!("{ind}    "));
    let tail = if doc.is_empty() { " ...\n".to_owned() } else { format!("\n{doc}") };
    let (head, name, ret) = match m.kind {
        MethodKind::Function | MethodKind::Call => ("def", m.name.as_str(), ret),
        MethodKind::Await => ("async def", m.name.as_str(), ret),
        MethodKind::New => ("def", "__init__", "None".to_owned()),
        MethodKind::Getter => {
            s.push_str("    @property\n");
            ("def", m.name.as_str(), ret)
        }
    };
    // One line when it fits the line length, else one parameter per line (ruff's layout).
    let line = format!("{ind}{head} {name}({}) -> {ret}:", params.join(", "));
    if line.len() + if doc.is_empty() { 4 } else { 0 } <= 100 {
        s.push_str(&line);
    } else {
        s.push_str(&format!("{ind}{head} {name}(\n"));
        for p in &params {
            s.push_str(&format!("{ind}    {p},\n"));
        }
        s.push_str(&format!("{ind}) -> {ret}:"));
    }
    s.push_str(&tail);
    s
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
    In,
    Out,
}

/// What the host passes (`In`) or receives (`Out`): an emitted record is a dict of its fields.
fn annotation(c: &Class, ty: &Ty, dir: Dir, imports: &mut Imports) -> String {
    match ty {
        Ty::Unit => "None".into(),
        Ty::Bool => "bool".into(),
        Ty::Int(_) => "int".into(),
        Ty::Float => "float".into(),
        Ty::Str => "str".into(),
        Ty::Json => "object".into(),
        Ty::JsonObject => "dict[str, object]".into(),
        Ty::Optional(t) => format!("{} | None", annotation(c, t, dir, imports)),
        Ty::List(t) => format!("list[{}]", annotation(c, t, dir, imports)),
        Ty::Tuple(ts) => format!(
            "tuple[{}]",
            ts.iter().map(|t| annotation(c, t, dir, imports)).collect::<Vec<_>>().join(", ")
        ),
        Ty::Record(r)
            if dir == Dir::Out
                && !matches!(r.host, HostForm::CoreClass(_) | HostForm::Annotation(_)) =>
        {
            "dict[str, object]".into()
        }
        Ty::Record(r) => match &r.host {
            HostForm::Model { module, name } => {
                imports.add(&format!(".{module}"), name);
                name.clone()
            }
            HostForm::CoreClass(name) => {
                imports.add(".._core", name);
                name.clone()
            }
            HostForm::External { module, name } => {
                imports.add(module, name);
                name.clone()
            }
            HostForm::Dict => "dict[str, object]".into(),
            HostForm::Annotation(text) => text.clone(),
        },
        Ty::This => c.name.clone(),
        Ty::Proxy { host, generic, .. } => {
            let (module, name) = host.rsplit_once('.').expect("a proxy's host class is dotted");
            imports.add(module, name);
            if *generic {
                imports.add("typing", "Any");
                format!("{name}[Any]")
            } else {
                name.to_owned()
            }
        }
        Ty::Source { class, .. } => class.clone(),
    }
}

fn py_literal(d: &Literal) -> String {
    match d {
        Literal::None => "None".into(),
        Literal::Bool(true) => "True".into(),
        Literal::Bool(false) => "False".into(),
        Literal::Int(i) => i.to_string(),
        Literal::Float(f) => format!("{f:?}"),
        Literal::Str(s) => format!("{s:?}"),
    }
}

fn docstring(doc: &[String], indent: &str) -> String {
    if doc.is_empty() {
        return String::new();
    }
    let mut s = format!("{indent}\"\"\"");
    for (i, line) in doc.iter().enumerate() {
        if i > 0 {
            s.push('\n');
            if !line.is_empty() {
                s.push_str(indent);
            }
        }
        s.push_str(&line.replace("\"\"\"", "\\\"\\\"\\\""));
    }
    if doc.len() > 1 {
        s.push('\n');
        s.push_str(indent);
    }
    s.push_str("\"\"\"\n");
    s
}
