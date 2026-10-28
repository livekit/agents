//! The `.pyi` of the declared classes, one per ABI subject (`_generated/_core_<subject>.pyi`),
//! re-exported by `_core.pyi`. Written from the same IR as the adapters.

use crate::decl::{
    Class, Declarations, Form, HostForm, Literal, Method, MethodKind, Proxy, ProxyCall, RecordDecl,
    Ty,
};
use crate::python::Out;
use crate::surface::Stub;

pub fn emit(decls: &Declarations) -> Result<Vec<Stub>, String> {
    let mut families: Vec<&str> = decls
        .classes
        .iter()
        .map(|c| c.family.as_str())
        .chain(
            decls
                .records
                .iter()
                .filter(|r| crate::adapter::has_functions(r))
                .map(|r| r.family.as_str()),
        )
        .collect();
    families.sort();
    families.dedup();
    let mut out = Vec::new();
    for family in families {
        let classes: Vec<&Class> = decls.classes.iter().filter(|c| c.family == family).collect();
        let mut imports = Imports::new(decls);
        let mut body = String::new();
        let mut own: Vec<String> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for c in &classes {
            body.push_str(&class(c, &mut imports));
            if c.form == Form::Functions {
                names.extend(c.methods.iter().map(|m| m.name.clone()));
            } else {
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
        for r in decls.records.iter().filter(|r| r.persisted && r.family == family) {
            let (persist, restore) = r.persistence();
            imports.add(&format!(".{}", r.module), &r.name);
            body.push_str(&format!(
                "\ndef {persist}(record: {name}) -> str: ...\ndef {restore}(data: str) -> {name}: ...\n",
                name = r.name
            ));
            names.extend([persist, restore]);
        }
        for r in decls.records.iter().filter(|r| r.family == family) {
            for (f, function) in r.omitted() {
                imports.add(&format!(".{}", r.module), &r.name);
                let ty = match f.ty {
                    Ty::Bool => "bool",
                    Ty::Int(_) => "int",
                    Ty::Float | Ty::Duration => "float",
                    _ => "str",
                };
                body.push_str(&format!("\ndef {function}(record: {}) -> {ty}: ...\n", r.name));
                names.push(function);
            }
        }
        names.extend(own.iter().cloned());
        let own: Vec<&str> = own.iter().map(String::as_str).collect();
        let mut text = String::from(
            "# GENERATED FILE -- DO NOT EDIT.\n#\n# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the\n# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.\n\nfrom __future__ import annotations\n",
        );
        let imports = imports.render(&own);
        if !imports.is_empty() {
            text.push('\n');
            text.push_str(&imports);
        }
        text.push_str(&body);
        out.push(Stub { out: Out { path: format!("_core_{family}.pyi"), text }, names });
    }
    Ok(out)
}

/// `from <module> import <name>` lines, sorted.
#[derive(Default)]
pub(crate) struct Imports(Vec<(String, String)>, Vec<RecordDecl>, Vec<Proxy>);

impl Imports {
    pub(crate) fn new(declarations: &Declarations) -> Self {
        Self(Vec::new(), declarations.records.clone(), declarations.proxies.clone())
    }

    pub(crate) fn entries(&self) -> &[(String, String)] {
        &self.0
    }

    /// `name` may be a nested model (`Owner.Name`): its owner is imported.
    pub(crate) fn add(&mut self, module: &str, name: &str) {
        let name = name.split('.').next().unwrap_or(name);
        let entry = (module.to_owned(), name.to_owned());
        if !self.0.contains(&entry) {
            self.0.push(entry);
        }
    }

    /// A handwritten scalar union can refer to the generated records beside its declaration.
    fn annotation(&mut self, text: &str, rust: &str) {
        let Some((module, _)) = rust.rsplit_once("::") else { return };
        let records: Vec<_> = self
            .1
            .iter()
            .filter(|record| {
                record.rust.rsplit_once("::").is_some_and(|(scope, _)| scope == module)
                    && text
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .any(|name| name == record.name)
            })
            .map(|record| (record.module.clone(), record.name.clone()))
            .collect();
        for (module, name) in records {
            self.add(&format!(".{module}"), &name);
        }
    }

    /// `import module` (a host callable named by its module, `time.time`).
    pub(crate) fn add_module(&mut self, module: &str) {
        self.add(module, "");
    }

    /// isort's order: the standard library, other absolute imports, relative ones; each block
    /// sorted, plain imports first, and separated by a blank line.
    pub(crate) fn render(&self, own: &[&str]) -> String {
        let block = |m: &str| match m {
            _ if m.starts_with('.') => 2,
            "typing" | "collections.abc" | "enum" | "asyncio" | "dataclasses" | "time" => 0,
            _ => 1,
        };
        let mut statements: Vec<(u8, bool, &str)> =
            self.0.iter().map(|(m, n)| (block(m), !n.is_empty(), m.as_str())).collect();
        statements.sort();
        statements.dedup();
        let mut s = String::new();
        let mut last = None;
        for (b, from, m) in statements {
            let line = if from {
                let mut names: Vec<&str> = self
                    .0
                    .iter()
                    .filter(|(mm, n)| {
                        mm == m && !n.is_empty() && !(m == ".._core" && own.contains(&n.as_str()))
                    })
                    .map(|(_, n)| n.as_str())
                    .collect();
                // ruff's order by type: CONSTANTS, then Classes, then the rest.
                names.sort_by_key(|n| {
                    let kind = if n.chars().all(|c| c.is_uppercase() || c == '_') {
                        0
                    } else if n.starts_with(char::is_uppercase) {
                        1
                    } else {
                        2
                    };
                    (kind, *n)
                });
                if names.is_empty() {
                    continue;
                }
                format!("from {m} import {}\n", names.join(", "))
            } else {
                format!("import {m}\n")
            };
            if last.is_some_and(|l| l != b) {
                s.push('\n');
            }
            last = Some(b);
            s.push_str(&line);
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
    let mut params = if matches!(m.kind, MethodKind::Function | MethodKind::AsyncFunction) {
        Vec::new()
    } else {
        vec!["self".to_owned()]
    };
    let (parameters, ret) = signature_parts(c, m, imports);
    let mut star = false;
    for (parameter, kw_only) in parameters {
        if kw_only && !star {
            params.push("*".into());
            star = true;
        }
        params.push(parameter);
    }
    let mut s = String::new();
    let ind = if matches!(m.kind, MethodKind::Function | MethodKind::AsyncFunction) {
        ""
    } else {
        "    "
    };
    let doc = docstring(&m.doc, &format!("{ind}    "));
    let tail = if doc.is_empty() { " ...\n".to_owned() } else { format!("\n{doc}") };
    let (head, name, ret) = match m.kind {
        MethodKind::Function | MethodKind::Call => ("def", m.name.as_str(), ret),
        MethodKind::Await | MethodKind::AsyncFunction => ("async def", m.name.as_str(), ret),
        MethodKind::New => ("def", "__init__", "None".to_owned()),
        MethodKind::Getter => {
            s.push_str("    @property\n");
            ("def", m.name.as_str(), ret)
        }
        MethodKind::Next => {
            imports.add("typing", "Self");
            let item = match &m.ret.ok {
                Ty::Optional(item) => annotation(c, item, Dir::Out, imports),
                other => unreachable!("next returns an option, not {other:?}"),
            };
            s.push_str("    def __aiter__(self) -> Self: ...\n");
            ("async def", "__anext__", item)
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

/// One callable's parameters and result, shared by stubs and generated forwarding methods.
pub(crate) fn signature_parts(
    c: &Class,
    m: &Method,
    imports: &mut Imports,
) -> (Vec<(String, bool)>, String) {
    (parameters(c, m, imports), annotation(c, &m.ret.ok, Dir::Out, imports))
}

/// Parameters shared by native stubs and authoring-class constructors.
pub(crate) fn parameters(c: &Class, m: &Method, imports: &mut Imports) -> Vec<(String, bool)> {
    m.params
        .iter()
        .filter(|p| p.ty != Ty::CallContext)
        .map(|p| {
            let ty = if p.shape_error.as_ref().is_some_and(|policy| policy.items) {
                strict_list_input(c, &p.ty, imports)
            } else {
                annotation(c, &p.ty, Dir::In, imports)
            };
            let name = p.name.trim_start_matches("r#");
            let text = match &p.default {
                Some(default) => format!("{name}: {ty} = {}", py_literal(default)),
                None if matches!(p.ty, Ty::Maybe(_)) => format!("{name}: {ty} = ..."),
                None => format!("{name}: {ty}"),
            };
            (text, p.kw_only)
        })
        .collect()
}

/// The existing item-shape policy reads a concrete PyList; child records keep input types.
fn strict_list_input(c: &Class, ty: &Ty, imports: &mut Imports) -> String {
    match ty {
        Ty::Optional(inner) => format!("{} | None", strict_list_input(c, inner, imports)),
        Ty::List(item) => format!("list[{}]", annotation(c, item, Dir::In, imports)),
        other => annotation(c, other, Dir::In, imports),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
    In,
    Out,
    /// A proxy result decoded by from_object: list values may be any iterable.
    ProxyResult,
}

/// What the host passes (`In`) or receives (`Out`): an emitted record is a dict of its fields.
fn annotation(c: &Class, ty: &Ty, dir: Dir, imports: &mut Imports) -> String {
    match ty {
        Ty::Unit => "None".into(),
        // Hidden from the host's signature: the binding captures it.
        Ty::CallContext => "object".into(),
        Ty::Bool => "bool".into(),
        Ty::Int(_) => "int".into(),
        Ty::Float | Ty::Duration => "float".into(),
        Ty::Str => "str".into(),
        Ty::Json => "object".into(),
        Ty::JsonObject => "dict[str, object]".into(),
        Ty::Optional(t) => format!("{} | None", annotation(c, t, dir, imports)),
        Ty::Maybe(t) => {
            imports.add("..types", "NotGivenOr");
            format!("NotGivenOr[{}]", annotation(c, t, dir, imports))
        }
        Ty::List(t) if dir == Dir::In && matches!(**t, Ty::Int(crate::decl::Int::U8)) => {
            "bytes | bytearray | memoryview | list[int]".into()
        }
        Ty::List(t) if matches!(**t, Ty::Int(crate::decl::Int::U8)) => "bytes".into(),
        Ty::List(t) if dir == Dir::In => {
            imports.add("collections.abc", "Sequence");
            format!("Sequence[{}]", annotation(c, t, dir, imports))
        }
        Ty::List(t) if dir == Dir::ProxyResult => {
            imports.add("collections.abc", "Iterable");
            format!("Iterable[{}]", annotation(c, t, dir, imports))
        }
        Ty::List(t) => format!("list[{}]", annotation(c, t, dir, imports)),
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => {
            let map = if dir == Dir::In && matches!(ty, Ty::Map(..)) {
                imports.add("collections.abc", "Mapping");
                "Mapping"
            } else {
                "dict"
            };
            format!("{map}[{}, {}]", annotation(c, k, dir, imports), annotation(c, t, dir, imports))
        }
        Ty::Set(t) => format!("frozenset[{}]", annotation(c, t, dir, imports)),
        Ty::Tuple(ts) => format!(
            "tuple[{}]",
            ts.iter().map(|t| annotation(c, t, dir, imports)).collect::<Vec<_>>().join(", ")
        ),
        Ty::Record(r)
            if dir == Dir::Out
                && !matches!(
                    r.host,
                    HostForm::CoreClass(_)
                        | HostForm::Annotation(_)
                        | HostForm::WireStrs
                        | HostForm::Record { .. }
                ) =>
        {
            "dict[str, object]".into()
        }
        Ty::Record(r) => match &r.host {
            HostForm::Model { module, name } | HostForm::Record { module, name } => {
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
            HostForm::WireStrs => "str".into(),
            HostForm::Annotation(text) => {
                imports.annotation(text, &r.rust);
                text.clone()
            }
        },
        Ty::This => c.name.clone(),
        Ty::Proxy { host, generic, declared, .. } => {
            if host == "collections.abc.Callable"
                && let Some((family, name)) = declared
                && let Some(method) = imports
                    .2
                    .iter()
                    .find(|proxy| &proxy.family == family && &proxy.rust_name == name)
                    .and_then(|proxy| proxy.methods.iter().find(|method| method.name == "__call__"))
                    .cloned()
            {
                imports.add("collections.abc", "Callable");
                let params = method
                    .params
                    .iter()
                    .filter(|parameter| parameter.ty != Ty::CallContext)
                    .map(|parameter| annotation(c, &parameter.ty, Dir::Out, imports))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut result = annotation(c, &method.ret.ok, Dir::ProxyResult, imports);
                if method.kind == ProxyCall::Await {
                    imports.add("collections.abc", "Awaitable");
                    result = format!("Awaitable[{result}]");
                }
                return format!("Callable[[{params}], {result}]");
            }
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
        Ty::HostSource(item) => {
            imports.add("collections.abc", "AsyncIterable");
            format!("AsyncIterable[{}]", annotation(c, item, dir, imports))
        }
        Ty::Error(_) => "BaseException".into(),
        Ty::HostDuplex { output, .. } => {
            imports.add("collections.abc", "AsyncIterable");
            format!("AsyncIterable[{}]", annotation(c, output, dir, imports))
        }
        Ty::Handle { class, .. } | Ty::Instance { class, .. } | Ty::Shared { class, .. } => {
            imports.add(".._core", class);
            class.clone()
        }
        Ty::Native { classes, .. } => classes
            .iter()
            .map(|class| {
                let (module, name) = class.rsplit_once('.').expect("a host class is dotted");
                imports.add(module, name);
                name
            })
            .collect::<Vec<_>>()
            .join(" | "),
    }
}

pub(crate) fn py_literal(d: &Literal) -> String {
    match d {
        Literal::None => "None".into(),
        Literal::Bool(true) => "True".into(),
        Literal::Bool(false) => "False".into(),
        Literal::Int(i) => i.to_string(),
        Literal::Float(f) => format!("{f:?}"),
        Literal::Str(s) => format!("{s:?}"),
    }
}

pub(crate) fn docstring(doc: &[String], indent: &str) -> String {
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
