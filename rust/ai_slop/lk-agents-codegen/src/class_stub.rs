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
        .chain(
            decls
                .proxies
                .iter()
                .filter(|p| !crate::adapter::proxy_sources(decls, p).is_empty())
                .map(|p| p.family.as_str()),
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
                body.push_str(&source_stub(
                    &name,
                    &annotation(&c.name, &item, Dir::Out, &mut imports),
                ));
                own.push(name);
            }
            for (sink, output) in crate::adapter::duplexes(c) {
                let input = annotation(&c.name, &sink.item, Dir::In, &mut imports);
                let output = annotation(&c.name, &output, Dir::Out, &mut imports);
                let control = Class {
                    name: sink.class.clone(),
                    methods: sink.methods.clone(),
                    ..(*c).clone()
                };
                body.push_str(&format!(
                    "\nclass {}:\n    def push(self, item: {input}) -> None: ...\n    def flush(self) -> None: ...\n    def end_input(self) -> None: ...\n",
                    sink.class
                ));
                for m in &control.methods {
                    body.push_str(&method(&control, m, &mut imports));
                }
                body.push_str(&format!(
                    "    async def recv(self) -> {output} | None: ...\n    async def aclose(self) -> None: ...\n"
                ));
                own.push(sink.class.clone());
            }
        }
        for p in decls.proxies.iter().filter(|p| p.family == family) {
            for (name, item) in crate::adapter::proxy_sources(decls, p) {
                body.push_str(&source_stub(&name, &annotation("", &item, Dir::Out, &mut imports)));
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
                    Ty::Float | Ty::Duration | Ty::Timestamp => "float",
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
pub(crate) struct Imports {
    entries: Vec<(String, String)>,
    records: Vec<RecordDecl>,
    proxies: Vec<Proxy>,
    defined: Vec<String>,
}

impl Imports {
    pub(crate) fn new(declarations: &Declarations) -> Self {
        Self {
            entries: Vec::new(),
            records: declarations.records.clone(),
            proxies: declarations.proxies.clone(),
            defined: declarations
                .classes
                .iter()
                .filter(|class| class.form != Form::Functions)
                .flat_map(|class| {
                    std::iter::once(class.name.clone())
                        .chain(crate::adapter::sources(class).into_iter().map(|(name, _)| name))
                })
                .collect(),
        }
    }

    pub(crate) fn entries(&self) -> &[(String, String)] {
        &self.entries
    }

    /// `name` may be a nested model (`Owner.Name`): its owner is imported.
    pub(crate) fn add(&mut self, module: &str, name: &str) {
        let name = name.split('.').next().unwrap_or(name);
        let entry = (module.to_owned(), name.to_owned());
        if !self.entries.contains(&entry) {
            self.entries.push(entry);
        }
    }

    /// Import colliding host classes under distinct local names.
    fn external(&mut self, module: &str, name: &str) -> String {
        let occupied = |bound: &str| {
            self.defined.iter().any(|defined| defined == bound)
                || self.entries.iter().any(|(source, import)| {
                    let (imported, alias) = import.split_once(" as ").unwrap_or((import, import));
                    alias == bound && !(source == module && imported == name)
                })
        };
        if occupied(name) {
            let mut alias = format!("Host{name}");
            while occupied(&alias) {
                alias.insert_str(0, "Host");
            }
            self.add(module, &format!("{name} as {alias}"));
            alias
        } else {
            self.add(module, name);
            name.to_owned()
        }
    }

    /// Resolve names in a handwritten annotation through their declarations or qualified paths.
    pub(crate) fn annotation(&mut self, text: &str, rust: &str) -> String {
        let scope = rust.rsplit_once("::").map(|(scope, _)| scope);
        let mut result = String::new();
        let mut end = 0;
        for (range, name) in annotation_names(text) {
            result.push_str(&text[end..range.start]);
            let record = self
                .records
                .iter()
                .find(|record| {
                    record.rust.rsplit_once("::").map(|(scope, _)| scope) == scope
                        && record.name == name
                })
                .map(|record| (record.module.clone(), record.name.clone()));
            if let Some((module, name)) = record {
                self.add(&format!(".{module}"), &name);
                result.push_str(&name);
            } else if name.starts_with("rtc.") {
                self.add("livekit", "rtc");
                result.push_str(name);
            } else if let Some((module, name)) = name.rsplit_once('.') {
                result.push_str(&self.external(module, name));
            } else {
                result.push_str(name);
            }
            end = range.end;
        }
        result.push_str(&text[end..]);
        result
    }

    /// `import module` (a host callable named by its module, `time.time`).
    pub(crate) fn add_module(&mut self, module: &str) {
        self.add(module, "");
    }

    /// isort's order: the standard library, other absolute imports, relative ones; each block
    /// sorted, plain imports first, and separated by a blank line.
    pub(crate) fn render(&self, own: &[&str]) -> String {
        let block = |m: &str| match m {
            _ if m.starts_with('.') => 3,
            "typing" | "collections.abc" | "enum" | "asyncio" | "dataclasses" | "time" => 0,
            _ if m == "livekit" || m.starts_with("livekit.") => 2,
            _ => 1,
        };
        let mut statements: Vec<(u8, bool, &str)> =
            self.entries.iter().map(|(m, n)| (block(m), !n.is_empty(), m.as_str())).collect();
        statements.sort();
        statements.dedup();
        let mut s = String::new();
        let mut last = None;
        for (b, from, m) in statements {
            let line = if from {
                let mut names: Vec<&str> = self
                    .entries
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
                crate::imports::render(
                    m,
                    &names.iter().map(|name| name.to_string()).collect::<Vec<_>>(),
                )
                .join("\n")
                    + "\n"
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

/// Identifier paths, excluding quoted literal values and forward-reference strings.
pub(crate) fn annotation_names(text: &str) -> Vec<(std::ops::Range<usize>, &str)> {
    let chars: Vec<_> = text.char_indices().collect();
    let mut names = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (start, ch) = chars[i];
        i += 1;
        if ch == '\'' || ch == '"' {
            while i < chars.len() {
                let next = chars[i].1;
                i += 1;
                if next == '\\' {
                    i = (i + 1).min(chars.len());
                } else if next == ch {
                    break;
                }
            }
        } else if ch.is_alphabetic()
            || ch == '_'
            || (ch == '.' && chars.get(i).is_some_and(|c| c.1 == '.' || c.1.is_alphabetic()))
        {
            // A relative path (`..llm.tool_context.Tool`) keeps its leading dots.
            while i < chars.len() && chars[i].1 == '.' {
                i += 1;
            }
            // `...` (a variadic tuple) names nothing.
            if ch == '.' && !chars.get(i).is_some_and(|c| c.1.is_alphabetic() || c.1 == '_') {
                continue;
            }
            while i < chars.len()
                && (chars[i].1.is_alphanumeric() || matches!(chars[i].1, '_' | '.'))
            {
                i += 1;
            }
            let end = chars.get(i).map_or(text.len(), |(index, _)| *index);
            names.push((start..end, &text[start..end]));
        }
    }
    names
}

/// A stream class: `recv()` the next item, `None` at the end.
fn source_stub(name: &str, item: &str) -> String {
    format!(
        "\nclass {name}:\n    async def recv(self) -> {item} | None: ...\n    async def aclose(self) -> None: ...\n"
    )
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
    if c.doc.is_empty() && c.methods.is_empty() {
        s.push_str("    ...\n");
    }
    s
}

fn method(c: &Class, m: &Method, imports: &mut Imports) -> String {
    let mut params = if matches!(
        m.kind,
        MethodKind::Function | MethodKind::AsyncFunction | MethodKind::StaticCall
    ) {
        Vec::new()
    } else if m.kind == MethodKind::New {
        vec!["cls".to_owned()]
    } else {
        vec!["self".to_owned()]
    };
    let (mut parameters, ret) = signature_parts(c, m, imports);
    // Python compares with any object; pyo3 answers `NotImplemented` for a foreign type.
    if matches!(m.name.as_str(), "__eq__" | "__ne__") {
        for (parameter, _) in &mut parameters {
            if let Some((name, _)) = parameter.split_once(':') {
                *parameter = format!("{name}: object");
            }
        }
    }
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
        MethodKind::StaticCall => {
            s.push_str("    @staticmethod\n");
            ("def", m.name.as_str(), ret)
        }
        MethodKind::Await | MethodKind::AsyncFunction => ("async def", m.name.as_str(), ret),
        // pyo3 builds the object in `__new__`: a Python subclass overrides that, not `__init__`.
        MethodKind::New => {
            imports.add("typing", "Self");
            ("def", "__new__", "Self".to_owned())
        }
        MethodKind::Getter => {
            s.push_str("    @property\n");
            ("def", m.name.as_str(), ret)
        }
        MethodKind::Setter => {
            s.push_str(&format!("    @{}.setter\n", m.name));
            ("def", m.name.as_str(), ret)
        }
        MethodKind::Next => {
            imports.add("typing", "Self");
            let item = match &m.ret.ok {
                Ty::Optional(item) => annotation(&c.name, item, Dir::Out, imports),
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
    let result = match &m.future {
        Some(ty) => {
            imports.add("asyncio", "Future");
            format!("Future[{}]", annotation(&c.name, ty, Dir::Out, imports))
        }
        None => annotation(&c.name, &m.ret.ok, Dir::Out, imports),
    };
    (parameters(c, m, imports), result)
}

/// Parameters shared by native stubs and authoring-class constructors.
pub(crate) fn parameters(c: &Class, m: &Method, imports: &mut Imports) -> Vec<(String, bool)> {
    m.params
        .iter()
        .filter(|p| !p.ty.hidden())
        .map(|p| {
            let ty = if let (true, Ty::List(item)) = (p.one_or_many, &p.ty) {
                let item = annotation(&c.name, item, Dir::In, imports);
                format!("{item} | list[{item}]")
            } else if p.pcm {
                "bytes | bytearray | memoryview".to_owned()
            } else if p.shape_error.as_ref().is_some_and(|policy| policy.items) {
                strict_list_input(c, &p.ty, imports)
            } else {
                annotation(&c.name, &p.ty, Dir::In, imports)
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
        Ty::List(item) => format!("list[{}]", annotation(&c.name, item, Dir::In, imports)),
        other => annotation(&c.name, other, Dir::In, imports),
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
/// `this`: the class whose method it annotates (`Self`).
fn annotation(this: &str, ty: &Ty, dir: Dir, imports: &mut Imports) -> String {
    match ty {
        Ty::Unit => "None".into(),
        // Hidden from the host's signature: the binding captures it.
        Ty::CallContext | Ty::Env => "object".into(),
        Ty::Bool => "bool".into(),
        Ty::Int(_) => "int".into(),
        Ty::Float | Ty::Duration | Ty::Timestamp => "float".into(),
        Ty::Str => "str".into(),
        Ty::Json => "object".into(),
        Ty::JsonObject => "dict[str, object]".into(),
        Ty::Optional(t) => format!("{} | None", annotation(this, t, dir, imports)),
        Ty::Boxed(t) => annotation(this, t, dir, imports),
        Ty::Maybe(t) => {
            imports.add("..types", "NotGivenOr");
            format!("NotGivenOr[{}]", annotation(this, t, dir, imports))
        }
        Ty::List(t) if dir == Dir::Out && matches!(**t, Ty::Int(crate::decl::Int::U8)) => {
            "bytes".into()
        }
        Ty::List(t) if dir == Dir::In && matches!(**t, Ty::Int(crate::decl::Int::U8)) => {
            "bytes | bytearray | memoryview | list[int]".into()
        }
        Ty::List(t) if matches!(**t, Ty::Int(crate::decl::Int::U8)) => "bytes".into(),
        Ty::List(t) if dir == Dir::In => {
            imports.add("collections.abc", "Sequence");
            format!("Sequence[{}]", annotation(this, t, dir, imports))
        }
        Ty::List(t) if dir == Dir::ProxyResult => {
            imports.add("collections.abc", "Iterable");
            format!("Iterable[{}]", annotation(this, t, dir, imports))
        }
        Ty::List(t) => format!("list[{}]", annotation(this, t, dir, imports)),
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => {
            let map = if dir == Dir::In && matches!(ty, Ty::Map(..)) {
                imports.add("collections.abc", "Mapping");
                "Mapping"
            } else {
                "dict"
            };
            format!(
                "{map}[{}, {}]",
                annotation(this, k, dir, imports),
                annotation(this, t, dir, imports)
            )
        }
        Ty::Set(t) => format!("frozenset[{}]", annotation(this, t, dir, imports)),
        Ty::Tuple(ts) => format!(
            "tuple[{}]",
            ts.iter().map(|t| annotation(this, t, dir, imports)).collect::<Vec<_>>().join(", ")
        ),
        Ty::Record(r)
            if dir == Dir::Out
                && !matches!(
                    r.host,
                    HostForm::CoreClass(_)
                        | HostForm::Annotation(_)
                        | HostForm::Converted(_)
                        | HostForm::WireStrs
                        | HostForm::Str
                        | HostForm::Record { .. }
                ) =>
        {
            "dict[str, object]".into()
        }
        Ty::Record(r) => match &r.host {
            // A nested model is named through its owner.
            HostForm::Model { module, name } | HostForm::Record { module, name }
                if name.contains('.') =>
            {
                imports.add(&format!(".{module}"), name);
                name.clone()
            }
            // Two modules' models of one name are told apart by an alias.
            HostForm::Model { module, name } | HostForm::Record { module, name } => {
                imports.external(&format!(".{module}"), name)
            }
            HostForm::CoreClass(name) => {
                imports.add(".._core", name);
                name.clone()
            }
            HostForm::External { module, name } => imports.external(module, name),
            HostForm::Dict => "dict[str, object]".into(),
            HostForm::WireStrs | HostForm::Str => "str".into(),
            HostForm::Annotation(text) | HostForm::Converted(text) => {
                imports.annotation(text, &r.rust)
            }
        },
        Ty::This => this.to_owned(),
        Ty::Proxy { host, generic, declared, .. } => {
            if host == "collections.abc.Callable"
                && let Some((family, name)) = declared
                && let Some(method) = imports
                    .proxies
                    .iter()
                    .find(|proxy| &proxy.family == family && &proxy.rust_name == name)
                    .and_then(|proxy| proxy.methods.iter().find(|method| method.name == "__call__"))
                    .cloned()
            {
                imports.add("collections.abc", "Callable");
                let params = method
                    .params
                    .iter()
                    .filter(|parameter| !parameter.ty.hidden())
                    .map(|parameter| annotation(this, &parameter.ty, Dir::Out, imports))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut result = annotation(this, &method.ret.ok, Dir::ProxyResult, imports);
                if method.kind == ProxyCall::Await {
                    imports.add("collections.abc", "Awaitable");
                    result = format!("Awaitable[{result}]");
                }
                return format!("Callable[[{params}], {result}]");
            }
            // A dotted class, with its type arguments when it is generic (`aio.Chan[..]`).
            let name = imports.annotation(host, "");
            if *generic {
                imports.add("typing", "Any");
                format!("{name}[Any]")
            } else {
                name
            }
        }
        Ty::Source { class, .. } | Ty::Sink { class, .. } | Ty::Duplex { class, .. } => {
            class.clone()
        }
        Ty::HostSource(item) => {
            imports.add("collections.abc", "AsyncIterable");
            format!("AsyncIterable[{}]", annotation(this, item, dir, imports))
        }
        Ty::Error(_) => "BaseException".into(),
        Ty::HostDuplex { output, .. } => {
            imports.add("collections.abc", "AsyncIterable");
            format!("AsyncIterable[{}]", annotation(this, output, dir, imports))
        }
        Ty::Handle { class, .. } | Ty::Instance { class, .. } | Ty::Shared { class, .. } => {
            imports.add(".._core", class);
            class.clone()
        }
        Ty::Native { classes, .. } => classes
            .iter()
            .map(|class| {
                let (module, name) = class.rsplit_once('.').expect("a host class is dotted");
                // A `_core` class: relative, so a family stub never imports the class it defines.
                let module =
                    if module == lk_agents_bindgen::DEFAULT_MODULE { ".._core" } else { module };
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
