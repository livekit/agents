//! The `#[host(record)]` records: their pyo3 conversions, free functions in the registration shell
//! (`lk-agents-py/src/generated/records.rs`, a module per model module: the ABI crate holds no host
//! code), and their models, written into `_generated/<module>.py` as frozen keyword-only
//! dataclasses. The declared proxies' `HostProxy` bridges for the in-place exports are written into
//! the ABI crate (`lk-agents-abi/src/python/generated/`) until those exports are declarations.
//!
//! A record crosses as an instance of its model, never a dict: a struct is one class; a
//! data-variant enum one class per variant, the class being the tag, plus a union alias; a
//! unit-only enum a `Literal` of its strings. Reading an object of another class, or `None` where
//! the field is not optional, fails naming `Record.field`; a declared default is the model's, so it
//! applies to an omitted keyword only.

mod emit;
mod legacy;
mod models;
pub(crate) use emit::emit;
pub use models::{link_legacy, merge, python};

use std::collections::BTreeSet;

use crate::decl::{
    Declarations, Field, HostForm, Record, RecordBody, RecordDecl, Ty, Variant, VariantShape,
};
use crate::python::Out;

pub const DIR: &str = "rust/crates/lk-agents-abi/src/python/generated";

const PACKAGE: &str = "livekit.agents._generated";

const HEADER: &str = "//! GENERATED FILE -- DO NOT EDIT.\n//!\n//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the\n//! `#[host(record)]` declarations of `lk-agents-abi`.\n";

/// What stays in ABI until the in-place exports are declarations: the declared proxies'
/// `HostProxy` bridges and the in-place records' trait impls, `mod.rs` and one module per subject.
pub fn rust(decls: &Declarations) -> Result<Vec<Out>, String> {
    let in_place = || decls.records.iter().filter(|r| r.in_place);
    let proxies = || decls.proxies.iter().filter(|proxy| legacy::proxy_in_place(&proxy.trait_path));
    let mut families: Vec<&str> =
        proxies().map(|p| p.family.as_str()).chain(in_place().map(|r| r.family.as_str())).collect();
    families.sort();
    families.dedup();
    let mut root = format!("{HEADER}\n");
    for family in &families {
        root.push_str(&format!("mod {family};\n"));
    }
    if in_place().next().is_some() {
        root.push_str(NOT_A);
    }
    let mut out = vec![Out { path: "mod.rs".into(), text: root }];
    for family in families {
        let mut text =
            format!("{HEADER}//!\n//! The `{family}` proxy bridges and in-place records.\n");
        if in_place().any(|r| r.family == family) {
            text.push_str("\nuse pyo3::prelude::*;\n");
        }
        for r in in_place().filter(|r| r.family == family) {
            text.push_str(&format!("\n// Declared in {}.\n", r.source));
            text.push_str(&record_rust(r, true, decls)?);
        }
        for p in proxies().filter(|p| p.family == family) {
            text.push_str(&format!("\n// Declared in {}.\n", p.source));
            text.push_str(&crate::proxy::legacy_bridge(p));
        }
        out.push(Out { path: format!("{family}.rs"), text });
    }
    Ok(out)
}

/// The conversions of every record, `records.rs` of the registration shell's generated adapters:
/// one module per model module, `<record>_from_py` and `<record>_to_py` in each.
pub fn conversions(decls: &Declarations) -> Result<Out, String> {
    let mut modules: Vec<&str> = decls.records.iter().map(|r| r.module.as_str()).collect();
    modules.sort();
    modules.dedup();
    let mut text = format!(
        "{HEADER}//!\n//! Each record crosses as an instance of its generated model.\n{NOT_A}"
    );
    for module in modules {
        text.push_str(&format!("\npub mod {module} {{\n    use pyo3::prelude::*;\n"));
        for r in decls.records.iter().filter(|r| r.module == module) {
            text.push_str(&format!("\n    // Declared in {}.\n", r.source));
            let body = if r.in_place { in_place_rust(r) } else { record_rust(r, false, decls)? };
            for line in body.lines() {
                text.push_str(&if line.is_empty() {
                    "\n".to_owned()
                } else {
                    format!("    {line}\n")
                });
            }
        }
        text.push_str("}\n");
    }
    Ok(Out { path: "records.rs".into(), text })
}

/// Records the in-place exports still convert by pyo3 trait (an in-place export's parameter, a
/// `#[derive(Host)]` record's field, handwritten runtime glue): their conversions stay trait impls
/// written into ABI, which the shell's functions call. The list never grows by itself: an entry
/// is only added by the commit that declares a record which was a `#[derive(Host)]` record (its
/// trait impls move from the derive to the generated module), and leaves when its in-place
/// consumers are declarations (`check_in_place`).
pub const IN_PLACE: &[&str] = &[
    "::lk_agents_abi::EvaluationRecord",
    "::lk_agents_abi::telemetry::report::RecordingOptions",
    "::lk_agents_abi::telemetry::report::SpanUpdateWire",
    "::lk_agents_abi::a2a::client::Message",
    "::lk_agents_abi::agent::run::RunError",
    "::lk_agents_abi::LogRecordWire",
    "::lk_agents_abi::llm::tool_executor::ToolOutcome",
    "::lk_agents_abi::llm::tool_executor::ToolReport",
    "::lk_agents_abi::telemetry::loop_monitor::LoopMonitorEmission",
    "::lk_agents_abi::worker::endpoint::EndpointRequest",
    "::lk_agents_abi::worker::ipc::ProcessCompletion",
    "::lk_agents_abi::worker::worker::RegisteredInfo",
];

/// A record field defaulting to `None` whose type is a declared record: that record is
/// transparent with a unit variant, the host's `None`.
pub fn check_none_defaults(decls: &Declarations) -> Result<(), String> {
    for r in &decls.records {
        for f in fields(&r.body) {
            let (Some(crate::decl::Literal::None), Ty::Record(held)) = (&f.default, &f.ty) else {
                continue;
            };
            let none = decls.records.iter().any(|o| {
                o.rust == held.rust
                    && o.transparent
                    && matches!(&o.body, RecordBody::Variants(vs) if vs.iter().any(|v| matches!(v.shape, VariantShape::Unit)))
            });
            if !none {
                return Err(format!(
                    "{}: {}.{}: default None is not a value of `{}`",
                    r.source,
                    r.name,
                    f.host_name(),
                    held.rust
                ));
            }
        }
    }
    Ok(())
}

/// Marks the in-place records, and the records their fields hold, wherever a type names them.
pub fn mark_in_place(decls: &mut Declarations) {
    let mut held: BTreeSet<String> = IN_PLACE
        .iter()
        .filter(|rust| decls.records.iter().any(|r| r.rust == **rust))
        .map(|rust| (*rust).to_owned())
        .collect();
    loop {
        let before = held.len();
        let mut found = Vec::new();
        for r in decls.records.iter().filter(|r| held.contains(&r.rust)) {
            for f in fields(&r.body) {
                records_in(&f.ty, &mut found);
            }
        }
        held.extend(found);
        if held.len() == before {
            break;
        }
    }
    for r in &mut decls.records {
        r.in_place = held.contains(&r.rust);
        for f in fields_mut(&mut r.body) {
            mark(&mut f.ty, &held);
        }
    }
    for c in &mut decls.classes {
        let sinks = c.sinks.iter_mut().flat_map(|s| s.methods.iter_mut());
        for m in c.methods.iter_mut().chain(sinks) {
            m.params.iter_mut().for_each(|p| mark(&mut p.ty, &held));
            mark(&mut m.ret.ok, &held);
        }
        for s in &mut c.sinks {
            mark(&mut s.item, &held);
        }
    }
    for p in &mut decls.proxies {
        for m in &mut p.methods {
            m.params.iter_mut().for_each(|q| mark(&mut q.ty, &held));
            mark(&mut m.ret.ok, &held);
        }
    }
}

/// Every [`IN_PLACE`] entry is a declared record, and one the list did not hold at `HEAD` was a
/// `#[derive(Host)]` record there: the list only trades a derive's trait impls for generated ones.
pub fn check_in_place(root: &std::path::Path, decls: &Declarations) -> Result<(), String> {
    legacy::check_proxies(decls)?;
    let at_head = |path: &str| -> Option<String> {
        let out = std::process::Command::new("git")
            .args(["show", &format!("HEAD:{path}")])
            .current_dir(root)
            .output()
            .ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let held = at_head("rust/crates/lk-agents-codegen/src/records.rs").unwrap_or_default();
    for rust in IN_PLACE {
        let Some(r) = decls.records.iter().find(|r| r.rust == *rust) else {
            return Err(format!("records::IN_PLACE: `{rust}` is not a declared record"));
        };
        if held.contains(&format!("\"{rust}\"")) {
            continue;
        }
        let ident = rust.rsplit("::").next().unwrap_or(rust);
        let derived = at_head(&r.source)
            .and_then(|text| syn::parse_file(&text).ok())
            .is_some_and(|file| derived_at(&file.items, ident));
        if !derived {
            return Err(format!(
                "records::IN_PLACE: `{rust}` is new, and was not a #[derive(Host)] record at HEAD: an in-place consumer of a declared record converts through its generated functions"
            ));
        }
    }
    Ok(())
}

/// `ident` is a `#[derive(Host)]` struct or enum among `items` (inline modules included).
pub(crate) fn derived_at(items: &[syn::Item], ident: &str) -> bool {
    items.iter().any(|item| match item {
        syn::Item::Struct(s) => s.ident == ident && crate::decl::resolve::derives_host(&s.attrs),
        syn::Item::Enum(e) => e.ident == ident && crate::decl::resolve::derives_host(&e.attrs),
        syn::Item::Mod(m) => m.content.as_ref().is_some_and(|(_, items)| derived_at(items, ident)),
        _ => false,
    })
}

fn records_in(ty: &Ty, out: &mut Vec<String>) {
    match ty {
        Ty::Record(r) if matches!(r.host, HostForm::Record { .. }) => out.push(r.rust.clone()),
        Ty::Optional(t) | Ty::List(t) | Ty::Set(t) | Ty::HostSource(t) => records_in(t, out),
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => {
            records_in(k, out);
            records_in(t, out);
        }
        Ty::Tuple(ts) => ts.iter().for_each(|t| records_in(t, out)),
        _ => {}
    }
}

fn mark(ty: &mut Ty, held: &BTreeSet<String>) {
    match ty {
        Ty::Record(r) => r.in_place = held.contains(&r.rust),
        Ty::Optional(t) | Ty::List(t) | Ty::Set(t) | Ty::HostSource(t) => mark(t, held),
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => {
            mark(k, held);
            mark(t, held);
        }
        Ty::Tuple(ts) => ts.iter_mut().for_each(|t| mark(t, held)),
        Ty::Source { item, .. } | Ty::Sink { item, .. } => mark(item, held),
        Ty::HostDuplex { input, output } | Ty::Duplex { input, output, .. } => {
            mark(input, held);
            mark(output, held);
        }
        _ => {}
    }
}

pub(crate) fn fields(body: &RecordBody) -> Vec<&Field> {
    match body {
        RecordBody::Struct(fields) => fields.iter().collect(),
        RecordBody::Variants(variants) => variants
            .iter()
            .flat_map(|v| match &v.shape {
                VariantShape::Unit | VariantShape::Tuple(None) => Vec::new(),
                VariantShape::Tuple(Some(f)) => vec![f],
                VariantShape::Named(fields) => fields.iter().collect(),
            })
            .collect(),
        RecordBody::Literals(_) => Vec::new(),
    }
}

fn fields_mut(body: &mut RecordBody) -> Vec<&mut Field> {
    match body {
        RecordBody::Struct(fields) => fields.iter_mut().collect(),
        RecordBody::Variants(variants) => variants
            .iter_mut()
            .flat_map(|v| match &mut v.shape {
                VariantShape::Unit | VariantShape::Tuple(None) => Vec::new(),
                VariantShape::Tuple(Some(f)) => vec![f],
                VariantShape::Named(fields) => fields.iter_mut().collect(),
            })
            .collect(),
        RecordBody::Literals(_) => Vec::new(),
    }
}

/// An in-place record's shell functions: calls of its ABI trait impls.
fn in_place_rust(r: &RecordDecl) -> String {
    let (rust, snake) = (&r.rust, cell(&r.name).to_lowercase());
    let mut s = String::new();
    if r.extract {
        s.push_str(&format!(
            "\npub fn {snake}_from_py(obj: &Bound<'_, PyAny>) -> PyResult<{rust}> {{\n    obj.extract::<{rust}>()\n}}\n"
        ));
    }
    if r.emit {
        s.push_str(&format!(
            "\npub fn {snake}_to_py<'py>(py: Python<'py>, value: {rust}) -> PyResult<Bound<'py, PyAny>> {{\n    ::pyo3::IntoPyObjectExt::into_bound_py_any(value, py)\n}}\n"
        ));
    }
    s
}

const NOT_A: &str = "\n/// `obj` is not an instance of the model(s) `expected` names.\nfn not_a(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, expected: &str) -> ::pyo3::PyErr {\n    use ::pyo3::types::{PyAnyMethods as _, PyTypeMethods as _};\n    let got = obj.get_type().name().map(|n| n.to_string()).unwrap_or_default();\n    ::pyo3::exceptions::PyTypeError::new_err(format!(\"expected {expected}, got {got}\"))\n}\n";

/// Whether `ty` holds a declared record: it then crosses by the records' conversion functions.
pub(crate) fn holds_record(ty: &Ty) -> bool {
    match ty {
        Ty::Record(r) => {
            matches!(r.host, HostForm::Record { .. } | HostForm::WireStrs) && !r.in_place
        }
        Ty::Optional(t) | Ty::List(t) | Ty::Set(t) => holds_record(t),
        Ty::Map(k, t) => holds_record(k) || holds_record(t),
        Ty::OrderedMap(..) => true,
        Ty::Tuple(ts) => ts.iter().any(holds_record),
        _ => false,
    }
}

/// The path of a declared record's conversions, less their `_from_py` / `_to_py` suffix.
pub(crate) fn conversion(module: &str, name: &str) -> String {
    format!("crate::generated::records::{module}::{}", cell(name).to_lowercase())
}

fn record_rust(r: &RecordDecl, as_trait: bool, decls: &Declarations) -> Result<String, String> {
    if r.transparent {
        if as_trait {
            return Err(format!("{}: a transparent record is not converted in place", r.rust));
        }
        return transparent_rust(r, decls);
    }
    let module = format!("{PACKAGE}.{}", r.module);
    let rust = &r.rust;
    let mut s = String::new();
    let mut classes = Vec::new();
    let (extract, emit) = match &r.body {
        RecordBody::Struct(fields) => {
            classes.push(r.name.clone());
            let class = class_ref(&module, &r.name);
            let reads = struct_reads(&r.name, fields, "obj")?;
            let check = format!(
                "        if !obj.is_instance({class})? {{\n            return Err(super::not_a(obj, {name:?}));\n        }}\n",
                name = r.name
            );
            let extract = if r.omitted().is_empty() {
                format!("{check}        Ok(Self {{\n{reads}        }})\n")
            } else {
                // An omitted field reads as `None`, then is computed from the record read.
                let mut first = String::new();
                let mut then = String::new();
                for (f, _) in r.omitted() {
                    let name = &f.rust_name;
                    let path = format!("{}.{}", r.name, f.host_name());
                    first.push_str(&format!(
                        "        let {name} = ::lk_agents_abi::python::attr_with(obj, {:?}, {:?}, |v| if v.is_none() {{ Ok(None) }} else {{ ({}).map(Some) }})?;\n",
                        r.name,
                        f.host_name(),
                        read(&f.ty, "v", &path)?
                    ));
                    let compute = f
                        .omitted
                        .as_deref()
                        .expect("omitted")
                        .replace("Self::", &format!("{}::", r.rust));
                    then.push_str(&format!(
                        "        record.{name} = match {name} {{ Some(v) => v, None => {compute}(&record) }};\n"
                    ));
                }
                format!(
                    "{check}{first}        let mut record = Self {{\n{reads}        }};\n{then}        Ok(record)\n"
                )
            };
            // An extract-only record may hold what never returns to the host (a proxy).
            let emit = if r.emit {
                let (pattern, sets) = struct_emits(fields)?;
                format!(
                    "        let Self {{ {pattern} }} = self;\n        let kwargs = ::pyo3::types::PyDict::new(py);\n{sets}        {}\n",
                    construct(&class_ref(&module, &r.name), "()", "Some(&kwargs)", as_trait)
                )
            } else {
                String::new()
            };
            (extract, emit)
        }
        RecordBody::Variants(variants) => {
            let mut extract = String::new();
            let mut emit = String::from("        match self {\n");
            for v in variants {
                classes.push(v.name.clone());
                let class = class_ref(&module, &v.name);
                let (read, arm, sets) = variant(v)?;
                extract.push_str(&format!(
                    "        if obj.is_instance({class})? {{\n            return Ok(Self::{}{read});\n        }}\n",
                    v.rust_name
                ));
                let call = if sets.is_empty() {
                    construct(&class_ref(&module, &v.name), "()", "None", as_trait)
                } else {
                    format!(
                        "{{\n                let kwargs = ::pyo3::types::PyDict::new(py);\n{sets}                {}\n            }}",
                        construct(&class_ref(&module, &v.name), "()", "Some(&kwargs)", as_trait)
                    )
                };
                emit.push_str(&format!("            Self::{}{arm} => {call},\n", v.rust_name));
            }
            let names: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
            extract.push_str(&format!("        Err(super::not_a(obj, {:?}))\n", names.join(" | ")));
            emit.push_str("        }\n");
            (extract, emit)
        }
        RecordBody::Literals(values) => {
            let arms: String = values
                .iter()
                .map(|(v, host)| format!("            {host:?} => Ok(Self::{v}),\n"))
                .collect();
            let expected: Vec<String> = values.iter().map(|(_, h)| format!("{h:?}")).collect();
            let extract = format!(
                "        let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;\n        match &*value {{\n{arms}            other => Err(::pyo3::exceptions::PyValueError::new_err(format!(\"{name}: {{other:?}} is not one of {expected}\"))),\n        }}\n",
                name = r.name,
                expected = expected.join(", ").replace('"', "'")
            );
            let arms: String = values
                .iter()
                .map(|(v, host)| format!("            Self::{v} => {host:?},\n"))
                .collect();
            let value = match r.enumeration {
                true => {
                    classes.push(r.name.clone());
                    format!(
                        "{}?",
                        construct(&class_ref(&module, &r.name), "(value,)", "None", as_trait)
                    )
                }
                false => "::pyo3::types::PyString::new(py, value).into_any()".to_owned(),
            };
            let emit = format!(
                "        let value = match self {{\n{arms}        }};\n        Ok({value})\n"
            );
            (extract, emit)
        }
    };
    for c in &classes {
        s.push_str(&format!("static {}: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =\n    ::pyo3::sync::PyOnceLock::new();\n", cell(c)));
    }
    if as_trait {
        if r.extract {
            s.push_str(&format!(
                "\nimpl<'a, 'py> FromPyObject<'a, 'py> for {rust} {{\n    type Error = PyErr;\n\n    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {{\n        let obj = &*obj;\n{}{extract}    }}\n}}\n",
                if classes.is_empty() { "" } else { "        let py = obj.py();\n" }
            ));
        }
        if r.emit {
            s.push_str(&format!(
                "\nimpl<'py> IntoPyObject<'py> for {rust} {{\n    type Target = PyAny;\n    type Output = Bound<'py, PyAny>;\n    type Error = PyErr;\n\n    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n{emit}    }}\n}}\n"
            ));
        }
        // A legacy record holding it emits its fields by reference.
        if r.legacy && r.emit && r.clone {
            s.push_str(&format!(
                "\nimpl<'py> IntoPyObject<'py> for &{rust} {{\n    type Target = PyAny;\n    type Output = Bound<'py, PyAny>;\n    type Error = PyErr;\n\n    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n        ::std::clone::Clone::clone(self).into_pyobject(py)\n    }}\n}}\n"
            ));
        }
        // A legacy union picks it by its constant discriminant, read off the model.
        if let (true, true, [(key, value)]) = (r.legacy, r.extract, &r.constants[..]) {
            s.push_str(&format!(
                "\nimpl ::lk_agents_abi::python::HostReadTag for {rust} {{\n    const KEY: &'static str = {key:?};\n    const TAG: &'static str = {value:?};\n    const ITEMS: bool = false;\n}}\n"
            ));
        }
        return Ok(s);
    }
    let snake = cell(&r.name).to_lowercase();
    if r.extract {
        let extract = extract
            .replace("Ok(Self {", &format!("Ok({rust} {{"))
            .replace("= Self {", &format!("= {rust} {{"))
            .replace("Self::", &format!("{rust}::"));
        s.push_str(&format!(
            "\npub fn {snake}_from_py(obj: &Bound<'_, PyAny>) -> PyResult<{rust}> {{\n{}{extract}}}\n",
            if classes.is_empty() { "" } else { "    let py = obj.py();\n" }
        ));
    }
    if r.emit {
        let emit = emit
            .replace("let Self {", &format!("let {rust} {{"))
            .replace("} = self;", "} = value;")
            .replace("match self {", "match value {")
            .replace("Self::", &format!("{rust}::"));
        s.push_str(&format!(
            "\npub fn {snake}_to_py<'py>(py: Python<'py>, value: {rust}) -> PyResult<Bound<'py, PyAny>> {{\n{emit}}}\n"
        ));
    }
    Ok(s)
}

fn cell(class: &str) -> String {
    let mut out = String::new();
    for (i, c) in class.chars().filter(|c| *c != '.').enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_uppercase());
    }
    out
}

fn class_ref(module: &str, class: &str) -> String {
    match class.split_once('.') {
        None => format!("{}.import(py, {module:?}, {class:?})?", cell(class)),
        // A nested model, read off its owner.
        Some((owner, name)) => format!(
            "{}.get_or_try_init(py, || -> PyResult<_> {{ Ok(py.import({module:?})?.getattr({owner:?})?.getattr({name:?})?.cast_into::<::pyo3::types::PyType>()?.unbind()) }})?.bind(py)",
            cell(class)
        ),
    }
}

/// `class(*args, **kwargs)`: a model built in place. The registration shell's conversions build
/// it through HostInvoker's Python entry; an in-place ABI conversion (`IN_PLACE`) still calls it.
fn construct(class: &str, args: &str, kwargs: &str, as_trait: bool) -> String {
    match as_trait {
        true if (args, kwargs) == ("()", "None") => format!("{class}.call0()"),
        true if kwargs == "None" => format!("{class}.call1({args})"),
        true => format!("{class}.call({args}, {kwargs})"),
        false => format!("crate::host::construct({class}.as_any(), {args}, {kwargs})"),
    }
}

/// `field: read,` lines of a struct literal read off `obj`.
fn struct_reads(owner: &str, fields: &[Field], obj: &str) -> Result<String, String> {
    let mut s = String::new();
    for f in fields.iter().filter(|f| f.modeled()) {
        let read = match f.omitted {
            Some(_) => "::std::default::Default::default()".to_owned(),
            None => attr_read(owner, f, obj)?,
        };
        s.push_str(&format!("            {}: {read},\n", f.rust_name));
    }
    Ok(s)
}

fn attr_read(owner: &str, f: &Field, obj: &str) -> Result<String, String> {
    let path = format!("{owner}.{}", f.host_name());
    Ok(format!(
        "::lk_agents_abi::python::attr_with({obj}, {owner:?}, {:?}, |v| {})?",
        f.host_name(),
        read(&f.ty, "v", &path)?
    ))
}

/// The destructuring pattern and the `kwargs.set_item` lines of a struct's fields.
fn struct_emits(fields: &[Field]) -> Result<(String, String), String> {
    let mut pattern = Vec::new();
    let mut sets = String::new();
    for f in fields {
        if f.skip {
            pattern.push(format!("{}: _", f.rust_name));
            continue;
        }
        pattern.push(f.rust_name.clone());
        sets.push_str(&set(f, &f.rust_name, "        ")?);
    }
    Ok((pattern.join(", "), sets))
}

fn set(f: &Field, value: &str, indent: &str) -> Result<String, String> {
    Ok(format!("{indent}kwargs.set_item({:?}, {})?;\n", f.host_name(), emit(&f.ty, value)?))
}

/// A variant's read suffix (`{ .. }` / `(..)` / nothing), its match-arm pattern and its sets.
fn variant(v: &Variant) -> Result<(String, String, String), String> {
    let indent = "                ";
    Ok(match &v.shape {
        VariantShape::Unit => (String::new(), String::new(), String::new()),
        VariantShape::Tuple(None) => (String::new(), "(_)".into(), String::new()),
        VariantShape::Tuple(Some(f)) => (
            format!("({})", attr_read(&v.name, f, "obj")?),
            format!("({})", f.rust_name),
            set(f, &f.rust_name, indent)?,
        ),
        VariantShape::Named(fields) => {
            let reads: Vec<String> = fields
                .iter()
                .filter(|f| f.modeled())
                .map(|f| Ok(format!("{}: {}", f.rust_name, attr_read(&v.name, f, "obj")?)))
                .collect::<Result<_, String>>()?;
            let mut pattern = Vec::new();
            let mut sets = String::new();
            for f in fields {
                if f.skip {
                    pattern.push(format!("{}: _", f.rust_name));
                } else {
                    pattern.push(f.rust_name.clone());
                    sets.push_str(&set(f, &f.rust_name, indent)?);
                }
            }
            (format!(" {{ {} }}", reads.join(", ")), format!(" {{ {} }}", pattern.join(", ")), sets)
        }
    })
}

/// `PyResult<T>` of the host value `v: &Bound<PyAny>`; `path` names it in errors.
fn read(ty: &Ty, v: &str, path: &str) -> Result<String, String> {
    Ok(match ty {
        Ty::Bool => format!("{v}.extract::<bool>()"),
        Ty::Int(i) => {
            format!("::lk_agents_abi::python::number_from_py::<{}>({v}, {path:?})", i.rust())
        }
        Ty::Float => format!("::lk_agents_abi::python::number_from_py::<f64>({v}, {path:?})"),
        Ty::Duration => format!(
            "::lk_agents_abi::python::number_from_py::<f64>({v}, {path:?}).and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!(\"{path}: {{e}}\"))))"
        ),
        Ty::Str => format!("{v}.extract::<::std::string::String>()"),
        Ty::Json => format!("::lk_agents_abi::python::value_from_py({v}, {path:?})"),
        Ty::JsonObject => format!(
            "match ::lk_agents_abi::python::value_from_py({v}, {path:?})? {{ ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(\"{path}: expected a dict\")) }}"
        ),
        Ty::Optional(t) => {
            format!("if {v}.is_none() {{ Ok(None) }} else {{ ({}).map(Some) }}", read(t, v, path)?)
        }
        Ty::Tuple(_) | Ty::OrderedMap(..) => {
            crate::adapter::from_object(v, ty, &format!("{path:?}"), path)?
        }
        Ty::List(t) if **t == Ty::Int(crate::decl::Int::U8) => {
            crate::adapter::from_object(v, ty, &format!("{path:?}"), path)?
        }
        Ty::List(t) => format!(
            "{v}.cast::<::pyo3::types::PyList>()?.iter().map(|item| {{ let item = &item; {} }}).collect::<PyResult<Vec<_>>>()",
            read(t, "item", path)?
        ),
        Ty::Map(k, t) => format!(
            "{v}.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| {{ let (key, item) = (&key, &item); Ok(({}?, {}?)) }}).collect::<PyResult<::std::collections::BTreeMap<_, _>>>()",
            read(k, "key", path)?,
            read(t, "item", path)?
        ),
        Ty::Set(t) => format!(
            "{v}.try_iter()?.map(|item| {{ let item = &item?; {} }}).collect::<PyResult<::std::collections::BTreeSet<_>>>()",
            read(t, "item", path)?
        ),
        Ty::Record(Record { host: HostForm::Record { module, name }, in_place: false, .. }) => {
            format!("{}_from_py({v})", conversion(module, name))
        }
        Ty::Record(r) => {
            format!("{v}.extract::<{}>().map_err(::std::convert::Into::<PyErr>::into)", r.rust)
        }
        Ty::Proxy { boxed: true, .. } => {
            return Err(format!(
                "{path}: a record holds a shared `Arc<dyn ..>` proxy, not an owned one"
            ));
        }
        Ty::Proxy { declared: Some(_), .. } | Ty::Native { .. } => {
            crate::adapter::from_object(v, ty, &format!("{path:?}"), path)?
        }
        Ty::Proxy { rust, .. } => {
            format!("<dyn {rust} as ::lk_agents_abi::python::HostProxy>::from_host({v}, None)")
        }
        other => return Err(format!("{path}: `{}` is not a record field", other.rust())),
    })
}

/// A transparent record's conversions: each variant's value crosses as itself (a unit variant as
/// `None`); reading tries the variants in declaration order by the value's class (a record's model
/// class, a literal record's `str`, else its Python type: `bool` apart from `int`, a `float` taking
/// an `int` too).
fn transparent_rust(r: &RecordDecl, decls: &Declarations) -> Result<String, String> {
    let RecordBody::Variants(variants) = &r.body else {
        return Err(format!("{}: a transparent record is an enum", r.rust));
    };
    let (rust, snake) = (&r.rust, cell(&r.name).to_lowercase());
    let literal = |held: &str| {
        decls.records.iter().any(|o| o.rust == held && matches!(o.body, RecordBody::Literals(_)))
    };
    let mut reads = String::new();
    let mut arms = String::new();
    let mut kinds: Vec<String> = Vec::new();
    for v in variants {
        let f = match &v.shape {
            VariantShape::Tuple(Some(f)) => f,
            VariantShape::Unit => {
                reads.push_str(&format!(
                    "    if obj.is_none() {{\n        return Ok({rust}::{});\n    }}\n",
                    v.rust_name
                ));
                arms.push_str(&format!(
                    "        {rust}::{} => py.None().into_bound(py),\n",
                    v.rust_name
                ));
                continue;
            }
            _ => {
                return Err(format!(
                    "{}::{}: a transparent variant carries its value",
                    r.rust, v.rust_name
                ));
            }
        };
        let (kind, test) = match &f.ty {
            Ty::Record(Record { rust: held, host: HostForm::Record { .. }, .. })
                if literal(held) =>
            {
                ("str", "obj.is_instance_of::<::pyo3::types::PyString>()".to_owned())
            }
            Ty::Record(Record { host: HostForm::Record { module, name }, .. }) => (
                "",
                format!(
                    "obj.is_instance(&obj.py().import({:?})?.getattr({name:?})?)?",
                    format!("{PACKAGE}.{module}")
                ),
            ),
            Ty::Tuple(_) => ("tuple", "obj.is_instance_of::<::pyo3::types::PyTuple>()".into()),
            Ty::Str => ("str", "obj.is_instance_of::<::pyo3::types::PyString>()".into()),
            Ty::Bool => ("bool", "obj.is_instance_of::<::pyo3::types::PyBool>()".into()),
            Ty::Int(_) => ("number", "(obj.is_instance_of::<::pyo3::types::PyInt>() && !obj.is_instance_of::<::pyo3::types::PyBool>())".into()),
            Ty::Float | Ty::Duration => ("number", "(obj.is_instance_of::<::pyo3::types::PyFloat>() || (obj.is_instance_of::<::pyo3::types::PyInt>() && !obj.is_instance_of::<::pyo3::types::PyBool>()))".into()),
            Ty::Map(..) | Ty::OrderedMap(..) | Ty::JsonObject => ("dict", "obj.is_instance_of::<::pyo3::types::PyDict>()".into()),
            Ty::List(_) => ("list", "obj.is_instance_of::<::pyo3::types::PyList>()".into()),
            other => {
                return Err(format!(
                    "{}::{}: `{}` is not told apart by its class in a transparent record",
                    r.rust,
                    v.rust_name,
                    other.rust()
                ));
            }
        };
        // A model class is its own kind; the Python types are shared (`float` takes an `int`).
        let kind = if kind.is_empty() { test.clone() } else { kind.to_owned() };
        if kinds.contains(&kind) {
            return Err(format!("{}::{}: two variants share a class", r.rust, v.rust_name));
        }
        kinds.push(kind);
        reads.push_str(&format!(
            "    if {test} {{\n        return Ok({rust}::{}({}?));\n    }}\n",
            v.rust_name,
            read(&f.ty, "obj", &format!("{}.{}", r.name, v.rust_name))?
        ));
        arms.push_str(&format!("        {rust}::{}(x) => {},\n", v.rust_name, emit(&f.ty, "x")?));
    }
    let mut s = String::new();
    if r.extract {
        s.push_str(&format!(
            "\npub fn {snake}_from_py(obj: &Bound<'_, PyAny>) -> PyResult<{rust}> {{\n{reads}    Err(super::not_a(obj, {:?}))\n}}\n",
            r.name
        ));
    }
    if r.emit {
        s.push_str(&format!(
            "\npub fn {snake}_to_py<'py>(py: Python<'py>, value: {rust}) -> PyResult<Bound<'py, PyAny>> {{\n    Ok(match value {{\n{arms}    }})\n}}\n"
        ));
    }
    Ok(s)
}
