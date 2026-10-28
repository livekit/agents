//! The `#[host(record)]` records: their pyo3 conversions, written into the ABI crate
//! (`lk-agents-abi/src/python/generated/`, one module per ABI subject), and their models, written
//! into `_generated/<module>.py` as frozen keyword-only dataclasses. The declared proxies'
//! `HostProxy` bridges for the in-place exports go into the same modules.
//!
//! A record crosses as an instance of its model, never a dict: a struct is one class; a
//! data-variant enum one class per variant, the class being the tag, plus a union alias; a
//! unit-only enum a `Literal` of its strings. Reading an object of another class, or `None` where
//! the field is not optional, fails naming `Record.field`; a declared default is the model's, so it
//! applies to an omitted keyword only.

use crate::class_stub::{Imports, docstring, py_literal};
use crate::decl::{
    Declarations, Field, HostForm, RecordBody, RecordDecl, Ty, Variant, VariantShape,
};
use crate::python::Out;

pub const DIR: &str = "rust/crates/lk-agents-abi/src/python/generated";

const PACKAGE: &str = "livekit.agents._generated";

const HEADER: &str = "//! GENERATED FILE -- DO NOT EDIT.\n//!\n//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the\n//! `#[host(record)]` declarations of `lk-agents-abi`.\n";

/// The conversions: `mod.rs` and one module per subject.
pub fn rust(decls: &Declarations) -> Result<Vec<Out>, String> {
    let mut families: Vec<&str> = decls
        .records
        .iter()
        .map(|r| r.family.as_str())
        .chain(decls.proxies.iter().map(|p| p.family.as_str()))
        .collect();
    families.sort();
    families.dedup();
    let mut root =
        format!("{HEADER}//!\n//! Each record crosses as an instance of its generated model.\n\n");
    for family in &families {
        root.push_str(&format!("mod {family};\n"));
    }
    root.push_str("\n/// `obj` is not an instance of the model(s) `expected` names.\nfn not_a(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, expected: &str) -> ::pyo3::PyErr {\n    use ::pyo3::types::{PyAnyMethods as _, PyTypeMethods as _};\n    let got = obj.get_type().name().map(|n| n.to_string()).unwrap_or_default();\n    ::pyo3::exceptions::PyTypeError::new_err(format!(\"expected {expected}, got {got}\"))\n}\n");
    let mut out = vec![Out { path: "mod.rs".into(), text: root }];
    for family in families {
        let mut text = format!("{HEADER}//!\n//! The `{family}` records.\n");
        if decls.records.iter().any(|r| r.family == family) {
            text.push_str("\nuse pyo3::prelude::*;\n");
        }
        for r in decls.records.iter().filter(|r| r.family == family) {
            text.push_str(&format!("\n// Declared in {}.\n", r.source));
            text.push_str(&record_rust(r)?);
        }
        for p in decls.proxies.iter().filter(|p| p.family == family) {
            text.push_str(&format!("\n// Declared in {}.\n", p.source));
            text.push_str(&crate::proxy::legacy_bridge(p));
        }
        out.push(Out { path: format!("{family}.rs"), text });
    }
    Ok(out)
}

fn record_rust(r: &RecordDecl) -> Result<String, String> {
    let module = format!("{PACKAGE}.{}", r.module);
    let rust = &r.rust;
    let mut s = String::new();
    let mut classes = Vec::new();
    let (extract, emit) = match &r.body {
        RecordBody::Struct(fields) => {
            classes.push(r.name.clone());
            let class = class_ref(&module, &r.name);
            let reads = struct_reads(&r.name, fields, "obj")?;
            let extract = format!(
                "        if !obj.is_instance({class})? {{\n            return Err(super::not_a(obj, {name:?}));\n        }}\n        Ok(Self {{\n{reads}        }})\n",
                name = r.name
            );
            // An extract-only record may hold what never returns to the host (a proxy).
            let emit = if r.emit {
                let (pattern, sets) = struct_emits(fields)?;
                format!(
                    "        let Self {{ {pattern} }} = self;\n        let kwargs = ::pyo3::types::PyDict::new(py);\n{sets}        {class_py}.call((), Some(&kwargs))\n",
                    class_py = class_ref(&module, &r.name)
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
                    format!("{}.call0()", class_ref(&module, &v.name))
                } else {
                    format!(
                        "{{\n                let kwargs = ::pyo3::types::PyDict::new(py);\n{sets}                {}.call((), Some(&kwargs))\n            }}",
                        class_ref(&module, &v.name)
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
            let emit = format!(
                "        let value = match self {{\n{arms}        }};\n        Ok(::pyo3::types::PyString::new(py, value).into_any())\n"
            );
            (extract, emit)
        }
    };
    for c in &classes {
        s.push_str(&format!("static {}: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =\n    ::pyo3::sync::PyOnceLock::new();\n", cell(c)));
    }
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
    Ok(s)
}

fn cell(class: &str) -> String {
    let mut out = String::new();
    for (i, c) in class.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_uppercase());
    }
    out
}

fn class_ref(module: &str, class: &str) -> String {
    format!("{}.import(py, {module:?}, {class:?})?", cell(class))
}

/// `field: read,` lines of a struct literal read off `obj`.
fn struct_reads(owner: &str, fields: &[Field], obj: &str) -> Result<String, String> {
    let mut s = String::new();
    for f in fields.iter().filter(|f| !f.skip) {
        s.push_str(&format!("            {}: {},\n", f.rust_name, attr_read(owner, f, obj)?));
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
                .filter(|f| !f.skip)
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
        Ty::Record(r) => {
            format!("{v}.extract::<{}>().map_err(::std::convert::Into::<PyErr>::into)", r.rust)
        }
        Ty::Proxy { boxed: true, .. } => {
            return Err(format!(
                "{path}: a record holds a shared `Arc<dyn ..>` proxy, not an owned one"
            ));
        }
        Ty::Proxy { rust, .. } => {
            format!("<dyn {rust} as crate::python::HostProxy>::from_host({v}, None)")
        }
        other => return Err(format!("{path}: `{}` is not a record field", other.rust())),
    })
}

/// A `Bound<'py, PyAny>` expression of the owned value `x` (fallible steps use `?`).
fn emit(ty: &Ty, x: &str) -> Result<String, String> {
    Ok(match ty {
        Ty::Json => format!("::lk_agents_abi::python::value_to_py(py, &{x})?"),
        Ty::JsonObject => format!("::lk_agents_abi::python::extra_to_py(py, &{x})?.into_any()"),
        Ty::Optional(t) => format!(
            "match {x} {{ None => py.None().into_bound(py), Some(x) => {} }}",
            emit(t, "x")?
        ),
        Ty::List(t) => format!(
            "::pyo3::types::PyList::new(py, {x}.into_iter().map(|x| PyResult::Ok({})).collect::<PyResult<Vec<_>>>()?)?.into_any()",
            emit(t, "x")?
        ),
        Ty::Map(k, t) => format!(
            "{{ let d = ::pyo3::types::PyDict::new(py); for (k, x) in {x} {{ d.set_item({}, {})?; }} d.into_any() }}",
            emit(k, "k")?,
            emit(t, "x")?
        ),
        Ty::Set(t) => format!(
            "::pyo3::types::PyFrozenSet::new(py, {x}.into_iter().map(|x| PyResult::Ok({})).collect::<PyResult<Vec<_>>>()?)?.into_any()",
            emit(t, "x")?
        ),
        Ty::Bool | Ty::Int(_) | Ty::Float | Ty::Str | Ty::Record(_) => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({x}, py)?")
        }
        Ty::Duration => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({x}.as_secs_f64(), py)?")
        }
        other => return Err(format!("`{}` is not a record field", other.rust())),
    })
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
                RecordBody::Variants(vs) => std::iter::once(r.name.clone())
                    .chain(vs.iter().map(|v| v.name.clone()))
                    .collect::<Vec<_>>(),
                _ => vec![r.name.clone()],
            })
            .collect();
        for r in &records {
            body.push_str(&model(r, module, &own, &mut runtime, &mut typing));
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

fn model(
    r: &RecordDecl,
    module: &str,
    own: &[String],
    runtime: &mut Imports,
    typing: &mut Imports,
) -> String {
    // An emit-only record's derived models (`#[derive(Host)]`) reach the host as dicts.
    let out = r.emit && !r.extract;
    let mut class = |name: &str, doc: &[String], fields: &[&Field]| {
        let mut s =
            format!("\n\n@dataclass(frozen=True, kw_only=True, slots=True)\nclass {name}:\n");
        s.push_str(&docstring(doc, "    "));
        let fields: Vec<&&Field> = fields.iter().filter(|f| !f.skip).collect();
        if fields.is_empty() {
            if doc.is_empty() {
                s.push_str("    pass\n");
            }
            return s;
        }
        if !doc.is_empty() {
            s.push('\n');
        }
        for f in fields {
            let default =
                f.default.as_ref().map(|d| format!(" = {}", py_literal(d))).unwrap_or_default();
            let ann = annotation(&f.ty, module, own, out, typing);
            s.push_str(&format!("    {}: {ann}{default}\n", f.host_name()));
            s.push_str(&docstring(&f.doc, "    "));
        }
        s
    };
    match &r.body {
        RecordBody::Struct(fields) => {
            runtime.add("dataclasses", "dataclass");
            let mut s = class(&r.name, &r.doc, &fields.iter().collect::<Vec<_>>());
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
                s.push_str(&class(&v.name, &v.doc, &fields));
            }
            let names: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
            s.push_str(&format!("\n\n{}\n", union_alias(&r.name, &names)));
            s.push_str(&docstring(&r.doc, ""));
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
        Ty::List(t) => format!("list[{}]", annotation(t, module, own, out, typing)),
        Ty::Map(k, t) => format!(
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
            HostForm::Annotation(text) => text.clone(),
        },
        // Refused by the resolver (`check_record_field`).
        other => unreachable!("`{}` is not a record field", other.rust()),
    }
}
