//! The `#[host(record)]` records: their pyo3 conversions, free functions in the registration shell
//! (`lk-agents-py/src/generated/records.rs`, a module per model module: the ABI crate holds no host
//! code), and their models, written into `_generated/<module>.py` as frozen keyword-only
//! dataclasses.
//!
//! A record crosses as an instance of its model, never a dict: a struct is one class; a
//! data-variant enum one class per variant, the class being the tag, plus a union alias; a
//! unit-only enum a `Literal` of its strings. Reading an object of another class, or `None` where
//! the field is not optional, fails naming `Record.field`; a declared default is the model's, so it
//! applies to an omitted keyword only.

mod emit;
mod models;
pub(crate) use emit::emit;
pub use models::{link_legacy, merge, python};

use std::collections::BTreeSet;

use crate::decl::{
    Declarations, Field, HostForm, Record, RecordBody, RecordDecl, Ty, Variant, VariantShape,
};
use crate::python::Out;

const PACKAGE: &str = "livekit.agents._generated";

const HEADER: &str = "//! GENERATED FILE -- DO NOT EDIT.\n//!\n//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the\n//! `#[host(record)]` declarations of `lk-agents-abi`.\n";

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
            let body = record_rust(r, decls)?;
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
                    && matches!(&o.body, RecordBody::Variants(vs) if vs.iter().any(|v| matches!(v.shape, VariantShape::Unit) && v.host_type.is_none()))
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

const NOT_A: &str = "\n/// `obj` is not an instance of the model(s) `expected` names.\nfn not_a(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, expected: &str) -> ::pyo3::PyErr {\n    use ::pyo3::types::{PyAnyMethods as _, PyTypeMethods as _};\n    let got = obj.get_type().name().map(|n| n.to_string()).unwrap_or_default();\n    ::pyo3::exceptions::PyTypeError::new_err(format!(\"expected {expected}, got {got}\"))\n}\n";

/// Whether `ty` holds a declared record: it then crosses by the records' conversion functions.
pub(crate) fn holds_record(ty: &Ty) -> bool {
    match ty {
        Ty::Record(r) => {
            matches!(r.host, HostForm::Record { .. } | HostForm::WireStrs)
        }
        Ty::Optional(t) | Ty::Maybe(t) | Ty::List(t) | Ty::Set(t) => holds_record(t),
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

fn record_rust(r: &RecordDecl, decls: &Declarations) -> Result<String, String> {
    if r.transparent {
        return transparent_rust(r, decls);
    }
    let module = format!("{PACKAGE}.{}", r.module);
    let rust = &r.rust;
    let mut s = String::new();
    let mut classes = Vec::new();
    let (extract, emit) = match &r.body {
        RecordBody::Struct(fields) if r.typed_dict => typed_dict_rust(r, fields)?,
        RecordBody::Struct(fields) => {
            classes.push(r.name.clone());
            let class = class_ref(&module, &r.name);
            // An emit-only record's fields need no reader (an emitted failure has none).
            let reads =
                if r.extract { struct_reads(&r.name, fields, "obj")? } else { String::new() };
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
                    construct(&class_ref(&module, &r.name), "()", "Some(&kwargs)")
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
                    construct(&class_ref(&module, &v.name), "()", "None")
                } else {
                    format!(
                        "{{\n                let kwargs = ::pyo3::types::PyDict::new(py);\n{sets}                {}\n            }}",
                        construct(&class_ref(&module, &v.name), "()", "Some(&kwargs)")
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
                    format!("{}?", construct(&class_ref(&module, &r.name), "(value,)", "None"))
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

/// `class(*args, **kwargs)`: a model built through HostInvoker's Python entry.
fn construct(class: &str, args: &str, kwargs: &str) -> String {
    format!("crate::host::construct({class}.as_any(), {args}, {kwargs})")
}

/// A `typed_dict` record's extract and emit bodies: keys of a plain dict, an optional field's key
/// absent (or `None`) when it holds nothing.
fn typed_dict_rust(r: &RecordDecl, fields: &[Field]) -> Result<(String, String), String> {
    let mut reads = String::new();
    let mut pattern = Vec::new();
    let mut sets = String::new();
    for f in fields {
        if f.factory.is_some() || f.omitted.is_some() || f.skip {
            return Err(format!(
                "{}: {}: a typed_dict key is read or defaulted",
                r.rust, f.rust_name
            ));
        }
        let (key, path) = (f.host_name(), format!("{}.{}", r.name, f.host_name()));
        let read = match &f.ty {
            Ty::Optional(inner) => format!(
                "match obj.get_item({key:?})? {{ Some(v) if !v.is_none() => Some(({})?), _ => None }}",
                read(inner, "(&v)", &path)?
            ),
            // A defaulted key reads absent and `None` alike as its default.
            ty if let Some(default) = &f.default => format!(
                "match obj.get_item({key:?})? {{ Some(v) if !v.is_none() => ({})?, _ => {} }}",
                read(ty, "(&v)", &path)?,
                crate::adapter::rust_literal(default, ty)
            ),
            ty => format!(
                "{{ let v = obj.get_item({key:?})?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err({path:?}))?; ({})? }}",
                read(ty, "(&v)", &path)?
            ),
        };
        reads.push_str(&format!(
            "            {}: {read},
",
            f.rust_name
        ));
        pattern.push(f.rust_name.clone());
        sets.push_str(&match &f.ty {
            Ty::Optional(inner) => format!(
                "        if let Some(x) = {} {{
            dict.set_item({key:?}, {})?;
        }}
",
                f.rust_name,
                emit(inner, "x")?
            ),
            ty => format!(
                "        dict.set_item({key:?}, {})?;
",
                emit(ty, &f.rust_name)?
            ),
        });
    }
    let extract = format!(
        "        let obj = obj.cast::<::pyo3::types::PyDict>().map_err(|_| super::not_a(obj, \"dict\"))?;\n        Ok(Self {{\n{reads}        }})\n"
    );
    let emit = format!(
        "        let Self {{ {} }} = self;\n        let dict = ::pyo3::types::PyDict::new(py);\n{sets}        Ok(dict.into_any())\n",
        pattern.join(", ")
    );
    Ok((extract, emit))
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
        Ty::Maybe(t) => format!(
            "if ::lk_agents_abi::python::is_not_given({v})? {{ Ok(::lk_agents_abi::Maybe::NotGiven) }} else {{ ({}).map(::lk_agents_abi::Maybe::Given) }}",
            read(t, v, path)?
        ),
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
        Ty::Record(Record { host: HostForm::Record { module, name }, .. }) => {
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
        Ty::Proxy { .. } | Ty::Native { .. } => {
            crate::adapter::from_object(v, ty, &format!("{path:?}"), path)?
        }
        other => return Err(format!("{path}: `{}` is not a record field", other.rust())),
    })
}

/// A transparent record's conversions: each value crosses as itself; a unit emits `None` or its
/// declared host marker. Reading tries the variants in declaration order by the value's class:
/// a record's model class, a literal record's `str`, else its Python type (`bool` apart from `int`,
/// a `float` taking an `int` too).
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
                if let Some((module, name)) = &v.host_type {
                    if r.extract {
                        reads.push_str(&format!(
                            "    if obj.is_instance(&obj.py().import({module:?})?.getattr({name:?})?)? {{\n        return Ok({rust}::{});\n    }}\n",
                            v.rust_name
                        ));
                    }
                    let class = format!("py.import({module:?})?.getattr({name:?})?");
                    arms.push_str(&format!(
                        "        {rust}::{} => {}?,\n",
                        v.rust_name,
                        construct(&class, "()", "None")
                    ));
                    continue;
                }
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
        arms.push_str(&format!("        {rust}::{}(x) => {},\n", v.rust_name, emit(&f.ty, "x")?));
        if !r.extract {
            continue;
        }
        let (kind, test) = match &f.ty {
            Ty::Record(Record { rust: held, host: HostForm::Record { .. }, .. })
                if literal(held) =>
            {
                ("str", "obj.is_instance_of::<::pyo3::types::PyString>()".to_owned())
            }
            Ty::Record(Record {
                host: HostForm::Record { module, name } | HostForm::Model { module, name },
                ..
            }) => (
                "",
                format!(
                    "obj.is_instance(&obj.py().import({:?})?.getattr({name:?})?)?",
                    format!("{PACKAGE}.{module}")
                ),
            ),
            Ty::Record(Record { host: HostForm::Annotation(text), .. }) => {
                let names = crate::class_stub::annotation_names(text);
                let [(range, path)] = names.as_slice() else {
                    return Err(format!("{}::{}: an opaque transparent member names one host class", r.rust, v.rust_name));
                };
                if range.start != 0 || range.end != text.len() {
                    return Err(format!("{}::{}: an opaque transparent member names one host class", r.rust, v.rust_name));
                }
                let Some((module, name)) = path.rsplit_once('.') else {
                    return Err(format!("{}::{}: an opaque transparent member names a qualified host class", r.rust, v.rust_name));
                };
                ("", format!("obj.is_instance(&obj.py().import({module:?})?.getattr({name:?})?)?"))
            }
            Ty::Tuple(_) => ("tuple", "obj.is_instance_of::<::pyo3::types::PyTuple>()".into()),
            Ty::Str => ("str", "obj.is_instance_of::<::pyo3::types::PyString>()".into()),
            Ty::Bool => ("bool", "obj.is_instance_of::<::pyo3::types::PyBool>()".into()),
            Ty::Int(_) => ("number", "(obj.is_instance_of::<::pyo3::types::PyInt>() && !obj.is_instance_of::<::pyo3::types::PyBool>())".into()),
            Ty::Float | Ty::Duration => ("number", "(obj.is_instance_of::<::pyo3::types::PyFloat>() || (obj.is_instance_of::<::pyo3::types::PyInt>() && !obj.is_instance_of::<::pyo3::types::PyBool>()))".into()),
            Ty::Map(..)
            | Ty::OrderedMap(..)
            | Ty::JsonObject
            | Ty::Record(Record { host: HostForm::Dict, .. }) => {
                ("dict", "obj.is_instance_of::<::pyo3::types::PyDict>()".into())
            }
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
