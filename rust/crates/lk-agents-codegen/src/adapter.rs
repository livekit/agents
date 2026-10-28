//! The pyo3 adapters of the declared classes, written into the registration shell
//! (`lk-agents-py/src/generated/`): one module per ABI subject, plus `mod.rs` with the
//! `Implementations` trait the shell implements to select each class's implementation. A class
//! the shell selects nothing for, or an implementation missing a declared method, fails to compile.
//!
//! Each parameter crosses by its declared kind: a kind pyo3 extracts itself (scalars, strings,
//! lists, declared records) is a typed parameter; JSON, JSON objects and host proxies are taken as
//! objects and converted here with the parameter's name as the error path.

use crate::decl::{Class, Declarations, Form, Literal, Method, MethodKind, Param, Ty};
use crate::python::Out;

pub const DIR: &str = "rust/crates/lk-agents-py/src/generated";

/// The first line is the marker the placement and file-length checks skip generated files by.
const HEADER: &str = "//! GENERATED FILE -- DO NOT EDIT.\n//!\n//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the\n//! `#[host(class)]` declarations of `lk-agents-abi`.\n";

pub fn emit(decls: &Declarations) -> Result<Vec<Out>, String> {
    let mut families: Vec<&str> = decls.classes.iter().map(|c| c.family.as_str()).collect();
    families.dedup();
    let mut sorted = families.clone();
    sorted.sort();
    sorted.dedup();
    if sorted.len() != families.len() {
        return Err("classes of one subject are read together".into());
    }
    let mut out = Vec::new();
    let mut root = format!(
        "{HEADER}//!\n//! The declared classes' pyo3 adapters, one module per ABI subject.\n\n"
    );
    root.push_str("use pyo3::prelude::*;\n\n");
    for family in &sorted {
        root.push_str(&format!("pub mod {family};\n"));
    }
    root.push_str("\n/// The implementation of each declared class. The registration shell implements it.\npub trait Implementations: 'static {\n");
    for c in &decls.classes {
        root.push_str(&format!("    type {}: {};\n", c.rust_name, c.trait_path));
    }
    root.push_str("}\n\n/// Adds every declared class to `_core`.\npub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {\n");
    for c in &decls.classes {
        if c.form == Form::Functions {
            for f in &c.methods {
                root.push_str(&format!(
                    "    m.add_function(wrap_pyfunction!({}::{}, m)?)?;\n",
                    c.family, f.name
                ));
            }
            continue;
        }
        root.push_str(&format!("    m.add_class::<{}::{}>()?;\n", c.family, c.name));
        for (class, _) in sources(c) {
            root.push_str(&format!("    m.add_class::<{}::{class}>()?;\n", c.family));
        }
    }
    root.push_str("    Ok(())\n}\n");
    out.push(Out { path: "mod.rs".into(), text: root });
    for family in &sorted {
        let classes: Vec<&Class> = decls.classes.iter().filter(|c| c.family == *family).collect();
        let mut text =
            format!("{HEADER}//!\n//! The `{family}` classes.\n\nuse pyo3::prelude::*;\n");
        for c in classes {
            text.push('\n');
            text.push_str(&class(c)?);
            for (name, item) in sources(c) {
                text.push('\n');
                text.push_str(&source_class(&name, &item));
            }
        }
        out.push(Out { path: format!("{family}.rs"), text });
    }
    Ok(out)
}

fn class(c: &Class) -> Result<String, String> {
    let name = &c.name;
    let mut s = format!("// Declared in {}.\n", c.source);
    s.push_str(&format!(
        "type {name}Impl = <crate::Selected as super::Implementations>::{};\n",
        c.rust_name
    ));
    if c.form == Form::Functions {
        for m in &c.methods {
            s.push('\n');
            s.push_str(&method(c, m)?);
        }
        return Ok(s);
    }
    s.push('\n');
    s.push_str(&doc(&c.doc, ""));
    let (frozen, held) = match c.form {
        Form::MutableClass => ("", format!("{name}Impl")),
        _ => ("frozen, ", format!("::std::sync::Arc<{name}Impl>")),
    };
    s.push_str(&format!(
        "#[pyclass({frozen}name = \"{name}\", module = \"livekit.agents._core\")]\npub struct {name}({held});\n\n#[pymethods]\nimpl {name} {{\n"
    ));
    let methods = c.methods.iter().map(|m| method(c, m)).collect::<Result<Vec<_>, _>>()?;
    s.push_str(&methods.join("\n"));
    s.push_str("}\n");
    for tr in &c.native {
        s.push_str(&format!(
            "\n// Wherever the host passes a `{tr}`, an instance of this class is itself.\n::lk_agents_abi::python::inventory::submit! {{\n    ::lk_agents_abi::python::HostNative {{\n        trait_id: || ::std::any::TypeId::of::<dyn {tr}>(),\n        read: |obj| {{\n            let this = obj.cast::<{name}>().ok()?;\n            let native: ::std::sync::Arc<dyn {tr}> = this.get().0.clone();\n            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)\n        }},\n    }}\n}}\n"
        ));
    }
    Ok(s)
}

fn method(c: &Class, m: &Method) -> Result<String, String> {
    let at = format!("{}::{}", c.trait_path, m.rust_name);
    let call = format!("<{}Impl as {}>::{}", c.name, c.trait_path, m.rust_name);
    // Functions are module items, methods are indented in their `impl`.
    let ind = if m.kind == MethodKind::Function { "" } else { "    " };
    let mut s = doc(&m.doc, ind);
    match m.kind {
        MethodKind::New => s.push_str(&format!("{ind}#[new]\n")),
        MethodKind::Getter => s.push_str(&format!("{ind}#[getter]\n")),
        MethodKind::Function => s.push_str("#[pyfunction]\n"),
        MethodKind::Call | MethodKind::Await => {}
    }
    if m.name != m.rust_name && m.kind != MethodKind::Function {
        s.push_str(&format!("{ind}#[pyo3(name = \"{}\")]\n", m.name));
    }
    if !m.params.is_empty() {
        s.push_str(&format!("{ind}#[pyo3(signature = ({}))]\n", signature(&m.params)));
    }
    let mut args = Vec::new();
    let mut prelude = String::new();
    for p in &m.params {
        let (ty, convert) = input(&at, p)?;
        args.push(format!("{}: {ty}", p.name));
        if let Some(convert) = convert {
            prelude.push_str(&format!("{ind}    let {} = {convert};\n", p.name));
        }
    }
    let receiver = match m.kind {
        MethodKind::New | MethodKind::Function => None,
        _ if m.mutable => Some("&mut self"),
        _ => Some("&self"),
    };
    let passed: Vec<String> = m
        .params
        .iter()
        .map(|p| if p.by_ref { format!("&{}", p.name) } else { p.name.clone() })
        .collect();
    let err = m.ret.err.as_ref().map(|e| format!(".map_err({})?", e.with)).unwrap_or_default();
    let (ret_ty, wrap) = output(&at, c, &m.ret.ok)?;
    let fn_name = match m.kind {
        MethodKind::New => "new".to_owned(),
        MethodKind::Function => m.name.clone(),
        _ => m.rust_name.clone(),
    };
    match m.kind {
        MethodKind::Await => {
            s.push_str(&format!(
                "    fn {fn_name}<'py>({}) -> PyResult<Bound<'py, PyAny>> {{\n        ::lk_agents_abi::python::check_pid()?;\n{prelude}        let this = ::std::sync::Arc::clone(&self.0);\n        ::lk_agents_abi::python::future_into_py(py, async move {{\n            let out = {call}(&this, {})\n                .await{err};\n            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({}, py))\n        }})\n    }}\n",
                ["&self".to_owned(), "py: Python<'py>".to_owned()].into_iter().chain(args).collect::<Vec<_>>().join(", "),
                passed.join(", "),
                wrap.replace("{}", "out").replace("{py}", "py")
            ));
        }
        _ => {
            let needs_py = ret_ty.contains("'py");
            let generics = if needs_py { "<'py>" } else { "" };
            let this = match m.kind {
                MethodKind::New | MethodKind::Function => None,
                _ if m.mutable => Some("&mut self.0".to_owned()),
                _ => Some("&self.0".to_owned()),
            };
            let all_args = receiver
                .map(str::to_owned)
                .into_iter()
                .chain(needs_py.then(|| "py: Python<'py>".to_owned()))
                .chain(args)
                .collect::<Vec<_>>()
                .join(", ");
            let value = format!(
                "{call}({}){err}",
                this.into_iter().chain(passed).collect::<Vec<_>>().join(", ")
            );
            s.push_str(&format!(
                "{ind}fn {fn_name}{generics}({all_args}) -> PyResult<{ret_ty}> {{\n{ind}    ::lk_agents_abi::python::check_pid()?;\n{prelude}{ind}    let out = {value};\n{ind}    Ok({})\n{ind}}}\n",
                wrap.replace("{}", "out").replace("{py}", "py")
            ));
        }
    }
    Ok(s)
}

/// `///` lines at `indent`.
fn doc(lines: &[String], indent: &str) -> String {
    lines
        .iter()
        .map(|l| format!("{indent}///{}{l}\n", if l.is_empty() { "" } else { " " }))
        .collect()
}

/// The stream classes a class's methods hand out, once each.
pub fn sources(c: &Class) -> Vec<(String, Ty)> {
    let mut out: Vec<(String, Ty)> = Vec::new();
    for m in &c.methods {
        if let Ty::Source { class, item } = &m.ret.ok
            && !out.iter().any(|(n, _)| n == class)
        {
            out.push((class.clone(), (**item).clone()));
        }
    }
    out
}

/// A pull source as the host consumes it: the runtime's `NativeSource` does the work.
fn source_class(name: &str, item: &Ty) -> String {
    format!(
        "/// One stream of `{item}` items: `recv()` resolves to the next, `None` at the end.\n#[pyclass(frozen, name = \"{name}\", module = \"livekit.agents._core\")]\npub struct {name}(::lk_agents_runtime::source::NativeSource<{item}>);\n\n#[pymethods]\nimpl {name} {{\n    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n        ::lk_agents_abi::python::check_pid()?;\n        self.0.recv(py)\n    }}\n\n    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n        ::lk_agents_abi::python::check_pid()?;\n        self.0.aclose(py)\n    }}\n}}\n",
        item = item.rust()
    )
}

/// pyo3's signature: positional parameters, then `*` before the first keyword-only one.
fn signature(params: &[Param]) -> String {
    let mut parts = Vec::new();
    let mut star = false;
    for p in params {
        if p.kw_only && !star {
            parts.push("*".to_owned());
            star = true;
        }
        match &p.default {
            None => parts.push(p.name.clone()),
            Some(d) => parts.push(format!("{} = {}", p.name, rust_literal(d, &p.ty))),
        }
    }
    parts.join(", ")
}

fn rust_literal(d: &Literal, ty: &Ty) -> String {
    match d {
        Literal::None => "None".into(),
        Literal::Bool(b) => b.to_string(),
        Literal::Int(i)
            if matches!(ty, Ty::Float) || matches!(ty, Ty::Optional(t) if **t == Ty::Float) =>
        {
            format!("{i}.0")
        }
        Literal::Int(i) => i.to_string(),
        Literal::Float(f) => format!("{f:?}"),
        Literal::Str(s) => format!("{s:?}.to_owned()"),
    }
}

/// A kind pyo3 extracts and emits by itself: the conversion is the type's own.
fn native(ty: &Ty) -> bool {
    match ty {
        Ty::Bool | Ty::Int(_) | Ty::Float | Ty::Str | Ty::Record(_) => true,
        Ty::Optional(t) | Ty::List(t) => native(t),
        Ty::Tuple(ts) => ts.iter().all(native),
        _ => false,
    }
}

/// The adapter's parameter type, and the conversion (an expression of the parameter) when it is
/// taken as an object.
fn input(at: &str, p: &Param) -> Result<(String, Option<String>), String> {
    if native(&p.ty) {
        return Ok((p.ty.rust(), None));
    }
    let path = format!("{:?}", p.name);
    match &p.ty {
        Ty::Optional(inner) => {
            if p.default.as_ref().is_some_and(|d| *d != Literal::None) {
                return Err(format!(
                    "{at}({}): an object-converted parameter defaults only to None",
                    p.name
                ));
            }
            let convert = from_object("v", inner, &path, at)?;
            Ok((
                "Option<Bound<'_, PyAny>>".into(),
                Some(format!("{}.map(|v| {convert}).transpose()?", p.name)),
            ))
        }
        ty => {
            if p.default.is_some() {
                return Err(format!(
                    "{at}({}): an object-converted parameter has no default",
                    p.name
                ));
            }
            Ok((
                "Bound<'_, PyAny>".into(),
                Some(format!("{}?", from_object(&p.name, ty, &path, at)?)),
            ))
        }
    }
}

/// `PyResult<T>` of the object `v`.
fn from_object(v: &str, ty: &Ty, path: &str, at: &str) -> Result<String, String> {
    Ok(match ty {
        Ty::Json => format!("::lk_agents_abi::python::value_from_py(&{v}, {path})"),
        Ty::JsonObject => format!(
            "match ::lk_agents_abi::python::value_from_py(&{v}, {path})? {{ ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!({path}, \": expected a dict\"))) }}"
        ),
        Ty::Proxy { rust, .. } => {
            format!("<dyn {rust} as ::lk_agents_abi::python::HostProxy>::from_host(&{v}, None)")
        }
        Ty::List(inner) => format!(
            "{v}.try_iter()?.map(|item| {{ let item = item?; {} }}).collect::<PyResult<Vec<_>>>()",
            from_object("item", inner, path, at)?
        ),
        other if native(other) => format!("{v}.extract::<{}>()", other.rust()),
        other => return Err(format!("{at}: `{}` has no input conversion", other.rust())),
    })
}

/// The adapter's return type and the expression of `{}` (the trait's value) it returns.
fn output(at: &str, c: &Class, ty: &Ty) -> Result<(String, String), String> {
    if native(ty) {
        return Ok((ty.rust(), "{}".into()));
    }
    Ok(match ty {
        Ty::Unit => ("()".into(), "{}".into()),
        Ty::This if c.form == Form::MutableClass => (c.name.clone(), format!("{}({{}})", c.name)),
        Ty::This => (c.name.clone(), format!("{}(::std::sync::Arc::new({{}}))", c.name)),
        Ty::Json => {
            ("Bound<'py, PyAny>".into(), "::lk_agents_abi::python::value_to_py({py}, &{})?".into())
        }
        Ty::JsonObject => (
            "Bound<'py, PyAny>".into(),
            "::lk_agents_abi::python::extra_to_py({py}, &{})?.into_any()".into(),
        ),
        Ty::Source { class, .. } => (
            class.clone(),
            format!(
                "{class}(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new({{}})))"
            ),
        ),
        other => return Err(format!("{at}: `{}` has no output conversion", other.rust())),
    })
}
