//! The pyo3 adapters of the declared classes, and the declared proxies ([`crate::proxy`]), written
//! into the registration shell (`lk-agents-py/src/generated/`): one module per ABI subject, plus `mod.rs` with the
//! `Implementations` trait the shell implements to select each class's implementation. A class
//! the shell selects nothing for, or an implementation missing a declared method, fails to compile.
//!
//! Each parameter crosses by its declared kind: a kind pyo3 extracts itself (scalars, strings,
//! lists, declared records) is a typed parameter; JSON, JSON objects and host proxies are taken as
//! objects and converted here with the parameter's name as the error path.

use crate::decl::{
    Class, Declarations, Form, HostForm, Literal, Method, MethodKind, Param, Proxy, Record,
    RecordDecl, Sink, Ty,
};
use crate::python::Out;

pub const DIR: &str = "rust/crates/lk-agents-py/src/generated";

/// The first line is the marker the placement and file-length checks skip generated files by.
const HEADER: &str = "//! GENERATED FILE -- DO NOT EDIT.\n//!\n//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the\n//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.\n";

pub fn emit(decls: &Declarations) -> Result<Vec<Out>, String> {
    let mut sorted: Vec<&str> = decls
        .classes
        .iter()
        .map(|c| c.family.as_str())
        .chain(decls.proxies.iter().map(|p| p.family.as_str()))
        .chain(decls.records.iter().filter(|r| has_functions(r)).map(|r| r.family.as_str()))
        .collect();
    sorted.sort();
    sorted.dedup();
    let mut out = Vec::new();
    let mut root = format!(
        "{HEADER}//!\n//! The declared classes' pyo3 adapters, one module per ABI subject.\n\n"
    );
    root.push_str("use pyo3::prelude::*;\n\n");
    for family in &sorted {
        root.push_str(&format!("pub mod {family};\n"));
    }
    // The declared records' conversions, which every family's adapters call.
    root.push_str("pub mod records;\n");
    root.push_str("\n/// The implementation of each declared class. The registration shell implements it.\npub trait Implementations: 'static {\n");
    for c in &decls.classes {
        // An instance of another class is that class's implementation.
        let mut bounds = Vec::new();
        for (assoc, class) in &c.instances {
            let target =
                decls.classes.iter().find(|t| t.name == *class).ok_or_else(|| {
                    format!("{}: `{class}` is not a declared class", c.trait_path)
                })?;
            bounds.push(format!("{assoc} = Self::{}", target.rust_name));
        }
        let bounds =
            if bounds.is_empty() { String::new() } else { format!("<{}>", bounds.join(", ")) };
        root.push_str(&format!("    type {}: {}{bounds};\n", c.rust_name, c.trait_path));
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
        for sink in &c.sinks {
            root.push_str(&format!("    m.add_class::<{}::{}>()?;\n", c.family, sink.class));
        }
    }
    for p in &decls.proxies {
        for (class, _) in proxy_sources(decls, p) {
            root.push_str(&format!("    m.add_class::<{}::{class}>()?;\n", p.family));
        }
    }
    for r in &decls.records {
        for f in functions(r) {
            root.push_str(&format!(
                "    m.add_function(wrap_pyfunction!({}::{f}, m)?)?;\n",
                r.family
            ));
        }
    }
    root.push_str("    Ok(())\n}\n");
    root.push_str(&native_to_host(decls)?);
    out.push(Out { path: "mod.rs".into(), text: root });
    for family in &sorted {
        let classes: Vec<&Class> = decls.classes.iter().filter(|c| c.family == *family).collect();
        let mut text = format!(
            "{HEADER}//!\n//! The `{family}` classes.\n\n// A host signature takes the parameters it declares.\n#![allow(clippy::too_many_arguments)]\n\nuse pyo3::prelude::*;\n"
        );
        for c in classes {
            text.push('\n');
            text.push_str(&class(decls, c)?);
            for (name, item) in sources(c) {
                text.push('\n');
                text.push_str(&source_class(&name, &item)?);
            }
            for (sink, output) in duplexes(c) {
                text.push('\n');
                text.push_str(&duplex_class(decls, c, sink, &output)?);
            }
        }
        for p in decls.proxies.iter().filter(|p| p.family == *family) {
            text.push('\n');
            text.push_str(&crate::proxy::proxy(p)?);
            for (name, item) in proxy_sources(decls, p) {
                text.push('\n');
                text.push_str(&source_class(&name, &item)?);
            }
        }
        for r in decls.records.iter().filter(|r| r.persisted && r.family == *family) {
            text.push('\n');
            text.push_str(&persistence(r));
        }
        for r in decls.records.iter().filter(|r| r.family == *family) {
            for (f, function) in r.omitted() {
                text.push('\n');
                text.push_str(&format!(
                    "// Declared in {source}.\n\n/// `{name}.{field}` when the host leaves it out.\n#[pyfunction]\npub fn {function}<'py>(py: Python<'py>, record: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {{\n    let record = {conv}_from_py(&record)?;\n    Ok({value})\n}}\n",
                    source = r.source,
                    name = r.name,
                    field = f.host_name(),
                    conv = crate::records::conversion(&r.module, &r.name),
                    value = crate::records::emit(&f.ty, &format!("record.{}", f.rust_name))?,
                ));
            }
        }
        out.push(Out { path: format!("{family}.rs"), text });
    }
    Ok(out)
}

/// The native traits handed to the host (a proxy call's argument, a record's field, a class
/// method's result), each with the declared classes implementing it.
fn native_arguments(decls: &Declarations) -> std::collections::BTreeMap<&str, Vec<&Class>> {
    fn native(ty: &Ty) -> Option<&str> {
        match ty {
            Ty::Native { rust, .. } => Some(rust),
            Ty::Optional(t) | Ty::List(t) => native(t),
            _ => None,
        }
    }
    let mut out = std::collections::BTreeMap::new();
    let params = decls.proxies.iter().flat_map(|p| &p.methods).flat_map(|m| &m.params);
    let fields =
        decls.records.iter().filter(|r| r.emit).flat_map(|r| crate::records::fields(&r.body));
    let results = decls.classes.iter().flat_map(|c| &c.methods).map(|m| &m.ret.ok);
    let handed = params.map(|q| &q.ty).chain(fields.map(|f| &f.ty)).chain(results);
    for rust in handed.filter_map(native) {
        out.entry(rust).or_insert_with(|| {
            decls.classes.iter().filter(|c| c.native.iter().any(|t| t == rust)).collect()
        });
    }
    out
}

/// `NativeToHost` for each native trait a proxy call passes: the implementation crosses as the
/// declared class it is.
fn native_to_host(decls: &Declarations) -> Result<String, String> {
    let arguments = native_arguments(decls);
    if arguments.is_empty() {
        return Ok(String::new());
    }
    let mut s = String::from(
        "\n/// A native implementation a proxy call passes, handed to the host as its declared class.\npub(crate) trait NativeToHost {\n    fn to_host<'py>(py: Python<'py>, native: ::std::sync::Arc<Self>) -> PyResult<Bound<'py, PyAny>>;\n}\n",
    );
    for (rust, classes) in arguments {
        if classes.is_empty() {
            return Err(format!(
                "`{rust}` is passed to the host, but no declared class implements it"
            ));
        }
        // Each class in turn; the last refusal names the trait.
        let mut body = format!(
            "Err(::pyo3::exceptions::PyTypeError::new_err(\"this `{rust}` is no declared class\"))"
        );
        for (i, c) in classes.iter().enumerate().rev() {
            let rest = if i + 1 == classes.len() { "_" } else { "native" };
            body = format!(
                "match native.downcast::<<crate::Selected as Implementations>::{}>() {{\n            Ok(native) => Ok(Py::new(py, {}::{}(native))?.into_bound(py).into_any()),\n            Err({rest}) => {body},\n        }}",
                c.rust_name, c.family, c.name
            );
        }
        s.push_str(&format!(
            "\nimpl NativeToHost for dyn {rust} {{\n    fn to_host<'py>(py: Python<'py>, native: ::std::sync::Arc<Self>) -> PyResult<Bound<'py, PyAny>> {{\n        let native: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = native;\n        {body}\n    }}\n}}\n"
        ));
    }
    Ok(s)
}

fn class(decls: &Declarations, c: &Class) -> Result<String, String> {
    let name = &c.name;
    let mut s = format!("// Declared in {}.\n", c.source);
    s.push_str(&format!(
        "type {name}Impl = <crate::Selected as super::Implementations>::{};\n",
        c.rust_name
    ));
    s.push_str(&numeric_extractors(c)?);
    if c.form == Form::Functions {
        for m in &c.methods {
            s.push('\n');
            s.push_str(&method(decls, c, m)?);
        }
        return Ok(s);
    }
    s.push('\n');
    s.push_str(&doc(&c.doc, ""));
    let (frozen, held) = match c.form {
        Form::MutableClass => ("", format!("{name}Impl")),
        _ => ("frozen, ", format!("::std::sync::Arc<{name}Impl>")),
    };
    // Associated class factories can construct a wrapper from another generated family.
    // Held fields are crate-wide: another family wraps or reads instances of this class.
    let visibility = "pub(crate) ";
    let host_module = &c.host_module;
    let subclass = if c.subclass { "subclass, " } else { "" };
    s.push_str(&format!(
        "#[pyclass({frozen}{subclass}name = \"{name}\", module = \"{host_module}\")]\npub struct {name}({visibility}{held});\n\n#[pymethods]\nimpl {name} {{\n"
    ));
    let methods = c.methods.iter().map(|m| method(decls, c, m)).collect::<Result<Vec<_>, _>>()?;
    s.push_str(&methods.join("\n"));
    s.push_str("}\n");
    for tr in &c.native {
        s.push_str(&format!(
            "\n// Wherever the host passes a `{tr}`, an instance of this class is itself.\n::lk_agents_abi::python::inventory::submit! {{\n    ::lk_agents_abi::python::HostNative {{\n        trait_id: || ::std::any::TypeId::of::<dyn {tr}>(),\n        read: |obj| {{\n            let this = obj.cast::<{name}>().ok()?;\n            let native: ::std::sync::Arc<dyn {tr}> = this.get().0.clone();\n            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)\n        }},\n    }}\n}}\n"
        ));
    }
    Ok(s)
}

/// PyO3's `from_py_with` accepts a function path, so extractors live beside the adapters.
fn extractor_name(c: &Class, m: &Method, p: &Param) -> String {
    format!(
        "_extract_{}_{}_{}",
        crate::decl::resolve::snake(&c.rust_name),
        m.rust_name.trim_start_matches("r#").trim_start_matches('_'),
        p.name.trim_start_matches("r#")
    )
}

fn numeric_extractors(c: &Class) -> Result<String, String> {
    let mut source = String::new();
    for method in &c.methods {
        for parameter in &method.params {
            if !native(&parameter.ty) || !contains_number(&parameter.ty) {
                continue;
            }
            let at = format!("{}::{}", c.trait_path, method.rust_name);
            let path = format!("{:?}", parameter.name);
            let conversion = match parameter.pcm {
                true => format!("::lk_agents_abi::python::pcm_bytes(obj, {path})"),
                false => from_object("obj", &parameter.ty, &path, &at)?,
            };
            let conversion = unblock(&conversion);
            source.push_str(&format!(
                "\nfn {}(obj: &Bound<'_, PyAny>) -> PyResult<{}> {{\n    {conversion}\n}}\n",
                extractor_name(c, method, parameter),
                parameter.ty.rust()
            ));
        }
    }
    Ok(source)
}

fn method(decls: &Declarations, c: &Class, m: &Method) -> Result<String, String> {
    let at = format!("{}::{}", c.trait_path, m.rust_name);
    let call = format!("<{}Impl as {}>::{}", c.name, c.trait_path, m.rust_name);
    // Functions are module items, methods are indented in their `impl`.
    let ind = if matches!(m.kind, MethodKind::Function | MethodKind::AsyncFunction) {
        ""
    } else {
        "    "
    };
    let guard = if m.pid.unwrap_or(c.pid) {
        format!("{ind}    crate::host::check_pid()?;\n")
    } else {
        String::new()
    };
    let mut s = doc(&m.doc, ind);
    match m.kind {
        MethodKind::New => s.push_str(&format!("{ind}#[new]\n")),
        MethodKind::StaticCall => s.push_str(&format!("{ind}#[staticmethod]\n")),
        MethodKind::Getter => s.push_str(&format!("{ind}#[getter]\n")),
        MethodKind::Setter => s.push_str(&format!("{ind}#[setter({})]\n", m.name)),
        MethodKind::Function | MethodKind::AsyncFunction => s.push_str("#[pyfunction]\n"),
        MethodKind::Call | MethodKind::Await | MethodKind::Next => {}
    }
    if m.name != m.rust_name
        && !matches!(
            m.kind,
            MethodKind::Function
                | MethodKind::AsyncFunction
                | MethodKind::Next
                | MethodKind::Setter
        )
    {
        s.push_str(&format!("{ind}#[pyo3(name = \"{}\")]\n", m.name));
    }
    // Protocol slots have fixed call shapes; only constructors and __call__ take kwargs.
    let fixed_protocol = !matches!(m.kind, MethodKind::Function | MethodKind::AsyncFunction)
        && m.name.starts_with("__")
        && m.name.ends_with("__")
        && !matches!(m.name.as_str(), "__new__" | "__init__" | "__call__");
    if m.kind != MethodKind::Setter && !fixed_protocol && m.params.iter().any(|p| !p.ty.hidden()) {
        s.push_str(&format!("{ind}#[pyo3(signature = ({}))]\n", signature(&m.params)));
    }
    let mut args = Vec::new();
    let mut prelude = String::new();
    for p in &m.params {
        // Captured synchronously at the call's entry, before any native work is scheduled.
        if p.ty == Ty::CallContext {
            prelude.push_str(&format!("{ind}    let {} = crate::host::entered();\n", p.name));
            continue;
        }
        // The runtime's, never the host's.
        if p.ty == Ty::Env {
            prelude.push_str(&format!("{ind}    let {} = crate::process::env();\n", p.name));
            continue;
        }
        let (ty, convert) = if p.ty == Ty::This {
            // Borrowed out of its `PyRef` before the GIL is released.
            let deref = if c.form == Form::MutableClass { "" } else { "*" };
            (format!("PyRef<'_, {}>", c.name), Some(format!("&{deref}{}.0", p.name)))
        } else {
            input(&at, p)?
        };
        let extraction = if native(&p.ty) && contains_number(&p.ty) {
            format!("#[pyo3(from_py_with = {})] ", extractor_name(c, m, p))
        } else {
            String::new()
        };
        args.push(format!("{extraction}{}: {ty}", p.name));
        if let Some(convert) = convert {
            prelude.push_str(&format!("{ind}    let {} = {convert};\n", p.name));
        }
    }
    let receiver = match m.kind {
        MethodKind::New
        | MethodKind::Function
        | MethodKind::AsyncFunction
        | MethodKind::StaticCall => None,
        _ if m.mutable => Some("&mut self"),
        _ => Some("&self"),
    };
    let passed: Vec<String> = m
        .params
        .iter()
        .map(|p| {
            if p.ty == Ty::This {
                p.name.clone()
            } else if p.by_ref {
                format!("&{}", p.name)
            } else {
                p.name.clone()
            }
        })
        .collect();
    let err = m.ret.err.as_ref().map(|e| format!(".map_err({})?", e.with)).unwrap_or_default();
    let (ret_ty, wrap) = output(&at, decls, c, &m.ret.ok)?;
    let fn_name = match m.kind {
        MethodKind::New => "new".to_owned(),
        MethodKind::Function | MethodKind::AsyncFunction => m.name.clone(),
        _ => m.rust_name.clone(),
    };
    match m.kind {
        MethodKind::Next => {
            let item = match &m.ret.ok {
                Ty::Optional(item) => output(&at, decls, c, item)?.1,
                other => {
                    return Err(format!("{at}: next returns an option, not `{}`", other.rust()));
                }
            };
            s.push_str(&format!(
                "    fn __aiter__(slf: Py<Self>) -> Py<Self> {{\n        slf\n    }}\n\n    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n{guard}        let this = ::std::sync::Arc::clone(&self.0);\n        crate::host::future_into_py(py, async move {{\n            match {call}(&this).await{err} {{\n                Some(out) => Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({}, py)),\n                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),\n            }}\n        }})\n    }}\n",
                item.replace("{}", "out").replace("{py}", "py")
            ));
        }
        MethodKind::AsyncFunction => {
            s.push_str(&format!(
                "pub fn {fn_name}<'py>({}) -> PyResult<Bound<'py, PyAny>> {{\n{guard}{prelude}    crate::host::future_into_py(py, async move {{\n        let out = {call}({}).await{err};\n        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({}, py))\n    }})\n}}\n",
                ["py: Python<'py>".to_owned()].into_iter().chain(args).collect::<Vec<_>>().join(", "),
                passed.join(", "),
                wrap.replace("{}", "out").replace("{py}", "py")
            ));
        }
        MethodKind::Await if m.eager => {
            let refused =
                m.refused.as_ref().map(|e| format!(".map_err({})?", e.with)).unwrap_or_default();
            s.push_str(&format!(
                "    fn {fn_name}<'py>({}) -> PyResult<Bound<'py, PyAny>> {{\n{guard}{prelude}        let call = py.detach(|| {call}(&self.0, {})){refused};\n        crate::host::future_into_py(py, async move {{\n            let out = call.await{err};\n            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({}, py))\n        }})\n    }}\n",
                ["&self".to_owned(), "py: Python<'py>".to_owned()].into_iter().chain(args).collect::<Vec<_>>().join(", "),
                passed.join(", "),
                wrap.replace("{}", "out").replace("{py}", "py")
            ));
        }
        MethodKind::Await => {
            s.push_str(&format!(
                "    fn {fn_name}<'py>({}) -> PyResult<Bound<'py, PyAny>> {{\n{guard}{prelude}        let this = ::std::sync::Arc::clone(&self.0);\n        crate::host::future_into_py(py, async move {{\n            let out = {call}(&this, {})\n                .await{err};\n            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({}, py))\n        }})\n    }}\n",
                ["&self".to_owned(), "py: Python<'py>".to_owned()].into_iter().chain(args).collect::<Vec<_>>().join(", "),
                passed.join(", "),
                wrap.replace("{}", "out").replace("{py}", "py")
            ));
        }
        _ => {
            // A mutable class's state is behind PyO3's borrow flag, which the GIL serializes:
            // released, another thread's call would be refused as already borrowed.
            let detach = !m.attached && !(c.form == Form::MutableClass && receiver.is_some());
            let needs_py = ret_ty.contains("'py") || detach;
            // A function is registered from `mod.rs`.
            let vis = if matches!(m.kind, MethodKind::Function | MethodKind::AsyncFunction) {
                "pub "
            } else {
                ""
            };
            let lifetime = ret_ty.contains("'py");
            let generics = if lifetime { "<'py>" } else { "" };
            let this = match m.kind {
                MethodKind::New
                | MethodKind::Function
                | MethodKind::AsyncFunction
                | MethodKind::StaticCall => None,
                _ if m.mutable => Some("&mut self.0".to_owned()),
                _ => Some("&self.0".to_owned()),
            };
            let all_args = receiver
                .map(str::to_owned)
                .into_iter()
                .chain(
                    needs_py.then(|| format!("py: Python<'{}>", if lifetime { "py" } else { "_" })),
                )
                .chain(args)
                .collect::<Vec<_>>()
                .join(", ");
            let invocation = format!(
                "{call}({})",
                this.into_iter().chain(passed).collect::<Vec<_>>().join(", ")
            );
            // The native call runs without the GIL: a Rust task holding a lock it waits for may
            // itself be waiting for the GIL.
            let value = if detach {
                format!("py.detach(|| {invocation}){err}")
            } else {
                format!("{invocation}{err}")
            };
            let body = if m.ret.ok == Ty::Unit {
                format!("{ind}    {value};\n{ind}    Ok(())\n")
            } else {
                format!(
                    "{ind}    let out = {value};\n{ind}    Ok({})\n",
                    wrap.replace("{}", "out").replace("{py}", "py")
                )
            };
            s.push_str(&format!(
                "{ind}{vis}fn {fn_name}{generics}({all_args}) -> PyResult<{ret_ty}> {{\n{guard}{prelude}{body}{ind}}}\n"
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
/// A record's `_core` functions: its persistence codec, its omitted fields' computations.
pub(crate) fn functions(r: &RecordDecl) -> Vec<String> {
    let mut out = Vec::new();
    if r.persisted {
        let (persist, restore) = r.persistence();
        out.extend([persist, restore]);
    }
    out.extend(r.omitted().into_iter().map(|(_, f)| f));
    out
}

pub(crate) fn has_functions(r: &RecordDecl) -> bool {
    r.persisted || !r.omitted().is_empty()
}

/// A persisted record's codec: its serde form in Rust, crossing as its generated model.
fn persistence(r: &RecordDecl) -> String {
    let (persist, restore) = r.persistence();
    let (rust, name) = (&r.rust, &r.name);
    format!(
        "// Declared in {source}.\n\n/// `{name}.to_json()`.\n#[pyfunction]\npub fn {persist}(record: Bound<'_, PyAny>) -> PyResult<String> {{\n    let record = {conv}_from_py(&record)?;\n    ::serde_json::to_string(&record)\n        .map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!(\"{name}: {{e}}\")))\n}}\n\n/// `{name}.from_json(data)`.\n#[pyfunction]\npub fn {restore}<'py>(py: Python<'py>, data: &str) -> PyResult<Bound<'py, PyAny>> {{\n    let record: {rust} = ::serde_json::from_str(data)\n        .map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!(\"{name}: {{e}}\")))?;\n    {conv}_to_py(py, record)\n}}\n",
        source = r.source,
        conv = crate::records::conversion(&r.module, &r.name)
    )
}

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

/// The stream classes a proxy's calls hand the host (`source(class = ..)` parameters) that no
/// class of its family hands out already.
pub fn proxy_sources(decls: &Declarations, p: &Proxy) -> Vec<(String, Ty)> {
    let mut out: Vec<(String, Ty)> = Vec::new();
    let handed_out = |name: &str| {
        decls
            .classes
            .iter()
            .filter(|c| c.family == p.family)
            .any(|c| sources(c).iter().any(|(n, _)| n == name))
    };
    let earlier = decls.proxies.iter().take_while(|o| !std::ptr::eq(*o, p));
    let earlier: Vec<String> = earlier
        .filter(|o| o.family == p.family)
        .flat_map(|o| o.methods.iter().flat_map(|m| m.params.iter()))
        .filter_map(|q| match &q.ty {
            Ty::Source { class, .. } => Some(class.clone()),
            _ => None,
        })
        .collect();
    for q in p.methods.iter().flat_map(|m| m.params.iter()) {
        if let Ty::Source { class, item } = &q.ty
            && !handed_out(class)
            && !earlier.contains(class)
            && !out.iter().any(|(n, _)| n == class)
        {
            out.push((class.clone(), (**item).clone()));
        }
    }
    out
}

/// The duplex stream classes a class's methods hand out: each sink with its source's item.
pub fn duplexes(c: &Class) -> Vec<(&Sink, Ty)> {
    c.sinks
        .iter()
        .filter_map(|sink| {
            c.methods.iter().find_map(|m| match &m.ret.ok {
                Ty::Duplex { class, output, .. } if *class == sink.class => {
                    Some((sink, (**output).clone()))
                }
                _ => None,
            })
        })
        .collect()
}

/// A duplex stream as the host drives it: input pushed synchronously into the sink, output pulled
/// through the runtime's `NativeSource`, closing ends the input. The sink trait's own `call`s are
/// methods of the class.
fn duplex_class(
    decls: &Declarations,
    c: &Class,
    sink: &Sink,
    output: &Ty,
) -> Result<String, String> {
    let name = &sink.class;
    let at = format!("{}::{name}", c.trait_path);
    let to_py = if native(output) {
        "::pyo3::IntoPyObjectExt::into_py_any(item, py)".to_owned()
    } else {
        format!("Ok({}.unbind())", crate::records::emit(output, "item")?)
    };
    let param = Param {
        name: "item".into(),
        ty: sink.item.clone(),
        by_ref: false,
        kw_only: false,
        default: None,
        shape_error: None,
        one_or_many: false,
        pcm: false,
    };
    let (item_ty, convert) = input(&at, &param)?;
    let convert = convert.map(|c| format!("        let item = {c};\n")).unwrap_or_default();
    let err = &sink.err.with;
    let guard = "        crate::host::check_pid()?;\n";
    let control = Class {
        name: name.clone(),
        rust_name: name.clone(),
        trait_path: sink.trait_path.clone(),
        methods: sink.methods.clone(),
        sinks: Vec::new(),
        ..c.clone()
    };
    let mut s = format!(
        "type {name}Impl = dyn {trait_path};\n{extractors}\n/// One duplex stream: `push()` input, `recv()` resolves to the next output, `None` at the end.\n#[pyclass(frozen, name = \"{name}\", module = \"livekit.agents._core\")]\npub struct {name}(::std::boxed::Box<{name}Impl>, ::lk_agents_runtime::net::source::NativeSource<{output}>);\n\n#[pymethods]\nimpl {name} {{\n",
        trait_path = sink.trait_path,
        extractors = numeric_extractors(&control)?,
        output = output.rust(),
    );
    s.push_str(&format!(
        "    fn push(&self, py: Python<'_>, item: {item_ty}) -> PyResult<()> {{\n{guard}{convert}        py.detach(|| ::lk_agents_abi::StreamSink::push(&*self.0, item)).map_err({err})\n    }}\n\n    fn flush(&self, py: Python<'_>) -> PyResult<()> {{\n{guard}        py.detach(|| ::lk_agents_abi::StreamSink::flush(&*self.0)).map_err({err})\n    }}\n\n    fn end_input(&self, py: Python<'_>) -> PyResult<()> {{\n{guard}        py.detach(|| ::lk_agents_abi::StreamSink::end_input(&*self.0));\n        Ok(())\n    }}\n\n    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n{guard}        crate::host::source::recv(&self.1, py, |py, item| {to_py})\n    }}\n\n    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n{guard}        py.detach(|| ::lk_agents_abi::StreamSink::end_input(&*self.0));\n        crate::host::source::aclose(&self.1, py)\n    }}\n"
    ));
    for m in &control.methods {
        s.push('\n');
        // The input half is boxed: its calls take the trait object.
        s.push_str(&method(decls, &control, m)?.replace("(&self.0", "(&*self.0"));
    }
    s.push_str("}\n");
    Ok(s)
}

/// A pull source as the host consumes it: the runtime's `NativeSource` does the work.
fn source_class(name: &str, item: &Ty) -> Result<String, String> {
    let to_py = if native(item) {
        "::pyo3::IntoPyObjectExt::into_py_any(item, py)".to_owned()
    } else {
        format!("Ok({}.unbind())", crate::records::emit(item, "item")?)
    };
    Ok(format!(
        "/// One stream of `{item}` items: `recv()` resolves to the next, `None` at the end.\n#[pyclass(frozen, name = \"{name}\", module = \"livekit.agents._core\")]\npub struct {name}(::lk_agents_runtime::net::source::NativeSource<{item}>);\n\n#[pymethods]\nimpl {name} {{\n    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n        crate::host::check_pid()?;\n        crate::host::source::recv(&self.0, py, |py, item| {to_py})\n    }}\n\n    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {{\n        crate::host::check_pid()?;\n        crate::host::source::aclose(&self.0, py)\n    }}\n}}\n",
        item = item.rust()
    ))
}

/// pyo3's signature: positional parameters, then `*` before the first keyword-only one.
fn signature(params: &[Param]) -> String {
    let mut parts = Vec::new();
    let mut star = false;
    for p in params.iter().filter(|p| !p.ty.hidden()) {
        if p.kw_only && !star {
            parts.push("*".to_owned());
            star = true;
        }
        match &p.default {
            None if matches!(p.ty, Ty::Maybe(_)) => {
                parts.push(format!("{} = crate::conversions::Omittable::NOT_GIVEN", p.name))
            }
            None => parts.push(p.name.clone()),
            Some(d) => parts.push(format!("{} = {}", p.name, rust_literal(d, &p.ty))),
        }
    }
    parts.join(", ")
}

pub(crate) fn rust_literal(d: &Literal, ty: &Ty) -> String {
    if let Ty::Optional(inner) = ty
        && *d != Literal::None
    {
        return format!("Some({})", rust_literal(d, inner));
    }
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
        // A declared record crosses by its generated conversion functions, not a pyo3 trait.
        Ty::Record(r) => {
            !matches!(r.host, HostForm::Record { .. } | HostForm::WireStrs | HostForm::Converted(_))
        }
        Ty::Bool | Ty::Int(_) | Ty::Float | Ty::Str => true,
        Ty::Optional(t) | Ty::List(t) => native(t),
        Ty::Map(k, t) => native(k) && native(t),
        Ty::Tuple(ts) => ts.iter().all(native),
        _ => false,
    }
}

/// Numeric leaves use the ABI extractor so Python booleans cannot stand in for numbers.
fn contains_number(ty: &Ty) -> bool {
    match ty {
        Ty::Int(_) | Ty::Float => true,
        Ty::Optional(inner) | Ty::Maybe(inner) | Ty::List(inner) => contains_number(inner),
        Ty::Map(key, value) | Ty::OrderedMap(key, value) => {
            contains_number(key) || contains_number(value)
        }
        Ty::Tuple(items) => items.iter().any(contains_number),
        _ => false,
    }
}

/// The adapter's parameter type, and the conversion (an expression of the parameter) when it is
/// taken as an object.
fn input(at: &str, p: &Param) -> Result<(String, Option<String>), String> {
    let path = format!("{:?}", p.name);
    if let (true, Ty::List(item)) = (p.one_or_many, &p.ty) {
        return Ok((
            "Bound<'_, PyAny>".into(),
            Some(format!(
                "(if {n}.is_instance_of::<::pyo3::types::PyList>() {{ {} }} else {{ ({}).map(|item| vec![item]) }})?",
                from_object(&p.name, &p.ty, &path, at)?,
                from_object(&p.name, item, &path, at)?,
                n = p.name,
            )),
        ));
    }
    if p.shape_error.is_none() && native(&p.ty) {
        return Ok((p.ty.rust(), None));
    }
    match &p.ty {
        Ty::Maybe(inner) => Ok((
            "crate::conversions::Omittable<'_>".into(),
            Some(maybe_input(&p.name, inner, &path, at)?),
        )),
        Ty::Optional(inner) => {
            if p.default.as_ref().is_some_and(|d| *d != Literal::None) {
                return Err(format!(
                    "{at}({}): an object-converted parameter defaults only to None",
                    p.name
                ));
            }
            let convert =
                crate::shape_error::convert("v", inner, &path, at, p.shape_error.as_ref())?;
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
                Some(format!(
                    "{}?",
                    crate::shape_error::convert(&p.name, ty, &path, at, p.shape_error.as_ref())?
                )),
            ))
        }
    }
}

/// Convert a supplied value while preserving the existing omission sentinel.
fn maybe_input(value: &str, inner: &Ty, path: &str, at: &str) -> Result<String, String> {
    Ok(format!("{value}.map(|value| {})?", from_object("value", inner, path, at)?))
}

/// `PyResult<T>` of the object `v`.
pub(crate) fn from_object(v: &str, ty: &Ty, path: &str, at: &str) -> Result<String, String> {
    Ok(match ty {
        Ty::Instance { family, class, .. } => format!(
            "{v}.cast::<super::{family}::{class}>().map(|value| value.get().0.as_ref().clone()).map_err(::pyo3::PyErr::from)"
        ),
        Ty::Int(integer) => {
            format!("::lk_agents_abi::python::number_from_py::<{}>(&{v}, {path})", integer.rust())
        }
        Ty::Float => format!("::lk_agents_abi::python::number_from_py::<f64>(&{v}, {path})"),
        Ty::Maybe(inner) => format!(
            "Ok({})",
            maybe_input(
                &format!("{v}.extract::<crate::conversions::Omittable<'_>>()?"),
                inner,
                path,
                at
            )?
        ),
        Ty::Json => format!("::lk_agents_abi::python::value_from_py(&{v}, {path})"),
        Ty::JsonObject => format!(
            "match ::lk_agents_abi::python::value_from_py(&{v}, {path})? {{ ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!({path}, \": expected a dict\"))) }}"
        ),
        Ty::Record(Record { host: HostForm::Record { module, name }, .. }) => {
            format!("{}_from_py(&{v})", crate::records::conversion(module, name))
        }
        Ty::Record(r @ Record { host: HostForm::Converted(_), .. }) => {
            format!("{}_from_py(&{v})", crate::records::converted(r))
        }
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => {
            let collection = if matches!(ty, Ty::OrderedMap(..)) {
                "::indexmap::IndexMap"
            } else {
                "::std::collections::BTreeMap"
            };
            format!(
                "{v}.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| dict.iter().map(|(key, item)| Ok(({}?, {}?))).collect::<PyResult<{collection}<_, _>>>())",
                from_object("key", k, path, at)?,
                from_object("item", t, path, at)?
            )
        }
        Ty::Native { rust, refused, .. } => format!(
            "::lk_agents_abi::python::native::<dyn {rust}>(&{v}).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!({path}, {message:?})))",
            message = format!(": {refused}"),
        ),
        Ty::Proxy { declared: Some((family, name)), boxed: true, .. } => {
            format!("crate::generated::{family}::{name}Proxy::owned(&{v})")
        }
        Ty::Proxy { declared: Some((family, name)), .. } => {
            format!("crate::generated::{family}::{name}Proxy::from_host(&{v})")
        }
        Ty::Shared { class, family, value: false, .. } => format!(
            "{v}.cast::<crate::generated::{family}::{class}>().map(|this| ::std::sync::Arc::clone(&this.get().0)).map_err(::pyo3::PyErr::from)"
        ),
        Ty::Shared { class, family, value: true, .. } => format!(
            "{v}.cast::<crate::generated::{family}::{class}>().map(|this| ::std::clone::Clone::clone(&*this.get().0)).map_err(::pyo3::PyErr::from)"
        ),
        // The host's async iterable, pulled in a copy of the context the call was made in.
        Ty::HostSource(item) => format!(
            "crate::host::capture_context({v}.py()).and_then(|context| crate::host::stream::HostStream::open({v}.py(), &{v}, &context, |o| Ok({}?), ::lk_agents_runtime::process::host_clock())).map(|s| ::std::boxed::Box::new(s) as {})",
            from_object("o", item, path, at)?,
            ty.rust()
        ),
        Ty::Proxy { rust, declared: None, .. } => {
            return Err(format!(
                "{at}: `{rust}` crosses behind its generated proxy: declare its methods' calls"
            ));
        }
        Ty::List(inner) if **inner == Ty::Int(crate::decl::Int::U8) => format!(
            "{{ if {v}.is_instance_of::<::pyo3::types::PyBytes>() || {v}.is_instance_of::<::pyo3::types::PyByteArray>() || {v}.is_instance_of::<::pyo3::types::PyMemoryView>() {{ ::lk_agents_abi::python::owned_bytes(&{v}, {path}) }} else {{ {v}.try_iter().and_then(|items| items.map(|item| {{ let item = item?; {} }}).collect::<PyResult<Vec<_>>>()) }} }}",
            from_object("item", inner, path, at)?
        ),
        Ty::List(inner) => format!(
            "{v}.try_iter().and_then(|items| items.map(|item| {{ let item = item?; {} }}).collect::<PyResult<Vec<_>>>())",
            from_object("item", inner, path, at)?
        ),
        Ty::Tuple(items) => {
            let mut reads = String::new();
            let mut fields = Vec::new();
            for (index, item) in items.iter().enumerate() {
                let name = format!("item_{index}");
                reads.push_str(&format!(" let {name} = tuple.get_item({index})?;"));
                fields.push(format!("({})?", from_object(&name, item, path, at)?));
            }
            format!(
                "(|| -> ::pyo3::PyResult<_> {{ let tuple = {v}.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != {len} {{ return Err(::pyo3::exceptions::PyTypeError::new_err(format!(\"{{}}: expected a tuple of {len} items\", {path}))); }}{reads} Ok(({},)) }})()",
                fields.join(", "),
                len = items.len(),
            )
        }
        Ty::Duration => format!(
            "::lk_agents_abi::python::number_from_py::<f64>(&{v}, {path}).and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!(\"{{}}: {{e}}\", {path}))))"
        ),
        Ty::Optional(inner) => format!(
            "if {v}.is_none() {{ Ok(None) }} else {{ ({}).map(Some) }}",
            from_object(v, inner, path, at)?
        ),
        // A `_core` class is taken by a clone, whose borrow failure converts.
        other if native(other) && holds_core_class(other) => {
            format!("{v}.extract::<{}>().map_err(::pyo3::PyErr::from)", other.rust())
        }
        other if native(other) => format!("{v}.extract::<{}>()", other.rust()),
        // A host exception handed over as an argument, classified as a raised one is.
        Ty::Error(e) if e.rust.ends_with("::ProviderError") => {
            format!("PyResult::Ok(::lk_agents_abi::python::capture_value(&{v}))")
        }
        other => return Err(format!("{at}: `{}` has no input conversion", other.rust())),
    })
}

fn holds_core_class(ty: &Ty) -> bool {
    match ty {
        Ty::Record(Record { host: HostForm::CoreClass(_), .. }) => true,
        Ty::Optional(t) | Ty::List(t) => holds_core_class(t),
        Ty::Tuple(ts) => ts.iter().any(holds_core_class),
        _ => false,
    }
}

/// The adapter's return type and the expression of `{}` (the trait's value) it returns.
fn output(at: &str, decls: &Declarations, c: &Class, ty: &Ty) -> Result<(String, String), String> {
    if native(ty) {
        return Ok((ty.rust(), "{}".into()));
    }
    Ok(match ty {
        Ty::Optional(inner) => {
            let (item_ty, item) = output(at, decls, c, inner)?;
            (
                format!("::std::option::Option<{item_ty}>"),
                format!(
                    "{{}}.map(|value| Ok::<_, ::pyo3::PyErr>({})).transpose()?",
                    item.replace("{}", "value"),
                ),
            )
        }
        Ty::Unit => ("()".into(), "{}".into()),
        Ty::Error(_) => (
            "Bound<'py, PyAny>".into(),
            "::lk_agents_abi::python::ExceptionValue::to_exception({}, {py})?".into(),
        ),
        Ty::This if c.form == Form::MutableClass => (c.name.clone(), format!("{}({{}})", c.name)),
        Ty::This => (c.name.clone(), format!("{}(::std::sync::Arc::new({{}}))", c.name)),
        Ty::Json => {
            ("Bound<'py, PyAny>".into(), "::lk_agents_abi::python::value_to_py({py}, &{})?".into())
        }
        Ty::JsonObject => (
            "Bound<'py, PyAny>".into(),
            "::lk_agents_abi::python::extra_to_py({py}, &{})?.into_any()".into(),
        ),
        Ty::Instance { class, .. } => {
            let target = decls
                .classes
                .iter()
                .find(|t| t.name == *class && t.form != Form::Functions)
                .ok_or_else(|| format!("{at}: `{class}` is not a declared class"))?;
            let path = format!("super::{}::{class}", target.family);
            let held = if target.form == Form::MutableClass {
                "{}".to_owned()
            } else {
                "::std::sync::Arc::new({})".to_owned()
            };
            (path.clone(), format!("{path}({held})"))
        }
        Ty::Handle { class, .. } => {
            (format!("crate::handles::{class}"), format!("crate::handles::{class}::from({{}})"))
        }
        Ty::Shared { class, family, value, .. } => (
            "Bound<'py, PyAny>".into(),
            format!(
                "::pyo3::Bound::new({{py}}, crate::generated::{family}::{class}({}))?.into_any()",
                if *value { "::std::sync::Arc::new({})" } else { "{}" }
            ),
        ),
        Ty::Duplex { class, .. } => (
            class.clone(),
            format!(
                "{{ let (sink, source) = {{}}; {class}(::std::boxed::Box::new(sink), ::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new(source))) }}"
            ),
        ),
        Ty::Source { class, .. } => (
            class.clone(),
            format!(
                "{class}(::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new({{}})))"
            ),
        ),
        other => (
            "Bound<'py, PyAny>".into(),
            crate::records::emit(other, "{}").map_err(|e| format!("{at}: {e}"))?,
        ),
    })
}

/// `{ expr }` as `expr` when the braces enclose the whole expression: a function body is a block.
fn unblock(expr: &str) -> &str {
    let Some(inner) = expr.strip_prefix("{ ").and_then(|e| e.strip_suffix(" }")) else {
        return expr;
    };
    let mut depth = 0i32;
    for c in inner.chars() {
        depth += match c {
            '{' => 1,
            '}' => -1,
            _ => 0,
        };
        if depth < 0 {
            return expr;
        }
    }
    if depth == 0 && !inner.starts_with("let ") { inner } else { expr }
}

/// Rust names for the private `_core_X` classes and their helpers: `CoreX` types and
/// snake-case functions. String literals (the Python names) are left as they are.
pub fn rust_names(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    let mut in_str = false;
    while let Some((i, c)) = chars.next() {
        if in_str {
            out.push(c);
            if c == '\\' {
                if let Some((_, next)) = chars.next() {
                    out.push(next);
                }
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
            out.push(c);
            continue;
        }
        if c.is_alphanumeric() || c == '_' {
            let start = i;
            let mut end = i + c.len_utf8();
            while let Some(&(j, d)) = chars.peek() {
                if !(d.is_alphanumeric() || d == '_') {
                    break;
                }
                end = j + d.len_utf8();
                chars.next();
            }
            let word = &text[start..end];
            let prev = text[..start].chars().next_back();
            // A lifetime or a char literal is not an identifier.
            if prev == Some('\'') {
                out.push_str(word);
                continue;
            }
            out.push_str(&match word.strip_prefix("_core_") {
                Some(rest) if rest.starts_with(char::is_uppercase) => format!("Core{rest}"),
                _ => word.replace("__core_", "_core_").replace("_core__", "_core_"),
            });
            continue;
        }
        out.push(c);
    }
    out
}
