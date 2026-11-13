//! The two-step constructor of an `init` class.

use super::{
    class_ident, contains_number, doc, extractor_name, input, native, parameter_type, signature,
};
use crate::decl::{Class, Method, Ty};

/// An `init` constructor: an empty object (`__new__`, taking whatever its host subclass is called
/// with), then `__init__`, which builds the implementation once, its events through a weak proxy
/// of the object and its label the object's host class.
pub(super) fn init(c: &Class, m: &Method) -> Result<String, String> {
    let at = format!("{}::{}", c.trait_path, m.rust_name);
    let call = format!("<{}Impl as {}>::{}", class_ident(&c.name), c.trait_path, m.rust_name);
    let host = &c.name;
    let guard = match m.pid.unwrap_or(c.pid) {
        true => "        crate::py_calls::check_pid()?;\n",
        false => "",
    };
    let mut args = Vec::new();
    let mut prelude = String::new();
    for p in &m.params {
        let filled = match &p.ty {
            Ty::Env => Some("crate::process::env()".to_owned()),
            Ty::Cx => Some("crate::py_calls::entry_cx()?".to_owned()),
            Ty::CallContext => Some("crate::py_calls::entered()".to_owned()),
            Ty::ClassLabel => Some("crate::py_objects::class_label(slf.as_any())?".to_owned()),
            Ty::Proxy { weak: true, declared: Some((family, name)), .. } => {
                Some(format!("crate::generated::{family}::{name}Proxy::weak(slf.as_any())?"))
            }
            _ => None,
        };
        if let Some(value) = filled {
            prelude.push_str(&format!("        let {} = {value};\n", p.name));
            continue;
        }
        let (ty, convert) = input(&at, p)?;
        let numeric = native(&p.ty) && contains_number(&p.ty);
        let ty = if numeric { parameter_type(c, m, p) } else { ty };
        let extraction = match numeric {
            true => format!("#[pyo3(from_py_with = {})] ", extractor_name(c, m, p)),
            false => String::new(),
        };
        args.push(format!("{extraction}{}: {ty}", p.name));
        if let Some(convert) = convert {
            prelude.push_str(&format!("        let {} = {convert};\n", p.name));
        }
    }
    let passed: Vec<String> = m
        .params
        .iter()
        .map(|p| if p.by_ref { format!("&{}", p.name) } else { p.name.clone() })
        .collect();
    let err = m.ret.err.as_ref().map(|e| format!(".map_err({})?", e.with)).unwrap_or_default();
    let mut s = String::from(
        "    #[new]\n    #[pyo3(signature = (*_args, **_kwargs))]\n    fn new(_args: &Bound<'_, ::pyo3::types::PyTuple>, _kwargs: Option<&Bound<'_, ::pyo3::types::PyDict>>) -> Self {\n        Self(::std::sync::OnceLock::new())\n    }\n\n",
    );
    s.push_str(&doc(&m.doc, "    "));
    if m.params.iter().any(|p| !p.ty.hidden()) {
        s.push_str(&format!("    #[pyo3(signature = ({}))]\n", signature(&m.params)));
    }
    let all_args = ["slf: &Bound<'_, Self>".to_owned(), "py: Python<'_>".to_owned()]
        .into_iter()
        .chain(args)
        .collect::<Vec<_>>()
        .join(", ");
    s.push_str(&format!(
        "    fn __init__({all_args}) -> PyResult<()> {{\n{guard}        let once = || ::pyo3::exceptions::PyRuntimeError::new_err(\"{host}.__init__ runs once\");\n        if slf.get().0.get().is_some() {{\n            return Err(once());\n        }}\n{prelude}        let out = py.detach(|| {call}({})){err};\n        slf.get().0.set(::std::sync::Arc::new(out)).map_err(|_| once())\n    }}\n",
        passed.join(", ")
    ));
    Ok(s)
}
