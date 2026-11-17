use super::{class_ident, native};
use crate::decl::{Class, Declarations, Form, Ty};

/// The adapter's return type and the expression of `{}` (the trait's value) it returns.
pub(super) fn output(
    at: &str,
    decls: &Declarations,
    c: &Class,
    ty: &Ty,
) -> Result<(String, String), String> {
    if native(ty) {
        return Ok((ty.rust(), "{}".into()));
    }
    Ok(match ty {
        Ty::Map(key, value) if matches!(value.as_ref(), Ty::Instance { .. }) => {
            let (_, value) = output(at, decls, c, value)?;
            let value = value.replace("{}", "value").replace("{py}", "py");
            let key = crate::records::emit(key, "key")?;
            (
                "Bound<'py, PyAny>".into(),
                format!(
                    "{{ let dict = ::pyo3::types::PyDict::new(py); for (key, value) in {{}} {{ dict.set_item({key}, ::pyo3::IntoPyObjectExt::into_bound_py_any({value}, py)?)?; }} dict.into_any() }}"
                ),
            )
        }
        Ty::Optional(inner) => {
            let (item_ty, item) = output(at, decls, c, inner)?;
            (
                format!("::std::option::Option<{item_ty}>"),
                format!(
                    "{{}}.map(|value| {}).transpose()?",
                    crate::records::as_result(&item.replace("{}", "value")),
                ),
            )
        }
        Ty::Unit => ("()".into(), "{}".into()),
        Ty::Error(_) => (
            "Bound<'py, PyAny>".into(),
            "crate::py_errors::ExceptionValue::to_exception({}, {py})?".into(),
        ),
        Ty::This if c.form == Form::MutableClass => {
            (class_ident(&c.name), format!("{}({{}})", class_ident(&c.name)))
        }
        Ty::This => {
            (class_ident(&c.name), format!("{}(::std::sync::Arc::new({{}}))", class_ident(&c.name)))
        }
        Ty::Json => {
            ("Bound<'py, PyAny>".into(), "crate::py_values::value_to_py({py}, &{})?".into())
        }
        Ty::JsonObject => (
            "Bound<'py, PyAny>".into(),
            "crate::py_values::extra_to_py({py}, &{})?.into_any()".into(),
        ),
        Ty::Instance { class, shared, .. } => {
            let target = decls
                .classes
                .iter()
                .find(|t| t.name == *class && t.form != Form::Functions)
                .ok_or_else(|| format!("{at}: `{class}` is not a declared class"))?;
            let path = format!("super::{}::{}", target.family, class_ident(class));
            let held = if *shared || target.form == Form::MutableClass {
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
                if *value { "::std::sync::Arc::new({})" } else { "{}" },
                class = class_ident(class),
            ),
        ),
        Ty::Duplex { class, .. } => (
            class_ident(class),
            format!(
                "{{ let (sink, source) = {{}}.split(); {class}(::std::boxed::Box::new(sink), ::lk_agents_runtime::net::source::SharedStream::new(::std::boxed::Box::new(source), ::lk_agents_runtime::process::host_clock())) }}",
                class = class_ident(class),
            ),
        ),
        Ty::Source { class, .. } => (
            class_ident(class),
            format!(
                "{class}(::lk_agents_runtime::net::source::SharedStream::new(::std::boxed::Box::new({{}}), ::lk_agents_runtime::process::host_clock()))",
                class = class_ident(class),
            ),
        ),
        other => (
            "Bound<'py, PyAny>".into(),
            crate::records::emit(other, "{}").map_err(|e| format!("{at}: {e}"))?,
        ),
    })
}
