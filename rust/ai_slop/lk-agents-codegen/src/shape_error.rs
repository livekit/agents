//! Explicit outer-shape diagnostics for declared input containers.

use crate::decl::Ty;
use crate::decl::vocabulary as voc;

#[derive(Debug, Clone)]
pub struct ShapeError {
    pub items: bool,
    pub kind: String,
    pub message: String,
}

pub fn read(at: &str, entries: &[voc::Entry], ty: &Ty) -> Result<Option<ShapeError>, String> {
    let Some(entry) = voc::find(entries, "shape_error") else { return Ok(None) };
    voc::only(at, &entry.nested, &["items", "kind", "message"])?;
    let items = voc::find(&entry.nested, "items");
    if items.is_some_and(|item| item.value.is_some() || !item.nested.is_empty()) {
        return Err(format!("{at}: shape_error(items) is a flag"));
    }
    let kind = voc::text(at, &entry.nested, "kind")?
        .ok_or_else(|| format!("{at}: shape_error requires kind"))?;
    if !matches!(kind.as_str(), "ValueError" | "TypeError") {
        return Err(format!("{at}: shape_error kind is ValueError or TypeError"));
    }
    let message = voc::text(at, &entry.nested, "message")?
        .ok_or_else(|| format!("{at}: shape_error requires message"))?;
    let remainder = message.replace("{value}", "");
    if remainder.contains(['{', '}']) {
        return Err(format!("{at}: shape_error message accepts only {{value}}"));
    }
    let ty = match ty {
        Ty::Optional(inner) => inner.as_ref(),
        other => other,
    };
    let items = items.is_some();
    let valid = if items {
        matches!(ty, Ty::List(inner) if matches!(inner.as_ref(), Ty::Record(_)))
            && !message.contains("{value}")
    } else {
        matches!(ty, Ty::JsonObject)
    };
    if !valid {
        return Err(format!(
            "{at}: shape_error applies to a JSON object, or items of a list of dict records; {{value}} belongs only to the object"
        ));
    }
    Ok(Some(ShapeError { items, kind, message }))
}

pub fn convert(
    value: &str,
    ty: &Ty,
    path: &str,
    at: &str,
    policy: Option<&ShapeError>,
) -> Result<String, String> {
    let Some(policy) = policy else {
        return crate::adapter::from_object(value, ty, path, at);
    };
    let exception = format!("::pyo3::exceptions::Py{}", policy.kind);
    if policy.items {
        let Ty::List(inner) = ty else {
            return Err(format!("{at}: shape_error items requires a list"));
        };
        let read = crate::adapter::from_object("item", inner, path, at)?;
        Ok(format!(
            "{value}.cast::<::pyo3::types::PyList>().map_err(::pyo3::PyErr::from).and_then(|items| items.iter().map(|item| {{ if !item.is_instance_of::<::pyo3::types::PyDict>() {{ return Err({exception}::new_err({message:?})); }} {read} }}).collect::<PyResult<Vec<_>>>())",
            message = policy.message,
        ))
    } else {
        Ok(format!(
            "match crate::python::value_from_py({value}.as_any(), {path})? {{ ::serde_json::Value::Object(map) => Ok(map), value => Err({exception}::new_err(format!({message:?}))) }}",
            message = policy.message,
        ))
    }
}
