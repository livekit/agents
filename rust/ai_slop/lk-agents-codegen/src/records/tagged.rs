use super::{emit, read};
use crate::decl::{Field, RecordBody, RecordDecl, Ty, Variant, VariantShape};

// A variant reads from any of its tags and emits its first; variants sharing a tag are told apart
// by which one's first key holds a value.

pub(super) fn record(r: &RecordDecl) -> Result<String, String> {
    let (RecordBody::Variants(variants), Some(key)) = (&r.body, &r.tag) else {
        return Err(format!("{}: a tagged dict union is an enum with a tag key", r.rust));
    };
    let rust = &r.rust;
    let name = super::cell(&r.name).to_lowercase();
    let mut text = String::new();
    if r.extract {
        text.push_str(&extract(r, variants, key)?);
    }
    if r.emit {
        let mut arms = String::new();
        for v in variants {
            let tag = v.tags.first().ok_or_else(|| format!("{rust}::{}: no tag", v.rust_name))?;
            let mut sets = String::new();
            let pattern = match &v.shape {
                VariantShape::Unit => String::new(),
                VariantShape::Named(fields) => {
                    for f in fields {
                        sets.push_str(&format!(
                            "dict.set_item({:?}, {})?;\n",
                            f.host_name(),
                            emit(&f.ty, &f.rust_name)?
                        ));
                    }
                    let names = fields.iter().map(|f| f.rust_name.as_str()).collect::<Vec<_>>();
                    format!(" {{ {} }}", names.join(", "))
                }
                _ => {
                    return Err(format!(
                        "{rust}::{}: a tagged variant has named keys",
                        v.rust_name
                    ));
                }
            };
            arms.push_str(&format!(
                "{rust}::{}{pattern} => {{ dict.set_item({key:?}, {tag:?})?; {sets} }}\n",
                v.rust_name
            ));
        }
        text.push_str(&format!(
            "pub fn {name}_to_py<'py>(py: Python<'py>, value: {rust}) -> PyResult<Bound<'py, PyAny>> {{\n    let dict = ::pyo3::types::PyDict::new(py);\n    match value {{\n{arms}    }}\n    Ok(dict.into_any())\n}}\n"
        ));
    }
    Ok(text)
}

fn extract(r: &RecordDecl, variants: &[Variant], key: &str) -> Result<String, String> {
    let (rust, owner) = (&r.rust, &r.name);
    let mut tags: Vec<&String> = variants.iter().flat_map(|v| &v.tags).collect();
    tags.sort();
    tags.dedup();
    let mut arms = String::new();
    for tag in tags {
        let sharing: Vec<&Variant> = variants.iter().filter(|v| v.tags.contains(tag)).collect();
        let body = if let [only] = &sharing[..] {
            build(r, only)?
        } else {
            // Told apart by which variant's first key holds a value.
            let mut keys = Vec::new();
            for v in &sharing {
                let VariantShape::Named(fields) = &v.shape else {
                    return Err(format!(
                        "{rust}::{}: a shared tag's variant has keys",
                        v.rust_name
                    ));
                };
                let first = fields.first().ok_or("a shared tag's variant has keys")?.host_name();
                if keys.contains(&first) {
                    return Err(format!("{rust}: variants sharing {tag:?} start with one key"));
                }
                keys.push(first);
            }
            let given = keys
                .iter()
                .map(|k| format!("item({k:?})?.is_some_and(|v| !v.is_none())"))
                .collect::<Vec<_>>();
            let mut arms = String::new();
            for (i, v) in sharing.iter().enumerate() {
                let pattern =
                    (0..keys.len()).map(|j| (i == j).to_string()).collect::<Vec<_>>().join(", ");
                arms.push_str(&format!("({pattern}) => {},\n", build(r, v)?));
            }
            let message = format!("{owner}: {tag} carries exactly one of {}", keys.join(", "));
            format!(
                "match ({}) {{\n{arms}_ => return Err(::pyo3::exceptions::PyValueError::new_err({message:?})),\n}}",
                given.join(", ")
            )
        };
        arms.push_str(&format!("{tag:?} => {body},\n"));
    }
    let name = super::cell(&r.name).to_lowercase();
    Ok(format!(
        "pub fn {name}_from_py<'py>(obj: &Bound<'py, PyAny>) -> PyResult<{rust}> {{
    let item = |key: &str| -> PyResult<Option<Bound<'py, PyAny>>> {{
        match obj.get_item(key) {{
            Ok(v) => Ok(Some(v)),
            Err(e) if e.is_instance_of::<::pyo3::exceptions::PyKeyError>(obj.py()) => Ok(None),
            Err(e) => Err(e),
        }}
    }};
    let tag = item({key:?})?.ok_or_else(|| ::pyo3::exceptions::PyKeyError::new_err({path:?}))?;
    let tag = tag.extract::<::pyo3::pybacked::PyBackedStr>()?;
    Ok(match &*tag {{
{arms}        other => return Err(::pyo3::exceptions::PyValueError::new_err(format!(\"{owner}: unexpected {key} {{other:?}}\"))),
    }})
}}
",
        path = format!("{owner}.{key}"),
    ))
}

/// The variant built from the dict's keys.
fn build(r: &RecordDecl, v: &Variant) -> Result<String, String> {
    let rust = &r.rust;
    let fields: &[Field] = match &v.shape {
        VariantShape::Unit => return Ok(format!("{rust}::{}", v.rust_name)),
        VariantShape::Named(fields) => fields,
        _ => return Err(format!("{rust}::{}: a tagged variant has named keys", v.rust_name)),
    };
    let mut reads = Vec::new();
    for f in fields {
        let key = f.host_name();
        let path = format!("{}.{key}", r.name);
        let reader = read(&f.ty, "v", &path)?;
        let absent = if let Some(default) = &f.default {
            crate::adapter::rust_literal(default, &f.ty)
        } else if f.conversion.missing_default || matches!(f.ty, Ty::Optional(_)) {
            "::std::default::Default::default()".to_owned()
        } else {
            format!("return Err(::pyo3::exceptions::PyKeyError::new_err({path:?}))")
        };
        // `missing_default`: a `None` reads as absent too.
        let present = if f.conversion.missing_default { ".filter(|v| !v.is_none())" } else { "" };
        reads.push(format!(
            "{}: match item({key:?})?{present} {{ Some(v) => {{ let v = &v; ({reader})? }} None => {absent} }}",
            f.rust_name
        ));
    }
    Ok(format!("{rust}::{} {{ {} }}", v.rust_name, reads.join(", ")))
}
