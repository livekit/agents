//! Whole boundary records: field access, omission and codecs are generated as one conversion.
use super::{emit, read};
use crate::decl::{Field, RecordBody, RecordDecl, Ty};

pub(super) fn record(r: &RecordDecl) -> Result<String, String> {
    let RecordBody::Struct(fields) = &r.body else {
        return Err(format!("{}: a boundary record is a struct", r.rust));
    };
    validate(r, fields)?;
    let name = super::cell(&r.name).to_lowercase();
    let rust = &r.rust;
    let output = r.output.as_deref().unwrap_or(rust);
    let owner = r.error_owner.as_deref().unwrap_or(&r.name);
    let mut text = String::new();
    if r.extract {
        let mut body = String::new();
        if r.default_on_null {
            body.push_str("if obj.is_none() || crate::py_values::is_not_given(obj)? { return Ok(Default::default()); }\n");
        }
        for attrs in source_paths(fields) {
            let variable = source_variable(&attrs);
            let parent = if attrs.len() == 1 {
                "obj".to_owned()
            } else {
                source_variable(&attrs[..attrs.len() - 1])
            };
            let key = attrs.last().expect("source path");
            body.push_str(&format!("let {variable} = {parent}.getattr({key:?})?;\n"));
        }
        for field in fields
            .iter()
            .filter(|f| !f.conversion.late)
            .chain(fields.iter().filter(|f| f.conversion.late))
        {
            let value = field_read(r, field, owner)?;
            body.push_str(&format!("let {} = {value};\n", field.rust_name));
        }
        let names = fields.iter().map(|f| f.rust_name.as_str()).collect::<Vec<_>>().join(", ");
        body.push_str(&format!("let value = {rust} {{ {names} }};\n"));
        if let Some(validate) = &r.validate {
            body.push_str(&format!("{validate}(&value).map_err(|error| crate::py_errors::HostException::to_py_err(&error))?;\n"));
        }
        if let Some(finish) = &r.finish {
            let finish = if r.finish_fallible {
                format!(
                    "{finish}(value).map_err(|error| crate::py_errors::HostException::to_py_err(&error))?"
                )
            } else {
                format!("{finish}(value)")
            };
            body.push_str(&format!("let value = {finish};\n"));
        }
        body.push_str("Ok(value)\n");
        text.push_str(&format!("pub fn {name}_from_py(obj: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<{output}> {{ {body} }}\n"));
    }
    if r.emit {
        let mut sets = String::new();
        let source_roots: Vec<_> = fields
            .iter()
            .filter_map(|f| f.conversion.source.as_ref().and_then(|s| s.attrs.first()))
            .collect();
        if let Some(classes) = &r.emit_classes {
            let field = &classes.field;
            sets.push_str(&format!("let selected_some = value.{field}.is_some();\n"));
        }
        for field in fields.iter().filter(|f| !f.skip) {
            let name = &field.rust_name;
            let key = field.host_name();
            if field.conversion.flatten {
                return Err(format!("{rust}: flattened emission is not declared"));
            }
            let (ty, value) = if field.conversion.omit_none {
                let Ty::Optional(inner) = &field.ty else {
                    return Err(format!("{rust}.{name}: omit_none requires Option"));
                };
                (&**inner, "x".to_owned())
            } else {
                (
                    &field.ty,
                    if field.conversion.getter {
                        format!("value.{name}()")
                    } else {
                        format!("value.{name}")
                    },
                )
            };
            let emitted = match &field.conversion.emit_with {
                Some(codec) => format!("crate::{codec}(py, {value})?"),
                None => emit(ty, &value)?,
            };
            let mut set = if let Some(source) = &field.conversion.source {
                let mut parent =
                    format!("__emitted_{}", source.attrs.first().ok_or("source needs root")?);
                for attr in source.attrs.iter().skip(1) {
                    parent = format!("{parent}.getattr({attr:?})?");
                }
                let (module, key) =
                    source.constant.rsplit_once('.').ok_or("source constant needs dotted path")?;
                format!(
                    "{parent}.set_item(py.import({module:?})?.getattr({key:?})?, {emitted})?;\n"
                )
            } else if source_roots.contains(&name) {
                format!(
                    "let __emitted_{name} = {emitted}; kwargs.set_item({key:?}, &__emitted_{name})?;\n"
                )
            } else {
                format!("kwargs.set_item({key:?}, {emitted})?;\n")
            };
            if field.conversion.omit_empty {
                set = format!("if !value.{name}.is_empty() {{ {set} }}\n");
            }
            if field.conversion.emit_if_none.is_some() {
                set = format!("if !selected_some {{ {set} }}\n");
            }
            if field.conversion.omit_none {
                sets.push_str(&format!("if let Some(x) = value.{name} {{ {set} }}\n"));
            } else {
                sets.push_str(&set);
            }
        }
        let generated_host = format!("livekit.agents._generated.{}.{}", r.module, r.name);
        let host = r.host_type.as_ref().unwrap_or(&generated_host);
        let import = |slot: &str, host: &str| -> Result<String, String> {
            let (module, class) = host.rsplit_once('.').ok_or("host_type needs dotted path")?;
            Ok(format!(
                "{{ static {slot}: pyo3::sync::PyOnceLock<pyo3::Py<pyo3::types::PyType>> = pyo3::sync::PyOnceLock::new(); {slot}.import(py, {module:?}, {class:?})?.as_any() }}"
            ))
        };
        let class = match &r.emit_classes {
            Some(classes) => format!(
                "if selected_some {{ {} }} else {{ {} }}",
                import("SOME", &classes.some)?,
                import("NONE", &classes.none)?
            ),
            None => import("CLASS", host)?,
        };
        text.push_str(&format!("pub fn {name}_to_py<'py>(py: pyo3::Python<'py>, value: {output}) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {{
            let kwargs = pyo3::types::PyDict::new(py); {sets}
            crate::py_calls::call({class}, (), Some(&kwargs))
        }}\n"));
    }
    Ok(text)
}

fn field_read(r: &RecordDecl, f: &Field, owner: &str) -> Result<String, String> {
    let c = &f.conversion;
    if f.skip {
        return Ok("Default::default()".into());
    }
    let path = format!("{}.{}", r.error_owner.as_deref().unwrap_or(&r.name), f.host_name());
    let convert = |value: &str| -> Result<String, String> {
        Ok(match &c.extract_with {
            Some(codec) => format!("crate::{codec}({value}, {path:?})"),
            None => read(&f.ty, value, &path)?,
        })
    };
    let value = if let Some(source) = &c.source {
        let parent = source_variable(&source.attrs);
        let (module, key) =
            source.constant.rsplit_once('.').ok_or("source constant needs dotted path")?;
        let reader = convert("v")?;
        format!(
            "{parent}.cast::<pyo3::types::PyDict>()?.get_item(obj.py().import({module:?})?.getattr({key:?})?)?.filter(|v| !v.is_none()).map(|v| -> PyResult<_> {{ let v = &v; {reader} }}).transpose()?.unwrap_or_default()"
        )
    } else if c.flatten {
        format!("({})?", convert("obj")?)
    } else if c.own_value {
        // A string subclass's value is distinct from any of its attributes.
        if c.extract_with.is_some() {
            format!("({})?", convert("obj")?)
        } else {
            "obj.str()?.extract::<String>()?".into()
        }
    } else {
        let key = f.host_name();
        let get = if r.mapping { "get_item" } else { "getattr" };
        let mut reader = convert("v")?;
        if c.when.is_some() || c.none_default {
            let condition = c
                .when
                .as_ref()
                .map(|(field, variant)| format!("{field} == {variant}"))
                .unwrap_or_else(|| "true".into());
            reader = format!(
                "if {condition} && !v.is_none() {{ {reader} }} else {{ Ok(Default::default()) }}"
            );
        }
        if r.dict_keys {
            format!(
                "{{ let member = obj.cast::<pyo3::types::PyDict>()?.get_item({key:?})?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err({path:?}))?; let v = &member; ({reader})? }}"
            )
        } else if c.key_error {
            format!(
                "{{ let member = obj.{get}({key:?}).map_err(|_| pyo3::exceptions::PyKeyError::new_err({path:?}))?; let v = &member; ({reader})? }}"
            )
        } else if c.lenient {
            format!(
                "obj.{get}({key:?}).ok().and_then(|v| {{ let v = &v; ({reader}).ok() }}).flatten()"
            )
        } else if c.missing_default {
            format!(
                "obj.{get}({key:?}).ok().filter(|v| !v.is_none()).map(|v| -> PyResult<_> {{ let v = &v; {reader} }}).transpose()?.unwrap_or_default()"
            )
        } else if source_paths_from_record(r).iter().any(|p| p.len() == 1 && p[0] == key.as_ref()) {
            let cached = source_variable(&[key.to_string()]);
            format!("{{ let v = &{cached}; ({reader})? }}")
        } else {
            let helper = if r.mapping { "item" } else { "attr" };
            if let Some(default) = &f.default {
                format!(
                    "crate::py_values::{helper}_or(obj, {owner:?}, {key:?}, |v| {{ {reader} }}, || {})?",
                    crate::adapter::rust_literal(default, &f.ty)
                )
            } else {
                format!(
                    "crate::py_values::{helper}_with(obj, {owner:?}, {key:?}, |v| {{ {reader} }})?"
                )
            }
        }
    };
    Ok(match &c.normalize {
        Some(normalize) if c.fallible => format!(
            "{normalize}({value}).map_err(|error| crate::py_errors::HostException::to_py_err(&error))?"
        ),
        Some(normalize) => format!("{normalize}({value})"),
        None => value,
    })
}

fn source_variable(attrs: &[String]) -> String {
    format!("__source_{}", attrs.join("_"))
}
fn source_paths(fields: &[Field]) -> Vec<Vec<String>> {
    let mut paths = Vec::new();
    for source in fields.iter().filter_map(|field| field.conversion.source.as_ref()) {
        for n in 1..=source.attrs.len() {
            let path = source.attrs[..n].to_vec();
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    paths
}
fn source_paths_from_record(record: &RecordDecl) -> Vec<Vec<String>> {
    match &record.body {
        RecordBody::Struct(fields) => source_paths(fields),
        _ => Vec::new(),
    }
}

fn validate(record: &RecordDecl, fields: &[Field]) -> Result<(), String> {
    let at = &record.rust;
    if record.output.is_some() != record.finish.is_some() {
        return Err(format!("{at}: output and finish/try_finish are declared together"));
    }
    if let Some(classes) = &record.emit_classes {
        if !record.emit
            || !fields.iter().any(|field| {
                field.rust_name == classes.field && matches!(field.ty, Ty::Optional(_))
            })
        {
            return Err(format!("{at}: classes selects an optional field of an emitted record"));
        }
        for class in [&classes.some, &classes.none] {
            if !class
                .rsplit_once('.')
                .is_some_and(|(module, name)| !module.is_empty() && !name.is_empty())
            {
                return Err(format!("{at}: an emitted class has a dotted path"));
            }
        }
    }
    for (index, field) in fields.iter().enumerate() {
        let conversion = &field.conversion;
        if conversion.lenient && !matches!(field.ty, Ty::Optional(_)) {
            return Err(format!("{at}.{}: lenient extraction requires Option", field.rust_name));
        }
        if conversion.lenient && conversion.missing_default {
            return Err(format!("{at}.{}: choose lenient or missing_default", field.rust_name));
        }
        if let Some((dependency, _)) = &conversion.when
            && !fields[..index]
                .iter()
                .any(|field| field.rust_name == *dependency && !field.conversion.late)
        {
            return Err(format!(
                "{at}.{}: when refers to an earlier extracted field",
                field.rust_name
            ));
        }
        if let Some(selector) = &conversion.emit_if_none
            && record.emit_classes.as_ref().is_none_or(|classes| classes.field != *selector)
        {
            return Err(format!(
                "{at}.{}: emit_if_none uses the declared class selector",
                field.rust_name
            ));
        }
        if let Some(source) = &conversion.source {
            if source.attrs.is_empty()
                || !fields.iter().any(|field| {
                    field.host_name() == source.attrs[0] && field.conversion.source.is_none()
                })
            {
                return Err(format!(
                    "{at}.{}: source starts at a declared host field",
                    field.rust_name
                ));
            }
            if !source
                .constant
                .rsplit_once('.')
                .is_some_and(|(module, name)| !module.is_empty() && !name.is_empty())
            {
                return Err(format!("{at}.{}: source constant has a dotted path", field.rust_name));
            }
        }
        for codec in [&conversion.extract_with, &conversion.emit_with].into_iter().flatten() {
            if syn::parse_str::<syn::Path>(codec).is_err() {
                return Err(format!(
                    "{at}.{}: a codec is a binding function path",
                    field.rust_name
                ));
            }
        }
    }
    Ok(())
}
