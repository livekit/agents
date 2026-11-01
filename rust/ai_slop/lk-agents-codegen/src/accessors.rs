//! Interpreter-local interface readers and the Python emitter. The emitter consumes the same
//! resolved declarations as classes, proxies and values; it never scans source or infers keys.
use crate::decl::tree::Crate;
use crate::decl::vocabulary::{self as voc, Entry};
use crate::decl::{
    Declarations, LocalIdentity, LocalInterface, LocalMember, LocalMethod, LocalProbe,
};
use crate::python::Out;

pub fn read(backend: &Crate) -> Result<Vec<LocalInterface>, String> {
    let mut interfaces = Vec::new();
    for (module, scope) in backend.modules.iter().enumerate() {
        for (index, item) in scope.items.iter().enumerate() {
            let syn::Item::Struct(item) = item else { continue };
            let entries = voc::read(&item.attrs)?;
            let Some(interface) = voc::find(&entries, "interface") else { continue };
            voc::only(&scope.file, &entries, &["interface"])?;
            if !matches!(item.fields, syn::Fields::Unit) || !item.generics.params.is_empty() {
                return Err(format!("{}: an interpreter interface is a unit struct", item.ident));
            }
            let name = item.ident.to_string();
            if interfaces.iter().any(|i: &LocalInterface| i.name == name) {
                return Err(format!("duplicate local interface {name}"));
            }
            let mut group = interface.clone();
            group.key = name;
            let mut resolved = resolve(&group)?;
            resolved.rust_path =
                backend.extern_path(module, index)?.replacen("::lk_agents_py", "crate", 1);
            interfaces.push(resolved);
        }
    }
    Ok(interfaces)
}

fn resolve(group: &Entry) -> Result<LocalInterface, String> {
    let name = &group.key;
    voc::only(
        name,
        &group.nested,
        &["module", "host", "protocol", "attrs", "items", "methods", "protocols"],
    )?;
    let module = voc::text(name, &group.nested, "module")?;
    let host = voc::text(name, &group.nested, "host")?;
    let protocol = voc::text(name, &group.nested, "protocol")?;
    let identity = match (module, host, protocol) {
        (Some(module), None, None) => LocalIdentity::Module(module),
        (None, Some(host), None) => {
            let (module, name) =
                host.rsplit_once('.').ok_or_else(|| format!("{name}: a type has a dotted path"))?;
            LocalIdentity::Type { module: module.into(), name: name.into() }
        }
        (None, None, Some(protocol)) => LocalIdentity::Structural(protocol),
        _ => {
            return Err(format!(
                "{name}: declare exactly one module, host type or structural protocol"
            ));
        }
    };
    let mut out = LocalInterface {
        name: name.clone(),
        rust_path: String::new(),
        identity,
        members: Vec::new(),
        methods: Vec::new(),
        probes: Vec::new(),
    };
    for kind in &group.nested {
        if ["module", "host", "protocol"].contains(&kind.key.as_str()) {
            continue;
        }
        for field in &kind.nested {
            match kind.key.as_str() {
                "attrs" | "items" => {
                    voc::only(name, &field.nested, &["name", "optional", "write"])?;
                    let optional = voc::find(&field.nested, "optional").is_some();
                    if optional && kind.key != "items" {
                        return Err(format!("{name}: optional access is a dictionary item"));
                    }
                    out.members.push(LocalMember {
                        name: field.key.clone(),
                        key: voc::text(name, &field.nested, "name")?.unwrap_or(field.key.clone()),
                        item: kind.key == "items",
                        optional,
                        write: voc::find(&field.nested, "write").is_some(),
                    });
                }
                "methods" => {
                    voc::only(name, &field.nested, &["args"])?;
                    let args = voc::find(&field.nested, "args").ok_or("method needs args")?;
                    let params = args
                        .nested
                        .iter()
                        .map(|arg| {
                            voc::only(name, &arg.nested, &["kw"])?;
                            Ok((arg.key.clone(), voc::find(&arg.nested, "kw").is_some()))
                        })
                        .collect::<Result<_, String>>()?;
                    out.methods.push(LocalMethod { name: field.key.clone(), params });
                }
                "protocols" => {
                    if !matches!(out.identity, LocalIdentity::Structural(_)) {
                        return Err(format!(
                            "{name}: nominal interfaces do not probe guaranteed members"
                        ));
                    }
                    voc::only(name, &field.nested, &["required"])?;
                    let required = voc::find(&field.nested, "required")
                        .ok_or("protocol needs required members")?;
                    if required.nested.is_empty() {
                        return Err("protocol needs at least one required member".into());
                    }
                    out.probes.push(LocalProbe {
                        name: field.key.clone(),
                        required: required.nested.iter().map(|m| m.key.clone()).collect(),
                    });
                }
                _ => unreachable!("validated kind"),
            }
        }
    }
    Ok(out)
}

pub fn emit(decls: &Declarations) -> Out {
    let mut text = String::from(
        "//! GENERATED FILE -- DO NOT EDIT.\n//! Interpreter interfaces resolved with the binding declarations.\nuse pyo3::prelude::*;\n",
    );
    for interface in &decls.local_interfaces {
        let name = &interface.name;
        let identity = match &interface.identity {
            LocalIdentity::Module(module) => module.clone(),
            LocalIdentity::Type { module, name } => format!("{module}.{name}"),
            LocalIdentity::Structural(protocol) => protocol.clone(),
        };
        text.push_str(&format!(
            "\n/// Python interface: `{identity}`.\npub use {} as {name};\nimpl {name} {{\n",
            interface.rust_path
        ));
        match &interface.identity {
            LocalIdentity::Module(module) => text.push_str(&format!("pub fn import(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {{ py.import({module:?}) }}\n")),
            LocalIdentity::Type { module, name } => text.push_str(&format!("pub fn import(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {{ static CLASS: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> = pyo3::sync::PyOnceLock::new(); Ok(CLASS.import(py, {module:?}, {name:?})?.as_any().clone()) }}\n")),
            LocalIdentity::Structural(_) => {},
        }
        for m in &interface.members {
            let name = &m.name;
            let key = &m.key;
            let get = if m.item { "get_item" } else { "getattr" };
            if m.optional {
                text.push_str(&format!("pub fn {name}<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Option<Bound<'py, PyAny>>> {{ obj.as_any().cast::<pyo3::types::PyDict>()?.get_item({key:?}) }}\n"));
            } else {
                text.push_str(&format!("pub fn {name}<'py>(obj: &Bound<'py, impl pyo3::PyTypeInfo>) -> PyResult<Bound<'py, PyAny>> {{ obj.as_any().{get}({key:?}) }}\n"));
            }
            if m.write {
                let (prefix, set) = if m.item { ("put", "set_item") } else { ("set", "setattr") };
                text.push_str(&format!("pub fn {prefix}_{name}<'py, V>(obj: &Bound<'py, PyAny>, value: V) -> PyResult<()> where V: IntoPyObject<'py> {{ obj.{set}({key:?}, value) }}\n"));
            }
        }
        for method in &interface.methods {
            let name = &method.name;
            let params = method
                .params
                .iter()
                .map(|(p, _)| format!("{p}: &Bound<'py, PyAny>"))
                .collect::<Vec<_>>()
                .join(", ");
            let positional = method
                .params
                .iter()
                .filter(|(_, kw)| !kw)
                .map(|(p, _)| p.as_str())
                .collect::<Vec<_>>();
            let arguments = if positional.is_empty() {
                "()".into()
            } else {
                format!("({},)", positional.join(", "))
            };
            let keywords = method
                .params
                .iter()
                .filter(|(_, kw)| *kw)
                .map(|(p, _)| format!("kwargs.set_item({p:?}, {p})?;"))
                .collect::<String>();
            let kwargs = if keywords.is_empty() { "None" } else { "Some(&kwargs)" };
            text.push_str(&format!("pub fn {name}<'py>(obj: &Bound<'py, PyAny>, {params}) -> PyResult<Bound<'py, PyAny>> {{ let py = obj.py(); let kwargs = pyo3::types::PyDict::new(py); {keywords} crate::py_calls::call(&obj.getattr({name:?})?, {arguments}, {kwargs}) }}\n"));
        }
        for probe in &interface.probes {
            let checks = probe
                .required
                .iter()
                .map(|member| format!("obj.hasattr({member:?})?"))
                .collect::<Vec<_>>()
                .join(" && ");
            let checks = if probe.required.len() == 1 {
                format!("obj.hasattr({:?})", probe.required[0])
            } else {
                format!("Ok({checks})")
            };
            text.push_str(&format!(
                "pub fn {}(obj: &Bound<'_, PyAny>) -> PyResult<bool> {{ {checks} }}\n",
                probe.name
            ));
        }
        text.push_str("}\n");
    }
    Out { path: "accessors.rs".into(), text }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn build(source: &str) -> Result<Declarations, String> {
        let backend = Crate::from_files("lk_agents_py", &[("lib.rs", source)])?;
        Ok(Declarations { local_interfaces: read(&backend)?, ..Declarations::default() })
    }
    #[test]
    fn nominal_identity_mutation_and_keyword_names_are_explicit() {
        let model = build(r#"
            #[host(interface(host = "asyncio.AbstractEventLoop", methods(create_task(args(coro, context(kw))))))]
            pub struct EventLoop;
            #[host(interface(host = "logging.LogRecord", attrs(created(write), name)))]
            pub struct LogRecord;
        "#).unwrap();
        let text = emit(&model).text;
        assert!(text.contains("CLASS.import(py, \"asyncio\", \"AbstractEventLoop\")"));
        assert!(text.contains("kwargs.set_item(\"context\", context)"));
        assert!(text.contains("pub fn set_created"));
        assert!(!text.contains("pub fn set_name"));
        assert!(text.contains("pub use crate::EventLoop as EventLoop"));
        assert!(!text.contains("pub struct EventLoop"));
        for source in [
            r#"#[host(interface(host = "sdk.X", typo(field)))] pub struct X;"#,
            r#"#[host(interface(host = "sdk.X", protocols(matches(required(name)))))] pub struct X;"#,
            r#"#[host(interface(module = "sdk", host = "sdk.X")))] pub struct X;"#,
            r#"#[host(interface(host = "sdk.X"))] pub struct X { field: u8 }"#,
            r#"#[host(interface(host = "sdk.X"), export)] pub struct X;"#,
        ] {
            assert!(build(source).is_err(), "{source}");
        }
    }
}
