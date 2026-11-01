//! Python presentation over the resolved declarations: authored plugin bases and constants.
//! Record models and their codecs are emitted by `records` from that same model.
use crate::decl::ConstValue;
use crate::decl::{Declarations, RecordBody};

pub struct Out {
    pub path: String,
    pub text: String,
}

pub fn emit(decls: &Declarations, bases: &[crate::bases::Bases]) -> Result<Vec<Out>, String> {
    let modules = decls
        .constants
        .iter()
        .map(|c| c.module.as_str())
        .chain(bases.iter().map(|b| b.module.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    let mut out = Vec::new();
    for module in modules {
        let base = bases.iter().find(|b| b.module == module);
        let constants = decls.constants.iter().filter(|c| c.module == module).collect::<Vec<_>>();
        let sources = constants
            .iter()
            .map(|c| c.source.as_str())
            .chain(base.into_iter().flat_map(|b| b.sources.iter().map(String::as_str)))
            .collect::<std::collections::BTreeSet<_>>();
        let mut header = String::from(
            "# GENERATED FILE -- DO NOT EDIT.\n# From ABI declarations and authored plugin bases:\n",
        );
        for source in sources {
            header.push_str(&format!("#     {source}\n"));
        }
        header.push_str("\nfrom __future__ import annotations\n");
        let mut imports = String::new();
        if !constants.is_empty() {
            imports.push_str("from typing import Final\n");
        }
        if let Some(base) = base {
            let mut checking = String::new();
            for (module, name) in &base.imports {
                if module.is_empty() {
                    imports.push_str(&format!("import {name}\n"));
                } else if let Some(module) = module.strip_prefix('?') {
                    checking.push_str(&format!("    from {module} import {name}\n"));
                } else {
                    imports.push_str(&format!("from {module} import {name}\n"));
                }
            }
            if !checking.is_empty() {
                imports.push_str("from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n");
                imports.push_str(&checking);
            }
        }
        let mut body = String::new();
        if let Some(base) = base {
            body.push_str(base.body.trim());
            body.push_str("\n\n\n");
        }
        for constant in &constants {
            let value = match &constant.value {
                ConstValue::Literal(value) => value.clone(),
                ConstValue::Record { rust, fields } => {
                    let record =
                        decls.records.iter().find(|record| record.rust == *rust).ok_or_else(
                            || {
                                format!(
                                    "{}: {}: {rust} is not a declared record",
                                    constant.source, constant.name
                                )
                            },
                        )?;
                    let RecordBody::Struct(members) = &record.body else {
                        return Err("a record constant constructs a struct".into());
                    };
                    if record.module != module {
                        imports
                            .push_str(&format!("from .{} import {}\n", record.module, record.name));
                    }
                    let params = fields
                        .iter()
                        .map(|(name, value)| {
                            let field = members
                                .iter()
                                .find(|field| field.rust_name == *name)
                                .ok_or_else(|| format!("{rust} has no field {name}"))?;
                            Ok(format!("{}={value}", field.host_name()))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let call = format!("{}({})", record.name, params.join(", "));
                    if constant.name.len() + call.len() + ": Final = ".len() > 100 {
                        format!("{}(\n    {},\n)", record.name, params.join(",\n    "))
                    } else {
                        call
                    }
                }
            };
            body.push_str(&format!("{}: Final = {value}\n", constant.name));
            docstring(&mut body, &constant.doc, 0);
        }
        if let Some(base) = base {
            body.push_str(
                "\n\n# The public module identity is assigned after all classes are built.\n",
            );
            for (name, path) in &base.paths {
                body.push_str(&format!("{name}.__module__ = {path:?}\n"));
            }
        }
        let text = crate::imports::merge(&header, &imports)?;
        let gap = if base.is_some() { "\n\n\n" } else { "\n\n" };
        out.push(Out {
            path: format!("{module}.py"),
            text: format!("{}{gap}{}", text.trim_end(), body.trim_end()) + "\n",
        });
    }
    Ok(out)
}

/// Returns whether one was written.
pub fn docstring(b: &mut String, doc: &[String], indent: usize) -> bool {
    if doc.is_empty() {
        return false;
    }
    let pad = " ".repeat(indent);
    // Escape `"""`, and a trailing quote, which would close the literal early.
    let esc = |l: &str| l.replace("\"\"\"", "\\\"\"\"");
    if doc.len() == 1 {
        let mut line = esc(&doc[0]);
        if line.ends_with('"') {
            line.pop();
            line.push_str("\\\"");
        }
        b.push_str(&format!("{pad}\"\"\"{line}\"\"\"\n"));
        return true;
    }
    b.push_str(&format!("{pad}\"\"\"{}\n", esc(&doc[0])));
    for line in &doc[1..] {
        if line.is_empty() {
            b.push('\n');
        } else {
            b.push_str(&format!("{pad}{}\n", esc(line)));
        }
    }
    b.push_str(&format!("{pad}\"\"\"\n"));
    true
}
