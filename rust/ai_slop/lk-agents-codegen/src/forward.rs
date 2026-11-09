use crate::decl::*;
use crate::python::{Out, docstring};

pub fn emit(declared: &crate::decl::Declarations) -> Result<Vec<Out>, String> {
    declared.str_classes.iter().map(|c| str_class(declared, c)).collect()
}

fn header(b: &mut String, sources: &[&str]) {
    b.push_str("# GENERATED FILE -- DO NOT EDIT.\n#\n");
    b.push_str(
        "# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the declarations in:\n",
    );
    for s in sources {
        b.push_str(&format!("#     {s}\n"));
    }
    b.push_str(
        "#\n# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a\n\
         # non-empty diff, so a hand edit here is reverted rather than merged.\n\n",
    );
    b.push_str("from __future__ import annotations\n\n");
}

/// A function a str class applies to its string: its host name, doc, and stub parameters and result.
struct Applied {
    name: String,
    doc: Vec<String>,
    params: Vec<(String, bool)>,
    ret: String,
}

/// `name` as an export or a declared function (`#[host(functions)]`).
fn applied(
    declared: &crate::decl::Declarations,
    c: &StrClass,
    name: &str,
) -> Result<Applied, String> {
    let found = declared
        .classes
        .iter()
        .filter(|class| class.form == crate::decl::Form::Functions)
        .find_map(|class| class.methods.iter().find(|m| m.name == name).map(|m| (class, m)));
    let Some((class, m)) = found else {
        return Err(format!("{}: {}: `{name}` is not an exported fn", c.source, c.name));
    };
    let mut imports = crate::class_stub::Imports::new(declared);
    let (params, ret) = crate::class_stub::signature_parts(class, m, &mut imports);
    Ok(Applied { name: m.name.clone(), doc: m.doc.clone(), params, ret })
}

/// The one string parameter an applied fn takes: `(its stub text, its name)`.
fn one_str(c: &StrClass, f: &Applied) -> Result<(String, String), String> {
    let [(p, false)] = f.params.as_slice() else {
        return Err(format!("{}: {}: `{}` must take the string alone", c.source, c.name, f.name));
    };
    let name = p.split(':').next().unwrap_or(p).trim().to_owned();
    Ok((p.clone(), name))
}

fn str_class(declared: &crate::decl::Declarations, c: &StrClass) -> Result<Out, String> {
    let mut b = String::new();
    header(&mut b, &[c.source.as_str()]);
    b.push_str("from typing import Any\n\n");
    b.push_str("from pydantic import GetCoreSchemaHandler\nfrom pydantic_core import CoreSchema, core_schema\n\n");
    b.push_str("from .. import _core\n\n\n");
    b.push_str(&format!("class {}(str):\n", c.name));
    if docstring(&mut b, &c.doc, 4) {
        b.push('\n');
    }
    let new = applied(declared, c, &c.new)?;
    let (param, arg) = one_str(c, &new)?;
    b.push_str(&format!("    def __new__(cls, {param}) -> {}:\n", c.name));
    b.push_str(&format!("        return super().__new__(cls, _core.{}({arg}))\n\n", new.name));
    b.push_str(
        "    @classmethod\n    def __get_pydantic_core_schema__(\n        cls, source_type: Any, handler: GetCoreSchemaHandler\n    ) -> CoreSchema:\n        return core_schema.no_info_plain_validator_function(\n            cls,\n            serialization=core_schema.to_string_ser_schema(),\n        )\n",
    );
    for (members, property) in [(&c.properties, true), (&c.methods, false)] {
        for (py, f) in members {
            let f = applied(declared, c, f)?;
            one_str(c, &f)?;
            b.push('\n');
            if property {
                b.push_str("    @property\n");
            }
            b.push_str(&format!("    def {py}(self) -> {}:\n", f.ret));
            docstring(&mut b, &f.doc, 8);
            b.push_str(&format!("        return _core.{}(self)\n", f.name));
        }
    }
    if let Some(path) = &c.path {
        b.push_str(&format!("\n\n{}.__module__ = \"{path}\"\n", c.name));
    }
    Ok(Out { path: format!("{}.py", c.module), text: b })
}
