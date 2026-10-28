//! Forwarding modules: Python classes whose every member is one core call, written from the ABI
//! declarations, so no hand-written module only forwards.

use crate::ir::*;
use crate::python::{Out, docstring};
use crate::stub::signature_parts;

pub fn emit(decls: &[Decl], x: &Exposed) -> Result<Vec<Out>, String> {
    x.str_classes.iter().map(|c| str_class(decls, x, c)).collect()
}

fn export<'a>(x: &'a Exposed, c: &StrClass, name: &str) -> Result<&'a ExportFn, String> {
    x.fns
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| format!("{}: {}: `{name}` is not an exported fn", c.source, c.name))
}

/// The one string parameter an applied fn takes: `(its stub text, its name)`.
fn one_str(decls: &[Decl], x: &Exposed, c: &StrClass, f: &ExportFn) -> Result<(String, String), String> {
    let (params, _) = signature_parts(decls, x, &f.sig)?;
    let [(p, false)] = params.as_slice() else {
        return Err(format!("{}: {}: `{}` must take the string alone", c.source, c.name, f.name));
    };
    let name = p.split(':').next().unwrap_or(p).trim().to_owned();
    Ok((p.clone(), name))
}

fn str_class(decls: &[Decl], x: &Exposed, c: &StrClass) -> Result<Out, String> {
    let mut b = String::new();
    b.push_str("# GENERATED FILE -- DO NOT EDIT.\n#\n");
    b.push_str(
        "# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[python(str_class(..))]` declaration in:\n",
    );
    b.push_str(&format!("#     {}\n", c.source));
    b.push_str(
        "#\n# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a\n\
         # non-empty diff, so a hand edit here is reverted rather than merged.\n\n",
    );
    b.push_str("from __future__ import annotations\n\nfrom typing import Any\n\n");
    b.push_str("from pydantic import GetCoreSchemaHandler\nfrom pydantic_core import CoreSchema, core_schema\n\n");
    b.push_str("from .. import _core\n\n\n");
    b.push_str(&format!("class {}(str):\n", c.name));
    if docstring(&mut b, &c.doc, 4) {
        b.push('\n');
    }
    let new = export(x, c, &c.new)?;
    let (param, arg) = one_str(decls, x, c, new)?;
    b.push_str(&format!("    def __new__(cls, {param}) -> {}:\n", c.name));
    b.push_str(&format!("        return super().__new__(cls, _core.{}({arg}))\n\n", new.name));
    b.push_str(
        "    @classmethod\n    def __get_pydantic_core_schema__(\n        cls, source_type: Any, handler: GetCoreSchemaHandler\n    ) -> CoreSchema:\n        return core_schema.no_info_plain_validator_function(\n            cls,\n            serialization=core_schema.to_string_ser_schema(),\n        )\n",
    );
    for (members, property) in [(&c.properties, true), (&c.methods, false)] {
        for (py, f) in members {
            let f = export(x, c, f)?;
            one_str(decls, x, c, f)?;
            let (_, ret) = signature_parts(decls, x, &f.sig)?;
            b.push('\n');
            if property {
                b.push_str("    @property\n");
            }
            b.push_str(&format!("    def {py}(self) -> {ret}:\n"));
            docstring(&mut b, &f.doc, 8);
            b.push_str(&format!("        return _core.{}(self)\n", f.name));
        }
    }
    if let Some(path) = &c.path {
        b.push_str(&format!("\n\n{}.__module__ = \"{path}\"\n", c.name));
    }
    Ok(Out { path: format!("{}.py", c.module), text: b })
}
