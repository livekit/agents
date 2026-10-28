//! Forwarding modules: Python classes whose every member is one core call, written from the ABI
//! declarations, so no hand-written module only forwards.

use crate::ir::*;
use crate::python::{Out, docstring};
use crate::stub::signature_parts;

pub fn emit(decls: &[Decl], x: &Exposed) -> Result<Vec<Out>, String> {
    let mut out: Vec<Out> =
        x.str_classes.iter().map(|c| str_class(decls, x, c)).collect::<Result<_, _>>()?;
    let mut modules: Vec<&str> = Vec::new();
    for m in
        x.wrappers.iter().map(|w| w.module.as_str()).chain(x.forwards.iter().map(|f| f.0.as_str()))
    {
        if !modules.contains(&m) {
            modules.push(m);
        }
    }
    for m in modules {
        if out.iter().any(|o| o.path == format!("{m}.py")) {
            return Err(format!(
                "{m}: a forwarding module holds a str class or wrappers, not both"
            ));
        }
        out.push(wrapper_module(decls, x, m)?);
    }
    Ok(out)
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

/// One `from X import a, b`, wrapped as ruff formats it.
fn from_import(b: &mut String, module: &str, names: &[String]) {
    let flat = format!("from {module} import {}\n", names.join(", "));
    if flat.len() <= 101 {
        b.push_str(&flat);
        return;
    }
    b.push_str(&format!("from {module} import (\n"));
    for n in names {
        b.push_str(&format!("    {n},\n"));
    }
    b.push_str(")\n");
}

/// The wrappers and forwarded exports of one module: the classes, then the re-exported fns.
fn wrapper_module(decls: &[Decl], x: &Exposed, module: &str) -> Result<Out, String> {
    let wrappers: Vec<&Wrapper> = x.wrappers.iter().filter(|w| w.module == module).collect();
    let mut forwards: Vec<String> =
        x.forwards.iter().filter(|f| f.0 == module).map(|f| f.1.clone()).collect();
    forwards.sort();
    let mut sources: Vec<&str> = Vec::new();
    for s in wrappers
        .iter()
        .map(|w| w.source.as_str())
        .chain(x.forwards.iter().filter(|f| f.0 == module).map(|f| f.2.as_str()))
    {
        if !sources.contains(&s) {
            sources.push(s);
        }
    }
    let mut b = String::new();
    header(&mut b, &sources);
    // `base` is "module:Name"; bases from one module share an import.
    let mut bases: Vec<(String, Vec<String>)> = Vec::new();
    for w in &wrappers {
        let (m, n) = w.base.split_once(':').ok_or_else(|| {
            format!("{}: {}: base is \"module:Name\", not `{}`", w.source, w.name, w.base)
        })?;
        match bases.iter_mut().find(|(bm, _)| bm == m) {
            Some((_, names)) => names.push(n.to_owned()),
            None => bases.push((m.to_owned(), vec![n.to_owned()])),
        }
    }
    let mut rel: Vec<(String, Vec<String>)> = Vec::new();
    if !wrappers.is_empty() {
        rel.push(("..".to_owned(), vec!["_core".to_owned()]));
    }
    if !forwards.is_empty() {
        rel.push((".._core".to_owned(), forwards.clone()));
    }
    rel.extend(bases);
    rel.sort_by(|a, b| a.0.cmp(&b.0));
    for (m, names) in &mut rel {
        names.sort();
        from_import(&mut b, m, names);
    }
    let mut all: Vec<String> = wrappers.iter().map(|w| w.name.clone()).collect();
    all.extend(forwards.iter().cloned());
    b.push_str("\n__all__ = [\n");
    for n in &all {
        b.push_str(&format!("    \"{n}\",\n"));
    }
    b.push_str("]\n");
    for w in &wrappers {
        let o = x.objects.iter().find(|o| o.name == w.core).expect("the wrapper's own object");
        let new =
            o.methods.iter().find(|m| m.kind == MethodKind::New).ok_or_else(|| {
                format!("{}: {}: the object has no #[host(new)]", w.source, w.name)
            })?;
        let (params, _) = signature_parts(decls, x, &new.sig)?;
        let base = w.base.split_once(':').map_or(w.base.as_str(), |(_, n)| n);
        b.push_str(&format!("\n\nclass {}({base}):\n", w.name));
        if docstring(&mut b, &w.doc, 4) {
            b.push('\n');
        }
        let mut parts: Vec<String> = vec!["self".to_owned()];
        let mut star = false;
        let mut args: Vec<String> = Vec::new();
        for (p, kw) in &params {
            if *kw && !star {
                parts.push("*".to_owned());
                star = true;
            }
            parts.push(p.clone());
            let name = p.split(':').next().unwrap_or(p).trim().to_owned();
            args.push(if star { format!("{name}={name}") } else { name });
        }
        let flat = format!("    def __init__({}) -> None:\n", parts.join(", "));
        if flat.len() <= 101 {
            b.push_str(&flat);
        } else {
            b.push_str("    def __init__(\n");
            for p in &parts {
                b.push_str(&format!("        {p},\n"));
            }
            b.push_str("    ) -> None:\n");
        }
        let call = format!("        self._core = _core.{}({})\n", w.core, args.join(", "));
        if call.len() <= 101 {
            b.push_str(&call);
        } else {
            b.push_str(&format!("        self._core = _core.{}(\n", w.core));
            for a in &args {
                b.push_str(&format!("            {a},\n"));
            }
            b.push_str("        )\n");
        }
    }
    let mut fixups: Vec<String> = Vec::new();
    for w in &wrappers {
        if let Some(path) = &w.path {
            fixups.push(format!("{}.__module__ = \"{path}\"\n", w.name));
        }
    }
    if !fixups.is_empty() {
        b.push_str("\n\n");
        b.push_str(&fixups.concat());
    }
    Ok(Out { path: format!("{module}.py"), text: b })
}

fn export<'a>(x: &'a Exposed, c: &StrClass, name: &str) -> Result<&'a ExportFn, String> {
    x.fns
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| format!("{}: {}: `{name}` is not an exported fn", c.source, c.name))
}

/// The one string parameter an applied fn takes: `(its stub text, its name)`.
fn one_str(
    decls: &[Decl],
    x: &Exposed,
    c: &StrClass,
    f: &ExportFn,
) -> Result<(String, String), String> {
    let (params, _) = signature_parts(decls, x, &f.sig)?;
    let [(p, false)] = params.as_slice() else {
        return Err(format!("{}: {}: `{}` must take the string alone", c.source, c.name, f.name));
    };
    let name = p.split(':').next().unwrap_or(p).trim().to_owned();
    Ok((p.clone(), name))
}

fn str_class(decls: &[Decl], x: &Exposed, c: &StrClass) -> Result<Out, String> {
    let mut b = String::new();
    header(&mut b, &[c.source.as_str()]);
    b.push_str("from typing import Any\n\n");
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
