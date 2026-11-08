//! Public re-export modules: every generated name whose declaration places it in a public module
//! (`path = "livekit.agents.<module>"`) is re-exported by `_generated/public_<module>.py`,
//! which that public module star-imports and lists in its literal `__all__`. The public modules
//! hold no hand-written import list of generated names.

use std::collections::BTreeMap;

use crate::decl::{Declarations, Form, RecordBody};
use crate::python::Out;

const PACKAGE: &str = "livekit.agents";
/// The native module, and its import from `_generated`.
const NATIVE_MODULE: &str = "livekit.agents._core";
const CORE: &str = ".._core";

/// The generated module re-exporting `path`'s names: `public_inference_stt` for
/// `livekit.agents.inference.stt`, `public` for the package itself.
pub fn module(path: &str) -> Option<String> {
    let rest = path.strip_prefix(PACKAGE)?;
    match rest.strip_prefix('.') {
        Some(rest) => Some(format!("public_{}", rest.replace('.', "_"))),
        None if rest.is_empty() => Some("public".to_owned()),
        None => None,
    }
}

/// Every placed name by public path: `(module it is imported from, relative to `_generated`,
/// name)`, in declaration order. A native class reporting a public module (`class(module = ..)`)
/// is placed there, and so are a plugin base file's public classes (their host class's module).
pub fn placed(
    declared: &Declarations,
    bases: &[crate::bases::Bases],
) -> BTreeMap<String, Vec<(String, String)>> {
    let mut placed = BTreeMap::<String, Vec<(String, String)>>::new();
    let mut add = |path: &str, source: String, name: &str| {
        // A private name stays the public module's own import.
        if name.starts_with('_') {
            return;
        }
        let names = placed.entry(path.to_owned()).or_default();
        if !names.iter().any(|(_, n)| n == name) {
            names.push((source, name.to_owned()));
        }
    };
    let generated = |module: &str| format!(".{module}");
    for record in &declared.records {
        let Some(path) = &record.path else { continue };
        if record.name.contains('.') || record.host_type.is_some() {
            continue;
        }
        add(path, generated(&record.module), &record.name);
        if let Some(missing) = &record.missing {
            add(path, generated(&record.module), missing);
        }
        if let RecordBody::Variants(variants) = &record.body
            && !record.transparent
        {
            for variant in variants {
                add(path, generated(&record.module), &variant.name);
            }
        }
    }
    for family in &declared.exception_families {
        for class in family.class_names() {
            add(family.path(), generated(family.module()), class);
        }
    }
    for flag in &declared.flags {
        add(&flag.path, generated(&flag.module), &flag.name);
    }
    for constant in &declared.constants {
        if let Some(path) = &constant.path {
            add(path, generated(&constant.module), &constant.name);
        }
    }
    for class in &declared.str_classes {
        if let Some(path) = &class.path {
            add(path, generated(&class.module), &class.name);
        }
    }
    for base in bases {
        for (class, path) in &base.paths {
            add(path, generated(&base.module), class);
        }
    }
    for class in &declared.classes {
        if class.form != Form::Functions && class.host_module != NATIVE_MODULE {
            add(&class.host_module, CORE.to_owned(), &class.name);
        }
        for method in &class.methods {
            if let Some(path) = &method.path {
                add(path, CORE.to_owned(), &method.name);
            }
        }
    }
    placed
}

pub fn emit(declared: &Declarations, bases: &[crate::bases::Bases]) -> Result<Vec<Out>, String> {
    let mut out = Vec::new();
    for (path, names) in placed(declared, bases) {
        let Some(module) = module(&path) else {
            return Err(format!("{path}: a public path lies under {PACKAGE}"));
        };
        let mut text = format!(
            "# GENERATED FILE -- DO NOT EDIT.\n# The names lk-agents-abi places in `{path}`, which star-imports this module.\n\nfrom __future__ import annotations\n\n"
        );
        let mut by_module = BTreeMap::<&str, Vec<&str>>::new();
        for (source, name) in &names {
            by_module.entry(source).or_default().push(name);
        }
        for (source, mut items) in by_module {
            items.sort_unstable();
            let items: Vec<String> = items.iter().map(|n| format!("{n} as {n}")).collect();
            let flat = format!("from {source} import {}\n", items.join(", "));
            if flat.len() <= 101 {
                text.push_str(&flat);
            } else {
                text.push_str(&format!("from {source} import (\n"));
                for item in items {
                    text.push_str(&format!("    {item},\n"));
                }
                text.push_str(")\n");
            }
        }
        text.push_str("\n__all__: list[str] = [\n");
        for (_, name) in &names {
            text.push_str(&format!("    {name:?},\n"));
        }
        text.push_str("]\n");
        out.push(Out { path: format!("{module}.py"), text });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_placed_record_is_re_exported_from_its_public_module() {
        let declarations = crate::decl::tests::read(
            r#"
            #[crate::host(record(module = "turn", path = "livekit.agents.voice.io", dataclass))]
            pub struct EndOfTurn { pub delay: f64 }
            #[crate::host(record(module = "turn", path = "livekit.agents", dataclass))]
            pub struct Usage { pub tokens: u32 }
            #[crate::host(record(module = "plain", dataclass))]
            pub struct Private { pub tokens: u32 }
            "#,
        )
        .unwrap();
        let out = super::emit(&declarations, &[]).unwrap();
        let paths: Vec<&str> = out.iter().map(|o| o.path.as_str()).collect();
        assert_eq!(paths, ["public.py", "public_voice_io.py"]);
        assert!(out[1].text.contains("from .turn import EndOfTurn as EndOfTurn\n"));
        assert!(out[1].text.contains("__all__: list[str] = [\n    \"EndOfTurn\",\n]\n"));
        assert!(!out.iter().any(|o| o.text.contains("Private")));
    }
}
