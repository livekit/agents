//! Public re-export modules: every generated name whose declaration places it in a public module
//! (`path = "livekit.agents.<module>"`) is re-exported by `_generated/public_<module>.py`,
//! which that public module star-imports and lists in its literal `__all__`. The public modules
//! hold no hand-written import list of generated names.

use std::collections::BTreeMap;

use crate::decl::{Declarations, RecordBody};
use crate::python::Out;

const PACKAGE: &str = "livekit.agents";

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

/// Every placed name by public path: `(generated module, name)`, in declaration order.
pub fn placed(declared: &Declarations) -> BTreeMap<String, Vec<(String, String)>> {
    let mut placed = BTreeMap::<String, Vec<(String, String)>>::new();
    let mut add = |path: &str, module: &str, name: &str| {
        // A private name stays the public module's own import.
        if name.starts_with('_') {
            return;
        }
        let names = placed.entry(path.to_owned()).or_default();
        if !names.iter().any(|(_, n)| n == name) {
            names.push((module.to_owned(), name.to_owned()));
        }
    };
    for record in &declared.records {
        let Some(path) = &record.path else { continue };
        if record.name.contains('.') || record.host_type.is_some() {
            continue;
        }
        add(path, &record.module, &record.name);
        if let RecordBody::Variants(variants) = &record.body
            && !record.transparent
        {
            for variant in variants {
                add(path, &record.module, &variant.name);
            }
        }
    }
    for family in &declared.exception_families {
        for class in family.class_names() {
            add(family.path(), family.module(), class);
        }
    }
    for flag in &declared.flags {
        add(&flag.path, &flag.module, &flag.name);
    }
    placed
}

pub fn emit(declared: &Declarations) -> Result<Vec<Out>, String> {
    let mut out = Vec::new();
    for (path, names) in placed(declared) {
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
            let flat = format!("from .{source} import {}\n", items.join(", "));
            if flat.len() <= 101 {
                text.push_str(&flat);
            } else {
                text.push_str(&format!("from .{source} import (\n"));
                for item in items {
                    text.push_str(&format!("    {item},\n"));
                }
                text.push_str(")\n");
            }
        }
        text.push_str("\n__all__ = [\n");
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
        let out = super::emit(&declarations).unwrap();
        let paths: Vec<&str> = out.iter().map(|o| o.path.as_str()).collect();
        assert_eq!(paths, ["public.py", "public_voice_io.py"]);
        assert!(out[1].text.contains("from .turn import EndOfTurn as EndOfTurn\n"));
        assert!(out[1].text.contains("__all__ = [\n    \"EndOfTurn\",\n]\n"));
        assert!(!out.iter().any(|o| o.text.contains("Private")));
    }
}
