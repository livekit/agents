//! Export placement is checked on parsed handwritten source, including inactive cfg branches.
//! Identities use lexical item paths rather than line numbers so unrelated edits need no waiver.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    process::Command,
};

use quote::ToTokens as _;
use syn::{parse::Parser as _, punctuated::Punctuated, spanned::Spanned as _};

const BASELINE: &str = "rust/ai_slop/lk-agents-codegen/export-placement-baseline.json";

pub fn check(root: &Path) -> Result<(), String> {
    let output = Command::new("git")
        .args(["ls-files", "-c", "-o", "--exclude-standard", "-z", "--", "rust/crates/**/*.rs"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("export placement: git ls-files: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "export placement: git ls-files: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let paths = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let mut current = BTreeSet::new();
    let mut locations = Vec::new();
    let mut scanned = 0;
    for path in paths.split('\0').filter(|p| !p.is_empty()).collect::<BTreeSet<_>>() {
        if !root.join(path).exists() {
            continue; // Deleted tracked files are no longer declarations.
        }
        let source =
            std::fs::read_to_string(root.join(path)).map_err(|e| format!("{path}: {e}"))?;
        scanned += 1;
        for (identity, line) in violations(path, &source)? {
            locations.push((identity.clone(), line));
            current.insert(identity);
        }
    }
    if scanned == 0 {
        return Err("export placement: no Rust source files found".to_owned());
    }
    let source =
        std::fs::read_to_string(root.join(BASELINE)).map_err(|e| format!("{BASELINE}: {e}"))?;
    let baseline = baseline(&source)?;
    validate(locations, &baseline)?;
    println!(
        "export placement: {} existing identities; baseline {}",
        current.len(),
        baseline.len()
    );
    Ok(())
}

/// A baseline row no current violation matches is stale: it would let that violation return.
fn validate(locations: Vec<(String, usize)>, baseline: &BTreeSet<String>) -> Result<(), String> {
    let current: BTreeSet<_> = locations.iter().map(|(identity, _)| identity.as_str()).collect();
    let stale = baseline
        .iter()
        .filter(|identity| !current.contains(identity.as_str()))
        .map(|identity| format!("stale baseline row: {identity}"));
    let errors: Vec<_> = locations
        .iter()
        .filter(|(identity, _)| !baseline.contains(identity))
        .map(|(identity, line)| format!("{identity} (line {line})"))
        .chain(stale)
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("handwritten export placement violations:\n{}", errors.join("\n")))
    }
}

fn baseline(source: &str) -> Result<BTreeSet<String>, String> {
    // A JSON array of strings is also a Rust array expression; use the existing syntax reader.
    let array: syn::ExprArray = syn::parse_str(source).map_err(|e| format!("{BASELINE}: {e}"))?;
    let mut entries = BTreeSet::new();
    for expression in array.elems {
        let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(value), .. }) = expression else {
            return Err(format!("{BASELINE}: expected string identities"));
        };
        if !entries.insert(value.value()) {
            return Err(format!("{BASELINE}: duplicate identity {}", value.value()));
        }
    }
    Ok(entries)
}

fn violations(path: &str, source: &str) -> Result<Vec<(String, usize)>, String> {
    if source.lines().next().is_some_and(|s| s.starts_with("//! GENERATED FILE -- DO NOT EDIT.")) {
        return Ok(Vec::new());
    }
    let mut visitor = crate::architecture::Scan::source(path, source)?.exports;
    if let Some(error) = visitor.error {
        return Err(format!("{path}: {error}"));
    }
    let mut occurrences = BTreeMap::new();
    for (identity, _) in &mut visitor.found {
        let occurrence = occurrences.entry(identity.clone()).or_insert(0);
        *occurrence += 1;
        if *occurrence > 1 {
            identity.push_str(&format!(" #{}", *occurrence));
        }
    }
    Ok(visitor.found)
}

#[derive(Default)]
pub(crate) struct Exports {
    pub(crate) path: String,
    pub(crate) scope: Vec<String>,
    pub(crate) found: Vec<(String, usize)>,
    pub(crate) error: Option<syn::Error>,
}

impl Exports {
    pub(crate) fn generator(&self) -> bool {
        self.path.starts_with("rust/ai_slop/")
    }

    pub(crate) fn attribute(&mut self, meta: &syn::Meta) -> syn::Result<()> {
        let Some(last) = meta.path().segments.last() else { return Ok(()) };
        let name = last.ident.to_string();
        if name == "cfg_attr" {
            if let syn::Meta::List(list) = meta {
                let metas = Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated
                    .parse2(list.tokens.clone())?;
                for nested in metas.iter().skip(1) {
                    self.attribute(nested)?;
                }
            }
            return Ok(());
        }
        let abi = self.path.starts_with("rust/crates/lk-agents-abi/");
        let kind = match name.as_str() {
            "host" | "python" | "node" | "export" | "class" | "binding" if !abi => Some(name),
            "derive" if !abi => {
                if let syn::Meta::List(list) = meta {
                    let paths = Punctuated::<syn::Path, syn::Token![,]>::parse_terminated
                        .parse2(list.tokens.clone())?;
                    paths
                        .iter()
                        .any(|p| p.segments.last().is_some_and(|s| s.ident == "Host"))
                        .then(|| "derive(Host)".to_owned())
                } else {
                    None
                }
            }
            "pyclass" | "pymethods" | "pyfunction" | "pymodule" => {
                let shell = self.path == "rust/crates/lk-agents-py/src/lib.rs"
                    && self.scope == ["fn _core"]
                    && name == "pymodule";
                (!shell).then_some(name)
            }
            _ => None,
        };
        if let Some(kind) = kind {
            let item =
                if self.scope.is_empty() { "<file>".to_owned() } else { self.scope.join("::") };
            self.found.push((
                format!("{kind}|{}|{item}|{}", self.path, meta.to_token_stream()),
                meta.span().start().line,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_rejects_export_attributes_in_macro_tokens_without_reading_strings() {
        let path = "rust/crates/lk-agents-voice/src/lib.rs";
        for source in [
            "macro_rules! declare { () => { #[pyclass] struct Sneaky; }; } declare!();",
            "macro_rules! declare { () => { #[lk_agents_abi::class] struct Sneaky; }; } declare!();",
            "macro_rules! declare { () => { #[binding(emit)] struct Sneaky; }; } declare!();",
            "fn emit() { quote! { #[host(export)] fn sneaky() {} }; }",
        ] {
            let found = violations(path, source).unwrap();
            assert!(!found.is_empty(), "{source}");
            assert!(validate(found, &BTreeSet::new()).is_err());
        }
        assert!(
            violations(path, "fn emit() { quote!(\"#[pyclass] struct Text;\"); }")
                .unwrap()
                .is_empty()
        );
        let tool = "rust/ai_slop/lk-agents-codegen/src/python.rs";
        assert!(
            violations(tool, "fn emit() { quote! { #[pyclass] struct Generated; }; }")
                .unwrap()
                .is_empty()
        );
        assert_eq!(violations(tool, "#[pyclass] struct Handwritten;").unwrap().len(), 1);
    }

    #[test]
    fn placement_rejects_every_handwritten_export_family_with_item_and_location() {
        for (source, kind, item) in [
            ("#[host(export)] fn call() {}", "host", "fn call"),
            ("#[lk_agents_abi::export] fn call() {}", "export", "fn call"),
            ("#[derive(Clone, abi::Host)] struct Record;", "derive(Host)", "struct Record"),
            ("#[python(pydantic)] struct Record;", "python", "struct Record"),
            ("#[lk_agents_abi::class(name = \"Record\")] struct Record;", "class", "struct Record"),
            ("#[binding(emit)] struct Record;", "binding", "struct Record"),
            ("#[cfg_attr(any(), lk_agents_abi::class)] struct Record;", "class", "struct Record"),
            (
                "#[cfg_attr(any(), abi::binding(extract))] struct Record;",
                "binding",
                "struct Record",
            ),
            ("#[pyclass] struct Record;", "pyclass", "struct Record"),
            ("#[pymethods] impl Record {}", "pymethods", "impl  for Record"),
            ("#[pyfunction] fn call() {}", "pyfunction", "fn call"),
            ("#[pymodule] fn module() {}", "pymodule", "fn module"),
            ("#[cfg_attr(any(), abi::host(export))] fn call() {}", "host", "fn call"),
            (
                "mod nested { #[cfg_attr(any(), cfg_attr(all(), pyo3::pyclass))] struct Record; }",
                "pyclass",
                "mod nested::struct Record",
            ),
        ] {
            for path in
                ["rust/crates/lk-agents-voice/src/lib.rs", "rust/crates/lk-agents-py/src/lib.rs"]
            {
                let found = violations(path, source).unwrap();
                assert_eq!(found.len(), 1, "{source}");
                assert!(found[0].0.starts_with(&format!("{kind}|{path}|{item}|")), "{source}");
                assert_eq!(found[0].1, 1);
                let error = validate(found, &BTreeSet::new()).unwrap_err();
                assert!(error.contains(path) && error.contains(item) && error.contains("line 1"));
            }
        }
    }

    #[test]
    fn placement_allows_only_abi_declarations_generated_files_and_module_registration() {
        assert!(
            violations(
                "rust/crates/lk-agents-abi/src/lib.rs",
                "#[derive(Host)] #[python(pydantic)] #[binding(emit)] #[crate::class] struct Record;"
            )
            .unwrap()
            .is_empty()
        );
        assert!(
            violations("rust/crates/lk-agents-py/src/lib.rs", "#[pymodule] fn _core() {}")
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            violations(
                "rust/crates/lk-agents-py/src/lib.rs",
                "#[host(export)] #[pymodule] fn _core() {}"
            )
            .unwrap()
            .len(),
            1
        );
        assert_eq!(
            violations(
                "rust/crates/lk-agents-py/src/lib.rs",
                "mod nested { #[pymodule] fn _core() {} }"
            )
            .unwrap()
            .len(),
            1
        );
        assert!(
            violations(
                "rust/crates/lk-agents-py/src/generated.rs",
                "//! GENERATED FILE -- DO NOT EDIT.\n#[pyclass] struct Record;"
            )
            .unwrap()
            .is_empty()
        );
        assert_eq!(
            violations("rust/crates/lk-agents-abi/src/lib.rs", "#[pyclass] struct Record;")
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn placement_baselines_do_not_trade_one_identity_for_another() {
        for (old, new) in [
            ("#[pyclass] enum Role {}", "#[pyclass] enum Another {}"),
            ("fn call(#[host(lenient)] old: u8) {}", "fn call(#[host(lenient)] new: u8) {}"),
            ("#[host(object)] impl Record {}", "#[host(export)] impl Record {}"),
        ] {
            let path = "rust/crates/lk-agents-py/src/runtime.rs";
            let existing = violations(path, old).unwrap();
            let baseline = existing.iter().map(|(identity, _)| identity.clone()).collect();
            assert!(validate(existing, &baseline).is_ok());
            assert!(validate(violations(path, new).unwrap(), &baseline).is_err());
            assert!(validate(Vec::new(), &baseline).is_err(), "a stale row passed");
        }
        assert!(super::baseline("[\"same\", \"same\"]").is_err());
        assert!(super::baseline("[1]").is_err());
    }
}
