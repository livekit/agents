//! Export placement is checked on parsed handwritten source, including inactive cfg branches.
//! Identities use lexical item paths rather than line numbers so unrelated edits need no waiver.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    process::Command,
};

use quote::ToTokens as _;
use syn::{parse::Parser as _, punctuated::Punctuated, spanned::Spanned as _, visit::Visit as _};

const BASELINE: &str = "rust/crates/lk-agents-codegen/export-placement-baseline.json";

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

fn validate(locations: Vec<(String, usize)>, baseline: &BTreeSet<String>) -> Result<(), String> {
    let errors: Vec<_> = locations
        .into_iter()
        .filter(|(identity, _)| !baseline.contains(identity))
        .map(|(identity, line)| format!("{identity} (line {line})"))
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
    let parsed = syn::parse_file(source).map_err(|e| format!("{path}: {e}"))?;
    let mut visitor = Exports { path, scope: Vec::new(), found: Vec::new(), error: None };
    visitor.visit_file(&parsed);
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

struct Exports<'a> {
    path: &'a str,
    scope: Vec<String>,
    found: Vec<(String, usize)>,
    error: Option<syn::Error>,
}

impl Exports<'_> {
    fn generator(&self) -> bool {
        ["lk-agents-codegen", "lk-agents-bindgen", "lk-agents-host-derive"]
            .iter()
            .any(|name| self.path.starts_with(&format!("rust/crates/{name}/")))
    }

    fn macro_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        use proc_macro2::{Delimiter, TokenTree};
        let tokens: Vec<_> = tokens.into_iter().collect();
        let mut index = 0;
        while index < tokens.len() {
            if matches!(&tokens[index], TokenTree::Punct(p) if p.as_char() == '#') {
                let mut next = index + 1;
                if matches!(tokens.get(next), Some(TokenTree::Punct(p)) if p.as_char() == '!') {
                    next += 1;
                }
                if let Some(TokenTree::Group(group)) = tokens.get(next)
                    && group.delimiter() == Delimiter::Bracket
                {
                    let contents = group.stream();
                    if matches!(contents.clone().into_iter().next(), Some(TokenTree::Punct(p)) if p.as_char() == '$')
                    {
                        index = next + 1; // Forwarded metadata is checked at its concrete call site.
                        continue;
                    }
                    let meta = syn::parse2::<syn::Meta>(contents.clone()).or_else(|_| {
                        // Metavariables may prevent parsing arguments; the attribute's path
                        // still identifies a handwritten export and must not be hidden.
                        let parser = |input: syn::parse::ParseStream<'_>| {
                            let path = input.call(syn::Path::parse_mod_style)?;
                            let _: proc_macro2::TokenStream = input.parse()?;
                            Ok(syn::Meta::Path(path))
                        };
                        parser.parse2(contents.clone())
                    });
                    match meta {
                        Ok(meta) => {
                            if let Err(error) = self.attribute(&meta) {
                                self.error = Some(error);
                            }
                        }
                        Err(error) => self.error = Some(error),
                    }
                    index = next + 1;
                    continue;
                }
            }
            if let TokenTree::Group(group) = &tokens[index] {
                let template = self.generator()
                    && index >= 2
                    && matches!(&tokens[index - 1], TokenTree::Punct(p) if p.as_char() == '!')
                    && matches!(&tokens[index - 2], TokenTree::Ident(i) if ["quote", "quote_spanned", "parse_quote", "parse_quote_spanned"].iter().any(|name| i == name));
                if !template {
                    self.macro_tokens(group.stream());
                }
            }
            index += 1;
        }
    }

    fn attribute(&mut self, meta: &syn::Meta) -> syn::Result<()> {
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
            "host" | "python" | "node" | "export" if !abi => Some(name),
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

impl<'ast> syn::visit::Visit<'ast> for Exports<'_> {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        let name = match item {
            syn::Item::Const(v) => format!("const {}", v.ident),
            syn::Item::Enum(v) => format!("enum {}", v.ident),
            syn::Item::Fn(v) => format!("fn {}", v.sig.ident),
            syn::Item::Impl(v) => format!(
                "impl {} for {}",
                v.trait_
                    .as_ref()
                    .map(|(_, p, _)| p.to_token_stream().to_string())
                    .unwrap_or_default(),
                v.self_ty.to_token_stream()
            ),
            syn::Item::Mod(v) => format!("mod {}", v.ident),
            syn::Item::Macro(v) => format!(
                "macro {}",
                v.ident.as_ref().map(ToString::to_string).unwrap_or_else(|| v
                    .mac
                    .path
                    .to_token_stream()
                    .to_string())
            ),
            syn::Item::Static(v) => format!("static {}", v.ident),
            syn::Item::Struct(v) => format!("struct {}", v.ident),
            syn::Item::Trait(v) => format!("trait {}", v.ident),
            syn::Item::TraitAlias(v) => format!("trait {}", v.ident),
            syn::Item::Type(v) => format!("type {}", v.ident),
            syn::Item::Union(v) => format!("union {}", v.ident),
            _ => "item".to_owned(),
        };
        self.scope.push(name);
        syn::visit::visit_item(self, item);
        self.scope.pop();
    }

    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        let name = match item {
            syn::ImplItem::Const(v) => v.ident.to_string(),
            syn::ImplItem::Fn(v) => v.sig.ident.to_string(),
            syn::ImplItem::Type(v) => v.ident.to_string(),
            _ => "item".to_owned(),
        };
        self.scope.push(name);
        syn::visit::visit_impl_item(self, item);
        self.scope.pop();
    }

    fn visit_trait_item(&mut self, item: &'ast syn::TraitItem) {
        let name = match item {
            syn::TraitItem::Const(v) => v.ident.to_string(),
            syn::TraitItem::Fn(v) => v.sig.ident.to_string(),
            syn::TraitItem::Type(v) => v.ident.to_string(),
            _ => "item".to_owned(),
        };
        self.scope.push(name);
        syn::visit::visit_trait_item(self, item);
        self.scope.pop();
    }

    fn visit_field(&mut self, field: &'ast syn::Field) {
        self.scope.push(
            field.ident.as_ref().map(ToString::to_string).unwrap_or_else(|| "field".to_owned()),
        );
        syn::visit::visit_field(self, field);
        self.scope.pop();
    }

    fn visit_fn_arg(&mut self, argument: &'ast syn::FnArg) {
        self.scope.push(match argument {
            syn::FnArg::Receiver(_) => "arg self".to_owned(),
            syn::FnArg::Typed(value) => format!("arg {}", value.pat.to_token_stream()),
        });
        syn::visit::visit_fn_arg(self, argument);
        self.scope.pop();
    }

    fn visit_variant(&mut self, variant: &'ast syn::Variant) {
        self.scope.push(variant.ident.to_string());
        syn::visit::visit_variant(self, variant);
        self.scope.pop();
    }

    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        if let Err(error) = self.attribute(&attr.meta) {
            self.error = Some(error);
        }
    }

    fn visit_macro(&mut self, value: &'ast syn::Macro) {
        let template = self.generator()
            && value.path.segments.last().is_some_and(|s| {
                ["quote", "quote_spanned", "parse_quote", "parse_quote_spanned"]
                    .iter()
                    .any(|name| s.ident == name)
            });
        if template {
            return; // These tools emit the generated adapters; their quoted attributes are output.
        }
        self.scope.push(format!("tokens {}", value.path.to_token_stream()));
        self.macro_tokens(value.tokens.clone());
        self.scope.pop();
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
        let tool = "rust/crates/lk-agents-codegen/src/python.rs";
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
                "#[derive(Host)] #[python(pydantic)] struct Record;"
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
            assert!(validate(Vec::new(), &baseline).is_ok());
        }
        assert!(super::baseline("[\"same\", \"same\"]").is_err());
        assert!(super::baseline("[1]").is_err());
    }
}
