//! Parse architecture lint exemptions; used by the Python gate, never product code.
use proc_macro2::Span;
use quote::ToTokens;
use std::collections::BTreeMap;
use syn::{spanned::Spanned, visit::Visit};

struct LintScope {
    span: Option<Span>,
    attrs: Vec<syn::Attribute>,
    module: bool,
    exemptions_apply: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum LintLevel {
    Permitted,
    Denied,
    Forbidden,
}

/// Proves source exemptions separately from the forced-warning compiler capture.
struct Coverage {
    scopes: Vec<LintScope>,
    enclosing: Vec<usize>,
    features: Vec<String>,
}

impl Coverage {
    fn new(attrs: &[syn::Attribute], features: &[String]) -> Self {
        Self {
            scopes: vec![LintScope {
                span: None,
                attrs: attrs.to_vec(),
                module: true,
                exemptions_apply: true,
            }],
            enclosing: vec![0],
            features: features.to_vec(),
        }
    }

    fn enter(&mut self, span: Span, module: bool, exemptions_apply: bool) {
        self.enclosing.push(self.scopes.len());
        self.scopes.push(LintScope {
            span: Some(span),
            attrs: Vec::new(),
            module,
            exemptions_apply,
        });
    }

    fn exit(&mut self) {
        self.enclosing.pop();
    }

    fn condition(&self, meta: &syn::Meta) -> Option<bool> {
        match meta {
            syn::Meta::NameValue(value) if value.path.is_ident("feature") => {
                let syn::Expr::Lit(value) = &value.value else { return None };
                let syn::Lit::Str(feature) = &value.lit else { return None };
                Some(self.features.contains(&feature.value()))
            }
            syn::Meta::List(list) => {
                use syn::parse::Parser;
                let items =
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated
                        .parse2(list.tokens.clone())
                        .ok()?;
                let values: Vec<_> = items.iter().map(|item| self.condition(item)).collect();
                if list.path.is_ident("all") {
                    if values.contains(&Some(false)) {
                        Some(false)
                    } else if values.contains(&None) {
                        None
                    } else {
                        Some(true)
                    }
                } else if list.path.is_ident("any") {
                    if values.contains(&Some(true)) {
                        Some(true)
                    } else if values.contains(&None) {
                        None
                    } else {
                        Some(false)
                    }
                } else if list.path.is_ident("not") && values.len() == 1 {
                    values[0].map(|value| !value)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn level(&self, meta: &syn::Meta, lint: &str, level: &mut Option<LintLevel>) {
        use syn::parse::Parser;
        let syn::Meta::List(list) = meta else { return };
        let kind = list.path.get_ident().map(ToString::to_string).unwrap_or_default();
        if !matches!(kind.as_str(), "cfg_attr" | "allow" | "expect" | "warn" | "deny" | "forbid") {
            return;
        }
        let Ok(items) = syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated
            .parse2(list.tokens.clone())
        else {
            *level = Some(LintLevel::Forbidden);
            return;
        };
        if list.path.is_ident("cfg_attr") {
            let Some(condition) = items.first() else { return };
            match self.condition(condition) {
                Some(true) => {
                    for item in items.iter().skip(1) {
                        self.level(item, lint, level);
                    }
                }
                Some(false) => {}
                None => {
                    let mut conditional = None;
                    for item in items.iter().skip(1) {
                        self.level(item, lint, &mut conditional);
                    }
                    if conditional.is_some() {
                        *level = Some(LintLevel::Forbidden);
                    }
                }
            }
            return;
        }
        if matches!(kind.as_str(), "deny" | "forbid")
            && items.iter().any(|item| item.path().is_ident("warnings"))
        {
            *level = Some(LintLevel::Forbidden);
            return;
        }
        if items.iter().any(|item| {
            let name = item.path().to_token_stream().to_string().replace(' ', "");
            name == format!("clippy::{lint}") || name == "clippy::restriction"
        }) {
            if *level != Some(LintLevel::Forbidden) {
                *level = Some(match kind.as_str() {
                    "allow" | "expect" | "warn" => LintLevel::Permitted,
                    "forbid" => LintLevel::Forbidden,
                    _ => LintLevel::Denied,
                });
            }
        }
    }

    fn allowed(&self, line: usize, column: usize, lint: &str, expanded: bool) -> bool {
        let position = (line, column.saturating_sub(1));
        let mut level = None;
        for (index, scope) in self.scopes.iter().enumerate() {
            if scope.attrs.is_empty() {
                continue;
            }
            if lint == "disallowed_macros" && index != 0 {
                continue;
            }
            if scope.span.is_some_and(|span| {
                position < (span.start().line, span.start().column)
                    || position > (span.end().line, span.end().column)
            }) {
                continue;
            }
            let authorizes = scope.exemptions_apply && (!expanded || scope.module);
            for attr in &scope.attrs {
                if authorizes {
                    self.level(&attr.meta, lint, &mut level);
                } else {
                    let mut restriction = None;
                    self.level(&attr.meta, lint, &mut restriction);
                    if matches!(restriction, Some(LintLevel::Denied | LintLevel::Forbidden)) {
                        level = Some(LintLevel::Forbidden);
                    }
                }
            }
        }
        level == Some(LintLevel::Permitted)
    }
}

struct SourceIndex {
    root: std::path::PathBuf,
    parsed: BTreeMap<std::path::PathBuf, std::rc::Rc<syn::File>>,
    ancestors: BTreeMap<std::path::PathBuf, Option<Vec<Vec<syn::Attribute>>>>,
}

impl SourceIndex {
    fn new(root: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut index = Self {
            root: root.canonicalize()?,
            parsed: BTreeMap::new(),
            ancestors: BTreeMap::new(),
        };
        index.file(root, Some(Vec::new()), &mut Default::default())?;
        Ok(index)
    }

    fn file(
        &mut self,
        path: &std::path::Path,
        inherited: Option<Vec<Vec<syn::Attribute>>>,
        visiting: &mut std::collections::BTreeSet<std::path::PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.canonicalize()?;
        if !visiting.insert(path.clone()) {
            return Err(format!("{}: cyclic module ancestry", path.display()).into());
        }
        let parsed = if let Some(parsed) = self.parsed.get(&path) {
            parsed.clone()
        } else {
            let parsed = std::rc::Rc::new(syn::parse_file(&std::fs::read_to_string(&path)?)?);
            self.parsed.insert(path.clone(), parsed.clone());
            parsed
        };
        let attrs = inherited.map(|mut attrs| {
            attrs.push(parsed.attrs.clone());
            attrs
        });
        let attrs = if let Some(previous) = self.ancestors.get(&path) {
            if previous.is_none() {
                visiting.remove(&path);
                return Ok(());
            }
            None
        } else {
            attrs
        };
        self.ancestors.insert(path.clone(), attrs.clone());
        let parent = path.parent().ok_or("source has no parent directory")?;
        let directory = if path == self.root
            || matches!(
                path.file_name().and_then(|s| s.to_str()),
                Some("lib.rs" | "main.rs" | "mod.rs")
            ) {
            parent.to_path_buf()
        } else {
            parent.join(path.file_stem().ok_or("source has no file stem")?)
        };
        self.modules(&parsed.items, &directory, parent, attrs, visiting)?;
        visiting.remove(&path);
        Ok(())
    }

    fn modules(
        &mut self,
        items: &[syn::Item],
        directory: &std::path::Path,
        path_directory: &std::path::Path,
        inherited: Option<Vec<Vec<syn::Attribute>>>,
        visiting: &mut std::collections::BTreeSet<std::path::PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for item in items {
            let syn::Item::Mod(module) = item else { continue };
            let name = module.ident.to_string();
            let attrs = inherited.clone().map(|mut attrs| {
                attrs.push(module.attrs.clone());
                attrs
            });
            if let Some((_, children)) = &module.content {
                let child = directory.join(name);
                self.modules(children, &child, &child, attrs, visiting)?;
            } else {
                let custom = module.attrs.iter().find_map(|attr| {
                    if !attr.path().is_ident("path") {
                        return None;
                    }
                    let syn::Meta::NameValue(value) = &attr.meta else { return None };
                    let syn::Expr::Lit(value) = &value.value else { return None };
                    let syn::Lit::Str(path) = &value.lit else { return None };
                    Some(path_directory.join(path.value()))
                });
                let candidates = custom.map_or_else(
                    || {
                        vec![
                            directory.join(format!("{name}.rs")),
                            directory.join(&name).join("mod.rs"),
                        ]
                    },
                    |path| vec![path],
                );
                for candidate in candidates {
                    if candidate.is_file() {
                        self.file(&candidate, attrs.clone(), visiting)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn attributes(
        &self,
        target: &std::path::Path,
    ) -> Result<&[Vec<syn::Attribute>], Box<dyn std::error::Error>> {
        match self.ancestors.get(target) {
            Some(Some(attrs)) => Ok(attrs),
            Some(None) => Err(format!("{}: ambiguous module ancestry", target.display()).into()),
            None => Err(format!(
                "{}: cannot prove its compiler target module ancestry",
                target.display()
            )
            .into()),
        }
    }

    fn coverage(
        &self,
        path: &str,
        positions: &[String],
        features: &[String],
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let target = std::path::Path::new(path).canonicalize()?;
        let attrs = self.attributes(&target)?;
        let mut coverage = Coverage::new(&attrs[0], features);
        coverage.scopes.extend(attrs.iter().skip(1).map(|attrs| LintScope {
            span: None,
            attrs: attrs.clone(),
            module: true,
            exemptions_apply: true,
        }));
        let mut scan = Scan { coverage: Some(coverage), ..Scan::default() };
        scan.visit_file(&self.parsed[&target]);
        let coverage = scan.coverage.unwrap();
        positions
            .iter()
            .map(|value| {
                let parts: Vec<_> = value.split(':').collect();
                if !(parts.len() == 3 || parts.len() == 4) {
                    return Err("expected line:column:lint[:expanded]".into());
                }
                Ok(coverage.allowed(
                    parts[0].parse()?,
                    parts[1].parse()?,
                    parts[2],
                    parts.get(3) == Some(&"1"),
                ))
            })
            .collect()
    }
}

fn coverage_root(path: &str) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    if let Some(root) = std::env::var_os("ARCHITECTURE_CRATE_ROOT") {
        return Ok(root.into());
    }
    let target = std::path::Path::new(path).canonicalize()?;
    let package = target.ancestors().skip(1).find(|directory| {
        directory.join("Cargo.toml").is_file() && directory.join("src/lib.rs").is_file()
    });
    if let Some(package) = package {
        if target.starts_with(package.join("src")) {
            return Ok(if target == package.join("src/main.rs") {
                target
            } else {
                package.join("src/lib.rs")
            });
        }
        for directory in ["tests", "examples"] {
            if let Ok(relative) = target.strip_prefix(package.join(directory)) {
                if relative.components().count() > 1 {
                    return Err(
                        format!("{path}: nested source requires its compiler target root").into()
                    );
                }
            }
        }
    }
    Ok(target)
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    fn allowed(source: &str, line: usize, lint: &str) -> bool {
        let file = syn::parse_file(source).unwrap();
        let mut scan = Scan { coverage: Some(Coverage::new(&file.attrs, &[])), ..Scan::default() };
        scan.visit_file(&file);
        scan.coverage.unwrap().allowed(line, 1, lint, false)
    }

    #[test]
    fn deleting_the_exemption_rejects_an_unchanged_legacy_site() {
        let site = "fn legacy() { std::thread::spawn(|| {}); }";
        assert!(!allowed(site, 1, "disallowed_methods"));
        assert!(allowed(
            &format!("#[allow(clippy::disallowed_methods)]\n{site}"),
            2,
            "disallowed_methods"
        ));
    }

    #[test]
    fn a_child_deny_overrides_an_inherited_exemption() {
        assert!(!allowed(
            "#![allow(clippy::disallowed_methods)]\n#[deny(clippy::disallowed_methods)]\nfn f() {}",
            3,
            "disallowed_methods"
        ));
    }

    #[test]
    fn signature_types_require_an_enclosing_function_exemption() {
        for asynchronous in ["", "async "] {
            let source = format!(
                "{asynchronous}fn f(#[allow(clippy::disallowed_types)] value: std::sync::Mutex<()>) {{}}"
            );
            let parsed = syn::parse_file(&source).unwrap();
            let mut scan =
                Scan { coverage: Some(Coverage::new(&parsed.attrs, &[])), ..Scan::default() };
            scan.visit_file(&parsed);
            let column = source.find("std::").unwrap() + 1;
            assert!(!scan.coverage.unwrap().allowed(1, column, "disallowed_types", false));
            assert!(allowed(
                &format!(
                    "#[allow(clippy::disallowed_types)]\n{asynchronous}fn f(value: std::sync::Mutex<()>) {{}}"
                ),
                2,
                "disallowed_types"
            ));
        }
    }

    #[test]
    fn moving_an_exemption_to_a_generic_or_macro_cannot_authorize_the_function_body() {
        for source in [
            "fn f<#[allow(clippy::disallowed_methods)] T>() { std::thread::spawn(|| {}); }",
            "fn f() { #[allow(clippy::disallowed_methods)] harmless!(); std::thread::spawn(|| {}); }",
        ] {
            let parsed = syn::parse_file(source).unwrap();
            let mut scan =
                Scan { coverage: Some(Coverage::new(&parsed.attrs, &[])), ..Scan::default() };
            scan.visit_file(&parsed);
            let column = source.find("std::").unwrap() + 1;
            assert!(!scan.coverage.unwrap().allowed(1, column, "disallowed_methods", false));
        }
    }

    #[test]
    fn narrow_scopes_still_restrict_inherited_exemptions() {
        for (source, expanded) in [
            (
                "#[allow(clippy::disallowed_types)] fn f<#[deny(clippy::disallowed_types)] T: Into<std::sync::Mutex<()>>>() {}",
                false,
            ),
            (
                "#[allow(clippy::disallowed_types)] fn f(value: fn(#[deny(clippy::disallowed_types)] std::sync::Mutex<()>)) {}",
                false,
            ),
            (
                "#![allow(clippy::disallowed_types)]\n#[deny(clippy::disallowed_types)]\nfn f(value: std::sync::Mutex<()>) {}",
                true,
            ),
        ] {
            let parsed = syn::parse_file(source).unwrap();
            let mut scan =
                Scan { coverage: Some(Coverage::new(&parsed.attrs, &[])), ..Scan::default() };
            scan.visit_file(&parsed);
            let before = &source[..source.find("std::").unwrap()];
            let line = before.bytes().filter(|b| *b == b'\n').count() + 1;
            let column = before.rsplit('\n').next().unwrap().len() + 1;
            assert!(!scan.coverage.unwrap().allowed(line, column, "disallowed_types", expanded));
        }
    }

    #[test]
    fn unrelated_attribute_arguments_cannot_revoke_an_exemption() {
        assert!(allowed(
            "#[allow(clippy::disallowed_methods)]\n#[pyo3(signature = (value, *, optional = None))]\nfn f() {}",
            3,
            "disallowed_methods"
        ));
    }

    #[test]
    fn inactive_or_unknown_conditions_cannot_authorize_a_site() {
        for condition in ["feature = \"python\"", "unknown_predicate"] {
            assert!(!allowed(
                &format!(
                    "#[cfg_attr({condition}, allow(clippy::disallowed_methods))]\nfn f() {{}}"
                ),
                2,
                "disallowed_methods"
            ));
        }
        let file = syn::parse_file(
            "#[cfg_attr(feature = \"python\", allow(clippy::disallowed_methods))]\nfn f() {}",
        )
        .unwrap();
        let mut scan = Scan {
            coverage: Some(Coverage::new(&file.attrs, &["python".into()])),
            ..Scan::default()
        };
        scan.visit_file(&file);
        assert!(scan.coverage.unwrap().allowed(2, 1, "disallowed_methods", false));
    }

    #[test]
    fn narrow_macro_exemptions_do_not_replace_the_crate_exemption() {
        assert!(!allowed("#[allow(clippy::disallowed_macros)]\nfn f() {}", 2, "disallowed_macros"));
        assert!(allowed("#![allow(clippy::disallowed_macros)]\nfn f() {}", 2, "disallowed_macros"));
    }

    #[test]
    fn exemptions_cannot_cross_sibling_scopes_or_override_forbid() {
        assert!(!allowed(
            "#[allow(clippy::disallowed_methods)] fn old() {}\nfn new() {}",
            2,
            "disallowed_methods"
        ));
        assert!(!allowed(
            "#![forbid(clippy::disallowed_methods)]\n#[allow(clippy::disallowed_methods)]\nfn f() {}",
            3,
            "disallowed_methods"
        ));
        assert!(!allowed("#![allow(clippy::all)]\nfn f() {}", 2, "disallowed_methods"));
        assert!(!allowed("#![allow(warnings)]\nfn f() {}", 2, "disallowed_methods"));
        assert!(!allowed(
            "#[rustfmt::allow(clippy::disallowed_methods)]\nfn f() {}",
            2,
            "disallowed_methods"
        ));
        assert!(!allowed(
            "#![deny(warnings)]\n#[warn(clippy::disallowed_methods)]\nfn f() {}",
            3,
            "disallowed_methods"
        ));
    }

    #[test]
    fn generated_sibling_items_require_a_module_exemption() {
        let parsed = syn::parse_file(
            "#[allow(clippy::disallowed_types, clippy::disallowed_methods)]\nfn f() {}",
        )
        .unwrap();
        let mut scan =
            Scan { coverage: Some(Coverage::new(&parsed.attrs, &[])), ..Scan::default() };
        scan.visit_file(&parsed);
        let coverage = scan.coverage.unwrap();
        for lint in ["disallowed_types", "disallowed_methods"] {
            assert!(coverage.allowed(2, 1, lint, false));
            assert!(!coverage.allowed(2, 1, lint, true));
        }
    }

    #[test]
    fn module_ancestry_propagates_exemptions_and_refuses_ambiguous_aliases() {
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let directory =
            std::env::temp_dir().join(format!("architecture-module-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let _cleanup = Cleanup(directory.clone());
        let root = directory.join("integration.rs");
        let target = directory.join("shared.rs");
        std::fs::write(
            &root,
            r#"
#![allow(clippy::disallowed_types)]
#[allow(clippy::disallowed_methods)]
mod shared;
"#,
        )
        .unwrap();
        std::fs::write(&target, "#![deny(clippy::disallowed_types)]\nfn f() {}").unwrap();
        let canonical = target.canonicalize().unwrap();
        let attrs = SourceIndex::new(&root).unwrap().attributes(&canonical).unwrap().to_vec();
        let mut coverage = Coverage::new(&attrs[0], &[]);
        coverage.scopes.extend(attrs.iter().skip(1).map(|attrs| LintScope {
            span: None,
            attrs: attrs.clone(),
            module: true,
            exemptions_apply: true,
        }));
        let mut scan = Scan { coverage: Some(coverage), ..Scan::default() };
        scan.visit_file(&syn::parse_file(&std::fs::read_to_string(&target).unwrap()).unwrap());
        let coverage = scan.coverage.unwrap();
        assert!(coverage.allowed(2, 1, "disallowed_methods", false));
        assert!(!coverage.allowed(2, 1, "disallowed_types", false));
        std::fs::write(
            &root,
            r#"
#[cfg(any())]
#[allow(clippy::disallowed_methods)]
#[path = "shared.rs"]
mod inactive;
#[path = "shared.rs"]
mod active;
"#,
        )
        .unwrap();
        std::fs::write(&target, "fn f() { std::thread::spawn(|| {}); }").unwrap();
        let index = SourceIndex::new(&root).unwrap();
        assert!(
            index
                .attributes(&target.canonicalize().unwrap())
                .unwrap_err()
                .to_string()
                .contains("ambiguous module ancestry")
        );
    }
}

#[derive(Default)]
struct Scan {
    path: String,
    scopes: Vec<String>,
    locations: Vec<(Span, String)>,
    exemptions: Vec<String>,
    occurrences: BTreeMap<String, usize>,
    tuple_fields: BTreeMap<String, usize>,
    errors: Vec<String>,
    coverage: Option<Coverage>,
}

impl Scan {
    fn coverage_scope(
        &mut self,
        span: Span,
        exemptions_apply: bool,
        visit: impl FnOnce(&mut Self),
    ) {
        if let Some(coverage) = &mut self.coverage {
            coverage.enter(span, false, exemptions_apply);
        }
        visit(self);
        if let Some(coverage) = &mut self.coverage {
            coverage.exit();
        }
    }

    fn enter(&mut self, name: String, span: Span, f: impl FnOnce(&mut Self)) {
        if let Some(coverage) = &mut self.coverage {
            coverage.enter(span, name.starts_with("mod "), !name.starts_with("argument "));
        }
        self.scopes.push(name);
        self.locations.push((span, self.scopes.join("/")));
        f(self);
        self.scopes.pop();
        if let Some(coverage) = &mut self.coverage {
            coverage.exit();
        }
    }

    fn macro_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        use syn::parse::Parser;
        let tokens: Vec<_> = tokens.into_iter().collect();
        for (index, token) in tokens.iter().enumerate() {
            if matches!(token, proc_macro2::TokenTree::Punct(p) if p.as_char() == '#') {
                if let Some(proc_macro2::TokenTree::Group(group)) = tokens.get(index + 1) {
                    if group.delimiter() == proc_macro2::Delimiter::Bracket {
                        if let Ok(attrs) =
                            syn::Attribute::parse_outer.parse2(quote::quote!(#token #group))
                        {
                            for attr in attrs {
                                self.meta(&attr.meta, "always");
                            }
                        }
                    }
                }
            }
            if let proc_macro2::TokenTree::Group(group) = token {
                self.macro_tokens(group.stream());
            }
        }
    }

    fn meta(&mut self, meta: &syn::Meta, context: &str) {
        let syn::Meta::List(list) = meta else { return };
        let kind = list.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
        if kind == "cfg_attr" {
            use syn::parse::Parser;
            if let Ok(items) =
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated
                    .parse2(list.tokens.clone())
            {
                let condition = items
                    .first()
                    .map(|item| item.to_token_stream().to_string())
                    .unwrap_or_default();
                let nested = format!("{context}/cfg({condition})");
                for item in items.iter().skip(1) {
                    self.meta(item, &nested);
                }
            }
        } else if matches!(kind.as_str(), "allow" | "expect" | "warn") {
            use syn::parse::Parser;
            if let Ok(items) =
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated
                    .parse2(list.tokens.clone())
            {
                let reason = items.iter().any(|item| {
                    matches!(item, syn::Meta::NameValue(value)
                        if value.path.is_ident("reason") && matches!(&value.value,
                            syn::Expr::Lit(value) if matches!(&value.lit,
                                syn::Lit::Str(text) if !text.value().trim().is_empty())))
                });
                for item in items {
                    let lint = item.path().to_token_stream().to_string().replace(' ', "");
                    if matches!(
                        lint.as_str(),
                        "warnings"
                            | "clippy::all"
                            | "clippy::restriction"
                            | "clippy::disallowed_methods"
                            | "clippy::disallowed_types"
                            | "clippy::disallowed_macros"
                    ) {
                        if !reason {
                            self.errors.push(format!(
                                "{}:{}: {kind}({lint}) requires a nonempty reason",
                                self.path,
                                self.scopes.join("/")
                            ));
                        }
                        let key = format!(
                            "{}:{}:{lint}:{kind}:{context}",
                            self.path,
                            self.scopes.join("/")
                        );
                        let occurrence = self.occurrences.entry(key.clone()).or_default();
                        *occurrence += 1;
                        self.exemptions.push(format!("{key}:{}", *occurrence));
                    }
                }
            }
        }
    }
}

macro_rules! coverage_visit {
    ($method:ident, $ty:ty, $exemptions_apply:expr) => {
        fn $method(&mut self, node: &'ast $ty) {
            self.coverage_scope(node.span(), $exemptions_apply, |s| syn::visit::$method(s, node));
        }
    };
}

impl<'ast> Visit<'ast> for Scan {
    fn visit_file(&mut self, node: &'ast syn::File) {
        if self.coverage.is_none() {
            for attr in &node.attrs {
                self.visit_attribute(attr);
            }
        }
        for item in &node.items {
            self.visit_item(item);
        }
    }

    coverage_visit!(visit_expr, syn::Expr, true);
    coverage_visit!(visit_local, syn::Local, true);
    coverage_visit!(visit_arm, syn::Arm, true);
    coverage_visit!(visit_generic_param, syn::GenericParam, false);
    coverage_visit!(visit_pat, syn::Pat, false);
    coverage_visit!(visit_field_pat, syn::FieldPat, false);
    coverage_visit!(visit_field_value, syn::FieldValue, true);
    coverage_visit!(visit_stmt_macro, syn::StmtMacro, true);
    coverage_visit!(visit_foreign_item, syn::ForeignItem, true);
    coverage_visit!(visit_bare_fn_arg, syn::BareFnArg, false);
    coverage_visit!(visit_bare_variadic, syn::BareVariadic, false);
    coverage_visit!(visit_variadic, syn::Variadic, false);

    fn visit_item(&mut self, node: &'ast syn::Item) {
        let name = match node {
            syn::Item::Fn(i) => format!("fn {}", i.sig.ident),
            syn::Item::Struct(i) => format!("struct {}", i.ident),
            syn::Item::Enum(i) => format!("enum {}", i.ident),
            syn::Item::Mod(i) => format!("mod {}", i.ident),
            syn::Item::Type(i) => format!("type {}", i.ident),
            syn::Item::Const(i) => format!("const {}", i.ident),
            syn::Item::Static(i) => format!("static {}", i.ident),
            syn::Item::Trait(i) => format!("trait {}", i.ident),
            syn::Item::Impl(i) => format!(
                "impl {} {}",
                i.trait_
                    .as_ref()
                    .map(|(_, p, _)| p.to_token_stream().to_string())
                    .unwrap_or_default(),
                i.self_ty.to_token_stream()
            ),
            syn::Item::Macro(i) => format!(
                "macro {} {}",
                i.mac.path.to_token_stream(),
                i.ident.as_ref().map(ToString::to_string).unwrap_or_default()
            ),
            syn::Item::Use(i) => format!("use {}", i.tree.to_token_stream()),
            _ => "item".into(),
        };
        self.enter(name, node.span(), |s| syn::visit::visit_item(s, node));
    }
    fn visit_impl_item(&mut self, node: &'ast syn::ImplItem) {
        let name = match node {
            syn::ImplItem::Fn(i) => format!("fn {}", i.sig.ident),
            syn::ImplItem::Const(i) => format!("const {}", i.ident),
            syn::ImplItem::Type(i) => format!("type {}", i.ident),
            _ => "item".into(),
        };
        self.enter(name, node.span(), |s| syn::visit::visit_impl_item(s, node));
    }
    fn visit_trait_item(&mut self, node: &'ast syn::TraitItem) {
        let name = match node {
            syn::TraitItem::Fn(i) => format!("fn {}", i.sig.ident),
            syn::TraitItem::Const(i) => format!("const {}", i.ident),
            syn::TraitItem::Type(i) => format!("type {}", i.ident),
            _ => "item".into(),
        };
        self.enter(name, node.span(), |s| syn::visit::visit_trait_item(s, node));
    }
    fn visit_variant(&mut self, node: &'ast syn::Variant) {
        self.enter(format!("variant {}", node.ident), node.span(), |s| {
            syn::visit::visit_variant(s, node)
        });
    }
    fn visit_fn_arg(&mut self, node: &'ast syn::FnArg) {
        let name = match node {
            syn::FnArg::Receiver(_) => "self".into(),
            syn::FnArg::Typed(i) => i.pat.to_token_stream().to_string(),
        };
        self.enter(format!("argument {name}"), node.span(), |s| syn::visit::visit_fn_arg(s, node));
    }
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        self.macro_tokens(node.tokens.clone());
        syn::visit::visit_macro(self, node);
    }
    fn visit_field(&mut self, node: &'ast syn::Field) {
        let name = if let Some(ident) = &node.ident {
            format!("field {ident}")
        } else {
            let ordinal = self.tuple_fields.entry(self.scopes.join("/")).or_default();
            let name = format!("field {} {}", *ordinal, node.ty.to_token_stream());
            *ordinal += 1;
            name
        };
        self.enter(name, node.span(), |s| syn::visit::visit_field(s, node));
    }
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        if let Some(coverage) = &mut self.coverage {
            let current = *coverage.enclosing.last().expect("root scope");
            coverage.scopes[current].attrs.push(attr.clone());
        }
        self.meta(&attr.meta, "always");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mode = args.first().ok_or("expected inventory or locate")?;
    if matches!(mode.as_str(), "coverage" | "coverage-batch") {
        use std::io::BufRead;
        let requests: Vec<(String, Vec<String>)> = if mode == "coverage-batch" {
            std::io::stdin()
                .lock()
                .lines()
                .map(|line| {
                    let line = line?;
                    let mut columns = line.split('\t');
                    let path = columns.next().ok_or("missing coverage source path")?.to_owned();
                    Ok((path, columns.map(str::to_owned).collect()))
                })
                .collect::<Result<_, Box<dyn std::error::Error>>>()?
        } else {
            vec![(
                args.get(1).ok_or("missing coverage source path")?.clone(),
                args.iter().skip(2).cloned().collect(),
            )]
        };
        let first = requests.first().ok_or("empty coverage request")?;
        let root = coverage_root(&first.0)?;
        let features: Vec<_> = std::env::var("ARCHITECTURE_FEATURES")?
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        let index = SourceIndex::new(&root)?;
        for (path, positions) in requests {
            if positions.is_empty() {
                return Err("empty coverage positions".into());
            }
            for permitted in index.coverage(&path, &positions, &features)? {
                println!("{}", if permitted { "allowed" } else { "denied" });
            }
        }
        return Ok(());
    }
    for path in args.iter().skip(1).take(if matches!(mode.as_str(), "locate" | "locations") {
        1
    } else {
        usize::MAX
    }) {
        let source = std::fs::read_to_string(path)?;
        let parsed = syn::parse_file(&source).map_err(|e| format!("{path}: {e}"))?;
        let mut scan = Scan { path: path.clone(), ..Scan::default() };
        scan.visit_file(&parsed);
        if mode == "locate" || mode == "locations" {
            let positions: Vec<(usize, usize)> = if mode == "locations" {
                args.iter()
                    .skip(2)
                    .map(|value| {
                        let (line, column) = value.split_once(':').ok_or("expected line:column")?;
                        Ok((line.parse()?, column.parse()?))
                    })
                    .collect::<Result<_, Box<dyn std::error::Error>>>()?
            } else {
                vec![(
                    args.get(2).ok_or("missing line")?.parse()?,
                    args.get(3).ok_or("missing column")?.parse()?,
                )]
            };
            for (line, column) in positions {
                let position = (line, column.saturating_sub(1));
                let candidate = scan
                    .locations
                    .iter()
                    .filter(|(span, _)| {
                        (span.start().line, span.start().column) <= position
                            && position <= (span.end().line, span.end().column)
                    })
                    .min_by_key(|(span, _)| {
                        (
                            span.end().line - span.start().line,
                            span.end().column.saturating_sub(span.start().column),
                        )
                    });
                let (span, name) = candidate
                    .ok_or_else(|| format!("{path}:{line}:{column}: no enclosing item"))?;
                println!("{}\t{name}", span.start().line);
            }
        } else if mode == "inventory" {
            if !scan.errors.is_empty() {
                return Err(scan.errors.join("\n").into());
            }
            for exemption in scan.exemptions {
                println!("{exemption}");
            }
        } else {
            return Err("expected inventory or locate".into());
        }
    }
    Ok(())
}
