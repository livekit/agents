//! Shared handwritten-source visitor for lint exemptions, compiler locations and export placement.
use proc_macro2::Span;
use quote::ToTokens;
use std::collections::BTreeMap;
use syn::{parse::Parser as _, spanned::Spanned, visit::Visit};

mod coverage;
pub(crate) mod host_members;
use crate::placement::Exports;
use coverage::{Coverage, SourceIndex, coverage_root};

#[derive(Default)]
pub(crate) struct Scan {
    path: String,
    scopes: Vec<String>,
    locations: Vec<(Span, String)>,
    exemptions: Vec<String>,
    occurrences: BTreeMap<String, usize>,
    tuple_fields: BTreeMap<String, usize>,
    errors: Vec<String>,
    coverage: Option<Coverage>,
    pub(crate) exports: Exports,
    check_exports: bool,
    check_exemptions: bool,
    locate: bool,
}

impl Scan {
    pub(crate) fn new(path: &str) -> Self {
        Self {
            path: path.to_owned(),
            check_exemptions: true,
            locate: true,
            exports: Exports { path: path.to_owned(), ..Exports::default() },
            ..Self::default()
        }
    }
    pub(crate) fn source(path: &str, source: &str) -> Result<Self, String> {
        let parsed = syn::parse_file(source).map_err(|e| format!("{path}: {e}"))?;
        let mut scan = Self::new(path);
        scan.check_exports = true;
        scan.check_exemptions = false;
        scan.locate = false;
        scan.visit_file(&parsed);
        Ok(scan)
    }

    fn coverage_scope(
        &mut self,
        span: Span,
        module: bool,
        exemptions_apply: bool,
        visit: impl FnOnce(&mut Self),
    ) {
        if let Some(coverage) = &mut self.coverage {
            coverage.enter(span, module, exemptions_apply);
        }
        visit(self);
        if let Some(coverage) = &mut self.coverage {
            coverage.exit();
        }
    }

    fn enter(&mut self, name: String, export_name: String, span: Span, f: impl FnOnce(&mut Self)) {
        if self.check_exemptions || self.locate {
            self.scopes.push(name);
        }
        if self.check_exports {
            self.exports.scope.push(export_name);
        }
        if self.locate {
            self.locations.push((span, self.scopes.join("/")));
        }
        f(self);
        if self.check_exemptions || self.locate {
            self.scopes.pop();
        }
        if self.check_exports {
            self.exports.scope.pop();
        }
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
                            self.meta(&meta, "always");
                            if self.check_exports
                                && let Err(error) = self.exports.attribute(&meta)
                            {
                                self.exports.error = Some(error);
                            }
                        }
                        Err(error) => self.exports.error = Some(error),
                    }
                    index = next + 1;
                    continue;
                }
            }
            if let TokenTree::Group(group) = &tokens[index] {
                let template = self.exports.generator()
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

    fn meta(&mut self, meta: &syn::Meta, context: &str) {
        if !self.check_exemptions {
            return;
        }
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
            self.coverage_scope(node.span(), false, $exemptions_apply, |s| {
                syn::visit::$method(s, node)
            });
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
        if self.coverage.is_some() {
            return self.coverage_scope(
                node.span(),
                matches!(node, syn::Item::Mod(_)),
                true,
                |s| syn::visit::visit_item(s, node),
            );
        }
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
        let export_name = match node {
            syn::Item::Impl(v) => format!(
                "impl {} for {}",
                v.trait_
                    .as_ref()
                    .map(|(_, p, _)| p.to_token_stream().to_string())
                    .unwrap_or_default(),
                v.self_ty.to_token_stream()
            ),
            syn::Item::Macro(v) => format!(
                "macro {}",
                v.ident.as_ref().map(ToString::to_string).unwrap_or_else(|| v
                    .mac
                    .path
                    .to_token_stream()
                    .to_string())
            ),
            syn::Item::TraitAlias(v) => format!("trait {}", v.ident),
            syn::Item::Union(v) => format!("union {}", v.ident),
            syn::Item::Use(_) => "item".to_owned(),
            _ => name.clone(),
        };
        self.enter(name, export_name, node.span(), |s| syn::visit::visit_item(s, node));
    }
    fn visit_impl_item(&mut self, node: &'ast syn::ImplItem) {
        if self.coverage.is_some() {
            return self.coverage_scope(node.span(), false, true, |s| {
                syn::visit::visit_impl_item(s, node)
            });
        }
        let name = match node {
            syn::ImplItem::Fn(i) => format!("fn {}", i.sig.ident),
            syn::ImplItem::Const(i) => format!("const {}", i.ident),
            syn::ImplItem::Type(i) => format!("type {}", i.ident),
            _ => "item".into(),
        };
        let export_name = match node {
            syn::ImplItem::Fn(i) => i.sig.ident.to_string(),
            syn::ImplItem::Const(i) => i.ident.to_string(),
            syn::ImplItem::Type(i) => i.ident.to_string(),
            _ => "item".to_owned(),
        };
        self.enter(name, export_name, node.span(), |s| syn::visit::visit_impl_item(s, node));
    }
    fn visit_trait_item(&mut self, node: &'ast syn::TraitItem) {
        if self.coverage.is_some() {
            return self.coverage_scope(node.span(), false, true, |s| {
                syn::visit::visit_trait_item(s, node)
            });
        }
        let name = match node {
            syn::TraitItem::Fn(i) => format!("fn {}", i.sig.ident),
            syn::TraitItem::Const(i) => format!("const {}", i.ident),
            syn::TraitItem::Type(i) => format!("type {}", i.ident),
            _ => "item".into(),
        };
        let export_name = match node {
            syn::TraitItem::Fn(i) => i.sig.ident.to_string(),
            syn::TraitItem::Const(i) => i.ident.to_string(),
            syn::TraitItem::Type(i) => i.ident.to_string(),
            _ => "item".to_owned(),
        };
        self.enter(name, export_name, node.span(), |s| syn::visit::visit_trait_item(s, node));
    }
    fn visit_variant(&mut self, node: &'ast syn::Variant) {
        if self.coverage.is_some() {
            return self
                .coverage_scope(node.span(), false, true, |s| syn::visit::visit_variant(s, node));
        }
        self.enter(format!("variant {}", node.ident), node.ident.to_string(), node.span(), |s| {
            syn::visit::visit_variant(s, node)
        });
    }
    fn visit_fn_arg(&mut self, node: &'ast syn::FnArg) {
        if self.coverage.is_some() {
            return self
                .coverage_scope(node.span(), false, false, |s| syn::visit::visit_fn_arg(s, node));
        }
        let name = match node {
            syn::FnArg::Receiver(_) => "self".into(),
            syn::FnArg::Typed(i) => i.pat.to_token_stream().to_string(),
        };
        self.enter(format!("argument {name}"), format!("arg {name}"), node.span(), |s| {
            syn::visit::visit_fn_arg(s, node)
        });
    }
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if !self.check_exemptions && !self.check_exports {
            return;
        }
        let template = self.exports.generator()
            && node.path.segments.last().is_some_and(|s| {
                ["quote", "quote_spanned", "parse_quote", "parse_quote_spanned"]
                    .iter()
                    .any(|name| s.ident == name)
            });
        if !template {
            self.exports.scope.push(format!("tokens {}", node.path.to_token_stream()));
            self.macro_tokens(node.tokens.clone());
            self.exports.scope.pop();
        }
    }
    fn visit_field(&mut self, node: &'ast syn::Field) {
        if self.coverage.is_some() {
            return self
                .coverage_scope(node.span(), false, true, |s| syn::visit::visit_field(s, node));
        }
        let name = if let Some(ident) = &node.ident {
            format!("field {ident}")
        } else {
            let ordinal = self.tuple_fields.entry(self.scopes.join("/")).or_default();
            let name = format!("field {} {}", *ordinal, node.ty.to_token_stream());
            *ordinal += 1;
            name
        };
        self.enter(
            name,
            node.ident.as_ref().map(ToString::to_string).unwrap_or_else(|| "field".to_owned()),
            node.span(),
            |s| syn::visit::visit_field(s, node),
        );
    }
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        if let Some(coverage) = &mut self.coverage {
            let current = *coverage.enclosing.last().expect("root scope");
            coverage.scopes[current].attrs.push(attr.clone());
        }
        self.meta(&attr.meta, "always");
        if self.check_exports
            && let Err(error) = self.exports.attribute(&attr.meta)
        {
            self.exports.error = Some(error);
        }
    }
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
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
        let mut scan = Scan::new(path);
        scan.locate = matches!(mode.as_str(), "locate" | "locations");
        scan.check_exemptions = mode == "inventory";
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
