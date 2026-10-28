//! Parse architecture lint exemptions; used by the Python gate, never product code.
use proc_macro2::Span;
use quote::ToTokens;
use std::collections::BTreeMap;
use syn::{spanned::Spanned, visit::Visit};

#[derive(Default)]
struct Scan {
    path: String,
    scopes: Vec<String>,
    locations: Vec<(Span, String)>,
    exemptions: Vec<String>,
    occurrences: BTreeMap<String, usize>,
}

impl Scan {
    fn enter(&mut self, name: String, span: Span, f: impl FnOnce(&mut Self)) {
        self.scopes.push(name);
        self.locations.push((span, self.scopes.join("/")));
        f(self);
        self.scopes.pop();
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
                                self.meta(&attr.meta, &attr.to_token_stream().to_string());
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
                for item in items.iter().skip(1) {
                    self.meta(item, context);
                }
            }
        } else if matches!(kind.as_str(), "allow" | "expect" | "warn") {
            use syn::parse::Parser;
            if let Ok(items) =
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated
                    .parse2(list.tokens.clone())
            {
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

impl<'ast> Visit<'ast> for Scan {
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
        let name = format!(
            "field {}",
            node.ident
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| node.ty.to_token_stream().to_string())
        );
        self.enter(name, node.span(), |s| syn::visit::visit_field(s, node));
    }
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.meta(&attr.meta, &attr.to_token_stream().to_string());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mode = args.first().ok_or("expected inventory or locate")?;
    for path in args.iter().skip(1).take(if mode == "locate" || mode == "locations" {
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
            for exemption in scan.exemptions {
                println!("{exemption}");
            }
        } else {
            return Err("expected inventory or locate".into());
        }
    }
    Ok(())
}
