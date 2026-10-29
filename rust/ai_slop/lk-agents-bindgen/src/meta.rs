//! The one reader of the attribute language, shared by the expansions and `lk-agents-codegen`.
//! Nesting is a key prefix (`method.defaults.size`). Rust values are unquoted ([`Metas::expr`],
//! `ty`, `path`, `ident`); host text is a string literal ([`Metas::text`] refuses anything else).

use proc_macro2::{Span, TokenStream, TokenTree};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A bare word: `kw_only`, `emit`.
    Word,
    /// A string literal: host text.
    Text(String),
    /// Anything else after `=`, as Rust tokens: `-1`, `None`, `crate::X`, `false`.
    Rust(String),
}

#[derive(Clone, Debug)]
pub struct Meta {
    pub key: String,
    pub value: Value,
}

/// Read once: `parse_nested_meta` stops at the first unconsumed entry, so per-key lookups would
/// drop the rest.
#[derive(Clone, Default, Debug)]
pub struct Metas(pub Vec<Meta>);

/// `#[host(..)]` by any path (`host`, `crate::host`, `lk_agents_abi::host`).
pub fn is_host(a: &syn::Attribute) -> bool {
    a.path().segments.last().is_some_and(|s| s.ident == "host")
}

fn err(message: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(Span::call_site(), message)
}

impl Metas {
    /// The attribute's own tokens (what an attribute macro receives).
    pub fn parse(tokens: TokenStream) -> syn::Result<Metas> {
        let mut out = Vec::new();
        let parser = syn::meta::parser(|m| walk(&m, "", &mut out));
        syn::parse::Parser::parse2(parser, tokens)?;
        Ok(Metas(out))
    }

    pub fn of(attrs: &[syn::Attribute]) -> syn::Result<Metas> {
        Self::named(attrs, "host")
    }

    /// By the path's last segment; a bare `#[<name>]` contributes nothing.
    pub fn named(attrs: &[syn::Attribute], name: &str) -> syn::Result<Metas> {
        let mut out = Vec::new();
        for a in attrs.iter().filter(|a| a.path().segments.last().is_some_and(|s| s.ident == name)) {
            if matches!(a.meta, syn::Meta::Path(_)) {
                continue;
            }
            a.parse_nested_meta(|m| walk(&m, "", &mut out))?;
        }
        Ok(Metas(out))
    }

    /// Each on its own: a type may carry two `stream(..)`s.
    pub fn each(attrs: &[syn::Attribute]) -> syn::Result<Vec<Metas>> {
        attrs.iter().filter(|a| is_host(a)).map(|a| Self::named(std::slice::from_ref(a), "host")).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `object` for `object(name = ..)`.
    pub fn head(&self) -> &str {
        self.0.first().map(|m| m.key.split('.').next().unwrap_or("")).unwrap_or("")
    }

    fn last(&self, key: &str) -> Option<&Value> {
        self.0.iter().rev().find(|m| m.key == key).map(|m| &m.value)
    }

    /// With or without a value.
    pub fn has(&self, key: &str) -> bool {
        self.0.iter().any(|m| m.key == key)
    }

    /// A bare word (`default`, not `default = ..`).
    pub fn flag(&self, key: &str) -> bool {
        self.0.iter().any(|m| m.key == key && m.value == Value::Word)
    }

    /// A string literal, refused in any other form.
    pub fn text(&self, key: &str) -> syn::Result<Option<&str>> {
        match self.last(key) {
            None | Some(Value::Word) => Ok(None),
            Some(Value::Text(s)) => Ok(Some(s)),
            Some(Value::Rust(r)) => Err(err(format!("`{key} = {r}`: this value is host text, write it as a string"))),
        }
    }

    /// A repeatable key's values, in order (`stub = ".."`).
    pub fn texts<'a>(&'a self, key: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.0.iter().filter(move |m| m.key == key).filter_map(|m| match &m.value {
            Value::Text(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// `true` for a bare word, else the written `true` / `false`.
    pub fn bool(&self, key: &str) -> syn::Result<Option<bool>> {
        match self.last(key) {
            None => Ok(None),
            Some(Value::Word) => Ok(Some(true)),
            Some(Value::Rust(v)) if v == "true" => Ok(Some(true)),
            Some(Value::Rust(v)) if v == "false" => Ok(Some(false)),
            Some(Value::Text(v) | Value::Rust(v)) => Err(err(format!("`{key} = {v}`: write true or false"))),
        }
    }

    /// The unquoted tokens; a string literal on a Rust key is a string expression.
    pub fn rust(&self, key: &str) -> Option<String> {
        rust_of(self.last(key)?)
    }

    pub fn expr(&self, key: &str) -> syn::Result<Option<TokenStream>> {
        match self.rust(key) {
            None => Ok(None),
            Some(v) => v.parse().map(Some).map_err(|_| err(format!("`{key} = {v}` is not a Rust expression"))),
        }
    }

    pub fn ty(&self, key: &str) -> syn::Result<Option<syn::Type>> {
        self.unquoted(key, "a type")
    }

    pub fn path(&self, key: &str) -> syn::Result<Option<syn::Path>> {
        self.unquoted(key, "a path")
    }

    pub fn ident(&self, key: &str) -> syn::Result<Option<syn::Ident>> {
        self.unquoted(key, "an identifier")
    }

    fn unquoted<T: syn::parse::Parse>(&self, key: &str, what: &str) -> syn::Result<Option<T>> {
        match self.last(key) {
            None | Some(Value::Word) => Ok(None),
            Some(Value::Rust(v)) => syn::parse_str(v).map(Some).map_err(|_| err(format!("`{key} = {v}` is not {what}"))),
            Some(Value::Text(v)) => Err(err(format!("`{key} = \"{v}\"`: {what} is Rust, write it unquoted"))),
        }
    }

    /// `defaults.` -> `[("size", Rust("- 1"))]`.
    pub fn under<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = (&'a str, &'a Value)> + 'a {
        self.0.iter().filter_map(move |m| m.key.strip_prefix(prefix).map(|k| (k, &m.value)))
    }

    /// Refuse any key not in `known` (key or `group.` prefix): unread keys would be dropped silently.
    pub fn only(&self, what: &str, known: &[&str]) -> syn::Result<()> {
        for m in &self.0 {
            let ok = known.iter().any(|k| if k.ends_with('.') { m.key.starts_with(k) } else { m.key == *k });
            if !ok {
                return Err(err(format!("{what}: `{}` is not a key here", m.key)));
            }
        }
        Ok(())
    }
}

pub fn rust_of(v: &Value) -> Option<String> {
    match v {
        Value::Word => None,
        Value::Rust(r) => Some(r.clone()),
        Value::Text(s) => Some(quote::quote!(#s).to_string()),
    }
}

fn walk(m: &syn::meta::ParseNestedMeta<'_>, prefix: &str, out: &mut Vec<Meta>) -> syn::Result<()> {
    let name = m.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
    let key = format!("{prefix}{name}");
    if m.input.peek(syn::Token![=]) {
        let input = m.value()?;
        let value = if input.peek(syn::LitStr) && (input.peek2(syn::Token![,]) || peek_end(input)) {
            Value::Text(input.parse::<syn::LitStr>()?.value())
        } else {
            Value::Rust(rust_source(input)?)
        };
        out.push(Meta { key, value });
    } else if m.input.peek(syn::token::Paren) {
        let inner = format!("{key}.");
        m.parse_nested_meta(|n| walk(&n, &inner, out))?;
    } else {
        out.push(Meta { key, value: Value::Word });
    }
    Ok(())
}

/// Whether the literal just ahead is the last token of the value.
fn peek_end(input: syn::parse::ParseStream<'_>) -> bool {
    let fork = input.fork();
    fork.parse::<syn::LitStr>().is_ok() && fork.is_empty()
}

/// Up to the next comma outside `<..>`. The text as written when spans know it (the codegen spells
/// Python defaults from it), else the tokens' own spelling.
fn rust_source(input: syn::parse::ParseStream<'_>) -> syn::Result<String> {
    let mut out = TokenStream::new();
    let mut angle = 0usize;
    let (mut first, mut last) = (None, None);
    while !input.is_empty() {
        if angle == 0 && input.peek(syn::Token![,]) {
            break;
        }
        let tt: TokenTree = input.parse()?;
        if let TokenTree::Punct(p) = &tt {
            match p.as_char() {
                '<' => angle += 1,
                '>' if angle > 0 => angle -= 1,
                _ => {}
            }
        }
        first.get_or_insert(tt.span());
        last = Some(tt.span());
        out.extend([tt]);
    }
    let (Some(first), Some(last)) = (first, last) else {
        return Err(input.error("a value after `=`"));
    };
    Ok(first.join(last).and_then(|s| s.source_text()).unwrap_or_else(|| out.to_string()))
}

/// A `wire_enum` variant's default string: `RoomIOEvent` -> `room_io_event`.
pub fn to_snake(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase() {
            let prev_lower = i > 0 && (chars[i - 1].is_ascii_lowercase() || chars[i - 1].is_ascii_digit());
            let next_lower = i + 1 < chars.len() && chars[i + 1].is_ascii_lowercase();
            let prev_upper = i > 0 && chars[i - 1].is_ascii_uppercase();
            if i > 0 && (prev_lower || (prev_upper && next_lower)) {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(*c);
        }
    }
    out
}

/// `///` lines with the single leading space removed, trailing blank lines dropped.
pub fn doc(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut out = Vec::new();
    for a in attrs.iter().filter(|a| a.path().is_ident("doc")) {
        if let syn::Meta::NameValue(nv) = &a.meta
            && let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = &nv.value
        {
            let v = s.value();
            out.push(v.strip_prefix(' ').unwrap_or(&v).to_owned());
        }
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    out
}
