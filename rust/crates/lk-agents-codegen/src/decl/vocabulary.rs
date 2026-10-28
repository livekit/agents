//! The `#[host(..)]` vocabulary of the declaration model, read here and nowhere else. The macros
//! that make the attributes legal Rust expand them to nothing.
//!
//! | where | key | meaning |
//! |---|---|---|
//! | trait | `class(name = "X", native = Trait)` | a host class `_core.X` the runtime or a core implements; `native`: it is taken as itself where the host passes that `#[host(proxy)]` trait |
//! | trait | `proxy(host = "module.Class")` | the core calling a host object of that class; declared (its proxy generated) when its methods declare `await`/`notify` |
//! | proxy method | `await` | an `async fn(&self, ..) -> Result<T, E>`: the host call awaited in its context; dropping it cancels the host task |
//! | proxy method | `notify` | a `fn(&self, ..)` returning nothing: scheduled on the host, not awaited; a failure is logged |
//! | proxy parameter | `context` | the one parameter whose `HostContext` is the context the call runs in (it is passed too) |
//! | error type | `error(raised = "module.Class")` | the host refuses a proxy call by raising that class: `Result<Result<T, E>, _>`'s `E`, from its message |
//! | associated type | `source(item = T, class = "X")` | a pull source of `T`, handed out as stream class `X` |
//! | trait method | `new` | the constructor: returns `Self` or `Result<Self, E>` |
//! | trait method | `call` | a synchronous call on `&self` |
//! | trait method | `await` | an `async fn` on `&self`, awaited by the host |
//! | trait method | `getter` | a read-only property on `&self` |
//! | trait method | `name = "x"` | the host name, when the Rust one cannot be it |
//! | parameter | `kw` | keyword-only from this parameter on |
//! | parameter | `default = <literal>` | omittable: `None`, `true`/`false`, a number or a string |
//! | error type | `error(with = path)` | `Result<_, E>` raises through `path(E) -> PyErr` |
//! | error type | `error(kind = "ValueError")` | `Result<_, E>` raises that builtin with E's message |
//! | record | `host_type = "module.Name"` | the host class it is read from, when not a generated model |
//! | type | `handwritten(extract, emit, host = "..")` | crosses by its own pyo3 conversions; `host` is its annotation (to be replaced by a record declaration) |
//!
//! Rust is written unquoted (`item = LlmChunk`, `with = crate::python::raise`), host text quoted
//! (`name = "InferenceLlm"`). Keys outside this table fail generation.

use syn::parse::Parser as _;

use super::Literal;

/// One key of a `#[host(..)]` list: `key`, `key = value` or `key(nested..)`.
#[derive(Debug, Clone)]
pub struct Entry {
    pub key: String,
    pub value: Option<Value>,
    pub nested: Vec<Entry>,
}

#[derive(Debug, Clone)]
pub enum Value {
    /// Host text.
    Text(String),
    /// A Rust type or path.
    Rust(syn::Type),
    Literal(Literal),
}

/// Every `#[host(..)]` (or `#[crate::host(..)]`, `#[lk_agents_abi::host(..)]`) on `attrs`, flattened.
pub fn read(attrs: &[syn::Attribute]) -> Result<Vec<Entry>, String> {
    let mut out = Vec::new();
    for attr in attrs.iter().filter(|a| is_host(a.path())) {
        if let syn::Meta::Path(_) = &attr.meta {
            continue;
        }
        let list = attr.meta.require_list().map_err(|e| e.to_string())?;
        out.extend(
            entries(list.tokens.clone()).map_err(|e| format!("#[host({})]: {e}", list.tokens))?,
        );
    }
    Ok(out)
}

/// Reads `allowed` keys only, each at most once.
pub fn only(at: &str, entries: &[Entry], allowed: &[&str]) -> Result<(), String> {
    for (i, e) in entries.iter().enumerate() {
        if !allowed.contains(&e.key.as_str()) {
            return Err(format!(
                "{at}: `{}` is not a key here (allowed: {})",
                e.key,
                allowed.join(", ")
            ));
        }
        if entries[..i].iter().any(|o| o.key == e.key) {
            return Err(format!("{at}: `{}` is given twice", e.key));
        }
    }
    Ok(())
}

pub fn find<'a>(entries: &'a [Entry], key: &str) -> Option<&'a Entry> {
    entries.iter().find(|e| e.key == key)
}

pub fn text(at: &str, entries: &[Entry], key: &str) -> Result<Option<String>, String> {
    match find(entries, key).map(|e| &e.value) {
        None => Ok(None),
        Some(Some(Value::Text(t))) => Ok(Some(t.clone())),
        Some(_) => Err(format!("{at}: `{key}` takes host text, quoted")),
    }
}

pub fn rust(at: &str, entries: &[Entry], key: &str) -> Result<Option<syn::Type>, String> {
    match find(entries, key).map(|e| &e.value) {
        None => Ok(None),
        Some(Some(Value::Rust(t))) => Ok(Some(t.clone())),
        Some(_) => Err(format!("{at}: `{key}` takes a Rust type or path, unquoted")),
    }
}

pub fn is_host(path: &syn::Path) -> bool {
    path.segments.last().is_some_and(|s| s.ident == "host")
        && matches!(
            path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().as_slice(),
            [_] | [_, _]
        )
}

fn entries(tokens: proc_macro2::TokenStream) -> syn::Result<Vec<Entry>> {
    let mut out = Vec::new();
    let parser = syn::meta::parser(|meta| {
        let key = meta
            .path
            .get_ident()
            .map(ToString::to_string)
            .ok_or_else(|| meta.error("a key is one word"))?;
        if meta.input.peek(syn::Token![=]) {
            let value = meta.value()?;
            let parsed = if value.peek(syn::LitStr) {
                Value::Text(value.parse::<syn::LitStr>()?.value())
            } else if value.peek(syn::Lit) || value.peek(syn::Token![-]) {
                Value::Literal(literal(&value.parse::<syn::Expr>()?)?)
            } else if value.peek(syn::Ident)
                && value.fork().parse::<syn::Ident>().is_ok_and(|i| i == "None")
            {
                value.parse::<syn::Ident>()?;
                Value::Literal(Literal::None)
            } else {
                Value::Rust(value.parse::<syn::Type>()?)
            };
            out.push(Entry { key, value: Some(parsed), nested: Vec::new() });
        } else if meta.input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in meta.input);
            let nested = entries(content.parse::<proc_macro2::TokenStream>()?)?;
            out.push(Entry { key, value: None, nested });
        } else {
            out.push(Entry { key, value: None, nested: Vec::new() });
        }
        Ok(())
    });
    parser.parse2(tokens)?;
    Ok(out)
}

fn literal(e: &syn::Expr) -> syn::Result<Literal> {
    let (negative, inner) = match e {
        syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => (true, &*u.expr),
        other => (false, other),
    };
    let syn::Expr::Lit(lit) = inner else {
        return Err(syn::Error::new_spanned(e, "a default is a literal"));
    };
    let sign = if negative { -1.0 } else { 1.0 };
    Ok(match &lit.lit {
        syn::Lit::Bool(b) if !negative => Literal::Bool(b.value),
        syn::Lit::Int(i) => Literal::Int(i.base10_parse::<i64>()? * if negative { -1 } else { 1 }),
        syn::Lit::Float(f) => Literal::Float(f.base10_parse::<f64>()? * sign),
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "a default is None, a bool, a number or a string",
            ));
        }
    })
}
