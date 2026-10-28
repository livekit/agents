//! The `#[host(..)]` vocabulary of the declaration model, read here and nowhere else. The macros
//! that make the attributes legal Rust expand them to nothing.
//!
//! | where | key | meaning |
//! |---|---|---|
//! | trait | `class(name = "X", native = Trait)` | a host class `_core.X` the runtime or a core implements; `native`: it is taken as itself where the host passes that `#[host(proxy)]` trait |
//! | trait | `proxy(host = "module.Class")` | the core calling a host object of that class; declared (its proxy generated) when its methods declare `await`/`notify`/`read_once`/`bind` |
//! | proxy method | `await` | an `async fn(&self, ..) -> Result<T, E>`: the host call awaited in its context; dropping it cancels the host task. `T` may be `Box<dyn StreamSource<I>>`: the host's async iterable, pulled in a copy of the call's context |
//! | proxy method | `notify` | a `fn(&self, ..)` returning nothing: scheduled on the host, not awaited; a failure is logged |
//! | proxy method | `read_once` | a `fn(&self) -> T`: read when the proxy is taken from the host (a failure refuses the object), then answered from that value |
//! | proxy method | `bind` | a `fn(&self, x: &X) -> Option<Arc<dyn Self>>`: a copy of the proxy passing `x` after the arguments of every awaited call; no host call |
//! | proxy method | `bind(wrap = T)` | `bind`, the copy handed out as `T::bound(copy, x, clock)`: a native wrapper the core runs around the host object (a host STT's retry loop) |
//! | proxy parameter | `context` | the one parameter whose `HostContext` is the context the call runs in (it is passed too); a call without one runs in the context the proxy captured when it was taken (the host's running loop and a copy of its context) |
//! | error type | `error(raised = "module.Class")` | the host refuses a proxy call by raising that class: `Result<Result<T, E>, _>`'s `E`, from its message |
//! | associated type | `source(item = T, class = "X")` | a pull source of `T`, handed out as stream class `X` |
//! | associated type | `class(name = "X")` | an instance of the declared class `X`: the generated `Implementations` binds this type to `X`'s implementation |
//! | associated type | `handle(class = "X")` | an implementation's own object, handed out (never taken in) as the registration shell's class `crate::handles::X`, built `From` it |
//! | trait method | `new` | the constructor: returns `Self` or `Result<Self, E>` |
//! | trait method | `call` | a synchronous call on `&self` |
//! | trait method | `await` | an `async fn` on `&self`, awaited by the host |
//! | trait method | `getter` | a read-only property on `&self` |
//! | trait method | `next` | `async fn next(&self) -> Result<Option<T>, E>`: the class is an async iterator of `T` (`__aiter__`/`__anext__`); `None` ends it |
//! | trait method | `name = "x"` | the host name, when the Rust one cannot be it |
//! | parameter | `kw` | keyword-only from this parameter on |
//! | parameter | `default = <literal>` | omittable: `None`, `true`/`false`, a number or a string |
//! | error type | `error(with = path)`, `error(kind = "ValueError")`, `error(original)`, per-variant `kind`/`delegate` | `Result<_, E>` raises through E's `HostException`, the one conversion its declaration expands to |
//! | record | `host_type = "module.Name"` | the host class it is read from, when not a generated model |
//! | type | `string` | crosses as its string: `Display` to the host, `FromStr` from it |
//! | type | `handwritten(extract, emit, host = "..")` | crosses by its own pyo3 conversions; `host` is its annotation (to be replaced by a record declaration) |
//! | `#[derive(Host)]` record | `handwritten(extract)` | read through its own `FromPyObject` beside the derive's emit |
//! | struct, enum | `record(module = "..", name = "..", extract, emit)` | a schema-defined record crossing as its generated model (`_generated/<module>.py`): a struct is a frozen dataclass, a data-variant enum one class per variant (the class is the tag) and a union alias, a unit-only enum a `Literal`; conversions written into `lk-agents-abi/src/python/generated/` |
//! | record struct | `record(.., persisted)` | application-saved state: the model gains `to_json()`/`from_json()` through the record's serde codec in Rust (it derives `Serialize`/`Deserialize`) |
//! | record field, tuple variant | `default = <literal>` | the model's default: an omitted keyword only; `None` stays a value an `Option` field takes and any other refuses |
//! | record field, tuple variant | `skip` | does not cross (emit-only records) |
//! | tuple variant | `field = "x"` | the name its one value crosses under |
//! | variant | `name = "X"` | its model class, when not `<Enum><Variant>` |
//! | unit variant of a literal record | `rename = "x"` | its string, when not the snake-cased identifier |
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
