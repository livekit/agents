//! The `#[host(..)]` vocabulary of the declaration model, read here and nowhere else. The macros
//! that make the attributes legal Rust expand them to nothing.
//!
//! | where | key | meaning |
//! |---|---|---|
//! | trait | `class(name = "X", native = Trait, module = "module", subclass)` | a host class `_core.X` the runtime or a core implements; `native`: it is taken as itself where the host passes that `#[host(proxy)]` trait; `module` preserves Python identity, `subclass` allows Python subclasses |
//! | trait | `native` | accepts only the registered native class (`class(native = Trait)` or legacy `object(as_dyn(Trait = method))`); the reader and type error are generated in Python bindings, with no ABI proxy; `native(class = "X")`: `_core.X`, a class the registration shell registers itself |
//! | trait | `proxy(host = "module.Class")` | the core calling a host object of that class; declared (its proxy generated) when its methods declare `await`/`notify`/`read_once`/`bind`; `view = "module.Class"`: the proxy calls that class's instance over the host object (a host class without the `_core_*` surface) |
//! | proxy method | `await` | an `async fn(&self, ..) -> Result<T, E>`, or returning nothing (a close: its failure is logged) (`self: Box<Self>` consumes the proxy, as a writer's close does): the host call awaited in its context; dropping it cancels the host task. `T` may be `Box<dyn StreamSource<I>>`: the host's async iterable, pulled in a copy of the call's context |
//! | proxy method | `call` | a `fn(&self, ..) -> Result<T, E>`: the host method called in place under the GIL, never awaiting (a read the host answers at once); a failure is the declared error; a host stream it returns is pulled in the call's context (its `context` parameter, else the captured one) |
//! | proxy method | `notify` | a `fn(&self, ..)` returning nothing: scheduled on the host, not awaited, delivered in call order and never dropped; a failure is logged. `notify(lossy)`: a frame of a stream, dropped (and logged) while the host loop is far behind |
//! | proxy method | `read_once` | a `fn(&self) -> T`: read when the proxy is taken from the host (a failure refuses the object), then answered from that value; `-> &T` lends it |
//! | proxy method | `bind` | a `fn(&self, x: &X) -> Option<Arc<dyn Self>>`: a copy of the proxy passing `x` after the arguments of every awaited call; no host call |
//! | proxy method | (none, with a default body) | native only: the proxy keeps the default (what only a native implementation answers) |
//! | proxy method | `bind(wrap = T)` | `bind`, the copy handed out as `T::bound(copy, x, clock)`: a native wrapper the core runs around the host object (a host STT's retry loop) |
//! | proxy trait | `: Any + Send + Sync` | a proxy of it passed to another proxy's host call (`Arc<dyn X>`, optional or not) is handed back as the host object it was taken from (before any `view`); a native implementation is refused there |
//! | proxy parameter | `source(class = "X")` | a `Box<dyn StreamSource<T>>` the core hands the host call as the generated stream class `X` (`recv`/`aclose`), written into the proxy's family |
//! | proxy parameter | `context` | the one parameter whose `HostContext` is the context the call runs in (it is passed too); a call without one runs in the context the proxy captured when it was taken (the host's running loop and a copy of its context) |
//! | proxy or class parameter | `context` on a `CallContext` | what the call runs in (`lk_agents_abi::invoke::CallContext`), never crossing: a class method's binding captures it at the call's entry (`crate::host::entered()`), hidden from the host's signature; a proxy call runs in it without passing it |
//! | proxy method | `requires(tool_call, ..)` | scopes (`job`, `agent`, `speech`, `tool_call`, `approval`) the awaited or notify call's `context` parameter must hold; without one the runtime refuses the call before the host is entered, with the declared error |
//! | error type | `error(raised = "module.Class")` | the host refuses a proxy call by raising that class: `Result<Result<T, E>, _>`'s `E`, from its message |
//! | associated type | `source(item = T, class = "X")` | a pull source of `T`, handed out as stream class `X` |
//! | associated type | `sink(item = I, class = "X")` | the input half of stream class `X`, returned as `(Self::Sink, Self::Source)` beside the `source(.., class = "X")`: one duplex class (`push`/`flush`/`end_input`, `recv`, `aclose` ending the input); its bound is `StreamSink<I>` or a `#[host(sink)]` trait over it whose `call`s are methods of `X` |
//! | trait | `sink` | a duplex stream class's input half: extends `StreamSink<I>`; every method is a `call` of that class |
//! | associated type | `env` | `type Env`: what the hosting process lends the class (`lk_agents_net::env::ProcessEnv`); a `&Self::Env` parameter is passed the runtime's (`crate::process::env()`), hidden from the host's signature |
//! | associated type | `class(name = "X")` | an instance of the declared class `X`: the generated `Implementations` binds this type to `X`'s implementation |
//! | associated type | `handle(class = "X")` | an implementation's own object, handed out (never taken in) as the registration shell's class `crate::handles::X`, built `From` it |
//! | trait method | `new` | the constructor: returns `Self` or `Result<Self, E>` |
//! | trait method | `call` | a synchronous call on `&self` or `&mut self`; no receiver declares a static class method. Synchronous class calls may borrow another instance as `&Self` |
//! | trait method | `await` | an `async fn` on `&self`, awaited by the host; or `fn(&self, ..) -> BoxFuture<'static, T>`, called at the host's call (a command sent in call order) and its future awaited |
//! | class method | `call, detach` | releases the GIL while the synchronous native operation executes; arguments are converted before release and results afterward |
//! | trait method | `getter` | a read-only property on `&self` |
//! | trait method | `call, future = T` | a synchronous method returning `HostRef` containing a host-owned future; its completion type `T` gives the generated `Future[T]` annotation |
//! | trait method | `setter` | a writable property on `&self` with one value parameter and unit result |
//! | trait method | `next` | `async fn next(&self) -> Result<Option<T>, E>`: the class is an async iterator of `T` (`__aiter__`/`__anext__`); `None` ends it |
//! | trait method | `name = "x"` | the host name, when the Rust one cannot be it |
//! | class/function trait | `class(.., pid = false)`, `functions(pid = false)` | retain the pure export opt-out from the fork guard; defaults to true |
//! | trait method | `pid = true/false` | override its class or function group fork guard |
//! | parameter | `kw` | keyword-only from this parameter on |
//! | parameter | `default = <literal>` | omittable: `None`, `true`/`false`, a number or a string |
//! | parameter | `one_or_many` | a list the host may also pass as one item (`T \| list[T]`) |
//! | parameter | `pcm` | a class method's bytes the host may also pass as a 16-bit PCM view, a partial sample included |
//! | parameter | `shape_error(kind = "ValueError", message = "..", items)` | only the outer JSON-object shape, or with `items` each dict-record shape in a strict list; `{value}` formats a nonobject JSON value; nested conversion failures keep their own errors |
//! | error type | `error(with = path)`, `error(kind = "ValueError")`, `error(original)`, per-variant `kind`/`delegate`/`delegate = field`, `kind = "module.Class", kwargs`, a struct's `by = field, kinds(Variant(class = "module.Class", kwargs(a, b)), ..)` | `Result<_, E>` raises through E's `HostException`, the one conversion its declaration expands to; `kwargs` also passes a named-field variant's fields as that class's keyword arguments; `by` raises the class the field's variant names, with the listed fields as keyword arguments |
//! | record | `host_type = "module.Name"` | the host class it is read from, when not a generated model |
//! | type | `string` | crosses as its string: `Display` to the host, `FromStr` from it |
//! | type | `handwritten(extract, emit, host = "..")` | crosses by its own pyo3 conversions; `host` is its annotation (to be replaced by a record declaration) |
//! | `#[derive(Host)]` record | `handwritten(extract)` | read through its own `FromPyObject` beside the derive's emit; with `host = ".."`, crosses both ways by its own conversions |
//! | struct, enum | `record(module = "..", name = "..", extract, emit)` | a schema-defined record crossing as its generated model (`_generated/<module>.py`): a struct is a frozen dataclass, a data-variant enum one class per variant (the class is the tag) and a union alias, a unit-only enum a `Literal`; conversions are free functions written into `lk-agents-py/src/generated/records.rs` |
//! | record struct | `record(.., constant(type = "x"))` | a model field at that fixed value (a discriminator its readers compare), never Rust state: read past, emitted by its default |
//! | transparent unit variant | `host_type = "module.Name"` | read and emit that existing marker class; without it the unit crosses as `None` |
//! | record enum | `record(.., transparent)` | every variant is one value crossing as itself (`str \| Message`, `Policy \| dict[str, Policy] \| None`), told apart by its class (a model class, a literal record's `str`, else its Python type; a `float` takes an `int`); one unit variant may cross as `None` (a field of the record then takes `default = None`): the model is a union alias, no class per variant |
//! | record struct | `record(.., nested_in = "Owner")` | its model is a class nested in the record struct `Owner`'s model of the same module, named `Owner.Name` (`RealtimeModelMetrics.InputTokenDetails`) |
//! | record enum | `record(.., transparent, dispatch = "session_dispatch")` | the host routes its members by model class through `_generated/<dispatch>.py` (`PUBLIC_EVENTS`, `MIRRORED_STATES`, `IGNORED_EVENTS`) |
//! | dispatched variant | `event`, `event(mirror = "_attr")`, `ignored` | emitted as the public event its model's `type` constant names (`mirror`: its `new_state` also kept on the session's `_attr`), or received and dropped; an unrouted member has host behaviour |
//! | record struct | `record(.., default)` | Rust `Default` from the fields' `default = <literal>` (a `Maybe` field: not given): one list of defaults for both languages; the one Rust the record marker expands to |
//! | record struct | `record(.., persisted)` | application-saved state: the model gains private `_to_json()`/`_from_json()` through the record's serde codec in Rust (it derives `Serialize`/`Deserialize`) |
//! | record field, tuple variant | `default = <literal>` | the model's default: an omitted keyword only; `None` stays a value an `Option` field takes and any other refuses |
//! | record field, tuple variant | `skip` | does not cross (emit-only records) |
//! | record field | `default_factory = "module.name"` | the model's default: that host callable, called at construction (`time.time`); with `skip`, Rust never sends it and the host stamps it, read back on extract |
//! | record enum | `record(.., enum)` | a unit-only enum crossing as a `str` `Enum` class (`CloseReason.ERROR`, `.value` its string), not a `Literal` |
//! | record struct field | `omitted = path` | an extracted scalar the host may leave out: `path(&record)` computes it from the record read (the model's `__post_init__` asks `_core` for it), an explicit value stays |
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
    /// A declaration macro's native diagnostic, evaluated by the conversion it generates.
    Expression(syn::Expr),
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

/// A typed boolean option, distinct from a bare marker or quoted text.
pub fn boolean(at: &str, entries: &[Entry], key: &str) -> Result<Option<bool>, String> {
    match find(entries, key).map(|entry| &entry.value) {
        None => Ok(None),
        Some(Some(Value::Literal(Literal::Bool(value)))) => Ok(Some(*value)),
        Some(_) => Err(format!("{at}: `{key}` takes a boolean")),
    }
}

pub fn rust(at: &str, entries: &[Entry], key: &str) -> Result<Option<syn::Type>, String> {
    match find(entries, key).map(|e| &e.value) {
        None => Ok(None),
        Some(Some(Value::Rust(t))) => Ok(Some(t.clone())),
        Some(Some(Value::Expression(expr))) => Err(format!(
            "{at}: `{key}` takes a Rust type or path, found expression `{}`",
            quote::quote!(#expr),
        )),
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
            let parsed = if key == "msg" {
                Value::Expression(value.parse::<syn::Expr>()?)
            } else if value.peek(syn::LitStr) {
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
