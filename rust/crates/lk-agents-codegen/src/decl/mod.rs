//! The declaration model: every `#[host(..)]` contract of the ABI crate, read once and resolved
//! into one IR that every emitter consumes.
//!
//! - [`tree`]: the ABI crate's modules, found from `src/lib.rs`, and path resolution over them.
//! - [`vocabulary`]: the attribute vocabulary, the one reader of `#[host(..)]`.
//! - [`resolve`]: declarations into the IR below; a type means what its declaration says, never
//!   what its last path segment spells. An unsupported or ambiguous declaration is an error naming it.
//!
//! The IR holds no `syn` types and no maps: generated output must be byte-stable.

pub mod resolve;
pub mod tree;
pub mod vocabulary;

#[cfg(test)]
mod tests;

/// Everything the ABI declares, in source order.
#[derive(Debug, Clone, Default)]
pub struct Declarations {
    pub classes: Vec<Class>,
    pub proxies: Vec<Proxy>,
    pub records: Vec<RecordDecl>,
}

/// `#[host(record(module = "..", extract, emit))]` on a struct or enum: a schema-defined record
/// crossing as a generated model (`_generated/<module>.py`), never a dict. Its pyo3 conversions
/// are generated into the ABI crate (`src/python/generated/`).
#[derive(Debug, Clone)]
pub struct RecordDecl {
    /// `::lk_agents_abi::agent::ValidatedOutput`.
    pub rust: String,
    /// The host name: the model class, the union alias or the literal alias.
    pub name: String,
    /// The generated Python module (`agent_run`: `_generated/agent_run.py`).
    pub module: String,
    /// The ABI subject: the generated Rust module it is written into.
    pub family: String,
    pub source: String,
    pub doc: Vec<String>,
    pub extract: bool,
    pub emit: bool,
    /// `persisted`: application-saved state. The model's `to_json()`/`from_json()` go through
    /// the record's own serde codec in Rust (`_core._core_persist_<name>`/`_core_restore_<name>`),
    /// so every variant, empty ones included, round-trips by its tag.
    pub persisted: bool,
    /// Computed views declared with the generated models' property vocabulary.
    pub members: Vec<crate::ir::Member>,
    pub body: RecordBody,
}

#[derive(Debug, Clone)]
pub enum RecordBody {
    /// A model class of these fields.
    Struct(Vec<Field>),
    /// A data-variant enum: one model class per variant, the class is the tag.
    Variants(Vec<Variant>),
    /// A unit-only enum: a `Literal` of each variant's host string.
    Literals(Vec<(String, String)>),
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub rust_name: String,
    /// The model class: `<Enum><Variant>` unless `name = ".."`.
    pub name: String,
    pub doc: Vec<String>,
    pub shape: VariantShape,
}

#[derive(Debug, Clone)]
pub enum VariantShape {
    Unit,
    /// `Variant(T)`: one field, named by `#[host(field = "..")]`; `None` when it is `skip`ped.
    Tuple(Option<Field>),
    Named(Vec<Field>),
}

/// A record field. Presence is declared: an `Option<T>` field takes the explicit null, any other
/// refuses it naming `Record.field`; `default` is what an omitted keyword gives the model, and
/// nothing else; a `skip`ped field does not cross (emit-only records).
#[derive(Debug, Clone)]
pub struct Field {
    pub rust_name: String,
    pub ty: Ty,
    pub default: Option<Literal>,
    pub skip: bool,
    pub doc: Vec<String>,
}

impl Field {
    /// The Python attribute: the Rust name without a raw identifier's `r#` (`r#type` is `type`).
    pub fn host_name(&self) -> &str {
        self.rust_name.trim_start_matches("r#")
    }
}

/// `#[host(proxy(host = ".."))] trait X` whose methods declare their calls: the core calling a
/// host object. The generated proxy implements the trait over the object; each call runs in the
/// host context its `#[host(context)]` parameter carries, or, with none, the one the proxy
/// captured when it was taken from the host.
#[derive(Debug, Clone)]
pub struct Proxy {
    /// The trait's identifier: the generated proxy is `<rust_name>Proxy`.
    pub rust_name: String,
    pub trait_path: String,
    /// The ABI subject: the generated module it is written into.
    pub family: String,
    pub source: String,
    /// The host class the object is an instance of: its annotation.
    pub host: String,
    /// `view = "module.Class"`: the class whose instance over the host object the proxy calls,
    /// for host classes that do not carry the `_core_*` surface themselves (the io legs).
    pub view: Option<(String, String)>,
    /// `#[async_trait]` on the trait: the implementation carries it too.
    pub async_trait: bool,
    pub methods: Vec<ProxyMethod>,
}

impl RecordDecl {
    /// The `_core` functions saving and restoring a persisted record.
    pub fn persistence(&self) -> (String, String) {
        let name = resolve::snake(&self.name);
        (format!("_core_persist_{name}"), format!("_core_restore_{name}"))
    }
}

impl Proxy {
    /// Some call has no `#[host(context)]` parameter: the proxy captures the host's context
    /// (running loop and a copy of its context) when it is taken, and those calls run in it.
    pub fn captures_context(&self) -> bool {
        self.methods
            .iter()
            .any(|m| m.context.is_none() && matches!(m.kind, ProxyCall::Await | ProxyCall::Notify))
    }

    /// The parameter a `bind` method appends to every awaited call of its proxy.
    pub fn bound(&self) -> Option<&Param> {
        self.methods.iter().find(|m| m.kind == ProxyCall::Bind).map(|m| &m.params[0])
    }
}

#[derive(Debug, Clone)]
pub struct ProxyMethod {
    pub rust_name: String,
    /// The host method called; the Rust name unless `name = ".."`.
    pub name: String,
    pub kind: ProxyCall,
    /// Passed to the host in order, the context parameter included.
    pub params: Vec<Param>,
    /// The `#[host(context)]` parameter: its type implements `HostContext`. `None`: the context
    /// the proxy captured.
    pub context: Option<usize>,
    pub ret: ProxyRet,
    /// `bind(wrap = T)`: the bound copy is handed out as `T::bound(copy, &x, clock)`, a native
    /// wrapper the core runs around it (the retry loop of a host STT).
    pub wrap: Option<String>,
    /// `self: Box<Self>`: an awaited call that consumes the proxy.
    pub consumes: bool,
    /// `read_once` returning `&T`: lends the value the proxy read.
    pub lends: bool,
}

/// How the core calls the host: declared, then checked against the signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyCall {
    /// `async fn`: awaited; dropping the call cancels the host's task, which still settles.
    Await,
    /// A sync `fn(&self, ..) -> Result<T, E>`: the host method called in place under the GIL,
    /// never awaiting; for a read the host answers at once (a room's connection state).
    Call,
    /// A sync `fn` returning nothing: scheduled on the host and not awaited; a failure is logged.
    Notify,
    /// `fn(&self) -> T`: read from the host once, when the proxy is taken; a failure to read it
    /// refuses the object.
    ReadOnce,
    /// `fn(&self, x: &X) -> Option<Arc<dyn Self>>`: this proxy with `x` appended to the arguments
    /// of every awaited call. No host call.
    Bind,
}

/// What an awaited call returns: `Result<T, E>`, or `Result<Result<T, R>, E>` when the host may
/// refuse the call with the declared exception `R` is raised as.
#[derive(Debug, Clone)]
pub struct ProxyRet {
    pub ok: Ty,
    pub refusal: Option<Refusal>,
    /// `E`: any other host failure, kept as the host raised it (`FromHostErr`).
    pub err: Option<String>,
}

/// `#[host(error(raised = "module.Class"))]`: the host raises an instance of that class; the
/// error is built from its message (`From<String>`).
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub rust: String,
    pub module: String,
    pub class: String,
}

/// `#[host(class(..))] trait X`: a host class implemented natively; or `#[host(functions)] trait
/// X`: module functions. The runtime or a core
/// implements the trait; the generated adapter calls it through the selected implementation.
#[derive(Debug, Clone)]
pub struct Class {
    pub form: Form,
    /// The host class name (`_core.InferenceLlm`); the trait's for functions.
    pub name: String,
    /// The trait's identifier: the `Implementations` associated type is named after it.
    pub rust_name: String,
    /// `::lk_agents_abi::llm::inference::InferenceLlm`.
    pub trait_path: String,
    /// The ABI subject the class belongs to (`llm`): its generated adapter module.
    pub family: String,
    /// Repo-relative declaring file.
    pub source: String,
    pub doc: Vec<String>,
    /// `native = Trait`: the `#[host(proxy)]` traits it implements; wherever the host passes
    /// one of those, an instance of this class is taken as itself rather than proxied.
    pub native: Vec<String>,
    /// `(assoc, class)`: the associated types that are instances of another declared class.
    pub instances: Vec<(String, String)>,
    pub methods: Vec<Method>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// Shared by the host: `&self` methods only, any of them awaitable.
    Class,
    /// Owned by one host object: some method takes `&mut self` (pyo3's borrow flag guards it).
    MutableClass,
    /// Module functions: every method is a `function`.
    Functions,
}

/// An associated type of a class, by its declared kind (`type Chunks` as a source).
#[derive(Debug, Clone)]
pub struct Assoc {
    pub rust_name: String,
    pub ty: Ty,
}

#[derive(Debug, Clone)]
pub struct Method {
    pub rust_name: String,
    /// `&mut self`.
    pub mutable: bool,
    /// The host name; the Rust one unless `name = ".."`.
    pub name: String,
    pub kind: MethodKind,
    pub params: Vec<Param>,
    pub ret: Ret,
    pub doc: Vec<String>,
}

/// How the host calls a method: declared, then checked against the signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodKind {
    /// The constructor: returns `Self` or `Result<Self, E>`.
    New,
    /// A synchronous call on `&self` (or `&mut self`: [`Form::MutableClass`]).
    Call,
    /// A module function: no receiver, synchronous.
    Function,
    /// An `async fn` on `&self`, awaited by the host.
    Await,
    /// A read-only property on `&self`.
    Getter,
    /// `async fn(&self) -> Result<Option<T>, E>`: the class is an async iterator of `T`
    /// (`__aiter__`/`__anext__`); `None` ends it.
    Next,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
    /// The trait takes it by reference (`&str`, `&[T]`, `&T`).
    pub by_ref: bool,
    /// Keyword-only: this parameter and every one after it.
    pub kw_only: bool,
    /// The value the host may omit to get. `None` is the explicit null, and only an optional
    /// parameter declares it.
    pub default: Option<Literal>,
}

/// A declared default, written once as a host literal and converted for both sides.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

#[derive(Debug, Clone)]
pub struct Ret {
    /// [`Ty::Unit`] for none.
    pub ok: Ty,
    /// `Result<_, E>`: E's declared conversion.
    pub err: Option<ErrorDecl>,
}

/// `#[host(error(..))]`: `with` is the expression converting the error to the host's
/// (a function of the error's crate, or a builtin exception carrying its message).
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorDecl {
    pub rust: String,
    pub with: String,
}

/// A resolved type. Each variant is a declared conversion kind with one rule per direction.
#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Unit,
    Bool,
    Int(Int),
    Float,
    /// `std::time::Duration`: seconds, as a float on the host.
    Duration,
    Str,
    /// `serde_json::Value`: any JSON value.
    Json,
    /// `serde_json::Map<String, serde_json::Value>`: a JSON object.
    JsonObject,
    /// `Option<T>`: the value or the explicit null.
    Optional(Box<Ty>),
    List(Box<Ty>),
    /// `BTreeMap<K, V>`: a dict, its keys converted by their declared kind.
    Map(Box<Ty>, Box<Ty>),
    /// `BTreeSet<T>`: a set (a `frozenset` on a model).
    Set(Box<Ty>),
    Tuple(Vec<Ty>),
    /// A `#[derive(Host)]` record, crossing by its generated conversions.
    Record(Record),
    /// The declaring class itself (`Self`).
    This,
    /// `Arc<dyn Trait>` of a `#[host(proxy)]` trait: a native class as itself, any other host
    /// object behind the generated proxy.
    Proxy {
        rust: String,
        host: String,
        /// A generated provider base (`#[python(provider(..))]`): generic over its event names, so
        /// any instantiation is accepted.
        generic: bool,
        /// A declaration-model proxy: `(family, trait)`, generated as `<family>::<trait>Proxy`.
        /// `None`: the trait's in-place `HostProxy`.
        declared: Option<(String, String)>,
        /// `Box<dyn Trait>`: an object the host hands over for the core to own (and consume, as a
        /// writer's `self: Box<Self>` close does); always behind the declared proxy.
        boxed: bool,
    },
    /// A pull source of `item`, handed to the host as the stream class `class`.
    Source {
        item: Box<Ty>,
        class: String,
    },
    /// `Box<dyn StreamSource<T>>` a host call returns: the host's async iterable of `T`, pulled
    /// in its own context (`lk_agents_runtime::host::stream::HostStream`). Input only.
    HostSource(Box<Ty>),
    /// `(Box<dyn StreamSink<I>>, Box<dyn StreamSource<T>>)` a host call returns: the host's duplex
    /// stream, input through its `_core_push`/`_core_flush`/`_core_end_input` and output by
    /// iterating it, in one context (`HostStream::duplex`). Input only.
    HostDuplex {
        input: Box<Ty>,
        output: Box<Ty>,
    },
    /// A declared error (`#[host(error(with = ..))]`) passed to the host as the exception it
    /// raises as. Output only.
    Error(ErrorDecl),
    /// `Self::<assoc>` declared `class(name = "X")`: an instance of the declared class `X` (its
    /// implementation is the `Implementations` type the generated bound ties `assoc` to). Output.
    Instance {
        assoc: String,
        class: String,
    },
    /// `Arc<T>` of an `object(cell = "arc")` class: the `_core` object itself, shared both ways.
    Shared {
        rust: String,
        class: String,
    },
    /// `Self::<assoc>` declared `handle(class = "X")`: an implementation's own object handed to
    /// the host as the registration shell's class `crate::handles::X` (`From` the implementation's
    /// type). Output only.
    Handle {
        assoc: String,
        class: String,
    },
}

/// What the host holds a record as.
#[derive(Debug, Clone, PartialEq)]
pub enum HostForm {
    /// A generated model: `_generated/<module>.py`'s `name`.
    Model { module: String, name: String },
    /// A `_core` class.
    CoreClass(String),
    /// A host class the framework does not generate (`#[host(host_type = "module.Name")]`), read
    /// from its attributes.
    External { module: String, name: String },
    /// A dict of the record's fields (`#[binding(from_items)]` with no model).
    Dict,
    /// `#[host(handwritten(.., host = ".."))]`: the annotation the declaration gives.
    Annotation(String),
    /// A `#[host(record)]`: instances of the generated model (`_generated/<module>.py`), both ways.
    Record { module: String, name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int {
    U8,
    U16,
    U32,
    U64,
    Usize,
    I32,
    I64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// `::lk_agents_abi::llm::chat_context::ChatItem`.
    pub rust: String,
    pub host: HostForm,
    /// Declared directions: the host may pass it in (`extract`) / receive it (`emit`).
    pub extract: bool,
    pub emit: bool,
}

impl Ty {
    /// The Rust type, written from outside the ABI crate.
    pub fn rust(&self) -> String {
        match self {
            Ty::Unit => "()".into(),
            Ty::Bool => "bool".into(),
            Ty::Int(i) => i.rust().into(),
            Ty::Float => "f64".into(),
            Ty::Duration => "::std::time::Duration".into(),
            Ty::Str => "::std::string::String".into(),
            Ty::Json => "::serde_json::Value".into(),
            Ty::JsonObject => {
                "::serde_json::Map<::std::string::String, ::serde_json::Value>".into()
            }
            Ty::Optional(t) => format!("::std::option::Option<{}>", t.rust()),
            Ty::List(t) => format!("::std::vec::Vec<{}>", t.rust()),
            Ty::Map(k, v) => format!("::std::collections::BTreeMap<{}, {}>", k.rust(), v.rust()),
            Ty::Set(t) => format!("::std::collections::BTreeSet<{}>", t.rust()),
            Ty::Tuple(ts) if ts.len() == 1 => format!("({},)", ts[0].rust()),
            Ty::Tuple(ts) => {
                format!("({})", ts.iter().map(Ty::rust).collect::<Vec<_>>().join(", "))
            }
            Ty::Record(r) => r.rust.clone(),
            Ty::Shared { rust, .. } => format!("::std::sync::Arc<{rust}>"),
            Ty::This => "Self".into(),
            Ty::Proxy { rust, boxed: true, .. } => format!("::std::boxed::Box<dyn {rust}>"),
            Ty::Proxy { rust, .. } => format!("::std::sync::Arc<dyn {rust}>"),
            Ty::Handle { assoc, .. } | Ty::Instance { assoc, .. } => format!("Self::{assoc}"),
            Ty::Source { item, .. } | Ty::HostSource(item) => {
                format!("::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<{}>>", item.rust())
            }
            Ty::Error(e) => e.rust.clone(),
            Ty::HostDuplex { input, output } => format!(
                "(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<{}>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<{}>>)",
                input.rust(),
                output.rust()
            ),
        }
    }
}

impl Int {
    pub fn rust(self) -> &'static str {
        match self {
            Int::U8 => "u8",
            Int::U16 => "u16",
            Int::U32 => "u32",
            Int::U64 => "u64",
            Int::Usize => "usize",
            Int::I32 => "i32",
            Int::I64 => "i64",
        }
    }
}
