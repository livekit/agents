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

/// `#[host(error(with = path))]`: the error converts through that function of its own crate.
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
    Str,
    /// `serde_json::Value`: any JSON value.
    Json,
    /// `serde_json::Map<String, serde_json::Value>`: a JSON object.
    JsonObject,
    /// `Option<T>`: the value or the explicit null.
    Optional(Box<Ty>),
    List(Box<Ty>),
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
    },
    /// A pull source of `item`, handed to the host as the stream class `class`.
    Source {
        item: Box<Ty>,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int {
    U8,
    U16,
    U32,
    U64,
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
            Ty::Str => "::std::string::String".into(),
            Ty::Json => "::serde_json::Value".into(),
            Ty::JsonObject => {
                "::serde_json::Map<::std::string::String, ::serde_json::Value>".into()
            }
            Ty::Optional(t) => format!("::std::option::Option<{}>", t.rust()),
            Ty::List(t) => format!("::std::vec::Vec<{}>", t.rust()),
            Ty::Tuple(ts) if ts.len() == 1 => format!("({},)", ts[0].rust()),
            Ty::Tuple(ts) => {
                format!("({})", ts.iter().map(Ty::rust).collect::<Vec<_>>().join(", "))
            }
            Ty::Record(r) => r.rust.clone(),
            Ty::This => "Self".into(),
            Ty::Proxy { rust, .. } => format!("::std::sync::Arc<dyn {rust}>"),
            Ty::Source { item, .. } => {
                format!("::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<{}>>", item.rust())
            }
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
            Int::I32 => "i32",
            Int::I64 => "i64",
        }
    }
}
