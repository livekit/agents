//! What a `#[derive(Host)]` declaration is, once parsed. Not `syn` types, so every emitter gets
//! one answer from the parser. No maps anywhere: generated output must be byte-stable.

/// One `#[derive(Host)]` type, in the order it was declared.
#[derive(Debug, Clone)]
pub struct Decl {
    pub rust_name: String,
    /// e.g. `lk-agents-voice/src/activity/events.rs`; goes into the generated file's header.
    pub source: String,
    /// `///` lines, verbatim, leading space stripped: the Python docstring.
    pub doc: Vec<String>,
    /// The `// ---` group this declaration sits under: the Rust layout is the generated layout.
    pub section: Option<String>,
    pub body: Body,
    /// Empty is rejected by the parser.
    pub targets: Vec<Target>,
    /// `#[python(dispatch = "module")]` on a union: where its `PUBLIC_EVENTS`/`IGNORED_EVENTS`
    /// table is generated.
    pub dispatch: Option<String>,
    /// `#[python(dispatch_also = "Record")]`: a record the same stream carries outside the union
    /// (`StreamClosedWire`), whose kind joins the table's `EVENT_KINDS`. Repeatable.
    pub dispatch_also: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Body {
    Struct(Vec<Field>),
    /// Fieldless enum. Python `(str, Enum)`, so `ev.reason == "job_shutdown"` stays `True`.
    Unit(Vec<UnitVariant>),
    /// One type per variant: the discriminated union. The tag is the member's own discriminant.
    /// A member may be [`Ty::Foreign`] (`OverlappingSpeechEvent`), named but never generated.
    Union(Vec<Variant>),
}

/// One arm of a [`Body::Union`]: the Rust variant and the one type it carries.
#[derive(Debug, Clone)]
pub struct Variant {
    /// Never a host name: the host tells members apart by their discriminant.
    pub rust_name: String,
    pub ty: Ty,
    /// `#[binding(twin = "..")]`: which of the member's emit structs this arm writes
    /// (`InterruptionMetrics` has two).
    pub twin: Option<String>,
    /// `#[python(event)]` / `#[python(ignored)]`.
    pub dispatch: Option<Dispatch>,
    /// `#[python(event, mirror = "_attr")]`: the session attribute the event's `new_state` is
    /// written to before it is emitted, so the session's property reads what the event said.
    pub mirror: Option<String>,
    /// `#[python(kind = "..")]`: the wire kind of a host-stream member whose record serialises by
    /// hand (`ToolsExecutedEvent`), so no declaration carries its tag. Only for such a member.
    pub kind: Option<String>,
}

/// A host-stream member's Python routing, generated into the union's `dispatch` module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    /// A public session event, nothing more.
    Event,
    Ignored,
}

#[derive(Debug, Clone)]
pub struct UnitVariant {
    /// The wire value, from `#[host(rename = "..")]`, never derived from the variant name. The
    /// only name carried: Python upper-cases it.
    pub value: String,
    pub doc: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub rust_name: String,
    /// Differs from `rust_name` only via `#[host(rename)]`, for keywords like `type`.
    pub name: String,
    pub doc: Vec<String>,
    pub ty: Ty,
    /// The discriminant: `#[python(literal = "..")]`, or `literal_const = "TYPE"` read from the
    /// `impl` so the identity is not spelled twice.
    pub literal: Option<String>,
    /// Verbatim Python source.
    pub default: Option<String>,
    /// A Python callable, so it crosses as a string.
    pub default_factory: Option<String>,
    /// `#[python(serializer = "..")]`: a `@field_serializer` calling that name in `_bindings`.
    pub serializer: Option<String>,
    /// pydantic `Field(exclude=True)`: still a visible field. Not [`Field::py_skip`].
    pub exclude: bool,
    /// `#[python(skip)]`: not on the host class at all (`LlmChunk.flush`, an in-band sentinel the
    /// reference's `ChatChunk` lacks).
    pub py_skip: bool,
    /// `#[python(private)]`: a `PrivateAttr`.
    pub private: bool,
    /// `#[python(import = "module:Name")]` on the field, whatever its type: a default factory can
    /// name anything, and [`Foreign::python_from`] only covers `ty` overrides.
    pub imports: Vec<String>,
    /// Same, under `if TYPE_CHECKING:`.
    pub type_imports: Vec<String>,
    /// `#[binding(skip)]`: a host-supplied field the emit struct omits (`created_at`: the core has
    /// no wall clock, and a zero would date every event to 1970).
    pub emit_skip: bool,
    /// `#[binding(default)]`: a missing attribute is `Default::default()`. Opt-in per field
    /// because pyo3's `default` collapses absent and `None` on an `Option`.
    pub extract_default: bool,
    /// `#[binding(default = "<expr>")]`, emitted as `#[pyo3(default = <expr>)]`. For fields where
    /// `Default` is wrong: `ToolExecutionOutput.reply_required` must default to `true`.
    pub extract_default_value: Option<String>,
}

/// A field's type, resolved as far as the parser can and no further.
#[derive(Debug, Clone)]
pub enum Ty {
    Str,
    Int,
    Float,
    Bool,
    /// Legal only with a `literal`.
    Unit,
    /// `serde_json::Value`.
    Any,
    Option(Box<Ty>),
    List(Box<Ty>),
    Map(Box<Ty>, Box<Ty>),
    Tuple(Vec<Ty>),
    /// By Rust name, resolved at emit time so a rename lands everywhere.
    Declared(String),
    /// A type we do not own, e.g. `#[python(ty = "ChatMessage", from = "..llm")]`.
    Foreign(Foreign),
}

/// A host type the declaration names but does not own.
#[derive(Debug, Clone)]
pub struct Foreign {
    /// Verbatim: `ChatMessage`, `LLMError | STTError`.
    pub python: String,
    /// `module:Name,Name` groups, one per `#[python(import)]`; a list because one annotation can
    /// span several modules (`ErrorEvent.error`).
    pub python_from: Vec<String>,
    /// Under `if TYPE_CHECKING:`: `SpeechCreatedEvent.speech_handle` would be an import cycle.
    pub python_type_from: Vec<String>,
}

/// A field the host class has and the core struct does not.
#[derive(Debug, Clone)]
pub struct HostField {
    pub name: String,
    pub ty: String,
    /// `None` emits a required field.
    pub default: Option<String>,
    pub default_factory: Option<String>,
    /// A `PrivateAttr` (`ImageContent._cache`).
    pub private: bool,
    pub doc: String,
    /// `type_import = "module:Name"`: what the annotation names, imported under `TYPE_CHECKING`.
    pub type_import: Option<String>,
}

/// A generated property or method, forwarded to its declared hook.
#[derive(Debug, Clone)]
pub struct Member {
    pub name: String,
    pub is_property: bool,
    /// After `self`, verbatim Python. Empty for a property.
    pub args: String,
    pub ret: String,
    pub doc: String,
    /// The hook name; defaults to `<snake_type>_<name>` so two models can share a member.
    pub hook: String,
}

/// One host's view of one declaration.
#[derive(Debug, Clone)]
pub struct Target {
    pub host: Host,
    pub kind: Kind,
    /// Relative to the host's output root, without an extension.
    pub module: String,
    pub name: String,
    /// The frozen public module, emitted as `__module__`: `repr`, `pickle` and the equivalence
    /// gate read it, and the generator cannot see the re-export.
    pub path: Option<String>,
    /// `#[python(total = "False")]` on a [`Kind::TypedDict`]. API, not style: `total=True` makes
    /// every key required. Rejected on other kinds.
    pub total: bool,
    /// `model_config = ConfigDict(..)`, verbatim.
    pub config: Option<String>,
    /// `#[host(validator = "..")]`: a `@model_validator(mode="after")` body in `_bindings`.
    pub validator: Option<String>,
    /// `#[python(tag = "type", tag_const = "TYPE")]`: `(name, value)`, emitted as the first field.
    /// Type-level, not a `kind: ()` marker field every Rust construction site would have to write.
    pub tag: Option<(String, String)>,
    /// `#[python(host_field(..))]`: fields only the host class has, emitted after the declared
    /// ones.
    pub host_fields: Vec<HostField>,
    /// `#[python(tag_after = "id")]`: pydantic field order is `model_dump()` and `repr` order, and
    /// `chat_context.py` puts `id` before `type`.
    pub tag_after: Option<String>,
    /// `#[python(base = "..")]`: the gate compares `bases` (`_BaseMetrics`).
    pub base: Option<String>,
    /// `#[python(nested_in = "..")]`: nested classes are public members of the outer one.
    pub nested_in: Option<String>,
    /// `#[python(post_init = "..")]`: a `__post_init__` calling that `_bindings` name.
    pub post_init: Option<String>,
    /// `#[python(repr = "..")]`: a `__repr__` calling that `_bindings` name.
    pub repr: Option<String>,
    /// `#[python(discriminator = "type")]`: `Annotated[.., Field(discriminator=..)]` rather than
    /// `A | B`; pydantic validates the two differently.
    pub discriminator: Option<String>,
    /// `#[python(property(..))]` / `#[python(method(..))]`, each a call into `_bindings`. Not
    /// `_core`: `ChatChunk.has_response` runs per token on Python data, a GIL crossing each time.
    pub members: Vec<Member>,
    /// `#[python(inline)]`: no host name; expanded at every use (`ToolExecutionUpdatedEvent.update`).
    pub inline: bool,
    /// `#[binding(class, new, ..)]`: a keyword constructor. Opt-in, since constructibility is API.
    pub constructor: bool,
    /// `#[python(order = "a, b, c")]`: the Python field order when it differs from the Rust (emit)
    /// order (`InterruptionOptions`). Empty means Rust order; otherwise names every field once.
    pub order: Vec<String>,
    /// `#[python(typed_dict_from = "typing")]`, default `typing_extensions`: they are different
    /// classes before 3.13 and `is_typeddict()` tells them apart.
    pub typed_dict_from: Option<String>,
    /// `#[binding(from_items)]`: the binding reads it as a dict, so a host hands it `asdict(..)`.
    pub from_items: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Host {
    Python,
    /// The pyo3 binding: `FromPyObject`/`IntoPyObject` twins emitted into `lk-agents-py`, since
    /// the declaring crates may not take pyo3.
    Binding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Pydantic,
    Dataclass,
    TypedDict,
    /// `(str, Enum)` in Python.
    Enum,
    Plain,
    Union,
    /// `Literal["a", "b"]`, not a class. Distinct from [`Kind::Enum`]: `UserState` has no members.
    Literal,
    /// Binding only: a `#[derive(FromPyObject)]` struct plus `From<_>` for the declared type.
    Extract,
    /// Binding only: a `#[derive(IntoPyObject)]` struct plus `From<&Core>`. Dict-shaped, since
    /// `IntoPyObject` rejects `#[pyo3(attribute(..))]`.
    Emit,
    /// Binding only: `#[pyclass] pub struct XPy(pub Core)` with one `#[getter]` per field. Shape
    /// only: a class with its own methods is hand-written, since a stub cannot span two files.
    Class,
}

pub fn kind_word(k: Kind) -> &'static str {
    match k {
        Kind::Pydantic => "pydantic",
        Kind::Dataclass => "dataclass",
        Kind::TypedDict => "typed_dict",
        Kind::Enum => "enum",
        Kind::Plain => "plain",
        Kind::Union => "union",
        Kind::Literal => "literal",
        Kind::Extract => "extract",
        Kind::Emit => "emit",
        Kind::Class => "class",
    }
}

impl Decl {
    /// Linear over at most two entries; a map would add an iteration order to the emit path.
    pub fn target(&self, host: Host) -> Option<&Target> {
        self.targets.iter().find(|t| t.host == host)
    }

    /// The binding has two targets for a type crossing both ways: its `IntoPyObject` and
    /// `FromPyObject` structs disagree field by field.
    pub fn targets_for(&self, host: Host) -> impl Iterator<Item = &Target> {
        self.targets.iter().filter(move |t| t.host == host)
    }

    pub fn fields(&self) -> &[Field] {
        match &self.body {
            Body::Struct(f) => f,
            _ => &[],
        }
    }
}

// Exposed core items (`#[host(export)]`, `#[host(object)]`, ...), read from `EXPORT_SOURCES`.
// [`RTy`] keeps Rust borrowing: the wrapper must know whether it passes `&name` or `name`.

/// `#[host(export)]` on a free fn or a `pub const` (a const becomes a zero-argument function).
#[derive(Debug, Clone)]
pub struct ExportFn {
    /// `name = ".."`, else the Rust identifier (lower-cased for a const).
    pub name: String,
    pub doc: Vec<String>,
    pub sig: Sig,
    /// `export(attr)`: a module attribute holding the value, not a function.
    pub is_attr: bool,
    /// The wrapper returns an awaitable; its arguments are owned.
    pub is_async: bool,
}

/// One callable's host signature.
#[derive(Debug, Clone)]
pub struct Sig {
    pub params: Vec<Param>,
    /// The `Ok` type when the fn returns a `Result`.
    pub ret: RTy,
    /// `defaults(x = "..")`: a Rust expression for pyo3's `signature`.
    pub defaults: Vec<(String, String)>,
    pub kw_only: KwOnly,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KwOnly {
    None,
    All,
    /// `kw_only = "x"`: `x` and every parameter after it.
    From(String),
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: RTy,
}

/// A Rust type as the wrapper must spell it.
#[derive(Debug, Clone, PartialEq)]
pub enum RTy {
    Unit,
    /// Verbatim: width matters to pyo3's range check.
    Prim(String),
    Str,
    StrRef,
    IntoString,
    Bytes,
    Option(Box<RTy>),
    Vec(Box<RTy>),
    /// `BTreeSet<T>`/`HashSet<T>`: a host `set`.
    Set(Box<RTy>),
    Slice(Box<RTy>),
    /// `impl Iterator<Item = T>` (return only): collected into a list.
    Iter(Box<RTy>),
    Tuple(Vec<RTy>),
    Json {
        by_ref: bool,
    },
    JsonMap {
        by_ref: bool,
    },
    /// Seconds as a float.
    Duration,
    /// Unix seconds as a float.
    SystemTime,
    Map(Box<RTy>),
    /// `impl IntoIterator<Item = T>` (argument only): a `list[T]`.
    IntoIter(Box<RTy>),
    /// `lk_agents_abi::Maybe<T>`: `NOT_GIVEN` (or `None`) is `NotGiven`; `Maybe<Option<T>>` keeps
    /// `NOT_GIVEN`, `None` and a value apart.
    Maybe(Box<RTy>),
    /// A unit enum crosses as its string, a record as its binding twin.
    Declared {
        name: String,
        by_ref: bool,
    },
    /// `Arc<dyn Trait>` of a `#[host(proxy)]` trait: the host object in that proxy.
    Dyn(String),
    /// `Arc<T>` of an `arc`-cell object (returned only): that class over the same `Arc`.
    Shared(String),
    /// A host object interned in the side table.
    HostRef,
    /// One half of a stream pair (returned only).
    StreamHalf {
        sink: bool,
    },
    SelfTy,
}

/// `#[host(object)]` on a core `impl` block: a `#[pyclass]` holding the core value.
#[derive(Debug, Clone)]
pub struct Object {
    /// The core type's identifier.
    pub core: String,
    /// `name = ".."`, else the core identifier.
    pub name: String,
    pub doc: Vec<String>,
    pub methods: Vec<Method>,
    /// `stub = ".."`: stub lines for hand-written methods (`multiple-pymethods`), so the stub
    /// stays one class.
    pub stub_extra: Vec<String>,
    pub stub_imports: Vec<String>,
    /// `#[host(const)]` associated consts: class attributes.
    pub consts: Vec<ObjConst>,
}

#[derive(Debug, Clone)]
pub struct ObjConst {
    pub name: String,
    pub ty: RTy,
}

#[derive(Debug, Clone)]
pub struct Method {
    pub name: String,
    pub kind: MethodKind,
    pub sig: Sig,
    /// `Reply<T>` (or `Result<Reply<T>, E>`): the host gets an awaitable of `T`; `sig.ret` is `()`.
    pub reply: Option<ReplySig>,
    /// The method's `///` lines (a generated Python method forwarding to it carries them).
    pub doc: Vec<String>,
}

/// `T` of `Reply<T>`, or `U` of `Reply<Result<U, E>>`.
#[derive(Debug, Clone)]
pub struct ReplySig {
    pub ret: RTy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodKind {
    New,
    Getter,
    Setter,
    Method,
    Static,
    Async,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Receiver {
    None,
    Ref,
    Mut,
}

/// `#[host(constant)]` on a `pub const`: a module-level `Final` in its file's Python module
/// (`#[host(file(module, path))]`), declared once in Rust and read by the core under its own name.
#[derive(Debug, Clone)]
pub struct Constant {
    pub name: String,
    pub module: String,
    pub source: String,
    pub doc: Vec<String>,
    pub value: ConstValue,
}

#[derive(Debug, Clone)]
pub enum ConstValue {
    /// A Python literal, written as is: a string, an integer, a float or a boolean.
    Literal(String),
    /// A declared record built from literals: its Rust name and `(rust field, literal)` pairs.
    Record(String, Vec<(String, String)>),
}

#[derive(Debug, Clone, Default)]
pub struct Exposed {
    pub fns: Vec<ExportFn>,
    pub objects: Vec<Object>,
    pub wire_enums: Vec<WireEnum>,
    pub kwargs: Vec<KwRecord>,
    pub proxies: Vec<Proxy>,
    pub providers: Vec<Provider>,
    pub str_classes: Vec<StrClass>,
    pub wrappers: Vec<Wrapper>,
    /// `#[host(error(kind, declare))]`: an exception class of `_core`'s own.
    pub exceptions: Vec<Exception>,
    /// `#[python(forward(module = ..))]` on an export: `(module, exported name, source)`.
    pub forwards: Vec<(String, String, String)>,
}

/// `class <name>(<base>)`, raised for the Rust error it is declared on.
#[derive(Debug, Clone)]
pub struct Exception {
    pub name: String,
    pub base: String,
    pub doc: Vec<String>,
}

/// `#[python(wrapper(module, path, name, base))]` on a `#[host(object)]` impl: a host class
/// subclassing `base` (`"module:Name"`) whose constructor takes the object's `new` parameters and
/// keeps the core object as `self._core`; `base` reads it.
#[derive(Debug, Clone)]
pub struct Wrapper {
    pub name: String,
    pub module: String,
    pub path: Option<String>,
    pub source: String,
    pub doc: Vec<String>,
    pub base: String,
    /// The object's `_core` class name.
    pub core: String,
}

/// `#[python(str_class(module, path, new, property(..), method(..)))]` on a `#[host(string)]`
/// type: a Python `str` subclass whose constructor normalises through the exported fn `new`, and
/// whose properties and methods each apply one exported fn to the string.
#[derive(Debug, Clone)]
pub struct StrClass {
    pub name: String,
    pub module: String,
    pub path: Option<String>,
    pub source: String,
    pub doc: Vec<String>,
    pub new: String,
    /// `(python name, exported fn)`.
    pub properties: Vec<(String, String)>,
    pub methods: Vec<(String, String)>,
}

impl Exposed {
    pub fn is_empty(&self) -> bool {
        self.fns.is_empty()
            && self.objects.is_empty()
            && self.wire_enums.is_empty()
            && self.kwargs.is_empty()
            && self.proxies.is_empty()
            && self.exceptions.is_empty()
    }

    pub fn extend(&mut self, other: Exposed) {
        self.fns.extend(other.fns);
        self.objects.extend(other.objects);
        self.wire_enums.extend(other.wire_enums);
        self.kwargs.extend(other.kwargs);
        self.proxies.extend(other.proxies);
        self.providers.extend(other.providers);
        self.str_classes.extend(other.str_classes);
        self.wrappers.extend(other.wrappers);
        self.exceptions.extend(other.exceptions);
        self.forwards.extend(other.forwards);
    }
}

/// `#[host(wire_enum)]`: crosses as its string without being a `#[derive(Host)]` record.
#[derive(Debug, Clone)]
pub struct WireEnum {
    pub name: String,
}

/// `#[host(kwargs)]` / `#[host(client(..))]`: fields spread into keyword arguments.
#[derive(Debug, Clone)]
pub struct KwRecord {
    pub name: String,
    pub fields: Vec<KwField>,
}

#[derive(Debug, Clone)]
pub struct KwField {
    pub name: String,
    pub ty: RTy,
    /// A Rust expression for pyo3's signature.
    pub default: Option<String>,
}

/// `#[host(proxy(..))]` on a core trait: an `impl Trait` over a host object, whose every method
/// calls `_core_<method>` on it.
#[derive(Debug, Clone)]
pub struct Proxy {
    /// The trait's identifier.
    pub trait_name: String,
    /// The trait's method names.
    pub methods: Vec<String>,
}

/// `#[python(provider(..))]` on a `#[host(proxy)]` trait: the Python base class plugins subclass,
/// generated with the `_core_*` surface the proxy calls, so the two cannot drift.
#[derive(Debug, Clone)]
pub struct Provider {
    pub source: String,
    /// `proxy(host)`'s last segment, else `provider(name)`.
    pub class: String,
    /// The `_generated` module it is written into, after the module's records.
    pub module: String,
    /// The public module, as `__module__`.
    pub path: String,
    pub doc: Vec<String>,
    /// `events = ".."`: the `rtc.EventEmitter` events; empty is no emitter.
    pub events: Vec<String>,
    /// `error(name = "..", kind = "..")`: the `error` event's class and its `type`, which
    /// `_emit_error` emits.
    pub error: Option<(String, String)>,
    pub methods: Vec<ProviderMethod>,
}

/// One trait method: its `_core_<rust>` method, and the Python face its role gives it.
#[derive(Debug, Clone)]
pub struct ProviderMethod {
    pub rust: String,
    pub doc: Vec<String>,
    pub role: Role,
    pub params: Vec<ProviderParam>,
    /// Verbatim; `None` returns nothing.
    pub ret: Option<String>,
    pub is_async: bool,
    /// The trait gives it a body: the Python hook has a default.
    pub has_default: bool,
    /// `when = "cap"`: the `_core_` method answers `None` unless `capabilities.<cap>`.
    pub when: Option<String>,
    /// `stream(..)`: the class this method returns, generated after the provider.
    pub stream: Option<StreamClass>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    /// `label`: the class path.
    Label,
    /// `init`: a constructor keyword, stored, read by a property of the method's name.
    Init,
    /// `identity`: the `ProviderId` fields but `label` as properties plugins override, and
    /// `metrics_metadata`.
    Identity,
    /// `hook = "name"` (default: the Rust name): a method plugins override. `impl = ".."`: the
    /// hook calls that abstract method. `unsupported = ".."`: the default raises
    /// `NotImplementedError` with that text. `store = "attr", default = ".."`: the default keeps
    /// its one argument in `self.<attr>`, which holds `default` until then.
    Hook {
        name: String,
        implementation: Option<String>,
        /// `retry = "helper"`: the hook is `_bindings.<helper>`, which runs `impl` in the attempts
        /// the core's retry policy decides and reports its metrics; the framework's own call is
        /// one attempt.
        retry: Option<String>,
        unsupported: Option<String>,
        store: Option<(String, String)>,
    },
    /// `conn`: the provider under one `conn_options`, which the declared proxy binds and passes
    /// to each awaited `_core_` call; no host surface of its own.
    Conn,
    /// `markup`: `markup._provider_key()`, the nested `Markup` class plugins override.
    Markup,
    /// `chat`: a `_core.ChatRequest` read into the hook's keywords.
    Chat { name: String },
    /// `report`: the framework's report of a failed attempt, the provider's `error` event.
    Report,
    /// `value = "expr"`: the `_core_` method answers a Python expression.
    Value(String),
}

#[derive(Debug, Clone)]
pub struct ProviderParam {
    pub name: String,
    pub rust_ty: String,
    /// `ty = ".."`: the hook's annotation, when it is not the core's.
    pub py_ty: Option<String>,
    /// `import = "module:Name"`: what `ty` names.
    pub import: Option<String>,
    /// `default = ".."`: the hook takes it as a keyword with this default (verbatim Python).
    pub default: Option<String>,
    /// `attr`: set on the method's result rather than passed to the hook.
    pub attr: bool,
}

/// `stream(name = "..", kind = "..", alias = "..", doc = "..")` on the method returning it.
#[derive(Debug, Clone)]
pub struct StreamClass {
    pub name: String,
    /// The stream idiom: `recognize`, `synthesize`, `chunked`, `chat`, `vad`.
    pub kind: String,
    /// A deprecated second name.
    pub alias: Option<String>,
    /// `proxy = "Trait"`: the `#[host(proxy)]` trait the class answers (every method's `_core_`).
    pub proxy: Option<String>,
    pub doc: Vec<String>,
}
