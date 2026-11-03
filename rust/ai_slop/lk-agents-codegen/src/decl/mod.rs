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
#[path = "../decl_tests.rs"]
pub(crate) mod tests;

/// Everything the ABI declares, in source order.
#[derive(Debug, Clone, Default)]
pub struct Declarations {
    pub classes: Vec<Class>,
    pub proxies: Vec<Proxy>,
    pub records: Vec<RecordDecl>,
    pub errors: Vec<DeclaredError>,
    pub exports: Vec<Export>,
    pub local_interfaces: Vec<LocalInterface>,
    pub native_enums: Vec<NativeEnum>,
    pub flags: Vec<Flag>,
    pub constants: Vec<Constant>,
    pub str_classes: Vec<StrClass>,
    pub exception_families: Vec<crate::exceptions::Family>,
}

#[derive(Debug, Clone)]
pub struct NativeEnum {
    pub name: String,
    pub rust: String,
    pub variants: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Flag {
    pub name: String,
    pub module: String,
    pub path: String,
    pub values: Vec<(String, u64)>,
}

#[derive(Debug, Clone)]
pub struct Member {
    pub name: String,
    pub is_property: bool,
    pub args: String,
    pub arguments: Vec<String>,
    pub ret: String,
    pub doc: String,
    pub hook: String,
}

#[derive(Debug, Clone)]
pub struct Wrapper {
    pub name: String,
    pub module: String,
    pub path: Option<String>,
    pub source: String,
    pub doc: Vec<String>,
    pub base: String,
    pub core: String,
}

#[derive(Debug, Clone)]
pub struct StrClass {
    pub name: String,
    pub module: String,
    pub path: Option<String>,
    pub source: String,
    pub doc: Vec<String>,
    pub new: String,
    pub properties: Vec<(String, String)>,
    pub methods: Vec<(String, String)>,
}

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
    Literal(String),
    Record { rust: String, fields: Vec<(String, String)> },
}

/// Backend-local interpreter interfaces. They cannot declare public native exports or provider
/// calls; those belong to ABI classes and proxies. Names and operations are resolved before emit.
#[derive(Debug, Clone)]
pub struct LocalInterface {
    pub name: String,
    /// The named Rust namespace whose inherent operations are generated.
    pub rust_path: String,
    pub identity: LocalIdentity,
    pub members: Vec<LocalMember>,
    pub methods: Vec<LocalMethod>,
    pub probes: Vec<LocalProbe>,
}

#[derive(Debug, Clone)]
pub enum LocalIdentity {
    Module(String),
    Type { module: String, name: String },
    Structural(String),
}

#[derive(Debug, Clone)]
pub struct LocalMember {
    pub name: String,
    pub key: String,
    pub item: bool,
    pub optional: bool,
    pub write: bool,
}

#[derive(Debug, Clone)]
pub struct LocalMethod {
    pub name: String,
    pub params: Vec<(String, bool)>,
}

#[derive(Debug, Clone)]
pub struct LocalProbe {
    pub name: String,
    pub required: Vec<String>,
}

/// `#[host(error(..))]` on an error type: the host exception it raises, written into its family.
#[derive(Debug, Clone)]
pub struct DeclaredError {
    pub family: String,
    /// `impl crate::py_errors::HostException for <error>`, and the class `declare` creates.
    pub binding: String,
    /// `declare = ".."`: the exception class of `_core`'s own.
    pub declared: Option<String>,
    pub base: Option<String>,
    pub doc: Vec<String>,
}

/// `#[host(export)]` on a free fn or a const: a `_core` function, or a module attribute when its
/// host name is UPPER_CASE (a const only).
#[derive(Debug, Clone)]
pub struct Export {
    pub family: String,
    pub name: String,
    /// `::lk_agents_abi::telemetry::policy::MAX_DEPTH`.
    pub rust: String,
    pub doc: Vec<String>,
    /// A const's value, not a call.
    pub constant: bool,
    pub attr: bool,
    pub ty: Ty,
    /// `export(pid = false)` drops the fork guard.
    pub pid: bool,
}

/// `#[host(record(module = "..", extract, emit))]` on a struct or enum: a schema-defined record
/// crossing as a generated model (`_generated/<module>.py`), never a dict. Its pyo3 conversions
/// are generated free functions of the registration shell (`lk-agents-py/src/generated/records.rs`).
#[derive(Debug, Clone)]
pub struct RecordDecl {
    /// An identity-only Python marker class, with no value equality or dataclass machinery.
    pub marker: bool,
    /// A missing-value marker and the public singleton name it defines.
    pub missing: Option<String>,
    pub output: Option<String>,
    pub finish: Option<String>,
    pub finish_fallible: bool,
    pub dict_keys: bool,
    pub emit_classes: Option<EmitClasses>,
    pub mapping: bool,
    pub boundary: bool,
    pub default_on_null: bool,
    pub validate: Option<String>,
    pub error_owner: Option<String>,
    /// An existing host class: generate structural field conversions and an alias, not a model.
    pub host_type: Option<String>,
    /// `::lk_agents_abi::agent::ValidatedOutput`.
    pub rust: String,
    /// The host name: the model class, the union alias or the literal alias; `Owner.Name` for a
    /// model nested in its owner's (`record(.., nested_in = "Owner")`).
    pub name: String,
    /// The generated Python module (`agent_run`: `_generated/agent_run.py`).
    pub module: String,
    /// The public module its classes report as `__module__` (`record(.., path = "..")`, else the
    /// file's `file(path = "..")` for a record of the file's module): `repr`, `pickle` and the docs read it.
    pub path: Option<String>,
    /// The ABI subject: the generated Rust module it is written into.
    pub family: String,
    pub source: String,
    pub doc: Vec<String>,
    pub extract: bool,
    /// `extract(py)`: the read is `lk-agents-py`'s hand-written `py_values::<record>_from_py`
    /// (a field the model does not hold as such, like a frame's `userdata`).
    pub extract_by_hand: bool,
    pub emit: bool,
    /// `emit(py)`: the host value is `py_values::<record>_to_py`'s.
    pub emit_by_hand: bool,
    /// `record(.., transparent)`: an enum whose tuple variants cross as their values themselves
    /// (`str | Message`), told apart by the value's class; no wrapper class per variant.
    pub transparent: bool,
    /// `record(.., typed_dict)`: a struct crossing as a plain dict typed by a generated `TypedDict`
    /// (an optional field is a `NotRequired` key, absent when `None`), not a model instance.
    pub typed_dict: bool,
    /// `record(.., enum)`: a unit-only enum crossing as a `str` `Enum` class (`CloseReason.ERROR`,
    /// its `.value` the string) rather than a `Literal` of its strings.
    pub enumeration: bool,
    pub enum_base: String,
    /// `record(.., pydantic)`: a struct whose model is a mutable pydantic `BaseModel` (the public
    /// records users and plugins build, mutate and validate), not a keyword-only dataclass.
    pub pydantic: Option<Pydantic>,
    /// `record(.., dataclass)`: a struct whose model is a mutable, positional `@dataclass` (the
    /// plugin-facing records built by position and read with `dataclasses.asdict`), not a keyword-only one.
    pub dataclass: Option<Dataclass>,
    /// The Rust type derives `Hash`: an identity or value used as a key, so its default
    /// dataclass is frozen (hashable). Other records are mutable; freezing costs every import.
    pub hashable: bool,
    /// `record(.., constant(type = "stt_metrics"))`: fields the model holds at a fixed value (a
    /// discriminator its readers compare), never Rust state; read past, emitted by default.
    pub constants: Vec<(String, String)>,
    /// `persisted`: application-saved state. The model's private `_to_json()`/`_from_json()` go through
    /// the record's own serde codec in Rust (`_core._core_persist_<name>`/`_core_restore_<name>`),
    /// so every variant, empty ones included, round-trips by its tag.
    pub persisted: bool,
    /// Computed views declared with the generated models' property vocabulary.
    pub members: Vec<Member>,
    /// `record(.., transparent, dispatch = "session_dispatch")`: the host routes its members by
    /// class through `_generated/<dispatch>.py` (each variant's [`Route`]).
    pub dispatch: Option<String>,
    /// `record(.., transparent, discriminator = "type")`: the union alias is pydantic's
    /// `Annotated[.., Field(discriminator=..)]` over its model members.
    pub discriminator: Option<String>,
    pub body: RecordBody,
}

#[derive(Debug, Clone)]
pub struct EmitClasses {
    pub field: String,
    pub some: String,
    pub none: String,
}

/// A pydantic record's model class: `base = ".."` names the pydantic record of the same module it
/// subclasses (else `BaseModel`); `repr = ".."` the `_bindings` function its `__repr__` calls;
/// `validator = ".."` the `_bindings` function its after-validator calls (a field defaulted from
/// the others, computed on the host without crossing).
#[derive(Debug, Clone, Default)]
pub struct Pydantic {
    pub base: Option<String>,
    pub repr: Option<String>,
    pub validator: Option<String>,
    pub serializer: Option<String>,
    pub arbitrary_types: bool,
    /// `constants_after = "id"`: the constants follow that field instead of leading.
    pub constants_after: Option<String>,
    /// `frozen`: an immutable model (`model_config = ConfigDict(frozen=True)`).
    pub frozen: bool,
    /// `private(_cache = "dict[object, object]")`: host-only private attributes no Rust field
    /// holds, each defaulting to an empty value of its annotation's container.
    pub private: Vec<(String, String)>,
}

/// A mutable dataclass model: `post_init = ".."` names the `_bindings` function its
/// `__post_init__` calls (a host-side normalisation of what was given).
#[derive(Debug, Clone, Default)]
pub struct Dataclass {
    pub frozen: bool,
    /// Keep postponed annotations for Python callbacks whose facade is still initializing.
    pub deferred: bool,
    pub kw_only: bool,
    pub slots: bool,
    pub arbitrary_types: bool,
    pub post_init: Option<String>,
    /// `host_fields(name(ty = "bool", default = "False", doc = ".."), ..)`: fields only the host
    /// holds, trailing the declared ones; never read nor emitted. A relative dotted default
    /// names a constant in another generated module, without copying its value.
    pub host_fields: Vec<HostField>,
}

#[derive(Debug, Clone)]
pub struct HostField {
    pub name: String,
    pub annotation: String,
    pub default: String,
    pub doc: Option<String>,
}

#[derive(Debug, Clone)]
pub enum RecordBody {
    /// A model class of these fields.
    Struct(Vec<Field>),
    /// A data-variant enum: one model class per variant, the class is the tag.
    Variants(Vec<Variant>),
    /// A unit-only enum: a `Literal` of each variant's host string, by Rust variant, with its doc.
    Literals(Vec<(String, String, Vec<String>)>),
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub rust_name: String,
    /// The model class: `<Enum><Variant>` unless `name = ".."`.
    pub name: String,
    pub doc: Vec<String>,
    pub shape: VariantShape,
    /// A transparent unit variant read and emitted as an existing host marker class.
    pub host_type: Option<(String, String)>,
    /// A dispatched record's member: how the host routes its value.
    pub route: Option<Route>,
}

/// `#[host(event)]`: emitted as the public event its model's `type` names (`mirror`: the host also
/// keeps the event's `new_state` on that attribute); `#[host(ignored)]`: received, not acted on.
/// Other members have host behaviour.
#[derive(Debug, Clone, PartialEq)]
pub enum Route {
    Event { mirror: Option<String> },
    Ignored,
}

#[derive(Debug, Clone)]
pub enum VariantShape {
    Unit,
    /// `Variant(T)`: one field, named by `#[host(field = "..")]`; `None` when it is `skip`ped.
    Tuple(Option<Box<Field>>),
    Named(Vec<Field>),
    /// `#[host(native)]`: a variant only native code produces or takes; it never crosses (no
    /// model, never read, emitting it raises).
    Native,
}

/// A record field. Presence is declared: an `Option<T>` field takes the explicit null, any other
/// refuses it naming `Record.field`; `default` is what an omitted keyword gives the model, and
/// nothing else; a `skip`ped field does not cross (emit-only records).
#[derive(Debug, Clone)]
pub struct Field {
    pub kw_only: bool,
    pub hide_repr: bool,
    pub conversion: FieldConversion,
    pub rust_name: String,
    pub ty: Ty,
    pub default: Option<Literal>,
    pub skip: bool,
    /// `omitted = path`: when the host leaves it out, `path(&record)` computes it from the rest
    /// (`STTMetrics.total_tokens`: input plus output). The function's Rust path.
    pub omitted: Option<String>,
    /// `default_factory = "time.time"`: the model's default, the host callable called at
    /// construction (a host timestamp). With `skip`, Rust never sends it: the host stamps it.
    pub factory: Option<String>,
    /// `host_type = "module.Name"`: the class the host holds, when the type's own model is not it
    /// (a legacy model beside its handwritten public class).
    pub host_type: Option<(String, String)>,
    /// `private`: a pydantic `PrivateAttr` named `_<field>`, read by attribute and set after the
    /// model is built; `None` leaves its default.
    pub private: bool,
    /// `annotation = ".."`: the model's annotation, when the type's own is not the one the host
    /// declares (one side-table key standing for `list[rtc.AudioFrame]`).
    pub annotation: Option<String>,
    /// `rename = "type"`: the Python attribute, when the Rust name cannot be it.
    pub rename: Option<String>,
    pub doc: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FieldSource {
    pub attrs: Vec<String>,
    pub constant: String,
}

/// Boundary conversion behavior, declared beside the native field. A codec handles one value;
/// it never chooses a Python member name. Structural access belongs to the generated reader.
#[derive(Debug, Clone, Default)]
pub struct FieldConversion {
    pub getter: bool,
    pub key_error: bool,
    pub source: Option<FieldSource>,
    pub hidden: bool,
    pub omit_empty: bool,
    pub when: Option<(String, String)>,
    pub none_default: bool,
    pub emit_if_none: Option<String>,
    pub late: bool,
    pub normalize: Option<String>,
    pub fallible: bool,
    pub extract_with: Option<String>,
    pub emit_with: Option<String>,
    pub lenient: bool,
    pub missing_default: bool,
    pub flatten: bool,
    pub own_value: bool,
    pub omit_none: bool,
}

impl Field {
    /// In the model and read from it: crossing, or stamped by the host (`skip` with a factory).
    pub fn modeled(&self) -> bool {
        (!self.skip || self.factory.is_some()) && !self.conversion.hidden
    }

    /// The Python attribute: its `rename`, else the Rust name without a raw identifier's `r#`
    /// (`r#type` is `type`).
    pub fn host_name(&self) -> std::borrow::Cow<'_, str> {
        let name = self.rename.as_deref().unwrap_or(self.rust_name.trim_start_matches("r#"));
        match self.private {
            true => format!("_{name}").into(),
            false => name.into(),
        }
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
    /// `base = "tts.py"`: the host class is a plugin base class, hand-written in that file beside
    /// the declaring one (see `crate::bases`).
    pub base: Option<String>,
    /// `generic`: the base class is an `EventEmitter` generic over the plugin's own events; an
    /// annotation names it `Class[Any]`.
    pub generic: bool,
    /// `#[async_trait]` on the trait: the implementation carries it too.
    pub async_trait: bool,
    /// The trait declares `Any`: a proxy of it passed to a host call is handed back as the host
    /// object it was taken from.
    pub handed_back: bool,
    pub methods: Vec<ProxyMethod>,
    /// A synchronous factory returns a capability that needs a captured host context.
    pub returned_context: bool,
}

impl RecordDecl {
    /// The fields an omitted keyword computes, each with its `_core` function.
    pub fn omitted(&self) -> Vec<(&Field, String)> {
        let RecordBody::Struct(fields) = &self.body else { return Vec::new() };
        fields
            .iter()
            .filter(|f| f.omitted.is_some())
            .map(|f| {
                let record = resolve::snake(self.rust.rsplit("::").next().unwrap_or(&self.rust));
                (f, format!("_core_omitted_{record}_{}", f.host_name()))
            })
            .collect()
    }

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
        self.returned_context
            || self.methods.iter().any(|m| {
                // A trace context runs the call in the context the proxy captured.
                m.context.is_none_or(|i| m.params[i].ty == Ty::TraceContext)
                    && match m.kind {
                        ProxyCall::Await | ProxyCall::Notify => true,
                        // A host stream it returns is pulled in a context.
                        ProxyCall::Call => m.ret.ok.holds_host_stream(),
                        ProxyCall::ReadOnce | ProxyCall::Bind => false,
                    }
            })
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
    /// `requires(..)`: the `lk_agents_abi::invoke::Scope` variants the call's context must hold.
    pub requires: Vec<&'static str>,
    pub ret: ProxyRet,
    /// `bind(wrap = T)`: the bound copy is handed out as `T::bound(copy, &x, clock)`, a native
    /// wrapper the core runs around it (the retry loop of a host STT).
    pub wrap: Option<String>,
    /// `self: Box<Self>`: an awaited call that consumes the proxy.
    pub consumes: bool,
    /// `read_once` returning `&T`: lends the value the proxy read.
    pub lends: bool,
    /// `notify(lossy)`: a frame of a stream, dropped while the host loop is far behind.
    pub lossy: bool,
    /// An awaited call declared `fn ..(..) -> BoxFuture<'lt, T>` (a trait without
    /// `#[async_trait]`): the future's lifetime, `'_`, or `'static` for a consuming call.
    pub boxed: Option<String>,
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
    /// A Python authoring class whose constructor retains this native implementation.
    pub wrapper: Option<Wrapper>,
    /// Functions re-exported by generated authoring modules.
    pub forwards: Vec<(String, String, String)>,
    /// Whether this class or function group checks for fork inheritance by default.
    pub pid: bool,
    /// Python identity and subclass support declared by the class contract.
    pub host_module: String,
    pub subclass: bool,
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
    /// The input halves of the duplex stream classes its methods hand out.
    pub sinks: Vec<Sink>,
    pub methods: Vec<Method>,
}

/// `#[host(sink(item = I, class = "X"))] type S: Trait`: the input half of the duplex stream
/// class `X`, whose output half is the `source(.., class = "X")` returned beside it
/// (`(Self::S, Self::Events)`). `Trait` is `StreamInput<I>`, or an ABI trait over it whose `call`
/// methods are `X`'s too (a live recognition's `update_options`).
#[derive(Debug, Clone)]
pub struct Sink {
    pub class: String,
    pub item: Ty,
    /// The trait the stream holds its input half as: `dyn <trait_path>`.
    pub trait_path: String,
    /// `ProviderError`'s declared conversion: what a refused push raises.
    pub err: ErrorDecl,
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
    /// An authoring facade's direct native delegation, generated with this signature.
    pub facade: Option<Facade>,
    pub rust_name: String,
    /// `&mut self`.
    pub mutable: bool,
    /// The host name; the Rust one unless `name = ".."`.
    pub name: String,
    pub kind: MethodKind,
    /// An `await` declared `fn(&self, ..) -> BoxFuture<'static, T>`: called synchronously at the
    /// host's call, its future awaited after.
    pub eager: bool,
    /// An eager await declared `Result<BoxFuture<'static, T>, E>`: `E`, raised at the call.
    pub refused: Option<ErrorDecl>,
    /// A getter that provably takes no lock runs with the GIL held; every other synchronous
    /// entry releases it for the native call, except a mutable class's methods.
    pub attached: bool,
    /// Override the class or function group fork guard for this entry.
    pub pid: Option<bool>,
    pub params: Vec<Param>,
    pub ret: Ret,
    /// A host-owned future returned as HostRef, with its declared completion type.
    pub future: Option<Ty>,
    pub doc: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Facade {
    pub module: String,
    pub class: String,
    pub target: String,
    pub future: bool,
    pub discard: bool,
    pub omit: Option<String>,
}

/// How the host calls a method: declared, then checked against the signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodKind {
    /// The constructor: returns `Self` or `Result<Self, E>`.
    New,
    /// A synchronous call on `&self` (or `&mut self`: [`Form::MutableClass`]).
    Call,
    /// A synchronous class call with no receiver.
    StaticCall,
    /// A module function: no receiver, synchronous.
    Function,
    /// An async module function with no receiver.
    AsyncFunction,
    /// An `async fn` on `&self`, awaited by the host.
    Await,
    /// A read-only property on `&self`.
    Getter,
    /// A writable property on `&self`, taking its new value.
    Setter,
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
    /// Only the outer declared container shape, never nested value conversion failures.
    pub shape_error: Option<crate::shape_error::ShapeError>,
    /// `#[host(one_or_many)]` on a list: the host passes one item or a `list` of them.
    pub one_or_many: bool,
    /// `#[host(pcm)]` on bytes: a 16-bit PCM view is read as its bytes, a partial sample included.
    pub pcm: bool,
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
    /// `std::time::SystemTime`: a wall clock, as the host's float of Unix seconds.
    Timestamp,
    Str,
    /// `serde_json::Value`: any JSON value.
    Json,
    /// `serde_json::Map<String, serde_json::Value>`: a JSON object.
    JsonObject,
    /// `Option<T>`: the value or the explicit null.
    Optional(Box<Ty>),
    /// Input omission is distinct from a given value, including an explicit null.
    Maybe(Box<Ty>),
    List(Box<Ty>),
    /// `BTreeMap<K, V>`: a dict, its keys converted by their declared kind.
    Map(Box<Ty>, Box<Ty>),
    /// `indexmap::IndexMap<K, V>`: a typed dict retaining host insertion order.
    OrderedMap(Box<Ty>, Box<Ty>),
    /// `BTreeSet<T>`: a set (a `frozenset` on a model).
    Set(Box<Ty>),
    /// `Box<T>` of a record field: crosses as `T`.
    Boxed(Box<Ty>),
    Tuple(Vec<Ty>),
    /// A declared record or codec, crossing through its resolved conversion.
    Record(Record),
    /// `lk_agents_abi::invoke::CallContext`, a `#[host(context)]` parameter: what a call runs in.
    /// Never crosses: a class method's binding captures it at the call's entry (hidden from the
    /// host's signature); a proxy call runs in it and does not pass it.
    CallContext,
    /// `lk_agents_abi::telemetry::record::TraceContext`, a proxy's `#[host(context)]` parameter:
    /// the call runs in the context the proxy captured, with the trace's span current. Never
    /// crosses.
    TraceContext,
    /// An awaited proxy operation's native cancellation, never passed to the host callable.
    Cancellation,
    /// `Self::Env`, a `#[host(env)]` associated type: what the hosting process lends the class
    /// (`lk_agents_net::env::ProcessEnv`). Never crosses: the binding passes the runtime's
    /// (`crate::process::env()`), hidden from the host's signature.
    Env,
    /// The declaring class itself (`Self`).
    This,
    /// `Arc<dyn Trait>` of a `#[host(proxy)]` trait: a native class as itself, any other host
    /// object behind the generated proxy.
    Proxy {
        rust: String,
        host: String,
        /// `proxy(generic)`: a plugin base class generic over its event names, so any
        /// instantiation is accepted.
        generic: bool,
        /// A declaration-model proxy: `(family, trait)`, generated as `<family>::<trait>Proxy`.
        /// `None`: the trait's in-place `HostProxy`.
        declared: Option<(String, String)>,
        /// `Box<dyn Trait>`: an object the host hands over for the core to own (and consume, as a
        /// writer's `self: Box<Self>` close does); always behind the declared proxy.
        boxed: bool,
    },
    /// `Arc<dyn Trait>` of a `#[host(native)]` trait: an instance of one of the registered host
    /// `classes` (dotted) implementing it, as itself; any other host object is refused with
    /// `refused`. No host proxy exists. Input only.
    Native {
        rust: String,
        classes: Vec<String>,
        refused: String,
    },
    /// A pull source of `item`, handed to the host as the stream class `class`.
    Source {
        item: Box<Ty>,
        class: String,
    },
    /// The input half of the duplex stream class `class` ([`Sink`]); returned only beside its
    /// source, as [`Ty::Duplex`].
    Sink {
        item: Box<Ty>,
        class: String,
    },
    /// `(Self::Sink, Self::Source)` of one stream class: handed to the host as that class, input
    /// pushed synchronously and output pulled, under one lifetime. Output only.
    Duplex {
        input: Box<Ty>,
        output: Box<Ty>,
        class: String,
    },
    /// `Box<dyn StreamOutput<T>>` a host call returns: the host's async iterable of `T`, pulled
    /// in its own context (`crate::py_streams::HostStream`). Input only.
    HostSource {
        item: Box<Ty>,
        provider: bool,
    },
    /// `(Box<dyn StreamInput<I>>, Box<dyn StreamOutput<T>>)` a host call returns: the host's duplex
    /// stream, input through its `_core_push`/`_core_flush`/`_core_end_input` and output by
    /// iterating it, in one context (`HostStream::duplex`). Input only.
    HostDuplex {
        input: Box<Ty>,
        output: Box<Ty>,
        provider: bool,
    },
    /// A declared error (`#[host(error(..))]`) passed to the host as the exception it
    /// raises as. Output only.
    Error(ErrorDecl),
    /// `Self::<assoc>` declared `class(name = "X")`: an instance of the declared class `X` (its
    /// implementation is the `Implementations` type the generated bound ties `assoc` to). Output.
    Instance {
        assoc: String,
        class: String,
        family: String,
        shared: bool,
    },
    /// `Arc<T>` of an ABI type implementing a declared class: the `_core` object itself, shared
    /// both ways (`<family>::<class>` holds the `Arc`). `value`: a plain `T`, the host holding a
    /// handle over a clone and handing a clone back.
    Shared {
        rust: String,
        class: String,
        family: String,
        value: bool,
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
    /// Pure `#[host(wire_strs)]`: output through its existing `as_str`, no host input converter.
    WireStrs,
    /// `#[host(string)]` / `#[host(wire_enum)]`: a `str` both ways, by its `Display` and `FromStr`
    /// (`crate::py_values::str_to_py` / `str_from_py`).
    Str,
    /// `#[host(codec(.., host = ".."))]`: the annotation the declaration gives.
    Annotation(String),
    /// `codec(.., py = "module::converter")`: the annotation and declared binding primitive path.
    Converted(String, String),
    /// Checked conversion through an explicitly declared data representation (`TryFrom` / `Into`).
    Via { host: String, via: Box<Ty>, attribute: Option<String> },
    /// A `#[host(record)]`: instances of the generated model (`_generated/<module>.py`), both ways.
    Record { module: String, name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int {
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    I16,
    I32,
    I64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// `::lk_agents_abi::llm::chat_context::ChatItem`.
    pub rust: String,
    pub host: HostForm,
    /// A unit-only record enum: it crosses as its string (or `Enum` member), not as a
    /// dict of fields.
    pub literal: bool,
    /// Declared directions: the host may pass it in (`extract`) / receive it (`emit`).
    pub extract: bool,
    pub emit: bool,
}

impl Ty {
    /// A host stream (`HostSource`, `HostDuplex`) somewhere in it: pulled in a host context.
    pub fn holds_proxy(&self) -> bool {
        match self {
            Ty::Proxy { boxed: false, declared: Some(_), .. } => true,
            Ty::Optional(inner) | Ty::List(inner) => inner.holds_proxy(),
            Ty::Tuple(items) => items.iter().any(Ty::holds_proxy),
            Ty::Map(key, value) => key.holds_proxy() || value.holds_proxy(),
            _ => false,
        }
    }

    pub fn holds_host_stream(&self) -> bool {
        match self {
            Ty::HostSource { .. } | Ty::HostDuplex { .. } => true,
            Ty::Optional(t) | Ty::List(t) => t.holds_host_stream(),
            Ty::Tuple(ts) => ts.iter().any(Ty::holds_host_stream),
            _ => false,
        }
    }

    /// A parameter the binding supplies itself, absent from the host's signature.
    pub fn hidden(&self) -> bool {
        matches!(self, Ty::CallContext | Ty::TraceContext | Ty::Cancellation | Ty::Env)
    }

    /// The Rust type, written from outside the ABI crate.
    pub fn rust(&self) -> String {
        match self {
            Ty::Unit => "()".into(),
            Ty::CallContext => "::lk_agents_abi::invoke::CallContext".into(),
            Ty::TraceContext => "::lk_agents_abi::telemetry::record::TraceContext".into(),
            Ty::Cancellation => {
                "::std::sync::Arc<dyn ::lk_agents_abi::invoke::Cancellation>".into()
            }
            Ty::Env => "::lk_agents_runtime::net::env::ProcessEnv".into(),
            Ty::Bool => "bool".into(),
            Ty::Int(i) => i.rust().into(),
            Ty::Float => "f64".into(),
            Ty::Duration => "::std::time::Duration".into(),
            Ty::Timestamp => "::std::time::SystemTime".into(),
            Ty::Str => "::std::string::String".into(),
            Ty::Json => "::serde_json::Value".into(),
            Ty::JsonObject => {
                "::serde_json::Map<::std::string::String, ::serde_json::Value>".into()
            }
            Ty::Optional(t) => format!("::std::option::Option<{}>", t.rust()),
            Ty::Maybe(t) => format!("::lk_agents_abi::Maybe<{}>", t.rust()),
            Ty::List(t) => format!("::std::vec::Vec<{}>", t.rust()),
            Ty::Map(k, v) => format!("::std::collections::BTreeMap<{}, {}>", k.rust(), v.rust()),
            Ty::OrderedMap(k, v) => format!("::indexmap::IndexMap<{}, {}>", k.rust(), v.rust()),
            Ty::Set(t) => format!("::std::collections::BTreeSet<{}>", t.rust()),
            Ty::Boxed(t) => format!("::std::boxed::Box<{}>", t.rust()),
            Ty::Tuple(ts) if ts.len() == 1 => format!("({},)", ts[0].rust()),
            Ty::Tuple(ts) => {
                format!("({})", ts.iter().map(Ty::rust).collect::<Vec<_>>().join(", "))
            }
            Ty::Record(r) => r.rust.clone(),
            Ty::Shared { rust, value: true, .. } => rust.clone(),
            Ty::Shared { rust, .. } => format!("::std::sync::Arc<{rust}>"),
            Ty::This => "Self".into(),
            Ty::Proxy { rust, boxed: true, .. } => format!("::std::boxed::Box<dyn {rust}>"),
            Ty::Proxy { rust, .. } | Ty::Native { rust, .. } => {
                format!("::std::sync::Arc<dyn {rust}>")
            }
            Ty::Handle { assoc, .. } | Ty::Instance { assoc, shared: false, .. } => {
                format!("Self::{assoc}")
            }
            Ty::Instance { assoc, shared: true, .. } => format!("::std::sync::Arc<Self::{assoc}>"),
            Ty::HostSource { item, provider } => format!(
                "::std::boxed::Box<dyn ::lk_agents_abi::{}<{}>>",
                if *provider { "ProviderStream" } else { "StreamOutput" },
                item.rust()
            ),
            Ty::Source { item, .. } => {
                format!("::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<{}>>", item.rust())
            }
            Ty::Sink { item, .. } => {
                format!("::std::boxed::Box<dyn ::lk_agents_abi::StreamInput<{}>>", item.rust())
            }
            Ty::Error(e) => e.rust.clone(),
            Ty::HostDuplex { input, output, provider } => format!(
                "::lk_agents_abi::DuplexStream<::std::boxed::Box<dyn ::lk_agents_abi::StreamInput<{}>>, ::std::boxed::Box<dyn ::lk_agents_abi::{}<{}>>>",
                input.rust(),
                if *provider { "ProviderStream" } else { "StreamOutput" },
                output.rust()
            ),
            Ty::Duplex { input, output, .. } => format!(
                "::lk_agents_abi::DuplexStream<::std::boxed::Box<dyn ::lk_agents_abi::StreamInput<{}>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<{}>>>",
                input.rust(),
                output.rust()
            ),
        }
    }
}

impl Int {
    pub fn rust(self) -> &'static str {
        match self {
            Int::Isize => "isize",
            Int::U8 => "u8",
            Int::U16 => "u16",
            Int::U32 => "u32",
            Int::U64 => "u64",
            Int::U128 => "u128",
            Int::Usize => "usize",
            Int::I16 => "i16",
            Int::I32 => "i32",
            Int::I64 => "i64",
        }
    }
}
