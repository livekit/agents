use super::{Method, Ty};

/// `#[host(facade(..))] trait X`: the public Python class over a native class, one method per
/// Python method, in order.
#[derive(Debug, Clone)]
pub struct FacadeClass {
    /// The Python class name: the trait's, or `name = ".."`.
    pub name: String,
    /// `path = "livekit.agents.x"`: the public module re-exporting it.
    pub path: String,
    /// `extended`: the base of the handwritten subclass of its name in `path`, which is the
    /// public class; it is not re-exported.
    pub extended: bool,
    /// The generated module under `_generated/` (`facades_<x>` for `livekit.agents.x`).
    pub module: String,
    /// `base = "module.Name"`, relative to `livekit.agents`.
    pub base: Option<String>,
    /// `foreign`: `base` is a fully qualified class outside `livekit.agents`.
    pub foreign: bool,
    /// `generic = "T"`: the class is `Generic[T]` over its own TypeVar `T`, which annotations
    /// name.
    pub generic: Option<String>,
    /// The native class's host name (`_core.<native>`); empty when every method goes `via` a
    /// delegate.
    pub native: String,
    /// `from_native`: the constructor takes the native object, then its `#[host(new)]`
    /// parameters, kept as `self._<param>`.
    pub from_native: bool,
    /// The provider methods whose override keeps the host object as the leg.
    pub legs: Vec<String>,
    /// The host method whose value is the leg, in place of `_native`.
    pub leg: Option<String>,
    pub availability: Option<Availability>,
    /// `base_args(a, ..)`: the base constructor's arguments, read from the native getters of the
    /// same name once the native object is built.
    pub base_args: Vec<String>,
    /// `keep(a, ..)`: constructor parameters also kept as `self._<a>` for the extending class.
    pub keep: Vec<String>,
    /// `host_args(p = "module.object")`: a native constructor parameter the host object fills.
    pub host_args: Vec<(String, String)>,
    pub stream: Option<StreamForm>,
    pub delegates: Vec<Delegate>,
    pub source: String,
    pub doc: Vec<String>,
    pub methods: Vec<FacadeMethod>,
}

/// `#[host(delegate)] type X: Trait`: an object the class holds as `self._x` and forwards to;
/// `delegate = "a.b()"`: read from that path on `self` at each call.
#[derive(Debug, Clone)]
pub struct Delegate {
    pub assoc: String,
    /// The path on `self`: `_x`, or the declared one.
    pub attribute: String,
    /// Held as `self._x`, not read from a declared path.
    pub held: bool,
    pub ty: Ty,
}

/// `stream(op = m, owner = "_tts", native_retries, resolve(..))`: a provider stream whose `_run`
/// opens the owner's native `op`, its arguments the stream's `self._<param>` attributes (passed
/// through a `resolve` function when named), and the base pumps it.
#[derive(Debug, Clone)]
pub struct StreamForm {
    pub op: String,
    pub owner: String,
    /// The native operation retries on its own: the base's attempts do not.
    pub native_retries: bool,
    pub resolve: Vec<(String, String)>,
    /// The owner's provider method taking the per-call parameters (all but `state`, read from
    /// the base stream); the stream passes `base_args` to its base and keeps the rest.
    pub method: Option<String>,
    pub base_args: Vec<String>,
    pub state: Vec<String>,
    /// A per-call parameter's host default (`module.NAME`) and annotation in the method.
    pub defaults: Vec<(String, String)>,
    pub annotations: Vec<(String, String)>,
    /// An `Option` parameter defaulting to `None` is `NotGivenOr[T] = NOT_GIVEN` in the method.
    pub not_given: bool,
    /// `rename(op_param = "method_param")`: the provider method's name for an operation parameter.
    pub rename: Vec<(String, String)>,
    /// `provider_default(p, ..)`: a parameter not given takes the owner's native getter `p`.
    pub provider_default: Vec<String>,
}

/// One method of a facade trait.
#[derive(Debug, Clone)]
pub struct FacadeMethod {
    /// The Python signature, resolved like a class method's.
    pub method: Method,
    /// The native method it calls: the same name unless `native = m`.
    pub native: String,
    /// `via = Assoc`: the delegate it forwards to, not the native object.
    pub via: Option<Delegate>,
    /// `(parameter, attribute)`: the parameters declared as a delegate (`Self::X`).
    pub delegated: Vec<(String, String)>,
    /// A native `None` raises `StopAsyncIteration`.
    pub none_ends_iteration: bool,
    /// `wrap = Record::field`: the native value becomes that record's field, `(module, name,
    /// field)`.
    pub wrap: Option<(String, String, String)>,
    /// `exit = m`: an `async with` whose entry is the native method.
    pub exit: Option<Exit>,
    pub adapt: Adapt,
}

/// A facade context manager's exit (`exit = m`, `abort = m`, `yields = "path"`). The target
/// (`self._native`, or the delegate read once at entry) enters it; the exit is idempotent
/// natively, a no-op once the target closed.
#[derive(Debug, Clone)]
pub struct Exit {
    /// The method run as the block ends (on success only when `abort` is declared).
    pub method: ExitCall,
    /// `abort = m`: run instead when the block raises, which then propagates.
    pub abort: Option<ExitCall>,
    /// `yields = "path"`: the block receives `self.<path>`, not the entered value.
    pub yields: Option<String>,
}

/// An exit method: the target's (`m`, passed the entered value when it takes one) or the
/// entered native value's own (`Class::m`).
#[derive(Debug, Clone)]
pub struct ExitCall {
    pub method: String,
    /// `Class::m`: the entered value's class.
    pub on_value: Option<String>,
}

/// The per-method adaptations of a facade forward.
#[derive(Debug, Clone, Default)]
pub struct Adapt {
    pub future: bool,
    pub discard: bool,
    pub not_given: bool,
    pub origin: Option<String>,
    pub returns: Option<String>,
    /// `each = Record::field`: with `returns`, each element of the native list as that field.
    pub each: Option<String>,
    pub annotations: Vec<(String, String)>,
    pub resolve: Vec<(String, String)>,
    /// `bind(p = "a.b")`: the native's `p` is `self.a.b`, not a Python parameter.
    pub bind: Vec<(String, String)>,
    /// `chain`: the method answers `self`.
    pub chain: bool,
    /// `setter = prop`: assigns the property `prop`, a getter declared before it.
    pub setter: Option<String>,
}

/// `availability(observer = "p", members = "m", kind = "..", event = "..", record = "module.Name")`
/// of a facade: the host object is the chain observer its native constructor takes as `p`, keeps
/// its constructor's `m` as `_members` and emits `event` as `record(entry of _members,
/// available)`; the serving entry names its `model`, `provider` and `metrics_metadata`, and every
/// entry unavailable is logged as `all <kind>s are unavailable`.
#[derive(Debug, Clone)]
pub struct Availability {
    pub observer: String,
    pub members: String,
    pub kind: String,
    pub event: String,
    pub record: String,
}
