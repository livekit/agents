//! How a failure crosses to JavaScript: a JS `Error` whose `code` and `name` are the **Python
//! exception kind** the same core item raises there (`ValueError`, `TypeError`, `OverflowError`,
//! ...), so the two hosts report one failure one way.
//!
//! The generated wrappers (`generated/`, from `napi.rs` in `lk-agents-codegen`) compute in
//! `Result<_, JsError>` and convert at the edge with [`run`]. `JsError` is `Send` and holds
//! no JS value, so a core future can carry one off the JS thread; the JS `Error` is built only
//! where an `Env` is in hand ([`JsError::into_napi`]). napi's own `Status` cannot carry the
//! kind -- its `code` is a fixed enum -- so the error object is made here and handed to napi as a
//! thrown value, which napi rethrows as the same object.

use napi::bindgen_prelude::{JsObjectValue, JsValue};
use napi::{Env, Status};

/// A failure, as its Python kind and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsError {
    pub kind: std::borrow::Cow<'static, str>,
    pub message: String,
}

impl JsError {
    pub fn new(kind: impl Into<std::borrow::Cow<'static, str>>, message: impl Into<String>) -> Self {
        JsError { kind: kind.into(), message: message.into() }
    }

    /// The JS `Error` for this failure. Falls back to a plain napi error if the object cannot be
    /// made (a pending exception), which still carries the message.
    pub fn into_napi(self, env: &Env) -> napi::Error {
        match self.js(env) {
            Ok(e) => e,
            Err(_) => napi::Error::new(Status::GenericFailure, format!("{}: {}", self.kind, self.message)),
        }
    }

    fn js(&self, env: &Env) -> napi::Result<napi::Error> {
        let mut obj = env.create_error(napi::Error::new(Status::GenericFailure, self.message.clone()))?;
        obj.set_named_property("code", &*self.kind)?;
        obj.set_named_property("name", &*self.kind)?;
        Ok(napi::Error::from(obj.to_unknown()))
    }
}

/// A generated `From` conversion is a `TryFrom` whose error is `Infallible`; `?` on one inside a
/// fallible body needs this.
impl From<std::convert::Infallible> for JsError {
    fn from(e: std::convert::Infallible) -> Self {
        match e {}
    }
}

/// Run a wrapper body and turn its `JsError` into the JS one.
pub fn run<T>(env: &Env, f: impl FnOnce() -> Result<T, JsError>) -> napi::Result<T> {
    f().map_err(|e| e.into_napi(env))
}

/// `SchemaError`'s Node map: the kind the core names for it, as `lk-agents-py`'s `schema_err` does.
pub fn schema_error_to_node(e: lk_agents_abi::llm::schema::SchemaError) -> JsError {
    JsError::new(e.host_kind().to_owned(), e.to_string())
}

/// `ProviderError`'s Node map: the agents error class name for its API kind (`APIStatusError`,
/// `APIConnectionError`, `APITimeoutError`, `APIError`), the names `lk-agents-py` raises for a
/// native failure. A host-captured failure keeps the kind JS classified it with.
pub fn provider_error_to_node(e: lk_agents_abi::errors::ProviderError) -> JsError {
    use lk_agents_abi::errors::{ApiErrorKind, ProviderError};
    let kind = match (&e, e.api_kind()) {
        (_, Some(ApiErrorKind::Status)) => "APIStatusError",
        (_, Some(ApiErrorKind::Connection)) => "APIConnectionError",
        (_, Some(ApiErrorKind::Timeout)) => "APITimeoutError",
        (ProviderError::Host { .. }, None) => "Error",
        _ => "APIError",
    };
    JsError::new(kind, e.message().to_owned())
}

/// `IdleRefused`'s Node map: `wait_for_idle()` / `hold_floor()` on a closing session, the error
/// `lk-agents-py` raises as `SessionClosing`.
pub fn idle_refused_to_node(_: lk_agents_voice::agent_session::IdleRefused) -> JsError {
    JsError::new("SessionClosing", "session is closing")
}

/// `CommitSuperseded`'s Node map: a `commitUserTurn()` a newer one replaced, rejected as the
/// cancellation `lk-agents-py` raises for it.
pub fn commit_superseded_to_node(e: lk_agents_voice::agent_session::CommitSuperseded) -> JsError {
    JsError::new("CancelledError", e.to_string())
}

/// `#[host(lenient)]`: a JS value that is not a `T` reads as `T::default()`, never an error --
/// pyo3's `extract().unwrap_or_default()`.
pub fn lenient<T: napi::bindgen_prelude::FromNapiValue + Default>(v: &napi::bindgen_prelude::Unknown<'_>) -> T {
    let raw = v.value();
    unsafe { T::from_napi_value(raw.env, raw.value) }.unwrap_or_default()
}

/// `Maybe<Option<T>>` from JS: `undefined` (or an omitted argument) is `NotGiven`, `null` is
/// `Given(None)`, a value is `Given(Some(..))` -- the three answers napi's `Option` folds to two.
/// `lk-agents-py`'s `convert::MaybeNull`, for Node.
pub enum MaybeNull<T> {
    NotGiven,
    Null,
    Given(T),
}

impl<T> MaybeNull<T> {
    pub fn into_core(self) -> lk_agents_abi::Maybe<Option<T>> {
        match self {
            MaybeNull::NotGiven => lk_agents_abi::Maybe::NotGiven,
            MaybeNull::Null => lk_agents_abi::Maybe::Given(None),
            MaybeNull::Given(v) => lk_agents_abi::Maybe::Given(Some(v)),
        }
    }

    pub fn try_map<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<MaybeNull<U>, E> {
        Ok(match self {
            MaybeNull::NotGiven => MaybeNull::NotGiven,
            MaybeNull::Null => MaybeNull::Null,
            MaybeNull::Given(v) => MaybeNull::Given(f(v)?),
        })
    }
}

impl<T: napi::bindgen_prelude::FromNapiValue> napi::bindgen_prelude::FromNapiValue for MaybeNull<T> {
    unsafe fn from_napi_value(env: napi::sys::napi_env, value: napi::sys::napi_value) -> napi::Result<Self> {
        let mut ty = 0;
        unsafe { napi::sys::napi_typeof(env, value, &mut ty) };
        Ok(match ty {
            napi::sys::ValueType::napi_undefined => MaybeNull::NotGiven,
            napi::sys::ValueType::napi_null => MaybeNull::Null,
            _ => MaybeNull::Given(unsafe { T::from_napi_value(env, value)? }),
        })
    }
}

impl<T> napi::bindgen_prelude::TypeName for MaybeNull<T> {
    fn type_name() -> &'static str {
        "MaybeNull"
    }

    fn value_type() -> napi::ValueType {
        napi::ValueType::Unknown
    }
}

impl<T: napi::bindgen_prelude::FromNapiValue> napi::bindgen_prelude::ValidateNapiValue for MaybeNull<T> {}
