//! The Node `HostBridge`: how a generated `#[host(proxy)]` impl (`generated/proxies.rs`, from
//! `napi.rs` in `lk-agents-codegen`) calls the JS object behind it.
//!
//! `lk-agents-py`'s bridge calls a method by name on a Python object from any thread, because the
//! GIL makes that legal. A JS object may only be touched on its own thread, so here the host
//! hands over **one dispatcher function per object** and every crossing goes through it:
//!
//! ```ts
//! (op: 'soon' | 'call' | 'anext', name: string, ...args: any[]) => unknown
//! // e.g. (op, name, ...args) => op === 'anext'
//! //        ? it.next().then((r) => (r.done ? null : r.value))
//! //        : obj[name](...args)
//! ```
//!
//! **One function, so one queue.** napi keeps a FIFO per threadsafe function and none between
//! them; a transport whose `send_message` and `close` rode two functions could close before the
//! send it followed (`providers/host.rs` found the same hazard the hard way). Python's twin is the
//! loop's single ready queue.
//!
//! A `call` whose JS function returns a Promise is awaited on the Rust side; a plain value is
//! taken as is. A JS `throw` (or a rejection) arrives as `Err`, and [`host_err`] turns it into the
//! trait's error type -- the one place that happens, as `lk-agents-py`'s `runtime::host_err` is.

use std::sync::Arc;

use napi::bindgen_prelude::{FromNapiValue, JsValue, JsValuesTupleIntoVec, Promise, ToNapiValue, Unknown};
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{Env, Status, sys};

/// One argument, converted to JS on the JS thread.
pub struct JsArg(Box<dyn FnOnce(sys::napi_env) -> napi::Result<sys::napi_value> + Send>);

impl JsArg {
    pub fn new<T: ToNapiValue + Send + 'static>(value: T) -> Self {
        JsArg(Box::new(move |env| unsafe { T::to_napi_value(env, value) }))
    }
}

/// One crossing: spread into the dispatcher as `(op, name, ...args)`.
pub struct JsCall {
    op: &'static str,
    name: &'static str,
    args: Vec<JsArg>,
}

impl JsValuesTupleIntoVec for JsCall {
    fn into_vec(self, env: sys::napi_env) -> napi::Result<Vec<sys::napi_value>> {
        let mut out = Vec::with_capacity(self.args.len() + 2);
        out.push(unsafe { <&str as ToNapiValue>::to_napi_value(env, self.op)? });
        out.push(unsafe { <&str as ToNapiValue>::to_napi_value(env, self.name)? });
        for a in self.args {
            out.push((a.0)(env)?);
        }
        Ok(out)
    }
}

/// The dispatcher, as napi holds it. `CalleeHandled = false`: it is an ordinary function, not a
/// node-style `(err, ..)` callback.
pub type DispatchFn = ThreadsafeFunction<JsCall, Unknown<'static>, JsCall, Status, false>;

/// A host object, as its dispatcher. `Clone` shares the one function (and so the one queue).
#[derive(Clone)]
pub struct NodeBridge {
    dispatch: Arc<DispatchFn>,
}

/// A JS return value nobody reads: anything is accepted.
pub struct Ignored;

impl FromNapiValue for Ignored {
    unsafe fn from_napi_value(_env: sys::napi_env, _value: sys::napi_value) -> napi::Result<Self> {
        Ok(Ignored)
    }
}

enum Reply<R: FromNapiValue + 'static> {
    Done(napi::Result<R>),
    Later(Promise<R>),
}

impl NodeBridge {
    pub fn new(dispatch: DispatchFn) -> Self {
        NodeBridge { dispatch: Arc::new(dispatch) }
    }

    /// Fire and forget, in FIFO order behind everything already queued on this object. `Ok` means
    /// handed over, not accepted -- `call_soon_threadsafe`'s contract.
    pub fn soon(&self, name: &'static str, args: Vec<JsArg>) -> napi::Result<()> {
        match self.dispatch.call(JsCall { op: "soon", name, args }, ThreadsafeFunctionCallMode::NonBlocking) {
            Status::Ok => Ok(()),
            status => Err(napi::Error::new(status, format!("`{name}` could not be handed to the host"))),
        }
    }

    /// Call `name`, awaiting the Promise it returns (if it returns one), and read the result as `R`.
    pub async fn call<R: FromNapiValue + Send + 'static>(&self, name: &'static str, args: Vec<JsArg>) -> napi::Result<R> {
        self.ask("call", name, args).await
    }

    /// The next item of the object's async iterator; `None` at its end.
    pub async fn anext<R: FromNapiValue + Send + 'static>(&self) -> napi::Result<Option<R>> {
        self.ask("anext", "", Vec::new()).await
    }

    async fn ask<R: FromNapiValue + Send + 'static>(&self, op: &'static str, name: &'static str, args: Vec<JsArg>) -> napi::Result<R> {
        let (tx, rx) = tokio::sync::oneshot::channel::<Reply<R>>();
        let status = self.dispatch.call_with_return_value(
            JsCall { op, name, args },
            ThreadsafeFunctionCallMode::NonBlocking,
            move |ret: napi::Result<Unknown<'static>>, env: Env| {
                let reply = match ret {
                    Err(e) => Reply::Done(Err(e)),
                    Ok(v) => unsafe {
                        let (env, raw) = (env.raw(), v.raw());
                        let mut is_promise = false;
                        sys::napi_is_promise(env, raw, &mut is_promise);
                        if is_promise {
                            match Promise::<R>::from_napi_value(env, raw) {
                                Ok(p) => Reply::Later(p),
                                Err(e) => Reply::Done(Err(e)),
                            }
                        } else {
                            Reply::Done(R::from_napi_value(env, raw))
                        }
                    },
                };
                let _ = tx.send(reply);
                Ok(())
            },
        );
        if status != Status::Ok {
            return Err(napi::Error::new(status, format!("`{name}` could not be handed to the host")));
        }
        match rx.await {
            Ok(Reply::Done(r)) => r,
            Ok(Reply::Later(p)) => p.await,
            Err(_) => Err(napi::Error::new(Status::Closing, "the host environment stopped responding")),
        }
    }
}

/// A JS failure, as the error a proxied trait method returns.
pub trait FromJsErr {
    fn from_js(err: napi::Error) -> Self;
}

fn host_error(err: &napi::Error) -> lk_agents_abi::errors::HostError {
    // The JS error object stays in JS (it is thread-affine); its message crosses.
    lk_agents_abi::errors::HostError::new(err.reason.clone())
}

impl FromJsErr for lk_agents_abi::errors::HostError {
    fn from_js(err: napi::Error) -> Self {
        host_error(&err)
    }
}

impl FromJsErr for lk_agents_voice::io::IoError {
    fn from_js(err: napi::Error) -> Self {
        lk_agents_voice::io::IoError::Host(host_error(&err))
    }
}

impl FromJsErr for lk_agents_abi::errors::ProviderError {
    fn from_js(err: napi::Error) -> Self {
        lk_agents_abi::errors::ProviderError::Host { error: host_error(&err), api_kind: None, retryable: false, status_code: None }
    }
}

/// `napi::Error` -> the caller's error type. The generated proxies' `map_err`.
pub fn host_err<E: FromJsErr>(err: napi::Error) -> E {
    E::from_js(err)
}

/// A value JS returned that does not convert, as a transport-level error.
pub fn conv_err(e: crate::host_error::JsError) -> napi::Error {
    napi::Error::new(Status::InvalidArg, format!("{}: {}", e.kind, e.message))
}
