//! The per-process runtime the in-place pyo3 expansions need (registration table, fork guard, error
//! conversion). In ABI because every core crate depends on it and it names ABI's own types.

mod bridge;
mod host_errors;
#[cfg(test)]
mod host_tests;
mod loop_time;
mod values;

pub use bridge::*;
pub use host_errors::*;
pub use loop_time::{LoopRuntime, VIRTUAL_TIME_ENV, future_into_py};
pub use values::*;

use std::sync::RwLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Re-exported for the expansions, so a core crate needs no dependency of its own.
pub use inventory;
pub use pyo3_async_runtimes;
pub use log;
/// Awaited in `recv`, taken in `aclose`.
pub use tokio::sync::Mutex as AsyncMutex;

/// For a Rust-native stream, whose teardown consults no clock. Host-backed streams never reach it.
pub const NATIVE_CLOSE_DEADLINE: crate::Ts = crate::Ts::from_nanos(u64::MAX);

/// Submitted by `#[lk_agents_abi::export]` and every in-place `#[host(..)]` expansion.
pub struct Registration(pub fn(&Bound<'_, PyModule>) -> PyResult<()>);

inventory::collect!(Registration);

/// In link order (nothing depends on it). The `#[pymodule]` body.
pub fn register_all(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for registration in inventory::iter::<Registration> {
        (registration.0)(m)?;
    }
    Ok(())
}

/// `const` equality: asserts a plain object's `#[host(object(name))]` (what the `.pyi` uses) is its
/// `#[pyclass(name)]`.
pub const fn same_name(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

// Importing `_core` must start nothing: the forkserver imports then forks, and `fork()` clones only
// the calling thread, so a runtime started at import would hang the child on its first `block_on`.

/// Not a `OnceLock`: a forked child must be able to re-declare itself.
static STATE: RwLock<Option<u32>> = RwLock::new(None);

fn poisoned() -> PyErr {
    PyRuntimeError::new_err("livekit.agents._core's process state is poisoned")
}

fn forked_without_init() -> PyErr {
    PyRuntimeError::new_err(
        "livekit.agents._core was inherited across a fork. The Rust core cannot be used in a \
         forked child without re-initialising it: call _core.init(role) in the child.",
    )
}

/// Overwriting is safe: nothing has started.
pub fn declare() -> PyResult<()> {
    *STATE.write().map_err(|_| poisoned())? = Some(std::process::id());
    Ok(())
}

/// Turns a silent deadlock on a runtime whose threads did not survive `fork()` into an exception.
pub fn check_pid() -> PyResult<()> {
    let pid = std::process::id();
    // Never declared: self-declare (nothing has started; raising would break plain scripts the
    // reference supports). The write re-checks so an explicit `init` in between is not overwritten.
    if let Some(p) = *STATE.read().map_err(|_| poisoned())? {
        return if p == pid { Ok(()) } else { Err(forked_without_init()) };
    }
    let mut state = STATE.write().map_err(|_| poisoned())?;
    match *state {
        Some(p) if p == pid => Ok(()),
        Some(_) => Err(forked_without_init()),
        None => {
            *state = Some(pid);
            Ok(())
        }
    }
}

// Autoref specialisation: `(&Fail(&e)).py_err()` resolves to [`Declared`] when `E: HostException`
// (a `#[host(error)]` expansion), else to [`Fallback`], `ValueError(str(e))`.

/// Implemented by a `#[host(error)]` declaration's expansion.
pub trait HostException {
    fn to_py_err(&self) -> PyErr;
}

#[doc(hidden)]
pub struct Fail<'a, E>(pub &'a E);

#[doc(hidden)]
pub trait Declared {
    fn py_err(&self) -> PyErr;
}

impl<E: HostException> Declared for Fail<'_, E> {
    fn py_err(&self) -> PyErr {
        self.0.to_py_err()
    }
}

#[doc(hidden)]
pub trait Fallback {
    fn py_err(&self) -> PyErr;
}

impl<E: std::fmt::Display> Fallback for &Fail<'_, E> {
    fn py_err(&self) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(self.0.to_string())
    }
}

#[cfg(test)]
mod tests {
    //! One process per test (nextest): `STATE` is a process global.
    use super::*;

    /// `check_pid`'s refusal is a `PyErr`, so the interpreter has to exist.
    fn with_py(f: impl FnOnce(Python<'_>)) {
        Python::initialize();
        Python::attach(f)
    }

    // N402: the reference needs no initialiser, so first use must not raise.
    #[test]
    fn a_process_that_never_declared_itself_is_usable() {
        with_py(|_| {
            assert_eq!(*STATE.read().unwrap(), None, "nothing should be declared yet");
            check_pid().expect("first use must self-declare rather than raise");
            assert_eq!(*STATE.read().unwrap(), Some(std::process::id()));
        });
    }

    // The failure the guard exists for, and its remedy: with `STATE` as a `OnceLock`, `init`
    // returned the error whose remedy is this call.
    #[test]
    fn a_forked_child_is_refused_until_it_re_declares_itself() {
        with_py(|_| {
            // What a forked child inherits: a declaration from a *different* process.
            *STATE.write().unwrap() = Some(std::process::id() ^ 0x5555);
            let err = check_pid().expect_err("a stale declaration must be refused");
            assert!(format!("{err}").contains("inherited across a fork"), "{err}");
            declare().expect("the child must be able to re-declare itself");
            declare().expect("idempotently");
            check_pid().expect("and be usable afterwards");
        });
    }

    struct Refused(&'static str);
    impl HostException for Refused {
        fn to_py_err(&self) -> PyErr {
            pyo3::exceptions::PyKeyError::new_err(self.0)
        }
    }
    impl std::fmt::Display for Refused {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("the fallback must not be chosen")
        }
    }

    // Catches: the dispatch picking the fallback for a declared error (it is `Display` too, so both
    // arms apply and only the autoref order decides), or the fallback raising anything but a
    // `ValueError` with the error's text -- the codegen's default the expansions must keep.
    #[test]
    fn a_declared_error_raises_its_own_class_and_any_other_a_value_error() {
        use super::{Declared as _, Fallback as _};
        with_py(|py| {
            let declared = (&Fail(&Refused("k"))).py_err();
            assert!(declared.is_instance_of::<pyo3::exceptions::PyKeyError>(py), "{declared}");
            let plain = (&Fail(&"bad input")).py_err();
            assert!(plain.is_instance_of::<pyo3::exceptions::PyValueError>(py), "{plain}");
            assert_eq!(plain.value(py).to_string(), "bad input");
        });
    }
}

// For a hand-written type that keeps its own `as_str`/`FromStr`.

macro_rules! into_host_str {
    ($t:ty, $to:expr) => {
        impl<'py> ::pyo3::IntoPyObject<'py> for $t {
            type Target = ::pyo3::types::PyString;
            type Output = ::pyo3::Bound<'py, ::pyo3::types::PyString>;
            type Error = ::std::convert::Infallible;
            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> Result<::pyo3::Bound<'py, ::pyo3::types::PyString>, ::std::convert::Infallible> {
                (&self).into_pyobject(py)
            }
        }
        impl<'py> ::pyo3::IntoPyObject<'py> for &$t {
            type Target = ::pyo3::types::PyString;
            type Output = ::pyo3::Bound<'py, ::pyo3::types::PyString>;
            type Error = ::std::convert::Infallible;
            fn into_pyobject(self, py: ::pyo3::Python<'py>) -> Result<::pyo3::Bound<'py, ::pyo3::types::PyString>, ::std::convert::Infallible> {
                let to: fn(&$t) -> &str = $to;
                Ok(::pyo3::types::PyString::new(py, to(self)))
            }
        }
    };
}
pub(crate) use into_host_str;

/// Parse failure is a `ValueError` carrying the error; `HostStr` names the argument in it.
macro_rules! from_host_str {
    ($t:ty) => {
        impl crate::python::HostStr for $t {}
        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for $t {
            type Error = ::pyo3::PyErr;
            fn extract(ob: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                let s = ob.extract::<::pyo3::pybacked::PyBackedStr>()?;
                <$t as ::std::str::FromStr>::from_str(&s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(e.to_string()))
            }
        }
    };
}
pub(crate) use from_host_str;

/// `register_host!(class T)`, `register_host!(fn f)`, `register_host!(exception E)`: hand-written
/// pyo3 items on `_core`, behind the caller's `python` feature.
#[macro_export]
macro_rules! register_host {
    (class $t:ty) => {
        $crate::python::inventory::submit! {
            $crate::python::Registration(|m| {
                use ::pyo3::types::PyModuleMethods as _;
                m.add_class::<$t>()
            })
        }
    };
    (fn $f:path) => {
        $crate::python::inventory::submit! {
            $crate::python::Registration(|m| {
                use ::pyo3::types::PyModuleMethods as _;
                m.add_function(::pyo3::wrap_pyfunction!($f, m)?)
            })
        }
    };
    (exception $t:ident) => {
        $crate::python::inventory::submit! {
            $crate::python::Registration(|m| {
                use ::pyo3::types::PyModuleMethods as _;
                m.add(stringify!($t), m.py().get_type::<$t>())
            })
        }
    };
}
