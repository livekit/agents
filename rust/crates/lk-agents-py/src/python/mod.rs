//! What the generated conversions and the hand-written classes cross the boundary with: values as
//! the host spells them, host failures as the core's errors and back, buffers, and the registration
//! table of the hand-written classes.

mod bridge;
mod buffers;
#[cfg(test)]
mod conversion_tests;
mod host_errors;
#[cfg(test)]
mod host_tests;
mod payload_table;
mod values;

pub use bridge::*;
pub use buffers::*;
pub use host_errors::*;
pub use values::*;

use pyo3::prelude::*;

/// For a Rust-native stream, whose teardown consults no clock. Host-backed streams never reach it.
pub const NATIVE_CLOSE_DEADLINE: lk_agents_abi::Ts = lk_agents_abi::Ts::from_nanos(u64::MAX);

/// Submitted by [`register_host!`](crate::register_host) for a hand-written `_core` item.
pub struct Registration(pub fn(&Bound<'_, PyModule>) -> PyResult<()>);

inventory::collect!(Registration);

/// In link order (nothing depends on it). The `#[pymodule]` body.
pub fn register_all(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for registration in inventory::iter::<Registration> {
        (registration.0)(m)?;
    }
    Ok(())
}

/// The host exception a declared error raises (`#[host(error)]`, generated).
pub trait HostException {
    fn to_py_err(&self) -> PyErr;
}

/// `register_host!(class T)`, `register_host!(fn f)`: hand-written pyo3 items on `_core`.
#[macro_export]
macro_rules! register_host {
    (class $t:ty) => {
        ::inventory::submit! {
            $crate::python::Registration(|m| {
                use ::pyo3::types::PyModuleMethods as _;
                m.add_class::<$t>()
            })
        }
    };
    (fn $f:path) => {
        ::inventory::submit! {
            $crate::python::Registration(|m| {
                use ::pyo3::types::PyModuleMethods as _;
                m.add_function(::pyo3::wrap_pyfunction!($f, m)?)
            })
        }
    };
}
