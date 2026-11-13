//! `livekit.agents._core`, the PyO3 binding generated from `lk-agents-abi` declarations.
//! Importing it starts nothing: everything that starts lives behind `init(role)`, and every runtime
//! entry calls `check_pid()` first, so a fork-inherited core raises instead of deadlocking.
#![cfg_attr(
    test,
    allow(
        clippy::disallowed_methods,
        reason = "tests drive Tokio's paused clock and spawn directly; the library is linted without"
    )
)]

use pyo3::prelude::*;

pub mod generated;
pub mod py_calls;
pub mod py_errors;
pub mod py_logging;
pub mod py_objects;
pub mod py_stall;
pub mod py_streams;
pub mod py_values;

pub use lk_agents_runtime::{inference, process, worker};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Nothing here may spawn, connect, or allocate a runtime.
    register(m)
}

/// Shared module initialization for Python imports and Cargo's embedded-interpreter tests.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    generated::register(m)?;
    py_logging::install(m)
}

#[cfg(test)]
mod surface_tests {
    use super::*;

    /// Every name `dir(livekit.agents._core)` returns, minus dunders.
    fn frozen_names() -> Vec<String> {
        let file: serde_json::Value =
            serde_json::from_str(include_str!("generated/core_names.json")).unwrap();
        file["names"].as_array().unwrap().iter().map(|n| n.as_str().unwrap().to_owned()).collect()
    }

    // Catches: a class or function added to, dropped from or renamed on `_core`, a crate whose
    // registrations are not linked, or a registration that fails to build.
    #[test]
    fn the_export_table_is_the_frozen_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let m = PyModule::new(py, "_core").unwrap();
            crate::generated::register(&m).unwrap();
            let mut names: Vec<String> = m
                .dir()
                .unwrap()
                .iter()
                .map(|n| n.extract::<String>().unwrap())
                .filter(|n| !n.starts_with("__"))
                .collect();
            names.sort();
            let expected = frozen_names();
            assert_eq!(names, expected);
        });
    }
}

#[cfg(test)]
pub mod py_test_support;
