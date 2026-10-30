//! The process runtime as `_core` exposes it: `init(role)`.
//!
//! Importing the module starts nothing: every entry point goes through the runtime's fork guard
//! (`crate::host::check_pid`).

use pyo3::prelude::*;

// ============================================================================================
// Process roles and fork safety.
//
// Importing `livekit.agents._core` must start nothing: everything starts in `init(role)`. The
// forkserver imports the plugin set and then forks, and `fork()` clones only the calling thread, so
// a runtime started at import would hang the child on its first `block_on`.
//
// The host need not call `init`: a process that never declares a role is a `Job` (the reference
// has no initialisation call).
// ============================================================================================

/// Which process this is. The host knows; Rust is told.
#[pyclass(eq, eq_int, from_py_object, module = "livekit.agents._core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The `/agent` websocket, registration, load reporting, the proc pool, signal handlers.
    Worker,
    /// One or more `AgentSession`s. **No worker connection** -- this is also the `console` and
    /// `connect` CLI path, so nothing here may assume a registration or a job assignment.
    Job,
}

// The pid a process declared, and the guard every entry point calls, are the runtime's
// (`lk_agents_runtime::process::check_pid`). The role is not kept: nothing in the core branches on
// it yet, and a process that never declared one is a `Job`.

/// Start the core for this process. Idempotent, and re-declares across a fork.
///
/// Overwriting is safe because `init` starts nothing; what must not survive a fork (threads,
/// mutexes) is caught by `check_pid` on a state `init` has not replaced.
#[pyfunction]
pub fn init(role: Role) -> PyResult<()> {
    let _ = role;
    // A forked child declaring itself abandons the parent's Tokio runtime; its own starts on use.
    // the Rust `log` lines' route to Python (installed at import) survives a fork
    lk_agents_runtime::process::reinit();
    Ok(())
}

crate::register_host!(class Role);
crate::register_host!(fn init);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name `dir(livekit.agents._core)` returns, minus dunders. The file's `about` says how
    /// to regenerate it.
    fn frozen_names() -> Vec<String> {
        let file: serde_json::Value =
            serde_json::from_str(include_str!("core_names.json")).unwrap();
        file["names"].as_array().unwrap().iter().map(|n| n.as_str().unwrap().to_owned()).collect()
    }

    // Catches: a class or function added to, dropped from or renamed on `_core`, a crate whose
    // registrations are not linked, or a registration that fails to build.
    #[test]
    fn the_export_table_is_the_frozen_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let m = PyModule::new(py, "_core").unwrap();
            crate::python::register_all(&m).unwrap();
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
