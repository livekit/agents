//! The process runtime as `_core` exposes it: `init(role)`. Everything else `_core` holds is
//! declared beside its Rust item, in the core crate that owns it, and registers itself on the
//! table [`register_all`] drains; the generic bridge -- the fork guard, the host handle, the side
//! tables, host-error capture -- is `lk_agents_abi::python`, re-exported here whole.
//!
//! Importing the module starts nothing: every entry point goes through [`check_pid`].

use pyo3::prelude::*;

// The fork guard and the registration table.
pub use lk_agents_abi::python::*;

// ============================================================================================
// Process roles and fork safety (§1.13).
//
// Importing `livekit.agents._core` must start nothing: everything starts in `init(role)`. The
// forkserver imports the plugin set and then forks, and `fork()` clones only the calling thread, so
// a runtime started at import would hang the child on its first `block_on`.
//
// The host need not call `init`: a process that never declares a role is a `Job` (the reference
// has no initialisation call; see DECISIONS-FOR-REVIEW N402).
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
    /// The `_InferenceRunner`s.
    Inference,
}

// The pid a process declared, and the guard every entry point calls, live in
// `lk_agents_abi::python` since in-place bindings (2026-09-23): a core crate's expanded items call
// `check_pid` too, and there must be one declaration per process. The role is not kept: nothing in
// the core branches on it yet, and a process that never declared one is a `Job`.

/// Start the core for this process. Idempotent, and re-declares across a fork.
///
/// Overwriting is safe because `init` starts nothing; what must not survive a fork (threads,
/// mutexes) is caught by `check_pid` on a state `init` has not replaced.
#[pyfunction]
pub fn init(role: Role) -> PyResult<()> {
    let _ = role;
    // Routes Rust `log` lines to the Python logger of the same name; a forked child inherits it.
    let _ = pyo3_log::try_init();
    lk_agents_abi::python::declare()
}

/// The CPU count the worker sizes its defaults by: the cgroup quota when the process runs
/// under one, else the machine's (`lk_agents_worker::CpuMonitor`). Reads, starts nothing.
#[pyfunction]
pub fn cpu_count() -> f32 {
    lk_agents_worker::CpuMonitor::new().cpu_count()
}

lk_agents_abi::register_host!(class Role);
lk_agents_abi::register_host!(fn init);
lk_agents_abi::register_host!(fn cpu_count);

#[cfg(test)]
mod tests {
    use super::*;

    // The fork guard's own tests (never-declared is usable; a forked child is refused until it
    // re-declares) moved with `STATE` to `lk_agents_abi::python`.

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
            register_all(&m).unwrap();
            let mut names: Vec<String> = m
                .dir()
                .unwrap()
                .iter()
                .map(|n| n.extract::<String>().unwrap())
                .filter(|n| !n.starts_with("__"))
                .collect();
            names.sort();
            let mut expected = frozen_names();
            // `SessionRecorder` is `ffmpeg`-gated (in `media.rs`); the workspace test build lacks it.
            if cfg!(not(feature = "ffmpeg")) {
                expected.retain(|n| n != "SessionRecorder");
            }
            assert_eq!(names, expected);
        });
    }
}
