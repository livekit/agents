//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `worker` proxy bridges and in-place records.

// Declared in rust/crates/lk-agents-abi/src/worker/job.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::worker::job::SessionJob {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::worker::job::SessionJob>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/worker/remote_session.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::worker::remote_session::SessionPort {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::worker::remote_session::SessionPort>(obj)
    }
}
