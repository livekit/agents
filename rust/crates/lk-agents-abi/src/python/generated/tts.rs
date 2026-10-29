//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `tts` proxy bridges.

// Declared in rust/crates/lk-agents-abi/src/tts/tts.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::Tts {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::Tts>(obj)
    }
}
