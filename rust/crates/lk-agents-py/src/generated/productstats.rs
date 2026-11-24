//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `productstats` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/productstats/mod.rs.
type ProductStatsHostImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ProductStatsHost;

/// A plugin registered on import: a `livekit.plugins.*` package is listed on the resource of
/// every later export; another package is not.
#[pyfunction]
#[pyo3(signature = (package, version))]
pub fn _core_product_stats_plugin(
    py: Python<'_>,
    package: ::std::string::String,
    version: ::std::string::String,
) -> PyResult<()> {
    py.detach(|| <ProductStatsHostImpl as ::lk_agents_abi::productstats::ProductStatsHost>::_core_product_stats_plugin(package, version));
    Ok(())
}
