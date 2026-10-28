//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The declared classes' pyo3 adapters, one module per ABI subject.

use pyo3::prelude::*;

pub mod agent;
pub mod llm;

/// The implementation of each declared class. The registration shell implements it.
pub trait Implementations: 'static {
    type InferenceLlm: ::lk_agents_abi::llm::inference::InferenceLlm;
    type ChoiceParser: ::lk_agents_abi::llm::inference::ChoiceParser;
    type CompletionWire: ::lk_agents_abi::llm::inference::CompletionWire;
}

/// Adds every declared class to `_core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<llm::InferenceLlm>()?;
    m.add_class::<llm::InferenceLlmChunks>()?;
    m.add_class::<llm::LlmChoiceParser>()?;
    m.add_function(wrap_pyfunction!(llm::llm_usage_chunk, m)?)?;
    Ok(())
}
