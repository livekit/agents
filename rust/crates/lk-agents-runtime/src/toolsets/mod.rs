//! Shared lifecycle execution for agent and session toolsets.
pub mod group;
pub mod lifetime;
pub mod scopes;

#[cfg(test)]
#[path = "../toolsets_tests.rs"]
mod tests;
