//! Shared toolset domain: lifecycle owners in [`machine`], requested runtime work in [`effects`].

pub mod effects;
pub mod machine;

pub use effects::ToolsetEffect;
pub use machine::{Group, Lifetime, Scopes};
