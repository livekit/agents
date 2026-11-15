//! Native attachment authority. A declaration owns no SDK resources; an activation
//! owns its resource aggregate through `Owned`, including after caller cancellation.

mod owner;
mod resource;

pub use owner::{BindingOwner, DeclaredBinding};
pub use resource::ActiveBinding;

#[cfg(test)]
mod binding_tests;
