//! Attachment ownership for sessions sharing one SDK room connection.

pub mod effects;
pub mod machine;

pub use lk_agents_abi::voice::room_binding::{BindingId, BindingKey, BindingPhase, BindingRefused};
