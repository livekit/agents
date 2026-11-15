//! Decisions shared by every binding of one room connection. Consumer input shape,
//! processing, output scheduling and room administration remain separate capabilities.

pub mod effects;
pub mod machine;

pub use lk_agents_abi::voice::room_binding::BindingCapabilities;
