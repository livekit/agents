//! The existing single-session client protocol projects one compatible binding.
//! This role does not select audio inputs, arbitrate audio output or own the job's report.

pub mod effects;
pub mod machine;

pub use lk_agents_abi::voice::room_binding::{LegacyProjection, LegacyTargetUnavailable};
