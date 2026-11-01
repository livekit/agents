//! SDK capability execution and owned room stream admission.

pub mod events;
pub mod playout;
pub mod synchronizer;
pub mod wait;

pub use lk_agents_abi::voice::rtc::*;

pub type Result<T> = std::result::Result<T, RtcError>;
