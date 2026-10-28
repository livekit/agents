//! Native test peers use the ABI room capabilities and modeled stream admission.

use super::Result;
pub use lk_agents_abi::voice::rtc::capabilities::*;
use std::sync::Arc;

pub trait RtcHost: Send + Sync {
    fn connect<'a>(
        &'a self,
        url: &'a str,
        token: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Arc<dyn RtcRoom>>>;
}
