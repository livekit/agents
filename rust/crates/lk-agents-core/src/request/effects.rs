use lk_agents_abi::Ts;
use lk_agents_abi::errors::ApiError;

/// What the owner performs or reports, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// Start this attempt's exchange.
    Launch(u32),
    /// Stop this attempt's exchange; its release is still reported.
    Stop(u32),
    /// The attempt failed and the request will retry after its backoff.
    Retry { attempt: u32, error: ApiError },
    /// The request failed: no attempt follows.
    GiveUp(ApiError),
    /// A retiring request's outcome was still unknown at its deadline; no attempt follows.
    Unresolved,
    /// The attempt missed its cleanup deadline; the runtime still owns it.
    CleanupIncomplete(u32),
}

/// The retry decision for a failed, fully released provider attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum Retry {
    At(Ts),
    GiveUp(ApiError),
}
