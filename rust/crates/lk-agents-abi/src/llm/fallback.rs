//! The fallback adapters' availability records: `AvailabilityChangedEvent` (all three
//! adapters), the chain's report to its adapter and `RealtimeModelFallbackAdapter`'s error plan.

use lk_agents_host_derive::Host;

/// `AvailabilityChangedEvent`, identical in all three fallback adapters: an entry, by position in
/// the host's list, went unavailable or recovered. The host maps the index back to its instance.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "llm_fallback", emit))]
pub struct AvailabilityChanged {
    pub index: usize,
    pub label: String,
    pub available: bool,
}

/// What a fallback chain reports to the adapter that built it (each host `FallbackAdapter`, the
/// realtime one included): the adapter maps the index back to its instance.
#[crate::host(proxy(host = "livekit.agents._core_view.AvailabilityObserver"))]
pub trait ChainObserver: Send + Sync {
    #[host(notify, name = "_core_availability_changed")]
    fn availability_changed(&self, ev: AvailabilityChanged);

    /// `logger.error("all X are unavailable, retrying..")`. The chain then tries every entry
    /// anyway rather than failing fast, which is what `all_failed` does in all three adapters.
    #[host(notify, name = "_core_all_unavailable")]
    fn all_unavailable(&self) {}
}

/// One `realtime_availability_changed` a `RealtimeModelFallbackAdapter` emits: the host maps the
/// index back to the model object it passed in, because users compare that object with `is`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct AvailabilityChange {
    pub index: usize,
    pub available: bool,
}

/// What `RealtimeModelFallbackAdapter._on_child_error` decided.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct ErrorPlan {
    /// Emit these `realtime_availability_changed` events first, in order.
    pub changes: Vec<AvailabilityChange>,
    /// The rewrite: emit with `recoverable=True` because a fallback exists and the swap will
    /// rescue the call.
    pub rewrite_recoverable: bool,
    /// The model to swap to, or `None`: the error was recoverable, every model is down, or a swap
    /// is already in flight.
    pub swap_to: Option<usize>,
}
