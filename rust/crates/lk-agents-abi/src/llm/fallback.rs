//! The fallback adapters' availability records: `AvailabilityChangedEvent` (all three
//! adapters) and `RealtimeModelFallbackAdapter`'s error plan.

use lk_agents_host_derive::Host;

/// `AvailabilityChangedEvent`, identical in all three fallback adapters: an entry, by position in
/// the host's list, went unavailable or recovered. The host maps the index back to its instance.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct AvailabilityChanged {
    pub index: usize,
    pub label: String,
    pub available: bool,
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
