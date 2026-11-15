//! Shared room attachment identity and lifecycle records, independent of a session engine.

/// An incarnation supplied by the room owner. Reusing a readable key requires a new ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BindingId(pub u64);

/// Application keys and convenience-start identities occupy separate namespaces.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum BindingKey {
    Named(String),
    Automatic(BindingId),
}

/// The owner holds a key until every resource of its incarnation has settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, crate::Vocabulary)]
pub enum BindingPhase {
    Declared,
    Starting,
    Active,
    Closing,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BindingRefused {
    #[error("room I/O is closed")]
    OwnerClosed,
    #[error("room I/O binding key {0:?} is already reserved")]
    KeyInUse(BindingKey),
    #[error("room I/O binding is unavailable or has already been activated")]
    Unavailable,
}

/// Fixed at binding construction; a non-conversational consumer does not
/// become eligible to receive legacy voice controls just by starting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BindingCapabilities {
    pub legacy_voice: bool,
}

/// Captured at admission. A request retains its target; unsolicited events must
/// still match the current projection before they can be published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegacyProjection {
    pub binding: BindingId,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the binding is not active and compatible with the legacy protocol")]
pub struct LegacyTargetUnavailable;
