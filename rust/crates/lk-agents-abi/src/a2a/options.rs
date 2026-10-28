//! A text door's arrival options, declared once for native execution and the host API.

use std::collections::BTreeMap;
use std::time::Duration;

pub const LEASE_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// A text door's turn options; a field left out is the default.
#[derive(Debug, Clone, Default, PartialEq)]
#[crate::host(record(module = "storage_options", emit, extract))]
pub struct TurnOptions {
    /// `inbox`: one policy for every channel, or a map by channel over the
    /// default map; `None` is the default map.
    #[host(default = None)]
    pub policy: PolicySpec,
    /// The wait before a turn starts, in seconds: one for every channel, or a map by channel;
    /// `None` is the defaults.
    #[host(default = None)]
    pub debounce: DebounceSpec,
    /// How long a waiter polls for the lane before its task fails `lease-timeout`, in seconds;
    /// `None` is a minute.
    #[host(default = None)]
    pub wait_timeout: Option<Duration>,
}

impl TurnOptions {
    /// `wait_timeout`, or the crate default.
    pub fn wait_timeout(&self) -> Duration {
        self.wait_timeout.unwrap_or(LEASE_WAIT_TIMEOUT)
    }
}

/// A message during a run: one of the five policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[crate::host(record(module = "storage_options", emit, extract))]
#[crate::host(wire_strs)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Policy {
    #[host(rename = "batch")]
    Batch,
    #[host(rename = "interrupt")]
    Interrupt,
    #[host(rename = "queue")]
    Queue,
    #[host(rename = "concurrent")]
    Concurrent,
    #[host(rename = "reject")]
    Reject,
}

impl Policy {
    /// `None` for a string no policy spells.
    pub fn parse(s: &str) -> Option<Policy> {
        Policy::from_wire(s)
    }
}

/// `inbox` as a door declares it: nothing (the default map), one policy for every
/// channel, or a map that overrides the default map channel by channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "storage_options", emit, extract, transparent))]
pub enum PolicySpec {
    #[default]
    Default,
    All(Policy),
    ByChannel(BTreeMap<String, Policy>),
}

/// The debounce as a door declares it: nothing (the defaults), one wait for every channel, or a
/// map per channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "storage_options", emit, extract, transparent))]
pub enum DebounceSpec {
    #[default]
    Default,
    All(Duration),
    ByChannel(BTreeMap<String, Duration>),
}
