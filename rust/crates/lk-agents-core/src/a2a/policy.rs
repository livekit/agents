//! The door's arrival options: the five inbox policies and the debounce, as declared (one
//! value, or per channel over the default map) and as resolved for a message's channel.

use lk_agents_abi::a2a_protocol as a2a;
use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;

use lk_agents_abi::a2a::metadata::{CHANNEL_KEY, KIND_KEY};

use crate::a2a::Policy;

pub use lk_agents_abi::a2a::options::{DebounceSpec, PolicySpec};

/// the default map, keyed by `livekit/channel`.
pub const DEFAULT_POLICY_MAP: [(&str, Policy); 5] = [
    ("sms", Policy::Batch),
    ("whatsapp", Policy::Batch),
    ("email", Policy::Batch),
    ("a2a", Policy::Interrupt),
    ("delegation", Policy::Queue),
];

/// two seconds for the inbox channels, zero for `a2a`; starting points.
pub const DEFAULT_DEBOUNCE: [(&str, Duration); 5] = [
    ("sms", Duration::from_secs(2)),
    ("whatsapp", Duration::from_secs(2)),
    ("email", Duration::from_secs(2)),
    ("a2a", Duration::ZERO),
    ("delegation", Duration::ZERO),
];

pub const INTERRUPTION_NEW_MESSAGE: &str = "new-message";
/// The run reason when a client canceled the task with `tasks/cancel`.
pub const CLIENT_CANCEL_REASON: &str = "client-cancel";
pub const A2A_CHANNEL: &str = "a2a";
pub const DELEGATION_CHANNEL: &str = "delegation";

/// `livekit/channel` of an A2A message; `livekit/kind = delegation` counts as `delegation`, and a
/// message with no channel counts as `a2a`.
pub fn channel_of(message: &a2a::Message) -> String {
    let metadata = message.metadata.as_ref();
    let field = |key: &str| metadata.and_then(|m| m.get(key)).and_then(Value::as_str);
    if field(KIND_KEY) == Some(DELEGATION_CHANNEL) {
        return DELEGATION_CHANNEL.to_owned();
    }
    match field(CHANNEL_KEY) {
        Some(c) if !c.is_empty() => c.to_owned(),
        _ => A2A_CHANNEL.to_owned(),
    }
}

/// One policy for a channel: a value applies to every channel, a map overrides the default map;
/// a channel in neither gets the `a2a` policy.
pub fn resolve_policy(policy: &PolicySpec, channel: &str) -> Policy {
    if let PolicySpec::All(p) = policy {
        return *p;
    }
    let mut table: BTreeMap<&str, Policy> = DEFAULT_POLICY_MAP.into_iter().collect();
    if let PolicySpec::ByChannel(map) = policy {
        for (k, v) in map {
            table.insert(k.as_str(), *v);
        }
    }
    table.get(channel).or_else(|| table.get(A2A_CHANNEL)).copied().unwrap_or(Policy::Interrupt)
}

/// The wait before a turn starts so a burst has landed; per channel, or one value for all. A
/// channel the defaults do not name waits the `sms` default under `batch` and nothing otherwise.
pub fn resolve_debounce(debounce: &DebounceSpec, channel: &str, policy: Policy) -> Duration {
    match debounce {
        DebounceSpec::All(d) => return *d,
        DebounceSpec::ByChannel(map) => {
            if let Some(d) = map.get(channel) {
                return *d;
            }
        }
        DebounceSpec::Default => {}
    }
    let defaults: BTreeMap<&str, Duration> = DEFAULT_DEBOUNCE.into_iter().collect();
    if let Some(d) = defaults.get(channel) {
        return *d;
    }
    if policy == Policy::Batch { defaults["sms"] } else { Duration::ZERO }
}
