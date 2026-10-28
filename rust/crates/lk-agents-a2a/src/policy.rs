//! The door's arrival options: the five inbox policies and the debounce, as declared (one
//! value, or per channel over the default map) and as resolved for a message's channel.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use lk_agents_storage::run::{CHANNEL_KEY, KIND_KEY};

use crate::Policy;

/// `inbox` as a door declares it: nothing (the default map), one policy for every
/// channel, or a map that overrides the default map channel by channel. On the wire `null`, a
/// policy string, or an object of them by channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PolicySpec {
    #[default]
    Default,
    All(Policy),
    ByChannel(BTreeMap<String, Policy>),
}

/// `PolicySpec` as JSON carries it: the policy strings, read by name so an unknown one is named
/// in the error rather than reported as "no variant matched".
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum PolicyWire {
    All(String),
    ByChannel(BTreeMap<String, String>),
}

impl Serialize for PolicySpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            PolicySpec::Default => None,
            PolicySpec::All(p) => Some(PolicyWire::All(p.as_str().to_owned())),
            PolicySpec::ByChannel(m) => Some(PolicyWire::ByChannel(
                m.iter().map(|(k, v)| (k.clone(), v.as_str().to_owned())).collect(),
            )),
        };
        wire.serialize(s)
    }
}

impl<'de> Deserialize<'de> for PolicySpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let named = |s: &str| {
            Policy::parse(s)
                .ok_or_else(|| serde::de::Error::custom(format!("unknown policy {s:?}")))
        };
        Ok(match Option::<PolicyWire>::deserialize(d)? {
            None => PolicySpec::Default,
            Some(PolicyWire::All(s)) => PolicySpec::All(named(&s)?),
            Some(PolicyWire::ByChannel(m)) => PolicySpec::ByChannel(
                m.into_iter().map(|(k, v)| named(&v).map(|p| (k, p))).collect::<Result<_, _>>()?,
            ),
        })
    }
}

/// The debounce as a door declares it: nothing (the defaults), one wait for every channel, or a
/// map per channel. On the wire `null`, seconds, or an object of seconds by channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DebounceSpec {
    #[default]
    Default,
    All(Duration),
    ByChannel(BTreeMap<String, Duration>),
}

/// `DebounceSpec` as JSON carries it: seconds as a number.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum DebounceWire {
    All(f64),
    ByChannel(BTreeMap<String, f64>),
}

/// Seconds as a `Duration`; negative, infinite or NaN is the caller's error.
pub(crate) fn duration_from_secs(secs: f64) -> Result<Duration, String> {
    Duration::try_from_secs_f64(secs).map_err(|e| e.to_string())
}

impl Serialize for DebounceSpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            DebounceSpec::Default => None,
            DebounceSpec::All(d) => Some(DebounceWire::All(d.as_secs_f64())),
            DebounceSpec::ByChannel(m) => Some(DebounceWire::ByChannel(
                m.iter().map(|(k, v)| (k.clone(), v.as_secs_f64())).collect(),
            )),
        };
        wire.serialize(s)
    }
}

impl<'de> Deserialize<'de> for DebounceSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let secs = |s: f64| duration_from_secs(s).map_err(serde::de::Error::custom);
        Ok(match Option::<DebounceWire>::deserialize(d)? {
            None => DebounceSpec::Default,
            Some(DebounceWire::All(s)) => DebounceSpec::All(secs(s)?),
            Some(DebounceWire::ByChannel(m)) => DebounceSpec::ByChannel(
                m.into_iter().map(|(k, v)| secs(v).map(|d| (k, d))).collect::<Result<_, _>>()?,
            ),
        })
    }
}

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

