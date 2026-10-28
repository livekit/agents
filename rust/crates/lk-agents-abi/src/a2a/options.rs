//! A text door's arrival options, declared once for native execution and the host API.

use std::collections::BTreeMap;
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub const LEASE_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// A text door's turn options; a field left out is the default.
#[derive(Debug, Clone, Default, PartialEq)]
#[crate::host(record(module = "storage_options", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TurnOptions {
    /// `inbox`: one policy for every channel, or a map by channel over the
    /// default map; `None` is the default map.
    #[cfg_attr(feature = "serde", serde(default))]
    #[host(default = None)]
    pub policy: PolicySpec,
    /// The wait before a turn starts, in seconds: one for every channel, or a map by channel;
    /// `None` is the defaults.
    #[cfg_attr(feature = "serde", serde(default))]
    #[host(default = None)]
    pub debounce: DebounceSpec,
    /// How long a waiter polls for the lane before its task fails `lease-timeout`, in seconds;
    /// `None` is a minute.
    #[cfg_attr(feature = "serde", serde(default, with = "opt_seconds"))]
    #[host(default = None)]
    pub wait_timeout: Option<Duration>,
}

impl TurnOptions {
    /// `wait_timeout`, or the crate default.
    pub fn wait_timeout(&self) -> Duration {
        self.wait_timeout.unwrap_or(LEASE_WAIT_TIMEOUT)
    }
}

/// `Option<Duration>` as seconds, `null` for none.
#[cfg(feature = "serde")]
mod opt_seconds {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        d.map(|d| d.as_secs_f64()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        Option::<f64>::deserialize(d)?
            .map(|s| super::duration_from_secs(s).map_err(serde::de::Error::custom))
            .transpose()
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
/// channel, or a map that overrides the default map channel by channel. On the wire `null`, a
/// policy string, or an object of them by channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "storage_options", emit, extract, transparent))]
pub enum PolicySpec {
    #[default]
    Default,
    All(Policy),
    ByChannel(BTreeMap<String, Policy>),
}

/// `PolicySpec` as JSON carries it: the policy strings, read by name so an unknown one is named
/// in the error rather than reported as "no variant matched".
#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum PolicyWire {
    All(String),
    ByChannel(BTreeMap<String, String>),
}

#[cfg(feature = "serde")]
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

#[cfg(feature = "serde")]
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
#[crate::host(record(module = "storage_options", emit, extract, transparent))]
pub enum DebounceSpec {
    #[default]
    Default,
    All(Duration),
    ByChannel(BTreeMap<String, Duration>),
}

/// `DebounceSpec` as JSON carries it: seconds as a number.
#[cfg(feature = "serde")]
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

#[cfg(feature = "serde")]
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

#[cfg(feature = "serde")]
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
