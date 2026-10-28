//! Expressive tags reported by speech preprocessing.

use lk_agents_host_derive::Host;

/// An expressive markup tag stripped from a transcript, surfaced for the frontend.
// A `TypedDict` in the reference; no field defaults, since `t["type"]` must raise on a missing key.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, extract, from_items)]
pub struct ExpressiveTag {
    /// The markup tag name (`"emotion"`, `"sound"`, ...) or the expr marker type.
    #[host(rename = "type")]
    pub kind: String,
    /// The payload: the `value="..."` attribute, the expr `label`, or the tag's inner text.
    pub value: String,
}

impl ExpressiveTag {
    pub fn new(kind: impl Into<String>, value: impl Into<String>) -> Self {
        ExpressiveTag { kind: kind.into(), value: value.into() }
    }
}
