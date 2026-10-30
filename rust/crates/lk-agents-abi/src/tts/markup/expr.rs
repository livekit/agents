//! Expressive tags reported by speech preprocessing.

/// An expressive markup tag stripped from a transcript: the tag name (or the expr marker's type)
/// and its payload (the ``value`` attribute, the expr ``label`` or the inner text).
// A `TypedDict` in the reference; no field defaults, since `t["type"]` must raise on a missing key.
#[crate::host(record(module = "tts", path = "livekit.agents.tts", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpressiveTag {
    /// The markup tag name (``"emotion"``, ``"sound"``, ...) or the expr marker type.
    #[host(rename = "type")]
    pub kind: String,
    /// The payload: the ``value="..."`` attribute, the expr ``label``, or the tag's inner text.
    pub value: String,
}

impl ExpressiveTag {
    pub fn new(kind: impl Into<String>, value: impl Into<String>) -> Self {
        ExpressiveTag { kind: kind.into(), value: value.into() }
    }
}
