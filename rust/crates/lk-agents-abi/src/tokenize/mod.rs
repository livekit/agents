//! Language identifiers and tokenization export contracts.

pub mod bindings;
pub mod language;
pub mod search;

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "tokenizer",
    path = "livekit.agents.tokenize",
    dataclass,
    extract,
    emit
))]
pub struct TokenData {
    #[host(default = "")]
    pub segment_id: String,
    #[host(default = "")]
    pub token: String,
}

/// A token and its start and end offsets, in characters, in the text it was cut from: what a
/// tokenizer's `split` gives a stream to batch, and what a supplied tokenizer may return.
#[crate::host(record(
    module = "tokenizer",
    path = "livekit.agents.tokenize",
    dataclass,
    frozen,
    slots,
    extract,
    emit
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpan {
    pub token: String,
    pub start: usize,
    pub end: usize,
}
