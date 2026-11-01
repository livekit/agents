//! Language identifiers and tokenization export contracts.

pub mod bindings;
pub mod language;
pub mod search;

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "tokenizer", path = "livekit.agents.tokenize", dataclass))]
pub struct TokenData {
    #[host(default = "")]
    pub segment_id: String,
    #[host(default = "")]
    pub token: String,
}
