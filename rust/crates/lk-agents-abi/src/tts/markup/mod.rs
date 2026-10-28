//! Speech markup vocabulary and language-neutral export contracts.

pub mod bindings;
pub mod expr;
pub mod mood;
pub mod provider;
pub mod steering;

pub use expr::ExpressiveTag;
pub use mood::Mood;
pub use provider::{NonverbalField, Provider};
pub use steering::{NonverbalOptions, NonverbalSounds, Pace, SpeechSteering};

/// Returned because the blingfire tokenizer lives outside this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SentenceTokenizerConfig {
    pub max_token_len: Option<usize>,
    /// Under expressive this equals `max_token_len`, so sentences batch up to that size.
    pub min_token_len: Option<usize>,
    /// Only under expressive: xml-aware tokenization would stall streaming on a stray `<`.
    pub xml_aware: bool,
}

/// The selected provider dialect, retained in resource descriptions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Markup(pub Option<Provider>);

impl Markup {
    pub fn from_provider_key(provider_key: &str) -> Self {
        Self(Provider::from_wire(provider_key))
    }

    pub fn from_provider(provider: Provider) -> Self {
        Self(Some(provider))
    }

    pub fn provider_key(&self) -> &'static str {
        self.0.map_or("", Provider::as_str)
    }
}
