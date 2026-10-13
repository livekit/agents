//! Expressive markup: the LLM is taught one tag, `<expr type label/>`, lowered per provider. Square
//! brackets are never stripped (they are prose links, #6590).

mod chat;
mod expr;
mod instructions;
mod mood;
mod mood_data;
mod prompts;
mod provider;
mod steering;
mod strip;
pub mod summarize;
mod transcript;

pub use chat::{
    EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID, positions_of, remove_expressive_instructions,
    strip_assistant_markup,
};
pub use expr::ExpressiveTag;
pub use summarize::{SummarizePlan, summarize_plan, summarize_plan_of, summary_content, to_xml, to_xml_of};
pub use mood::{DEFAULT_MOOD, MOOD_PRIORITY, Mood, match_mood};
pub use provider::{NonverbalField, Provider, supported_nonverbals};
pub use steering::{
    NonverbalOptions, NonverbalSounds, Pace, SpeechSteering, allowed_prosody, allowed_sounds,
    steering_instructions,
};
pub use strip::{convert_expression_tags, extract_and_strip};
pub use transcript::{
    CuePlan, TranscriptMarkupStripper, drop_bracket_cues, drop_bracket_cues_plan,
    expression_attribute, expression_attribute_map, split_all_markup, strip_all_markup,
    strip_expr_markup,
};

/// Returned because the blingfire tokenizer lives outside this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SentenceTokenizerConfig {
    pub max_token_len: Option<usize>,
    /// Under expressive this equals `max_token_len`, so sentences batch up to that size.
    pub min_token_len: Option<usize>,
    /// Only under expressive: xml-aware tokenization would stall streaming on a stray `<`.
    pub xml_aware: bool,
}

/// Declares TTS markup capabilities for the expressive pipeline (`tts.py`'s `TTS.Markup`).
#[crate::class]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Markup(Option<Provider>);

/// `TTS.Markup`, keyed by `_provider_key()`. An unknown key is the no-dialect markup.
#[crate::host(object(
    pid = false
))]
impl Markup {
    /// Resolve a `_provider_key()` string; an unrecognised key is the no-dialect markup.
    #[host(new)]
    pub fn from_provider_key(provider_key: &str) -> Self {
        Markup(Provider::from_wire(provider_key))
    }

    pub fn from_provider(provider: Provider) -> Self {
        Markup(Some(provider))
    }

    /// `_provider_key()` --- `""` when there is no dialect.
    #[host(method(name = "_provider_key"))]
    pub fn provider_key(&self) -> &'static str {
        self.0.map_or("", Provider::as_str)
    }

    /// Instructions for the LLM describing available markup tags, or `None` if this TTS has no
    /// markup support. `None` also turns expressive off for the turn (`_resolve_expressive_options`).
    #[host(method(kw_only))]
    pub fn llm_instructions(&self, speech_steering: Option<&SpeechSteering>) -> Option<String> {
        instructions::llm_instructions(self.0, speech_steering)
    }

    /// Fix common LLM markup mistakes (e.g. unclosed self-closing tags). A no-op with no dialect.
    #[host(method)]
    pub fn normalize(&self, text: &str) -> String {
        expr::normalize_markup(self.0, text)
    }

    /// Convert framework-standard markup to the provider's native format.
    #[host(method)]
    pub fn convert(&self, text: &str) -> String {
        expr::convert_markup(self.0, text)
    }

    /// The max text chunk length for this provider, or `None` if unlimited.
    pub fn max_input_len(self) -> Option<usize> {
        self.0.and_then(Provider::max_input_len)
    }

    /// Default blingfire settings for this provider's streamed TTS input.
    pub fn sentence_tokenizer(self, expressive: bool) -> SentenceTokenizerConfig {
        let max_len = self.max_input_len();
        SentenceTokenizerConfig {
            max_token_len: max_len,
            min_token_len: if expressive { max_len } else { None },
            xml_aware: expressive,
        }
    }

    /// Render a [`SpeechSteering`] into delivery guidelines for this provider; `""` if no change.
    pub fn steering_instructions(self, steering: &SpeechSteering) -> String {
        steering::steering_instructions(self.0, steering)
    }
}

/// `MarkupInfo.nonverbals` for a `_provider_key()`: the field -> labels matrix, in insertion order.
#[crate::host(export(pid = false, name = "supported_nonverbals"))]
pub fn nonverbals_of(provider: &str) -> serde_json::Map<String, serde_json::Value> {
    supported_nonverbals(Provider::from_wire(provider))
        .into_iter()
        .map(|(field, labels)| (field.as_str().to_owned(), labels.into()))
        .collect()
}

/// The max text chunk length for a `_provider_key()`, or `None` if unlimited.
#[crate::host(export(pid = false, name = "max_input_len"))]
pub fn max_input_len_of(provider: &str) -> Option<usize> {
    Markup::from_provider_key(provider).max_input_len()
}

/// `(max_token_len, min_token_len, xml_aware)` for `sentence_tokenizer(provider, expressive=)`; the
/// shim builds the blingfire tokenizer from it.
#[crate::host(export(pid = false, name = "sentence_tokenizer_config", kw_only = "expressive"))]
pub fn sentence_tokenizer_of(provider: &str, expressive: bool) -> (Option<usize>, Option<usize>, bool) {
    let c = Markup::from_provider_key(provider).sentence_tokenizer(expressive);
    (c.max_token_len, c.min_token_len, c.xml_aware)
}

/// [`Markup::steering_instructions`] for a `_provider_key()`.
#[crate::host(export(pid = false, name = "steering_instructions"))]
pub fn steering_instructions_of(provider: &str, steering: &SpeechSteering) -> String {
    Markup::from_provider_key(provider).steering_instructions(steering)
}

#[cfg(test)]
mod tests;
