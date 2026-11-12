use lk_agents_abi::tts::markup as abi;
use lk_agents_abi::tts::markup::{MarkupClass, MarkupFunctions};
pub(crate) use provider::ProviderExt;

mod chat;
mod instructions;
mod lower;
mod mood;
mod mood_data;
mod prompts;
mod provider;
mod transcript;

pub use chat::{
    ChatMessageExt, EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID, chat_message_text_content,
    remove_expressive_instructions,
};
pub use instructions::{
    NonverbalOptions, NonverbalSounds, Pace, SpeechSteering, allowed_prosody, allowed_sounds,
    steering_instructions,
};
pub use lk_agents_abi::tts::markup::ExpressiveTag;
pub use mood::{DEFAULT_MOOD, MOOD_PRIORITY, Mood, match_mood};
pub use provider::{NonverbalField, Provider, supported_nonverbals};
pub use transcript::{
    CuePlan, TranscriptMarkupStripper, delivery, delivery_attribute, drop_bracket_cues,
    drop_bracket_cues_plan, expression_attribute, expression_attribute_map, expression_markup,
    split_all_markup, split_expr_markup, strip_all_markup, strip_expr_markup,
};

pub use lk_agents_abi::tts::markup::{Markup, SentenceTokenizerConfig};

// Square brackets are never stripped: they are prose links.

pub trait MarkupExt {
    fn llm_instructions(&self, speech_steering: Option<&SpeechSteering>) -> Option<String>;
    fn normalize(&self, text: &str) -> String;
    fn convert(&self, text: &str) -> String;
    fn max_input_len(self) -> Option<usize>;
    fn sentence_tokenizer(self, expressive: bool) -> SentenceTokenizerConfig;
    fn steering_instructions(self, steering: &SpeechSteering) -> String;
}

impl MarkupExt for Markup {
    fn llm_instructions(&self, speech_steering: Option<&SpeechSteering>) -> Option<String> {
        instructions::llm_instructions(self.0, speech_steering)
    }

    fn normalize(&self, text: &str) -> String {
        lower::normalize(self.0, text)
    }

    fn convert(&self, text: &str) -> String {
        lower::convert(self.0, text)
    }

    fn max_input_len(self) -> Option<usize> {
        self.0.and_then(ProviderExt::max_input_len)
    }

    fn sentence_tokenizer(self, expressive: bool) -> SentenceTokenizerConfig {
        let max_len = self.max_input_len();
        SentenceTokenizerConfig {
            max_token_len: max_len,
            batch_to_max: expressive && max_len.is_some(),
        }
    }

    fn steering_instructions(self, steering: &SpeechSteering) -> String {
        instructions::steering_instructions(self.0, steering)
    }
}

/// `MarkupInfo.nonverbals` for a `_provider_key()`: field -> labels, in wire field order.
pub fn nonverbals_of(provider: &str) -> std::collections::BTreeMap<NonverbalField, Vec<String>> {
    supported_nonverbals(Provider::from_wire(provider))
        .into_iter()
        .map(|(field, labels)| (field, labels.into_iter().map(str::to_owned).collect()))
        .collect()
}

/// The markup a voice speaking `markup_provider`'s dialect keeps whole, while it flows.
pub fn markup_vocabulary(
    markup_provider: &str,
    expressive: bool,
) -> Option<crate::lexer::Vocabulary> {
    expressive.then(|| crate::lexer::Vocabulary::of(Provider::from_wire(markup_provider)))
}

/// `sentence_tokenizer(provider, expressive=)` for a `_provider_key()`.
pub fn sentence_tokenizer_of(provider: &str, expressive: bool) -> SentenceTokenizerConfig {
    Markup::from_provider_key(provider).sentence_tokenizer(expressive)
}

#[cfg(test)]
#[path = "markup_tests.rs"]
mod tests;

/// Supplies the non-streaming TTS adapter with this voice's text transformations.
pub fn text_processor_factory() -> lk_agents_abi::tts::stream_adapter::TextProcessorFactory {
    std::sync::Arc::new(text_processor)
}

/// Select the provider dialect without allocating another factory.
pub fn text_processor(
    provider: &str,
) -> Option<Box<dyn lk_agents_abi::tts::stream_adapter::SentenceTextProcessor>> {
    let markup = Markup::from_provider_key(provider);
    (!markup.provider_key().is_empty()).then(|| {
        Box::new(SentenceMarkup(markup))
            as Box<dyn lk_agents_abi::tts::stream_adapter::SentenceTextProcessor>
    })
}

struct SentenceMarkup(Markup);

impl lk_agents_abi::tts::stream_adapter::SentenceTextProcessor for SentenceMarkup {
    fn normalize(&self, text: &str) -> String {
        self.0.normalize(text)
    }
    fn convert(&self, text: &str) -> String {
        self.0.convert(text)
    }
}

pub struct SpeechMarkup;

impl MarkupFunctions for SpeechMarkup {
    fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String> {
        self::chat_message_text_content(parts, assistant)
    }
    fn supported_nonverbals(provider: String) -> lk_agents_abi::MarkupInfo {
        lk_agents_abi::MarkupInfo { nonverbals: self::nonverbals_of(&provider) }
    }
    fn strip_expr_markup(text: String) -> String {
        self::strip_expr_markup(&text)
    }
    fn split_expr_markup(text: String) -> lk_agents_abi::tts::markup::MarkupSplit {
        let (text, tags) = self::split_expr_markup(&text);
        lk_agents_abi::tts::markup::MarkupSplit { text, tags }
    }
    fn to_xml(
        tag_name: String,
        content: Option<String>,
        attrs: Option<Vec<(String, String)>>,
    ) -> String {
        to_xml_of(&tag_name, content.as_deref(), attrs)
    }
}

/// One provider's markup dialect, as the host's `Markup` class.
pub struct ProviderMarkup(abi::Markup);

impl MarkupClass for ProviderMarkup {
    fn from_provider_key(provider_key: String) -> Self {
        Self(abi::Markup::from_provider_key(&provider_key))
    }
    fn provider_key(&self) -> String {
        self.0.provider_key().to_owned()
    }
    fn llm_instructions(&self, speech_steering: Option<SpeechSteering>) -> Option<String> {
        self.0.llm_instructions(speech_steering.as_ref())
    }
    fn normalize(&self, text: String) -> String {
        self.0.normalize(&text)
    }
    fn convert(&self, text: String) -> String {
        self.0.convert(&text)
    }
}


/// Render an element with owned attribute pairs, retaining their supplied order.
fn to_xml_of(
    tag_name: &str,
    content: Option<&str>,
    attrs: Option<Vec<(String, String)>>,
) -> String {
    let attrs = attrs.unwrap_or_default();
    to_xml(
        tag_name,
        content,
        &attrs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<Vec<_>>(),
    )
}

/// An absent or empty body renders a self-closing element.
fn to_xml(tag_name: &str, content: Option<&str>, attrs: &[(&str, &str)]) -> String {
    let attrs_str = if attrs.is_empty() {
        None
    } else {
        Some(attrs.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect::<Vec<_>>().join(" "))
    };

    match content.filter(|c| !c.is_empty()) {
        Some(content) => {
            let open = match &attrs_str {
                Some(a) => format!("<{tag_name} {a}>"),
                None => format!("<{tag_name}>"),
            };
            format!("{open}\n{content}\n</{tag_name}>")
        }
        None => match &attrs_str {
            Some(a) => format!("<{tag_name} {a} />"),
            None => format!("<{tag_name} />"),
        },
    }
}

#[cfg(test)]
mod xml_tests {
    use super::*;

    #[test]
    fn empty_content_is_self_closing() {
        assert_eq!(to_xml("t", Some(""), &[]), "<t />");
        assert_eq!(to_xml("t", None, &[("k", "v")]), r#"<t k="v" />"#);
    }
}
