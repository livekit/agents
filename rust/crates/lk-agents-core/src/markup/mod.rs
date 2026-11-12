use lk_agents_abi::tts::markup as abi;
use lk_agents_abi::tts::markup::{MarkupClass, MarkupFunctions};
use lk_agents_abi::tts::types::SpeechChunk;
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
    TranscriptMarkupStripper, delivery, delivery_attribute, expression_attribute,
    expression_attribute_map, expression_markup, split_all_markup, split_expr_markup,
    strip_all_markup, strip_expr_markup,
};

pub use lk_agents_abi::tts::markup::Markup;

// Square brackets are never stripped: they are prose links.

pub trait MarkupExt {
    fn llm_instructions(&self, speech_steering: Option<&SpeechSteering>) -> Option<String>;
    /// A delta with its settled tags canonical and its point markers self-closed, so a splitter
    /// keeps only real wrappers open.
    fn normalize(&self, text: &str) -> String;
    /// A chunk in the provider's syntax, with the words it says.
    fn lower(&self, text: &str) -> SpeechChunk;
    /// The provider's request limit in characters, where the dialect knows it.
    fn max_input_len(self) -> Option<usize>;
    fn steering_instructions(self, steering: &SpeechSteering) -> String;
}

impl MarkupExt for Markup {
    fn llm_instructions(&self, speech_steering: Option<&SpeechSteering>) -> Option<String> {
        instructions::llm_instructions(self.0, speech_steering)
    }

    fn normalize(&self, text: &str) -> String {
        lower::normalize(self.0, text)
    }

    fn lower(&self, text: &str) -> SpeechChunk {
        lower::lower(self.0, text)
    }

    fn max_input_len(self) -> Option<usize> {
        self.0.and_then(ProviderExt::max_input_len)
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

/// The markup a voice speaking `markup_provider`'s dialect keeps whole: every tag its stage
/// lowers, `<expr>` included.
pub fn markup_vocabulary(markup_provider: &str) -> crate::lexer::Vocabulary {
    crate::lexer::Vocabulary::of(Provider::from_wire(markup_provider))
}

#[cfg(test)]
#[path = "markup_tests.rs"]
mod tests;

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
