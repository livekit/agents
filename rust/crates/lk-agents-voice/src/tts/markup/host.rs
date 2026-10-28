//! The host's expressive-markup functions and classes (`tts.markup`).

use std::collections::BTreeMap;

use lk_agents_abi::ChatItem;
use lk_agents_abi::tts::markup::bindings::{
    CuePlan, MarkupClass, MarkupFunctions, TranscriptStripper,
};
use lk_agents_abi::tts::markup::{
    self as abi, ExpressiveTag, Mood, NonverbalField, SpeechSteering,
};

use super::MarkupExt;

pub struct SpeechMarkup;

impl MarkupFunctions for SpeechMarkup {
    fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String> {
        super::chat_message_text_content(parts, assistant)
    }
    fn supported_nonverbals(provider: String) -> BTreeMap<NonverbalField, Vec<String>> {
        super::nonverbals_of(&provider)
    }
    fn max_input_len(provider: String) -> Option<usize> {
        super::max_input_len_of(&provider)
    }
    fn sentence_tokenizer_config(
        provider: String,
        expressive: bool,
    ) -> (Option<usize>, Option<usize>, bool) {
        super::sentence_tokenizer_of(&provider, expressive)
    }
    fn steering_instructions(provider: String, steering: SpeechSteering) -> String {
        super::steering_instructions_of(&provider, &steering)
    }
    fn convert_expression_tags(text: String) -> String {
        super::convert_expression_tags(&text)
    }
    fn extract_and_strip(text: String, xml_tags: Vec<String>) -> (String, Vec<(String, String)>) {
        super::extract_and_strip(&text, &xml_tags.iter().map(String::as_str).collect::<Vec<_>>())
    }
    fn split_all_markup(text: String) -> (String, Vec<ExpressiveTag>) {
        super::split_all_markup(&text)
    }
    fn strip_all_markup(text: String) -> String {
        super::strip_all_markup(&text)
    }
    fn strip_expr_markup(text: String) -> String {
        super::strip_expr_markup(&text)
    }
    fn split_expr_markup(text: String) -> (String, Vec<ExpressiveTag>) {
        super::split_expr_markup(&text)
    }
    fn expression_attribute(tags: Vec<ExpressiveTag>) -> Option<BTreeMap<String, String>> {
        super::expression_attribute_map(tags)
    }
    fn drop_bracket_cues(texts: Vec<String>, end_of_stream: bool) -> (CuePlan, CuePlan) {
        super::drop_bracket_cues_plan(&texts, end_of_stream)
    }
    fn match_mood(label: String, fallback: Option<Mood>) -> Option<Mood> {
        super::match_mood(&label, fallback)
    }
    fn to_xml(
        tag_name: String,
        content: Option<String>,
        attrs: Option<Vec<(String, String)>>,
    ) -> String {
        super::to_xml_of(&tag_name, content.as_deref(), attrs)
    }
    fn chat_history_summary_xml(summary: String) -> String {
        super::summary_content(&summary)
    }
    fn chat_ctx_summarize_plan(
        items: Vec<ChatItem>,
        keep_last_turns: usize,
    ) -> Option<(usize, Vec<usize>, String, String, f64)> {
        super::summarize_plan(&items, keep_last_turns).map(|p| {
            (
                p.split_idx,
                p.preserved,
                p.system_prompt.to_owned(),
                p.user_content,
                lk_agents_abi::epoch_secs(p.created_at),
            )
        })
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

impl TranscriptStripper for super::TranscriptMarkupStripper {
    fn new() -> Self {
        Self::default()
    }
    fn push(&mut self, text: String) -> String {
        super::TranscriptMarkupStripper::push(self, &text)
    }
    fn flush(&mut self) -> String {
        super::TranscriptMarkupStripper::flush(self)
    }
    fn tags(&self) -> Vec<ExpressiveTag> {
        super::TranscriptMarkupStripper::tags(self).to_vec()
    }
    fn expression_attribute(&self) -> Option<BTreeMap<String, String>> {
        self.expression_attribute_map()
    }
}
