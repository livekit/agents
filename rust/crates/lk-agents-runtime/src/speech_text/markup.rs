//! Host export implementations for expressive speech text.

use lk_agents_abi::ChatItem;
use lk_agents_abi::tts::markup::bindings::{
    CuePlan, MarkupClass, MarkupFunctions, TranscriptStripper,
};
use lk_agents_abi::tts::markup::{
    self as abi, ExpressiveTag, Mood, NonverbalField, SpeechSteering,
};
use lk_agents_voice::tts::markup::{self as voice, MarkupExt};

pub use lk_agents_voice::tts::markup::text_processor_factory;
use std::collections::BTreeMap;

pub struct SpeechMarkup;

impl MarkupFunctions for SpeechMarkup {
    fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String> {
        voice::chat_message_text_content(parts, assistant)
    }
    fn supported_nonverbals(provider: String) -> BTreeMap<NonverbalField, Vec<String>> {
        voice::nonverbals_of(&provider)
    }
    fn max_input_len(provider: String) -> Option<usize> {
        voice::max_input_len_of(&provider)
    }
    fn sentence_tokenizer_config(
        provider: String,
        expressive: bool,
    ) -> (Option<usize>, Option<usize>, bool) {
        voice::sentence_tokenizer_of(&provider, expressive)
    }
    fn steering_instructions(provider: String, steering: SpeechSteering) -> String {
        voice::steering_instructions_of(&provider, &steering)
    }
    fn convert_expression_tags(text: String) -> String {
        voice::convert_expression_tags(&text)
    }
    fn extract_and_strip(text: String, xml_tags: Vec<String>) -> (String, Vec<(String, String)>) {
        voice::extract_and_strip(&text, &xml_tags.iter().map(String::as_str).collect::<Vec<_>>())
    }
    fn split_all_markup(text: String) -> (String, Vec<ExpressiveTag>) {
        voice::split_all_markup(&text)
    }
    fn strip_all_markup(text: String) -> String {
        voice::strip_all_markup(&text)
    }
    fn strip_expr_markup(text: String) -> String {
        voice::strip_expr_markup(&text)
    }
    fn split_expr_markup(text: String) -> (String, Vec<ExpressiveTag>) {
        voice::split_expr_markup(&text)
    }
    fn expression_attribute(tags: Vec<ExpressiveTag>) -> Option<BTreeMap<String, String>> {
        voice::expression_attribute_map(tags)
    }
    fn drop_bracket_cues(texts: Vec<String>, end_of_stream: bool) -> (CuePlan, CuePlan) {
        voice::drop_bracket_cues_plan(&texts, end_of_stream)
    }
    fn match_mood(label: String, fallback: Option<Mood>) -> Option<Mood> {
        voice::match_mood(&label, fallback)
    }
    fn to_xml(
        tag_name: String,
        content: Option<String>,
        attrs: Option<Vec<(String, String)>>,
    ) -> String {
        voice::to_xml_of(&tag_name, content.as_deref(), attrs)
    }
    fn chat_history_summary_xml(summary: String) -> String {
        voice::summary_content(&summary)
    }
    fn chat_ctx_summarize_plan(
        items: Vec<ChatItem>,
        keep_last_turns: usize,
    ) -> Option<(usize, Vec<usize>, String, String, f64)> {
        voice::summarize_plan(&items, keep_last_turns).map(|p| {
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

pub struct Markup(abi::Markup);

impl MarkupClass for Markup {
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

pub struct TranscriptMarkupStripper(voice::TranscriptMarkupStripper);

impl TranscriptStripper for TranscriptMarkupStripper {
    fn new() -> Self {
        Self(voice::TranscriptMarkupStripper::new())
    }
    fn push(&mut self, text: String) -> String {
        self.0.push(&text)
    }
    fn flush(&mut self) -> String {
        self.0.flush()
    }
    fn tags(&self) -> Vec<ExpressiveTag> {
        self.0.tags().to_vec()
    }
    fn expression_attribute(&self) -> Option<BTreeMap<String, String>> {
        self.0.expression_attribute_map()
    }
}
