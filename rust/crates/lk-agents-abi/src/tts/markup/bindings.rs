//! Host-visible speech transformations; implementations live with voice algorithms.

use super::{ExpressiveTag, Mood, NonverbalField, SpeechSteering};
use crate::ChatItem;
use std::collections::BTreeMap;

#[crate::host(functions(pid = false))]
pub trait MarkupFunctions {
    #[host(function, pid = true)]
    fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String>;
    #[host(function)]
    fn supported_nonverbals(provider: String) -> BTreeMap<NonverbalField, Vec<String>>;
    #[host(function)]
    fn max_input_len(provider: String) -> Option<usize>;
    #[host(function)]
    fn sentence_tokenizer_config(
        provider: String,
        #[host(kw)] expressive: bool,
    ) -> (Option<usize>, Option<usize>, bool);
    #[host(function)]
    fn steering_instructions(provider: String, steering: SpeechSteering) -> String;
    #[host(function)]
    fn convert_expression_tags(text: String) -> String;
    #[host(function)]
    fn extract_and_strip(text: String, xml_tags: Vec<String>) -> (String, Vec<(String, String)>);
    #[host(function)]
    fn split_all_markup(text: String) -> (String, Vec<ExpressiveTag>);
    #[host(function)]
    fn strip_all_markup(text: String) -> String;
    #[host(function)]
    fn strip_expr_markup(text: String) -> String;
    #[host(function)]
    fn split_expr_markup(text: String) -> (String, Vec<ExpressiveTag>);
    #[host(function)]
    fn expression_attribute(tags: Vec<ExpressiveTag>) -> Option<BTreeMap<String, String>>;
    #[host(function)]
    fn drop_bracket_cues(texts: Vec<String>, end_of_stream: bool) -> (CuePlan, CuePlan);
    #[host(function)]
    fn match_mood(label: String, #[host(default = None)] fallback: Option<Mood>) -> Option<Mood>;
    #[host(function)]
    fn to_xml(
        tag_name: String,
        #[host(default = None)] content: Option<String>,
        #[host(default = None)] attrs: Option<Vec<(String, String)>>,
    ) -> String;
    #[host(function)]
    fn chat_history_summary_xml(summary: String) -> String;
    #[host(function)]
    fn chat_ctx_summarize_plan(
        items: Vec<ChatItem>,
        keep_last_turns: usize,
    ) -> Option<(usize, Vec<usize>, String, String, f64)>;
}

pub type CuePlan = Vec<(usize, Option<String>)>;

/// The markup dialect a TTS speaks, for the expressive pipeline: what the LLM is taught to write
/// and how it is normalized and lowered before synthesis.
///
/// Plugins opt in by overriding `_provider_key`.
#[crate::host(class(name = "Markup", pid = false))]
pub trait MarkupClass: Send + Sized + 'static {
    #[host(new)]
    fn from_provider_key(provider_key: String) -> Self;
    /// The provider's key into the markup tables, or `""` for no markup.
    #[host(call, name = "_provider_key")]
    fn provider_key(&self) -> String;
    /// The instructions teaching the LLM this voice's markup, `None` without markup.
    #[host(call)]
    fn llm_instructions(
        &self,
        #[host(kw, default = None)] speech_steering: Option<SpeechSteering>,
    ) -> Option<String>;
    /// Fix common LLM markup mistakes (e.g. unclosed self-closing tags).
    #[host(call)]
    fn normalize(&self, text: String) -> String;
    /// Lower the framework's markup to the provider's native syntax.
    #[host(call)]
    fn convert(&self, text: String) -> String;
}

/// Stateful, provider-agnostic markup stripper for one transcript segment.
#[crate::host(class(name = "TranscriptMarkupStripper", pid = false))]
pub trait TranscriptStripper: Send + Sized + 'static {
    #[host(new)]
    fn new() -> Self;
    #[host(call)]
    fn push(&mut self, text: String) -> String;
    #[host(call)]
    fn flush(&mut self) -> String;
    #[host(getter)]
    fn tags(&self) -> Vec<ExpressiveTag>;
    #[host(call)]
    fn expression_attribute(&self) -> Option<BTreeMap<String, String>>;
}
