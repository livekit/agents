//! Host-visible speech transformations; implementations live with voice algorithms.

use super::{ExpressiveTag, SpeechSteering};
use crate::ChatItem;
use std::collections::BTreeMap;

#[crate::host(functions(pid = false))]
pub trait MarkupFunctions {
    #[host(function, pid = true)]
    fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String>;
    #[host(function)]
    fn supported_nonverbals(provider: String) -> crate::MarkupInfo;
    #[host(function)]
    fn split_all_markup(text: String) -> (String, Vec<ExpressiveTag>);
    #[host(function)]
    fn strip_all_markup(text: String) -> String;
    #[host(function)]
    fn strip_expr_markup(text: String) -> String;
    #[host(function)]
    fn split_expr_markup(text: String) -> (String, Vec<ExpressiveTag>);
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
