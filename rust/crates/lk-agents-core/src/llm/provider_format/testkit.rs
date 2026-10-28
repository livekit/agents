//! Builders shared by the provider test modules. Nothing here is compiled into the library.

use std::time::{Duration, SystemTime};

use lk_agents_abi::{
    AgentHandoffItem, ChatContent, ChatContext, ChatItem, ChatMessage, FunctionCall,
    FunctionCallOutput, ImageContent, ImageSource, InferenceDetail, Role,
};

use crate::llm::provider_format::image::{PreResolved, SerializedImage};

pub fn t(secs: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
}

pub fn ctx(items: Vec<ChatItem>) -> ChatContext {
    ChatContext::new(items)
}

pub fn msg(id: &str, role: Role, parts: &[&str]) -> ChatItem {
    ChatMessage::new(id, role, parts.iter().map(|p| ChatContent::Text((*p).into())).collect(), t(1))
        .into()
}

pub fn msg_with(id: &str, role: Role, content: Vec<ChatContent>) -> ChatItem {
    ChatMessage::new(id, role, content, t(1)).into()
}

pub fn call(id: &str, call_id: &str, name: &str, arguments: &str) -> ChatItem {
    FunctionCall {
        id: id.into(),
        call_id: call_id.into(),
        arguments: arguments.into(),
        name: name.into(),
        created_at: t(1),
        extra: Default::default(),
        group_id: None,
    }
    .into()
}

pub fn out(id: &str, call_id: &str, name: &str, output: &str) -> ChatItem {
    let mut o = FunctionCallOutput::new(id, call_id, output, false, t(1));
    o.name = name.into();
    o.into()
}

pub fn handoff(id: &str) -> ChatItem {
    AgentHandoffItem {
        id: id.into(),
        old_agent_id: None,
        new_agent_id: "b".into(),
        created_at: t(1),
    }
    .into()
}

pub fn text_content(s: &str) -> ChatContent {
    ChatContent::Text(s.into())
}

/// An image part plus a resolver that knows how to serialize it.
pub fn image(id: &str) -> ChatContent {
    ChatContent::Image(ImageContent {
        id: id.into(),
        image: ImageSource::Url(format!("https://example.test/{id}.png")),
        inference_width: None,
        inference_height: None,
        inference_detail: InferenceDetail::Auto,
        mime_type: None,
    })
}

/// Resolves every `image(id)` to inline JPEG bytes; anything else is a miss.
pub fn inline_images(ids: &[&str]) -> PreResolved {
    PreResolved::from_pairs(ids.iter().map(|id| {
        (
            id.to_string(),
            SerializedImage::inline(b"\xff\xd8\xff".to_vec(), "image/jpeg", InferenceDetail::Auto),
        )
    }))
}

/// Resolves every `image(id)` to its external URL.
pub fn url_images(ids: &[&str]) -> PreResolved {
    PreResolved::from_pairs(ids.iter().map(|id| {
        (
            id.to_string(),
            SerializedImage::external(
                format!("https://example.test/{id}.png"),
                Some("image/png".into()),
                InferenceDetail::High,
            ),
        )
    }))
}

/// A resolver for contexts with no images at all.
pub fn no_images() -> PreResolved {
    PreResolved::default()
}
