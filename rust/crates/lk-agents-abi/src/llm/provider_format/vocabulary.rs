#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConversationRole {
    User,
    Assistant,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GeminiRole {
    User,
    Model,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ImageMime {
    #[host(rename = "image/jpeg")]
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[host(rename = "image/png")]
    #[serde(rename = "image/png")]
    Png,
    #[host(rename = "image/gif")]
    #[serde(rename = "image/gif")]
    Gif,
    #[host(rename = "image/webp")]
    #[serde(rename = "image/webp")]
    Webp,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BedrockImageFormat {
    Jpeg,
    Png,
    Gif,
    Webp,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultStatus {
    Success,
    Error,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesPhase {
    Commentary,
    FinalAnswer,
}

/// Gemini's function `behavior`; `UNSPECIFIED` is spelled by omitting it.
#[crate::host(record(module = "provider_data", name = "TOOL_BEHAVIOR", extract))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GeminiToolBehavior {
    #[host(rename = "UNSPECIFIED")]
    #[serde(rename = "UNSPECIFIED")]
    Unspecified,
    #[host(rename = "BLOCKING")]
    #[serde(rename = "BLOCKING")]
    Blocking,
    #[host(rename = "NON_BLOCKING")]
    #[serde(rename = "NON_BLOCKING")]
    NonBlocking,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MistralEntryObject {
    Entry,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    User,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssistantRole {
    Assistant,
}
