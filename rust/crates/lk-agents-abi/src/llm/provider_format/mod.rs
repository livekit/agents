//! Image, tool and formatting error boundary contracts.
pub mod host;
pub mod image;
pub mod tools;
pub use image::{BYTES_TAG, ImageResolver, SerializedImage};

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "provider_data",
    path = "livekit.agents.llm._provider_format",
    dataclass
))]
pub struct AnthropicFormatData {
    pub system_messages: Option<Vec<String>>,
}

#[crate::host(record(
    module = "provider_data",
    path = "livekit.agents.llm._provider_format",
    dataclass
))]
pub struct BedrockFormatData {
    pub system_messages: Option<Vec<String>>,
}

#[crate::host(record(
    module = "provider_data",
    path = "livekit.agents.llm._provider_format",
    dataclass
))]
pub struct GoogleFormatData {
    pub system_messages: Option<Vec<String>>,
}

#[crate::host(record(
    module = "provider_data",
    path = "livekit.agents.llm._provider_format",
    dataclass
))]
pub struct MistralFormatData {
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FormatError {
    /// `utils.py:117`'s assert.
    #[error("two messages landed in tool-call group {group_id:?}")]
    MultipleMessagesInGroup { group_id: String },

    #[error("cannot serialize image {image_id}: {reason}")]
    Image { image_id: String, reason: String },

    /// `aws.py:103`: Bedrock takes image bytes only.
    #[error("external_url is not supported by {provider}")]
    ExternalUrlUnsupported { provider: &'static str },

    /// `aws.py:118`.
    #[error("unsupported mime_type {mime_type:?} for {provider} images")]
    UnsupportedImageMime { provider: &'static str, mime_type: String },

    /// A `KeyError`/`TypeError` in the reference.
    #[error("tool schema is not shaped as expected: {detail}")]
    MalformedToolSchema { detail: String },
}

pub const MAX_DEPTH: usize = 100;

/// A strict-schema fault; the kind table picks the host exception. Plugins catch the reference's
/// exceptions per site; `AssertionError` (a non-dict `resolve_ref` segment) folds into `KeyError`.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StrictError {
    #[error("expected a JSON object at {path}")]
    #[host(kind = "TypeError")]
    NotAnObject { path: String },
    #[error("non-string $ref at {path}")]
    NonStringRef { path: String },
    #[error("unexpected $ref format {reference:?}; does not start with #/")]
    RefNotRooted { reference: String },
    #[error("$ref {reference:?} does not resolve: no object at segment {segment:?}")]
    #[host(kind = "KeyError")]
    RefUnresolvable { reference: String, segment: String },
    #[error("nullable union at {path} has no non-null object branch")]
    NullableUnionBranch { path: String },
    #[error("schema nests deeper than {MAX_DEPTH} levels at {path}; is a $ref self-referential?")]
    TooDeep { path: String },
}
