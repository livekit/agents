//! One error for all five dialects; the binding picks the Python exception.

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
