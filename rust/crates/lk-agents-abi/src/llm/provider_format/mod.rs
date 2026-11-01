//! Image, tool and formatting error boundary contracts.
mod error;
pub mod host;
pub mod image;
mod strict;
pub mod tools;
pub use error::FormatError;
pub use image::{BYTES_TAG, ImageResolver, SerializedImage};
pub use strict::{MAX_DEPTH, StrictError};

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
