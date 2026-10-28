//! The image seam. Nothing here encodes: images resolve through a caller-supplied [`ImageResolver`],
//! keeping the `image` crate out of this tier.

use lk_agents_abi::{ImageContent, InferenceDetail};
use lk_agents_host_derive::Host;

/// `llm/utils.py:141`'s `SerializedImage`, as plain data.
///
/// `inference_detail` is [`InferenceDetail`] rather than a bare `str`. Stays here: the encoder
/// crate sits below this one.
// Read from the `serialize_image` dataclass, attribute by attribute.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[host(host_type = "livekit.agents.llm.utils.SerializedImage")]
#[binding(extract)]
pub struct SerializedImage {
    // Its string: an unknown value raises rather than sending the wrong fidelity.
    pub inference_detail: lk_agents_abi::InferenceDetail,
    /// `None` is reachable via the URL arm; Google and AWS each have a fallback.
    pub mime_type: Option<String>,
    /// In practice exactly one of this and `external_url` is set.
    pub data_bytes: Option<Vec<u8>>,
    pub external_url: Option<String>,
}

impl SerializedImage {
    pub fn inline(
        data_bytes: Vec<u8>,
        mime_type: impl Into<String>,
        inference_detail: InferenceDetail,
    ) -> Self {
        SerializedImage {
            inference_detail,
            mime_type: Some(mime_type.into()),
            data_bytes: Some(data_bytes),
            external_url: None,
        }
    }

    /// `mime_type` is whatever the caller set on `ImageContent`, often nothing.
    pub fn external(
        url: impl Into<String>,
        mime_type: Option<String>,
        inference_detail: InferenceDetail,
    ) -> Self {
        SerializedImage {
            inference_detail,
            mime_type,
            data_bytes: None,
            external_url: Some(url.into()),
        }
    }
}

/// Implemented above the purity line. A `String` error so the encoder need not know `FormatError`.
pub trait ImageResolver {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String>;
}

/// Tagged bytes leaf consumed by generated provider-value conversions.
pub const BYTES_TAG: &str = "$lk_bytes_b64";
