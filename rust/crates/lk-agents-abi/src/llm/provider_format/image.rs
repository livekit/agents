//! The image seam. Nothing here encodes: images resolve through a caller-supplied [`ImageResolver`],
//! keeping the `image` crate out of this tier.

use lk_agents_host_derive::Host;
use lk_agents_abi::{ImageContent, InferenceDetail};
use serde_json::Value;

/// `llm/utils.py:141`'s `SerializedImage`, as plain data.
///
/// `inference_detail` is [`InferenceDetail`] rather than a bare `str`. Stays here: the encoder
/// crate sits below this one.
// Read from the `serialize_image` dataclass, attribute by attribute.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
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

/// Keyed by `ImageContent.id`: one frame under two `ImageContent`s may have different resize hints.
#[derive(Debug, Clone, Default)]
pub struct PreResolved(pub std::collections::HashMap<String, SerializedImage>);

impl PreResolved {
    pub fn from_pairs<K: Into<String>>(
        pairs: impl IntoIterator<Item = (K, SerializedImage)>,
    ) -> Self {
        PreResolved(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
}

impl ImageResolver for PreResolved {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String> {
        self.0
            .get(&image.id)
            .cloned()
            .ok_or_else(|| format!("no SerializedImage was resolved for {}", image.id))
    }
}

/// Gemini's and Bedrock's image leaves are `bytes`: they emit `{"$lk_bytes_b64": "<base64>"}` and
/// the binding materialises it.
pub const BYTES_TAG: &str = "$lk_bytes_b64";

pub fn bytes_value(data: &[u8]) -> Value {
    serde_json::json!({ BYTES_TAG: b64_encode(data) })
}

/// `base64.b64encode`.
pub fn b64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// A `None` mime_type formats as the reference's `data:None;...`, which only a driver bug reaches.
pub(crate) fn data_uri(mime_type: Option<&str>, data: &[u8]) -> String {
    format!("data:{};base64,{}", mime_type.unwrap_or("None"), b64_encode(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a malformed data URI prefix.
    #[test]
    fn data_uri_has_the_exact_prefix_providers_parse() {
        assert_eq!(data_uri(Some("image/png"), b"foo"), "data:image/png;base64,Zm9v");
        // The reference f-strings a `None` mime type in literally; see the doc comment.
        assert_eq!(data_uri(None, b"foo"), "data:None;base64,Zm9v");
    }
}
