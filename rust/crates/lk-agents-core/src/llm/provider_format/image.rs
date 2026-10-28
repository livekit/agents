//! Pre-resolved image lookup and provider byte projections.
use lk_agents_abi::ImageContent;
pub use lk_agents_abi::llm::provider_format::image::{BYTES_TAG, ImageResolver, SerializedImage};
use serde_json::Value;

/// Keyed by `ImageContent.id`: one frame under two `ImageContent`s may have different resize hints.
#[derive(Debug, Clone, Default)]
pub struct PreResolved(pub std::collections::BTreeMap<String, SerializedImage>);

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
