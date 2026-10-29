//! The image seam. Nothing here encodes: images resolve through a caller-supplied [`ImageResolver`],
//! keeping the `image` crate out of this tier.

use lk_agents_abi::{ImageContent, InferenceDetail};

// `llm/utils.py`'s `SerializedImage`: what `serialize_image` returns. `inference_detail` is
// [`InferenceDetail`] rather than a bare `str`: an unknown value raises rather than sending the
// wrong fidelity. Stays here: the encoder crate sits below this one.
#[crate::host(record(module = "llm", path = "livekit.agents.llm.utils", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializedImage {
    pub inference_detail: InferenceDetail,
    // `None` is reachable via the URL arm; Google and AWS each have a fallback.
    pub mime_type: Option<String>,
    // In practice exactly one of this and `external_url` is set.
    #[host(default = None)]
    pub data_bytes: Option<Vec<u8>>,
    #[host(default = None)]
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

    /// Encodes `input` as the host asked (`encode_image`, `serialize_image`): CPU work for an
    /// admitted image worker.
    fn prepare(
        &self,
        input: crate::media::images::ImageInput,
        options: crate::media::images::ImageEncodeOptions,
    ) -> Result<crate::media::images::PreparedImage, String> {
        let _ = (input, options);
        Err("no image preparation in this process".into())
    }
}

/// Tagged bytes leaf consumed by generated provider-value conversions.
pub const BYTES_TAG: &str = "$lk_bytes_b64";
