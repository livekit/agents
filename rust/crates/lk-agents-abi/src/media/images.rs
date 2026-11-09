use std::sync::Arc;

use crate::utils::video::{RtcVideoFrame, VideoFrame};

pub enum ImageInput {
    Frame(VideoFrame),
    HostFrame(Arc<dyn RtcVideoFrame>),
    Url { url: String, mime_type: Option<String> },
}

/// PIL's format names, upper case. A typo must not default to JPEG.
#[crate::host(wire_enum(msg = format!("unknown image format {s:?}: expected 'JPEG' or 'PNG'")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    #[default]
    #[host(rename = "JPEG")]
    Jpeg,
    #[host(rename = "PNG")]
    Png,
}

/// `ResizeOptions.strategy` (`image.py:47`). The `Scale*` strategies do not letterbox, so the
/// result is usually smaller (fit) or larger (cover) than the box in one dimension.
#[crate::host(wire_enum(msg = format!(
    "unknown resize strategy {s:?}: expected one of 'center_aspect_fit', 'center_aspect_cover', \
     'scale_aspect_fit', 'scale_aspect_cover', 'skew'"
)))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeStrategy {
    CenterAspectFit,
    CenterAspectCover,
    ScaleAspectFit,
    ScaleAspectCover,
    Skew,
}

#[derive(Clone)]
pub struct ImageEncodeOptions {
    pub format: ImageFormat,
    pub quality: Option<u8>,
    pub resize_width: Option<u32>,
    pub resize_height: Option<u32>,
    pub resize_strategy: Option<ResizeStrategy>,
}

impl Default for ImageEncodeOptions {
    fn default() -> Self {
        Self {
            format: ImageFormat::Jpeg,
            quality: Some(75),
            resize_width: None,
            resize_height: None,
            resize_strategy: None,
        }
    }
}

/// A prepared image: inline bytes with their MIME type, or an external URL passed through.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PreparedImage {
    pub data: Option<Vec<u8>>,
    pub mime_type: Option<String>,
    pub external_url: Option<String>,
    /// A `data:` URL header's MIME type that a declared one overrode.
    pub overridden_mime: Option<String>,
}

/// Frames as portable images, encoded on an admitted image worker.
#[crate::host(functions)]
pub trait ImageFunctions {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(function)]
    async fn encode_image(
        env: &Self::Env,
        frame: Arc<dyn RtcVideoFrame>,
        #[host(kw, default = "JPEG")] format: ImageFormat,
        #[host(default = None)] quality: Option<u8>,
        #[host(default = None)] resize_width: Option<u32>,
        #[host(default = None)] resize_height: Option<u32>,
        #[host(default = None)] resize_strategy: Option<ResizeStrategy>,
    ) -> Result<Vec<u8>, ImageError>;

    /// A frame as JPEG bytes, resized to its inference size when it has one.
    #[host(function)]
    async fn serialize_image_frame(
        env: &Self::Env,
        frame: Arc<dyn RtcVideoFrame>,
        inference_detail: crate::InferenceDetail,
        #[host(default = None)] inference_width: Option<u32>,
        #[host(default = None)] inference_height: Option<u32>,
    ) -> Result<crate::llm::provider_format::image::SerializedImage, ImageError>;

    /// A URL's inline bytes and MIME type, or the URL itself when it is external.
    #[host(function)]
    async fn serialize_image_url(
        env: &Self::Env,
        url: String,
        inference_detail: crate::InferenceDetail,
        mime_type: Option<String>,
    ) -> Result<crate::llm::provider_format::image::SerializedImage, ImageError>;
}

/// An image that could not be prepared.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ImageError(pub String);

// Public data contracts. Python behavior remains in its adapters.

/// How a frame is resized as it is encoded.
#[crate::host(record(module = "image_options", path = "livekit.agents.utils.media", dataclass))]
pub struct ResizeOptions {
    pub width: i64,
    pub height: i64,
    /// ``center_aspect_fit`` letterboxes into the size, ``center_aspect_cover`` fills it
    ///     cropping, ``scale_aspect_fit``/``scale_aspect_cover`` keep the aspect ratio inside/around it,
    ///     ``skew`` resizes to it exactly.
    pub strategy: ResizeStrategy,
}

/// How a frame is encoded.
#[crate::host(record(module = "image_options", path = "livekit.agents.utils.media", dataclass))]
pub struct EncodeOptions {
    #[host(default = "JPEG")]
    pub format: ImageFormat,
    #[host(default = None)]
    pub resize_options: Option<ResizeOptions>,
    /// JPEG quality, 0-100.
    #[host(default = 75)]
    pub quality: Option<i64>,
}
