//! Image preparation's records and its host contracts; media prepares, the runtime admits.

use crate::MediaRef;
use crate::utils::video::VideoFrame;

pub enum ImageInput {
    Frame(VideoFrame),
    HostFrame(MediaRef),
    Url { url: String, mime_type: Option<String> },
}

#[derive(Clone)]
pub struct ImageEncodeOptions {
    pub format: String,
    pub quality: Option<u8>,
    pub resize_width: Option<u32>,
    pub resize_height: Option<u32>,
    pub resize_strategy: Option<String>,
}

impl Default for ImageEncodeOptions {
    fn default() -> Self {
        Self {
            format: "JPEG".into(),
            quality: Some(75),
            resize_width: None,
            resize_height: None,
            resize_strategy: None,
        }
    }
}

pub type PreparedImage = (Option<Vec<u8>>, Option<String>, Option<String>, Option<String>);

/// Reads a host video frame as an RGBA [`VideoFrame`], converting it when it is in another
/// buffer type. The host hands one to the runtime as its process starts.
#[crate::host(proxy(host = "livekit.agents.utils.images._FrameReader"))]
pub trait FrameReader: Send + Sync {
    #[host(call, name = "_core_rgba")]
    fn rgba(&self, frame: &MediaRef) -> Result<VideoFrame, crate::errors::HostError>;
}

/// The host hands the process its frame reader as it starts.
#[crate::host(functions)]
pub trait FrameReaderInstall {
    /// Hands the runtime the host's frame reader; images of host frames need it.
    #[host(function, name = "_core_install_frame_reader", pid = false)]
    fn install_frame_reader(reader: std::sync::Arc<dyn FrameReader>);
}

/// Frames as portable images, encoded on an admitted image worker.
#[crate::host(functions)]
pub trait ImageFunctions {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(function)]
    #[allow(clippy::too_many_arguments)]
    async fn encode_image(
        env: &Self::Env,
        frame: MediaRef,
        #[host(kw, default = "JPEG")] format: String,
        #[host(default = None)] quality: Option<u8>,
        #[host(default = None)] resize_width: Option<u32>,
        #[host(default = None)] resize_height: Option<u32>,
        #[host(default = None)] resize_strategy: Option<String>,
    ) -> Result<Vec<u8>, ImageError>;

    /// A frame as JPEG bytes, resized to its inference size when it has one.
    #[host(function)]
    async fn serialize_image_frame(
        env: &Self::Env,
        frame: MediaRef,
        #[host(default = None)] inference_width: Option<u32>,
        #[host(default = None)] inference_height: Option<u32>,
    ) -> Result<(Vec<u8>, String), ImageError>;

    /// A URL's inline bytes and MIME type, or the URL itself when it is external.
    #[host(function)]
    async fn serialize_image_url(
        env: &Self::Env,
        url: String,
        mime_type: Option<String>,
    ) -> Result<(Option<Vec<u8>>, Option<String>, Option<String>, Option<String>), ImageError>;
}

/// An image that could not be prepared.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ImageError(pub String);
