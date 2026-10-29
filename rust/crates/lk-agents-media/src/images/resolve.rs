//! Images prepared for a provider: host frames read through the host's [`FrameReader`]
//! (decision 94). Synchronous: the caller runs it on an image worker.

use std::sync::Arc;

use lk_agents_abi::llm::provider_format::{ImageResolver, SerializedImage};
use lk_agents_abi::media::images::{FrameReader, ImageEncodeOptions, ImageInput, PreparedImage};
use lk_agents_abi::{ImageContent, ImageSource};

use super::{MAX_IMAGE_BYTES, prepare};

/// Image preparation with the host's frame reader, when the process has one.
#[derive(Clone, Default)]
pub struct Images {
    frames: Option<Arc<dyn FrameReader>>,
}

impl Images {
    pub fn new(frames: Option<Arc<dyn FrameReader>>) -> Self {
        Self { frames }
    }

    /// Prepares `input` here, a host frame read through the host's reader: CPU work for an
    /// admitted image worker.
    pub fn prepare(
        &self,
        input: ImageInput,
        options: ImageEncodeOptions,
    ) -> Result<PreparedImage, String> {
        prepare(input, options, |frame| {
            let frames = self.frames.as_ref().ok_or("host frame conversion requires a host")?;
            frames.rgba(&frame).map_err(|e| e.to_string())
        })
    }
}

/// How a realtime model prepares its images: on the image worker it already runs on.
impl ImageResolver for Images {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String> {
        let input = match &image.image {
            ImageSource::Url(url) => {
                if url.len() > MAX_IMAGE_BYTES {
                    return Err("realtime image URL exceeds 16 MiB".into());
                }
                ImageInput::Url { url: url.clone(), mime_type: image.mime_type.clone() }
            }
            ImageSource::Frame(frame) => ImageInput::HostFrame(frame.clone()),
        };
        let resize = image
            .inference_width
            .zip(image.inference_height)
            .filter(|(width, height)| *width > 0 && *height > 0);
        let (data_bytes, mime_type, external_url, overridden_mime) = self
            .prepare(
                input,
                ImageEncodeOptions {
                    format: "JPEG".into(),
                    resize_width: resize.map(|(width, _)| width),
                    resize_height: resize.map(|(_, height)| height),
                    resize_strategy: resize.map(|_| "scale_aspect_fit".into()),
                    ..Default::default()
                },
            )
            .map_err(|cause| format!("image {}: {cause}", image.id))?;
        if let Some(header) = overridden_mime {
            log::warn!(target: "livekit.agents", declared_mime = image.mime_type.as_deref().unwrap_or_default(), header_mime = header.as_str(); "image MIME declaration overrides data URL header");
        }
        Ok(SerializedImage {
            data_bytes,
            mime_type,
            external_url,
            inference_detail: image.inference_detail,
        })
    }
}
