//! The native model consumes the shared preparation service on its admitted image worker.

use lk_agents_abi::llm::provider_format::{ImageResolver, SerializedImage};
use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput};
use lk_agents_abi::{ImageContent, ImageSource};

pub const MAX_IMAGE_URL_BYTES: usize = crate::images::MAX_IMAGE_BYTES;

pub struct NativeImages;

impl ImageResolver for NativeImages {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String> {
        let input = match &image.image {
            ImageSource::Url(url) => {
                if url.len() > MAX_IMAGE_URL_BYTES {
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
        let (data_bytes, mime_type, external_url, overridden_mime) =
            crate::images::prepare_on_worker(
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
