//! Image preparation bounds: pixel and byte caps, resize validation and encoding of one frame or
//! data URL, as the admitted image worker runs it. The provider resolver uses the same bounds
//! and reads a host frame's pixels from the frame itself.

use super::{EncodeOptions, ImageFormat, MediaError, ResizeOptions, RgbaFrame};
use lk_agents_abi::llm::provider_format::{ImageResolver, SerializedImage};
use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput, PreparedImage, ResizeStrategy};
use lk_agents_abi::utils::video::VideoBufferType;
use lk_agents_abi::{ImageContent, ImageSource};
use std::io::Write;

const MAX_PIXELS: u64 = 4096 * 4096;
pub const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;

/// Prepares `input`: CPU work for an admitted image worker. A host frame hands over its RGBA
/// pixels first.
pub fn prepare(
    input: ImageInput,
    options: ImageEncodeOptions,
) -> Result<PreparedImage, MediaError> {
    let frame = match input {
        ImageInput::Url { url, mime_type } => {
            if url.len() > MAX_IMAGE_BYTES {
                return Err(MediaError::UrlTooLarge);
            }
            return super::serialize_image_url(&url, mime_type.as_deref());
        }
        ImageInput::Frame(frame) => frame,
        ImageInput::HostFrame(frame) => {
            frame.rgba().map_err(|e| MediaError::HostFrame(e.to_string()))?
        }
    };
    validate_dimensions(frame.width, frame.height)?;
    if frame.buffer_type != VideoBufferType::Rgba {
        return Err(MediaError::NotRgba(frame.buffer_type));
    }
    let resize_options =
        match (options.resize_width, options.resize_height, options.resize_strategy) {
            (Some(width), Some(height), Some(strategy)) => {
                validate_dimensions(width, height)?;
                Some(ResizeOptions { width, height, strategy })
            }
            (None, None, None) => None,
            _ => return Err(MediaError::PartialResize),
        };
    let rgba = RgbaFrame { width: frame.width, height: frame.height, data: &frame.data };
    if let Some(resize) = resize_options {
        let (width, height) = super::resized_dimensions(rgba.width, rgba.height, resize)?;
        validate_dimensions(width, height)?;
    }
    let format = options.format;
    let options = EncodeOptions { format, quality: options.quality, resize_options };
    let mut output = ImageBytes(Vec::new());
    super::encode_into(rgba, &options, &mut output)?;
    let mime = match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
    };
    Ok(PreparedImage { data: Some(output.0), mime_type: Some(mime.into()), ..Default::default() })
}

/// Between 1 and 4096 x 4096 pixels.
pub fn validate_dimensions(width: u32, height: u32) -> Result<(), MediaError> {
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(MediaError::DimensionsOutOfRange);
    }
    Ok(())
}

/// The encoder's output, refused past [`MAX_IMAGE_BYTES`].
struct ImageBytes(Vec<u8>);

impl Write for ImageBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.0.len().saturating_add(bytes.len()) > MAX_IMAGE_BYTES {
            return Err(std::io::Error::other("encoded image exceeds 16 MiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// How a realtime model prepares its images: on the image worker it already runs on.
#[derive(Clone, Copy, Default)]
pub struct Images;

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
        let prepared = prepare(
            input,
            ImageEncodeOptions {
                format: ImageFormat::Jpeg,
                resize_width: resize.map(|(width, _)| width),
                resize_height: resize.map(|(_, height)| height),
                resize_strategy: resize.map(|_| ResizeStrategy::ScaleAspectFit),
                ..Default::default()
            },
        )
        .map_err(|cause| format!("image {}: {cause}", image.id))?;
        if let Some(header) = prepared.overridden_mime {
            log::warn!(target: lk_agents_abi::LOG_TARGET, declared_mime = image.mime_type.as_deref().unwrap_or_default(), header_mime = header.as_str(); "image MIME declaration overrides data URL header");
        }
        Ok(SerializedImage {
            data_bytes: prepared.data,
            mime_type: prepared.mime_type,
            external_url: prepared.external_url,
            inference_detail: image.inference_detail,
        })
    }
}
