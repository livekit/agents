//! Image preparation bounds: pixel and byte caps, resize validation and encoding of one frame or
//! data URL, as the admitted image worker runs it.

use std::io::Write;

use lk_agents_abi::MediaRef;
use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput, PreparedImage};
use lk_agents_abi::utils::video::VideoFrame;

use super::{EncodeOptions, ImageFormat, ResizeOptions, ResizeStrategy, RgbaFrame};

const MAX_PIXELS: u64 = 4096 * 4096;
pub const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;

/// Prepares `input`; a host frame is converted by `host_frame` (the host's RGBA conversion),
/// once its dimensions passed [`validate_dimensions`].
pub fn prepare(
    input: ImageInput,
    options: ImageEncodeOptions,
    host_frame: impl FnOnce(MediaRef) -> Result<VideoFrame, String>,
) -> Result<PreparedImage, String> {
    let frame = match input {
        ImageInput::Url { url, mime_type } => {
            if url.len() > MAX_IMAGE_BYTES {
                return Err("image URL exceeds 16 MiB".into());
            }
            return super::serialize_image_url(&url, mime_type.as_deref())
                .map_err(|e| e.to_string());
        }
        ImageInput::Frame(frame) => frame,
        ImageInput::HostFrame(frame) => host_frame(frame)?,
    };
    validate_dimensions(frame.width, frame.height)?;
    if frame.buffer_type != 0 {
        return Err("image encoder requires an RGBA frame".into());
    }
    let format = options.format.parse::<ImageFormat>()?;
    let resize_options =
        match (options.resize_width, options.resize_height, options.resize_strategy) {
            (Some(width), Some(height), Some(strategy)) => {
                validate_dimensions(width, height)?;
                let strategy = strategy.parse::<ResizeStrategy>()?;
                Some(ResizeOptions { width, height, strategy })
            }
            (None, None, None) => None,
            _ => {
                return Err(
                    "resize needs all three of width, height and strategy, or none of them".into(),
                );
            }
        };
    let frame = RgbaFrame { width: frame.width, height: frame.height, data: &frame.data };
    if let Some(resize) = resize_options {
        let (width, height) = super::resized_dimensions(frame.width, frame.height, resize)
            .map_err(|e| e.to_string())?;
        validate_dimensions(width, height)?;
    }
    let options = EncodeOptions { format, quality: options.quality, resize_options };
    let mut output = ImageBytes(Vec::new());
    super::encode_into(frame, &options, &mut output).map_err(|e| e.to_string())?;
    let mime = match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
    };
    Ok((Some(output.0), Some(mime.into()), None, None))
}

/// Between 1 and 4096 x 4096 pixels.
pub fn validate_dimensions(width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err("image dimensions must contain between 1 and 16777216 pixels".into());
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
