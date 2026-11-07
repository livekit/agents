//! `utils/images/image.py`: RGBA frame in, JPEG or PNG out. Kept from PIL: `convert("RGB")` drops
//! alpha rather than compositing, and resize dimensions are float-computed and truncated. Output
//! bytes differ (Catmull-Rom for BICUBIC, the `image` crate's JPEG writer); geometry does not.

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::imageops::{self, FilterType};
use image::{ExtendedColorType, ImageEncoder, RgbImage};
pub use lk_agents_abi::media::images::{ImageFormat, ResizeStrategy};
use lk_agents_abi::utils::video::VideoBufferType;

/// PIL's default `resample` for an RGB image is `Resampling.BICUBIC`; Catmull-Rom is that kernel.
const FILTER: FilterType = FilterType::CatmullRom;

/// libjpeg's default; the reference reaches it both via `quality=None` and by omitting the kwarg.
const DEFAULT_JPEG_QUALITY: u8 = 75;

/// Which Python exception each becomes is the binding's decision.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MediaError {
    #[error("frame is {width}x{height}; a zero dimension has no pixels to encode")]
    ZeroSizedFrame { width: u32, height: u32 },
    #[error("frame buffer is {actual} bytes; {width}x{height} RGBA needs exactly {expected}")]
    FrameBufferMismatch { width: u32, height: u32, expected: usize, actual: usize },
    #[error("resize target is {width}x{height}; a zero dimension has no pixels to encode")]
    ZeroSizedTarget { width: u32, height: u32 },
    #[error(
        "{strategy:?} scales {source_width}x{source_height} to {width}x{height}, which has no pixels"
    )]
    DegenerateResize {
        strategy: ResizeStrategy,
        source_width: u32,
        source_height: u32,
        width: u32,
        height: u32,
    },
    /// JPEG cannot express a dimension above 65535, and the encoder can fail on write.
    #[error("encoder rejected the image: {0}")]
    Encode(String),
    #[error("`data:` URL has no comma separating the header from the payload")]
    MalformedDataUrl,
    // The reference's `ValueError` text, verbatim.
    #[error("Unsupported mime_type {0}. Must be jpeg, png, webp, or gif")]
    UnsupportedMimeType(String),
    /// `binascii.Error`'s own text (`Incorrect padding`, ...).
    #[error("{0}")]
    InvalidBase64(String),
    /// The caller passed a frame's `ImageContent` without its RGBA buffer.
    #[error("image is a frame reference but no resolved RGBA buffer was supplied")]
    UnresolvedFrame,
    #[error("image URL exceeds 16 MiB")]
    UrlTooLarge,
    #[error("image dimensions must contain between 1 and 16777216 pixels")]
    DimensionsOutOfRange,
    #[error("image encoder requires an RGBA frame, not {0:?}")]
    NotRgba(VideoBufferType),
    #[error("resize needs all three of width, height and strategy, or none of them")]
    PartialResize,
    /// The host could not hand over the frame's RGBA pixels.
    #[error("{0}")]
    HostFrame(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeOptions {
    pub format: ImageFormat,
    pub resize_options: Option<ResizeOptions>,
    /// Clamped by the encoder as libjpeg clamps it; `None` resolves to 75. PIL rejects values
    /// outside 0-100, so `u8` loses nothing.
    pub quality: Option<u8>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            format: ImageFormat::Jpeg,
            resize_options: None,
            quality: Some(DEFAULT_JPEG_QUALITY),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeOptions {
    pub width: u32,
    pub height: u32,
    pub strategy: ResizeStrategy,
}

/// A frame the host already converted to packed RGBA (`image.py:90` does it inline).
#[derive(Debug, Clone, Copy)]
pub struct RgbaFrame<'a> {
    pub width: u32,
    pub height: u32,
    /// Exactly `width * height * 4` bytes, no stride padding: ignoring a tail would skew the image.
    pub data: &'a [u8],
}

/// `encode` (`image.py:73`).
pub fn encode(frame: RgbaFrame<'_>, options: &EncodeOptions) -> Result<Vec<u8>, MediaError> {
    let mut buffer = Vec::new();
    encode_into(frame, options, &mut buffer)?;
    Ok(buffer)
}

/// Encodes into supplied storage, so the runtime can enforce an encoded-byte limit while writing.
pub fn encode_into(
    frame: RgbaFrame<'_>,
    options: &EncodeOptions,
    output: impl std::io::Write,
) -> Result<(), MediaError> {
    let img = rgb_from_frame(frame)?;
    let resized = resize_image(img, options.resize_options)?;
    encode_buffer(&resized, options, output)
}

/// `_image_from_frame` (`image.py:89`). Alpha is dropped, not composited.
fn rgb_from_frame(frame: RgbaFrame<'_>) -> Result<RgbImage, MediaError> {
    let RgbaFrame { width, height, data } = frame;
    if width == 0 || height == 0 {
        return Err(MediaError::ZeroSizedFrame { width, height });
    }

    let pixels = (width as usize) * (height as usize);
    let expected = pixels * 4;
    if data.len() != expected {
        return Err(MediaError::FrameBufferMismatch {
            width,
            height,
            expected,
            actual: data.len(),
        });
    }

    let mut rgb = Vec::with_capacity(pixels * 3);
    for px in data.as_chunks::<4>().0 {
        rgb.extend_from_slice(&px[..3]);
    }
    Ok(RgbImage::from_raw(width, height, rgb).expect("rgb buffer is width * height * 3"))
}

/// `_resize_image` (`image.py:99`).
fn resize_image(image: RgbImage, options: Option<ResizeOptions>) -> Result<RgbImage, MediaError> {
    let Some(opts) = options else { return Ok(image) };

    let (iw, ih) = (image.width(), image.height());
    let (new_width, new_height) = resized_dimensions(iw, ih, opts)?;

    match opts.strategy {
        ResizeStrategy::CenterAspectFit | ResizeStrategy::CenterAspectCover => {
            let mut canvas = RgbImage::new(opts.width, opts.height);
            if new_width > 0 && new_height > 0 {
                let resized = imageops::resize(&image, new_width, new_height, FILTER);
                let x = centered_offset(opts.width, new_width);
                let y = centered_offset(opts.height, new_height);
                imageops::replace(&mut canvas, &resized, x, y);
            }
            Ok(canvas)
        }
        _ => {
            if new_width == 0 || new_height == 0 {
                return Err(MediaError::DegenerateResize {
                    strategy: opts.strategy,
                    source_width: iw,
                    source_height: ih,
                    width: new_width,
                    height: new_height,
                });
            }
            Ok(imageops::resize(&image, new_width, new_height, FILTER))
        }
    }
}

/// Intermediate resize dimensions, also used to admit the runtime's pixel allocation.
pub fn resized_dimensions(iw: u32, ih: u32, opts: ResizeOptions) -> Result<(u32, u32), MediaError> {
    if iw == 0 || ih == 0 {
        return Err(MediaError::ZeroSizedFrame { width: iw, height: ih });
    }
    if opts.width == 0 || opts.height == 0 {
        // The reference fails with `ZeroDivisionError` or inside `save` depending on the branch.
        return Err(MediaError::ZeroSizedTarget { width: opts.width, height: opts.height });
    }

    Ok(match opts.strategy {
        ResizeStrategy::Skew => (opts.width, opts.height),
        ResizeStrategy::CenterAspectFit => {
            // Start with assuming the new image is narrower than the original.
            if ratio(opts.width, opts.height) > ratio(iw, ih) {
                (scaled(iw, opts.height, ih), opts.height)
            } else {
                (opts.width, scaled(ih, opts.width, iw))
            }
        }
        ResizeStrategy::CenterAspectCover => {
            // Start with assuming the new image is shorter than the original.
            if ratio(opts.height, opts.width) > ratio(ih, iw) {
                (scaled(iw, opts.height, ih), opts.height)
            } else {
                (opts.width, scaled(ih, opts.width, iw))
            }
        }
        ResizeStrategy::ScaleAspectCover => {
            // Start with assuming width is the limiting dimension.
            let candidate = scaled(ih, opts.width, iw);
            if candidate < opts.height {
                (scaled(iw, opts.height, ih), opts.height)
            } else {
                (opts.width, candidate)
            }
        }
        ResizeStrategy::ScaleAspectFit => {
            // Start with assuming width is the limiting dimension.
            let candidate = scaled(ih, opts.width, iw);
            if candidate > opts.height {
                (scaled(iw, opts.height, ih), opts.height)
            } else {
                (opts.width, candidate)
            }
        }
    })
}

/// `(target - actual) // 2`: Python's floor division, which differs from `/` on odd negative overhangs.
fn centered_offset(target: u32, actual: u32) -> i64 {
    (i64::from(target) - i64::from(actual)).div_euclid(2)
}

/// True division, as the reference computes `a / b`.
fn ratio(a: u32, b: u32) -> f64 {
    f64::from(a) / f64::from(b)
}

/// `int(value * (num / den))`: float multiply then truncate; integer math or `round()` is off by a pixel.
fn scaled(value: u32, num: u32, den: u32) -> u32 {
    (f64::from(value) * (f64::from(num) / f64::from(den))) as u32
}

fn encode_buffer(
    image: &RgbImage,
    options: &EncodeOptions,
    mut buffer: impl std::io::Write,
) -> Result<(), MediaError> {
    let (w, h) = (image.width(), image.height());
    let result = match options.format {
        ImageFormat::Jpeg => {
            // `quality` is only passed for JPEG; omitting it lands on 75 too.
            let quality = options.quality.unwrap_or(DEFAULT_JPEG_QUALITY);
            JpegEncoder::new_with_quality(&mut buffer, quality).encode(
                image.as_raw(),
                w,
                h,
                ExtendedColorType::Rgb8,
            )
        }
        ImageFormat::Png => {
            PngEncoder::new(&mut buffer).write_image(image.as_raw(), w, h, ExtendedColorType::Rgb8)
        }
    };
    result.map_err(|err| MediaError::Encode(err.to_string()))?;
    Ok(())
}

#[cfg(test)]
#[path = "encode_tests.rs"]
mod tests;
