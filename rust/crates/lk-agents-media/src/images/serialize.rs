//! `serialize_image` (`llm/utils.py:148`): the producer half of the media seam. The type lives in
//! the format crate, which must not reach an encoder.

use lk_agents_abi::llm::provider_format::SerializedImage;
use lk_agents_abi::media::images::PreparedImage;
use lk_agents_abi::{ImageContent, ImageSource};

use super::encode::{EncodeOptions, MediaError, ResizeOptions, ResizeStrategy, RgbaFrame, encode};

/// The four types the reference admits from a `data:` URL (`utils.py:165`); plain URLs are never checked.
pub const SUPPORTED_DATA_URL_MIME_TYPES: [&str; 4] =
    ["image/jpeg", "image/png", "image/webp", "image/gif"];

/// The encode never consults `EncodeOptions.format`; the reference hardcodes the label to match.
pub const FRAME_MIME_TYPE: &str = "image/jpeg";

/// A declared `mime_type` that disagrees with the `data:` header; the declared one wins
/// (`utils.py:157`) and the binding logs the reference's warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimeMismatch {
    pub declared: String,
    pub header: String,
}

/// `serialize_image` (`llm/utils.py:148`), minus the `_cache` lookup. `frame` is ignored for a
/// URL image, as the reference ignores the inference size there.
pub fn serialize_image(
    image: &ImageContent,
    frame: Option<RgbaFrame<'_>>,
) -> Result<(SerializedImage, Option<MimeMismatch>), MediaError> {
    let detail = image.inference_detail;
    match &image.image {
        ImageSource::Url(url) => {
            let (data, mime_type, external, mismatch) =
                serialize_url_parts(url, image.mime_type.as_deref())?;
            let out = match (data, external) {
                (Some(data), _) => {
                    SerializedImage::inline(data, mime_type.unwrap_or_default(), detail)
                }
                (None, url) => {
                    SerializedImage::external(url.unwrap_or_default(), mime_type, detail)
                }
            };
            Ok((out, mismatch))
        }
        ImageSource::Frame(_) => {
            let frame = frame.ok_or(MediaError::UnresolvedFrame)?;
            let data = encode_frame(frame, image.inference_width, image.inference_height)?;
            Ok((SerializedImage::inline(data, FRAME_MIME_TYPE, detail), None))
        }
    }
}

/// `serialize_image`'s `str` arm, for a host that owns `ImageContent`. Exactly one of `data` and
/// `external_url` is set; the host logs the reference's warning for an `overridden_mime`.
pub fn serialize_image_url(
    url: &str,
    mime_type: Option<&str>,
) -> Result<PreparedImage, MediaError> {
    let (data, mime_type, external_url, mismatch) = serialize_url_parts(url, mime_type)?;
    Ok(PreparedImage { data, mime_type, external_url, overridden_mime: mismatch.map(|m| m.header) })
}

fn encode_frame(
    frame: RgbaFrame<'_>,
    inference_width: Option<u32>,
    inference_height: Option<u32>,
) -> Result<Vec<u8>, MediaError> {
    let mut options = EncodeOptions::default();
    // A truthiness test: zero on either side means "no resize".
    if let (Some(width), Some(height)) = (inference_width, inference_height)
        && width > 0
        && height > 0
    {
        options.resize_options =
            Some(ResizeOptions { width, height, strategy: ResizeStrategy::ScaleAspectFit });
    }
    encode(frame, &options)
}

type UrlParts = (Option<Vec<u8>>, Option<String>, Option<String>, Option<MimeMismatch>);

fn serialize_url_parts(url: &str, declared: Option<&str>) -> Result<UrlParts, MediaError> {
    if !url.starts_with("data:") {
        return Ok((None, declared.map(str::to_owned), Some(url.to_owned()), None));
    }
    // `header, b64_data = image.image.split(",", 1)` -- no comma is an unpacking ValueError.
    let (header, payload) = url.split_once(',').ok_or(MediaError::MalformedDataUrl)?;

    // Decoded before the mime type is read, as the reference does.
    let data = b64decode(payload)?;

    // `header.split(";")[0].split(":")[1]`; an empty result (`data:,x`) then fails the gate below.
    let header_mime =
        header.split(';').next().unwrap_or("").split(':').nth(1).unwrap_or("").to_owned();

    // Truthiness again: an empty declared mime type defers to the header.
    let (mime_type, mismatch) = match declared.filter(|m| !m.is_empty()) {
        Some(declared) if declared != header_mime => (
            declared.to_owned(),
            Some(MimeMismatch { declared: declared.to_owned(), header: header_mime }),
        ),
        _ => (header_mime, None),
    };

    if !SUPPORTED_DATA_URL_MIME_TYPES.contains(&mime_type.as_str()) {
        return Err(MediaError::UnsupportedMimeType(mime_type));
    }
    Ok((Some(data), Some(mime_type), None, mismatch))
}

/// CPython's non-strict `base64.b64decode`, ported: non-ASCII is refused, characters outside the
/// alphabet are skipped, a completing `=` ends decoding, trailing bits are accepted, and a
/// dangling quantum raises `binascii`'s messages. Decoded whether or not the header said `;base64`.
fn b64decode(payload: &str) -> Result<Vec<u8>, MediaError> {
    if !payload.is_ascii() {
        return Err(MediaError::InvalidBase64(
            "string argument should contain only ASCII characters".into(),
        ));
    }
    fn digit(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(payload.len() / 4 * 3);
    let (mut quad_pos, mut left, mut pads) = (0u8, 0u8, 0u8);
    for c in payload.bytes() {
        if c == b'=' {
            if quad_pos >= 2 {
                pads += 1;
                if quad_pos + pads >= 4 {
                    return Ok(out);
                }
            }
            continue;
        }
        let Some(v) = digit(c) else { continue };
        pads = 0;
        match quad_pos {
            0 => left = v,
            1 => {
                out.push((left << 2) | (v >> 4));
                left = v & 0x0f;
            }
            2 => {
                out.push((left << 4) | (v >> 2));
                left = v & 0x03;
            }
            _ => out.push((left << 6) | v),
        }
        quad_pos = (quad_pos + 1) % 4;
    }
    match quad_pos {
        0 => Ok(out),
        1 => Err(MediaError::InvalidBase64(format!(
            "Invalid base64-encoded string: number of data characters ({}) cannot be 1 more \
             than a multiple of 4",
            out.len() / 3 * 4 + 1
        ))),
        _ => Err(MediaError::InvalidBase64("Incorrect padding".into())),
    }
}

#[cfg(test)]
#[path = "serialize_tests.rs"]
mod tests;
