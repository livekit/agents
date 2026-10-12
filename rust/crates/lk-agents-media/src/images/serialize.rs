//! `serialize_image` (`llm/utils.py:148`): the producer half of the media seam. The type lives in
//! the format crate, which must not reach an encoder.

use lk_agents_abi::llm::provider_format::SerializedImage;
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
                (Some(data), _) => SerializedImage::inline(data, mime_type.unwrap_or_default(), detail),
                (None, url) => SerializedImage::external(url.unwrap_or_default(), mime_type, detail),
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

/// `serialize_image`'s `str` arm, for a host that owns `ImageContent`:
/// `(data_bytes, mime_type, external_url, header_mime_if_overridden)`.
///
/// Exactly one of `data_bytes`/`external_url` is set. The fourth element is the `data:` header's
/// mime type when a declared `mime_type` overrode it -- the host logs the reference's warning.
#[lk_agents_abi::host(export(pid = false))]
#[allow(clippy::type_complexity)]
pub fn serialize_image_url(
    url: &str,
    #[host(required)] mime_type: Option<&str>,
) -> Result<(Option<Vec<u8>>, Option<String>, Option<String>, Option<String>), MediaError> {
    let (data, mime, external, mismatch) = serialize_url_parts(url, mime_type)?;
    Ok((data, mime, external, mismatch.map(|m| m.header)))
}

/// `serialize_image`'s `rtc.VideoFrame` arm over tight RGBA: `(data_bytes, mime_type)`.
///
/// The resize decision is here: `scale_aspect_fit` to the inference size when both are truthy.
#[lk_agents_abi::host(export)]
pub fn serialize_image_frame(
    data: &[u8],
    width: u32,
    height: u32,
    inference_width: Option<u32>,
    inference_height: Option<u32>,
) -> Result<(Vec<u8>, String), MediaError> {
    let bytes = encode_frame(RgbaFrame { width, height, data }, inference_width, inference_height)?;
    Ok((bytes, FRAME_MIME_TYPE.to_owned()))
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
mod tests {
    use std::sync::Arc;

    use lk_agents_abi::{InferenceDetail, MediaRef, RefRegistry};

    use super::*;

    struct Noop;
    impl RefRegistry for Noop {
        fn release(&self, _id: u64) {}
    }

    fn url_image(url: &str, mime_type: Option<&str>) -> ImageContent {
        ImageContent {
            id: "img_1".into(),
            image: ImageSource::Url(url.into()),
            inference_width: None,
            inference_height: None,
            inference_detail: InferenceDetail::High,
            mime_type: mime_type.map(str::to_owned),
        }
    }

    fn frame_image(width: Option<u32>, height: Option<u32>) -> ImageContent {
        let registry: Arc<dyn RefRegistry> = Arc::new(Noop);
        ImageContent {
            id: "img_2".into(),
            image: ImageSource::Frame(MediaRef::new(7, &registry)),
            inference_width: width,
            inference_height: height,
            inference_detail: InferenceDetail::Low,
            mime_type: None,
        }
    }

    fn rgba(width: u32, height: u32) -> Vec<u8> {
        [4, 5, 6, 255].repeat((width * height) as usize)
    }

    // Catches: the mime type parsed off the wrong side of the header (`image/png`, not
    // `image/png;base64` or `data`), the header winning over a declared `mime_type` (the reference
    // warns, then uses the declared one; the reported mismatch is what the binding logs), a
    // mismatch reported when the two agree (a warning on every careful caller's image), and `""`
    // read as a declaration -- it is falsy in Python, so it defers to the header.
    #[test]
    fn a_declared_mime_type_wins_over_the_data_url_header_and_a_disagreement_is_reported() {
        let mismatch = |declared: &str| Some(MimeMismatch { declared: declared.into(), header: "image/png".into() });
        for (declared, want, want_mismatch) in [
            (None, "image/png", None),
            (Some("image/jpeg"), "image/jpeg", mismatch("image/jpeg")),
            (Some("image/png"), "image/png", None),
            (Some(""), "image/png", None),
        ] {
            let (out, got) = serialize_image(&url_image("data:image/png;base64,aGVsbG8=", declared), None).unwrap();
            assert_eq!(out.mime_type.as_deref(), Some(want), "{declared:?}");
            assert_eq!(out.data_bytes.as_deref(), Some(&b"hello"[..]), "the payload is decoded here");
            assert_eq!(out.external_url, None);
            assert_eq!(out.inference_detail, InferenceDetail::High);
            assert_eq!(got, want_mismatch, "{declared:?}");
        }
    }

    // Catches: a payload decoder that is not CPython's non-strict `b64decode`. It discards
    // characters outside the alphabet (a line-wrapped URL works) but keeps `+` and `/` (two of the
    // 64 digits -- stripping them corrupts silently), stops at the `=` that completes a quantum,
    // refuses bad padding and non-ASCII rather than handing a provider a corrupt image, and the
    // header is split off at the first comma, which must exist.
    #[test]
    fn a_data_url_payload_decodes_the_way_binascii_decodes_it() {
        for (url, want) in [
            ("data:image/png;base64,+/8=", Ok(&[0xFB, 0xFF][..])),
            ("data:image/png;base64,aGVs\n bG8=", Ok(&b"hello"[..])),
            ("data:image/png;base64,aGVsbG8=aGVs", Ok(&b"hello"[..])),
            ("data:image/png;base64,aGVsbG8", Err("InvalidBase64")),
            ("data:image/png;base64,aGVsb", Err("InvalidBase64")),
            ("data:image/png;base64,aGVs\u{e9}bG8=", Err("InvalidBase64")),
            ("data:image/png;base64", Err("MalformedDataUrl")),
        ] {
            let got = serialize_image(&url_image(url, None), None);
            match want {
                Ok(bytes) => assert_eq!(got.unwrap().0.data_bytes.as_deref(), Some(bytes), "{url:?}"),
                Err(kind) => assert!(format!("{:?}", got.unwrap_err()).starts_with(kind), "{url:?}"),
            }
        }
    }

    // Catches: treating every string as a data URL (a plain URL is passed through for the provider
    // to fetch, with no bytes and no mime sniffed off the extension), including gating on
    // `starts_with("data")` -- `https://data.example.com/..` is an ordinary URL.
    #[test]
    fn a_plain_url_is_passed_through_with_no_bytes_and_no_mime_inference() {
        for url in ["https://example.com/cat.jpg", "https://data.example.com/x.png"] {
            let (out, mismatch) = serialize_image(&url_image(url, None), None).unwrap();
            assert_eq!(out.external_url.as_deref(), Some(url));
            assert_eq!(out.data_bytes, None);
            assert_eq!(out.mime_type, None, "nothing sniffs the extension");
            assert_eq!(mismatch, None);
        }
    }

    // Catches: dropping the supported-types gate, or applying it to the header when the declared
    // type is the one that wins. `image/bmp` reaches a provider as an unsupported inline image and
    // comes back as an opaque API error; the reference refuses it locally.
    #[test]
    fn an_unsupported_data_url_mime_type_is_refused() {
        let image = url_image("data:image/bmp;base64,aGVsbG8=", None);
        assert_eq!(
            serialize_image(&image, None),
            Err(MediaError::UnsupportedMimeType("image/bmp".into()))
        );

        let overridden = url_image("data:image/png;base64,aGVsbG8=", Some("image/bmp"));
        assert_eq!(
            serialize_image(&overridden, None),
            Err(MediaError::UnsupportedMimeType("image/bmp".into())),
            "the gate applies to the type that wins, not the header"
        );
    }

    // Catches: forgetting that the frame arm always produces JPEG. The reference hardcodes both
    // the encode format and the mime type, and they have to agree -- a PNG labelled `image/jpeg`
    // is rejected by Bedrock and mis-decoded by Anthropic.
    #[test]
    fn a_frame_is_encoded_as_jpeg_and_labelled_as_jpeg() {
        let data = rgba(8, 8);
        let (out, _) = serialize_image(
            &frame_image(None, None),
            Some(RgbaFrame { width: 8, height: 8, data: &data }),
        )
        .unwrap();
        assert_eq!(out.mime_type.as_deref(), Some("image/jpeg"));
        assert_eq!(&out.data_bytes.unwrap()[..3], &[0xFF, 0xD8, 0xFF]);
        assert_eq!(out.inference_detail, InferenceDetail::Low);
    }

    // Catches: translating `if image.inference_width and image.inference_height` into
    // `if both.is_some()`. A zero is falsy in Python, so it means "no resize" -- read as
    // `Some(0)` it becomes a zero-sized target and turns a working call into an error.
    #[test]
    fn a_zero_inference_dimension_means_no_resize_not_a_zero_sized_one() {
        let data = rgba(8, 8);
        let frame = || RgbaFrame { width: 8, height: 8, data: &data };

        let unresized = serialize_image(&frame_image(None, None), Some(frame())).unwrap().0;
        let zero = serialize_image(&frame_image(Some(0), Some(4)), Some(frame())).unwrap().0;
        assert_eq!(zero.data_bytes, unresized.data_bytes, "a zero hint must be ignored");

        let zero_height = serialize_image(&frame_image(Some(4), Some(0)), Some(frame())).unwrap().0;
        assert_eq!(zero_height.data_bytes, unresized.data_bytes, "either side may be the zero");

        let one_sided = serialize_image(&frame_image(Some(4), None), Some(frame())).unwrap().0;
        assert_eq!(one_sided.data_bytes, unresized.data_bytes, "both hints are required");

        let resized = serialize_image(&frame_image(Some(4), Some(4)), Some(frame())).unwrap().0;
        assert_ne!(resized.data_bytes, unresized.data_bytes, "both hints present must resize");
    }

    // Catches: silently encoding an empty image when the driver forgot to resolve the `MediaRef`.
    // §1.11 gives the pure tier `release` and never `resolve`, so this crate cannot go and get the
    // buffer itself; the only correct response is to say so.
    #[test]
    fn a_frame_reference_without_its_resolved_buffer_is_an_error() {
        assert_eq!(
            serialize_image(&frame_image(None, None), None),
            Err(MediaError::UnresolvedFrame)
        );
    }
}
