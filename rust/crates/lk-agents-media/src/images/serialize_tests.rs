use lk_agents_abi::InferenceDetail;
use lk_agents_abi::utils::video::FakeFrame;

use super::*;

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
    ImageContent {
        id: "img_2".into(),
        image: ImageSource::Frame(FakeFrame::shared(7)),
        inference_width: width,
        inference_height: height,
        inference_detail: InferenceDetail::Low,
        mime_type: None,
    }
}

fn rgba(width: u32, height: u32) -> Vec<u8> {
    [4, 5, 6, 255].repeat((width * height) as usize)
}

// The mime type comes off the header's `image/png` part, a declared `mime_type` wins (reported
// as a mismatch only when they differ), and `""` is falsy in Python, so it defers to the header.
#[test]
fn a_declared_mime_type_wins_over_the_data_url_header_and_a_disagreement_is_reported() {
    let mismatch = |declared: &str| {
        Some(MimeMismatch { declared: declared.into(), header: "image/png".into() })
    };
    for (declared, want, want_mismatch) in [
        (None, "image/png", None),
        (Some("image/jpeg"), "image/jpeg", mismatch("image/jpeg")),
        (Some("image/png"), "image/png", None),
        (Some(""), "image/png", None),
    ] {
        let (out, got) =
            serialize_image(&url_image("data:image/png;base64,aGVsbG8=", declared), None).unwrap();
        assert_eq!(out.mime_type.as_deref(), Some(want), "{declared:?}");
        assert_eq!(out.data_bytes.as_deref(), Some(&b"hello"[..]), "the payload is decoded here");
        assert_eq!(out.external_url, None);
        assert_eq!(out.inference_detail, InferenceDetail::High);
        assert_eq!(got, want_mismatch, "{declared:?}");
    }
}

// CPython's non-strict `b64decode`: skips characters outside the alphabet, keeps `+` and `/`,
// stops at a completing `=`, refuses bad padding and non-ASCII; the header ends at the first comma.
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
            Ok(bytes) => {
                assert_eq!(got.unwrap().0.data_bytes.as_deref(), Some(bytes), "{url:?}")
            }
            Err(kind) => {
                assert!(format!("{:?}", got.unwrap_err()).starts_with(kind), "{url:?}")
            }
        }
    }
}

// A plain URL is passed through for the provider to fetch, including `https://data.example...`.
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

// Unsupported declared types such as `image/bmp` are refused locally, not by the provider.
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

// The frame arm always encodes JPEG; a PNG labelled `image/jpeg` is rejected by some providers.
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

// A zero inference dimension is falsy in Python, so it means "no resize", not a zero target.
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

// This crate cannot resolve a `MediaRef`, so a frame image without pixels is an error.
#[test]
fn a_frame_reference_without_its_resolved_buffer_is_an_error() {
    assert_eq!(serialize_image(&frame_image(None, None), None), Err(MediaError::UnresolvedFrame));
}
