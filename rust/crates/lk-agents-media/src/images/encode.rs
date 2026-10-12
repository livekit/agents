//! `utils/images/image.py`: RGBA frame in, JPEG or PNG out. Kept from PIL: `convert("RGB")` drops
//! alpha rather than compositing, and resize dimensions are float-computed and truncated. Output
//! bytes differ (Catmull-Rom for BICUBIC, the `image` crate's JPEG writer); geometry does not.

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::imageops::{self, FilterType};
use image::{ExtendedColorType, ImageEncoder, RgbImage};

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
    /// Only the driver can resolve a `MediaRef`; the caller passed `ImageContent` without its buffer.
    #[error("image is a frame reference but no resolved RGBA buffer was supplied")]
    UnresolvedFrame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    #[default]
    Jpeg,
    Png,
}

impl ImageFormat {
    pub const ALL: [ImageFormat; 2] = [ImageFormat::Jpeg, ImageFormat::Png];

    /// PIL's format names, upper case.
    pub fn as_str(self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "JPEG",
            ImageFormat::Png => "PNG",
        }
    }
}

/// `Display` is the binding's message, verbatim; a typo must not default to JPEG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownImageFormat(pub String);

impl std::fmt::Display for UnknownImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown image format {:?}: expected 'JPEG' or 'PNG'", self.0)
    }
}

impl std::error::Error for UnknownImageFormat {}

impl std::str::FromStr for ImageFormat {
    type Err = UnknownImageFormat;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|f| f.as_str() == s)
            .ok_or_else(|| UnknownImageFormat(s.to_owned()))
    }
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

/// `ResizeOptions.strategy` (`image.py:47`). The `Scale*` strategies do not letterbox, so the
/// result is usually smaller (fit) or larger (cover) than the box in one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeStrategy {
    CenterAspectFit,
    CenterAspectCover,
    ScaleAspectFit,
    ScaleAspectCover,
    Skew,
}

impl ResizeStrategy {
    pub const ALL: [ResizeStrategy; 5] = [
        ResizeStrategy::CenterAspectFit,
        ResizeStrategy::CenterAspectCover,
        ResizeStrategy::ScaleAspectFit,
        ResizeStrategy::ScaleAspectCover,
        ResizeStrategy::Skew,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ResizeStrategy::CenterAspectFit => "center_aspect_fit",
            ResizeStrategy::CenterAspectCover => "center_aspect_cover",
            ResizeStrategy::ScaleAspectFit => "scale_aspect_fit",
            ResizeStrategy::ScaleAspectCover => "scale_aspect_cover",
            ResizeStrategy::Skew => "skew",
        }
    }
}

/// `Display` is the binding's message, verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownResizeStrategy(pub String);

impl std::fmt::Display for UnknownResizeStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown resize strategy {:?}: expected one of 'center_aspect_fit', \
             'center_aspect_cover', 'scale_aspect_fit', 'scale_aspect_cover', 'skew'",
            self.0
        )
    }
}

impl std::error::Error for UnknownResizeStrategy {}

impl std::str::FromStr for ResizeStrategy {
    type Err = UnknownResizeStrategy;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|r| r.as_str() == s)
            .ok_or_else(|| UnknownResizeStrategy(s.to_owned()))
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

/// `utils.images.encode(frame, options)` over tight RGBA, the options as the reference's
/// `Literal` strings. Unknown values raise rather than default.
#[lk_agents_abi::host(export(
    kw_only = "format",
    defaults(format = "JPEG")
))]
#[allow(clippy::too_many_arguments)]
pub fn encode_image(
    data: &[u8],
    width: u32,
    height: u32,
    format: &str,
    quality: Option<u8>,
    resize_width: Option<u32>,
    resize_height: Option<u32>,
    resize_strategy: Option<&str>,
) -> Result<Vec<u8>, String> {
    let format = format.parse::<ImageFormat>().map_err(|e| e.to_string())?;
    let resize_options = match (resize_width, resize_height, resize_strategy) {
        (Some(width), Some(height), Some(strategy)) => {
            let strategy = strategy.parse::<ResizeStrategy>().map_err(|e| e.to_string())?;
            Some(ResizeOptions { width, height, strategy })
        }
        (None, None, None) => None,
        _ => return Err("resize needs all three of width, height and strategy, or none of them".to_owned()),
    };
    let options = EncodeOptions { format, resize_options, quality };
    encode(RgbaFrame { width, height, data }, &options).map_err(|e| e.to_string())
}

/// `encode` (`image.py:73`).
pub fn encode(frame: RgbaFrame<'_>, options: &EncodeOptions) -> Result<Vec<u8>, MediaError> {
    let img = rgb_from_frame(frame)?;
    let resized = resize_image(img, options.resize_options)?;
    encode_buffer(&resized, options)
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
    for px in data.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    Ok(RgbImage::from_raw(width, height, rgb).expect("rgb buffer is width * height * 3"))
}

/// `_resize_image` (`image.py:99`).
fn resize_image(image: RgbImage, options: Option<ResizeOptions>) -> Result<RgbImage, MediaError> {
    let Some(opts) = options else { return Ok(image) };

    let (iw, ih) = (image.width(), image.height());
    if opts.width == 0 || opts.height == 0 {
        // The reference fails with `ZeroDivisionError` or inside `save` depending on the branch.
        return Err(MediaError::ZeroSizedTarget { width: opts.width, height: opts.height });
    }

    let (new_width, new_height) = match opts.strategy {
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
    };

    match opts.strategy {
        ResizeStrategy::CenterAspectFit | ResizeStrategy::CenterAspectCover => {
            // Black fill; a maskless `paste` clips the overhang, which is how "cover" crops.
            let mut canvas = RgbImage::new(opts.width, opts.height);
            // Defensive: the image crate does not promise a zero-sized paste is a no-op.
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
                // Extreme aspect ratios (1920x1 fitted to 512 wide -> 512x0); the reference dies in `save`.
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

fn encode_buffer(image: &RgbImage, options: &EncodeOptions) -> Result<Vec<u8>, MediaError> {
    let mut buffer = Vec::new();
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
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `width * height` RGBA pixels, all the same colour.
    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
        rgba.repeat((width * height) as usize)
    }

    fn frame<'a>(width: u32, height: u32, data: &'a [u8]) -> RgbaFrame<'a> {
        RgbaFrame { width, height, data }
    }

    fn resize(source: (u32, u32), target: (u32, u32), strategy: ResizeStrategy) -> RgbImage {
        let data = solid(source.0, source.1, [255, 0, 0, 255]);
        let image = rgb_from_frame(frame(source.0, source.1, &data)).unwrap();
        resize_image(image, Some(ResizeOptions { width: target.0, height: target.1, strategy }))
            .unwrap()
    }

    // Catches: `FromStr` and `as_str` disagreeing about a `Literal` spelling, a variant missing
    // from `ALL`, or a typo defaulting instead of refusing -- defaulting to JPEG produces a
    // plausible image in the wrong format, and a wrong strategy resizes every frame an LLM sees.
    // The two `match`es keep the lists exhaustive.
    #[test]
    fn image_format_and_resize_strategy_round_trip_and_refuse_unknown_spellings() {
        for f in ImageFormat::ALL {
            match f {
                ImageFormat::Jpeg | ImageFormat::Png => {}
            }
            assert_eq!(f.as_str().parse::<ImageFormat>(), Ok(f));
        }
        assert!("jpeg".parse::<ImageFormat>().is_err(), "PIL's names are upper case and the case is not folded");

        for s in ResizeStrategy::ALL {
            match s {
                ResizeStrategy::CenterAspectFit
                | ResizeStrategy::CenterAspectCover
                | ResizeStrategy::ScaleAspectFit
                | ResizeStrategy::ScaleAspectCover
                | ResizeStrategy::Skew => {}
            }
            assert_eq!(s.as_str().parse::<ResizeStrategy>(), Ok(s));
        }
        assert!("stretch".parse::<ResizeStrategy>().is_err());
    }

    // Catches: compositing alpha instead of discarding it. PIL's `convert("RGB")` on an RGBA image
    // drops the channel -- a fully transparent red pixel stays (255, 0, 0). Anything that blends
    // against the black canvas turns every screenshot the anthropic and browser plugins send
    // darker, and does it silently.
    #[test]
    fn alpha_is_discarded_not_composited() {
        let data = [255, 0, 0, 0, 0, 255, 0, 128];
        let image = rgb_from_frame(frame(2, 1, &data)).unwrap();
        assert_eq!(image.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(image.get_pixel(1, 0).0, [0, 255, 0]);
    }

    // Catches: accepting a buffer whose length does not match the dimensions -- a strided frame,
    // or an RGB buffer passed where RGBA was meant. `chunks_exact` would silently drop the
    // remainder and `from_raw` would then fail an assertion deep inside the image crate; the frame
    // has to be rejected at the door with both numbers named.
    #[test]
    fn a_buffer_that_is_not_exactly_width_times_height_times_four_is_rejected() {
        let short = vec![0u8; 15];
        assert_eq!(
            rgb_from_frame(frame(2, 2, &short)),
            Err(MediaError::FrameBufferMismatch { width: 2, height: 2, expected: 16, actual: 15 })
        );
        let long = vec![0u8; 20];
        assert!(rgb_from_frame(frame(2, 2, &long)).is_err(), "a strided buffer is not tightened");
    }

    // Catches: letting a zero-dimension frame through to the encoder. The reference dies inside
    // PIL with a `ZeroDivisionError` or a `SystemError` depending on which branch it reaches
    // first; either way there is nothing to encode, and `f64` division would hand us `inf` here
    // rather than a diagnosis.
    #[test]
    fn a_zero_sized_frame_is_refused_before_any_division() {
        let empty: Vec<u8> = Vec::new();
        assert_eq!(
            rgb_from_frame(frame(0, 10, &empty)),
            Err(MediaError::ZeroSizedFrame { width: 0, height: 10 })
        );
    }

    // Catches: dropping the `None` short-circuit and running the resize machinery with a
    // zero-by-zero target, which is what `resize_options=None` would otherwise look like.
    #[test]
    fn no_resize_options_returns_the_frame_untouched() {
        let data = solid(3, 2, [1, 2, 3, 255]);
        let image = rgb_from_frame(frame(3, 2, &data)).unwrap();
        let out = resize_image(image, None).unwrap();
        assert_eq!(out.dimensions(), (3, 2));
    }

    // Catches: the `scale_*` aspect comparison done in integer arithmetic or with its branches
    // swapped -- `fit` must land inside the box from either orientation (an oversized image is
    // billed), `cover` outside it, and `<` vs `>` is the only difference between them; `skew`
    // respecting the aspect ratio (it is the one strategy that must not); and a comparison
    // loosened to `>=`: at an exact tie `int()` truncation makes the losing branch come out square.
    #[test]
    fn the_scale_strategies_land_inside_outside_or_exactly_on_the_box() {
        use ResizeStrategy::*;
        for (source, target, strategy, want) in [
            ((200, 100), (50, 50), ScaleAspectFit, (50, 25)),
            ((100, 200), (50, 50), ScaleAspectFit, (25, 50)),
            ((200, 100), (50, 50), ScaleAspectCover, (100, 50)),
            ((100, 200), (50, 50), ScaleAspectCover, (50, 100)),
            ((200, 100), (33, 77), Skew, (33, 77)),
            ((7, 7), (61, 60), ScaleAspectCover, (61, 60)),
            ((7, 7), (61, 60), ScaleAspectFit, (61, 60)),
        ] {
            assert_eq!(resize(source, target, strategy).dimensions(), want, "{source:?} -> {target:?} {strategy:?}");
        }
    }

    // Catches: the two halves of the degenerate-scale rule conflated, or guarded in one
    // orientation only. A 192x1 frame fitted into 51x51 scales to 51x0 (and 1x192 to 0x51): for
    // `scale_aspect_fit` that is an empty image and an error; for `center_aspect_fit` the
    // reference pastes nothing onto the black canvas and succeeds.
    #[test]
    fn a_scale_that_rounds_a_dimension_to_zero_errors_but_a_centered_one_stays_black() {
        for (w, h, scaled) in [(192, 1, (51, 0)), (1, 192, (0, 51))] {
            let data = solid(w, h, [255, 255, 255, 255]);
            let image = rgb_from_frame(frame(w, h, &data)).unwrap();
            let opts = |strategy| Some(ResizeOptions { width: 51, height: 51, strategy });
            assert_eq!(
                resize_image(image.clone(), opts(ResizeStrategy::ScaleAspectFit)),
                Err(MediaError::DegenerateResize {
                    strategy: ResizeStrategy::ScaleAspectFit,
                    source_width: w,
                    source_height: h,
                    width: scaled.0,
                    height: scaled.1,
                })
            );
            let centered = resize_image(image, opts(ResizeStrategy::CenterAspectFit)).unwrap();
            assert_eq!(centered.dimensions(), (51, 51));
            assert!(centered.pixels().all(|px| px.0 == [0, 0, 0]), "nothing was pasted, so it is black");
        }
    }

    // Catches: returning the scaled image instead of the canvas for the `center_*` strategies.
    // Both must come out exactly the requested size -- that is the entire difference from the
    // `scale_*` pair -- and `fit` must letterbox the leftover with black rather than stretching.
    #[test]
    fn center_aspect_fit_letterboxes_onto_a_black_canvas_of_the_requested_size() {
        let out = resize((200, 100), (50, 50), ResizeStrategy::CenterAspectFit);
        assert_eq!(out.dimensions(), (50, 50));
        // 50x25 pasted at y = (50 - 25) // 2 = 12.
        assert_eq!(out.get_pixel(25, 0).0, [0, 0, 0], "the letterbox is black");
        assert_eq!(out.get_pixel(25, 11).0, [0, 0, 0]);
        assert_eq!(
            out.get_pixel(25, 12).0,
            [255, 0, 0],
            "the image starts at the floor-halved offset"
        );
        assert_eq!(out.get_pixel(25, 36).0, [255, 0, 0]);
        assert_eq!(out.get_pixel(25, 37).0, [0, 0, 0]);
    }

    // Catches: `center_aspect_cover` leaving a letterbox. It scales up so the canvas is fully
    // covered and the overhang is cropped by the paste, so no pixel may stay black.
    #[test]
    fn center_aspect_cover_leaves_no_black_canvas_showing() {
        let out = resize((200, 100), (50, 50), ResizeStrategy::CenterAspectCover);
        assert_eq!(out.dimensions(), (50, 50));
        assert!(
            out.pixels().all(|px| px.0 == [255, 0, 0]),
            "a cover crop that shows canvas has been scaled with the wrong branch"
        );
    }

    // Catches: `(target - actual) / 2` instead of `// 2`. Rust truncates towards zero and Python
    // floors, so every *odd* overhang -- the normal case for a cover crop -- is placed one pixel
    // to the right and down of where the reference puts it. Even overhangs agree, which is why a
    // test with round numbers would miss this entirely.
    #[test]
    fn the_paste_offset_floors_rather_than_truncating_when_the_overhang_is_odd() {
        assert_eq!(centered_offset(51, 100), -25, "Python: (51 - 100) // 2 == -25, not -24");
        assert_eq!(centered_offset(50, 25), 12, "positive offsets floor too: 25 // 2 == 12");
        assert_eq!(centered_offset(100, 50), 25, "an even gap is unambiguous");
    }

    // Catches: treating a zero target as "no resize", or dividing by it. Every strategy needs a
    // non-empty box: three of the five divide by one of the two dimensions, and the other two hand
    // PIL an empty image that fails at save time.
    #[test]
    fn a_zero_sized_target_is_refused_for_every_strategy() {
        for strategy in [
            ResizeStrategy::Skew,
            ResizeStrategy::CenterAspectFit,
            ResizeStrategy::CenterAspectCover,
            ResizeStrategy::ScaleAspectFit,
            ResizeStrategy::ScaleAspectCover,
        ] {
            let data = solid(4, 4, [9, 9, 9, 255]);
            let image = rgb_from_frame(frame(4, 4, &data)).unwrap();
            assert_eq!(
                resize_image(image, Some(ResizeOptions { width: 10, height: 0, strategy })),
                Err(MediaError::ZeroSizedTarget { width: 10, height: 0 }),
                "{strategy:?} accepted a zero-height target"
            );
        }
    }

    // Catches: the two ways this line is normally "cleaned up". `value * num / den` in integers
    // gives 61 where the reference's `int(7 * (61 / 7))` gives 60, because the double nearest to
    // 61/7 is below it; and `round()` gives 67 where the reference gives 66. Both are off-by-one
    // pixel errors on every resize that hits them, and neither shows up on round numbers.
    #[test]
    fn scaled_dimensions_truncate_a_float_product_rather_than_rounding_an_integer_one() {
        assert_eq!(scaled(7, 61, 7), 60, "integer arithmetic would say 61");
        assert_eq!(scaled(100, 200, 300), 66, "rounding would say 67");
        assert_eq!(scaled(10, 1, 100), 0, "a dimension really can round away to nothing");
    }

    // Catches: any of the four aspect comparisons being loosened to `>=` or `<=`. Equal ratios are
    // not a tie: `int()` truncation makes the two branches disagree by a pixel even when the
    // ratios match exactly -- a 7x7 frame into a 61x61 box is 61x60 one way and 60x61 the other --
    // so which side of the comparison wins is observable, and round numbers never show it.
    #[test]
    fn the_aspect_comparisons_are_strict_so_an_exact_tie_takes_the_second_branch() {
        for strategy in [ResizeStrategy::CenterAspectFit, ResizeStrategy::CenterAspectCover] {
            let out = resize((7, 7), (61, 61), strategy);
            assert_eq!(out.dimensions(), (61, 61), "{strategy:?} returns the canvas");
            assert_eq!(
                out.get_pixel(60, 0).0,
                [255, 0, 0],
                "{strategy:?} scaled to 61x60, so the last column is covered"
            );
            assert_eq!(
                out.get_pixel(0, 60).0,
                [0, 0, 0],
                "{strategy:?} scaled to 61x60, so the last row is not"
            );
        }
    }

    // Catches: writing PNG when JPEG was asked for, or ignoring `EncodeOptions.format` entirely.
    // The two are told apart by their magic bytes, which is also what every provider sniffs.
    #[test]
    fn the_format_option_selects_the_container() {
        let data = solid(8, 8, [10, 200, 30, 255]);
        let jpeg = encode(frame(8, 8, &data), &EncodeOptions::default()).unwrap();
        assert_eq!(&jpeg[..3], &[0xFF, 0xD8, 0xFF], "JPEG SOI");

        let png = encode(
            frame(8, 8, &data),
            &EncodeOptions { format: ImageFormat::Png, ..EncodeOptions::default() },
        )
        .unwrap();
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A], "PNG signature");
    }

    // Catches: dropping the `quality` option on the floor, and picking a default other than 75.
    // `quality=None` means "do not pass the kwarg", and PIL's own default is 75 -- so `None` and
    // `Some(75)` have to produce the same bytes, while a different quality must not.
    #[test]
    fn jpeg_quality_is_honoured_and_none_means_seventy_five() {
        let data: Vec<u8> = (0..64 * 64 * 4).map(|i| (i % 251) as u8).collect();
        let at = |quality| {
            encode(frame(64, 64, &data), &EncodeOptions { quality, ..EncodeOptions::default() })
                .unwrap()
        };
        assert_eq!(at(None), at(Some(75)), "None must land on the reference's default");
        assert!(at(Some(5)).len() < at(Some(95)).len(), "quality changed nothing");
    }

    // Catches: passing `quality` to the PNG encoder. The reference guards the kwarg with
    // `if options.format == "JPEG"`, and PNG is lossless -- a quality-sensitive PNG path would
    // mean the option had been wired into the wrong encoder.
    #[test]
    fn quality_is_ignored_for_png() {
        let data = solid(8, 8, [1, 2, 3, 255]);
        let png = |quality| {
            encode(
                frame(8, 8, &data),
                &EncodeOptions { format: ImageFormat::Png, quality, resize_options: None },
            )
            .unwrap()
        };
        assert_eq!(png(Some(1)), png(None));
    }

}
