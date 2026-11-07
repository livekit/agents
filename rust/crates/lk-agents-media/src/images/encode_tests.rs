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
        assert_eq!(
            resize(source, target, strategy).dimensions(),
            want,
            "{source:?} -> {target:?} {strategy:?}"
        );
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
        assert!(
            centered.pixels().all(|px| px.0 == [0, 0, 0]),
            "nothing was pasted, so it is black"
        );
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
    assert_eq!(out.get_pixel(25, 12).0, [255, 0, 0], "the image starts at the floor-halved offset");
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
