use super::*;
use std::f64::consts::TAU;

/// 20 ms of a sine at `hz`, interleaved across `nch` identical channels.
fn sine(hz: f64, rate: u32, ms: usize, nch: u32) -> Vec<i16> {
    let n = rate as usize * ms / 1000;
    let mut v = Vec::with_capacity(n * nch as usize);
    for i in 0..n {
        let s = ((TAU * hz * i as f64 / f64::from(rate)).sin() * 16000.0) as i16;
        for _ in 0..nch {
            v.push(s);
        }
    }
    v
}

/// Zero crossings -> frequency. Crude, but it is the property that matters: if the filter or
/// the rate ratio is wrong the tone moves or aliases, and that is exactly what this catches.
fn dominant_hz(pcm: &[i16], rate: u32) -> f64 {
    let crossings = pcm.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count();
    crossings as f64 / 2.0 * f64::from(rate) / pcm.len() as f64
}

#[test]
fn zero_rates_and_channels_are_rejected() {
    for (input, output, channels) in [(0, 16_000, 1), (48_000, 0, 1), (48_000, 16_000, 0)] {
        assert!(Resampler::new(input, output, channels, Quality::Medium).is_err());
    }
}

#[test]
fn a_1khz_tone_survives_48k_to_16k_at_the_same_pitch() {
    // The reason this crate exists: the VAD and EOT models were tuned on audio that went
    // through soxr. A wrong ratio or a broken filter shows up here as a shifted or aliased
    // tone, which no length assertion would notice.
    let mut r = Resampler::new(48_000, 16_000, 1, Quality::Medium).unwrap();
    let mut out = Vec::new();
    for _ in 0..10 {
        out.extend_from_slice(r.push(&sine(1000.0, 48_000, 20, 1)).unwrap());
    }
    out.extend_from_slice(r.flush().unwrap());

    let hz = dominant_hz(&out, 16_000);
    assert!((hz - 1000.0).abs() < 25.0, "tone moved to {hz} Hz");
}

#[test]
fn flush_drains_the_tail_and_then_has_nothing_left() {
    let mut r = Resampler::new(48_000, 16_000, 1, Quality::Medium).unwrap();
    r.push(&sine(440.0, 48_000, 20, 1)).unwrap();
    assert!(!r.flush().unwrap().is_empty(), "the filter tail was dropped");
    // flush also clears, so the second one has no history to drain.
    assert!(r.flush().unwrap().is_empty());
}

#[test]
fn concurrent_sessions_neither_crash_nor_corrupt_each_others_audio() {
    // The B25 shape, run as a stress test with **no test-side lock**: libsoxr's FFT cache is
    // grown by whichever thread first constructs a resampler needing a longer DFT, while other
    // threads are mid-`soxr_process` reading it. Without `FFT_CACHE` that is a SIGSEGV or an
    // `assert()` inside libsoxr (measured 4 runs in 12 for the whole suite), or -- the worse
    // failure -- a 1 kHz tone silently coming out at 1.0-1.7 kHz (1 run in 30). With it, 0 in
    // 100 consecutive runs of this binary.
    //
    // Shape matters more than volume, and the obvious strengthenings backfire:
    //   - batching (each thread builds several) made it *less* likely, because after round
    //     one the coefficient cache is warm and later creates skip the racy grow entirely;
    //   - without the barrier, spawn overhead serialises the threads and the test passes
    //     even with the lock removed (verified: 0 failures in 6 runs).
    // So: many threads, one construction each, a distinct output rate per thread so the
    // filters differ, all released at one instant -- and one *victim* already pushing before
    // they are released, which is the reader whose table moves. Every thread checks the pitch
    // of its own output, since corruption is a value error and not a crash. Probabilistic by
    // nature: treat a single green run as weak evidence and loop it.
    //
    // `cargo nextest` gives every test its own process and cannot see any of this; the threads
    // are spawned here so it reproduces under both runners.
    const N: usize = 24;
    let media_pool = crate::pool::MediaPool::new(Default::default()).unwrap();
    let gate = std::sync::Arc::new(std::sync::Barrier::new(N + 1));

    let victim = {
        let gate = std::sync::Arc::clone(&gate);
        media_pool
            .try_spawn(crate::pool::JobClass::Decoder, move || {
                let mut r = Resampler::new(48_000, 16_000, 1, Quality::Medium).unwrap();
                let tone = sine(1000.0, 48_000, 20, 1);
                let mut out = Vec::new();
                // Warm, so the victim is mid-stream when the creators are released.
                for _ in 0..5 {
                    out.extend_from_slice(r.push(&tone).unwrap());
                }
                gate.wait();
                for _ in 0..200 {
                    out.extend_from_slice(r.push(&tone).unwrap());
                }
                out.extend_from_slice(r.flush().unwrap());
                let hz = dominant_hz(&out, 16_000);
                assert!((hz - 1000.0).abs() < 25.0, "victim's tone moved to {hz} Hz");
            })
            .unwrap()
    };

    let mut creators = Vec::new();
    for i in 0..N {
        let gate = std::sync::Arc::clone(&gate);
        creators.push(
            media_pool
                .try_spawn(crate::pool::JobClass::Decoder, move || {
                    gate.wait();
                    // A distinct rate per thread: every one designs its own filter.
                    let out_rate = 8_000 + (i as u32) * 1_000;
                    let mut r = Resampler::new(48_000, out_rate, 1, Quality::Medium)
                        .expect("construction must not race");
                    let tone = sine(1000.0, 48_000, 20, 1);
                    let mut out = Vec::new();
                    for _ in 0..10 {
                        out.extend_from_slice(r.push(&tone).unwrap());
                    }
                    out.extend_from_slice(r.flush().unwrap());
                    let hz = dominant_hz(&out, out_rate);
                    assert!((hz - 1000.0).abs() < 25.0, "tone at {out_rate} Hz moved to {hz} Hz");
                })
                .unwrap(),
        );
    }
    for h in creators {
        h.wait().expect("a construction thread panicked or crashed");
    }
    victim.wait().expect("the victim's audio was corrupted or it crashed");
}

#[test]
fn stereo_sizing_is_per_frame_not_per_sample() {
    // Every other test in this file is mono, which makes `* nch` and `/ nch` invisible --
    // multiplying by one. That is why the buffer arithmetic went unpinned: the tests could
    // not distinguish the correct expression from four mutations of it.
    //
    // Catches: sizing `out` in samples where it should be frames (or the reverse). Too small
    // and soxr silently emits less than it had ready -- a slow drift, never an error.
    let mut r = Resampler::new(48_000, 16_000, 2, Quality::Medium).unwrap();
    let mut total = 0usize;
    for _ in 0..50 {
        let out = r.push(&sine(440.0, 48_000, 20, 2)).unwrap();
        assert!(out.len().is_multiple_of(2), "output is not whole stereo frames");
        total += out.len();
    }
    total += r.flush().unwrap().len();

    // 1 s in at 48 k stereo -> 16 000 frames -> 32 000 interleaved samples.
    let expected = 50 * 48_000 * 20 / 1000 / 3 * 2;
    let drift = (total as i64 - expected as i64).abs();
    assert!(drift <= 4, "stereo produced {total}, expected ~{expected} (drift {drift})");

    // An odd sample count is half a frame. soxr would consume it and swap L/R for the rest of
    // the stream, which is silent and permanent.
    let err = r.push(&[1, 2, 3]).unwrap_err();
    assert!(matches!(err, ResampleError::PartialFrame { len: 3, num_channels: 2 }));
}

#[test]
fn a_long_stream_does_not_drift_once_the_filter_is_holding_history() {
    // Catches the `soxr_delay` term in `push`'s sizing. `soxr_delay` only becomes non-zero
    // once the filter has history, so a short run cannot see it -- 100 pushes was not enough
    // to pin it, which is why four mutations of this expression survived.
    //
    // Under-allocate and soxr emits less than it has ready on each call; the shortfall
    // accumulates and never errors.
    let mut r = Resampler::new(44_100, 16_000, 1, Quality::High).unwrap();
    let mut total = 0usize;
    for _ in 0..500 {
        total += r.push(&sine(440.0, 44_100, 20, 1)).unwrap().len();
    }
    total += r.flush().unwrap().len();

    // 10 s in -> 160 000 out. A ratio that is not a round fraction is the point: it keeps the
    // filter's residual non-zero for the whole run.
    let expected = 500 * 16_000 * 20 / 1000;
    let drift = (total as i64 - expected as i64).abs();
    assert!(drift <= 8, "produced {total} over 10 s, expected ~{expected} (drift {drift})");
}

#[test]
fn flushing_a_long_stereo_tail_stays_inside_the_buffer() {
    // `flush` tells soxr how many FRAMES the output buffer holds: `self.out.len() / nch`.
    // Get that wrong upward and soxr writes past the end of a `Vec<i16>` -- heap corruption,
    // not an error return. The other stereo test does not reach it because its tail is a few
    // samples; a long filter (VeryHigh) holds a large residual, which is what fills the tail.
    //
    // Catches: `/ nch` becoming `* nch` in `flush`, which is a 4x over-report at stereo.
    let mut r = Resampler::new(48_000, 44_100, 2, Quality::VeryHigh).unwrap();
    for _ in 0..100 {
        r.push(&sine(440.0, 48_000, 20, 2)).unwrap();
    }
    let tail = r.flush().unwrap();
    assert!(tail.len().is_multiple_of(2), "tail is not whole stereo frames");
    // A residual, not a whole second of audio: if `cap` were over-reported soxr would happily
    // hand back far more than the filter actually held.
    assert!(tail.len() < 44_100, "flush returned {} samples, which is not a tail", tail.len());
}

#[test]
fn independent_resamplers_produce_identical_pcm() {
    let input = sine(731.0, 48_000, 1000, 1);
    let mut first = Resampler::new(48_000, 24_000, 1, Quality::Medium).unwrap();
    let mut second = Resampler::new(48_000, 24_000, 1, Quality::Medium).unwrap();
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for chunk in input.chunks(997) {
        left.extend_from_slice(first.push(chunk).unwrap());
        right.extend_from_slice(second.push(chunk).unwrap());
    }
    left.extend_from_slice(first.flush().unwrap());
    right.extend_from_slice(second.flush().unwrap());
    assert_eq!(left.len(), 24_000);
    assert_eq!(left.len(), right.len());
    let different = left.iter().zip(&right).position(|(a, b)| a != b);
    assert_eq!(different, None, "identical input depended on the native handle's seed");
}

#[test]
fn the_quality_ordinals_are_the_host_sdks_and_not_soxrs_own_recipes() {
    // Pins the deliberate bug-for-bug mapping documented at the top of this file. If someone
    // "fixes" High to SOXR_HQ (4) our audio silently stops matching `rtc.AudioResampler`.
    assert_eq!(Quality::Quick as u32, 0);
    assert_eq!(Quality::Low as u32, 1);
    assert_eq!(Quality::Medium as u32, 2);
    assert_eq!(Quality::High as u32, 3); // soxr's SOXR_16_BITQ, NOT SOXR_HQ (= 4)
    assert_eq!(Quality::VeryHigh as u32, 4); // soxr's SOXR_20_BITQ, NOT SOXR_VHQ (= 6)
    assert_eq!(Quality::default(), Quality::Medium);
}
