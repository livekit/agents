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

/// Zero crossings -> frequency: a wrong filter or rate ratio moves or aliases the tone.
fn dominant_hz(pcm: &[i16], rate: u32) -> f64 {
    let crossings = pcm.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count();
    crossings as f64 / 2.0 * f64::from(rate) / pcm.len() as f64
}

/// Rate conversion preserves duration, pitch and channel framing. https://github.com/livekit/agents/issues/4244 https://github.com/livekit/agents/pull/1643
#[test]
fn realtime_rate_pairs_preserve_tone_duration_and_stereo_headers() {
    for (source, target) in [(22_050, 24_000), (24_000, 48_000), (48_000, 24_000)] {
        let mut resampler =
            crate::AudioResampler::new(target, Quality::Medium, crate::OnChange::Drain);
        let input = sine(440.0, source, 1000, 2);
        let mut frames = Vec::new();
        for chunk in input.chunks(source as usize / 50 * 2) {
            frames.extend(
                resampler
                    .push_frame(&lk_agents_abi::AudioFrame::new(
                        crate::pcm_bytes(chunk),
                        source,
                        2,
                        (chunk.len() / 2) as u32,
                    ))
                    .unwrap(),
            );
        }
        frames.extend(resampler.flush().unwrap());
        assert!(frames.iter().all(|f| f.sample_rate == target
            && f.num_channels == 2
            && f.data.len() == f.samples_per_channel as usize * 4));
        let samples: Vec<_> = frames.iter().flat_map(|f| crate::pcm_samples(&f.data)).collect();
        assert!(samples.len().abs_diff(target as usize * 2) <= 2);
        let left: Vec<_> = samples
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                assert_eq!(pair[0], pair[1]);
                pair[0]
            })
            .collect();
        assert!((dominant_hz(&left, target) - 440.0).abs() < 2.0);
    }
}

#[test]
fn zero_rates_and_channels_are_rejected() {
    for (input, output, channels) in [(0, 16_000, 1), (48_000, 0, 1), (48_000, 16_000, 0)] {
        assert!(Resampler::new(input, output, channels, Quality::Medium).is_err());
    }
}

#[test]
fn a_1khz_tone_survives_48k_to_16k_at_the_same_pitch() {
    // The VAD and EOT models were tuned on soxr output; a wrong ratio or broken filter shows as a
    // shifted or aliased tone, which no length assertion would notice.
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
    // libsoxr's FFT cache is grown by the first thread to need a longer DFT while others read it
    // mid-`soxr_process`; without `FFT_CACHE` that crashes or silently shifts a tone's pitch.
    // Shape matters: many threads, one construction each, a distinct rate per thread, all
    // released at one barrier, plus a victim already pushing. Batching warms the cache and hides
    // the race; without the barrier spawn overhead serialises the threads. Probabilistic: loop it.
    // Threads are spawned here so it reproduces under `cargo nextest` too.
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
    // Stereo, because mono makes `* nch` and `/ nch` invisible. Sizing `out` in samples instead
    // of frames (or the reverse) makes soxr silently emit less than it had ready.
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
    // `soxr_delay` only becomes non-zero once the filter has history, so this runs long enough
    // to pin it; under-allocation accumulates a shortfall that never errors.
    let mut r = Resampler::new(44_100, 16_000, 1, Quality::High).unwrap();
    let mut total = 0usize;
    for _ in 0..500 {
        total += r.push(&sine(440.0, 44_100, 20, 1)).unwrap().len();
    }
    total += r.flush().unwrap().len();

    // 10 s in -> 160 000 out. A ratio that is not a round fraction keeps the filter's residual
    // non-zero for the whole run.
    let expected = 500 * 16_000 * 20 / 1000;
    let drift = (total as i64 - expected as i64).abs();
    assert!(drift <= 8, "produced {total} over 10 s, expected ~{expected} (drift {drift})");
}

#[test]
fn flushing_a_long_stereo_tail_stays_inside_the_buffer() {
    // `flush` tells soxr how many frames the output holds: `self.out.len() / nch`. Over-reporting
    // lets soxr write past the end of the `Vec<i16>`; a long VeryHigh filter fills the tail.
    let mut r = Resampler::new(48_000, 44_100, 2, Quality::VeryHigh).unwrap();
    for _ in 0..100 {
        r.push(&sine(440.0, 48_000, 20, 2)).unwrap();
    }
    let tail = r.flush().unwrap();
    assert!(tail.len().is_multiple_of(2), "tail is not whole stereo frames");
    // A residual, not a whole second: an over-reported `cap` would hand back far more.
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
    // High is SOXR_16_BITQ, not SOXR_HQ, so the audio matches `rtc.AudioResampler`.
    assert_eq!(Quality::Quick as u32, 0);
    assert_eq!(Quality::Low as u32, 1);
    assert_eq!(Quality::Medium as u32, 2);
    assert_eq!(Quality::High as u32, 3); // soxr's SOXR_16_BITQ, NOT SOXR_HQ (= 4)
    assert_eq!(Quality::VeryHigh as u32, 4); // soxr's SOXR_20_BITQ, NOT SOXR_VHQ (= 6)
    assert_eq!(Quality::default(), Quality::Medium);
}
