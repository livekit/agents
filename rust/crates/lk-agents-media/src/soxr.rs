//! Sample-rate conversion over `soxr-sys`, the library behind `rtc.AudioResampler`, so output is
//! identical (the VAD/EOT models were tuned on it). Upstream bugs: `livekit-ffi`'s `soxr_create`
//! error check is dead code (we check a real out-param); its quality ordinal is passed straight as
//! the recipe, one notch low for HIGH/VERY_HIGH, and we reproduce that to match the SDK.

use std::{
    ffi::{CStr, c_char},
    os::raw::{c_ulong, c_void},
    sync::RwLock,
};

/// libsoxr's FFT coefficient cache is process-global and unlocked without OpenMP; growing it under
/// a running transform crashes or silently corrupts output. Only `soxr_create` grows it, so `new`
/// takes this for writing and `push`/`flush` (`soxr_clear`) for reading.
static FFT_CACHE: RwLock<()> = RwLock::new(());

/// `Quality`'s discriminants are protobuf ordinals, not soxr recipes (see module docs).
pub use lk_agents_abi::{Quality, ResampleError, Resampler};

/// Not `Sync`; distinct handles on distinct threads are safe only because of `FFT_CACHE`.
pub struct SoxrResampler {
    ptr: soxr_sys::soxr_t,
    out: Vec<i16>,
    input_rate: f64,
    output_rate: f64,
    num_channels: u32,
}

unsafe impl Send for SoxrResampler {}

impl SoxrResampler {
    pub fn new(
        input_rate: u32,
        output_rate: u32,
        num_channels: u32,
        quality: Quality,
    ) -> Result<Self, ResampleError> {
        let input_rate = f64::from(input_rate);
        let output_rate = f64::from(output_rate);

        // A real out-param, unlike upstream.
        let mut err: *const c_char = std::ptr::null();

        // The writer. A poisoned lock guards no state of ours, so it is not a reason to refuse.
        let _cache = FFT_CACHE.write().unwrap_or_else(|e| e.into_inner());

        let ptr = unsafe {
            let io = soxr_sys::soxr_io_spec(
                soxr_sys::soxr_datatype_t_SOXR_INT16_I,
                soxr_sys::soxr_datatype_t_SOXR_INT16_I,
            );
            // flags 0 and 1 thread: what `livekit-ffi` hard-codes.
            let q = soxr_sys::soxr_quality_spec(quality as c_ulong, 0);
            let rt = soxr_sys::soxr_runtime_spec(1);
            soxr_sys::soxr_create(input_rate, output_rate, num_channels, &mut err, &io, &q, &rt)
        };

        if !err.is_null() {
            return Err(ResampleError::Create(
                unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned(),
            ));
        }
        if ptr.is_null() {
            // soxr documents that it sets `err`; a null handle would otherwise segfault on `push`.
            return Err(ResampleError::Create("soxr_create returned null".into()));
        }

        Ok(Self {
            ptr,
            // A valid pointer for soxr even before the first resize.
            out: Vec::with_capacity(output_rate as usize / 100),
            input_rate,
            output_rate,
            num_channels,
        })
    }

    pub fn input_rate(&self) -> u32 {
        self.input_rate as u32
    }

    pub fn output_rate(&self) -> u32 {
        self.output_rate as u32
    }

    pub fn num_channels(&self) -> u32 {
        self.num_channels
    }

    /// May return an empty slice: soxr buffers.
    pub fn push(&mut self, input: &[i16]) -> Result<&[i16], ResampleError> {
        let nch = self.num_channels as usize;
        if !input.len().is_multiple_of(nch) {
            return Err(ResampleError::PartialFrame {
                len: input.len(),
                num_channels: self.num_channels,
            });
        }
        let in_frames = input.len() / nch;
        let ratio = self.output_rate / self.input_rate;

        // A reader of the FFT cache.
        let _cache = FFT_CACHE.read().unwrap_or_else(|e| e.into_inner());

        // `soxr_delay` is the filter's residual; sizing on ratio alone makes soxr silently emit less.
        let delay = unsafe { soxr_sys::soxr_delay(self.ptr) };
        // `+ 1` is a margin against a `ceil` rounding edge, not load-bearing.
        let max_out = ((in_frames as f64 * ratio).ceil() as usize) + (delay.ceil() as usize) + 1;

        let needed = max_out * nch;
        if self.out.len() < needed {
            self.out.resize(needed, 0);
        }

        let mut idone: usize = 0;
        let mut odone: usize = 0;
        let err = unsafe {
            soxr_sys::soxr_process(
                self.ptr,
                input.as_ptr() as *const c_void,
                in_frames,
                &mut idone,
                self.out.as_mut_ptr() as *mut c_void,
                max_out,
                &mut odone,
            )
        };
        if !err.is_null() {
            return Err(ResampleError::Process(
                unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned(),
            ));
        }
        Ok(&self.out[..odone * nch])
    }

    /// Flushes and clears, as the host SDK does, so no filter history carries across a silence.
    pub fn flush(&mut self) -> Result<&[i16], ResampleError> {
        let nch = self.num_channels as usize;
        if self.out.is_empty() {
            // Nothing pushed: no tail, but soxr still needs a real buffer, not a dangling pointer.
            self.out.resize(self.output_rate as usize / 100 * nch, 0);
        }
        let cap = self.out.len() / nch;

        // A reader: `soxr_clear` re-initialises at lengths construction already reached.
        let _cache = FFT_CACHE.read().unwrap_or_else(|e| e.into_inner());

        let mut odone: usize = 0;
        let err = unsafe {
            soxr_sys::soxr_process(
                self.ptr,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                self.out.as_mut_ptr() as *mut c_void,
                cap,
                &mut odone,
            )
        };
        if !err.is_null() {
            return Err(ResampleError::Process(
                unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned(),
            ));
        }

        let err = unsafe { soxr_sys::soxr_clear(self.ptr) };
        if !err.is_null() {
            return Err(ResampleError::Process(
                unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned(),
            ));
        }
        Ok(&self.out[..odone * nch])
    }
}

impl Resampler for SoxrResampler {
    fn input_rate(&self) -> u32 {
        SoxrResampler::input_rate(self)
    }

    fn output_rate(&self) -> u32 {
        SoxrResampler::output_rate(self)
    }

    fn num_channels(&self) -> u32 {
        SoxrResampler::num_channels(self)
    }

    fn push(&mut self, input: &[i16]) -> Result<&[i16], ResampleError> {
        SoxrResampler::push(self, input)
    }

    fn flush(&mut self) -> Result<&[i16], ResampleError> {
        SoxrResampler::flush(self)
    }
}

impl Drop for SoxrResampler {
    fn drop(&mut self) {
        unsafe { soxr_sys::soxr_delete(self.ptr) }
    }
}

#[cfg(test)]
mod tests {
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
    fn a_1khz_tone_survives_48k_to_16k_at_the_same_pitch() {
        // The reason this crate exists: the VAD and EOT models were tuned on audio that went
        // through soxr. A wrong ratio or a broken filter shows up here as a shifted or aliased
        // tone, which no length assertion would notice.
        let mut r = SoxrResampler::new(48_000, 16_000, 1, Quality::Medium).unwrap();
        let mut out = Vec::new();
        for _ in 0..10 {
            out.extend_from_slice(r.push(&sine(1000.0, 48_000, 20, 1)).unwrap());
        }
        out.extend_from_slice(r.flush().unwrap());

        let hz = dominant_hz(&out, 16_000);
        assert!((hz - 1000.0).abs() < 25.0, "tone moved to {hz} Hz");
    }

    #[test]
    fn a_tone_above_the_new_nyquist_is_filtered_out_rather_than_aliased() {
        // 12 kHz cannot exist at 16 kHz output. A resampler that merely decimated would fold it
        // down to 4 kHz and hand the VAD a tone nobody spoke.
        let mut r = SoxrResampler::new(48_000, 16_000, 1, Quality::Medium).unwrap();
        let mut out = Vec::new();
        for _ in 0..10 {
            out.extend_from_slice(r.push(&sine(12_000.0, 48_000, 20, 1)).unwrap());
        }
        out.extend_from_slice(r.flush().unwrap());

        // Measure the steady state, not the whole buffer. The filter's startup and flush
        // transients ring at ~3100 peak here, which is real and expected -- an assertion over
        // the full buffer measures the transient, not the stopband, and would pass just as
        // happily against a resampler that aliased.
        let n = out.len();
        let steady = &out[n / 4..3 * n / 4];
        let peak = steady.iter().map(|s| s.unsigned_abs()).max().unwrap();
        assert!(peak < 50, "12 kHz leaked into the 16 kHz output at peak {peak}");
    }

    #[test]
    fn flush_drains_the_tail_and_then_has_nothing_left() {
        let mut r = SoxrResampler::new(48_000, 16_000, 1, Quality::Medium).unwrap();
        r.push(&sine(440.0, 48_000, 20, 1)).unwrap();
        assert!(!r.flush().unwrap().is_empty(), "the filter tail was dropped");
        // flush also clears, so the second one has no history to drain.
        assert!(r.flush().unwrap().is_empty());
    }

    #[test]
    fn upsampling_works_too_since_tts_output_has_to_reach_the_room_rate() {
        let mut r = SoxrResampler::new(16_000, 48_000, 1, Quality::Medium).unwrap();
        let mut total = 0usize;
        for _ in 0..50 {
            total += r.push(&sine(440.0, 16_000, 20, 1)).unwrap().len();
        }
        total += r.flush().unwrap().len();
        let expected = 50 * 16_000 * 20 / 1000 * 3;
        assert!((total as i64 - expected as i64).abs() <= 2, "produced {total}, want ~{expected}");
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
        let gate = std::sync::Arc::new(std::sync::Barrier::new(N + 1));

        let victim = {
            let gate = std::sync::Arc::clone(&gate);
            std::thread::spawn(move || {
                let mut r = SoxrResampler::new(48_000, 16_000, 1, Quality::Medium).unwrap();
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
        };

        let mut creators = Vec::new();
        for i in 0..N {
            let gate = std::sync::Arc::clone(&gate);
            creators.push(std::thread::spawn(move || {
                gate.wait();
                // A distinct rate per thread: every one designs its own filter.
                let out_rate = 8_000 + (i as u32) * 1_000;
                let mut r = SoxrResampler::new(48_000, out_rate, 1, Quality::Medium)
                    .expect("construction must not race");
                let tone = sine(1000.0, 48_000, 20, 1);
                let mut out = Vec::new();
                for _ in 0..10 {
                    out.extend_from_slice(r.push(&tone).unwrap());
                }
                out.extend_from_slice(r.flush().unwrap());
                let hz = dominant_hz(&out, out_rate);
                assert!((hz - 1000.0).abs() < 25.0, "tone at {out_rate} Hz moved to {hz} Hz");
            }));
        }
        for h in creators {
            h.join().expect("a construction thread panicked or crashed");
        }
        victim.join().expect("the victim's audio was corrupted or it crashed");
    }

    #[test]
    fn stereo_sizing_is_per_frame_not_per_sample() {
        // Every other test in this file is mono, which makes `* nch` and `/ nch` invisible --
        // multiplying by one. That is why the buffer arithmetic went unpinned: the tests could
        // not distinguish the correct expression from four mutations of it.
        //
        // Catches: sizing `out` in samples where it should be frames (or the reverse). Too small
        // and soxr silently emits less than it had ready -- a slow drift, never an error.
        let mut r = SoxrResampler::new(48_000, 16_000, 2, Quality::Medium).unwrap();
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
        let mut r = SoxrResampler::new(44_100, 16_000, 1, Quality::High).unwrap();
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
        let mut r = SoxrResampler::new(48_000, 44_100, 2, Quality::VeryHigh).unwrap();
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
}
