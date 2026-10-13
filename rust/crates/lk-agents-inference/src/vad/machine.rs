//! `inference/vad.py`'s `_VADStream._main_task`, as a synchronous accumulator: the model runs
//! in-process behind [`VadModel`], no socket and no clock. The slow-inference lag is computed and
//! reported on [`VadOutput::extra_inference_time`]; the host logs it.

use crate::abi::AudioFrame;
use lk_agents_media::{AudioFormat, FrameResampler, Quality};
use lk_agents_abi::{VadEvent, VadEventType};

/// `VAD_WINDOW_SAMPLES`, from `livekit.local_inference`. Silero's frame size at 16 kHz.
pub const VAD_WINDOW_SAMPLES: usize = 512;

pub const MODEL_SAMPLE_RATE: u32 = 16_000;

pub const SLOW_INFERENCE_THRESHOLD: f64 = 0.2;

/// The Silero model behind `livekit.local_inference`; `predict` also returns how long it took.
pub trait VadModel: Send {
    fn predict(&mut self, window: &[i16]) -> (f32, f64);
    fn reset(&mut self);
}

/// `rtc.AudioResampler`'s default quality, mono: every frame is read as one channel. A construction
/// failure yields `None` rather than aborting.
fn new_resampler(input_rate: u32) -> Option<FrameResampler> {
    FrameResampler::fixed(AudioFormat::new(input_rate, 1), MODEL_SAMPLE_RATE, Quality::Medium).ok()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadOptions {
    pub min_speech_duration: f64,
    pub min_silence_duration: f64,
    pub prefix_padding_duration: f64,
    pub max_buffered_speech: f64,
    pub activation_threshold: f64,
    pub deactivation_threshold: f64,
}

impl Default for VadOptions {
    fn default() -> Self {
        VadOptions {
            min_speech_duration: 0.05,
            min_silence_duration: 0.25,
            prefix_padding_duration: 0.5,
            max_buffered_speech: 60.0,
            activation_threshold: 0.5,
            // `max(activation_threshold - 0.15, 0.01)`: hysteresis.
            deactivation_threshold: 0.35,
        }
    }
}

/// One emitted event, plus its PCM: `VadEvent.frames` is a side-table handle (§1.11), so whoever
/// installs the audio sets it.
#[derive(Debug, Clone, PartialEq)]
pub struct VadOutput {
    pub event: VadEvent,
    pub frames: Vec<AudioFrame>,
    /// Accumulated lag behind realtime, in seconds.
    pub extra_inference_time: f64,
    /// True on the call whose inference exceeded [`SLOW_INFERENCE_THRESHOLD`].
    pub slow_inference: bool,
}

pub struct VadMachine {
    opts: VadOptions,
    model: Box<dyn VadModel>,
    resampler: Option<FrameResampler>,

    input_sample_rate: u32,
    /// Sized `max_buffered_speech * input_rate + prefix_padding`, in **samples**.
    speech_buffer: Vec<i16>,
    speech_buffer_index: usize,
    speech_buffer_max_reached: bool,
    prefix_padding_samples: usize,

    // "pub_" means public: these are what reach the events.
    pub_speaking: bool,
    pub_speech_duration: f64,
    pub_silence_duration: f64,
    pub_current_sample: u64,
    pub_timestamp: f64,

    speech_threshold_duration: f64,
    silence_threshold_duration: f64,

    input_frames: Vec<AudioFrame>,
    inference_frames: Vec<AudioFrame>,
    /// Avoids drift when the sample-rate ratio is not an integer.
    input_copy_remaining_fract: f64,
    extra_inference_time: f64,
}

impl VadMachine {
    pub fn new(opts: VadOptions, model: Box<dyn VadModel>) -> Self {
        VadMachine {
            opts,
            model,
            resampler: None,
            input_sample_rate: 0,
            speech_buffer: Vec::new(),
            speech_buffer_index: 0,
            speech_buffer_max_reached: false,
            prefix_padding_samples: 0,
            pub_speaking: false,
            pub_speech_duration: 0.0,
            pub_silence_duration: 0.0,
            pub_current_sample: 0,
            pub_timestamp: 0.0,
            speech_threshold_duration: 0.0,
            silence_threshold_duration: 0.0,
            input_frames: Vec::new(),
            inference_frames: Vec::new(),
            input_copy_remaining_fract: 0.0,
            extra_inference_time: 0.0,
        }
    }

    /// The `_FlushSentinel` branch: reset everything, including the model's LSTM state.
    pub fn reset(&mut self) {
        self.model.reset();
        self.speech_buffer_index = 0;
        self.speech_buffer_max_reached = false;
        self.speech_buffer.fill(0);
        self.pub_speaking = false;
        self.pub_speech_duration = 0.0;
        self.pub_silence_duration = 0.0;
        self.pub_current_sample = 0;
        self.pub_timestamp = 0.0;
        self.speech_threshold_duration = 0.0;
        self.silence_threshold_duration = 0.0;
        self.input_frames.clear();
        self.inference_frames.clear();
        self.input_copy_remaining_fract = 0.0;
        self.extra_inference_time = 0.0;
        self.resampler = (self.input_sample_rate != 0
            && self.input_sample_rate != MODEL_SAMPLE_RATE)
            .then(|| new_resampler(self.input_sample_rate))
            .flatten();
    }

    /// `_VADStream.update_options`: the thresholds apply from the next window; once the sample rate
    /// is known, the prefix padding and the speech buffer are resized, and a grown buffer may fill
    /// again.
    pub fn update_options(&mut self, opts: VadOptions) {
        let grew = opts.max_buffered_speech > self.opts.max_buffered_speech;
        self.opts = opts;
        if self.input_sample_rate == 0 {
            return;
        }
        let rate = f64::from(self.input_sample_rate);
        self.prefix_padding_samples = (opts.prefix_padding_duration * rate) as usize;
        // `ndarray.resize`: truncated, or zero-filled at the end.
        self.speech_buffer.resize((opts.max_buffered_speech * rate) as usize + self.prefix_padding_samples, 0);
        // The reference leaves the cursor past a shrunk buffer, where its next rewind or copy
        // raises; clamped instead.
        self.speech_buffer_index = self.speech_buffer_index.min(self.speech_buffer.len());
        if grew {
            self.speech_buffer_max_reached = false;
        }
    }

    /// Push one frame and run inference over every whole window it completes. A frame at a
    /// different sample rate from the first is `Err` (the reference logs and drops it).
    pub fn push(&mut self, frame: AudioFrame) -> Result<Vec<VadOutput>, SampleRateMismatch> {
        if self.input_sample_rate == 0 {
            self.input_sample_rate = frame.sample_rate;
            self.prefix_padding_samples =
                (self.opts.prefix_padding_duration * f64::from(self.input_sample_rate)) as usize;
            self.speech_buffer = vec![
                0i16;
                (self.opts.max_buffered_speech * f64::from(self.input_sample_rate))
                    as usize
                    + self.prefix_padding_samples
            ];
            if self.input_sample_rate != MODEL_SAMPLE_RATE {
                self.resampler = new_resampler(self.input_sample_rate);
            }
        } else if self.input_sample_rate != frame.sample_rate {
            return Err(SampleRateMismatch {
                expected: self.input_sample_rate,
                got: frame.sample_rate,
            });
        }

        match self.resampler.as_mut() {
            // Empty is normal (soxr buffers), and a failed push is also empty rather than a panic
            // on a tokio worker.
            Some(resampler) => self.inference_frames.extend(resampler.push(&frame).unwrap_or_default()),
            None => self.inference_frames.push(frame.clone()),
        }
        self.input_frames.push(frame);

        Ok(self.run_windows())
    }

    fn run_windows(&mut self) -> Vec<VadOutput> {
        let mut out = Vec::new();
        loop {
            let available: usize =
                self.inference_frames.iter().map(|f| f.samples_per_channel as usize).sum();
            if available < VAD_WINDOW_SAMPLES {
                break;
            }

            let input_frame = combine(&self.input_frames, self.input_sample_rate);
            let inference_frame = combine(&self.inference_frames, MODEL_SAMPLE_RATE);

            let window = to_i16(&inference_frame.data[..VAD_WINDOW_SAMPLES * 2]);
            let (probability, inference_duration) = self.model.predict(&window);
            let probability = f64::from(probability);

            let window_duration = VAD_WINDOW_SAMPLES as f64 / f64::from(MODEL_SAMPLE_RATE);
            self.pub_current_sample += VAD_WINDOW_SAMPLES as u64;
            self.pub_timestamp += window_duration;

            let resampling_ratio = f64::from(self.input_sample_rate) / f64::from(MODEL_SAMPLE_RATE);
            let to_copy =
                VAD_WINDOW_SAMPLES as f64 * resampling_ratio + self.input_copy_remaining_fract;
            let to_copy_int = to_copy as usize;
            // The fractional carry keeps the event timeline from drifting at non-integer ratios
            // (44.1 kHz).
            self.input_copy_remaining_fract = to_copy - to_copy_int as f64;

            let input_samples = to_i16(&input_frame.data);
            let available_space = self.speech_buffer.len() - self.speech_buffer_index;
            let to_copy_buffer = to_copy_int.min(available_space).min(input_samples.len());
            if to_copy_buffer > 0 {
                self.speech_buffer
                    [self.speech_buffer_index..self.speech_buffer_index + to_copy_buffer]
                    .copy_from_slice(&input_samples[..to_copy_buffer]);
                self.speech_buffer_index += to_copy_buffer;
            } else if !self.speech_buffer_max_reached {
                // Reached `max_buffered_speech` (padding included).
                self.speech_buffer_max_reached = true;
            }

            self.extra_inference_time =
                (self.extra_inference_time + inference_duration - window_duration).max(0.0);
            let slow_inference = inference_duration > SLOW_INFERENCE_THRESHOLD;

            if self.pub_speaking {
                self.pub_speech_duration += window_duration;
            } else {
                self.pub_silence_duration += window_duration;
            }

            let window_frame = mono(
                input_frame.data[..(to_copy_int * 2).min(input_frame.data.len())].to_vec(),
                self.input_sample_rate,
            );
            out.push(VadOutput {
                event: VadEvent {
                    kind: VadEventType::InferenceDone,
                    samples_index: self.pub_current_sample,
                    timestamp: self.pub_timestamp,
                    speech_duration: self.pub_speech_duration,
                    silence_duration: self.pub_silence_duration,
                    frames: None,
                    probability,
                    inference_duration,
                    speaking: self.pub_speaking,
                    raw_accumulated_silence: self.silence_threshold_duration,
                    raw_accumulated_speech: self.speech_threshold_duration,
                },
                frames: vec![window_frame],
                extra_inference_time: self.extra_inference_time,
                slow_inference,
            });

            // Hysteresis: `p >= activation` starts speech; once speaking, `p > deactivation`
            // keeps it.
            if probability >= self.opts.activation_threshold
                || (self.pub_speaking && probability > self.opts.deactivation_threshold)
            {
                self.speech_threshold_duration += window_duration;
                self.silence_threshold_duration = 0.0;

                if !self.pub_speaking
                    && self.speech_threshold_duration >= self.opts.min_speech_duration
                {
                    self.pub_speaking = true;
                    self.pub_silence_duration = 0.0;
                    self.pub_speech_duration = self.speech_threshold_duration;
                    out.push(self.edge(VadEventType::StartOfSpeech, self.pub_speech_duration));
                }
            } else {
                self.silence_threshold_duration += window_duration;
                self.speech_threshold_duration = 0.0;

                if !self.pub_speaking {
                    self.reset_write_cursor();
                }

                if self.pub_speaking
                    && self.silence_threshold_duration >= self.opts.min_silence_duration
                {
                    self.pub_speaking = false;
                    self.pub_silence_duration = self.silence_threshold_duration;
                    // Trailing silence is subtracted: speech ended when silence started.
                    let speech = (self.pub_speech_duration - self.silence_threshold_duration).max(0.0);
                    out.push(self.edge(VadEventType::EndOfSpeech, speech));
                    self.pub_speech_duration = 0.0;
                    self.reset_write_cursor();
                }
            }

            let input_tail =
                input_frame.data[(to_copy_int * 2).min(input_frame.data.len())..].to_vec();
            let inference_tail = inference_frame.data[VAD_WINDOW_SAMPLES * 2..].to_vec();
            self.input_frames.clear();
            self.inference_frames.clear();
            if !input_tail.is_empty() {
                self.input_frames.push(mono(input_tail, self.input_sample_rate));
            }
            if !inference_tail.is_empty() {
                self.inference_frames.push(mono(inference_tail, MODEL_SAMPLE_RATE));
            }
        }
        out
    }

    /// Slide the last `prefix_padding` samples to the front and rewind the write cursor, so
    /// `start_of_speech` carries the run-up before the onset.
    fn reset_write_cursor(&mut self) {
        if self.speech_buffer_index <= self.prefix_padding_samples {
            return;
        }
        let start = self.speech_buffer_index - self.prefix_padding_samples;
        self.speech_buffer.copy_within(start..self.speech_buffer_index, 0);
        self.speech_buffer_max_reached = false;
        self.speech_buffer_index = self.prefix_padding_samples;
    }

    fn edge(&self, kind: VadEventType, speech_duration: f64) -> VadOutput {
        VadOutput {
            event: VadEvent {
                kind,
                samples_index: self.pub_current_sample,
                timestamp: self.pub_timestamp,
                speech_duration,
                silence_duration: self.pub_silence_duration,
                frames: None,
                probability: 0.0,
                inference_duration: 0.0,
                speaking: kind == VadEventType::StartOfSpeech,
                raw_accumulated_silence: 0.0,
                raw_accumulated_speech: 0.0,
            },
            frames: vec![self.copy_speech_buffer()],
            extra_inference_time: self.extra_inference_time,
            slow_inference: false,
        }
    }

    fn copy_speech_buffer(&self) -> AudioFrame {
        let mut data = Vec::with_capacity(self.speech_buffer_index * 2);
        for sample in &self.speech_buffer[..self.speech_buffer_index] {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        mono(data, self.input_sample_rate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a frame at {got} Hz was pushed after the stream settled on {expected} Hz")]
pub struct SampleRateMismatch {
    pub expected: u32,
    pub got: u32,
}

fn combine(frames: &[AudioFrame], sample_rate: u32) -> AudioFrame {
    let mut data = Vec::with_capacity(frames.iter().map(|f| f.data.len()).sum());
    for frame in frames {
        data.extend_from_slice(&frame.data);
    }
    mono(data, sample_rate)
}

pub(crate) fn mono(data: Vec<u8>, sample_rate: u32) -> AudioFrame {
    let samples_per_channel = (data.len() / 2) as u32;
    AudioFrame::new(data, sample_rate, 1, samples_per_channel)
}

fn to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that replays a scripted probability per window.
    struct ScriptedModel {
        probabilities: std::collections::VecDeque<f32>,
        duration: f64,
    }

    impl VadModel for ScriptedModel {
        fn predict(&mut self, _window: &[i16]) -> (f32, f64) {
            (self.probabilities.pop_front().unwrap_or(0.0), self.duration)
        }
        fn reset(&mut self) {}
    }

    /// Each window scores the next probability and takes `duration` seconds to infer.
    fn timed_machine(probabilities: &[f32], duration: f64) -> VadMachine {
        let probabilities = probabilities.iter().copied().collect();
        VadMachine::new(VadOptions::default(), Box::new(ScriptedModel { probabilities, duration }))
    }

    /// Fed at [`MODEL_SAMPLE_RATE`], so no resampler is built.
    fn machine(probabilities: &[f32]) -> VadMachine {
        timed_machine(probabilities, 0.001)
    }

    /// One window's worth of 16 kHz silence.
    fn window() -> AudioFrame {
        filled_window(0)
    }

    /// One window of audio whose every byte is `fill`, so a buffer's *contents* are checkable.
    fn filled_window(fill: u8) -> AudioFrame {
        mono(vec![fill; VAD_WINDOW_SAMPLES * 2], MODEL_SAMPLE_RATE)
    }

    fn kinds(out: &[VadOutput]) -> Vec<VadEventType> {
        out.iter().map(|o| o.event.kind).collect()
    }

    // Catches: building a resampler for audio already at the model's rate (soxr is not a no-op).
    #[test]
    fn no_resampler_is_built_when_the_input_is_already_at_the_model_rate() {
        let mut m = machine(&[0.0]);
        let out = m.push(filled_window(7)).unwrap();
        let done = out.iter().find(|o| o.event.kind == VadEventType::InferenceDone).unwrap();
        assert!(m.resampler.is_none(), "the model rate needs no conversion");
        assert!(
            done.frames.iter().all(|f| f.data.iter().all(|b| *b == 7)),
            "the frames the model scored were rewritten by a resampler that should not exist"
        );
    }

    // Catches: firing START_OF_SPEECH on the first loud window (`min_speech_duration` needs two).
    #[test]
    fn speech_starts_only_after_min_speech_duration_of_windows() {
        let mut m = machine(&[0.9, 0.9]);
        let out = m.push(window()).unwrap();
        assert_eq!(kinds(&out), [VadEventType::InferenceDone], "32ms < 50ms");
        let out = m.push(window()).unwrap();
        assert_eq!(kinds(&out), [VadEventType::InferenceDone, VadEventType::StartOfSpeech]);
    }

    // Catches: collapsing the two thresholds into one.
    #[test]
    fn hysteresis_keeps_a_turn_open_between_the_two_thresholds() {
        let mut m = machine(&[0.9, 0.9, 0.4, 0.4, 0.4]);
        m.push(window()).unwrap();
        m.push(window()).unwrap();
        for _ in 0..3 {
            let out = m.push(window()).unwrap();
            assert_eq!(kinds(&out), [VadEventType::InferenceDone], "0.4 > deactivation");
            assert!(out[0].event.speaking);
        }
    }

    // Catches: reporting the trailing silence as speech.
    #[test]
    fn end_of_speech_subtracts_the_trailing_silence_from_the_speech_duration() {
        let mut probabilities = vec![0.9; 20];
        probabilities.extend(std::iter::repeat_n(0.0, 10));
        let mut m = machine(&probabilities);
        let mut end = None;
        for _ in 0..30 {
            for out in m.push(window()).unwrap() {
                if out.event.kind == VadEventType::EndOfSpeech {
                    end = Some(out);
                }
            }
        }
        let end = end.expect("speech ended");
        let window_duration = VAD_WINDOW_SAMPLES as f64 / f64::from(MODEL_SAMPLE_RATE);
        assert!(end.event.silence_duration >= 0.25);
        // `pub_speech_duration` counted 28 windows by end-of-speech; subtracting the silence
        // leaves exactly the 20 loud ones.
        let expected = 20.0 * window_duration;
        assert!(
            (end.event.speech_duration - expected).abs() < 1e-9,
            "got {} expected {expected}",
            end.event.speech_duration
        );
    }

    // Catches: dropping the fractional carry on the input-copy count.
    #[test]
    fn the_input_copy_fraction_carries_across_windows() {
        // A non-integer ratio (44100 -> 16000) through real soxr; the property is the carry.
        let mut m = timed_machine(&[0.0; 20], 0.0);
        // 100 ms of 44.1 kHz audio at a time.
        let mut consumed = Vec::new();
        for _ in 0..6 {
            let frame = mono(vec![0u8; 4410 * 2], 44_100);
            for out in m.push(frame).unwrap() {
                if out.event.kind == VadEventType::InferenceDone {
                    consumed.push(out.frames[0].samples_per_channel);
                }
            }
        }
        assert!(consumed.len() >= 3, "got {consumed:?}");
        // 512 * 44100/16000 = 1411.2, so the counts must alternate.
        assert!(consumed.contains(&1412), "the carry never produced a long window: {consumed:?}");
    }

    // Catches: a flush not resetting the model's LSTM state.
    #[test]
    fn a_flush_resets_the_model_as_well_as_the_counters() {
        struct Counting(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl VadModel for Counting {
            fn predict(&mut self, _: &[i16]) -> (f32, f64) {
                (0.0, 0.0)
            }
            fn reset(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let resets = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut m = VadMachine::new(
            VadOptions::default(),
            Box::new(Counting(resets.clone())),
        );
        m.push(window()).unwrap();
        m.reset();
        assert_eq!(resets.load(std::sync::atomic::Ordering::SeqCst), 1);
        let out = m.push(window()).unwrap();
        assert_eq!(out[0].event.samples_index, VAD_WINDOW_SAMPLES as u64, "the counters restarted");
    }

    // Catches: the speech-buffer bookkeeping (write cursor, prefix rewind, available space);
    // the other tests run on silence, where a wrong buffer looks right.
    #[test]
    fn start_of_speech_carries_the_prefix_padding_plus_the_speech_so_far() {
        let mut probabilities = vec![0.0f32; 20];
        probabilities.extend([0.9, 0.9]);
        let mut m = machine(&probabilities);

        for _ in 0..20 {
            m.push(filled_window(0)).unwrap();
        }
        let mut onset = None;
        for _ in 0..2 {
            for out in m.push(filled_window(7)).unwrap() {
                if out.event.kind == VadEventType::StartOfSpeech {
                    onset = Some(out);
                }
            }
        }
        let onset = onset.expect("speech started");
        let frame = &onset.frames[0];

        // 0.5 s of prefix padding at 16 kHz, plus the two windows that crossed
        // `min_speech_duration`.
        let prefix = (0.5 * 16_000.0) as u32;
        assert_eq!(frame.samples_per_channel, prefix + 2 * VAD_WINDOW_SAMPLES as u32);
        assert_eq!(frame.data.len(), frame.samples_per_channel as usize * 2);
        // The tail is the speech; the head is the run-up that was rewound into place.
        let speech_bytes = VAD_WINDOW_SAMPLES * 2 * 2;
        assert!(frame.data[frame.data.len() - speech_bytes..].iter().all(|&b| b == 7));
        assert!(frame.data[..frame.data.len() - speech_bytes].iter().all(|&b| b == 0));
    }

    // Catches: a running machine ignoring `update_options` (the stream's options read once, at
    // start): with the activation threshold raised past 0.6, 0.6 windows must not start speech.
    #[test]
    fn a_threshold_updated_mid_stream_applies_from_the_next_window() {
        let mut m = machine(&[0.6; 4]);
        m.push(window()).unwrap();
        m.update_options(VadOptions { activation_threshold: 0.9, ..VadOptions::default() });
        for _ in 0..3 {
            assert_eq!(kinds(&m.push(window()).unwrap()), [VadEventType::InferenceDone]);
        }
    }

    // Catches: the speech buffer keeping its first size, or a grown buffer staying "full".
    #[test]
    fn a_buffer_updated_mid_stream_is_resized_and_may_fill_again() {
        let none = VadOptions { max_buffered_speech: 0.0, prefix_padding_duration: 0.0, ..VadOptions::default() };
        let mut m = VadMachine::new(none, Box::new(ScriptedModel { probabilities: [0.9; 4].into(), duration: 0.0 }));
        m.push(window()).unwrap();
        assert!(m.speech_buffer_max_reached, "a zero-length buffer is full at once");

        m.update_options(VadOptions { max_buffered_speech: 1.0, ..none });
        assert_eq!(m.speech_buffer.len(), 16_000);
        assert!(!m.speech_buffer_max_reached);
        m.push(window()).unwrap();
        assert_eq!(m.speech_buffer_index, VAD_WINDOW_SAMPLES, "the grown buffer took the next window");
    }

    // Catches: a mismatched sample rate silently accepted or dropped.
    #[test]
    fn a_frame_at_a_different_sample_rate_is_rejected() {
        let mut m = machine(&[0.0]);
        m.push(window()).unwrap();
        let err = m.push(mono(vec![0u8; 64], 48_000)).expect_err("mismatched rate");
        assert_eq!(err, SampleRateMismatch { expected: 16_000, got: 48_000 });
    }

    // Catches: dropping the slow-inference signal.
    #[test]
    fn slow_inference_is_reported_with_its_accumulated_lag() {
        let mut m = timed_machine(&[0.0; 4], 0.3);
        let out = m.push(window()).unwrap();
        assert!(out[0].slow_inference, "300ms > the 200ms threshold");
        let window_duration = VAD_WINDOW_SAMPLES as f64 / f64::from(MODEL_SAMPLE_RATE);
        assert!((out[0].extra_inference_time - (0.3 - window_duration)).abs() < 1e-9);
        // ...and the lag accumulates rather than being reported per call.
        let out = m.push(window()).unwrap();
        assert!(out[0].extra_inference_time > 0.5);
    }
}
