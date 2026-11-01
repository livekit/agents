//! `inference/vad.py`'s `_VADStream._main_task` as a synchronous machine: it cuts the input into
//! model windows ([`VadPredict`]), one in flight at a time so a recurrent model sees them in order,
//! and interprets each supplied [`VadResult`] and its measured duration. The runtime executes the
//! model; no model, clock or task appears here. The slow-inference lag is reported on
//! [`VadOutput::extra_inference_time`]; the host logs it.

use std::collections::VecDeque;

use crate::abi::AudioFrame;
use lk_agents_abi::{VadEvent, VadEventType};
use lk_agents_media::{AudioFormat, AudioResampler, Quality, pcm_bytes, pcm_samples};

/// `VAD_WINDOW_SAMPLES`, from `livekit.local_inference`. Silero's frame size at 16 kHz.
pub const VAD_WINDOW_SAMPLES: usize = 512;

pub const MODEL_SAMPLE_RATE: u32 = 16_000;

pub const SLOW_INFERENCE_THRESHOLD: f64 = 0.2;

/// `rtc.AudioResampler`'s default quality, mono: every frame is read as one channel. A construction
/// failure yields `None` rather than aborting.
fn new_resampler(input_rate: u32) -> Option<AudioResampler> {
    AudioResampler::fixed(AudioFormat::new(input_rate, 1), MODEL_SAMPLE_RATE, Quality::Medium).ok()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InvalidVadOptions {
    #[error("VAD durations must be finite and non-negative")]
    Duration,
    #[error("VAD thresholds must lie in [0, 1] with deactivation at most activation")]
    Threshold,
}

impl VadOptions {
    /// Checked before any stream uses them: a negative or non-finite duration would size the
    /// speech buffer from garbage, and inverted thresholds disable the hysteresis.
    pub fn validate(&self) -> Result<(), InvalidVadOptions> {
        let durations = [
            self.min_speech_duration,
            self.min_silence_duration,
            self.prefix_padding_duration,
            self.max_buffered_speech,
        ];
        if durations.iter().any(|d| !d.is_finite() || *d < 0.0) {
            return Err(InvalidVadOptions::Duration);
        }
        let unit = |t: f64| (0.0..=1.0).contains(&t);
        if !unit(self.activation_threshold)
            || !unit(self.deactivation_threshold)
            || self.deactivation_threshold > self.activation_threshold
        {
            return Err(InvalidVadOptions::Threshold);
        }
        Ok(())
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

/// A model window to score: [`VAD_WINDOW_SAMPLES`] samples at [`MODEL_SAMPLE_RATE`].
#[derive(Debug, Clone, PartialEq)]
pub struct VadPredict {
    pub window: u64,
    pub samples: Vec<i16>,
}

/// The model's answer for a window and how long the runtime measured it took.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadResult {
    pub window: u64,
    pub probability: f32,
    pub inference_duration: f64,
}

pub struct VadMachine {
    opts: VadOptions,
    resampler: Option<AudioResampler>,

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

    /// Input-rate samples not yet attributed to a scored window.
    input: VecDeque<i16>,
    /// Model-rate samples not yet cut into a window.
    inference: VecDeque<i16>,
    /// Avoids drift when the sample-rate ratio is not an integer.
    input_copy_remaining_fract: f64,
    extra_inference_time: f64,

    next_window: u64,
    /// The window whose result is awaited; a result for any other is stale.
    in_flight: Option<u64>,
}

impl VadMachine {
    pub fn new(opts: VadOptions) -> Self {
        VadMachine {
            opts,
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
            input: VecDeque::new(),
            inference: VecDeque::new(),
            input_copy_remaining_fract: 0.0,
            extra_inference_time: 0.0,
            next_window: 0,
            in_flight: None,
        }
    }

    /// The `_FlushSentinel` branch: every counter restarts and the window in flight goes stale.
    /// The runtime resets the model's recurrent state with it.
    pub fn reset(&mut self) {
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
        self.input.clear();
        self.inference.clear();
        self.input_copy_remaining_fract = 0.0;
        self.extra_inference_time = 0.0;
        self.in_flight = None;
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
        self.speech_buffer
            .resize((opts.max_buffered_speech * rate) as usize + self.prefix_padding_samples, 0);
        // The reference leaves the cursor past a shrunk buffer, where its next rewind or copy
        // raises; clamped instead.
        self.speech_buffer_index = self.speech_buffer_index.min(self.speech_buffer.len());
        if grew {
            self.speech_buffer_max_reached = false;
        }
    }

    /// Buffer one frame; the next window to score once one is complete and none is in flight. A
    /// frame at a different sample rate from the first is `Err` (the reference logs and drops it).
    pub fn push(&mut self, frame: AudioFrame) -> Result<Option<VadPredict>, SampleRateMismatch> {
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

        let frame = if frame.num_channels > 1 {
            let pcm = pcm_samples(&frame.data);
            let mono = lk_agents_media::downmix_mono(&pcm, frame.num_channels);
            AudioFrame::new(pcm_bytes(&mono), frame.sample_rate, 1, mono.len() as u32)
        } else {
            frame
        };
        match self.resampler.as_mut() {
            // Empty is normal (soxr buffers), and a failed push is also empty rather than a panic.
            Some(resampler) => {
                for converted in resampler.push_frame(&frame).unwrap_or_default() {
                    self.inference.extend(pcm_samples(&converted.data));
                }
            }
            None => self.inference.extend(pcm_samples(&frame.data)),
        }
        self.input.extend(pcm_samples(&frame.data));
        Ok(self.next_predict())
    }

    /// Apply the model's answer for the window in flight; the outputs it produced and the next
    /// window to score. A stale result (an earlier window, or one cut before a reset) changes
    /// nothing.
    pub fn predicted(&mut self, result: VadResult) -> (Vec<VadOutput>, Option<VadPredict>) {
        if self.in_flight != Some(result.window) {
            return (Vec::new(), None);
        }
        self.in_flight = None;
        let outputs = self.apply(f64::from(result.probability), result.inference_duration);
        (outputs, self.next_predict())
    }

    fn next_predict(&mut self) -> Option<VadPredict> {
        if self.in_flight.is_some() || self.inference.len() < VAD_WINDOW_SAMPLES {
            return None;
        }
        let window = self.next_window;
        self.next_window += 1;
        self.in_flight = Some(window);
        Some(VadPredict { window, samples: self.inference.drain(..VAD_WINDOW_SAMPLES).collect() })
    }

    fn apply(&mut self, probability: f64, inference_duration: f64) -> Vec<VadOutput> {
        let mut out = Vec::new();
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

        let window_input: Vec<i16> =
            self.input.drain(..to_copy_int.min(self.input.len())).collect();
        let available_space = self.speech_buffer.len() - self.speech_buffer_index;
        let to_copy_buffer = window_input.len().min(available_space);
        if to_copy_buffer > 0 {
            self.speech_buffer[self.speech_buffer_index..self.speech_buffer_index + to_copy_buffer]
                .copy_from_slice(&window_input[..to_copy_buffer]);
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

        out.push(VadOutput {
            event: VadEvent {
                kind: VadEventType::InferenceDone,
                samples_index: self.pub_current_sample,
                timestamp: self.pub_timestamp,
                speech_duration: self.pub_speech_duration,
                silence_duration: self.pub_silence_duration,
                frames: Vec::new(),
                probability,
                inference_duration,
                speaking: self.pub_speaking,
                raw_accumulated_silence: self.silence_threshold_duration,
                raw_accumulated_speech: self.speech_threshold_duration,
            },
            frames: vec![mono(pcm_bytes(&window_input), self.input_sample_rate)],
            extra_inference_time: self.extra_inference_time,
            slow_inference,
        });

        // Hysteresis: `p >= activation` starts speech; once speaking, `p > deactivation` keeps it.
        if probability >= self.opts.activation_threshold
            || (self.pub_speaking && probability > self.opts.deactivation_threshold)
        {
            self.speech_threshold_duration += window_duration;
            self.silence_threshold_duration = 0.0;

            if !self.pub_speaking && self.speech_threshold_duration >= self.opts.min_speech_duration
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
                frames: Vec::new(),
                probability: 0.0,
                inference_duration: 0.0,
                speaking: kind == VadEventType::StartOfSpeech,
                raw_accumulated_silence: 0.0,
                raw_accumulated_speech: 0.0,
            },
            frames: vec![mono(
                pcm_bytes(&self.speech_buffer[..self.speech_buffer_index]),
                self.input_sample_rate,
            )],
            extra_inference_time: self.extra_inference_time,
            slow_inference: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a frame at {got} Hz was pushed after the stream settled on {expected} Hz")]
pub struct SampleRateMismatch {
    pub expected: u32,
    pub got: u32,
}

pub(crate) fn mono(data: Vec<u8>, sample_rate: u32) -> AudioFrame {
    let samples_per_channel = (data.len() / 2) as u32;
    AudioFrame::new(data, sample_rate, 1, samples_per_channel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The machine with every window answered at once: each scores the next scripted
    /// probability, measured at `duration` seconds.
    struct Scripted {
        m: VadMachine,
        probabilities: VecDeque<f32>,
        duration: f64,
    }

    impl Scripted {
        fn push(&mut self, frame: AudioFrame) -> Result<Vec<VadOutput>, SampleRateMismatch> {
            let mut out = Vec::new();
            let mut next = self.m.push(frame)?;
            while let Some(predict) = next {
                let probability = self.probabilities.pop_front().unwrap_or(0.0);
                let result = VadResult {
                    window: predict.window,
                    probability,
                    inference_duration: self.duration,
                };
                let (outputs, following) = self.m.predicted(result);
                out.extend(outputs);
                next = following;
            }
            Ok(out)
        }
    }

    fn scripted(opts: VadOptions, probabilities: &[f32], duration: f64) -> Scripted {
        Scripted {
            m: VadMachine::new(opts),
            probabilities: probabilities.iter().copied().collect(),
            duration,
        }
    }

    fn timed_machine(probabilities: &[f32], duration: f64) -> Scripted {
        scripted(VadOptions::default(), probabilities, duration)
    }

    /// Fed at [`MODEL_SAMPLE_RATE`], so no resampler is built.
    fn machine(probabilities: &[f32]) -> Scripted {
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
        assert!(m.m.resampler.is_none(), "the model rate needs no conversion");
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

    // Catches: a flush leaving the counters running, or a result for a window cut before the
    // flush applied to the stream after it (the model's state was reset in between).
    #[test]
    fn vad_eot_supplied_result_vectors_ignore_results_from_before_a_flush() {
        let mut m = VadMachine::new(VadOptions::default());
        let stale = m.push(window()).unwrap().expect("a whole window is scored");
        m.reset();
        let current = m.push(window()).unwrap().expect("the flushed stream scores again");
        let late = VadResult { window: stale.window, probability: 1.0, inference_duration: 0.0 };
        assert_eq!(m.predicted(late), (Vec::new(), None), "a stale result changes nothing");
        let (out, next) = m.predicted(VadResult {
            window: current.window,
            probability: 0.0,
            inference_duration: 0.0,
        });
        assert_eq!(out[0].event.samples_index, VAD_WINDOW_SAMPLES as u64, "the counters restarted");
        assert_eq!(next, None);
    }

    // Catches: a second window handed out while one is in flight: a recurrent model would score
    // them out of order. Each window follows the previous result.
    #[test]
    fn windows_are_scored_one_at_a_time_in_order() {
        let mut m = VadMachine::new(VadOptions::default());
        let first = m.push(filled_window(1)).unwrap().unwrap();
        assert_eq!(m.push(filled_window(2)).unwrap(), None, "one window in flight");
        let (_, second) = m.predicted(VadResult {
            window: first.window,
            probability: 0.0,
            inference_duration: 0.0,
        });
        let second = second.expect("the queued window follows the result");
        assert!(first.samples.iter().all(|s| *s == 0x0101));
        assert!(second.samples.iter().all(|s| *s == 0x0202));
    }

    // Catches: options that size the speech buffer from garbage or invert the hysteresis
    // accepted.
    #[test]
    fn vad_options_are_validated() {
        assert_eq!(VadOptions::default().validate(), Ok(()));
        let bad = [
            VadOptions { min_silence_duration: -0.1, ..VadOptions::default() },
            VadOptions { max_buffered_speech: f64::NAN, ..VadOptions::default() },
        ];
        for opts in bad {
            assert_eq!(opts.validate(), Err(InvalidVadOptions::Duration));
        }
        let inverted = VadOptions {
            activation_threshold: 0.3,
            deactivation_threshold: 0.4,
            ..VadOptions::default()
        };
        assert_eq!(inverted.validate(), Err(InvalidVadOptions::Threshold));
        let above = VadOptions { activation_threshold: 1.5, ..VadOptions::default() };
        assert_eq!(above.validate(), Err(InvalidVadOptions::Threshold));
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
        m.m.update_options(VadOptions { activation_threshold: 0.9, ..VadOptions::default() });
        for _ in 0..3 {
            assert_eq!(kinds(&m.push(window()).unwrap()), [VadEventType::InferenceDone]);
        }
    }

    // Catches: the speech buffer keeping its first size, or a grown buffer staying "full".
    #[test]
    fn a_buffer_updated_mid_stream_is_resized_and_may_fill_again() {
        let none = VadOptions {
            max_buffered_speech: 0.0,
            prefix_padding_duration: 0.0,
            ..VadOptions::default()
        };
        let mut m = scripted(none, &[0.9; 4], 0.0);
        m.push(window()).unwrap();
        assert!(m.m.speech_buffer_max_reached, "a zero-length buffer is full at once");

        m.m.update_options(VadOptions { max_buffered_speech: 1.0, ..none });
        assert_eq!(m.m.speech_buffer.len(), 16_000);
        assert!(!m.m.speech_buffer_max_reached);
        m.push(window()).unwrap();
        assert_eq!(
            m.m.speech_buffer_index, VAD_WINDOW_SAMPLES,
            "the grown buffer took the next window"
        );
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
