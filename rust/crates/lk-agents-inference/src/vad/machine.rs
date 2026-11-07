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
    /// The runtime resets the model's recurrent state with it. Speech in progress ends first, so
    /// the session is never left speaking.
    pub fn reset(&mut self) -> Option<VadOutput> {
        let ended = self.pub_speaking.then(|| self.end_at_boundary());
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
        ended
    }

    /// Speech in progress ends at the reset, after the audio received but not yet scored (the
    /// window in flight and an incomplete last window): it is the tail of the speech.
    fn end_at_boundary(&mut self) -> VadOutput {
        let tail = self.input.make_contiguous();
        let copied = tail.len().min(self.speech_buffer.len() - self.speech_buffer_index);
        self.speech_buffer[self.speech_buffer_index..self.speech_buffer_index + copied]
            .copy_from_slice(&tail[..copied]);
        self.speech_buffer_index += copied;
        let duration = tail.len() as f64 / f64::from(self.input_sample_rate);
        self.pub_current_sample += (duration * f64::from(MODEL_SAMPLE_RATE)).round() as u64;
        self.pub_timestamp += duration;
        self.pub_speech_duration += duration;
        if self.silence_threshold_duration > 0.0 {
            self.silence_threshold_duration += duration;
        }
        self.pub_silence_duration = self.silence_threshold_duration;
        let speech = (self.pub_speech_duration - self.silence_threshold_duration).max(0.0);
        self.edge(VadEventType::EndOfSpeech, speech)
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
#[path = "machine_tests.rs"]
mod tests;
