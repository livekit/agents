//! `PrimarySpeakerDetector` (`stt/multi_speaker_adapter.py`): the primary speaker by RMS, and the
//! others formatted or suppressed.

use lk_agents_abi::{AudioByteStream, SpeechData, SpeechEvent, SpeechEventType};

/// `PrimarySpeakerDetectionOptions`, and how a transcript is rendered per speaker class.
pub use lk_agents_abi::voice::audio::{PrimarySpeakerOptions, SpeakerFormats};

#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum SpeakerClass {
    Primary,
    Background,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Forward,
    /// Drop it; a dropped final is replaced by [`clearing_event`].
    Suppress,
}

#[derive(Debug, Clone, PartialEq)]
struct SpeakerData {
    speaker_id: String,
    last_activity_time: f64,
    rms: f64,
}

/// Detects the primary speaker from RMS (closest to the mic), and formats or suppresses the
/// others. Switching away costs a `threshold_multiplier`, which decays while the primary is silent.
pub struct PrimarySpeakerDetector {
    opt: PrimarySpeakerOptions,
    formats: SpeakerFormats,
    detect_primary: bool,
    suppress_background: bool,

    pushed_duration: f64,
    primary_speaker: Option<String>,
    speakers: Vec<SpeakerData>,
    bstream: Option<AudioByteStream>,
    rms_buffer: Vec<f64>,
    frame_size: f64,
    max_rms_size: usize,
}

impl PrimarySpeakerDetector {
    /// Background suppression without primary detection is downgraded with a warning, as in the
    /// reference.
    pub fn new(
        detect_primary: bool,
        suppress_background: bool,
        opt: PrimarySpeakerOptions,
        formats: SpeakerFormats,
    ) -> Self {
        let frame_size = f64::from(opt.frame_size_ms) / 1000.0;
        PrimarySpeakerDetector {
            opt,
            formats,
            detect_primary,
            suppress_background: suppress_background && detect_primary,
            pushed_duration: 0.0,
            primary_speaker: None,
            speakers: Vec::new(),
            bstream: None,
            rms_buffer: Vec::new(),
            frame_size,
            // A memory bound from the nominal frame size, not recomputed.
            max_rms_size: (opt.rms_buffer_duration / frame_size) as usize,
        }
    }

    /// One frame of the single track, RMS over interleaved i16 as the reference does.
    pub fn push_audio(&mut self, data: &[u8], sample_rate: u32, num_channels: u16) {
        if !self.detect_primary {
            // `pushed_duration` is the clock every timestamp here is measured against.
            let spc = data.len() / (2 * num_channels.max(1) as usize);
            self.pushed_duration += spc as f64 / f64::from(sample_rate);
            return;
        }

        if self.bstream.is_none() {
            let samples_per_channel = (f64::from(sample_rate) * self.frame_size) as u32;
            self.bstream = Some(AudioByteStream::new(
                sample_rate,
                num_channels,
                Some(samples_per_channel),
                false,
            ));
            // The frame size the chunker actually produces.
            self.frame_size = f64::from(samples_per_channel) / f64::from(sample_rate);
        }

        let frames = self.bstream.as_mut().expect("just created").push(data);
        for f in frames {
            self.rms_buffer.push(rms_of(&f.data));
            self.pushed_duration += f64::from(f.samples_per_channel) / f64::from(sample_rate);
        }

        if self.rms_buffer.len() > self.max_rms_size {
            self.rms_buffer.drain(..self.rms_buffer.len() - self.max_rms_size);
        }
    }

    /// Label, rewrite or suppress one STT event in place. Only finals update the primary.
    pub fn on_stt_event(&mut self, ev: &mut SpeechEvent) -> Verdict {
        if ev.alternatives.is_empty() {
            return Verdict::Forward;
        }

        if ev.kind == SpeechEventType::FinalTranscript {
            let sd = ev.alternatives[0].clone();
            self.update_primary(&sd);
        }

        let sd = &mut ev.alternatives[0];
        let (Some(speaker_id), Some(primary)) =
            (sd.speaker_id.clone(), self.primary_speaker.clone())
        else {
            // `is_primary_speaker` stays unset rather than becoming `false`.
            return Verdict::Forward;
        };

        let is_primary = speaker_id == primary;
        sd.is_primary_speaker = Some(is_primary);
        if is_primary {
            sd.text = format_transcript(&self.formats.primary, &sd.text, &speaker_id);
            return Verdict::Forward;
        }
        if self.suppress_background {
            return Verdict::Suppress;
        }
        sd.text = format_transcript(&self.formats.background, &sd.text, &speaker_id);
        Verdict::Forward
    }

    /// `_update_primary_speaker`.
    fn update_primary(&mut self, sd: &SpeechData) {
        if !self.detect_primary {
            self.primary_speaker = None;
            return;
        }
        // An unattributed segment keeps the current primary (#6460).
        let Some(speaker_id) = sd.speaker_id.clone() else { return };
        let Some(rms) = self.rms_for_timerange(sd.start_time, sd.end_time) else { return };

        match self.speakers.iter_mut().find(|s| s.speaker_id == speaker_id) {
            Some(data) => {
                data.last_activity_time = sd.end_time;
                data.rms = data.rms * self.opt.rms_smoothing_factor
                    + rms * (1.0 - self.opt.rms_smoothing_factor);
            }
            None => self.speakers.push(SpeakerData {
                speaker_id: speaker_id.clone(),
                last_activity_time: sd.end_time,
                rms,
            }),
        }

        if self.primary_speaker.as_deref() == Some(speaker_id.as_str()) {
            return;
        }

        let primary = self
            .primary_speaker
            .as_ref()
            .and_then(|id| self.speakers.iter().find(|s| &s.speaker_id == id))
            .map(|s| (s.rms, s.last_activity_time));
        let Some((primary_rms, primary_last_activity)) = primary else {
            self.primary_speaker = Some(speaker_id);
            return;
        };

        let silence_duration = self.pushed_duration - primary_last_activity;
        // Decay so a departed primary doesn't hold the floor for the rest of the call.
        let decay_rate = if self.opt.threshold_multiplier > 1.0 {
            (self.opt.threshold_multiplier - 1.0) / self.opt.decay_to_equal_time
        } else {
            0.0
        };
        let multiplier = (self.opt.threshold_multiplier - decay_rate * silence_duration)
            .max(self.opt.threshold_min_multiplier);
        if rms > primary_rms * multiplier {
            self.primary_speaker = Some(speaker_id);
        }
    }

    /// Median RMS over a transcript's window, measured backwards from `pushed_duration`; `None`
    /// when too little audio covers it. Median, so one door slam doesn't crown a primary.
    fn rms_for_timerange(&self, start_time: f64, end_time: f64) -> Option<f64> {
        if self.rms_buffer.is_empty() {
            return None;
        }
        let len = self.rms_buffer.len() as i64;
        // `int()` truncates toward zero, not `floor`.
        let start_back = ((self.pushed_duration - start_time) / self.frame_size).trunc() as i64;
        let end_back = ((self.pushed_duration - end_time) / self.frame_size).trunc() as i64;
        let start = len - start_back - 1;
        let end = len - end_back;

        if end < 0 || start >= len {
            return None;
        }
        let start = start.max(0);
        if end - start < self.opt.min_rms_samples as i64 {
            return None;
        }
        // Python's slice clamps; the guard above uses the unclamped `end`.
        let window = &self.rms_buffer[start as usize..(end.min(len)) as usize];
        median(window)
    }
}

/// The empty final sent in place of a suppressed one, to clear the consumer's interims.
pub fn clearing_event() -> SpeechEvent {
    let mut ev = SpeechEvent::new(SpeechEventType::FinalTranscript);
    ev.alternatives.push(SpeechData::new("", ""));
    ev
}

/// `sqrt(mean(x**2))` over interleaved i16, in `f64` as numpy does.
pub fn rms_of(data: &[u8]) -> f64 {
    let n = data.len() / 2;
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = data
        .chunks_exact(2)
        .map(|b| {
            let s = f64::from(i16::from_le_bytes([b[0], b[1]]));
            s * s
        })
        .sum();
    (sum / n as f64).sqrt()
}

/// `np.median`: the middle value, or the mean of the two middle values.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        // numpy returns `nan`; unreachable unless `min_rms_samples <= 0`.
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("RMS values are never NaN"));
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

/// `"{text}"`/`"{speaker_id}"` plus `{{`/`}}`. An unknown field is left verbatim rather than
/// raising (a typo'd user format would kill the stream in Python).
pub fn format_transcript(fmt: &str, text: &str, speaker_id: &str) -> String {
    let mut out = String::with_capacity(fmt.len() + text.len());
    let mut rest = fmt;
    while let Some(i) = rest.find(['{', '}']) {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        if let Some(t) = tail.strip_prefix("{{") {
            out.push('{');
            rest = t;
        } else if let Some(t) = tail.strip_prefix("}}") {
            out.push('}');
            rest = t;
        } else if let Some(t) = tail.strip_prefix("{text}") {
            out.push_str(text);
            rest = t;
        } else if let Some(t) = tail.strip_prefix("{speaker_id}") {
            out.push_str(speaker_id);
            rest = t;
        } else {
            out.push_str(&tail[..1]);
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}
