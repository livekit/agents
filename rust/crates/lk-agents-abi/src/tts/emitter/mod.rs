//! `AudioEmitter` (`tts/tts.py:835`). A state machine, not a push sink: after 0.15 s sent it
//! compares wall clock against audio duration and arms its own flush timer ([`on_drift`](AudioEmitter::on_drift)).
//! PCM only: the driver decodes. No channel, so a misuse error surfaces from the call that made it.

use std::time::Duration;

use crate::TimedString;
use crate::{AudioByteStream, Frame, Ts};
use crate::{AudioFrame, SynthesizedAudio};

pub mod driver;
pub use driver::{Cmd, EmitterDriver, EmitterHost, HostEffect, PumpError};

/// What the machine asked for, in order: recorded, not called, so the machine lock is never held
/// across a host call.
pub type Effects = Vec<HostEffect>;

/// Below this much audio sent, the drift check is skipped (`tts/tts.py:1063`).
const MIN_DRIFT_AUDIO: f64 = 0.15;
/// Margin so the drift flush lands before the buffer runs dry (`tts/tts.py:1065`).
const DRIFT_MARGIN: f64 = 0.02;
/// Held back so the last audio of a segment can be tagged `is_final` (`tts/tts.py:1072`).
const TAIL_MS: u32 = 10;
/// The end-of-segment marker is 10 ms of silence: `sample_rate // 100` (`tts/tts.py:1155`).
const MARKER_DIVISOR: u32 = 100;

/// Every misuse the reference raises `RuntimeError` for. Messages are verbatim: plugin tests
/// assert on them.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmitterError {
    #[error("AudioEmitter already started")]
    AlreadyStarted,
    #[error("AudioEmitter isn't started")]
    NotStarted,
    #[error(
        "start_segment() can only be called when SynthesizeStream is initialized with stream=True"
    )]
    StartSegmentNotStreaming,
    #[error(
        "end_segment() can only be called when SynthesizeStream is initialized with stream=True"
    )]
    EndSegmentNotStreaming,
    #[error("start_segment() called before the previous segment was ended")]
    SegmentAlreadyOpen,
    #[error("start_segment() must be called before pushing audio data")]
    NoSegment,
}

/// Pulled out by the driver when `LK_DUMP_TTS` is on; the binding supplies the local-time stamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentDump {
    pub label: String,
    pub request_id: String,
    /// Empty when `stream=False`; the reference leaves it out of the filename in that case.
    pub segment_id: String,
    pub streaming: bool,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub frames: Vec<Frame>,
}

impl SegmentDump {
    pub const DIR: &'static str = "lk_dump";

    /// `dump_segment()`'s file name; `stamp` is the host's local-time `strftime`.
    pub fn file_name(&self, stamp: &str) -> String {
        if self.streaming {
            format!("{}_{}_{}_{}.wav", self.label, self.request_id, self.segment_id, stamp)
        } else {
            format!("{}_{}_{}.wav", self.label, self.request_id, stamp)
        }
    }

    pub fn path(&self, stamp: &str) -> std::path::PathBuf {
        std::path::Path::new(Self::DIR).join(self.file_name(stamp))
    }

    /// `rtc.combine_audio_frames(frames).to_wav_bytes()`.
    pub fn wav_bytes(&self) -> Vec<u8> {
        let data_len: usize = self.frames.iter().map(|f| f.data.len()).sum();
        let data_len = u32::try_from(data_len).unwrap_or(u32::MAX);
        let channels = u32::from(self.num_channels);
        let block_align = channels * 2;
        let mut out = Vec::with_capacity(44 + data_len as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36u32.saturating_add(data_len)).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&self.num_channels.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.saturating_mul(block_align).to_le_bytes());
        out.extend_from_slice(&(block_align as u16).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for f in &self.frames {
            out.extend_from_slice(&f.data);
        }
        out
    }

    /// Creates the directory (the reference would raise inside the emitter task).
    pub fn write(&self, stamp: &str) -> std::io::Result<std::path::PathBuf> {
        std::fs::create_dir_all(Self::DIR)?;
        let path = self.path(stamp);
        std::fs::write(&path, self.wav_bytes())?;
        Ok(path)
    }
}

#[derive(Debug, Clone)]
struct SegmentCtx {
    segment_id: String,
    audio_duration: f64,
}

pub struct AudioEmitter {
    label: String,
    request_id: String,
    started: bool,
    /// Every mutator returns silently once set, as after `end_input()`.
    closed: bool,
    streaming: bool,
    sample_rate: u32,
    num_channels: u16,
    frame_size_ms: u32,
    tail_samples: u32,

    num_segments: usize,
    /// Indexed by `pushed_duration`.
    audio_durations: Vec<f64>,
    provider_request_ids: Vec<String>,
    /// Pulled by the driver so the span write survives a `start_segment()` that is then rejected
    /// (the reference's order).
    pending_ids: Option<Vec<String>>,

    seg: Option<SegmentCtx>,
    abs: Option<AudioByteStream>,
    last_frame: Option<Frame>,
    timed_transcripts: Vec<TimedString>,

    // --- the drift watchdog ---
    sent_start: Option<Ts>,
    sent_duration: f64,
    timer_armed: bool,

    // --- LK_DUMP_TTS ---
    dump: bool,
    debug_frames: Vec<Frame>,
    pending_dumps: Vec<SegmentDump>,

    /// Bytes `AudioByteStream::flush` discarded as a partial sample; the binding logs them.
    dropped_bytes: usize,
}

impl AudioEmitter {
    pub fn new(label: impl Into<String>) -> Self {
        AudioEmitter {
            label: label.into(),
            request_id: String::new(),
            started: false,
            closed: false,
            streaming: false,
            sample_rate: 0,
            num_channels: 0,
            frame_size_ms: 0,
            tail_samples: 0,
            num_segments: 0,
            audio_durations: Vec::new(),
            provider_request_ids: Vec::new(),
            pending_ids: None,
            seg: None,
            abs: None,
            last_frame: None,
            timed_transcripts: Vec::new(),
            sent_start: None,
            sent_duration: 0.0,
            timer_armed: false,
            dump: false,
            debug_frames: Vec::new(),
            pending_dumps: Vec::new(),
            dropped_bytes: 0,
        }
    }

    // ----------------------------------------------------------------- lifecycle

    /// PCM only. An empty `request_id` becomes `"unknown"`.
    pub fn initialize(
        &mut self,
        request_id: &str,
        sample_rate: u32,
        num_channels: u16,
        frame_size_ms: u32,
        stream: bool,
    ) -> Result<(), EmitterError> {
        if self.started {
            return Err(EmitterError::AlreadyStarted);
        }
        self.started = true;
        self.request_id =
            if request_id.is_empty() { "unknown".to_string() } else { request_id.to_string() };
        self.sample_rate = sample_rate;
        self.num_channels = num_channels;
        self.frame_size_ms = frame_size_ms;
        self.streaming = stream;
        self.tail_samples = sample_rate * TAIL_MS / 1000;

        if !self.streaming {
            // "always start a segment with stream=False" (`tts/tts.py:917`).
            self.begin_segment("")?;
        }
        Ok(())
    }

    pub fn started(&self) -> bool {
        self.started
    }

    pub fn num_segments(&self) -> usize {
        self.num_segments
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn provider_request_ids(&self) -> &[String] {
        &self.provider_request_ids
    }

    /// Python indexing; the reference default is `idx=-1`. Out-of-range is `0.0`.
    pub fn pushed_duration(&self, idx: i64) -> f64 {
        let n = self.audio_durations.len() as i64;
        if idx >= -n && idx < n {
            let i = if idx < 0 { (n + idx) as usize } else { idx as usize };
            self.audio_durations[i]
        } else {
            0.0
        }
    }

    // ----------------------------------------------------------------- segments

    /// Streaming only. `true` when `segment_id` was new (the binding then writes the span attr).
    pub fn start_segment(&mut self, segment_id: &str) -> Result<bool, EmitterError> {
        if !self.streaming {
            return Err(EmitterError::StartSegmentNotStreaming);
        }
        let noted = self.note_provider_request_id(segment_id);
        self.begin_segment(segment_id)?;
        Ok(noted)
    }

    /// A plugin may learn the id after `start_segment` (`tts/tts.py:947`).
    pub fn note_provider_request_id(&mut self, context_id: &str) -> bool {
        if context_id.is_empty() || self.provider_request_ids.iter().any(|id| id == context_id) {
            return false;
        }
        self.provider_request_ids.push(context_id.to_string());
        self.pending_ids = Some(self.provider_request_ids.clone());
        true
    }

    /// If it changed since the last call.
    pub fn take_provider_ids_update(&mut self) -> Option<Vec<String>> {
        self.pending_ids.take()
    }

    fn begin_segment(&mut self, segment_id: &str) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        // Incremented before validation, as the reference: `SynthesizeStream` compares
        // `num_segments` on the error path.
        self.num_segments += 1;
        if self.seg.is_some() {
            return Err(EmitterError::SegmentAlreadyOpen);
        }
        self.audio_durations.push(0.0);
        self.seg = Some(SegmentCtx { segment_id: segment_id.to_string(), audio_duration: 0.0 });
        Ok(())
    }

    /// Streaming only.
    pub fn end_segment(&mut self, now: Ts, eff: &mut Effects) -> Result<(), EmitterError> {
        if !self.streaming {
            return Err(EmitterError::EndSegmentNotStreaming);
        }
        if !self.live()? {
            return Ok(());
        }
        self.handle_end_segment(now, eff)
    }

    /// `false` once closed (mutators are then silent no-ops); an error before `initialize()`.
    fn live(&self) -> Result<bool, EmitterError> {
        if !self.started {
            return Err(EmitterError::NotStarted);
        }
        Ok(!self.closed)
    }

    // ----------------------------------------------------------------- input

    /// Bytes, not `&[i16]`: providers routinely push half a sample.
    pub fn push(&mut self, now: Ts, data: &[u8], eff: &mut Effects) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        if self.seg.is_none() {
            return Err(EmitterError::NoSegment);
        }
        if self.abs.is_none() {
            // `progressive=True`: the first frame is 20 ms so the agent starts talking sooner.
            let spc = self.sample_rate / 1000 * self.frame_size_ms;
            self.abs =
                Some(AudioByteStream::new(self.sample_rate, self.num_channels, Some(spc), true));
        }
        let frames = self.abs.as_mut().expect("just set").push(data);
        for f in frames {
            self.emit_frame(Some(f), false, now, eff);
        }
        Ok(())
    }

    /// The words ride on the *next* frame this emitter sends.
    pub fn push_timed_transcript(
        &mut self,
        words: impl IntoIterator<Item = TimedString>,
    ) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        self.timed_transcripts.extend(words);
        Ok(())
    }

    /// Releases the held-back tail without ending the segment.
    pub fn flush(&mut self, now: Ts, eff: &mut Effects) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        self.handle_flush(now, eff)
    }

    /// Every later mutator is a silent no-op.
    pub fn end_input(&mut self, now: Ts, eff: &mut Effects) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        let r = self.handle_end_segment(now, eff);
        self.closed = true;
        r
    }

    /// Safe on an emitter never initialised.
    pub fn stop(&mut self, eff: &mut Effects) {
        if self.timer_armed {
            eff.push(HostEffect::Cancel);
            self.timer_armed = false;
        }
        self.closed = true;
    }

    /// The drift watchdog firing (`tts/tts.py:1057`): releases the held-back tail before the
    /// consumer runs dry.
    pub fn on_drift(&mut self, now: Ts, eff: &mut Effects) -> Result<(), EmitterError> {
        // The timer fired; forget it so `_flush_frame` does not cancel it.
        self.timer_armed = false;
        self.flush(now, eff)
    }

    // ----------------------------------------------------------------- LK_DUMP_TTS

    /// A real field so the reference's monkeypatching tests keep working through the binding.
    pub fn set_dump(&mut self, on: bool) {
        self.dump = on;
    }

    pub fn dump(&self) -> bool {
        self.dump
    }

    /// Always empty when dumping is off.
    pub fn take_dumps(&mut self) -> Vec<SegmentDump> {
        std::mem::take(&mut self.pending_dumps)
    }

    pub fn take_dropped_bytes(&mut self) -> usize {
        std::mem::replace(&mut self.dropped_bytes, 0)
    }

    // ----------------------------------------------------------------- the loop body

    fn handle_flush(&mut self, now: Ts, eff: &mut Effects) -> Result<(), EmitterError> {
        if self.seg.is_none() {
            // Streaming: an empty segment, ignored (`tts/tts.py:1259`).
            return if self.streaming { Ok(()) } else { Err(EmitterError::NoSegment) };
        }
        if self.abs.is_none() {
            // `elif audio_byte_stream:`: a flush before the first byte does nothing.
            return Ok(());
        }
        let flushed = self.abs.as_mut().expect("checked").flush();
        self.dropped_bytes += flushed.dropped_bytes;
        for f in flushed.frames {
            self.emit_frame(Some(f), false, now, eff);
        }
        self.flush_frame(now, eff);
        self.abs.as_mut().expect("checked").clear();
        Ok(())
    }

    fn handle_end_segment(&mut self, now: Ts, eff: &mut Effects) -> Result<(), EmitterError> {
        if self.seg.is_none() {
            return if self.streaming { Ok(()) } else { Err(EmitterError::NoSegment) };
        }
        if self.abs.is_none() {
            // Preserved reference quirk: a segment that never received a byte is never closed, so
            // the next `start_segment()` raises. Pinned by a test.
            return Ok(());
        }
        let flushed = self.abs.as_mut().expect("checked").flush();
        self.dropped_bytes += flushed.dropped_bytes;
        for f in flushed.frames {
            self.emit_frame(Some(f), false, now, eff);
        }
        self.emit_frame(None, true, now, eff);
        self.dump_segment();
        self.seg = None;
        self.abs = None;
        self.last_frame = None;
        Ok(())
    }

    // ----------------------------------------------------------------- framing

    fn emit_frame(&mut self, frame: Option<Frame>, is_final: bool, now: Ts, eff: &mut Effects) {
        if is_final {
            match (self.last_frame.take(), frame) {
                (Some(tail), Some(f)) => {
                    let combined = combine(tail, f);
                    self.do_send(combined, true, now, eff);
                }
                (Some(tail), None) => self.do_send(tail, true, now, eff),
                (None, Some(f)) => self.do_send(f, true, now, eff),
                (None, None) => {
                    // The segment produced audio: send 10 ms of silence so it gets an `is_final`.
                    // Not counted in `pushed_duration`.
                    if self.seg.as_ref().is_some_and(|s| s.audio_duration > 0.0) {
                        let marker = self.marker_frame();
                        let dur = self.frame_duration(&marker);
                        let ev = self.build_event(marker, true);
                        self.send_audio(ev, false, dur, now, eff);
                    }
                }
            }
            return;
        }

        let Some(f) = frame else { return };
        let combined = match self.last_frame.take() {
            Some(tail) => combine(tail, f),
            None => f,
        };
        let (head, tail) = self.split_tail(combined);
        if let Some(head) = head {
            self.do_send(head, false, now, eff);
        }
        self.last_frame = Some(tail);
    }

    /// `(None, frame)` when the frame is too small to give a tail away. The reference's
    /// `split_idx` is in int16 samples; doubled here for bytes.
    fn split_tail(&self, frame: Frame) -> (Option<Frame>, Frame) {
        if frame.samples_per_channel <= self.tail_samples {
            return (None, frame);
        }
        let head_samples = frame.samples_per_channel - self.tail_samples;
        let split = (head_samples as usize)
            .saturating_mul(self.num_channels as usize)
            .saturating_mul(size_of::<i16>())
            .min(frame.data.len());
        let tail_data = frame.data[split..].to_vec();
        let mut head_data = frame.data;
        head_data.truncate(split);
        (
            Some(Frame { data: head_data, samples_per_channel: head_samples }),
            Frame { data: tail_data, samples_per_channel: self.tail_samples },
        )
    }

    fn do_send(&mut self, frame: Frame, is_final: bool, now: Ts, eff: &mut Effects) {
        let dur = self.frame_duration(&frame);
        if self.dump {
            self.debug_frames.push(frame.clone());
        }
        let ev = self.build_event(frame, is_final);
        // A final frame has nothing held back for a drift flush to release.
        self.send_audio(ev, !is_final, dur, now, eff);
        self.account(dur);
    }

    fn account(&mut self, dur: f64) {
        if let Some(seg) = self.seg.as_mut() {
            seg.audio_duration += dur;
        }
        if let Some(d) = self.audio_durations.last_mut() {
            *d += dur;
        }
    }

    fn flush_frame(&mut self, now: Ts, eff: &mut Effects) {
        let Some(tail) = self.last_frame.take() else {
            return;
        };
        let dur = self.frame_duration(&tail);
        if self.dump {
            self.debug_frames.push(tail.clone());
        }
        let ev = self.build_event(tail, false);
        self.send_audio(ev, false, dur, now, eff);
        self.account(dur);

        // Restart the drift measurement, or the next frame would flush immediately, forever.
        self.sent_start = None;
        self.sent_duration = 0.0;
        if self.timer_armed {
            eff.push(HostEffect::Cancel);
            self.timer_armed = false;
        }
    }

    fn send_audio(
        &mut self,
        ev: SynthesizedAudio,
        flush_if_delayed: bool,
        dur: f64,
        now: Ts,
        eff: &mut Effects,
    ) {
        eff.push(HostEffect::Send(Box::new(ev)));

        if self.sent_start.is_none() {
            self.sent_start = Some(now);
        }
        self.sent_duration += dur;

        if self.timer_armed {
            eff.push(HostEffect::Cancel);
            self.timer_armed = false;
        }

        if flush_if_delayed && self.sent_duration > MIN_DRIFT_AUDIO {
            let start = self.sent_start.unwrap_or(now);
            let elapsed = now.saturating_duration_since(start).as_secs_f64();
            let delay = self.sent_duration - elapsed - DRIFT_MARGIN;
            // `delay == 0.0` is unreachable in doubles here; `>` kept as the reference writes it.
            if delay > 0.0 {
                // Relative, so the host's clock need not agree with `now`.
                eff.push(HostEffect::Arm(Duration::from_secs_f64(delay)));
                self.timer_armed = true;
            }
            // `delay <= 0`: the provider is already behind; the next frame re-evaluates.
        }
    }

    fn build_event(&mut self, frame: Frame, is_final: bool) -> SynthesizedAudio {
        SynthesizedAudio {
            frame: AudioFrame::new(
                frame.data,
                self.sample_rate,
                self.num_channels,
                frame.samples_per_channel,
            ),
            request_id: self.request_id.clone(),
            is_final,
            segment_id: self.seg.as_ref().map(|s| s.segment_id.clone()).unwrap_or_default(),
            delta_text: String::new(),
            // Stamped by `ChunkedStream`/`SynthesizeStream` on the way out (`tts.py:502`).
            started_time: None,
            // Consumed, not copied, or the synchroniser speaks the words twice.
            timed_transcript: std::mem::take(&mut self.timed_transcripts),
            encoding: crate::AudioEncoding::Pcm,
        }
    }

    fn marker_frame(&self) -> Frame {
        let samples = self.sample_rate / MARKER_DIVISOR;
        let len = (samples as usize)
            .saturating_mul(self.num_channels as usize)
            .saturating_mul(size_of::<i16>());
        Frame { data: vec![0u8; len], samples_per_channel: samples }
    }

    fn frame_duration(&self, f: &Frame) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        f64::from(f.samples_per_channel) / f64::from(self.sample_rate)
    }

    fn dump_segment(&mut self) {
        if !self.dump || self.debug_frames.is_empty() {
            return;
        }
        self.pending_dumps.push(SegmentDump {
            label: self.label.clone(),
            request_id: self.request_id.clone(),
            segment_id: self.seg.as_ref().map(|s| s.segment_id.clone()).unwrap_or_default(),
            streaming: self.streaming,
            sample_rate: self.sample_rate,
            num_channels: self.num_channels,
            frames: std::mem::take(&mut self.debug_frames),
        });
    }
}

fn combine(mut a: Frame, b: Frame) -> Frame {
    a.data.extend_from_slice(&b.data);
    a.samples_per_channel += b.samples_per_channel;
    a
}

#[cfg(test)]
mod tests;
