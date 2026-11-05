//! The synchronous PCM-framing half of TTS `AudioEmitter`.
//!
//! Provider bytes become progressively sized audio frames. A short tail is retained for the
//! final-segment marker; if generation falls behind, a scheduled flush releases it before the
//! consumer runs dry. Transcript alignment and provider usage ride on emitted audio.
//!
//! This owner performs no decoding, I/O, host call, spawn, or clock read. Its caller supplies
//! time and performs `AudioEmitterEffect` requests in order. The inference crate owns decoding,
//! delivery, and the flush timer; `on_flush_deadline` is the actual timer input, not a generic
//! progress callback. It produces synthesized audio, not playback/room delivery decisions.

use std::time::Duration;

use super::effects::AudioEmitterEffect;

use crate::{AudioByteStream, Frame};
use lk_agents_abi::media::emitter::EmitterError;
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{AudioEncoding, AudioFrame, SynthesizedAudio};
use lk_agents_abi::{TranscriptChunk, Ts};

/// Below this much audio sent, the drift check is skipped (`tts/tts.py:1063`).
const MIN_DRIFT_AUDIO: f64 = 0.15;
/// Margin so the drift flush lands before the buffer runs dry (`tts/tts.py:1065`).
const DRIFT_MARGIN: f64 = 0.02;
/// Held back so the last audio of a segment can be tagged `is_final` (`tts/tts.py:1072`).
const TAIL_MS: u32 = 10;
/// The end-of-segment marker is 10 ms of silence: `sample_rate // 100` (`tts/tts.py:1155`).
const MARKER_DIVISOR: u32 = 100;

#[derive(Debug, Clone)]
struct SegmentCtx {
    segment_id: String,
    audio_duration: f64,
    /// What the provider reported, carried on the final frame.
    usage: Usage,
}

/// What the provider reported billing; reports add up.
#[derive(Debug, Clone, Default)]
struct Usage {
    tokens: Option<TokenUsage>,
    characters_count: Option<u64>,
}

impl Usage {
    fn add(&mut self, tokens: Option<TokenUsage>, characters_count: Option<u64>) {
        if let Some(tokens) = tokens {
            match &mut self.tokens {
                Some(total) => total.add(&tokens),
                None => self.tokens = Some(tokens),
            }
        }
        if let Some(n) = characters_count {
            *self.characters_count.get_or_insert(0) += n;
        }
    }
}

#[derive(Default)]
pub struct AudioEmitter {
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
    /// Learnt since the last frame: the next frame carries them, even past a `start_segment()`
    /// that is then rejected (the reference's order).
    pending_ids: Vec<String>,

    seg: Option<SegmentCtx>,
    /// Reported while no segment was open; it rides on the next segment.
    unattached: Usage,
    abs: Option<AudioByteStream>,
    last_frame: Option<Frame>,
    timed_transcripts: Vec<TranscriptChunk>,

    // --- the drift watchdog ---
    sent_start: Option<Ts>,
    sent_duration: f64,
    flush_scheduled: bool,

    /// Bytes of a partial sample an ended segment discarded; the binding logs them.
    dropped_bytes: usize,
}

impl AudioEmitter {
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

    /// The PCM format `push` takes, set by `initialize`.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn num_channels(&self) -> u16 {
        self.num_channels
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
        self.pending_ids.push(context_id.to_string());
        true
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
        self.seg = Some(SegmentCtx {
            segment_id: segment_id.to_string(),
            audio_duration: 0.0,
            usage: std::mem::take(&mut self.unattached),
        });
        Ok(())
    }

    /// Streaming only.
    pub fn end_segment(
        &mut self,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) -> Result<(), EmitterError> {
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
    pub fn push(
        &mut self,
        now: Ts,
        data: &[u8],
        eff: &mut Vec<AudioEmitterEffect>,
    ) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        if self.seg.is_none() {
            return Err(EmitterError::NoSegment);
        }
        let frames = self.byte_stream().push(data);
        for f in frames {
            self.emit_frame(Some(f), false, now, eff);
        }
        Ok(())
    }

    /// `push_frame`: an already framed chunk, sent as it is, skipping the progressive re-chunking
    /// of [`push`](Self::push). For adapters over a TTS whose emitter already framed its audio.
    pub fn push_frame(
        &mut self,
        now: Ts,
        frame: Frame,
        eff: &mut Vec<AudioEmitterEffect>,
    ) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        if self.seg.is_none() {
            return Err(EmitterError::NoSegment);
        }
        // The segment has audio now, so its end is sent.
        self.byte_stream();
        self.emit_frame(Some(frame), false, now, eff);
        Ok(())
    }

    /// The segment's byte stream, made on its first audio. `progressive=True`: the first frame is
    /// 20 ms so the agent starts talking sooner.
    fn byte_stream(&mut self) -> &mut AudioByteStream {
        let spc = self.sample_rate / 1000 * self.frame_size_ms;
        self.abs.get_or_insert_with(|| {
            AudioByteStream::new(self.sample_rate, self.num_channels, Some(spc), true)
        })
    }

    /// The words ride on the *next* frame this emitter sends.
    pub fn push_timed_transcript(
        &mut self,
        words: impl IntoIterator<Item = TranscriptChunk>,
    ) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        self.timed_transcripts.extend(words);
        Ok(())
    }

    /// What the provider billed for the open segment, or the next one when none is open (a
    /// streaming provider can report after its segment ended); reports add up.
    pub fn report_usage(
        &mut self,
        usage: Option<TokenUsage>,
        characters_count: Option<u64>,
    ) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        let total = match self.seg.as_mut() {
            Some(seg) => &mut seg.usage,
            None => &mut self.unattached,
        };
        total.add(usage, characters_count);
        Ok(())
    }

    /// Releases the held-back tail without ending the segment.
    pub fn flush(
        &mut self,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        self.handle_flush(now, eff)
    }

    /// Every later mutator is a silent no-op.
    pub fn end_input(
        &mut self,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) -> Result<(), EmitterError> {
        if !self.live()? {
            return Ok(());
        }
        let r = self.handle_end_segment(now, eff);
        self.closed = true;
        r
    }

    /// Safe on an emitter never initialised.
    pub fn stop(&mut self, eff: &mut Vec<AudioEmitterEffect>) {
        if self.flush_scheduled {
            eff.push(AudioEmitterEffect::CancelScheduledFlush);
            self.flush_scheduled = false;
        }
        self.closed = true;
    }

    /// The scheduled flush deadline (`tts/tts.py:1057`): releases the held-back tail before the
    /// consumer runs dry.
    pub fn on_flush_deadline(
        &mut self,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) -> Result<(), EmitterError> {
        // The timer fired; forget it so `_flush_frame` does not cancel it.
        self.flush_scheduled = false;
        self.flush(now, eff)
    }

    pub fn take_dropped_bytes(&mut self) -> usize {
        std::mem::replace(&mut self.dropped_bytes, 0)
    }

    // ----------------------------------------------------------------- the loop body

    fn handle_flush(
        &mut self,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) -> Result<(), EmitterError> {
        if self.seg.is_none() {
            // Streaming: an empty segment, ignored (`tts/tts.py:1259`).
            return if self.streaming { Ok(()) } else { Err(EmitterError::NoSegment) };
        }
        if self.abs.is_none() {
            // `elif audio_byte_stream:`: a flush before the first byte does nothing.
            return Ok(());
        }
        let abs = self.abs.as_mut().expect("checked");
        let flushed = abs.flush();
        // More bytes can follow this flush: a partial sample stays for them.
        abs.reset_progressive();
        for f in flushed {
            self.emit_frame(Some(f), false, now, eff);
        }
        self.flush_frame(now, eff);
        Ok(())
    }

    fn handle_end_segment(
        &mut self,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) -> Result<(), EmitterError> {
        if self.seg.is_none() {
            return if self.streaming { Ok(()) } else { Err(EmitterError::NoSegment) };
        }
        if self.abs.is_none() {
            // Preserved reference quirk: a segment that never received a byte is never closed, so
            // the next `start_segment()` raises. Pinned by a test.
            return Ok(());
        }
        let abs = self.abs.as_mut().expect("checked");
        let flushed = abs.flush();
        self.dropped_bytes += abs.buffered_bytes();
        for f in flushed {
            self.emit_frame(Some(f), false, now, eff);
        }
        self.emit_frame(None, true, now, eff);
        self.seg = None;
        self.abs = None;
        self.last_frame = None;
        Ok(())
    }

    // ----------------------------------------------------------------- framing

    fn emit_frame(
        &mut self,
        frame: Option<Frame>,
        is_final: bool,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) {
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

    fn do_send(
        &mut self,
        frame: Frame,
        is_final: bool,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) {
        let dur = self.frame_duration(&frame);
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

    fn flush_frame(&mut self, now: Ts, eff: &mut Vec<AudioEmitterEffect>) {
        let Some(tail) = self.last_frame.take() else {
            return;
        };
        let dur = self.frame_duration(&tail);
        let ev = self.build_event(tail, false);
        self.send_audio(ev, false, dur, now, eff);
        self.account(dur);

        // Restart the drift measurement, or the next frame would flush immediately, forever.
        self.sent_start = None;
        self.sent_duration = 0.0;
        if self.flush_scheduled {
            eff.push(AudioEmitterEffect::CancelScheduledFlush);
            self.flush_scheduled = false;
        }
    }

    fn send_audio(
        &mut self,
        ev: SynthesizedAudio,
        flush_if_delayed: bool,
        dur: f64,
        now: Ts,
        eff: &mut Vec<AudioEmitterEffect>,
    ) {
        eff.push(AudioEmitterEffect::EmitAudio(Box::new(ev)));

        if self.sent_start.is_none() {
            self.sent_start = Some(now);
        }
        self.sent_duration += dur;

        if self.flush_scheduled {
            eff.push(AudioEmitterEffect::CancelScheduledFlush);
            self.flush_scheduled = false;
        }

        if flush_if_delayed && self.sent_duration > MIN_DRIFT_AUDIO {
            let start = self.sent_start.unwrap_or(now);
            let elapsed = now.saturating_duration_since(start).as_secs_f64();
            let delay = self.sent_duration - elapsed - DRIFT_MARGIN;
            // `delay == 0.0` is unreachable in doubles here; `>` kept as the reference writes it.
            if delay > 0.0 {
                // Relative, so the host's clock need not agree with `now`.
                eff.push(AudioEmitterEffect::ScheduleFlushAfter(Duration::from_secs_f64(delay)));
                self.flush_scheduled = true;
            }
            // `delay <= 0`: the provider is already behind; the next frame re-evaluates.
        }
    }

    fn build_event(&mut self, frame: Frame, is_final: bool) -> SynthesizedAudio {
        let Usage { tokens: usage, characters_count } = match self.seg.as_mut().filter(|_| is_final)
        {
            Some(seg) => std::mem::take(&mut seg.usage),
            None => Usage::default(),
        };
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
            usage: usage.map(Box::new),
            characters_count,
            provider_request_ids: std::mem::take(&mut self.pending_ids),
            // Stamped by `ChunkedStream`/`SynthesizeStream` on the way out (`tts.py:502`).
            started_time: None,
            // Consumed, not copied, or the synchroniser speaks the words twice.
            timed_transcript: std::mem::take(&mut self.timed_transcripts),
            encoding: AudioEncoding::Pcm,
            connection: None,
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
}

fn combine(mut a: Frame, b: Frame) -> Frame {
    a.data.extend_from_slice(&b.data);
    a.samples_per_channel += b.samples_per_channel;
    a
}

#[cfg(test)]
#[path = "../emitter_tests.rs"]
mod tests;
