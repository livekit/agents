//! `tts.AudioEmitter`: the native framing machine with its decoding, drift timer and segment
//! finalization. Effects reach the host through [`EmitterOutput`], whose calls queue in order on
//! the host loop: no host code runs inside an emitter call, so a consumer that calls back into
//! the emitter never nests in it.
//!
//! PCM and WAV are framed inside the call that pushed them. A compressed container decodes on an
//! admitted native worker, one decoder per segment as in the reference: its calls queue for the
//! emitter's task, which ends a segment only after the segment's audio is decoded and framed.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use lk_agents_abi::media::emitter::EmitterOutput;
use lk_agents_abi::tts::emitter::{AudioEmitter as Machine, Effects, EmitterError, HostEffect};
use lk_agents_abi::{AbortHandle, AudioFrame, Clock, Frame, Spawner, SpawnerExt, TimedString, Ts};
use lk_agents_media::workers::MediaWorkers;
use lk_agents_media::{DecodePlan, Decoder};
use tokio::sync::{Notify, watch};

use crate::codecs::{self, DecoderInput, decoder_limits, whole_pushes};

/// What the emitter runs on, lent by the process (decision 88).
pub struct EmitterEnv {
    pub spawner: Arc<dyn Spawner>,
    pub clock: Arc<dyn Clock>,
    pub workers: MediaWorkers,
}

pub struct AudioEmitter(Arc<Inner>);

struct Inner {
    label: String,
    output: Arc<dyn EmitterOutput>,
    env: EmitterEnv,
    machine: Mutex<Machine>,
    outbox: Mutex<Outbox>,
    decode: OnceLock<Decode>,
    timer: Mutex<Option<AbortHandle>>,
    /// The first failure off the calling thread, reported by the calls after it and by `join`.
    failure: Mutex<Option<EmitterError>>,
    ended: AtomicBool,
    closed: AtomicBool,
}

/// Effects in machine order, delivered by one caller at a time with no lock held.
#[derive(Default)]
struct Outbox {
    effects: VecDeque<HostEffect>,
    delivering: bool,
}

enum Decode {
    /// `audio/pcm` and `audio/raw`.
    Pcm,
    /// Decoded inside `push`; a segment's resampler tail is framed when it ends.
    Wav(Mutex<Option<Decoder>>),
    /// Calls queued for the emitter's task in arrival order.
    Container(Queue),
}

struct Queue {
    mime: String,
    ops: Mutex<VecDeque<Op>>,
    ready: Notify,
    /// The task has framed everything up to the end of input, or stopped.
    drained: watch::Sender<bool>,
    /// The open segment's decoder input, reachable to cancel it.
    segment: Mutex<Option<Arc<DecoderInput>>>,
}

enum Op {
    StartSegment(String),
    EndSegment,
    Push(Vec<u8>),
    Frame(Frame),
    Words(Vec<TimedString>),
    Flush,
    EndInput,
}

/// A segment's container decoder.
struct Segment {
    input: Arc<DecoderInput>,
    output: codecs::AudioDecoder,
    /// Until its output ends; a container may end before its segment does.
    decoding: bool,
}

/// `LK_DUMP_TTS`, read once; a non-integer value disables dumping.
fn dump_default() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LK_DUMP_TTS")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .is_some_and(|v| v != 0)
    })
}

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl AudioEmitter {
    pub fn with_env(label: String, output: Arc<dyn EmitterOutput>, env: EmitterEnv) -> Self {
        let mut machine = Machine::new(&label);
        machine.set_dump(dump_default());
        Self(Arc::new(Inner {
            label,
            output,
            env,
            machine: Mutex::new(machine),
            outbox: Mutex::default(),
            decode: OnceLock::new(),
            timer: Mutex::new(None),
            failure: Mutex::new(None),
            ended: AtomicBool::new(false),
            closed: AtomicBool::new(false),
        }))
    }
}

impl Inner {
    fn failed(&self) -> Result<(), EmitterError> {
        locked(&self.failure).clone().map_or(Ok(()), Err)
    }

    fn fail(&self, error: EmitterError) {
        locked(&self.failure).get_or_insert(error);
    }

    /// Runs `op` on the machine, then delivers what it produced, in order.
    fn run(
        self: &Arc<Self>,
        op: impl FnOnce(&mut Machine, Ts, &mut Effects) -> Result<(), EmitterError>,
    ) -> Result<(), EmitterError> {
        let now = self.env.clock.now();
        let result = {
            let mut machine = locked(&self.machine);
            let mut effects = Effects::new();
            let result = op(&mut machine, now, &mut effects);
            if let Some(ids) = machine.take_provider_ids_update() {
                effects.push(HostEffect::ProviderIds(ids));
            }
            effects.extend(machine.take_dumps().into_iter().map(|d| HostEffect::Dump(Box::new(d))));
            let dropped = machine.take_dropped_bytes();
            if dropped > 0 {
                effects.push(HostEffect::Dropped(dropped));
            }
            locked(&self.outbox).effects.extend(effects);
            result
        };
        self.deliver();
        result
    }

    fn deliver(self: &Arc<Self>) {
        if std::mem::replace(&mut locked(&self.outbox).delivering, true) {
            return;
        }
        loop {
            let effects = {
                let mut outbox = locked(&self.outbox);
                if outbox.effects.is_empty() {
                    outbox.delivering = false;
                    return;
                }
                std::mem::take(&mut outbox.effects)
            };
            effects.into_iter().for_each(|effect| self.effect(effect));
        }
    }

    fn effect(self: &Arc<Self>, effect: HostEffect) {
        match effect {
            HostEffect::Send(audio) => {
                let audio = *audio;
                self.output.send(
                    audio.frame,
                    audio.request_id,
                    audio.segment_id,
                    audio.is_final,
                    audio.timed_transcript,
                );
            }
            HostEffect::Arm(delay) => self.arm(delay),
            HostEffect::Cancel => self.disarm(),
            HostEffect::Dump(dump) => self.output.dump(dump.name(), dump.wav_bytes()),
            HostEffect::Dropped(bytes) => log::warn!(
                target: "livekit.agents",
                "incomplete PCM sample at end of segment, discarding {bytes} trailing bytes"
            ),
            HostEffect::ProviderIds(ids) => self.output.span_attribute(
                lk_agents_abi::telemetry::attrs::PROVIDER_REQUEST_IDS.to_owned(),
                ids,
            ),
        }
    }

    /// The drift flush, on the host's clock.
    fn arm(self: &Arc<Self>, delay: Duration) {
        self.disarm();
        let at = self.env.clock.now().saturating_add(delay);
        let clock = self.env.clock.clone();
        let emitter = Arc::downgrade(self);
        let timer = self.env.spawner.spawn_unboxed(async move {
            clock.wait_until(at).await;
            let Some(emitter) = emitter.upgrade() else { return };
            log::debug!(target: "livekit.agents", "flush audio emitter due to slow audio generation");
            if let Err(error) = emitter.run(|m, now, eff| m.on_drift(now, eff)) {
                emitter.fail(error);
            }
        });
        *locked(&self.timer) = Some(timer);
    }

    fn disarm(&self) {
        if let Some(timer) = locked(&self.timer).take() {
            timer.abort();
        }
    }

    /// PCM to frame for `data`, from the WAV decoder when there is one.
    fn decoded(&self, wav: &Mutex<Option<Decoder>>, data: &[u8]) -> Result<Vec<u8>, EmitterError> {
        let mut decoder = locked(wav);
        let decoder = match decoder.as_mut() {
            Some(decoder) => decoder,
            None => {
                let (rate, channels) = self.format();
                let limits = whole_pushes(decoder_limits(rate, channels).format);
                decoder.insert(Decoder::wav(Some(rate), limits).map_err(decode_error)?)
            }
        };
        decoder.push(data).map(|pcm| lk_agents_media::pcm_bytes(&pcm)).map_err(decode_error)
    }

    /// The open WAV segment's resampler tail, framed before the segment ends.
    fn wav_tail(self: &Arc<Self>, wav: &Mutex<Option<Decoder>>) -> Result<(), EmitterError> {
        let tail = match locked(wav).take() {
            Some(mut decoder) => decoder.flush().map_err(decode_error)?,
            None => return Ok(()),
        };
        if tail.is_empty() {
            return Ok(());
        }
        self.run(|m, now, eff| m.push(now, &lk_agents_media::pcm_bytes(&tail), eff))
    }

    /// The PCM format the machine frames, set by `initialize`.
    fn format(&self) -> (u32, u16) {
        let machine = locked(&self.machine);
        (machine.sample_rate(), machine.num_channels())
    }

    /// Applies one call in arrival order; for a container, the emitter's task applies it.
    fn apply(self: &Arc<Self>, op: Op) -> Result<(), EmitterError> {
        self.failed()?;
        match self.decode.get() {
            Some(Decode::Container(queue)) => {
                locked(&queue.ops).push_back(op);
                queue.ready.notify_one();
                Ok(())
            }
            Some(Decode::Wav(wav)) => match op {
                Op::Push(data) => {
                    let pcm = self.decoded(wav, &data)?;
                    self.run(|m, now, eff| m.push(now, &pcm, eff))
                }
                Op::EndSegment | Op::EndInput => {
                    self.wav_tail(wav)?;
                    self.apply_now(op)
                }
                op => self.apply_now(op),
            },
            Some(Decode::Pcm) | None => self.apply_now(op),
        }
    }

    fn apply_now(self: &Arc<Self>, op: Op) -> Result<(), EmitterError> {
        match op {
            Op::StartSegment(id) => self.run(|m, _, _| m.start_segment(&id).map(drop)),
            Op::EndSegment => self.run(|m, now, eff| m.end_segment(now, eff)),
            Op::Push(data) => self.run(|m, now, eff| m.push(now, &data, eff)),
            Op::Frame(frame) => self.run(|m, now, eff| m.push_frame(now, frame, eff)),
            Op::Words(words) => self.run(|m, _, _| m.push_timed_transcript(words)),
            Op::Flush => self.run(|m, now, eff| m.flush(now, eff)),
            Op::EndInput => self.run(|m, now, eff| m.end_input(now, eff)),
        }
    }

    /// The emitter's task for a container: applies queued calls in order, decoding each
    /// segment's bytes on its own admitted worker and ending the segment once they are framed.
    async fn serve(self: Arc<Self>, queue: &Queue) {
        let mut segment: Option<Segment> = None;
        let result = self.serve_ops(queue, &mut segment).await;
        if let Some(mut segment) = segment {
            let _ = segment.output.close().await;
        }
        *locked(&queue.segment) = None;
        if let Err(error) = result
            && !self.closed.load(Ordering::Acquire)
        {
            self.fail(error);
        }
        queue.drained.send_replace(true);
    }

    async fn serve_ops(
        self: &Arc<Self>,
        queue: &Queue,
        segment: &mut Option<Segment>,
    ) -> Result<(), EmitterError> {
        loop {
            let op = match segment.as_mut() {
                Some(open) => tokio::select! {
                    biased;
                    op = next_op(queue, &self.closed) => op,
                    frame = open.output.next(), if open.decoding => {
                        open.decoding = self.frame(frame)?;
                        continue;
                    }
                },
                None => next_op(queue, &self.closed).await,
            };
            let Some(op) = op else { return Ok(()) };
            match op {
                Op::Push(data) => {
                    let open = match segment {
                        Some(open) => open,
                        None => segment.insert(self.open_segment(queue)?),
                    };
                    open.input.push(data).map_err(decode_error)?;
                }
                op @ (Op::EndSegment | Op::EndInput) => {
                    if let Some(mut open) = segment.take() {
                        open.input.end_input();
                        while open.decoding {
                            open.decoding = self.frame(open.output.next().await)?;
                        }
                        *locked(&queue.segment) = None;
                        open.output.close().await.map_err(decode_error)?;
                    }
                    let end = matches!(op, Op::EndInput);
                    self.apply_now(op)?;
                    if end {
                        return Ok(());
                    }
                }
                op => self.apply_now(op)?,
            }
        }
    }

    fn open_segment(&self, queue: &Queue) -> Result<Segment, EmitterError> {
        let (rate, channels) = self.format();
        let (output, input) = codecs::AudioDecoder::open_stream(
            &self.env.workers,
            &queue.mime,
            Some(rate),
            channels,
            decoder_limits(rate, channels),
        )
        .map_err(decode_error)?;
        let input = Arc::new(input);
        *locked(&queue.segment) = Some(input.clone());
        Ok(Segment { input, output, decoding: true })
    }

    /// Frames one decoded frame; `false` at the end of the segment's audio.
    fn frame(
        self: &Arc<Self>,
        frame: Result<Option<AudioFrame>, codecs::DecoderError>,
    ) -> Result<bool, EmitterError> {
        match frame.map_err(decode_error)? {
            Some(frame) => self.run(|m, now, eff| m.push(now, &frame.data, eff)).map(|()| true),
            None => Ok(false),
        }
    }
}

/// The next queued call; `None` once the emitter is closed.
async fn next_op(queue: &Queue, closed: &AtomicBool) -> Option<Op> {
    loop {
        let ready = queue.ready.notified();
        if closed.load(Ordering::Acquire) {
            return None;
        }
        if let Some(op) = locked(&queue.ops).pop_front() {
            return Some(op);
        }
        ready.await;
    }
}

fn decode_error(error: impl std::fmt::Display) -> EmitterError {
    EmitterError::Decode(error.to_string())
}

impl lk_agents_abi::media::emitter::AudioEmitter for AudioEmitter {
    fn new(label: String, dst_ch: Arc<dyn EmitterOutput>) -> Result<Self, EmitterError> {
        use lk_agents_net::env::Env as _;
        let unavailable = |error: String| EmitterError::Unavailable(error);
        let env = EmitterEnv {
            spawner: crate::process::Process::spawner().map_err(unavailable)?,
            clock: crate::process::Process::clock(),
            workers: crate::process::workers().map_err(|e| unavailable(e.to_string()))?,
        };
        Ok(Self::with_env(label, dst_ch, env))
    }

    fn initialize(
        &self,
        request_id: String,
        sample_rate: u32,
        num_channels: u16,
        mime_type: String,
        frame_size_ms: u32,
        stream: bool,
    ) -> Result<(), EmitterError> {
        let me = &self.0;
        let decode = match lk_agents_media::decode_plan(&mime_type)
            .map_err(|e| EmitterError::UnsupportedMime(e.to_string()))?
        {
            DecodePlan::Passthrough => Decode::Pcm,
            DecodePlan::Decode(mime)
                if lk_agents_media::mime_to_av_format(Some(&mime)) == Some("wav") =>
            {
                Decode::Wav(Mutex::new(None))
            }
            DecodePlan::Decode(mime) => {
                if !cfg!(feature = "media-ffmpeg") {
                    return Err(EmitterError::UnsupportedMime(format!(
                        "{mime:?} needs the runtime media-ffmpeg feature to decode"
                    )));
                }
                Decode::Container(Queue {
                    mime,
                    ops: Mutex::default(),
                    ready: Notify::new(),
                    drained: watch::Sender::new(false),
                    segment: Mutex::new(None),
                })
            }
        };
        me.run(|m, _, _| {
            m.initialize(&request_id, sample_rate, num_channels, frame_size_ms, stream)
        })?;
        if request_id.is_empty() {
            log::warn!(target: "livekit.agents", "no request_id provided for TTS {}", me.label);
        }
        if me.decode.set(decode).is_err() {
            return Err(EmitterError::AlreadyStarted);
        }
        if let Some(Decode::Container(_)) = me.decode.get() {
            let emitter = me.clone();
            me.env.spawner.spawn_unboxed(async move {
                let Some(Decode::Container(queue)) = emitter.decode.get() else { return };
                emitter.clone().serve(queue).await;
            });
        }
        Ok(())
    }

    fn start_segment(&self, segment_id: String) -> Result<(), EmitterError> {
        self.0.apply(Op::StartSegment(segment_id))
    }

    fn end_segment(&self) -> Result<(), EmitterError> {
        self.0.apply(Op::EndSegment)
    }

    fn push(&self, data: &[u8]) -> Result<(), EmitterError> {
        self.0.apply(Op::Push(data.to_vec()))
    }

    fn push_frame(&self, frame: AudioFrame) -> Result<(), EmitterError> {
        let samples_per_channel = frame.samples_per_channel;
        self.0.apply(Op::Frame(Frame { data: frame.data.to_vec(), samples_per_channel }))
    }

    fn push_timed_transcript(&self, delta_text: Vec<TimedString>) -> Result<(), EmitterError> {
        if delta_text.is_empty() {
            return Ok(());
        }
        self.0.apply(Op::Words(delta_text))
    }

    fn flush(&self) -> Result<(), EmitterError> {
        self.0.apply(Op::Flush)
    }

    fn end_input(&self) -> Result<(), EmitterError> {
        self.0.ended.store(true, Ordering::Release);
        self.0.apply(Op::EndInput)
    }

    async fn join(&self) -> Result<(), EmitterError> {
        if !locked(&self.0.machine).started() {
            return Err(EmitterError::NotStarted);
        }
        if let Some(Decode::Container(queue)) = self.0.decode.get()
            && self.0.ended.load(Ordering::Acquire)
        {
            let _ = queue.drained.subscribe().wait_for(|drained| *drained).await;
        }
        self.0.failed()
    }

    async fn aclose(&self) -> Result<(), EmitterError> {
        let me = &self.0;
        me.closed.store(true, Ordering::Release);
        let _ = me.run(|m, _, eff| {
            m.stop(eff);
            Ok(())
        });
        me.disarm();
        if let Some(Decode::Container(queue)) = me.decode.get() {
            locked(&queue.ops).clear();
            if let Some(segment) = locked(&queue.segment).take() {
                segment.cancel();
            }
            queue.ready.notify_one();
            // Closing never reports a synthesis failure, but owns native decoding to settlement.
            let _ = queue.drained.subscribe().wait_for(|drained| *drained).await;
        }
        Ok(())
    }

    fn pushed_duration(&self, idx: i64) -> f64 {
        locked(&self.0.machine).pushed_duration(idx)
    }

    fn num_segments(&self) -> usize {
        locked(&self.0.machine).num_segments()
    }

    fn _note_provider_request_id(&self, context_id: String) -> Result<(), EmitterError> {
        self.0.run(|m, _, _| {
            m.note_provider_request_id(&context_id);
            Ok(())
        })
    }

    fn output_closed(&self) -> Result<bool, EmitterError> {
        self.0.output.closed().map_err(|error| EmitterError::Unavailable(error.to_string()))
    }
}

#[cfg(test)]
mod tests;
