use crate::agent_activity::GateSnapshot;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::Out;
use crate::agent_session::{DriverError, Timebase, Tx};
use crate::trace::{ChannelEndCause, InputStep};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::voice::io::AudioInput;
use lk_agents_abi::{
    AudioByteStream, AudioFrame, Closed, Spawner, SpawnerExt, StreamSink, StreamSource,
};
use lk_agents_media::{FrameResampler, OnChange, Quality};
use lk_agents_rtc::silence_frame;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// 20 ms. What every STT and VAD in the tree is written against.
pub const DEFAULT_FRAME_MS: u32 = 20;

/// `first_input_at` before any frame: a session instant can be zero, so "none" is not zero.
pub(super) const NO_INPUT_YET: u64 = u64::MAX;

pub(super) const AUDIO_IN_LEG: &str = "audio_in";

pub(super) const VIDEO_IN_LEG: &str = "video_in";

/// The reads [`audio_gate`] makes. Every one is driver-owned state; none is a host call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateInputs {
    /// `self._session.agent_state`.
    pub agent_state: AgentState,
    /// `self._session._aec_warmup_remaining`.
    pub aec_warmup_remaining: Duration,
    /// `self._session._aec_warmup_timer is not None`.
    pub aec_warmup_armed: bool,
    /// `self._current_speech is not None and not done and not interrupted and not allow_interruptions`.
    pub uninterruptible_speech_active: bool,
    /// `session.options.interruption["discard_audio_if_uninterruptible"]`.
    pub discard_audio_if_uninterruptible: bool,
}

/// `AgentActivity.push_audio`'s decision half, pure: whether the substituted legs (see
/// [`AudioChannel::substituted`]) get silence for this frame.
pub fn audio_gate(i: &GateInputs) -> bool {
    // Warmup only contaminates while the agent is actually speaking.
    let aec_warmup_active = i.agent_state == AgentState::Speaking
        && !i.aec_warmup_remaining.is_zero()
        && i.aec_warmup_armed;

    let uninterruptible = i.uninterruptible_speech_active && i.discard_audio_if_uninterruptible;

    aec_warmup_active || uninterruptible
}

/// `soxr` is stateful, so one instance per stream (alternating resamplers click at every boundary).
/// Its output is bursty (48 -> 16 kHz: `0, 526, 0, 525, ...` samples), so it goes through
/// [`AudioByteStream`] and every consumer sees steady 20 ms frames.
pub struct AudioFanOut {
    target_rate: u32,
    frame_samples: u32,
    /// Built on the first off-rate frame (the rate is the track's) and rebuilt, unflushed, if the
    /// rate changes; a frame already at the target rate leaves it be. MEDIUM is
    /// `rtc.AudioResampler`'s default; see `lk-agents-media` for the ordinal quirk.
    resampler: FrameResampler,
    chunker: AudioByteStream,
}

impl AudioFanOut {
    /// `target_rate` is what the STT and VAD legs want -- 16 kHz for every model in the tree.
    pub fn new(target_rate: u32) -> Self {
        let frame_samples = target_rate / 1000 * DEFAULT_FRAME_MS;
        AudioFanOut {
            target_rate,
            frame_samples,
            resampler: FrameResampler::new(target_rate, Quality::Medium, OnChange::Discard),
            chunker: AudioByteStream::new(target_rate, 1, Some(frame_samples), false),
        }
    }

    pub fn frame_shape(&self) -> FrameShape {
        FrameShape {
            sample_rate: self.target_rate,
            num_channels: 1,
            samples_per_channel: self.frame_samples,
        }
    }

    /// Convert one room frame and return the whole target-rate frames now ready (zero, one or two).
    pub fn convert(&mut self, frame: &AudioFrame) -> Result<Vec<AudioFrame>, DriverError> {
        // Explicit fast path: 1:1 soxr isn't the identity (filter delay would skew STT vs VAD).
        if frame.num_channels == 1
            && frame.sample_rate == self.target_rate
            && frame.samples_per_channel == self.frame_samples
        {
            return Ok(vec![frame.clone()]);
        }
        let ready =
            self.resampler.push(&mono(frame)).map_err(|e| DriverError::Resample(e.to_string()))?;
        Ok(match ready.first() {
            Some(converted) => self.rechunk(&converted.data),
            None => Vec::new(),
        })
    }

    fn rechunk(&mut self, bytes: &[u8]) -> Vec<AudioFrame> {
        let rate = self.target_rate;
        self.chunker
            .push(bytes)
            .into_iter()
            .map(|f| AudioFrame::new(f.data, rate, 1, f.samples_per_channel))
            .collect()
    }
}

/// A silence frame shaped like `frame`, so substituted legs keep that leg's rate.
fn silence_like(frame: &AudioFrame) -> AudioFrame {
    silence_frame(frame.samples_per_channel, frame.sample_rate, frame.num_channels)
}

/// Interleaved `i16` -> the same frame averaged to mono.
fn mono(frame: &AudioFrame) -> AudioFrame {
    let ch = frame.num_channels.max(1) as usize;
    let mut bytes = Vec::with_capacity(frame.data.len() / ch);
    // Sum in i32: a wrapping i16 add turns a loud stereo frame into white noise.
    for group in frame.data.chunks_exact(2 * ch) {
        let sum: i32 =
            group.chunks_exact(2).map(|b| i32::from(i16::from_le_bytes([b[0], b[1]]))).sum();
        bytes.extend_from_slice(&((sum / ch as i32) as i16).to_le_bytes());
    }
    let spc = (bytes.len() / 2) as u32;
    AudioFrame::new(bytes, frame.sample_rate, 1, spc)
}

/// [`substituted`](Self::substituted) is the only place the substitution side is decided.
///
/// | leg            | feed        | gate side     | reference                             |
/// |----------------|-------------|---------------|---------------------------------------|
/// | `Stt`          | `Raw`       | `Substituted` | `self._stt_pipeline.audio_ch`         |
/// | `Vad`          | `Raw`       | `Real`        | `self._vad_ch`                        |
/// | `Realtime`     | `Raw`       | `Substituted` | `self._rt_session.push_audio`         |
/// | `Eot`          | `Converted` | `Substituted` | `self._turn_detector_stream.push_audio` |
/// | `Interruption` | `Converted` | `Real`        | `self._interruption_ch` (hears the user while the agent speaks) |
/// | `Amd`          | `Raw`       | `Real`        | `self._session.amd.push_audio`        |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, lk_agents_abi::Vocabulary)]
pub enum AudioChannel {
    Stt,
    Vad,
    Realtime,
    Eot,
    Interruption,
    Amd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feed {
    /// The source's own frame, unconverted, as the reference forwards it; each plugin resamples.
    Raw,
    /// The one 16 kHz mono conversion, for the two Rust-native detectors written against it.
    Converted,
}

impl AudioChannel {
    pub const ALL: [AudioChannel; 6] = [
        AudioChannel::Stt,
        AudioChannel::Vad,
        AudioChannel::Realtime,
        AudioChannel::Eot,
        AudioChannel::Interruption,
        AudioChannel::Amd,
    ];

    pub const fn feed(self) -> Feed {
        match self {
            AudioChannel::Stt | AudioChannel::Vad | AudioChannel::Realtime | AudioChannel::Amd => {
                Feed::Raw
            }
            AudioChannel::Eot | AudioChannel::Interruption => Feed::Converted,
        }
    }

    /// Which side of the gate this leg is on: a substituted leg gets silence while the gate
    /// discards, or it would answer the agent's own voice; the others always hear the user.
    /// **The only place the substitution is decided.**
    pub const fn substituted(self) -> bool {
        matches!(self, AudioChannel::Stt | AudioChannel::Realtime | AudioChannel::Eot)
    }

    pub const fn label(self) -> &'static str {
        match self {
            AudioChannel::Stt => "stt",
            AudioChannel::Vad => "vad",
            AudioChannel::Realtime => "realtime",
            AudioChannel::Eot => "eot",
            AudioChannel::Interruption => "interruption",
            AudioChannel::Amd => "amd",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameShape {
    pub sample_rate: u32,
    pub num_channels: u16,
    pub samples_per_channel: u32,
}

impl FrameShape {
    pub fn of(frame: &AudioFrame) -> FrameShape {
        FrameShape {
            sample_rate: frame.sample_rate,
            num_channels: frame.num_channels,
            samples_per_channel: frame.samples_per_channel,
        }
    }
}

#[derive(Default)]
pub struct LegSlot {
    sink: Option<Arc<dyn StreamSink<AudioFrame>>>,
    /// The shape of the last frame pushed here; journalled at `Closing`.
    shape: Option<FrameShape>,
    /// Sinks installed here. Each install's drain gets it as a [`Ticket`]; [`AudioChannels::ended`] clears
    /// the row only if that install is still in place.
    opens: u32,
    /// The open whose items the loop still consumes; a drain's own end leaves it.
    live: Option<u32>,
    frames: u64,
    silent: u64,
    /// `silent` as of the last [`AudioChannels::take_silenced`]: the journal reports the delta.
    silent_reported: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    pub(super) leg: AudioChannel,
    opens: u32,
}

impl Ticket {
    /// Which install of the row this was (1-based): the stream's identity for what it delivers.
    pub(super) fn open(&self) -> u32 {
        self.opens
    }
}

/// One item a drain delivered, tagged with its stream's open so the loop drops what a retired
/// stream still sends ([`AudioChannels::installed_open`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Tagged<T> {
    pub open: u32,
    pub item: T,
}

struct TaggedSource<T> {
    open: u32,
    inner: Box<dyn StreamSource<T>>,
}

#[async_trait::async_trait]
impl<T: Send + 'static> StreamSource<Tagged<T>> for TaggedSource<T> {
    async fn next(&mut self) -> Result<Option<Tagged<T>>, lk_agents_abi::errors::ProviderError> {
        Ok(self.inner.next().await?.map(|item| Tagged { open: self.open, item }))
    }
    fn cancel_safe(&self) -> bool {
        self.inner.cancel_safe()
    }
    async fn close(self: Box<Self>, deadline: lk_agents_abi::Ts) -> Closed {
        self.inner.close(deadline).await
    }
}

pub(super) fn tagged<T: Send + 'static>(
    ticket: Ticket,
    source: Box<dyn StreamSource<T>>,
) -> Box<dyn StreamSource<Tagged<T>>> {
    Box::new(TaggedSource { open: ticket.open(), inner: source })
}

/// The one table of legs, held by the pump; the loop mutates it through [`ChannelOp`].
pub struct AudioChannels {
    slots: Mutex<[LegSlot; 6]>,
    input: Mutex<Option<(FrameShape, FrameShape)>>,
}

impl Default for AudioChannels {
    fn default() -> Self {
        AudioChannels {
            slots: Mutex::new(std::array::from_fn(|_| LegSlot::default())),
            input: Mutex::new(None),
        }
    }
}

impl AudioChannels {
    pub fn new() -> Arc<AudioChannels> {
        Arc::new(AudioChannels::default())
    }

    /// Put `sink` in `leg`, then end the one it replaced. Installing first means the pump never
    /// sees an empty row; ending (not dropping) because dropping a host stream doesn't close it.
    pub fn install(&self, leg: AudioChannel, sink: Arc<dyn StreamSink<AudioFrame>>) -> Ticket {
        let (previous, opens) = {
            let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
            let slot = &mut slots[leg.index()];
            slot.opens += 1;
            slot.live = Some(slot.opens);
            (std::mem::replace(&mut slot.sink, Some(sink)), slot.opens)
        };
        if let Some(previous) = previous {
            previous.end_input();
        }
        Ticket { leg, opens }
    }

    /// Empty `leg` and end what it held. `Agent(stt=None)`: this agent does not listen.
    pub fn clear(&self, leg: AudioChannel) {
        let previous = {
            let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
            let slot = &mut slots[leg.index()];
            slot.live = None;
            slot.sink.take()
        };
        if let Some(previous) = previous {
            previous.end_input();
        }
    }

    /// The open whose items the loop still consumes on `leg`; `None` once cleared. A drain's own
    /// end does not retire it.
    pub fn installed_open(&self, leg: AudioChannel) -> Option<u32> {
        self.slots.lock().unwrap_or_else(|p| p.into_inner())[leg.index()].live
    }

    /// The drain under `ticket` ended: forget its sink only if it is still installed, or the live
    /// successor would be removed. Returns whether the row was cleared.
    pub fn ended(&self, ticket: Ticket) -> bool {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        let slot = &mut slots[ticket.leg.index()];
        if slot.opens == ticket.opens && slot.sink.is_some() {
            slot.sink = None;
            true
        } else {
            false
        }
    }

    /// The sink `leg` currently holds. Read rather than borrowed: a handoff can replace it.
    pub fn get(&self, leg: AudioChannel) -> Option<Arc<dyn StreamSink<AudioFrame>>> {
        self.slots.lock().unwrap_or_else(|p| p.into_inner())[leg.index()].sink.clone()
    }

    pub fn present(&self, leg: AudioChannel) -> bool {
        self.slots.lock().unwrap_or_else(|p| p.into_inner())[leg.index()].sink.is_some()
    }

    pub fn take_silenced(&self) -> [(AudioChannel, u64); 6] {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        AudioChannel::ALL.map(|id| {
            let slot = &mut slots[id.index()];
            let delta = slot.silent - slot.silent_reported;
            slot.silent_reported = slot.silent;
            (id, delta)
        })
    }

    pub fn shape(&self, leg: AudioChannel) -> Option<FrameShape> {
        self.slots.lock().unwrap_or_else(|p| p.into_inner())[leg.index()].shape
    }

    pub fn note_input(&self, input: FrameShape, converted: FrameShape) {
        *self.input.lock().unwrap_or_else(|p| p.into_inner()) = Some((input, converted));
    }

    pub fn input_shape(&self) -> Option<(FrameShape, FrameShape)> {
        *self.input.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn counters(&self, leg: AudioChannel) -> (u64, u64) {
        let slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        (slots[leg.index()].frames, slots[leg.index()].silent)
    }

    /// One frame to every installed leg on the `feed` side: `silence` (given only while the gate
    /// discards) to the substituted legs, the frame to the rest. Snapshot under the lock, push
    /// outside it.
    pub(super) fn push(&self, feed: Feed, frame: &AudioFrame, silence: Option<&AudioFrame>) {
        let mut targets: [Option<(Arc<dyn StreamSink<AudioFrame>>, &AudioFrame)>; 6] =
            std::array::from_fn(|_| None);
        {
            let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
            for id in AudioChannel::ALL {
                if id.feed() != feed {
                    continue;
                }
                let slot = &mut slots[id.index()];
                let Some(sink) = slot.sink.clone() else { continue };
                let heard = match silence {
                    Some(s) if id.substituted() => {
                        slot.silent += 1;
                        s
                    }
                    _ => frame,
                };
                slot.frames += 1;
                slot.shape = Some(FrameShape::of(heard));
                targets[id.index()] = Some((sink, heard));
            }
        }
        // A dead host stream is the drain's to report, not the pump's per frame.
        for (sink, heard) in targets.into_iter().flatten() {
            let _ = sink.push(heard.clone());
        }
    }
}

/// How the loop changes the table, applied by `Driver::apply_leg_op`.
pub enum ChannelOp {
    Install {
        leg: AudioChannel,
        sink: Arc<dyn StreamSink<AudioFrame>>,
    },
    Clear(AudioChannel),
    /// Point the pump at a new source, or stop it; the legs stay.
    SetSource(Option<Arc<dyn AudioInput>>),
}

impl std::fmt::Debug for ChannelOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelOp::Install { leg, sink: _ } => {
                f.debug_struct("Install").field("leg", leg).finish()
            }
            ChannelOp::Clear(leg) => f.debug_tuple("Clear").field(leg).finish(),
            ChannelOp::SetSource(input) => {
                f.debug_tuple("SetSource").field(&input.is_some()).finish()
            }
        }
    }
}

/// A speech about to push its first frame while the user speaks keeps it until the user stops. No
/// timer. Level, not edge: `publish` runs every iteration; the task reads the level before its
/// first push and parks while it names its speech.
pub struct LaunchGate {
    /// The speeches held at launch. A set: a speech blocked while parked is still held when the
    /// next is authorized behind it.
    held: tokio::sync::watch::Sender<std::collections::BTreeSet<SpeechId>>,
    /// Every speech authorized whose first frame is not out yet: the speeches in their launch.
    launching: std::sync::Mutex<std::collections::BTreeSet<SpeechId>>,
}

impl Default for LaunchGate {
    fn default() -> Self {
        LaunchGate::new()
    }
}

impl LaunchGate {
    pub fn new() -> Self {
        LaunchGate {
            held: tokio::sync::watch::channel(std::collections::BTreeSet::new()).0,
            launching: std::sync::Mutex::new(std::collections::BTreeSet::new()),
        }
    }

    /// The loop, at `Authorized`: this speech is in its launch from now until its first frame.
    pub fn note_authorized(&self, speech: SpeechId) {
        self.launching.lock().unwrap().insert(speech);
    }

    pub fn launching(&self) -> Vec<SpeechId> {
        self.launching.lock().unwrap().iter().copied().collect()
    }

    /// The loop: the level as of the end of this iteration.
    pub fn publish(&self, held: std::collections::BTreeSet<SpeechId>) {
        self.held.send_if_modified(|h| {
            if *h == held {
                false
            } else {
                *h = held;
                true
            }
        });
    }

    pub fn note_started(&self, speech: SpeechId) {
        self.launching.lock().unwrap().remove(&speech);
    }

    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<std::collections::BTreeSet<SpeechId>> {
        self.held.subscribe()
    }
}

#[derive(Clone)]
pub(super) struct AudioForwarderParts {
    /// `(sample_rate, num_channels)` of the last mic frame (`AudioRecognition._sample_rate`), for
    /// the close's silence flush. Zero before any audio.
    pub(super) last_input: Arc<std::sync::atomic::AtomicU64>,
    /// The session instant of the first frame, as nanos ([`NO_INPUT_YET`] until then).
    pub(super) first_input_at: Arc<std::sync::atomic::AtomicU64>,
    /// `input.set_audio_enabled(..)`: while `false` the pump drops every frame at the source; the
    /// track stays subscribed.
    pub(super) input_enabled: Arc<std::sync::atomic::AtomicBool>,
    /// The session recorder's tap: every frame the pump hears is the user's channel.
    pub(super) recorder: Arc<std::sync::RwLock<Option<crate::recorder_io::RecorderTap>>>,
    pub(super) gate: Arc<GateSnapshot>,
    pub(super) legs: Arc<AudioChannels>,
    pub(super) rate: u32,
    pub(super) closing: tokio::sync::watch::Receiver<bool>,
    pub(super) ended: Tx<ChannelEnd>,
    pub(super) time: Timebase,
}

impl AudioForwarderParts {
    pub(super) async fn pump(self, input: Arc<dyn AudioInput>) {
        let mut p = self;
        let mut fan = AudioFanOut::new(p.rate);
        // Exits on track end or close, `biased` so an arrived frame is fed first.
        let cause = loop {
            let frame = tokio::select! {
                biased;
                frame = input.next_frame() => match frame {
                    Some(frame) => frame,
                    None => break ChannelEndCause::SourceEnded,
                },
                _ = p.closing.wait_for(|c| *c) => break ChannelEndCause::Closing,
            };
            // For the close-time STT flush's frame shape.
            p.last_input.store(
                (u64::from(frame.sample_rate) << 32) | u64::from(frame.num_channels),
                std::sync::atomic::Ordering::Relaxed,
            );
            // Muted: dropped at the source, before any leg or counter sees it.
            if !p.input_enabled.load(std::sync::atomic::Ordering::Relaxed) {
                continue;
            }
            // Cloned out from under the lock.
            let tap = p.recorder.read().unwrap_or_else(|p| p.into_inner()).clone();
            if let Some(tap) = tap {
                tap.recorder.on_input_frame(tap.now(), &frame);
            }
            if p.first_input_at.load(std::sync::atomic::Ordering::Relaxed) == NO_INPUT_YET {
                p.first_input_at
                    .store(p.time.now().as_nanos(), std::sync::atomic::Ordering::Relaxed);
            }
            // Read once per frame so every leg is gated by the same verdict.
            let discard = p.gate.discard();
            p.legs.note_input(FrameShape::of(&frame), fan.frame_shape());

            // `Raw` rows: the source frame; the substitute has the room frame's shape.
            let raw_silence = discard.then(|| silence_like(&frame));
            p.legs.push(Feed::Raw, &frame, raw_silence.as_ref());

            // `Converted` rows: the Rust detectors expect 16 kHz mono.
            let Ok(ready) = fan.convert(&frame) else { continue };
            for converted in ready {
                let substitute = discard.then(|| silence_like(&converted));
                p.legs.push(Feed::Converted, &converted, substitute.as_ref());
            }
        };
        // The input is the host's, not a `HostStream`: nothing to close here, so no `Closed`.
        let _ = p.ended.send(ChannelEnd { leg: AUDIO_IN_LEG, cause, closed: None, error: None });
    }
}

#[derive(Clone)]
pub(super) struct VideoForwarderParts {
    pub(super) closing: tokio::sync::watch::Receiver<bool>,
    pub(super) ended: Tx<ChannelEnd>,
    /// The loop's queue; the sampler runs in the loop, on the session clock.
    pub(super) frames: Tx<lk_agents_rtc::VideoFrameRef>,
}

impl VideoForwarderParts {
    pub(super) async fn pump(self, input: Arc<dyn lk_agents_abi::voice::io::VideoInput>) {
        let mut p = self;
        let cause = loop {
            let frame = tokio::select! {
                biased;
                frame = input.next_frame() => match frame {
                    Some(frame) => frame,
                    None => break ChannelEndCause::SourceEnded,
                },
                _ = p.closing.wait_for(|c| *c) => break ChannelEndCause::Closing,
            };
            // A queue that is gone is a loop that has returned: nothing to report to.
            if p.frames.send(frame).is_err() {
                return;
            }
        };
        let _ = p.ended.send(ChannelEnd { leg: VIDEO_IN_LEG, cause, closed: None, error: None });
    }
}

/// A drain or the pump exited, sent to the loop so it is journalled there.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelEnd {
    pub(super) leg: &'static str,
    pub(super) cause: ChannelEndCause,
    /// How the host stream's close went, for a drain; `None` for the pump, which owns no source.
    pub(super) closed: Option<Closed>,
    pub(super) error: Option<lk_agents_abi::errors::ProviderError>,
}

/// Journal a [`ChannelEnd`]; the drain already cleared the row ([`AudioChannels::ended`]).
pub(super) fn on_leg_ended(end: ChannelEnd, out: &mut Out) {
    out.trace(InputStep::ChannelEnded { leg: end.leg, cause: end.cause });
    if let Some(how) = end.closed {
        out.trace(InputStep::HostStreamClosed { label: end.leg, how });
    }
}

/// Drain a provider source into the loop, close it once, then forget its row. A row holding a
/// closed sink would raise on every frame for the rest of the call. The ticket makes the forget
/// conditional ([`AudioChannels::ended`]), since the replacement is installed before the old drain exits.
pub(super) fn spawn_drain<T: Send + 'static>(
    spawner: &Arc<dyn Spawner>,
    time: Timebase,
    mut closing: tokio::sync::watch::Receiver<bool>,
    mut source: Box<dyn StreamSource<T>>,
    tx: Tx<T>,
    legs: Arc<AudioChannels>,
    ticket: Ticket,
    ended: Tx<ChannelEnd>,
) {
    spawner.spawn_unboxed(async move {
        let mut error = None;
        let cause = loop {
            // `biased` so an arrived item is delivered before a simultaneous close.
            let next = tokio::select! {
                biased;
                item = source.next() => Some(item),
                _ = closing.wait_for(|c| *c) => None,
            };
            match next {
                Some(Ok(Some(item))) => {
                    if tx.send(item).is_err() {
                        break ChannelEndCause::Closing; // the loop is gone
                    }
                }
                Some(Ok(None)) => break ChannelEndCause::SourceEnded,
                Some(Err(e)) => {
                    error = Some(e);
                    break ChannelEndCause::Error;
                }
                None => break ChannelEndCause::Closing,
            }
        };
        legs.ended(ticket);
        // Always, bounded on the session clock (dropping doesn't cancel a host stream).
        let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let closed = source.close(deadline).await;
        let _ =
            ended.send(ChannelEnd { leg: ticket.leg.label(), cause, closed: Some(closed), error });
    });
}
