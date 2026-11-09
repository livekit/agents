use super::effects::{AudioInputRouterEffect, Delivery};
use crate::agent_session::{DriverError, VoiceEvent};
use crate::audio::{AudioChannel, DEFAULT_FRAME_MS, Feed};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::rtc::silence_frame;
use lk_agents_abi::{AudioFrame, Ts};
use lk_agents_media::AudioByteStream;
use lk_agents_media::{AudioResampler, OnChange, Quality};
use std::collections::VecDeque;
use std::time::Duration;

// Routing state changes only through the decision core's `AudioInputRouterCommand`s.

/// The length of each silence frame `FlushSttWithSilence` pushes, in seconds.
const SILENCE_FLUSH_FRAME_SECONDS: f64 = 0.2;

/// The audio kept from before a press (pre-roll) and fed after a release (tail): the press and
/// release travel apart from the audio, so either can arrive a little early or late.
pub const TURN_EDGE: Duration = Duration::from_millis(200);

/// The audio an open turn holds for an STT stream still opening; beyond it the oldest goes.
const TURN_BACKLOG: Duration = Duration::from_secs(30);

/// Control inputs to the microphone fan-out. External commands enter through the owner's queue;
/// the decision core's effects are applied in their batch before another frame is forwarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioInputRouterCommand {
    /// Put `sink` on `channel`. The sink it replaces has its input ended.
    Install { channel: AudioChannel, sink: ResourceId },
    /// `sink` hears nothing more and has its input ended, if a channel still holds it.
    Remove(ResourceId),
    /// The microphone was replaced or detached. The runtime swaps the reader; the forwarder
    /// reports the new source's first frame.
    SetSource(Option<ResourceId>),
    /// The gate's verdict: while `true` the substituted channels hear silence.
    SetGateDiscard(bool),
    /// `input.set_audio_enabled(..)`: while `false` every frame is dropped at the source.
    SetInputEnabled(bool),
    /// `_commit_user_turn`'s "flush the stt by pushing silence": `frames` frames of 0.2 s, shaped
    /// like the microphone's last frame, as filler. Bypasses the counters.
    FlushSttWithSilence { frames: u32 },
    /// Answered with [`VoiceEvent::InputCensus`].
    ReportCensus,
    /// While `true` and no user turn is open, no frame reaches any channel; the last
    /// [`TURN_EDGE`] of audio is kept for the next turn.
    SetPushToTalk(bool),
    /// A user turn opens: the kept audio goes first, then the live audio. Until an STT sink is
    /// installed, the turn's STT audio is held for it.
    StartTurn,
    /// The open turn is released: its STT sink leaves the channel and hears [`TURN_EDGE`] more
    /// of audio, then has its input ended.
    EndTurn,
    /// The open turn is dropped: its STT sink leaves the channel and is aborted, its queued audio
    /// never reaching the provider.
    CancelTurn,
    /// A released turn's STT sink is aborted, in its tail or after it.
    Abort(ResourceId),
    /// A released turn's tail ends now: the microphone went quiet (a client that stops sending
    /// on release).
    EndTail,
}

/// The counters the close journals, answered to [`AudioInputRouterCommand::ReportCensus`].
#[derive(Debug, Clone, PartialEq)]
pub struct InputCensus {
    /// Each channel that carried frames: frames pushed, how many were silence, the last one's
    /// shape.
    pub channels: Vec<(AudioChannel, u64, u64, FrameShape)>,
    /// The microphone's last frame shape and the conversion's.
    pub input: Option<(FrameShape, FrameShape)>,
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
struct LegSlot {
    sink: Option<ResourceId>,
    /// The shape of the last frame pushed here.
    shape: Option<FrameShape>,
    frames: u64,
    silent: u64,
    /// `silent` as of the last gate change: the report carries the delta.
    silent_reported: u64,
}

/// The microphone fan-out: one sink per [`AudioChannel`], the one 16 kHz conversion, the gate's
/// verdict and the input switch.
pub struct AudioInputRouter {
    slots: [LegSlot; 6],
    conversion: AudioFanOut,
    discard: bool,
    pub(crate) input_enabled: bool,
    /// Whether this source's first frame was reported (`VoiceEvent::InputStarted`).
    input_started: bool,
    /// `(sample_rate, num_channels)` of the last frame, muted or not (`AudioRecognition.
    /// _sample_rate`), for the STT flush.
    last_input: Option<(u32, u16)>,
    /// The last forwarded frame's shape and the conversion's.
    input_shape: Option<(FrameShape, FrameShape)>,
    turns: TurnGate,
}

/// Push-to-talk and user turns at the microphone.
#[derive(Default)]
struct TurnGate {
    push_to_talk: bool,
    open: bool,
    /// Push-to-talk outside a turn: the last [`TURN_EDGE`] of microphone frames.
    pre_roll: Edge,
    /// The open turn's STT audio while no STT sink is installed.
    backlog: Edge,
    /// A released turn's STT sink and the audio it still hears.
    tail: Option<(ResourceId, Duration)>,
}

impl TurnGate {
    fn listening(&self) -> bool {
        !self.push_to_talk || self.open
    }
}

/// Frames and their total duration, trimmed from the front to a bound.
#[derive(Default)]
struct Edge {
    frames: VecDeque<AudioFrame>,
    held: Duration,
}

impl Edge {
    fn push(&mut self, frame: AudioFrame, bound: Duration) {
        self.held += length(&frame);
        self.frames.push_back(frame);
        while self.held > bound
            && let Some(old) = self.frames.pop_front()
        {
            self.held = self.held.saturating_sub(length(&old));
        }
    }

    fn take(&mut self) -> VecDeque<AudioFrame> {
        self.held = Duration::ZERO;
        std::mem::take(&mut self.frames)
    }
}

impl AudioInputRouter {
    /// `rate` is what the converted channels want.
    pub fn new(rate: u32) -> Self {
        AudioInputRouter {
            slots: std::array::from_fn(|_| LegSlot::default()),
            conversion: AudioFanOut::new(rate),
            discard: false,
            input_enabled: true,
            input_started: false,
            last_input: None,
            input_shape: None,
            turns: TurnGate::default(),
        }
    }

    /// Whether the microphone is heard: `AgentInput.audio_enabled`.
    pub fn input_enabled(&self) -> bool {
        self.input_enabled
    }

    pub fn handle(&mut self, command: AudioInputRouterCommand) -> Vec<AudioInputRouterEffect> {
        let mut effects = Vec::new();
        match command {
            AudioInputRouterCommand::Install { channel, sink } => {
                let previous = self.slots[channel as usize].sink.replace(sink);
                if let Some(previous) = previous.filter(|p| *p != sink) {
                    effects.push(AudioInputRouterEffect::EndInput(previous));
                }
                if channel == AudioChannel::Stt {
                    let held = self.turns.backlog.take().into_iter();
                    effects.extend(held.map(|frame| AudioInputRouterEffect::PushFrame {
                        sink,
                        frame,
                        delivery: Delivery::Head,
                    }));
                }
            }
            AudioInputRouterCommand::Remove(sink) => {
                if let Some(slot) = self.slots.iter_mut().find(|s| s.sink == Some(sink)) {
                    slot.sink = None;
                    effects.push(AudioInputRouterEffect::EndInput(sink));
                }
            }
            AudioInputRouterCommand::SetSource(_) => self.input_started = false,
            // The silence the closing verdict produced, reported before the verdict that ends it.
            AudioInputRouterCommand::SetGateDiscard(discard) => {
                self.report_gate_silence(&mut effects);
                self.discard = discard;
            }
            AudioInputRouterCommand::SetInputEnabled(enabled) => {
                self.input_enabled = enabled;
                // Nothing more will be heard: the tail ends now.
                if !enabled {
                    self.end_tail(&mut effects);
                }
            }
            AudioInputRouterCommand::FlushSttWithSilence { frames } => {
                self.flush_stt_with_silence(frames, &mut effects)
            }
            AudioInputRouterCommand::ReportCensus => effects.push(self.census()),
            AudioInputRouterCommand::SetPushToTalk(on) => {
                self.turns.push_to_talk = on;
                self.turns.pre_roll.take();
            }
            AudioInputRouterCommand::StartTurn => {
                self.turns.open = true;
                for frame in self.turns.pre_roll.take() {
                    self.fan_out(frame, Delivery::Live, &mut effects);
                }
            }
            AudioInputRouterCommand::EndTurn => {
                self.end_tail(&mut effects);
                self.turns.open = false;
                self.turns.backlog.take();
                if let Some(sink) = self.slots[AudioChannel::Stt as usize].sink.take() {
                    self.turns.tail = Some((sink, TURN_EDGE));
                }
            }
            AudioInputRouterCommand::EndTail => self.end_tail(&mut effects),
            AudioInputRouterCommand::Abort(sink) => {
                if self.turns.tail.is_some_and(|(tail, _)| tail == sink) {
                    self.turns.tail = None;
                }
                effects.push(AudioInputRouterEffect::AbortInput(sink));
            }
            AudioInputRouterCommand::CancelTurn => {
                self.turns.open = false;
                self.turns.backlog.take();
                if let Some(sink) = self.slots[AudioChannel::Stt as usize].sink.take() {
                    effects.push(AudioInputRouterEffect::AbortInput(sink));
                }
            }
        }
        effects
    }

    fn end_tail(&mut self, effects: &mut Vec<AudioInputRouterEffect>) {
        if let Some((sink, _)) = self.turns.tail.take() {
            effects.push(AudioInputRouterEffect::EndInput(sink));
        }
    }

    /// A released turn's STT sink hears `frame`; the tail ends once it heard [`TURN_EDGE`].
    fn feed_tail(&mut self, frame: &AudioFrame, effects: &mut Vec<AudioInputRouterEffect>) {
        let Some((sink, left)) = &mut self.turns.tail else { return };
        let frame = frame.clone();
        *left = left.saturating_sub(length(&frame));
        let (sink, done) = (*sink, left.is_zero());
        effects.push(AudioInputRouterEffect::PushFrame { sink, frame, delivery: Delivery::Live });
        if done {
            self.end_tail(effects);
        }
    }

    fn report_gate_silence(&mut self, effects: &mut Vec<AudioInputRouterEffect>) {
        let silenced: Vec<(AudioChannel, u64)> = AudioChannel::ALL
            .into_iter()
            .filter_map(|channel| {
                let slot = &mut self.slots[channel as usize];
                let delta = slot.silent - slot.silent_reported;
                slot.silent_reported = slot.silent;
                (delta > 0).then_some((channel, delta))
            })
            .collect();
        if !silenced.is_empty() {
            let event = VoiceEvent::GateSilenced(silenced);
            effects.push(AudioInputRouterEffect::ReportToSession(Box::new(event)));
        }
    }

    /// `frames` frames of silence into the STT, shaped as the last microphone frame. Filler: it
    /// never pushes the person's audio still queued out of a bounded input.
    fn flush_stt_with_silence(&self, frames: u32, effects: &mut Vec<AudioInputRouterEffect>) {
        let stt = self.slots[AudioChannel::Stt as usize].sink;
        let (Some(sink), Some((rate, channels))) = (stt, self.last_input) else { return };
        let per_frame = (f64::from(rate) * SILENCE_FLUSH_FRAME_SECONDS) as u32;
        if per_frame == 0 {
            return;
        }
        effects.extend((0..frames).map(|_| {
            let frame = silence_frame(per_frame, rate, channels);
            AudioInputRouterEffect::PushFrame { sink, frame, delivery: Delivery::Filler }
        }));
    }

    fn census(&self) -> AudioInputRouterEffect {
        let channels = AudioChannel::ALL
            .into_iter()
            .filter_map(|channel| {
                let slot = &self.slots[channel as usize];
                Some((channel, slot.frames, slot.silent, slot.shape?))
            })
            .collect();
        let census = InputCensus { channels, input: self.input_shape };
        AudioInputRouterEffect::ReportToSession(Box::new(VoiceEvent::InputCensus(census)))
    }

    /// The session ended: every sink still installed has its input ended, and an open turn's.
    pub fn close(&mut self) -> Vec<AudioInputRouterEffect> {
        let tail = self.turns.tail.take().map(|(sink, _)| sink);
        let sinks = self.slots.iter_mut().filter_map(|slot| slot.sink.take()).chain(tail);
        sinks.map(AudioInputRouterEffect::EndInput).collect()
    }

    /// One microphone frame to every installed sink on its feed side: silence to the substituted
    /// channels while the gate discards, the frame to the rest. A `head` frame replays pre-connect
    /// audio, and every sink keeps it in full.
    pub fn on_frame(
        &mut self,
        frame: AudioFrame,
        head: bool,
        now: Ts,
    ) -> Vec<AudioInputRouterEffect> {
        let mut effects = Vec::new();
        self.last_input = Some((frame.sample_rate, frame.num_channels));
        // Muted: dropped at the source, before any channel or counter sees it.
        if !self.input_enabled {
            return effects;
        }
        if !std::mem::replace(&mut self.input_started, true) {
            effects.push(AudioInputRouterEffect::ReportToSession(Box::new(
                VoiceEvent::InputStarted { at: now },
            )));
        }
        self.feed_tail(&frame, &mut effects);
        // Push-to-talk outside a turn: no channel hears it, the pre-roll keeps it.
        if !self.turns.listening() {
            self.turns.pre_roll.push(frame, TURN_EDGE);
            return effects;
        }
        let delivery = if head { Delivery::Head } else { Delivery::Live };
        self.fan_out(frame, delivery, &mut effects);
        effects
    }

    fn fan_out(
        &mut self,
        frame: AudioFrame,
        delivery: Delivery,
        effects: &mut Vec<AudioInputRouterEffect>,
    ) {
        self.input_shape = Some((FrameShape::of(&frame), self.conversion.frame_shape()));
        // An open turn whose STT stream is still opening keeps its audio for it.
        if self.turns.open && self.slots[AudioChannel::Stt as usize].sink.is_none() {
            self.turns.backlog.push(frame.clone(), TURN_BACKLOG);
        }
        // `Raw` channels: the source frame; the substitute has the room frame's shape.
        let raw_silence = self.discard.then(|| silence_like(&frame));
        self.push(Feed::Raw, &frame, raw_silence.as_ref(), delivery, effects);
        // `Converted` channels: the Rust detectors expect 16 kHz mono.
        let Ok(ready) = self.conversion.convert(&frame) else { return };
        for converted in ready {
            let substitute = self.discard.then(|| silence_like(&converted));
            self.push(Feed::Converted, &converted, substitute.as_ref(), delivery, effects);
        }
    }

    fn push(
        &mut self,
        feed: Feed,
        frame: &AudioFrame,
        silence: Option<&AudioFrame>,
        delivery: Delivery,
        effects: &mut Vec<AudioInputRouterEffect>,
    ) {
        for channel in AudioChannel::ALL {
            if channel.feed() != feed {
                continue;
            }
            let slot = &mut self.slots[channel as usize];
            let Some(sink) = slot.sink else { continue };
            let heard = match silence {
                Some(s) if channel.substituted() => {
                    slot.silent += 1;
                    s
                }
                _ => frame,
            };
            slot.frames += 1;
            slot.shape = Some(FrameShape::of(heard));
            let frame = heard.clone();
            effects.push(AudioInputRouterEffect::PushFrame { sink, frame, delivery });
        }
    }
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
    resampler: AudioResampler,
    chunker: AudioByteStream,
}

impl AudioFanOut {
    /// `target_rate` is what the STT and VAD legs want -- 16 kHz for every model in the tree.
    pub fn new(target_rate: u32) -> Self {
        let frame_samples = target_rate / 1000 * DEFAULT_FRAME_MS;
        AudioFanOut {
            target_rate,
            frame_samples,
            resampler: AudioResampler::new(target_rate, Quality::Medium, OnChange::Discard),
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
        let ready = self
            .resampler
            .push_frame(&mono(frame))
            .map_err(|e| DriverError::Resample(e.to_string()))?;
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

fn length(frame: &AudioFrame) -> Duration {
    Duration::from_secs_f64(frame.duration().max(0.0))
}

/// A silence frame shaped like `frame`, so substituted channels keep that channel's rate.
fn silence_like(frame: &AudioFrame) -> AudioFrame {
    silence_frame(frame.samples_per_channel, frame.sample_rate, frame.num_channels)
}

/// Interleaved `i16` -> the same frame averaged to mono.
fn mono(frame: &AudioFrame) -> AudioFrame {
    let ch = frame.num_channels.max(1) as usize;
    let mut bytes = Vec::with_capacity(frame.data.len() / ch);
    // Sum in i32: a wrapping i16 add turns a loud stereo frame into white noise.
    for group in frame.data.chunks_exact(2 * ch) {
        let sum: i32 = group
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i32::from(i16::from_le_bytes([b[0], b[1]])))
            .sum();
        bytes.extend_from_slice(&((sum / ch as i32) as i16).to_le_bytes());
    }
    let spc = (bytes.len() / 2) as u32;
    AudioFrame::new(bytes, frame.sample_rate, 1, spc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pushed(effects: &[AudioInputRouterEffect]) -> Vec<(ResourceId, bool)> {
        let pushes = effects.iter().filter_map(|effect| match effect {
            AudioInputRouterEffect::PushFrame { sink, delivery, .. } => {
                Some((*sink, *delivery == Delivery::Head))
            }
            _ => None,
        });
        pushes.collect()
    }

    // Catches: pre-connect audio reaching a converted or raw sink as live audio, which a `Now`
    // sink cuts to its last 500 ms.
    #[test]
    fn a_head_frame_reaches_every_sink_as_head() {
        let mut router = AudioInputRouter::new(16_000);
        let (vad, eot) = (ResourceId(1), ResourceId(2));
        router.handle(AudioInputRouterCommand::Install { channel: AudioChannel::Vad, sink: vad });
        router.handle(AudioInputRouterCommand::Install { channel: AudioChannel::Eot, sink: eot });
        let frame = || silence_frame(320, 16_000, 1);
        let head = router.on_frame(frame(), true, Ts::ZERO);
        assert_eq!(pushed(&head), [(vad, true), (eot, true)]);
        let live = router.on_frame(frame(), false, Ts::ZERO);
        assert_eq!(pushed(&live), [(vad, false), (eot, false)]);
    }

    /// A 20 ms live frame at 16 kHz mono, the shape that skips the conversion.
    fn mic() -> AudioFrame {
        silence_frame(320, 16_000, 1)
    }

    fn ended(effects: &[AudioInputRouterEffect]) -> Vec<(ResourceId, bool)> {
        let ends = effects.iter().filter_map(|effect| match effect {
            AudioInputRouterEffect::EndInput(sink) => Some((*sink, false)),
            AudioInputRouterEffect::AbortInput(sink) => Some((*sink, true)),
            _ => None,
        });
        ends.collect()
    }

    fn turn_router() -> (AudioInputRouter, ResourceId, ResourceId) {
        let mut router = AudioInputRouter::new(16_000);
        let (stt, vad) = (ResourceId(1), ResourceId(2));
        router.handle(AudioInputRouterCommand::Install { channel: AudioChannel::Stt, sink: stt });
        router.handle(AudioInputRouterCommand::Install { channel: AudioChannel::Vad, sink: vad });
        router.handle(AudioInputRouterCommand::SetPushToTalk(true));
        (router, stt, vad)
    }

    // Catches: audio outside a push-to-talk turn reaching the STT, the VAD or the interruption
    // detector, or the press losing the words said just before it.
    #[test]
    fn push_to_talk_hears_nothing_outside_a_turn_and_keeps_a_pre_roll() {
        let (mut router, stt, vad) = turn_router();
        for _ in 0..20 {
            assert!(pushed(&router.on_frame(mic(), false, Ts::ZERO)).is_empty());
        }
        let start = router.handle(AudioInputRouterCommand::StartTurn);
        // 200 ms of 20 ms frames, to both sinks.
        assert_eq!(pushed(&start).len(), 2 * 10);
        assert!(pushed(&start).iter().all(|(sink, _)| *sink == stt || *sink == vad));
        assert_eq!(pushed(&router.on_frame(mic(), false, Ts::ZERO)), [(stt, false), (vad, false)]);
    }

    // Catches: a release cutting the turn's last words, or a turn's STT left open after it.
    #[test]
    fn a_released_turn_hears_its_tail_then_its_input_ends() {
        let (mut router, stt, _) = turn_router();
        router.handle(AudioInputRouterCommand::StartTurn);
        assert!(ended(&router.handle(AudioInputRouterCommand::EndTurn)).is_empty());
        let mut tail = Vec::new();
        for _ in 0..10 {
            tail.extend(router.on_frame(mic(), false, Ts::ZERO));
        }
        assert_eq!(pushed(&tail), vec![(stt, false); 10], "the tail reaches the STT alone");
        assert_eq!(ended(&tail), [(stt, false)]);
        assert!(ended(&router.on_frame(mic(), false, Ts::ZERO)).is_empty());
    }

    // Catches: a cancelled turn's audio finalized by the provider (its input ended, not dropped).
    #[test]
    fn a_cancelled_turn_aborts_its_stt_input() {
        let (mut router, stt, _) = turn_router();
        router.handle(AudioInputRouterCommand::StartTurn);
        router.on_frame(mic(), false, Ts::ZERO);
        assert_eq!(ended(&router.handle(AudioInputRouterCommand::CancelTurn)), [(stt, true)]);
        assert!(pushed(&router.on_frame(mic(), false, Ts::ZERO)).is_empty());
    }

    // Catches: a turn's first words lost while its STT stream is still opening.
    #[test]
    fn a_turn_holds_its_audio_for_an_stt_still_opening() {
        let mut router = AudioInputRouter::new(16_000);
        router.handle(AudioInputRouterCommand::StartTurn);
        for _ in 0..3 {
            router.on_frame(mic(), false, Ts::ZERO);
        }
        let stt = ResourceId(7);
        let install = router
            .handle(AudioInputRouterCommand::Install { channel: AudioChannel::Stt, sink: stt });
        assert_eq!(pushed(&install), vec![(stt, true); 3]);
    }
}
