//! The data plane: what moves frames, stepped beside the decision core by the
//! runtime queue owner. Sans-IO: no lock, no task, no clock read. Its state
//! changes only by the decision core's [`AudioInputRouterCommand`]s, and it answers with
//! [`AudioInputRouterEffect`]s the runtime performs, reports to the decision core among them.
//!
//! Today it is the microphone fan-out, [`AudioInputRouter`] (`AgentActivity.push_audio`).

use super::effects::AudioInputRouterEffect;
use crate::agent_session::{DriverError, VoiceEvent};
use crate::audio::{AudioChannel, DEFAULT_FRAME_MS, Feed};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::rtc::silence_frame;
use lk_agents_abi::{AudioFrame, Ts};
use lk_agents_media::AudioByteStream;
use lk_agents_media::{AudioResampler, OnChange, Quality};

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
    /// like the microphone's last frame. Bypasses the counters.
    FlushSttWithSilence { frames: u32 },
    /// Answered with [`VoiceEvent::InputCensus`].
    ReportCensus,
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
                    effects.push(AudioInputRouterEffect::ReportToSession(Box::new(
                        VoiceEvent::GateSilenced(silenced),
                    )));
                }
                self.discard = discard;
            }
            AudioInputRouterCommand::SetInputEnabled(enabled) => self.input_enabled = enabled,
            AudioInputRouterCommand::FlushSttWithSilence { frames } => {
                let stt = self.slots[AudioChannel::Stt as usize].sink;
                if let (Some(sink), Some((rate, channels))) = (stt, self.last_input) {
                    let per_frame = (f64::from(rate) * 0.2) as u32;
                    if per_frame > 0 {
                        for _ in 0..frames {
                            let frame = silence_frame(per_frame, rate, channels);
                            let head = false;
                            effects.push(AudioInputRouterEffect::PushFrame { sink, frame, head });
                        }
                    }
                }
            }
            AudioInputRouterCommand::ReportCensus => {
                let channels = AudioChannel::ALL
                    .into_iter()
                    .filter_map(|channel| {
                        let slot = &self.slots[channel as usize];
                        Some((channel, slot.frames, slot.silent, slot.shape?))
                    })
                    .collect();
                let census = InputCensus { channels, input: self.input_shape };
                effects.push(AudioInputRouterEffect::ReportToSession(Box::new(
                    VoiceEvent::InputCensus(census),
                )));
            }
        }
        effects
    }

    /// The session ended: every sink still installed has its input ended.
    pub fn close(&mut self) -> Vec<AudioInputRouterEffect> {
        let sinks = self.slots.iter_mut().filter_map(|slot| slot.sink.take());
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
        self.input_shape = Some((FrameShape::of(&frame), self.conversion.frame_shape()));
        // `Raw` channels: the source frame; the substitute has the room frame's shape.
        let raw_silence = self.discard.then(|| silence_like(&frame));
        self.push(Feed::Raw, &frame, raw_silence.as_ref(), head, &mut effects);
        // `Converted` channels: the Rust detectors expect 16 kHz mono.
        let Ok(ready) = self.conversion.convert(&frame) else { return effects };
        for converted in ready {
            let substitute = self.discard.then(|| silence_like(&converted));
            self.push(Feed::Converted, &converted, substitute.as_ref(), head, &mut effects);
        }
        effects
    }

    fn push(
        &mut self,
        feed: Feed,
        frame: &AudioFrame,
        silence: Option<&AudioFrame>,
        head: bool,
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
            effects.push(AudioInputRouterEffect::PushFrame { sink, frame: heard.clone(), head });
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
            AudioInputRouterEffect::PushFrame { sink, head, .. } => Some((*sink, *head)),
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
}
