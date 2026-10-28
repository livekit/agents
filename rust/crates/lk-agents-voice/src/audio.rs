use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::Out;
use crate::trace::{ChannelEndCause, InputStep};
use lk_agents_abi::Closed;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::events::AgentState;
use std::time::Duration;

/// 20 ms. What every STT and VAD in the tree is written against.
pub const DEFAULT_FRAME_MS: u32 = 20;

pub const AUDIO_IN_LEG: &str = "audio_in";

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
}

/// One item a drain delivered, tagged with its stream (the id of the sink it was opened with) so
/// the loop drops what a retired stream still sends.
#[derive(Debug, Clone, PartialEq)]
pub struct Tagged<T> {
    pub stream: ResourceId,
    pub item: T,
}

/// A speech about to push its first frame while the user speaks keeps it until the user stops. No
/// timer. Level, not edge: `publish` runs every iteration; the task reads the level before its
/// first push and parks while it names its speech.
pub struct LaunchGate {
    /// The speeches held at launch. A set: a speech blocked while parked is still held when the
    /// next is authorized behind it.
    held: tokio::sync::watch::Sender<std::collections::BTreeSet<SpeechId>>,
}

impl Default for LaunchGate {
    fn default() -> Self {
        LaunchGate::new()
    }
}

impl LaunchGate {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new() -> Self {
        LaunchGate { held: tokio::sync::watch::channel(std::collections::BTreeSet::new()).0 }
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

    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<std::collections::BTreeSet<SpeechId>> {
        self.held.subscribe()
    }
}

/// A drain or a reader exited, sent to the loop so it is journalled there.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelEnd {
    pub leg: &'static str,
    pub cause: ChannelEndCause,
    /// The sink of the stream a drain read, so the loop removes it if it is still installed.
    pub sink: Option<ResourceId>,
    /// How the host stream's close went, for a drain; `None` for a reader, which owns no source.
    pub closed: Option<Closed>,
    pub error: Option<lk_agents_abi::errors::ProviderError>,
}

impl ChannelEnd {
    /// The leg whose reader or drain ended.
    pub fn leg(&self) -> &'static str {
        self.leg
    }

    /// The stream a drain read: what its items were tagged with, sent before this end.
    pub fn sink(&self) -> Option<ResourceId> {
        self.sink
    }
}

/// Journal a [`ChannelEnd`].
pub(super) fn on_leg_ended(end: ChannelEnd, out: &mut Out) {
    out.trace(InputStep::ChannelEnded { leg: end.leg, cause: end.cause });
    if let Some(how) = end.closed {
        out.trace(InputStep::HostStreamClosed { label: end.leg, how });
    }
}
