use super::MeasureProduct;
use super::features::{
    Feature, FeatureBackgroundAudio, FeatureLanguageSwitch, FeatureMcp, FeatureModelSwitch,
    FeatureOverrideOptions, FeatureUpdateOptions,
};
use crate::agent::FunctionToolInfo;
use crate::telemetry::decl::{Attrs, SpanValue, metric};
use std::collections::BTreeSet;

// A session is one call (`rtc_session`) or one text request (`a2a_session`).

/// The door a session came through.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entrypoint {
    RtcSession,
    A2aSession,
}

/// How a session's agents were arranged.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// One agent for the whole session.
    Single,
    /// One agent replaced another.
    Handoff,
    /// A local agent was called as a tool.
    AgentAsTool,
    /// A handoff and an agent as a tool.
    Both,
}

/// Whether a session left its process.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delegation {
    /// Everything ran in this process.
    None,
    /// A `RemoteAgent` was called.
    Remote,
}

/// Sessions, counted once each when the session ends.
#[metric(
    name = "lk.product.agents.sessions",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct Sessions {
    /// A call or a text request.
    #[attr("lk.product.agents.session.entrypoint")]
    pub entrypoint: Entrypoint,
    /// One agent, a handoff, an agent as a tool, or both.
    #[attr("lk.product.agents.session.topology")]
    pub topology: Topology,
    /// Whether a `RemoteAgent` was called.
    #[attr("lk.product.agents.session.delegation")]
    pub delegation: Delegation,
}

/// One session's summary: what its agents did and the features it used, each once. The voice
/// session and the text request both fill one and emit it once, when the session ends.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionStats {
    /// One agent replaced another.
    pub handoff: bool,
    /// A local agent ran as a tool.
    pub agent_as_tool: bool,
    /// A `RemoteAgent` was called.
    pub remote: bool,
    /// The features the session used.
    pub features: BTreeSet<Feature>,
}

impl SessionStats {
    /// The body of `tool` starts: a local agent's, a `RemoteAgent`'s, or neither.
    pub fn tool_started(&mut self, tool: &FunctionToolInfo) {
        self.agent_as_tool |= tool.local_agent.is_some();
        self.remote |= tool.remote.is_some() && tool.local_agent.is_none();
    }

    /// The session used `feature`.
    pub fn used(&mut self, feature: Feature) {
        self.features.insert(feature);
    }

    /// What another part of the same session did, too.
    pub fn merge(&mut self, other: SessionStats) {
        self.handoff |= other.handoff;
        self.agent_as_tool |= other.agent_as_tool;
        self.remote |= other.remote;
        self.features.extend(other.features);
    }

    /// The session ended: its one count, and one for each feature it used.
    pub fn emit(self, entrypoint: Entrypoint, mut into: impl MeasureProduct) {
        let topology = match (self.handoff, self.agent_as_tool) {
            (false, false) => Topology::Single,
            (true, false) => Topology::Handoff,
            (false, true) => Topology::AgentAsTool,
            (true, true) => Topology::Both,
        };
        let delegation = if self.remote { Delegation::Remote } else { Delegation::None };
        into.measure_product(&Sessions { entrypoint, topology, delegation }, 1u64.into());
        for feature in self.features {
            let once = 1u64.into();
            match feature {
                Feature::Mcp => into.measure_product(&FeatureMcp, once),
                Feature::UpdateOptions => into.measure_product(&FeatureUpdateOptions, once),
                Feature::OverrideOptions => into.measure_product(&FeatureOverrideOptions, once),
                Feature::BackgroundAudio => into.measure_product(&FeatureBackgroundAudio, once),
                Feature::LanguageSwitch => into.measure_product(&FeatureLanguageSwitch, once),
                Feature::ModelSwitch(kind) => {
                    into.measure_product(&FeatureModelSwitch { kind }, once)
                }
            }
        }
    }
}

/// The models a voice session runs on.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pipeline {
    SttLlmTts,
    Realtime,
    /// A realtime model with a separate TTS.
    RealtimeTts,
    Duplex,
}

/// What ends the person's turn.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetection {
    Vad,
    Stt,
    RealtimeLlm,
    Manual,
    /// A turn detection model.
    Model,
}

/// How a session decides that the person is interrupting: the mode it runs, not the one asked.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interruption {
    /// The interruption model classifies overlapping speech.
    Adaptive,
    /// Any detected speech interrupts.
    Vad,
    /// Interruptions are disabled.
    Off,
}

/// Whether a false interruption pauses the agent's speech and resumes it.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalseInterruption {
    On,
    Off,
}

/// Voice sessions, counted once each when the session starts, with what it starts with.
#[metric(
    name = "lk.product.agents.voice.sessions",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct VoiceSessions {
    /// The models the session runs on.
    #[attr("lk.product.agents.voice.pipeline")]
    pub pipeline: Pipeline,
    /// What ends the person's turn.
    #[attr("lk.product.agents.voice.turn_detection")]
    pub turn_detection: TurnDetection,
    /// How the session decides that the person is interrupting.
    #[attr("lk.product.agents.voice.interruption")]
    pub interruption: Interruption,
    /// Whether a false interruption pauses the agent's speech and resumes it.
    #[attr("lk.product.agents.voice.false_interruption")]
    pub false_interruption: FalseInterruption,
}
