use super::models::ModelKind;
use crate::telemetry::decl::{Attrs, metric};

// One counter per feature, each counting uses: one measurement where the feature is used, and
// nothing carried or summarised.

/// An agent replaced another in a voice session.
#[metric(
    name = "lk.product.agents.feature.handoff",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureHandoff;

/// A local agent was invoked as a tool.
#[metric(
    name = "lk.product.agents.feature.agent_as_tool",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureAgentAsTool;

/// A `RemoteAgent` call was made.
#[metric(
    name = "lk.product.agents.feature.remote_agent",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureRemoteAgent;

/// An MCP toolset was set up.
#[metric(
    name = "lk.product.agents.feature.mcp",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureMcp;

/// `session.update_options(...)` was called.
#[metric(
    name = "lk.product.agents.feature.update_options",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureUpdateOptions;

/// `session.override_options(...)` was entered.
#[metric(
    name = "lk.product.agents.feature.override_options",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureOverrideOptions;

/// A background audio player was started, with or without a session.
#[metric(
    name = "lk.product.agents.feature.background_audio",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureBackgroundAudio;

/// The STT reported another language than the one before.
#[metric(
    name = "lk.product.agents.feature.language_switch",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureLanguageSwitch;

/// A model was replaced with another while a session ran.
#[metric(
    name = "lk.product.agents.feature.model_switch",
    kind = Counter,
    unit = "{use}",
    audience = ProductStats
)]
#[derive(Attrs, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FeatureModelSwitch {
    /// The kind of the model installed just before the switch.
    #[attr("lk.product.agents.feature.model_switch.from.kind")]
    pub from: ModelKind,
    /// The kind of the model the switch installed.
    #[attr("lk.product.agents.feature.model_switch.to.kind")]
    pub to: ModelKind,
}
