use super::models::ModelKind;
use crate::telemetry::decl::{Attrs, metric};

// One counter per feature, each counting the sessions that used it: a session's summary holds
// each feature it used once, and adds 1 for it when the session ends.

/// Sessions that ran an agent that has an MCP toolset.
#[metric(
    name = "lk.product.agents.feature.mcp",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureMcp;

/// Sessions that called `session.update_options(...)`.
#[metric(
    name = "lk.product.agents.feature.update_options",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureUpdateOptions;

/// Sessions that entered `session.override_options(...)`.
#[metric(
    name = "lk.product.agents.feature.override_options",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureOverrideOptions;

/// Background audio players started, with or without a session: the one feature counted per use
/// and not per session.
#[metric(
    name = "lk.product.agents.feature.background_audio",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureBackgroundAudio;

/// Sessions that heard the person change language.
#[metric(
    name = "lk.product.agents.feature.language_switch",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct FeatureLanguageSwitch;

/// Sessions that replaced a model with another while running, once for each distinct pair of
/// kinds.
#[metric(
    name = "lk.product.agents.feature.model_switch",
    kind = Counter,
    unit = "{session}",
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

/// A feature a session can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Feature {
    Mcp,
    UpdateOptions,
    OverrideOptions,
    LanguageSwitch,
    ModelSwitch(FeatureModelSwitch),
}
