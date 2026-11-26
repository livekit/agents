use crate::ProviderId;
use crate::telemetry::decl::{Attrs, Bounded, Label, Measure, SpanValue, metric};
use crate::telemetry::instruments::DURATION_BOUNDS;

// What every model is counted under. The framework derives the labels from the identity a model
// declares; no table maps a name, and a plugin records nothing.

/// The kind of model a statistic is about.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Stt,
    Tts,
    Llm,
    Realtime,
    Duplex,
}

/// The module path of LiveKit's plugins, each named by the next segment.
const PLUGINS: &str = "livekit.plugins.";
/// The module path of the LiveKit Inference models.
const INFERENCE: &str = "livekit.agents.inference.";
/// The plugin id of the LiveKit Inference models.
pub const INFERENCE_PLUGIN: &str = "livekit-inference";
/// The plugin id of a class outside LiveKit's plugins.
pub const CUSTOM_PLUGIN: &str = "custom";
/// A provider or model name the model does not declare.
pub const UNKNOWN: &str = "unknown";

/// The code, the vendor and the model a statistic is about.
#[derive(Attrs, Bounded, Debug, Clone, PartialEq, Eq)]
pub struct ModelLabels {
    /// STT, TTS, LLM, realtime or duplex.
    #[attr("lk.product.agents.model.kind")]
    pub kind: ModelKind,
    /// The code that runs: a plugin id, `livekit-inference`, or `custom`.
    #[attr("lk.product.agents.model.plugin")]
    pub plugin: Label,
    /// The vendor whose model runs, as the model declares it.
    #[attr("lk.product.agents.model.provider")]
    pub provider: Label,
    /// The vendor's own id for the model.
    #[attr("lk.product.agents.model.name")]
    pub name: Label,
}

impl ModelLabels {
    /// The labels of the model `id` declares.
    ///
    /// The plugin is read off the class path (`id.label`): the segment after `livekit.plugins.`
    /// with hyphens, `livekit-inference` for a `livekit.agents.inference` class, `custom` for
    /// any other. A LiveKit Inference model string names its vendor and options
    /// (`deepgram/nova-3:en`): the provider is the vendor and the name loses both. Any other
    /// provider and name are the declared ones, `unknown` when there is none; a provider that
    /// reads as a host, a URL or an address is `unknown`, never sent.
    pub fn of(kind: ModelKind, id: &ProviderId) -> Self {
        let declared = |v: &str| if v.is_empty() { UNKNOWN } else { v }.to_owned();
        let (plugin, provider, name) = if id.label.starts_with(INFERENCE) {
            let (vendor, model) = id.model.split_once('/').unwrap_or(("", &id.model));
            let model = model.split_once(':').map_or(model, |(model, _options)| model);
            (INFERENCE_PLUGIN.to_owned(), declared(vendor), declared(model))
        } else {
            let plugin = id.label.strip_prefix(PLUGINS).and_then(|path| path.split('.').next());
            let plugin = plugin.filter(|p| !p.is_empty()).map(|p| p.replace('_', "-"));
            (
                plugin.unwrap_or_else(|| CUSTOM_PLUGIN.to_owned()),
                id.vendor().unwrap_or(UNKNOWN).to_owned(),
                declared(&id.model),
            )
        };
        let provider =
            if provider.contains(['.', '/', ':', '@']) { UNKNOWN.into() } else { provider };
        ModelLabels { kind, plugin: Label(plugin), provider: Label(provider), name: Label(name) }
    }
}

/// Requests made to a model: one LLM completion, one TTS synthesis, one STT stream or
/// recognition, one realtime response.
#[metric(
    name = "lk.product.agents.model.requests",
    kind = Counter,
    unit = "{request}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct ModelRequests {
    #[attrs]
    pub model: ModelLabels,
}

/// Tokens a model consumed, as its provider reports them.
#[metric(
    name = "lk.product.agents.model.input_tokens",
    kind = Counter,
    unit = "{token}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct ModelInputTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// Tokens a model produced, as its provider reports them.
#[metric(
    name = "lk.product.agents.model.output_tokens",
    kind = Counter,
    unit = "{token}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct ModelOutputTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// Characters of text a model was given to speak.
#[metric(
    name = "lk.product.agents.model.characters",
    kind = Counter,
    unit = "{character}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct ModelCharacters {
    #[attrs]
    pub model: ModelLabels,
}

/// Audio a model heard (an STT) or spoke (a TTS).
#[metric(
    name = "lk.product.agents.model.audio_duration",
    kind = Counter,
    unit = "s",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct ModelAudioDuration {
    #[attrs]
    pub model: ModelLabels,
}

/// What a model was measured in. It adds to each unit it reported, and to no other.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub characters: u64,
    pub audio_duration: f64,
}

impl ModelUsage {
    pub fn tokens(tokens: &crate::usage::TokenUsage) -> Self {
        ModelUsage {
            input_tokens: tokens.input_tokens,
            output_tokens: tokens.output_tokens,
            ..ModelUsage::default()
        }
    }

    /// Adds this usage to `model`'s statistics: each unit it was measured in, and no other.
    pub fn measure(self, model: &ModelLabels, mut into: impl Measure) {
        let model = || model.clone();
        if self.input_tokens > 0 {
            let tokens = self.input_tokens.into();
            into.measure(&ModelInputTokens { model: model() }, tokens);
        }
        if self.output_tokens > 0 {
            let tokens = self.output_tokens.into();
            into.measure(&ModelOutputTokens { model: model() }, tokens);
        }
        if self.characters > 0 {
            into.measure(&ModelCharacters { model: model() }, self.characters.into());
        }
        if self.audio_duration > 0.0 {
            let seconds = self.audio_duration.into();
            into.measure(&ModelAudioDuration { model: model() }, seconds);
        }
    }
}

/// Time to the first token of an LLM, realtime or duplex response.
#[metric(
    name = "lk.product.agents.model.ttft",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS,
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct ModelTtft {
    #[attrs]
    pub model: ModelLabels,
}

/// Time to the first audio of a TTS request.
#[metric(
    name = "lk.product.agents.model.ttfb",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS,
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct ModelTtfb {
    #[attrs]
    pub model: ModelLabels,
}

/// What ended the speech a transcription delay is measured from.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointedBy {
    /// The session's VAD.
    Vad,
    /// The STT itself.
    Stt,
}

/// Time from the end of the person's speech to the STT's final transcript, once per turn; a
/// turn whose final came first, or whose end of speech was not detected, records nothing.
#[metric(
    name = "lk.product.agents.model.transcription_delay",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS,
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct ModelTranscriptionDelay {
    #[attrs]
    pub model: ModelLabels,
    /// What ended the speech.
    #[attr("lk.product.agents.model.transcription_delay.endpointing")]
    pub endpointing: EndpointedBy,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(label: &str, provider: &str, model: &str) -> (String, String, String) {
        let id = ProviderId { label: label.into(), model: model.into(), provider: provider.into() };
        let labels = ModelLabels::of(ModelKind::Llm, &id);
        (labels.plugin.0, labels.provider.0, labels.name.0)
    }

    // Catches: a plugin counted under shared code instead of its own id (an OpenAI-compatible
    // plugin as `openai`, an Inference model as its vendor's plugin), a hyphenated package losing
    // its hyphen, an Inference model keeping `livekit`, its vendor prefix or its options, a model
    // name renamed, or a host name, URL or address sent as a provider.
    #[test]
    fn a_model_is_named_by_its_plugin_its_declared_vendor_and_its_own_id() {
        let named = |plugin: &str, provider: &str, name: &str| {
            (plugin.to_owned(), provider.to_owned(), name.to_owned())
        };
        for ((label, provider, model), expected) in [
            (
                ("livekit.plugins.deepgram.stt.STT", "deepgram", "nova-3"),
                named("deepgram", "deepgram", "nova-3"),
            ),
            (
                ("livekit.agents.inference.stt.STT", "livekit", "deepgram/nova-3:en"),
                named("livekit-inference", "deepgram", "nova-3"),
            ),
            (
                ("livekit.agents.inference.llm.LLM", "livekit", "openai/gpt-4.1-mini"),
                named("livekit-inference", "openai", "gpt-4.1-mini"),
            ),
            (
                ("livekit.agents.inference.llm.LLM", "livekit", "gpt-4.1-mini"),
                named("livekit-inference", "unknown", "gpt-4.1-mini"),
            ),
            (
                ("livekit.plugins.openai.llm.LLM", "cerebras", "llama-3.3-70b"),
                named("openai", "cerebras", "llama-3.3-70b"),
            ),
            (
                ("livekit.plugins.cerebras.llm.LLM", "cerebras", "gpt-4o-mini-2024-07-18"),
                named("cerebras", "cerebras", "gpt-4o-mini-2024-07-18"),
            ),
            (
                ("livekit.plugins.boson_avatar.tts.TTS", "boson", "higgs"),
                named("boson-avatar", "boson", "higgs"),
            ),
            (("my_app.models.MyLLM", "acme", "m1"), named("custom", "acme", "m1")),
            (("livekit.agents.llm.LLM", "", ""), named("custom", "unknown", "unknown")),
            (
                ("livekit.plugins.openai.llm.LLM", "unknown", "unknown"),
                named("openai", "unknown", "unknown"),
            ),
            (
                ("livekit.plugins.openai.llm.LLM", "llm.internal.example.com", "m"),
                named("openai", "unknown", "m"),
            ),
            (
                ("livekit.plugins.openai.llm.LLM", "https://example", "m"),
                named("openai", "unknown", "m"),
            ),
            (
                ("livekit.plugins.openai.llm.LLM", "localhost:8080", "m"),
                named("openai", "unknown", "m"),
            ),
            (("livekit.plugins.openai.llm.LLM", "me@host", "m"), named("openai", "unknown", "m")),
        ] {
            assert_eq!(labels(label, provider, model), expected, "{label} {provider} {model}");
        }
    }
}
