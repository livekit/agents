use crate::ProviderId;
use crate::telemetry::decl::{Attrs, Bounded, BoundedLabel, Measure, SpanValue, metric};
use crate::telemetry::instruments::DURATION_BOUNDS;

// What every model is counted under. The framework derives the labels from the identity a model
// declares; no table maps a name, and a plugin records nothing.

/// The kind of model a statistic is about.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelKind {
    Stt,
    Tts,
    Llm,
    Realtime,
    Duplex,
}

/// The code, the vendor and the model a statistic is about.
#[derive(Attrs, Bounded, Debug, Clone, PartialEq, Eq)]
pub struct ModelId {
    /// STT, TTS, LLM, realtime or duplex.
    #[attr("lk.product.agents.model.kind")]
    pub kind: ModelKind,
    /// The code that runs: a plugin id, `livekit-inference`, or `custom`.
    #[attr("lk.product.agents.model.plugin")]
    pub plugin: BoundedLabel,
    /// The vendor whose model runs, as the model declares it.
    #[attr("lk.product.agents.model.provider")]
    pub provider: BoundedLabel,
    /// The vendor's own id for the model.
    #[attr("lk.product.agents.model.name")]
    pub name: BoundedLabel,
}

impl ModelId {
    /// The identity `id` declares, copied: its plugin, its provider (the vendor behind a gateway
    /// when it declares one) and its model's own id; a blank one is `unknown`. Nothing is parsed
    /// or mapped here: the model, or its host's base class, declared each value.
    pub fn of(kind: ModelKind, id: &ProviderId) -> ModelId {
        let declared = |upstream: &str, own: &str| {
            BoundedLabel::new(if upstream.trim().is_empty() { own } else { upstream })
        };
        ModelId {
            kind,
            plugin: BoundedLabel::new(&id.plugin),
            provider: declared(&id.upstream_provider, &id.provider),
            name: declared(&id.upstream_model, &id.model),
        }
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
    pub model: ModelId,
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
    pub model: ModelId,
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
    pub model: ModelId,
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
    pub model: ModelId,
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
    pub model: ModelId,
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
    pub fn measure(self, model: &ModelId, mut into: impl Measure) {
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
    pub model: ModelId,
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
    pub model: ModelId,
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
    pub model: ModelId,
    /// What ended the speech.
    #[attr("lk.product.agents.model.transcription_delay.endpointing")]
    pub endpointing: EndpointedBy,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(plugin: &str, provider: &str, model: &str) -> ProviderId {
        ProviderId {
            plugin: plugin.into(),
            provider: provider.into(),
            model: model.into(),
            ..ProviderId::default()
        }
    }

    fn named(id: &ProviderId) -> (String, String, String) {
        let model = ModelId::of(ModelKind::Llm, id);
        let text = |label: &BoundedLabel| label.as_str().to_owned();
        (text(&model.plugin), text(&model.provider), text(&model.name))
    }

    fn expected(plugin: &str, provider: &str, name: &str) -> (String, String, String) {
        (plugin.to_owned(), provider.to_owned(), name.to_owned())
    }

    // Catches: a declared identity respelled, mapped or parsed on its way to a statistic, the
    // gateway reported in place of the vendor a gateway model declares, or a blank value sent.
    #[test]
    fn a_model_id_copies_what_the_model_declares() {
        assert_eq!(
            named(&id("openai", "cerebras", "llama-3.3-70b")),
            expected("openai", "cerebras", "llama-3.3-70b")
        );
        let inference = ProviderId {
            upstream_provider: "deepgram".into(),
            upstream_model: "nova-3".into(),
            ..id("livekit-inference", "livekit", "deepgram/nova-3:en")
        };
        assert_eq!(named(&inference), expected("livekit-inference", "deepgram", "nova-3"));
        assert_eq!(named(&id("custom", "", " ")), expected("custom", "unknown", "unknown"));
        assert_eq!(named(&id("", "acme", "m1")), expected("unknown", "acme", "m1"));
    }

    // Catches: a third-party plugin's free text reaching LiveKit as a label: a URL with a query,
    // a sentence, a user name, or anything long.
    #[test]
    fn a_value_that_is_not_a_short_identifier_is_unknown() {
        let long = "m".repeat(BoundedLabel::MAX_BYTES + 1);
        for unfit in [
            "https://llm.internal.example.com/v1?key=sk-123",
            "My company's fine-tuned model, do not share",
            "me@host",
            "modèle",
            long.as_str(),
        ] {
            assert_eq!(named(&id("acme", unfit, unfit)), expected("acme", "unknown", "unknown"));
        }
        let fits = "m".repeat(BoundedLabel::MAX_BYTES);
        assert_eq!(named(&id("acme", "aws", &fits)), expected("acme", "aws", &fits));
        let dated = "us.anthropic.claude-3-5-sonnet-20241022-v2:0";
        assert_eq!(named(&id("aws", "aws", dated)), expected("aws", "aws", dated));
    }
}
