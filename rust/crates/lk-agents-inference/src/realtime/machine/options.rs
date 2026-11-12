use crate::realtime::dialect::Dialect;
use lk_agents_abi::llm::realtime_options::{
    AudioTranscription, NoiseReduction, Reasoning, Tracing, Truncation, TurnDetection, Voice,
};
use lk_agents_abi::{Maybe, ToolChoice};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub model: String,
    pub voice: Voice,
    pub instructions: Option<String>,
    pub tool_choice: Option<ToolChoice>,
    pub turn_detection: Option<TurnDetection>,
    /// `None` is off (`null` on the wire), which also makes `capabilities.user_transcription`
    /// false.
    pub transcription: Option<AudioTranscription>,
    /// `None` is off.
    pub noise_reduction: Option<NoiseReduction>,
    pub audio_output: bool,
    pub speed: f64,
    /// The reply's token cap; `None` is the model's limit (`"inf"` on the wire).
    pub max_output_tokens: Option<u32>,
    /// Always on the wire, `null` when `None`.
    pub tracing: Option<Tracing>,
    /// On the wire only when set.
    pub truncation: Option<Truncation>,
    /// On the wire only when set.
    pub reasoning: Option<Reasoning>,
    pub max_session_duration: Option<Duration>,
}

/// Transcription by `model`, with the server's other defaults.
pub fn transcription(model: &str) -> AudioTranscription {
    AudioTranscription { model: Some(model.to_owned()), language: None, prompt: None }
}

impl SessionOptions {
    /// The dialect's defaults, with `turn_detection_disabled` applied when the dialect honours it.
    pub fn from_dialect(d: &Dialect, turn_detection_disabled: bool) -> SessionOptions {
        let disable = turn_detection_disabled && d.supports_turn_detection_disabled;
        SessionOptions {
            model: d.default_model.to_owned(),
            voice: Voice::Name(d.default_voice.to_owned()),
            instructions: None,
            tool_choice: None,
            turn_detection: if disable { None } else { d.default_turn_detection.clone() },
            transcription: d.default_transcription_model.map(transcription),
            noise_reduction: None,
            // `"audio" in modalities`, and the dialect may not let the caller choose.
            audio_output: d.forced_modalities.is_none_or(|m| m.contains(&"audio")),
            speed: 1.0,
            max_output_tokens: None,
            tracing: None,
            truncation: None,
            reasoning: None,
            max_session_duration: d.default_max_session_duration,
        }
    }

    /// The model's half of `update_options`: the values are taken, the sessions are told.
    pub fn apply(&mut self, p: &OptionsPatch) {
        if let Maybe::Given(v) = &p.tool_choice {
            self.tool_choice = v.clone();
        }
        if let Maybe::Given(v) = &p.voice {
            self.voice = v.clone();
        }
        if let Maybe::Given(v) = &p.turn_detection {
            self.turn_detection = v.clone();
        }
        if let Maybe::Given(v) = &p.max_output_tokens {
            self.max_output_tokens = *v;
        }
        if let Maybe::Given(v) = &p.transcription {
            self.transcription = v.clone();
        }
        if let Maybe::Given(v) = &p.noise_reduction {
            self.noise_reduction = v.clone();
        }
        if let Maybe::Given(v) = p.speed {
            self.speed = v;
        }
        if let Maybe::Given(v) = &p.tracing {
            self.tracing = v.clone();
        }
        if let Maybe::Given(v) = &p.truncation {
            self.truncation = v.clone();
        }
        if let Maybe::Given(v) = &p.reasoning {
            self.reasoning = v.clone();
        }
    }
}

/// `update_options(..)`'s keywords; each `Given` is compared against the session's own value and
/// only a change goes on the wire.
#[derive(Clone, Default)]
pub struct OptionsPatch {
    pub tool_choice: Maybe<Option<ToolChoice>>,
    pub voice: Maybe<Voice>,
    pub turn_detection: Maybe<Option<TurnDetection>>,
    pub max_output_tokens: Maybe<Option<u32>>,
    pub transcription: Maybe<Option<AudioTranscription>>,
    pub noise_reduction: Maybe<Option<NoiseReduction>>,
    pub speed: Maybe<f64>,
    pub tracing: Maybe<Option<Tracing>>,
    pub truncation: Maybe<Option<Truncation>>,
    pub reasoning: Maybe<Option<Reasoning>>,
}

impl OptionsPatch {
    pub fn is_empty(&self) -> bool {
        matches!(self.tool_choice, Maybe::NotGiven)
            && matches!(self.voice, Maybe::NotGiven)
            && matches!(self.turn_detection, Maybe::NotGiven)
            && matches!(self.max_output_tokens, Maybe::NotGiven)
            && matches!(self.transcription, Maybe::NotGiven)
            && matches!(self.noise_reduction, Maybe::NotGiven)
            && matches!(self.speed, Maybe::NotGiven)
            && matches!(self.tracing, Maybe::NotGiven)
            && matches!(self.truncation, Maybe::NotGiven)
            && matches!(self.reasoning, Maybe::NotGiven)
    }
}

impl std::fmt::Debug for OptionsPatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fields = f.debug_struct("OptionsPatch");
        macro_rules! given { ($($field:ident),*) => { $(fields.field(stringify!($field), &self.$field.is_given());)* }; }
        given!(
            tool_choice,
            voice,
            turn_detection,
            max_output_tokens,
            transcription,
            noise_reduction,
            speed,
            tracing,
            truncation,
            reasoning
        );
        fields.finish()
    }
}
