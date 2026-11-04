use crate::realtime::dialect::{DEFAULT_MAX_OUTPUT_TOKENS, Dialect, TurnDetection};
use lk_agents_abi::{Maybe, ToolChoice};
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub model: String,
    /// A voice name, or the provider's voice object.
    pub voice: Value,
    pub instructions: Option<String>,
    pub tool_choice: Option<ToolChoice>,
    pub turn_detection: Option<TurnDetection>,
    /// The `AudioTranscription` object as given. `None` is off (`null` on the wire), which also
    /// makes `capabilities.user_transcription` false.
    pub transcription: Option<Value>,
    /// The `NoiseReduction` object; `None` is off.
    pub noise_reduction: Option<Value>,
    pub audio_output: bool,
    pub speed: f64,
    /// `max_response_output_tokens`: `"inf"`, a count, or `null`.
    pub max_output_tokens: Value,
    /// Always on the wire, `null` when `None`.
    pub tracing: Option<Value>,
    /// On the wire only when set.
    pub truncation: Option<Value>,
    /// On the wire only when set.
    pub reasoning: Option<Value>,
    pub max_session_duration: Option<Duration>,
}

impl SessionOptions {
    /// The dialect's defaults, with `turn_detection_disabled` applied when the dialect honours it.
    pub fn from_dialect(d: &Dialect, turn_detection_disabled: bool) -> SessionOptions {
        let disable = turn_detection_disabled && d.supports_turn_detection_disabled;
        SessionOptions {
            model: d.default_model.to_owned(),
            voice: json!(d.default_voice),
            instructions: None,
            tool_choice: None,
            turn_detection: if disable { None } else { d.default_turn_detection.clone() },
            transcription: d.default_transcription_model.map(|m| json!({"model": m})),
            noise_reduction: None,
            // `"audio" in modalities`, and the dialect may not let the caller choose.
            audio_output: d.forced_modalities.is_none_or(|m| m.contains(&"audio")),
            speed: 1.0,
            max_output_tokens: json!(DEFAULT_MAX_OUTPUT_TOKENS),
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
            self.max_output_tokens = v.clone();
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
    pub voice: Maybe<Value>,
    pub turn_detection: Maybe<Option<TurnDetection>>,
    pub max_output_tokens: Maybe<Value>,
    pub transcription: Maybe<Option<Value>>,
    pub noise_reduction: Maybe<Option<Value>>,
    pub speed: Maybe<f64>,
    pub tracing: Maybe<Option<Value>>,
    pub truncation: Maybe<Option<Value>>,
    pub reasoning: Maybe<Option<Value>>,
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
