use lk_agents_abi::inference::models::InferenceProvider;
use lk_agents_abi::stt::inference::{
    AssemblyaiMode, AssemblyaiVoiceFocus, DeepgramFluxRedact, InworldAudioEncoding, OpenaiDelay,
    OpenaiNoiseReduction,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Options {
    Deepgram(Deepgram),
    DeepgramFlux(DeepgramFlux),
    Cartesia(Cartesia),
    Assemblyai(Assemblyai),
    Speechmatics(Speechmatics),
    Xai(Xai),
    Inworld(Inworld),
    Google(Google),
    Openai(Openai),
    Soniox(Soniox),
    Elevenlabs(Elevenlabs),
    Other(Map<String, Value>),
}

impl Options {
    pub fn parse(model: &str, extra: Map<String, Value>) -> Result<Self, serde_json::Error> {
        let value = Value::Object(extra);
        match InferenceProvider::of_model(model) {
            Some(InferenceProvider::Deepgram) if model.starts_with("deepgram/flux-") => {
                serde_json::from_value(value).map(Self::DeepgramFlux)
            }
            Some(InferenceProvider::Deepgramflux) => {
                serde_json::from_value(value).map(Self::DeepgramFlux)
            }
            Some(InferenceProvider::Deepgram) => serde_json::from_value(value).map(Self::Deepgram),
            Some(InferenceProvider::Cartesia) => serde_json::from_value(value).map(Self::Cartesia),
            Some(InferenceProvider::Assemblyai) => {
                serde_json::from_value(value).map(Self::Assemblyai)
            }
            Some(InferenceProvider::Speechmatics) => {
                serde_json::from_value(value).map(Self::Speechmatics)
            }
            Some(InferenceProvider::Xai) => serde_json::from_value(value).map(Self::Xai),
            Some(InferenceProvider::Inworld) => serde_json::from_value(value).map(Self::Inworld),
            Some(InferenceProvider::Google) => serde_json::from_value(value).map(Self::Google),
            Some(InferenceProvider::Openai) => serde_json::from_value(value).map(Self::Openai),
            Some(InferenceProvider::Soniox) => serde_json::from_value(value).map(Self::Soniox),
            Some(InferenceProvider::Elevenlabs) => {
                serde_json::from_value(value).map(Self::Elevenlabs)
            }
            _ => serde_json::from_value(value).map(Self::Other),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Deepgram {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filler_words: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interim_results: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpointing: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub punctuate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smart_format: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<(String, f64)>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyterm: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_hint: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profanity_filter: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numerals: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mip_opt_out: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vad_events: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dictation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detect_language: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_delay: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utterance_end: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redact: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replace: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeepgramFlux {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eager_eot_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eot_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eot_timeout_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyterm: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mip_opt_out: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_hint: Option<OneOrMany>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numerals: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redact: Option<DeepgramFluxRedact>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cartesia {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_volume: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_silence_duration_secs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_start_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_eager_end_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_end_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_end_timeout_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyterm: Option<OneOrMany>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Assemblyai {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format_turns: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_of_turn_confidence_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_end_of_turn_silence_when_confident: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turn_silence: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyterms_prompt: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_context_n_turns: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_focus: Option<AssemblyaiVoiceFocus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_focus_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<AssemblyaiMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_detection: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inactivity_timeout: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_labels: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuous_partials: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interruption_delay: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_codes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_speakers: Option<i64>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Vocabulary {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sounds_like: Option<Vec<String>>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Speechmatics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_locale: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_delay: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_delay_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_sensitivity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_speakers: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefer_current_speaker: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_partials: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_entities: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub punctuation_overrides: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_vocab: Option<Vec<Vocabulary>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_of_utterance_silence_trigger: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_filtering_config: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_filtering_config: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operating_point: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_hints: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_preflight_on_finalized: Option<bool>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Xai {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpointing: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interim_results: Option<bool>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Inworld {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_voice_profile: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_profile_top_n: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_word_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_encoding: Option<InworldAudioEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inactivity_timeout_seconds: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_of_turn_confidence_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_end_of_turn_silence_when_confident: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vad_threshold: Option<f64>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Google {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_vocabulary: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_codes: Option<Vec<String>>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Openai {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delay: Option<OpenaiDelay>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_reduction: Option<OpenaiNoiseReduction>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Soniox {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_hints: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_hints_strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_speaker_diarization: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_language_identification: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_endpoint_detection: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_endpoint_delay_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_sensitivity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_latency_adjustment_level: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_reference_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Value>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Elevenlabs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_speech_duration_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_silence_duration_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_strategy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_language_detection: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_logging: Option<bool>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_options_send_the_gateway_fields_and_extensions() {
        let extra = serde_json::from_value(json!({
            "keywords": ["LiveKit"], "languages": ["en"], "delay": "low", "future": true
        }))
        .unwrap();
        let options = Options::parse("openai/gpt-live-transcribe", extra).unwrap();
        assert_eq!(
            serde_json::to_value(options).unwrap(),
            json!({"keywords": ["LiveKit"], "languages": ["en"], "delay": "low", "future": true})
        );
    }

    #[test]
    fn deepgram_language_hints_accept_one_or_many() {
        for hint in [json!("en"), json!(["en", "es"])] {
            let extra = serde_json::from_value(json!({"language_hint": hint})).unwrap();
            let options = Options::parse("deepgram/flux-general-multi", extra).unwrap();
            assert_eq!(serde_json::to_value(options).unwrap(), json!({"language_hint": hint}));
        }
    }

    #[test]
    fn cartesia_ink_two_sends_keyterms_without_another_providers_fields() {
        let extra = serde_json::from_value(json!({"keyterm": ["LiveKit"]})).unwrap();
        let options = Options::parse("cartesia/ink-2", extra).unwrap();
        assert_eq!(serde_json::to_value(options).unwrap(), json!({"keyterm": ["LiveKit"]}));
    }
}
