//! Provider-specific gateway TTS options. Unknown keys remain gateway-owned.

use lk_agents_abi::inference::models::InferenceProvider;
use lk_agents_abi::inference::tts_client::{
    CartesiaSpeedPreset, FishAudioLatency, InworldDeliveryMode, InworldTextNormalization,
    InworldTimestampType,
};
use lk_agents_abi::inference::tts_wire::{
    AutoModeStrategy, ElevenlabsTextNormalization, RimeSegment, TimestampTransportStrategy,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Options {
    Cartesia(Cartesia),
    Deepgram(Deepgram),
    DeepgramFlux(DeepgramFlux),
    Rime(Rime),
    Inworld(Inworld),
    Xai(Xai),
    FishAudio(FishAudio),
    Assemblyai(Assemblyai),
    Elevenlabs(Elevenlabs),
    Google(Google),
    Gradium(Gradium),
    Other(Map<String, Value>),
}

impl Options {
    pub fn parse(model: &str, mut extra: Map<String, Value>) -> Result<Self, serde_json::Error> {
        if InferenceProvider::of_model(model) == Some(InferenceProvider::Rime) {
            for (legacy, canonical) in [
                ("speed_alpha", "speedAlpha"),
                ("reduce_latency", "reduceLatency"),
                ("pause_between_brackets", "pauseBetweenBrackets"),
                ("phonemize_between_brackets", "phonemizeBetweenBrackets"),
                ("inline_speed_alpha", "inlineSpeedAlpha"),
                ("no_text_normalization", "noTextNormalization"),
                ("time_scale_factor", "timeScaleFactor"),
            ] {
                if extra.contains_key(canonical) {
                    extra.remove(legacy);
                }
            }
        }
        let value = Value::Object(extra);
        match InferenceProvider::of_model(model) {
            Some(InferenceProvider::Cartesia) => serde_json::from_value(value).map(Self::Cartesia),
            Some(InferenceProvider::Deepgram) => serde_json::from_value(value).map(Self::Deepgram),
            Some(InferenceProvider::Deepgramflux) => {
                serde_json::from_value(value).map(Self::DeepgramFlux)
            }
            Some(InferenceProvider::Rime) => serde_json::from_value(value).map(Self::Rime),
            Some(InferenceProvider::Inworld) => serde_json::from_value(value).map(Self::Inworld),
            Some(InferenceProvider::Xai) => serde_json::from_value(value).map(Self::Xai),
            Some(InferenceProvider::Fishaudio) => {
                serde_json::from_value(value).map(Self::FishAudio)
            }
            Some(InferenceProvider::Assemblyai) => {
                serde_json::from_value(value).map(Self::Assemblyai)
            }
            Some(InferenceProvider::Elevenlabs) => {
                serde_json::from_value(value).map(Self::Elevenlabs)
            }
            Some(InferenceProvider::Google) => serde_json::from_value(value).map(Self::Google),
            Some(InferenceProvider::Gradium) => serde_json::from_value(value).map(Self::Gradium),
            _ => serde_json::from_value(value).map(Self::Other),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Speed {
    Preset(CartesiaSpeedPreset),
    Value(f64),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cartesia {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<Speed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_buffer_delay_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub add_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub add_phoneme_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_normalized_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalization: Option<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Deepgram {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mip_opt_out: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeepgramFlux {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mip_opt_out: Option<bool>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Rime {
    #[serde(rename = "speedAlpha", alias = "speed_alpha", skip_serializing_if = "Option::is_none")]
    pub speed_alpha: Option<f64>,
    #[serde(
        rename = "reduceLatency",
        alias = "reduce_latency",
        skip_serializing_if = "Option::is_none"
    )]
    pub reduce_latency: Option<bool>,
    #[serde(
        rename = "pauseBetweenBrackets",
        alias = "pause_between_brackets",
        skip_serializing_if = "Option::is_none"
    )]
    pub pause_between_brackets: Option<bool>,
    #[serde(
        rename = "phonemizeBetweenBrackets",
        alias = "phonemize_between_brackets",
        skip_serializing_if = "Option::is_none"
    )]
    pub phonemize_between_brackets: Option<bool>,
    #[serde(
        rename = "inlineSpeedAlpha",
        alias = "inline_speed_alpha",
        skip_serializing_if = "Option::is_none"
    )]
    pub inline_speed_alpha: Option<String>,
    #[serde(
        rename = "noTextNormalization",
        alias = "no_text_normalization",
        skip_serializing_if = "Option::is_none"
    )]
    pub no_text_normalization: Option<bool>,
    #[serde(
        rename = "timeScaleFactor",
        alias = "time_scale_factor",
        skip_serializing_if = "Option::is_none"
    )]
    pub time_scale_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment: Option<RimeSegment>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Inworld {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaking_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivery_mode: Option<InworldDeliveryMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_type: Option<InworldTimestampType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_text_normalization: Option<InworldTextNormalization>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_transport_strategy: Option<TimestampTransportStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_mode_strategy: Option<AutoModeStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_buffer_delay_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_char_threshold: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Xai {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bit_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimize_streaming_latency: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_normalization: Option<bool>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FishAudio {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency: Option<FishAudioLatency>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize_loudness: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_length: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_chunk_length: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition_on_previous_chunks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub early_stop_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_guard: Option<bool>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Assemblyai {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_timestamps: Option<bool>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Elevenlabs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_text_normalization: Option<ElevenlabsTextNormalization>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inactivity_timeout: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_ssml_parsing: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_alignment: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_length_schedule: Option<Vec<f64>>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Google {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Gradium {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pronunciation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewrite_rules: Option<String>,
    #[serde(alias = "temp", skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_coef: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub padding_bonus: Option<f64>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn provider_options_keep_gateway_extensions_without_null_defaults() {
        let extra =
            serde_json::from_value(json!({"word_timestamps": false, "future": {"x": 1}})).unwrap();
        let options = Options::parse("assemblyai/assemblyai-tts-1", extra).unwrap();
        assert_eq!(
            serde_json::to_value(options).unwrap(),
            json!({"word_timestamps": false, "future": {"x": 1}})
        );
    }

    #[test]
    fn rime_canonical_setting_wins_over_legacy_spelling() {
        let extra = serde_json::from_value(json!({"speed_alpha": 0.5, "speedAlpha": 1.5})).unwrap();
        let options = Options::parse("rime/mistv2", extra).unwrap();
        assert_eq!(serde_json::to_value(options).unwrap(), json!({"speedAlpha": 1.5}));
    }

    #[test]
    fn rime_legacy_settings_emit_the_canonical_gateway_keys() {
        let extra = serde_json::from_value(json!({"reduce_latency": true})).unwrap();
        let options = Options::parse("rime/mistv3", extra).unwrap();
        assert_eq!(serde_json::to_value(options).unwrap(), json!({"reduceLatency": true}));
    }
}
