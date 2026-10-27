//! GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`); see `napi.rs` in `lk-agents-codegen`.

#![allow(clippy::all, dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]

use napi::bindgen_prelude::JsObjectValue as _;
use napi_derive::napi;

#[napi(object, js_name = "UserStateChangedEvent")]
pub struct UserStateChangedEvent {
    #[napi(ts_type = "'speaking' | 'listening' | 'away'")]
    pub old_state: String,
    #[napi(ts_type = "'speaking' | 'listening' | 'away'")]
    pub new_state: String,
    #[napi(ts_type = "number")]
    pub created_at: Option<f64>,
}

impl From<&lk_agents_abi::voice::events::UserStateChanged> for UserStateChangedEvent {
    fn from(v: &lk_agents_abi::voice::events::UserStateChanged) -> Self {
        UserStateChangedEvent {
            old_state: v.old_state.as_str().to_owned(),
            new_state: v.new_state.as_str().to_owned(),
            created_at: Some(v.created_at),
        }
    }
}

#[napi(object, js_name = "AgentStateChangedEvent")]
pub struct AgentStateChangedEvent {
    #[napi(ts_type = "'initializing' | 'idle' | 'listening' | 'thinking' | 'speaking'")]
    pub old_state: String,
    #[napi(ts_type = "'initializing' | 'idle' | 'listening' | 'thinking' | 'speaking'")]
    pub new_state: String,
    #[napi(ts_type = "number")]
    pub created_at: Option<f64>,
}

impl From<&lk_agents_abi::voice::events::AgentStateChanged> for AgentStateChangedEvent {
    fn from(v: &lk_agents_abi::voice::events::AgentStateChanged) -> Self {
        AgentStateChangedEvent {
            old_state: v.old_state.as_str().to_owned(),
            new_state: v.new_state.as_str().to_owned(),
            created_at: Some(lk_agents_abi::epoch_secs(std::time::SystemTime::now())),
        }
    }
}

#[napi(object, js_name = "UserInputTranscribedEvent")]
pub struct UserInputTranscribedEvent {
    #[napi(ts_type = "string")]
    pub transcript: String,
    #[napi(ts_type = "boolean")]
    pub is_final: bool,
    /// Provider-specific ID for the transcribed input item, when available.
    #[napi(ts_type = "string | null")]
    pub item_id: Option<String>,
    #[napi(ts_type = "string | null")]
    pub speaker_id: Option<String>,
    #[napi(ts_type = "string | null")]
    pub language: Option<String>,
    #[napi(ts_type = "number")]
    pub created_at: Option<f64>,
}

impl From<&lk_agents_abi::voice::events::UserInputTranscribed> for UserInputTranscribedEvent {
    fn from(v: &lk_agents_abi::voice::events::UserInputTranscribed) -> Self {
        UserInputTranscribedEvent {
            transcript: v.transcript.to_string(),
            is_final: v.is_final,
            item_id: v.item_id.as_ref().map(|v| v.to_string()),
            speaker_id: v.speaker_id.as_ref().map(|v| v.to_string()),
            language: v.language.as_ref().map(|v| v.to_string()),
            created_at: Some(lk_agents_abi::epoch_secs(std::time::SystemTime::now())),
        }
    }
}

impl From<UserInputTranscribedEvent> for lk_agents_abi::voice::events::UserInputTranscribed {
    fn from(v: UserInputTranscribedEvent) -> Self {
        lk_agents_abi::voice::events::UserInputTranscribed {
            transcript: v.transcript,
            is_final: v.is_final,
            item_id: v.item_id,
            speaker_id: v.speaker_id,
            language: v.language,
            created_at: match v.created_at { Some(v) => v, None => lk_agents_abi::epoch_secs(std::time::SystemTime::now()) },
        }
    }
}

#[napi(object, js_name = "UserTranscriptionTimeoutEvent")]
pub struct UserTranscriptionTimeoutEvent {
    /// Total VAD-detected speech (s) in the turn that produced no transcript.
    #[napi(ts_type = "number")]
    pub speech_duration: f64,
    /// When VAD first detected speech for this (untranscribed) turn.
    #[napi(ts_type = "number")]
    pub vad_speech_started_at: f64,
    #[napi(ts_type = "number")]
    pub created_at: Option<f64>,
}

impl From<&lk_agents_abi::voice::events::UserTranscriptionTimeout> for UserTranscriptionTimeoutEvent {
    fn from(v: &lk_agents_abi::voice::events::UserTranscriptionTimeout) -> Self {
        UserTranscriptionTimeoutEvent {
            speech_duration: v.speech_duration,
            vad_speech_started_at: v.vad_speech_started_at,
            created_at: Some(lk_agents_abi::epoch_secs(std::time::SystemTime::now())),
        }
    }
}

impl From<UserTranscriptionTimeoutEvent> for lk_agents_abi::voice::events::UserTranscriptionTimeout {
    fn from(v: UserTranscriptionTimeoutEvent) -> Self {
        lk_agents_abi::voice::events::UserTranscriptionTimeout {
            speech_duration: v.speech_duration,
            vad_speech_started_at: v.vad_speech_started_at,
            created_at: match v.created_at { Some(v) => v, None => lk_agents_abi::epoch_secs(std::time::SystemTime::now()) },
        }
    }
}

#[napi(object, js_name = "EotPredictionEvent")]
pub struct EotPredictionEvent {
    #[napi(ts_type = "number")]
    pub probability: f64,
    #[napi(ts_type = "number")]
    pub threshold: f64,
    /// Server-side model inference time.
    #[napi(ts_type = "number")]
    pub inference_duration: f64,
    /// End of user speech → prediction received latency (s), anchored on the
    /// VAD-backdated last_speaking_time.
    #[napi(ts_type = "number")]
    pub delay: f64,
    #[napi(ts_type = "number")]
    pub created_at: Option<f64>,
}

impl From<&lk_agents_abi::voice::events::EotPrediction> for EotPredictionEvent {
    fn from(v: &lk_agents_abi::voice::events::EotPrediction) -> Self {
        EotPredictionEvent {
            probability: v.probability,
            threshold: v.threshold,
            inference_duration: v.inference_duration,
            delay: v.delay,
            created_at: Some(lk_agents_abi::epoch_secs(std::time::SystemTime::now())),
        }
    }
}

impl From<EotPredictionEvent> for lk_agents_abi::voice::events::EotPrediction {
    fn from(v: EotPredictionEvent) -> Self {
        lk_agents_abi::voice::events::EotPrediction {
            probability: v.probability,
            threshold: v.threshold,
            inference_duration: v.inference_duration,
            delay: v.delay,
            created_at: match v.created_at { Some(v) => v, None => lk_agents_abi::epoch_secs(std::time::SystemTime::now()) },
        }
    }
}

#[napi(object, js_name = "DetectionOptions")]
pub struct DetectionOptions {
    #[napi(ts_type = "number")]
    pub human_speech_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub human_silence_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub machine_silence_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub no_speech_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub timeout: Option<f64>,
    #[napi(ts_type = "number")]
    pub max_endpointing_delay: Option<f64>,
    #[napi(ts_type = "string")]
    pub prompt: Option<String>,
}

impl From<&lk_agents_abi::voice::amd::AmdDetectionOptions> for DetectionOptions {
    fn from(v: &lk_agents_abi::voice::amd::AmdDetectionOptions) -> Self {
        DetectionOptions {
            human_speech_threshold: v.human_speech_threshold,
            human_silence_threshold: v.human_silence_threshold,
            machine_silence_threshold: v.machine_silence_threshold,
            no_speech_threshold: v.no_speech_threshold,
            timeout: v.timeout,
            max_endpointing_delay: v.max_endpointing_delay,
            prompt: v.prompt.as_ref().map(|v| v.to_string()),
        }
    }
}

impl From<DetectionOptions> for lk_agents_abi::voice::amd::AmdDetectionOptions {
    fn from(v: DetectionOptions) -> Self {
        lk_agents_abi::voice::amd::AmdDetectionOptions {
            human_speech_threshold: v.human_speech_threshold,
            human_silence_threshold: v.human_silence_threshold,
            machine_silence_threshold: v.machine_silence_threshold,
            no_speech_threshold: v.no_speech_threshold,
            timeout: v.timeout,
            max_endpointing_delay: v.max_endpointing_delay,
            prompt: v.prompt,
        }
    }
}

#[napi(object, js_name = "SpeechData")]
pub struct SpeechData {
    #[napi(ts_type = "string")]
    pub language: String,
    #[napi(ts_type = "string")]
    pub text: String,
    #[napi(ts_type = "number")]
    pub start_time: Option<f64>,
    #[napi(ts_type = "number")]
    pub end_time: Option<f64>,
    #[napi(ts_type = "number")]
    pub confidence: Option<f64>,
    #[napi(ts_type = "string | null")]
    pub speaker_id: Option<String>,
    #[napi(ts_type = "string[] | null")]
    pub words: Option<Vec<crate::generated::records::TimedString>>,
    /// the source languages spoken by the user. populated by STT services that support translation,
    /// where `language` holds the target language and `source_languages` holds the original spoken language(s),
    /// or by multi-language detection services where `language` holds the dominant language and
    /// `source_languages` holds all detected languages sorted by prevalence.
    /// may contain multiple entries when a single utterance spans multiple source languages.
    #[napi(ts_type = "string[] | null")]
    pub source_languages: Option<Vec<String>>,
    /// the original transcription segments in the source language(s), when translation is active.
    /// each entry corresponds to the same-indexed entry in `source_languages`.
    #[napi(ts_type = "string[] | null")]
    pub source_texts: Option<Vec<String>>,
    /// the target language(s) produced by a translation-capable STT service, one entry per
    /// consecutive same-language run, parallel to `target_texts`. mirrors `source_languages` /
    /// `source_texts` on the source side: `language` holds the dominant (first) target language
    /// and `target_languages` carries the fine-grained per-run breakdown.
    /// populated when translation is active; None otherwise.
    #[napi(ts_type = "string[] | null")]
    pub target_languages: Option<Vec<String>>,
    /// the translated transcription segments in the target language(s).
    /// each entry corresponds to the same-indexed entry in `target_languages`.
    #[napi(ts_type = "string[] | null")]
    pub target_texts: Option<Vec<String>>,
    /// optional plugin-specific metadata (e.g. voice profile, provider diagnostics).
    /// plugins may populate this with provider-specific data that doesn't map to standard fields.
    #[napi(ts_type = "Record<string, unknown> | null")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

impl From<&lk_agents_abi::SpeechData> for SpeechData {
    fn from(v: &lk_agents_abi::SpeechData) -> Self {
        SpeechData {
            language: (&v.language).as_str().to_owned(),
            text: v.text.to_string(),
            start_time: Some(v.start_time),
            end_time: Some(v.end_time),
            confidence: Some(v.confidence),
            speaker_id: v.speaker_id.as_ref().map(|v| v.to_string()),
            words: v.words.as_ref().map(|v| v.iter().map(|v| crate::generated::records::TimedString::from(v)).collect::<Vec<_>>()),
            source_languages: v.source_languages.as_ref().map(|v| v.iter().map(|v| (v).as_str().to_owned()).collect::<Vec<_>>()),
            source_texts: v.source_texts.as_ref().map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
            target_languages: v.target_languages.as_ref().map(|v| v.iter().map(|v| (v).as_str().to_owned()).collect::<Vec<_>>()),
            target_texts: v.target_texts.as_ref().map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
            metadata: v.metadata.as_ref().map(|v| v.clone()),
        }
    }
}

impl TryFrom<SpeechData> for lk_agents_abi::SpeechData {
    type Error = crate::host_error::JsError;

    fn try_from(v: SpeechData) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::SpeechData {
            language: lk_agents_abi::LanguageCode::new(&v.language),
            text: v.text,
            start_time: match v.start_time { Some(v) => v, None => 0.0_f64 },
            end_time: match v.end_time { Some(v) => v, None => 0.0_f64 },
            confidence: match v.confidence { Some(v) => v, None => 0.0_f64 },
            speaker_id: v.speaker_id,
            words: v.words.map(|v| Ok::<_, crate::host_error::JsError>(v.into_iter().map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).collect::<Result<Vec<_>, _>>()?)).transpose()?,
            source_languages: v.source_languages.map(|v| v.into_iter().map(|v| lk_agents_abi::LanguageCode::new(&v)).collect::<Vec<_>>()),
            source_texts: v.source_texts,
            target_languages: v.target_languages.map(|v| v.into_iter().map(|v| lk_agents_abi::LanguageCode::new(&v)).collect::<Vec<_>>()),
            target_texts: v.target_texts,
            metadata: v.metadata,
        })
    }
}

#[napi(object, js_name = "RecognitionUsage")]
pub struct RecognitionUsage {
    /// Incremental audio duration/usage in seconds
    #[napi(ts_type = "number")]
    pub audio_duration: f64,
    #[napi(ts_type = "number")]
    pub input_tokens: Option<u32>,
    #[napi(ts_type = "number")]
    pub output_tokens: Option<u32>,
}

impl From<&lk_agents_abi::RecognitionUsage> for RecognitionUsage {
    fn from(v: &lk_agents_abi::RecognitionUsage) -> Self {
        RecognitionUsage {
            audio_duration: v.audio_duration,
            input_tokens: Some(v.input_tokens),
            output_tokens: Some(v.output_tokens),
        }
    }
}

impl From<RecognitionUsage> for lk_agents_abi::RecognitionUsage {
    fn from(v: RecognitionUsage) -> Self {
        lk_agents_abi::RecognitionUsage {
            audio_duration: v.audio_duration,
            input_tokens: match v.input_tokens { Some(v) => v, None => 0u32 },
            output_tokens: match v.output_tokens { Some(v) => v, None => 0u32 },
        }
    }
}

#[napi(object, js_name = "SpeechEvent")]
pub struct SpeechEvent {
    #[napi(ts_type = "'start_of_speech' | 'interim_transcript' | 'preflight_transcript' | 'final_transcript' | 'recognition_usage' | 'end_of_speech'")]
    pub r#type: String,
    #[napi(ts_type = "string")]
    pub request_id: Option<String>,
    #[napi(ts_type = "SpeechData[]")]
    pub alternatives: Option<Vec<crate::generated::records::SpeechData>>,
    #[napi(ts_type = "RecognitionUsage | null")]
    pub recognition_usage: Option<crate::generated::records::RecognitionUsage>,
    /// server-reported wall-clock time of speech onset, when the provider sends
    /// a separate speech-start signal carrying onset timing.
    #[napi(ts_type = "number | null")]
    pub speech_start_time: Option<f64>,
    /// Wall-clock time when this event was created.
    #[napi(ts_type = "number")]
    pub created_at: Option<f64>,
    /// Wall-clock time when the recognized speech ended, when known.
    #[napi(ts_type = "number | null")]
    pub speech_end_time: Option<f64>,
}

impl From<&lk_agents_abi::SpeechEvent> for SpeechEvent {
    fn from(v: &lk_agents_abi::SpeechEvent) -> Self {
        SpeechEvent {
            r#type: v.kind.as_str().to_owned(),
            request_id: Some(v.request_id.to_string()),
            alternatives: Some(v.alternatives.iter().map(|v| crate::generated::records::SpeechData::from(v)).collect::<Vec<_>>()),
            recognition_usage: v.recognition_usage.as_ref().map(|v| crate::generated::records::RecognitionUsage::from(v)),
            speech_start_time: v.speech_start_time,
            created_at: v.created_at,
            speech_end_time: v.speech_end_time,
        }
    }
}

#[napi(object, js_name = "TextOutputOptions")]
pub struct TextOutputOptions {
    /// False to disable transcription synchronization with audio output.
    /// Otherwise, transcription is emitted as quickly as available.
    #[napi(ts_type = "boolean")]
    pub sync_transcription: Option<bool>,
    /// Speed factor of transcription synchronization with audio output.
    /// Only effective if `sync_transcription` is True.
    #[napi(ts_type = "number")]
    pub transcription_speed_factor: Option<f64>,
    /// Send the transcription as JSON dict for each chunk, including start and end timestamps if it's a TimedString.
    #[napi(ts_type = "boolean")]
    pub json_format: Option<bool>,
}

impl From<&lk_agents_abi::voice::room_io::TextOutputConfig> for TextOutputOptions {
    fn from(v: &lk_agents_abi::voice::room_io::TextOutputConfig) -> Self {
        TextOutputOptions {
            sync_transcription: v.sync_transcription,
            transcription_speed_factor: Some(v.transcription_speed_factor),
            json_format: Some(v.json_format),
        }
    }
}

impl From<TextOutputOptions> for lk_agents_abi::voice::room_io::TextOutputConfig {
    fn from(v: TextOutputOptions) -> Self {
        lk_agents_abi::voice::room_io::TextOutputConfig {
            sync_transcription: v.sync_transcription,
            transcription_speed_factor: match v.transcription_speed_factor { Some(v) => v, None => 1.0_f64 },
            json_format: match v.json_format { Some(v) => v, None => false },
        }
    }
}

#[napi(object, js_name = "VideoInputOptions")]
pub struct VideoInputOptions {
}

impl From<&lk_agents_abi::voice::room_io::VideoInputOptions> for VideoInputOptions {
    fn from(v: &lk_agents_abi::voice::room_io::VideoInputOptions) -> Self {
        VideoInputOptions {
        }
    }
}

impl From<VideoInputOptions> for lk_agents_abi::voice::room_io::VideoInputOptions {
    fn from(v: VideoInputOptions) -> Self {
        lk_agents_abi::voice::room_io::VideoInputOptions {
        }
    }
}

/// Inference Fallback Adapter: configuration for a fallback STT model that runs server-side in LiveKit Inference, providing automatic fallback between providers.
#[napi(object, js_name = "SttFallbackModel")]
pub struct SttFallbackModel {
    /// Model name (e.g. "deepgram/nova-3", "assemblyai/universal-streaming", "cartesia/ink-whisper").
    #[napi(ts_type = "string")]
    pub model: String,
    /// Extra configuration for the model.
    #[napi(ts_type = "Record<string, unknown>")]
    pub extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

impl From<&lk_agents_abi::inference::fallback::SttFallbackModel> for SttFallbackModel {
    fn from(v: &lk_agents_abi::inference::fallback::SttFallbackModel) -> Self {
        SttFallbackModel {
            model: v.model.to_string(),
            extra_kwargs: v.extra_kwargs.as_ref().map(|v| v.clone()),
        }
    }
}

impl From<SttFallbackModel> for lk_agents_abi::inference::fallback::SttFallbackModel {
    fn from(v: SttFallbackModel) -> Self {
        lk_agents_abi::inference::fallback::SttFallbackModel {
            model: v.model,
            extra_kwargs: v.extra_kwargs,
        }
    }
}

/// Inference Fallback Adapter: configuration for a fallback TTS model that runs server-side in LiveKit Inference, providing automatic fallback between providers.
#[napi(object, js_name = "TtsFallbackModel")]
pub struct TtsFallbackModel {
    /// Model name (e.g. "cartesia/sonic", "inworld/inworld-tts-1", "rime/coda").
    #[napi(ts_type = "string")]
    pub model: String,
    /// Voice to use for the model.
    #[napi(ts_type = "string")]
    pub voice: String,
    /// Extra configuration for the model.
    #[napi(ts_type = "Record<string, unknown>")]
    pub extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

impl From<&lk_agents_abi::inference::fallback::TtsFallbackModel> for TtsFallbackModel {
    fn from(v: &lk_agents_abi::inference::fallback::TtsFallbackModel) -> Self {
        TtsFallbackModel {
            model: v.model.to_string(),
            voice: v.voice.to_string(),
            extra_kwargs: v.extra_kwargs.as_ref().map(|v| v.clone()),
        }
    }
}

impl From<TtsFallbackModel> for lk_agents_abi::inference::fallback::TtsFallbackModel {
    fn from(v: TtsFallbackModel) -> Self {
        lk_agents_abi::inference::fallback::TtsFallbackModel {
            model: v.model,
            voice: v.voice,
            extra_kwargs: v.extra_kwargs,
        }
    }
}

/// GetAddressResult(address: 'str')
#[napi(object, js_name = "GetAddressResult")]
pub struct GetAddressResult {
    #[napi(ts_type = "string")]
    pub address: String,
}

impl From<&lk_agents_abi::voice::workflows::GetAddressResult> for GetAddressResult {
    fn from(v: &lk_agents_abi::voice::workflows::GetAddressResult) -> Self {
        GetAddressResult {
            address: v.address.to_string(),
        }
    }
}

impl From<GetAddressResult> for lk_agents_abi::voice::workflows::GetAddressResult {
    fn from(v: GetAddressResult) -> Self {
        lk_agents_abi::voice::workflows::GetAddressResult {
            address: v.address,
        }
    }
}

/// GetEmailResult(email_address: 'str')
#[napi(object, js_name = "GetEmailResult")]
pub struct GetEmailResult {
    #[napi(ts_type = "string")]
    pub email_address: String,
}

impl From<&lk_agents_abi::voice::workflows::GetEmailResult> for GetEmailResult {
    fn from(v: &lk_agents_abi::voice::workflows::GetEmailResult) -> Self {
        GetEmailResult {
            email_address: v.email_address.to_string(),
        }
    }
}

impl From<GetEmailResult> for lk_agents_abi::voice::workflows::GetEmailResult {
    fn from(v: GetEmailResult) -> Self {
        lk_agents_abi::voice::workflows::GetEmailResult {
            email_address: v.email_address,
        }
    }
}

/// GetPhoneNumberResult(phone_number: 'str')
#[napi(object, js_name = "GetPhoneNumberResult")]
pub struct GetPhoneNumberResult {
    #[napi(ts_type = "string")]
    pub phone_number: String,
}

impl From<&lk_agents_abi::voice::workflows::GetPhoneNumberResult> for GetPhoneNumberResult {
    fn from(v: &lk_agents_abi::voice::workflows::GetPhoneNumberResult) -> Self {
        GetPhoneNumberResult {
            phone_number: v.phone_number.to_string(),
        }
    }
}

impl From<GetPhoneNumberResult> for lk_agents_abi::voice::workflows::GetPhoneNumberResult {
    fn from(v: GetPhoneNumberResult) -> Self {
        lk_agents_abi::voice::workflows::GetPhoneNumberResult {
            phone_number: v.phone_number,
        }
    }
}

/// GetNameResult(first_name: 'str | None' = None, middle_name: 'str | None' = None, last_name: 'str | None' = None)
#[napi(object, js_name = "GetNameResult")]
pub struct GetNameResult {
    #[napi(ts_type = "string | null")]
    pub first_name: Option<String>,
    #[napi(ts_type = "string | null")]
    pub middle_name: Option<String>,
    #[napi(ts_type = "string | null")]
    pub last_name: Option<String>,
}

impl From<&lk_agents_abi::voice::workflows::GetNameResult> for GetNameResult {
    fn from(v: &lk_agents_abi::voice::workflows::GetNameResult) -> Self {
        GetNameResult {
            first_name: v.first_name.as_ref().map(|v| v.to_string()),
            middle_name: v.middle_name.as_ref().map(|v| v.to_string()),
            last_name: v.last_name.as_ref().map(|v| v.to_string()),
        }
    }
}

impl From<GetNameResult> for lk_agents_abi::voice::workflows::GetNameResult {
    fn from(v: GetNameResult) -> Self {
        lk_agents_abi::voice::workflows::GetNameResult {
            first_name: v.first_name,
            middle_name: v.middle_name,
            last_name: v.last_name,
        }
    }
}

/// GetCardNumberResult(issuer: 'str', card_number: 'str')
#[napi(object, js_name = "GetCardNumberResult")]
pub struct GetCardNumberResult {
    #[napi(ts_type = "string")]
    pub issuer: String,
    #[napi(ts_type = "string")]
    pub card_number: String,
}

impl From<&lk_agents_abi::voice::workflows::GetCardNumberResult> for GetCardNumberResult {
    fn from(v: &lk_agents_abi::voice::workflows::GetCardNumberResult) -> Self {
        GetCardNumberResult {
            issuer: v.issuer.to_string(),
            card_number: v.card_number.to_string(),
        }
    }
}

impl From<GetCardNumberResult> for lk_agents_abi::voice::workflows::GetCardNumberResult {
    fn from(v: GetCardNumberResult) -> Self {
        lk_agents_abi::voice::workflows::GetCardNumberResult {
            issuer: v.issuer,
            card_number: v.card_number,
        }
    }
}

/// GetSecurityCodeResult(security_code: 'str')
#[napi(object, js_name = "GetSecurityCodeResult")]
pub struct GetSecurityCodeResult {
    #[napi(ts_type = "string")]
    pub security_code: String,
}

impl From<&lk_agents_abi::voice::workflows::GetSecurityCodeResult> for GetSecurityCodeResult {
    fn from(v: &lk_agents_abi::voice::workflows::GetSecurityCodeResult) -> Self {
        GetSecurityCodeResult {
            security_code: v.security_code.to_string(),
        }
    }
}

impl From<GetSecurityCodeResult> for lk_agents_abi::voice::workflows::GetSecurityCodeResult {
    fn from(v: GetSecurityCodeResult) -> Self {
        lk_agents_abi::voice::workflows::GetSecurityCodeResult {
            security_code: v.security_code,
        }
    }
}

/// GetExpirationDateResult(date: 'str')
#[napi(object, js_name = "GetExpirationDateResult")]
pub struct GetExpirationDateResult {
    #[napi(ts_type = "string")]
    pub date: String,
}

impl From<&lk_agents_abi::voice::workflows::GetExpirationDateResult> for GetExpirationDateResult {
    fn from(v: &lk_agents_abi::voice::workflows::GetExpirationDateResult) -> Self {
        GetExpirationDateResult {
            date: v.date.to_string(),
        }
    }
}

impl From<GetExpirationDateResult> for lk_agents_abi::voice::workflows::GetExpirationDateResult {
    fn from(v: GetExpirationDateResult) -> Self {
        lk_agents_abi::voice::workflows::GetExpirationDateResult {
            date: v.date,
        }
    }
}

/// GetCreditCardResult(cardholder_name: 'str', issuer: 'str', card_number: 'str', security_code: 'str', expiration_date: 'str')
#[napi(object, js_name = "GetCreditCardResult")]
pub struct GetCreditCardResult {
    #[napi(ts_type = "string")]
    pub cardholder_name: String,
    #[napi(ts_type = "string")]
    pub issuer: String,
    #[napi(ts_type = "string")]
    pub card_number: String,
    #[napi(ts_type = "string")]
    pub security_code: String,
    #[napi(ts_type = "string")]
    pub expiration_date: String,
}

impl From<&lk_agents_abi::voice::workflows::GetCreditCardResult> for GetCreditCardResult {
    fn from(v: &lk_agents_abi::voice::workflows::GetCreditCardResult) -> Self {
        GetCreditCardResult {
            cardholder_name: v.cardholder_name.to_string(),
            issuer: v.issuer.to_string(),
            card_number: v.card_number.to_string(),
            security_code: v.security_code.to_string(),
            expiration_date: v.expiration_date.to_string(),
        }
    }
}

impl From<GetCreditCardResult> for lk_agents_abi::voice::workflows::GetCreditCardResult {
    fn from(v: GetCreditCardResult) -> Self {
        lk_agents_abi::voice::workflows::GetCreditCardResult {
            cardholder_name: v.cardholder_name,
            issuer: v.issuer,
            card_number: v.card_number,
            security_code: v.security_code,
            expiration_date: v.expiration_date,
        }
    }
}

#[napi(object, js_name = "WarmTransferResult")]
pub struct WarmTransferResult {
    #[napi(ts_type = "string")]
    pub human_agent_identity: String,
}

impl From<&lk_agents_abi::voice::workflows::WarmTransferResult> for WarmTransferResult {
    fn from(v: &lk_agents_abi::voice::workflows::WarmTransferResult) -> Self {
        WarmTransferResult {
            human_agent_identity: v.human_agent_identity.to_string(),
        }
    }
}

impl From<WarmTransferResult> for lk_agents_abi::voice::workflows::WarmTransferResult {
    fn from(v: WarmTransferResult) -> Self {
        lk_agents_abi::voice::workflows::WarmTransferResult {
            human_agent_identity: v.human_agent_identity,
        }
    }
}

#[napi(object, js_name = "VADCapabilities")]
pub struct VADCapabilities {
    #[napi(ts_type = "number")]
    pub update_interval: f64,
}

impl From<&lk_agents_abi::VadCapabilities> for VADCapabilities {
    fn from(v: &lk_agents_abi::VadCapabilities) -> Self {
        VADCapabilities {
            update_interval: v.update_interval,
        }
    }
}

impl From<VADCapabilities> for lk_agents_abi::VadCapabilities {
    fn from(v: VADCapabilities) -> Self {
        lk_agents_abi::VadCapabilities {
            update_interval: v.update_interval,
        }
    }
}

#[napi(object, js_name = "SttCartesiaOptions")]
pub struct SttCartesiaOptions {
    #[napi(ts_type = "number")]
    pub min_volume: Option<f64>,
    #[napi(ts_type = "number")]
    pub max_silence_duration_secs: Option<f64>,
    #[napi(ts_type = "number")]
    pub turn_start_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub turn_eager_end_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub turn_end_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub turn_end_timeout_ms: Option<i64>,
    #[napi(ts_type = "string | string[]")]
    pub keyterm: Option<serde_json::Value>,
}

impl From<&lk_agents_abi::inference::gateway::stt::CartesiaOptions> for SttCartesiaOptions {
    fn from(v: &lk_agents_abi::inference::gateway::stt::CartesiaOptions) -> Self {
        SttCartesiaOptions {
            min_volume: v.min_volume,
            max_silence_duration_secs: v.max_silence_duration_secs,
            turn_start_threshold: v.turn_start_threshold,
            turn_eager_end_threshold: v.turn_eager_end_threshold,
            turn_end_threshold: v.turn_end_threshold,
            turn_end_timeout_ms: v.turn_end_timeout_ms,
            keyterm: v.keyterm.as_ref().map(|v| v.clone()),
        }
    }
}

impl From<SttCartesiaOptions> for lk_agents_abi::inference::gateway::stt::CartesiaOptions {
    fn from(v: SttCartesiaOptions) -> Self {
        lk_agents_abi::inference::gateway::stt::CartesiaOptions {
            min_volume: v.min_volume,
            max_silence_duration_secs: v.max_silence_duration_secs,
            turn_start_threshold: v.turn_start_threshold,
            turn_eager_end_threshold: v.turn_eager_end_threshold,
            turn_end_threshold: v.turn_end_threshold,
            turn_end_timeout_ms: v.turn_end_timeout_ms,
            keyterm: v.keyterm,
        }
    }
}

#[napi(object, js_name = "SttDeepgramOptions")]
pub struct SttDeepgramOptions {
    #[napi(ts_type = "boolean")]
    pub filler_words: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub interim_results: Option<bool>,
    #[napi(ts_type = "number")]
    pub endpointing: Option<i64>,
    #[napi(ts_type = "boolean")]
    pub punctuate: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub smart_format: Option<bool>,
    #[napi(ts_type = "[string, number][]")]
    pub keywords: Option<serde_json::Value>,
    #[napi(ts_type = "string | string[]")]
    pub keyterm: Option<serde_json::Value>,
    #[napi(ts_type = "boolean")]
    pub profanity_filter: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub numerals: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub mip_opt_out: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub vad_events: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub diarize: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub dictation: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub detect_language: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub no_delay: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub utterance_end: Option<bool>,
    #[napi(ts_type = "string | string[]")]
    pub redact: Option<serde_json::Value>,
    #[napi(ts_type = "string | string[]")]
    pub replace: Option<serde_json::Value>,
    #[napi(ts_type = "string | string[]")]
    pub search: Option<serde_json::Value>,
    #[napi(ts_type = "string | string[]")]
    pub tag: Option<serde_json::Value>,
    #[napi(ts_type = "number")]
    pub channels: Option<i64>,
    #[napi(ts_type = "string")]
    pub version: Option<String>,
    #[napi(ts_type = "string")]
    pub callback: Option<String>,
    #[napi(ts_type = "string")]
    pub callback_method: Option<String>,
    #[napi(ts_type = "string")]
    pub extra: Option<String>,
}

impl From<&lk_agents_abi::inference::gateway::stt::DeepgramOptions> for SttDeepgramOptions {
    fn from(v: &lk_agents_abi::inference::gateway::stt::DeepgramOptions) -> Self {
        SttDeepgramOptions {
            filler_words: v.filler_words,
            interim_results: v.interim_results,
            endpointing: v.endpointing,
            punctuate: v.punctuate,
            smart_format: v.smart_format,
            keywords: v.keywords.as_ref().map(|v| v.clone()),
            keyterm: v.keyterm.as_ref().map(|v| v.clone()),
            profanity_filter: v.profanity_filter,
            numerals: v.numerals,
            mip_opt_out: v.mip_opt_out,
            vad_events: v.vad_events,
            diarize: v.diarize,
            dictation: v.dictation,
            detect_language: v.detect_language,
            no_delay: v.no_delay,
            utterance_end: v.utterance_end,
            redact: v.redact.as_ref().map(|v| v.clone()),
            replace: v.replace.as_ref().map(|v| v.clone()),
            search: v.search.as_ref().map(|v| v.clone()),
            tag: v.tag.as_ref().map(|v| v.clone()),
            channels: v.channels,
            version: v.version.as_ref().map(|v| v.to_string()),
            callback: v.callback.as_ref().map(|v| v.to_string()),
            callback_method: v.callback_method.as_ref().map(|v| v.to_string()),
            extra: v.extra.as_ref().map(|v| v.to_string()),
        }
    }
}

impl From<SttDeepgramOptions> for lk_agents_abi::inference::gateway::stt::DeepgramOptions {
    fn from(v: SttDeepgramOptions) -> Self {
        lk_agents_abi::inference::gateway::stt::DeepgramOptions {
            filler_words: v.filler_words,
            interim_results: v.interim_results,
            endpointing: v.endpointing,
            punctuate: v.punctuate,
            smart_format: v.smart_format,
            keywords: v.keywords,
            keyterm: v.keyterm,
            profanity_filter: v.profanity_filter,
            numerals: v.numerals,
            mip_opt_out: v.mip_opt_out,
            vad_events: v.vad_events,
            diarize: v.diarize,
            dictation: v.dictation,
            detect_language: v.detect_language,
            no_delay: v.no_delay,
            utterance_end: v.utterance_end,
            redact: v.redact,
            replace: v.replace,
            search: v.search,
            tag: v.tag,
            channels: v.channels,
            version: v.version,
            callback: v.callback,
            callback_method: v.callback_method,
            extra: v.extra,
        }
    }
}

#[napi(object, js_name = "DeepgramFluxOptions")]
pub struct DeepgramFluxOptions {
    #[napi(ts_type = "number")]
    pub eager_eot_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub eot_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub eot_timeout_ms: Option<i64>,
    #[napi(ts_type = "string | string[]")]
    pub keyterm: Option<serde_json::Value>,
    #[napi(ts_type = "boolean")]
    pub mip_opt_out: Option<bool>,
    #[napi(ts_type = "string | string[]")]
    pub tag: Option<serde_json::Value>,
    #[napi(ts_type = "boolean")]
    pub detect_language: Option<bool>,
}

impl From<&lk_agents_abi::inference::gateway::stt::DeepgramFluxOptions> for DeepgramFluxOptions {
    fn from(v: &lk_agents_abi::inference::gateway::stt::DeepgramFluxOptions) -> Self {
        DeepgramFluxOptions {
            eager_eot_threshold: v.eager_eot_threshold,
            eot_threshold: v.eot_threshold,
            eot_timeout_ms: v.eot_timeout_ms,
            keyterm: v.keyterm.as_ref().map(|v| v.clone()),
            mip_opt_out: v.mip_opt_out,
            tag: v.tag.as_ref().map(|v| v.clone()),
            detect_language: v.detect_language,
        }
    }
}

impl From<DeepgramFluxOptions> for lk_agents_abi::inference::gateway::stt::DeepgramFluxOptions {
    fn from(v: DeepgramFluxOptions) -> Self {
        lk_agents_abi::inference::gateway::stt::DeepgramFluxOptions {
            eager_eot_threshold: v.eager_eot_threshold,
            eot_threshold: v.eot_threshold,
            eot_timeout_ms: v.eot_timeout_ms,
            keyterm: v.keyterm,
            mip_opt_out: v.mip_opt_out,
            tag: v.tag,
            detect_language: v.detect_language,
        }
    }
}

#[napi(object, js_name = "AssemblyaiOptions")]
pub struct AssemblyaiOptions {
    #[napi(ts_type = "boolean")]
    pub format_turns: Option<bool>,
    #[napi(ts_type = "number")]
    pub end_of_turn_confidence_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub min_end_of_turn_silence_when_confident: Option<i64>,
    #[napi(ts_type = "number")]
    pub max_turn_silence: Option<i64>,
    #[napi(ts_type = "string[]")]
    pub keyterms_prompt: Option<Vec<String>>,
    #[napi(ts_type = "boolean")]
    pub language_detection: Option<bool>,
    #[napi(ts_type = "number")]
    pub inactivity_timeout: Option<f64>,
    #[napi(ts_type = "string")]
    pub prompt: Option<String>,
    #[napi(ts_type = "boolean")]
    pub speaker_labels: Option<bool>,
    #[napi(ts_type = "string")]
    pub agent_context: Option<String>,
    #[napi(ts_type = "number")]
    pub previous_context_n_turns: Option<i64>,
    #[napi(ts_type = "'near-field' | 'far-field'")]
    pub voice_focus: Option<serde_json::Value>,
    #[napi(ts_type = "number")]
    pub voice_focus_threshold: Option<f64>,
    #[napi(ts_type = "'min_latency' | 'balanced' | 'max_accuracy'")]
    pub mode: Option<serde_json::Value>,
}

impl From<&lk_agents_abi::inference::gateway::stt::AssemblyaiOptions> for AssemblyaiOptions {
    fn from(v: &lk_agents_abi::inference::gateway::stt::AssemblyaiOptions) -> Self {
        AssemblyaiOptions {
            format_turns: v.format_turns,
            end_of_turn_confidence_threshold: v.end_of_turn_confidence_threshold,
            min_end_of_turn_silence_when_confident: v.min_end_of_turn_silence_when_confident,
            max_turn_silence: v.max_turn_silence,
            keyterms_prompt: v.keyterms_prompt.as_ref().map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
            language_detection: v.language_detection,
            inactivity_timeout: v.inactivity_timeout,
            prompt: v.prompt.as_ref().map(|v| v.to_string()),
            speaker_labels: v.speaker_labels,
            agent_context: v.agent_context.as_ref().map(|v| v.to_string()),
            previous_context_n_turns: v.previous_context_n_turns,
            voice_focus: v.voice_focus.as_ref().map(|v| v.clone()),
            voice_focus_threshold: v.voice_focus_threshold,
            mode: v.mode.as_ref().map(|v| v.clone()),
        }
    }
}

impl From<AssemblyaiOptions> for lk_agents_abi::inference::gateway::stt::AssemblyaiOptions {
    fn from(v: AssemblyaiOptions) -> Self {
        lk_agents_abi::inference::gateway::stt::AssemblyaiOptions {
            format_turns: v.format_turns,
            end_of_turn_confidence_threshold: v.end_of_turn_confidence_threshold,
            min_end_of_turn_silence_when_confident: v.min_end_of_turn_silence_when_confident,
            max_turn_silence: v.max_turn_silence,
            keyterms_prompt: v.keyterms_prompt,
            language_detection: v.language_detection,
            inactivity_timeout: v.inactivity_timeout,
            prompt: v.prompt,
            speaker_labels: v.speaker_labels,
            agent_context: v.agent_context,
            previous_context_n_turns: v.previous_context_n_turns,
            voice_focus: v.voice_focus,
            voice_focus_threshold: v.voice_focus_threshold,
            mode: v.mode,
        }
    }
}

#[napi(object, js_name = "SpeechmaticsOptions")]
pub struct SpeechmaticsOptions {
    #[napi(ts_type = "string")]
    pub domain: Option<String>,
    #[napi(ts_type = "string")]
    pub output_locale: Option<String>,
    #[napi(ts_type = "number")]
    pub max_delay: Option<f64>,
    #[napi(ts_type = "string")]
    pub max_delay_mode: Option<String>,
    #[napi(ts_type = "string")]
    pub diarization: Option<String>,
    #[napi(ts_type = "number")]
    pub speaker_sensitivity: Option<f64>,
    #[napi(ts_type = "number")]
    pub max_speakers: Option<i64>,
    #[napi(ts_type = "boolean")]
    pub prefer_current_speaker: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub enable_partials: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub enable_entities: Option<bool>,
    #[napi(ts_type = "Record<string, unknown>")]
    pub punctuation_overrides: Option<std::collections::HashMap<String, serde_json::Value>>,
    #[napi(ts_type = "Record<string, unknown>[]")]
    pub additional_vocab: Option<Vec<std::collections::HashMap<String, serde_json::Value>>>,
    #[napi(ts_type = "number")]
    pub end_of_utterance_silence_trigger: Option<f64>,
    #[napi(ts_type = "Record<string, unknown>")]
    pub audio_filtering_config: Option<std::collections::HashMap<String, serde_json::Value>>,
    #[napi(ts_type = "Record<string, unknown>")]
    pub transcript_filtering_config: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl From<&lk_agents_abi::inference::gateway::stt::SpeechmaticsOptions> for SpeechmaticsOptions {
    fn from(v: &lk_agents_abi::inference::gateway::stt::SpeechmaticsOptions) -> Self {
        SpeechmaticsOptions {
            domain: v.domain.as_ref().map(|v| v.to_string()),
            output_locale: v.output_locale.as_ref().map(|v| v.to_string()),
            max_delay: v.max_delay,
            max_delay_mode: v.max_delay_mode.as_ref().map(|v| v.to_string()),
            diarization: v.diarization.as_ref().map(|v| v.to_string()),
            speaker_sensitivity: v.speaker_sensitivity,
            max_speakers: v.max_speakers,
            prefer_current_speaker: v.prefer_current_speaker,
            enable_partials: v.enable_partials,
            enable_entities: v.enable_entities,
            punctuation_overrides: v.punctuation_overrides.as_ref().map(|v| v.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<std::collections::HashMap<_, _>>()),
            additional_vocab: v.additional_vocab.as_ref().map(|v| v.iter().map(|v| v.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<std::collections::HashMap<_, _>>()).collect::<Vec<_>>()),
            end_of_utterance_silence_trigger: v.end_of_utterance_silence_trigger,
            audio_filtering_config: v.audio_filtering_config.as_ref().map(|v| v.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<std::collections::HashMap<_, _>>()),
            transcript_filtering_config: v.transcript_filtering_config.as_ref().map(|v| v.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<std::collections::HashMap<_, _>>()),
        }
    }
}

impl From<SpeechmaticsOptions> for lk_agents_abi::inference::gateway::stt::SpeechmaticsOptions {
    fn from(v: SpeechmaticsOptions) -> Self {
        lk_agents_abi::inference::gateway::stt::SpeechmaticsOptions {
            domain: v.domain,
            output_locale: v.output_locale,
            max_delay: v.max_delay,
            max_delay_mode: v.max_delay_mode,
            diarization: v.diarization,
            speaker_sensitivity: v.speaker_sensitivity,
            max_speakers: v.max_speakers,
            prefer_current_speaker: v.prefer_current_speaker,
            enable_partials: v.enable_partials,
            enable_entities: v.enable_entities,
            punctuation_overrides: v.punctuation_overrides.map(|v| v.into_iter().map(|(k, v)| (k, v)).collect()),
            additional_vocab: v.additional_vocab.map(|v| v.into_iter().map(|v| v.into_iter().map(|(k, v)| (k, v)).collect()).collect::<Vec<_>>()),
            end_of_utterance_silence_trigger: v.end_of_utterance_silence_trigger,
            audio_filtering_config: v.audio_filtering_config.map(|v| v.into_iter().map(|(k, v)| (k, v)).collect()),
            transcript_filtering_config: v.transcript_filtering_config.map(|v| v.into_iter().map(|(k, v)| (k, v)).collect()),
        }
    }
}

#[napi(object, js_name = "SttXaiOptions")]
pub struct SttXaiOptions {
    #[napi(ts_type = "boolean")]
    pub diarize: Option<bool>,
    #[napi(ts_type = "number")]
    pub endpointing: Option<i64>,
    #[napi(ts_type = "boolean")]
    pub format: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub interim_results: Option<bool>,
}

impl From<&lk_agents_abi::inference::gateway::stt::XaiOptions> for SttXaiOptions {
    fn from(v: &lk_agents_abi::inference::gateway::stt::XaiOptions) -> Self {
        SttXaiOptions {
            diarize: v.diarize,
            endpointing: v.endpointing,
            format: v.format,
            interim_results: v.interim_results,
        }
    }
}

impl From<SttXaiOptions> for lk_agents_abi::inference::gateway::stt::XaiOptions {
    fn from(v: SttXaiOptions) -> Self {
        lk_agents_abi::inference::gateway::stt::XaiOptions {
            diarize: v.diarize,
            endpointing: v.endpointing,
            format: v.format,
            interim_results: v.interim_results,
        }
    }
}

#[napi(object, js_name = "SttInworldOptions")]
pub struct SttInworldOptions {
    #[napi(ts_type = "boolean")]
    pub enable_voice_profile: Option<bool>,
    #[napi(ts_type = "number")]
    pub voice_profile_top_n: Option<i64>,
    #[napi(ts_type = "boolean")]
    pub include_word_timestamps: Option<bool>,
    #[napi(ts_type = "'LINEAR16' | 'AUTO_DETECT'")]
    pub audio_encoding: Option<serde_json::Value>,
    #[napi(ts_type = "number")]
    pub inactivity_timeout_seconds: Option<i64>,
    #[napi(ts_type = "number")]
    pub end_of_turn_confidence_threshold: Option<f64>,
    #[napi(ts_type = "number")]
    pub min_end_of_turn_silence_when_confident: Option<i64>,
    #[napi(ts_type = "string[]")]
    pub prompts: Option<Vec<String>>,
    #[napi(ts_type = "number")]
    pub vad_threshold: Option<f64>,
}

impl From<&lk_agents_abi::inference::gateway::stt::InworldOptions> for SttInworldOptions {
    fn from(v: &lk_agents_abi::inference::gateway::stt::InworldOptions) -> Self {
        SttInworldOptions {
            enable_voice_profile: v.enable_voice_profile,
            voice_profile_top_n: v.voice_profile_top_n,
            include_word_timestamps: v.include_word_timestamps,
            audio_encoding: v.audio_encoding.as_ref().map(|v| v.clone()),
            inactivity_timeout_seconds: v.inactivity_timeout_seconds,
            end_of_turn_confidence_threshold: v.end_of_turn_confidence_threshold,
            min_end_of_turn_silence_when_confident: v.min_end_of_turn_silence_when_confident,
            prompts: v.prompts.as_ref().map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
            vad_threshold: v.vad_threshold,
        }
    }
}

impl From<SttInworldOptions> for lk_agents_abi::inference::gateway::stt::InworldOptions {
    fn from(v: SttInworldOptions) -> Self {
        lk_agents_abi::inference::gateway::stt::InworldOptions {
            enable_voice_profile: v.enable_voice_profile,
            voice_profile_top_n: v.voice_profile_top_n,
            include_word_timestamps: v.include_word_timestamps,
            audio_encoding: v.audio_encoding,
            inactivity_timeout_seconds: v.inactivity_timeout_seconds,
            end_of_turn_confidence_threshold: v.end_of_turn_confidence_threshold,
            min_end_of_turn_silence_when_confident: v.min_end_of_turn_silence_when_confident,
            prompts: v.prompts,
            vad_threshold: v.vad_threshold,
        }
    }
}

#[napi(object, js_name = "GoogleOptions")]
pub struct GoogleOptions {
    #[napi(ts_type = "string[]")]
    pub language_codes: Option<Vec<String>>,
    #[napi(ts_type = "string[]")]
    pub custom_vocabulary: Option<Vec<String>>,
}

impl From<&lk_agents_abi::inference::gateway::stt::GoogleOptions> for GoogleOptions {
    fn from(v: &lk_agents_abi::inference::gateway::stt::GoogleOptions) -> Self {
        GoogleOptions {
            language_codes: v.language_codes.as_ref().map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
            custom_vocabulary: v.custom_vocabulary.as_ref().map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
        }
    }
}

impl From<GoogleOptions> for lk_agents_abi::inference::gateway::stt::GoogleOptions {
    fn from(v: GoogleOptions) -> Self {
        lk_agents_abi::inference::gateway::stt::GoogleOptions {
            language_codes: v.language_codes,
            custom_vocabulary: v.custom_vocabulary,
        }
    }
}

#[napi(object, js_name = "TtsCartesiaOptions")]
pub struct TtsCartesiaOptions {
    #[napi(ts_type = "string")]
    pub emotion: Option<String>,
    #[napi(ts_type = "'slow' | 'normal' | 'fast' | number")]
    pub speed: Option<serde_json::Value>,
    #[napi(ts_type = "number")]
    pub volume: Option<f64>,
    #[napi(ts_type = "number")]
    pub duration: Option<f64>,
    #[napi(ts_type = "number")]
    pub max_buffer_delay_ms: Option<i64>,
    #[napi(ts_type = "boolean")]
    pub add_timestamps: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub add_phoneme_timestamps: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub use_normalized_timestamps: Option<bool>,
}

impl From<&lk_agents_abi::inference::gateway::tts::CartesiaOptions> for TtsCartesiaOptions {
    fn from(v: &lk_agents_abi::inference::gateway::tts::CartesiaOptions) -> Self {
        TtsCartesiaOptions {
            emotion: v.emotion.as_ref().map(|v| v.to_string()),
            speed: v.speed.as_ref().map(|v| v.clone()),
            volume: v.volume,
            duration: v.duration,
            max_buffer_delay_ms: v.max_buffer_delay_ms,
            add_timestamps: v.add_timestamps,
            add_phoneme_timestamps: v.add_phoneme_timestamps,
            use_normalized_timestamps: v.use_normalized_timestamps,
        }
    }
}

impl From<TtsCartesiaOptions> for lk_agents_abi::inference::gateway::tts::CartesiaOptions {
    fn from(v: TtsCartesiaOptions) -> Self {
        lk_agents_abi::inference::gateway::tts::CartesiaOptions {
            emotion: v.emotion,
            speed: v.speed,
            volume: v.volume,
            duration: v.duration,
            max_buffer_delay_ms: v.max_buffer_delay_ms,
            add_timestamps: v.add_timestamps,
            add_phoneme_timestamps: v.add_phoneme_timestamps,
            use_normalized_timestamps: v.use_normalized_timestamps,
        }
    }
}

#[napi(object, js_name = "TtsDeepgramOptions")]
pub struct TtsDeepgramOptions {
    #[napi(ts_type = "boolean")]
    pub mip_opt_out: Option<bool>,
}

impl From<&lk_agents_abi::inference::gateway::tts::DeepgramOptions> for TtsDeepgramOptions {
    fn from(v: &lk_agents_abi::inference::gateway::tts::DeepgramOptions) -> Self {
        TtsDeepgramOptions {
            mip_opt_out: v.mip_opt_out,
        }
    }
}

impl From<TtsDeepgramOptions> for lk_agents_abi::inference::gateway::tts::DeepgramOptions {
    fn from(v: TtsDeepgramOptions) -> Self {
        lk_agents_abi::inference::gateway::tts::DeepgramOptions {
            mip_opt_out: v.mip_opt_out,
        }
    }
}

/// Rime-specific WebSocket JSON query parameters.
/// See: https://docs.rime.ai/api-reference/endpoint/websockets-json
#[napi(object, js_name = "RimeOptions")]
pub struct RimeOptions {
    #[napi(ts_type = "number")]
    pub speed_alpha: Option<f64>,
    #[napi(ts_type = "boolean")]
    pub pause_between_brackets: Option<bool>,
    #[napi(ts_type = "boolean")]
    pub phonemize_between_brackets: Option<bool>,
    #[napi(ts_type = "string")]
    pub inline_speed_alpha: Option<String>,
    #[napi(ts_type = "boolean")]
    pub no_text_normalization: Option<bool>,
}

impl From<&lk_agents_abi::inference::gateway::tts::RimeOptions> for RimeOptions {
    fn from(v: &lk_agents_abi::inference::gateway::tts::RimeOptions) -> Self {
        RimeOptions {
            speed_alpha: v.speed_alpha,
            pause_between_brackets: v.pause_between_brackets,
            phonemize_between_brackets: v.phonemize_between_brackets,
            inline_speed_alpha: v.inline_speed_alpha.as_ref().map(|v| v.to_string()),
            no_text_normalization: v.no_text_normalization,
        }
    }
}

impl From<RimeOptions> for lk_agents_abi::inference::gateway::tts::RimeOptions {
    fn from(v: RimeOptions) -> Self {
        lk_agents_abi::inference::gateway::tts::RimeOptions {
            speed_alpha: v.speed_alpha,
            pause_between_brackets: v.pause_between_brackets,
            phonemize_between_brackets: v.phonemize_between_brackets,
            inline_speed_alpha: v.inline_speed_alpha,
            no_text_normalization: v.no_text_normalization,
        }
    }
}

#[napi(object, js_name = "TtsInworldOptions")]
pub struct TtsInworldOptions {
    #[napi(ts_type = "number")]
    pub speaking_rate: Option<f64>,
    #[napi(ts_type = "number")]
    pub temperature: Option<f64>,
    #[napi(ts_type = "'DELIVERY_MODE_UNSPECIFIED' | 'STABLE' | 'BALANCED' | 'CREATIVE'")]
    pub delivery_mode: Option<serde_json::Value>,
    #[napi(ts_type = "'TIMESTAMP_TYPE_UNSPECIFIED' | 'WORD' | 'CHARACTER'")]
    pub timestamp_type: Option<serde_json::Value>,
    #[napi(ts_type = "'APPLY_TEXT_NORMALIZATION_UNSPECIFIED' | 'ON' | 'OFF'")]
    pub apply_text_normalization: Option<serde_json::Value>,
}

impl From<&lk_agents_abi::inference::gateway::tts::InworldOptions> for TtsInworldOptions {
    fn from(v: &lk_agents_abi::inference::gateway::tts::InworldOptions) -> Self {
        TtsInworldOptions {
            speaking_rate: v.speaking_rate,
            temperature: v.temperature,
            delivery_mode: v.delivery_mode.as_ref().map(|v| v.clone()),
            timestamp_type: v.timestamp_type.as_ref().map(|v| v.clone()),
            apply_text_normalization: v.apply_text_normalization.as_ref().map(|v| v.clone()),
        }
    }
}

impl From<TtsInworldOptions> for lk_agents_abi::inference::gateway::tts::InworldOptions {
    fn from(v: TtsInworldOptions) -> Self {
        lk_agents_abi::inference::gateway::tts::InworldOptions {
            speaking_rate: v.speaking_rate,
            temperature: v.temperature,
            delivery_mode: v.delivery_mode,
            timestamp_type: v.timestamp_type,
            apply_text_normalization: v.apply_text_normalization,
        }
    }
}

#[napi(object, js_name = "TtsXaiOptions")]
pub struct TtsXaiOptions {
    #[napi(ts_type = "32000 | 64000 | 96000 | 128000 | 192000")]
    pub bit_rate: Option<serde_json::Value>,
    #[napi(ts_type = "number")]
    pub speed: Option<f64>,
    #[napi(ts_type = "0 | 1 | 2")]
    pub optimize_streaming_latency: Option<serde_json::Value>,
}

impl From<&lk_agents_abi::inference::gateway::tts::XaiOptions> for TtsXaiOptions {
    fn from(v: &lk_agents_abi::inference::gateway::tts::XaiOptions) -> Self {
        TtsXaiOptions {
            bit_rate: v.bit_rate.as_ref().map(|v| v.clone()),
            speed: v.speed,
            optimize_streaming_latency: v.optimize_streaming_latency.as_ref().map(|v| v.clone()),
        }
    }
}

impl From<TtsXaiOptions> for lk_agents_abi::inference::gateway::tts::XaiOptions {
    fn from(v: TtsXaiOptions) -> Self {
        lk_agents_abi::inference::gateway::tts::XaiOptions {
            bit_rate: v.bit_rate,
            speed: v.speed,
            optimize_streaming_latency: v.optimize_streaming_latency,
        }
    }
}

/// See https://docs.fish.audio/api-reference/endpoint/websocket/tts-live
#[napi(object, js_name = "FishAudioOptions")]
pub struct FishAudioOptions {
    #[napi(ts_type = "number")]
    pub speed: Option<f64>,
    #[napi(ts_type = "number")]
    pub volume: Option<f64>,
    #[napi(ts_type = "boolean")]
    pub normalize_loudness: Option<bool>,
    #[napi(ts_type = "number")]
    pub temperature: Option<f64>,
    #[napi(ts_type = "number")]
    pub top_p: Option<f64>,
    #[napi(ts_type = "boolean")]
    pub normalize: Option<bool>,
    #[napi(ts_type = "'normal' | 'balanced' | 'low'")]
    pub latency: Option<serde_json::Value>,
    #[napi(ts_type = "number")]
    pub chunk_length: Option<i64>,
    #[napi(ts_type = "number")]
    pub max_new_tokens: Option<i64>,
    #[napi(ts_type = "number")]
    pub min_chunk_length: Option<i64>,
    #[napi(ts_type = "boolean")]
    pub condition_on_previous_chunks: Option<bool>,
    #[napi(ts_type = "number")]
    pub early_stop_threshold: Option<f64>,
}

impl From<&lk_agents_abi::inference::gateway::tts::FishAudioOptions> for FishAudioOptions {
    fn from(v: &lk_agents_abi::inference::gateway::tts::FishAudioOptions) -> Self {
        FishAudioOptions {
            speed: v.speed,
            volume: v.volume,
            normalize_loudness: v.normalize_loudness,
            temperature: v.temperature,
            top_p: v.top_p,
            normalize: v.normalize,
            latency: v.latency.as_ref().map(|v| v.clone()),
            chunk_length: v.chunk_length,
            max_new_tokens: v.max_new_tokens,
            min_chunk_length: v.min_chunk_length,
            condition_on_previous_chunks: v.condition_on_previous_chunks,
            early_stop_threshold: v.early_stop_threshold,
        }
    }
}

impl From<FishAudioOptions> for lk_agents_abi::inference::gateway::tts::FishAudioOptions {
    fn from(v: FishAudioOptions) -> Self {
        lk_agents_abi::inference::gateway::tts::FishAudioOptions {
            speed: v.speed,
            volume: v.volume,
            normalize_loudness: v.normalize_loudness,
            temperature: v.temperature,
            top_p: v.top_p,
            normalize: v.normalize,
            latency: v.latency,
            chunk_length: v.chunk_length,
            max_new_tokens: v.max_new_tokens,
            min_chunk_length: v.min_chunk_length,
            condition_on_previous_chunks: v.condition_on_previous_chunks,
            early_stop_threshold: v.early_stop_threshold,
        }
    }
}

#[napi(object, js_name = "ChatCompletionOptions")]
pub struct ChatCompletionOptions {
    #[napi(ts_type = "number | null")]
    pub frequency_penalty: Option<Option<f64>>,
    #[napi(ts_type = "Record<string, number> | null")]
    pub logit_bias: Option<Option<std::collections::HashMap<String, i64>>>,
    #[napi(ts_type = "boolean | null")]
    pub logprobs: Option<Option<bool>>,
    #[napi(ts_type = "number | null")]
    pub max_completion_tokens: Option<Option<i64>>,
    #[napi(ts_type = "number | null")]
    pub max_tokens: Option<Option<i64>>,
    #[napi(ts_type = "Record<string, string> | null")]
    pub metadata: Option<serde_json::Value>,
    #[napi(ts_type = "('text' | 'audio')[] | null")]
    pub modalities: Option<serde_json::Value>,
    #[napi(ts_type = "number | null")]
    pub n: Option<Option<i64>>,
    #[napi(ts_type = "boolean")]
    pub parallel_tool_calls: Option<bool>,
    #[napi(ts_type = "unknown")]
    pub prediction: Option<serde_json::Value>,
    #[napi(ts_type = "number | null")]
    pub presence_penalty: Option<Option<f64>>,
    #[napi(ts_type = "string")]
    pub prompt_cache_key: Option<String>,
    #[napi(ts_type = "'in_memory' | '24h' | null")]
    pub prompt_cache_retention: Option<serde_json::Value>,
    #[napi(ts_type = "unknown")]
    pub reasoning_effort: Option<serde_json::Value>,
    #[napi(ts_type = "string")]
    pub safety_identifier: Option<String>,
    #[napi(ts_type = "number | null")]
    pub seed: Option<Option<i64>>,
    #[napi(ts_type = "'auto' | 'default' | 'flex' | 'scale' | 'priority' | null")]
    pub service_tier: Option<serde_json::Value>,
    #[napi(ts_type = "string | null | string[]")]
    pub stop: Option<serde_json::Value>,
    #[napi(ts_type = "boolean | null")]
    pub store: Option<Option<bool>>,
    #[napi(ts_type = "number | null")]
    pub temperature: Option<Option<f64>>,
    #[napi(ts_type = "number | null")]
    pub top_logprobs: Option<Option<i64>>,
    #[napi(ts_type = "number | null")]
    pub top_p: Option<Option<f64>>,
    #[napi(ts_type = "string")]
    pub user: Option<String>,
    #[napi(ts_type = "'low' | 'medium' | 'high' | null")]
    pub verbosity: Option<serde_json::Value>,
    #[napi(ts_type = "unknown")]
    pub web_search_options: Option<serde_json::Value>,
    #[napi(ts_type = "unknown")]
    pub tool_choice: Option<serde_json::Value>,
}

impl From<&lk_agents_abi::inference::gateway::llm::ChatCompletionOptions> for ChatCompletionOptions {
    fn from(v: &lk_agents_abi::inference::gateway::llm::ChatCompletionOptions) -> Self {
        ChatCompletionOptions {
            frequency_penalty: v.frequency_penalty,
            logit_bias: v.logit_bias.as_ref().map(|v| v.as_ref().map(|v| v.iter().map(|(k, v)| (k.clone(), *v)).collect::<std::collections::HashMap<_, _>>())),
            logprobs: v.logprobs,
            max_completion_tokens: v.max_completion_tokens,
            max_tokens: v.max_tokens,
            metadata: v.metadata.as_ref().map(|v| v.clone()),
            modalities: v.modalities.as_ref().map(|v| v.clone()),
            n: v.n,
            parallel_tool_calls: v.parallel_tool_calls,
            prediction: v.prediction.as_ref().map(|v| v.clone()),
            presence_penalty: v.presence_penalty,
            prompt_cache_key: v.prompt_cache_key.as_ref().map(|v| v.to_string()),
            prompt_cache_retention: v.prompt_cache_retention.as_ref().map(|v| v.clone()),
            reasoning_effort: v.reasoning_effort.as_ref().map(|v| v.clone()),
            safety_identifier: v.safety_identifier.as_ref().map(|v| v.to_string()),
            seed: v.seed,
            service_tier: v.service_tier.as_ref().map(|v| v.clone()),
            stop: v.stop.as_ref().map(|v| v.clone()),
            store: v.store,
            temperature: v.temperature,
            top_logprobs: v.top_logprobs,
            top_p: v.top_p,
            user: v.user.as_ref().map(|v| v.to_string()),
            verbosity: v.verbosity.as_ref().map(|v| v.clone()),
            web_search_options: v.web_search_options.as_ref().map(|v| v.clone()),
            tool_choice: v.tool_choice.as_ref().map(|v| v.clone()),
        }
    }
}

impl From<ChatCompletionOptions> for lk_agents_abi::inference::gateway::llm::ChatCompletionOptions {
    fn from(v: ChatCompletionOptions) -> Self {
        lk_agents_abi::inference::gateway::llm::ChatCompletionOptions {
            frequency_penalty: v.frequency_penalty,
            logit_bias: v.logit_bias.map(|v| v.map(|v| v.into_iter().map(|(k, v)| (k, v)).collect())),
            logprobs: v.logprobs,
            max_completion_tokens: v.max_completion_tokens,
            max_tokens: v.max_tokens,
            metadata: v.metadata,
            modalities: v.modalities,
            n: v.n,
            parallel_tool_calls: v.parallel_tool_calls,
            prediction: v.prediction,
            presence_penalty: v.presence_penalty,
            prompt_cache_key: v.prompt_cache_key,
            prompt_cache_retention: v.prompt_cache_retention,
            reasoning_effort: v.reasoning_effort,
            safety_identifier: v.safety_identifier,
            seed: v.seed,
            service_tier: v.service_tier,
            stop: v.stop,
            store: v.store,
            temperature: v.temperature,
            top_logprobs: v.top_logprobs,
            top_p: v.top_p,
            user: v.user,
            verbosity: v.verbosity,
            web_search_options: v.web_search_options,
            tool_choice: v.tool_choice,
        }
    }
}

/// Provider-specific options for a LemonSlice inference avatar.
#[napi(object, js_name = "LemonSliceOptions")]
pub struct LemonSliceOptions {
    #[napi(ts_type = "string")]
    pub image_url: Option<String>,
    #[napi(ts_type = "string")]
    pub prompt: Option<String>,
    #[napi(ts_type = "string")]
    pub idle_prompt: Option<String>,
    #[napi(ts_type = "number")]
    pub idle_timeout: Option<i64>,
}

impl From<&lk_agents_abi::inference::gateway::avatar::LemonSliceOptions> for LemonSliceOptions {
    fn from(v: &lk_agents_abi::inference::gateway::avatar::LemonSliceOptions) -> Self {
        LemonSliceOptions {
            image_url: v.image_url.as_ref().map(|v| v.to_string()),
            prompt: v.prompt.as_ref().map(|v| v.to_string()),
            idle_prompt: v.idle_prompt.as_ref().map(|v| v.to_string()),
            idle_timeout: v.idle_timeout,
        }
    }
}

impl From<LemonSliceOptions> for lk_agents_abi::inference::gateway::avatar::LemonSliceOptions {
    fn from(v: LemonSliceOptions) -> Self {
        lk_agents_abi::inference::gateway::avatar::LemonSliceOptions {
            image_url: v.image_url,
            prompt: v.prompt,
            idle_prompt: v.idle_prompt,
            idle_timeout: v.idle_timeout,
        }
    }
}

/// An expressive markup tag stripped from a transcript, surfaced for the frontend.
#[napi(object, js_name = "ExpressiveTag")]
pub struct ExpressiveTag {
    /// The markup tag name (`"emotion"`, `"sound"`, ...) or the expr marker type.
    pub r#type: String,
    /// The payload: the `value="..."` attribute, the expr `label`, or the tag's inner text.
    pub value: String,
}

impl From<&lk_agents_abi::tts::markup::ExpressiveTag> for ExpressiveTag {
    fn from(v: &lk_agents_abi::tts::markup::ExpressiveTag) -> Self {
        ExpressiveTag {
            r#type: v.kind.to_string(),
            value: v.value.to_string(),
        }
    }
}

impl From<ExpressiveTag> for lk_agents_abi::tts::markup::ExpressiveTag {
    fn from(v: ExpressiveTag) -> Self {
        lk_agents_abi::tts::markup::ExpressiveTag {
            kind: v.r#type,
            value: v.value,
        }
    }
}

/// The detector's constants, as the thirteen keys the binding's `amd_constants` answers with.
#[napi(object, js_name = "AmdConstants")]
pub struct AmdConstants {
    pub default_llm_model: String,
    pub default_stt_model: String,
    pub prompt: String,
    pub evaluated_llm_models: Vec<String>,
    pub evaluated_stt_models: Vec<String>,
    pub human_speech_threshold: f64,
    pub human_silence_threshold: f64,
    pub machine_silence_threshold: f64,
    pub no_speech_threshold: f64,
    pub detection_timeout: f64,
    pub max_endpointing_delay: f64,
    /// The extension budget: `MAX_EXTENSIONS` requests, each clamped to `MAX_EXTENSION_SECS`.
    pub max_extensions: u32,
    pub max_extension_secs: f64,
}

impl From<&lk_agents_abi::voice::amd::AmdConstants> for AmdConstants {
    fn from(v: &lk_agents_abi::voice::amd::AmdConstants) -> Self {
        AmdConstants {
            default_llm_model: v.default_llm_model.to_string(),
            default_stt_model: v.default_stt_model.to_string(),
            prompt: v.prompt.to_string(),
            evaluated_llm_models: v.evaluated_llm_models.iter().map(|v| v.to_string()).collect::<Vec<_>>(),
            evaluated_stt_models: v.evaluated_stt_models.iter().map(|v| v.to_string()).collect::<Vec<_>>(),
            human_speech_threshold: v.human_speech_threshold,
            human_silence_threshold: v.human_silence_threshold,
            machine_silence_threshold: v.machine_silence_threshold,
            no_speech_threshold: v.no_speech_threshold,
            detection_timeout: v.detection_timeout,
            max_endpointing_delay: v.max_endpointing_delay,
            max_extensions: v.max_extensions,
            max_extension_secs: v.max_extension_secs,
        }
    }
}

impl From<AmdConstants> for lk_agents_abi::voice::amd::AmdConstants {
    fn from(v: AmdConstants) -> Self {
        lk_agents_abi::voice::amd::AmdConstants {
            default_llm_model: v.default_llm_model,
            default_stt_model: v.default_stt_model,
            prompt: v.prompt,
            evaluated_llm_models: v.evaluated_llm_models,
            evaluated_stt_models: v.evaluated_stt_models,
            human_speech_threshold: v.human_speech_threshold,
            human_silence_threshold: v.human_silence_threshold,
            machine_silence_threshold: v.machine_silence_threshold,
            no_speech_threshold: v.no_speech_threshold,
            detection_timeout: v.detection_timeout,
            max_endpointing_delay: v.max_endpointing_delay,
            max_extensions: v.max_extensions,
            max_extension_secs: v.max_extension_secs,
        }
    }
}

#[napi(object, js_name = "MetricsReport")]
pub struct MetricsReport {
    pub started_speaking_at: Option<f64>,
    pub stopped_speaking_at: Option<f64>,
    /// Time taken to obtain the transcript after the end of the user's speech
    pub transcription_delay: Option<f64>,
    /// Amount of time between the end of speech and the decision to end the user's turn
    pub end_of_turn_delay: Option<f64>,
    /// Time taken to invoke the developer's `Agent.on_user_turn_completed` callback.
    pub on_user_turn_completed_delay: Option<f64>,
    /// Time taken for the `llm_node` to return the first token
    pub llm_node_ttft: Option<f64>,
    /// LLM output tokens per second for this turn, measured at the `llm_node` over the
    /// streaming window (first to last text chunk). Absent for a reply that arrived in a
    /// single chunk, which has no measurable rate
    pub llm_node_tps: Option<f64>,
    /// Time from LLM generation start until the first sentence reached the TTS provider, as
    /// segmented by that TTS. Absent when no audio came from a LiveKit TTS this turn: no TTS,
    /// an interruption before the first frame, or a `tts_node` synthesizing audio on its own
    pub llm_node_ttfs: Option<f64>,
    /// Time taken for the `tts_node` to return the first chunk of audio (after the first text token has been sent)
    pub tts_node_ttfb: Option<f64>,
    /// Delay between forwarding the first audio frame and the `AudioOutput` reporting
    /// playback started. Near-zero for the default room output (self-reported when the frame
    /// is pushed to the track, so it doesn't account for network delivery to the client);
    /// meaningful when a remote avatar worker is in the chain and reports playback via
    /// the `lk.playback_started` RPC.
    pub playback_latency: Option<f64>,
    /// Time from when the user finished speaking to when the agent began responding
    #[napi(js_name = "e2eLatency")]
    pub e2e_latency: Option<f64>,
    /// Provider-known request or response IDs associated with this turn.
    pub provider_request_ids: Option<Vec<String>>,
    pub llm_metadata: Option<crate::generated::records::MetricsMetadata>,
    pub tts_metadata: Option<crate::generated::records::MetricsMetadata>,
    pub stt_metadata: Option<crate::generated::records::MetricsMetadata>,
}

impl From<&lk_agents_abi::MetricsReport> for MetricsReport {
    fn from(v: &lk_agents_abi::MetricsReport) -> Self {
        MetricsReport {
            started_speaking_at: v.started_speaking_at,
            stopped_speaking_at: v.stopped_speaking_at,
            transcription_delay: v.transcription_delay,
            end_of_turn_delay: v.end_of_turn_delay,
            on_user_turn_completed_delay: v.on_user_turn_completed_delay,
            llm_node_ttft: v.llm_node_ttft,
            llm_node_tps: v.llm_node_tps,
            llm_node_ttfs: v.llm_node_ttfs,
            tts_node_ttfb: v.tts_node_ttfb,
            playback_latency: v.playback_latency,
            e2e_latency: v.e2e_latency,
            provider_request_ids: v.provider_request_ids.as_ref().map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
            llm_metadata: v.llm_metadata.as_ref().map(|v| crate::generated::records::MetricsMetadata::from(v)),
            tts_metadata: v.tts_metadata.as_ref().map(|v| crate::generated::records::MetricsMetadata::from(v)),
            stt_metadata: v.stt_metadata.as_ref().map(|v| crate::generated::records::MetricsMetadata::from(v)),
        }
    }
}

impl TryFrom<MetricsReport> for lk_agents_abi::MetricsReport {
    type Error = crate::host_error::JsError;

    fn try_from(v: MetricsReport) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::MetricsReport {
            started_speaking_at: v.started_speaking_at,
            stopped_speaking_at: v.stopped_speaking_at,
            transcription_delay: v.transcription_delay,
            end_of_turn_delay: v.end_of_turn_delay,
            on_user_turn_completed_delay: v.on_user_turn_completed_delay,
            llm_node_ttft: v.llm_node_ttft,
            llm_node_tps: v.llm_node_tps,
            llm_node_ttfs: v.llm_node_ttfs,
            tts_node_ttfb: v.tts_node_ttfb,
            playback_latency: v.playback_latency,
            e2e_latency: v.e2e_latency,
            provider_request_ids: v.provider_request_ids,
            llm_metadata: v.llm_metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
            tts_metadata: v.tts_metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
            stt_metadata: v.stt_metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "TtsCapabilities")]
pub struct TtsCapabilities {
    /// Whether this TTS supports streaming (generally using websockets)
    pub streaming: bool,
    /// Whether this TTS supports aligned transcripts with word timestamps
    pub aligned_transcript: bool,
}

impl From<&lk_agents_abi::TtsCapabilities> for TtsCapabilities {
    fn from(v: &lk_agents_abi::TtsCapabilities) -> Self {
        TtsCapabilities {
            streaming: v.streaming,
            aligned_transcript: v.aligned_transcript,
        }
    }
}

impl From<TtsCapabilities> for lk_agents_abi::TtsCapabilities {
    fn from(v: TtsCapabilities) -> Self {
        lk_agents_abi::TtsCapabilities {
            streaming: v.streaming,
            aligned_transcript: v.aligned_transcript,
        }
    }
}

#[napi(object, js_name = "RealtimeCapabilities")]
pub struct RealtimeCapabilities {
    /// Whether generated assistant messages can be truncated after interruption
    pub message_truncation: bool,
    /// Whether the model emits server-side speech start and stop events for turn taking
    pub turn_detection: bool,
    /// Whether the model emits user audio transcription events
    pub user_transcription: bool,
    /// Whether the model automatically generates a reply after receiving tool results
    pub auto_tool_reply_generation: bool,
    /// Whether the model can produce audio output directly
    pub audio_output: bool,
    /// Whether function call items already in the chat context can be resumed
    pub manual_function_calls: bool,
    /// Whether server-side turn detection can be disabled so the client drives turn-taking.
    pub can_disable_turn_detection: bool,
    /// Whether the chat context can be updated mid-session
    pub mutable_chat_context: bool,
    /// Whether the instructions can be updated mid-session
    pub mutable_instructions: bool,
    /// Whether the tools can be updated mid-session
    pub mutable_tools: bool,
    /// Whether the tool and tool choice can be specified per response
    pub per_response_tool_choice: bool,
    /// Whether session.say() can use the realtime session directly, without TTS.
    pub supports_say: bool,
    /// Whether the server generates the reply by itself when its turn detection ends a user turn.
    pub auto_turn_reply_generation: bool,
}

impl From<&lk_agents_abi::llm::realtime::RealtimeCapabilities> for RealtimeCapabilities {
    fn from(v: &lk_agents_abi::llm::realtime::RealtimeCapabilities) -> Self {
        RealtimeCapabilities {
            message_truncation: v.message_truncation,
            turn_detection: v.turn_detection,
            user_transcription: v.user_transcription,
            auto_tool_reply_generation: v.auto_tool_reply_generation,
            audio_output: v.audio_output,
            manual_function_calls: v.manual_function_calls,
            can_disable_turn_detection: v.can_disable_turn_detection,
            mutable_chat_context: v.mutable_chat_context,
            mutable_instructions: v.mutable_instructions,
            mutable_tools: v.mutable_tools,
            per_response_tool_choice: v.per_response_tool_choice,
            supports_say: v.supports_say,
            auto_turn_reply_generation: v.auto_turn_reply_generation,
        }
    }
}

impl From<RealtimeCapabilities> for lk_agents_abi::llm::realtime::RealtimeCapabilities {
    fn from(v: RealtimeCapabilities) -> Self {
        lk_agents_abi::llm::realtime::RealtimeCapabilities {
            message_truncation: v.message_truncation,
            turn_detection: v.turn_detection,
            user_transcription: v.user_transcription,
            auto_tool_reply_generation: v.auto_tool_reply_generation,
            audio_output: v.audio_output,
            manual_function_calls: v.manual_function_calls,
            can_disable_turn_detection: v.can_disable_turn_detection,
            mutable_chat_context: v.mutable_chat_context,
            mutable_instructions: v.mutable_instructions,
            mutable_tools: v.mutable_tools,
            per_response_tool_choice: v.per_response_tool_choice,
            supports_say: v.supports_say,
            auto_turn_reply_generation: v.auto_turn_reply_generation,
        }
    }
}

/// One entry of `Tagger.evaluations` (`observability.py:120`).
#[napi(object, js_name = "EvaluationRecord")]
pub struct EvaluationRecord {
    pub name: String,
    pub tag: String,
    pub verdict: String,
    pub reasoning: String,
    pub instructions: String,
}

impl From<&lk_agents_abi::telemetry::report::EvaluationRecord> for EvaluationRecord {
    fn from(v: &lk_agents_abi::telemetry::report::EvaluationRecord) -> Self {
        EvaluationRecord {
            name: v.name.to_string(),
            tag: v.tag.to_string(),
            verdict: v.verdict.to_string(),
            reasoning: v.reasoning.to_string(),
            instructions: v.instructions.to_string(),
        }
    }
}

impl From<EvaluationRecord> for lk_agents_abi::telemetry::report::EvaluationRecord {
    fn from(v: EvaluationRecord) -> Self {
        lk_agents_abi::telemetry::report::EvaluationRecord {
            name: v.name,
            tag: v.tag,
            verdict: v.verdict,
            reasoning: v.reasoning,
            instructions: v.instructions,
        }
    }
}

#[napi(object, js_name = "LlmChunk")]
pub struct LlmChunk {
    pub id: String,
    pub delta: Option<crate::generated::records::ChoiceDelta>,
    pub usage: Option<crate::generated::records::CompletionUsage>,
    /// `FlushSentinel`: finish the current TTS segment at this point in the stream.
    pub flush: bool,
    /// Text an `Agent.llm_node` answered itself (a `str`), not a model's: spoken, never metered as
    /// a model call (the reference meters the `LLMStream`, and such a node opened none).
    pub node_text: bool,
}

impl From<&lk_agents_abi::LlmChunk> for LlmChunk {
    fn from(v: &lk_agents_abi::LlmChunk) -> Self {
        LlmChunk {
            id: v.id.to_string(),
            delta: v.delta.as_ref().map(|v| crate::generated::records::ChoiceDelta::from(v)),
            usage: v.usage.as_ref().map(|v| crate::generated::records::CompletionUsage::from(v)),
            flush: v.flush,
            node_text: v.node_text,
        }
    }
}

impl TryFrom<LlmChunk> for lk_agents_abi::LlmChunk {
    type Error = crate::host_error::JsError;

    fn try_from(v: LlmChunk) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::LlmChunk {
            id: v.id,
            delta: v.delta.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
            usage: v.usage.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
            flush: v.flush,
            node_text: v.node_text,
        })
    }
}

/// Union type for all model usage types.
pub enum ModelUsage {
    Llm(LlmModelUsage),
    Tts(TtsModelUsage),
    Stt(SttModelUsage),
    Interruption(InterruptionModelUsage),
    Eot(EotModelUsage),
}

impl napi::bindgen_prelude::TypeName for ModelUsage {
    fn type_name() -> &'static str {
        "ModelUsage"
    }

    fn value_type() -> napi::ValueType {
        napi::ValueType::Object
    }
}

impl napi::bindgen_prelude::ToNapiValue for ModelUsage {
    unsafe fn to_napi_value(env: napi::sys::napi_env, val: Self) -> napi::Result<napi::sys::napi_value> {
        match val {
            ModelUsage::Llm(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            ModelUsage::Tts(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            ModelUsage::Stt(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            ModelUsage::Interruption(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            ModelUsage::Eot(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
        }
    }
}

impl From<&lk_agents_abi::metrics::usage::records::ModelUsage> for ModelUsage {
    fn from(v: &lk_agents_abi::metrics::usage::records::ModelUsage) -> Self {
        match v {
            lk_agents_abi::metrics::usage::records::ModelUsage::Llm(x) => ModelUsage::Llm(LlmModelUsage::from(x)),
            lk_agents_abi::metrics::usage::records::ModelUsage::Tts(x) => ModelUsage::Tts(TtsModelUsage::from(x)),
            lk_agents_abi::metrics::usage::records::ModelUsage::Stt(x) => ModelUsage::Stt(SttModelUsage::from(x)),
            lk_agents_abi::metrics::usage::records::ModelUsage::Interruption(x) => ModelUsage::Interruption(InterruptionModelUsage::from(x)),
            lk_agents_abi::metrics::usage::records::ModelUsage::Eot(x) => ModelUsage::Eot(EotModelUsage::from(x)),
        }
    }
}

impl napi::bindgen_prelude::FromNapiValue for ModelUsage {
    unsafe fn from_napi_value(env: napi::sys::napi_env, value: napi::sys::napi_value) -> napi::Result<Self> {
        let obj = unsafe { <napi::bindgen_prelude::Object as napi::bindgen_prelude::FromNapiValue>::from_napi_value(env, value)? };
        let tag: String = obj.get_named_property("type")?;
        match tag.as_str() {
            "llm_usage" => Ok(ModelUsage::Llm(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "tts_usage" => Ok(ModelUsage::Tts(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "stt_usage" => Ok(ModelUsage::Stt(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "interruption_usage" => Ok(ModelUsage::Interruption(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "eot_usage" => Ok(ModelUsage::Eot(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            other => Err(napi::Error::new(napi::Status::InvalidArg, format!("ModelUsage: unknown type `{other}`"))),
        }
    }
}

impl napi::bindgen_prelude::ValidateNapiValue for ModelUsage {}

impl TryFrom<ModelUsage> for lk_agents_abi::metrics::usage::records::ModelUsage {
    type Error = crate::host_error::JsError;

    fn try_from(v: ModelUsage) -> Result<Self, Self::Error> {
        Ok(match v {
            ModelUsage::Llm(x) => lk_agents_abi::metrics::usage::records::ModelUsage::Llm(x.try_into()?),
            ModelUsage::Tts(x) => lk_agents_abi::metrics::usage::records::ModelUsage::Tts(x.try_into()?),
            ModelUsage::Stt(x) => lk_agents_abi::metrics::usage::records::ModelUsage::Stt(x.try_into()?),
            ModelUsage::Interruption(x) => lk_agents_abi::metrics::usage::records::ModelUsage::Interruption(x.try_into()?),
            ModelUsage::Eot(x) => lk_agents_abi::metrics::usage::records::ModelUsage::Eot(x.try_into()?),
        })
    }
}

/// The resolved `AgentSessionOptions` as the host reads it: every public option of
/// `session.options`, seconds as `f64`. Built only by `resolved_session_options`, whose exhaustive
/// destructure keeps it complete.
#[napi(object, js_name = "ResolvedSessionOptions")]
pub struct ResolvedSessionOptions {
    pub endpointing: crate::generated::records::ResolvedEndpointing,
    pub interruption: crate::generated::records::ResolvedInterruption,
    pub preemptive_generation: crate::generated::records::ResolvedPreemptiveGeneration,
    pub user_turn_limit: crate::generated::records::ResolvedUserTurnLimit,
    /// The `turn_detection` the session runs on, as the public literal; `None` is `Auto`.
    pub turn_detection_mode: Option<String>,
    pub endpointing_overrides: crate::generated::records::EndpointingOverrides,
    pub max_tool_steps: u32,
    pub min_consecutive_speech_delay: f64,
    pub use_tts_aligned_transcript: Option<bool>,
    pub tts_text_transforms: Option<Vec<String>>,
    pub ivr_detection: bool,
    pub user_away_timeout: Option<f64>,
    pub transcription_timeout: Option<f64>,
    pub aec_warmup_duration: Option<f64>,
    pub session_close_transcript_timeout: f64,
}

impl From<&lk_agents_abi::voice::session_options::ResolvedSessionOptions> for ResolvedSessionOptions {
    fn from(v: &lk_agents_abi::voice::session_options::ResolvedSessionOptions) -> Self {
        ResolvedSessionOptions {
            endpointing: crate::generated::records::ResolvedEndpointing::from(&v.endpointing),
            interruption: crate::generated::records::ResolvedInterruption::from(&v.interruption),
            preemptive_generation: crate::generated::records::ResolvedPreemptiveGeneration::from(&v.preemptive_generation),
            user_turn_limit: crate::generated::records::ResolvedUserTurnLimit::from(&v.user_turn_limit),
            turn_detection_mode: v.turn_detection_mode.as_ref().map(|v| v.to_string()),
            endpointing_overrides: crate::generated::records::EndpointingOverrides::from(&v.endpointing_overrides),
            max_tool_steps: v.max_tool_steps,
            min_consecutive_speech_delay: v.min_consecutive_speech_delay,
            use_tts_aligned_transcript: v.use_tts_aligned_transcript,
            tts_text_transforms: v.tts_text_transforms.as_ref().map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<_>>()),
            ivr_detection: v.ivr_detection,
            user_away_timeout: v.user_away_timeout,
            transcription_timeout: v.transcription_timeout,
            aec_warmup_duration: v.aec_warmup_duration,
            session_close_transcript_timeout: v.session_close_transcript_timeout,
        }
    }
}

impl TryFrom<ResolvedSessionOptions> for lk_agents_abi::voice::session_options::ResolvedSessionOptions {
    type Error = crate::host_error::JsError;

    fn try_from(v: ResolvedSessionOptions) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::voice::session_options::ResolvedSessionOptions {
            endpointing: v.endpointing.try_into()?,
            interruption: v.interruption.try_into()?,
            preemptive_generation: v.preemptive_generation.try_into()?,
            user_turn_limit: v.user_turn_limit.try_into()?,
            turn_detection_mode: v.turn_detection_mode,
            endpointing_overrides: v.endpointing_overrides.try_into()?,
            max_tool_steps: v.max_tool_steps,
            min_consecutive_speech_delay: v.min_consecutive_speech_delay,
            use_tts_aligned_transcript: v.use_tts_aligned_transcript,
            tts_text_transforms: v.tts_text_transforms,
            ivr_detection: v.ivr_detection,
            user_away_timeout: v.user_away_timeout,
            transcription_timeout: v.transcription_timeout,
            aec_warmup_duration: v.aec_warmup_duration,
            session_close_transcript_timeout: v.session_close_transcript_timeout,
        })
    }
}

/// What a driver-run DTMF collection came to: `complete`, `incomplete`, or `stopped`.
pub enum DtmfOutcomeWire {
    Complete(DtmfComplete),
    Incomplete(DtmfIncomplete),
    Stopped(DtmfStopped),
}

impl napi::bindgen_prelude::TypeName for DtmfOutcomeWire {
    fn type_name() -> &'static str {
        "DtmfOutcomeWire"
    }

    fn value_type() -> napi::ValueType {
        napi::ValueType::Object
    }
}

impl napi::bindgen_prelude::ToNapiValue for DtmfOutcomeWire {
    unsafe fn to_napi_value(env: napi::sys::napi_env, val: Self) -> napi::Result<napi::sys::napi_value> {
        match val {
            DtmfOutcomeWire::Complete(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            DtmfOutcomeWire::Incomplete(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            DtmfOutcomeWire::Stopped(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
        }
    }
}

impl From<&lk_agents_abi::voice::workflows::DtmfOutcomeWire> for DtmfOutcomeWire {
    fn from(v: &lk_agents_abi::voice::workflows::DtmfOutcomeWire) -> Self {
        match v {
            lk_agents_abi::voice::workflows::DtmfOutcomeWire::Complete(x) => DtmfOutcomeWire::Complete(DtmfComplete::from(x)),
            lk_agents_abi::voice::workflows::DtmfOutcomeWire::Incomplete(x) => DtmfOutcomeWire::Incomplete(DtmfIncomplete::from(x)),
            lk_agents_abi::voice::workflows::DtmfOutcomeWire::Stopped(x) => DtmfOutcomeWire::Stopped(DtmfStopped::from(x)),
        }
    }
}

impl napi::bindgen_prelude::FromNapiValue for DtmfOutcomeWire {
    unsafe fn from_napi_value(env: napi::sys::napi_env, value: napi::sys::napi_value) -> napi::Result<Self> {
        let obj = unsafe { <napi::bindgen_prelude::Object as napi::bindgen_prelude::FromNapiValue>::from_napi_value(env, value)? };
        let tag: String = obj.get_named_property("kind")?;
        match tag.as_str() {
            "complete" => Ok(DtmfOutcomeWire::Complete(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "incomplete" => Ok(DtmfOutcomeWire::Incomplete(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "stopped" => Ok(DtmfOutcomeWire::Stopped(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            other => Err(napi::Error::new(napi::Status::InvalidArg, format!("DtmfOutcomeWire: unknown kind `{other}`"))),
        }
    }
}

impl napi::bindgen_prelude::ValidateNapiValue for DtmfOutcomeWire {}

impl TryFrom<DtmfOutcomeWire> for lk_agents_abi::voice::workflows::DtmfOutcomeWire {
    type Error = crate::host_error::JsError;

    fn try_from(v: DtmfOutcomeWire) -> Result<Self, Self::Error> {
        Ok(match v {
            DtmfOutcomeWire::Complete(x) => lk_agents_abi::voice::workflows::DtmfOutcomeWire::Complete(x.try_into()?),
            DtmfOutcomeWire::Incomplete(x) => lk_agents_abi::voice::workflows::DtmfOutcomeWire::Incomplete(x.try_into()?),
            DtmfOutcomeWire::Stopped(x) => lk_agents_abi::voice::workflows::DtmfOutcomeWire::Stopped(x.try_into()?),
        })
    }
}

/// `metrics/base.py:224`'s `AgentMetrics` union, in the reference's order.
pub enum AgentMetrics {
    Stt(SttMetrics),
    Llm(LlmMetrics),
    Tts(TtsMetrics),
    Vad(VadMetrics),
    Eou(EouMetrics),
    EotInference(EotInferenceMetrics),
    RealtimeModel(RealtimeModelMetrics),
    Interruption(InterruptionMetrics),
    Avatar(AvatarMetrics),
}

impl napi::bindgen_prelude::TypeName for AgentMetrics {
    fn type_name() -> &'static str {
        "AgentMetrics"
    }

    fn value_type() -> napi::ValueType {
        napi::ValueType::Object
    }
}

impl napi::bindgen_prelude::ToNapiValue for AgentMetrics {
    unsafe fn to_napi_value(env: napi::sys::napi_env, val: Self) -> napi::Result<napi::sys::napi_value> {
        match val {
            AgentMetrics::Stt(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            AgentMetrics::Llm(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            AgentMetrics::Tts(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            AgentMetrics::Vad(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            AgentMetrics::Eou(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            AgentMetrics::EotInference(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            AgentMetrics::RealtimeModel(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            AgentMetrics::Interruption(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
            AgentMetrics::Avatar(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },
        }
    }
}

impl From<&lk_agents_abi::metrics::base::AgentMetrics> for AgentMetrics {
    fn from(v: &lk_agents_abi::metrics::base::AgentMetrics) -> Self {
        match v {
            lk_agents_abi::metrics::base::AgentMetrics::Stt(x) => AgentMetrics::Stt(SttMetrics::from(x)),
            lk_agents_abi::metrics::base::AgentMetrics::Llm(x) => AgentMetrics::Llm(LlmMetrics::from(x)),
            lk_agents_abi::metrics::base::AgentMetrics::Tts(x) => AgentMetrics::Tts(TtsMetrics::from(x)),
            lk_agents_abi::metrics::base::AgentMetrics::Vad(x) => AgentMetrics::Vad(VadMetrics::from(x)),
            lk_agents_abi::metrics::base::AgentMetrics::Eou(x) => AgentMetrics::Eou(EouMetrics::from(x)),
            lk_agents_abi::metrics::base::AgentMetrics::EotInference(x) => AgentMetrics::EotInference(EotInferenceMetrics::from(x)),
            lk_agents_abi::metrics::base::AgentMetrics::RealtimeModel(x) => AgentMetrics::RealtimeModel(RealtimeModelMetrics::from(x)),
            lk_agents_abi::metrics::base::AgentMetrics::Interruption(x) => AgentMetrics::Interruption(InterruptionMetrics::from(x)),
            lk_agents_abi::metrics::base::AgentMetrics::Avatar(x) => AgentMetrics::Avatar(AvatarMetrics::from(x)),
        }
    }
}

impl napi::bindgen_prelude::FromNapiValue for AgentMetrics {
    unsafe fn from_napi_value(env: napi::sys::napi_env, value: napi::sys::napi_value) -> napi::Result<Self> {
        let obj = unsafe { <napi::bindgen_prelude::Object as napi::bindgen_prelude::FromNapiValue>::from_napi_value(env, value)? };
        let tag: String = obj.get_named_property("type")?;
        match tag.as_str() {
            "stt_metrics" => Ok(AgentMetrics::Stt(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "llm_metrics" => Ok(AgentMetrics::Llm(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "tts_metrics" => Ok(AgentMetrics::Tts(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "vad_metrics" => Ok(AgentMetrics::Vad(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "eou_metrics" => Ok(AgentMetrics::Eou(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "eot_inference_metrics" => Ok(AgentMetrics::EotInference(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "realtime_model_metrics" => Ok(AgentMetrics::RealtimeModel(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "interruption_metrics" => Ok(AgentMetrics::Interruption(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            "avatar_metrics" => Ok(AgentMetrics::Avatar(unsafe { napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? })),
            other => Err(napi::Error::new(napi::Status::InvalidArg, format!("AgentMetrics: unknown type `{other}`"))),
        }
    }
}

impl napi::bindgen_prelude::ValidateNapiValue for AgentMetrics {}

impl TryFrom<AgentMetrics> for lk_agents_abi::metrics::base::AgentMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: AgentMetrics) -> Result<Self, Self::Error> {
        Ok(match v {
            AgentMetrics::Stt(x) => lk_agents_abi::metrics::base::AgentMetrics::Stt(x.try_into()?),
            AgentMetrics::Llm(x) => lk_agents_abi::metrics::base::AgentMetrics::Llm(x.try_into()?),
            AgentMetrics::Tts(x) => lk_agents_abi::metrics::base::AgentMetrics::Tts(x.try_into()?),
            AgentMetrics::Vad(x) => lk_agents_abi::metrics::base::AgentMetrics::Vad(x.try_into()?),
            AgentMetrics::Eou(x) => lk_agents_abi::metrics::base::AgentMetrics::Eou(x.try_into()?),
            AgentMetrics::EotInference(x) => lk_agents_abi::metrics::base::AgentMetrics::EotInference(x.try_into()?),
            AgentMetrics::RealtimeModel(x) => lk_agents_abi::metrics::base::AgentMetrics::RealtimeModel(x.try_into()?),
            AgentMetrics::Interruption(x) => lk_agents_abi::metrics::base::AgentMetrics::Interruption(x.try_into()?),
            AgentMetrics::Avatar(x) => lk_agents_abi::metrics::base::AgentMetrics::Avatar(x.try_into()?),
        })
    }
}

/// `rtc.TranscriptionSegment`.
#[napi(object, js_name = "TranscriptionSegment")]
pub struct TranscriptionSegment {
    pub id: String,
    pub text: String,
    /// Milliseconds. `room_io` always writes 0 for both; kept because the field is on the wire.
    pub start_time: i64,
    pub end_time: i64,
    pub language: String,
    pub r#final: bool,
}

impl From<&lk_agents_abi::voice::rtc::TranscriptionSegment> for TranscriptionSegment {
    fn from(v: &lk_agents_abi::voice::rtc::TranscriptionSegment) -> Self {
        TranscriptionSegment {
            id: v.id.to_string(),
            text: v.text.to_string(),
            start_time: i64::try_from(v.start_time).unwrap_or(i64::MAX),
            end_time: i64::try_from(v.end_time).unwrap_or(i64::MAX),
            language: v.language.to_string(),
            r#final: v.is_final,
        }
    }
}

impl TryFrom<TranscriptionSegment> for lk_agents_abi::voice::rtc::TranscriptionSegment {
    type Error = crate::host_error::JsError;

    fn try_from(v: TranscriptionSegment) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::voice::rtc::TranscriptionSegment {
            id: v.id,
            text: v.text,
            start_time: <u64>::try_from(v.start_time).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("TranscriptionSegment.start_time: {e}")))?,
            end_time: <u64>::try_from(v.end_time).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("TranscriptionSegment.end_time: {e}")))?,
            language: v.language,
            is_final: v.r#final,
        })
    }
}

#[napi(object, js_name = "LlmMetrics")]
pub struct LlmMetrics {
    #[napi(js_name = "type", ts_type = "'llm_metrics'")]
    pub r#type: String,
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    pub duration: f64,
    /// Time to first generated token in seconds. -1 if the response generated none.
    pub ttft: f64,
    pub cancelled: bool,
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub prompt_cached_tokens: u32,
    /// The number of tokens used to write to the prompt cache (e.g. Anthropic cache writes).
    pub cache_creation_tokens: u32,
    /// The number of completion tokens spent on hidden reasoning.
    pub reasoning_tokens: u32,
    pub total_tokens: u32,
    pub tokens_per_second: f64,
    pub speech_id: Option<String>,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::LlmMetrics> for LlmMetrics {
    fn from(v: &lk_agents_abi::metrics::base::LlmMetrics) -> Self {
        LlmMetrics {
            r#type: "llm_metrics".to_owned(),
            label: v.label.to_string(),
            request_id: v.request_id.to_string(),
            timestamp: v.timestamp,
            duration: v.duration,
            ttft: v.ttft,
            cancelled: v.cancelled,
            completion_tokens: v.completion_tokens,
            prompt_tokens: v.prompt_tokens,
            prompt_cached_tokens: v.prompt_cached_tokens,
            cache_creation_tokens: v.cache_creation_tokens,
            reasoning_tokens: v.reasoning_tokens,
            total_tokens: v.total_tokens,
            tokens_per_second: v.tokens_per_second,
            speech_id: v.speech_id.as_ref().map(|v| v.to_string()),
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<LlmMetrics> for lk_agents_abi::metrics::base::LlmMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: LlmMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::LlmMetrics {
            label: v.label,
            request_id: v.request_id,
            timestamp: v.timestamp,
            duration: v.duration,
            ttft: v.ttft,
            cancelled: v.cancelled,
            completion_tokens: v.completion_tokens,
            prompt_tokens: v.prompt_tokens,
            prompt_cached_tokens: v.prompt_cached_tokens,
            cache_creation_tokens: v.cache_creation_tokens,
            reasoning_tokens: v.reasoning_tokens,
            total_tokens: v.total_tokens,
            tokens_per_second: v.tokens_per_second,
            speech_id: v.speech_id,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "SynthesizedAudio")]
pub struct SynthesizedAudio {
    /// Synthesized audio frame
    pub frame: crate::providers::host::JsAudioFrame,
    /// Request ID (one segment could be made up of multiple requests)
    pub request_id: String,
    /// Whether this is latest frame of the segment
    pub is_final: bool,
    /// Segment ID, each segment is separated by a flush (streaming only)
    pub segment_id: String,
    /// Current segment of the synthesized audio (streaming only)
    pub delta_text: String,
    pub started_time: Option<f64>,
    pub timed_transcript: Vec<crate::generated::records::TimedString>,
    pub encoding: String,
}

impl From<&lk_agents_abi::SynthesizedAudio> for SynthesizedAudio {
    fn from(v: &lk_agents_abi::SynthesizedAudio) -> Self {
        SynthesizedAudio {
            frame: crate::providers::host::audio_frame_to_js(&v.frame),
            request_id: v.request_id.to_string(),
            is_final: v.is_final,
            segment_id: v.segment_id.to_string(),
            delta_text: v.delta_text.to_string(),
            started_time: v.started_time,
            timed_transcript: v.timed_transcript.iter().map(|v| crate::generated::records::TimedString::from(v)).collect::<Vec<_>>(),
            encoding: crate::providers::host::encoding_to_js(&v.encoding),
        }
    }
}

impl TryFrom<SynthesizedAudio> for lk_agents_abi::SynthesizedAudio {
    type Error = crate::host_error::JsError;

    fn try_from(v: SynthesizedAudio) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::SynthesizedAudio {
            frame: crate::providers::host::audio_frame_from_js(v.frame),
            request_id: v.request_id,
            is_final: v.is_final,
            segment_id: v.segment_id,
            delta_text: v.delta_text,
            started_time: v.started_time,
            timed_transcript: v.timed_transcript.into_iter().map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).collect::<Result<Vec<_>, _>>()?,
            encoding: crate::providers::host::encoding_from_js(v.encoding),
        })
    }
}

#[napi(object, js_name = "TtsMetrics")]
pub struct TtsMetrics {
    #[napi(js_name = "type", ts_type = "'tts_metrics'")]
    pub r#type: String,
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    pub ttfb: f64,
    pub duration: f64,
    pub audio_duration: f64,
    pub cancelled: bool,
    /// Number of characters synthesized (for character-based billing).
    pub characters_count: u32,
    /// Input text tokens (for token-based billing, e.g., OpenAI TTS).
    pub input_tokens: u32,
    /// Output audio tokens (for token-based billing, e.g., OpenAI TTS).
    pub output_tokens: u32,
    pub streamed: bool,
    /// Time in seconds to acquire the connection. (WebSocket only)
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    pub connection_reused: bool,
    pub segment_id: Option<String>,
    pub speech_id: Option<String>,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::TtsMetrics> for TtsMetrics {
    fn from(v: &lk_agents_abi::metrics::base::TtsMetrics) -> Self {
        TtsMetrics {
            r#type: "tts_metrics".to_owned(),
            label: v.label.to_string(),
            request_id: v.request_id.to_string(),
            timestamp: v.timestamp,
            ttfb: v.ttfb,
            duration: v.duration,
            audio_duration: v.audio_duration,
            cancelled: v.cancelled,
            characters_count: v.characters_count,
            input_tokens: v.input_tokens,
            output_tokens: v.output_tokens,
            streamed: v.streamed,
            acquire_time: v.acquire_time,
            connection_reused: v.connection_reused,
            segment_id: v.segment_id.as_ref().map(|v| v.to_string()),
            speech_id: v.speech_id.as_ref().map(|v| v.to_string()),
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<TtsMetrics> for lk_agents_abi::metrics::base::TtsMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: TtsMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::TtsMetrics {
            label: v.label,
            request_id: v.request_id,
            timestamp: v.timestamp,
            ttfb: v.ttfb,
            duration: v.duration,
            audio_duration: v.audio_duration,
            cancelled: v.cancelled,
            characters_count: v.characters_count,
            input_tokens: v.input_tokens,
            output_tokens: v.output_tokens,
            streamed: v.streamed,
            acquire_time: v.acquire_time,
            connection_reused: v.connection_reused,
            segment_id: v.segment_id,
            speech_id: v.speech_id,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

/// A word (or chunk) with optional start/end times for alignment.
#[napi(object, js_name = "TimedString")]
pub struct TimedString {
    /// The text itself: `TimedString(text=word, ..)` out, and the `str` the object is on the way in.
    pub text: String,
    /// Word start time in seconds.
    pub start_time: Option<f64>,
    /// Word end time in seconds.
    pub end_time: Option<f64>,
    /// Per-word confidence score.
    pub confidence: Option<f64>,
    /// Offset in seconds relative to the start of the audio input stream or session. Used by STT
    /// plugins to align words against the session timeline.
    pub start_time_offset: Option<f64>,
    /// Speaker identifier when the provider supports diarization.
    pub speaker_id: Option<String>,
}

impl From<&lk_agents_abi::TimedString> for TimedString {
    fn from(v: &lk_agents_abi::TimedString) -> Self {
        TimedString {
            text: v.text.to_string(),
            start_time: v.start_time,
            end_time: v.end_time,
            confidence: v.confidence,
            start_time_offset: v.start_time_offset,
            speaker_id: v.speaker_id.as_ref().map(|v| v.to_string()),
        }
    }
}

impl From<TimedString> for lk_agents_abi::TimedString {
    fn from(v: TimedString) -> Self {
        lk_agents_abi::TimedString {
            text: v.text,
            start_time: v.start_time,
            end_time: v.end_time,
            confidence: v.confidence,
            start_time_offset: v.start_time_offset,
            speaker_id: v.speaker_id,
        }
    }
}

#[napi(object, js_name = "MetricsMetadata")]
pub struct MetricsMetadata {
    pub model_name: Option<String>,
    pub model_provider: Option<String>,
}

impl From<&lk_agents_abi::MetricsMetadata> for MetricsMetadata {
    fn from(v: &lk_agents_abi::MetricsMetadata) -> Self {
        MetricsMetadata {
            model_name: v.model_name.as_ref().map(|v| v.to_string()),
            model_provider: v.model_provider.as_ref().map(|v| v.to_string()),
        }
    }
}

impl From<MetricsMetadata> for lk_agents_abi::MetricsMetadata {
    fn from(v: MetricsMetadata) -> Self {
        lk_agents_abi::MetricsMetadata {
            model_name: v.model_name,
            model_provider: v.model_provider,
        }
    }
}

#[napi(object, js_name = "ChoiceDelta")]
pub struct ChoiceDelta {
    #[napi(ts_type = "'developer' | 'system' | 'user' | 'assistant' | null")]
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<crate::generated::records::FunctionToolCall>,
    /// Provider-specific extra data (e.g., Google thought signatures).
    pub extra: Option<serde_json::Map<String, serde_json::Value>>,
}

impl From<&lk_agents_abi::ChoiceDelta> for ChoiceDelta {
    fn from(v: &lk_agents_abi::ChoiceDelta) -> Self {
        ChoiceDelta {
            role: v.role.as_ref().map(|v| (v).as_str().to_owned()),
            content: v.content.as_ref().map(|v| v.to_string()),
            tool_calls: v.tool_calls.iter().map(|v| crate::generated::records::FunctionToolCall::from(v)).collect::<Vec<_>>(),
            extra: v.extra.as_ref().map(|v| v.clone()),
        }
    }
}

impl TryFrom<ChoiceDelta> for lk_agents_abi::ChoiceDelta {
    type Error = crate::host_error::JsError;

    fn try_from(v: ChoiceDelta) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::ChoiceDelta {
            role: v.role.map(|v| Ok::<_, crate::host_error::JsError>(lk_agents_abi::Role::from_literal(&v).ok_or_else(|| crate::host_error::JsError::new("ValueError", format!("not a chat role: {}", v)))?)).transpose()?,
            content: v.content,
            tool_calls: v.tool_calls.into_iter().map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).collect::<Result<Vec<_>, _>>()?,
            extra: v.extra,
        })
    }
}

#[napi(object, js_name = "CompletionUsage")]
pub struct CompletionUsage {
    /// The number of tokens in the completion.
    pub completion_tokens: u32,
    /// The number of input tokens used (includes cached tokens).
    pub prompt_tokens: u32,
    /// The number of cached input tokens used.
    pub prompt_cached_tokens: u32,
    /// The number of tokens used to create the cache.
    pub cache_creation_tokens: u32,
    /// The number of tokens read from the cache.
    pub cache_read_tokens: u32,
    /// The number of completion tokens spent on hidden reasoning.
    /// Already counted in ``completion_tokens``; do not add it to totals. Only reported by
    /// providers that break reasoning out separately (e.g. OpenAI-compatible responses
    /// carrying ``completion_tokens_details.reasoning_tokens``), 0 everywhere else.
    pub reasoning_tokens: u32,
    /// The total number of tokens used (completion + prompt tokens).
    pub total_tokens: u32,
    /// The service tier used for processing the request (e.g. 'default', 'priority', 'flex').
    /// Returned by providers that support tiered processing (e.g. OpenAI).
    pub service_tier: Option<String>,
}

impl From<&lk_agents_abi::CompletionUsage> for CompletionUsage {
    fn from(v: &lk_agents_abi::CompletionUsage) -> Self {
        CompletionUsage {
            completion_tokens: v.completion_tokens,
            prompt_tokens: v.prompt_tokens,
            prompt_cached_tokens: v.prompt_cached_tokens,
            cache_creation_tokens: v.cache_creation_tokens,
            cache_read_tokens: v.cache_read_tokens,
            reasoning_tokens: v.reasoning_tokens,
            total_tokens: v.total_tokens,
            service_tier: v.service_tier.as_ref().map(|v| v.to_string()),
        }
    }
}

impl From<CompletionUsage> for lk_agents_abi::CompletionUsage {
    fn from(v: CompletionUsage) -> Self {
        lk_agents_abi::CompletionUsage {
            completion_tokens: v.completion_tokens,
            prompt_tokens: v.prompt_tokens,
            prompt_cached_tokens: v.prompt_cached_tokens,
            cache_creation_tokens: v.cache_creation_tokens,
            cache_read_tokens: v.cache_read_tokens,
            reasoning_tokens: v.reasoning_tokens,
            total_tokens: v.total_tokens,
            service_tier: v.service_tier,
        }
    }
}

/// Usage summary for LLM models.
#[napi(object, js_name = "LlmModelUsage")]
pub struct LlmModelUsage {
    #[napi(js_name = "type", ts_type = "'llm_usage'")]
    pub r#type: String,
    /// The provider name (e.g., 'openai', 'anthropic').
    pub provider: String,
    /// The model name (e.g., 'gpt-4o', 'claude-3-5-sonnet').
    pub model: String,
    /// Total input tokens.
    pub input_tokens: i64,
    /// Input tokens served from cache.
    pub input_cached_tokens: i64,
    /// Input tokens used to write to the prompt cache (e.g. Anthropic cache writes).
    pub input_cache_creation_tokens: i64,
    /// Input audio tokens (for multimodal models).
    pub input_audio_tokens: i64,
    /// Cached input audio tokens.
    pub input_cached_audio_tokens: i64,
    /// Input text tokens.
    pub input_text_tokens: i64,
    /// Cached input text tokens.
    pub input_cached_text_tokens: i64,
    /// Input image tokens (for multimodal models).
    pub input_image_tokens: i64,
    /// Cached input image tokens.
    pub input_cached_image_tokens: i64,
    /// Total output tokens.
    pub output_tokens: i64,
    /// Output audio tokens (for multimodal models).
    pub output_audio_tokens: i64,
    /// Output text tokens.
    pub output_text_tokens: i64,
    /// Output tokens spent on hidden reasoning. Already counted in ``output_tokens``.
    pub output_reasoning_tokens: i64,
    /// Total session connection duration in seconds (for session-based billing like xAI).
    pub session_duration: f64,
}

impl From<&lk_agents_abi::metrics::usage::records::LlmModelUsage> for LlmModelUsage {
    fn from(v: &lk_agents_abi::metrics::usage::records::LlmModelUsage) -> Self {
        LlmModelUsage {
            r#type: "llm_usage".to_owned(),
            provider: v.provider.to_string(),
            model: v.model.to_string(),
            input_tokens: i64::try_from(v.input_tokens).unwrap_or(i64::MAX),
            input_cached_tokens: i64::try_from(v.input_cached_tokens).unwrap_or(i64::MAX),
            input_cache_creation_tokens: i64::try_from(v.input_cache_creation_tokens).unwrap_or(i64::MAX),
            input_audio_tokens: i64::try_from(v.input_audio_tokens).unwrap_or(i64::MAX),
            input_cached_audio_tokens: i64::try_from(v.input_cached_audio_tokens).unwrap_or(i64::MAX),
            input_text_tokens: i64::try_from(v.input_text_tokens).unwrap_or(i64::MAX),
            input_cached_text_tokens: i64::try_from(v.input_cached_text_tokens).unwrap_or(i64::MAX),
            input_image_tokens: i64::try_from(v.input_image_tokens).unwrap_or(i64::MAX),
            input_cached_image_tokens: i64::try_from(v.input_cached_image_tokens).unwrap_or(i64::MAX),
            output_tokens: i64::try_from(v.output_tokens).unwrap_or(i64::MAX),
            output_audio_tokens: i64::try_from(v.output_audio_tokens).unwrap_or(i64::MAX),
            output_text_tokens: i64::try_from(v.output_text_tokens).unwrap_or(i64::MAX),
            output_reasoning_tokens: i64::try_from(v.output_reasoning_tokens).unwrap_or(i64::MAX),
            session_duration: v.session_duration,
        }
    }
}

impl TryFrom<LlmModelUsage> for lk_agents_abi::metrics::usage::records::LlmModelUsage {
    type Error = crate::host_error::JsError;

    fn try_from(v: LlmModelUsage) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::usage::records::LlmModelUsage {
            provider: v.provider,
            model: v.model,
            input_tokens: <u64>::try_from(v.input_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_tokens: {e}")))?,
            input_cached_tokens: <u64>::try_from(v.input_cached_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_cached_tokens: {e}")))?,
            input_cache_creation_tokens: <u64>::try_from(v.input_cache_creation_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_cache_creation_tokens: {e}")))?,
            input_audio_tokens: <u64>::try_from(v.input_audio_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_audio_tokens: {e}")))?,
            input_cached_audio_tokens: <u64>::try_from(v.input_cached_audio_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_cached_audio_tokens: {e}")))?,
            input_text_tokens: <u64>::try_from(v.input_text_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_text_tokens: {e}")))?,
            input_cached_text_tokens: <u64>::try_from(v.input_cached_text_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_cached_text_tokens: {e}")))?,
            input_image_tokens: <u64>::try_from(v.input_image_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_image_tokens: {e}")))?,
            input_cached_image_tokens: <u64>::try_from(v.input_cached_image_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.input_cached_image_tokens: {e}")))?,
            output_tokens: <u64>::try_from(v.output_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.output_tokens: {e}")))?,
            output_audio_tokens: <u64>::try_from(v.output_audio_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.output_audio_tokens: {e}")))?,
            output_text_tokens: <u64>::try_from(v.output_text_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.output_text_tokens: {e}")))?,
            output_reasoning_tokens: <u64>::try_from(v.output_reasoning_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("LlmModelUsage.output_reasoning_tokens: {e}")))?,
            session_duration: v.session_duration,
        })
    }
}

/// Usage summary for TTS models.
#[napi(object, js_name = "TtsModelUsage")]
pub struct TtsModelUsage {
    #[napi(js_name = "type", ts_type = "'tts_usage'")]
    pub r#type: String,
    /// The provider name (e.g., 'elevenlabs', 'cartesia').
    pub provider: String,
    /// The model name (e.g., 'eleven_turbo_v2', 'sonic').
    pub model: String,
    /// Input text tokens (for token-based TTS billing, e.g., OpenAI TTS).
    pub input_tokens: i64,
    /// Output audio tokens (for token-based TTS billing, e.g., OpenAI TTS).
    pub output_tokens: i64,
    /// Number of characters synthesized (for character-based TTS billing).
    pub characters_count: i64,
    /// Duration of generated audio in seconds.
    pub audio_duration: f64,
}

impl From<&lk_agents_abi::metrics::usage::records::TtsModelUsage> for TtsModelUsage {
    fn from(v: &lk_agents_abi::metrics::usage::records::TtsModelUsage) -> Self {
        TtsModelUsage {
            r#type: "tts_usage".to_owned(),
            provider: v.provider.to_string(),
            model: v.model.to_string(),
            input_tokens: i64::try_from(v.input_tokens).unwrap_or(i64::MAX),
            output_tokens: i64::try_from(v.output_tokens).unwrap_or(i64::MAX),
            characters_count: i64::try_from(v.characters_count).unwrap_or(i64::MAX),
            audio_duration: v.audio_duration,
        }
    }
}

impl TryFrom<TtsModelUsage> for lk_agents_abi::metrics::usage::records::TtsModelUsage {
    type Error = crate::host_error::JsError;

    fn try_from(v: TtsModelUsage) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::usage::records::TtsModelUsage {
            provider: v.provider,
            model: v.model,
            input_tokens: <u64>::try_from(v.input_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("TtsModelUsage.input_tokens: {e}")))?,
            output_tokens: <u64>::try_from(v.output_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("TtsModelUsage.output_tokens: {e}")))?,
            characters_count: <u64>::try_from(v.characters_count).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("TtsModelUsage.characters_count: {e}")))?,
            audio_duration: v.audio_duration,
        })
    }
}

/// Usage summary for STT models.
#[napi(object, js_name = "SttModelUsage")]
pub struct SttModelUsage {
    #[napi(js_name = "type", ts_type = "'stt_usage'")]
    pub r#type: String,
    /// The provider name (e.g., 'deepgram', 'assemblyai').
    pub provider: String,
    /// The model name (e.g., 'nova-2', 'best').
    pub model: String,
    /// Total input tokens, including both audio and text (for token-based STT billing).
    pub input_tokens: i64,
    /// Audio input tokens, a subset of input_tokens when reported by the provider.
    pub input_audio_tokens: i64,
    /// Output text tokens (for token-based STT billing).
    pub output_tokens: i64,
    /// Duration of processed audio in seconds.
    pub audio_duration: f64,
}

impl From<&lk_agents_abi::metrics::usage::records::SttModelUsage> for SttModelUsage {
    fn from(v: &lk_agents_abi::metrics::usage::records::SttModelUsage) -> Self {
        SttModelUsage {
            r#type: "stt_usage".to_owned(),
            provider: v.provider.to_string(),
            model: v.model.to_string(),
            input_tokens: i64::try_from(v.input_tokens).unwrap_or(i64::MAX),
            input_audio_tokens: i64::try_from(v.input_audio_tokens).unwrap_or(i64::MAX),
            output_tokens: i64::try_from(v.output_tokens).unwrap_or(i64::MAX),
            audio_duration: v.audio_duration,
        }
    }
}

impl TryFrom<SttModelUsage> for lk_agents_abi::metrics::usage::records::SttModelUsage {
    type Error = crate::host_error::JsError;

    fn try_from(v: SttModelUsage) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::usage::records::SttModelUsage {
            provider: v.provider,
            model: v.model,
            input_tokens: <u64>::try_from(v.input_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("SttModelUsage.input_tokens: {e}")))?,
            input_audio_tokens: <u64>::try_from(v.input_audio_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("SttModelUsage.input_audio_tokens: {e}")))?,
            output_tokens: <u64>::try_from(v.output_tokens).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("SttModelUsage.output_tokens: {e}")))?,
            audio_duration: v.audio_duration,
        })
    }
}

/// Usage summary for interruption detection models.
#[napi(object, js_name = "InterruptionModelUsage")]
pub struct InterruptionModelUsage {
    #[napi(js_name = "type", ts_type = "'interruption_usage'")]
    pub r#type: String,
    /// The provider name (e.g., 'livekit').
    pub provider: String,
    /// The model name (e.g., 'adaptive').
    pub model: String,
    /// Total number of requests sent to the interruption detection model.
    pub total_requests: i64,
}

impl From<&lk_agents_abi::metrics::usage::records::InterruptionModelUsage> for InterruptionModelUsage {
    fn from(v: &lk_agents_abi::metrics::usage::records::InterruptionModelUsage) -> Self {
        InterruptionModelUsage {
            r#type: "interruption_usage".to_owned(),
            provider: v.provider.to_string(),
            model: v.model.to_string(),
            total_requests: i64::try_from(v.total_requests).unwrap_or(i64::MAX),
        }
    }
}

impl TryFrom<InterruptionModelUsage> for lk_agents_abi::metrics::usage::records::InterruptionModelUsage {
    type Error = crate::host_error::JsError;

    fn try_from(v: InterruptionModelUsage) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::usage::records::InterruptionModelUsage {
            provider: v.provider,
            model: v.model,
            total_requests: <u64>::try_from(v.total_requests).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("InterruptionModelUsage.total_requests: {e}")))?,
        })
    }
}

/// Usage summary for end-of-turn detection models.
#[napi(object, js_name = "EotModelUsage")]
pub struct EotModelUsage {
    #[napi(js_name = "type", ts_type = "'eot_usage'")]
    pub r#type: String,
    /// The provider name (e.g., 'livekit').
    pub provider: String,
    /// The model name (e.g., 'turn-detector-v1').
    pub model: String,
    /// Total number of inference requests sent to the EOT model.
    pub total_requests: i64,
}

impl From<&lk_agents_abi::metrics::usage::records::EotModelUsage> for EotModelUsage {
    fn from(v: &lk_agents_abi::metrics::usage::records::EotModelUsage) -> Self {
        EotModelUsage {
            r#type: "eot_usage".to_owned(),
            provider: v.provider.to_string(),
            model: v.model.to_string(),
            total_requests: i64::try_from(v.total_requests).unwrap_or(i64::MAX),
        }
    }
}

impl TryFrom<EotModelUsage> for lk_agents_abi::metrics::usage::records::EotModelUsage {
    type Error = crate::host_error::JsError;

    fn try_from(v: EotModelUsage) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::usage::records::EotModelUsage {
            provider: v.provider,
            model: v.model,
            total_requests: <u64>::try_from(v.total_requests).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("EotModelUsage.total_requests: {e}")))?,
        })
    }
}

/// The resolved `endpointing` group, seconds as `f64`.
#[napi(object, js_name = "ResolvedEndpointing")]
pub struct ResolvedEndpointing {
    pub mode: String,
    pub min_delay: f64,
    pub max_delay: f64,
    pub alpha: f64,
}

impl From<&lk_agents_abi::voice::session_options::ResolvedEndpointing> for ResolvedEndpointing {
    fn from(v: &lk_agents_abi::voice::session_options::ResolvedEndpointing) -> Self {
        ResolvedEndpointing {
            mode: v.mode.to_string(),
            min_delay: v.min_delay,
            max_delay: v.max_delay,
            alpha: v.alpha,
        }
    }
}

impl From<ResolvedEndpointing> for lk_agents_abi::voice::session_options::ResolvedEndpointing {
    fn from(v: ResolvedEndpointing) -> Self {
        lk_agents_abi::voice::session_options::ResolvedEndpointing {
            mode: v.mode,
            min_delay: v.min_delay,
            max_delay: v.max_delay,
            alpha: v.alpha,
        }
    }
}

/// The resolved `interruption` group. `mode` is absent when the session picks.
#[napi(object, js_name = "ResolvedInterruption")]
pub struct ResolvedInterruption {
    pub enabled: bool,
    pub mode: Option<String>,
    pub discard_audio_if_uninterruptible: bool,
    pub min_duration: f64,
    pub min_words: u32,
    pub resume_false_interruption: bool,
    pub false_interruption_timeout: Option<f64>,
    pub backchannel_boundary: Option<(f64, f64)>,
}

impl From<&lk_agents_abi::voice::session_options::ResolvedInterruption> for ResolvedInterruption {
    fn from(v: &lk_agents_abi::voice::session_options::ResolvedInterruption) -> Self {
        ResolvedInterruption {
            enabled: v.enabled,
            mode: v.mode.as_ref().map(|v| v.to_string()),
            discard_audio_if_uninterruptible: v.discard_audio_if_uninterruptible,
            min_duration: v.min_duration,
            min_words: v.min_words,
            resume_false_interruption: v.resume_false_interruption,
            false_interruption_timeout: v.false_interruption_timeout,
            backchannel_boundary: v.backchannel_boundary.as_ref().map(|v| { let (t0, t1,) = v; (*t0, *t1) }),
        }
    }
}

impl From<ResolvedInterruption> for lk_agents_abi::voice::session_options::ResolvedInterruption {
    fn from(v: ResolvedInterruption) -> Self {
        lk_agents_abi::voice::session_options::ResolvedInterruption {
            enabled: v.enabled,
            mode: v.mode,
            discard_audio_if_uninterruptible: v.discard_audio_if_uninterruptible,
            min_duration: v.min_duration,
            min_words: v.min_words,
            resume_false_interruption: v.resume_false_interruption,
            false_interruption_timeout: v.false_interruption_timeout,
            backchannel_boundary: v.backchannel_boundary,
        }
    }
}

/// The resolved `preemptive_generation` group.
#[napi(object, js_name = "ResolvedPreemptiveGeneration")]
pub struct ResolvedPreemptiveGeneration {
    pub enabled: bool,
    pub preemptive_tts: bool,
    pub max_speech_duration: f64,
    pub max_retries: u32,
}

impl From<&lk_agents_abi::voice::session_options::ResolvedPreemptiveGeneration> for ResolvedPreemptiveGeneration {
    fn from(v: &lk_agents_abi::voice::session_options::ResolvedPreemptiveGeneration) -> Self {
        ResolvedPreemptiveGeneration {
            enabled: v.enabled,
            preemptive_tts: v.preemptive_tts,
            max_speech_duration: v.max_speech_duration,
            max_retries: v.max_retries,
        }
    }
}

impl From<ResolvedPreemptiveGeneration> for lk_agents_abi::voice::session_options::ResolvedPreemptiveGeneration {
    fn from(v: ResolvedPreemptiveGeneration) -> Self {
        lk_agents_abi::voice::session_options::ResolvedPreemptiveGeneration {
            enabled: v.enabled,
            preemptive_tts: v.preemptive_tts,
            max_speech_duration: v.max_speech_duration,
            max_retries: v.max_retries,
        }
    }
}

/// The resolved `user_turn_limit` group.
#[napi(object, js_name = "ResolvedUserTurnLimit")]
pub struct ResolvedUserTurnLimit {
    pub max_words: Option<u32>,
    pub max_duration: Option<f64>,
}

impl From<&lk_agents_abi::voice::session_options::ResolvedUserTurnLimit> for ResolvedUserTurnLimit {
    fn from(v: &lk_agents_abi::voice::session_options::ResolvedUserTurnLimit) -> Self {
        ResolvedUserTurnLimit {
            max_words: v.max_words,
            max_duration: v.max_duration,
        }
    }
}

impl From<ResolvedUserTurnLimit> for lk_agents_abi::voice::session_options::ResolvedUserTurnLimit {
    fn from(v: ResolvedUserTurnLimit) -> Self {
        lk_agents_abi::voice::session_options::ResolvedUserTurnLimit {
            max_words: v.max_words,
            max_duration: v.max_duration,
        }
    }
}

/// The session layer's sparse endpointing (`AgentSessionOptions.endpointing_overrides`).
#[napi(object, js_name = "EndpointingOverrides")]
pub struct EndpointingOverrides {
    pub mode: Option<String>,
    pub min_delay: Option<f64>,
    pub max_delay: Option<f64>,
    pub alpha: Option<f64>,
}

impl From<&lk_agents_abi::voice::session_options::EndpointingOverrides> for EndpointingOverrides {
    fn from(v: &lk_agents_abi::voice::session_options::EndpointingOverrides) -> Self {
        EndpointingOverrides {
            mode: v.mode.as_ref().map(|v| v.to_string()),
            min_delay: v.min_delay,
            max_delay: v.max_delay,
            alpha: v.alpha,
        }
    }
}

impl From<EndpointingOverrides> for lk_agents_abi::voice::session_options::EndpointingOverrides {
    fn from(v: EndpointingOverrides) -> Self {
        lk_agents_abi::voice::session_options::EndpointingOverrides {
            mode: v.mode,
            min_delay: v.min_delay,
            max_delay: v.max_delay,
            alpha: v.alpha,
        }
    }
}

/// `self.complete(GetDtmfResult(user_input))`.
#[napi(object, js_name = "DtmfComplete")]
pub struct DtmfComplete {
    #[napi(js_name = "kind", ts_type = "'complete'")]
    pub kind: String,
    pub user_input: String,
}

impl From<&lk_agents_abi::voice::workflows::DtmfComplete> for DtmfComplete {
    fn from(v: &lk_agents_abi::voice::workflows::DtmfComplete) -> Self {
        DtmfComplete {
            kind: "complete".to_owned(),
            user_input: v.user_input.to_string(),
        }
    }
}

impl From<DtmfComplete> for lk_agents_abi::voice::workflows::DtmfComplete {
    fn from(v: DtmfComplete) -> Self {
        lk_agents_abi::voice::workflows::DtmfComplete {
            user_input: v.user_input,
        }
    }
}

/// `self.complete(ToolError(message))`: the debounce fired before `num_digits` arrived.
#[napi(object, js_name = "DtmfIncomplete")]
pub struct DtmfIncomplete {
    #[napi(js_name = "kind", ts_type = "'incomplete'")]
    pub kind: String,
    pub message: String,
}

impl From<&lk_agents_abi::voice::workflows::DtmfIncomplete> for DtmfIncomplete {
    fn from(v: &lk_agents_abi::voice::workflows::DtmfIncomplete) -> Self {
        DtmfIncomplete {
            kind: "incomplete".to_owned(),
            message: v.message.to_string(),
        }
    }
}

impl From<DtmfIncomplete> for lk_agents_abi::voice::workflows::DtmfIncomplete {
    fn from(v: DtmfIncomplete) -> Self {
        lk_agents_abi::voice::workflows::DtmfIncomplete {
            message: v.message,
        }
    }
}

/// The collection was stopped, replaced or closed first.
#[napi(object, js_name = "DtmfStopped")]
pub struct DtmfStopped {
    #[napi(js_name = "kind", ts_type = "'stopped'")]
    pub kind: String,
}

impl From<&lk_agents_abi::voice::workflows::DtmfStopped> for DtmfStopped {
    fn from(v: &lk_agents_abi::voice::workflows::DtmfStopped) -> Self {
        DtmfStopped {
            kind: "stopped".to_owned(),
        }
    }
}

impl From<DtmfStopped> for lk_agents_abi::voice::workflows::DtmfStopped {
    fn from(v: DtmfStopped) -> Self {
        lk_agents_abi::voice::workflows::DtmfStopped {
        }
    }
}

#[napi(object, js_name = "SttMetrics")]
pub struct SttMetrics {
    #[napi(js_name = "type", ts_type = "'stt_metrics'")]
    pub r#type: String,
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    /// The request duration in seconds, 0.0 if the STT is streaming.
    pub duration: f64,
    /// The duration of the pushed audio in seconds, or 0.0 when unknown.
    pub audio_duration: f64,
    /// Whether the STT is streaming (e.g using websocket).
    pub streamed: bool,
    /// Total input tokens, including both audio and text (for token-based billing).
    pub input_tokens: u32,
    /// Total output tokens (for token-based billing).
    pub output_tokens: u32,
    /// Sum of input and output tokens.
    pub total_tokens: u32,
    /// Audio input tokens, a subset of input_tokens when reported by the provider.
    pub input_audio_tokens: u32,
    /// Time in seconds to acquire the connection. (WebSocket only)
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    pub connection_reused: bool,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::SttMetrics> for SttMetrics {
    fn from(v: &lk_agents_abi::metrics::base::SttMetrics) -> Self {
        SttMetrics {
            r#type: "stt_metrics".to_owned(),
            label: v.label.to_string(),
            request_id: v.request_id.to_string(),
            timestamp: v.timestamp,
            duration: v.duration,
            audio_duration: v.audio_duration,
            streamed: v.streamed,
            input_tokens: v.input_tokens,
            output_tokens: v.output_tokens,
            total_tokens: v.total_tokens,
            input_audio_tokens: v.input_audio_tokens,
            acquire_time: v.acquire_time,
            connection_reused: v.connection_reused,
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<SttMetrics> for lk_agents_abi::metrics::base::SttMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: SttMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::SttMetrics {
            label: v.label,
            request_id: v.request_id,
            timestamp: v.timestamp,
            duration: v.duration,
            audio_duration: v.audio_duration,
            streamed: v.streamed,
            input_tokens: v.input_tokens,
            output_tokens: v.output_tokens,
            total_tokens: v.total_tokens,
            input_audio_tokens: v.input_audio_tokens,
            acquire_time: v.acquire_time,
            connection_reused: v.connection_reused,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "VadMetrics")]
pub struct VadMetrics {
    #[napi(js_name = "type", ts_type = "'vad_metrics'")]
    pub r#type: String,
    pub label: String,
    pub timestamp: f64,
    pub idle_time: f64,
    pub inference_duration_total: f64,
    pub inference_count: u32,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::VadMetrics> for VadMetrics {
    fn from(v: &lk_agents_abi::metrics::base::VadMetrics) -> Self {
        VadMetrics {
            r#type: "vad_metrics".to_owned(),
            label: v.label.to_string(),
            timestamp: v.timestamp,
            idle_time: v.idle_time,
            inference_duration_total: v.inference_duration_total,
            inference_count: v.inference_count,
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<VadMetrics> for lk_agents_abi::metrics::base::VadMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: VadMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::VadMetrics {
            label: v.label,
            timestamp: v.timestamp,
            idle_time: v.idle_time,
            inference_duration_total: v.inference_duration_total,
            inference_count: v.inference_count,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "EouMetrics")]
pub struct EouMetrics {
    #[napi(js_name = "type", ts_type = "'eou_metrics'")]
    pub r#type: String,
    pub timestamp: f64,
    /// Amount of time between the end of speech from VAD and the decision to end the user's turn.
    /// Set to 0.0 if the end of speech was not detected.
    pub end_of_utterance_delay: f64,
    /// Time taken to obtain the transcript after the end of the user's speech.
    /// Set to 0.0 if the end of speech was not detected.
    pub transcription_delay: f64,
    /// Time taken to invoke the user's `Agent.on_user_turn_completed` callback.
    pub on_user_turn_completed_delay: f64,
    pub speech_id: Option<String>,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::EouMetrics> for EouMetrics {
    fn from(v: &lk_agents_abi::metrics::base::EouMetrics) -> Self {
        EouMetrics {
            r#type: "eou_metrics".to_owned(),
            timestamp: v.timestamp,
            end_of_utterance_delay: v.end_of_utterance_delay,
            transcription_delay: v.transcription_delay,
            on_user_turn_completed_delay: v.on_user_turn_completed_delay,
            speech_id: v.speech_id.as_ref().map(|v| v.to_string()),
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<EouMetrics> for lk_agents_abi::metrics::base::EouMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: EouMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::EouMetrics {
            timestamp: v.timestamp,
            end_of_utterance_delay: v.end_of_utterance_delay,
            transcription_delay: v.transcription_delay,
            on_user_turn_completed_delay: v.on_user_turn_completed_delay,
            speech_id: v.speech_id,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

/// Per-inference metrics emitted by the EOT model on each prediction.
#[napi(object, js_name = "EotInferenceMetrics")]
pub struct EotInferenceMetrics {
    #[napi(js_name = "type", ts_type = "'eot_inference_metrics'")]
    pub r#type: String,
    pub timestamp: f64,
    /// Earliest audio creation time in an inference to response receive time.
    pub total_duration: f64,
    /// Latest audio creation time in an inference to response receive time.
    pub detection_delay: f64,
    /// Server side model inference time.
    pub prediction_duration: f64,
    /// Number of inference requests made during one inference.
    pub num_requests: u32,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::EotInferenceMetrics> for EotInferenceMetrics {
    fn from(v: &lk_agents_abi::metrics::base::EotInferenceMetrics) -> Self {
        EotInferenceMetrics {
            r#type: "eot_inference_metrics".to_owned(),
            timestamp: v.timestamp,
            total_duration: v.total_duration,
            detection_delay: v.detection_delay,
            prediction_duration: v.prediction_duration,
            num_requests: v.num_requests,
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<EotInferenceMetrics> for lk_agents_abi::metrics::base::EotInferenceMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: EotInferenceMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::EotInferenceMetrics {
            timestamp: v.timestamp,
            total_duration: v.total_duration,
            detection_delay: v.detection_delay,
            prediction_duration: v.prediction_duration,
            num_requests: v.num_requests,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "RealtimeModelMetrics")]
pub struct RealtimeModelMetrics {
    #[napi(js_name = "type", ts_type = "'realtime_model_metrics'")]
    pub r#type: String,
    pub label: String,
    pub request_id: String,
    /// The timestamp of the response creation.
    pub timestamp: f64,
    /// The duration of the response from created to done in seconds.
    pub duration: f64,
    /// The duration of the session connection in seconds (for session-based billing like xAI).
    pub session_duration: f64,
    /// Time to first audio token in seconds. -1 if no audio token was sent.
    pub ttft: f64,
    /// Whether the request was cancelled.
    pub cancelled: bool,
    /// The number of input tokens used in the Response, including text and audio tokens.
    pub input_tokens: u32,
    /// The number of output tokens sent in the Response, including text and audio tokens.
    pub output_tokens: u32,
    /// The total number of tokens in the Response.
    pub total_tokens: u32,
    /// The number of tokens per second.
    pub tokens_per_second: f64,
    /// Details about the input tokens used in the Response.
    pub input_token_details: crate::generated::records::InputTokenDetails,
    /// Details about the output tokens used in the Response.
    pub output_token_details: crate::generated::records::OutputTokenDetails,
    /// Time in seconds to acquire the connection. (WebSocket only)
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    pub connection_reused: bool,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::RealtimeModelMetrics> for RealtimeModelMetrics {
    fn from(v: &lk_agents_abi::metrics::base::RealtimeModelMetrics) -> Self {
        RealtimeModelMetrics {
            r#type: "realtime_model_metrics".to_owned(),
            label: v.label.to_string(),
            request_id: v.request_id.to_string(),
            timestamp: v.timestamp,
            duration: v.duration,
            session_duration: v.session_duration,
            ttft: v.ttft,
            cancelled: v.cancelled,
            input_tokens: v.input_tokens,
            output_tokens: v.output_tokens,
            total_tokens: v.total_tokens,
            tokens_per_second: v.tokens_per_second,
            input_token_details: crate::generated::records::InputTokenDetails::from(&v.input_token_details),
            output_token_details: crate::generated::records::OutputTokenDetails::from(&v.output_token_details),
            acquire_time: v.acquire_time,
            connection_reused: v.connection_reused,
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<RealtimeModelMetrics> for lk_agents_abi::metrics::base::RealtimeModelMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: RealtimeModelMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::RealtimeModelMetrics {
            label: v.label,
            request_id: v.request_id,
            timestamp: v.timestamp,
            duration: v.duration,
            session_duration: v.session_duration,
            ttft: v.ttft,
            cancelled: v.cancelled,
            input_tokens: v.input_tokens,
            output_tokens: v.output_tokens,
            total_tokens: v.total_tokens,
            tokens_per_second: v.tokens_per_second,
            input_token_details: v.input_token_details.try_into()?,
            output_token_details: v.output_token_details.try_into()?,
            acquire_time: v.acquire_time,
            connection_reused: v.connection_reused,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "InterruptionMetrics")]
pub struct InterruptionMetrics {
    #[napi(js_name = "type", ts_type = "'interruption_metrics'")]
    pub r#type: String,
    pub timestamp: f64,
    /// Latest RTT (Round Trip Time) time taken to perform the inference, in seconds.
    pub total_duration: f64,
    /// Latest time taken to perform the inference from the model side, in seconds.
    pub prediction_duration: f64,
    /// Latest total time from the onset of the speech to the final prediction, in seconds.
    pub detection_delay: f64,
    /// Number of interruptions detected, incrementally counted.
    pub num_interruptions: u32,
    /// Number of backchannels detected, incrementally counted.
    pub num_backchannels: u32,
    /// Number of requests sent to the interruption detection model, incrementally counted.
    pub num_requests: u32,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::InterruptionMetrics> for InterruptionMetrics {
    fn from(v: &lk_agents_abi::metrics::base::InterruptionMetrics) -> Self {
        InterruptionMetrics {
            r#type: "interruption_metrics".to_owned(),
            timestamp: v.timestamp,
            total_duration: v.total_duration,
            prediction_duration: v.prediction_duration,
            detection_delay: v.detection_delay,
            num_interruptions: v.num_interruptions,
            num_backchannels: v.num_backchannels,
            num_requests: v.num_requests,
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<InterruptionMetrics> for lk_agents_abi::metrics::base::InterruptionMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: InterruptionMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::InterruptionMetrics {
            timestamp: v.timestamp,
            total_duration: v.total_duration,
            prediction_duration: v.prediction_duration,
            detection_delay: v.detection_delay,
            num_interruptions: v.num_interruptions,
            num_backchannels: v.num_backchannels,
            num_requests: v.num_requests,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "AvatarMetrics")]
pub struct AvatarMetrics {
    #[napi(js_name = "type", ts_type = "'avatar_metrics'")]
    pub r#type: String,
    pub timestamp: f64,
    /// Delay between forwarding the first audio frame to the avatar and the playback started.
    pub playback_latency: f64,
    /// Time when the avatar session was started.
    pub session_started_time: Option<f64>,
    /// Time when the avatar participant joined and started video track.
    pub avatar_joined_time: Option<f64>,
    pub metadata: Option<crate::generated::records::Metadata>,
}

impl From<&lk_agents_abi::metrics::base::AvatarMetrics> for AvatarMetrics {
    fn from(v: &lk_agents_abi::metrics::base::AvatarMetrics) -> Self {
        AvatarMetrics {
            r#type: "avatar_metrics".to_owned(),
            timestamp: v.timestamp,
            playback_latency: v.playback_latency,
            session_started_time: v.session_started_time,
            avatar_joined_time: v.avatar_joined_time,
            metadata: v.metadata.as_ref().map(|v| crate::generated::records::Metadata::from(v)),
        }
    }
}

impl TryFrom<AvatarMetrics> for lk_agents_abi::metrics::base::AvatarMetrics {
    type Error = crate::host_error::JsError;

    fn try_from(v: AvatarMetrics) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::AvatarMetrics {
            timestamp: v.timestamp,
            playback_latency: v.playback_latency,
            session_started_time: v.session_started_time,
            avatar_joined_time: v.avatar_joined_time,
            metadata: v.metadata.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "Metadata")]
pub struct Metadata {
    pub model_name: Option<String>,
    pub model_provider: Option<String>,
}

impl From<&lk_agents_abi::metrics::base::Metadata> for Metadata {
    fn from(v: &lk_agents_abi::metrics::base::Metadata) -> Self {
        Metadata {
            model_name: v.model_name.as_ref().map(|v| v.to_string()),
            model_provider: v.model_provider.as_ref().map(|v| v.to_string()),
        }
    }
}

impl From<Metadata> for lk_agents_abi::metrics::base::Metadata {
    fn from(v: Metadata) -> Self {
        lk_agents_abi::metrics::base::Metadata {
            model_name: v.model_name,
            model_provider: v.model_provider,
        }
    }
}

#[napi(object, js_name = "FunctionToolCall")]
pub struct FunctionToolCall {
    #[napi(js_name = "type", ts_type = "'function'")]
    pub r#type: String,
    pub name: String,
    pub arguments: String,
    pub call_id: String,
    /// Provider-specific extra data (e.g., Google thought signatures).
    pub extra: Option<serde_json::Map<String, serde_json::Value>>,
}

impl From<&lk_agents_abi::FunctionToolCall> for FunctionToolCall {
    fn from(v: &lk_agents_abi::FunctionToolCall) -> Self {
        FunctionToolCall {
            r#type: "function".to_owned(),
            name: v.name.to_string(),
            arguments: v.arguments.to_string(),
            call_id: v.call_id.to_string(),
            extra: v.extra.as_ref().map(|v| v.clone()),
        }
    }
}

impl From<FunctionToolCall> for lk_agents_abi::FunctionToolCall {
    fn from(v: FunctionToolCall) -> Self {
        lk_agents_abi::FunctionToolCall {
            name: v.name,
            arguments: v.arguments,
            call_id: v.call_id,
            extra: v.extra,
        }
    }
}

#[napi(object, js_name = "InputTokenDetails")]
pub struct InputTokenDetails {
    pub audio_tokens: u32,
    pub text_tokens: u32,
    pub image_tokens: u32,
    pub cached_tokens: u32,
    pub cached_tokens_details: Option<crate::generated::records::CachedTokenDetails>,
}

impl From<&lk_agents_abi::metrics::base::InputTokenDetails> for InputTokenDetails {
    fn from(v: &lk_agents_abi::metrics::base::InputTokenDetails) -> Self {
        InputTokenDetails {
            audio_tokens: v.audio_tokens,
            text_tokens: v.text_tokens,
            image_tokens: v.image_tokens,
            cached_tokens: v.cached_tokens,
            cached_tokens_details: v.cached_tokens_details.as_ref().map(|v| crate::generated::records::CachedTokenDetails::from(v)),
        }
    }
}

impl TryFrom<InputTokenDetails> for lk_agents_abi::metrics::base::InputTokenDetails {
    type Error = crate::host_error::JsError;

    fn try_from(v: InputTokenDetails) -> Result<Self, Self::Error> {
        Ok(lk_agents_abi::metrics::base::InputTokenDetails {
            audio_tokens: v.audio_tokens,
            text_tokens: v.text_tokens,
            image_tokens: v.image_tokens,
            cached_tokens: v.cached_tokens,
            cached_tokens_details: v.cached_tokens_details.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?,
        })
    }
}

#[napi(object, js_name = "OutputTokenDetails")]
pub struct OutputTokenDetails {
    pub text_tokens: u32,
    pub audio_tokens: u32,
    pub image_tokens: u32,
}

impl From<&lk_agents_abi::metrics::base::OutputTokenDetails> for OutputTokenDetails {
    fn from(v: &lk_agents_abi::metrics::base::OutputTokenDetails) -> Self {
        OutputTokenDetails {
            text_tokens: v.text_tokens,
            audio_tokens: v.audio_tokens,
            image_tokens: v.image_tokens,
        }
    }
}

impl From<OutputTokenDetails> for lk_agents_abi::metrics::base::OutputTokenDetails {
    fn from(v: OutputTokenDetails) -> Self {
        lk_agents_abi::metrics::base::OutputTokenDetails {
            text_tokens: v.text_tokens,
            audio_tokens: v.audio_tokens,
            image_tokens: v.image_tokens,
        }
    }
}

#[napi(object, js_name = "CachedTokenDetails")]
pub struct CachedTokenDetails {
    pub audio_tokens: u32,
    pub text_tokens: u32,
    pub image_tokens: u32,
}

impl From<&lk_agents_abi::metrics::base::CachedTokenDetails> for CachedTokenDetails {
    fn from(v: &lk_agents_abi::metrics::base::CachedTokenDetails) -> Self {
        CachedTokenDetails {
            audio_tokens: v.audio_tokens,
            text_tokens: v.text_tokens,
            image_tokens: v.image_tokens,
        }
    }
}

impl From<CachedTokenDetails> for lk_agents_abi::metrics::base::CachedTokenDetails {
    fn from(v: CachedTokenDetails) -> Self {
        lk_agents_abi::metrics::base::CachedTokenDetails {
            audio_tokens: v.audio_tokens,
            text_tokens: v.text_tokens,
            image_tokens: v.image_tokens,
        }
    }
}
