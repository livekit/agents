use super::TtsOptions;
use lk_agents_abi::TranscriptChunk;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::inference::models::InferenceProvider;
pub use lk_agents_abi::inference::tts_wire::{ErrorCode, FallbackCause, FallbackType};
use lk_agents_abi::tts::markup::Provider as MarkupProvider;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const DEFAULT_ENCODING: &str = "pcm_s16le";
pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;

pub use lk_agents_abi::inference::fallback::TtsFallbackModel as FallbackModel;

/// `"provider/model:voice"` -> `("provider/model", Some("voice"))`.
pub fn parse_model_string(model: &str) -> (&str, Option<&str>) {
    crate::stt::options::parse_model_string(model)
}

/// `_normalize_fallback`'s string arm.
pub fn normalize_fallback_str(model: &str) -> lk_agents_abi::inference::fallback::TtsFallbackModel {
    let (name, voice) = parse_model_string(model);
    FallbackModel {
        model: name.to_owned(),
        // `voice if voice else ""`.
        voice: voice.filter(|v| !v.is_empty()).unwrap_or("").to_owned(),
        extra_kwargs: None,
    }
}

/// `_has_aligned_transcript`: whether this model and these options produce timings (recomputed
/// on `update_options`).
pub fn has_aligned_transcript(model: &str, extra_kwargs: &Map<String, Value>) -> bool {
    match InferenceProvider::of_model(model) {
        // `bool(extra_kwargs.get("add_timestamps"))` -- `false`, `0` and `""` are all "no".
        Some(InferenceProvider::Cartesia) => truthy(extra_kwargs.get("add_timestamps")),
        Some(InferenceProvider::Inworld) => matches!(
            extra_kwargs.get("timestamp_type").and_then(Value::as_str),
            Some("WORD") | Some("CHARACTER")
        ),
        Some(InferenceProvider::Assemblyai) => {
            extra_kwargs.get("word_timestamps") != Some(&Value::Bool(false))
        }
        _ => false,
    }
}

fn truthy(value: Option<&Value>) -> bool {
    value.is_some_and(lk_agents_core::llm::provider_format::is_truthy)
}

/// `TTS.Markup._provider_key()`: the markup dialect, `None` for a model with no markup.
pub fn markup_provider(model: &str) -> Option<MarkupProvider> {
    match InferenceProvider::of_model(model)? {
        InferenceProvider::Cartesia => Some(MarkupProvider::Cartesia),
        InferenceProvider::Inworld if model.contains("tts-2") => Some(MarkupProvider::Inworld),
        InferenceProvider::Xai => Some(MarkupProvider::Xai),
        InferenceProvider::Fishaudio if model.contains("drama") => {
            Some(MarkupProvider::FishAudioDrama)
        }
        InferenceProvider::Fishaudio => Some(MarkupProvider::FishAudio),
        _ => None,
    }
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ClientMessage<'a> {
    #[serde(rename = "session.create")]
    SessionCreate {
        sample_rate: u32,
        encoding: &'a str,
        extra: super::provider_options::Options,
        #[serde(flatten)]
        generation: GenerationConfig<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        fallback: Option<Fallback<'a>>,
        connection: crate::extra::Connection,
    },
    #[serde(rename = "input_transcript")]
    InputTranscript {
        transcript: String,
        generation_config: GenerationConfig<'a>,
        extra: super::provider_options::Options,
    },
    #[serde(rename = "session.flush")]
    SessionFlush,
}

#[derive(Serialize)]
struct GenerationConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
}

impl<'a> From<&'a TtsOptions> for GenerationConfig<'a> {
    fn from(opts: &'a TtsOptions) -> Self {
        Self {
            voice: opts.voice.as_deref().filter(|s| !s.is_empty()),
            model: (!opts.model.is_empty()).then_some(opts.model.as_str()),
            language: opts.language.as_deref().filter(|s| !s.is_empty()),
        }
    }
}

#[derive(Serialize)]
struct Fallback<'a> {
    models: Vec<FallbackEntry<'a>>,
}

#[derive(Serialize)]
struct FallbackEntry<'a> {
    model: &'a str,
    voice: &'a str,
    extra: super::provider_options::Options,
}

fn invalid_options(_: serde_json::Error) -> ApiError {
    let mut error = ApiError::new("Invalid LiveKit Inference TTS provider options");
    error.retryable = false;
    error
}

pub fn session_create(opts: &TtsOptions) -> Result<Value, ApiError> {
    let models = opts
        .fallback
        .iter()
        .map(|f| {
            Ok(FallbackEntry {
                model: &f.model,
                voice: &f.voice,
                extra: super::provider_options::Options::parse(
                    &f.model,
                    f.extra_kwargs.clone().unwrap_or_default(),
                )
                .map_err(invalid_options)?,
            })
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    serde_json::to_value(ClientMessage::SessionCreate {
        sample_rate: opts.sample_rate,
        encoding: &opts.encoding,
        extra: super::provider_options::Options::parse(&opts.model, opts.extra_kwargs.clone())
            .map_err(invalid_options)?,
        generation: opts.into(),
        fallback: (!models.is_empty()).then_some(Fallback { models }),
        connection: crate::extra::Connection::from(&opts.conn_options),
    })
    .map_err(invalid_options)
}

/// The trailing space matters: the gateway concatenates transcripts.
pub fn input_transcript(text: &str, opts: &TtsOptions) -> Result<Value, ApiError> {
    serde_json::to_value(ClientMessage::InputTranscript {
        transcript: format!("{text} "),
        generation_config: opts.into(),
        extra: super::provider_options::Options::parse(&opts.model, opts.extra_kwargs.clone())
            .map_err(invalid_options)?,
    })
    .map_err(invalid_options)
}

pub fn session_flush() -> Value {
    serde_json::to_value(ClientMessage::SessionFlush).expect("a unit message serializes")
}

#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    SessionCreated {
        session_id: Option<String>,
    },
    OutputAudio {
        session_id: Option<String>,
        audio: String,
    },
    OutputAlignment {
        session_id: Option<String>,
        tokens: Vec<TranscriptChunk>,
    },
    Done {
        session_id: Option<String>,
    },
    Error {
        code: ErrorCode,
        retryable: Option<bool>,
    },
    FallbackActivated {
        session_id: Option<String>,
        fallback_type: FallbackType,
        provider: String,
        model: String,
        voice: String,
        cause: FallbackCause,
    },
    Malformed {
        detail: String,
    },
    Unknown {
        session_id: Option<String>,
    },
}

/// Decode one text frame. `provider` selects Cartesia's re-added trailing space on word alignment
/// (`f"{w['word']} "`).
pub fn decode(text: &str, provider: Option<InferenceProvider>) -> Option<Incoming> {
    if text.len() > MAX_FRAME_BYTES {
        return Some(Incoming::Malformed { detail: "TTS gateway frame exceeds its limit".into() });
    }
    let envelope: MessageType = serde_json::from_str(text).ok()?;
    envelope.kind?;
    let frame: ServerFrame = match serde_json::from_str(text) {
        Ok(frame) => frame,
        Err(_) => {
            return Some(Incoming::Malformed {
                detail: "TTS returned a malformed gateway frame".into(),
            });
        }
    };
    let session_id = frame.session_id;
    Some(match frame.message {
        ServerMessage::SessionCreated => Incoming::SessionCreated { session_id },
        ServerMessage::OutputAudio { audio } => Incoming::OutputAudio { session_id, audio },
        ServerMessage::OutputAlignment { words, chars } => {
            match alignment_tokens(words, chars, provider) {
                Some(tokens) => Incoming::OutputAlignment { session_id, tokens },
                None => Incoming::Malformed { detail: "TTS returned malformed alignment".into() },
            }
        }
        ServerMessage::Done => Incoming::Done { session_id },
        ServerMessage::Error { code, retryable } => Incoming::Error { code, retryable },
        ServerMessage::FallbackActivated { fallback_type, provider, model, voice, cause } => {
            Incoming::FallbackActivated { session_id, fallback_type, provider, model, voice, cause }
        }
        _ => Incoming::Unknown { session_id },
    })
}

/// The largest gateway frame decoded; a larger one is malformed.
const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
struct MessageType {
    #[serde(rename = "type")]
    kind: Option<String>,
}

#[derive(Deserialize)]
struct ServerFrame {
    session_id: Option<String>,
    #[serde(flatten)]
    message: ServerMessage,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ServerMessage {
    #[serde(rename = "session.created")]
    SessionCreated,
    #[serde(rename = "output_audio")]
    OutputAudio { audio: String },
    #[serde(rename = "output_alignment")]
    OutputAlignment { words: Option<Vec<Word>>, chars: Option<Vec<Character>> },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error { code: ErrorCode, retryable: Option<bool> },
    #[serde(rename = "fallback_activated")]
    FallbackActivated {
        fallback_type: FallbackType,
        provider: String,
        model: String,
        voice: String,
        cause: FallbackCause,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct Word {
    word: String,
    #[serde(flatten)]
    time: AlignmentTime,
}

#[derive(Deserialize)]
struct Character {
    char: String,
    #[serde(flatten)]
    time: AlignmentTime,
}

#[derive(Deserialize)]
struct AlignmentTime {
    start: Option<f64>,
    end: Option<f64>,
}

impl AlignmentTime {
    fn token(self, text: String) -> Option<TranscriptChunk> {
        if self.start.into_iter().chain(self.end).any(|t| !t.is_finite() || t < 0.0) {
            return None;
        }
        Some(TranscriptChunk {
            text,
            start_time: self.start,
            end_time: self.end,
            confidence: None,
            start_time_offset: None,
            speaker_id: None,
        })
    }
}

fn alignment_tokens(
    words: Option<Vec<Word>>,
    chars: Option<Vec<Character>>,
    provider: Option<InferenceProvider>,
) -> Option<Vec<TranscriptChunk>> {
    if let Some(words) = words.as_ref().filter(|w| !w.is_empty()) {
        return words
            .iter()
            .map(|w| {
                let text = if provider == Some(InferenceProvider::Cartesia) {
                    format!("{} ", w.word)
                } else {
                    w.word.clone()
                };
                AlignmentTime { start: w.time.start, end: w.time.end }.token(text)
            })
            .collect();
    }
    if let Some(chars) = chars {
        return chars.into_iter().map(|c| c.time.token(c.char)).collect();
    }
    words.map(|_| Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn session_create(opts: &TtsOptions) -> Value {
        super::session_create(opts).unwrap()
    }

    fn input_transcript(text: &str, opts: &TtsOptions) -> Value {
        super::input_transcript(text, opts).unwrap()
    }

    // Catches: R2b on the empty voice.
    #[test]
    fn an_empty_voice_is_omitted_from_both_frames() {
        let opts = TtsOptions {
            model: "cartesia/sonic".into(),
            voice: Some(String::new()),
            ..Default::default()
        };
        assert!(session_create(&opts).get("voice").is_none());
        assert!(input_transcript("hi", &opts)["generation_config"].get("voice").is_none());
        assert_eq!(normalize_fallback_str("cartesia/sonic:").voice, "");
        assert_eq!(normalize_fallback_str("cartesia/sonic:bob").voice, "bob");
    }

    // Catches: R2b on the language, in both frames (including generation_config).
    #[test]
    fn an_empty_language_is_omitted_from_session_create_and_from_every_transcript() {
        let opts = TtsOptions {
            model: "cartesia/sonic-3".into(),
            language: Some(String::new()),
            ..Default::default()
        };
        assert!(session_create(&opts).get("language").is_none());
        assert!(input_transcript("hi", &opts)["generation_config"].get("language").is_none());

        let opts = TtsOptions { language: Some("fr".into()), ..opts };
        assert_eq!(session_create(&opts)["language"], json!("fr"));
        assert_eq!(input_transcript("hi", &opts)["generation_config"]["language"], json!("fr"));
    }

    // Catches: an empty model reaching the wire.
    #[test]
    fn an_empty_model_is_omitted() {
        let opts = TtsOptions::default();
        assert!(session_create(&opts).get("model").is_none());
        assert!(input_transcript("hi", &opts)["generation_config"].get("model").is_none());
    }

    // Catches: the object and array arms of Python truthiness in the alignment flag.
    #[test]
    fn an_empty_container_is_a_falsy_alignment_flag() {
        let mut extra = Map::new();
        for empty in [json!({}), json!([]), json!(0), json!("")] {
            extra.insert("add_timestamps".into(), empty.clone());
            assert!(!has_aligned_transcript("cartesia/sonic-3", &extra), "{empty} must be falsy");
        }
        extra.insert("add_timestamps".into(), json!({"a": 1}));
        assert!(has_aligned_transcript("cartesia/sonic-3", &extra));
    }

    // Catches: normalising the fallback extra to `null` like STT.
    #[test]
    fn a_tts_fallback_without_extras_sends_an_empty_object_not_null() {
        let opts = TtsOptions {
            model: "cartesia/sonic-3".into(),
            fallback: vec![FallbackModel {
                model: "inworld/inworld-tts-1".into(),
                voice: "ashley".into(),
                extra_kwargs: None,
            }],
            ..Default::default()
        };
        assert_eq!(
            session_create(&opts)["fallback"],
            json!({"models": [{"model": "inworld/inworld-tts-1", "voice": "ashley", "extra": {}}]})
        );
    }

    // Catches: R2b on `_has_aligned_transcript` (`{"add_timestamps": false}`).
    #[test]
    fn alignment_capability_follows_the_flags_value_not_its_presence() {
        let mut extra = Map::new();
        extra.insert("add_timestamps".into(), json!(false));
        assert!(!has_aligned_transcript("cartesia/sonic-3", &extra));
        extra.insert("add_timestamps".into(), json!(true));
        assert!(has_aligned_transcript("cartesia/sonic-3", &extra));
        // ...and the flag is per provider: the same key means nothing to Rime.
        assert!(!has_aligned_transcript("rime/mistv3", &extra));

        let mut extra = Map::new();
        extra.insert("timestamp_type".into(), json!("TIMESTAMP_TYPE_UNSPECIFIED"));
        assert!(!has_aligned_transcript("inworld/inworld-tts-2", &extra));
        extra.insert("timestamp_type".into(), json!("CHARACTER"));
        assert!(has_aligned_transcript("inworld/inworld-tts-2", &extra));
    }

    // Catches: R2b on `if words := data.get("words")`.
    #[test]
    fn an_empty_words_list_falls_through_to_character_alignment() {
        let frame = r#"{"type":"output_alignment","words":[],"chars":[{"char":"h","start":0.0,"end":0.1}]}"#;
        let Some(Incoming::OutputAlignment { tokens, .. }) =
            decode(frame, Some(InferenceProvider::Inworld))
        else {
            panic!("not an alignment")
        };
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "h");
    }

    // Catches: dropping Cartesia's re-added trailing space.
    #[test]
    fn cartesia_word_tokens_get_their_space_back_and_other_providers_do_not() {
        let frame =
            r#"{"type":"output_alignment","words":[{"word":"hello","start":0.0,"end":0.4}]}"#;
        let Some(Incoming::OutputAlignment { tokens, .. }) =
            decode(frame, Some(InferenceProvider::Cartesia))
        else {
            panic!()
        };
        assert_eq!(tokens[0].text, "hello ");
        let Some(Incoming::OutputAlignment { tokens, .. }) =
            decode(frame, Some(InferenceProvider::Inworld))
        else {
            panic!()
        };
        assert_eq!(tokens[0].text, "hello");
    }

    // Catches: an older Inworld model mapping to "inworld" instead of "".
    #[test]
    fn older_inworld_models_report_no_markup_dialect() {
        assert_eq!(markup_provider("inworld/inworld-tts-2"), Some(MarkupProvider::Inworld));
        assert_eq!(markup_provider("inworld/inworld-tts-1.5"), None);
        assert_eq!(markup_provider("cartesia/sonic-3"), Some(MarkupProvider::Cartesia));
    }

    #[test]
    fn gateway_options_have_numeric_sample_rates_and_keep_new_provider_fields() {
        let opts = TtsOptions {
            model: "assemblyai/assemblyai-tts-1".into(),
            extra_kwargs: serde_json::from_value(
                json!({"word_timestamps": true, "future_option": 4}),
            )
            .unwrap(),
            ..Default::default()
        };
        let create = session_create(&opts);
        assert_eq!(create["sample_rate"], 24_000);
        assert_eq!(create["extra"]["future_option"], 4);
        assert!(has_aligned_transcript(&opts.model, &opts.extra_kwargs));
        let mut extra = opts.extra_kwargs;
        extra.insert("word_timestamps".into(), Value::Bool(false));
        assert!(!has_aligned_transcript(&opts.model, &extra));
        assert_eq!(markup_provider("fishaudio/drama-3"), Some(MarkupProvider::FishAudioDrama));
    }
}
