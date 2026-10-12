//! The TTS gateway's JSON protocol. It differs from STT's on purpose (`"type"` first; fallback
//! extra defaults to `{}`), so the two are not shared.

use super::TtsOptions;
use lk_agents_abi::TimedString;
use serde_json::{Map, Value, json};

pub const DEFAULT_ENCODING: &str = "pcm_s16le";
pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;

pub use lk_agents_abi::inference::fallback::TtsFallbackModel as FallbackModel;

/// `"provider/model:voice"` -> `("provider/model", Some("voice"))`.
#[lk_agents_abi::host(export(name = "tts_parse_model_string"))]
pub fn parse_model_string(model: &str) -> (&str, Option<&str>) {
    crate::stt::options::parse_model_string(model)
}

/// `_normalize_fallback`'s string arm.
#[lk_agents_abi::host(export(name = "tts_normalize_fallback_str"))]
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
#[lk_agents_abi::host(export(name = "tts_has_aligned_transcript"))]
pub fn has_aligned_transcript(model: &str, extra_kwargs: &Map<String, Value>) -> bool {
    let provider = model.split('/').next().unwrap_or("");
    match provider {
        // `bool(extra_kwargs.get("add_timestamps"))` -- `false`, `0` and `""` are all "no".
        "cartesia" => truthy(extra_kwargs.get("add_timestamps")),
        "elevenlabs" => truthy(extra_kwargs.get("sync_alignment")),
        "inworld" => matches!(
            extra_kwargs.get("timestamp_type").and_then(Value::as_str),
            Some("WORD") | Some("CHARACTER")
        ),
        _ => false,
    }
}

fn truthy(value: Option<&Value>) -> bool {
    value.is_some_and(|v| !crate::stt::options::is_falsy(v))
}

/// `TTS.Markup._provider_key()`: the markup dialect; "" means the model has no markup.
#[lk_agents_abi::host(export(name = "tts_markup_provider_key"))]
pub fn markup_provider_key(model: &str) -> &str {
    let provider = model.split('/').next().unwrap_or("");
    if provider == "inworld" {
        return if model.contains("tts-2") { "inworld" } else { "" };
    }
    provider
}

/// The opening `session.create` frame. `"type"` first, unlike STT's.
pub fn session_create(opts: &TtsOptions) -> Value {
    let mut params = Map::new();
    params.insert("type".into(), json!("session.create"));
    params.insert("sample_rate".into(), json!(opts.sample_rate.to_string()));
    params.insert("encoding".into(), json!(opts.encoding));
    params.insert("extra".into(), Value::Object(opts.extra_kwargs.clone()));

    // An empty voice / model / language sends no key.
    if let Some(voice) = opts.voice.as_deref()
        && !voice.is_empty()
    {
        params.insert("voice".into(), json!(voice));
    }
    if !opts.model.is_empty() {
        params.insert("model".into(), json!(opts.model));
    }
    if let Some(language) = opts.language.as_deref()
        && !language.is_empty()
    {
        params.insert("language".into(), json!(language));
    }
    if !opts.fallback.is_empty() {
        let models: Vec<Value> = opts
            .fallback
            .iter()
            .map(|f| {
                json!({
                    "model": f.model,
                    "voice": f.voice,
                    // `m.get("extra_kwargs", {})` -- **with** a default here, unlike STT's.
                    "extra": f.extra_kwargs.clone().map(Value::Object).unwrap_or_else(|| json!({})),
                })
            })
            .collect();
        params.insert("fallback".into(), json!({"models": models}));
    }
    params.insert(
        "connection".into(),
        json!({"timeout": opts.conn_options.timeout, "retries": opts.conn_options.max_retry}),
    );
    Value::Object(params)
}

/// One sentence of input. The trailing space matters: the gateway concatenates transcripts.
pub fn input_transcript(text: &str, opts: &TtsOptions) -> Value {
    let mut generation_config = Map::new();
    if let Some(voice) = opts.voice.as_deref()
        && !voice.is_empty()
    {
        generation_config.insert("voice".into(), json!(voice));
    }
    if !opts.model.is_empty() {
        generation_config.insert("model".into(), json!(opts.model));
    }
    if let Some(language) = opts.language.as_deref()
        && !language.is_empty()
    {
        generation_config.insert("language".into(), json!(language));
    }

    let mut pkt = Map::new();
    pkt.insert("type".into(), json!("input_transcript"));
    pkt.insert("transcript".into(), json!(format!("{text} ")));
    pkt.insert("generation_config".into(), Value::Object(generation_config));
    pkt.insert("extra".into(), Value::Object(opts.extra_kwargs.clone()));
    Value::Object(pkt)
}

/// `{"type": "session.flush"}` -- sent once the sentence stream ends, including when the input was
/// empty.
pub fn session_flush() -> Value {
    json!({"type": "session.flush"})
}

#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    SessionCreated { session_id: Option<String> },
    OutputAudio { session_id: Option<String>, audio: String },
    OutputAlignment { session_id: Option<String>, tokens: Vec<TimedString> },
    Done { session_id: Option<String> },
    Error { raw: String },
    Unknown { session_id: Option<String> },
}

impl Incoming {
    /// Every message can carry the session id, and the *first* one that does starts the segment.
    pub fn session_id(&self) -> Option<&str> {
        match self {
            Incoming::SessionCreated { session_id }
            | Incoming::OutputAudio { session_id, .. }
            | Incoming::OutputAlignment { session_id, .. }
            | Incoming::Done { session_id }
            | Incoming::Unknown { session_id } => session_id.as_deref(),
            Incoming::Error { .. } => None,
        }
    }
}

/// Decode one text frame. `provider` selects Cartesia's re-added trailing space on word alignment
/// (`f"{w['word']} "`).
pub fn decode(text: &str, provider: &str) -> Option<Incoming> {
    let value: Value = serde_json::from_str(text).ok()?;
    let obj = value.as_object()?;
    let session_id = obj.get("session_id").and_then(Value::as_str).map(str::to_owned);

    Some(match obj.get("type").and_then(Value::as_str) {
        Some("session.created") => Incoming::SessionCreated { session_id },
        Some("output_audio") => Incoming::OutputAudio {
            session_id,
            audio: obj.get("audio").and_then(Value::as_str).unwrap_or("").to_owned(),
        },
        Some("output_alignment") => {
            Incoming::OutputAlignment { session_id, tokens: alignment_tokens(obj, provider) }
        }
        Some("done") => Incoming::Done { session_id },
        Some("error") => Incoming::Error { raw: text.to_owned() },
        _ => Incoming::Unknown { session_id },
    })
}

fn alignment_tokens(obj: &Map<String, Value>, provider: &str) -> Vec<TimedString> {
    let token = |text: String, v: &Value| TimedString {
        text,
        start_time: v.get("start").and_then(Value::as_f64),
        end_time: v.get("end").and_then(Value::as_f64),
        confidence: None,
        start_time_offset: None,
        speaker_id: None,
    };
    let str_of = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).unwrap_or("").to_owned();
    // `if words := data.get("words")` -- an empty list falls through to `chars`.
    let non_empty = |key: &str| obj.get(key).and_then(Value::as_array).filter(|a| !a.is_empty());
    if let Some(words) = non_empty("words") {
        return words
            .iter()
            .map(|w| {
                let word = str_of(w, "word");
                token(if provider == "cartesia" { format!("{word} ") } else { word }, w)
            })
            .collect();
    }
    non_empty("chars")
        .map(|chars| chars.iter().map(|c| token(str_of(c, "char"), c)).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

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
            extra.insert("sync_alignment".into(), empty.clone());
            assert!(
                !has_aligned_transcript("elevenlabs/eleven_flash_v2", &extra),
                "{empty} must be falsy"
            );
        }
        extra.insert("sync_alignment".into(), json!({"a": 1}));
        assert!(has_aligned_transcript("elevenlabs/eleven_flash_v2", &extra));
    }

    // Catches: normalising the fallback extra to `null` like STT.
    #[test]
    fn a_tts_fallback_without_extras_sends_an_empty_object_not_null() {
        let opts = TtsOptions {
            model: "cartesia/sonic-3".into(),
            fallback: vec![FallbackModel {
                model: "elevenlabs/eleven_flash_v2".into(),
                voice: "rachel".into(),
                extra_kwargs: None,
            }],
            ..Default::default()
        };
        assert_eq!(
            session_create(&opts)["fallback"],
            json!({"models": [{"model": "elevenlabs/eleven_flash_v2", "voice": "rachel", "extra": {}}]})
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
        // ...and the flag is per provider: the same key means nothing to ElevenLabs.
        assert!(!has_aligned_transcript("elevenlabs/eleven_flash_v2", &extra));

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
        let Some(Incoming::OutputAlignment { tokens, .. }) = decode(frame, "elevenlabs") else {
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
        let Some(Incoming::OutputAlignment { tokens, .. }) = decode(frame, "cartesia") else {
            panic!()
        };
        assert_eq!(tokens[0].text, "hello ");
        let Some(Incoming::OutputAlignment { tokens, .. }) = decode(frame, "elevenlabs") else {
            panic!()
        };
        assert_eq!(tokens[0].text, "hello");
    }

    // Catches: an older Inworld model mapping to "inworld" instead of "".
    #[test]
    fn older_inworld_models_report_no_markup_dialect() {
        assert_eq!(markup_provider_key("inworld/inworld-tts-2"), "inworld");
        assert_eq!(markup_provider_key("inworld/inworld-tts-1.5"), "");
        assert_eq!(markup_provider_key("cartesia/sonic-3"), "cartesia");
    }
}
