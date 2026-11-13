use super::SttOptions;
use super::options::keyterms_extra_for_model;
use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::inference::models::InferenceProvider;
use serde::Serialize;
use serde_json::{Map, Value};

#[derive(Serialize)]
#[serde(tag = "type")]
enum Outgoing<'a> {
    #[serde(rename = "session.create")]
    Create {
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<&'a str>,
        settings: Settings<'a>,
        #[serde(skip_serializing_if = "Vec::is_empty", serialize_with = "serialize_fallback")]
        fallback: Vec<Fallback<'a>>,
        connection: crate::extra::Connection,
    },
    #[serde(rename = "input_audio")]
    Audio { audio: &'a str },
    #[serde(rename = "session.finalize")]
    Finalize,
    #[serde(rename = "session.close")]
    Close,
    #[serde(rename = "session.update")]
    Update { settings: Update<'a> },
}

#[derive(Serialize)]
struct Settings<'a> {
    sample_rate: u32,
    encoding: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    extra: super::provider_options::Options,
}

#[derive(Serialize)]
struct Fallback<'a> {
    model: &'a str,
    extra: super::provider_options::Options,
}

fn serialize_fallback<S: serde::Serializer>(
    models: &[Fallback<'_>],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct Fallbacks<'a, 'b> {
        models: &'a [Fallback<'b>],
    }
    Fallbacks { models }.serialize(serializer)
}

#[derive(Serialize)]
struct Update<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra: Option<super::provider_options::Options>,
}

pub fn session_create(opts: &SttOptions, session_keyterms: &[String]) -> Result<Value, ApiError> {
    let mut extra = opts.extra_kwargs.clone();
    if let Some(keyterms) =
        keyterms_extra_for_model(opts.model.as_deref(), Some(&opts.extra_kwargs), session_keyterms)
    {
        for (k, v) in keyterms {
            extra.insert(k, v);
        }
    }
    let model = opts.model.as_deref().filter(|m| !m.is_empty() && *m != "auto");
    let extra = provider_options(model.unwrap_or(""), opts.language.as_deref(), extra)?;
    let fallback = opts
        .fallback
        .iter()
        .map(|entry| {
            Ok(Fallback {
                model: &entry.model,
                extra: provider_options(
                    &entry.model,
                    opts.language.as_deref(),
                    entry.extra_kwargs.clone().unwrap_or_default(),
                )?,
            })
        })
        .collect::<Result<_, ApiError>>()?;
    Ok(serde_json::to_value(Outgoing::Create {
        model,
        settings: Settings {
            sample_rate: opts.sample_rate,
            encoding: &opts.encoding,
            language: opts.language.as_deref().filter(|language| !language.is_empty()),
            extra,
        },
        fallback,
        connection: (&opts.conn_options).into(),
    })
    .expect("STT request serialization"))
}

fn provider_options(
    model: &str,
    language: Option<&str>,
    mut extra: Map<String, Value>,
) -> Result<super::provider_options::Options, ApiError> {
    if InferenceProvider::of_model(model) == Some(InferenceProvider::Deepgram) {
        extra.remove("callback");
        extra.remove("callback_method");
        if !model.starts_with("deepgram/flux-") && language != Some("multi") {
            extra.remove("language_hint");
        }
    }
    super::provider_options::Options::parse(model, extra)
        .map_err(|_| ApiError::new(ErrorKind::InvalidRequest, "Invalid STT provider options"))
}

pub fn input_audio(base64_pcm: &str) -> Value {
    serde_json::to_value(Outgoing::Audio { audio: base64_pcm }).expect("STT audio serialization")
}

/// `{"type": "session.finalize"}` -- sent when the input ends, and by the VAD task on every end
/// of speech.
pub fn session_finalize() -> Value {
    serde_json::to_value(Outgoing::Finalize).expect("STT finalize serialization")
}

pub fn session_close() -> Value {
    serde_json::to_value(Outgoing::Close).expect("STT close serialization")
}

pub fn session_update(
    provider_model: &str,
    provider_language: Option<&str>,
    model: Option<&str>,
    language: Option<&str>,
    extra: Option<&Map<String, Value>>,
) -> Result<Option<Value>, ApiError> {
    if model.is_none() && language.is_none() && extra.is_none() {
        return Ok(None);
    }
    let extra = extra
        .map(|extra| provider_options(provider_model, provider_language, extra.clone()))
        .transpose()?;
    Ok(Some(
        serde_json::to_value(Outgoing::Update { settings: Update { model, language, extra } })
            .expect("STT update serialization"),
    ))
}

/// Known transcript fields are validated before the protocol sees a frame. `extra` remains
/// provider-owned metadata.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Transcript {
    pub transcript: String,
    pub request_id: Option<String>,
    pub language: Option<String>,
    #[serde(default)]
    pub start: f64,
    #[serde(default)]
    pub duration: f64,
    #[serde(default = "utterance_confidence")]
    pub confidence: f64,
    pub speaker_id: Option<String>,
    #[serde(default, deserialize_with = "crate::llm::parse::null_as_empty")]
    pub words: Vec<Word>,
    pub extra: Option<Map<String, Value>>,
}

fn utterance_confidence() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Word {
    pub word: String,
    #[serde(default)]
    pub start: f64,
    #[serde(default)]
    pub end: f64,
    #[serde(default)]
    pub confidence: f64,
    pub speaker_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(tag = "type")]
pub enum Incoming {
    #[serde(rename = "session.created")]
    SessionCreated,
    #[serde(rename = "session.finalized")]
    SessionFinalized {
        #[serde(default)]
        truncated: bool,
    },
    #[serde(rename = "session.closed")]
    SessionClosed {
        #[serde(default)]
        truncated: bool,
    },
    #[serde(rename = "session.updated")]
    SessionUpdated { success: bool },
    #[serde(rename = "start_of_speech")]
    StartOfSpeech,
    #[serde(rename = "interim_transcript")]
    InterimTranscript(Transcript),
    #[serde(rename = "preflight_transcript")]
    PreflightTranscript(Transcript),
    #[serde(rename = "final_transcript")]
    FinalTranscript(Transcript),
    #[serde(rename = "error")]
    /// Its `message` can carry customer content: only the code is read.
    Error {
        #[serde(default = "missing_code")]
        code: i64,
    },
    #[serde(other)]
    Unknown,
}

fn missing_code() -> i64 {
    -1
}

pub fn decode(text: &str) -> Option<Incoming> {
    // Missing types are forward-compatible control frames; malformed known fields are failures.
    #[derive(serde::Deserialize)]
    struct Envelope {
        #[serde(rename = "type")]
        kind: Option<String>,
    }
    let envelope: Envelope = serde_json::from_str(text).ok()?;
    if envelope.kind.is_none() {
        return Some(Incoming::Unknown);
    }
    serde_json::from_str(text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::options::FallbackModel;
    use serde_json::json;

    // Catches: sending `model: "auto"` to the gateway.
    #[test]
    fn the_auto_model_is_not_sent_as_a_model_name() {
        let opts = SttOptions { model: Some("auto".into()), ..Default::default() };
        let frame = session_create(&opts, &[]).unwrap();
        assert!(frame.get("model").is_none());

        let opts = SttOptions { model: Some("deepgram/nova-3".into()), ..Default::default() };
        assert_eq!(session_create(&opts, &[]).unwrap()["model"], json!("deepgram/nova-3"));
    }

    // Catches: R2b on `if self._opts.language:`.
    #[test]
    fn an_empty_language_is_not_sent() {
        let opts = SttOptions {
            model: Some("deepgram/nova-3".into()),
            language: Some(String::new()),
            ..Default::default()
        };
        assert!(session_create(&opts, &[]).unwrap()["settings"].get("language").is_none());
    }

    // Catches: omitting the `connection` block when the caller configured no connect options, or
    // sending a disabled connect limit (the gateway refuses `null`).
    #[test]
    fn the_connection_block_carries_the_connect_limit_and_retries() {
        let frame = session_create(&SttOptions::default(), &[]).unwrap();
        assert_eq!(frame["connection"], json!({"timeout": 10.0, "retries": 3}));
        let opts = SttOptions {
            conn_options: lk_agents_abi::ApiConnectOptions {
                connect_timeout: None,
                max_retry: 1,
                ..Default::default()
            },
            ..SttOptions::default()
        };
        assert_eq!(session_create(&opts, &[]).unwrap()["connection"], json!({"retries": 1}));
    }

    #[test]
    fn the_sample_rate_matches_the_server_integer_field() {
        let frame = session_create(&SttOptions::default(), &[]).unwrap();
        assert_eq!(frame["settings"]["sample_rate"], json!(16000));
    }

    #[test]
    fn a_fallback_with_no_extra_kwargs_sends_an_empty_object() {
        let opts = SttOptions {
            model: Some("deepgram/nova-3".into()),
            fallback: vec![FallbackModel {
                model: "cartesia/ink-whisper".into(),
                extra_kwargs: None,
            }],
            ..Default::default()
        };
        let frame = session_create(&opts, &[]).unwrap();
        assert_eq!(
            frame["fallback"],
            json!({"models": [{"model": "cartesia/ink-whisper", "extra": {}}]})
        );
    }

    // Catches: session keyterms not merged last over the user's `extra`, or not at all.
    #[test]
    fn session_keyterms_merge_into_extra_without_disturbing_the_rest() {
        let mut extra = Map::new();
        extra.insert("punctuate".into(), json!(true));
        extra.insert("keyterm".into(), json!(["mine"]));
        let opts = SttOptions {
            model: Some("deepgram/nova-3".into()),
            extra_kwargs: extra,
            ..Default::default()
        };
        let frame = session_create(&opts, &["theirs".into()]).unwrap();
        assert_eq!(frame["settings"]["extra"]["punctuate"], json!(true));
        assert_eq!(frame["settings"]["extra"]["keyterm"], json!(["mine", "theirs"]));
    }

    // Catches: sending an empty `session.update`.
    #[test]
    fn an_update_with_nothing_given_is_not_sent() {
        assert!(session_update("", None, None, None, None).unwrap().is_none());
        let frame = session_update("deepgram/nova-3", None, Some("deepgram/nova-3"), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            frame,
            json!({"type": "session.update", "settings": {"model": "deepgram/nova-3"}})
        );
    }

    // Catches: treating an unrecognised message type as an error.
    #[test]
    fn an_unrecognised_message_type_is_ignored() {
        assert_eq!(decode(r#"{"type": "future_thing"}"#), Some(Incoming::Unknown));
        assert_eq!(decode(r#"{}"#), Some(Incoming::Unknown));
        assert_eq!(decode("not json"), None);
    }

    // Catches: dropping the `-1` sentinel for an error with no code.
    #[test]
    fn an_error_without_a_code_reports_minus_one() {
        let got = decode(r#"{"type": "error", "message": "nope"}"#).unwrap();
        match got {
            Incoming::Error { code } => assert_eq!(code, -1),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn deepgram_hints_only_reach_multilingual_nova_or_flux() {
        for (model, language, expected) in [
            ("deepgram/nova-3", Some("en"), false),
            ("deepgram/nova-3", Some("multi"), true),
            ("deepgram/flux-general-multi", None, true),
        ] {
            let extra = serde_json::from_value(json!({
                "language_hint": ["fr"], "callback": "https://ignored", "callback_method": "POST"
            }))
            .unwrap();
            let opts = SttOptions {
                model: Some(model.into()),
                language: language.map(str::to_owned),
                extra_kwargs: extra,
                ..Default::default()
            };
            let frame = session_create(&opts, &[]).unwrap();
            assert_eq!(frame["settings"]["extra"].get("language_hint").is_some(), expected);
            assert!(frame["settings"]["extra"].get("callback").is_none());
            assert!(frame["settings"]["extra"].get("callback_method").is_none());
            let update = session_update(model, language, None, None, Some(&opts.extra_kwargs))
                .unwrap()
                .unwrap();
            assert_eq!(update["settings"]["extra"].get("language_hint").is_some(), expected);
        }
    }

    #[test]
    fn invalid_known_options_are_refused_without_customer_text() {
        let opts = SttOptions {
            model: Some("openai/gpt-live-transcribe".into()),
            extra_kwargs: serde_json::from_value(json!({"delay": "customer words"})).unwrap(),
            ..Default::default()
        };
        let error = session_create(&opts, &[]).unwrap_err();
        assert!(!error.retryable);
        assert!(!error.message.contains("customer"));
    }
}
