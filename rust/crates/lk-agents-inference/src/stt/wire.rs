//! The STT gateway's JSON protocol: the frames the client sends and the messages it receives.

use super::SttOptions;
use super::options::keyterms_extra_for_model;
use serde_json::{Map, Value, json};

/// The opening `session.create` frame; `"type"` is the last key, as in the reference.
pub fn session_create(opts: &SttOptions, session_keyterms: &[String]) -> Value {
    let mut settings = Map::new();
    // `str(sample_rate)`, as the captured frames have it.
    settings.insert("sample_rate".into(), json!(opts.sample_rate.to_string()));
    settings.insert("encoding".into(), json!(opts.encoding));

    let mut extra = opts.extra_kwargs.clone();
    if let Some(keyterms) =
        keyterms_extra_for_model(opts.model.as_deref(), Some(&opts.extra_kwargs), session_keyterms)
    {
        for (k, v) in keyterms {
            extra.insert(k, v);
        }
    }
    settings.insert("extra".into(), Value::Object(extra));

    let mut params = Map::new();
    params.insert("settings".into(), Value::Object(settings));

    // `if self._opts.model and self._opts.model != "auto"` -- neither "" nor "auto" is sent.
    if let Some(model) = opts.model.as_deref()
        && !model.is_empty()
        && model != "auto"
    {
        params.insert("model".into(), json!(model));
    }

    // `if self._opts.language:` -- an empty language is not sent.
    if let Some(language) = opts.language.as_deref()
        && !language.is_empty()
    {
        params
            .get_mut("settings")
            .and_then(Value::as_object_mut)
            .expect("settings was just inserted")
            .insert("language".into(), json!(language));
    }

    // `if self._opts.fallback:` -- an empty list sends no `fallback` key.
    if !opts.fallback.is_empty() {
        let models: Vec<Value> = opts
            .fallback
            .iter()
            .map(|f| {
                // `m.get("extra_kwargs")` with no default: `"extra": null` (TTS passes `{}`).
                json!({
                    "model": f.model,
                    "extra": f.extra_kwargs.clone().map(Value::Object).unwrap_or(Value::Null),
                })
            })
            .collect();
        params.insert("fallback".into(), json!({"models": models}));
    }

    params.insert(
        "connection".into(),
        json!({"timeout": opts.conn_options.timeout, "retries": opts.conn_options.max_retry}),
    );

    params.insert("type".into(), json!("session.create"));
    Value::Object(params)
}

pub fn input_audio(base64_pcm: &str) -> Value {
    json!({"type": "input_audio", "audio": base64_pcm})
}

/// `{"type": "session.finalize"}` -- sent when the input ends, and by the VAD task on every end
/// of speech.
pub fn session_finalize() -> Value {
    json!({"type": "session.finalize"})
}

pub fn session_update(
    model: Option<&str>,
    language: Option<&str>,
    extra: Option<&Map<String, Value>>,
) -> Option<Value> {
    let mut settings = Map::new();
    if let Some(model) = model {
        settings.insert("model".into(), json!(model));
    }
    if let Some(language) = language {
        settings.insert("language".into(), json!(language));
    }
    if let Some(extra) = extra {
        settings.insert("extra".into(), Value::Object(extra.clone()));
    }
    if settings.is_empty() {
        return None;
    }
    Some(json!({"type": "session.update", "settings": settings}))
}

/// One decoded message from the gateway. `Unknown` is ignored: the gateway adds types ahead of
/// the SDKs.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    StartOfSpeech,
    InterimTranscript(Map<String, Value>),
    PreflightTranscript(Map<String, Value>),
    FinalTranscript(Map<String, Value>),
    Error { message: Option<String>, code: i64 },
    Unknown,
}

/// Decode one text frame. A frame that is not a JSON object is `None` (a stream-ending error).
pub fn decode(text: &str) -> Option<Incoming> {
    let value: Value = serde_json::from_str(text).ok()?;
    let obj = value.as_object()?.clone();
    Some(match obj.get("type").and_then(Value::as_str) {
        Some("start_of_speech") => Incoming::StartOfSpeech,
        Some("interim_transcript") => Incoming::InterimTranscript(obj),
        Some("preflight_transcript") => Incoming::PreflightTranscript(obj),
        Some("final_transcript") => Incoming::FinalTranscript(obj),
        Some("error") => Incoming::Error {
            message: obj.get("message").and_then(Value::as_str).map(str::to_owned),
            // `data.get("code", -1)` -- the reference's sentinel for "the gateway did not say".
            code: obj.get("code").and_then(Value::as_i64).unwrap_or(-1),
        },
        _ => Incoming::Unknown,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::options::FallbackModel;

    // Catches: sending `model: "auto"` to the gateway.
    #[test]
    fn the_auto_model_is_not_sent_as_a_model_name() {
        let opts = SttOptions { model: Some("auto".into()), ..Default::default() };
        let frame = session_create(&opts, &[]);
        assert!(frame.get("model").is_none());

        let opts = SttOptions { model: Some("deepgram/nova-3".into()), ..Default::default() };
        assert_eq!(session_create(&opts, &[])["model"], json!("deepgram/nova-3"));
    }

    // Catches: R2b on `if self._opts.language:`.
    #[test]
    fn an_empty_language_is_not_sent() {
        let opts = SttOptions {
            model: Some("deepgram/nova-3".into()),
            language: Some(String::new()),
            ..Default::default()
        };
        assert!(session_create(&opts, &[])["settings"].get("language").is_none());
    }

    // Catches: omitting the `connection` block when the caller configured no connect options.
    #[test]
    fn the_connection_block_is_always_sent() {
        let frame = session_create(&SttOptions::default(), &[]);
        let defaults = lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS;
        assert_eq!(
            frame["connection"],
            json!({"timeout": defaults.timeout, "retries": defaults.max_retry})
        );
    }

    // Catches: sending the sample rate as a number.
    #[test]
    fn the_sample_rate_is_a_string() {
        let frame = session_create(&SttOptions::default(), &[]);
        assert_eq!(frame["settings"]["sample_rate"], json!("16000"));
    }

    // Catches: "normalising" the fallback entry's missing extra to `{}`.
    #[test]
    fn a_fallback_with_no_extra_kwargs_sends_null_not_an_empty_object() {
        let opts = SttOptions {
            model: Some("deepgram/nova-3".into()),
            fallback: vec![FallbackModel {
                model: "cartesia/ink-whisper".into(),
                extra_kwargs: None,
            }],
            ..Default::default()
        };
        let frame = session_create(&opts, &[]);
        assert_eq!(
            frame["fallback"],
            json!({"models": [{"model": "cartesia/ink-whisper", "extra": null}]})
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
        let frame = session_create(&opts, &["theirs".into()]);
        assert_eq!(frame["settings"]["extra"]["punctuate"], json!(true));
        assert_eq!(frame["settings"]["extra"]["keyterm"], json!(["mine", "theirs"]));
    }

    // Catches: sending an empty `session.update`.
    #[test]
    fn an_update_with_nothing_given_is_not_sent() {
        assert!(session_update(None, None, None).is_none());
        let frame = session_update(Some("deepgram/nova-3"), None, None).unwrap();
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
            Incoming::Error { message, code, .. } => {
                assert_eq!(message.as_deref(), Some("nope"));
                assert_eq!(code, -1);
            }
            other => panic!("{other:?}"),
        }
    }
}
