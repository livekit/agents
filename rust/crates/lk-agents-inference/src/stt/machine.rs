//! `SpeechStream`'s `_process_*` methods: gateway messages in, `SpeechEvent`s out, as a clock-free
//! accumulator. `speaking` is written from three places, and its end-of-speech edge releases a
//! deferred keyterm update.

use super::wire::Incoming;
use lk_agents_abi::{RecognitionUsage, SpeechData, SpeechEvent, SpeechEventType, TimedString};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Processed {
    pub events: Vec<SpeechEvent>,
    /// The `_on_end_of_speech()` edge: apply a deferred keyterm update (reported, not performed).
    pub end_of_speech: bool,
    /// The gateway sent `{"type": "error"}`; the stream must fail with this.
    pub error: Option<lk_agents_abi::errors::ApiError>,
}

#[derive(Debug, Clone)]
pub struct TranscriptMachine {
    speaking: bool,
    speech_duration: f64,
    /// `self._request_id`, a `shortuuid("stt_request_")` replaced by the connection's session id
    /// once the socket reports one.
    request_id: String,
    /// `self._opts.language or "en"` — the fallback when a transcript carries no language.
    default_language: String,
}

impl TranscriptMachine {
    pub fn new(request_id: impl Into<String>, language: Option<&str>) -> Self {
        TranscriptMachine {
            speaking: false,
            speech_duration: 0.0,
            request_id: request_id.into(),
            // `self._opts.language or "en"`.
            default_language: match language {
                Some(l) if !l.is_empty() => l.to_owned(),
                _ => "en".to_owned(),
            },
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    /// `_speech_duration` is accumulated on the send side.
    pub fn record_audio(&mut self, duration: f64) {
        self.speech_duration += duration;
    }

    pub fn process(&mut self, msg: Incoming) -> Processed {
        match msg {
            Incoming::StartOfSpeech => {
                let mut out = Processed::default();
                self.start_of_speech(&mut out.events);
                out
            }
            Incoming::PreflightTranscript(data) => self.preflight(&data),
            Incoming::InterimTranscript(data) => self.transcript(&data, false),
            Incoming::FinalTranscript(data) => self.transcript(&data, true),
            Incoming::Error { message, code, .. } => Processed {
                error: Some(lk_agents_abi::errors::ApiError::status(
                    format!(
                        "LiveKit Inference STT returned error: {}",
                        message.as_deref().unwrap_or("None")
                    ),
                    code,
                    None,
                    None,
                )),
                ..Default::default()
            },
            // `session.created`, `session.finalized`, `session.closed` and anything unrecognised
            // are all `pass` in the reference.
            _ => Processed::default(),
        }
    }

    /// Onset reported by a provider that detects it server-side (e.g. Cartesia Ink-2).
    fn start_of_speech(&mut self, events: &mut Vec<SpeechEvent>) {
        if self.speaking {
            return;
        }
        self.speaking = true;
        // No request id: the reference constructs this event with the kind alone.
        events.push(SpeechEvent::new(SpeechEventType::StartOfSpeech));
    }

    fn preflight(&mut self, data: &Map<String, Value>) -> Processed {
        let text = data.get("transcript").and_then(Value::as_str).unwrap_or("");
        // Drop an empty preflight, and any preflight outside a turn (a speculative final).
        if text.is_empty() || !self.speaking {
            return Processed::default();
        }
        let event = SpeechEvent {
            kind: SpeechEventType::PreflightTranscript,
            request_id: self.request_id_of(data),
            alternatives: vec![self.speech_data(data)],
            recognition_usage: None,
            speech_start_time: None,
        };
        Processed { events: vec![event], ..Default::default() }
    }

    fn transcript(&mut self, data: &Map<String, Value>, is_final: bool) -> Processed {
        let request_id = self.request_id_of(data);
        let text = data.get("transcript").and_then(Value::as_str).unwrap_or("");

        // `if not text and not is_final: return`. An empty final still closes the turn.
        if text.is_empty() && !is_final {
            return Processed::default();
        }

        let mut out = Processed::default();
        self.start_of_speech(&mut out.events);
        let speech_data = self.speech_data(data);

        if !is_final {
            out.events.push(SpeechEvent {
                kind: SpeechEventType::InterimTranscript,
                request_id,
                alternatives: vec![speech_data],
                recognition_usage: None,
                speech_start_time: None,
            });
            return out;
        }

        // Usage goes before the final transcript, and only when audio was pushed.
        if self.speech_duration > 0.0 {
            out.events.push(SpeechEvent {
                kind: SpeechEventType::RecognitionUsage,
                request_id: request_id.clone(),
                alternatives: Vec::new(),
                recognition_usage: Some(RecognitionUsage {
                    audio_duration: self.speech_duration,
                    ..Default::default()
                }),
                speech_start_time: None,
            });
            self.speech_duration = 0.0;
        }

        out.events.push(SpeechEvent {
            kind: SpeechEventType::FinalTranscript,
            request_id,
            alternatives: vec![speech_data],
            recognition_usage: None,
            speech_start_time: None,
        });

        if self.speaking {
            self.speaking = false;
            out.events.push(SpeechEvent::new(SpeechEventType::EndOfSpeech));
            out.end_of_speech = true;
        }
        out
    }

    fn request_id_of(&self, data: &Map<String, Value>) -> String {
        data.get("request_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| self.request_id.clone())
    }

    /// `_build_speech_data`.
    fn speech_data(&self, data: &Map<String, Value>) -> SpeechData {
        let num = |key: &str| data.get(key).and_then(Value::as_f64);
        let start = num("start").unwrap_or(0.0);
        let duration = num("duration").unwrap_or(0.0);
        let language =
            data.get("language").and_then(Value::as_str).unwrap_or(&self.default_language);

        let words: Vec<TimedString> = data
            .get("words")
            .and_then(Value::as_array)
            .map(|words| {
                words
                    .iter()
                    .map(|w| TimedString {
                        text: w.get("word").and_then(Value::as_str).unwrap_or("").to_owned(),
                        // `stt.SpeechStream.start_time_offset` is never set on this path: 0.
                        start_time: Some(w.get("start").and_then(Value::as_f64).unwrap_or(0.0)),
                        end_time: Some(w.get("end").and_then(Value::as_f64).unwrap_or(0.0)),
                        // 0.0, unlike the utterance-level 1.0, as in the reference.
                        confidence: Some(
                            w.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
                        ),
                        start_time_offset: Some(0.0),
                        speaker_id: w.get("speaker_id").and_then(Value::as_str).map(str::to_owned),
                    })
                    .collect()
            })
            .unwrap_or_default();

        SpeechData {
            language: lk_agents_abi::tokenize::language::LanguageCode::new(language),
            text: data.get("transcript").and_then(Value::as_str).unwrap_or("").to_owned(),
            start_time: start,
            end_time: start + duration,
            confidence: num("confidence").unwrap_or(1.0),
            speaker_id: data.get("speaker_id").and_then(Value::as_str).map(str::to_owned),
            // An absent `words` is an empty list, not `None`.
            words: Some(words),
            source_languages: None,
            source_texts: None,
            target_languages: None,
            target_texts: None,
            // An empty `extra` object is `None`, not `Some({})`.
            metadata: data
                .get("extra")
                .and_then(Value::as_object)
                .filter(|m| !m.is_empty())
                .cloned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msg(json: Value) -> Incoming {
        super::super::wire::decode(&json.to_string()).unwrap()
    }

    /// The final transcript, which is never `events[0]` -- the inferred START_OF_SPEECH is.
    fn final_of(p: &Processed) -> &SpeechEvent {
        p.events.iter().find(|e| e.kind == SpeechEventType::FinalTranscript).expect("a final")
    }

    fn kinds(p: &Processed) -> Vec<SpeechEventType> {
        p.events.iter().map(|e| e.kind).collect()
    }

    // Catches: dropping an empty final transcript along with the empty interims.
    #[test]
    fn an_empty_final_transcript_still_ends_the_turn() {
        let mut m = TranscriptMachine::new("stt_request_x", None);
        m.record_audio(1.5);
        m.process(msg(json!({"type": "start_of_speech"})));
        let out = m.process(msg(json!({"type": "final_transcript", "transcript": ""})));
        assert_eq!(
            kinds(&out),
            [
                SpeechEventType::RecognitionUsage,
                SpeechEventType::FinalTranscript,
                SpeechEventType::EndOfSpeech
            ]
        );
        assert!(out.end_of_speech);
        assert!(!m.is_speaking());

        // ...while an empty interim produces nothing at all.
        let mut m = TranscriptMachine::new("r", None);
        assert!(
            m.process(msg(json!({"type": "interim_transcript", "transcript": ""})))
                .events
                .is_empty()
        );
    }

    // Catches: swallowing the gateway's `error` frame.
    #[test]
    fn a_gateway_error_frame_becomes_a_terminal_error() {
        let mut m = TranscriptMachine::new("r", None);
        let out = m.process(msg(json!({"type": "error", "message": "bad model", "code": 400})));
        let err = out.error.expect("an error");
        assert_eq!(err.status_code, Some(400));
        assert!(err.message.contains("bad model"), "{}", err.message);
        assert!(!err.retryable, "a 400 is not retried");
        assert!(out.events.is_empty());

        // ...and one with no message still reports, rather than being dropped for lack of detail.
        let out = m.process(msg(json!({"type": "error"})));
        let err = out.error.expect("an error");
        assert_eq!(err.status_code, Some(-1));
    }

    // Catches: emitting START_OF_SPEECH twice (missing `if self._speaking: return`).
    #[test]
    fn start_of_speech_is_emitted_once_per_turn() {
        let mut m = TranscriptMachine::new("r", None);
        assert_eq!(
            kinds(&m.process(msg(json!({"type": "start_of_speech"})))),
            [SpeechEventType::StartOfSpeech]
        );
        assert!(m.process(msg(json!({"type": "start_of_speech"}))).events.is_empty());
        let out = m.process(msg(json!({"type": "interim_transcript", "transcript": "hi"})));
        assert_eq!(kinds(&out), [SpeechEventType::InterimTranscript], "no second onset");
    }

    // Catches: the first transcript not opening the turn, or opening it after the transcript.
    #[test]
    fn the_first_transcript_opens_the_turn_when_the_provider_sends_no_onset() {
        let mut m = TranscriptMachine::new("r", None);
        let out = m.process(msg(json!({"type": "interim_transcript", "transcript": "hel"})));
        assert_eq!(
            kinds(&out),
            [SpeechEventType::StartOfSpeech, SpeechEventType::InterimTranscript]
        );
    }

    // Catches: forwarding a preflight transcript outside a turn.
    #[test]
    fn a_preflight_outside_a_turn_is_dropped() {
        let mut m = TranscriptMachine::new("r", None);
        assert!(
            m.process(msg(json!({"type": "preflight_transcript", "transcript": "hi"})))
                .events
                .is_empty()
        );
        m.process(msg(json!({"type": "start_of_speech"})));
        let out = m.process(msg(json!({"type": "preflight_transcript", "transcript": "hi"})));
        assert_eq!(kinds(&out), [SpeechEventType::PreflightTranscript]);
        // ...and an empty one is dropped even inside a turn.
        assert!(
            m.process(msg(json!({"type": "preflight_transcript", "transcript": ""})))
                .events
                .is_empty()
        );
    }

    // Catches: a zero-duration usage record, or not resetting the accumulator.
    #[test]
    fn the_usage_record_is_emitted_once_per_turn_and_then_reset() {
        let mut m = TranscriptMachine::new("r", None);
        m.record_audio(0.05);
        m.record_audio(0.05);
        let out = m.process(msg(json!({"type": "final_transcript", "transcript": "hi"})));
        let usage =
            out.events.iter().find(|e| e.kind == SpeechEventType::RecognitionUsage).unwrap();
        assert!((usage.recognition_usage.unwrap().audio_duration - 0.1).abs() < 1e-9);

        let out = m.process(msg(json!({"type": "final_transcript", "transcript": "again"})));
        assert!(
            !out.events.iter().any(|e| e.kind == SpeechEventType::RecognitionUsage),
            "no audio was pushed since the last final"
        );
    }

    // Catches: `end_time` other than `start + duration`, or word confidence defaulting to 1.0.
    #[test]
    fn timings_add_up_and_word_confidence_defaults_to_zero() {
        let mut m = TranscriptMachine::new("r", None);
        let out = m.process(msg(json!({
            "type": "final_transcript",
            "transcript": "hi there",
            "start": 1.0,
            "duration": 2.0,
            "words": [{"word": "hi", "start": 1.0, "end": 1.5}]
        })));
        let data = out.events.iter().find(|e| e.kind == SpeechEventType::FinalTranscript).unwrap();
        let alt = data.best_alternative().unwrap();
        assert_eq!(alt.start_time, 1.0);
        assert_eq!(alt.end_time, 3.0, "start + duration");
        assert_eq!(alt.confidence, 1.0, "the utterance default");
        let word = &alt.words.as_ref().unwrap()[0];
        assert_eq!((word.start_time, word.end_time), (Some(1.0), Some(1.5)));
        assert_eq!(word.confidence, Some(0.0), "the *word* default differs from the utterance's");
        assert_eq!(word.start_time_offset, Some(0.0));
    }

    // Catches: R2b on the metadata guard.
    #[test]
    fn an_empty_extra_object_is_not_metadata() {
        let mut m = TranscriptMachine::new("r", None);
        let out =
            m.process(msg(json!({"type": "final_transcript", "transcript": "x", "extra": {}})));
        assert_eq!(final_of(&out).best_alternative().unwrap().metadata, None);
        let out = m.process(msg(
            json!({"type": "final_transcript", "transcript": "x", "extra": {"a": 1}}),
        ));
        assert!(final_of(&out).best_alternative().unwrap().metadata.is_some());
    }

    // Catches: using the connection's request id when the message carries its own.
    #[test]
    fn a_message_request_id_wins_over_the_connections() {
        let mut m = TranscriptMachine::new("stt_request_local", None);
        let out = m.process(msg(json!({"type": "final_transcript", "transcript": "x"})));
        assert_eq!(final_of(&out).request_id, "stt_request_local");
        let out = m.process(msg(
            json!({"type": "final_transcript", "transcript": "y", "request_id": "gw_42"}),
        ));
        assert_eq!(final_of(&out).request_id, "gw_42");
    }

    // Catches: R2b on the language fallback.
    #[test]
    fn an_empty_configured_language_falls_back_to_english() {
        let mut m = TranscriptMachine::new("r", Some(""));
        let out = m.process(msg(json!({"type": "final_transcript", "transcript": "x"})));
        assert_eq!(final_of(&out).best_alternative().unwrap().language.as_str(), "en");

        let mut m = TranscriptMachine::new("r", Some("fr"));
        let out = m.process(msg(json!({"type": "final_transcript", "transcript": "x"})));
        assert_eq!(final_of(&out).best_alternative().unwrap().language.as_str(), "fr");
    }
}
