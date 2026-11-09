use crate::stt::wire::{Incoming, Transcript};
use lk_agents_abi::{RecognitionUsage, SpeechData, SpeechEvent, SpeechEventType, TranscriptChunk};

// One connection's transcript assembly. `speaking` is written from three places, and its end-of-speech edge releases a deferred keyterm
// update.

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Processed {
    pub events: Vec<SpeechEvent>,
    /// The `_on_end_of_speech()` edge: apply a deferred keyterm update (reported, not performed).
    pub end_of_speech: bool,
    /// The gateway sent `{"type": "error"}`; the stream must fail with this.
    pub error: Option<lk_agents_abi::errors::ApiError>,
}

#[derive(Debug, Clone)]
pub(super) struct TranscriptMachine {
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
            Incoming::Error { code } => Processed {
                error: Some(lk_agents_abi::errors::ApiError::status(
                    "LiveKit Inference STT returned an error",
                    code,
                    None,
                    lk_agents_abi::inference::stt_wire::SttErrorCode::from_wire(&code.to_string())
                        .map(|code| code.retryable()),
                )),
                ..Default::default()
            },
            Incoming::SessionUpdated { success: false } => Processed {
                error: Some(
                    lk_agents_abi::errors::ApiError::new(
                        "LiveKit Inference STT session update was rejected",
                    )
                    .with_retryable(false),
                ),
                ..Default::default()
            },
            Incoming::SessionFinalized { truncated: true }
            | Incoming::SessionClosed { truncated: true } => Processed {
                error: Some(
                    lk_agents_abi::errors::ApiError::new(
                        "LiveKit Inference STT transcript was truncated",
                    )
                    .with_retryable(false),
                ),
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

    fn preflight(&mut self, data: &Transcript) -> Processed {
        let text = &data.transcript;
        // Drop an empty preflight, and any preflight outside a turn (a speculative final).
        if text.is_empty() || !self.speaking {
            return Processed::default();
        }
        let event = SpeechEvent {
            kind: SpeechEventType::PreflightTranscript,
            request_id: self.request_id_of(data),
            alternatives: vec![self.speech_data(data)],
            recognition_usage: None,
            ..SpeechEvent::new(SpeechEventType::PreflightTranscript)
        };
        Processed { events: vec![event], ..Default::default() }
    }

    fn transcript(&mut self, data: &Transcript, is_final: bool) -> Processed {
        let request_id = self.request_id_of(data);
        let text = &data.transcript;

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
                created_at: None,
                speech_end_time: None,
            });
            return out;
        }

        // Usage goes before the final transcript, and only when audio was pushed.
        if self.speech_duration > 0.0 {
            out.events.push(SpeechEvent {
                kind: SpeechEventType::RecognitionUsage,
                request_id: request_id.clone(),
                alternatives: Vec::new(),
                recognition_usage: Some(Box::new(RecognitionUsage {
                    audio_duration: Some(self.speech_duration),
                    ..Default::default()
                })),
                speech_start_time: None,
                created_at: None,
                speech_end_time: None,
            });
            self.speech_duration = 0.0;
        }

        out.events.push(SpeechEvent {
            kind: SpeechEventType::FinalTranscript,
            request_id,
            alternatives: vec![speech_data],
            recognition_usage: None,
            speech_start_time: None,
            created_at: None,
            speech_end_time: None,
        });

        if self.speaking {
            self.speaking = false;
            out.events.push(SpeechEvent::new(SpeechEventType::EndOfSpeech));
            out.end_of_speech = true;
        }
        out
    }

    fn request_id_of(&self, data: &Transcript) -> String {
        data.request_id.clone().unwrap_or_else(|| self.request_id.clone())
    }

    /// `_build_speech_data`.
    fn speech_data(&self, data: &Transcript) -> SpeechData {
        let start = data.start;
        let duration = data.duration;
        let language = data.language.as_deref().unwrap_or(&self.default_language);
        let words = data
            .words
            .iter()
            .map(|word| TranscriptChunk {
                text: word.word.clone(),
                start_time: Some(word.start),
                end_time: Some(word.end),
                confidence: Some(word.confidence),
                start_time_offset: Some(0.0),
                speaker_id: word.speaker_id.clone(),
            })
            .collect();

        SpeechData {
            language: lk_agents_abi::tokenize::language::LanguageCode::new(language),
            text: data.transcript.clone(),
            start_time: start,
            end_time: start + duration,
            confidence: data.confidence,
            speaker_id: data.speaker_id.clone(),
            // An absent `words` is an empty list, not `None`.
            words: Some(words),
            source_languages: None,
            source_texts: None,
            target_languages: None,
            target_texts: None,
            // An empty `extra` object is `None`, not `Some({})`.
            metadata: data.extra.as_ref().filter(|extra| !extra.is_empty()).cloned(),
        }
    }
}

#[cfg(test)]
#[path = "transcript_tests.rs"]
mod tests;
