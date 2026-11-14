use crate::stt::speech::Speech;
use crate::stt::wire::{Incoming, Transcript};
use lk_agents_abi::{RecognitionUsage, SpeechEvent, SpeechEventType, TranscriptChunk};

// One connection's transcript assembly; its end-of-speech edge releases a deferred keyterm update.

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
    speech: Speech,
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
            // the gateway reports onsets, but each final ends the turn
            speech: Speech::new(false, false),
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
        self.speech.speaking()
    }

    /// The connection ended: its open speech ends with it.
    pub fn boundary(&mut self) -> Vec<SpeechEvent> {
        let mut events = Vec::new();
        self.speech.boundary(&mut events);
        events
    }

    /// `_speech_duration` is accumulated on the send side.
    pub fn record_audio(&mut self, duration: f64) {
        self.speech_duration += duration;
    }

    pub fn process(&mut self, msg: Incoming) -> Processed {
        match msg {
            Incoming::StartOfSpeech => {
                let mut out = Processed::default();
                self.speech.started(None, &mut out.events);
                out
            }
            Incoming::PreflightTranscript(data) => self.transcript(&data, false, true),
            Incoming::InterimTranscript(data) => self.transcript(&data, false, false),
            Incoming::FinalTranscript(data) => self.transcript(&data, true, false),
            Incoming::Error { code } => Processed {
                error: Some(
                    lk_agents_abi::errors::ApiError::status(
                        "LiveKit Inference STT returned an error",
                        code,
                        None,
                    )
                    .with_should_retry(
                        lk_agents_abi::inference::stt_wire::SttErrorCode::from_wire(
                            &code.to_string(),
                        )
                        .map(|code| code.retryable()),
                    ),
                ),
                ..Default::default()
            },
            Incoming::SessionUpdated { success: false } => Processed {
                error: Some(lk_agents_abi::errors::ApiError::new(
                    lk_agents_abi::errors::ErrorKind::InvalidRequest,
                    "LiveKit Inference STT session update was rejected",
                )),
                ..Default::default()
            },
            Incoming::SessionFinalized { truncated: true }
            | Incoming::SessionClosed { truncated: true } => Processed {
                error: Some(
                    // the transcript before the cut was delivered: a retry would repeat it
                    lk_agents_abi::errors::ApiError::new(
                        lk_agents_abi::errors::ErrorKind::Server,
                        "LiveKit Inference STT transcript was truncated",
                    )
                    .into_final(),
                ),
                ..Default::default()
            },
            // `session.created`, `session.finalized`, `session.closed` and anything unrecognised
            // are all `pass` in the reference.
            _ => Processed::default(),
        }
    }

    fn transcript(&mut self, data: &Transcript, is_final: bool, preflight: bool) -> Processed {
        let request_id = self.request_id_of(data);
        let report = lk_agents_abi::stt::events::Transcript {
            r#final: is_final,
            preflight,
            ..self.report(data, request_id.clone())
        };
        let mut out = Processed::default();
        out.end_of_speech = self.speech.transcript(report, &mut out.events);
        // Usage goes before the final transcript, and only when audio was pushed.
        if is_final && self.speech_duration > 0.0 {
            let at = usize::from(
                out.events.first().is_some_and(|e| e.kind == SpeechEventType::StartOfSpeech),
            );
            out.events.insert(
                at,
                SpeechEvent {
                    request_id,
                    recognition_usage: Some(Box::new(RecognitionUsage {
                        audio_duration: Some(self.speech_duration),
                        ..Default::default()
                    })),
                    ..SpeechEvent::new(SpeechEventType::RecognitionUsage)
                },
            );
            self.speech_duration = 0.0;
        }
        out
    }

    fn request_id_of(&self, data: &Transcript) -> String {
        data.request_id.clone().unwrap_or_else(|| self.request_id.clone())
    }

    /// `_build_speech_data`.
    fn report(
        &self,
        data: &Transcript,
        request_id: String,
    ) -> lk_agents_abi::stt::events::Transcript {
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
                ..TranscriptChunk::default()
            })
            .collect();

        lk_agents_abi::stt::events::Transcript {
            request_id,
            start_time: data.start,
            end_time: data.start + data.duration,
            confidence: data.confidence,
            speaker_id: data.speaker_id.clone(),
            // An absent `words` is an empty list, not `None`.
            words: Some(words),
            // An empty `extra` object is `None`, not `Some({})`.
            metadata: data.extra.as_ref().filter(|extra| !extra.is_empty()).cloned(),
            ..lk_agents_abi::stt::events::Transcript::new(language, data.transcript.clone())
        }
    }
}

#[cfg(test)]
#[path = "transcript_tests.rs"]
mod tests;
