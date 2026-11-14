use lk_agents_abi::stt::events::{SpeechData, Transcript};
use lk_agents_abi::{SpeechEvent, SpeechEventType};

/// Whether the person is speaking, from what a recognition reports. Every stream's
/// START_OF_SPEECH and END_OF_SPEECH come from here: the first words start a speech, a final ends
/// it (unless the provider detects speech itself), the provider's own boundaries are deduplicated
/// and a connection boundary ends the open speech, its pending interim made final.
///
/// Each speech is one utterance, numbered on its events. A provider that detects speech may end
/// one while words are still owed (an interim not yet final, or no words yet from a provider that
/// transcribes after the speech ends): the speech stays open until a final, a new speech or the
/// connection's end, so the final is still that utterance's.
#[derive(Debug, Clone, Default)]
pub struct Speech {
    provider_vad: bool,
    /// The provider transcribes a speech after its end and always sends a final for it.
    transcribes_after_speech_ends: bool,
    speaking: bool,
    /// The open speech produced words.
    heard: bool,
    /// The interim no final replaced yet.
    interim: Option<Transcript>,
    /// The request id of the open speech's words, on the events minted for it.
    request_id: String,
    /// The provider ended the speech while words were owed: the end waits for them.
    ending: bool,
    /// The number of the open (or last) speech.
    utterance: u64,
}

impl Speech {
    pub fn new(provider_vad: bool, transcribes_after_speech_ends: bool) -> Self {
        Speech { provider_vad, transcribes_after_speech_ends, ..Speech::default() }
    }

    pub fn speaking(&self) -> bool {
        self.speaking
    }

    pub fn settled(&self) -> bool {
        !self.speaking && self.interim.is_none()
    }

    /// The provider heard speech start: an ended speech still waiting for its final ends first.
    pub fn started(&mut self, speech_start_time: Option<f64>, out: &mut Vec<SpeechEvent>) {
        if self.ending {
            self.end(out);
        }
        if !self.speaking {
            self.open();
            out.push(SpeechEvent {
                speech_start_time,
                ..self.mark(SpeechEvent::new(SpeechEventType::StartOfSpeech))
            });
        }
    }

    /// The provider heard speech end: while words are still owed, the speech stays open for them;
    /// otherwise it ends now. Whether it ended an open speech.
    pub fn ended(&mut self, out: &mut Vec<SpeechEvent>) -> bool {
        let owed = self.interim.is_some() || (self.transcribes_after_speech_ends && !self.heard);
        if self.provider_vad && self.speaking && owed {
            self.ending = true;
            return false;
        }
        self.end(out)
    }

    /// A connection ended: its pending interim is made final and the open speech ends. Whether
    /// it ended an open speech.
    pub fn boundary(&mut self, out: &mut Vec<SpeechEvent>) -> bool {
        if let Some(interim) = self.interim.take() {
            let promoted = Transcript { r#final: true, ..interim };
            out.push(self.mark(event(SpeechEventType::FinalTranscript, promoted)));
        }
        self.end(out)
    }

    /// Whether it ended an open speech.
    pub fn transcript(&mut self, transcript: Transcript, out: &mut Vec<SpeechEvent>) -> bool {
        let is_final = transcript.r#final;
        if transcript.text.is_empty() {
            if !is_final {
                return false;
            }
            // a provider that detects speech still ends one no words were heard in, or one it
            // already ended
            let ends = self.speaking && (!self.provider_vad || !self.heard || self.ending);
            if self.interim.take().is_some() {
                out.push(self.mark(event(SpeechEventType::InterimTranscript, transcript)));
            } else if ends {
                out.push(self.mark(event(SpeechEventType::FinalTranscript, transcript)));
            }
            return ends && self.end(out);
        }
        let kind = match (is_final, transcript.preflight) {
            (true, _) => SpeechEventType::FinalTranscript,
            (false, true) if !self.speaking => return false,
            (false, true) => SpeechEventType::PreflightTranscript,
            (false, false) => SpeechEventType::InterimTranscript,
        };
        if !transcript.request_id.is_empty() {
            self.request_id.clone_from(&transcript.request_id);
        }
        if !self.speaking {
            self.open();
            // words outside the provider's speech are of one it already ended: its final ends it
            self.ending = self.provider_vad && self.transcribes_after_speech_ends;
            let start = SpeechEvent {
                request_id: self.request_id.clone(),
                ..SpeechEvent::new(SpeechEventType::StartOfSpeech)
            };
            out.push(self.mark(start));
        }
        self.heard = true;
        self.interim = (!is_final).then(|| transcript.clone());
        out.push(self.mark(event(kind, transcript)));
        is_final && (!self.provider_vad || self.ending) && self.end(out)
    }

    fn open(&mut self) {
        self.speaking = true;
        self.utterance += 1;
    }

    /// The event, as the open speech's.
    fn mark(&self, event: SpeechEvent) -> SpeechEvent {
        SpeechEvent { utterance: Some(self.utterance), ..event }
    }

    fn end(&mut self, out: &mut Vec<SpeechEvent>) -> bool {
        self.heard = false;
        self.interim = None;
        self.ending = false;
        let request_id = std::mem::take(&mut self.request_id);
        if !self.speaking {
            return false;
        }
        self.speaking = false;
        let end = SpeechEvent { request_id, ..SpeechEvent::new(SpeechEventType::EndOfSpeech) };
        out.push(self.mark(end));
        true
    }
}

fn event(kind: SpeechEventType, transcript: Transcript) -> SpeechEvent {
    let Transcript {
        text,
        language,
        request_id,
        start_time,
        end_time,
        confidence,
        speaker_id,
        words,
        source_languages,
        source_texts,
        target_languages,
        target_texts,
        metadata,
        ..
    } = transcript;
    SpeechEvent {
        request_id,
        alternatives: vec![SpeechData {
            language,
            text,
            start_time,
            end_time,
            confidence,
            speaker_id,
            words,
            source_languages,
            source_texts,
            target_languages,
            target_texts,
            metadata,
        }],
        ..SpeechEvent::new(kind)
    }
}

impl lk_agents_abi::stt::events::SpeechState for Speech {
    fn new(provider_vad: bool, transcribes_after_speech_ends: bool) -> Self {
        Speech::new(provider_vad, transcribes_after_speech_ends)
    }

    fn speaking(&self) -> bool {
        self.speaking
    }

    fn settled(&self) -> bool {
        Speech::settled(self)
    }

    fn transcript(&mut self, transcript: Transcript) -> Vec<SpeechEvent> {
        let mut out = Vec::new();
        Speech::transcript(self, transcript, &mut out);
        out
    }

    fn started(&mut self, speech_start_time: Option<f64>) -> Vec<SpeechEvent> {
        let mut out = Vec::new();
        Speech::started(self, speech_start_time, &mut out);
        out
    }

    fn ended(&mut self) -> Vec<SpeechEvent> {
        let mut out = Vec::new();
        Speech::ended(self, &mut out);
        out
    }

    fn boundary(&mut self) -> Vec<SpeechEvent> {
        let mut out = Vec::new();
        Speech::boundary(self, &mut out);
        out
    }
}

#[cfg(test)]
#[path = "speech_tests.rs"]
mod tests;
