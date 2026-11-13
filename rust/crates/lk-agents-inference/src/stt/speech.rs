use lk_agents_abi::stt::events::{SpeechData, Transcript};
use lk_agents_abi::{SpeechEvent, SpeechEventType};

/// Whether the person is speaking, from what a recognition reports. Every stream's
/// START_OF_SPEECH and END_OF_SPEECH come from here: the first words start a speech, a final ends
/// it (unless the provider detects speech itself), the provider's own boundaries are deduplicated
/// and a connection boundary ends the open speech.
#[derive(Debug, Clone, Default)]
pub struct Speech {
    provider_vad: bool,
    speaking: bool,
    /// The open speech produced words.
    heard: bool,
    /// An interim stands that no final replaced.
    interim: bool,
}

impl Speech {
    pub fn new(provider_vad: bool) -> Self {
        Speech { provider_vad, ..Speech::default() }
    }

    pub fn speaking(&self) -> bool {
        self.speaking
    }

    pub fn started(&mut self, out: &mut Vec<SpeechEvent>) {
        if !self.speaking {
            self.speaking = true;
            out.push(SpeechEvent::new(SpeechEventType::StartOfSpeech));
        }
    }

    /// Whether it ended an open speech.
    pub fn ended(&mut self, out: &mut Vec<SpeechEvent>) -> bool {
        self.heard = false;
        self.interim = false;
        if !self.speaking {
            return false;
        }
        self.speaking = false;
        out.push(SpeechEvent::new(SpeechEventType::EndOfSpeech));
        true
    }

    /// Whether it ended an open speech.
    pub fn transcript(&mut self, transcript: Transcript, out: &mut Vec<SpeechEvent>) -> bool {
        let is_final = transcript.r#final;
        if transcript.text.is_empty() {
            if !is_final {
                return false;
            }
            // a provider that detects speech still ends one no words were heard in
            let ends = self.speaking && (!self.provider_vad || !self.heard);
            if std::mem::take(&mut self.interim) {
                out.push(event(SpeechEventType::InterimTranscript, transcript));
            } else if ends {
                out.push(event(SpeechEventType::FinalTranscript, transcript));
            }
            return ends && self.ended(out);
        }
        let kind = match (is_final, transcript.preflight) {
            (true, _) => SpeechEventType::FinalTranscript,
            (false, true) if !self.speaking => return false,
            (false, true) => SpeechEventType::PreflightTranscript,
            (false, false) => SpeechEventType::InterimTranscript,
        };
        self.started(out);
        self.heard = true;
        self.interim = !is_final;
        out.push(event(kind, transcript));
        is_final && !self.provider_vad && self.ended(out)
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
            source_texts: None,
            target_languages: None,
            target_texts: None,
            metadata,
        }],
        ..SpeechEvent::new(kind)
    }
}

impl lk_agents_abi::stt::events::SpeechState for Speech {
    fn new(provider_vad: bool) -> Self {
        Speech::new(provider_vad)
    }

    fn speaking(&self) -> bool {
        self.speaking
    }

    fn transcript(&mut self, transcript: Transcript) -> Vec<SpeechEvent> {
        let mut out = Vec::new();
        Speech::transcript(self, transcript, &mut out);
        out
    }

    fn started(&mut self) -> Vec<SpeechEvent> {
        let mut out = Vec::new();
        Speech::started(self, &mut out);
        out
    }

    fn ended(&mut self) -> Vec<SpeechEvent> {
        let mut out = Vec::new();
        Speech::ended(self, &mut out);
        out
    }

    fn boundary(&mut self) -> Vec<SpeechEvent> {
        let mut out = Vec::new();
        Speech::ended(self, &mut out);
        out
    }
}

#[cfg(test)]
#[path = "speech_tests.rs"]
mod tests;
