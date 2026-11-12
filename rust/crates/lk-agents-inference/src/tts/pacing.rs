use std::collections::VecDeque;

/// The pacing decisions, on supplied time: `now` and the audio pushed so far, in seconds.
#[derive(Debug)]
pub struct SentencePacer {
    minimum_audio: f64,
    maximum_text: usize,
    /// The provider's request limit: no batch is joined past it.
    limit: Option<usize>,
    sentences: VecDeque<String>,
    audio_start: Option<f64>,
    first: bool,
    previous_audio: f64,
    previous_check: f64,
    generation_started: bool,
    generation_stopped: bool,
    ended: bool,
}

/// What one [`SentencePacer::poll`] decided.
#[derive(Debug, PartialEq)]
pub struct Paced {
    /// The batch to send now.
    pub text: Option<String>,
    /// Seconds of audio left to play, for diagnostics.
    pub remaining: f64,
    /// Seconds until the next poll; `None` once every sentence went and the input ended.
    pub delay: Option<f64>,
}

impl SentencePacer {
    pub fn new(minimum_audio: f64, maximum_text: usize, limit: Option<usize>) -> Self {
        Self {
            minimum_audio,
            maximum_text,
            limit,
            sentences: VecDeque::new(),
            audio_start: None,
            first: true,
            previous_audio: 0.0,
            previous_check: 0.0,
            generation_started: false,
            generation_stopped: false,
            ended: false,
        }
    }

    pub fn push(&mut self, text: String) {
        self.sentences.push_back(text);
    }

    /// No further sentence will be pushed.
    pub fn end(&mut self) {
        self.ended = true;
    }

    /// A new segment: its first sentence goes at once, while the audio already synthesized still
    /// plays out on the same clock.
    pub fn next_segment(&mut self) {
        self.first = true;
        self.ended = false;
        self.generation_started = false;
        self.generation_stopped = false;
    }

    /// Whether a sentence is still held.
    pub fn holds_any(&self) -> bool {
        !self.sentences.is_empty()
    }

    /// Whether a held sentence still has words to speak.
    pub fn holds_speech(&self) -> bool {
        self.sentences.iter().any(|s| !trim_python_whitespace(s).is_empty())
    }

    /// The first sentence goes at once. Later ones wait until the audio stopped growing (two
    /// checks at least 0.1 s apart saw no new audio) and at most `minimum_audio` seconds of it
    /// remain, then go as one batch of whole sentences up to `maximum_text` characters, never
    /// joined past the provider's `limit` (a sentence is within it on its own).
    pub fn poll(&mut self, now: f64, audio: f64) -> Paced {
        if self.ended && self.sentences.is_empty() {
            return Paced { text: None, remaining: 0.0, delay: None };
        }
        if audio > 0.0 && self.audio_start.is_none() {
            self.audio_start = Some(now);
        }
        if now - self.previous_check >= 0.1 {
            if self.previous_audio < audio {
                self.generation_started = true;
            } else if self.generation_started {
                self.generation_stopped = true;
            }
            self.previous_audio = audio;
            self.previous_check = now;
        }
        let remaining = self.audio_start.map_or(0.0, |start| start + audio - now);
        let mut text = None;
        if self.first || (self.generation_stopped && remaining <= self.minimum_audio) {
            let mut batch = Vec::new();
            let mut length = 0;
            while let Some(sentence) = self.sentences.pop_front() {
                let joined = length + usize::from(!batch.is_empty()) + sentence.chars().count();
                if !batch.is_empty() && self.limit.is_some_and(|limit| joined > limit) {
                    self.sentences.push_front(sentence);
                    break;
                }
                length = joined;
                batch.push(sentence);
                if self.first || length >= self.maximum_text {
                    break;
                }
            }
            if !batch.is_empty() {
                text = Some(batch.join(" "));
                self.generation_started = false;
                self.generation_stopped = false;
                self.first = false;
            }
        }
        let delay = if self.generation_started && !self.generation_stopped {
            0.2
        } else {
            (remaining - self.minimum_audio).max(0.5)
        };
        Paced { text, remaining, delay: Some(delay) }
    }
}

pub(crate) fn trim_python_whitespace(text: &str) -> &str {
    text.trim_matches(|c| lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c))
}

pub(crate) fn sent(text: &str, remaining: f64) {
    log::debug!(target: lk_agents_abi::LOG_TARGET, "sent text to tts: {text:?}, remaining audio {remaining}");
}
