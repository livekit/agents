//! Sentence batching driven by observed synthesis duration and an injected clock.
use std::collections::VecDeque;

#[crate::class(name = "SentencePacer", mutable)]
pub struct SentencePacer {
    minimum_audio: f64,
    maximum_text: usize,
    sentences: VecDeque<String>,
    audio_start: Option<f64>,
    first: bool,
    previous_audio: f64,
    previous_check: f64,
    generation_started: bool,
    generation_stopped: bool,
    ended: bool,
}

#[crate::host(object)]
impl SentencePacer {
    #[host(new)]
    pub fn new(minimum_audio: f64, maximum_text: usize) -> Self {
        Self {
            minimum_audio,
            maximum_text,
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
    #[host(method)]
    pub fn push(&mut self, text: String) {
        self.sentences.push_back(text);
    }
    #[host(method)]
    pub fn end(&mut self) {
        self.ended = true;
    }
    /// Text to emit, remaining audio for diagnostics, and next wakeup (None means done).
    #[host(method)]
    pub fn poll(&mut self, now: f64, audio: f64) -> (Option<String>, f64, Option<f64>) {
        if self.ended && self.sentences.is_empty() {
            return (None, 0.0, None);
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
                length += sentence.chars().count();
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
        (text, remaining, Some(delay))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_sentence_is_immediate_then_wait_for_audio_and_batch() {
        let mut p = SentencePacer::new(5.0, 4);
        p.push("hello".into());
        p.push("你好".into());
        p.push("ab".into());
        p.end();
        assert_eq!(p.poll(100.0, 0.0).0.as_deref(), Some("hello"));
        assert_eq!(p.poll(101.0, 10.0).0, None);
        assert_eq!(p.poll(102.0, 10.0).0, None);
        assert_eq!(p.poll(106.0, 10.0).0.as_deref(), Some("你好 ab"));
        assert_eq!(p.poll(107.0, 10.0).2, None);
    }
}
