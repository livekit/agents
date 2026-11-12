use std::collections::VecDeque;
use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::tts::stream_pacer::{
    HostSentenceStream, PacedSentences as Port, PushedAudio, SentenceStreamPacer,
};
use lk_agents_abi::utils::clock::{Clock, before, secs};
use lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT;
use lk_agents_net::env::ProcessEnv;

use super::sentences::HostSentences;
use crate::telemetry;

/// The pacing decisions, on supplied time: `now` and the audio pushed so far, in seconds.
#[derive(Debug)]
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

    pub fn push(&mut self, text: String) {
        self.sentences.push_back(text);
    }

    /// No further sentence will be pushed.
    pub fn end(&mut self) {
        self.ended = true;
    }

    /// Whether a held sentence still has words to speak.
    pub fn holds_speech(&self) -> bool {
        self.sentences.iter().any(|s| !trim_python_whitespace(s).is_empty())
    }

    /// The first sentence goes at once. Later ones wait until the audio stopped growing (two
    /// checks at least 0.1 s apart saw no new audio) and at most `minimum_audio` seconds of it
    /// remain, then go as one batch of whole sentences up to `maximum_text` characters.
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
        Paced { text, remaining, delay: Some(delay) }
    }
}

pub(crate) fn trim_python_whitespace(text: &str) -> &str {
    text.trim_matches(|c| lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c))
}

pub(crate) fn sent(text: &str, remaining: f64) {
    log::debug!(target: lk_agents_abi::LOG_TARGET, "sent text to tts: {text:?}, remaining audio {remaining}");
}

/// [`SentencePacer`] over a host sentence stream, paced by an emitter's pushed audio.
pub struct PacedSentences {
    stream: Arc<dyn HostSentenceStream>,
    clock: Arc<dyn Clock>,
    audio: Arc<dyn PushedAudio>,
    state: tokio::sync::Mutex<PacedState>,
    /// Ends a pending `next` when the stream is closed.
    closed: tokio_util::sync::CancellationToken,
}

struct PacedState {
    pacer: SentencePacer,
    input: PacedInput,
}

enum PacedInput {
    /// Read from the first `next`, under its span.
    Unread(Arc<dyn HostSentenceStream>),
    Reading(HostSentences),
    Ended,
}

impl PacedSentences {
    async fn next_batch(
        &self,
        state: &mut PacedState,
        cx: &telemetry::Cx,
    ) -> Result<Option<String>, ProviderError> {
        if let PacedInput::Unread(stream) = &state.input {
            state.input = PacedInput::Reading(HostSentences::read(stream.clone(), cx));
        }
        loop {
            let now = self.clock.now();
            let paced = state.pacer.poll(now.as_secs_f64(), self.audio.pushed_duration());
            if let Some(text) = paced.text {
                sent(&text, paced.remaining);
                return Ok(Some(text));
            }
            let Some(delay) = paced.delay else { return Ok(None) };
            let wake = self.clock.wait_until(now.saturating_add(secs(delay)));
            let PacedInput::Reading(input) = &mut state.input else {
                wake.await;
                continue;
            };
            tokio::select! {
                biased;
                next = input.next() => match next? {
                    Some(sentence) => state.pacer.push(sentence),
                    None => {
                        state.pacer.end();
                        state.input = PacedInput::Ended;
                    }
                },
                () = wake => {}
            }
        }
    }
}

impl Port for PacedSentences {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        sentences: Arc<dyn HostSentenceStream>,
        audio: Arc<dyn PushedAudio>,
        pacer: SentenceStreamPacer,
    ) -> Self {
        Self {
            stream: sentences.clone(),
            clock: env.clock.clone(),
            audio,
            state: tokio::sync::Mutex::new(PacedState {
                pacer: SentencePacer::new(pacer.min_remaining_audio, pacer.max_text_length),
                input: PacedInput::Unread(sentences),
            }),
            closed: tokio_util::sync::CancellationToken::new(),
        }
    }

    async fn next(&self, cx: telemetry::Cx) -> Result<Option<String>, ProviderError> {
        tokio::select! {
            biased;
            () = self.closed.cancelled() => Ok(None),
            next = async {
                let mut state = self.state.lock().await;
                self.next_batch(&mut state, &cx).await
            } => next,
        }
    }

    async fn aclose(&self, cx: telemetry::Cx) {
        if self.closed.is_cancelled() {
            return;
        }
        self.closed.cancel();
        let input = std::mem::replace(&mut self.state.lock().await.input, PacedInput::Ended);
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        match input {
            PacedInput::Reading(sentences) => sentences.close(&cx, &*self.clock, deadline).await,
            PacedInput::Unread(_) | PacedInput::Ended => {
                let closed = self.stream.aclose(&cx);
                if before(&*self.clock, Some(deadline), closed).await.is_err() {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "sentence stream close timed out");
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "pacing_tests.rs"]
mod tests;
