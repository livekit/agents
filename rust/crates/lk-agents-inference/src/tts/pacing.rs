//! Sentence pacing (`tts.SentenceStreamPacer`): sentences held back until the audio already
//! synthesized is nearly played, then sent in batches. [`SentencePacer`] decides on supplied
//! time; [`PacedSentences`] runs it over a host sentence stream for a plugin
//! (`SentenceStreamPacer.wrap()`); the sentence adapter runs it over its own sentences.

use std::collections::VecDeque;
use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::tts::stream_adapter::{SentenceSplitter, SentenceTextProcessor};
use lk_agents_abi::tts::stream_pacer::{HostSentenceStream, PacedSentences as Port, PushedAudio};
use lk_agents_abi::utils::clock::{Clock, Ts, before, secs};
use lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT;
use lk_agents_net::env::ProcessEnv;
use tokio::sync::mpsc;

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
    log::debug!(target: "livekit.agents", "sent text to tts: {text:?}, remaining audio {remaining}");
}

/// A host sentence stream read by a task of its own, so a wait elsewhere never cancels a pending
/// read of the host's stream.
pub(crate) struct HostSentences {
    stream: Arc<dyn HostSentenceStream>,
    out: mpsc::UnboundedReceiver<Result<Option<String>, ProviderError>>,
    reader: tokio::task::JoinHandle<()>,
}

impl HostSentences {
    pub(crate) fn read(stream: Arc<dyn HostSentenceStream>, cx: &telemetry::Cx) -> Self {
        let (tx, out) = mpsc::unbounded_channel();
        let (host, cx) = (stream.clone(), cx.clone());
        let reader = tokio::spawn(async move {
            loop {
                let next = host.next(&cx).await;
                let more = matches!(next, Ok(Some(_)));
                if tx.send(next).is_err() || !more {
                    return;
                }
            }
        });
        Self { stream, out, reader }
    }

    pub(crate) fn stream(&self) -> &dyn HostSentenceStream {
        &*self.stream
    }

    /// The next sentence read, without waiting; `Some(Ok(None))` once the stream ended.
    pub(crate) fn try_next(&mut self) -> Option<Result<Option<String>, ProviderError>> {
        match self.out.try_recv() {
            Ok(next) => Some(next),
            Err(mpsc::error::TryRecvError::Empty) => None,
            Err(mpsc::error::TryRecvError::Disconnected) => Some(Ok(None)),
        }
    }

    /// The next sentence, `None` once the stream ended. Cancel-safe.
    pub(crate) async fn next(&mut self) -> Result<Option<String>, ProviderError> {
        self.out.recv().await.unwrap_or(Ok(None))
    }

    /// Stop reading and close the host stream, its close bounded by `deadline`.
    pub(crate) async fn close(&self, cx: &telemetry::Cx, clock: &dyn Clock, deadline: Ts) {
        self.reader.abort();
        let closed = self.stream.aclose(cx);
        if before(clock, Some(deadline), closed).await.is_err() {
            log::warn!(target: "livekit.agents", "sentence stream close timed out");
        }
    }
}

/// The sentence adapter's text, as pushed or flushed.
pub(crate) enum AdapterInput {
    Text(String),
    Flush,
}

/// How the adapter's text is split into sentences.
pub(crate) enum Split {
    /// Natively, as the input arrives.
    Native(Box<dyn SentenceSplitter>),
    /// By a host tokenizer's stream: the input forwarded to it, its sentences read back.
    Host(HostSentences),
}

enum Woke {
    Input(Option<AdapterInput>),
    Read(Result<Option<String>, ProviderError>),
    Paced,
}

/// The sentence adapter's sentences: its input split, then, when paced, held until the audio
/// already synthesized is nearly played. Every wait is cancel-safe.
pub(crate) struct Sentences {
    rx: mpsc::UnboundedReceiver<AdapterInput>,
    input_closed: bool,
    split: Split,
    /// Markup normalized before splitting, so a marker is never split across two sentences.
    normalize: Option<Box<dyn SentenceTextProcessor>>,
    /// Split, not yet paced.
    split_out: VecDeque<String>,
    split_done: bool,
    pacer: Option<SentencePacer>,
    /// When the pacer is polled next; `None` once it released everything.
    pace_at: Option<Ts>,
    /// Ready to synthesize.
    ready: VecDeque<String>,
    /// A host stream's failure, raised by the next `next`.
    failed: Option<ProviderError>,
    clock: Arc<dyn Clock>,
}

impl Sentences {
    pub(crate) fn new(
        rx: mpsc::UnboundedReceiver<AdapterInput>,
        split: Split,
        normalize: Option<Box<dyn SentenceTextProcessor>>,
        pacing: Option<(f64, usize)>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let pacer = pacing.map(|(audio, text)| SentencePacer::new(audio, text));
        let pace_at = pacer.as_ref().map(|_| clock.now());
        Self {
            rx,
            input_closed: false,
            split,
            normalize,
            split_out: VecDeque::new(),
            split_done: false,
            pacer,
            pace_at,
            ready: VecDeque::new(),
            failed: None,
            clock,
        }
    }

    /// Take what arrived without waiting, and pace it against `audio` seconds synthesized.
    pub(crate) fn drain(&mut self, audio: f64) {
        while let Ok(input) = self.rx.try_recv() {
            self.input(Some(input));
        }
        if !self.input_closed && self.rx.is_closed() && self.rx.is_empty() {
            self.input(None);
        }
        while !self.split_done {
            let Split::Host(host) = &mut self.split else { break };
            let Some(next) = host.try_next() else { break };
            self.split_read(next);
        }
        self.pace(audio);
    }

    /// The next sentence to synthesize, `None` once there is none left.
    pub(crate) async fn next(&mut self, audio: f64) -> Result<Option<String>, ProviderError> {
        loop {
            self.drain(audio);
            if let Some(error) = self.failed.take() {
                return Err(error);
            }
            if let Some(sentence) = self.ready.pop_front() {
                return Ok(Some(sentence));
            }
            if self.split_done && self.split_out.is_empty() && self.pace_at.is_none() {
                return Ok(None);
            }
            match self.wake().await {
                Woke::Input(input) => self.input(input),
                Woke::Read(next) => self.split_read(next),
                Woke::Paced => {}
            }
        }
    }

    /// The next input, host sentence or pacing time, whichever comes first.
    async fn wake(&mut self) -> Woke {
        let (rx, input_open, pace_at, clock) =
            (&mut self.rx, !self.input_closed, self.pace_at, &self.clock);
        let host = match &mut self.split {
            Split::Host(host) if !self.split_done => Some(host),
            _ => None,
        };
        let reading = host.is_some();
        tokio::select! {
            biased;
            input = rx.recv(), if input_open => Woke::Input(input),
            next = async move {
                match host {
                    Some(host) => host.next().await,
                    None => std::future::pending().await,
                }
            }, if reading => Woke::Read(next),
            () = async {
                match pace_at {
                    Some(at) => clock.wait_until(at).await,
                    None => std::future::pending().await,
                }
            }, if pace_at.is_some() => Woke::Paced,
            else => Woke::Paced,
        }
    }

    /// Whether more speech can still come.
    pub(crate) fn more_coming(&self) -> bool {
        let speaks = |s: &String| !trim_python_whitespace(s).is_empty();
        !self.split_done
            || self.split_out.iter().any(speaks)
            || self.ready.iter().any(speaks)
            || self.pacer.as_ref().is_some_and(SentencePacer::holds_speech)
    }

    /// Stop reading a host tokenizer's stream and close it, by `deadline`.
    pub(crate) async fn close(&mut self, cx: &telemetry::Cx, deadline: Ts) {
        self.rx.close();
        if let Split::Host(host) = &self.split {
            host.close(cx, &*self.clock, deadline).await;
        }
    }

    /// One input, `None` once the input ended.
    fn input(&mut self, input: Option<AdapterInput>) {
        let text = match (&input, &self.normalize) {
            (Some(AdapterInput::Text(text)), Some(markup)) => markup.normalize(text),
            (Some(AdapterInput::Text(text)), None) => text.clone(),
            _ => String::new(),
        };
        if input.is_none() {
            self.input_closed = true;
        }
        match (&mut self.split, input) {
            (Split::Native(splitter), Some(AdapterInput::Text(_))) => {
                self.split_out.extend(splitter.push(&text));
            }
            (Split::Native(splitter), Some(AdapterInput::Flush)) => {
                self.split_out.extend(splitter.flush());
            }
            (Split::Native(splitter), None) => {
                self.split_out.extend(splitter.flush());
                self.split_done = true;
            }
            (Split::Host(host), Some(AdapterInput::Text(_))) => host.stream().push_text(text),
            (Split::Host(host), Some(AdapterInput::Flush)) => host.stream().flush(),
            (Split::Host(host), None) => host.stream().end_input(),
        }
    }

    fn split_read(&mut self, next: Result<Option<String>, ProviderError>) {
        match next {
            Ok(Some(sentence)) => self.split_out.push_back(sentence),
            Ok(None) => self.split_done = true,
            Err(error) => {
                self.failed = Some(error);
                self.split_done = true;
            }
        }
    }

    fn pace(&mut self, audio: f64) {
        let Some(pacer) = self.pacer.as_mut() else {
            self.ready.extend(self.split_out.drain(..));
            return;
        };
        for sentence in self.split_out.drain(..) {
            pacer.push(sentence);
        }
        if self.split_done {
            pacer.end();
        }
        let now = self.clock.now();
        let paced = pacer.poll(now.as_secs_f64(), audio);
        if let Some(text) = paced.text {
            sent(&text, paced.remaining);
            self.ready.push_back(text);
        }
        self.pace_at = paced.delay.map(|delay| now.saturating_add(secs(delay)));
    }
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
        min_remaining_audio: f64,
        max_text_length: usize,
    ) -> Self {
        Self {
            stream: sentences.clone(),
            clock: env.clock.clone(),
            audio,
            state: tokio::sync::Mutex::new(PacedState {
                pacer: SentencePacer::new(min_remaining_audio, max_text_length),
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
                    log::warn!(target: "livekit.agents", "sentence stream close timed out");
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "pacing_tests.rs"]
mod tests;
