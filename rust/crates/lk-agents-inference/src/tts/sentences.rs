//! The sentence adapter's sentences: its text split natively or by a host tokenizer's stream,
//! then paced.

use std::collections::VecDeque;
use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::tts::stream_adapter::{SentenceSplitter, SentenceTextProcessor};
use lk_agents_abi::tts::stream_pacer::{HostSentenceStream, SentenceStreamPacer};
use lk_agents_abi::utils::clock::{Clock, Ts, before, secs};
use lk_agents_core::lexer::Vocabulary;
use lk_agents_core::tokenize::sentences::speakable_alone;
use tokio::sync::mpsc;

use super::pacing::{SentencePacer, sent, trim_python_whitespace};
use crate::telemetry;

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
        let reader = tokio::spawn(lk_agents_net::task::inherit(async move {
            loop {
                let next = host.next(&cx).await.map(|sentence| sentence.map(|s| s.token));
                let more = matches!(next, Ok(Some(_)));
                if tx.send(next).is_err() || !more {
                    return;
                }
            }
        }));
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
            log::warn!(target: lk_agents_abi::LOG_TARGET, "sentence stream close timed out");
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
    /// The voice's tags: a host tokenizer's sentence that ends inside one, or says nothing but
    /// markup, waits in `glued` for the next.
    markup: Option<Vocabulary>,
    glued: String,
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
        markup: Option<Vocabulary>,
        pacing: Option<SentenceStreamPacer>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let pacer = pacing.map(|p| SentencePacer::new(p.min_remaining_audio, p.max_text_length));
        let pace_at = pacer.as_ref().map(|_| clock.now());
        Self {
            rx,
            input_closed: false,
            split,
            normalize,
            markup,
            glued: String::new(),
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
            Ok(Some(sentence)) => match &self.markup {
                Some(markup) => {
                    self.glued.push_str(&sentence);
                    if speakable_alone(&self.glued, markup) {
                        self.split_out.push_back(std::mem::take(&mut self.glued));
                    }
                }
                None => self.split_out.push_back(sentence),
            },
            Ok(None) => {
                if !self.glued.is_empty() {
                    self.split_out.push_back(std::mem::take(&mut self.glued));
                }
                self.split_done = true;
            }
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
