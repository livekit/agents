//! The text-to-speech stage: what a synthesis was pushed, as the chunks its provider receives.
//! Every TTS reads its input through it: the text split into sentences (or words, for a provider
//! that buffers text itself) by the native splitter or a host tokenizer, the chunks paced as the
//! audio plays out, and each lowered to the provider's syntax with the words it says.
//!
//! A flush ends a segment: what follows waits until the segment's chunks all went, then the flush
//! goes and the next segment starts afresh (its first chunk at once).

use std::collections::VecDeque;
use std::sync::Arc;

use futures_util::FutureExt;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::tokenize::bindings::{SentenceSettings, SentenceTokenizerLeg};
use lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer;
use lk_agents_abi::tts::stream_pacer::{HostSentenceStream, SentenceStreamPacer};
use lk_agents_abi::tts::types::{SpeechChunk, TextUnit, Tts, TtsOptions};
use lk_agents_abi::utils::clock::{Clock, Ts, before, secs};
use lk_agents_abi::utils::stream::Entry;
use lk_agents_core::lexer::Vocabulary;
use lk_agents_core::markup::{Markup, MarkupExt, markup_vocabulary};
use lk_agents_core::tokenize::sentences::{SentenceOptions, SentenceStream, speakable_alone};

use super::fed::TextLog;
use super::pacing::{SentencePacer, sent, trim_python_whitespace};
use crate::telemetry;

/// A whole text as `tts` takes it: one request, or as few as its request limit allows (cut at a
/// clause mark or word boundary, never inside a tag), each lowered to its syntax with the words it
/// says; not paced. A TTS that prepares its text per provider ([`TextUnit::Text`]) takes it as
/// written.
pub(crate) fn one_shot(tts: &dyn Tts, text: &str) -> Vec<SpeechChunk> {
    if tts.text_unit() == TextUnit::Text {
        return vec![SpeechChunk { text: text.to_owned(), said: String::new() }];
    }
    let key = tts.markup_provider();
    let dialect = Markup::from_provider_key(&key);
    let text = dialect.normalize(trim_python_whitespace(text));
    let limit = tts.max_text_len().or_else(|| dialect.max_input_len());
    let pieces = match limit.filter(|limit| text.chars().count() > *limit) {
        Some(limit) => {
            let mut cut = SentenceStream::new(SentenceOptions {
                min_words: usize::MAX,
                max_token_len: Some(limit),
                retain_format: true,
                markup: Some(markup_vocabulary(&key)),
                ..SentenceOptions::default()
            });
            let mut pieces = cut.push(&text);
            pieces.extend(cut.flush());
            pieces
        }
        None => vec![text],
    };
    pieces
        .iter()
        .map(|piece| {
            let lowered = dialect.lower(piece);
            SpeechChunk {
                text: trim_python_whitespace(&lowered.text).to_owned(),
                said: trim_python_whitespace(&lowered.said).to_owned(),
            }
        })
        .collect()
}

type Read = BoxFuture<'static, Result<Option<String>, ProviderError>>;

/// A host sentence stream and its pending read, held here so a wait elsewhere never cancels a
/// pending read of the host's stream.
pub(crate) struct HostSentences {
    stream: Arc<dyn HostSentenceStream>,
    cx: telemetry::Cx,
    /// `None` once the stream ended or failed.
    reading: Option<Read>,
}

impl HostSentences {
    pub(crate) fn read(stream: Arc<dyn HostSentenceStream>, cx: &telemetry::Cx) -> Self {
        let reading = Some(Self::start(&stream, cx));
        Self { stream, cx: cx.clone(), reading }
    }

    fn start(stream: &Arc<dyn HostSentenceStream>, cx: &telemetry::Cx) -> Read {
        let (stream, cx) = (stream.clone(), cx.clone());
        Box::pin(async move { stream.next(&cx).await.map(|sentence| sentence.map(|s| s.token)) })
    }

    pub(crate) fn stream(&self) -> &dyn HostSentenceStream {
        &*self.stream
    }

    /// The next sentence read, without waiting; `Some(Ok(None))` once the stream ended.
    pub(crate) fn try_next(&mut self) -> Option<Result<Option<String>, ProviderError>> {
        self.next().now_or_never()
    }

    /// The next sentence, `None` once the stream ended. Cancel-safe: the read stays pending.
    pub(crate) async fn next(&mut self) -> Result<Option<String>, ProviderError> {
        let Some(reading) = self.reading.as_mut() else { return Ok(None) };
        let next = reading.await;
        self.reading = matches!(next, Ok(Some(_))).then(|| Self::start(&self.stream, &self.cx));
        next
    }

    /// Stop reading and close the host stream, its close bounded by `deadline`.
    pub(crate) async fn close(mut self, cx: &telemetry::Cx, clock: &dyn Clock, deadline: Ts) {
        self.reading = None;
        let closed = self.stream.aclose(cx);
        if before(clock, Some(deadline), closed).await.is_err() {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "sentence stream close timed out");
        }
    }
}

/// How a segment's text is split.
enum Split {
    /// Not at all: the TTS prepares its text itself ([`TextUnit::Text`]).
    Through,
    Native(Box<SentenceStream>),
    /// A host tokenizer: one stream of it per segment, opened at the segment's first text and
    /// ended with it, so its end tells the segment's last sentence apart.
    Host {
        tokenizer: Arc<dyn HostSentenceTokenizer>,
        current: Option<Box<HostSentences>>,
        opening: Option<BoxFuture<'static, Result<Arc<dyn HostSentenceStream>, ProviderError>>>,
        /// Text pushed before the segment's stream opened.
        early: Vec<Entry<String>>,
        /// Earlier segments' streams, ended: closed with the stage.
        ended: Vec<HostSentences>,
    },
}

enum Woke {
    Entry(Entry<String>),
    Read(Result<Option<String>, ProviderError>),
    Opened(Result<Arc<dyn HostSentenceStream>, ProviderError>),
    Paced,
}

pub(crate) struct Stage {
    log: Arc<dyn TextLog>,
    /// The next log entry to read.
    cursor: usize,
    /// Entries of a later segment, read while the current one is still going out.
    waiting: VecDeque<Entry<String>>,
    /// How the current segment ends, once its end was read.
    segment_end: Option<Entry<String>>,
    split: Split,
    split_done: bool,
    split_out: VecDeque<String>,
    /// The provider's dialect: what each chunk is lowered to.
    dialect: Markup,
    /// The provider's request limit: no chunk, and no paced batch, passes it.
    limit: Option<usize>,
    /// The voice's tags: a host sentence that ends inside one waits in `glued` for the next.
    tags: Vocabulary,
    glued: String,
    pacing: Option<SentenceStreamPacer>,
    pacer: Option<SentencePacer>,
    pace_at: Option<Ts>,
    ready: VecDeque<Entry<SpeechChunk>>,
    /// The input ended and its end went out.
    done: bool,
    failed: Option<ProviderError>,
    cx: telemetry::Cx,
    clock: Arc<dyn Clock>,
}

impl Stage {
    /// The stage `tts` reads its input through: its dialect, its text unit, its text options.
    /// Unless `paced`, each chunk goes as soon as it is split (a probe has nothing playing).
    pub(crate) fn new(
        tts: &dyn Tts,
        log: Arc<dyn TextLog>,
        paced: bool,
        cx: &telemetry::Cx,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let key = tts.markup_provider();
        let dialect = Markup::from_provider_key(&key);
        let unit = tts.text_unit();
        let limit = tts.max_text_len().or_else(|| dialect.max_input_len());
        let TtsOptions { sentence_tokenizer, pacing } = tts.options().current();
        let pacing = pacing.filter(|_| paced && unit != TextUnit::Text);
        // a word provider takes words whatever splits the sentences
        let native = |settings| {
            let options = SentenceOptions::for_voice(settings, &key, unit, limit);
            Split::Native(Box::new(SentenceStream::new(options)))
        };
        let split = match sentence_tokenizer {
            _ if unit == TextUnit::Text => Split::Through,
            SentenceTokenizerLeg::Native(tokenizer) => native(tokenizer.settings()),
            SentenceTokenizerLeg::Host(_) if unit == TextUnit::Words => {
                native(SentenceSettings::default())
            }
            SentenceTokenizerLeg::Host(tokenizer) => Split::Host {
                tokenizer,
                current: None,
                opening: None,
                early: Vec::new(),
                ended: Vec::new(),
            },
        };
        let mut stage = Stage {
            log,
            cursor: 0,
            waiting: VecDeque::new(),
            segment_end: None,
            split,
            split_done: false,
            split_out: VecDeque::new(),
            dialect,
            limit,
            tags: markup_vocabulary(&key),
            glued: String::new(),
            pacing,
            pacer: None,
            pace_at: None,
            ready: VecDeque::new(),
            done: false,
            failed: None,
            cx: cx.clone(),
            clock,
        };
        stage.start_segment();
        stage
    }

    fn start_segment(&mut self) {
        let limit = self.limit;
        match &mut self.pacer {
            Some(pacer) => pacer.next_segment(),
            None => {
                self.pacer = self
                    .pacing
                    .map(|p| SentencePacer::new(p.min_remaining_audio, p.max_text_length, limit));
            }
        }
        self.pace_at = self.pacer.as_ref().map(|_| self.clock.now());
        self.segment_end = None;
        self.split_done = false;
    }

    /// The next entry for the provider, against `audio` seconds synthesized so far; `None` once
    /// the input's end went out. Cancel-safe.
    pub(crate) async fn next(
        &mut self,
        audio: f64,
    ) -> Result<Option<Entry<SpeechChunk>>, ProviderError> {
        loop {
            self.settle(audio);
            if let Some(error) = self.failed.take() {
                return Err(error);
            }
            if let Some(entry) = self.ready.pop_front() {
                if matches!(entry, Entry::EndInput) {
                    self.done = true;
                }
                return Ok(Some(entry));
            }
            if self.done {
                return Ok(None);
            }
            match self.wake().await {
                Woke::Entry(entry) => {
                    self.cursor += 1;
                    self.input(entry);
                }
                Woke::Read(next) => self.split_read(next),
                Woke::Opened(opened) => self.opened(opened),
                Woke::Paced => {}
            }
        }
    }

    /// Stop reading the host tokenizer's streams and close them, by `deadline`.
    pub(crate) async fn close(&mut self, deadline: Ts) {
        if let Split::Host { current, ended, .. } = &mut self.split {
            for host in ended.drain(..).chain(current.take().map(|host| *host)) {
                host.close(&self.cx, &*self.clock, deadline).await;
            }
        }
    }

    /// What can be decided without waiting: host sentences read, chunks paced, a finished
    /// segment's end handed on and the next segment started.
    fn settle(&mut self, audio: f64) {
        loop {
            while !self.split_done {
                let Split::Host { current: Some(host), .. } = &mut self.split else { break };
                let Some(next) = host.try_next() else { break };
                self.split_read(next);
            }
            self.pace(audio);
            let drained = self.split_done
                && self.split_out.is_empty()
                && self.pacer.as_ref().is_none_or(|p| !p.holds_any());
            let Some(end) = self.segment_end.take_if(|_| drained) else { return };
            let last = matches!(end, Entry::EndInput);
            self.ready.push_back(if last { Entry::EndInput } else { Entry::Flush });
            if last {
                self.split_done = true;
                self.pace_at = None;
                return;
            }
            self.start_segment();
            while self.segment_end.is_none() {
                let Some(entry) = self.waiting.pop_front() else { break };
                self.input(entry);
            }
        }
    }

    /// The next log entry, host sentence, opened host stream or pacing time.
    async fn wake(&mut self) -> Woke {
        let (log, cursor, pace_at, clock) = (&self.log, self.cursor, self.pace_at, &self.clock);
        let reading_log = !self.done;
        let (opening, host) = match &mut self.split {
            Split::Host { opening, current, .. } => (opening.as_mut(), current.as_mut()),
            Split::Native(_) | Split::Through => (None, None),
        };
        let host = host.filter(|_| !self.split_done);
        let (is_opening, is_reading) = (opening.is_some(), host.is_some());
        tokio::select! {
            biased;
            entry = log.entry(cursor), if reading_log => Woke::Entry(entry),
            opened = async move {
                match opening {
                    Some(opening) => opening.await,
                    None => std::future::pending().await,
                }
            }, if is_opening => Woke::Opened(opened),
            next = async move {
                match host {
                    Some(host) => host.next().await,
                    None => std::future::pending().await,
                }
            }, if is_reading => Woke::Read(next),
            () = async {
                match pace_at {
                    Some(at) => clock.wait_until(at).await,
                    None => std::future::pending().await,
                }
            }, if pace_at.is_some() => Woke::Paced,
        }
    }

    /// One log entry: a later segment's waits; this segment's is split.
    fn input(&mut self, entry: Entry<String>) {
        if matches!(self.split, Split::Through) {
            self.ready.push_back(match entry {
                Entry::Item(text) => Entry::Item(SpeechChunk { text, said: String::new() }),
                Entry::Flush => Entry::Flush,
                Entry::EndInput => Entry::EndInput,
            });
            return;
        }
        if self.segment_end.is_some() {
            self.waiting.push_back(entry);
            return;
        }
        let entry = match entry {
            Entry::Item(text) => Entry::Item(self.dialect.normalize(&text)),
            entry => entry,
        };
        let ends = !matches!(entry, Entry::Item(_));
        match &mut self.split {
            Split::Through => {}
            Split::Native(splitter) => {
                let chunks = match &entry {
                    Entry::Item(text) => splitter.push(text),
                    Entry::Flush | Entry::EndInput => splitter.flush(),
                };
                self.split_out.extend(chunks);
                self.split_done = ends;
            }
            Split::Host { tokenizer, current, opening, early, .. } => match current {
                Some(host) => match &entry {
                    Entry::Item(text) => host.stream().push_text(text.clone()),
                    Entry::Flush | Entry::EndInput => host.stream().end_input(),
                },
                // A segment with no text has nothing for a tokenizer to split.
                None if ends && opening.is_none() && early.is_empty() => self.split_done = true,
                None => {
                    if opening.is_none() {
                        let (tokenizer, cx) = (tokenizer.clone(), self.cx.clone());
                        *opening = Some(Box::pin(async move {
                            tokenizer.stream(&cx).await.map_err(ProviderError::from)
                        }));
                    }
                    early.push(entry.clone());
                }
            },
        }
        if ends {
            self.segment_end = Some(entry);
        }
    }

    fn opened(&mut self, opened: Result<Arc<dyn HostSentenceStream>, ProviderError>) {
        let Split::Host { current, opening, early, .. } = &mut self.split else { return };
        *opening = None;
        let stream = match opened {
            Ok(stream) => stream,
            Err(error) => {
                self.failed = Some(error);
                return;
            }
        };
        for entry in early.drain(..) {
            match entry {
                Entry::Item(text) => stream.push_text(text),
                Entry::Flush | Entry::EndInput => stream.end_input(),
            }
        }
        *current = Some(Box::new(HostSentences::read(stream, &self.cx)));
    }

    fn split_read(&mut self, next: Result<Option<String>, ProviderError>) {
        match next {
            Ok(Some(sentence)) => {
                self.glued.push_str(&sentence);
                if speakable_alone(&self.glued, &self.tags) {
                    let sentence = std::mem::take(&mut self.glued);
                    self.split_out.extend(self.within_limit(sentence));
                }
            }
            Ok(None) => {
                if !self.glued.is_empty() {
                    let sentence = std::mem::take(&mut self.glued);
                    self.split_out.extend(self.within_limit(sentence));
                }
                self.split_done = true;
                if let Split::Host { current, ended, .. } = &mut self.split {
                    ended.extend(current.take().map(|host| *host));
                }
            }
            Err(error) => {
                self.failed = Some(error);
                self.split_done = true;
            }
        }
    }

    /// A host tokenizer's sentence, cut where it passes the provider's request limit (at a clause
    /// mark or word boundary, never inside a tag) and otherwise as the tokenizer chunked it.
    fn within_limit(&self, sentence: String) -> Vec<String> {
        let Some(limit) = self.limit.filter(|limit| sentence.chars().count() > *limit) else {
            return vec![sentence];
        };
        let mut cut = SentenceStream::new(SentenceOptions {
            min_words: usize::MAX,
            max_token_len: Some(limit),
            retain_format: true,
            markup: Some(self.tags.clone()),
            ..SentenceOptions::default()
        });
        let mut pieces = cut.push(&sentence);
        pieces.extend(cut.flush());
        pieces
    }

    /// A chunk as the provider takes it, lowered and trimmed; `None` when it sends nothing.
    fn lowered(&self, chunk: &str) -> Option<SpeechChunk> {
        let lowered = self.dialect.lower(&self.dialect.normalize(trim_python_whitespace(chunk)));
        let text = trim_python_whitespace(&lowered.text);
        (!text.is_empty()).then(|| SpeechChunk {
            text: text.to_owned(),
            said: trim_python_whitespace(&lowered.said).to_owned(),
        })
    }

    fn pace(&mut self, audio: f64) {
        let chunks: Vec<String> = self.split_out.drain(..).collect();
        let Some(pacer) = self.pacer.as_mut() else {
            let ready: Vec<_> = chunks.iter().filter_map(|c| self.lowered(c)).collect();
            self.ready.extend(ready.into_iter().map(Entry::Item));
            return;
        };
        for chunk in chunks {
            pacer.push(chunk);
        }
        if self.split_done {
            pacer.end();
        }
        let now = self.clock.now();
        let paced = pacer.poll(now.as_secs_f64(), audio);
        self.pace_at = paced.delay.map(|delay| now.saturating_add(secs(delay)));
        if let Some(text) = paced.text {
            sent(&text, paced.remaining);
            if let Some(text) = self.lowered(&text) {
                self.ready.push_back(Entry::Item(text));
            }
        }
    }
}

#[cfg(test)]
#[path = "stage_tests.rs"]
mod tests;
