use crate::generation::effects::{
    AlignedTranscriptEffect, DrainedText, StreamingTtsEffect, SynthesisLossEffect,
};
use crate::transcription::{Heard, Visible};
use lk_agents_abi::TranscriptChunk;
use lk_agents_abi::errors::ProviderError;

/// About five seconds of speculative 20 ms frames. Runtime backpressures the provider at this bound.
pub const MAX_BUFFERED_TTS_FRAMES: usize = 256;

/// Whether a subsequent source read is still needed.
pub enum Read {
    Provider,
    End,
    Failed(ProviderError),
}

pub struct StreamingTts {
    pub words: Visible,
    pub input_done: bool,
    ended: bool,
    send_error: Option<ProviderError>,
}

impl StreamingTts {
    pub fn new(words: Visible, input_done: bool) -> Self {
        Self { words, input_done, ended: false, send_error: None }
    }

    pub fn before_read(&mut self) -> Read {
        if self.ended {
            Read::End
        } else if let Some(error) = self.send_error.take() {
            Read::Failed(error)
        } else {
            Read::Provider
        }
    }

    /// The transcript delta precedes the same delta's synthesis input.
    pub fn text(&mut self, chunk: Option<Visible>) -> Vec<StreamingTtsEffect> {
        match chunk {
            Some(chunk) => vec![StreamingTtsEffect::Capture(chunk)],
            None => {
                self.input_done = true;
                vec![StreamingTtsEffect::EndInput, StreamingTtsEffect::FlushText]
            }
        }
    }

    /// Capturing a transcript delta completed; only now does the feed include its words.
    pub fn captured(&mut self, chunk: Visible) -> StreamingTtsEffect {
        let said = chunk.said.clone();
        self.words.extend(chunk);
        StreamingTtsEffect::Push(said)
    }

    /// A rejected text push ends feeding. Its error follows the source read already in flight.
    pub fn push_failed(&mut self, error: ProviderError) {
        self.send_error = Some(error);
        self.input_done = true;
    }

    /// A rejected opening feed is reported before the first source read.
    pub fn opening_pushed(&mut self, error: Option<ProviderError>) -> bool {
        let accepted = error.is_none();
        self.send_error = error;
        self.input_done && accepted
    }

    pub fn source_ended(&mut self) {
        self.ended = true;
    }

    /// A pooled source can stay open after its final frame. End audio once input is complete.
    pub fn frame(&mut self, is_final: bool) -> bool {
        if is_final && self.input_done {
            self.ended = true;
            true
        } else {
            false
        }
    }
}

/// Stamped words own an aligned transcript after the first stamp: each of the voice's timed words
/// is mapped onto the next spoken word of the transcript, so the transcript keeps its own text
/// (markdown, emoji, code) and paces it on the voice's timings. An unstamped segment falls back to
/// the transcript whole. Closing flushes it exactly once.
#[derive(Clone, Default)]
pub struct AlignedTranscript {
    /// The transcript shown so far, as the transcript chain got it.
    pub words: String,
    pub stamped: bool,
    pub closed: bool,
    /// Where the next stamp maps: the chunk of `said.shown`, and the byte within it.
    chunk: usize,
    byte: usize,
    /// Stamps that found no word yet (a voice that expanded "$5" to "five dollars"): the words
    /// the next matching stamp skips over take their time.
    held: Vec<TranscriptChunk>,
    /// Where the last word shown started: silent text after it (an emoji, closing markup) shows
    /// with it at the end.
    last_start: Option<f64>,
}

/// How far ahead a stamp may find its word: a voice that read "$5" as "five dollars" stamps words
/// the transcript does not have, and one that merged two words skips one.
const LOOKAHEAD: usize = 4;

/// A piece of the transcript still to show: silent text, or one spoken word with its lead. A word
/// the reply's deltas split (`Re` + `f`), or markup glued to it (`**` + `Upgrade`), is one piece of
/// several transcript chunks.
struct Pending {
    chunks: Vec<TranscriptChunk>,
    /// The word, normalized for matching; `None` for silent text.
    word: Option<String>,
    /// Where the cursor is once this piece is shown.
    next: (usize, usize),
}

impl Pending {
    fn text(&self) -> &str {
        self.chunks.last().map_or("", |c| c.text.as_str())
    }

    /// Whether `next` continues the word this piece ends.
    fn continued_by(&self, next: &Pending) -> bool {
        crate::transcription::continues_word(
            self.text(),
            next.chunks.first().map_or("", |c| &c.text),
        )
    }
}

fn normalized(word: &str) -> String {
    word.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

impl AlignedTranscript {
    /// The pieces from the cursor on: silent chunks whole, spoken chunks word by word, a word
    /// split across chunks joined.
    fn pending(&self, said: &Visible) -> Vec<Pending> {
        let mut out: Vec<Pending> = Vec::new();
        for slice in self.slices(said) {
            match out.last_mut() {
                Some(last) if last.continued_by(&slice) => {
                    last.chunks.extend(slice.chunks);
                    last.word = match (last.word.take(), slice.word) {
                        (None, None) => None,
                        (a, b) => Some(a.unwrap_or_default() + &b.unwrap_or_default()),
                    };
                    last.next = slice.next;
                }
                _ => out.push(slice),
            }
        }
        out
    }

    /// The pieces from the cursor on as each chunk cuts them.
    fn slices(&self, said: &Visible) -> Vec<Pending> {
        let mut out = Vec::new();
        let (mut chunk, mut byte) = (self.chunk, self.byte);
        while let Some(piece) = said.shown.get(chunk) {
            let rest = &piece.text[byte.min(piece.text.len())..];
            if !piece.spoken || rest.is_empty() {
                if !rest.is_empty() || byte == 0 {
                    out.push(Pending {
                        chunks: vec![TranscriptChunk { text: rest.to_owned(), ..piece.clone() }],
                        word: None,
                        next: (chunk + 1, 0),
                    });
                }
                (chunk, byte) = (chunk + 1, 0);
                continue;
            }
            // one word with the blank before it; a trailing blank goes with the last word
            let lead = rest.len() - rest.trim_start().len();
            let len = rest[lead..].find(char::is_whitespace).map_or(rest.len(), |i| lead + i);
            let tail = rest[len..].len() - rest[len..].trim_start().len();
            let end = if len + tail == rest.len() { rest.len() } else { len };
            let expression = if byte == 0 { piece.expression.clone() } else { None };
            out.push(Pending {
                chunks: vec![TranscriptChunk { expression, ..TranscriptChunk::new(&rest[..end]) }],
                word: Some(normalized(&rest[lead..len])),
                next: if byte + end == piece.text.len() {
                    (chunk + 1, 0)
                } else {
                    (chunk, byte + end)
                },
            });
            if byte + end == piece.text.len() {
                (chunk, byte) = (chunk + 1, 0);
            } else {
                byte += end;
            }
        }
        out
    }

    /// `said` is what the voice was given so far, as the transcript shows it. The stamp releases
    /// the transcript up to the word it says, its silent text first, at no cost. A stamp that
    /// finds no word nearby is held; the words the next matching stamp skips over take the held
    /// stamps' time. Too many held stamps map by position, so the transcript never drifts far.
    pub fn word(
        &mut self,
        stamp: &TranscriptChunk,
        said: &Visible,
    ) -> Vec<AlignedTranscriptEffect> {
        self.stamped = true;
        let pending = self.pending(said);
        let words: Vec<usize> =
            pending.iter().enumerate().filter(|(_, p)| p.word.is_some()).map(|(i, _)| i).collect();
        // a word the reply has not finished writing (its text ends mid-word) takes only a stamp
        // that says it whole: position never maps a stamp onto it, the rest of it may yet come
        let unfinished = |i: usize| {
            i + 1 == pending.len() && pending[i].text().ends_with(|c: char| !c.is_whitespace())
        };
        let said_word = normalized(&stamp.text);
        let found = words
            .iter()
            .take(LOOKAHEAD)
            .copied()
            .find(|&i| pending[i].word.as_deref() == Some(said_word.as_str()));
        let upto = match found {
            Some(i) => i,
            // the voice is ahead of the transcript it was given: show what it said
            None if words.is_empty() => {
                self.words.push_str(&stamp.text);
                self.last_start = stamp.start_time;
                return vec![AlignedTranscriptEffect::CaptureTimed(stamp.clone())];
            }
            None if self.held.len() < LOOKAHEAD || unfinished(words[0]) => {
                self.held.push(stamp.clone());
                return Vec::new();
            }
            // nothing matches for long: map the oldest held stamp onto the next word
            None => {
                self.held.push(stamp.clone());
                let oldest = self.held.remove(0);
                return self.release(pending, words[0], &oldest, &oldest);
            }
        };
        // the words skipped over take the held stamps' span, the matched one its own stamp
        let skipped = std::mem::take(&mut self.held);
        let span = TranscriptChunk {
            start_time: skipped.first().map_or(stamp.start_time, |h| h.start_time),
            end_time: skipped.last().map_or(stamp.start_time, |h| h.end_time),
            ..TranscriptChunk::default()
        };
        self.release(pending, upto, &span, stamp)
    }

    /// Show `pending` up to `upto`: silent text at its stamp's start, the last word at `stamp`,
    /// any words before it over `skipped`.
    fn release(
        &mut self,
        pending: Vec<Pending>,
        upto: usize,
        skipped: &TranscriptChunk,
        stamp: &TranscriptChunk,
    ) -> Vec<AlignedTranscriptEffect> {
        let mut effects = Vec::new();
        for (i, piece) in pending.into_iter().take(upto + 1).enumerate() {
            let (start_time, end_time) = match piece.word {
                Some(_) if i == upto => (stamp.start_time, stamp.end_time),
                Some(_) => (skipped.start_time, skipped.end_time),
                None => (skipped.start_time, skipped.start_time),
            };
            (self.chunk, self.byte) = piece.next;
            self.last_start = start_time;
            for chunk in piece.chunks {
                self.words.push_str(&chunk.text);
                // syntax glued to its word shows when the word starts, at no cost
                let end_time = if chunk.spoken { end_time } else { start_time };
                effects.push(AlignedTranscriptEffect::CaptureTimed(TranscriptChunk {
                    start_time,
                    end_time,
                    ..chunk
                }));
            }
        }
        effects
    }

    /// With nothing stamped, the transcript is forwarded whole, as off the aligned path; after
    /// stamps, what no stamp reached is shown at the end.
    pub fn end(&mut self, said: &Visible) -> Vec<AlignedTranscriptEffect> {
        if std::mem::replace(&mut self.closed, true) {
            return Vec::new();
        }
        let mut effects = Vec::new();
        if self.stamped {
            let mut pending = self.pending(said).into_iter().peekable();
            // silent text after the last word shown is the voice's too: it shows with that word
            while let Some(start_time) = self.last_start
                && let Some(piece) = pending.next_if(|p| p.word.is_none())
            {
                for chunk in piece.chunks.into_iter().filter(|c| !c.text.is_empty()) {
                    self.words.push_str(&chunk.text);
                    effects.push(AlignedTranscriptEffect::CaptureTimed(TranscriptChunk {
                        start_time: Some(start_time),
                        end_time: Some(start_time),
                        ..chunk
                    }));
                }
            }
            let rest: Vec<TranscriptChunk> =
                pending.flat_map(|p| p.chunks).filter(|c| !c.text.is_empty()).collect();
            for chunk in &rest {
                self.words.push_str(&chunk.text);
            }
            if !rest.is_empty() {
                effects.push(AlignedTranscriptEffect::Capture(rest));
            }
        } else {
            self.words.push_str(&said.shown_text());
            effects.push(AlignedTranscriptEffect::WarnUnaligned);
            effects.push(AlignedTranscriptEffect::Capture(said.shown.clone()));
        }
        effects.push(AlignedTranscriptEffect::Flush);
        effects
    }
}

/// One synthesis request's observations. All timestamps are supplied by the executor.
pub struct SynthesisMetrics {
    pub segment: usize,
    synthesis_started_at: f64,
    started_at: f64,
    metrics: lk_agents_core::metrics::provider::TtsMetricState,
    metered_calls: bool,
}

impl SynthesisMetrics {
    pub fn new(
        provider: lk_agents_abi::ProviderId,
        segment: usize,
        synthesis_started_at: f64,
        started_at: f64,
    ) -> Self {
        Self {
            segment,
            synthesis_started_at,
            started_at,
            metrics: lk_agents_core::metrics::provider::TtsMetricState::new(
                true,
                provider.label.clone(),
                Some(provider.metadata()),
            ),
            metered_calls: false,
        }
    }

    pub fn produced(&self) -> bool {
        self.metrics.ttfb().is_some()
    }

    /// `now` is read only for the first audio, the one reading kept.
    pub fn note(
        &mut self,
        chunk: &lk_agents_abi::SynthesizedAudio,
        now: impl FnOnce() -> f64,
    ) -> Option<super::TtsStats> {
        let first = (!self.produced()).then(now);
        self.metrics.observe(chunk, || first.map_or(0.0, |now| now - self.started_at));
        first.map(|now| super::TtsStats {
            ttfb: Some(now - chunk.started_time.unwrap_or(self.synthesis_started_at)),
            synthesis_started_at: chunk.started_time,
        })
    }

    pub fn calls(
        &mut self,
        records: Vec<lk_agents_abi::metrics::base::AgentMetrics>,
        speech_id: &str,
    ) -> Vec<lk_agents_abi::metrics::base::AgentMetrics> {
        self.metered_calls = true;
        records
            .into_iter()
            .map(|mut record| {
                if let lk_agents_abi::metrics::base::AgentMetrics::Tts(tts) = &mut record {
                    tts.speech_id = Some(speech_id.to_owned());
                }
                record
            })
            .collect()
    }

    pub fn finish(
        mut self,
        text: &str,
        now: f64,
        interrupted: bool,
        speech_id: &str,
    ) -> Option<lk_agents_abi::metrics::base::AgentMetrics> {
        if self.metered_calls {
            return None;
        }
        let mut record = self.metrics.finish(now, self.started_at, text, now, interrupted)?;
        record.speech_id = Some(speech_id.to_owned());
        Some(lk_agents_abi::metrics::base::AgentMetrics::Tts(record))
    }
}

/// The outcome of synthesis that did not produce a completed segment.
pub struct Lost {
    pub failure: Option<crate::agent_session::DriverError>,
    pub lost: bool,
    pub opening: Visible,
    pub synchronized: Option<String>,
    pub played: bool,
    /// What the sink confirmed of the failed attempt's audio: its frames and playhead.
    pub heard: Heard,
}

/// The input and playout accumulated during one synthesis.
pub struct SynthesisAttempt {
    pub opening: Visible,
    pub input_done: bool,
    synchronized: Option<String>,
    heard: Heard,
}

impl SynthesisAttempt {
    pub fn new(opening: Visible) -> Self {
        Self { opening, input_done: false, synchronized: None, heard: Heard::default() }
    }

    pub fn before_open(&self, interrupted: bool, closing: bool) -> Option<Lost> {
        (interrupted || closing).then(|| self.stopped())
    }

    pub fn stream_failed(
        &mut self,
        opening: Visible,
        input_done: bool,
        synchronized: Option<String>,
        heard: Option<Heard>,
    ) {
        self.opening = opening;
        self.input_done = input_done;
        self.synchronized = synchronized;
        self.heard = heard.unwrap_or_default();
    }

    pub fn failed(&mut self, error: crate::agent_session::DriverError, produced: bool) -> Lost {
        Lost {
            lost: matches!(error, crate::agent_session::DriverError::Provider(_)),
            failure: match error {
                crate::agent_session::DriverError::Provider(_) => None,
                error => Some(error),
            },
            opening: self.opening.clone(),
            synchronized: self.synchronized.clone(),
            played: produced,
            heard: self.heard.clone(),
        }
    }

    /// Stopped by an interrupt or the close: nothing failed and nothing was lost.
    pub fn stopped(&self) -> Lost {
        Lost {
            failure: None,
            lost: false,
            opening: self.opening.clone(),
            synchronized: None,
            played: false,
            heard: Heard::default(),
        }
    }
}

/// How a lost synthesis settles the rest of its segment.
#[derive(Default)]
pub struct SynthesisLoss {
    pending: Option<LossText>,
    /// The sink's account of the audio that did play: the segment is voiced if any reached it.
    sink: Heard,
}

enum LossText {
    Unheard,
    Stamped(String),
    Generated { played: bool, synchronized: Option<String> },
}

pub enum SynthesisLossEvent {
    Failed { lost: Box<Lost>, aligned: Option<AlignedTranscript> },
    Drained(Visible),
}

impl SynthesisLoss {
    pub fn handle(&mut self, event: SynthesisLossEvent) -> Vec<SynthesisLossEffect> {
        match event {
            SynthesisLossEvent::Failed { lost, aligned } => self.failed(*lost, aligned),
            SynthesisLossEvent::Drained(words) => self.drained(words),
        }
    }

    /// The synthesis failed: a lost voice is released and cut short, and the drain forwards the
    /// model's text only when no accepted stamps stand for what was heard.
    fn failed(
        &mut self,
        lost: Lost,
        aligned: Option<AlignedTranscript>,
    ) -> Vec<SynthesisLossEffect> {
        if let Some(error) = lost.failure {
            return vec![SynthesisLossEffect::Fail(error)];
        }
        if !lost.lost {
            self.pending = Some(LossText::Unheard);
            return vec![SynthesisLossEffect::Drain {
                opening: lost.opening,
                text: DrainedText::Discard,
            }];
        }
        self.sink = lost.heard;
        let mut effects = vec![SynthesisLossEffect::ReleaseVoice, SynthesisLossEffect::CutShort];
        match aligned {
            Some(aligned) if aligned.stamped => {
                // Once stamps were accepted, model text can never become their fallback.
                // history keeps the transcript plain
                self.pending = Some(LossText::Stamped(crate::transcription::plain(&aligned.words)));
                effects.push(SynthesisLossEffect::Drain {
                    opening: lost.opening,
                    text: DrainedText::Discard,
                });
            }
            aligned => {
                if aligned.is_some_and(|a| !a.closed) {
                    effects.push(SynthesisLossEffect::Capture(lost.opening.shown.clone()));
                }
                let (played, synchronized) = (lost.played, lost.synchronized);
                self.pending = Some(LossText::Generated { played, synchronized });
                effects.push(SynthesisLossEffect::Drain {
                    opening: lost.opening,
                    text: DrainedText::Forward,
                });
            }
        }
        effects
    }

    /// The drain is done: what was heard, from the stamps, the synchronized text or the words.
    fn drained(&mut self, words: Visible) -> Vec<SynthesisLossEffect> {
        let Some(pending) = self.pending.take() else {
            return Vec::new();
        };
        let text = match pending {
            LossText::Unheard => return vec![SynthesisLossEffect::Heard(Heard::default())],
            LossText::Stamped(text) => text,
            LossText::Generated { played: true, synchronized } => {
                synchronized.unwrap_or(words.text)
            }
            LossText::Generated { played: false, .. } => String::new(),
        };
        let sink = std::mem::take(&mut self.sink);
        vec![SynthesisLossEffect::Heard(Heard { text, complete: true, ..sink })]
    }
}

#[cfg(test)]
#[path = "tts_tests.rs"]
mod tests;
