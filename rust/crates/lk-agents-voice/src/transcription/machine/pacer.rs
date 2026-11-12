use crate::transcription::Heard;
use crate::transcription::effects::TranscriptPacerEffect;
use crate::transcription::machine::synchronizer::{Hyphenator, Liang};
use lk_agents_abi::TranscriptChunk;
use lk_agents_core::tokenize::{WordStream, WordTokenizerOptions, split_words};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

/// The synchroniser's tokenizer settings: tokens must reassemble the input and punctuation is
/// paced.
pub const SYNC_TOKENIZER: WordTokenizerOptions =
    WordTokenizerOptions { ignore_punctuation: false, retain_format: true };

/// Hyphens (syllables) per second at `speed = 1.0`, before a segment's own rate is known.
pub const STANDARD_SPEECH_RATE: f64 = 3.83;

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub(in crate::transcription) enum TranscriptPacerEvent {
    /// One piece of the transcript: a spoken word's text, or text with no speech. `start_time`
    /// and `end_time` are the TTS's alignment for it, when the TTS aligns.
    Text {
        chunk: TranscriptChunk,
        start_time: Option<f64>,
        end_time: Option<f64>,
    },
    TextEnd,
    Audio {
        secs: f64,
    },
    AudioEnd,
    Position(Duration),
    /// The audio paused where the last position left it.
    Paused,
    PlaybackFinished {
        position: Duration,
        interrupted: bool,
    },
    /// The sink cut its playout at `position` while the segment's audio was still coming: what
    /// was heard so far is reported, and the audio that follows plays on from there, so the words
    /// left (and those still to come) pace against it.
    PlaybackSplit {
        position: Duration,
    },
    /// The sink's audio was cleared at `position`: what was heard by then is reported, and its
    /// transcript stands there. Only the playout's finish (or a close) follows.
    Cleared {
        position: Duration,
    },
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum TranscriptPacerWarning {
    /// Text after `TextEnd`: a missing rotation upstream.
    TextAfterEnd,
    /// Audio after `AudioEnd`: a missing rotation upstream.
    AudioAfterEnd,
    /// `PlaybackFinished` before both inputs ended. The segment finishes anyway on what it has.
    PlaybackFinishedTooEarly,
    AfterFinished,
    AfterClose,
}

impl TranscriptPacerWarning {
    pub fn message(self) -> &'static str {
        match self {
            TranscriptPacerWarning::TextAfterEnd => {
                "transcript synchronizer: text pushed after its input ended"
            }
            TranscriptPacerWarning::AudioAfterEnd => {
                "transcript synchronizer: audio pushed after its input ended"
            }
            TranscriptPacerWarning::PlaybackFinishedTooEarly => {
                "transcript synchronizer: playback_finished called before text/audio input is done"
            }
            TranscriptPacerWarning::AfterFinished => {
                "transcript synchronizer: segment used after it finished"
            }
            TranscriptPacerWarning::AfterClose => {
                "transcript synchronizer: segment used after close"
            }
        }
    }
}

/// Where a segment is. The enum is the whole state; the accumulators beside it are data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::transcription) enum TranscriptPacerState {
    Streaming,
    TextEnded,
    AudioEnded,
    /// Both inputs ended; waiting for the playout to finish.
    InputsEnded,
    /// The sink's audio was cleared: no word is shown any more, and the playout's finish reports
    /// the words heard among those shown.
    Cleared,
    /// `Heard` was produced. Only `Close` is left.
    Finished,
    Closed,
}

#[derive(Debug, Clone, PartialEq)]
struct Planned {
    /// What the transcript shows for it.
    chunk: TranscriptChunk,
    hyphens: usize,
    offset: usize,
    /// The TTS's `start_time` for the delta this word came from, or the segment start.
    base: Option<f64>,
    /// The TTS's `end_time` for that delta: its audio is over by then whatever the pace says.
    limit: Option<f64>,
    /// The playhead when the word was planned: a word that arrives late starts no earlier.
    arrived: f64,
}

impl Planned {
    /// Its delta's onset plus the hyphens before it, at the pace; no later than the delta's end.
    fn due(&self, speed: f64) -> f64 {
        let paced = self.base.unwrap_or(0.0) + self.offset as f64 / speed;
        self.limit.map_or(paced, |limit| paced.min(limit))
    }

    fn end(&self, speed: f64) -> f64 {
        let (due, paced) = (self.due(speed), self.due(speed) + self.hyphens as f64 / speed);
        self.limit.map_or(paced, |limit| paced.min(limit).max(due))
    }
}

pub(in crate::transcription) struct TranscriptPacer {
    state: TranscriptPacerState,
    opts: TranscriptPacerOptions,
    speed: f64,
    /// A token is emitted once a following one proves it complete.
    words: WordStream,
    /// A delivery that arrived with no word yet: the next planned piece carries it.
    expression: Option<String>,
    queue: VecDeque<Planned>,
    /// Hyphens planned since `base` moved (or since the segment start).
    offset: usize,
    base: Option<f64>,
    planned_hyphens: usize,
    /// The TTS's `end_time` for the current aligned delta.
    limit: Option<f64>,
    /// The words shown, each with where its audio ends. Heard are the ones whose audio ended.
    forwarded: Vec<(String, f64)>,
    /// Where the next unaligned word starts: the end of the last one shown (`None` before any).
    cursor: Option<f64>,
    /// The front word's `(start, end)`, fixed at the speed known when it started.
    front: Option<(f64, f64)>,
    audio_secs: f64,
    frames: u64,
    /// The playhead in the segment's own time, every playout since the last split continuing it.
    position: Duration,
    /// Where the sink's current playout started in the segment's time: the playhead of the last
    /// split.
    origin: Duration,
    saw_text: bool,
}

impl TranscriptPacer {
    pub fn new(opts: TranscriptPacerOptions) -> TranscriptPacer {
        let speed = STANDARD_SPEECH_RATE * opts.speed;
        TranscriptPacer {
            state: TranscriptPacerState::Streaming,
            opts,
            speed,
            words: WordStream::new(SYNC_TOKENIZER),
            expression: None,
            queue: VecDeque::new(),
            offset: 0,
            base: None,
            planned_hyphens: 0,
            limit: None,
            forwarded: Vec::new(),
            cursor: None,
            front: None,
            audio_secs: 0.0,
            frames: 0,
            position: Duration::ZERO,
            origin: Duration::ZERO,
            saw_text: false,
        }
    }

    /// The last playhead the sink told, in its current playout.
    pub fn position(&self) -> Duration {
        self.position.saturating_sub(self.origin)
    }

    pub fn is_blank(&self) -> bool {
        self.frames == 0 && !self.saw_text
    }

    pub fn saw_text(&self) -> bool {
        self.saw_text
    }

    #[cfg(test)]
    fn text_ended(&self) -> bool {
        matches!(self.state, TranscriptPacerState::TextEnded | TranscriptPacerState::InputsEnded)
    }

    #[cfg(test)]
    fn audio_ended(&self) -> bool {
        matches!(self.state, TranscriptPacerState::AudioEnded | TranscriptPacerState::InputsEnded)
    }

    /// `Heard` was produced, or the segment was closed: it takes nothing more but `Close`.
    pub fn is_over(&self) -> bool {
        matches!(self.state, TranscriptPacerState::Finished | TranscriptPacerState::Closed)
    }

    /// Its audio was cleared: it shows nothing more and awaits its playout's finish.
    pub fn is_cleared(&self) -> bool {
        self.state == TranscriptPacerState::Cleared
    }

    /// One event in, its effects out. The state x event table:
    ///
    /// | state \ event | Text | TextEnd | Audio | AudioEnd | Position | Finished | Split | Close |
    /// |---|---|---|---|---|---|---|---|---|
    /// | Streaming | plan | -> TextEnded | count | -> AudioEnded (1) | release | early (3) | split | cut |
    /// | TextEnded | warn | warn | count | -> InputsEnded (1) | release | early (3) | split | cut |
    /// | AudioEnded | plan | -> InputsEnded (1) | warn | warn | release | early (3) | finish (2) | cut |
    /// | InputsEnded | warn | warn | warn | warn | release | finish | finish (2) | cut |
    /// | Cleared | - | - | - | - | - | settle (4) | - | cut |
    /// | Finished | warn | warn | warn | warn | warn | warn | warn | -> Closed |
    /// | Closed | warn | warn | warn | warn | warn | warn | warn | warn |
    ///
    /// (1) re-estimates the speed. A segment whose inputs ended with no audio shows its text whole
    /// at once: nothing will play it. (2) Interrupted: with its audio over, nothing plays on after
    /// the cut. (3) Finishes, warned too early. (4) Interrupted wherever the sink says it
    /// stopped: the words heard among those shown, with nothing more shown. `Cleared` comes from
    /// any state but `Finished` and `Closed`; it releases up to its position and reports the words
    /// heard there.
    ///
    /// Positions are the sink's, in its current playout; the pacer moves them past the last
    /// split.
    #[must_use = "an effect not performed is a word the user never sees"]
    pub fn step(
        mut self,
        ev: TranscriptPacerEvent,
    ) -> (TranscriptPacer, Vec<TranscriptPacerEffect>) {
        use TranscriptPacerState as S;
        let mut out = Vec::new();
        let warn =
            |out: &mut Vec<TranscriptPacerEffect>, w| out.push(TranscriptPacerEffect::Warn(w));
        match (self.state, ev) {
            (S::Closed, _) => warn(&mut out, TranscriptPacerWarning::AfterClose),
            (S::Finished, TranscriptPacerEvent::Close) => self.state = S::Closed,
            (S::Finished, _) => warn(&mut out, TranscriptPacerWarning::AfterFinished),
            (S::Cleared, TranscriptPacerEvent::Close) => self.cut(&mut out),
            (S::Cleared, TranscriptPacerEvent::PlaybackFinished { position, .. }) => {
                self.settle(self.origin + position, &mut out)
            }
            (S::Cleared, _) => {}
            (_, TranscriptPacerEvent::Cleared { position }) => {
                self.clear(self.origin + position, &mut out)
            }
            (_, TranscriptPacerEvent::Position(p)) => self.release(self.origin + p, &mut out),
            (_, TranscriptPacerEvent::Paused) => self.paused(&mut out),
            (_, TranscriptPacerEvent::Close) => self.cut(&mut out),
            (state, TranscriptPacerEvent::PlaybackFinished { position, interrupted }) => {
                if state != S::InputsEnded {
                    warn(&mut out, TranscriptPacerWarning::PlaybackFinishedTooEarly);
                }
                self.finish(self.origin + position, interrupted, &mut out);
            }
            (S::Streaming | S::TextEnded, TranscriptPacerEvent::PlaybackSplit { position }) => {
                self.split(self.origin + position, &mut out)
            }
            (_, TranscriptPacerEvent::PlaybackSplit { position }) => {
                self.finish(self.origin + position, true, &mut out)
            }
            (
                S::Streaming | S::AudioEnded,
                TranscriptPacerEvent::Text { chunk, start_time, end_time },
            ) => self.plan(chunk, start_time, end_time),
            (S::Streaming, TranscriptPacerEvent::TextEnd) => {
                self.end_text();
                self.state = S::TextEnded;
            }
            (S::AudioEnded, TranscriptPacerEvent::TextEnd) => {
                self.end_text();
                self.inputs_ended(&mut out);
            }
            (_, TranscriptPacerEvent::Text { .. } | TranscriptPacerEvent::TextEnd) => {
                warn(&mut out, TranscriptPacerWarning::TextAfterEnd)
            }
            (S::Streaming | S::TextEnded, TranscriptPacerEvent::Audio { secs }) => {
                self.count_audio(secs)
            }
            (S::Streaming, TranscriptPacerEvent::AudioEnd) => self.end_audio(),
            (S::TextEnded, TranscriptPacerEvent::AudioEnd) => self.inputs_ended(&mut out),
            (_, TranscriptPacerEvent::Audio { .. } | TranscriptPacerEvent::AudioEnd) => {
                warn(&mut out, TranscriptPacerWarning::AudioAfterEnd)
            }
        }
        (self, out)
    }

    fn plan(&mut self, chunk: TranscriptChunk, start_time: Option<f64>, end_time: Option<f64>) {
        self.saw_text = true;
        if let Some(t) = start_time {
            // An aligned delta is a word boundary: the held word belongs to the previous onset.
            let held = self.words.flush();
            self.plan_tokens(held);
            self.base = Some(t);
            self.limit = end_time.filter(|end| *end >= t);
            self.offset = 0;
        }
        if chunk.expression.is_some() {
            self.expression.clone_from(&chunk.expression);
        }
        if chunk.spoken {
            let tokens = self.words.push(&chunk.text);
            self.plan_tokens(tokens);
        } else {
            // text with no speech: the word before it is whole, and it costs no time
            let held = self.words.flush();
            self.plan_tokens(held);
            self.plan_piece(TranscriptChunk { expression: None, ..chunk }, 0);
        }
    }

    fn plan_tokens(&mut self, tokens: Vec<String>) {
        for word in tokens {
            let hyphens = self.hyphens(&word);
            self.plan_piece(TranscriptChunk::new(word), hyphens);
        }
    }

    fn plan_piece(&mut self, mut chunk: TranscriptChunk, hyphens: usize) {
        if chunk.text.is_empty() && self.expression.is_none() {
            return;
        }
        chunk.expression = self.expression.take();
        self.queue.push_back(Planned {
            chunk,
            hyphens,
            offset: self.offset,
            base: self.base,
            limit: self.limit,
            arrived: self.position.as_secs_f64(),
        });
        self.offset += hyphens;
        self.planned_hyphens += hyphens;
    }

    /// Whitespace and markup cost nothing; everything else costs its hyphens.
    fn hyphens(&self, clean: &str) -> usize {
        if clean.trim().is_empty() {
            return 0;
        }
        split_words(clean, SYNC_TOKENIZER)
            .iter()
            .map(|w| self.opts.hyphenator.hyphens(&w.text))
            .sum()
    }

    /// The text ended: the held partial word is a whole word now, and a delivery with no word
    /// after it still reaches the transcript.
    fn end_text(&mut self) {
        let tail = self.words.flush();
        self.plan_tokens(tail);
        if self.expression.is_some() {
            self.plan_piece(TranscriptChunk::silent(""), 0);
        }
    }

    fn count_audio(&mut self, secs: f64) {
        if secs.is_finite() && secs > 0.0 {
            self.audio_secs += secs;
        }
        self.frames += 1;
    }

    fn end_audio(&mut self) {
        self.state = TranscriptPacerState::AudioEnded;
    }

    /// Both inputs ended: the segment's own rate is known.
    fn inputs_ended(&mut self, out: &mut Vec<TranscriptPacerEffect>) {
        if self.frames == 0 {
            // Nothing will play it: shown whole now, as a segment that finished playing.
            self.finish(Duration::ZERO, false, out);
            return;
        }
        self.state = TranscriptPacerState::InputsEnded;
        // A markup- or emoji-only segment would otherwise set the speed to zero.
        if self.planned_hyphens > 0 && self.audio_secs > 0.0 {
            self.speed = self.planned_hyphens as f64 / self.audio_secs;
        }
    }

    /// Each word starts where the previous one ended (or when it arrived), lasts its hyphens at the
    /// speed known then, and is shown halfway through. An aligned word keeps the TTS's own timing.
    fn release(&mut self, position: Duration, out: &mut Vec<TranscriptPacerEffect>) {
        self.position = position;
        let now = position.as_secs_f64();
        // The aligned delta's audio is over: its held word is whole, as any continuation would
        // be a later delta starting at a new onset.
        if self.limit.is_some_and(|end| now >= end) {
            let held = self.words.flush();
            self.plan_tokens(held);
        }
        while let Some((start, end, shown_at)) = self.front_times() {
            // Only a word whose audio has started is in its delay (what a pause shows).
            if start > now + 1e-9 {
                break;
            }
            self.front = Some((start, end));
            if shown_at > now + 1e-9 {
                break;
            }
            let word = self.queue.pop_front().expect("peeked");
            self.front = None;
            self.cursor = Some(end);
            self.forward(word, start, end, out);
        }
    }

    /// The front word's `(start, end, shown_at)` in the segment's time, at the speed known now
    /// unless its start already fixed it.
    fn front_times(&self) -> Option<(f64, f64, f64)> {
        let front = self.queue.front()?;
        let (start, end) = match (self.front, front.base) {
            (Some(timed), _) => timed,
            (None, Some(base)) => {
                let (start, end) = (front.due(self.speed), front.end(self.speed));
                // The next stamped delta starts where this one's audio is over: a token paced
                // past it (the space after a word) must not hold it back.
                match self.queue.get(1).and_then(|w| w.base).filter(|next| *next > base) {
                    Some(next) => (start.min(next), end.min(next).max(start.min(next))),
                    None => (start, end),
                }
            }
            (None, None) => {
                let start = self.cursor.unwrap_or(0.0).max(front.arrived);
                (start, start + front.hyphens as f64 / self.speed)
            }
        };
        let shown_at = if front.base.is_some() { start } else { (start + end) / 2.0 };
        Some((start, end, shown_at))
    }

    /// The next playhead, in the sink's current playout, at which a position would release
    /// something: the front word's start or its showing, or the end of the aligned delta whose
    /// last word is held. `None` when nothing waits on the playhead.
    pub fn next_due(&self) -> Option<Duration> {
        if self.is_cleared() {
            return None;
        }
        let now = self.position.as_secs_f64();
        let front = self.front_times().map(
            |(start, _, shown_at)| {
                if start > now + 1e-9 { start } else { shown_at }
            },
        );
        let held = self.limit.filter(|end| *end > now);
        let due = match (front, held) {
            (Some(a), Some(b)) => a.min(b),
            (due, None) | (None, due) => due?,
        };
        Some(Duration::from_secs_f64(due.max(0.0)).saturating_sub(self.origin))
    }

    /// The audio the sink was sent in its current playout: no playhead estimate passes it.
    pub fn audio_sent(&self) -> Duration {
        Duration::from_secs_f64(self.audio_secs).saturating_sub(self.origin)
    }

    /// The word whose delay had begun is still shown; only the next word waits.
    fn paused(&mut self, out: &mut Vec<TranscriptPacerEffect>) {
        let Some((start, end)) = self.front.take() else { return };
        let word = self.queue.pop_front().expect("a started word is the queue's front");
        self.cursor = Some(end);
        self.forward(word, start, end, out);
    }

    fn forward(
        &mut self,
        word: Planned,
        start_time: f64,
        end_time: f64,
        out: &mut Vec<TranscriptPacerEffect>,
    ) {
        // The words are one voice: the last one's audio was over when this one's began, whatever
        // pace it was shown at (a sentence's first word paced before its rate was known).
        if let Some((_, end)) = self.forwarded.last_mut() {
            *end = end.min(start_time);
        }
        self.forwarded.push((word.chunk.text.clone(), end_time));
        out.push(TranscriptPacerEffect::Forward { chunk: word.chunk, start_time, end_time });
    }

    /// The words shown whose audio had ended by `position`.
    /// As history keeps it: plain, and ending at the last word heard.
    fn heard(&self, position: Duration) -> String {
        let at = position.as_secs_f64() + 1e-9;
        let shown: String = self
            .forwarded
            .iter()
            .take_while(|(_, end)| *end <= at)
            .map(|(w, _)| w.as_str())
            .collect();
        crate::transcription::plain(shown.trim_end())
    }

    /// The playout ended. Complete: everything left is heard, at once. Cut: what was forwarded by
    /// the playhead it reports. A sink that reports no progress (an avatar's data stream) paces
    /// nothing before this.
    fn finish(
        &mut self,
        position: Duration,
        interrupted: bool,
        out: &mut Vec<TranscriptPacerEffect>,
    ) {
        if interrupted {
            self.release(position, out);
            self.queue.clear();
        } else {
            self.end_text();
            let end = position.as_secs_f64();
            while let Some(word) = self.queue.pop_front() {
                // Stamped no later than the playout's own end: the audio is over.
                let start_time = word.due(self.speed);
                let end_time = word.end(self.speed).min(end.max(start_time));
                self.forward(word, start_time, end_time, out);
            }
        }
        out.push(TranscriptPacerEffect::Flush);
        let heard = if interrupted {
            Heard::cut(self.heard(position), position, self.frames)
        } else {
            let all: String = self.forwarded.iter().map(|(w, _)| w.as_str()).collect();
            Heard::complete(crate::transcription::plain(&all), position, self.frames)
        };
        out.push(TranscriptPacerEffect::Heard(heard));
        self.state = TranscriptPacerState::Finished;
    }

    /// The audio was cleared at `position`: the words due by then are shown, the rest never are,
    /// and the words heard are reported.
    fn clear(&mut self, position: Duration, out: &mut Vec<TranscriptPacerEffect>) {
        self.release(position, out);
        self.queue.clear();
        out.push(TranscriptPacerEffect::Flush);
        let heard = Heard::cut(self.heard(position), position, self.frames);
        out.push(TranscriptPacerEffect::Heard(heard));
        self.state = TranscriptPacerState::Cleared;
    }

    /// A cleared playout's finish at `position`: the words heard among those shown.
    fn settle(&mut self, position: Duration, out: &mut Vec<TranscriptPacerEffect>) {
        out.push(TranscriptPacerEffect::Flush);
        let heard = Heard::cut(self.heard(position), position, self.frames);
        out.push(TranscriptPacerEffect::Heard(heard));
        self.state = TranscriptPacerState::Finished;
    }

    /// The words heard by `position` are this playout's; the next playout starts there.
    fn split(&mut self, position: Duration, out: &mut Vec<TranscriptPacerEffect>) {
        self.release(position, out);
        out.push(TranscriptPacerEffect::Heard(Heard::cut(
            self.heard(position),
            position,
            self.frames,
        )));
        self.origin = position;
    }

    /// Closed before the playout finished: the words shown are the words heard.
    fn cut(&mut self, out: &mut Vec<TranscriptPacerEffect>) {
        self.queue.clear();
        out.push(TranscriptPacerEffect::Flush);
        let heard = self.heard(self.position);
        out.push(TranscriptPacerEffect::Heard(Heard::cut(heard, self.position, self.frames)));
        self.state = TranscriptPacerState::Closed;
    }
}

/// Pacing options; the word tokenizer is pinned to [`SYNC_TOKENIZER`].
#[derive(Clone)]
pub struct TranscriptPacerOptions {
    /// The caller's `transcription_speed_factor`. Multiplies [`STANDARD_SPEECH_RATE`].
    pub speed: f64,
    pub hyphenator: Arc<dyn Hyphenator>,
}

impl std::fmt::Debug for TranscriptPacerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptPacerOptions").field("speed", &self.speed).finish_non_exhaustive()
    }
}

impl Default for TranscriptPacerOptions {
    fn default() -> Self {
        TranscriptPacerOptions { speed: 1.0, hyphenator: Arc::new(Liang) }
    }
}

#[cfg(test)]
#[path = "pacer_tests.rs"]
mod tests;
