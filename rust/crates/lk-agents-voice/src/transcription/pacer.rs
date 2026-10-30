//! One segment's pacing: its words released against its audio's playhead, as the reference's
//! `_SegmentSynchronizerImpl` does.
use lk_agents_core::tokenize::{
    NativeTokenBuffer as TokenBuffer, TokenBufferOptions, WordTokenizerOptions, Words, split_words,
};
use crate::transcription::Heard;
use crate::transcription::synchronizer::{Hyphenator, Liang};
use lk_agents_core::tts::markup::TranscriptMarkupStripper;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

/// The synchroniser's tokenizer settings, from `TranscriptSynchronizer.__init__`: tokens must
/// reassemble the input, punctuation is paced, and CJK/Thai split per character.
pub const SYNC_TOKENIZER: WordTokenizerOptions =
    WordTokenizerOptions { ignore_punctuation: false, split_character: true, retain_format: true };

/// Hyphens (syllables) per second at `speed = 1.0`, before a segment's own rate is known.
pub const STANDARD_SPEECH_RATE: f64 = 3.83;

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum PaceEvent {
    /// One text delta, raw. `start_time` and `end_time` are the TTS's alignment for it, when the
    /// TTS aligns.
    Text {
        raw: String,
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
    Close,
}

#[must_use = "an effect not performed is a word the user never sees"]
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum PaceEffect {
    /// One raw token for the text output, with where its audio starts and ends.
    Forward {
        word: String,
        start_time: f64,
        end_time: f64,
    },
    Flush,
    /// Once per playout: at its finish, and at each split.
    Heard(Heard),
    /// An event the segment was not in a state to take. Nothing changed.
    Warn(PaceWarning),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum PaceWarning {
    /// Text after `TextEnd`: a missing rotation upstream.
    TextAfterEnd,
    /// Audio after `AudioEnd`: a missing rotation upstream.
    AudioAfterEnd,
    /// `PlaybackFinished` before both inputs ended. The segment finishes anyway on what it has.
    PlaybackFinishedTooEarly,
    AfterFinished,
    AfterClose,
}

impl PaceWarning {
    /// The reference's log line for it (`_SegmentSynchronizerImpl`).
    pub fn message(self) -> &'static str {
        match self {
            PaceWarning::TextAfterEnd => {
                "transcript synchronizer: text pushed after its input ended"
            }
            PaceWarning::AudioAfterEnd => {
                "transcript synchronizer: audio pushed after its input ended"
            }
            PaceWarning::PlaybackFinishedTooEarly => {
                "transcript synchronizer: playback_finished called before text/audio input is done"
            }
            PaceWarning::AfterFinished => "transcript synchronizer: segment used after it finished",
            PaceWarning::AfterClose => "transcript synchronizer: segment used after close",
        }
    }
}

/// Where a segment is. The enum is the whole state; the accumulators beside it are data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaceState {
    Streaming,
    TextEnded,
    AudioEnded,
    /// Both inputs ended; waiting for the playout to finish.
    InputsEnded,
    /// `Heard` was produced. Only `Close` is left.
    Finished,
    Closed,
}

#[derive(Debug, Clone, PartialEq)]
struct Planned {
    word: String,
    /// The token's visible text, from the segment's one stateful stripper.
    visible: String,
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

pub struct Pacer {
    state: PaceState,
    opts: PacerOptions,
    speed: f64,
    /// `basic.WordTokenizer(...).stream()`: a token is emitted once a following one proves it
    /// complete.
    words: TokenBuffer<Words>,
    stripper: TranscriptMarkupStripper,
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
    /// The front word's `(start, end)`, fixed at the speed known when it started, as the
    /// reference's pacing loop computes each word's delay as it reaches it.
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

impl Pacer {
    pub fn new(opts: PacerOptions) -> Pacer {
        let speed = STANDARD_SPEECH_RATE * opts.speed;
        Pacer {
            state: PaceState::Streaming,
            opts,
            speed,
            words: TokenBuffer::new(Words(SYNC_TOKENIZER), TokenBufferOptions::WORDS),
            stripper: TranscriptMarkupStripper::new(),
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

    pub fn text_ended(&self) -> bool {
        matches!(self.state, PaceState::TextEnded | PaceState::InputsEnded)
    }

    pub fn audio_ended(&self) -> bool {
        matches!(self.state, PaceState::AudioEnded | PaceState::InputsEnded)
    }

    /// `Heard` was produced, or the segment was closed: it takes nothing more but `Close`.
    pub fn is_over(&self) -> bool {
        matches!(self.state, PaceState::Finished | PaceState::Closed)
    }

    /// One event in, its effects out. The state × event table:
    ///
    /// | state \ event | Text | TextEnd | Audio | AudioEnd | Position | Finished | Split | Close |
    /// |---|---|---|---|---|---|---|---|---|
    /// | Streaming | plan | → TextEnded | count | → AudioEnded¹ | release | early³ | split | cut |
    /// | TextEnded | warn | warn | count | → InputsEnded¹ | release | early³ | split | cut |
    /// | AudioEnded | plan | → InputsEnded¹ | warn | warn | release | early³ | finish² | cut |
    /// | InputsEnded | warn | warn | warn | warn | release | finish | finish² | cut |
    /// | Finished | warn | warn | warn | warn | warn | warn | warn | → Closed |
    /// | Closed | warn | warn | warn | warn | warn | warn | warn | warn |
    ///
    /// ¹ re-estimates the speed. A segment whose inputs ended with no audio shows its text whole
    /// at once: nothing will play it. ² Interrupted: with its audio over, nothing plays on after
    /// the cut. ³ Finishes, warned too early.
    ///
    /// Positions are the sink's, in its current playout; the pacer moves them past the last
    /// split.
    #[must_use = "an effect not performed is a word the user never sees"]
    pub fn step(mut self, ev: PaceEvent) -> (Pacer, Vec<PaceEffect>) {
        use PaceEvent as E;
        use PaceState as S;
        let mut out = Vec::new();
        let warn = |out: &mut Vec<PaceEffect>, w| out.push(PaceEffect::Warn(w));
        match (self.state, ev) {
            (S::Closed, _) => warn(&mut out, PaceWarning::AfterClose),
            (S::Finished, E::Close) => self.state = S::Closed,
            (S::Finished, _) => warn(&mut out, PaceWarning::AfterFinished),
            (_, E::Position(p)) => self.release(self.origin + p, &mut out),
            (_, E::Paused) => self.paused(&mut out),
            (_, E::Close) => self.cut(&mut out),
            (state, E::PlaybackFinished { position, interrupted }) => {
                if state != S::InputsEnded {
                    warn(&mut out, PaceWarning::PlaybackFinishedTooEarly);
                }
                self.finish(self.origin + position, interrupted, &mut out);
            }
            (S::Streaming | S::TextEnded, E::PlaybackSplit { position }) => {
                self.split(self.origin + position, &mut out)
            }
            (_, E::PlaybackSplit { position }) => {
                self.finish(self.origin + position, true, &mut out)
            }
            (S::Streaming | S::AudioEnded, E::Text { raw, start_time, end_time }) => {
                self.plan(&raw, start_time, end_time)
            }
            (S::Streaming, E::TextEnd) => {
                self.end_text();
                self.state = S::TextEnded;
            }
            (S::AudioEnded, E::TextEnd) => {
                self.end_text();
                self.inputs_ended(&mut out);
            }
            (_, E::Text { .. } | E::TextEnd) => warn(&mut out, PaceWarning::TextAfterEnd),
            (S::Streaming | S::TextEnded, E::Audio { secs }) => self.count_audio(secs),
            (S::Streaming, E::AudioEnd) => self.end_audio(),
            (S::TextEnded, E::AudioEnd) => self.inputs_ended(&mut out),
            (_, E::Audio { .. } | E::AudioEnd) => warn(&mut out, PaceWarning::AudioAfterEnd),
        }
        (self, out)
    }

    fn plan(&mut self, raw: &str, start_time: Option<f64>, end_time: Option<f64>) {
        self.saw_text = true;
        if let Some(t) = start_time {
            // An aligned delta is a word boundary: the held word belongs to the previous onset.
            let held = self.words.flush();
            self.plan_tokens(held);
            self.base = Some(t);
            self.limit = end_time.filter(|end| *end >= t);
            self.offset = 0;
        }
        let tokens = self.words.push(raw);
        self.plan_tokens(tokens);
    }

    fn plan_tokens(&mut self, tokens: Vec<String>) {
        for word in tokens {
            let visible = self.stripper.push(&word);
            self.plan_word(word, visible);
        }
    }

    fn plan_word(&mut self, word: String, visible: String) {
        let hyphens = self.hyphens(&visible);
        self.queue.push_back(Planned {
            word,
            visible,
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

    /// The text ended: the held partial word is a whole word now, and so is a held tag.
    fn end_text(&mut self) {
        let tail = self.words.flush();
        self.plan_tokens(tail);
        let held = self.stripper.flush();
        if !held.is_empty() {
            // A fragment held as a possible tag that never closed: plain text after all.
            self.plan_word(String::new(), held);
        }
    }

    fn count_audio(&mut self, secs: f64) {
        if secs.is_finite() && secs > 0.0 {
            self.audio_secs += secs;
        }
        self.frames += 1;
    }

    fn end_audio(&mut self) {
        self.state = PaceState::AudioEnded;
    }

    /// Both inputs ended: the segment's own rate is known.
    fn inputs_ended(&mut self, out: &mut Vec<PaceEffect>) {
        if self.frames == 0 {
            // Nothing will play it: shown whole now, as a segment that finished playing.
            self.finish(Duration::ZERO, false, out);
            return;
        }
        self.state = PaceState::InputsEnded;
        // A markup- or emoji-only segment would otherwise set the speed to zero.
        if self.planned_hyphens > 0 && self.audio_secs > 0.0 {
            self.speed = self.planned_hyphens as f64 / self.audio_secs;
        }
    }

    /// The reference's pacing loop (`_SegmentSynchronizerImpl._main_task`): each word starts where
    /// the previous one ended (or when it arrived), lasts its hyphens at the speed known then, and
    /// is shown halfway through. An aligned word keeps the TTS's own timing.
    fn release(&mut self, position: Duration, out: &mut Vec<PaceEffect>) {
        self.position = position;
        let now = position.as_secs_f64();
        // The aligned delta's audio is over: its held word is whole, as any continuation would
        // be a later delta starting at a new onset.
        if self.limit.is_some_and(|end| now >= end) {
            let held = self.words.flush();
            self.plan_tokens(held);
        }
        while let Some(front) = self.queue.front() {
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
            // Only a word whose audio has started is in its delay (what a pause shows).
            if start > now + 1e-9 {
                break;
            }
            self.front = Some((start, end));
            let shown_at = if front.base.is_some() { start } else { (start + end) / 2.0 };
            if shown_at > now + 1e-9 {
                break;
            }
            let word = self.queue.pop_front().expect("peeked");
            self.front = None;
            self.cursor = Some(end);
            self.forward(word, start, end, out);
        }
    }

    /// The word whose delay had begun is still shown: the reference's pacing loop sleeps through
    /// a pause and waits only before the next word.
    fn paused(&mut self, out: &mut Vec<PaceEffect>) {
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
        out: &mut Vec<PaceEffect>,
    ) {
        // The words are one voice: the last one's audio was over when this one's began, whatever
        // pace it was shown at (a sentence's first word paced before its rate was known).
        if let Some((_, end)) = self.forwarded.last_mut() {
            *end = end.min(start_time);
        }
        self.forwarded.push((word.visible, end_time));
        if !word.word.is_empty() {
            out.push(PaceEffect::Forward { word: word.word, start_time, end_time });
        }
    }

    /// The words shown whose audio had ended by `position` (`forwarded_text`, which the reference
    /// extends only once a word's whole delay has passed).
    fn heard(&self, position: Duration) -> String {
        let at = position.as_secs_f64() + 1e-9;
        self.forwarded.iter().take_while(|(_, end)| *end <= at).map(|(w, _)| w.as_str()).collect()
    }

    /// The playout ended. Complete: everything left is heard, at once. Cut: what was forwarded by
    /// the playhead it reports. A sink that reports no progress (an avatar's data stream) paces
    /// nothing before this; the reference's loop paced its words on the wall clock meanwhile.
    fn finish(&mut self, position: Duration, interrupted: bool, out: &mut Vec<PaceEffect>) {
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
        out.push(PaceEffect::Flush);
        let heard = if interrupted {
            Heard::cut(self.heard(position), position, self.frames)
        } else {
            let all: String = self.forwarded.iter().map(|(w, _)| w.as_str()).collect();
            Heard::complete(all, position, self.frames)
        };
        out.push(PaceEffect::Heard(heard));
        self.state = PaceState::Finished;
    }

    /// The words heard by `position` are this playout's; the next playout starts there.
    fn split(&mut self, position: Duration, out: &mut Vec<PaceEffect>) {
        self.release(position, out);
        out.push(PaceEffect::Heard(Heard::cut(self.heard(position), position, self.frames)));
        self.origin = position;
    }

    /// Closed before the playout finished: the words shown are the words heard.
    fn cut(&mut self, out: &mut Vec<PaceEffect>) {
        self.queue.clear();
        out.push(PaceEffect::Flush);
        let heard = self.heard(self.position);
        out.push(PaceEffect::Heard(Heard::cut(heard, self.position, self.frames)));
        self.state = PaceState::Closed;
    }
}

/// `_TextSyncOptions`, minus the word tokenizer (pinned -- see [`SYNC_TOKENIZER`]).
#[derive(Clone)]
pub struct PacerOptions {
    /// The caller's `transcription_speed_factor`. Multiplies [`STANDARD_SPEECH_RATE`].
    pub speed: f64,
    pub hyphenator: Arc<dyn Hyphenator>,
}

impl std::fmt::Debug for PacerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PacerOptions").field("speed", &self.speed).finish_non_exhaustive()
    }
}

impl Default for PacerOptions {
    fn default() -> Self {
        PacerOptions { speed: 1.0, hyphenator: Arc::new(Liang) }
    }
}

#[cfg(test)]
#[path = "pacer_tests.rs"]
mod tests;
