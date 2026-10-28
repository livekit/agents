use crate::transcription::Heard;
use async_trait::async_trait;
use lk_agents_abi::tokenize::{
    TokenBuffer, TokenBufferOptions, WordTokenizerOptions, Words, split_words,
};
use lk_agents_abi::tts::markup::TranscriptMarkupStripper;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackFinished, PlaybackObserver,
    PlaybackTracker, Result, TextOutput,
};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, Clock, Monotone, Spawner, SpawnerExt, TimedString};
use lk_agents_voice::io::AudioSinkProxy;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::{HashMap, VecDeque};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;
use tokio::sync::mpsc;

/// The synchroniser's tokenizer settings, from `TranscriptSynchronizer.__init__`: tokens must
/// reassemble the input, punctuation is paced, and CJK/Thai split per character.
pub const SYNC_TOKENIZER: WordTokenizerOptions =
    WordTokenizerOptions { ignore_punctuation: false, split_character: true, retain_format: true };

/// Hyphens (syllables) per second at `speed = 1.0`, before a segment's own rate is known.
pub const STANDARD_SPEECH_RATE: f64 = 3.83;

const PATTERNS: &str = include_str!("liang/patterns.txt");

const EXCEPTIONS: &str = include_str!("liang/exceptions.txt");

/// How long `aclose` waits for the forwarder to hand the last words to the text sink.
pub const SEGMENT_ACLOSE_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// Exactly once per segment.
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
    position: Duration,
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
            saw_text: false,
        }
    }

    /// The last playhead the segment was told.
    pub fn position(&self) -> Duration {
        self.position
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
    /// | state \ event | Text | TextEnd | Audio | AudioEnd | Position | PlaybackFinished | Close |
    /// |---|---|---|---|---|---|---|---|
    /// | Streaming | plan | → TextEnded | count | → AudioEnded¹ | release | finish (too early) | cut |
    /// | TextEnded | warn | warn | count | → InputsEnded¹ | release | finish (too early) | cut |
    /// | AudioEnded | plan | → InputsEnded¹ | warn | warn | release | finish (too early) | cut |
    /// | InputsEnded | warn | warn | warn | warn | release | finish | cut |
    /// | Finished | warn | warn | warn | warn | warn | warn | → Closed |
    /// | Closed | warn | warn | warn | warn | warn | warn | warn |
    ///
    /// ¹ re-estimates the speed. A segment whose inputs ended with no audio shows its text whole
    /// at once: nothing will play it.
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
            (_, E::Position(p)) => self.release(p, &mut out),
            (_, E::Paused) => self.paused(&mut out),
            (_, E::Close) => self.cut(&mut out),
            (state, E::PlaybackFinished { position, interrupted }) => {
                if state != S::InputsEnded {
                    warn(&mut out, PaceWarning::PlaybackFinishedTooEarly);
                }
                self.finish(position, interrupted, &mut out);
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

    /// Closed before the playout finished: the words shown are the words heard.
    fn cut(&mut self, out: &mut Vec<PaceEffect>) {
        self.queue.clear();
        out.push(PaceEffect::Flush);
        let heard = self.heard(self.position);
        out.push(PaceEffect::Heard(Heard::cut(heard, self.position, self.frames)));
        self.state = PaceState::Closed;
    }
}

/// Counts syllable-like pieces used to pace transcript words. The default is [`Liang`]; a caller's
/// own `hyphenate_word=` is this trait through `synchronizer.py`'s `_Hyphenator` (a GIL hop per
/// word).
#[lk_agents_abi::host(proxy)]
pub trait Hyphenator: Send + Sync {
    /// `len(hyphenate_word(word))`. Never zero, including for the empty string.
    fn hyphens(&self, word: &str) -> usize;
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

#[derive(Default)]
struct Node {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    children: HashMap<char, Node>,
    /// The pattern's points when a pattern ends at this node (`t[None]`).
    points: Option<Vec<u8>>,
}

struct Table {
    tree: Node,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    exceptions: HashMap<String, Vec<u8>>,
}

impl Table {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn build() -> Table {
        let mut tree = Node::default();
        for pattern in PATTERNS.split_whitespace() {
            // One point before every character and one after the last, as the reference parses.
            let mut points = Vec::new();
            let mut digits = String::new();
            let mut node = &mut tree;
            for c in pattern.chars() {
                if c.is_ascii_digit() {
                    digits.push(c);
                } else {
                    points.push(digits.parse::<u8>().unwrap_or(0));
                    digits.clear();
                    node = node.children.entry(c).or_default();
                }
            }
            points.push(digits.parse::<u8>().unwrap_or(0));
            node.points = Some(points);
        }
        let mut exceptions = HashMap::new();
        for ex in EXCEPTIONS.split_whitespace() {
            // `[0] + [int(h == "-") for h in re.split(r"[a-z]", ex)]`.
            let mut points = vec![0u8];
            let mut between = String::new();
            for c in ex.chars() {
                if c.is_ascii_lowercase() {
                    points.push(u8::from(between == "-"));
                    between.clear();
                } else {
                    between.push(c);
                }
            }
            points.push(u8::from(between == "-"));
            exceptions.insert(ex.replace('-', ""), points);
        }
        Table { tree, exceptions }
    }
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(Table::build)
}

/// `hyphenate_word(word)`: the pieces, broken at the possible hyphenation points.
#[lk_agents_abi::host(export(pid = false))]
#[python(forward(module = "tokenize_basic"))]
pub fn hyphenate_word(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    // "Short words aren't hyphenated." Characters, not bytes.
    if chars.len() <= 4 {
        return vec![word.to_owned()];
    }
    let table = table();
    let lower = word.to_lowercase();
    let points: Vec<u8> = match table.exceptions.get(&lower) {
        Some(p) => p.clone(),
        None => {
            let work: Vec<char> =
                std::iter::once('.').chain(lower.chars()).chain(std::iter::once('.')).collect();
            let mut points = vec![0u8; work.len() + 1];
            for i in 0..work.len() {
                let mut node = &table.tree;
                for &c in &work[i..] {
                    let Some(next) = node.children.get(&c) else { break };
                    node = next;
                    if let Some(p) = &node.points {
                        for (j, pj) in p.iter().enumerate() {
                            if let Some(slot) = points.get_mut(i + j) {
                                *slot = (*slot).max(*pj);
                            }
                        }
                    }
                }
            }
            // "No hyphens in the first two chars or the last two."
            let n = points.len();
            points[1] = 0;
            points[2] = 0;
            points[n - 2] = 0;
            points[n - 3] = 0;
            points
        }
    };
    // `for c, p in zip(word, points[2:])`: the original characters, the folded word's points.
    let mut pieces = vec![String::new()];
    for (c, p) in chars.iter().zip(points.iter().skip(2)) {
        pieces.last_mut().expect("one piece at least").push(*c);
        if p % 2 == 1 {
            pieces.push(String::new());
        }
    }
    pieces
}

/// The pacer's hyphenator: Liang's syllable count, `len(hyphenate_word(word))`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Liang;

impl Hyphenator for Liang {
    fn hyphens(&self, word: &str) -> usize {
        hyphenate_word(word).len().max(1)
    }
}

/// What the two legs and the sink below tell the synchroniser.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncInput {
    Text {
        raw: String,
        start_time: Option<f64>,
        end_time: Option<f64>,
    },
    /// The text leg's flush: the segment's text is complete.
    TextEnd,
    /// A frame for the sink below.
    Audio {
        secs: f64,
    },
    /// The audio leg's flush: the segment's audio is complete, possibly with none.
    AudioEnd,
    /// `clear_buffer`: the newest segment, if none of its audio reached the sink, is cut.
    Clear,
    Started,
    Progressed(Duration),
    Paused,
    Finished {
        position: Duration,
        interrupted: bool,
    },
    /// The sink threw its playback count away: the segments it owed are gone where they stood.
    Reset,
    Enabled(bool),
    /// End every segment where it stands; the next input starts a fresh one.
    Rotate,
}

/// One segment as the synchroniser sees it.
struct Segment {
    /// `None` once closed while its audio is still the sink's (a rotation, synchronisation
    /// switched off), or opened while synchronisation was off: its playback moves no pacer.
    pacer: Option<Pacer>,
    text_ended: bool,
    /// A frame reached the sink: the sink owes this segment's playback finish.
    audio: bool,
    audio_ended: bool,
    /// The sink reported this segment's playback started.
    started: bool,
}

/// `_SegmentSynchronizerImpl`s in order. Each leg is an ordered stream of segments delimited by
/// its flushes: a segment's text is the next text not yet ended, its audio the next audio, so a
/// segment the producer ended with no audio is one by construction even when its words arrive
/// later (through a `transcription_node`). The sink's playback moves the oldest segment it owes.
pub struct SyncSegments {
    opts: PacerOptions,
    enabled: bool,
    segments: VecDeque<Segment>,
}

impl SyncSegments {
    pub fn new(opts: PacerOptions) -> SyncSegments {
        SyncSegments { opts, enabled: true, segments: VecDeque::new() }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// One input in, the words to show out. A `Heard` is kept only for a `Finished` segment that
    /// had text: its synchronised transcript.
    pub fn step(&mut self, input: SyncInput) -> Vec<PaceEffect> {
        let mut out = Vec::new();
        match input {
            SyncInput::Text { raw, start_time, end_time } => {
                let i = self.next(|s| !s.text_ended);
                if self.segments[i].pacer.is_none() {
                    // A segment closed mid-write: its words have no pace to keep.
                    out.push(PaceEffect::Forward { word: raw, start_time: 0.0, end_time: 0.0 });
                } else {
                    self.apply(i, PaceEvent::Text { raw, start_time, end_time }, &mut out);
                }
            }
            SyncInput::TextEnd => {
                if let Some(i) = self.segments.iter().position(|s| !s.text_ended) {
                    self.segments[i].text_ended = true;
                    self.apply(i, PaceEvent::TextEnd, &mut out);
                    self.settle(i);
                }
            }
            SyncInput::Audio { secs } => {
                let i = self.next(|s| !s.audio_ended);
                self.segments[i].audio = true;
                self.apply(i, PaceEvent::Audio { secs }, &mut out);
            }
            SyncInput::AudioEnd => {
                // Ended before any of its words arrived: a segment with no audio, unless its
                // words will pass unpaced.
                if !self.enabled && self.segments.iter().all(|s| s.audio_ended) {
                    return out;
                }
                let i = self.next(|s| !s.audio_ended);
                self.segments[i].audio_ended = true;
                self.apply(i, PaceEvent::AudioEnd, &mut out);
                self.settle(i);
            }
            SyncInput::Clear => {
                if self.segments.back().is_some_and(|s| !s.audio) {
                    let i = self.segments.len() - 1;
                    self.apply(i, PaceEvent::Close, &mut out);
                    self.segments.pop_back();
                }
            }
            SyncInput::Started => {
                // A sink plays its segments in order: one starting while an older one it had
                // started is still owed means that one's finish was lost. It ends where it stood.
                if let Some(i) = self.playing()
                    && self.segments[i].started
                    && self.segments.iter().skip(i + 1).any(|s| s.audio)
                {
                    self.apply(i, PaceEvent::Close, &mut out);
                    self.segments.remove(i);
                }
                // The reference's pacing loop takes its first word as the playback starts.
                if let Some(i) = self.playing() {
                    self.segments[i].started = true;
                    if let Some(at) = self.segments[i].pacer.as_ref().map(Pacer::position) {
                        self.apply(i, PaceEvent::Position(at), &mut out);
                    }
                }
            }
            SyncInput::Progressed(at) => {
                if let Some(i) = self.playing() {
                    self.apply(i, PaceEvent::Position(at), &mut out);
                }
            }
            SyncInput::Paused => {
                if let Some(i) = self.playing() {
                    self.apply(i, PaceEvent::Paused, &mut out);
                }
            }
            SyncInput::Finished { position, interrupted } => {
                let Some(i) = self.playing() else { return out };
                let segment = self.segments.remove(i).expect("playing is an index");
                if let Some(pacer) = segment.pacer.filter(|p| !p.is_over() && !p.is_blank()) {
                    // Audio with no words: no transcript, so the speech's own text stands.
                    let had_text = pacer.saw_text();
                    let (_, effects) =
                        pacer.step(PaceEvent::PlaybackFinished { position, interrupted });
                    let heard = |e: &PaceEffect| matches!(e, PaceEffect::Heard(_));
                    out.extend(effects.into_iter().filter(|e| had_text || !heard(e)));
                    return out;
                }
            }
            SyncInput::Reset => {
                for i in 0..self.segments.len() {
                    if self.segments[i].audio {
                        self.apply(i, PaceEvent::Close, &mut out);
                    }
                }
                self.segments.retain(|s| !s.audio);
            }
            SyncInput::Enabled(enabled) => {
                if self.enabled != enabled {
                    self.enabled = enabled;
                    self.rotate(&mut out);
                }
            }
            SyncInput::Rotate => self.rotate(&mut out),
        }
        out.retain(|e| !matches!(e, PaceEffect::Heard(_)));
        out
    }

    /// The oldest segment `open` holds for, else a new one.
    fn next(&mut self, open: impl Fn(&Segment) -> bool) -> usize {
        if let Some(i) = self.segments.iter().position(open) {
            return i;
        }
        let pacer = self.enabled.then(|| Pacer::new(self.opts.clone()));
        let segment =
            Segment { pacer, text_ended: false, audio: false, audio_ended: false, started: false };
        self.segments.push_back(segment);
        self.segments.len() - 1
    }

    /// A segment with no audio is gone once its pacer and its text are: nothing will report its
    /// playout.
    fn settle(&mut self, i: usize) {
        let segment = &self.segments[i];
        if !segment.audio && segment.text_ended && segment.pacer.as_ref().is_none_or(Pacer::is_over)
        {
            self.segments.remove(i);
        }
    }

    /// The oldest segment the sink owes a playout.
    fn playing(&self) -> Option<usize> {
        self.segments.iter().position(|s| s.audio)
    }

    /// Closing ends the text output's segment even with none open, as closing the reference's
    /// idle `_SegmentSynchronizerImpl` does.
    fn rotate(&mut self, out: &mut Vec<PaceEffect>) {
        if self.segments.is_empty() {
            out.push(PaceEffect::Flush);
        }
        for i in 0..self.segments.len() {
            self.apply(i, PaceEvent::Close, out);
            let segment = &mut self.segments[i];
            segment.pacer = None;
            segment.text_ended = true;
        }
        self.segments.retain(|s| s.audio);
    }

    fn apply(&mut self, i: usize, ev: PaceEvent, out: &mut Vec<PaceEffect>) {
        let slot = &mut self.segments[i].pacer;
        // An ended segment takes nothing more; its slot still accounts for its playout.
        let Some(pacer) = slot.take_if(|p| !p.is_over()) else { return };
        let (pacer, effects) = pacer.step(ev);
        *slot = Some(pacer);
        out.extend(effects);
    }
}

enum Out {
    Word(TimedString),
    Flush,
}

/// The segments and the gate around them, under one lock.
struct SyncState {
    segments: SyncSegments,
    audio_attached: bool,
    text_attached: bool,
    closed: bool,
    /// `_SyncedTextOutput._capturing`: whether a flush has anything to end. With nothing
    /// captured, ending the text would pace later words against the next segment's audio.
    capturing_text: bool,
}

/// What both chain nodes and the facade share; nothing holds the facade, so no reference cycle.
struct SyncCore {
    clock: Arc<dyn Clock>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    state: Mutex<SyncState>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `None` once closed: dropping the sender is what ends the forwarder.
    tx: Mutex<Option<mpsc::UnboundedSender<Out>>>,
    /// Shared with the forwarder task, which must not keep the core alive (or it never ends).
    forwarder_done: Arc<Monotone<bool>>,
    /// The audio node's tracker; released words are raised on it as `transcript_shown`.
    tracker: Arc<PlaybackTracker>,
}

impl SyncCore {
    fn enabled(&self) -> bool {
        self.state.lock().unwrap().segments.enabled()
    }

    /// Step the segments and perform what they say; the heard transcript of a finished segment.
    fn step(&self, input: SyncInput) -> Option<Heard> {
        let effects = self.state.lock().unwrap().segments.step(input);
        let mut heard = None;
        for effect in effects {
            match effect {
                PaceEffect::Forward { word, start_time, end_time } => {
                    let shown = TimedString {
                        start_time: Some(start_time),
                        end_time: Some(end_time),
                        ..TimedString::new(word)
                    };
                    self.tracker.transcript_shown(&shown);
                    // The text output gets where the word's audio ends, as the reference's does.
                    self.send(Out::Word(TimedString { start_time: None, ..shown }));
                }
                PaceEffect::Flush => self.send(Out::Flush),
                PaceEffect::Heard(h) => heard = Some(h),
                PaceEffect::Warn(w) => log::warn!(target: "livekit.agents", "{}", w.message()),
            }
        }
        heard
    }

    fn send(&self, out: Out) {
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send(out);
        }
    }

    /// `_on_attachment_changed`. `None` leaves a leg's attachment where it is.
    fn on_attachment_changed(&self, audio: Option<bool>, text: Option<bool>) {
        let enabled = {
            let mut state = self.state.lock().unwrap();
            state.audio_attached = audio.unwrap_or(state.audio_attached);
            state.text_attached = text.unwrap_or(state.text_attached);
            state.audio_attached && state.text_attached
        };
        self.step(SyncInput::Enabled(enabled));
    }

    /// The forwarder: the only task and the only place the text sink is awaited.
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    async fn forward(next: Option<Arc<dyn TextOutput>>, mut rx: mpsc::UnboundedReceiver<Out>) {
        while let Some(out) = rx.recv().await {
            let Some(next) = &next else { continue };
            match out {
                // A failure is the sink's. Dropping it keeps the rest of the transcript flowing.
                Out::Word(t) => {
                    let _ = next.capture_timed(&t).await;
                }
                Out::Flush => next.flush(),
            }
        }
    }
}

/// `_SyncedAudioOutput`.
pub struct SyncedAudioOutput {
    core: Arc<SyncCore>,
    tracker: Arc<PlaybackTracker>,
    next: Arc<dyn AudioOutput>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    observer: Mutex<Option<Arc<dyn PlaybackObserver>>>,
}

/// [`Weak`] so the tracker doesn't keep the chain alive.
struct AudioEvents(Weak<SyncedAudioOutput>);

impl PlaybackObserver for AudioEvents {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        if let Some(node) = self.0.upgrade() {
            node.handle_playback(ev);
        }
    }

    fn on_playback_reset(&self) {
        if let Some(node) = self.0.upgrade() {
            node.core.step(SyncInput::Reset);
            node.tracker.on_playback_reset();
        }
    }
}

impl SyncedAudioOutput {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn new(core: Arc<SyncCore>, next: Arc<dyn AudioOutput>) -> Arc<Self> {
        let node = Arc::new(SyncedAudioOutput {
            tracker: Arc::clone(&core.tracker),
            core,
            next: next.clone(),
            observer: Mutex::new(None),
        });
        let observer: Arc<dyn PlaybackObserver> = Arc::new(AudioEvents(Arc::downgrade(&node)));
        next.tracker().observe(observer.clone());
        *node.observer.lock().unwrap() = Some(observer);
        node
    }

    /// Synchronous: a wrapper's counter must move before the leaf's emit returns, or
    /// `wait_for_playout` races the event meant to release it.
    fn handle_playback(&self, ev: &PlaybackEvent) {
        match ev {
            PlaybackEvent::Started(e) => {
                self.tracker.playback_started(e.created_at);
                self.core.step(SyncInput::Started);
            }
            PlaybackEvent::Progressed(e) => {
                self.tracker.playback_progressed(e.started_at, e.offset, e.duration);
                self.core.step(SyncInput::Progressed(e.offset + e.duration));
            }
            PlaybackEvent::Finished(e) => {
                let position = e.playback_position;
                let heard =
                    self.core.step(SyncInput::Finished { position, interrupted: e.interrupted });
                let synchronized_transcript =
                    heard.map(|h| h.text).or(e.synchronized_transcript.clone());
                self.tracker
                    .playback_finished(PlaybackFinished { synchronized_transcript, ..e.clone() });
            }
        }
    }
}

#[async_trait]
impl AudioOutput for SyncedAudioOutput {
    fn label(&self) -> &str {
        "TranscriptSynchronizer"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.tracker
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }

    fn sample_rate(&self) -> Option<u32> {
        self.next.sample_rate()
    }

    fn next_in_chain(&self) -> Option<Arc<dyn AudioOutput>> {
        Some(self.next.clone())
    }

    async fn capture(&self, frame: &AudioFrame) -> Result<()> {
        // Counted before it passes: the sink below may report its playback from inside the
        // capture, and that playback is this segment's.
        let secs = lk_agents_rtc::frame_duration(frame).as_secs_f64();
        self.core.step(SyncInput::Audio { secs });
        self.next.capture_frame(frame).await
    }

    fn flush_segment(&self) {
        self.next.flush();
        self.core.step(SyncInput::AudioEnd);
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
        self.core.step(SyncInput::Clear);
    }

    fn on_attached(&self) {
        self.next.on_attached();
        self.core.on_attachment_changed(Some(true), None);
    }

    fn on_detached(&self) {
        self.next.on_detached();
        self.core.on_attachment_changed(Some(false), None);
    }

    fn pause(&self) {
        self.next.pause();
        self.core.step(SyncInput::Paused);
    }

    fn resume(&self) {
        self.next.resume();
    }
}

/// `_SyncedTextOutput`.
pub struct SyncedTextOutput {
    core: Arc<SyncCore>,
    next: Option<Arc<dyn TextOutput>>,
}

impl SyncedTextOutput {
    /// `aligned`: whether the caller used the timed API (see `Pacer::push_text`).
    async fn capture(&self, text: &TimedString, aligned: bool) -> Result<()> {
        if !self.core.enabled() {
            if let Some(next) = &self.next {
                if aligned {
                    next.capture_timed(text).await?;
                } else {
                    next.capture_text(&text.text).await?;
                }
            }
            return Ok(());
        }
        self.core.state.lock().unwrap().capturing_text = true;
        let (start_time, end_time) =
            if aligned { (text.start_time, text.end_time) } else { (None, None) };
        let raw = text.text.clone();
        self.core.step(SyncInput::Text { raw, start_time, end_time });
        Ok(())
    }
}

#[async_trait]
impl TextOutput for SyncedTextOutput {
    fn label(&self) -> &str {
        "TranscriptSynchronizer"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        self.next.clone()
    }

    async fn capture_timed(&self, text: &TimedString) -> Result<()> {
        self.capture(text, true).await
    }

    async fn capture_text(&self, text: &str) -> Result<()> {
        self.capture(&TimedString::new(text), false).await
    }

    fn flush(&self) {
        if !self.core.enabled() {
            if let Some(next) = &self.next {
                next.flush();
            }
            return;
        }
        if std::mem::take(&mut self.core.state.lock().unwrap().capturing_text) {
            self.core.step(SyncInput::TextEnd);
        }
    }

    fn on_attached(&self) {
        if let Some(next) = &self.next {
            next.on_attached();
        }
        self.core.on_attachment_changed(None, Some(true));
    }

    fn on_detached(&self) {
        if let Some(next) = &self.next {
            next.on_detached();
        }
        self.core.on_attachment_changed(None, Some(false));
    }
}

/// Everything `TranscriptSynchronizer.__init__` takes, plus the two ports Rust needs explicitly.
pub struct SyncConfig {
    /// The audio sink below; a bare leaf is wrapped in an [`AudioSinkProxy`].
    pub audio_next: Arc<dyn AudioOutput>,
    /// The transcript sink below this node. `None` for the console path, which prints locally.
    pub text_next: Option<Arc<dyn TextOutput>>,
    pub speed: f64,
    pub hyphenator: Arc<dyn crate::transcription::synchronizer::Hyphenator>,
    pub clock: Arc<dyn Clock>,
    pub spawner: Arc<dyn Spawner>,
}

pub struct TranscriptSynchronizer {
    core: Arc<SyncCore>,
    audio: Arc<SyncedAudioOutput>,
    text: Arc<SyncedTextOutput>,
    /// The clock's wall stamp, for a host listening to the audio node's playback.
    anchor: lk_agents_abi::WallAnchor,
}

/// `_core.TranscriptSynchronizer`: the node every native session builds, handed to a host that
/// assembles a synchroniser by hand (the console, a custom chain), so pacing is one implementation
/// on one clock.
#[lk_agents_abi::host(object(cell = "arc"))]
impl TranscriptSynchronizer {
    pub fn new(config: SyncConfig) -> Arc<Self> {
        Arc::new(Self::build(config))
    }

    /// `next_in_chain_audio` is the host `io.AudioOutput` the audio passes through to (its
    /// playback events pace the text); `next_in_chain_text` the host `io.TextOutput` the paced
    /// words go to, or `None`. `hyphenate_word` is the caller's, or `None` for Liang.
    #[host(new(kw_only = "speed", defaults(speed = 1.0)))]
    pub fn for_host(
        next_in_chain_audio: Arc<dyn AudioOutput>,
        next_in_chain_text: Option<Arc<dyn TextOutput>>,
        speed: f64,
        hyphenate_word: Option<Arc<dyn Hyphenator>>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let tracker = Arc::clone(next_in_chain_audio.tracker());
        let sync = Self::build(SyncConfig {
            audio_next: next_in_chain_audio,
            text_next: next_in_chain_text,
            speed,
            hyphenator: hyphenate_word.unwrap_or_else(|| Arc::new(Liang)),
            clock: Arc::new(lk_agents_abi::RealClock::new()),
            spawner,
        });
        // The host sink's playback, stamped on the clock the pacer runs on.
        tracker.anchor_to(sync.anchor);
        sync
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn build(config: SyncConfig) -> Self {
        let SyncConfig { audio_next, text_next, speed, hyphenator, clock, spawner } = config;
        let anchor = lk_agents_abi::WallAnchor::of(&*clock);

        // `io.AudioOutput.__init__`'s auto-wrap, so an avatar hand-off can replace the leaf later.
        let audio_next = if audio_next.next_in_chain().is_none() {
            AudioSinkProxy::new(audio_next) as Arc<dyn AudioOutput>
        } else {
            audio_next
        };

        let opts = PacerOptions { speed, hyphenator };
        let (tx, rx) = mpsc::unbounded_channel();
        let core = Arc::new(SyncCore {
            state: Mutex::new(SyncState {
                segments: SyncSegments::new(opts),
                audio_attached: true,
                text_attached: true,
                closed: false,
                capturing_text: false,
            }),
            clock,
            tx: Mutex::new(Some(tx)),
            forwarder_done: Arc::new(Monotone::new(false)),
            tracker: Arc::new(PlaybackTracker::new()),
        });
        // The task holds only the done flag, so it never keeps the core alive.
        let forwarder_done = Arc::clone(&core.forwarder_done);
        let sink = text_next.clone();
        spawner.spawn_unboxed(async move {
            SyncCore::forward(sink, rx).await;
            forwarder_done.set(true);
        });

        let audio = SyncedAudioOutput::new(core.clone(), audio_next);
        let text = Arc::new(SyncedTextOutput { core: core.clone(), next: text_next });
        TranscriptSynchronizer { core, audio, text, anchor }
    }

    /// The node to install as `session.output.transcription`, as a host object.
    #[host(getter(name = "text_output"))]
    pub fn text_leg(&self) -> crate::io::CoreTextOutput {
        crate::io::CoreTextOutput::new(self.text_output())
    }

    /// The node to install as `session.output.audio`.
    pub fn audio_output(&self) -> Arc<dyn AudioOutput> {
        self.audio.clone()
    }

    /// The node to install as `session.output.transcription`.
    pub fn text_output(&self) -> Arc<dyn TextOutput> {
        self.text.clone()
    }

    #[host(getter)]
    pub fn enabled(&self) -> bool {
        self.core.enabled()
    }

    #[host(method)]
    pub fn set_enabled(&self, enabled: bool) {
        self.core.step(SyncInput::Enabled(enabled));
    }

    /// End the live segment now and start a fresh one.
    #[host(method)]
    pub fn rotate_segment(&self) {
        if !self.core.state.lock().unwrap().closed {
            self.core.step(SyncInput::Rotate);
        }
    }

    /// Close: end the live segment and give the forwarder [`SEGMENT_ACLOSE_TIMEOUT`] to drain.
    /// `RoomIO.aclose` must close this before the transcription outputs it feeds.
    #[host(method)]
    pub async fn aclose(&self) {
        if std::mem::replace(&mut self.core.state.lock().unwrap().closed, true) {
            return;
        }
        self.core.step(SyncInput::Rotate);
        // Dropping the sender is what ends the forwarder's `recv`, after the words already queued.
        self.core.tx.lock().unwrap().take();
        let _ = lk_agents_abi::timeout(
            &*self.core.clock,
            SEGMENT_ACLOSE_TIMEOUT,
            self.core.forwarder_done.wait_for(|set| *set),
        )
        .await;
    }
}

#[cfg(test)]
#[path = "synchronizer_tests.rs"]
mod tests;
