use crate::transcription::effects::TranscriptPacerEffect;
use crate::transcription::machine::pacer::{
    TranscriptPacer, TranscriptPacerEvent, TranscriptPacerOptions,
};
use lk_agents_abi::Ts;
use lk_agents_abi::voice::io::PlaybackSegment;
use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;
use std::time::Duration;

const PATTERNS: &str = include_str!("../liang/patterns.txt");

const EXCEPTIONS: &str = include_str!("../liang/exceptions.txt");

/// Words of at most this many characters are never hyphenated.
const SHORT_WORD_CHARS: usize = 4;

/// The time stamped on a word forwarded with no pace to keep.
const UNPACED_TIME: f64 = 0.0;

/// A tick lands this much after the playhead reaches a word, so rounding never leaves it short.
const TICK_MARGIN: Duration = Duration::from_micros(1);

/// Counts syllable-like pieces used to pace transcript words: [`Liang`] by default.
pub trait Hyphenator: Send + Sync {
    /// The number of hyphenation pieces. Never zero, including for the empty string.
    fn hyphens(&self, word: &str) -> usize;
}

#[derive(Default)]
struct Node {
    children: BTreeMap<char, Node>,
    /// The pattern's points when a pattern ends at this node.
    points: Option<Vec<u8>>,
}

struct Table {
    tree: Node,
    exceptions: BTreeMap<String, Vec<u8>>,
}

impl Table {
    fn build() -> Table {
        let mut tree = Node::default();
        for pattern in PATTERNS.split_whitespace() {
            // One point before every character and one after the last.
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
        let mut exceptions = BTreeMap::new();
        for ex in EXCEPTIONS.split_whitespace() {
            // A point before the first letter, then 1 after each letter followed by a hyphen.
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

impl Node {
    /// Every pattern on the path from this node along `word[start..]` raises the points it
    /// covers, from `start`, to its own where higher.
    fn raise_points(&self, word: &[char], start: usize, points: &mut [u8]) {
        let mut node = self;
        for c in &word[start..] {
            let Some(next) = node.children.get(c) else { break };
            node = next;
            let Some(pattern) = &node.points else { continue };
            for (slot, p) in points[start..].iter_mut().zip(pattern) {
                *slot = (*slot).max(*p);
            }
        }
    }
}

/// Liang's points for a word with no exception: one before each character of `.word.`, and one
/// after the last.
fn hyphenation_points(table: &Table, lower: &str) -> Vec<u8> {
    let work: Vec<char> =
        std::iter::once('.').chain(lower.chars()).chain(std::iter::once('.')).collect();
    let mut points = vec![0u8; work.len() + 1];
    for start in 0..work.len() {
        table.tree.raise_points(&work, start, &mut points);
    }
    // No hyphens in the first two chars or the last two. Point `i` comes before `work[i]`, and
    // `work` has a `.` at each end.
    let n = points.len();
    for edge in [1, 2, n - 2, n - 3] {
        points[edge] = 0;
    }
    points
}

/// The pieces of `word`, broken at the possible hyphenation points.
fn hyphenate_word(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    // Short words are not hyphenated. Characters, not bytes.
    if chars.len() <= SHORT_WORD_CHARS {
        return vec![word.to_owned()];
    }
    let table = table();
    let lower = word.to_lowercase();
    let points = match table.exceptions.get(&lower) {
        Some(p) => p.clone(),
        None => hyphenation_points(table, &lower),
    };
    // The original characters with the lowercased word's points.
    let mut pieces = vec![String::new()];
    for (c, p) in chars.iter().zip(points.iter().skip(2)) {
        pieces.last_mut().expect("one piece at least").push(*c);
        if p % 2 == 1 {
            pieces.push(String::new());
        }
    }
    pieces
}

/// The pacer's hyphenator: Liang's syllable count.
#[derive(Debug, Clone, Copy, Default)]
pub struct Liang;

impl Hyphenator for Liang {
    fn hyphens(&self, word: &str) -> usize {
        hyphenate_word(word).len().max(1)
    }
}

/// What the two legs and the sink below tell the synchroniser.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::transcription) enum TranscriptSyncInput {
    Text {
        chunk: lk_agents_abi::TranscriptChunk,
        start_time: Option<f64>,
        end_time: Option<f64>,
    },
    /// The text leg's flush: the segment's text is complete.
    TextEnd,
    /// A frame for the sink below, and the playback it opened or continues.
    Audio {
        secs: f64,
        segment: Option<PlaybackSegment>,
    },
    /// The audio leg's flush: the segment's audio is complete, possibly with none.
    AudioEnd,
    /// The sink threw its audio away at `at`. The newest segment, if none of its audio reached
    /// the sink, is cut; the others it owed stop where their playout stood.
    Clear {
        at: Ts,
    },
    Started {
        at: Ts,
    },
    /// The sink's playhead in its current playout, reported at `at`.
    Progressed {
        position: Duration,
        at: Ts,
    },
    Paused {
        at: Ts,
    },
    Resumed {
        at: Ts,
    },
    /// The instant a [`TranscriptSyncSegments::wake`] asked for, or later.
    Tick {
        at: Ts,
    },
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
    pacer: Option<TranscriptPacer>,
    text_ended: bool,
    /// A frame reached the sink: the sink owes this segment's playback finish.
    audio: bool,
    /// The playback its latest frame went to, as the sink's tracker keyed it.
    key: Option<PlaybackSegment>,
    audio_ended: bool,
    /// The sink reported this segment's playback started.
    started: bool,
    /// Where its playout was last known to be, from its start or the sink's last report.
    playhead: Option<Playhead>,
    /// The sink cut this segment's playout while its audio was still coming. It owes no finish
    /// until a reset starts the rest of its audio as a playout of its own.
    cut_below: bool,
    /// A clear closed its words before any of its audio and before its audio leg ended: the
    /// leg's late end is its, not a new segment's. Gone at the next words or audio: a cut leg
    /// usually ends with no audio end at all, and the next speech's end is its own.
    cleared: bool,
}

/// A playout's position when it was last known: it moves on with elapsed time from there.
#[derive(Debug, Clone, Copy)]
struct Playhead {
    position: Duration,
    at: Ts,
}

/// Segment pacers in order. Each leg is an ordered stream of segments delimited by
/// its flushes: a segment's text is the next text not yet ended, its audio the next audio, so a
/// segment the producer ended with no audio is one by construction even when its words arrive
/// later (through a `transcription_node`). The sink's playback moves the oldest segment it owes.
pub(in crate::transcription) struct TranscriptSyncSegments {
    opts: TranscriptPacerOptions,
    enabled: bool,
    segments: VecDeque<Segment>,
    /// The sink was paused: no playhead moves with elapsed time until it resumes.
    paused: bool,
    /// The earliest tick asked for and not yet fed back.
    scheduled: Option<Ts>,
}

impl TranscriptSyncSegments {
    pub fn new(opts: TranscriptPacerOptions) -> TranscriptSyncSegments {
        TranscriptSyncSegments {
            opts,
            enabled: false,
            segments: VecDeque::new(),
            paused: false,
            scheduled: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// One input in, the words to show out. A `Heard` is kept only for a `Finished` segment that
    /// had text: its synchronised transcript.
    pub fn step(&mut self, input: TranscriptSyncInput) -> Vec<TranscriptPacerEffect> {
        let mut out = Vec::new();
        match input {
            TranscriptSyncInput::Text { chunk, start_time, end_time } => {
                self.on_text(chunk, start_time, end_time, &mut out)
            }
            TranscriptSyncInput::TextEnd => self.on_text_end(&mut out),
            TranscriptSyncInput::Audio { secs, segment } => self.on_audio(secs, segment, &mut out),
            TranscriptSyncInput::AudioEnd => self.on_audio_end(&mut out),
            TranscriptSyncInput::Clear { at } => self.on_clear(at, &mut out),
            TranscriptSyncInput::Started { at } => self.on_started(at, &mut out),
            TranscriptSyncInput::Progressed { position, at } => {
                // The sink's own report corrects the estimate.
                if let Some(i) = self.playing() {
                    self.segments[i].playhead = Some(Playhead { position, at });
                    self.apply(i, TranscriptPacerEvent::Position(position), &mut out);
                }
            }
            TranscriptSyncInput::Paused { at } => {
                if let Some(i) = self.playing() {
                    self.advance(i, at, &mut out);
                    self.apply(i, TranscriptPacerEvent::Paused, &mut out);
                }
                self.paused = true;
            }
            TranscriptSyncInput::Resumed { at } => {
                // The playhead stood still while paused: it moves on from where it stopped.
                if std::mem::take(&mut self.paused)
                    && let Some(i) = self.playing()
                    && let Some(head) = &mut self.segments[i].playhead
                {
                    head.at = at;
                }
            }
            TranscriptSyncInput::Tick { at } => {
                self.scheduled = None;
                if let Some(i) = self.playing() {
                    self.advance(i, at, &mut out);
                }
            }
            TranscriptSyncInput::Finished { position, interrupted } => {
                return self.on_finished(position, interrupted);
            }
            TranscriptSyncInput::Reset => self.on_reset(&mut out),
            TranscriptSyncInput::Enabled(enabled) => {
                if self.enabled != enabled {
                    self.enabled = enabled;
                    self.rotate(&mut out);
                }
            }
            TranscriptSyncInput::Rotate => self.rotate(&mut out),
        }
        out.retain(|e| !matches!(e, TranscriptPacerEffect::Heard(_)));
        out
    }

    fn on_text(
        &mut self,
        chunk: lk_agents_abi::TranscriptChunk,
        start_time: Option<f64>,
        end_time: Option<f64>,
        out: &mut Vec<TranscriptPacerEffect>,
    ) {
        self.segments.retain(|s| !s.cleared);
        let i = self.next(|s| !s.text_ended);
        if self.segments[i].pacer.is_none() {
            // A segment closed mid-write: its words have no pace to keep.
            let (start_time, end_time) = (UNPACED_TIME, UNPACED_TIME);
            out.push(TranscriptPacerEffect::Forward { chunk, start_time, end_time });
        } else {
            self.apply(i, TranscriptPacerEvent::Text { chunk, start_time, end_time }, out);
        }
    }

    fn on_text_end(&mut self, out: &mut Vec<TranscriptPacerEffect>) {
        if let Some(i) = self.segments.iter().position(|s| !s.text_ended) {
            self.segments[i].text_ended = true;
            self.apply(i, TranscriptPacerEvent::TextEnd, out);
            self.settle(i);
        }
    }

    fn on_audio(
        &mut self,
        secs: f64,
        key: Option<PlaybackSegment>,
        out: &mut Vec<TranscriptPacerEffect>,
    ) {
        self.segments.retain(|s| !s.cleared);
        let i = self.next(|s| !s.audio_ended);
        self.segments[i].audio = true;
        self.segments[i].key = key.or(self.segments[i].key);
        self.apply(i, TranscriptPacerEvent::Audio { secs }, out);
    }

    fn on_audio_end(&mut self, out: &mut Vec<TranscriptPacerEffect>) {
        // Ended before any of its words arrived: a segment with no audio, unless its
        // words will pass unpaced.
        if !self.enabled && self.segments.iter().all(|s| s.audio_ended) {
            return;
        }
        let i = self.next(|s| !s.audio_ended);
        if self.segments[i].cleared {
            // The cut speech's leg ended after its clear: a new segment here would pair the
            // next speech's words with no audio.
            self.segments.remove(i);
            return;
        }
        self.segments[i].audio_ended = true;
        self.apply(i, TranscriptPacerEvent::AudioEnd, out);
        if self.segments[i].cut_below {
            // Cut below and no reset came: the sink owes it no finish.
            self.apply(i, TranscriptPacerEvent::Close, out);
            self.segments.remove(i);
        } else {
            self.settle(i);
        }
    }

    fn on_clear(&mut self, at: Ts, out: &mut Vec<TranscriptPacerEffect>) {
        self.stop_playouts(at, out);
        if !self.segments.back().is_some_and(|s| !s.audio) {
            return;
        }
        let i = self.segments.len() - 1;
        self.apply(i, TranscriptPacerEvent::Close, out);
        let segment = &mut self.segments[i];
        if segment.audio_ended {
            self.segments.pop_back();
        } else {
            (segment.pacer, segment.text_ended, segment.cleared) = (None, true, true);
        }
    }

    /// The sink dropped the audio of every segment it owed: each one's words stand where its
    /// playout was at `at`, and what was heard of it by then is reported for its playback, which
    /// may never report its own finish.
    fn stop_playouts(&mut self, at: Ts, out: &mut Vec<TranscriptPacerEffect>) {
        let playing = self.playing();
        for i in 0..self.segments.len() {
            let segment = &self.segments[i];
            let Some(pacer) = segment.pacer.as_ref().filter(|p| live(p) && !p.is_cleared()) else {
                continue;
            };
            if !segment.audio || segment.cut_below {
                continue;
            }
            let (key, had_text) = (segment.key, pacer.saw_text());
            if playing == Some(i) {
                self.advance(i, at, out);
            }
            let position = self.segments[i].pacer.as_ref().map(TranscriptPacer::position);
            let position = position.unwrap_or_default();
            let mut cleared = Vec::new();
            self.apply(i, TranscriptPacerEvent::Cleared { position }, &mut cleared);
            for effect in cleared {
                match (effect, key) {
                    // Audio with no words: no transcript, so the speech's own text stands.
                    (TranscriptPacerEffect::Heard(heard), Some(segment)) if had_text => {
                        out.push(TranscriptPacerEffect::Cleared { segment, heard })
                    }
                    (TranscriptPacerEffect::Heard(_), _) => {}
                    (effect, _) => out.push(effect),
                }
            }
        }
    }

    /// The tick that next moves the playing segment's words, unless one at or before it is
    /// already asked for. `None` while paused, with nothing waiting on the playhead, or with the
    /// next word past the audio sent: the next input asks again.
    pub fn wake(&mut self) -> Option<Ts> {
        let wake = self.next_wake()?;
        if self.scheduled.is_some_and(|armed| armed <= wake) {
            return None;
        }
        self.scheduled = Some(wake);
        Some(wake)
    }

    fn next_wake(&self) -> Option<Ts> {
        if self.paused {
            return None;
        }
        let segment = &self.segments[self.playing()?];
        let head = segment.playhead?;
        let pacer = segment.pacer.as_ref().filter(|p| !p.is_over())?;
        let due = pacer.next_due()?;
        if due > pacer.audio_sent().max(head.position) {
            return None;
        }
        Some(head.at.saturating_add(due.saturating_sub(head.position) + TICK_MARGIN))
    }

    /// Moves segment `i` to where its playout is estimated to be at `at`: its last known
    /// position plus the time since, unless paused, and never past the audio sent.
    fn advance(&mut self, i: usize, at: Ts, out: &mut Vec<TranscriptPacerEffect>) {
        let segment = &mut self.segments[i];
        let (Some(head), Some(pacer)) = (&mut segment.playhead, &segment.pacer) else { return };
        if !self.paused {
            let moved = head.position + at.saturating_duration_since(head.at);
            let position = moved.min(pacer.audio_sent()).max(head.position);
            *head = Playhead { position, at };
        }
        let position = head.position;
        self.apply(i, TranscriptPacerEvent::Position(position), out);
    }

    fn on_started(&mut self, at: Ts, out: &mut Vec<TranscriptPacerEffect>) {
        // A sink plays its segments in order: one starting while an older one it had
        // started is still owed means that one's finish was lost. It ends where it stood.
        if let Some(i) = self.playing()
            && self.segments[i].started
            && self.segments.iter().skip(i + 1).any(|s| s.audio)
        {
            self.apply(i, TranscriptPacerEvent::Close, out);
            self.segments.remove(i);
        }
        // The first word is taken as the playback starts.
        let Some(i) = self.playing() else { return };
        self.segments[i].started = true;
        let position = self.segments[i].pacer.as_ref().map(TranscriptPacer::position);
        self.segments[i].playhead = Some(Playhead { position: position.unwrap_or_default(), at });
        if let Some(position) = position {
            self.apply(i, TranscriptPacerEvent::Position(position), out);
        }
    }

    /// The sink finished the playing segment. Its `Heard` is kept only when it had text.
    fn on_finished(&mut self, position: Duration, interrupted: bool) -> Vec<TranscriptPacerEffect> {
        let heard = |e: &TranscriptPacerEffect| matches!(e, TranscriptPacerEffect::Heard(_));
        let mut out = Vec::new();
        let Some(i) = self.playing() else { return out };
        let segment = &self.segments[i];
        let splits = |p: &TranscriptPacer| live(p) && !p.is_cleared();
        if interrupted && !segment.audio_ended && segment.pacer.as_ref().is_some_and(splits) {
            // Still capturing: a sink that cuts and resets its count plays the rest anew.
            let had_text = segment.pacer.as_ref().is_some_and(TranscriptPacer::saw_text);
            self.segments[i].cut_below = true;
            self.apply(i, TranscriptPacerEvent::PlaybackSplit { position }, &mut out);
            out.retain(|e| had_text || !heard(e));
            return out;
        }
        let segment = self.segments.remove(i).expect("playing is an index");
        let Some(pacer) = segment.pacer.filter(live) else { return out };
        // Audio with no words: no transcript, so the speech's own text stands.
        let had_text = pacer.saw_text();
        let (_, effects) =
            pacer.step(TranscriptPacerEvent::PlaybackFinished { position, interrupted });
        out.extend(effects.into_iter().filter(|e| had_text || !heard(e)));
        out
    }

    /// The sink threw its playback count away: a segment whose audio goes on starts the rest
    /// as a playout of its own, and the others it owed end where they stood.
    fn on_reset(&mut self, out: &mut Vec<TranscriptPacerEffect>) {
        for i in 0..self.segments.len() {
            let segment = &self.segments[i];
            let splits = |p: &TranscriptPacer| live(p) && !p.is_cleared();
            if segment.audio && !segment.audio_ended && segment.pacer.as_ref().is_some_and(splits) {
                self.split_playout(i, out);
            } else if segment.audio {
                self.apply(i, TranscriptPacerEvent::Close, out);
            }
        }
        self.segments.retain(|s| !s.audio);
    }

    /// Its audio goes on: the rest is a playout of its own, the words with it.
    fn split_playout(&mut self, i: usize, out: &mut Vec<TranscriptPacerEffect>) {
        let segment = &mut self.segments[i];
        if !std::mem::take(&mut segment.cut_below) {
            let position = segment.pacer.as_ref().map(TranscriptPacer::position);
            let position = position.unwrap_or_default();
            self.apply(i, TranscriptPacerEvent::PlaybackSplit { position }, out);
        }
        let segment = &mut self.segments[i];
        (segment.audio, segment.started, segment.playhead) = (false, false, None);
    }

    /// The oldest segment `open` holds for, else a new one.
    fn next(&mut self, open: impl Fn(&Segment) -> bool) -> usize {
        if let Some(i) = self.segments.iter().position(open) {
            return i;
        }
        let pacer = self.enabled.then(|| TranscriptPacer::new(self.opts.clone()));
        let segment = Segment {
            pacer,
            text_ended: false,
            audio: false,
            key: None,
            audio_ended: false,
            started: false,
            playhead: None,
            cut_below: false,
            cleared: false,
        };
        self.segments.push_back(segment);
        self.segments.len() - 1
    }

    /// A segment with no audio is gone once its pacer and its text are: nothing will report its
    /// playout.
    fn settle(&mut self, i: usize) {
        let segment = &self.segments[i];
        if !segment.audio
            && segment.text_ended
            && segment.pacer.as_ref().is_none_or(TranscriptPacer::is_over)
        {
            self.segments.remove(i);
        }
    }

    /// The oldest segment the sink owes a playout.
    fn playing(&self) -> Option<usize> {
        self.segments.iter().position(|s| s.audio && !s.cut_below)
    }

    /// Closing ends the text output's segment even with none open.
    fn rotate(&mut self, out: &mut Vec<TranscriptPacerEffect>) {
        if self.segments.is_empty() {
            out.push(TranscriptPacerEffect::Flush);
        }
        for i in 0..self.segments.len() {
            self.apply(i, TranscriptPacerEvent::Close, out);
            let segment = &mut self.segments[i];
            segment.pacer = None;
            segment.text_ended = true;
        }
        self.segments.retain(|s| s.audio && !s.cut_below);
    }

    fn apply(&mut self, i: usize, ev: TranscriptPacerEvent, out: &mut Vec<TranscriptPacerEffect>) {
        let slot = &mut self.segments[i].pacer;
        // An ended segment takes nothing more; its slot still accounts for its playout.
        let Some(pacer) = slot.take_if(|p| !p.is_over()) else { return };
        let (pacer, effects) = pacer.step(ev);
        *slot = Some(pacer);
        out.extend(effects);
    }
}

/// A pacer still owed its playout's end: not over, and with audio or text to account for.
fn live(pacer: &TranscriptPacer) -> bool {
    !pacer.is_over() && !pacer.is_blank()
}
