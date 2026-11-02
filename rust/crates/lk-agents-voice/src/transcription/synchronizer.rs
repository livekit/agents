use crate::transcription::pacer::{
    TranscriptPacer, TranscriptPacerEffect, TranscriptPacerEvent, TranscriptPacerOptions,
};
use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;
use std::time::Duration;

const PATTERNS: &str = include_str!("liang/patterns.txt");

const EXCEPTIONS: &str = include_str!("liang/exceptions.txt");

/// Counts syllable-like pieces used to pace transcript words: [`Liang`] by default.
pub trait Hyphenator: Send + Sync {
    /// `len(hyphenate_word(word))`. Never zero, including for the empty string.
    fn hyphens(&self, word: &str) -> usize;
}

#[derive(Default)]
struct Node {
    children: BTreeMap<char, Node>,
    /// The pattern's points when a pattern ends at this node (`t[None]`).
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
        let mut exceptions = BTreeMap::new();
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
pub(super) enum TranscriptSyncInput {
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
    pacer: Option<TranscriptPacer>,
    text_ended: bool,
    /// A frame reached the sink: the sink owes this segment's playback finish.
    audio: bool,
    audio_ended: bool,
    /// The sink reported this segment's playback started.
    started: bool,
    /// The sink cut this segment's playout while its audio was still coming. It owes no finish
    /// until a reset starts the rest of its audio as a playout of its own.
    cut_below: bool,
    /// A clear closed its words before any of its audio and before its audio leg ended: the
    /// leg's late end is its, not a new segment's. Gone at the next words or audio: a cut leg
    /// usually ends with no audio end at all, and the next speech's end is its own.
    cleared: bool,
}

/// `_SegmentSynchronizerImpl`s in order. Each leg is an ordered stream of segments delimited by
/// its flushes: a segment's text is the next text not yet ended, its audio the next audio, so a
/// segment the producer ended with no audio is one by construction even when its words arrive
/// later (through a `transcription_node`). The sink's playback moves the oldest segment it owes.
pub(super) struct TranscriptSyncSegments {
    opts: TranscriptPacerOptions,
    enabled: bool,
    segments: VecDeque<Segment>,
}

impl TranscriptSyncSegments {
    pub fn new(opts: TranscriptPacerOptions) -> TranscriptSyncSegments {
        TranscriptSyncSegments { opts, enabled: false, segments: VecDeque::new() }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// One input in, the words to show out. A `Heard` is kept only for a `Finished` segment that
    /// had text: its synchronised transcript.
    pub fn step(&mut self, input: TranscriptSyncInput) -> Vec<TranscriptPacerEffect> {
        let mut out = Vec::new();
        match input {
            TranscriptSyncInput::Text { raw, start_time, end_time } => {
                self.segments.retain(|s| !s.cleared);
                let i = self.next(|s| !s.text_ended);
                if self.segments[i].pacer.is_none() {
                    // A segment closed mid-write: its words have no pace to keep.
                    out.push(TranscriptPacerEffect::Forward {
                        word: raw,
                        start_time: 0.0,
                        end_time: 0.0,
                    });
                } else {
                    self.apply(
                        i,
                        TranscriptPacerEvent::Text { raw, start_time, end_time },
                        &mut out,
                    );
                }
            }
            TranscriptSyncInput::TextEnd => {
                if let Some(i) = self.segments.iter().position(|s| !s.text_ended) {
                    self.segments[i].text_ended = true;
                    self.apply(i, TranscriptPacerEvent::TextEnd, &mut out);
                    self.settle(i);
                }
            }
            TranscriptSyncInput::Audio { secs } => {
                self.segments.retain(|s| !s.cleared);
                let i = self.next(|s| !s.audio_ended);
                self.segments[i].audio = true;
                self.apply(i, TranscriptPacerEvent::Audio { secs }, &mut out);
            }
            TranscriptSyncInput::AudioEnd => {
                // Ended before any of its words arrived: a segment with no audio, unless its
                // words will pass unpaced.
                if !self.enabled && self.segments.iter().all(|s| s.audio_ended) {
                    return out;
                }
                let i = self.next(|s| !s.audio_ended);
                if self.segments[i].cleared {
                    // The cut speech's leg ended after its clear (seed 19354): a new segment
                    // here would pair the next speech's words with no audio.
                    self.segments.remove(i);
                    return out;
                }
                self.segments[i].audio_ended = true;
                self.apply(i, TranscriptPacerEvent::AudioEnd, &mut out);
                if self.segments[i].cut_below {
                    // Cut below and no reset came: the sink owes it no finish.
                    self.apply(i, TranscriptPacerEvent::Close, &mut out);
                    self.segments.remove(i);
                } else {
                    self.settle(i);
                }
            }
            TranscriptSyncInput::Clear => {
                if self.segments.back().is_some_and(|s| !s.audio) {
                    let i = self.segments.len() - 1;
                    self.apply(i, TranscriptPacerEvent::Close, &mut out);
                    let segment = &mut self.segments[i];
                    if segment.audio_ended {
                        self.segments.pop_back();
                    } else {
                        (segment.pacer, segment.text_ended, segment.cleared) = (None, true, true);
                    }
                }
            }
            TranscriptSyncInput::Started => {
                // A sink plays its segments in order: one starting while an older one it had
                // started is still owed means that one's finish was lost. It ends where it stood.
                if let Some(i) = self.playing()
                    && self.segments[i].started
                    && self.segments.iter().skip(i + 1).any(|s| s.audio)
                {
                    self.apply(i, TranscriptPacerEvent::Close, &mut out);
                    self.segments.remove(i);
                }
                // The reference's pacing loop takes its first word as the playback starts.
                if let Some(i) = self.playing() {
                    self.segments[i].started = true;
                    if let Some(at) = self.segments[i].pacer.as_ref().map(TranscriptPacer::position)
                    {
                        self.apply(i, TranscriptPacerEvent::Position(at), &mut out);
                    }
                }
            }
            TranscriptSyncInput::Progressed(at) => {
                if let Some(i) = self.playing() {
                    self.apply(i, TranscriptPacerEvent::Position(at), &mut out);
                }
            }
            TranscriptSyncInput::Paused => {
                if let Some(i) = self.playing() {
                    self.apply(i, TranscriptPacerEvent::Paused, &mut out);
                }
            }
            TranscriptSyncInput::Finished { position, interrupted } => {
                let Some(i) = self.playing() else { return out };
                let segment = &self.segments[i];
                if interrupted && !segment.audio_ended && segment.pacer.as_ref().is_some_and(live) {
                    // Still capturing: a sink that cuts and resets its count plays the rest anew.
                    let had_text = segment.pacer.as_ref().is_some_and(TranscriptPacer::saw_text);
                    self.segments[i].cut_below = true;
                    self.apply(i, TranscriptPacerEvent::PlaybackSplit { position }, &mut out);
                    out.retain(|e| had_text || !matches!(e, TranscriptPacerEffect::Heard(_)));
                    return out;
                }
                let segment = self.segments.remove(i).expect("playing is an index");
                if let Some(pacer) = segment.pacer.filter(live) {
                    // Audio with no words: no transcript, so the speech's own text stands.
                    let had_text = pacer.saw_text();
                    let (_, effects) = pacer
                        .step(TranscriptPacerEvent::PlaybackFinished { position, interrupted });
                    let heard =
                        |e: &TranscriptPacerEffect| matches!(e, TranscriptPacerEffect::Heard(_));
                    out.extend(effects.into_iter().filter(|e| had_text || !heard(e)));
                    return out;
                }
            }
            TranscriptSyncInput::Reset => {
                for i in 0..self.segments.len() {
                    let segment = &mut self.segments[i];
                    if segment.audio
                        && !segment.audio_ended
                        && segment.pacer.as_ref().is_some_and(live)
                    {
                        // Its audio goes on: the rest is a playout of its own, the words with it.
                        if !std::mem::take(&mut segment.cut_below) {
                            let position = segment.pacer.as_ref().map(TranscriptPacer::position);
                            let position = position.unwrap_or_default();
                            self.apply(
                                i,
                                TranscriptPacerEvent::PlaybackSplit { position },
                                &mut out,
                            );
                        }
                        let segment = &mut self.segments[i];
                        (segment.audio, segment.started) = (false, false);
                    } else if segment.audio {
                        self.apply(i, TranscriptPacerEvent::Close, &mut out);
                    }
                }
                self.segments.retain(|s| !s.audio);
            }
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
            audio_ended: false,
            started: false,
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

    /// Closing ends the text output's segment even with none open, as closing the reference's
    /// idle `_SegmentSynchronizerImpl` does.
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
