pub mod effects;
mod estimate;
pub mod machine;

use lk_agents_abi::TranscriptChunk;
use lk_agents_core::lexer::{Lexer, Projector, Rules, Unit, Units, View, Vocabulary};
use std::time::Duration;

/// What was heard of one segment: the words, where the playhead stopped, whether it played out.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Heard {
    pub text: String,
    pub position: Duration,
    pub complete: bool,
    /// Zero for a segment with no audio.
    pub frames: u64,
    /// The audio heard over every segment of a forwarding, the cut one up to the cut: where a
    /// realtime reply's messages are truncated. `position` alone is the last segment's.
    pub played: Duration,
}

impl Heard {
    pub fn complete(text: impl Into<String>, position: Duration, frames: u64) -> Heard {
        Heard { text: text.into(), position, complete: true, frames, played: position }
    }

    /// A segment cut at `position`; `text` is what was heard up to there.
    pub fn cut(text: impl Into<String>, position: Duration, frames: u64) -> Heard {
        Heard { text: text.into(), position, complete: false, frames, played: position }
    }
}

/// Whether a transcript output has text since its last flush: a flush closes a segment that had
/// text, and an empty one never reaches the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TranscriptSegment {
    open: bool,
}

impl TranscriptSegment {
    /// Text went to the output; empty text opens nothing.
    pub fn wrote(&mut self, text: &str) {
        self.open |= !text.is_empty();
    }

    /// Whether this flush reaches the output, closing the segment either way.
    pub fn flush(&mut self) -> bool {
        std::mem::take(&mut self.open)
    }
}

/// The plain text of what the transcript showed, as history keeps it: its markdown taken out.
pub fn plain(shown: &str) -> String {
    let rules = Rules { markdown: true, emoji: true, ..Rules::default() };
    Projector::new(View::Plain).text(&Lexer::all(rules, shown))
}

/// Whether `next` continues the word `text` ends: no blank between them. A word the reply's
/// deltas split (`Tuesda` + `y`) and markup glued to it (`**` + `Upgrade`) are one word to every
/// reader of transcript pieces.
pub fn continues_word(text: &str, next: &str) -> bool {
    let word = |c: char| !c.is_whitespace();
    text.ends_with(word) && next.starts_with(word)
}

/// The transcript's pieces as words, each with where its audio ends: what a cut keeps as heard.
/// Silent text (`- `, `**`) leads the word after it and is heard with that word; the words are
/// one voice, so a word's audio was over when the next one's began.
#[derive(Debug, Default)]
pub struct HeardWords {
    words: Vec<HeardWord>,
}

#[derive(Debug)]
struct HeardWord {
    text: String,
    end: f64,
    /// Whether it has a spoken word yet: silent text alone waits for the word it leads.
    spoken: bool,
}

impl HeardWords {
    /// A piece shown from `start` to `end` seconds into the segment.
    pub fn push(&mut self, text: &str, spoken: bool, start: f64, end: f64) {
        match self.words.last_mut() {
            Some(last) if !last.spoken => {
                last.text.push_str(text);
                (last.end, last.spoken) = (end, spoken);
            }
            Some(last) if continues_word(&last.text, text) => {
                last.text.push_str(text);
                last.end = last.end.max(end);
            }
            last => {
                if let Some(last) = last {
                    last.end = last.end.min(start);
                }
                self.words.push(HeardWord { text: text.to_owned(), end, spoken });
            }
        }
    }

    /// The words whose audio ended by `at` seconds, as shown.
    pub fn heard(&self, at: f64) -> String {
        self.words.iter().take_while(|w| w.end <= at && w.spoken).map(|w| w.text.as_str()).collect()
    }

    /// Every word, as shown.
    pub fn all(&self) -> String {
        self.words.iter().map(|w| w.text.as_str()).collect()
    }
}

/// A plain string a host hands the transcript chain (a `transcription_node` yielding strings, a
/// custom output): it never went through a voice's reading, so its markup (any provider's, the
/// voice unknown here) is taken out and its delivery lifted onto the chunk. Per segment.
#[derive(Debug, Default)]
pub struct HostText {
    stripper: lk_agents_core::markup::TranscriptMarkupStripper,
    seen: usize,
}

impl HostText {
    pub fn push(&mut self, text: &str) -> TranscriptChunk {
        let shown = self.stripper.push(text);
        let tags = &self.stripper.tags()[self.seen..];
        self.seen = self.stripper.tags().len();
        let expression = tags
            .iter()
            .find(|t| t.kind == "expression" || t.kind == lk_agents_abi::tts::markup::EMOTION_TAG)
            .map(|t| t.value.clone());
        TranscriptChunk { expression, ..TranscriptChunk::new(shown) }
    }

    /// The segment ended: a tag it was cut inside, and a held trailing blank, go.
    pub fn reset(&mut self) {
        *self = HostText::default();
    }
}

/// One piece of agent text, read once by its segment's lexer.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Visible {
    /// What the TTS is sent: the spoken text, its markup in canonical form for the provider.
    pub said: String,
    /// What the transcript shows, in order: words the voice says and text with no speech.
    pub shown: Vec<TranscriptChunk>,
    /// The plain text the item carries: no markdown, no markup.
    pub text: String,
}

impl Visible {
    /// A whole piece of text, under the voice's tags `vocab`; for deltas use [`SegmentText`].
    pub fn of(text: &str, vocab: &Vocabulary) -> Visible {
        let mut segment = SegmentText::new(vocab.clone());
        let mut visible = segment.push(text);
        visible.extend(segment.finish());
        visible
    }

    /// Nothing anyone sees or hears: a delta that was markup only, or text still being held.
    pub fn is_blank(&self) -> bool {
        self.said.is_empty() && self.shown.is_empty() && self.text.is_empty()
    }

    pub fn extend(&mut self, more: Visible) {
        self.said.push_str(&more.said);
        self.shown.extend(more.shown);
        self.text.push_str(&more.text);
    }

    /// The transcript's text of this piece.
    pub fn shown_text(&self) -> String {
        self.shown.iter().map(|c| c.text.as_str()).collect()
    }

    /// The first delivery this piece asks for.
    pub fn expression(&self) -> Option<&str> {
        self.shown.iter().find_map(|c| c.expression.as_deref())
    }
}

/// One segment's text, lexed once: every view of it comes from the same tokens, so the TTS, the
/// transcript and the history never disagree on what was markup.
#[derive(Debug, Clone)]
pub struct SegmentText {
    lexer: Lexer,
    said: Projector,
    plain: Projector,
    units: Units,
    /// A delivery met since the last chunk: it goes on the next one.
    expression: Option<String>,
}

impl SegmentText {
    /// A segment of a voice whose TTS speaks the tags `vocab`.
    pub fn new(vocab: Vocabulary) -> SegmentText {
        SegmentText {
            lexer: Lexer::new(Rules::speech(vocab)),
            said: Projector::new(View::Synthesis),
            plain: Projector::new(View::Plain),
            units: Units::new(View::Spoken, View::Display),
            expression: None,
        }
    }

    /// One delta in; what it settled out. Only a tag or delimiter still arriving is held.
    pub fn push(&mut self, delta: &str) -> Visible {
        let tokens = self.lexer.push(delta);
        let units = self.units.push(&tokens);
        self.read(&tokens, units)
    }

    /// The segment ended: whatever was held settles; a cut tag is dropped.
    pub fn finish(&mut self) -> Visible {
        let tokens = self.lexer.finish();
        let units = self.units.push(&tokens);
        let mut visible = self.read(&tokens, units);
        if let Some(expression) = self.expression.take() {
            visible.shown.push(TranscriptChunk {
                expression: Some(expression),
                ..TranscriptChunk::silent("")
            });
        }
        visible
    }

    /// The units as chunks: words and the blanks between them run together as spoken text (the
    /// pacer splits words itself); only text with no speech is a chunk of its own. A delivery
    /// starts a chunk.
    fn read(&mut self, tokens: &[lk_agents_core::lexer::Token], units: Vec<Unit>) -> Visible {
        let mut shown: Vec<TranscriptChunk> = Vec::new();
        for unit in units {
            let (text, spoken) = match unit {
                Unit::Said(said) => (said, true),
                Unit::Silent { shown, tags } => {
                    if let Some(delivery) =
                        tags.iter().rev().find_map(lk_agents_core::markup::delivery)
                    {
                        self.expression = Some(delivery);
                    }
                    let blank = shown.trim().is_empty();
                    (shown, blank)
                }
            };
            if text.is_empty() {
                continue;
            }
            match shown.last_mut() {
                Some(last) if spoken && last.spoken && self.expression.is_none() => {
                    last.text.push_str(&text);
                }
                _ => shown.push(TranscriptChunk {
                    spoken,
                    expression: self.expression.take(),
                    ..TranscriptChunk::new(text)
                }),
            }
        }
        Visible { said: self.said.text(tokens), shown, text: self.plain.text(tokens) }
    }
}
